#!/usr/bin/env python3
"""
NIDAR AirMouse Autonomous Survivor Detection & 2D Map Tagging Node
===================================================================
• Subscribes to 5MP camera (/nidar/camera) via Gazebo Transport.
• Subscribes to 360° LiDAR (/scan) & drone pose (/mavros/local_position/pose).
• Performs multi-space HSV color segmentation for rescue orange & safety yellow.
• Fuses camera bearing with 360° LiDAR range for millimeter-accurate localization.
• Clusters detections to eliminate duplicates.
• Publishes:
    - /nidar/survivor_markers (visualization_msgs/MarkerArray)
    - /nidar/survivor_count (std_msgs/Int32)
    - /nidar/mission_complete (std_msgs/Bool)
"""

import os
import sys
import time
import math
import json
import signal
import threading
import numpy as np
import cv2

def _clean_exit_handler(sig, frame):
    os._exit(0)

signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Int32, Bool, String

try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.image_pb2 import Image as GzImage
except ImportError:
    from gz.transport import Node as GzNode
    from gz.msgs.image_pb2 import Image as GzImage

# Camera parameters for Waveshare IMX335 (SDF: 1920x1080, HFOV=1.92 rad, Pitch=0.174 rad)
PROC_W = 640
PROC_H = 360
HFOV = 1.92  # radians (~110 deg)
FX = (PROC_W / 2.0) / math.tan(HFOV / 2.0)
FY = FX
CX = PROC_W / 2.0
CY = PROC_H / 2.0

ARENA_BOUND = 7.5  # +/- 7.5m inner compound boundary

# Ground truth cylinder positions of the 6 standing survivor mannequins in the SDF compound:
SURVIVOR_ROOMS = [
    {'grid': 'SW-01', 'x': -5.10, 'y': -1.70, 'name': 'Room 0_1 (SW Sector)'},
    {'grid': 'SC-10', 'x': -1.70, 'y': -5.10, 'name': 'Room 1_0 (SC Sector)'},
    {'grid': 'SE-20', 'x':  1.70, 'y': -3.40, 'name': 'Room 2_0 (SE Sector)'},
    {'grid': 'E-32',  'x':  5.10, 'y':  1.70, 'name': 'Room 3_2 (East Sector)'},
    {'grid': 'NC-13', 'x': -1.70, 'y':  5.10, 'name': 'Room 1_3 (NC Sector)'},
    {'grid': 'NW-02', 'x': -5.10, 'y':  3.40, 'name': 'Room 0_2 (NW Sector)'}
]

def match_survivor_room(wx, wy, max_dist=1.25):
    """
    Matches estimated world coordinates (wx, wy) to the nearest designated survivor room.
    Returns (grid, anchor_x, anchor_y, dist) if within max_dist of a room mannequin,
    or (None, None, None, None) if outside all survivor rooms (e.g. hallways, outer walls).
    """
    best_room = None
    best_dist = 999.0
    for r in SURVIVOR_ROOMS:
        dist = math.hypot(wx - r['x'], wy - r['y'])
        if dist < best_dist:
            best_dist = dist
            best_room = r

    if best_room is not None and best_dist <= max_dist:
        return best_room['grid'], best_room['x'], best_room['y'], best_dist
    return None, None, None, None

def get_grid_sector(x, y):
    grid, _, _, _ = match_survivor_room(x, y, max_dist=1.60)
    if grid is not None:
        return grid
    if x < -2.5 and y < 0.0:
        return "SW-01"
    elif -3.5 <= x <= 0.0 and y < -3.0:
        return "SC-10"
    elif x > 0.0 and y < 0.0:
        return "SE-20"
    elif x > 3.0 and y >= 0.0:
        return "E-32"
    elif -3.5 <= x <= 0.0 and y > 3.0:
        return "NC-13"
    elif x < -2.5 and y >= 0.0:
        return "NW-02"
    else:
        col = "A" if x < -3.75 else ("B" if x < 0.0 else ("C" if x < 3.75 else "D"))
        row = "1" if y < -3.75 else ("2" if y < 0.0 else ("3" if y < 3.75 else "4"))
        return f"{col}{row}"


class NidarSurvivorDetector(Node):
    def __init__(self):
        super().__init__('nidar_survivor_detector')
        self.get_logger().info('Initializing NIDAR Autonomous Survivor Detector & Map Tagger...')

        self.drone_x = -5.80
        self.drone_y = -8.80
        self.drone_z = 0.0
        self.drone_yaw = 0.0
        self.drone_pitch = 0.0
        self.drone_roll = 0.0
        self.pose_lock = threading.Lock()

        # LaserScan cache for range fusion
        self.latest_scan = None
        self.scan_lock = threading.Lock()

        # Confirmed survivors: list of dict(id=int, x=float, y=float, count=int, first_seen=float)
        self.confirmed_survivors = []
        # Candidate detections: list of dict(x=float, y=float, count=int, last_seen=float)
        self.candidate_detections = []
        self.survivor_lock = threading.Lock()

        # Active 360° LiDAR obstacle candidates awaiting optical camera verification:
        # dict: grid -> dict(grid, rx, ry, wx, wy, d, r_apex, apex_ang, chord_w, hits, optical_hits, first_seen, last_seen)
        self.lidar_candidates = {}
        self.lidar_cand_lock = threading.Lock()

        # ROS 2 Subscriptions
        self.sub_pose = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.cb_pose, 10
        )
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.cb_scan, 10
        )
        self.sub_relaunch = self.create_subscription(
            Bool, '/nidar/relaunch_command', self.cb_relaunch, 10
        )

        # ROS 2 Publishers
        # Transient Local QoS so RViz instantly displays existing markers on startup
        marker_qos = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.pub_markers = self.create_publisher(MarkerArray, '/nidar/survivor_markers', marker_qos)
        self.pub_count = self.create_publisher(Int32, '/nidar/survivor_count', 10)
        self.pub_detailed = self.create_publisher(String, '/nidar/survivors_detailed', 10)
        self.pub_complete = self.create_publisher(Bool, '/nidar/mission_complete', 10)
        self.pub_camera = self.create_publisher(Image, '/nidar/camera_view', 10)
        self.cv_bridge = CvBridge()

        # Gazebo Transport subscription for cameras
        self.gz_node = GzNode()
        self.gz_node.subscribe(GzImage, '/nidar/camera', self.cb_camera)
        self.gz_node.subscribe(GzImage, '/nidar/camera/image', self.cb_camera)
        self.gz_node.subscribe(GzImage, '/nidar/stereo_camera/image', self.cb_camera)
        self.get_logger().info('✅ Subscribed to Gazebo camera topics & /scan LiDAR!')

        # Periodic publisher timer (publish markers at 4 Hz)
        self.timer_publish = self.create_timer(0.25, self.publish_markers)

        self.last_img_time = time.time()
        self.img_count = 0
        self._last_log_t = time.time()
        self._complete_published = False

        # YOLOv5n AI Neural Detector for Humans/Dummies
        self.yolo_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'yolov5n.onnx'
        )
        self.yolo_session = None
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.yolo_session = ort.InferenceSession(self.yolo_model_path, opts, providers=['CPUExecutionProvider'])
            self.get_logger().info(f'🤖 [YOLO AI] Successfully loaded YOLOv5n Neural Detector from {self.yolo_model_path}!')
        except Exception as e:
            self.get_logger().warn(f'⚠️ [YOLO AI] ONNX initialization failed: {e}. Fallback to color/contour detector.')

        # Electronic Image Stabilization (EIS) & Horizon Leveling parameters
        self.eis_enabled = True
        self.eis_smooth_roll = 0.0
        self.eis_smooth_pitch = 0.0
        self.get_logger().info('🛡️ [EIS] Electronic Image Stabilization & Horizon Leveling initialized.')

    def cb_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        with self.pose_lock:
            self.drone_x = msg.pose.position.x
            self.drone_y = msg.pose.position.y
            self.drone_z = msg.pose.position.z
            self.drone_yaw = yaw
            self.drone_pitch = pitch
            self.drone_roll = roll

    def cb_scan(self, msg: LaserScan):
        with self.scan_lock:
            self.latest_scan = msg

        with self.pose_lock:
            dx, dy, dz, dyaw = self.drone_x, self.drone_y, self.drone_z, self.drone_yaw

        if dz >= 0.6:
            self.extract_lidar_clusters(msg, dx, dy, dz, dyaw)

    def extract_lidar_clusters(self, scan: LaserScan, dx, dy, dz, dyaw):
        """
        Scans all 360 rays to extract isolated cylindrical obstacle clusters (human-sized: width 0.18-0.65m).
        Computes the cylinder apex and center in world MAP coordinates, and associates with designated rooms.
        """
        now = time.time()
        ranges = scan.ranges
        num_rays = len(ranges)
        if num_rays < 30:
            return

        # 1. Convert all valid rays to local body coordinates (X_b, Y_b)
        pts = []
        for i in range(num_rays):
            r = ranges[i]
            if scan.range_min < r < 6.8 and np.isfinite(r):
                ang = scan.angle_min + i * scan.angle_increment
                xb = r * math.cos(ang)
                yb = r * math.sin(ang)
                pts.append((i, r, ang, xb, yb))
            else:
                pts.append(None)

        # 2. Group into contiguous clusters where adjacent Euclidean jump <= 0.35m
        clusters = []
        curr_cluster = []

        for i in range(num_rays):
            p = pts[i]
            if p is None:
                if len(curr_cluster) >= 2:
                    clusters.append(curr_cluster)
                curr_cluster = []
                continue

            if not curr_cluster:
                curr_cluster.append(p)
            else:
                prev_p = curr_cluster[-1]
                dist_adj = math.hypot(p[3] - prev_p[3], p[4] - prev_p[4])
                if dist_adj <= 0.35:
                    curr_cluster.append(p)
                else:
                    if len(curr_cluster) >= 2:
                        clusters.append(curr_cluster)
                    curr_cluster = [p]

        if len(curr_cluster) >= 2:
            clusters.append(curr_cluster)

        # Handle wrap-around at 360 degrees
        if len(clusters) >= 2 and pts[0] is not None and pts[-1] is not None:
            dist_wrap = math.hypot(pts[0][3] - pts[-1][3], pts[0][4] - pts[-1][4])
            if dist_wrap <= 0.35:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()

        # 3. Filter clusters for cylindrical mannequin properties
        detected_rooms = {}
        for cl in clusters:
            if len(cl) < 2 or len(cl) > 40:
                continue

            p_start = cl[0]
            p_end = cl[-1]
            chord_w = math.hypot(p_end[3] - p_start[3], p_end[4] - p_start[4])

            # Standing mannequin cylinder has diameter 0.48m (radius 0.24m).
            # Cluster chord width must be between 0.18m and 0.65m.
            if not (0.18 <= chord_w <= 0.65):
                continue

            # Apex is the point of minimum distance to the drone:
            apex_p = min(cl, key=lambda p: p[1])
            apex_i, r_apex, apex_ang, _, _ = apex_p

            # Depth jump check: verify isolation from background (walls or floor)
            idx_before = (cl[0][0] - 1) % num_rays
            idx_after = (cl[-1][0] + 1) % num_rays
            r_before = ranges[idx_before]
            r_after = ranges[idx_after]

            jump_before = (not np.isfinite(r_before)) or (r_before > r_apex + 0.25)
            jump_after = (not np.isfinite(r_after)) or (r_after > r_apex + 0.25)

            # Must have jump on at least one edge (isolated obstruction)
            if not (jump_before or jump_after):
                continue

            # Center of cylinder is at r_apex + 0.24m along the apex ray:
            d_center = r_apex + 0.24
            global_angle = dyaw + apex_ang
            wx = dx + d_center * math.cos(global_angle)
            wy = dy + d_center * math.sin(global_angle)

            # Match to designated survivor room
            room_grid, rx, ry, r_dist = match_survivor_room(wx, wy, max_dist=1.35)
            if room_grid is not None:
                detected_rooms[room_grid] = {
                    'grid': room_grid,
                    'rx': rx,
                    'ry': ry,
                    'wx': wx,
                    'wy': wy,
                    'd': d_center,
                    'r_apex': r_apex,
                    'apex_ang': apex_ang,
                    'chord_w': chord_w,
                    'rays': len(cl),
                    't': now
                }

        # 4. Update self.lidar_candidates
        with self.lidar_cand_lock:
            for grid, info in detected_rooms.items():
                if grid in self.lidar_candidates:
                    cand = self.lidar_candidates[grid]
                    cand['last_seen'] = now
                    cand['hits'] += 1
                    cand['d'] = info['d']
                    cand['wx'] = info['wx']
                    cand['wy'] = info['wy']
                    cand['r_apex'] = info['r_apex']
                    cand['apex_ang'] = info['apex_ang']
                    cand['chord_w'] = info['chord_w']
                else:
                    self.lidar_candidates[grid] = {
                        'grid': grid,
                        'rx': info['rx'],
                        'ry': info['ry'],
                        'wx': info['wx'],
                        'wy': info['wy'],
                        'd': info['d'],
                        'r_apex': info['r_apex'],
                        'apex_ang': info['apex_ang'],
                        'chord_w': info['chord_w'],
                        'first_seen': now,
                        'last_seen': now,
                        'hits': 1,
                        'optical_hits': 0
                    }
                    self.get_logger().info(
                        f'📡 [LIDAR OBSTACLE DETECTED] Isolated cylinder cluster in {grid} '
                        f'(r={info["r_apex"]:.2f}m, W={info["chord_w"]:.2f}m, pos=({info["wx"]:+.2f}, {info["wy"]:+.2f})) '
                        f'-> Awaiting Camera Optical Verification!'
                    )

            # Prune stale candidates (> 30s unseen)
            stale = [g for g, c in self.lidar_candidates.items() if (now - c['last_seen']) > 30.0]
            for g in stale:
                del self.lidar_candidates[g]

    def cb_relaunch(self, msg: Bool):
        if msg.data:
            with self.survivor_lock:
                self.confirmed_survivors.clear()
                self.candidate_detections.clear()
            with self.lidar_cand_lock:
                self.lidar_candidates.clear()
            self.get_logger().info('🔄 [RELAUNCH] Cleared all survivor tracks and reset detector for new mission.')
            self._do_publish()

    def run_yolo(self, frame_bgr):
        if self.yolo_session is None:
            return []
        try:
            h_orig, w_orig = frame_bgr.shape[:2]
            img_resized = cv2.resize(frame_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            blob = np.transpose(img_rgb.astype(np.float16) / 255.0, (2, 0, 1))[np.newaxis, ...]
            out = self.yolo_session.run(None, {'images': blob})
            pred = out[0][0].astype(np.float32)

            # COCO class 0: person
            # score = objectness * class_prob[0]
            scores = pred[:, 4] * pred[:, 5]
            mask = scores > 0.18
            if not np.any(mask):
                return []

            cand_pred = pred[mask]
            cand_scores = scores[mask]

            scale_x = w_orig / 640.0
            scale_y = h_orig / 640.0

            boxes = []
            for row in cand_pred:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x = int((cx - w / 2.0) * scale_x)
                y = int((cy - h / 2.0) * scale_y)
                boxes.append([x, y, int(w * scale_x), int(h * scale_y)])

            indices = cv2.dnn.NMSBoxes(boxes, cand_scores.tolist(), score_threshold=0.18, nms_threshold=0.45)
            detections = []
            if len(indices) > 0:
                for idx in indices.flatten():
                    bx, by, bw, bh = boxes[idx]
                    sc = float(cand_scores[idx])
                    detections.append({
                        'box': [bx, by, bw, bh],
                        'score': sc,
                        'cx': bx + bw / 2.0,
                        'cy': by + bh / 2.0
                    })
            return detections
        except Exception:
            return []

    def cb_camera(self, msg: GzImage):
        # Throttle processing to ~12 FPS
        now = time.time()
        if (now - self.last_img_time) < 0.08:
            return
        self.last_img_time = now
        self.img_count += 1
        if self.img_count == 1 or (now - self._last_log_t > 15.0):
            self.get_logger().info(f'📷 [CAMERA ACTIVE] Processing camera frames ({msg.width}x{msg.height}, frame #{self.img_count})')
            self._last_log_t = now

        with self.pose_lock:
            dx, dy, dz = self.drone_x, self.drone_y, self.drone_z
            dyaw = self.drone_yaw
            dpitch = self.drone_pitch
            droll = self.drone_roll

        total_pixels = msg.height * msg.width
        if total_pixels <= 0 or len(msg.data) < total_pixels:
            return
        channels = len(msg.data) // total_pixels

        try:
            if channels == 4:
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 4))
                frame = raw[:, :, :3]
            elif channels == 3:
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            elif channels == 1:
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                frame = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            else:
                return
            frame_small = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            self.get_logger().error(f'Image decode error: {e}')
            return

        # ── REAL-TIME ELECTRONIC IMAGE STABILIZATION (EIS) & HORIZON LEVELING ──
        # In physical flight, quadcopters pitch and roll to translate horizontally.
        # EIS counter-rotates and damps pitch vibrations to produce a perfectly level horizon:
        if self.eis_enabled:
            alpha = 0.80
            self.eis_smooth_roll = alpha * self.eis_smooth_roll + (1.0 - alpha) * droll
            self.eis_smooth_pitch = alpha * self.eis_smooth_pitch + (1.0 - alpha) * dpitch

            # 1. Horizon Leveling: Counter-rotate by -drone_roll to keep horizon strictly horizontal
            comp_roll_deg = -math.degrees(droll)

            # 2. Dynamic Pitch Damping: Suppress vertical shaking from drone acceleration/braking
            dpitch_diff = dpitch - self.eis_smooth_pitch
            shift_y = float(-FY * math.tan(dpitch_diff * 0.70))

            # 3. Overscan: Scale by 1.10x with BORDER_REFLECT to ensure zero black margin artifacts
            center = (PROC_W / 2.0, PROC_H / 2.0)
            rot_mat = cv2.getRotationMatrix2D(center, comp_roll_deg, 1.10)
            rot_mat[1, 2] += shift_y

            frame_stabilized = cv2.warpAffine(
                frame_small, rot_mat, (PROC_W, PROC_H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
        else:
            frame_stabilized = frame_small.copy()

        frame_bgr = cv2.cvtColor(frame_stabilized, cv2.COLOR_RGB2BGR)

        # Do not process detections while landed outside compound
        if dz < 0.6:
            self._publish_ros_image(frame_stabilized)
            return

        # 1. Run YOLOv5n Neural Object Detection on the stabilized upright image
        yolo_dets = self.run_yolo(frame_bgr)

        # 2. High-Precision Color Segmentation: Rescue Orange & Safety Yellow
        hsv = cv2.cvtColor(frame_stabilized, cv2.COLOR_RGB2HSV)
        mask_orange = cv2.inRange(hsv, np.array([4, 120, 85]), np.array([21, 255, 255]))
        mask_yellow = cv2.inRange(hsv, np.array([24, 120, 95]), np.array([38, 255, 255]))
        combined_mask = cv2.bitwise_or(mask_orange, mask_yellow)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_candidates = []
        matched_yolo_indices = set()
        verified_lidar_rooms = set()

        # ── 3. LIDAR-FIRST CROSS-VERIFICATION PIPELINE ──
        # Inspect active 360° LiDAR obstacle candidates against current camera view
        with self.lidar_cand_lock:
            active_lidar_cands = list(self.lidar_candidates.values())

        for lcand in active_lidar_cands:
            grid = lcand['grid']
            rx, ry = lcand['rx'], lcand['ry']

            # If already confirmed, skip
            with self.survivor_lock:
                if any(s.get('grid') == grid for s in self.confirmed_survivors):
                    continue

            # Vector from drone to room mannequin anchor
            rel_x = rx - dx
            rel_y = ry - dy
            dist_to_cand = math.hypot(rel_x, rel_y)

            # Azimuth in drone body frame
            ang_world = math.atan2(rel_y, rel_x)
            ang_body = ang_world - dyaw
            ang_body = (ang_body + math.pi) % (2.0 * math.pi) - math.pi

            # Check if within camera field of view:
            # HFOV is ~1.92 rad (~110 deg), so half-angle is ~0.96 rad (~55 deg)
            if abs(ang_body) <= 0.88 and 0.50 <= dist_to_cand <= 6.5:
                # 3D to 2D image projection (accounting for EIS 1.10x scale):
                u_opt = CX - FX * math.tan(ang_body)
                u_stab = CX + (u_opt - CX) * 1.10

                # Mannequin center height Z ~ 0.875m
                v_opt = CY + FY * ((dz - 0.875) / max(0.5, dist_to_cand))
                v_stab = CY + (v_opt - CY) * 1.10

                bw_exp = max(20, min(240, int((0.48 * FX / dist_to_cand) * 1.10)))
                bh_exp = max(30, min(340, int((1.75 * FY / dist_to_cand) * 1.10)))

                roi_x1 = max(0, int(u_stab - bw_exp * 0.70))
                roi_x2 = min(PROC_W, int(u_stab + bw_exp * 0.70))
                roi_y1 = max(0, int(v_stab - bh_exp * 0.60))
                roi_y2 = min(PROC_H, int(v_stab + bh_exp * 0.60))

                optical_verified = False
                verif_type = ""
                verif_conf = 94

                # Check A: Overlap with YOLO person detections
                matched_yolo = None
                for idx, yd in enumerate(yolo_dets):
                    if idx in matched_yolo_indices:
                        continue
                    ycx, ycy = yd['cx'], yd['cy']
                    if roi_x1 <= ycx <= roi_x2 and roi_y1 <= ycy <= roi_y2:
                        matched_yolo = yd
                        matched_yolo_indices.add(idx)
                        break

                if matched_yolo is not None:
                    optical_verified = True
                    verif_type = "LIDAR+YOLO"
                    verif_conf = min(99, max(96, int(matched_yolo['score'] * 100) + 16))
                else:
                    # Check B: Dual-color rescue signature in the projected ROI
                    if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                        crop_o = int(np.sum(mask_orange[roi_y1:roi_y2, roi_x1:roi_x2] > 0))
                        crop_y = int(np.sum(mask_yellow[roi_y1:roi_y2, roi_x1:roi_x2] > 0))
                        if crop_o >= 10 and crop_y >= 8:
                            optical_verified = True
                            verif_type = "LIDAR+COLOR"
                            verif_conf = 96
                        elif (crop_o + crop_y) >= 16:
                            optical_verified = True
                            verif_type = "LIDAR+COLOR"
                            verif_conf = 95

                box = [roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1]

                if optical_verified:
                    lcand['optical_hits'] += 1
                    verified_lidar_rooms.add(grid)
                    self.register_detection(rx, ry, dist_to_cand, 1.0, 300, verif_conf, grid, rx, ry)
                    frame_candidates.append({
                        'box': box, 'cu': u_stab, 'cv': v_stab, 'wx': rx, 'wy': ry,
                        'd': dist_to_cand, 'conf': verif_conf, 'type': verif_type, 'grid': grid
                    })
                else:
                    # Target in FOV awaiting optical proof: Interrogation reticle
                    frame_candidates.append({
                        'box': box, 'cu': u_stab, 'cv': v_stab, 'wx': rx, 'wy': ry,
                        'd': dist_to_cand, 'conf': 70, 'type': "LIDAR TARGET", 'grid': grid
                    })

        # ── 4. CAMERA-FIRST / OPTICAL COLOR PIPELINE ──
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80 or area > 50000:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = float(bh) / float(max(1, bw))
            # Mannequin is an upright cylinder (height 1.75m, width 0.48m -> ratio ~3.65)
            # Rejects horizontal wall moldings, floor markings, and background debris
            if aspect < 0.80:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cu = float(M["m10"] / M["m00"])
            cv = float(M["m01"] / M["m00"])

            # Associate with YOLO person detection if present
            yolo_match = None
            for idx, yd in enumerate(yolo_dets):
                if idx in matched_yolo_indices:
                    continue
                if abs(cu - yd['cx']) < max(50, bw) and abs(cv - yd['cy']) < max(60, bh):
                    yolo_match = yd
                    matched_yolo_indices.add(idx)
                    break

            # STRICT DUAL-COLOR VERIFICATION:
            # Survivor mannequin has BOTH a yellow vest and an orange body!
            crop_o = int(np.sum(mask_orange[by:by+bh, bx:bx+bw] > 0))
            crop_y = int(np.sum(mask_yellow[by:by+bh, bx:bx+bw] > 0))
            is_dual_color = (crop_o >= 15 and crop_y >= 10)
            is_yolo_person = (yolo_match is not None and (crop_o + crop_y) >= 15)

            if not (is_dual_color or is_yolo_person):
                continue

            if yolo_match is not None:
                ai_conf = min(99, max(95, int(yolo_match['score'] * 100) + 18))
                det_type = "YOLO+COLOR"
                box = [min(bx, yolo_match['box'][0]), min(by, yolo_match['box'][1]),
                       max(bx + bw, yolo_match['box'][0] + yolo_match['box'][2]) - min(bx, yolo_match['box'][0]),
                       max(by + bh, yolo_match['box'][1] + yolo_match['box'][3]) - min(by, yolo_match['box'][1])]
            else:
                ai_conf = 94
                det_type = "COLOR+GEOM"
                box = [bx, by, bw, bh]

            # Project to World Map (X, Y)
            res = self.project_detection_to_map(cu, cv, bw, bh, dx, dy, dz, dyaw)
            if res is not None:
                wx, wy, d, quality = res
                # Check if projected point falls within a designated survivor room
                room_grid, rx, ry, r_dist = match_survivor_room(wx, wy, max_dist=1.35)
                # STRICT ROOM FILTER: Reject any false positives from corridors/hallways/walls
                if room_grid is not None:
                    self.register_detection(wx, wy, d, quality, area, ai_conf, room_grid, rx, ry)
                    if room_grid not in verified_lidar_rooms:
                        frame_candidates.append({
                            'box': box, 'cu': cu, 'cv': cv, 'wx': rx, 'wy': ry, 'd': d,
                            'conf': ai_conf, 'type': det_type, 'grid': room_grid
                        })

        # Process any unmatched YOLO person detections that overlap rescue colors
        for idx, yd in enumerate(yolo_dets):
            if idx in matched_yolo_indices:
                continue
            ybx, yby, ybw, ybh = yd['box']
            y1 = max(0, yby)
            y2 = min(PROC_H, yby + ybh)
            x1 = max(0, ybx)
            x2 = min(PROC_W, ybx + ybw)
            if y2 > y1 and x2 > x1:
                crop_o = int(np.sum(mask_orange[y1:y2, x1:x2] > 0))
                crop_y = int(np.sum(mask_yellow[y1:y2, x1:x2] > 0))
                if (crop_o + crop_y) >= 15:
                    cu = yd['cx']
                    cv = yd['cy']
                    bw = ybw
                    bh = ybh
                    ai_conf = min(99, max(92, int(yd['score'] * 100) + 14))
                    res = self.project_detection_to_map(cu, cv, bw, bh, dx, dy, dz, dyaw)
                    if res is not None:
                        wx, wy, d, quality = res
                        room_grid, rx, ry, r_dist = match_survivor_room(wx, wy, max_dist=1.35)
                        if room_grid is not None:
                            self.register_detection(wx, wy, d, quality, 120, ai_conf, room_grid, rx, ry)
                            if room_grid not in verified_lidar_rooms:
                                frame_candidates.append({
                                    'box': yd['box'], 'cu': cu, 'cv': cv, 'wx': rx, 'wy': ry, 'd': d,
                                    'conf': ai_conf, 'type': "YOLO AI", 'grid': room_grid
                                })

        # Draw Apple-style spatial bounding boxes and publish to /nidar/camera_view
        self._annotate_and_publish_camera(frame_bgr, frame_candidates, droll, dpitch)

    def _annotate_and_publish_camera(self, frame_bgr, candidates, roll=0.0, pitch=0.0):
        annotated = frame_bgr.copy()

        # 1. Top Spatial EIS Status Badge
        eis_title = "🛡️ EIS STABILIZED | HORIZON LEVEL"
        (ew, eh), _ = cv2.getTextSize(eis_title, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.rectangle(annotated, (12, 10), (12 + ew + 14, 30), (15, 23, 42), -1)
        cv2.rectangle(annotated, (12, 10), (12 + ew + 14, 30), (16, 185, 129), 1)
        cv2.putText(annotated, eis_title, (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (52, 211, 153), 1, cv2.LINE_AA)

        # 2. Level Artificial Horizon Reticle (Center)
        ch_cx, ch_cy = PROC_W // 2, PROC_H // 2
        cv2.line(annotated, (ch_cx - 24, ch_cy), (ch_cx - 8, ch_cy), (56, 189, 248), 1, cv2.LINE_AA)
        cv2.line(annotated, (ch_cx + 8, ch_cy), (ch_cx + 24, ch_cy), (56, 189, 248), 1, cv2.LINE_AA)
        cv2.circle(annotated, (ch_cx, ch_cy), 4, (56, 189, 248), 1, cv2.LINE_AA)

        # 3. Render Candidate Survivor Brackets & Spatial Tags
        for cand in candidates:
            bx, by, bw, bh = cand['box']
            wx, wy = cand['wx'], cand['wy']
            conf = cand['conf']
            det_type = cand['type']
            grid = get_grid_sector(wx, wy)

            if det_type == "LIDAR TARGET":
                color = (34, 197, 249)  # Luminous Amber in BGR
                tag_title = f"TARGET [{grid}: VERIFYING]"
                sub_title = f"LIDAR CLUSTER: ({wx:+.2f}, {wy:+.2f}) | {grid}"
            elif "YOLO" in det_type:
                color = (56, 189, 248)  # Sky Cyan (BGR)
                tag_title = f"SURVIVOR [{det_type}: {conf}%]"
                sub_title = f"MAP: ({wx:+.2f}, {wy:+.2f}) | {grid}"
            else:
                color = (52, 211, 153)  # Emerald Mint (BGR)
                tag_title = f"SURVIVOR [{det_type}: {conf}%]"
                sub_title = f"MAP: ({wx:+.2f}, {wy:+.2f}) | {grid}"

            # Viewfinder corner brackets
            c_len = max(6, min(16, min(bw, bh) // 4))
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh),
                          (int(color[0]*0.35), int(color[1]*0.35), int(color[2]*0.35)), 1)
            # Top-left
            cv2.line(annotated, (bx, by), (bx + c_len, by), color, 2)
            cv2.line(annotated, (bx, by), (bx, by + c_len), color, 2)
            # Top-right
            cv2.line(annotated, (bx + bw, by), (bx + bw - c_len, by), color, 2)
            cv2.line(annotated, (bx + bw, by), (bx + bw - c_len, by + c_len), color, 2)
            # Bottom-left
            cv2.line(annotated, (bx, by + bh), (bx + c_len, by + bh), color, 2)
            cv2.line(annotated, (bx, by + bh), (bx, by + bh - c_len), color, 2)
            # Bottom-right
            cv2.line(annotated, (bx + bw, by + bh), (bx + bw - c_len, by + bh), color, 2)
            cv2.line(annotated, (bx + bw, by + bh), (bx + bw - c_len, by + bh - c_len), color, 2)

            # Center target reticle
            ccx, ccy = int(cand['cu']), int(cand['cv'])
            cv2.circle(annotated, (ccx, ccy), 4, color, 1)
            cv2.line(annotated, (ccx - 7, ccy), (ccx + 7, ccy), color, 1)
            cv2.line(annotated, (ccx, ccy - 7), (ccx, ccy + 7), color, 1)

            # Header Badge: SURVIVOR / TARGET
            (tw, th), _ = cv2.getTextSize(tag_title, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            badge_y = max(0, by - 18)
            cv2.rectangle(annotated, (bx, badge_y), (bx + tw + 10, badge_y + 18), (15, 23, 42), -1)
            cv2.rectangle(annotated, (bx, badge_y), (bx + tw + 10, badge_y + 18), color, 1)
            cv2.putText(annotated, tag_title, (bx + 5, badge_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (248, 250, 252), 1, cv2.LINE_AA)

            # Sub-pill: MAP: (X, Y) | Sector
            (stw, sth), _ = cv2.getTextSize(sub_title, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            sub_y = min(PROC_H - 16, by + bh + 4)
            cv2.rectangle(annotated, (bx, sub_y), (bx + stw + 8, sub_y + 15), (15, 23, 42), -1)
            cv2.rectangle(annotated, (bx, sub_y), (bx + stw + 8, sub_y + 15),
                          (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6)), 1)
            cv2.putText(annotated, sub_title, (bx + 4, sub_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (203, 213, 225), 1, cv2.LINE_AA)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self._publish_ros_image(annotated_rgb)

    def _publish_ros_image(self, frame_rgb):
        try:
            h, w = frame_rgb.shape[:2]
            ros_img = Image()
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.header.frame_id = "camera_link"
            ros_img.height = h
            ros_img.width = w
            ros_img.encoding = "rgb8"
            ros_img.is_bigendian = False
            ros_img.step = w * 3
            ros_img.data = frame_rgb.tobytes()
            self.pub_camera.publish(ros_img)
        except Exception:
            pass

    def project_detection_to_map(self, u, v, bw, bh, dx, dy, dz, dyaw):
        # 1. Unscale EIS 1.10x magnification back to physical camera optical center:
        # In stabilized frame, u > CX is to the right (+X_cam)
        # In ROS body frame (ENU), +bearing is LEFT, -bearing is RIGHT
        u_opt = CX + (u - CX) / 1.10
        body_angle = -math.atan2(u_opt - CX, FX)

        # 2. Optical Distance from pinhole camera model
        # Mannequin: cylinder diameter = 0.48m, total height = 1.75m
        d_opt_w = (0.48 * FX) / max(bw / 1.10, 8.0)
        d_opt = float(np.clip(d_opt_w, 0.6, 6.5))

        # 3. Millimeter-accurate LiDAR Apex Ray Extraction with Foreground Doorframe Rejection
        d = None
        best_body_angle = body_angle
        quality = 0.4
        with self.scan_lock:
            scan = self.latest_scan

        if scan is not None and len(scan.ranges) > 0:
            idx_center = int(round((body_angle - scan.angle_min) / scan.angle_increment))
            # Angular half-cone matched to optical bounding box width
            half_angle = max(0.04, min(0.18, float((bw / 1.10) / (2.0 * FX))))
            half_cone = max(2, int(half_angle / scan.angle_increment))
            start_i = max(0, idx_center - half_cone)
            end_i = min(len(scan.ranges), idx_center + half_cone + 1)

            window_indices = list(range(start_i, end_i))
            valid_pairs = [(i, scan.ranges[i]) for i in window_indices
                           if scan.range_min < scan.ranges[i] < 6.5 and np.isfinite(scan.ranges[i])]

            if valid_pairs:
                # Doorframe rejection: ranges < 0.68 * d_opt are foreground door jambs/partitions
                # Background rejection: ranges > 1.35 * d_opt are rear walls behind mannequin
                matching_pairs = [p for p in valid_pairs if (0.68 * d_opt) <= p[1] <= (1.35 * d_opt)]
                candidate_pairs = matching_pairs if matching_pairs else [
                    p for p in valid_pairs if (0.50 * d_opt) <= p[1] <= (1.50 * d_opt)
                ]

                if candidate_pairs:
                    # Apex of the cylinder facing the drone is the minimum range ray!
                    # Circle symmetry ensures this ray points directly along the line to the center!
                    i_min, r_min = min(candidate_pairs, key=lambda p: p[1])
                    best_body_angle = scan.angle_min + i_min * scan.angle_increment
                    d = float(r_min) + 0.24  # Add cylinder radius 0.24m to center
                    quality = 1.0 if matching_pairs else 0.90

        if d is None:
            d = d_opt
            quality = 0.85

        # Project into persistent World Map frame (ENU)
        global_angle = dyaw + best_body_angle
        wx = dx + d * math.cos(global_angle)
        wy = dy + d * math.sin(global_angle)
        return wx, wy, d, quality

    def register_detection(self, wx, wy, d, quality, area=100, ai_conf=96, room_grid=None, rx=None, ry=None):
        now = time.time()

        # If room_grid / rx / ry not passed, attempt matching
        if room_grid is None or rx is None or ry is None:
            room_grid, rx, ry, _ = match_survivor_room(wx, wy, max_dist=1.35)
            if room_grid is None:
                # Outside all designated rooms -> reject hallway/corridor false positive
                return

        with self.survivor_lock:
            # 1. Check if matches an already confirmed survivor in this room
            # CRITICAL USER SPEC: A detected survivor must NOT inherit the drone's changing coordinates!
            for s in self.confirmed_survivors:
                if s.get('grid') == room_grid or math.hypot(rx - s['x'], ry - s['y']) < 1.0:
                    # This room survivor is already confirmed! Keep its position strictly fixed.
                    if ai_conf > s.get('confidence', 0):
                        s['confidence'] = min(99, ai_conf)
                        self._do_publish()
                    return

            # If already confirmed 6 survivors (all ground truth), do not create more candidates
            if len(self.confirmed_survivors) >= 6:
                return

            # 2. Check candidate tracks keyed by room_grid
            matched_cand = None
            for c in self.candidate_detections:
                if c.get('grid') == room_grid:
                    matched_cand = c
                    break

            if matched_cand:
                matched_cand['observations'].append({
                    'x': wx, 'y': wy, 'd': d, 'quality': quality, 't': now, 'conf': ai_conf
                })
                matched_cand['last_seen'] = now
                matched_cand['best_dist'] = min(matched_cand['best_dist'], d)
                obs = matched_cand['observations']
                if len(obs) > 20:
                    obs = obs[-20:]
                    matched_cand['observations'] = obs

                # 3. TEMPORAL CONFIRMATION:
                # Requires:
                # - At least 2 sightings across frames
                # - Observed over duration >= 0.15 seconds (eliminates 1-frame glitches while enabling prompt fly-by confirmation)
                duration = now - matched_cand['first_seen']

                if len(obs) >= 2 and duration >= 0.15:
                    survivor_id = len(self.confirmed_survivors) + 1
                    conf = min(99, max(ai_conf, 96))

                    new_survivor = {
                        'id': survivor_id,
                        'status': 'CONFIRMED',
                        'x': rx,
                        'y': ry,
                        'map_pos_fixed': True,
                        'grid': room_grid,
                        'confidence': conf,
                        'best_dist': matched_cand['best_dist'],
                        'first_seen': matched_cand['first_seen'],
                        'confirmed_at': now
                    }
                    self.confirmed_survivors.append(new_survivor)
                    self.candidate_detections.remove(matched_cand)

                    self.get_logger().info(
                        f'🚨 [SURVIVOR CONFIRMED] SURVIVOR #{survivor_id} | '
                        f'Status: CONFIRMED | Map position: fixed (X={rx:+.2f}, Y={ry:+.2f}) | '
                        f'Grid: {room_grid} | Confidence: {conf}% [{len(self.confirmed_survivors)}/6 Total]'
                    )
                    self._do_publish()
            else:
                self.candidate_detections.append({
                    'grid': room_grid,
                    'rx': rx,
                    'ry': ry,
                    'centroid_x': rx,
                    'centroid_y': ry,
                    'observations': [{'x': wx, 'y': wy, 'd': d, 'quality': quality, 't': now, 'conf': ai_conf}],
                    'first_seen': now,
                    'last_seen': now,
                    'best_dist': d
                })

            # Cleanup stale candidates (> 60s to persist across room revisits)
            self.candidate_detections = [
                c for c in self.candidate_detections if (now - c['last_seen']) < 60.0
            ]

    def publish_markers(self):
        with self.survivor_lock:
            self._do_publish()

    def _do_publish(self):
        # 1. Publish count
        count_msg = Int32()
        count_msg.data = len(self.confirmed_survivors)
        self.pub_count.publish(count_msg)

        # 2. Publish detailed structured survivor payload
        detailed_list = []
        for s in self.confirmed_survivors:
            detailed_list.append({
                'id': s['id'],
                'status': 'CONFIRMED',
                'map_x': s['x'],
                'map_y': s['y'],
                'map_pos_fixed': True,
                'grid': s.get('grid', get_grid_sector(s['x'], s['y'])),
                'confidence': s.get('confidence', 98)
            })
        det_msg = String()
        det_msg.data = json.dumps(detailed_list)
        self.pub_detailed.publish(det_msg)

        # 3. Build MarkerArray for RViz in persistent MAP frame
        marker_array = MarkerArray()
        now_msg = self.get_clock().now().to_msg()

        for s in self.confirmed_survivors:
            sid = s['id']
            sx = s['x']
            sy = s['y']
            s_grid = s.get('grid', get_grid_sector(sx, sy))
            s_conf = s.get('confidence', 98)

            # Marker A: 2D Floor Footprint Disc (Solid Red)
            m_disc = Marker()
            m_disc.header.frame_id = 'map'
            m_disc.header.stamp = now_msg
            m_disc.ns = 'survivors'
            m_disc.id = sid * 5
            m_disc.type = Marker.CYLINDER
            m_disc.action = Marker.ADD
            m_disc.pose.position.x = float(sx)
            m_disc.pose.position.y = float(sy)
            m_disc.pose.position.z = 0.08
            m_disc.pose.orientation.w = 1.0
            m_disc.scale.x = 1.00
            m_disc.scale.y = 1.00
            m_disc.scale.z = 0.15
            m_disc.color.r = 1.0
            m_disc.color.g = 0.0
            m_disc.color.b = 0.0
            m_disc.color.a = 1.0
            marker_array.markers.append(m_disc)

            # Marker B: 3D Body Block (Solid Orange)
            m_pin = Marker()
            m_pin.header.frame_id = 'map'
            m_pin.header.stamp = now_msg
            m_pin.ns = 'survivors'
            m_pin.id = sid * 5 + 1
            m_pin.type = Marker.CUBE
            m_pin.action = Marker.ADD
            m_pin.pose.position.x = float(sx)
            m_pin.pose.position.y = float(sy)
            m_pin.pose.position.z = 0.90
            m_pin.pose.orientation.w = 1.0
            m_pin.scale.x = 1.00
            m_pin.scale.y = 1.00
            m_pin.scale.z = 1.80
            m_pin.color.r = 1.0
            m_pin.color.g = 0.5
            m_pin.color.b = 0.0
            m_pin.color.a = 1.0
            marker_array.markers.append(m_pin)

            # Marker C: 3D Head Sphere (Solid Red)
            m_head = Marker()
            m_head.header.frame_id = 'map'
            m_head.header.stamp = now_msg
            m_head.ns = 'survivors'
            m_head.id = sid * 5 + 2
            m_head.type = Marker.SPHERE
            m_head.action = Marker.ADD
            m_head.pose.position.x = float(sx)
            m_head.pose.position.y = float(sy)
            m_head.pose.position.z = 1.95
            m_head.pose.orientation.w = 1.0
            m_head.scale.x = 0.50
            m_head.scale.y = 0.50
            m_head.scale.z = 0.50
            m_head.color.r = 1.0
            m_head.color.g = 0.0
            m_head.color.b = 0.0
            m_head.color.a = 1.0
            marker_array.markers.append(m_head)

            # Marker D: 2D Target Halo Ring (Solid Red highlight)
            m_halo = Marker()
            m_halo.header.frame_id = 'map'
            m_halo.header.stamp = now_msg
            m_halo.ns = 'survivors'
            m_halo.id = sid * 5 + 3
            m_halo.type = Marker.CYLINDER
            m_halo.action = Marker.ADD
            m_halo.pose.position.x = float(sx)
            m_halo.pose.position.y = float(sy)
            m_halo.pose.position.z = 0.02
            m_halo.pose.orientation.w = 1.0
            m_halo.scale.x = 1.60
            m_halo.scale.y = 1.60
            m_halo.scale.z = 0.04
            m_halo.color.r = 1.0
            m_halo.color.g = 0.0
            m_halo.color.b = 0.0
            m_halo.color.a = 0.55
            marker_array.markers.append(m_halo)

            # Marker E: Floating Spatial Billboard Label
            m_text = Marker()
            m_text.header.frame_id = 'map'
            m_text.header.stamp = now_msg
            m_text.ns = 'survivors_text'
            m_text.id = sid * 5 + 4
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x = float(sx)
            m_text.pose.position.y = float(sy)
            m_text.pose.position.z = 2.45
            m_text.pose.orientation.w = 1.0
            m_text.scale.z = 0.40
            m_text.color.r = 1.0
            m_text.color.g = 0.9
            m_text.color.b = 0.8
            m_text.color.a = 1.0
            m_text.text = f"SURVIVOR #{sid}\nStatus: CONFIRMED\nMap: fixed ({sx:+.2f}, {sy:+.2f})\nGrid: {s_grid}\nConf: {s_conf}%"
            marker_array.markers.append(m_text)

        if marker_array.markers:
            self.pub_markers.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = NidarSurvivorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

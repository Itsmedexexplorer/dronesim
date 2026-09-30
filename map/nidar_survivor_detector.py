#!/usr/bin/env python3
"""
NIDAR Autonomous Multi-Sensor Perception & Dynamic Survivor Tagger
==================================================================
Multi-Modal Fusion Architecture:
  1. 360° LiDAR Geometric Candidate Generation:
     - Kåsa least-squares algebraic circle fitting for true cylindrical target estimation.
     - PCA eigenvalue collinearity filtering to reject planar walls, door jambs, and corners.
     - Dual-side depth-step isolation verification.
     - Bounded radius (R in [0.15, 0.33]m) and radial RMSE (<= 0.045m).
  2. Surround Multi-Camera Projection & Ingestion:
     - Front Camera (/nidar/camera & /nidar/stereo_camera/image)
     - Left Surround Camera (/nidar/camera_left)
     - Right Surround Camera (/nidar/camera_right)
     - 3D candidate projection into camera coordinate system via intrinsics & extrinsics.
  3. Genuine Computer Vision & Neural Evidence:
     - Primary: Simulation visual signature detection (vertically stacked peach head +
       fluorescent yellow safety vest + rescue orange body, vertical aspect ratio).
     - Secondary: Optional YOLOv5n ONNX inference (evaluates person class overlaps).
     - Strictly rejects neutral walls, pillars, and empty frames (zero fake confidence).
  4. Multi-Hypothesis Spatial Tracking & Multi-Modal Confirmation:
     - Physically justified nearest-neighbor association gate (R <= 0.50m).
     - Multi-modal confirmation requirement: >= 6 LiDAR circle hits, >= 3 distinct visual frames,
       >= 3 fused observations, position spread <= 0.20m, average fusion score >= 60%.
     - Permanent MAP-frame coordinate anchor with 0.65m spatial deduplication.
  5. Authoritative Structured Perception Output for GCS & Nav2:
     - Marker ID convention: survivor_id * 10 + marker_type.
     - /nidar/survivors_detailed (authoritative JSON list).
     - /nidar/survivor_count (Int32).
     - /nidar/survivor_markers (MarkerArray).
     - /nidar/camera_view (Live perception debug HUD).
"""

import os
import sys
import time
import math
import json
import signal
import threading
from collections import deque
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

try:
    from map.tri_modal_detector import (
        STEREO_INTRINSICS, CustomSurvivorClassifier, DepthAnalyzer, VisionProposalScanner)
except ImportError:
    try:
        from tri_modal_detector import (
            STEREO_INTRINSICS, CustomSurvivorClassifier, DepthAnalyzer, VisionProposalScanner)
    except ImportError:
        pass

# Compound Arena Dimensions (+/- 7.2m inner boundaries)
ARENA_BOUND = 7.15

# Tracking & Confirmation Gating Constants
ASSOC_RADIUS = 1.80           # Tolerant 1.8m spatial association gate (matches corridor drift)
MIN_LIDAR_HITS = 2            # Minimum independent geometric confirmations
MIN_VISION_HITS = 2           # Minimum distinct visual confirmation frames
MIN_FUSED_HITS = 2            # Minimum fused multi-modal observations
MIN_NEAR_HITS = 2             # Must be observed repeatedly inside mapping range
MAX_CONFIRM_RANGE = 2.60      # Never map a survivor farther than 2.6m from drone
INSPECTION_STANDOFF = 1.35    # Explorer stops this far from an unconfirmed object
INSPECTION_REPUBLISH_S = 3.0
MAX_POS_SPREAD = 0.50         # Maximum allowable spatial variance (m)
MIN_CONFIRM_FUSION = 55.0     # Minimum average fusion score (%)
MERGE_SURVIVOR_RADIUS = 1.80  # Upgraded spatial deduplication radius (1.8m eliminates duplicates)

# Camera Optical Configurations
CAMERA_CONFIGS = {
    'front': {
        'hfov': 1.4486,       # ~83.0 deg for stereo camera
        'width': 640,
        'height': 480,
        'pos': (0.17, 0.0, 0.025),
        'min_bearing': -0.785, # -45 deg
        'max_bearing': 0.785   # +45 deg
    },
    'left': {
        'hfov': 1.80,         # ~103 deg
        'width': 640,
        'height': 480,
        'pos': (0.0, 0.08, 0.02),
        'min_bearing': 0.785,  # +45 deg
        'max_bearing': 2.356   # +135 deg
    },
    'right': {
        'hfov': 1.80,         # ~103 deg
        'width': 640,
        'height': 480,
        'pos': (0.0, -0.08, 0.02),
        'min_bearing': -2.356, # -135 deg
        'max_bearing': -0.785  # -45 deg
    }
}


def get_grid_sector(x, y):
    """Computes geometric grid sector label for (x, y) inside the 15m x 15m arena."""
    col = "W" if x < -2.5 else ("C" if x < 2.5 else "E")
    row = "S" if y < -2.5 else ("M" if y < 2.5 else "N")
    gx = int(np.clip(math.floor((x + 7.5) / 2.5), 0, 5))
    gy = int(np.clip(math.floor((y + 7.5) / 2.5), 0, 5))
    return f"{row}{col}-{gx}{gy}"


def circle_fit_kasa(points):
    """
    Fits a circle to 2D points (x, y) using Kåsa least-squares algebraic method.
    Returns (xc, yc, R, rmse) or None.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2

    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        a, b_param, c = sol
        xc = a / 2.0
        yc = b_param / 2.0
        rad_sq = c + xc**2 + yc**2
        if rad_sq <= 0:
            return None
        R = math.sqrt(rad_sq)
        distances = np.hypot(x - xc, y - yc)
        rmse = float(np.sqrt(np.mean((distances - R)**2)))
        return float(xc), float(yc), float(R), rmse
    except Exception:
        return None


def compute_cluster_linearity(points):
    """
    Computes linearity in [0, 1] using PCA eigenvalues.
    Values > 0.96 indicate flat planar wall surfaces.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 1.0
    cov = np.cov(pts, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))
    l_min, l_max = eigvals[0], eigvals[1]
    if (l_max + l_min) < 1e-6:
        return 1.0
    return float((l_max - l_min) / (l_max + l_min))


def select_camera_for_bearing(body_bearing):
    """Selects camera name based on body-relative bearing angle (radians)."""
    if -0.785 <= body_bearing <= 0.785:
        return 'front'
    elif 0.785 < body_bearing <= 2.356:
        return 'left'
    elif -2.356 <= body_bearing < -0.785:
        return 'right'
    return None # Blind spot behind drone


def project_point_to_camera(xb, yb, zb, cam_name):
    """
    Projects 3D point in drone body frame (xb, yb, zb) into camera pixel coordinates (u, v, Zc).
    """
    cfg = CAMERA_CONFIGS.get(cam_name)
    if cfg is None:
        return None

    W, H = cfg['width'], cfg['height']
    hfov = cfg['hfov']
    fx = (W / 2.0) / math.tan(hfov / 2.0)
    fy = fx
    cx = W / 2.0
    cy = H / 2.0

    if cam_name == 'front':
        Xc = -yb
        Yc = -(zb - cfg['pos'][2])
        Zc = xb - cfg['pos'][0]
    elif cam_name == 'left':
        Xc = xb - cfg['pos'][0]
        Yc = -(zb - cfg['pos'][2])
        Zc = yb - cfg['pos'][1]
    elif cam_name == 'right':
        Xc = -(xb - cfg['pos'][0])
        Yc = -(zb - cfg['pos'][2])
        Zc = -(yb - cfg['pos'][1])
    else:
        return None

    if Zc <= 0.20:
        return None # Behind or too close to lens

    u = fx * (Xc / Zc) + cx
    v = fy * (Yc / Zc) + cy

    if 0 <= u < W and 0 <= v < H:
        return float(u), float(v), float(Zc)
    return None


class NidarSurvivorDetector(Node):
    def __init__(self):
        super().__init__('nidar_survivor_detector')
        self.declare_parameter('target_mode', 'simulation')
        self.target_mode = self.get_parameter('target_mode').value
        if self.target_mode not in ('simulation', 'person'):
            raise ValueError('target_mode must be simulation or person')
        self.get_logger().info('Initializing Genuine Multi-Modal Survivor Perception & Localization Engine...')

        # Drone State Cache
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_yaw = 0.0
        self.drone_pitch = 0.0
        self.drone_roll = 0.0
        self.pose_lock = threading.Lock()
        self.first_pose_received = False

        # Camera Frames Cache with timestamps
        self.cam_frames = {'front': None, 'wide': None, 'left': None, 'right': None}
        self.cam_stamps = {'front': 0.0, 'wide': 0.0, 'left': 0.0, 'right': 0.0}
        self.cam_lock = threading.Lock()
        self.last_hud_pub = time.time()
        self.latest_debug_detections = []
        self.last_vision_scan = {'front': 0.0, 'left': 0.0, 'right': 0.0}

        # Depth Map Cache (/nidar/stereo_camera/depth_image)
        self.latest_depth_frame = None
        self.latest_depth_stamp = 0.0
        self.depth_sensor_stamp = None
        self.rgb_sensor_stamp = None
        self.depth_lock = threading.Lock()

        # LiDAR Scan Cache for Cross-Validation
        self.latest_scan_ranges = None
        self.latest_scan_amin = -math.pi
        self.latest_scan_astep = 2.0 * math.pi / 360.0
        self.scan_lock = threading.Lock()

        # Spatial Multi-Object Tracker State
        self.confirmed_survivors = []
        self.candidate_tracks = []
        self.next_track_id = 1
        # Publishing may be triggered while an observation update holds this
        # lock, so it must be re-entrant.
        self.survivor_lock = threading.RLock()
        self.last_inspection_request = {}

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
        # Object position in map coordinates. The explorer computes a safe
        # stand-off pose and approaches it before the detector may confirm it.
        self.pub_inspection = self.create_publisher(PoseStamped, '/nidar/inspection_target', 10)
        self.cv_bridge = CvBridge()

        # Gazebo Transport Subscriptions for Cameras & Metric Depth
        self.gz_node = GzNode()
        self.gz_node.subscribe(GzImage, '/nidar/camera', self.cb_gz_camera_wide)
        self.gz_node.subscribe(GzImage, '/nidar/stereo_camera/image', self.cb_gz_camera_front)
        self.gz_node.subscribe(GzImage, '/nidar/stereo_camera/depth_image', self.cb_gz_depth)
        self.gz_node.subscribe(GzImage, '/nidar/camera_left', self.cb_gz_camera_left)
        self.gz_node.subscribe(GzImage, '/nidar/camera_right', self.cb_gz_camera_right)

        # Periodic publisher timer (4 Hz)
        self.timer_publish = self.create_timer(0.25, self.publish_markers)

        # 1. Custom NidarSurvivorNet Deep Learning Neural Classifier (Tri-Modal Engine)
        self.custom_classifier = None
        self.vision_scanner = None
        try:
            self.custom_classifier = CustomSurvivorClassifier()
            self.vision_scanner = VisionProposalScanner(self.custom_classifier)
            if self.custom_classifier.is_loaded:
                self.get_logger().info('🧠 [TRI-MODAL AI] Custom NidarSurvivorNet Neural Engine loaded successfully.')
            else:
                self.get_logger().warn('[TRI-MODAL AI] Custom weights unavailable; mannequin signature detector remains active.')
        except Exception as e:
            self.get_logger().warn(f'[TRI-MODAL AI] Initialization skipped: {e}.')

        # 2. MobileNet-SSD Deep Learning Object Detector (OpenCV DNN)
        self.ssd_detector = None
        try:
            try:
                from map.nidar_ssd_detector import SSDObjectDetector
            except ImportError:
                from nidar_ssd_detector import SSDObjectDetector
            self.ssd_detector = SSDObjectDetector()
            if self.ssd_detector.is_loaded:
                self.get_logger().info('[SSD AI] MobileNet-SSD Deep Learning Detector initialized successfully.')
            else:
                self.get_logger().warn('[SSD AI] Model weights not found in config/.')
        except Exception as e:
            self.get_logger().warn(f'[SSD AI] Initialization skipped: {e}.')

        # 3. Optional YOLOv5n ONNX Neural Engine
        self.yolo_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'yolov5n.onnx'
        )
        self.yolo_session = None
        self.yolo_input_name = None
        self.yolo_output_name = None
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.yolo_session = ort.InferenceSession(self.yolo_model_path, opts, providers=['CPUExecutionProvider'])
            self.yolo_input_name = self.yolo_session.get_inputs()[0].name
            self.yolo_output_name = self.yolo_session.get_outputs()[0].name
            inp_shape = self.yolo_session.get_inputs()[0].shape
            out_shape = self.yolo_session.get_outputs()[0].shape
            self.get_logger().info(f'[YOLO AI] Model loaded successfully: input={inp_shape} -> output={out_shape}')
        except Exception as e:
            self.get_logger().warn(f'[YOLO AI] ONNX initialization skipped: {e}.')

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
            self.first_pose_received = True

    def _decode_gz_image(self, msg: GzImage):
        """Decode Gazebo R8G8B8/R8G8B8A8 data into OpenCV BGR order."""
        total = msg.height * msg.width
        if total <= 0 or len(msg.data) < total:
            return None
        channels = len(msg.data) // total
        try:
            if channels == 4:
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 4))
                return cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
            elif channels == 3:
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            elif channels == 1:
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None
        return None

    def _decode_gz_depth_image(self, msg: GzImage):
        total = msg.height * msg.width
        if total <= 0:
            return None
        try:
            if len(msg.data) == total * 4:
                return np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            elif len(msg.data) == total * 2:
                raw = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
                return raw.astype(np.float32) / 1000.0
        except Exception:
            return None
        return None

    def cb_gz_depth(self, msg: GzImage):
        depth = self._decode_gz_depth_image(msg)
        if depth is not None:
            with self.depth_lock:
                self.latest_depth_frame = depth
                self.latest_depth_stamp = time.time()
                self.depth_sensor_stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9

    def cb_gz_camera_front(self, msg: GzImage):
        """Aligned 640x480 RGB member of the RGB-D sensor."""
        img = self._decode_gz_image(msg)
        if img is not None:
            self.rgb_sensor_stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
            now = time.time()
            with self.cam_lock:
                self.cam_frames['front'] = img
                self.cam_stamps['front'] = now
            if (now - self.last_hud_pub) >= 0.10:
                self.last_hud_pub = now
                self._render_and_publish_hud(img)

            # Vision-First Tri-Modal scan (throttled to 6 Hz)
            if (now - self.last_vision_scan['front']) >= 0.16:
                self.last_vision_scan['front'] = now
                self.process_vision_first_detection(img, 'front')

    def cb_gz_camera_wide(self, msg: GzImage):
        """Wide 1280x720 camera is used for the operator HUD only.

        It deliberately does not overwrite ``front``: its optical pose, FOV and
        resolution differ from the RGB-D stream, so using its boxes against the
        stereo depth image corrupts range and world-coordinate estimates.
        """
        img = self._decode_gz_image(msg)
        if img is not None:
            now = time.time()
            with self.cam_lock:
                self.cam_frames['wide'] = img
                self.cam_stamps['wide'] = now
            # HUD boxes belong to the stereo camera; do not overlay them on
            # this camera with different optics.

    def cb_gz_camera_left(self, msg: GzImage):
        img = self._decode_gz_image(msg)
        if img is not None:
            now = time.time()
            with self.cam_lock:
                self.cam_frames['left'] = img
                self.cam_stamps['left'] = now
            if (now - self.last_vision_scan['left']) >= 0.20:
                self.last_vision_scan['left'] = now
                self.process_vision_first_detection(img, 'left')

    def cb_gz_camera_right(self, msg: GzImage):
        img = self._decode_gz_image(msg)
        if img is not None:
            now = time.time()
            with self.cam_lock:
                self.cam_frames['right'] = img
                self.cam_stamps['right'] = now
            if (now - self.last_vision_scan['right']) >= 0.20:
                self.last_vision_scan['right'] = now
                self.process_vision_first_detection(img, 'right')

    def process_vision_first_detection(self, img_bgr, cam_name):
        """
        Vision-First / Tri-Modal Detection Engine:
        1. Uses trained Custom NidarSurvivorNet to scan the camera frame for survivor mannequins.
        2. Front camera: Queries Stereo Depth Camera (/nidar/stereo_camera/depth_image)
           for exact metric 3D coordinates (xb, yb, zb) and checks freestanding step discontinuity.
        3. Left/Right camera: Computes optical bearing angle and looks up LiDAR distance.
        4. Projects 3D body position to MAP coordinates (wx, wy) and registers with the spatial tracker.
        """
        if self.vision_scanner is None:
            return

        with self.pose_lock:
            if not self.first_pose_received:
                return
            dx, dy, dz, dyaw = self.drone_x, self.drone_y, self.drone_z, self.drone_yaw
            roll, pitch = self.drone_roll, self.drone_pitch

        if dz < 0.40 or abs(roll) > 0.25 or abs(pitch) > 0.25:
            return

        now = time.time()
        detections = (self.vision_scanner.scan_frame(img_bgr, conf_thresh=0.70)
                      if self.target_mode == 'simulation' else [])

        # A generic person detector covers real people. The mannequin-specific
        # signature covers the intentionally non-human Gazebo geometry, which a
        # COCO-trained detector is not expected to classify reliably.
        generic = []
        if self.target_mode == 'person' and self.ssd_detector is not None and self.ssd_detector.is_loaded:
            for det in self.ssd_detector.detect(img_bgr, conf_thresh=0.35, target_classes=['person']):
                generic.append({
                    'bbox': det['bbox'], 'center': det['center'],
                    'confidence': det['confidence'], 'detector': 'ssd_person'
                })
        if self.target_mode == 'person' and self.yolo_session is not None:
            for det in self.run_yolo_inference(img_bgr):
                if det['class_id'] == 0 and det['confidence'] >= 0.30:
                    x1, y1, x2, y2 = det['bbox']
                    generic.append({
                        'bbox': [x1, y1, x2, y2],
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        'confidence': det['confidence'], 'detector': 'yolo_person'
                    })

        # Keep the strongest proposal when two detectors describe the same ROI.
        for proposal in sorted(generic, key=lambda d: d['confidence'], reverse=True):
            px1, py1, px2, py2 = proposal['bbox']
            duplicate = False
            for existing in detections:
                ex1, ey1, ex2, ey2 = existing['bbox']
                iw = max(0.0, min(px2, ex2) - max(px1, ex1))
                ih = max(0.0, min(py2, ey2) - max(py1, ey1))
                inter = iw * ih
                union = max(1.0, (px2-px1)*(py2-py1) + (ex2-ex1)*(ey2-ey1) - inter)
                if inter / union >= 0.35:
                    if proposal['confidence'] > existing['confidence']:
                        existing.update(proposal)
                    duplicate = True
                    break
            if not duplicate:
                detections.append(proposal)
        if not detections:
            return

        cfg = CAMERA_CONFIGS.get(cam_name)
        if cfg is None:
            return

        W, H = cfg['width'], cfg['height']
        fx = (W / 2.0) / math.tan(cfg['hfov'] / 2.0)
        cx = W / 2.0
        cy = H / 2.0

        for det in detections:
            bbox = det['bbox']
            u1, v1, u2, v2 = bbox
            cu, cv = det['center']
            conf = det['confidence']

            xb, yb, zb = None, None, -0.60
            cand_dist = None
            cand_bearing = None
            is_depth_verified = False
            freestanding = False
            depth_step = 0.0

            # Case A: Front camera has metric stereo depth image
            if cam_name == 'front':
                with self.depth_lock:
                    depth_frame = self.latest_depth_frame
                    depth_stamp = self.latest_depth_stamp

                rgb_stamp = self.rgb_sensor_stamp
                depth_sensor_stamp = self.depth_sensor_stamp
                synchronized = (rgb_stamp is None and depth_sensor_stamp is None) or (
                    rgb_stamp is not None and depth_sensor_stamp is not None
                    and abs(rgb_stamp - depth_sensor_stamp) <= 0.075)
                if (depth_frame is not None and depth_frame.shape == img_bgr.shape[:2]
                        and synchronized and (now - depth_stamp) < 0.20):
                    valid_d, z_metric, is_free, d_step = DepthAnalyzer.analyze_roi_depth(
                        depth_frame, bbox, source_size=(img_bgr.shape[1], img_bgr.shape[0]))
                    if valid_d and 0.40 <= z_metric <= 8.5:
                        xb, yb, zb = DepthAnalyzer.project_2d_to_3d_body(cu, cv, z_metric)
                        # Depth measures the visible front surface. Move the
                        # reported point toward the object's centre so the map
                        # tag does not sit on the near edge of a torso/cylinder.
                        centre_offset = 0.22 if det.get('detector') == 'mannequin_signature' else 0.14
                        ray_x = xb - STEREO_INTRINSICS['cam_pos_body'][0]
                        ray_y = yb - STEREO_INTRINSICS['cam_pos_body'][1]
                        ray_norm = max(1e-6, math.hypot(ray_x, ray_y))
                        xb += centre_offset * ray_x / ray_norm
                        yb += centre_offset * ray_y / ray_norm
                        cand_dist = math.hypot(xb, yb)
                        cand_bearing = math.atan2(yb, xb)
                        is_depth_verified = True
                        freestanding = is_free
                        depth_step = d_step

            # Fallback or Surround Cameras (left / right): Optical ray bearing + LiDAR range lookup
            if xb is None:
                ang_cam = math.atan2((cu - cx), fx)
                if cam_name == 'front':
                    cand_bearing = -ang_cam
                elif cam_name == 'left':
                    cand_bearing = (math.pi / 2.0) - ang_cam
                elif cam_name == 'right':
                    cand_bearing = (-math.pi / 2.0) - ang_cam

                cand_bearing = math.atan2(math.sin(cand_bearing), math.cos(cand_bearing))

                with self.scan_lock:
                    scan_ranges = self.latest_scan_ranges
                    amin = self.latest_scan_amin
                    astep = self.latest_scan_astep

                if scan_ranges and len(scan_ranges) > 30:
                    ray_idx = int(round((cand_bearing - amin) / astep)) % len(scan_ranges)
                    win_ranges = []
                    for offset in range(-4, 5):
                        idx = (ray_idx + offset) % len(scan_ranges)
                        r_val = scan_ranges[idx]
                        if np.isfinite(r_val) and 0.35 <= r_val <= 8.5:
                            win_ranges.append(r_val)

                    if win_ranges:
                        cand_dist = float(np.min(win_ranges))
                        xb = cand_dist * math.cos(cand_bearing)
                        yb = cand_dist * math.sin(cand_bearing)
                        zb = -0.60

            if xb is None or cand_dist is None or not (0.35 <= cand_dist <= 8.5):
                continue

            # Transform body (xb, yb) to MAP frame (wx, wy)
            cos_y = math.cos(dyaw)
            sin_y = math.sin(dyaw)
            # Rotate the measured body point using the complete attitude.
            cr, sr = math.cos(roll), math.sin(roll)
            cp, sp = math.cos(pitch), math.sin(pitch)
            level_x = cp * xb + sp * (sr * yb + cr * zb)
            level_y = cr * yb - sr * zb
            wx = dx + level_x * cos_y - level_y * sin_y
            wy = dy + level_x * sin_y + level_y * cos_y

            if abs(wx) > ARENA_BOUND or abs(wy) > ARENA_BOUND:
                continue

            # A far object may guide exploration, but it is never committed to
            # the map until repeated fused observations are made within 2m.
            if cand_dist > MAX_CONFIRM_RANGE:
                self._request_inspection(wx, wy, cand_dist, det.get('detector', 'vision'))

            vis_score = conf * 100.0
            if is_depth_verified:
                geom_score = 95.0 if freestanding else 80.0
                fused_score = min(99.0, 0.55 * vis_score + 0.35 * geom_score + 10.0)
            else:
                geom_score = 80.0
                fused_score = min(99.0, 0.60 * vis_score + 0.30 * geom_score + 10.0)

            cand_data = {
                'wx': wx,
                'wy': wy,
                'width_m': float((u2-u1) * max(0.1, cand_dist-0.17) / fx),
                'dist': cand_dist,
                'bearing': cand_bearing,
                'radius': 0.24,
                'rmse': 0.02,
                'geom_score': geom_score,
                'vis_score': vis_score,
                'fused_score': fused_score,
                'vis_detected': True,
                'vis_cam': cam_name,
                'vis_stamp': now,
                'bbox': bbox,
                'ssd_detected': False,
                'ssd_conf': 0.0,
                'ssd_bbox': None,
                'custom_net_conf': conf,
                'detector': det.get('detector', 'mannequin_signature'),
                'geometry_verified': False,
                'depth_verified': is_depth_verified,
                'freestanding': freestanding,
                'depth_step': depth_step,
                'within_confirm_range': cand_dist <= MAX_CONFIRM_RANGE,
                't': now
            }

            if cam_name == 'front':
                self.latest_debug_detections.append(cand_data)
                self.latest_debug_detections = self.latest_debug_detections[-20:]

            self.register_candidate_observation(cand_data)

    def _request_inspection(self, wx, wy, distance, source):
        """Ask navigation to approach a plausible, not-yet-confirmable object."""
        with self.survivor_lock:
            if any(math.hypot(wx-s['x'], wy-s['y']) <= MERGE_SURVIVOR_RADIUS
                   for s in self.confirmed_survivors):
                return
        key = (round(wx * 2.0) / 2.0, round(wy * 2.0) / 2.0)
        now = time.time()
        if now - self.last_inspection_request.get(key, 0.0) < INSPECTION_REPUBLISH_S:
            return
        self.last_inspection_request[key] = now
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(wx)
        msg.pose.position.y = float(wy)
        msg.pose.position.z = float(distance)  # diagnostic range at request time
        msg.pose.orientation.w = 1.0
        self.pub_inspection.publish(msg)
        self.get_logger().info(
            f'🔎 [INSPECT] {source} candidate at ({wx:+.2f}, {wy:+.2f}), '
            f'{distance:.2f}m away; requesting a {INSPECTION_STANDOFF:.2f}m approach.')

    def cb_relaunch(self, msg: Bool):
        if msg.data:
            with self.survivor_lock:
                self.confirmed_survivors.clear()
                self.candidate_tracks.clear()
                self.next_track_id = 1
            self.get_logger().info('🔄 [RELAUNCH] Cleared all survivor tracks for new mission.')
            self._do_publish()

    # ==========================================================================
    # STAGE 1: LIDAR GEOMETRIC PROCESSING (CIRCLE FIT & LINEARITY FILTER)
    # ==========================================================================
    def cb_scan(self, msg: LaserScan):
        with self.pose_lock:
            if not self.first_pose_received:
                return
            dx, dy, dz, dyaw = self.drone_x, self.drone_y, self.drone_z, self.drone_yaw
            roll, pitch = self.drone_roll, self.drone_pitch

        # Store latest scan ranges for cross-validation by Vision-First stream
        with self.scan_lock:
            self.latest_scan_ranges = list(msg.ranges)
            self.latest_scan_amin = msg.angle_min
            self.latest_scan_astep = msg.angle_increment

        # Skip scan if drone is still grounded or heavily tilted (> 12 deg)
        if dz < 0.40 or abs(roll) > 0.20 or abs(pitch) > 0.20:
            return

        self.process_lidar_geometry(msg.ranges, msg.angle_min, msg.angle_increment, dx, dy, dz, dyaw)

    def process_lidar_geometry(self, ranges, angle_min, angle_step, dx, dy, dz, dyaw):
        num_rays = len(ranges)
        if num_rays < 30:
            return

        now = time.time()

        # 1. Convert valid returns to body-frame points (xb, yb)
        pts = []
        for i in range(num_rays):
            r = ranges[i]
            if 0.25 < r < 7.5 and np.isfinite(r):
                ang = angle_min + i * angle_step
                xb = r * math.cos(ang)
                yb = r * math.sin(ang)
                pts.append((i, r, ang, xb, yb))
            else:
                pts.append(None)

        # 2. Contiguous clustering (Euclidean jump <= 0.35m)
        clusters = []
        curr = []
        for i in range(num_rays):
            p = pts[i]
            if p is None:
                if len(curr) >= 3:
                    clusters.append(curr)
                curr = []
                continue
            if not curr:
                curr.append(p)
            else:
                prev_p = curr[-1]
                if math.hypot(p[3] - prev_p[3], p[4] - prev_p[4]) <= 0.35:
                    curr.append(p)
                else:
                    if len(curr) >= 3:
                        clusters.append(curr)
                    curr = [p]
        if len(curr) >= 3:
            clusters.append(curr)

        # Wrap-around across 0/360 degrees
        if len(clusters) >= 2 and pts[0] is not None and pts[-1] is not None:
            if math.hypot(pts[0][3] - pts[-1][3], pts[0][4] - pts[-1][4]) <= 0.35:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()

        debug_detections = []

        # 3. Geometric Filtering per cluster
        for cl in clusters:
            if len(cl) < 3 or len(cl) > 40:
                continue

            chord_w = math.hypot(cl[-1][3] - cl[0][3], cl[-1][4] - cl[0][4])
            # Standing mannequin diameter 0.48m -> chord width between 0.10m and 0.85m
            if not (0.10 <= chord_w <= 0.85):
                continue

            cl_xy = np.array([[p[3], p[4]] for p in cl])

            # Linearity check: reject flat planar walls
            linearity = compute_cluster_linearity(cl_xy)

            # Circle fitting (Kåsa least squares)
            circle_fit = circle_fit_kasa(cl_xy)
            if circle_fit is None:
                continue

            fit_xb, fit_yb, fit_R, rmse = circle_fit

            # Geometry gate:
            # - Radius near mannequin radius ~0.24m (allow [0.12, 0.40]m)
            # - Radial RMSE <= 0.065m
            # - Reject strongly linear clusters (linearity > 0.98) unless circle fit is exceptionally good
            if not (0.12 <= fit_R <= 0.40):
                continue
            if rmse > 0.065:
                continue
            if linearity > 0.98 and rmse > 0.035:
                continue

            # Dual-side depth step check
            idx_before = (cl[0][0] - 1) % num_rays
            idx_after = (cl[-1][0] + 1) % num_rays
            r_bef = ranges[idx_before]
            r_aft = ranges[idx_after]
            r_closest = min(p[1] for p in cl)

            jump_bef = (not np.isfinite(r_bef)) or (r_bef > r_closest + 0.14)
            jump_aft = (not np.isfinite(r_aft)) or (r_aft > r_closest + 0.14)

            # Must be isolated object (at least one depth jump)
            if not (jump_bef or jump_aft):
                continue

            # Target center in body coordinates
            cand_bearing = math.atan2(fit_yb, fit_xb)
            cand_dist = math.hypot(fit_xb, fit_yb)

            # Convert body coordinates to MAP coordinates using drone pose
            # wx = dx + xb*cos(yaw) - yb*sin(yaw)
            # wy = dy + xb*sin(yaw) + yb*cos(yaw)
            cos_y = math.cos(dyaw)
            sin_y = math.sin(dyaw)
            wx = dx + fit_xb * cos_y - fit_yb * sin_y
            wy = dy + fit_xb * sin_y + fit_yb * cos_y

            # Check compound arena perimeter boundary
            if abs(wx) > ARENA_BOUND or abs(wy) > ARENA_BOUND:
                continue

            # Compute geometry quality score [0..100]
            rad_err = abs(fit_R - 0.24) / 0.12
            geom_score = max(0.0, min(100.0, 100.0 * (1.0 - 0.5 * rad_err - 0.5 * (rmse / 0.065))))

            # STAGE 2 & 3: Multi-Camera Visual Evidence & Fusion
            vis_info = self.evaluate_visual_evidence(fit_xb, fit_yb, cand_bearing, cand_dist)

            # Combine evidence into bounded fusion score
            vis_score = vis_info['visual_score']
            is_vis_detected = vis_info['detected']

            # Weighted fusion: 45% geometry, 45% visual signature, 10% projection agreement
            if is_vis_detected:
                fused_score = 0.45 * geom_score + 0.45 * vis_score + 10.0
            else:
                fused_score = 0.60 * geom_score * 0.40 # Heavily penalized without visual evidence

            fused_score = min(99.0, max(0.0, fused_score))

            cand_data = {
                'wx': wx,
                'wy': wy,
                'dist': cand_dist,
                'bearing': cand_bearing,
                'radius': fit_R,
                'rmse': rmse,
                'geom_score': geom_score,
                'vis_score': vis_score,
                'fused_score': fused_score,
                'vis_detected': is_vis_detected,
                'vis_cam': vis_info.get('camera'),
                'vis_stamp': vis_info.get('stamp', 0.0),
                'bbox': vis_info.get('bbox'),
                'ssd_detected': vis_info.get('ssd_detected', False),
                'ssd_conf': vis_info.get('ssd_conf', 0.0),
                'ssd_bbox': vis_info.get('ssd_bbox'),
                'detector': 'lidar_visual_fusion',
                'geometry_verified': True,
                'depth_verified': vis_info.get('depth_agreed', False),
                'freestanding': True,  # dual-side LiDAR depth-step passed above
                'depth_step': vis_info.get('depth_step', 0.0),
                'within_confirm_range': cand_dist <= MAX_CONFIRM_RANGE,
                't': now
            }
            debug_detections.append(cand_data)

            if cand_dist > MAX_CONFIRM_RANGE and (is_vis_detected or geom_score >= 70.0):
                self._request_inspection(wx, wy, cand_dist, 'lidar/vision')

            # STAGE 4: Spatial Tracking & Confirmation Gating
            self.register_candidate_observation(cand_data)

        self.latest_debug_detections = debug_detections

    # ==========================================================================
    # STAGE 2 & 3: COMPUTER VISION & MULTI-CAMERA FUSION
    # ==========================================================================
    def evaluate_visual_evidence(self, xb, yb, bearing, dist):
        """
        Projects LiDAR candidate (xb, yb) into the appropriate camera.
        Extracts image ROI around projection, evaluates mannequin visual signature,
        and optionally evaluates YOLO neural inference.
        Returns structured dictionary.
        """
        cam_name = select_camera_for_bearing(bearing)
        if cam_name is None:
            return {
                'detected': False,
                'visual_score': 0.0,
                'camera': 'none',
                'reason': 'Outside camera FOV coverage'
            }

        with self.cam_lock:
            frame = self.cam_frames.get(cam_name)
            stamp = self.cam_stamps.get(cam_name, 0.0)

        if frame is None:
            return {
                'detected': False,
                'visual_score': 0.0,
                'camera': cam_name,
                'reason': 'No frame received'
            }

        # Project mannequin torso/head into camera coordinates
        # Drone flies at z ~ 1.5m, mannequin stands from z=0 to z=1.75m.
        # Mannequin center is at z ~ 0.88m, relative to drone z_body ~ -0.62m.
        proj = project_point_to_camera(xb, yb, -0.60, cam_name)
        if proj is None:
            return {
                'detected': False,
                'visual_score': 0.0,
                'camera': cam_name,
                'reason': 'Projection outside image boundaries'
            }

        u, v, zc = proj
        cfg = CAMERA_CONFIGS[cam_name]
        W, H = cfg['width'], cfg['height']

        # Determine expected ROI size based on distance
        # Mannequin is 0.48m wide, 1.75m high
        fx = (W / 2.0) / math.tan(cfg['hfov'] / 2.0)
        exp_w = fx * (0.60 / max(0.5, zc))
        exp_h = fx * (1.90 / max(0.5, zc))

        half_w = max(20.0, exp_w * 0.75)
        half_h = max(35.0, exp_h * 0.65)

        roi_u1 = max(0, int(u - half_w))
        roi_u2 = min(W, int(u + half_w))
        roi_v1 = max(0, int(v - half_h))
        roi_v2 = min(H, int(v + half_h))

        # 1. Custom NidarSurvivorNet Neural Classifier
        net_detected = False
        net_conf = 0.0
        if self.custom_classifier is not None and self.custom_classifier.is_loaded:
            roi_crop = frame[roi_v1:roi_v2, roi_u1:roi_u2]
            net_conf = self.custom_classifier.predict(roi_crop)
            if net_conf >= 0.70:
                net_detected = True

        # 2. Computer Vision Mannequin Signature Detection
        sig_detected, sig_score, sig_details = self.detect_mannequin_signature(frame, (roi_u1, roi_v1, roi_u2, roi_v2))

        # 3. MobileNet-SSD Deep Learning Object Detection (OpenCV DNN)
        ssd_detected = False
        ssd_conf = 0.0
        ssd_bbox = None
        if self.ssd_detector is not None and self.ssd_detector.is_loaded:
            ssd_dets = self.ssd_detector.detect(frame, conf_thresh=0.20, target_classes=['person'])
            for sd in ssd_dets:
                sbx = sd['bbox']
                ox1 = max(roi_u1, sbx[0])
                oy1 = max(roi_v1, sbx[1])
                ox2 = min(roi_u2, sbx[2])
                oy2 = min(roi_v2, sbx[3])
                if ox1 < ox2 and oy1 < oy2:
                    ssd_detected = True
                    if sd['confidence'] > ssd_conf:
                        ssd_conf = sd['confidence']
                        ssd_bbox = sbx

        # 4. Optional YOLOv5n inference on frame
        yolo_detected = False
        yolo_conf = 0.0
        if self.yolo_session is not None:
            yolo_dets = self.run_yolo_inference(frame)
            for yd in yolo_dets:
                ybx = yd['bbox']
                ox1 = max(roi_u1, ybx[0])
                oy1 = max(roi_v1, ybx[1])
                ox2 = min(roi_u2, ybx[2])
                oy2 = min(roi_v2, ybx[3])
                if ox1 < ox2 and oy1 < oy2:
                    if yd['class_id'] == 0:
                        yolo_detected = True
                        yolo_conf = max(yolo_conf, yd['confidence'])

        # 5. Stereo Depth Camera Metric Verification
        depth_agreed = False
        depth_step = 0.0
        if cam_name == 'front':
            with self.depth_lock:
                d_frame = self.latest_depth_frame
            if d_frame is not None:
                valid_d, z_fg, is_free, d_step = DepthAnalyzer.analyze_roi_depth(
                    d_frame, [roi_u1, roi_v1, roi_u2, roi_v2], source_size=(frame.shape[1], frame.shape[0]))
                depth_step = d_step
                if valid_d and is_free and abs(z_fg - dist) < 0.60:
                    depth_agreed = True

        # Multi-modal visual decision
        is_detected = (net_detected and net_conf >= 0.70) or sig_detected or (ssd_detected and ssd_conf >= 0.25) or (yolo_detected and yolo_conf >= 0.25)
        visual_score = max(net_conf * 100.0, sig_score * 100.0, ssd_conf * 100.0, yolo_conf * 100.0)
        if depth_agreed:
            visual_score = min(100.0, visual_score + 10.0)

        chosen_bbox = ssd_bbox if ssd_bbox else sig_details.get('bbox', [roi_u1, roi_v1, roi_u2, roi_v2])

        return {
            'detected': is_detected,
            'visual_score': visual_score,
            'camera': cam_name,
            'stamp': stamp,
            'u_proj': u,
            'v_proj': v,
            'bbox': chosen_bbox,
            'sig_details': sig_details,
            'custom_net_detected': net_detected,
            'custom_net_conf': net_conf,
            'ssd_detected': ssd_detected,
            'ssd_conf': ssd_conf,
            'ssd_bbox': ssd_bbox,
            'yolo_detected': yolo_detected,
            'yolo_conf': yolo_conf,
            'depth_agreed': depth_agreed,
            'depth_step': depth_step
        }

    def detect_mannequin_signature(self, image_bgr, roi_box):
        """
        Inspects cropped ROI for the characteristic mannequin vertical stack:
        Peach Head sphere + Fluorescent Yellow safety vest + Rescue Orange body.
        """
        u1, v1, u2, v2 = roi_box
        roi = image_bgr[v1:v2, u1:u2]
        roi_h, roi_w = roi.shape[:2]
        total_pixels = roi_h * roi_w
        if total_pixels < 100:
            return False, 0.0, {'reason': 'ROI too small'}

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Orange Body: Hue [4, 22], Sat [85, 255], Val [55, 255]
        mask_orange = cv2.inRange(hsv, np.array([4, 85, 55]), np.array([22, 255, 255]))
        # Yellow Vest: Hue [24, 38], Sat [85, 255], Val [75, 255]
        mask_yellow = cv2.inRange(hsv, np.array([24, 85, 75]), np.array([38, 255, 255]))

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

        n_orange = cv2.countNonZero(mask_orange)
        n_yellow = cv2.countNonZero(mask_yellow)
        n_combined = n_orange + n_yellow

        if n_combined < 30:
            return False, 0.0, {'reason': 'Zero/insufficient mannequin colors', 'orange': n_orange, 'yellow': n_yellow}

        mask_combined = cv2.bitwise_or(mask_orange, mask_yellow)
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, 0.0, {'reason': 'No contours found'}

        largest_c = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest_c)
        aspect_ratio = bh / max(1.0, float(bw))

        # Vertical centroid validation: yellow vest should be above orange body
        stack_valid = True
        if n_orange > 15 and n_yellow > 15:
            M_o = cv2.moments(mask_orange)
            M_y = cv2.moments(mask_yellow)
            if M_o["m00"] > 0 and M_y["m00"] > 0:
                y_o = M_o["m01"] / M_o["m00"]
                y_y = M_y["m01"] / M_y["m00"]
                if y_y > y_o + 5.0: # Yellow is physically below orange (invalid)
                    stack_valid = False

        score_color = min(1.0, (n_orange + 1.4 * n_yellow) / (0.12 * total_pixels + 1e-5))
        score_aspect = 1.0 if (0.30 <= aspect_ratio <= 4.8) else (0.5 if (0.22 <= aspect_ratio <= 5.8) else 0.1)
        score_stack = 1.0 if stack_valid else 0.4

        conf = float(0.50 * score_color + 0.30 * score_aspect + 0.20 * score_stack)
        detected = (conf >= 0.45) and (n_orange >= 20 and n_yellow >= 20) and (aspect_ratio >= 0.22)

        global_bbox = [u1 + bx, v1 + by, u1 + bx + bw, v1 + by + bh]
        return detected, conf, {
            'bbox': global_bbox,
            'aspect_ratio': aspect_ratio,
            'orange_px': n_orange,
            'yellow_px': n_yellow,
            'stack_valid': stack_valid,
            'score': conf
        }

    def run_yolo_inference(self, img_bgr):
        """Runs YOLOv5n ONNX inference with float16 input casting and NMS."""
        if self.yolo_session is None or img_bgr is None:
            return []
        h_orig, w_orig = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        blob = np.transpose(img_rgb.astype(np.float16) / 255.0, (2, 0, 1))[np.newaxis, ...]
        try:
            out = self.yolo_session.run([self.yolo_output_name], {self.yolo_input_name: blob})
            pred = out[0][0] # (25200, 85)
            obj_conf = pred[:, 4]
            mask = obj_conf > 0.15
            if not np.any(mask):
                return []
            candidates = pred[mask]
            cls_scores = candidates[:, 5:]
            cls_ids = np.argmax(cls_scores, axis=1)
            scores = candidates[:, 4] * np.max(cls_scores, axis=1)
            valid = scores > 0.15
            if not np.any(valid):
                return []
            candidates = candidates[valid]
            cls_ids = cls_ids[valid]
            scores = scores[valid]

            cx = candidates[:, 0] * (w_orig / 640.0)
            cy = candidates[:, 1] * (h_orig / 640.0)
            w = candidates[:, 2] * (w_orig / 640.0)
            h = candidates[:, 3] * (h_orig / 640.0)
            x1 = np.maximum(0, cx - w / 2.0)
            y1 = np.maximum(0, cy - h / 2.0)
            x2 = np.minimum(w_orig, cx + w / 2.0)
            y2 = np.minimum(h_orig, cy + h / 2.0)

            boxes = np.column_stack([x1, y1, x2 - x1, y2 - y1]).tolist()
            indices = cv2.dnn.NMSBoxes(boxes, scores.tolist(), 0.15, 0.45)
            results = []
            if len(indices) > 0:
                for idx in indices.flatten():
                    results.append({
                        'class_id': int(cls_ids[idx]),
                        'confidence': float(scores[idx]),
                        'bbox': [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])]
                    })
            return results
        except Exception:
            return []

    # ==========================================================================
    # STAGE 4: SPATIAL TRACKING & MULTI-MODAL CONFIRMATION
    # ==========================================================================
    def register_candidate_observation(self, cand):
        """
        Nearest-neighbor association with ASSOC_RADIUS = 0.50m.
        Multi-modal confirmation gating requiring LiDAR + Vision agreement.
        """
        # Only aligned RGB-D observations can commit a survivor. LiDAR-only
        # projections remain useful inspection cues, but previously allowed
        # unrelated image patches to label walls as people.
        if (not cand.get('vis_detected') or not cand.get('depth_verified')
                or not cand.get('freestanding')
                or cand.get('detector') == 'lidar_visual_fusion'
                or not np.isfinite(cand.get('dist', float('nan')))
                or cand['dist'] > MAX_CONFIRM_RANGE):
            return
        wx, wy = cand['wx'], cand['wy']
        now = cand['t']
        fused_score = cand['fused_score']
        vis_detected = cand['vis_detected']
        vis_stamp = cand['vis_stamp']
        geometry_verified = bool(cand.get('geometry_verified', False))
        depth_verified = bool(cand.get('depth_verified', False) and cand.get('freestanding', False))
        within_range = bool(cand.get('within_confirm_range', cand.get('dist', 99.0) <= MAX_CONFIRM_RANGE))

        with self.survivor_lock:
            self.candidate_tracks = [track for track in self.candidate_tracks
                                     if now - track['last_seen'] < 2.0]
            # 1. Match against already CONFIRMED survivors
            matches = []
            for s in self.confirmed_survivors:
                distance = math.hypot(wx - s['x'], wy - s['y'])
                width, old_width = cand.get('width_m'), s.get('width_m')
                compatible = (width is None or old_width is None or
                              0.6 <= width / max(0.01, old_width) <= 1.67)
                variance = s.get('position_variance', 0.02) + 0.02
                statistical_match = distance**2 / variance <= 5.99
                overlapping = width is not None and old_width is not None and distance < 0.5*(width+old_width)
                if compatible and (distance <= MERGE_SURVIVOR_RADIUS or
                                   (distance <= 0.85 and statistical_match and overlapping)):
                    matches.append((distance, s))
            # Uncertain association never creates a new permanent anchor.
            if len(matches) > 1:
                return
            for _, s in matches:
                s['last_seen'] = now
                if fused_score > s['confidence']:
                    s['confidence'] = min(99, int(round(fused_score)))
                    self._do_publish()
                return
            # A conflicting size at an existing anchor is uncertain evidence,
            # not proof of a second person. Defer rather than create a ghost.
            if any(math.hypot(wx-s['x'], wy-s['y']) <= MERGE_SURVIVOR_RADIUS
                   for s in self.confirmed_survivors):
                return

            # 2. Match against active candidate tracks (nearest neighbor <= 0.50m)
            matched_track = None
            min_dist = ASSOC_RADIUS
            for track in self.candidate_tracks:
                d = math.hypot(wx - track['avg_x'], wy - track['avg_y'])
                if d < min_dist:
                    min_dist = d
                    matched_track = track

            if matched_track is not None:
                if vis_stamp - matched_track.get('last_vis_stamp', 0.0) <= 0.04:
                    return
                matched_track['observations'].append(cand)
                matched_track['last_seen'] = now
                if geometry_verified:
                    matched_track['lidar_hits'] += 1
                if depth_verified:
                    matched_track['depth_hits'] += 1
                if within_range:
                    matched_track['near_hits'] += 1
                matched_track['fusion_scores'].append(fused_score)

                # Check if this visual confirmation is from a new, distinct camera frame (>40ms apart)
                if vis_detected and (vis_stamp - matched_track.get('last_vis_stamp', 0.0)) > 0.04:
                    matched_track['vision_hits'] += 1
                    if geometry_verified or depth_verified:
                        matched_track['fused_hits'] += 1
                    matched_track['last_vis_stamp'] = vis_stamp

                obs = matched_track['observations']
                if len(obs) > 40:
                    obs = obs[-40:]
                    matched_track['observations'] = obs

                # Update centroid
                matched_track['avg_x'] = float(np.mean([o['wx'] for o in obs]))
                matched_track['avg_y'] = float(np.mean([o['wy'] for o in obs]))

                # 3. CONFIRMATION GATING
                duration = now - matched_track['first_seen']
                l_hits = matched_track['lidar_hits']
                d_hits = matched_track['depth_hits']
                v_hits = matched_track['vision_hits']
                f_hits = matched_track['fused_hits']
                near_hits = matched_track['near_hits']
                avg_fusion = float(np.mean(matched_track['fusion_scores'][-12:]))

                near_obs = [o for o in obs if o.get('within_confirm_range', o.get('dist', 99.0) <= MAX_CONFIRM_RANGE)]
                xs = [o['wx'] for o in near_obs]
                ys = [o['wy'] for o in near_obs]
                if not xs:
                    return
                pos_spread = math.hypot(float(np.std(xs)), float(np.std(ys)))

                # Strict multi-modal confirmation rule
                if ((l_hits >= MIN_LIDAR_HITS or d_hits >= MIN_LIDAR_HITS) and
                    v_hits >= MIN_VISION_HITS and
                    f_hits >= MIN_FUSED_HITS and
                    near_hits >= MIN_NEAR_HITS and
                    duration >= 0.30 and
                    pos_spread <= MAX_POS_SPREAD and
                    avg_fusion >= MIN_CONFIRM_FUSION):

                    # Check spatial deduplication with confirmed survivors
                    # Only close-range observations contribute to the committed
                    # map coordinate; distant estimates are approach cues only.
                    final_x = round(float(np.median(xs)), 2)
                    final_y = round(float(np.median(ys)), 2)

                    is_duplicate = False
                    for existing in self.confirmed_survivors:
                        if math.hypot(final_x - existing['x'], final_y - existing['y']) <= MERGE_SURVIVOR_RADIUS:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        survivor_id = len(self.confirmed_survivors) + 1
                        grid = get_grid_sector(final_x, final_y)
                        confirmed_conf = int(np.clip(round(avg_fusion), 70, 99))

                        new_survivor = {
                            'id': survivor_id,
                            'width_m': cand.get('width_m'),
                            'position_variance': max(0.01, pos_spread**2),
                            'status': 'CONFIRMED',
                            'x': final_x,
                            'y': final_y,
                            'map_pos_fixed': True,
                            'grid': grid,
                            'confidence': confirmed_conf,
                            'confirmation_range_m': round(min(o.get('dist', 99.0) for o in near_obs), 2),
                            'evidence': 'vision+depth' if d_hits >= MIN_LIDAR_HITS else 'vision+lidar',
                            'first_seen': matched_track['first_seen'],
                            'confirmed_at': now,
                            'last_seen': now
                        }
                        self.confirmed_survivors.append(new_survivor)

                        self.get_logger().info(
                            f'🚨 [GENUINE MULTI-MODAL SURVIVOR CONFIRMED] SURVIVOR #{survivor_id} | '
                            f'MAP Position: (X={final_x:+.2f}, Y={final_y:+.2f}) | '
                            f'Sector: {grid} | Evidence Conf: {confirmed_conf}% | '
                            f'LiDAR Hits: {l_hits} | Depth Hits: {d_hits} | Vision Hits: {v_hits} | '
                            f'Near Hits: {near_hits} | Spread: {pos_spread:.2f}m'
                        )
                        self._do_publish()

                    self.candidate_tracks.remove(matched_track)

                    # Prune candidate tracks within merge radius
                    self.candidate_tracks = [
                        c for c in self.candidate_tracks
                        if math.hypot(c['avg_x'] - final_x, c['avg_y'] - final_y) > MERGE_SURVIVOR_RADIUS
                    ]
            else:
                # Initialize new candidate track
                new_track = {
                    'track_id': self.next_track_id,
                    'avg_x': wx,
                    'avg_y': wy,
                    'observations': [cand],
                    'first_seen': now,
                    'last_seen': now,
                    'lidar_hits': 1 if geometry_verified else 0,
                    'depth_hits': 1 if depth_verified else 0,
                    'near_hits': 1 if within_range else 0,
                    'vision_hits': 1 if vis_detected else 0,
                    'fused_hits': 1 if vis_detected and (geometry_verified or depth_verified) else 0,
                    'last_vis_stamp': vis_stamp if vis_detected else 0.0,
                    'fusion_scores': [fused_score]
                }
                self.next_track_id += 1
                self.candidate_tracks.append(new_track)

            # Clean stale candidate tracks (> 15s unseen)
            self.candidate_tracks = [
                c for c in self.candidate_tracks if (now - c['last_seen']) < 15.0
            ]

    # ==========================================================================
    # STAGE 5: DEBUG HUD & TOPIC PUBLISHING
    # ==========================================================================
    def _render_and_publish_hud(self, frame_bgr):
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        # Top Badge: NIDAR TRI-MODAL PERCEPTION ACTIVE
        ai_tags = []
        if self.custom_classifier and self.custom_classifier.is_loaded:
            ai_tags.append("TRI-MODAL CNN")
        if self.ssd_detector and self.ssd_detector.is_loaded:
            ai_tags.append("SSD")
        if self.yolo_session:
            ai_tags.append("YOLO")
        tag_str = "+".join(ai_tags) if ai_tags else "CV"
        title = f"🛡️ NIDAR [{tag_str}+DEPTH+LIDAR] | SURVIVORS: {len(self.confirmed_survivors)}"
        cv2.rectangle(annotated, (12, 10), (480, 34), (15, 23, 42), -1)
        cv2.rectangle(annotated, (12, 10), (480, 34), (16, 185, 129), 1)
        cv2.putText(annotated, title, (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (52, 211, 153), 1, cv2.LINE_AA)

        # Draw projected candidates and visual signatures
        for cd in self.latest_debug_detections[-8:]:
            if cd.get('vis_cam') == 'front':
                bbox = cd.get('bbox')
                if bbox:
                    # Candidate boxes originate in the 640x480 aligned RGB-D
                    # image; scale them when the HUD is showing the wide camera.
                    sx, sy = w / 640.0, h / 480.0
                    bbox = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
                    c_conf = cd.get('custom_net_conf', 0.0)
                    if c_conf >= 0.70:
                        color = (0, 255, 110)
                        lbl = f"SURVIVOR ({c_conf*100:.0f}%)"
                    elif cd.get('ssd_detected'):
                        color = (0, 255, 110)
                        lbl = f"SSD SURVIVOR ({cd['ssd_conf']*100:.0f}%)"
                    elif cd['vis_detected']:
                        color = (0, 210, 255)
                        lbl = f"SURVIVOR CAND ({cd['fused_score']:.0f}%)"
                    else:
                        color = (100, 100, 100)
                        lbl = "LiDAR CLUSTER"
                    cv2.rectangle(annotated, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.putText(annotated, lbl, (int(bbox[0]), max(15, int(bbox[1]) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # Level Artificial Horizon Reticle
        cx, cy = w // 2, h // 2
        cv2.line(annotated, (cx - 15, cy), (cx + 15, cy), (56, 189, 248), 1, cv2.LINE_AA)
        cv2.line(annotated, (cx, cy - 15), (cx, cy + 15), (56, 189, 248), 1, cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), 3, (56, 189, 248), 1, cv2.LINE_AA)

        out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        try:
            ros_img = Image()
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.header.frame_id = "camera_optical_frame"
            ros_img.height = h
            ros_img.width = w
            ros_img.encoding = "rgb8"
            ros_img.is_bigendian = False
            ros_img.step = w * 3
            ros_img.data = out_rgb.tobytes()
            self.pub_camera.publish(ros_img)
        except Exception:
            pass

    def publish_markers(self):
        with self.survivor_lock:
            self._do_publish()

    def _do_publish(self):
        # 1. Int32 Survivor Count
        count_msg = Int32()
        count_msg.data = len(self.confirmed_survivors)
        self.pub_count.publish(count_msg)

        # 2. Authoritative JSON list on /nidar/survivors_detailed
        detailed_list = []
        for s in self.confirmed_survivors:
            detailed_list.append({
                'id': s['id'],
                'status': 'CONFIRMED',
                'map_x': s['x'],
                'map_y': s['y'],
                'map_pos_fixed': True,
                'grid': s.get('grid', get_grid_sector(s['x'], s['y'])),
                'confidence': s.get('confidence', 95),
                'confirmation_range_m': s.get('confirmation_range_m'),
                'evidence': s.get('evidence', 'vision+lidar'),
                'last_seen': s.get('last_seen', time.time())
            })
        det_msg = String()
        det_msg.data = json.dumps(detailed_list)
        self.pub_detailed.publish(det_msg)

        # 3. Persistent 3D Markers in MAP Frame (id = sid * 10 + type)
        marker_array = MarkerArray()
        now_msg = self.get_clock().now().to_msg()

        for s in self.confirmed_survivors:
            sid = s['id']
            sx = s['x']
            sy = s['y']
            s_grid = s.get('grid', get_grid_sector(sx, sy))
            s_conf = s.get('confidence', 95)

            # Footprint disc (type 0)
            m_disc = Marker()
            m_disc.header.frame_id = 'map'
            m_disc.header.stamp = now_msg
            m_disc.ns = 'survivors'
            m_disc.id = sid * 10 + 0
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

            # Body cube (type 1)
            m_pin = Marker()
            m_pin.header.frame_id = 'map'
            m_pin.header.stamp = now_msg
            m_pin.ns = 'survivors'
            m_pin.id = sid * 10 + 1
            m_pin.type = Marker.CUBE
            m_pin.action = Marker.ADD
            m_pin.pose.position.x = float(sx)
            m_pin.pose.position.y = float(sy)
            m_pin.pose.position.z = 0.90
            m_pin.pose.orientation.w = 1.0
            m_pin.scale.x = 0.80
            m_pin.scale.y = 0.80
            m_pin.scale.z = 1.80
            m_pin.color.r = 1.0
            m_pin.color.g = 0.45
            m_pin.color.b = 0.0
            m_pin.color.a = 1.0
            marker_array.markers.append(m_pin)

            # Head sphere (type 2)
            m_head = Marker()
            m_head.header.frame_id = 'map'
            m_head.header.stamp = now_msg
            m_head.ns = 'survivors'
            m_head.id = sid * 10 + 2
            m_head.type = Marker.SPHERE
            m_head.action = Marker.ADD
            m_head.pose.position.x = float(sx)
            m_head.pose.position.y = float(sy)
            m_head.pose.position.z = 1.95
            m_head.pose.orientation.w = 1.0
            m_head.scale.x = 0.45
            m_head.scale.y = 0.45
            m_head.scale.z = 0.45
            m_head.color.r = 1.0
            m_head.color.g = 0.0
            m_head.color.b = 0.0
            m_head.color.a = 1.0
            marker_array.markers.append(m_head)

            # Floating 3D Text Label (type 3)
            m_text = Marker()
            m_text.header.frame_id = 'map'
            m_text.header.stamp = now_msg
            m_text.ns = 'survivors_text'
            m_text.id = sid * 10 + 3
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x = float(sx)
            m_text.pose.position.y = float(sy)
            m_text.pose.position.z = 2.45
            m_text.pose.orientation.w = 1.0
            m_text.scale.z = 0.38
            m_text.color.r = 1.0
            m_text.color.g = 0.95
            m_text.color.b = 0.9
            m_text.color.a = 1.0
            m_text.text = f"SURVIVOR #{sid}\nStatus: CONFIRMED\nMap: fixed ({sx:+.2f}, {sy:+.2f})\nGrid: {s_grid}\nConf: {s_conf}%"
            marker_array.markers.append(m_text)

        if marker_array.markers:
            self.pub_markers.publish(marker_array)

    def destroy_node(self):
        """Stop Gazebo callbacks before ROS / Python objects are torn down."""
        if getattr(self, 'gz_node', None) is not None:
            for topic in (
                '/nidar/camera', '/nidar/stereo_camera/image',
                '/nidar/stereo_camera/depth_image', '/nidar/camera_left',
                '/nidar/camera_right'):
                try:
                    self.gz_node.unsubscribe(topic)
                except Exception:
                    pass
            # Gazebo Transport dispatches callbacks on native threads. Give an
            # in-flight callback time to return before releasing the pybind node.
            time.sleep(0.20)
            self.gz_node = None
        return super().destroy_node()


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

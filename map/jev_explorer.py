#!/usr/bin/env python3

import os
import sys
import time
import math
import signal
import threading
import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_msgs.msg import Int32, Bool, String, Float32

ARENA_LIMIT = 7.1  # Comprehensive 15x15m indoor compound boundary (+/-7.5m outer walls)
UNKNOWN  = -1
FREE     = 0
OCCUPIED = 100


def calculate_search_confidence(map_coverage, visual_coverage,
                                pending_inspections, frontier_count,
                                sectors_toured, has_block_gap,
                                has_fine_visual_gap):
    """Conservative evidence score for count-independent search completion."""
    map_score = float(np.clip(map_coverage, 0.0, 100.0))
    visual_score = float(np.clip(visual_coverage, 0.0, 100.0))
    inspection_score = max(0.0, 100.0 - 20.0 * pending_inspections)
    # LiDAR coverage is the primary evidence that the bounded arena has been
    # searched. Camera coverage is independent corroboration, not a demand to
    # physically occupy every arbitrary grid block. A 97.8% map plus 94% of
    # free-space viewpoints observed by successful 360-degree RGB-D sweeps can
    # therefore cross 97%, provided no unresolved evidence remains.
    score = 0.70 * map_score + 0.20 * visual_score + 0.10 * inspection_score
    score -= min(10.0, 0.75 * max(0, frontier_count))
    if has_fine_visual_gap:
        score -= 5.0
    if not sectors_toured:
        score = min(score, 85.0)
    # A score above 97 has a strict meaning: no unresolved evidence remains.
    if (pending_inspections > 0 or frontier_count > 0 or has_block_gap or
            has_fine_visual_gap):
        score = min(score, 96.5)
    return float(np.clip(score, 0.0, 100.0))


def grid_line_is_clear(grid, start, end):
    """True when a camera ray crosses only known free occupancy cells.

    Unknown cells and occupied cells both occlude the camera. This is a small
    integer Bresenham traversal so audit confidence cannot see through walls.
    The first cell is ignored because it is the camera's own position.
    """
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    first = True
    while True:
        if not first and grid[y0, x0] != FREE:
            return False
        first = False
        if x0 == x1 and y0 == y1:
            return True
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy

def _clean_exit_handler(sig, frame):
    os._exit(0)

signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)

class JevDecisionLayer(Node):
    def __init__(self):
        super().__init__('jev_explorer')
        self.get_logger().info('🧠 [JEV] Initializing JEV Decision-Making Layer...')

        self.drone_x = -5.80
        self.drone_y = -8.80
        self.drone_z = 0.0
        self.home_x = None
        self.home_y = None
        self.pose_lock = threading.Lock()

        self.grid = None
        self.map_info = None
        self.grid_lock = threading.Lock()

        # JEV Short-term memory
        self.memory = {
            'visited_locations': [],
            'failed_actions': [],
            'current_objective': None,
            'expected_outcome': None,
            'coverage': 0.0,
            'frontiers': []
        }

        # Subscriptions
        self.sub_pose = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.cb_pose, 10)

        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=QoSReliabilityPolicy.RELIABLE)
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.cb_map, map_qos)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.survivor_count = 0
        self.mission_complete_received = False
        self.return_requested = False
        self.survivors_detailed = []
        self.inspection_lock = threading.Lock()
        # Keep a deduplicated queue instead of one overwriteable target.  A
        # burst of LiDAR prospects must not erase a person waiting for a
        # close-range RGB-D inspection.
        self.pending_inspections = {}
        self.inspection_cooldowns = {}
        self.sub_survivors = self.create_subscription(Int32, '/nidar/survivor_count', self.cb_survivor_count, 10)
        self.sub_surv_detailed = self.create_subscription(String, '/nidar/survivors_detailed', self.cb_survivors_detailed, 10)
        self.sub_complete = self.create_subscription(Bool, '/nidar/mission_complete', self.cb_mission_complete, 10)
        self.sub_land = self.create_subscription(Bool, '/nidar/land_command', self.cb_land_cmd, 10)
        self.sub_relaunch = self.create_subscription(Bool, '/nidar/relaunch_command', self.cb_relaunch_cmd, 10)
        self.sub_inspection = self.create_subscription(
            PoseStamped, '/nidar/inspection_target', self.cb_inspection_target, 10)
        self.relaunch_requested = False

        self.pub_land_cmd = self.create_publisher(Bool, '/nidar/land_command', 10)
        self.pub_complete = self.create_publisher(Bool, '/nidar/mission_complete', 10)
        self.pub_goal = self.create_publisher(PoseStamped, '/nidar/current_goal', 10)
        self.pub_status = self.create_publisher(String, '/nidar/jev_status', 10)
        self.pub_search_confidence = self.create_publisher(
            Float32, '/nidar/search_confidence', 10)

        self.active_goal_handle = None
        self.goal_status = None
        self.is_acting = False
        self.goal_generation = 0
        self.cancel_pending = False
        self.blocked_targets = []
        self.route = []
        self.blocked_scan_hits = 0
        self.current_target = None
        # The survivor detector deliberately uses the front RGB-D stream.  A
        # short in-place sweep therefore gives it true 360-degree coverage
        # without fabricating depth for the left/right RGB-only cameras.
        self.declare_parameter('camera_scan_enabled', True)
        self.declare_parameter('camera_scan_headings', 5)
        self.declare_parameter('camera_scan_dwell_s', 0.70)
        self.declare_parameter('camera_scan_goal_timeout_s', 6.0)
        self.declare_parameter('audit_start_coverage', 90.0)
        self.declare_parameter('audit_start_time_s', 600.0)
        self.declare_parameter('audit_block_size_m', 2.0)
        self.declare_parameter('camera_audit_visibility_m', 3.2)
        self.declare_parameter('camera_audit_completion_pct', 94.0)
        self.declare_parameter('completion_frontier_min_cells', 12)
        self.declare_parameter('completion_confidence_threshold', 97.0)
        self.declare_parameter('completion_stability_s', 15.0)
        self.rotation_in_place = False
        self.camera_scan_enabled = bool(
            self.get_parameter('camera_scan_enabled').value)
        self.scan_headings = max(2, int(
            self.get_parameter('camera_scan_headings').value))
        self.scan_dwell_s = max(0.25, float(
            self.get_parameter('camera_scan_dwell_s').value))
        self.scan_goal_timeout_s = max(2.0, float(
            self.get_parameter('camera_scan_goal_timeout_s').value))
        self.audit_start_coverage = float(
            self.get_parameter('audit_start_coverage').value)
        self.audit_start_time_s = max(60.0, float(
            self.get_parameter('audit_start_time_s').value))
        self.audit_block_size_m = max(1.0, float(
            self.get_parameter('audit_block_size_m').value))
        self.camera_audit_visibility_m = max(1.5, float(
            self.get_parameter('camera_audit_visibility_m').value))
        self.camera_audit_completion_pct = float(np.clip(
            self.get_parameter('camera_audit_completion_pct').value,
            85.0, 100.0))
        self.completion_frontier_min_cells = max(3, int(
            self.get_parameter('completion_frontier_min_cells').value))
        self.completion_confidence_threshold = float(
            self.get_parameter('completion_confidence_threshold').value)
        self.completion_stability_s = max(3.0, float(
            self.get_parameter('completion_stability_s').value))
        self.visual_scan_locations = []
        self.mission_started_at = None
        self.completion_confidence_since = None
        self.last_block_coverage_pct = 0.0
        self.sub_route = self.create_subscription(Path, '/plan', self.cb_route, 10)
        self.sub_obstacles = self.create_subscription(LaserScan, '/scan', self.cb_obstacles, qos_profile_sensor_data)

        self.thread = threading.Thread(target=self.jev_core_loop, daemon=True)
        self.thread.start()

    def cb_survivor_count(self, msg: Int32): self.survivor_count = msg.data

    def cb_route(self, msg):
        if msg.header.frame_id == 'map':
            self.route = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def cb_obstacles(self, msg):
        if (getattr(self, 'rotation_in_place', False) or not self.is_acting or
                not self.route or self.cancel_pending):
            return
        with self.pose_lock:
            x, y = self.drone_x, self.drone_y
            yaw = getattr(self, 'drone_yaw', 0.0)
        route = np.asarray(self.route)
        nearest = int(np.argmin(np.sum((route - [x, y]) ** 2, axis=1)))
        ahead = route[nearest:]
        ahead = ahead[np.linalg.norm(ahead - [x, y], axis=1) <= 1.0]
        if not len(ahead):
            return
        obstructed = False
        for i, distance in enumerate(msg.ranges):
            if not np.isfinite(distance) or not msg.range_min < distance < min(msg.range_max, 1.6):
                continue
            angle = yaw + msg.angle_min + i * msg.angle_increment
            point = [x + distance * math.cos(angle), y + distance * math.sin(angle)]
            if np.min(np.linalg.norm(ahead - point, axis=1)) < 0.38:
                obstructed = True
                break
        self.blocked_scan_hits = self.blocked_scan_hits + 1 if obstructed else 0
        threshold = getattr(self, 'blocked_scan_threshold', 2)
        if self.blocked_scan_hits >= threshold:
            if self.current_target:
                self.blocked_targets.append((*self.current_target, time.monotonic() + 25.0))
            try:
                self.get_logger().warning(f'⚡ [JEV] Dynamic obstacle/survivor detected on path! Rapid evasive replan triggered.')
            except Exception:
                pass
            self.cancel_current_action()
            self.goal_status = 'BLOCKED'
    def cb_survivors_detailed(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.survivors_detailed = data
            self.survivor_count = len(data)
        except Exception:
            pass

    def cb_inspection_target(self, msg: PoseStamped):
        """Queue a plausible object for close-range (<=2m) classification."""
        tx, ty = float(msg.pose.position.x), float(msg.pose.position.y)
        now = time.time()
        if abs(tx) > ARENA_LIMIT or abs(ty) > ARENA_LIMIT:
            return
        if any(math.hypot(tx - s.get('x', s.get('map_x', 99.0)),
                          ty - s.get('y', s.get('map_y', 99.0))) < 0.75
               for s in self.survivors_detailed):
            return
        key = (round(tx * 2.0) / 2.0, round(ty * 2.0) / 2.0)
        if now < self.inspection_cooldowns.get(key, 0.0):
            return
        with self.inspection_lock:
            self.pending_inspections[key] = (tx, ty, key, now)
            # Bound noise from imperfect geometric prospects while retaining
            # the newest observation for every spatial cell.
            if len(self.pending_inspections) > 24:
                oldest = min(self.pending_inspections,
                             key=lambda k: self.pending_inspections[k][3])
                del self.pending_inspections[oldest]

    def _next_inspection(self):
        """Return the nearest fresh queued prospect without losing the rest."""
        now = time.time()
        with self.pose_lock:
            cx, cy = self.drone_x, self.drone_y
        with self.inspection_lock:
            stale = [key for key, item in self.pending_inspections.items()
                     if now - item[3] >= 30.0]
            for key in stale:
                del self.pending_inspections[key]
            eligible = [item for key, item in self.pending_inspections.items()
                        if now >= self.inspection_cooldowns.get(key, 0.0)]
            if not eligible:
                return None
            chosen = min(eligible,
                         key=lambda item: math.hypot(item[0] - cx, item[1] - cy))
            self.pending_inspections.pop(chosen[2], None)
            return chosen

    def _inspection_goal(self, tx, ty):
        """Choose a free map cell 1.2-1.6m from an object, facing it."""
        with self.pose_lock:
            cx, cy = self.drone_x, self.drone_y
        base_angle = math.atan2(cy - ty, cx - tx)
        candidates = []
        for radius in (1.35, 1.60, 1.15):
            for offset in (0.0, 0.35, -0.35, 0.70, -0.70):
                angle = base_angle + offset
                gx = tx + radius * math.cos(angle)
                gy = ty + radius * math.sin(angle)
                candidates.append((gx, gy, math.hypot(gx - cx, gy - cy)))

        with self.grid_lock:
            grid = None if self.grid is None else self.grid.copy()
            info = self.map_info
        if grid is None or info is None:
            return None

        safe = []
        for gx, gy, travel in candidates:
            col = int((gx - info.origin.position.x) / info.resolution)
            row = int((gy - info.origin.position.y) / info.resolution)
            clearance = max(1, int(0.32 / info.resolution))
            if not (clearance <= row < info.height-clearance and clearance <= col < info.width-clearance):
                continue
            patch = grid[row-clearance:row+clearance+1, col-clearance:col+clearance+1]
            if patch.size and np.all((patch >= 0) & (patch < 65)):
                safe.append((travel, gx, gy))
        if not safe:
            return None
        _, gx, gy = min(safe)
        return gx, gy
    def cb_mission_complete(self, msg: Bool): self.mission_complete_received = msg.data
    def cb_land_cmd(self, msg: Bool):
        if msg.data:
            self.return_requested = True
            self.get_logger().info('🚨 [JEV] Return-to-Helipad command received from GCS / System! Aborting current action...')
            self.cancel_current_action()

    def cb_relaunch_cmd(self, msg: Bool):
        if msg.data:
            self.get_logger().info('🚀 [JEV] Relaunch signal received! Resetting exploration states...')
            self.return_requested = False
            self.relaunch_requested = True
            self.mission_complete_received = False
            with self.inspection_lock:
                self.pending_inspections.clear()
            self.inspection_cooldowns.clear()
            self.visual_scan_locations.clear()
            self.mission_started_at = time.monotonic()
            self.completion_confidence_since = None
            self.last_block_coverage_pct = 0.0
            self.cancel_current_action()

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)

    def cb_pose(self, msg: PoseStamped):
        with self.pose_lock:
            q = msg.pose.orientation
            self.drone_yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            self.drone_x = msg.pose.position.x
            self.drone_y = msg.pose.position.y
            self.drone_z = msg.pose.position.z
            if self.home_x is None:
                self.home_x = self.drone_x
                self.home_y = self.drone_y

    def cb_map(self, msg: OccupancyGrid):
        w = msg.info.width
        h = msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((h, w))
        with self.grid_lock:
            self.grid = data
            self.map_info = msg.info

    def get_coverage(self):
        with self.grid_lock:
            if self.grid is None: return 0.0
            info, g = self.map_info, self.grid
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        c0 = max(0, int((-ARENA_LIMIT - ox) / res))
        c1 = min(info.width, int((ARENA_LIMIT - ox) / res))
        r0 = max(0, int((-ARENA_LIMIT - oy) / res))
        r1 = min(info.height, int((ARENA_LIMIT - oy) / res))
        sub = g[r0:r1, c0:c1]
        total = sub.size
        explored = np.count_nonzero(sub != UNKNOWN)
        return (explored / float(total)) * 100.0 if total > 0 else 0.0

    def find_frontiers(self):
        with self.grid_lock:
            if self.grid is None: return []
            info, g = self.map_info, self.grid.copy()
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        h, w = g.shape
        free_cells = (g == FREE)
        unknown_cells = (g == UNKNOWN)
        f_mask = np.zeros_like(free_cells)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.zeros_like(unknown_cells)
            r_dst = slice(max(0, dr), min(h, h + dr))
            c_dst = slice(max(0, dc), min(w, w + dc))
            r_src = slice(max(0, -dr), min(h, h - dr))
            c_src = slice(max(0, -dc), min(w, w - dc))
            shifted[r_dst, c_dst] = unknown_cells[r_src, c_src]
            f_mask |= (free_cells & shifted)

        frontier_r, frontier_c = np.where(f_mask)
        f_set = set(zip(frontier_c, frontier_r))
        clusters, visited = [], set()

        for cell in f_set:
            if cell in visited: continue
            cluster, queue = [], [cell]
            visited.add(cell)
            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                cc, cr = curr
                for dc in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        nbr = (cc + dc, cr + dr)
                        if nbr in f_set and nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
            if len(cluster) >= 3:
                avg_c = sum(c for c, _ in cluster) / float(len(cluster))
                avg_r = sum(r for _, r in cluster) / float(len(cluster))
                wx = ox + (avg_c + 0.5) * res
                wy = oy + (avg_r + 0.5) * res
                if abs(wx) <= 6.9 and abs(wy) <= 6.9 and wy >= -6.8: # Full 15x15m indoor compound
                    # Medial axis corridor centering: buffer away from nearby walls into open space
                    safe_c, safe_r = int(round((wx - ox) / res)), int(round((wy - oy) / res))
                    check_rad = 10  # 0.50m safety check
                    r0 = max(0, safe_r - check_rad)
                    r1 = min(h, safe_r + check_rad + 1)
                    c0 = max(0, safe_c - check_rad)
                    c1 = min(w, safe_c + check_rad + 1)
                    occ_r, occ_c = np.where(g[r0:r1, c0:c1] == OCCUPIED)
                    if len(occ_r) > 0:
                        wall_c = c0 + float(np.mean(occ_c))
                        wall_r = r0 + float(np.mean(occ_r))
                        wall_wx = ox + (wall_c + 0.5) * res
                        wall_wy = oy + (wall_r + 0.5) * res
                        away_angle = math.atan2(wy - wall_wy, wx - wall_wx)
                        wx += 0.45 * math.cos(away_angle)
                        wy += 0.45 * math.sin(away_angle)
                    wx = float(np.clip(wx, -6.85, 6.85))
                    wy = float(np.clip(wy, -6.85, 6.85))
                    clusters.append((wx, wy, len(cluster)))
        return clusters

    def find_visual_recovery_goal(self):
        """Choose a safe map cell least covered by previous 360-degree scans.

        LiDAR can map a room through a doorway without the RGB-D camera ever
        obtaining a useful human view.  This independent visual-coverage pass
        samples known free space and moves the camera to the largest remaining
        viewpoint gap.  It uses only the live occupancy map, never survivor or
        ground-truth coordinates.
        """
        with self.grid_lock:
            if self.grid is None or self.map_info is None:
                return None
            grid = self.grid.copy()
            info = self.map_info
        with self.pose_lock:
            cx, cy = self.drone_x, self.drone_y

        resolution = float(info.resolution)
        stride = max(1, int(0.55 / resolution))
        clearance = max(1, int(0.38 / resolution))
        references = list(self.visual_scan_locations)
        visibility_m = getattr(self, 'camera_audit_visibility_m', 3.2)

        def visually_covered(wx, wy, row, col):
            for sx, sy in references:
                if math.hypot(wx - sx, wy - sy) > visibility_m:
                    continue
                scan_col = int((sx - info.origin.position.x) / resolution)
                scan_row = int((sy - info.origin.position.y) / resolution)
                if (0 <= scan_col < info.width and 0 <= scan_row < info.height and
                        grid_line_is_clear(
                            grid, (scan_col, scan_row), (col, row))):
                    return True
            return False

        candidates = []
        for row in range(clearance, info.height - clearance, stride):
            for col in range(clearance, info.width - clearance, stride):
                wx = info.origin.position.x + (col + 0.5) * resolution
                wy = info.origin.position.y + (row + 0.5) * resolution
                if abs(wx) > 6.85 or abs(wy) > 6.85 or wy < -6.8:
                    continue
                patch = grid[row-clearance:row+clearance+1,
                             col-clearance:col+clearance+1]
                if patch.size == 0 or not np.all((patch >= 0) & (patch < 65)):
                    continue
                if visually_covered(wx, wy, row, col):
                    continue
                travel = math.hypot(wx - cx, wy - cy)
                if travel < 0.70:
                    continue
                visual_gap = (min(math.hypot(wx - sx, wy - sy)
                                  for sx, sy in references)
                              if references else visibility_m)
                blocked = any(expiry > time.monotonic() and
                              math.hypot(wx - bx, wy - by) < 1.0
                              for bx, by, expiry in self.blocked_targets)
                if blocked:
                    continue
                # Maximise unseen visual coverage with a small travel penalty.
                candidates.append((visual_gap - 0.12 * travel, wx, wy))
        if not candidates:
            return None
        _, wx, wy = max(candidates)
        return float(wx), float(wy)

    def find_block_audit_goal(self):
        """Return a safe representative viewpoint in an unscanned map block.

        This is the fast 90%-coverage pass. Known free cells are divided into
        coarse blocks, each block is represented by its safest cell nearest the
        block centre, and blocks already containing a successful 360-degree
        scan are removed. Blocks with residual unknown space are prioritised,
        followed by camera distance and travel cost. No world or survivor
        coordinates participate in this selection.
        """
        with self.grid_lock:
            if self.grid is None or self.map_info is None:
                return None
            grid = self.grid.copy()
            info = self.map_info
        with self.pose_lock:
            cx, cy = self.drone_x, self.drone_y

        resolution = float(info.resolution)
        clearance = max(1, int(0.38 / resolution))
        stride = max(1, int(0.30 / resolution))
        block_size = self.audit_block_size_m
        visibility_m = getattr(self, 'camera_audit_visibility_m', 3.2)

        def block_key(wx, wy):
            return (int(math.floor((wx + ARENA_LIMIT) / block_size)),
                    int(math.floor((wy + ARENA_LIMIT) / block_size)))

        representatives = {}
        for row in range(clearance, info.height - clearance, stride):
            for col in range(clearance, info.width - clearance, stride):
                wx = info.origin.position.x + (col + 0.5) * resolution
                wy = info.origin.position.y + (row + 0.5) * resolution
                if abs(wx) > 6.85 or abs(wy) > 6.85 or wy < -6.8:
                    continue
                key = block_key(wx, wy)
                patch = grid[row-clearance:row+clearance+1,
                             col-clearance:col+clearance+1]
                if patch.size == 0 or not np.all((patch >= 0) & (patch < 65)):
                    continue
                centre_x = -ARENA_LIMIT + (key[0] + 0.5) * block_size
                centre_y = -ARENA_LIMIT + (key[1] + 0.5) * block_size
                centre_error = math.hypot(wx - centre_x, wy - centre_y)
                previous = representatives.get(key)
                if previous is None or centre_error < previous[0]:
                    representatives[key] = (centre_error, wx, wy, row, col)

        # A completed 360-degree scan observes every representative within
        # reliable RGB-D range that has an unobstructed free-space ray. The old
        # exact-block comparison severely under-counted real camera coverage.
        scanned_blocks = set()
        for key, (_, wx, wy, row, col) in representatives.items():
            for sx, sy in self.visual_scan_locations:
                if math.hypot(wx - sx, wy - sy) > visibility_m:
                    continue
                scan_col = int((sx - info.origin.position.x) / resolution)
                scan_row = int((sy - info.origin.position.y) / resolution)
                if (0 <= scan_col < info.width and 0 <= scan_row < info.height and
                        grid_line_is_clear(
                            grid, (scan_col, scan_row), (col, row))):
                    scanned_blocks.add(key)
                    break

        total_blocks = len(representatives)
        scanned_count = len(scanned_blocks)
        self.last_block_coverage_pct = (
            100.0 * scanned_count / total_blocks if total_blocks else 0.0)
        candidates = []
        search_px = max(1, int((block_size * 0.5) / resolution))
        for key, (_, wx, wy, row, col) in representatives.items():
            if key in scanned_blocks:
                continue
            if any(expiry > time.monotonic() and
                   math.hypot(wx - bx, wy - by) < 1.0
                   for bx, by, expiry in self.blocked_targets):
                continue
            r0, r1 = max(0, row-search_px), min(info.height, row+search_px+1)
            c0, c1 = max(0, col-search_px), min(info.width, col+search_px+1)
            area = grid[r0:r1, c0:c1]
            unknown_ratio = (float(np.count_nonzero(area == UNKNOWN)) / area.size
                             if area.size else 0.0)
            camera_gap = (min(math.hypot(wx-sx, wy-sy)
                              for sx, sy in self.visual_scan_locations)
                          if self.visual_scan_locations else block_size)
            travel = math.hypot(wx - cx, wy - cy)
            score = 4.0 * unknown_ratio + camera_gap - 0.10 * travel
            candidates.append((score, wx, wy))
        if not candidates:
            return None
        _, wx, wy = max(candidates)
        return float(wx), float(wy)

    def find_unknown_focus_near(self, wx, wy, radius_m=1.8):
        """Return the centroid of nearby unknown cells for camera orientation."""
        with self.grid_lock:
            if self.grid is None or self.map_info is None:
                return None
            grid = self.grid.copy()
            info = self.map_info
        resolution = float(info.resolution)
        col = int((wx - info.origin.position.x) / resolution)
        row = int((wy - info.origin.position.y) / resolution)
        radius = max(1, int(radius_m / resolution))
        r0, r1 = max(0, row-radius), min(info.height, row+radius+1)
        c0, c1 = max(0, col-radius), min(info.width, col+radius+1)
        unknown_r, unknown_c = np.where(grid[r0:r1, c0:c1] == UNKNOWN)
        if len(unknown_r) < 3:
            return None
        focus_col = c0 + float(np.mean(unknown_c))
        focus_row = r0 + float(np.mean(unknown_r))
        focus_x = info.origin.position.x + (focus_col + 0.5) * resolution
        focus_y = info.origin.position.y + (focus_row + 0.5) * resolution
        return float(focus_x), float(focus_y)

    def _act_navigate(self, x, y, face_target=None, target_yaw=None,
                      ignore_blocked=False):
        if self.cancel_pending:
            return
        now = time.monotonic()
        self.blocked_targets = [(bx, by, expiry) for bx, by, expiry in self.blocked_targets
                                if expiry > now]
        if (not ignore_blocked and
                any(math.hypot(x-bx, y-by) < 1.0 for bx, by, _ in self.blocked_targets)):
            self.goal_status = 'COOLDOWN'
            return
        self.goal_generation += 1
        generation = self.goal_generation
        self.current_target = (x, y)
        self.active_goal_handle = None
        self.route = []
        self.blocked_scan_hits = 0
        self.get_logger().info(f'🧠 [JEV] ACT: Dispatching Nav2 Goal to ({x:.2f}, {y:.2f})')
        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = 1.50
        with self.pose_lock: cx, cy = self.drone_x, self.drone_y
        yaw = math.atan2(y - cy, x - cx)
        if face_target is not None:
            yaw = math.atan2(face_target[1] - y, face_target[0] - x)
        if target_yaw is not None:
            yaw = float(target_yaw)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        # Publish current goal for GCS visualizer
        self.pub_goal.publish(goal.pose)

        self.is_acting = True
        self.goal_status = "PENDING"
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(lambda result: self._goal_response_cb(result, generation))

    def _scan_360_at_current_position(self):
        """Actively sweep the front RGB-D camera through the complete azimuth."""
        if (not self.camera_scan_enabled or self.return_requested or
                self.scan_headings < 2):
            return
        # Allow a just-cancelled proximity goal to finish its action callback.
        wait_until = time.time() + 3.0
        while self.cancel_pending and time.time() < wait_until and rclpy.ok():
            time.sleep(0.05)
        if self.cancel_pending:
            self.get_logger().warning(
                '[JEV] Skipping camera sweep: previous Nav2 cancellation is still pending.')
            return
        with self.pose_lock:
            x, y = self.drone_x, self.drone_y
            start_yaw = getattr(self, 'drone_yaw', 0.0)

        self.rotation_in_place = True
        self.publish_status('OBSERVE: Active 360° RGB-D survivor scan')
        self.get_logger().info(
            f'🔄 [JEV] Starting {self.scan_headings}-view 360° camera scan at ({x:.2f}, {y:.2f})')
        successful_views = 1  # Arrival heading was already observed.
        try:
            # The arrival heading is the first view. Visit the remaining evenly
            # spaced headings; 5 views overlap the stereo camera's 83° HFOV.
            for step in range(1, self.scan_headings):
                if self.return_requested or not rclpy.ok():
                    break
                yaw = start_yaw + (2.0 * math.pi * step / self.scan_headings)
                self._act_navigate(
                    x, y, target_yaw=yaw, ignore_blocked=True)
                deadline = time.time() + self.scan_goal_timeout_s
                while self.is_acting and time.time() < deadline and rclpy.ok():
                    time.sleep(0.10)
                if self.is_acting:
                    self.cancel_current_action()
                elif self.goal_status == "SUCCEEDED":
                    successful_views += 1
                # Three or more independent front RGB-D frames arrive here.
                time.sleep(self.scan_dwell_s)
        finally:
            self.rotation_in_place = False
            self.route = []
            with self.pose_lock:
                scan_position = (self.drone_x, self.drone_y)
            required_views = max(2, int(math.ceil(self.scan_headings * 0.60)))
            if successful_views >= required_views:
                if not self.visual_scan_locations or math.hypot(
                        scan_position[0] - self.visual_scan_locations[-1][0],
                        scan_position[1] - self.visual_scan_locations[-1][1]) > 0.25:
                    self.visual_scan_locations.append(scan_position)
                self.get_logger().info(
                    f'✅ [JEV] 360° RGB-D scan complete '
                    f'({successful_views}/{self.scan_headings} views).')
            else:
                self.get_logger().warning(
                    f'[JEV] Camera sweep incomplete ({successful_views}/'
                    f'{self.scan_headings} views); block remains pending for audit.')

    def _goal_response_cb(self, future, generation):
        handle = future.result()
        if generation != self.goal_generation:
            if handle.accepted:
                handle.cancel_goal_async().add_done_callback(self._cancel_done)
            else:
                self.cancel_pending = False
            return
        if not handle.accepted:
            self.get_logger().warning('[JEV] ACT FAILED: Nav2 rejected the goal.')
            self.goal_status = "REJECTED"
            self.is_acting = False
            return
        self.active_goal_handle = handle
        res_future = handle.get_result_async()
        res_future.add_done_callback(lambda result: self._goal_result_cb(result, generation))

    def _goal_result_cb(self, future, generation):
        if generation != self.goal_generation:
            if generation == self.goal_generation - 1:
                self.cancel_pending = False
            return
        res = future.result()
        if res.status == GoalStatus.STATUS_SUCCEEDED:
            self.goal_status = "SUCCEEDED"
        else:
            self.goal_status = f"FAILED_{res.status}"
        self.is_acting = False

    def cancel_current_action(self):
        if not self.is_acting:
            return
        self.goal_generation += 1
        self.cancel_pending = True
        if self.active_goal_handle:
            self.active_goal_handle.cancel_goal_async().add_done_callback(self._cancel_done)
            self.active_goal_handle = None
        self.is_acting = False

    def _cancel_done(self, future):
        try:
            response = future.result()
            self.cancel_pending = response.return_code not in (0, 3)
        except Exception as exc:
            self.get_logger().error(f'Navigation cancellation failed: {exc}')

    def jev_core_loop(self):
        self.get_logger().info('[JEV] Waiting for ROS 2 Subsystems (Nav2, Map, Pose)...')
        while not self.nav_client.wait_for_server(timeout_sec=1.0): pass
        while self.grid is None: time.sleep(0.5)

        self.get_logger().info('[JEV] Subsystems Online. Activating JEV Core Loop.')

        # Wait for drone to complete takeoff & reach cruising altitude
        self.get_logger().info('🧠 [JEV] Waiting for drone takeoff & climb to 1.5m cruise altitude...')
        self.publish_status("STANDBY: Awaiting Drone Takeoff & Climb to 1.5m")
        while rclpy.ok():
            with self.pose_lock:
                cz = self.drone_z
            if cz >= 1.20:
                self.get_logger().info(f'🛫 [JEV] Cruise altitude reached ({cz:.2f}m)! Engaging South Gate penetration...')
                break
            time.sleep(0.3)

        # Initial Penetration (Bypass exploration logic just to enter the maze)
        with self.pose_lock: start_y = self.drone_y
        if start_y < -7.0:
            self.get_logger().info('[JEV] PRE-MISSION: Drone is outside compound. Penetrating South Gate (-5.80, -7.20)...')
            self._act_navigate(-5.80, -7.20)
            while self.is_acting: time.sleep(0.5)

        # Ten minutes is a strategy-switch point, not a mission timeout.
        self.mission_started_at = time.monotonic()
        consecutive_failures = 0

        while rclpy.ok():
            time.sleep(0.8)
            if self.cancel_pending:
                continue

            # --- 1. OBSERVE ---
            with self.pose_lock:
                pos = (self.drone_x, self.drone_y)
            mission_elapsed = max(
                0.0, time.monotonic() - (self.mission_started_at or time.monotonic()))
            cov = self.get_coverage()
            frontiers = self.find_frontiers()
            completion_frontiers = [
                frontier for frontier in frontiers
                if frontier[2] >= self.completion_frontier_min_cells]
            self.memory['coverage'] = cov
            self.memory['frontiers'] = frontiers
            self.memory['position'] = pos
            self.publish_status(f"OBSERVE: Frontiers={len(frontiers)} | Cov={cov:.1f}%")

            self.get_logger().info(f'\n🧠 [JEV] --- COGNITIVE LOOP ---')
            self.get_logger().info(f'🧠 [JEV] OBSERVE: Pos=({pos[0]:.2f}, {pos[1]:.2f}), Coverage={cov:.1f}%, Frontiers={len(frontiers)}')

            # An appearance/geometric anomaly farther than 2m is an approach
            # cue, not a survivor. Move to a collision-checked stand-off pose so
            # RGB-D can classify it at close range before mapping it.
            inspection = self._next_inspection()
            if inspection is not None and not self.return_requested:
                tx, ty, key, requested_at = inspection
                if time.time() - requested_at < 20.0:
                    inspect_goal = self._inspection_goal(tx, ty)
                    if inspect_goal is not None:
                        gx, gy = inspect_goal
                        self.publish_status(
                            f"INSPECT: Approaching candidate ({tx:+.2f}, {ty:+.2f}) before classification")
                        self.get_logger().info(
                            f'🔎 [JEV] Candidate requires close verification; navigating to '
                            f'({gx:+.2f}, {gy:+.2f}), object=({tx:+.2f}, {ty:+.2f}).')
                        self._act_navigate(gx, gy, face_target=(tx, ty))
                        started = time.time()
                        while self.is_acting and rclpy.ok() and not self.return_requested and time.time() - started < 35.0:
                            time.sleep(0.25)
                            with self.pose_lock:
                                current = (self.drone_x, self.drone_y)
                        if self.is_acting:
                            self.cancel_current_action()
                        # Hold position briefly for independent RGB-D frames.
                        time.sleep(1.0)
                        self.inspection_cooldowns[key] = time.time() + 20.0
                        continue
                    self.inspection_cooldowns[key] = time.time() + 8.0

            # --- 2. QUESTION ---
            # Strictly require touring both West (X < -1.5m) and East (X > 1.5m) sectors of the compound,
            # plus >= 95.0% arena exploration before returning to base (ensuring all rooms are inspected)
            has_visited_east = any(vx > 1.5 for vx, vy in self.memory['visited_locations'])
            has_visited_west = any(vx < -1.5 for vx, vy in self.memory['visited_locations'])
            sectors_toured = has_visited_east and has_visited_west

            coverage_complete = sectors_toured and (
                cov >= 95.0 or (cov >= 92.0 and len(frontiers) == 0))
            audit_active = (
                cov >= self.audit_start_coverage or
                mission_elapsed >= self.audit_start_time_s)
            # Always calculate block evidence for the confidence graph, but do
            # not prioritise block visits until the 90%/10-minute strategy switch.
            block_audit_goal = self.find_block_audit_goal()
            recovery_goal = (self.find_visual_recovery_goal()
                             if coverage_complete and block_audit_goal is None else None)
            with self.inspection_lock:
                pending_inspection_count = len(self.pending_inspections)

            search_confidence = calculate_search_confidence(
                cov, self.last_block_coverage_pct, pending_inspection_count,
                len(completion_frontiers), sectors_toured,
                self.last_block_coverage_pct < self.camera_audit_completion_pct,
                recovery_goal is not None)
            self.pub_search_confidence.publish(Float32(data=search_confidence))

            confidence_ready = (
                search_confidence >= self.completion_confidence_threshold)
            if confidence_ready:
                if self.completion_confidence_since is None:
                    self.completion_confidence_since = time.monotonic()
            else:
                self.completion_confidence_since = None
            confidence_stable = (
                self.completion_confidence_since is not None and
                time.monotonic() - self.completion_confidence_since >=
                self.completion_stability_s)
            return_needed = self.return_requested or confidence_stable
            phase = ('GAP CLOSURE' if audit_active and frontiers else
                     'VISUAL AUDIT' if audit_active else 'EXPLORE')
            self.publish_status(
                f"{phase}: Trust={search_confidence:.1f}% | Map={cov:.1f}% | "
                f"Visual={self.last_block_coverage_pct:.1f}% | "
                f"Confirmed={self.survivor_count} | Pending={pending_inspection_count} | "
                f"Gaps={len(completion_frontiers)}")

            if return_needed:
                reason = ('MANUAL' if self.return_requested else
                          'CONFIDENCE_STABLE')
                self.get_logger().info(
                    f'🎯 [JEV] DECIDE: Return to Base triggered '
                    f'(Reason={reason}, Elapsed={mission_elapsed/60.0:.1f}min, '
                    f'Trust={search_confidence:.1f}%, Coverage={cov:.1f}%, '
                    f'Confirmed Survivors={self.survivor_count}).')

                # Step 1: Fly safely through the maze corridors to South Gate opening
                self.publish_status("ACT: Returning to Base — Navigating South Gate (-5.80, -7.20)")
                self.get_logger().info('🎯 [JEV] ACT: Step 1/2 — Navigating to South Gate opening (-5.80, -7.20)...')
                self._act_navigate(-5.80, -7.20)
                act_t0 = time.time()
                while self.is_acting and rclpy.ok() and (time.time() - act_t0 < 60.0):
                    time.sleep(0.4)
                    with self.pose_lock: cur = (self.drone_x, self.drone_y)
                    if math.hypot(-5.80 - cur[0], -7.20 - cur[1]) < 0.65 or cur[1] < -7.10:
                        self.cancel_current_action()
                        break

                # Step 2: Exit through the gate directly over the Helipad
                self.publish_status("ACT: Exiting South Gate to Helipad (-5.80, -8.75)")
                self.get_logger().info('🎯 [JEV] ACT: Step 2/2 — Exiting through South Gate to Helipad (-5.80, -8.75)...')
                self._act_navigate(-5.80, -8.75)
                act_t1 = time.time()
                while self.is_acting and rclpy.ok() and (time.time() - act_t1 < 35.0):
                    time.sleep(0.4)
                    with self.pose_lock: cur = (self.drone_x, self.drone_y)
                    if math.hypot(-5.80 - cur[0], -8.75 - cur[1]) < 0.45 or cur[1] <= -8.40:
                        self.cancel_current_action()
                        break

                # Step 3: Trigger Precision Touchdown
                self.publish_status("VERIFY: Over Helipad — Commencing Touchdown & Disarm")
                self.pub_land_cmd.publish(Bool(data=True))
                self.get_logger().info('🎯 [JEV] VERIFY: Centered over Helipad! Commencing touchdown descent...')

                # Stand by for optional relaunch command
                while rclpy.ok() and not self.relaunch_requested:
                    time.sleep(0.5)
                if self.relaunch_requested:
                    self.relaunch_requested = False
                    self.get_logger().info('🚀 [JEV] Relaunch triggered! Re-engaging exploration loop...')
                    continue
                break

            if confidence_ready and not confidence_stable:
                stable_for = time.monotonic() - self.completion_confidence_since
                self.publish_status(
                    f"VERIFY TRUST: {search_confidence:.1f}% stable for "
                    f"{stable_for:.1f}/{self.completion_stability_s:.0f}s")
                time.sleep(0.5)
                continue

            # --- 3. THINK ---
            self.publish_status("THINK: Evaluating Frontier Heuristics & Obstacle Safety")
            best_action = None

            # After ten minutes, meaningful black/unknown LiDAR gaps remain
            # the first priority. Existing frontier scoring handles them here.
            if best_action is None and len(frontiers) > 0:
                valid_options = []
                for fx, fy, sz in frontiers:
                    if any(expiry > time.monotonic() and math.hypot(fx-bx, fy-by) < 1.0
                           for bx, by, expiry in self.blocked_targets):
                        continue
                    dist = math.hypot(fx - pos[0], fy - pos[1])
                    # Skip frontiers immediately under the drone footprint (< 0.35m)
                    if dist < 0.35:
                        continue

                    # Soft penalty for areas with recent action failures
                    fails = sum(1 for gx, gy in self.memory['failed_actions'] if math.hypot(fx-gx, fy-gy) < 1.0)

                    # Soft penalty for areas already visited
                    visits = sum(1 for vx, vy in self.memory['visited_locations'] if math.hypot(fx-vx, fy-vy) < 1.0)

                    # Sector balancing bonus: strongly prioritize unvisited side of compound
                    sector_bonus = 0.0
                    if not has_visited_east and fx > 0.8:
                        sector_bonus = -6.0
                    elif not has_visited_west and fx < -0.8:
                        sector_bonus = -6.0

                    # Locality hysteresis / neighborhood commitment bonus:
                    # Prioritize finishing local sectors (< 3.2m) and penalize cross-arena jumps (> 6.0m)
                    locality_bonus = -3.5 if dist < 3.2 else (3.0 if dist > 6.0 else 0.0)

                    # Multi-objective heuristic score: distance, cluster size, locality, sector, failure & visit counts
                    score = dist - (0.18 * sz) + locality_bonus + sector_bonus + (fails * 8.0) + (visits * 12.0)
                    valid_options.append({'x': fx, 'y': fy, 'score': score, 'sz': sz})

                if valid_options:
                    valid_options.sort(key=lambda x: x['score'])
                    best_action = valid_options[0]
                    self.get_logger().info(f'🧠 [JEV] THINK: Selected optimal frontier at ({best_action["x"]:.2f}, {best_action["y"]:.2f}) [score={best_action["score"]:.2f}, size={best_action["sz"]}].')

            # After the black/unknown gaps, visit mapped blocks that never
            # received a successful camera sweep.
            if (best_action is None and audit_active and
                    block_audit_goal is not None):
                rx, ry = block_audit_goal
                self.publish_status(
                    f"BLOCK AUDIT: quick 360° check at ({rx:+.1f}, {ry:+.1f})")
                best_action = {'x': rx, 'y': ry, 'sz': 0, 'kind': 'audit'}

            # Finally close sub-block camera gaps before trust can exceed 97%.
            if best_action is None and recovery_goal is not None:
                rx, ry = recovery_goal
                best_action = {'x': rx, 'y': ry, 'sz': 0, 'kind': 'recovery'}

            if best_action is None:
                def available(tx, ty):
                    return not any(expiry > time.monotonic() and math.hypot(tx-bx, ty-by) < 1.0
                                   for bx, by, expiry in self.blocked_targets)
                # If East or West sector not toured yet, navigate directly to corridor transit hub to inspect rooms
                if not has_visited_east and available(3.40, 0.0):
                    self.get_logger().info('🧠 [JEV] THINK: Prioritizing East Corridor sector transit (X=+3.40, Y=0.0) to inspect eastern rooms...')
                    best_action = {'x': 3.40, 'y': 0.0, 'sz': 100}
                elif not has_visited_west and available(-3.40, 0.0):
                    self.get_logger().info('🧠 [JEV] THINK: Prioritizing West Corridor sector transit (X=-3.40, Y=0.0)...')
                    best_action = {'x': -3.40, 'y': 0.0, 'sz': 100}
                else:
                    # Exploratory wander behavior
                    angle = (time.time() % (2 * math.pi))
                    nx = float(np.clip(pos[0] + 1.5 * math.cos(angle), -6.2, 6.2))
                    ny = float(np.clip(pos[1] + 1.5 * math.sin(angle), -6.2, 6.2))
                    self.get_logger().info(f'🧠 [JEV] THINK: No frontiers nearby. Performing exploratory wander to ({nx:.2f}, {ny:.2f}).')
                    best_action = {'x': nx, 'y': ny, 'sz': 0}

            # --- 4. DECIDE ---
            target_x, target_y = best_action['x'], best_action['y']
            action_kind = best_action.get('kind', 'frontier')
            face_target = (self.find_unknown_focus_near(target_x, target_y)
                           if audit_active else None)
            self.publish_status(f"DECIDE: Selected target waypoint ({target_x:.2f}, {target_y:.2f})")
            self.get_logger().info(f'🧠 [JEV] DECIDE: Navigating to ({target_x:.2f}, {target_y:.2f}).')

            # --- 5. ACT ---
            self.publish_status(f"ACT: Navigating corridor to ({target_x:.2f}, {target_y:.2f})")
            self.memory['expected_outcome'] = (target_x, target_y)
            self._act_navigate(target_x, target_y, face_target=face_target)

            act_start_time = time.time()
            progress_pos = pos
            stuck_time = 0.0

            # Wait for action to complete or timeout
            while self.is_acting and rclpy.ok():
                if self.return_requested:
                    self.get_logger().info('🧠 [JEV] Immediate abort of exploration: Return requested by operator!')
                    self.cancel_current_action()
                    break

                time.sleep(0.4)
                with self.pose_lock:
                    cur_pos = (self.drone_x, self.drone_y)
                dist = math.hypot(target_x - cur_pos[0], target_y - cur_pos[1])

                # Check progress
                if math.hypot(cur_pos[0]-progress_pos[0], cur_pos[1]-progress_pos[1]) < 0.12:
                    stuck_time += 0.4
                else:
                    stuck_time = 0.0
                    progress_pos = cur_pos

                if dist < 0.40:
                    self.get_logger().info(f'🧠 [JEV] VERIFY: Reached frontier proximity ({dist:.2f}m). Smooth transition.')
                    self.cancel_current_action()
                    self.goal_status = "SUCCEEDED"
                    break

                # Rapid evasive abort: if progress stalls for 1.6s or total action exceeds 25s, replan immediately
                if stuck_time >= 1.6 or (time.time() - act_start_time) > 25.0:
                    self.blocked_targets.append((target_x, target_y, time.monotonic() + 30.0))
                    self.get_logger().warning(f'⚡ [JEV] Stalled / Blocked path ahead ({stuck_time:.1f}s stalled). Rapid evasive reroute!')
                    self.cancel_current_action()
                    self.goal_status = "BLOCKED"
                    break

            # --- 6. VERIFY & 7. REASSESS ---
            self.publish_status(f"VERIFY: Waypoint execution outcome: {self.goal_status}")
            self.get_logger().info(f'🧠 [JEV] VERIFY: Action finished with status: {self.goal_status}')

            if self.goal_status == "SUCCEEDED":
                self.get_logger().info(f'🧠 [JEV] REASSESS: Action succeeded. Updating visited memory.')
                self.memory['visited_locations'].append((target_x, target_y))
                consecutive_failures = 0
                self._scan_360_at_current_position()
            else:
                self.get_logger().info(f'🧠 [JEV] REASSESS: Action failed. Updating failed memory. Soft penalty applied.')
                self.memory['failed_actions'].append((target_x, target_y))
                consecutive_failures += 1

                if consecutive_failures >= 3:
                    if self.memory['visited_locations']:
                        idx = max(0, len(self.memory['visited_locations']) - 4)
                        rx, ry = self.memory['visited_locations'][idx]
                    else:
                        rx, ry = 0.0, 0.0
                    self.get_logger().warning(f'[JEV] REASSESS: Multiple consecutive failures! Retreating to safe visited location ({rx:.2f}, {ry:.2f})')
                    self._act_navigate(rx, ry)
                    time.sleep(3.0)
                    self.cancel_current_action()
                    consecutive_failures = 0

def main(args=None):
    rclpy.init(args=args)
    node = JevDecisionLayer()
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

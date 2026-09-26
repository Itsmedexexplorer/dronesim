#!/usr/bin/env python3
"""
Nidar MAVROS Autonomous Navigator
=================================
Autonomous exploration & 2D mapping drone controller running over standard MAVROS ROS 2 topics:
  • /mavros/state                     - Verifies connection, arming, and OFFBOARD mode
  • /mavros/cmd/arming                - Service call to arm/disarm motors
  • /mavros/set_mode                  - Service call to engage OFFBOARD flight mode
  • /mavros/local_position/pose       - Subscribes to 3D pose (X, Y, Z, yaw)
  • /mavros/setpoint_raw/local        - Publishes body/local velocity targets (PositionTarget)
  • /lidar (gz.transport)             - Receives 360° LiDAR ranges for SLAM & obstacle avoidance
"""

import os
import sys
import time
import math
import signal
import heapq
import threading
from collections import deque
import numpy as np

def _clean_exit_handler(sig, frame):
    os._exit(0)

signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan as RosLaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
import tf2_ros

# Gazebo Transport for raw LiDAR scan
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.laserscan_pb2 import LaserScan as GzLaserScan
except ImportError:
    from gz.transport import Node as GzNode
    from gz.msgs.laserscan_pb2 import LaserScan as GzLaserScan

# ── GRID & EXPLORATION CONSTANTS ──
GRID_RES   = 0.05       # 5cm resolution
GRID_SIZE  = 30.0       # 30x30m grid
GRID_N     = int(GRID_SIZE / GRID_RES) # 600x600 cells
ORIGIN_X   = -GRID_SIZE / 2.0
ORIGIN_Y   = -GRID_SIZE / 2.0

UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2

TARGET_ALTITUDE = 1.50   # 1.5m indoor flight altitude
ARENA_LIMIT     = 6.40   # ±6.40m compound grid limit (confines all exploration strictly inside)
COMPOUND_MIN_X  = -6.40
COMPOUND_MAX_X  =  6.40
COMPOUND_MIN_Y  = -6.40  # North of South Gate wall (-7.5m): drone MUST NEVER fly south of -6.4m in exploration
COMPOUND_MAX_Y  =  6.40
HELIPAD_X       = -5.80  # Designated launch & landing helipad outside South Gate
HELIPAD_Y       = -8.80

# 16 room geometric centers across the 4x4 compound layout (spaced every 3.4m)
ROOM_CENTERS = [
    (-5.10, -5.10), (-1.70, -5.10), (1.70, -5.10), (5.10, -5.10),
    (-5.10, -1.70), (-1.70, -1.70), (1.70, -1.70), (5.10, -1.70),
    (-5.10,  1.70), (-1.70,  1.70), (1.70,  1.70), (5.10,  1.70),
    (-5.10,  5.10), (-1.70,  5.10), (1.70,  5.10), (5.10,  5.10)
]


def find_unexplored_rooms(g, min_unknown=20):
    """
    Inspects each of the 16 rooms in the compound.
    Returns list of (target_x, target_y, unknown_count, room_x, room_y) for rooms
    that still contain grouped unexplored unknown space.
    """
    targets = []
    half_w = int(1.20 / GRID_RES)  # 2.4m inner room window
    for rx, ry in ROOM_CENTERS:
        c_pt, r_pt = world_to_cell(rx, ry)
        if c_pt is None or r_pt is None:
            continue
        r_low = max(0, r_pt - half_w); r_high = min(GRID_N, r_pt + half_w + 1)
        c_low = max(0, c_pt - half_w); c_high = min(GRID_N, c_pt + half_w + 1)
        sub = g[r_low:r_high, c_low:c_high]
        unknown_cnt = int(np.count_nonzero(sub == UNKNOWN))
        if unknown_cnt >= min_unknown:
            # Find the best target coordinate in this room:
            best_x, best_y = rx, ry
            if g[r_pt, c_pt] != FREE:
                free_rs, free_cs = np.where(sub == FREE)
                if len(free_rs) > 0:
                    mid = len(free_rs) // 2
                    best_x, best_y = cell_to_world(free_cs[mid] + c_low, free_rs[mid] + r_low)
            best_x = float(np.clip(best_x, COMPOUND_MIN_X + 0.30, COMPOUND_MAX_X - 0.30))
            best_y = float(np.clip(best_y, COMPOUND_MIN_Y + 0.30, COMPOUND_MAX_Y - 0.30))
            targets.append((best_x, best_y, unknown_cnt, rx, ry))
    return targets


def world_to_cell(wx, wy):
    c = int(round((wx - ORIGIN_X) / GRID_RES))
    r = int(round((wy - ORIGIN_Y) / GRID_RES))
    if 0 <= c < GRID_N and 0 <= r < GRID_N:
        return (c, r)
    return None


def cell_to_world(c, r):
    wx = ORIGIN_X + c * GRID_RES
    wy = ORIGIN_Y + r * GRID_RES
    return (wx, wy)


def astar_grid_path(g, start_xy, goal_xy, downsample=2, clamp_to_indoor=True):
    """
    Computes a fast, collision-free grid path between start and goal.
    Dilates obstacles by safety margins to guarantee body clearance.
    """
    p_start = world_to_cell(start_xy[0], start_xy[1])
    p_goal  = world_to_cell(goal_xy[0], goal_xy[1])
    if not p_start or not p_goal:
        return []
    sc, sr = p_start
    gc, gr = p_goal

    H, W = g.shape
    h_down = H // downsample
    w_down = W // downsample

    blocks = g[:h_down*downsample, :w_down*downsample].reshape(h_down, downsample, w_down, downsample)
    occ_mask = np.any(blocks == OCCUPIED, axis=(1, 3))
    unk_mask = np.any(blocks == UNKNOWN, axis=(1, 3))

    # 1 iteration of 8-neighbor dilation (10cm) + soft layer (10cm): guarantees ~15cm clearance from all walls
    dilated_occ = occ_mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0: continue
            shifted = np.zeros_like(occ_mask)
            r_dst = slice(max(0, dr), min(h_down, h_down+dr))
            c_dst = slice(max(0, dc), min(w_down, w_down+dc))
            r_src = slice(max(0, -dr), min(h_down, h_down-dr))
            c_src = slice(max(0, -dc), min(w_down, w_down-dc))
            shifted[r_dst, c_dst] = occ_mask[r_src, c_src]
            dilated_occ |= shifted

    # Soft cost gradient layer (adds mild penalty near walls to maintain 15cm-20cm clearance)
    soft_occ = dilated_occ.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0: continue
            shifted = np.zeros_like(dilated_occ)
            r_dst = slice(max(0, dr), min(h_down, h_down+dr))
            c_dst = slice(max(0, dc), min(w_down, w_down+dc))
            r_src = slice(max(0, -dr), min(h_down, h_down-dr))
            c_src = slice(max(0, -dc), min(w_down, w_down-dc))
            shifted[r_dst, c_dst] = dilated_occ[r_src, c_src]
            soft_occ |= shifted

    s_r = max(0, min(h_down - 1, sr // downsample))
    s_c = max(0, min(w_down - 1, sc // downsample))
    g_r = max(0, min(h_down - 1, gr // downsample))
    g_c = max(0, min(w_down - 1, gc // downsample))

    # Robust start snapping: find closest free cell if start is within dilated boundary
    if dilated_occ[s_r, s_c]:
        for rad in range(1, 16):
            found_s = False
            for dr in range(-rad, rad + 1):
                for dc in (-rad, rad):
                    nr, nc = s_r + dr, s_c + dc
                    if 0 <= nr < h_down and 0 <= nc < w_down and not dilated_occ[nr, nc]:
                        s_r, s_c = nr, nc; found_s = True; break
                if found_s: break
            if not found_s:
                for dc in range(-rad, rad + 1):
                    for dr in (-rad, rad):
                        nr, nc = s_r + dr, s_c + dc
                        if 0 <= nr < h_down and 0 <= nc < w_down and not dilated_occ[nr, nc]:
                            s_r, s_c = nr, nc; found_s = True; break
                    if found_s: break
            if found_s: break

    # Robust goal snapping: find closest free cell in front of wall/corner if goal is in dilation
    if dilated_occ[g_r, g_c]:
        for rad in range(1, 16):
            found_g = False
            for dr in range(-rad, rad + 1):
                for dc in (-rad, rad):
                    nr, nc = g_r + dr, g_c + dc
                    if 0 <= nr < h_down and 0 <= nc < w_down and not dilated_occ[nr, nc]:
                        g_r, g_c = nr, nc; found_g = True; break
                if found_g: break
            if not found_g:
                for dc in range(-rad, rad + 1):
                    for dr in (-rad, rad):
                        nr, nc = g_r + dr, g_c + dc
                        if 0 <= nr < h_down and 0 <= nc < w_down and not dilated_occ[nr, nc]:
                            g_r, g_c = nr, nc; found_g = True; break
                    if found_g: break
            if found_g: break

    pq = [(0.0, 0.0, s_r, s_c)]
    came_from = {}
    cost_so_far = {(s_r, s_c): 0.0}

    dirs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    found = False
    while pq:
        _, current_cost, r, c = heapq.heappop(pq)
        if (r, c) == (g_r, g_c):
            found = True
            break
        if current_cost > cost_so_far.get((r, c), float('inf')):
            continue
        for dr, dc, w in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h_down and 0 <= nc < w_down:
                if dilated_occ[nr, nc]:
                    continue
                step_cost = w + (0.50 if soft_occ[nr, nc] else 0.0) + (2.5 if unk_mask[nr, nc] else 0.0)
                new_cost = current_cost + step_cost
                if new_cost < cost_so_far.get((nr, nc), float('inf')):
                    cost_so_far[(nr, nc)] = new_cost
                    h = math.hypot(nr - g_r, nc - g_c)
                    heapq.heappush(pq, (new_cost + h, new_cost, nr, nc))
                    came_from[(nr, nc)] = (r, c)

    if not found:
        return []

    curr = (g_r, g_c)
    wpts_rc = []
    while curr in came_from:
        wpts_rc.append(curr)
        curr = came_from[curr]
    wpts_rc.append((s_r, s_c))
    wpts_rc.reverse()

    wpts = [start_xy]
    if clamp_to_indoor:
        min_x, max_x = COMPOUND_MIN_X, COMPOUND_MAX_X
        min_y, max_y = COMPOUND_MIN_Y, COMPOUND_MAX_Y
    else:
        min_x, max_x = ORIGIN_X + 0.5, -ORIGIN_X - 0.5
        min_y, max_y = ORIGIN_Y + 0.5, -ORIGIN_Y - 0.5

    for r, c in wpts_rc[1:]:
        wx, wy = cell_to_world(c * downsample + downsample // 2, r * downsample + downsample // 2)
        wx = float(np.clip(wx, min_x, max_x))
        wy = float(np.clip(wy, min_y, max_y))
        if math.hypot(wx - wpts[-1][0], wy - wpts[-1][1]) >= 0.35:
            wpts.append((wx, wy))
    g_clamped = (float(np.clip(goal_xy[0], min_x, max_x)),
                 float(np.clip(goal_xy[1], min_y, max_y)))
    if math.hypot(g_clamped[0] - wpts[-1][0], g_clamped[1] - wpts[-1][1]) >= 0.20:
        wpts.append(g_clamped)

    # Safe collinear point simplification (NEVER crosses corners or walls)
    if len(wpts) > 2:
        simplified = [wpts[0]]
        for i in range(1, len(wpts) - 1):
            p_prev = simplified[-1]
            p_curr = wpts[i]
            p_next = wpts[i + 1]
            dx1 = p_curr[0] - p_prev[0]; dy1 = p_curr[1] - p_prev[1]
            dx2 = p_next[0] - p_curr[0]; dy2 = p_next[1] - p_curr[1]
            cross = abs(dx1 * dy2 - dy1 * dx2)
            dot = dx1 * dx2 + dy1 * dy2
            if cross > 0.02 or dot <= 0:
                simplified.append(p_curr)
        simplified.append(wpts[-1])
        wpts = simplified

    return wpts


def find_frontier_clusters(g, min_cluster_size=3):
    """
    Finds clusters of unexplored frontiers strictly inside the compound (|X,Y| <= 6.3m).
    Guarantees targets have clearance from occupied walls.
    """
    c_min, r_min = world_to_cell(-ARENA_LIMIT, -ARENA_LIMIT) or (0, 0)
    c_max, r_max = world_to_cell( ARENA_LIMIT,  ARENA_LIMIT) or (GRID_N, GRID_N)
    c0 = max(1, min(c_min, c_max))
    c1 = min(GRID_N - 1, max(c_min, c_max))
    r0 = max(1, min(r_min, r_max))
    r1 = min(GRID_N - 1, max(r_min, r_max))

    frontier_cells = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if g[r, c] == FREE:
                sub = g[r-1:r+2, c-1:c+2]
                sub_occ = g[max(0, r-2):min(GRID_N, r+3), max(0, c-2):min(GRID_N, c+3)]
                if np.any(sub == UNKNOWN) and not np.any(sub_occ == OCCUPIED):
                    frontier_cells.append((c, r))

    if not frontier_cells:
        return []

    visited = set()
    clusters = []
    f_set = set(frontier_cells)

    for cell in frontier_cells:
        if cell in visited:
            continue
        cluster = []
        queue = [cell]
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
        if len(cluster) >= min_cluster_size:
            avg_c = int(round(sum(c for c, _ in cluster) / float(len(cluster))))
            avg_r = int(round(sum(r for _, r in cluster) / float(len(cluster))))
            sub_c = g[max(0, avg_r-2):min(GRID_N, avg_r+3), max(0, avg_c-2):min(GRID_N, avg_c+3)]
            if np.any(sub_c == OCCUPIED):
                best_cell = min(cluster, key=lambda p: math.hypot(p[0]-avg_c, p[1]-avg_r))
                avg_c, avg_r = best_cell
            wx, wy = cell_to_world(avg_c, avg_r)
            if (COMPOUND_MIN_X + 0.30 <= wx <= COMPOUND_MAX_X - 0.30) and (COMPOUND_MIN_Y + 0.30 <= wy <= COMPOUND_MAX_Y - 0.30):
                clusters.append((wx, wy, len(cluster)))

    return clusters


def compute_lidar_avoidance(ranges, angle_min=-math.pi, angle_step=None, d_influence=0.42):
    """
    Computes body-frame collision clearance and guidance vectors from 360° LiDAR rays.
    Separates:
      - fwd_clearance: true forward flight corridor (|y_body| <= 0.20m, x_body > 0)
      - side clearances: left (d_l) and right (d_r)
      - d_rear: rear obstacle clearance
    """
    n_rays = len(ranges) if ranges is not None else 0
    if n_rays == 0:
        return 0.0, 0.0, float('inf'), float('inf'), float('inf'), float('inf')

    if angle_step is None:
        angle_step = (2 * math.pi) / n_rays

    fx = 0.0
    fy = 0.0
    fwd_clearance = float('inf')
    d_l = float('inf')
    d_r = float('inf')
    d_rear = float('inf')

    # Half-width of drone flight envelope (drone rotor radius ~0.15m + 0.05m margin)
    CORRIDOR_HALF_WIDTH = 0.20

    for i, r in enumerate(ranges):
        if math.isinf(r) or math.isnan(r) or r < 0.05 or r > 16.0:
            continue
        theta = (angle_min + i * angle_step + math.pi) % (2 * math.pi) - math.pi
        td = math.degrees(theta)

        # Body Cartesian projection
        xb = r * math.cos(theta)
        yb = r * math.sin(theta)

        # 1. Forward Flight Corridor: ONLY rays directly blocking forward movement
        if xb > 0.0 and abs(yb) <= CORRIDOR_HALF_WIDTH:
            if xb < fwd_clearance:
                fwd_clearance = xb

        # 2. Side flanks
        if 40 <= td <= 140:
            if r < d_l: d_l = r
        elif -140 <= td <= -40:
            if r < d_r: d_r = r
        elif abs(td) >= 145:
            if r < d_rear: d_rear = r

        # 3. Continuous lateral repulsive field
        if r < d_influence:
            weight = ((d_influence - r) / d_influence) ** 2
            if abs(td) >= 25:
                fy -= weight * math.sin(theta)
            elif r < 0.30:
                fx -= weight * math.cos(theta)

    return fx, fy, fwd_clearance, d_l, d_r, d_rear


def compute_escape_vector(ranges):
    """
    Computes body-frame velocity (esc_vx, esc_vy) to actively pull drone off a wall
    towards the direction of maximum clearance.
    """
    fx, fy, fwd_clearance, d_l, d_r, d_rear = compute_lidar_avoidance(ranges)
    esc_vx = -0.28 if d_rear > 0.35 else 0.0
    esc_vy = 0.25 if d_l > d_r else -0.25
    return esc_vx, esc_vy


def apply_lidar_obstacle_avoidance(vx_cmd, vy_cmd, wz_cmd, yaw_err, ranges):
    fx, fy, fwd_clearance, d_l, d_r, d_rear = compute_lidar_avoidance(ranges)
    min_side = min(d_l, d_r)
    status_suffix = ""

    # Heading alignment: If large yaw error, turn in place before accelerating forward
    if abs(yaw_err) > 1.2:
        if fwd_clearance >= 0.38 and min_side >= 0.30:
            return 0.0, 0.0, wz_cmd, " | Aligning Heading", fwd_clearance
        else:
            rep_vx = -0.15 if fwd_clearance < 0.28 and d_rear > 0.35 else 0.0
            rep_vy = float(np.clip(fy * 0.18, -0.18, 0.18))
            return rep_vx, rep_vy, wz_cmd, f" | Turn Clearance ({min(fwd_clearance, min_side):.2f}m)", fwd_clearance

    # Head-on wall collision avoidance (~22cm trigger)
    if fwd_clearance < 0.24:
        vx_cmd = -0.16 if d_rear > 0.35 else 0.0
        vy_cmd = 0.20 if d_l > d_r else -0.20
        status_suffix = f" | HEAD-ON WALL ({fwd_clearance:.2f}m) -> Slide"

    elif fwd_clearance < 0.38:
        speed_factor = max(0.35, (fwd_clearance - 0.22) / (0.38 - 0.22))
        vx_cmd = min(vx_cmd, max(0.18, vx_cmd * speed_factor))
        status_suffix = f" | Decel Wall ({fwd_clearance:.2f}m)"

    # Doorway & Corridor Auto-Centering (Dual-Wall Gateway Transit)
    if d_l < 0.70 and d_r < 0.70:
        # Both posts/walls detected: Center precisely between them
        diff = d_l - d_r
        vy_cmd = float(np.clip(diff * 0.30, -0.18, 0.18))
        if fwd_clearance > 0.30:
            vx_cmd = max(0.25, vx_cmd)
        status_suffix += " | Doorway Centered"
    elif d_l < 0.22:
        # Too close to left wall (gap < 22cm): push right to maintain ~15cm+ gap
        push = -0.40 * (0.22 - d_l)
        vy_cmd = min(vy_cmd, float(np.clip(push, -0.20, 0.0)))
        status_suffix += f" | Left Wall Repel ({d_l:.2f}m)"
    elif d_r < 0.22:
        # Too close to right wall (gap < 22cm): push left to maintain ~15cm+ gap
        push = 0.40 * (0.22 - d_r)
        vy_cmd = max(vy_cmd, float(np.clip(push, 0.0, 0.20)))
        status_suffix += f" | Right Wall Repel ({d_r:.2f}m)"
    else:
        # Open space lateral smoothing
        vy_cmd += float(np.clip(fy * 0.15, -0.15, 0.15))

    return vx_cmd, vy_cmd, wz_cmd, status_suffix, fwd_clearance


def apply_indoor_boundary_fence(cx, cy, yaw, vx, vy):
    """
    Confines velocity commands strictly within the indoor compound grid during exploration.
    Converts body velocities to world frame, enforces pushback against arena borders,
    and maps back to body frame.
    """
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    vwx = vx * cos_y - vy * sin_y
    vwy = vx * sin_y + vy * cos_y

    MARGIN = 0.35
    PUSH = 0.25

    if cx < COMPOUND_MIN_X + MARGIN:
        vwx = max(vwx, PUSH)
    elif cx > COMPOUND_MAX_X - MARGIN:
        vwx = min(vwx, -PUSH)

    if cy < COMPOUND_MIN_Y + MARGIN:
        vwy = max(vwy, PUSH)
    elif cy > COMPOUND_MAX_Y - MARGIN:
        vwy = min(vwy, -PUSH)

    vx_fenced =  vwx * cos_y + vwy * sin_y
    vy_fenced = -vwx * sin_y + vwy * cos_y
    return vx_fenced, vy_fenced


class NidarMavrosNavigator(Node):
    def __init__(self):
        super().__init__('nidar_mavros_navigator')
        self.get_logger().info('Initializing Nidar MAVROS Autonomous Navigator...')

        # State
        self.state_connected = False
        self.state_armed = False
        self.state_mode = "MANUAL"

        # Pose & Odometry
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_roll = 0.0
        self.drone_pitch = 0.0
        self.drone_yaw = 0.0
        self.home_x = HELIPAD_X
        self.home_y = HELIPAD_Y
        self.is_searching_room = False
        self.pose_lock = threading.Lock()

        # Occupancy Grid SLAM with Hit Evidence Accumulation
        self.grid = np.zeros((GRID_N, GRID_N), dtype=np.uint8)
        self.hit_counts = np.zeros((GRID_N, GRID_N), dtype=np.int16)
        self.grid_lock = threading.Lock()
        self.visited_heatmap = np.zeros((GRID_N, GRID_N), dtype=np.int32)
        self.heatmap_lock = threading.Lock()

        # Raw LiDAR
        self.lidar_ranges = []
        self.lidar_lock = threading.Lock()

        # Mission state
        self.mission_state = "TAKEOFF"
        self.waypoints = []
        self.wpt_idx = 0
        self.latest_frontiers = []
        self.mission_lock = threading.Lock()

        # Stuck Detection & Recovery Constraints
        self.pose_history = deque(maxlen=250)   # (timestamp, x, y) samples
        self.stuck_blacklist = []               # [(x, y, expire_time)]

        # Survivor / Human Tracking (prevents laser raycasts from stamping black wall dots on survivors)
        self.survivors = []
        self.survivors_lock = threading.Lock()
        self.sub_survivors = self.create_subscription(
            MarkerArray, '/nidar/survivor_markers', self.cb_survivors, 10
        )

        # ── ROS 2 Subscriptions ──
        self.sub_state = self.create_subscription(State, '/mavros/state', self.cb_state, 10)
        self.sub_pose  = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.cb_pose, 10)

        # ── ROS 2 Publishers ──
        self.pub_setpoint = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)
        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=QoSReliabilityPolicy.RELIABLE)
        self.pub_map = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self.pub_path = self.create_publisher(Path, '/nidar/path', 1)
        self.pub_lidar = self.create_publisher(RosLaserScan, '/nidar/lidar_scan', 10)
        self.pub_frontiers = self.create_publisher(MarkerArray, '/nidar/frontiers', 1)
        self.pub_drone_marker = self.create_publisher(Marker, '/nidar/drone_marker', 1)
        self.pub_mission_complete = self.create_publisher(Bool, '/nidar/mission_complete', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.latest_frontiers = []

        # 2Hz Map & TF Visualization Timer
        self.timer_vis = self.create_timer(0.5, self.publish_visualization)

        # ── ROS 2 Service Clients ──
        self.cli_arm  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.cli_mode = self.create_client(SetMode, '/mavros/set_mode')

        # ── Gazebo LiDAR Subscription ──
        self.gz_node = GzNode()
        self.gz_node.subscribe(GzLaserScan, '/nidar/lidar', self.cb_lidar)

        # Main Mission Thread
        self.running = True
        self.thread = threading.Thread(target=self.mission_loop, daemon=True)
        self.thread.start()

        self.get_logger().info('✅ Navigator Node Ready! Map published on /map, TF on /tf')

    def cb_survivors(self, msg: MarkerArray):
        survs = []
        for m in msg.markers:
            if m.type == Marker.CYLINDER and m.color.r > 0.8:
                survs.append((m.pose.position.x, m.pose.position.y))
        with self.survivors_lock:
            self.survivors = survs

    def cb_state(self, msg: State):
        self.state_connected = msg.connected
        self.state_armed = msg.armed
        self.state_mode = msg.mode

    def cb_pose(self, msg: PoseStamped):
        with self.pose_lock:
            p = msg.pose.position
            o = msg.pose.orientation
            self.drone_x = p.x
            self.drone_y = p.y
            self.drone_z = p.z

            # Roll (x-axis)
            sinr_cosp = 2.0 * (o.w * o.x + o.y * o.z)
            cosr_cosp = 1.0 - 2.0 * (o.x * o.x + o.y * o.y)
            self.drone_roll = math.atan2(sinr_cosp, cosr_cosp)

            # Pitch (y-axis)
            sinp = 2.0 * (o.w * o.y - o.z * o.x)
            if abs(sinp) >= 1.0:
                self.drone_pitch = math.copysign(math.pi / 2.0, sinp)
            else:
                self.drone_pitch = math.asin(sinp)

            # Yaw (z-axis)
            siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
            cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
            self.drone_yaw = math.atan2(siny_cosp, cosy_cosp)

            if self.home_x is None and p.z >= 0.1:
                self.home_x = p.x
                self.home_y = p.y

            # Update visited heatmap (bounded to avoid infinite buildup)
            cell = world_to_cell(p.x, p.y)
            if cell:
                with self.heatmap_lock:
                    if self.visited_heatmap[cell[1], cell[0]] < 40:
                        self.visited_heatmap[cell[1], cell[0]] += 1

    def cb_lidar(self, msg: GzLaserScan):
        ranges = list(msg.ranges)
        with self.lidar_lock:
            self.lidar_ranges = ranges

        # Raycasting into Occupancy Grid
        with self.pose_lock:
            cx, cy = self.drone_x, self.drone_y
            yaw = self.drone_yaw
            roll = self.drone_roll
            pitch = self.drone_pitch

        # Tilt Gating: Skip scan if drone tilts > 4.5 deg (0.08 rad) during turns or hovering.
        # This completely prevents planar LiDAR rays from dipping into the floor and creating false phantom walls.
        if abs(roll) > 0.08 or abs(pitch) > 0.08:
            return

        p_drone = world_to_cell(cx, cy)
        if not p_drone: return

        angle_min = msg.angle_min
        angle_step = msg.angle_step
        max_r = min(msg.range_max, 10.0)

        with self.grid_lock:
            for i, r in enumerate(ranges):
                if math.isinf(r) or math.isnan(r) or r <= msg.range_min:
                    continue
                ray_ang = (yaw + angle_min + i * angle_step + math.pi) % (2 * math.pi) - math.pi
                hit_wall = (r < max_r - 0.10)
                dist_ray = min(r, max_r)

                cos_a = math.cos(ray_ang)
                sin_a = math.sin(ray_ang)

                # Dynamic Free space raycast: clears transient noise while strictly preserving confirmed walls
                n_steps = max(1, int(dist_ray / (GRID_RES * 1.5)))
                for step in range(1, n_steps):
                    frac = step / float(n_steps)
                    wx = cx + dist_ray * frac * cos_a
                    wy = cy + dist_ray * frac * sin_a
                    c = world_to_cell(wx, wy)
                    if c:
                        col, row = c
                        # Only clear if it is NOT a confirmed solid wall (hit_counts < 5)
                        if self.hit_counts[row, col] > 0 and self.hit_counts[row, col] < 5:
                            self.hit_counts[row, col] -= 1
                            if self.hit_counts[row, col] == 0:
                                self.grid[row, col] = FREE
                        elif self.hit_counts[row, col] == 0:
                            self.grid[row, col] = FREE

                # Occupied wall endpoint
                if hit_wall:
                    wx = cx + dist_ray * cos_a
                    wy = cy + dist_ray * sin_a

                    # Do not stamp wall occupancy onto detected humans/survivors
                    is_survivor = False
                    with self.survivors_lock:
                        for sx, sy in self.survivors:
                            if math.hypot(wx - sx, wy - sy) < 0.55:
                                is_survivor = True
                                break

                    c = world_to_cell(wx, wy)
                    if c:
                        col, row = c
                        if is_survivor:
                            self.hit_counts[row, col] = 0
                            self.grid[row, col] = FREE
                        else:
                            # Solid wall evidence accumulation: requires 5 hits to become confirmed OCCUPIED
                            self.hit_counts[row, col] = min(30, self.hit_counts[row, col] + 3)
                            if self.hit_counts[row, col] >= 5:
                                self.grid[row, col] = OCCUPIED

        # Publish standard ROS 2 LaserScan for RViz / Foxglove
        try:
            ros_scan = RosLaserScan()
            ros_scan.header.stamp = self.get_clock().now().to_msg()
            ros_scan.header.frame_id = "base_link"
            ros_scan.angle_min = float(msg.angle_min)
            ros_scan.angle_max = float(msg.angle_min + len(ranges) * msg.angle_step)
            ros_scan.angle_increment = float(msg.angle_step)
            ros_scan.time_increment = 0.0
            ros_scan.scan_time = 0.05
            ros_scan.range_min = float(msg.range_min)
            ros_scan.range_max = float(min(msg.range_max, 16.0))
            ros_scan.ranges = [float(r) if (not math.isinf(r) and not math.isnan(r)) else float('inf') for r in ranges]
            self.pub_lidar.publish(ros_scan)
        except Exception:
            pass

    def publish_visualization(self):
        try:
            now_msg = self.get_clock().now().to_msg()

            # 1. Publish nav_msgs/OccupancyGrid on /map
            with self.grid_lock:
                grid_copy = self.grid.copy()

            # Clear any black obstacle dots at confirmed human/survivor locations
            with self.survivors_lock:
                survs = list(self.survivors)
            for sx, sy in survs:
                sc, sr = world_to_cell(sx, sy)
                if sc is not None:
                    rad_cells = int(0.55 / GRID_RES)
                    r_min, r_max = max(0, sr - rad_cells), min(GRID_N, sr + rad_cells + 1)
                    c_min, c_max = max(0, sc - rad_cells), min(GRID_N, sc + rad_cells + 1)
                    for rr in range(r_min, r_max):
                        for cc in range(c_min, c_max):
                            if (rr - sr)**2 + (cc - sc)**2 <= rad_cells**2:
                                if grid_copy[rr, cc] == OCCUPIED:
                                    grid_copy[rr, cc] = FREE

            map_msg = OccupancyGrid()
            map_msg.header.stamp = now_msg
            map_msg.header.frame_id = "map"
            map_msg.info.map_load_time = now_msg
            map_msg.info.resolution = float(GRID_RES)
            map_msg.info.width = int(GRID_N)
            map_msg.info.height = int(GRID_N)
            map_msg.info.origin.position.x = float(ORIGIN_X)
            map_msg.info.origin.position.y = float(ORIGIN_Y)
            map_msg.info.origin.position.z = 0.0
            map_msg.info.origin.orientation.w = 1.0

            flat_data = np.full(GRID_N * GRID_N, -1, dtype=np.int8)
            raw = grid_copy.ravel()
            flat_data[raw == FREE] = 0
            flat_data[raw == OCCUPIED] = 100
            map_msg.data = flat_data.tolist()
            self.pub_map.publish(map_msg)

            # 2. Broadcast TF (map -> base_link)
            with self.pose_lock:
                cx, cy, cz, yaw = self.drone_x, self.drone_y, self.drone_z, self.drone_yaw

            t = TransformStamped()
            t.header.stamp = now_msg
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"
            t.transform.translation.x = float(cx)
            t.transform.translation.y = float(cy)
            t.transform.translation.z = float(cz)

            cyaw = math.cos(yaw * 0.5)
            syaw = math.sin(yaw * 0.5)
            t.transform.rotation.w = float(cyaw)
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = float(syaw)
            self.tf_broadcaster.sendTransform(t)

            # 3. Publish Planned Path on /nidar/path
            with self.mission_lock:
                wpts = list(self.waypoints)
                idx = self.wpt_idx

            path_msg = Path()
            path_msg.header.stamp = now_msg
            path_msg.header.frame_id = "map"
            if wpts:
                for (wx, wy) in wpts[max(0, idx - 1):]:
                    ps = PoseStamped()
                    ps.header.frame_id = "map"
                    ps.pose.position.x = float(wx)
                    ps.pose.position.y = float(wy)
                    ps.pose.position.z = 0.15
                    ps.pose.orientation.w = 1.0
                    path_msg.poses.append(ps)
            self.pub_path.publish(path_msg)

            # 4. Publish Frontier Cluster Markers on /nidar/frontiers
            ma = MarkerArray()
            del_marker = Marker()
            del_marker.action = Marker.DELETEALL
            ma.markers.append(del_marker)

            for i, cl in enumerate(self.latest_frontiers[:15]):
                fx, fy = cl[0], cl[1]
                m = Marker()
                m.header.stamp = now_msg
                m.header.frame_id = "map"
                m.ns = "frontiers"
                m.id = i + 1
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(fx)
                m.pose.position.y = float(fy)
                m.pose.position.z = float(TARGET_ALTITUDE)
                m.pose.orientation.w = 1.0
                m.scale.x = 0.35
                m.scale.y = 0.35
                m.scale.z = 0.35
                m.color.r = 1.0
                m.color.g = 0.85
                m.color.b = 0.0
                m.color.a = 0.85
                ma.markers.append(m)
            self.pub_frontiers.publish(ma)

            # 5. Publish Complete Solid Triangle on /nidar/drone_marker
            m_drone = Marker()
            m_drone.header.stamp = now_msg
            m_drone.header.frame_id = "base_link"
            m_drone.ns = "drone"
            m_drone.id = 0
            m_drone.type = Marker.TRIANGLE_LIST
            m_drone.action = Marker.ADD
            m_drone.scale.x = 1.0
            m_drone.scale.y = 1.0
            m_drone.scale.z = 1.0
            # High-visibility Electric Cyan solid body
            m_drone.color.r = 0.0
            m_drone.color.g = 0.90
            m_drone.color.b = 1.0
            m_drone.color.a = 0.95
            m_drone.pose.orientation.w = 1.0

            # Complete 3-Sided Triangle: 0.48m rear base, 0.62m total length to forward apex
            apex = Point(x=0.38, y=0.0, z=0.04)
            rear_left = Point(x=-0.24, y=0.24, z=0.04)
            rear_right = Point(x=-0.24, y=-0.24, z=0.04)

            m_drone.points = [
                apex, rear_left, rear_right,
                apex, rear_right, rear_left
            ]
            self.pub_drone_marker.publish(m_drone)

        except Exception:
            pass

    def arm_vehicle(self, arm=True):
        if not self.cli_arm.wait_for_service(timeout_sec=2.0):
            return False
        req = CommandBool.Request()
        req.value = arm
        future = self.cli_arm.call_async(req)
        start_t = time.time()
        while not future.done() and (time.time() - start_t) < 2.0:
            time.sleep(0.02)
        res = future.result()
        return res is not None and res.success

    def set_flight_mode(self, mode_str="OFFBOARD"):
        if not self.cli_mode.wait_for_service(timeout_sec=2.0):
            return False
        req = SetMode.Request()
        req.custom_mode = mode_str
        future = self.cli_mode.call_async(req)
        start_t = time.time()
        while not future.done() and (time.time() - start_t) < 2.0:
            time.sleep(0.02)
        res = future.result()
        return res is not None and res.mode_sent

    def send_target_vel(self, vx, vy, vz, wz):
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
        # Ignore position & acceleration, use velocity & yaw_rate
        msg.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
                         PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                         PositionTarget.IGNORE_YAW)
        msg.velocity.x = float(vx)
        msg.velocity.y = float(vy)
        msg.velocity.z = float(vz)
        msg.yaw_rate   = float(wz)
        self.pub_setpoint.publish(msg)

    def coverage_pct(self):
        with self.grid_lock:
            # Compound is inside ±7.5m
            c0, r0 = world_to_cell(-ARENA_LIMIT, -ARENA_LIMIT) or (0, 0)
            c1, r1 = world_to_cell( ARENA_LIMIT,  ARENA_LIMIT) or (GRID_N, GRID_N)
            sub = self.grid[r0:r1, c0:c1]
            total = sub.size
            explored = np.count_nonzero(sub != UNKNOWN)
            return (explored / float(total)) * 100.0 if total > 0 else 0.0

    def is_in_blacklist(self, x, y, radius=1.3):
        now_t = time.time()
        for bx, by, expiry in self.stuck_blacklist:
            if now_t < expiry and math.hypot(x - bx, y - by) < radius:
                return True
        return False

    def plan_rth_path(self, g, cx, cy):
        """
        Plans collision-free exit route from anywhere in the compound to the Launch Helipad:
          Step 1: If deep inside the compound (cy > -6.5), routes via A* to South Gate vestibule (-5.80, -6.00).
          Step 2: Appends South Gate gateway transit (-5.80, -7.20).
          Step 3: Appends Touchdown Helipad (-5.80, -8.80).
        """
        gate_inner = (-5.80, -6.00)
        gate_thresh = (-5.80, -7.20)
        helipad = (HELIPAD_X, HELIPAD_Y)

        if cy <= -6.5:
            return [(cx, cy), helipad]

        path_to_gate = astar_grid_path(g, (cx, cy), gate_inner, clamp_to_indoor=False)
        if path_to_gate and len(path_to_gate) >= 2:
            res = list(path_to_gate)
            res.append(gate_thresh)
            res.append(helipad)
            return res
        else:
            return [(cx, cy), gate_inner, gate_thresh, helipad]

    def perform_stuck_recovery(self, cx, cy, ranges):
        """
        Decisive 3-Stage Recovery Maneuver when stuck:
          1. Blacklist current coordinates AND destination goal for 60-90s to prevent looping back.
          2. Decisive Reverse & Obstacle Pushback (1.5s): Drives backwards out of corner/doorway.
          3. 360° LiDAR Sweep (2.5s): Rotates to scan surrounding space and clear occlusion.
          4. Clear active path & pose history to force fresh alternative corridor selection.
        """
        now_t = time.time()
        self.get_logger().warning(
            f'🚨 STUCK DETECTED near ({cx:.2f}, {cy:.2f}) [displacement < 0.22m over 6s]!'
        )
        self.get_logger().warning('🔄 Initiating 3-Stage Recovery: Blacklist -> Reverse Out -> 360° Sweep')

        # Stage 1: Blacklist current coordinates AND active destination goal
        self.stuck_blacklist.append((cx, cy, now_t + 60.0))
        with self.mission_lock:
            if self.waypoints and len(self.waypoints) > 0:
                dest = self.waypoints[-1]
                self.stuck_blacklist.append((dest[0], dest[1], now_t + 90.0))
                self.get_logger().warning(f'🚫 Blacklisted unreachable goal at ({dest[0]:.2f}, {dest[1]:.2f}) for 90s to prevent looping!')

        c_cell, r_cell = world_to_cell(cx, cy)
        if c_cell is not None and r_cell is not None:
            with self.heatmap_lock:
                r_low = max(0, r_cell - 4); r_high = min(GRID_N, r_cell + 5)
                c_low = max(0, c_cell - 4); c_high = min(GRID_N, c_cell + 5)
                self.visited_heatmap[r_low:r_high, c_low:c_high] = np.minimum(50, self.visited_heatmap[r_low:r_high, c_low:c_high] + 30)

        # Stage 2: Decisive Reverse & Wall Pushback for 1.5 seconds (~45 steps)
        self.get_logger().info('↩️ [Recovery 1/2] Reversing clear of obstacles...')
        with self.pose_lock:
            cur_yaw = self.drone_yaw
        for _ in range(45):
            esc_vx, esc_vy = compute_escape_vector(ranges)
            esc_vx = min(esc_vx, -0.28)  # ensure backward velocity
            esc_vx, esc_vy = apply_indoor_boundary_fence(cx, cy, cur_yaw, esc_vx, esc_vy)
            self.send_target_vel(esc_vx, esc_vy, 0.0, 0.0)
            time.sleep(0.033)

        # Stage 3: Sweep 360° Scan for 2.5 seconds (~75 steps) to update SLAM & discover open corridors
        self.get_logger().info('🔄 [Recovery 2/2] Sweeping 360° LiDAR scan to discover alternative routes...')
        for _ in range(75):
            self.send_target_vel(0.0, 0.0, 0.0, 1.4)
            time.sleep(0.033)

        # Stage 4: Reset path & watchdog history
        with self.mission_lock:
            self.waypoints = []
            self.wpt_idx = 0
        self.pose_history.clear()
        self.get_logger().info('✅ Recovery Complete: Previous zone & unreachable goal blacklisted; selecting fresh corridor.')

    def mission_loop(self):
        self.get_logger().info('🚀 Mission Loop Started: Waiting for MAVROS...')
        while self.running and not self.state_connected:
            time.sleep(0.1)

        # ── 1. Offboard Warmup: Send setpoints BEFORE requesting OFFBOARD mode (PX4 requirement) ──
        self.get_logger().info('📡 Sending 20 heartbeat setpoints to MAVROS...')
        for _ in range(20):
            self.send_target_vel(0.0, 0.0, 0.0, 0.0)
            time.sleep(0.05)

        # ── 2. Arm & Engage OFFBOARD mode ──
        self.get_logger().info('⚙️ Arming vehicle & engaging OFFBOARD mode...')
        self.set_flight_mode("OFFBOARD")
        self.arm_vehicle(True)

        while self.running:
            with self.pose_lock:
                cx, cy, cz = self.drone_x, self.drone_y, self.drone_z
                yaw = self.drone_yaw
            with self.lidar_lock:
                ranges = list(self.lidar_ranges)
            with self.grid_lock:
                g = self.grid.copy()

            cov = self.coverage_pct()

            # ── STATE 1: TAKEOFF TO 1.5M ──
            if self.mission_state == "TAKEOFF":
                alt_err = TARGET_ALTITUDE - cz
                if cz < 1.35:
                    vz = float(np.clip(1.3 * alt_err, 0.40, 0.70))
                    self.send_target_vel(0.0, 0.0, vz, 0.0)
                    self.get_logger().info(f'🛫 Takeoff climb: alt={cz:.2f}m / {TARGET_ALTITUDE:.2f}m (vz={vz:.2f} m/s)')
                else:
                    self.get_logger().info(f'🛫 Takeoff Complete at {cz:.2f}m!')
                    if cy < -6.5:
                        self.get_logger().info('🚪 [ENTRY SEQUENCE] Penetrating facility through 2.0m South Gate...')
                        self.mission_state = "ENTRY"
                        with self.mission_lock:
                            self.waypoints = [
                                (-5.80, -7.20),  # South Gate threshold
                                (-5.10, -5.50),  # Room 0_0 vestibule
                                (-5.10, -3.40),  # Main Corridor Hub
                            ]
                            self.wpt_idx = 0
                    else:
                        self.get_logger().info('Starting Frontier Exploration inside compound...')
                        self.mission_state = "EXPLORING"
                    self.pose_history.clear()
                time.sleep(0.03)
                continue

            # ── STATE 1B: ENTRY SEQUENCE THROUGH SOUTH GATE INTO COMPOUND ──
            elif self.mission_state == "ENTRY":
                with self.mission_lock:
                    wpts = list(self.waypoints)
                    idx  = self.wpt_idx

                if wpts and idx < len(wpts):
                    target = wpts[idx]
                    tx, ty = target
                    dx = tx - cx
                    dy = ty - cy
                    dist = math.hypot(dx, dy)

                    if dist <= 0.40:
                        idx += 1
                        with self.mission_lock:
                            self.wpt_idx = idx
                        if idx >= len(wpts):
                            self.get_logger().info('🏰 [INSIDE COMPOUND] Penetration successful! Active in main corridor. Launching exploration...')
                            self.mission_state = "EXPLORING"
                            with self.mission_lock:
                                self.waypoints = []
                                self.wpt_idx = 0
                            self.pose_history.clear()
                            continue

                    heading = math.atan2(dy, dx)
                    yaw_err = (heading - yaw + math.pi) % (2 * math.pi) - math.pi

                    if abs(yaw_err) < 0.08:
                        wz = 0.0
                    else:
                        wz = float(np.clip(1.2 * yaw_err, -0.65, 0.65))

                    heading_scale = max(0.20, math.cos(yaw_err))
                    speed = min(0.45, max(0.22, 0.80 * dist))
                    vx = speed * heading_scale
                    vx, vy, wz, _, _ = apply_lidar_obstacle_avoidance(vx, 0.0, wz, yaw_err, ranges)

                    alt_err = TARGET_ALTITUDE - cz
                    vz = float(np.clip(1.3 * alt_err, -0.45, 0.45))
                    self.send_target_vel(vx, vy, vz, wz)
                else:
                    self.get_logger().info('🏰 [INSIDE COMPOUND] Penetration complete. Launching exploration...')
                    self.mission_state = "EXPLORING"
                    with self.mission_lock:
                        self.waypoints = []
                        self.wpt_idx = 0
                    self.pose_history.clear()
                time.sleep(0.03)
                continue

            # ── STATE 2: EXPLORING (A* FRONTIER TRANSIT + TARGETED UNEXPLORED ROOM SEARCH) ──
            elif self.mission_state == "EXPLORING":
                with self.survivors_lock:
                    num_surv = len(self.survivors)

                # Sliding-window displacement watchdog: record history & prune expired blacklist
                now_t = time.time()
                self.pose_history.append((now_t, cx, cy))
                self.stuck_blacklist = [b for b in self.stuck_blacklist if now_t < b[2]]

                is_stuck = False
                for t_old, x_old, y_old in self.pose_history:
                    if 5.5 <= (now_t - t_old) <= 8.5:
                        disp = math.hypot(cx - x_old, cy - y_old)
                        if disp < 0.22:
                            is_stuck = True
                            break

                if is_stuck:
                    self.perform_stuck_recovery(cx, cy, ranges)
                    continue

                with self.mission_lock:
                    wpts = list(self.waypoints)
                    idx  = self.wpt_idx

                # If following an active A* transit path:
                if wpts and idx < len(wpts):
                    target = wpts[idx]
                    tx, ty = target
                    dx = tx - cx
                    dy = ty - cy
                    dist = math.hypot(dx, dy)

                    # Lookahead segment projection: advance past waypoints if we are already past them or closer to next
                    while idx < len(wpts) - 1:
                        next_target = wpts[idx + 1]
                        next_dist = math.hypot(next_target[0] - cx, next_target[1] - cy)
                        seg_dx = next_target[0] - wpts[idx][0]
                        seg_dy = next_target[1] - wpts[idx][1]
                        seg_len = math.hypot(seg_dx, seg_dy)
                        if seg_len > 0.01:
                            proj = ((cx - wpts[idx][0]) * seg_dx + (cy - wpts[idx][1]) * seg_dy) / (seg_len ** 2)
                            if proj >= 0.70 or next_dist < dist * 0.85:
                                idx += 1
                                target = wpts[idx]
                                tx, ty = target
                                dx = tx - cx; dy = ty - cy
                                dist = math.hypot(dx, dy)
                                continue
                        if dist <= 0.40:
                            idx += 1
                            target = wpts[idx]
                            tx, ty = target
                            dx = tx - cx; dy = ty - cy
                            dist = math.hypot(dx, dy)
                            continue
                        break

                    if dist <= 0.35 and idx == len(wpts) - 1:
                        idx += 1
                        wpts = []
                        if self.is_searching_room:
                            self.get_logger().info('🔄 Reached targeted room: performing 360° LiDAR & visual sweep for survivors...')
                            for _ in range(70):
                                self.send_target_vel(0.0, 0.0, 0.0, 1.2)
                                time.sleep(0.033)
                            self.is_searching_room = False

                    with self.mission_lock:
                        self.wpt_idx = idx

                    if wpts and idx < len(wpts):
                        target_heading = math.atan2(dy, dx)
                        yaw_err = (target_heading - yaw + math.pi) % (2 * math.pi) - math.pi

                        # Yaw deadband: small error (< 0.08 rad / ~4.5 deg) keeps zero yaw rate to prevent twitching
                        if abs(yaw_err) < 0.08:
                            wz = 0.0
                        else:
                            wz = float(np.clip(1.15 * yaw_err, -0.75, 0.75))

                        # Coordinated forward cruise: smooth speed scaled by heading alignment
                        heading_scale = max(0.20, math.cos(yaw_err))
                        cruise = min(0.55, max(0.25, 0.80 * dist))
                        vx = cruise * heading_scale
                        vy = 0.0

                        # Active 360° LiDAR Obstacle Tackling & Proportional Corridor Centering
                        vx, vy, wz, status, fwd_clr = apply_lidar_obstacle_avoidance(vx, vy, wz, yaw_err, ranges)

                        # Immediate path change & recovery upon encountering a wall:
                        # If an unexpected obstacle or wall blocks forward movement, abort blocked path and recover
                        if fwd_clr < 0.28 or ("HEAD-ON WALL" in status and dist > 0.40):
                            self.get_logger().warning(f'🚧 Wall encountered at {fwd_clr:.2f}m! Aborting blocked path and replanning...')
                            self.perform_stuck_recovery(cx, cy, ranges)
                            continue

                        # Enforce indoor compound boundary geofence
                        vx, vy = apply_indoor_boundary_fence(cx, cy, yaw, vx, vy)

                        alt_err = TARGET_ALTITUDE - cz
                        vz = float(np.clip(1.3 * alt_err, -0.45, 0.45))
                        self.send_target_vel(vx, vy, vz, wz)
                        time.sleep(0.03)
                        continue

                # Evaluate remaining grouped unexplored areas and boundary frontiers
                unexplored_groups = [rm for rm in find_unexplored_rooms(g, min_unknown=20)
                                     if not self.is_in_blacklist(rm[0], rm[1])]

                clusters = find_frontier_clusters(g, min_cluster_size=3)
                self.latest_frontiers = clusters
                valid_clusters = [cl for cl in clusters if not self.is_in_blacklist(cl[0], cl[1])]

                # ── PRIMARY OBJECTIVE CHECK: 97% COVERAGE MILESTONE OR ZERO UNEXPLORED ──
                # Triggers Return-To-Home as soon as ~97% coverage is achieved!
                if cov >= 97.0 or (not unexplored_groups and not valid_clusters):
                    self.get_logger().info(
                        f'🏁 [97% COVERAGE MILESTONE ACHIEVED] Compound {cov:.1f}% explored ({num_surv} survivors located)! Initiating Return-To-Home...'
                    )
                    self.mission_state = "RTH"
                    self.waypoints = self.plan_rth_path(g, cx, cy)
                    self.wpt_idx = 0
                    self.pose_history.clear()
                    continue

                # Priority 1: Clear closest grouped unexplored area (shortest path first)
                if unexplored_groups:
                    # Sort primarily by euclidean distance to the drone, with small bias for unknown cell volume
                    unexplored_groups.sort(key=lambda rm: math.hypot(rm[3] - cx, rm[4] - cy) - min(1.5, 0.015 * rm[2]))
                    chosen_room_path = []
                    chosen_rm = None
                    for bx, by, u_cnt, rx, ry in unexplored_groups:
                        path = astar_grid_path(g, (cx, cy), (bx, by))
                        if len(path) >= 2:
                            chosen_room_path = path
                            chosen_rm = (bx, by, u_cnt, rx, ry)
                            break
                    if chosen_room_path and chosen_rm is not None:
                        bx, by, u_cnt, rx, ry = chosen_rm
                        dist_rm = math.hypot(rx - cx, ry - cy)
                        self.get_logger().info(
                            f'🧭 [SHORTEST PATH TO ROOM] Routing to Room ({rx:+.2f}, {ry:+.2f}) [dist={dist_rm:.2f}m, {u_cnt} unknown cells] | Survivors: {num_surv} | Cov: {cov:.1f}%'
                        )
                        with self.mission_lock:
                            self.waypoints = chosen_room_path
                            self.wpt_idx = 0
                            self.is_searching_room = True
                        continue

                # Priority 3: Standard Frontier Exploration
                if valid_clusters:
                    # If closest frontier is already within sweep range (<= 0.45m), face and sweep it!
                    closest_cl = min(valid_clusters, key=lambda c: math.hypot(c[0]-cx, c[1]-cy))
                    d_closest = math.hypot(closest_cl[0]-cx, closest_cl[1]-cy)
                    if d_closest <= 0.45:
                        fx, fy, _ = closest_cl
                        target_heading = math.atan2(fy - cy, fx - cx)
                        yaw_err = (target_heading - yaw + math.pi) % (2 * math.pi) - math.pi
                        wz = float(np.clip(1.4 * yaw_err, -0.85, 0.85))
                        vx_cmd = 0.12 * math.cos(yaw_err)
                        vx, vy = apply_indoor_boundary_fence(cx, cy, yaw, vx_cmd, 0.0)
                        self.send_target_vel(vx, vy, 0.0, wz)
                        time.sleep(0.03)
                        continue

                    def cluster_cost(cl):
                        fx, fy, c_size = cl
                        d = math.hypot(fx - cx, fy - cy)
                        c_pt, r_pt = world_to_cell(fx, fy)
                        heat = 0
                        if c_pt is not None and r_pt is not None:
                            with self.heatmap_lock:
                                heat = int(self.visited_heatmap[r_pt, c_pt])
                        return d + min(6.0, 0.05 * heat) - min(4.0, 0.03 * c_size)

                    valid_clusters.sort(key=cluster_cost)
                    transit_path = []
                    chosen = None
                    for cl in valid_clusters:
                        fx, fy, c_size = cl
                        if math.hypot(fx - cx, fy - cy) > 0.40:
                            path = astar_grid_path(g, (cx, cy), (fx, fy))
                            if len(path) >= 2:
                                transit_path = path
                                chosen = cl
                                break

                    if transit_path and chosen is not None:
                        fx, fy, c_size = chosen
                        self.get_logger().info(f'🧭 [FRONTIER] Routing to ({fx:+.2f}, {fy:+.2f}) [{c_size} cells, {len(transit_path)} wpts] | Cov: {cov:.1f}%')
                        with self.mission_lock:
                            self.waypoints = transit_path
                            self.wpt_idx = 0
                            self.is_searching_room = False
                        continue
                    else:
                        # Sweep in place to discover openings
                        self.send_target_vel(0.0, 0.0, 0.0, 0.85)
                else:
                    self.get_logger().info(f'🎯 [SEARCH COMPLETE] No remaining frontiers ({num_surv} survivors located, Cov: {cov:.1f}%)! Initiating Return-To-Home...')
                    self.mission_state = "RTH"
                    self.waypoints = self.plan_rth_path(g, cx, cy)
                    self.wpt_idx = 0
                    self.pose_history.clear()
                    continue

            # ── STATE 3: RETURN TO HOME VIA CORRIDOR HIGHWAY ──
            elif self.mission_state == "RTH":
                dist_home = math.hypot(HELIPAD_X - cx, HELIPAD_Y - cy)
                if dist_home <= 0.45:
                    self.get_logger().info(f'🏠 Centered Over Helipad (dist={dist_home:.2f}m)! Initiating Precision Touchdown...')
                    self.mission_state = "LANDING"
                    self.send_target_vel(0.0, 0.0, 0.0, 0.0)
                    continue

                # Sliding-window displacement watchdog in RTH
                now_t = time.time()
                self.pose_history.append((now_t, cx, cy))
                is_stuck = False
                for t_old, x_old, y_old in self.pose_history:
                    if 5.0 <= (now_t - t_old) <= 8.0:
                        if math.hypot(cx - x_old, cy - y_old) < 0.20:
                            is_stuck = True
                            break
                if is_stuck:
                    self.get_logger().warning('🚨 Minor doorway pause in RTH: backing up and replanning exit path...')
                    for _ in range(30):
                        self.send_target_vel(-0.25, 0.0, 0.0, 0.0)
                        time.sleep(0.033)
                    self.pose_history.clear()
                    with self.mission_lock:
                        self.waypoints = self.plan_rth_path(g, cx, cy)
                        self.wpt_idx = 0
                    continue

                with self.mission_lock:
                    wpts = list(self.waypoints)
                    idx  = self.wpt_idx

                # If waypoints empty or exhausted but not yet at helipad, re-plan!
                if not wpts or idx >= len(wpts):
                    new_path = self.plan_rth_path(g, cx, cy)
                    with self.mission_lock:
                        self.waypoints = new_path
                        self.wpt_idx = 0
                    wpts = new_path
                    idx = 0

                if wpts and idx < len(wpts):
                    target = wpts[idx]
                    dx = target[0] - cx; dy = target[1] - cy
                    dist = math.hypot(dx, dy)

                    # Lookahead segment projection for RTH
                    while idx < len(wpts) - 1:
                        next_target = wpts[idx + 1]
                        next_dist = math.hypot(next_target[0] - cx, next_target[1] - cy)
                        if dist <= 0.40 or next_dist < dist * 0.85:
                            idx += 1
                            target = wpts[idx]
                            dx = target[0] - cx; dy = target[1] - cy
                            dist = math.hypot(dx, dy)
                            continue
                        break

                    with self.mission_lock:
                        self.wpt_idx = idx

                    heading = math.atan2(dy, dx)
                    yaw_err = (heading - yaw + math.pi) % (2 * math.pi) - math.pi

                    if abs(yaw_err) < 0.08:
                        wz = 0.0
                    else:
                        wz = float(np.clip(1.2 * yaw_err, -0.65, 0.65))

                    heading_scale = max(0.20, math.cos(yaw_err))
                    speed = min(0.50, max(0.22, 0.80 * dist))
                    vx = speed * heading_scale
                    vx, vy, wz, _, _ = apply_lidar_obstacle_avoidance(vx, 0.0, wz, yaw_err, ranges)
                    alt_err = TARGET_ALTITUDE - cz
                    vz = float(np.clip(1.3 * alt_err, -0.45, 0.45))
                    self.send_target_vel(vx, vy, vz, wz)
                else:
                    # Fallback direct drive towards Helipad
                    dx = HELIPAD_X - cx; dy = HELIPAD_Y - cy
                    heading = math.atan2(dy, dx)
                    yaw_err = (heading - yaw + math.pi) % (2 * math.pi) - math.pi
                    wz = float(np.clip(1.0 * yaw_err, -0.5, 0.5))
                    self.send_target_vel(0.25 * max(0.2, math.cos(yaw_err)), 0.0, 0.0, wz)

                time.sleep(0.03)
                continue

            # ── STATE 4: LANDING ON HELIPAD ──
            elif self.mission_state == "LANDING":
                if cz > 0.15:
                    # Closed-loop P-centering over helipad during descent
                    err_x = HELIPAD_X - cx
                    err_y = HELIPAD_Y - cy
                    vx = float(np.clip(1.0 * err_x, -0.20, 0.20))
                    vy = float(np.clip(1.0 * err_y, -0.20, 0.20))
                    self.send_target_vel(vx, vy, -0.25, 0.0)
                else:
                    self.send_target_vel(0.0, 0.0, 0.0, 0.0)
                    self.arm_vehicle(False)
                    self.set_flight_mode("AUTO.LAND")
                    complete_msg = Bool()
                    complete_msg.data = True
                    self.pub_mission_complete.publish(complete_msg)
                    self.get_logger().info('🏁 MISSION COMPLETE: ALL SURVIVORS SEARCHED & SAFELY TOUCHED DOWN ON HELIPAD!')
                    break
                time.sleep(0.03)

        self.get_logger().info('Mission loop ended.')


def main(args=None):
    rclpy.init(args=args)
    node = NidarMavrosNavigator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.running = False
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
NIDAR AI Drone — Unified Command Center
========================================
Single-file: Flight Control + Live 2D Map + Sensor HUD + Smart Collision-Free RTH

  python3 map/nidar_command_center.py

Features:
  • Live 2D occupancy grid map with dynamic home origin
  • Smart Return-To-Home (RTH):
      - Minimal collision-free corridor route planning (corridor shortcutting)
      - Coordinated forward flight & corner turning (no sideways sliding into walls)
      - Gentle lateral corridor centering from LiDAR sector ranges
      - Front obstacle emergency brake (< 0.45m)
      - Live visualization of planned path vector (cyan) & steering vector (green)
  • Comprehensive telemetry HUD (Altitude, Motors, LiDAR, Stereo, 5MP WDR, 5m Radar)
  • Zero-lag non-blocking keyboard flight controls

Controls (click map window, then press keys):
  ↑ / ↓         Forward / Backward
  ← / →         Strafe Left / Right
  W / S         Altitude Up / Down
  A / D         Yaw Left / Right
  SPACE         Hover at default altitude (1.5m)
  H             Smart Return-To-Home (collision-free corridor navigation)
  R             Reset simulation
  X             Land / Stop
  Q             Quit + save map
"""

import collections
import heapq
import math
import os
import re
import subprocess
import sys
import threading
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from gz.transport import Node
from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.image_pb2 import Image
from gz.msgs.laserscan_pb2 import LaserScan
from gz.msgs.twist_pb2 import Twist

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
MAX_CEILING     = 2.30    # hard ceiling (m)
DEFAULT_ALT     = 1.50    # default hover altitude (m)
MIN_ALT         = 0.05

OBS_MAX_DIST    = 5.0     # obstacle detection radar radius (m)
OBS_MIN_DIST    = 0.3
CRITICAL_DIST   = 1.5
WARNING_DIST    = 3.0
FWD_FOV_DEG     = 70.0

CELL_M          = 0.05    # map cell size (5 cm)
GRID_SPAN_M     = 30.0    # ±15 m from start
GRID_N          = int(GRID_SPAN_M / CELL_M)   # 600 × 600
HALF_SPAN       = GRID_SPAN_M / 2.0

MAP_REFRESH_S   = 0.5     # map display refresh rate

SAVE_DIR        = os.path.dirname(os.path.abspath(__file__))
SAVE_RAW        = os.path.join(SAVE_DIR, "drone_map_final.png")
SAVE_ANNOT      = os.path.join(SAVE_DIR, "drone_map_annotated.png")

# RTH navigation parameters
RTH_MAX_VEL     = 0.85    # max cruise speed during RTH (m/s)
RTH_WPT_THRESH  = 0.45    # waypoint acceptance radius (m)
RTH_HOME_THRESH = 0.35    # final home acceptance radius (m)

# Grid cell states
UNKNOWN  = 128
FREE     = 240
OCCUPIED = 0
PATH_C   = 180

# ══════════════════════════════════════════════════════════════════════
#  SHARED STATE
# ══════════════════════════════════════════════════════════════════════

# ── Flight ──
target_altitude   = DEFAULT_ALT
current_altitude  = 0.05
vertical_velocity = 0.0
cmd_vx = cmd_vy = cmd_wz = 0.0
last_key_time     = 0.0
motors_active     = True
running           = True
world_name        = "nidar_world"

# ── Smart RTH State ──
rth_active        = False
rth_done          = False
rth_waypoints     = []        # list of (wx, wy) world waypoints
rth_wpt_idx       = 0
rth_steer_vector  = (0.0, 0.0) # (vx_world, vy_world) current steering vector
rth_lock          = threading.Lock()
rth_status_msg    = ""

# ── Autonomous 15x15m Exploration State ──
MISSION_IDLE      = "IDLE"
MISSION_TAKEOFF   = "TAKEOFF"
MISSION_EXPLORING = "EXPLORING 15x15m GRID"
MISSION_RTH       = "RETURNING HOME"
MISSION_LANDING   = "LANDING ON HELIPAD"
MISSION_COMPLETE  = "MISSION COMPLETE"

auto_mode         = False
mission_state     = MISSION_IDLE
auto_waypoints    = []
auto_wpt_idx      = 0
auto_status_msg   = ""
auto_lock         = threading.Lock()

# Breadcrumb trail (corridor highway recorded during flight)
breadcrumbs       = []
last_bread_pos    = None
bread_lock        = threading.Lock()

# ── Sensors ──
lidar_connected   = False
lidar_rays_count  = 0
obstacle_detected = False
closest_dist      = float('inf')
closest_angle_deg = 0.0
sector_ranges     = {k: float('inf') for k in ("left","front_left","front","front_right","right")}

stereo_connected   = False
stereo_fps         = 0.0
stereo_res         = "640x480"
stereo_fc          = 0
stereo_lt          = time.time()

cam5_connected     = False
cam5_fps           = 0.0
cam5_res           = "1920x1080"
cam5_fc            = 0
cam5_lt            = time.time()

# ── Map ──
grid      = np.full((GRID_N, GRID_N), UNKNOWN, dtype=np.uint8)
grid_lock = threading.Lock()

# Visited Heatmap for anti-looping exploration & raw LiDAR buffer
visited_heatmap     = np.zeros((GRID_N, GRID_N), dtype=np.uint32)
heatmap_lock        = threading.Lock()
raw_lidar_ranges    = []
raw_lidar_lock      = threading.Lock()

# Dynamic origin (locked on first pose reading)
home_x    = None
home_y    = None
home_set  = threading.Event()

# Live drone pose
map_drone_x   = None
map_drone_y   = None
map_drone_yaw = 0.0
map_pose_lock = threading.Lock()

# ══════════════════════════════════════════════════════════════════════
#  COORDINATE CONVERSION & LINE OF SIGHT
# ══════════════════════════════════════════════════════════════════════
def world_to_cell(wx, wy):
    if home_x is None:
        return None
    col = int((wx - home_x + HALF_SPAN) / CELL_M)
    row = int((wy - home_y + HALF_SPAN) / CELL_M)
    if 0 <= col < GRID_N and 0 <= row < GRID_N:
        return col, row
    return None


def cell_to_world(col, row):
    if home_x is None:
        return 0.0, 0.0
    wx = (col * CELL_M) - HALF_SPAN + home_x
    wy = (row * CELL_M) - HALF_SPAN + home_y
    return wx, wy


def bresenham(c0, r0, c1, r1):
    dc = abs(c1 - c0); dr = abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc - dr
    c, r = c0, r0
    while True:
        yield c, r
        if c == c1 and r == r1:
            break
        e2 = 2 * err
        if e2 > -dr: err -= dr; c += sc
        if e2 < dc:  err += dc; r += sr


def line_is_clear_of_walls(p1, p2, occ_grid, margin_cells=3):
    """
    Check if straight line between p1 and p2 (world coords) stays at least
    margin_cells (15cm) away from all mapped OCCUPIED obstacles.
    """
    c0_r0 = world_to_cell(p1[0], p1[1])
    c1_r1 = world_to_cell(p2[0], p2[1])
    if c0_r0 is None or c1_r1 is None:
        return False
    c0, r0 = c0_r0
    c1, r1 = c1_r1

    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1; sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    H, W = occ_grid.shape
    while True:
        r_min = max(0, r - margin_cells); r_max = min(H, r + margin_cells + 1)
        c_min = max(0, c - margin_cells); c_max = min(W, c + margin_cells + 1)
        if np.any(occ_grid[r_min:r_max, c_min:c_max] == OCCUPIED):
            return False
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr: err -= dr; c += sc
        if e2 < dc:  err += dc; r += sr
    return True


def fmt_dist(d):
    return f"{d:.2f}m" if not math.isinf(d) else "CLEAR"


# ══════════════════════════════════════════════════════════════════════
#  SMART PATH PLANNING: CORRIDOR BREADCRUMB SHORTCUTTING
# ══════════════════════════════════════════════════════════════════════
def plan_rth_path(start_xy, goal_xy):
    """
    Computes the shortest collision-free corridor route to Home.
    Uses the recorded corridor breadcrumb highway that the drone already
    safely navigated, shortcutting unnecessary loops and turns without
    ever attempting to cut across unmapped buildings or through solid walls.
    """
    with bread_lock:
        trail_pts = list(breadcrumbs)
    with grid_lock:
        g = grid.copy()

    # Direct line-of-sight check to Home
    if line_is_clear_of_walls(start_xy, goal_xy, g, margin_cells=4):
        return [start_xy, goal_xy]

    if not trail_pts:
        return [start_xy, goal_xy]

    # Find closest breadcrumb point to current position
    closest_i = len(trail_pts) - 1
    min_d = float('inf')
    for i, p in enumerate(trail_pts):
        d = math.hypot(p[0] - start_xy[0], p[1] - start_xy[1])
        if d < min_d:
            min_d = d
            closest_i = i

    # Trace backward from closest breadcrumb to 0 (home)
    raw_path = [start_xy]
    for i in range(closest_i, -1, -1):
        pt = trail_pts[i]
        if math.hypot(pt[0] - raw_path[-1][0], pt[1] - raw_path[-1][1]) >= 0.20:
            raw_path.append(pt)
    if math.hypot(raw_path[-1][0] - goal_xy[0], raw_path[-1][1] - goal_xy[1]) >= 0.20:
        raw_path.append(goal_xy)

    # Greedily shortcut intermediate waypoints where clear line-of-sight exists
    shortcutted = [raw_path[0]]
    curr_i = 0
    while curr_i < len(raw_path) - 1:
        best_next = curr_i + 1
        for check_i in range(len(raw_path) - 1, curr_i, -1):
            if line_is_clear_of_walls(raw_path[curr_i], raw_path[check_i], g, margin_cells=3):
                best_next = check_i
                break
        shortcutted.append(raw_path[best_next])
        curr_i = best_next

    # Resample long straightaways every 0.50m - 0.70m for smooth guidance
    wpts = [shortcutted[0]]
    for p in shortcutted[1:]:
        prev_p = wpts[-1]
        dist = math.hypot(p[0] - prev_p[0], p[1] - prev_p[1])
        if dist > 0.75:
            steps = int(math.ceil(dist / 0.55))
            for s in range(1, steps):
                frac = s / float(steps)
                ix = prev_p[0] + frac * (p[0] - prev_p[0])
                iy = prev_p[1] + frac * (p[1] - prev_p[1])
                wpts.append((ix, iy))
        wpts.append(p)

    return wpts


# ══════════════════════════════════════════════════════════════════════
#  GAZEBO TRANSPORT NODE
# ══════════════════════════════════════════════════════════════════════
node      = Node()
twist_msg = Twist()
twist_pub = node.advertise('/X3/gazebo/command/twist', Twist)
enable_pub = node.advertise('/X3/enable', Boolean)
enable_msg = Boolean(); enable_msg.data = True


def detect_world():
    global world_name
    try:
        res = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True, timeout=3)
        if "/world/nidar_world/pose/info" in res.stdout:
            world_name = "nidar_world"
        elif "/world/quadcopter/pose/info" in res.stdout:
            world_name = "quadcopter"
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
#  SENSOR CALLBACKS
# ══════════════════════════════════════════════════════════════════════
def lidar_cb(msg: LaserScan):
    global lidar_connected, lidar_rays_count
    global obstacle_detected, closest_dist, closest_angle_deg, sector_ranges, raw_lidar_ranges

    lidar_connected  = True
    lidar_rays_count = len(msg.ranges)
    if lidar_rays_count == 0:
        return

    with raw_lidar_lock:
        raw_lidar_ranges = list(msg.ranges)

    angle_min  = msg.angle_min
    angle_step = msg.angle_step
    half_cone  = math.radians(FWD_FOV_DEG / 2.0)

    min_fd = float('inf'); min_fa = 0.0
    sl = sf_l = sf = sf_r = sr = float('inf')

    # ── Map ray casting & Visited Heatmap update ──
    if home_set.is_set():
        with map_pose_lock:
            ox = map_drone_x; oy = map_drone_y; yaw = map_drone_yaw
        if ox is not None:
            origin = world_to_cell(ox, oy)
            if origin:
                oc, or_ = origin
                with heatmap_lock:
                    visited_heatmap[max(0, or_-2):min(GRID_N, or_+3),
                                    max(0, oc-2):min(GRID_N, oc+3)] += 1
                with grid_lock:
                    for i, r in enumerate(msg.ranges):
                        theta_l = angle_min + i * angle_step
                        theta_w = theta_l + yaw
                        valid   = not (math.isinf(r) or math.isnan(r)) and msg.range_min <= r <= msg.range_max
                        ray_r   = r if valid else msg.range_max
                        ex = ox + ray_r * math.cos(theta_w)
                        ey = oy + ray_r * math.sin(theta_w)
                        ep = world_to_cell(ex, ey)
                        if ep is None:
                            continue
                        ec, er = ep
                        for c, ro in bresenham(oc, or_, ec, er):
                            if 0 <= c < GRID_N and 0 <= ro < GRID_N and grid[ro, c] == UNKNOWN:
                                grid[ro, c] = FREE
                        if valid and 0 <= ec < GRID_N and 0 <= er < GRID_N:
                            grid[er, ec] = OCCUPIED
                    if grid[or_, oc] != OCCUPIED:
                        grid[or_, oc] = PATH_C

    # ── Obstacle radar & surround sectors ──
    for i, r in enumerate(msg.ranges):
        if math.isinf(r) or math.isnan(r) or r < msg.range_min or r > msg.range_max:
            continue
        theta = (angle_min + i * angle_step + math.pi) % (2 * math.pi) - math.pi
        td    = math.degrees(theta)
        if -20 <= td <= 20:
            if r < sf:  sf  = r
        elif 20 < td <= 60:
            if r < sf_l: sf_l = r
        elif 60 < td <= 110:
            if r < sl:  sl  = r
        elif -60 <= td < -20:
            if r < sf_r: sf_r = r
        elif -110 <= td < -60:
            if r < sr:  sr  = r
        if -half_cone <= (angle_min + i * angle_step) <= half_cone:
            if r < min_fd:
                min_fd = r; min_fa = td

    sector_ranges.update({"left": sl, "front_left": sf_l,
                           "front": sf, "front_right": sf_r, "right": sr})
    closest_dist      = min_fd
    closest_angle_deg = min_fa
    obstacle_detected = (min_fd <= OBS_MAX_DIST)


def stereo_cb(msg: Image):
    global stereo_connected, stereo_fps, stereo_res, stereo_fc, stereo_lt
    stereo_connected = True
    stereo_fc += 1
    stereo_res = f"{msg.width}x{msg.height}"
    now = time.time()
    if now - stereo_lt >= 1.0:
        stereo_fps = stereo_fc / (now - stereo_lt)
        stereo_fc = 0; stereo_lt = now


def cam5_cb(msg: Image):
    global cam5_connected, cam5_fps, cam5_res, cam5_fc, cam5_lt
    cam5_connected = True
    cam5_fc += 1
    cam5_res = f"{msg.width}x{msg.height}"
    now = time.time()
    if now - cam5_lt >= 1.0:
        cam5_fps = cam5_fc / (now - cam5_lt)
        cam5_fc = 0; cam5_lt = now


node.subscribe(LaserScan, '/nidar/lidar', lidar_cb)
node.subscribe(Image, '/nidar/stereo_camera/image', stereo_cb)
node.subscribe(Image, '/nidar/camera', cam5_cb)


# ══════════════════════════════════════════════════════════════════════
#  POSE READER THREAD & CORRIDOR BREADCRUMB HIGHWAY
# ══════════════════════════════════════════════════════════════════════
def pose_thread():
    global current_altitude, vertical_velocity, running, last_bread_pos
    global home_x, home_y, map_drone_x, map_drone_y, map_drone_yaw, world_name

    detect_world()
    pose_topic = f"/world/{world_name}/pose/info"
    proc = subprocess.Popen(
        ["gz", "topic", "-t", pose_topic, "-e"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
    )

    curr_ent  = None
    in_pos    = in_orient = False
    last_z    = 0.05
    last_t    = time.time()
    tx = ty = tz = 0.0
    tqx = tqy = tqz = tqw = 1.0

    while running:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()

        m = re.match(r'name:\s*"([^"]+)"', line)
        if m:
            curr_ent  = m.group(1)
            in_pos    = in_orient = False

        if curr_ent != "x3":
            continue

        if line == "position {":    in_pos    = True
        if line == "orientation {": in_orient = True
        if line == "}" and in_pos:
            in_pos = False
        if line == "}" and in_orient:
            siny = 2.0 * (tqw * tqz + tqx * tqy)
            cosy = 1.0 - 2.0 * (tqy**2 + tqz**2)
            yaw  = math.atan2(siny, cosy)

            with map_pose_lock:
                map_drone_x   = tx
                map_drone_y   = ty
                map_drone_yaw = yaw

            # Lock home on first valid reading
            if not home_set.is_set() and tx is not None:
                home_x = tx; home_y = ty
                home_set.set()
                with bread_lock:
                    breadcrumbs.append((home_x, home_y))
                    last_bread_pos = (home_x, home_y)

            # Record breadcrumb every 0.20m of travel in corridor
            if home_set.is_set():
                with bread_lock:
                    if last_bread_pos is None:
                        breadcrumbs.append((tx, ty))
                        last_bread_pos = (tx, ty)
                    elif math.hypot(tx - last_bread_pos[0], ty - last_bread_pos[1]) >= 0.20:
                        breadcrumbs.append((tx, ty))
                        last_bread_pos = (tx, ty)

            in_orient = False

        if in_pos:
            for tag in ('x', 'y', 'z'):
                m2 = re.match(rf'{tag}:\s*([\-\d.eE+]+)', line)
                if m2:
                    val = float(m2.group(1))
                    if tag == 'x': tx = val
                    elif tag == 'y': ty = val
                    elif tag == 'z': tz = val

            if line.startswith("z:"):
                try:
                    z   = float(line.split(":")[1].strip())
                    now = time.time()
                    dt  = now - last_t
                    if dt > 0.015:
                        vertical_velocity = 0.7 * vertical_velocity + 0.3 * (z - last_z) / dt
                        current_altitude  = z
                        last_z = z; last_t = now
                except ValueError:
                    pass

        if in_orient:
            for tag in ('x', 'y', 'z', 'w'):
                m2 = re.match(rf'{tag}:\s*([\-\d.eE+]+)', line)
                if m2:
                    val = float(m2.group(1))
                    if tag == 'x': tqx = val
                    elif tag == 'y': tqy = val
                    elif tag == 'z': tqz = val
                    elif tag == 'w': tqw = val

    proc.kill()


# ══════════════════════════════════════════════════════════════════════
#  FLIGHT CONTROL LOOP (MANUAL TELEOP)
# ══════════════════════════════════════════════════════════════════════
def flight_ctrl_thread():
    global cmd_vx, cmd_vy, cmd_wz, running, motors_active

    enable_pub.publish(enable_msg)

    while running:
        with rth_lock:
            rth = rth_active
        with auto_lock:
            auto_active = auto_mode and mission_state in (MISSION_TAKEOFF, MISSION_EXPLORING)

        if rth or auto_active:
            time.sleep(0.02)
            continue   # Autonomous thread controls twist during autonomous flight

        if not motors_active or target_altitude < 0.1:
            twist_msg.linear.x = 0.0; twist_msg.linear.y = 0.0
            twist_msg.linear.z = -0.5 if current_altitude > 0.15 else 0.0
            twist_msg.angular.z = 0.0
            twist_pub.publish(twist_msg)
            time.sleep(0.02)
            continue

        if time.time() - last_key_time > 0.35:
            cmd_vx = cmd_vy = cmd_wz = 0.0

        alt_err = target_altitude - current_altitude
        cmd_vz  = max(-1.0, min(1.0, 1.3 * alt_err))
        if current_altitude >= MAX_CEILING:
            cmd_vz = -0.5
        elif current_altitude >= MAX_CEILING - 0.2 and cmd_vz > 0:
            cmd_vz = min(0.15, cmd_vz)
        if abs(alt_err) < 0.04 and current_altitude < MAX_CEILING:
            cmd_vz = 0.0

        twist_msg.linear.x  = cmd_vx
        twist_msg.linear.y  = cmd_vy
        twist_msg.linear.z  = cmd_vz
        twist_msg.angular.z = cmd_wz
        twist_pub.publish(twist_msg)
        time.sleep(0.02)


# ══════════════════════════════════════════════════════════════════════
#  REAL-TIME 360° LIDAR OBSTACLE TACKLING & COLLISION EVASION
# ══════════════════════════════════════════════════════════════════════
def compute_lidar_avoidance(ranges, angle_min=-math.pi, angle_step=None, d_influence=0.85):
    """
    Computes real-time continuous repulsive force vector (fx, fy) in body frame,
    and returns min distances across key sectors from 360° LiDAR rays.
    """
    n_rays = len(ranges) if ranges is not None else 0
    if n_rays == 0:
        return 0.0, 0.0, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')

    if angle_step is None:
        angle_step = (2 * math.pi) / n_rays

    fx = 0.0
    fy = 0.0
    d_f = float('inf')
    d_fl = float('inf')
    d_fr = float('inf')
    d_l = float('inf')
    d_r = float('inf')
    d_rear = float('inf')

    for i, r in enumerate(ranges):
        if math.isinf(r) or math.isnan(r) or r < 0.05 or r > 16.0:
            continue
        theta = (angle_min + i * angle_step + math.pi) % (2 * math.pi) - math.pi
        td = math.degrees(theta)

        if -25 <= td <= 25:
            if r < d_f: d_f = r
        elif 15 < td <= 75:
            if r < d_fl: d_fl = r
        elif -75 <= td < -15:
            if r < d_fr: d_fr = r
        elif 60 < td <= 120:
            if r < d_l: d_l = r
        elif -120 <= td < -60:
            if r < d_r: d_r = r
        elif abs(td) >= 135:
            if r < d_rear: d_rear = r

        # Full 360° potential field repulsive force for obstacles within d_influence
        if r < d_influence:
            weight = ((d_influence - r) / d_influence) ** 2
            if abs(td) <= 90:
                weight *= 1.4
            fx -= weight * math.cos(theta)
            fy -= weight * math.sin(theta)

    return fx, fy, d_f, d_fl, d_fr, d_l, d_r, d_rear


def compute_escape_vector(ranges):
    """
    Computes body-frame velocity (esc_vx, esc_vy) to actively pull drone off a wall
    towards the direction of maximum clearance.
    """
    fx, fy, d_f, d_fl, d_fr, d_l, d_r, d_rear = compute_lidar_avoidance(ranges)
    norm = math.hypot(fx, fy)
    if norm > 0.05:
        esc_vx = float(np.clip((fx / norm) * 0.35, -0.35, 0.35))
        esc_vy = float(np.clip((fy / norm) * 0.35, -0.35, 0.35))
    else:
        esc_vx = -0.25 if d_rear > 0.40 else 0.0
        esc_vy = 0.25 if d_l > d_r else -0.25
    return esc_vx, esc_vy


def apply_lidar_obstacle_avoidance(vx_cmd, vy_cmd, wz_cmd, yaw_err, ranges):
    """
    LiDAR-based active collision avoidance and wall push:
    • Uses continuous repulsive vector (fx, fy) to actively push away from walls.
    • CRITICAL: If an obstacle is close (< 0.48m), actively pushes away even while aligning/turning!
    • Decelerates approaching obstacles; hard stops and reverses if front is blocked (< 0.40m).
    """
    fx, fy, d_f, d_fl, d_fr, d_l, d_r, d_rear = compute_lidar_avoidance(ranges)
    min_fwd = min(d_f, d_fl, d_fr)
    min_side = min(d_l, d_r)
    min_overall = min(min_fwd, min_side, d_rear)
    status_suffix = ""

    # Calculate active repulsive velocities away from obstacles
    norm_f = math.hypot(fx, fy)
    if norm_f > 0.01:
        rep_vx = float(np.clip(fx * 0.35, -0.30, 0.30))
        rep_vy = float(np.clip(fy * 0.35, -0.30, 0.30))
    else:
        rep_vx = 0.0
        rep_vy = 0.0

    # Turning in place (large yaw_err):
    if abs(yaw_err) > 0.35:
        if min_overall >= 0.50:
            return 0.0, 0.0, wz_cmd, status_suffix
        else:
            # Dangerously close to a wall/corner! MUST push away while rotating!
            vx_out = rep_vx
            vy_out = rep_vy
            if min_fwd < 0.40 and d_rear > 0.45:
                vx_out = min(vx_out, -0.18)
            status_suffix = f" | Wall Push ({min_overall:.2f}m)"
            return vx_out, vy_out, wz_cmd, status_suffix

    # Moving forward toward waypoint:
    if min_fwd < 0.40:
        # Critical front proximity: Stop forward progress and push away
        vx_cmd = -0.20 if d_rear > 0.45 else 0.0
        # Lateral slide strongly away from obstacle
        if d_fl < d_fr:
            vy_cmd = -0.28  # slide right
        else:
            vy_cmd = 0.28   # slide left
        status_suffix = f" | WALL ({min_fwd:.2f}m) -> Reverse & Slide"

    elif min_fwd < 0.70:
        # Approaching obstacle: decelerate forward speed and deflect laterally
        speed_factor = max(0.0, (min_fwd - 0.40) / (0.70 - 0.40))
        vx_cmd = min(vx_cmd, 0.45 * speed_factor)
        vy_cmd += rep_vy
        status_suffix = f" | Wall Deflect ({min_fwd:.2f}m)"

    else:
        # Normal cruise: apply gentle corridor centering
        vy_cmd += float(np.clip(fy * 0.20, -0.18, 0.18))

    # Also check side clearance: if side wall is closer than 0.45m, push away laterally
    if min_side < 0.45:
        if d_l < 0.45:
            vy_cmd = min(vy_cmd, -0.22) # push right
        elif d_r < 0.45:
            vy_cmd = max(vy_cmd, 0.22)  # push left

    return vx_cmd, vy_cmd, wz_cmd, status_suffix


# ══════════════════════════════════════════════════════════════════════
#  SMART RETURN-TO-HOME THREAD (COORDINATED FLIGHT + CORRIDOR CENTERING)
# ══════════════════════════════════════════════════════════════════════
def rth_thread():
    """
    Autonomous Return-To-Home:
    • Follows the minimal collision-free corridor route.
    • Coordinated turning: Rotates to face the corridor first at corners
      rather than strafing sideways into walls.
    • Gentle lateral corridor centering via LiDAR clearance.
    • Emergency front obstacle brake (< 0.45m).
    """
    global rth_active, rth_done, rth_waypoints, rth_wpt_idx, rth_steer_vector
    global rth_status_msg, running, auto_mode, mission_state, auto_status_msg
    global target_altitude, motors_active

    while running:
        with rth_lock:
            active = rth_active

        if not active or home_x is None:
            time.sleep(0.03)
            continue

        with map_pose_lock:
            cx  = map_drone_x
            cy  = map_drone_y
            yaw = map_drone_yaw
        cz = current_altitude

        if cx is None:
            time.sleep(0.03)
            continue

        with rth_lock:
            wpts = list(rth_waypoints)
            idx  = rth_wpt_idx

        if not wpts:
            wpts = plan_rth_path((cx, cy), (home_x, home_y))
            idx = 0
            with rth_lock:
                rth_waypoints = wpts
                rth_wpt_idx   = idx

        # Target waypoint
        target_pt = wpts[idx]
        tx, ty = target_pt
        dx = tx - cx
        dy = ty - cy
        dist_to_wpt = math.hypot(dx, dy)
        dist_to_home = math.hypot(home_x - cx, home_y - cy)

        # Check if reached final Home destination -> Smoothly land & disarm
        if dist_to_home <= RTH_HOME_THRESH:
            with rth_lock:
                rth_status_msg = "TOUCHING DOWN ON HELIPAD"
            print("\n  🏠 REACHED HOME HELIPAD: Initiating smooth autonomous landing descent...")

            # Smooth landing descent phase down to helipad
            while running and current_altitude > 0.15:
                twist_msg.linear.x  = 0.0
                twist_msg.linear.y  = 0.0
                twist_msg.linear.z  = -0.35  # gentle 35 cm/s descent
                twist_msg.angular.z = 0.0
                twist_pub.publish(twist_msg)
                time.sleep(0.03)

            # Touchdown: completely stop and disarm motors on the pad
            for _ in range(5):
                twist_msg.linear.x  = 0.0
                twist_msg.linear.y  = 0.0
                twist_msg.linear.z  = 0.0
                twist_msg.angular.z = 0.0
                twist_pub.publish(twist_msg)
                time.sleep(0.02)

            motors_active = False
            target_altitude = 0.0

            with rth_lock:
                rth_active       = False
                rth_done         = True
                rth_steer_vector = (0.0, 0.0)
                rth_status_msg   = "LANDED & DISARMED ON HELIPAD"

            with auto_lock:
                if auto_mode:
                    mission_state   = MISSION_COMPLETE
                    auto_status_msg = "MISSION COMPLETE: FULL 15x15M GRID EXPLORED & LANDED"
                    auto_mode       = False

            print("  ✅ TOUCHDOWN COMPLETE: Drone has safely landed on Helipad and motors are disarmed!\n")
            save_map()
            continue

        # Advance to next waypoint if close enough
        if dist_to_wpt <= RTH_WPT_THRESH and idx < len(wpts) - 1:
            idx += 1
            with rth_lock:
                rth_wpt_idx = idx
            target_pt = wpts[idx]
            tx, ty = target_pt
            dx = tx - cx; dy = ty - cy
            dist_to_wpt = math.hypot(dx, dy)

        # Target heading to waypoint
        target_heading = math.atan2(dy, dx)
        yaw_err = (target_heading - yaw + math.pi) % (2 * math.pi) - math.pi

        # ── Coordinated flight dynamics ──
        # At corners or when misaligned: turn in place to face corridor first!
        if abs(yaw_err) > 0.60: # > ~35 degrees
            vx_cmd = 0.08 # minimal forward crawl
            vy_cmd = 0.0  # zero sideways slide
            wz_cmd = float(np.clip(1.5 * yaw_err, -1.1, 1.1))
            status = f"Turning corner (error {math.degrees(yaw_err):+.0f}°)"
        else:
            # Aligned with corridor: cruise smoothly forward
            cruise_speed = min(RTH_MAX_VEL, max(0.35, 0.9 * dist_to_wpt))
            vx_cmd = cruise_speed * math.cos(yaw_err)
            wz_cmd = float(np.clip(1.2 * yaw_err, -0.6, 0.6))
            vy_cmd = 0.0
            status = f"Cruising {vx_cmd:.2f} m/s"

        with raw_lidar_lock:
            ranges = list(raw_lidar_ranges)
        vx_cmd, vy_cmd, wz_cmd, obs_status = apply_lidar_obstacle_avoidance(
            vx_cmd, vy_cmd, wz_cmd, yaw_err, ranges
        )
        status += obs_status

        # Altitude hold at safe cruising altitude (1.50m)
        alt_err = DEFAULT_ALT - cz
        vz_cmd = float(np.clip(1.3 * alt_err, -0.5, 0.5))

        # Publish twist
        twist_msg.linear.x  = float(np.clip(vx_cmd, -RTH_MAX_VEL, RTH_MAX_VEL))
        twist_msg.linear.y  = float(np.clip(vy_cmd, -0.5, 0.5))
        twist_msg.linear.z  = vz_cmd
        twist_msg.angular.z = wz_cmd
        twist_pub.publish(twist_msg)

        # Store steer vector for map display (world frame)
        sv_world_x = math.cos(yaw) * vx_cmd - math.sin(yaw) * vy_cmd
        sv_world_y = math.sin(yaw) * vx_cmd + math.cos(yaw) * vy_cmd
        with rth_lock:
            rth_steer_vector = (sv_world_x, sv_world_y)
            rth_status_msg   = status

        time.sleep(0.03)


# ══════════════════════════════════════════════════════════════════════
#  RESET SIMULATION
# ══════════════════════════════════════════════════════════════════════
def reset_sim():
    global target_altitude, cmd_vx, cmd_vy, cmd_wz, motors_active
    global rth_active, rth_done, rth_waypoints, auto_mode, mission_state
    with rth_lock:
        rth_active = False
        rth_done   = False
        rth_waypoints = []
    with auto_lock:
        auto_mode     = False
        mission_state = MISSION_IDLE
    motors_active = True
    target_altitude = DEFAULT_ALT
    cmd_vx = cmd_vy = cmd_wz = 0.0
    subprocess.Popen(
        ["gz", "service", "-s", f"/world/{world_name}/control",
         "--reqtype", "gz.msgs.WorldControl",
         "--reptype", "gz.msgs.Boolean",
         "--timeout", "2000",
         "--req", "reset: { all: true }, pause: false"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


# ══════════════════════════════════════════════════════════════════════
#  DYNAMIC LIDAR PREFERENCE & FRONTIER EXPLORATION ENGINE
# ══════════════════════════════════════════════════════════════════════
def astar_grid_path(g, start_xy, goal_xy, downsample=2):
    """
    Computes a fast, collision-free grid path between start and goal.
    Dilates obstacles by safety margins to guarantee body clearance.
    """
    p_start = world_to_cell(start_xy[0], start_xy[1])
    p_goal  = world_to_cell(goal_xy[0], goal_xy[1])
    if not p_start or not p_goal:
        return [start_xy, goal_xy]
    sc, sr = p_start
    gc, gr = p_goal

    H, W = g.shape
    h_down = H // downsample
    w_down = W // downsample

    blocks = g[:h_down*downsample, :w_down*downsample].reshape(h_down, downsample, w_down, downsample)
    occ_mask = np.any(blocks == OCCUPIED, axis=(1, 3))

    dilated_occ = occ_mask.copy()
    for _ in range(3):
        nbrs = dilated_occ.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue
                shifted = np.zeros_like(nbrs)
                r_dst = slice(max(0, dr), min(h_down, h_down+dr))
                c_dst = slice(max(0, dc), min(w_down, w_down+dc))
                r_src = slice(max(0, -dr), min(h_down, h_down-dr))
                c_src = slice(max(0, -dc), min(w_down, w_down-dc))
                shifted[r_dst, c_dst] = nbrs[r_src, c_src]
                dilated_occ |= shifted

    s_r = max(0, min(h_down - 1, sr // downsample))
    s_c = max(0, min(w_down - 1, sc // downsample))
    g_r = max(0, min(h_down - 1, gr // downsample))
    g_c = max(0, min(w_down - 1, gc // downsample))

    # Only un-dilate the start cell if the drone itself is currently in dilation zone
    if dilated_occ[s_r, s_c]:
        dilated_occ[s_r, s_c] = False
        for dr, dc in ((-1,0), (1,0), (0,-1), (0,1)):
            if 0 <= s_r + dr < h_down and 0 <= s_c + dc < w_down:
                if not occ_mask[s_r + dr, s_c + dc]:
                    dilated_occ[s_r + dr, s_c + dc] = False
    # Only un-dilate the goal cell if it is dilated but not occupied
    if dilated_occ[g_r, g_c] and not occ_mask[g_r, g_c]:
        dilated_occ[g_r, g_c] = False

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
                new_cost = current_cost + w
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
    for r, c in wpts_rc[1:]:
        wx, wy = cell_to_world(c * downsample + downsample // 2, r * downsample + downsample // 2)
        wx = float(np.clip(wx, -6.3, 6.3))
        wy = float(np.clip(wy, -6.3, 6.3))
        if math.hypot(wx - wpts[-1][0], wy - wpts[-1][1]) >= 0.40:
            wpts.append((wx, wy))
    g_clamped = (float(np.clip(goal_xy[0], -6.3, 6.3)), float(np.clip(goal_xy[1], -6.3, 6.3)))
    if math.hypot(g_clamped[0] - wpts[-1][0], g_clamped[1] - wpts[-1][1]) >= 0.20:
        wpts.append(g_clamped)
    return wpts


def find_frontier_clusters(g, min_cluster_size=4):
    """
    Extracts frontier cells strictly within the 15x15m arena (|X| <= 6.3m, |Y| <= 6.3m).
    Completely ignores any open ground plane outside the compound walls.
    """
    c_min, r_min = world_to_cell(-6.3, -6.3) or (0, 0)
    c_max, r_max = world_to_cell( 6.3,  6.3) or (GRID_N, GRID_N)
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
            if abs(wx) <= 6.3 and abs(wy) <= 6.3:
                clusters.append((wx, wy, len(cluster)))

    return clusters


def evaluate_lidar_exploration_direction(cx, cy, current_yaw, lidar_ranges, g, v_map):
    """
    Evaluates 36 candidate heading angles around drone [0, 360 deg) based on:
    1. Raw LiDAR clearance in cone (safety: strictly forbids min_clearance < 0.60m)
    2. Number of UNKNOWN cells in cone (unexplored exploration reward)
    3. Visited count along ray (anti-looping penalty from visited heatmap)
    4. Heading continuity with current_yaw (momentum / anti-jitter)
    """
    n_rays = len(lidar_ranges) if lidar_ranges is not None else 0
    if n_rays == 0:
        return current_yaw, 0.0, False

    num_candidates = 36
    candidate_angles = np.linspace(-math.pi, math.pi, num_candidates, endpoint=False)
    best_heading = current_yaw
    best_score = -float('inf')
    found_valid = False

    for heading_w in candidate_angles:
        rel_angle = (heading_w - current_yaw + math.pi) % (2 * math.pi) - math.pi

        cone_half = math.radians(16.0)
        min_clearance = 16.0
        for i, r in enumerate(lidar_ranges):
            if math.isinf(r) or math.isnan(r) or r <= 0.10:
                continue
            ray_theta = -math.pi + i * (2 * math.pi / n_rays)
            diff = (ray_theta - rel_angle + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) <= cone_half:
                if r < min_clearance:
                    min_clearance = r

        # Strict safety margin: if obstacle is closer than 0.60m, heading is blocked!
        if min_clearance < 0.60:
            continue

        ray_dist = min(min_clearance, 4.0)
        ex = cx + ray_dist * math.cos(heading_w)
        ey = cy + ray_dist * math.sin(heading_w)

        p0 = world_to_cell(cx, cy)
        p1 = world_to_cell(ex, ey)
        if not p0 or not p1:
            continue

        pts = list(bresenham(p0[0], p0[1], p1[0], p1[1]))
        unknown_count = 0
        visited_penalty = 0.0
        free_count = 0
        blocked_by_wall = False

        for c, r in pts[1:]:
            if not (0 <= c < GRID_N and 0 <= r < GRID_N):
                break
            cell_val = g[r, c]
            if cell_val == OCCUPIED:
                blocked_by_wall = True
                break
            elif cell_val == UNKNOWN:
                unknown_count += 1
            else:
                free_count += 1
                visited_penalty += v_map[r, c]

        if blocked_by_wall and (free_count + unknown_count) < 6:
            continue

        clearance_score = min_clearance * 2.5
        unexplored_score = unknown_count * 4.5
        visit_cost = visited_penalty * 0.85
        momentum = math.cos(rel_angle) * 1.5

        total_score = unexplored_score + clearance_score - visit_cost + momentum

        if total_score > best_score:
            best_score = total_score
            best_heading = heading_w
            found_valid = True

    return best_heading, best_score, found_valid


def auto_mission_thread():
    """
    Fully Autonomous Dynamic LiDAR & Frontier Exploration Engine:
    1. TAKEOFF: Climbs to hover altitude (1.50m) and safely exits Room (0,0) into corridor.
    2. DYNAMIC EXPLORATION:
       • Continuously evaluates 36 LiDAR candidate headings.
       • Steers toward maximum free clearance and unexplored space.
       • Visited heatmap actively repels drone from already explored corridors (prevents looping).
       • When local area is mapped, routes via A* to nearest unexplored frontier.
       • Anti-looping circuit breaker detects and breaks any cycles.
    3. SMART RETURN-TO-HOME:
       • Upon reaching >= 95.5% coverage (or all frontiers cleared), plans collision-free A* shortcut to Helipad.
    4. PRECISION LANDING:
       • Soft touchdown on Helipad and disarms motors.
    """
    global auto_mode, mission_state, auto_status_msg, auto_waypoints, auto_wpt_idx
    global target_altitude, motors_active, rth_active, rth_done, rth_waypoints, rth_wpt_idx
    global rth_steer_vector, rth_status_msg, running

    takeoff_start_time = None
    pos_history = collections.deque(maxlen=350)
    last_prog_pos = None
    last_prog_time = time.time()
    active_turn_dir = None
    turn_start_time = 0.0

    while running:
        with auto_lock:
            active = auto_mode
            state  = mission_state

        if not active or home_x is None:
            takeoff_start_time = None
            pos_history.clear()
            last_prog_pos = None
            last_prog_time = time.time()
            active_turn_dir = None
            turn_start_time = 0.0
            time.sleep(0.04)
            continue

        with map_pose_lock:
            cx  = map_drone_x
            cy  = map_drone_y
            yaw = map_drone_yaw
        cz = current_altitude

        if cx is None or cy is None:
            time.sleep(0.04)
            continue

        # ── PHASE 1: AUTONOMOUS TAKEOFF ──
        if state == MISSION_TAKEOFF:
            motors_active   = True
            target_altitude = DEFAULT_ALT
            if takeoff_start_time is None:
                takeoff_start_time = time.time()
            elapsed_climb = time.time() - takeoff_start_time
            with auto_lock:
                auto_status_msg = f"Climbing to hover altitude ({cz:.2f}m / {DEFAULT_ALT:.2f}m)"

            if cz >= (DEFAULT_ALT - 0.25) or (elapsed_climb > 2.5 and cz >= 0.90):
                # If starting in Room (0,0) (helipad), step out cleanly through South doorway into corridor
                if cx < -4.2 and cy < -4.2:
                    exit_wpts = [(-5.10, -5.80), (-5.10, -6.80)]
                else:
                    exit_wpts = []
                with auto_lock:
                    auto_waypoints  = exit_wpts
                    auto_wpt_idx    = 0
                    mission_state   = MISSION_EXPLORING
                    auto_status_msg = "Takeoff Complete: Commencing Dynamic LiDAR Exploration"
                last_prog_pos = (cx, cy)
                last_prog_time = time.time()
                print("\n  🚀 TAKEOFF COMPLETE: Commencing Dynamic LiDAR Autonomous Exploration...")
                time.sleep(0.02)
                continue
            else:
                twist_msg.linear.x  = 0.0
                twist_msg.linear.y  = 0.0
                twist_msg.linear.z  = float(np.clip(1.3 * (DEFAULT_ALT - cz), 0.30, 0.65))
                twist_msg.angular.z = 0.0
                twist_pub.publish(twist_msg)
                time.sleep(0.03)
                continue

        # ── PHASE 2: DYNAMIC LIDAR & FRONTIER AUTONOMOUS EXPLORATION ──
        elif state == MISSION_EXPLORING:
            cov_pct = coverage_stats()[0]

            with grid_lock:
                g = grid.copy()
            with heatmap_lock:
                v_map = visited_heatmap.copy()
            with raw_lidar_lock:
                ranges = list(raw_lidar_ranges)

            # Check coverage threshold: >= 95.5% triggers automatic Return-To-Home!
            if cov_pct >= 95.5:
                print(f"\n  🎯 15x15M GRID EXPLORATION COMPLETE: Final Coverage {cov_pct:.1f}% (>=95% target achieved)!")
                print("  🏠 Initiating Autonomous Return-To-Home along explored corridor highway...")
                planned = astar_grid_path(g, (cx, cy), (home_x, home_y))
                with auto_lock:
                    mission_state   = MISSION_RTH
                    auto_waypoints  = planned
                    auto_wpt_idx    = 0
                    auto_status_msg = f"RTH Corridor Shortcut ({len(planned)} waypoints)"
                with rth_lock:
                    rth_waypoints   = planned
                    rth_wpt_idx     = 0
                    rth_active      = True
                    rth_done        = False
                continue

            with auto_lock:
                wpts = list(auto_waypoints)
                idx  = auto_wpt_idx

            # If there are active transit waypoints (e.g. initial South door exit or A* path to distant frontier):
            if wpts and idx < len(wpts):
                active_turn_dir = None
                target_pt = wpts[idx]
                tx, ty = target_pt
                dx = tx - cx; dy = ty - cy
                dist_to_wpt = math.hypot(dx, dy)

                _, _, d_f, d_fl, d_fr, _, _, _ = compute_lidar_avoidance(ranges)
                min_fwd = min(d_f, d_fl, d_fr)
                reached = (dist_to_wpt <= 0.38)
                if reached:
                    idx += 1
                    with auto_lock:
                        auto_wpt_idx = idx
                    last_prog_pos = (cx, cy)
                    last_prog_time = time.time()
                    if idx < len(wpts):
                        target_pt = wpts[idx]
                        tx, ty = target_pt
                        dx = tx - cx; dy = ty - cy
                        dist_to_wpt = math.hypot(dx, dy)
                    else:
                        wpts = []

                if wpts and idx < len(wpts):
                    target_heading = math.atan2(dy, dx)
                    yaw_err = (target_heading - yaw + math.pi) % (2 * math.pi) - math.pi
                    if abs(yaw_err) > 0.45:
                        vx_cmd = 0.08
                        vy_cmd = 0.0
                        wz_cmd = float(np.clip(1.6 * yaw_err, -1.2, 1.2))
                        status = f"[Transit A*] {cov_pct:.1f}% Cov | Aligning ({math.degrees(yaw_err):+.0f}°)"
                    else:
                        cruise_speed = min(0.65, max(0.35, 0.85 * dist_to_wpt))
                        vx_cmd = cruise_speed * math.cos(yaw_err)
                        wz_cmd = float(np.clip(1.3 * yaw_err, -0.65, 0.65))
                        vy_cmd = 0.0
                        status = f"[Transit A*] {cov_pct:.1f}% Cov | {vx_cmd:.2f} m/s ({idx+1}/{len(wpts)})"

                    # Apply real-time LiDAR obstacle tackling & repulsion
                    vx_cmd, vy_cmd, wz_cmd, obs_status = apply_lidar_obstacle_avoidance(
                        vx_cmd, vy_cmd, wz_cmd, yaw_err, ranges
                    )
                    status += obs_status

                    # Stuck progression watchdog for transit
                    now_t = time.time()
                    if last_prog_pos is None:
                        last_prog_pos = (cx, cy)
                        last_prog_time = now_t
                    dist_prog = math.hypot(cx - last_prog_pos[0], cy - last_prog_pos[1])
                    if dist_prog > 0.10:
                        last_prog_pos = (cx, cy)
                        last_prog_time = now_t
                    elif (now_t - last_prog_time) > 1.5:
                        # Stuck! Active escape push away from obstacle into open space
                        print(f"\n  ⚠️ WATCHDOG: Stuck near ({cx:+.2f}, {cy:+.2f}) -> Active escape push!")
                        esc_vx, esc_vy = compute_escape_vector(ranges)
                        for _ in range(12):
                            twist_msg.linear.x  = esc_vx
                            twist_msg.linear.y  = esc_vy
                            twist_msg.linear.z  = 0.0
                            twist_msg.angular.z = 0.0
                            twist_pub.publish(twist_msg)
                            time.sleep(0.03)

                        p_curr = world_to_cell(cx, cy)
                        if p_curr:
                            with heatmap_lock:
                                visited_heatmap[max(0, p_curr[1]-4):min(GRID_N, p_curr[1]+5),
                                                max(0, p_curr[0]-4):min(GRID_N, p_curr[0]+5)] += 80

                        with auto_lock:
                            auto_waypoints = []
                            auto_wpt_idx   = 0
                        last_prog_pos = (cx, cy)
                        last_prog_time = time.time()
                        continue

                    alt_err = DEFAULT_ALT - cz
                    twist_msg.linear.x  = float(np.clip(vx_cmd, -0.85, 0.85))
                    twist_msg.linear.y  = float(np.clip(vy_cmd, -0.5, 0.5))
                    twist_msg.linear.z  = float(np.clip(1.3 * alt_err, -0.5, 0.5))
                    twist_msg.angular.z = wz_cmd
                    twist_pub.publish(twist_msg)

                    with rth_lock:
                        rth_steer_vector = (math.cos(yaw)*vx_cmd - math.sin(yaw)*vy_cmd,
                                            math.sin(yaw)*vx_cmd + math.cos(yaw)*vy_cmd)
                    with auto_lock:
                        auto_status_msg = status
                    time.sleep(0.03)
                    continue

            # ── NO TRANSIT WAYPOINTS: PLAN PATH TO NEXT FRONTIER ──
            # Instead of blindly cruising forward, always plan a collision-free A* path to the
            # nearest reachable unexplored frontier.
            clusters = find_frontier_clusters(g, min_cluster_size=3)
            if clusters:
                # Sort clusters by distance from drone
                clusters.sort(key=lambda cl: math.hypot(cl[0]-cx, cl[1]-cy))

                transit_path = []
                chosen = None
                for cl in clusters:
                    fx, fy, c_size = cl
                    dist_to_frontier = math.hypot(fx - cx, fy - cy)
                    if dist_to_frontier > 0.40:
                        path = astar_grid_path(g, (cx, cy), (fx, fy))
                        if len(path) >= 2:
                            transit_path = path
                            chosen = cl
                            break

                if transit_path and chosen is not None:
                    fx, fy, c_size = chosen
                    print(f"\n  🧭 FRONTIER PLAN: Routing to ({fx:+.2f}, {fy:+.2f}) [{c_size} cells] ({len(transit_path)} waypoints)")
                    with auto_lock:
                        auto_waypoints = transit_path
                        auto_wpt_idx   = 0
                    last_prog_pos = (cx, cy)
                    last_prog_time = time.time()
                    continue
                else:
                    # No distant cluster has an unobstructed A* path yet:
                    # Steer toward local open space using real-time LiDAR clearance!
                    _, _, d_f, d_fl, d_fr, d_l, d_r, d_rear = compute_lidar_avoidance(ranges)
                    min_fwd = min(d_f, d_fl, d_fr)
                    if min_fwd >= 0.65:
                        vx_cmd = 0.35
                        vy_cmd = 0.0
                        wz_cmd = 0.0
                        status = f"[Local Scan] {cov_pct:.1f}% Cov | Cruising Open Space ({min_fwd:.2f}m)"
                    else:
                        # Turn toward the side with most clearance
                        vx_cmd = 0.0
                        vy_cmd = 0.0
                        wz_cmd = 1.0 if d_l > d_r else -1.0
                        status = f"[Local Scan] {cov_pct:.1f}% Cov | Turning to Open Corridor"
            else:
                # No frontiers remaining — exploration is complete!
                print(f"\n  🎯 Compound fully mapped (No remaining frontiers)! Final Coverage: {cov_pct:.1f}%")
                planned = astar_grid_path(g, (cx, cy), (home_x, home_y))
                with auto_lock:
                    mission_state   = MISSION_RTH
                    auto_waypoints  = planned
                    auto_wpt_idx    = 0
                    auto_status_msg = f"RTH Corridor Shortcut ({len(planned)} waypoints)"
                with rth_lock:
                    rth_waypoints   = planned
                    rth_wpt_idx     = 0
                    rth_active      = True
                    rth_done        = False
                continue

            # Command velocities
            alt_err = DEFAULT_ALT - cz
            vz_cmd = float(np.clip(1.3 * alt_err, -0.5, 0.5))

            twist_msg.linear.x  = float(np.clip(vx_cmd, -0.85, 0.85))
            twist_msg.linear.y  = float(np.clip(vy_cmd, -0.5, 0.5))
            twist_msg.linear.z  = vz_cmd
            twist_msg.angular.z = wz_cmd
            twist_pub.publish(twist_msg)

            sv_world_x = math.cos(yaw) * vx_cmd - math.sin(yaw) * vy_cmd
            sv_world_y = math.sin(yaw) * vx_cmd + math.cos(yaw) * vy_cmd
            with rth_lock:
                rth_steer_vector = (sv_world_x, sv_world_y)
                rth_status_msg   = status
            with auto_lock:
                auto_status_msg = status

            time.sleep(0.03)

        # ── PHASE 3 & 4: RETURN-TO-HOME & LANDING ──
        elif state in (MISSION_RTH, MISSION_LANDING):
            with rth_lock:
                done = rth_done
            if done:
                with auto_lock:
                    mission_state   = MISSION_COMPLETE
                    auto_status_msg = "MISSION COMPLETE: 100% EXPLORED & LANDED ON HELIPAD"
                    auto_mode       = False
            time.sleep(0.1)

        elif state == MISSION_COMPLETE:
            time.sleep(0.2)


# ══════════════════════════════════════════════════════════════════════
#  COVERAGE STATS (15x15M ARENA)
# ══════════════════════════════════════════════════════════════════════
def coverage_stats():
    with grid_lock:
        g = grid.copy()

    # 15m x 15m arena bounding box (world coords [-7.5, 7.5] x [-7.5, 7.5])
    c_min, r_min = world_to_cell(-7.5, -7.5) or (0, 0)
    c_max, r_max = world_to_cell( 7.5,  7.5) or (GRID_N, GRID_N)
    c0 = max(0, min(c_min, c_max))
    c1 = min(GRID_N, max(c_min, c_max))
    r0 = max(0, min(r_min, r_max))
    r1 = min(GRID_N, max(r_min, r_max))

    arena_sub = g[r0:r1, c0:c1]
    if arena_sub.size > 0:
        known_arena = np.count_nonzero(arena_sub != UNKNOWN)
        arena_pct = (known_arena / float(arena_sub.size)) * 100.0
    else:
        arena_pct = 0.0

    unk  = int(np.sum(g == UNKNOWN))
    occ  = int(np.sum(g == OCCUPIED))
    path = int(np.sum(g == PATH_C))
    free = int(np.sum(g == FREE))
    return arena_pct, free, path, occ, unk


# ══════════════════════════════════════════════════════════════════════
#  SAVE FINAL MAP
# ══════════════════════════════════════════════════════════════════════
def save_map():
    with grid_lock:
        g = grid.copy()
    with map_pose_lock:
        fx = map_drone_x or home_x or 0.0
        fy = map_drone_y or home_y or 0.0

    def make_rgb(g_arr):
        rgb = np.zeros((GRID_N, GRID_N, 3), dtype=np.uint8)
        rgb[g_arr == UNKNOWN]  = [160, 160, 160]
        rgb[g_arr == FREE]     = [235, 235, 235]
        rgb[g_arr == PATH_C]   = [80,  140, 255]
        rgb[g_arr == OCCUPIED] = [25,  25,  25]
        return rgb

    rgb = make_rgb(g)

    # Home marker (green cross)
    if home_x is not None:
        hc, hr = GRID_N // 2, GRID_N // 2
        for d in range(-5, 6):
            for nc, nr in [(hc + d, hr), (hc, hr + d)]:
                if 0 <= nc < GRID_N and 0 <= nr < GRID_N:
                    rgb[nr, nc] = [0, 220, 80]

    # Final drone position (red dot)
    fp = world_to_cell(fx, fy)
    if fp:
        fc, fr = fp
        for dc in range(-4, 5):
            for dr in range(-4, 5):
                if dc**2 + dr**2 <= 16:
                    nc, nr = fc + dc, fr + dr
                    if 0 <= nc < GRID_N and 0 <= nr < GRID_N:
                        rgb[nr, nc] = [255, 50, 50]

    pct, fc_c, path_c, occ_c, unk_c = coverage_stats()

    # Raw map save
    fig_r, ax_r = plt.subplots(figsize=(8, 8), dpi=150)
    ax_r.imshow(rgb, origin="lower"); ax_r.axis("off")
    fig_r.tight_layout(pad=0)
    fig_r.savefig(SAVE_RAW, bbox_inches="tight", pad_inches=0)
    plt.close(fig_r)

    # Annotated map save
    half = HALF_SPAN
    fig_a, ax_a = plt.subplots(figsize=(11, 11), dpi=150)
    ax_a.imshow(rgb, origin="lower", extent=[-half, half, -half, half])
    for v in np.arange(-half, half + 1, 1.0):
        ax_a.axhline(v, color="gray", lw=0.3, alpha=0.4)
        ax_a.axvline(v, color="gray", lw=0.3, alpha=0.4)
    ax_a.set_xlabel("X offset from start (m)", fontsize=11)
    ax_a.set_ylabel("Y offset from start (m)", fontsize=11)
    ax_a.set_title(
        f"NIDAR Drone — Final 2D Occupancy Map\n"
        f"Coverage: {pct:.1f}%  |  Free: {fc_c * CELL_M**2:.1f} m²  |  "
        f"Walls: {occ_c}  |  Start: ({home_x:+.2f}, {home_y:+.2f}) m",
        fontsize=11, fontweight="bold"
    )
    leg = [
        mpatches.Patch(color=[0.10,0.10,0.10], label="Wall"),
        mpatches.Patch(color=[0.92,0.92,0.92], label="Free"),
        mpatches.Patch(color=[0.31,0.55,1.00], label="Drone Path"),
        mpatches.Patch(color=[0.63,0.63,0.63], label="Unexplored"),
        mpatches.Patch(color=[0.00,0.86,0.31], label="Home"),
        mpatches.Patch(color=[1.00,0.20,0.20], label="Final Pos"),
    ]
    ax_a.legend(handles=leg, loc="upper right", fontsize=9, framealpha=0.9)
    ax_a.plot(0, 0, 'g^', markersize=13, zorder=6)
    fig_a.tight_layout()
    fig_a.savefig(SAVE_ANNOT, bbox_inches="tight")
    plt.close(fig_a)

    print(f"\n✅ Maps saved:\n   {SAVE_RAW}\n   {SAVE_ANNOT}")
    print(f"\n📊 Final Stats:\n   Coverage: {pct:.1f}%  |  Free: {fc_c*CELL_M**2:.1f} m²"
          f"  |  Walls: {occ_c}  |  Path: {path_c} cells")
    if home_x is not None:
        print(f"   Start pos: ({home_x:+.3f}, {home_y:+.3f}) m [world]")


# ══════════════════════════════════════════════════════════════════════
#  BUILD HUD TEXT CONTENT
# ══════════════════════════════════════════════════════════════════════
def build_hud_lines():
    lines = []

    # ── Header ──
    lines.append(("🚁  NIDAR DRONE  —  COMMAND CENTER", "#00e5ff", 11, True))
    lines.append(("━" * 36, "#444466", 9, False))

    # ── Flight Status ──
    with rth_lock:
        is_rth   = rth_active
        is_done  = rth_done
        wpts     = list(rth_waypoints)
        w_idx    = rth_wpt_idx
        st_msg   = rth_status_msg
    with auto_lock:
        is_auto  = auto_mode
        m_state  = mission_state
        a_status = auto_status_msg
        a_wpts   = list(auto_waypoints)
        a_idx    = auto_wpt_idx

    if not motors_active:
        fmode = "🔴  MOTORS DISARMED"
        fcolor = "#ff4444"
    elif is_auto:
        fmode = f"🚀  AUTO-EXPLORE: {m_state}"
        fcolor = "#00e5ff"
    elif is_rth:
        fmode = f"🏠  RTH: Wpt {w_idx+1}/{max(1, len(wpts))} [{st_msg}]"
        fcolor = "#00e5ff"
    elif is_done:
        fmode = "✅  HOME SAFELY REACHED"
        fcolor = "#44ff88"
    elif abs(current_altitude - target_altitude) < 0.12 and abs(cmd_vx) < 0.1 and abs(cmd_vy) < 0.1:
        fmode = "🟢  HOVER LOCKED (1.50m)"
        fcolor = "#44ff88"
    elif cmd_vx != 0 or cmd_vy != 0 or cmd_wz != 0:
        fmode = "🔵  MANUAL STEERING"
        fcolor = "#44aaff"
    elif current_altitude < target_altitude:
        fmode = "🟡  CLIMBING"
        fcolor = "#ffee44"
    else:
        fmode = "🟡  DESCENDING"
        fcolor = "#ffee44"

    lines.append(("FLIGHT STATUS", "#aaaacc", 9, True))
    lines.append((fmode, fcolor, 9, True))
    if is_auto:
        if m_state == MISSION_TAKEOFF:
            lines.append((f"  Takeoff: {a_status}", "#ffffaa", 8.5, False))
        elif a_wpts and m_state == MISSION_EXPLORING:
            lines.append((f"  Wpt {a_idx+1}/{len(a_wpts)}: {a_status}", "#ffffaa", 8.5, False))

    bar_n   = 20
    filled  = int(min(1.0, max(0.0, current_altitude / MAX_CEILING)) * bar_n)
    alt_bar = "█" * filled + "░" * (bar_n - filled)
    lines.append((f"  Alt: [{alt_bar}] {current_altitude:.2f}m / {MAX_CEILING:.1f}m", "#ccccee", 8.5, False))
    lines.append((f"  Target: {target_altitude:.2f}m   Climb: {vertical_velocity:+.2f} m/s", "#aaaacc", 8.5, False))
    lines.append((f"  Vx:{cmd_vx:+.2f}  Vy:{cmd_vy:+.2f}  Wz:{cmd_wz:+.2f}", "#aaaacc", 8.5, False))

    if current_altitude >= MAX_CEILING - 0.05:
        lines.append(("  ⚠ CEILING LOCK ACTIVE", "#ff6633", 8.5, True))

    lines.append(("━" * 36, "#444466", 9, False))

    # ── Sensors ──
    lines.append(("SENSORS", "#aaaacc", 9, True))
    ld = f"🟢 {lidar_rays_count} rays" if lidar_connected else "⚪ connecting..."
    sc = f"🟢 {stereo_res} @ {stereo_fps:.0f}fps" if stereo_connected else "⚪ connecting..."
    cc = f"🟢 {cam5_res} @ {cam5_fps:.0f}fps" if cam5_connected else "⚪ connecting..."
    lines.append((f"  LiDAR 360°       {ld}", "#ccffcc" if lidar_connected else "#888888", 8.5, False))
    lines.append((f"  Stereo Depth     {sc}", "#ccffcc" if stereo_connected else "#888888", 8.5, False))
    lines.append((f"  IMX335 5MP WDR   {cc}", "#ccffcc" if cam5_connected else "#888888", 8.5, False))
    lines.append(("━" * 36, "#444466", 9, False))

    # ── Obstacle Radar ──
    lines.append(("OBSTACLE RADAR  (5.0 m radius)", "#aaaacc", 9, True))
    if obstacle_detected:
        d   = closest_dist
        ang = closest_angle_deg
        if d <= CRITICAL_DIST:
            warn = f"🚨 CRITICAL  {d:.2f} m"
            wc   = "#ff3333"
        elif d <= WARNING_DIST:
            warn = f"⚠  WARNING   {d:.2f} m"
            wc   = "#ffaa00"
        else:
            warn = f"📢 CAUTION   {d:.2f} m"
            wc   = "#ffdd44"
        bear = f"{abs(ang):.1f}° {'L' if ang > 0 else 'R'}" if abs(ang) > 2.0 else "Center"
        lines.append((f"  {warn}  ({bear})", wc, 9, True))
    else:
        lines.append(("  🟢 PATH CLEAR", "#44ff88", 9, False))

    fl = fmt_dist(sector_ranges["left"])
    ffl= fmt_dist(sector_ranges["front_left"])
    ff = fmt_dist(sector_ranges["front"])
    ffr= fmt_dist(sector_ranges["front_right"])
    fr = fmt_dist(sector_ranges["right"])
    lines.append((f"  L:{fl}  FL:{ffl}  F:{ff}", "#aaaacc", 8.5, False))
    lines.append((f"  FR:{ffr}  R:{fr}", "#aaaacc", 8.5, False))
    lines.append(("━" * 36, "#444466", 9, False))

    # ── Navigation & Map Stats ──
    pct, fc_c, path_c, occ_c, unk_c = coverage_stats()
    with map_pose_lock:
        cx = map_drone_x; cy = map_drone_y
    dist_home = math.hypot(cx - home_x, cy - home_y) if (cx is not None and home_x is not None) else 0.0
    lines.append(("NAVIGATION & MAP", "#aaaacc", 9, True))
    lines.append((f"  Coverage: {pct:.1f}%  |  Walls Mapped: {occ_c}", "#ccccee", 8.5, False))
    lines.append((f"  Free Area: {fc_c * CELL_M**2:.1f} m²  |  Path: {path_c} cells", "#ccccee", 8.5, False))
    if home_x is not None and cx is not None:
        lines.append((f"  Drone: ({cx-home_x:+.1f}m, {cy-home_y:+.1f}m) from home", "#ccccee", 8.5, False))
        lines.append((f"  Straight Dist to Home: {dist_home:.2f} m", "#ccccee", 8.5, False))
    lines.append(("━" * 36, "#444466", 9, False))

    # ── Controls Guide ──
    lines.append(("CONTROLS", "#aaaacc", 9, True))
    ctrl = [
        ("  E",         "Toggle AUTONOMOUS 15x15m Exploration"),
        ("  ↑ / ↓",    "Fly Forward / Backward"),
        ("  ← / →",    "Strafe Left / Right"),
        ("  W / S",     "Altitude Up / Down"),
        ("  A / D",     "Yaw Left / Right"),
        ("  SPACE",     "Hover at safe 1.5 m"),
        ("  H",         "SMART Return-To-Home (Corridor Route)"),
        ("  R",         "Reset simulation to Helipad"),
        ("  X",         "Land / Touch down"),
        ("  Q",         "Quit + Save Map"),
    ]
    for key_str, desc in ctrl:
        lines.append((f"{key_str:<10} {desc}", "#ccccee", 8.5, False))

    return lines


# ══════════════════════════════════════════════════════════════════════
#  MAIN — MATPLOTLIB DASHBOARD
# ══════════════════════════════════════════════════════════════════════
def main():
    global running, rth_active, rth_done, rth_waypoints, rth_wpt_idx
    global target_altitude, motors_active, cmd_vx, cmd_vy, cmd_wz, last_key_time
    global auto_mode, mission_state, auto_status_msg

    detect_world()
    print("Starting NIDAR Command Center — waiting for Gazebo...")

    # Start threads
    for fn in (pose_thread, flight_ctrl_thread, rth_thread, auto_mission_thread):
        threading.Thread(target=fn, daemon=True).start()

    home_set.wait(timeout=15.0)
    if not home_set.is_set():
        print("⚠  No pose from Gazebo. Is the simulation running?")
        running = False
        return

    print(f"Home locked at ({home_x:+.3f}, {home_y:+.3f}) m — opening dashboard...")

    # Auto-start if --auto flag passed
    if "--auto" in sys.argv:
        with auto_lock:
            auto_mode     = True
            mission_state = MISSION_TAKEOFF
        print("  🚀 AUTO FLAG DETECTED: Autonomous 15x15m Exploration will begin immediately!")

    # ── Figure layout ──
    plt.ion()
    fig = plt.figure(figsize=(18, 9), facecolor="#0d1117")
    fig.canvas.manager.set_window_title(
        "NIDAR Command Center  [↑↓←→=Move  W/S=Alt  A/D=Yaw  H=Smart RTH  Q=Quit]"
    )

    gs   = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.15, 0.85], wspace=0.04)
    ax_m = fig.add_subplot(gs[0])   # left: map
    ax_h = fig.add_subplot(gs[1])   # right: HUD

    # ── Map axis setup ──
    ax_m.set_facecolor("#111827")
    half = HALF_SPAN
    init_rgb = np.full((GRID_N, GRID_N, 3), 160, dtype=np.uint8)
    im = ax_m.imshow(init_rgb, origin="lower",
                     extent=[-half, half, -half, half], interpolation="nearest")
    for v in np.arange(-half, half + 1, 1.0):
        ax_m.axhline(v, color="white", lw=0.25, alpha=0.18)
        ax_m.axvline(v, color="white", lw=0.25, alpha=0.18)

    # Home symbol
    ax_m.plot([0], [0], 'g^', markersize=11, zorder=6)
    ax_m.annotate("HOME", (0, 0), textcoords="offset points", xytext=(5, 4),
                  color="lime", fontsize=7.5, fontweight="bold")

    # Planned RTH path vector line (cyan)
    rth_path_line, = ax_m.plot([], [], color='#00f0ff', lw=2.4, linestyle='--',
                               marker='o', markersize=4, zorder=8, label='RTH Path')

    # Real-time steering vector arrow (bright green)
    steer_arrow = ax_m.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="-|>", color="#39ff14", lw=2.5, mutation_scale=14),
                                zorder=9)

    # Drone marker and heading arrow
    drone_dot, = ax_m.plot([], [], 'r^', markersize=10, zorder=7)
    drone_arr  = ax_m.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="->", color="red", lw=2))

    ax_m.set_xlabel("X from start (m)", color="white", fontsize=9)
    ax_m.set_ylabel("Y from start (m)", color="white", fontsize=9)
    ax_m.tick_params(colors="white")
    for sp in ax_m.spines.values():
        sp.set_edgecolor("#334455")

    map_legend = [
        mpatches.Patch(color=[0.10,0.10,0.10], label="Wall"),
        mpatches.Patch(color=[0.92,0.92,0.92], label="Free"),
        mpatches.Patch(color=[0.31,0.55,1.00], label="Path"),
        mpatches.Patch(color=[0.63,0.63,0.63], label="Unknown"),
        mpatches.Patch(color=[0.00,0.94,1.00], label="RTH Vector Path"),
        mpatches.Patch(color=[0.22,1.00,0.08], label="Steer Vector"),
    ]
    ax_m.legend(handles=map_legend, loc="upper right", fontsize=7.5,
                facecolor="#111827", edgecolor="#334455", labelcolor="white")
    map_title = ax_m.set_title("NIDAR — Live 2D Map", color="white",
                                fontsize=10, fontweight="bold", pad=6)

    # ── HUD axis setup ──
    ax_h.set_facecolor("#0d1117")
    ax_h.axis("off")

    # ── Key bindings ──
    def on_key_press(event):
        global target_altitude, motors_active, cmd_vx, cmd_vy, cmd_wz
        global last_key_time, running, rth_active, rth_done, rth_waypoints, rth_wpt_idx
        global auto_mode, mission_state

        k = event.key
        if k in ('q', 'Q'):
            running = False
            return

        # Cancel RTH and Auto-Exploration only if explicit manual flight control keys are pressed
        flight_keys = ('up', 'down', 'left', 'right', 'w', 'W', 's', 'S', 'a', 'A', 'd', 'D', ' ')
        if k in flight_keys:
            with rth_lock:
                if rth_active:
                    rth_active = False
                    print("\n  ⚠ Manual takeover: Smart RTH paused.")
            with auto_lock:
                if auto_mode:
                    auto_mode = False
                    mission_state = MISSION_IDLE
                    print("\n  ⚠ Manual takeover: Auto-exploration paused.")

        if k in ('e', 'E'):
            with rth_lock:
                rth_active = False
                rth_done   = False
                rth_waypoints = []
                rth_wpt_idx = 0
            with auto_lock:
                if not auto_mode:
                    auto_mode     = True
                    mission_state = MISSION_TAKEOFF
                    print("\n  🚀 AUTONOMOUS 15x15M EXPLORATION ACTIVATED: Auto-takeoff initiated...")
                else:
                    auto_mode     = False
                    mission_state = MISSION_IDLE
                    print("\n  ⏸ AUTONOMOUS EXPLORATION PAUSED.")
        elif k in ('w', 'W'):
            motors_active = True
            target_altitude = min(MAX_CEILING, round(target_altitude + 0.25, 2))
            last_key_time = time.time()
        elif k in ('s', 'S'):
            target_altitude = max(MIN_ALT, round(target_altitude - 0.25, 2))
            last_key_time = time.time()
        elif k == 'up':
            motors_active = True; cmd_vx = 1.0; last_key_time = time.time()
        elif k == 'down':
            motors_active = True; cmd_vx = -1.0; last_key_time = time.time()
        elif k == 'left':
            motors_active = True; cmd_vy = 0.8; last_key_time = time.time()
        elif k == 'right':
            motors_active = True; cmd_vy = -0.8; last_key_time = time.time()
        elif k in ('a', 'A'):
            motors_active = True; cmd_wz = 0.7; last_key_time = time.time()
        elif k in ('d', 'D'):
            motors_active = True; cmd_wz = -0.7; last_key_time = time.time()
        elif k == ' ':
            motors_active = True
            target_altitude = DEFAULT_ALT
            cmd_vx = cmd_vy = cmd_wz = 0.0
            last_key_time = time.time()
        elif k in ('h', 'H'):
            # Trigger Smart Collision-Free Return-To-Home along explored corridors
            with map_pose_lock:
                cx = map_drone_x; cy = map_drone_y
            if cx is not None and home_x is not None:
                planned = plan_rth_path((cx, cy), (home_x, home_y))
                with rth_lock:
                    rth_waypoints = planned
                    rth_wpt_idx   = 0
                    rth_active    = True
                    rth_done      = False
                print(f"\n  🏠 SMART RTH ACTIVATED: Planned {len(planned)} collision-free corridor waypoints to Home!")
        elif k in ('r', 'R'):
            reset_sim()
        elif k in ('x', 'X'):
            motors_active = False
            cmd_vx = cmd_vy = cmd_wz = 0.0
            target_altitude = 0.0

    def on_key_release(event):
        global cmd_vx, cmd_vy, cmd_wz
        k = event.key
        if k in ('up', 'down'):
            cmd_vx = 0.0
        elif k in ('left', 'right'):
            cmd_vy = 0.0
        elif k in ('a', 'A', 'd', 'D'):
            cmd_wz = 0.0

    fig.canvas.mpl_connect("key_press_event",   on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    start_t      = time.time()
    last_refresh = 0.0

    try:
        while running and plt.fignum_exists(fig.number):
            now = time.time()

            if now - last_refresh >= MAP_REFRESH_S:
                last_refresh = now

                # ── Rebuild map RGB ──
                with grid_lock:
                    g = grid.copy()

                rgb = np.zeros((GRID_N, GRID_N, 3), dtype=np.uint8)
                rgb[g == UNKNOWN]  = [160, 160, 160]
                rgb[g == FREE]     = [235, 235, 235]
                rgb[g == PATH_C]   = [80,  140, 255]
                rgb[g == OCCUPIED] = [25,  25,  25]

                # Home cross
                hc = hr = GRID_N // 2
                for d in range(-4, 5):
                    for nc, nr in [(hc + d, hr), (hc, hr + d)]:
                        if 0 <= nc < GRID_N and 0 <= nr < GRID_N:
                            rgb[nr, nc] = [0, 220, 80]

                # Drone position
                with map_pose_lock:
                    cx = map_drone_x; cy = map_drone_y; cyaw = map_drone_yaw

                if cx is not None and home_x is not None:
                    cp = world_to_cell(cx, cy)
                    if cp:
                        cc, cr = cp
                        for dc in range(-3, 4):
                            for dr in range(-3, 4):
                                if dc**2 + dr**2 <= 9:
                                    nc, nr = cc + dc, cr + dr
                                    if 0 <= nc < GRID_N and 0 <= nr < GRID_N:
                                        rgb[nr, nc] = [255, 50, 50]

                im.set_data(rgb)

                # Update drone marker and heading
                if cx is not None and home_x is not None:
                    rx = cx - home_x; ry = cy - home_y
                    drone_dot.set_data([rx], [ry])
                    al = 0.6
                    drone_arr.xy     = (rx + al * math.cos(cyaw), ry + al * math.sin(cyaw))
                    drone_arr.xytext = (rx, ry)

                    # Update planned path vector (RTH or Auto-Exploration) & real-time steering vector
                    with rth_lock:
                        is_rth   = rth_active
                        wpts     = list(rth_waypoints)
                        steer_v  = rth_steer_vector
                    with auto_lock:
                        is_auto  = auto_mode
                        m_state  = mission_state
                        a_wpts   = list(auto_waypoints)

                    active_wpts = wpts if is_rth else (a_wpts if (is_auto and m_state == MISSION_EXPLORING) else [])
                    if active_wpts:
                        path_rx = [wx - home_x for wx, _ in active_wpts]
                        path_ry = [wy - home_y for _, wy in active_wpts]
                        rth_path_line.set_data(path_rx, path_ry)

                        svx, svy = steer_v
                        v_len = math.hypot(svx, svy)
                        if v_len > 0.05:
                            arr_scale = 0.8 / max(v_len, 0.01)
                            steer_arrow.xy     = (rx + svx * arr_scale, ry + svy * arr_scale)
                            steer_arrow.xytext = (rx, ry)
                            steer_arrow.set_visible(True)
                        else:
                            steer_arrow.set_visible(False)
                    else:
                        rth_path_line.set_data([], [])
                        steer_arrow.set_visible(False)

                    pct = coverage_stats()[0]
                    elapsed = int(now - start_t); mm, ss = divmod(elapsed, 60)
                    if is_rth:
                        mode_lbl = " [🏠 SMART RTH ACTIVE]"
                    elif is_auto:
                        mode_lbl = f" [🚀 AUTO-EXPLORE: {m_state}]"
                    else:
                        mode_lbl = ""
                    map_title.set_text(
                        f"NIDAR — Live 2D Map{mode_lbl}  |  15x15m Coverage: {pct:.1f}%  |  ⏱ {mm:02d}:{ss:02d}"
                    )

                # ── Rebuild HUD panel ──
                ax_h.cla()
                ax_h.set_facecolor("#0d1117")
                ax_h.axis("off")
                ax_h.set_xlim(0, 1); ax_h.set_ylim(0, 1)

                hud_lines = build_hud_lines()
                n    = len(hud_lines)
                ypos = 0.98
                step = 0.98 / max(n, 1)

                for txt, color, fsize, bold in hud_lines:
                    ax_h.text(
                        0.03, ypos, txt,
                        transform=ax_h.transAxes,
                        color=color,
                        fontsize=fsize,
                        fontweight="bold" if bold else "normal",
                        fontfamily="monospace",
                        va="top", ha="left",
                    )
                    ypos -= step

                fig.canvas.draw_idle()

            plt.pause(0.04)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        twist_msg.linear.x = twist_msg.linear.y = twist_msg.linear.z = twist_msg.angular.z = 0.0
        twist_pub.publish(twist_msg)
        plt.ioff()
        plt.close("all")
        save_map()
        print("\nNIDAR Command Center closed.")


if __name__ == "__main__":
    main()

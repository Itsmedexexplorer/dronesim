#!/usr/bin/env python3
"""
NIDAR AI Drone — Live 2D Occupancy Grid Mapper (v3: Smart Collision-Free RTH)
=============================================================================
Runs alongside teleop_drone.py in a second terminal:

    Terminal 1:  python3 teleop_drone.py
    Terminal 2:  python3 map/drone_mapper.py

Features (v3):
  • Dynamic origin locked to drone's actual start position (no hardcoded center)
  • Smart Collision-Free Return-To-Home (Press H):
      - Minimal corridor route planning (shortcutted corridor breadcrumbs)
      - Coordinated forward flight & corner turning (no sideways sliding into walls)
      - Gentle lateral corridor centering via LiDAR
      - Live visualization of planned return path (cyan) & steering vector (green)
  • Saves final high-resolution map on exit (Q)
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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from gz.transport import Node
from gz.msgs.laserscan_pb2 import LaserScan
from gz.msgs.twist_pb2 import Twist

# ═══════════════════════════ Configuration ════════════════════════════
CELL_SIZE_M    = 0.05           # 5 cm per cell
GRID_SPAN_M    = 30.0           # grid covers ±15 m from start in each axis
GRID_N         = int(GRID_SPAN_M / CELL_SIZE_M)   # 600 × 600 cells
HALF_SPAN      = GRID_SPAN_M / 2.0                # 15 m each side from start

WORLD_NAME     = "nidar_world"
LIDAR_TOPIC    = "/nidar/lidar"
POSE_TOPIC     = f"/world/{WORLD_NAME}/pose/info"
TWIST_TOPIC    = "/X3/gazebo/command/twist"
DRONE_NAME     = "x3"

MAP_REFRESH_S  = 0.5

SAVE_DIR       = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH      = os.path.join(SAVE_DIR, "drone_map_final.png")
SAVE_ANNOT     = os.path.join(SAVE_DIR, "drone_map_annotated.png")

# RTH navigation parameters
RTH_MAX_VEL     = 0.85
RTH_WPT_THRESH  = 0.45
RTH_HOME_THRESH = 0.35

# Cell state values
UNKNOWN  = 128
FREE     = 240
OCCUPIED = 0
PATH     = 180

# ═══════════════════════ Dynamic-Origin Grid State ═════════════════════
grid      = np.full((GRID_N, GRID_N), UNKNOWN, dtype=np.uint8)
grid_lock = threading.Lock()

# Visited Heatmap for anti-looping exploration & raw LiDAR buffer
visited_heatmap     = np.zeros((GRID_N, GRID_N), dtype=np.uint32)
heatmap_lock        = threading.Lock()
raw_lidar_ranges    = []
raw_lidar_lock      = threading.Lock()

home_x   = None
home_y   = None
home_set = threading.Event()

drone_x   = None
drone_y   = None
drone_z   = 1.5
drone_yaw = 0.0
pose_lock = threading.Lock()

# Smart RTH state
rth_active        = False
rth_done          = False
rth_waypoints     = []
rth_wpt_idx       = 0
rth_steer_vector  = (0.0, 0.0)
rth_lock          = threading.Lock()

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

# Breadcrumbs
breadcrumbs    = []
last_bread_pos = None
bread_lock     = threading.Lock()

# Sector ranges
sector_ranges = {"left": float('inf'), "right": float('inf'), "front": float('inf')}

running = True

# ════════════════════════════ Helpers ══════════════════════════════════
def world_to_cell(wx, wy):
    if home_x is None:
        return None
    col = int((wx - home_x + HALF_SPAN) / CELL_SIZE_M)
    row = int((wy - home_y + HALF_SPAN) / CELL_SIZE_M)
    if 0 <= col < GRID_N and 0 <= row < GRID_N:
        return col, row
    return None


def cell_to_world(col, row):
    if home_x is None:
        return 0.0, 0.0
    wx = (col * CELL_SIZE_M) - HALF_SPAN + home_x
    wy = (row * CELL_SIZE_M) - HALF_SPAN + home_y
    return wx, wy


def bresenham(c0, r0, c1, r1):
    dc = abs(c1 - c0); dr = abs(r1 - r0)
    sc = 1 if c0 < c1 else -1; sr = 1 if r0 < r1 else -1
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


# ══════════════════════ Smart Path Planning ════════════════════════════
def plan_rth_path(start_xy, goal_xy):
    with bread_lock:
        trail_pts = list(breadcrumbs)
    with grid_lock:
        g = grid.copy()

    if line_is_clear_of_walls(start_xy, goal_xy, g, margin_cells=4):
        return [start_xy, goal_xy]

    if not trail_pts:
        return [start_xy, goal_xy]

    closest_i = len(trail_pts) - 1
    min_d = float('inf')
    for i, p in enumerate(trail_pts):
        d = math.hypot(p[0] - start_xy[0], p[1] - start_xy[1])
        if d < min_d:
            min_d = d
            closest_i = i

    raw_path = [start_xy]
    for i in range(closest_i, -1, -1):
        pt = trail_pts[i]
        if math.hypot(pt[0] - raw_path[-1][0], pt[1] - raw_path[-1][1]) >= 0.20:
            raw_path.append(pt)
    if math.hypot(raw_path[-1][0] - goal_xy[0], raw_path[-1][1] - goal_xy[1]) >= 0.20:
        raw_path.append(goal_xy)

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


# ════════════════════════ Pose Reader Thread ═══════════════════════════
def pose_reader_thread():
    global drone_x, drone_y, drone_z, drone_yaw, home_x, home_y, running, last_bread_pos

    proc = subprocess.Popen(
        ["gz", "topic", "-t", POSE_TOPIC, "-e"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )

    curr_entity = None
    in_pos = False; in_orient = False
    tmp_x = tmp_y = tmp_z = 0.0
    tmp_qx = tmp_qy = tmp_qz = tmp_qw = 1.0

    while running:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()

        m = re.match(r'name:\s*"([^"]+)"', line)
        if m:
            curr_entity = m.group(1)
            in_pos = False; in_orient = False

        if curr_entity != DRONE_NAME:
            continue

        if line == "position {":    in_pos = True
        if line == "orientation {": in_orient = True
        if line == "}" and in_pos:   in_pos = False
        if line == "}" and in_orient:
            siny = 2.0 * (tmp_qw * tmp_qz + tmp_qx * tmp_qy)
            cosy = 1.0 - 2.0 * (tmp_qy ** 2 + tmp_qz ** 2)
            yaw  = math.atan2(siny, cosy)

            with pose_lock:
                drone_x   = tmp_x
                drone_y   = tmp_y
                drone_z   = tmp_z
                drone_yaw = yaw

            if not home_set.is_set() and tmp_x is not None:
                home_x = tmp_x; home_y = tmp_y
                home_set.set()
                with bread_lock:
                    breadcrumbs.append((home_x, home_y))
                    last_bread_pos = (home_x, home_y)
                print(f"  📍 Home origin locked at X={home_x:+.3f}m, Y={home_y:+.3f}m")

            if home_set.is_set():
                with bread_lock:
                    if last_bread_pos is None:
                        breadcrumbs.append((tmp_x, tmp_y))
                        last_bread_pos = (tmp_x, tmp_y)
                    elif math.hypot(tmp_x - last_bread_pos[0], tmp_y - last_bread_pos[1]) >= 0.20:
                        breadcrumbs.append((tmp_x, tmp_y))
                        last_bread_pos = (tmp_x, tmp_y)

            in_orient = False

        if in_pos:
            for tag in ('x', 'y', 'z'):
                m2 = re.match(rf'{tag}:\s*([\-\d.eE+]+)', line)
                if m2:
                    val = float(m2.group(1))
                    if tag == 'x': tmp_x = val
                    elif tag == 'y': tmp_y = val
                    elif tag == 'z': tmp_z = val
        if in_orient:
            for tag in ('x', 'y', 'z', 'w'):
                m2 = re.match(rf'{tag}:\s*([\-\d.eE+]+)', line)
                if m2:
                    val = float(m2.group(1))
                    if tag == 'x': tmp_qx = val
                    elif tag == 'y': tmp_qy = val
                    elif tag == 'z': tmp_qz = val
                    elif tag == 'w': tmp_qw = val

    proc.kill()


# ══════════════════════ LiDAR Callback (Ray Caster) ════════════════════
def lidar_callback(msg: LaserScan):
    global sector_ranges, raw_lidar_ranges

    if not home_set.is_set():
        return

    with pose_lock:
        ox    = drone_x
        oy    = drone_y
        yaw   = drone_yaw

    if ox is None:
        return

    origin = world_to_cell(ox, oy)
    if origin is None:
        return

    oc, or_ = origin
    n_rays  = len(msg.ranges)
    if n_rays == 0:
        return

    with raw_lidar_lock:
        raw_lidar_ranges = list(msg.ranges)

    with heatmap_lock:
        visited_heatmap[max(0, or_-2):min(GRID_N, or_+3),
                        max(0, oc-2):min(GRID_N, oc+3)] += 1

    angle_min  = msg.angle_min
    angle_step = msg.angle_step

    sl = sf = sr = float('inf')

    with grid_lock:
        for i, r in enumerate(msg.ranges):
            theta_local = angle_min + i * angle_step
            theta_world = theta_local + yaw
            valid = not (math.isinf(r) or math.isnan(r)) and (msg.range_min <= r <= msg.range_max)
            ray_r = r if valid else msg.range_max

            ex = ox + ray_r * math.cos(theta_world)
            ey = oy + ray_r * math.sin(theta_world)
            ep = world_to_cell(ex, ey)
            if ep is None:
                continue

            ec, er = ep
            for c, ro in bresenham(oc, or_, ec, er):
                if 0 <= c < GRID_N and 0 <= ro < GRID_N:
                    if grid[ro, c] == UNKNOWN:
                        grid[ro, c] = FREE
            if valid and 0 <= ec < GRID_N and 0 <= er < GRID_N:
                grid[er, ec] = OCCUPIED

            if valid:
                td = math.degrees((theta_local + math.pi) % (2 * math.pi) - math.pi)
                if -25 <= td <= 25 and r < sf: sf = r
                elif 50 <= td <= 110 and r < sl: sl = r
                elif -110 <= td <= -50 and r < sr: sr = r

        if grid[or_, oc] != OCCUPIED:
            grid[or_, oc] = PATH

    sector_ranges["left"] = sl
    sector_ranges["front"] = sf
    sector_ranges["right"] = sr


# ══════════════════════════════════════════════════════════════════════
#  REAL-TIME 360° LIDAR OBSTACLE TACKLING & COLLISION EVASION
# ══════════════════════════════════════════════════════════════════════
def compute_lidar_avoidance(ranges, angle_min=-math.pi, angle_step=None, d_influence=0.80):
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


# ══════════════════════ Smart RTH Controller ══════════════════════════
def rth_thread(gz_node):
    global rth_active, rth_done, rth_waypoints, rth_wpt_idx, rth_steer_vector, running
    global auto_mode, mission_state, auto_status_msg

    twist_pub = gz_node.advertise(TWIST_TOPIC, Twist)
    msg       = Twist()

    while running:
        with rth_lock:
            active = rth_active

        if not active or home_x is None:
            time.sleep(0.03)
            continue

        with pose_lock:
            cx  = drone_x
            cy  = drone_y
            cz  = drone_z
            yaw = drone_yaw

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

        target_pt = wpts[idx]
        tx, ty = target_pt
        dx = tx - cx; dy = ty - cy
        dist_to_wpt  = math.hypot(dx, dy)
        dist_to_home = math.hypot(home_x - cx, home_y - cy)

        # Check if reached final Home destination -> Smoothly land & disarm
        if dist_to_home <= RTH_HOME_THRESH:
            print("\n  🏠 REACHED HOME HELIPAD: Initiating smooth autonomous landing descent...")

            # Smooth landing descent phase down to helipad
            while running and cz > 0.15:
                msg.linear.x  = 0.0
                msg.linear.y  = 0.0
                msg.linear.z  = -0.35  # gentle 35 cm/s descent
                msg.angular.z = 0.0
                twist_pub.publish(msg)
                time.sleep(0.03)
                with pose_lock:
                    cz = drone_z

            # Touchdown: completely stop and disarm
            for _ in range(5):
                msg.linear.x  = 0.0
                msg.linear.y  = 0.0
                msg.linear.z  = 0.0
                msg.angular.z = 0.0
                twist_pub.publish(msg)
                time.sleep(0.02)

            with rth_lock:
                rth_active       = False
                rth_done         = True
                rth_steer_vector = (0.0, 0.0)

            with auto_lock:
                if auto_mode:
                    mission_state   = MISSION_COMPLETE
                    auto_status_msg = "MISSION COMPLETE: FULL 15x15M GRID EXPLORED & LANDED"
                    auto_mode       = False

            print("  ✅ TOUCHDOWN COMPLETE: Drone has safely landed on Home Helipad!\n")
            save_final_map()
            continue

        if dist_to_wpt <= RTH_WPT_THRESH and idx < len(wpts) - 1:
            idx += 1
            with rth_lock:
                rth_wpt_idx = idx
            target_pt = wpts[idx]
            tx, ty = target_pt
            dx = tx - cx; dy = ty - cy
            dist_to_wpt = math.hypot(dx, dy)

        target_heading = math.atan2(dy, dx)
        yaw_err = (target_heading - yaw + math.pi) % (2 * math.pi) - math.pi

        if abs(yaw_err) > 0.60:
            vx_cmd = 0.08
            vy_cmd = 0.0
            wz_cmd = float(np.clip(1.5 * yaw_err, -1.1, 1.1))
        else:
            cruise_speed = min(RTH_MAX_VEL, max(0.35, 0.9 * dist_to_wpt))
            vx_cmd = cruise_speed * math.cos(yaw_err)
            wz_cmd = float(np.clip(1.2 * yaw_err, -0.6, 0.6))
            vy_cmd = 0.0

        with raw_lidar_lock:
            ranges = list(raw_lidar_ranges)
        vx_cmd, vy_cmd, wz_cmd, _ = apply_lidar_obstacle_avoidance(
            vx_cmd, vy_cmd, wz_cmd, yaw_err, ranges
        )

        alt_err = 1.50 - cz
        vz_cmd = float(np.clip(1.3 * alt_err, -0.5, 0.5))

        msg.linear.x  = float(np.clip(vx_cmd, -RTH_MAX_VEL, RTH_MAX_VEL))
        msg.linear.y  = float(np.clip(vy_cmd, -0.5, 0.5))
        msg.linear.z  = vz_cmd
        msg.angular.z = wz_cmd
        twist_pub.publish(msg)

        sv_world_x = math.cos(yaw) * vx_cmd - math.sin(yaw) * vy_cmd
        sv_world_y = math.sin(yaw) * vx_cmd + math.cos(yaw) * vy_cmd
        with rth_lock:
            rth_steer_vector = (sv_world_x, sv_world_y)

        time.sleep(0.03)


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
    for _ in range(2):
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


def auto_mission_thread(gz_node):
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
    global rth_active, rth_done, rth_waypoints, rth_wpt_idx, rth_steer_vector, running

    twist_pub = gz_node.advertise('/X3/gazebo/command/twist', Twist)
    msg = Twist()
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

        with pose_lock:
            cx  = drone_x
            cy  = drone_y
            cz  = drone_z
            yaw = drone_yaw

        if cx is None or cy is None:
            time.sleep(0.04)
            continue

        # ── PHASE 1: AUTONOMOUS TAKEOFF ──
        if state == MISSION_TAKEOFF:
            if takeoff_start_time is None:
                takeoff_start_time = time.time()
            elapsed_climb = time.time() - takeoff_start_time
            with auto_lock:
                auto_status_msg = f"Climbing to hover altitude ({cz:.2f}m / 1.50m)"

            if cz >= 1.25 or (elapsed_climb > 2.5 and cz >= 0.90):
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
                msg.linear.x  = 0.0
                msg.linear.y  = 0.0
                msg.linear.z  = float(np.clip(1.3 * (1.50 - cz), 0.30, 0.65))
                msg.angular.z = 0.0
                twist_pub.publish(msg)
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
                            msg.linear.x  = esc_vx
                            msg.linear.y  = esc_vy
                            msg.linear.z  = 0.0
                            msg.angular.z = 0.0
                            twist_pub.publish(msg)
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

                    alt_err = 1.50 - cz
                    msg.linear.x  = float(np.clip(vx_cmd, -0.85, 0.85))
                    msg.linear.y  = float(np.clip(vy_cmd, -0.5, 0.5))
                    msg.linear.z  = float(np.clip(1.3 * alt_err, -0.5, 0.5))
                    msg.angular.z = wz_cmd
                    twist_pub.publish(msg)

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
            alt_err = 1.50 - cz
            vz_cmd = float(np.clip(1.3 * alt_err, -0.5, 0.5))

            msg.linear.x  = float(np.clip(vx_cmd, -0.85, 0.85))
            msg.linear.y  = float(np.clip(vy_cmd, -0.5, 0.5))
            msg.linear.z  = vz_cmd
            msg.angular.z = wz_cmd
            twist_pub.publish(msg)

            sv_world_x = math.cos(yaw) * vx_cmd - math.sin(yaw) * vy_cmd
            sv_world_y = math.sin(yaw) * vx_cmd + math.cos(yaw) * vy_cmd
            with rth_lock:
                rth_steer_vector = (sv_world_x, sv_world_y)
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


# ═════════════════════════ Coverage Stats (15x15m Arena) ═══════════════
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

    unknown  = int(np.sum(g == UNKNOWN))
    occupied = int(np.sum(g == OCCUPIED))
    path_c   = int(np.sum(g == PATH))
    free_c   = int(np.sum(g == FREE))
    return arena_pct, free_c, path_c, occupied, unknown


# ═════════════════════════ Save Final Map ══════════════════════════════
def save_final_map():
    with grid_lock:
        g = grid.copy()
    with pose_lock:
        fx = drone_x or home_x or 0.0
        fy = drone_y or home_y or 0.0

    pct, free_c, path_c, occ_c, unk_c = coverage_stats()

    def make_rgb(g_arr):
        rgb = np.zeros((GRID_N, GRID_N, 3), dtype=np.uint8)
        rgb[g_arr == UNKNOWN]  = [160, 160, 160]
        rgb[g_arr == FREE]     = [235, 235, 235]
        rgb[g_arr == PATH]     = [80,  140, 255]
        rgb[g_arr == OCCUPIED] = [25,  25,  25]
        return rgb

    rgb = make_rgb(g)

    if home_x is not None:
        hc, hr = GRID_N // 2, GRID_N // 2
        for d in range(-5, 6):
            for nc, nr in [(hc + d, hr), (hc, hr + d)]:
                if 0 <= nc < GRID_N and 0 <= nr < GRID_N:
                    rgb[nr, nc] = [0, 220, 80]

    fp = world_to_cell(fx, fy)
    if fp:
        fc, fr = fp
        for dc in range(-4, 5):
            for dr in range(-4, 5):
                if dc**2 + dr**2 <= 16:
                    nc, nr = fc + dc, fr + dr
                    if 0 <= nc < GRID_N and 0 <= nr < GRID_N:
                        rgb[nr, nc] = [255, 50, 50]

    fig_r, ax_r = plt.subplots(figsize=(8, 8), dpi=150)
    ax_r.imshow(rgb, origin="lower"); ax_r.axis("off")
    fig_r.tight_layout(pad=0)
    fig_r.savefig(SAVE_PATH, bbox_inches="tight", pad_inches=0)
    plt.close(fig_r)

    half = HALF_SPAN
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    ax.imshow(rgb, origin="lower", extent=[-half, half, -half, half])
    for v in np.arange(-half, half + 1.0, 1.0):
        ax.axhline(v, color="gray", lw=0.3, alpha=0.4)
        ax.axvline(v, color="gray", lw=0.3, alpha=0.4)

    ax.set_xlabel("X offset from start (m)", fontsize=11)
    ax.set_ylabel("Y offset from start (m)", fontsize=11)
    ax.set_title(
        f"NIDAR Drone — Final 2D Occupancy Map\n"
        f"Coverage: {pct:.1f}%  |  Free: {free_c * CELL_SIZE_M**2:.1f} m²  |  "
        f"Walls: {occ_c} cells  |  Start: ({home_x:+.2f}, {home_y:+.2f}) m",
        fontsize=11, fontweight="bold"
    )

    legend_handles = [
        mpatches.Patch(color=[0.10, 0.10, 0.10], label="Wall / Obstacle"),
        mpatches.Patch(color=[0.92, 0.92, 0.92], label="Free Space"),
        mpatches.Patch(color=[0.31, 0.55, 1.00], label="Drone Path"),
        mpatches.Patch(color=[0.63, 0.63, 0.63], label="Unexplored"),
        mpatches.Patch(color=[0.00, 0.86, 0.31], label="Home (start)"),
        mpatches.Patch(color=[1.00, 0.20, 0.20], label="Final Position"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)
    ax.plot(0, 0, 'g^', markersize=14, zorder=6)

    fig.tight_layout()
    fig.savefig(SAVE_ANNOT, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✅ Maps saved:\n   {SAVE_PATH}\n   {SAVE_ANNOT}")


# ══════════════════════════════ Main ═══════════════════════════════════
def main():
    global running, rth_active, rth_waypoints, rth_wpt_idx, auto_mode, mission_state

    print("=" * 65)
    print("  🗺  NIDAR Drone — 2D Occupancy Mapper  (v3: Smart Collision-Free RTH)")
    print("=" * 65)
    print("  Controls (map window focused):")
    print("    E  → Toggle AUTONOMOUS 15x15m Exploration & Mapping")
    print("    H  → Activate SMART Return-To-Home (Corridor Route)")
    print("    Q  → Quit mapper + save final map")
    print("=" * 65)

    gz_node = Node()
    gz_node.subscribe(LaserScan, LIDAR_TOPIC, lidar_callback)

    th_pose = threading.Thread(target=pose_reader_thread, daemon=True)
    th_rth  = threading.Thread(target=rth_thread, args=(gz_node,), daemon=True)
    th_auto = threading.Thread(target=auto_mission_thread, args=(gz_node,), daemon=True)
    th_pose.start()
    th_rth.start()
    th_auto.start()

    home_set.wait(timeout=15.0)
    if not home_set.is_set():
        print("⚠  No pose from Gazebo.")
        running = False
        return

    if "--auto" in sys.argv:
        with auto_lock:
            auto_mode     = True
            mission_state = MISSION_TAKEOFF
        print("  🚀 AUTO FLAG DETECTED: Autonomous 15x15m Exploration will begin immediately!")

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.canvas.manager.set_window_title("NIDAR 2D Map  [E=Auto-Explore  H=Smart RTH  Q=Quit]")
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    half = HALF_SPAN
    display_rgb = np.full((GRID_N, GRID_N, 3), 160, dtype=np.uint8)
    im = ax.imshow(
        display_rgb, origin="lower",
        extent=[-half, half, -half, half], interpolation="nearest",
    )

    for v in np.arange(-half, half + 1, 1.0):
        ax.axhline(v, color="white", lw=0.3, alpha=0.2)
        ax.axvline(v, color="white", lw=0.3, alpha=0.2)

    ax.plot([0], [0], 'g^', markersize=12, zorder=6)
    ax.annotate("HOME", (0, 0), textcoords="offset points", xytext=(6, 5),
                color="lime", fontsize=8, fontweight="bold")

    rth_path_line, = ax.plot([], [], color='#00f0ff', lw=2.4, linestyle='--',
                             marker='o', markersize=4, zorder=8, label='RTH Path')

    steer_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                              arrowprops=dict(arrowstyle="-|>", color="#39ff14", lw=2.5, mutation_scale=14),
                              zorder=9)

    drone_marker, = ax.plot([], [], 'r^', markersize=11, zorder=7)
    arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="red", lw=2))

    ax.set_xlabel("X offset from start (m)", color="white", fontsize=9)
    ax.set_ylabel("Y offset from start (m)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor("white")

    legend_handles = [
        mpatches.Patch(color=[0.10, 0.10, 0.10], label="Wall"),
        mpatches.Patch(color=[0.92, 0.92, 0.92], label="Free"),
        mpatches.Patch(color=[0.31, 0.55, 1.00], label="Path"),
        mpatches.Patch(color=[0.63, 0.63, 0.63], label="Unknown"),
        mpatches.Patch(color=[0.00, 0.94, 1.00], label="RTH Vector Path"),
        mpatches.Patch(color=[0.22, 1.00, 0.08], label="Steer Vector"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              facecolor="#111827", edgecolor="white", labelcolor="white")

    title = ax.set_title("NIDAR Live Map", color="white", fontsize=10, fontweight="bold")

    def on_key(event):
        global rth_active, rth_waypoints, rth_wpt_idx, running, auto_mode, mission_state
        if event.key in ('e', 'E'):
            with auto_lock:
                if not auto_mode:
                    auto_mode     = True
                    mission_state = MISSION_TAKEOFF
                    print("\n  🚀 AUTONOMOUS 15x15M EXPLORATION ACTIVATED: Auto-takeoff initiated...")
                else:
                    auto_mode     = False
                    mission_state = MISSION_IDLE
                    print("\n  ⏸ AUTONOMOUS EXPLORATION PAUSED.")
        elif event.key in ('h', 'H'):
            with pose_lock:
                cx, cy = drone_x, drone_y
            if cx is not None and home_x is not None:
                planned = plan_rth_path((cx, cy), (home_x, home_y))
                with rth_lock:
                    rth_waypoints = planned
                    rth_wpt_idx   = 0
                    rth_active    = True
                print(f"\n  🏠 SMART RTH ACTIVATED: Planned {len(planned)} collision-free corridor waypoints!")
        elif event.key in ('q', 'Q'):
            running = False

    fig.canvas.mpl_connect("key_press_event", on_key)

    start_t = time.time()
    last_refresh = 0.0

    try:
        while running and plt.fignum_exists(fig.number):
            now = time.time()

            if now - last_refresh >= MAP_REFRESH_S:
                last_refresh = now

                with grid_lock:
                    g = grid.copy()
                with pose_lock:
                    cx  = drone_x
                    cy  = drone_y
                    cyaw = drone_yaw

                rgb = np.zeros((GRID_N, GRID_N, 3), dtype=np.uint8)
                rgb[g == UNKNOWN]  = [160, 160, 160]
                rgb[g == FREE]     = [235, 235, 235]
                rgb[g == PATH]     = [80,  140, 255]
                rgb[g == OCCUPIED] = [25,  25,  25]

                home_col = GRID_N // 2; home_row = GRID_N // 2
                for dc in range(-4, 5):
                    for nc, nr in [(home_col + dc, home_row), (home_col, home_row + dc)]:
                        if 0 <= nc < GRID_N and 0 <= nr < GRID_N:
                            rgb[nr, nc] = [0, 220, 80]

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

                if cx is not None and home_x is not None:
                    rx = cx - home_x; ry = cy - home_y
                    drone_marker.set_data([rx], [ry])
                    aend_x = rx + 0.5 * math.cos(cyaw)
                    aend_y = ry + 0.5 * math.sin(cyaw)
                    arrow.xy     = (aend_x, aend_y)
                    arrow.xytext = (rx, ry)

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
                            steer_arrow.xy = (rx + svx * arr_scale, ry + svy * arr_scale)
                            steer_arrow.xytext = (rx, ry)
                            steer_arrow.set_visible(True)
                        else:
                            steer_arrow.set_visible(False)
                    else:
                        rth_path_line.set_data([], [])
                        steer_arrow.set_visible(False)

                pct, free_c, path_c, occ_c, unk_c = coverage_stats()
                elapsed  = int(now - start_t)
                mm, ss   = divmod(elapsed, 60)
                if is_rth:
                    mode = "🏠 SMART RTH ACTIVE"
                elif is_auto:
                    mode = f"🚀 AUTO-EXPLORE: {m_state}"
                elif rth_done:
                    mode = "✅ HOME LANDED"
                else:
                    mode = "✈ FLYING"

                dist_home = math.hypot(cx - home_x, cy - home_y) if (cx is not None and home_x is not None) else 0.0
                title.set_text(
                    f"NIDAR 2D Map  [{mode}]  |  15x15m Coverage: {pct:.1f}%  |  ⏱ {mm:02d}:{ss:02d}\n"
                    f"Drone ({cx - home_x if cx else 0:+.1f}m, {cy - home_y if cy else 0:+.1f}m)  |  "
                    f"Dist to Home: {dist_home:.1f}m  |  [E]=Auto-Explore  [H]=Smart RTH  [Q]=Quit"
                )

                fig.canvas.draw_idle()

            plt.pause(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        plt.ioff()
        plt.close("all")
        save_final_map()


if __name__ == "__main__":
    main()

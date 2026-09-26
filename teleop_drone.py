#!/usr/bin/env python3
"""
NIDAR AI Drone - Unified Teleoperation & Live Sensor Telemetry HUD
Features:
  - 100% Stable: MulticopterVelocityControl (1000Hz attitude stabilization)
  - Automatic Hover Lock & Strict 3.0m Ceiling
  - Concurrent Live Sensor Telemetry:
      * LiDAR 360° LaserScan
      * Waveshare IMX219-83 Stereo Depth Camera
      * Waveshare IMX335 5MP WDR Camera
  - Real-time Forward Obstacle Detection (4.0m - 6.0m alert radius with precise distance)
"""

import os
import sys
import tty
import termios
import select
import threading
import time
import subprocess
import math
import re

from gz.transport import Node
from gz.msgs.twist_pb2 import Twist
from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.laserscan_pb2 import LaserScan
from gz.msgs.image_pb2 import Image

# ----------------- Parameters -----------------
MAX_CEILING = 2.30       # Hard ceiling: 2.30m (strictly under 8ft / 2.44m net ceiling)
DEFAULT_ALTITUDE = 1.50  # Default safe indoor cruising height
MIN_ALTITUDE = 0.05

# Obstacle Detection Config (0.3m - 5.0m max radius)
OBS_ALERT_MAX_DIST = 5.0  # Maximum detection radius reduced to 5.0m for precision
OBS_ALERT_MIN_DIST = 0.3  # Minimum safety radius (meters)
CRITICAL_ALERT_DIST = 1.5 # Critical alert threshold (1.5m)
WARNING_ALERT_DIST = 3.0  # Warning alert threshold (3.0m)
FORWARD_FOV_DEG = 70.0    # Forward sensing cone (+/- 35 deg from heading)

# ----------------- Drone State -----------------
target_altitude = DEFAULT_ALTITUDE
current_altitude = 0.05
vertical_velocity = 0.0

cmd_vx = 0.0
cmd_vy = 0.0
cmd_wz = 0.0
last_key_time = 0.0

motors_active = True
running = True
world_name = "nidar_world"

# ----------------- Live Sensor State -----------------
lidar_connected = False
lidar_rays_count = 0
lidar_last_time = 0.0

# Obstacle Tracking
obstacle_detected = False
closest_obstacle_dist = float('inf')
closest_obstacle_angle_deg = 0.0

# Sector Min Distances (Left, Front-Left, Front, Front-Right, Right)
sector_ranges = {
    "left": float('inf'),
    "front_left": float('inf'),
    "front": float('inf'),
    "front_right": float('inf'),
    "right": float('inf'),
}

stereo_connected = False
stereo_fps = 0.0
stereo_res = "640x480"
stereo_frame_count = 0
stereo_last_t = time.time()

camera_5mp_connected = False
camera_5mp_fps = 0.0
camera_5mp_res = "1920x1080"
camera_5mp_frame_count = 0
camera_5mp_last_t = time.time()

# ----------------- 1. Gazebo Transport Node Setup -----------------
node = Node()
twist_pub = node.advertise('/X3/gazebo/command/twist', Twist)
enable_pub = node.advertise('/X3/enable', Boolean)

twist_msg = Twist()
enable_msg = Boolean()
enable_msg.data = True

def detect_world_name():
    global world_name
    try:
        res = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True, timeout=2)
        if "/world/nidar_world/pose/info" in res.stdout:
            world_name = "nidar_world"
        elif "/world/quadcopter/pose/info" in res.stdout:
            world_name = "quadcopter"
    except Exception:
        pass

detect_world_name()

# ----------------- 2. Sensor Subscriber Callbacks -----------------
def lidar_callback(msg: LaserScan):
    global lidar_connected, lidar_rays_count, lidar_last_time
    global obstacle_detected, closest_obstacle_dist, closest_obstacle_angle_deg
    global sector_ranges

    lidar_connected = True
    lidar_last_time = time.time()
    lidar_rays_count = len(msg.ranges)
    if lidar_rays_count == 0:
        return

    angle_min = msg.angle_min
    angle_step = msg.angle_step

    min_front_dist = float('inf')
    min_front_angle = 0.0

    s_left = float('inf')
    s_fleft = float('inf')
    s_front = float('inf')
    s_fright = float('inf')
    s_right = float('inf')

    half_cone = math.radians(FORWARD_FOV_DEG / 2.0)

    for i, r in enumerate(msg.ranges):
        # Filter invalid or out-of-bound readings
        if math.isinf(r) or math.isnan(r) or r < msg.range_min or r > msg.range_max:
            continue

        theta = angle_min + i * angle_step
        # Normalize theta to [-pi, pi]
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
        theta_deg = math.degrees(theta)

        # Sector categorizations
        if -20.0 <= theta_deg <= 20.0:
            if r < s_front: s_front = r
        elif 20.0 < theta_deg <= 60.0:
            if r < s_fleft: s_fleft = r
        elif 60.0 < theta_deg <= 110.0:
            if r < s_left: s_left = r
        elif -60.0 <= theta_deg < -20.0:
            if r < s_fright: s_fright = r
        elif -110.0 <= theta_deg < -60.0:
            if r < s_right: s_right = r

        # Forward Cone Obstacle Detection (within 4-6m alert radius)
        if -half_cone <= theta <= half_cone:
            if r < min_front_dist:
                min_front_dist = r
                min_front_angle = theta_deg

    sector_ranges["left"] = s_left
    sector_ranges["front_left"] = s_fleft
    sector_ranges["front"] = s_front
    sector_ranges["front_right"] = s_fright
    sector_ranges["right"] = s_right

    closest_obstacle_dist = min_front_dist
    closest_obstacle_angle_deg = min_front_angle
    obstacle_detected = (min_front_dist <= OBS_ALERT_MAX_DIST)

def stereo_callback(msg: Image):
    global stereo_connected, stereo_fps, stereo_res, stereo_frame_count, stereo_last_t
    stereo_connected = True
    stereo_frame_count += 1
    stereo_res = f"{msg.width}x{msg.height}"
    now = time.time()
    dt = now - stereo_last_t
    if dt >= 1.0:
        stereo_fps = stereo_frame_count / dt
        stereo_frame_count = 0
        stereo_last_t = now

def camera_5mp_callback(msg: Image):
    global camera_5mp_connected, camera_5mp_fps, camera_5mp_res, camera_5mp_frame_count, camera_5mp_last_t
    camera_5mp_connected = True
    camera_5mp_frame_count += 1
    camera_5mp_res = f"{msg.width}x{msg.height}"
    now = time.time()
    dt = now - camera_5mp_last_t
    if dt >= 1.0:
        camera_5mp_fps = camera_5mp_frame_count / dt
        camera_5mp_frame_count = 0
        camera_5mp_last_t = now

# Subscribe with gz transport Node
node.subscribe(LaserScan, '/nidar/lidar', lidar_callback)
node.subscribe(Image, '/nidar/stereo_camera/image', stereo_callback)
node.subscribe(Image, '/nidar/camera', camera_5mp_callback)

# ----------------- 3. Background Pose Monitor (50 Hz) -----------------
def pose_reader_thread():
    global current_altitude, vertical_velocity, running, world_name
    
    pose_topic = f"/world/{world_name}/pose/info"
    proc = subprocess.Popen(
        ["gz", "topic", "-t", pose_topic, "-e"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    last_z = 0.05
    last_t = time.time()
    curr_entity = None
    in_pos = False

    while running:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        m = re.match(r'name:\s*"([^"]+)"', line)
        if m:
            curr_entity = m.group(1)

        if curr_entity == "x3":
            if line.startswith("position"):
                in_pos = True
            elif in_pos and line.startswith("}"):
                in_pos = False
            elif in_pos and line.startswith("z:"):
                try:
                    z = float(line.split(":")[1].strip())
                    now = time.time()
                    dt = now - last_t
                    if dt > 0.015:
                        vz = (z - last_z) / dt
                        current_altitude = z
                        vertical_velocity = 0.7 * vertical_velocity + 0.3 * vz
                        last_z = z
                        last_t = now
                except ValueError:
                    pass
    proc.terminate()

# ----------------- 4. High-Rate Flight Control Loop (50 Hz) -----------------
def flight_control_loop():
    global target_altitude, current_altitude, vertical_velocity
    global cmd_vx, cmd_vy, cmd_wz, last_key_time, motors_active, running

    enable_pub.publish(enable_msg)

    while running:
        if not motors_active or target_altitude < 0.1:
            twist_msg.linear.x = 0.0
            twist_msg.linear.y = 0.0
            twist_msg.linear.z = -0.5 if current_altitude > 0.15 else 0.0
            twist_msg.angular.z = 0.0
            twist_pub.publish(twist_msg)
            time.sleep(0.02)
            continue

        # Auto-decay horizontal velocity when keys are not being held
        if time.time() - last_key_time > 0.35:
            cmd_vx = 0.0
            cmd_vy = 0.0
            cmd_wz = 0.0

        # --- ALTITUDE HOLD & STRICT 3.0m CEILING ---
        alt_err = target_altitude - current_altitude
        cmd_vz = max(-1.0, min(1.0, 1.3 * alt_err))

        if current_altitude >= MAX_CEILING:
            cmd_vz = -0.5   # Force downward if above 3.0m
        elif current_altitude >= (MAX_CEILING - 0.2) and cmd_vz > 0:
            cmd_vz = min(0.15, cmd_vz)

        if abs(alt_err) < 0.04 and current_altitude < MAX_CEILING:
            cmd_vz = 0.0

        twist_msg.linear.x = cmd_vx
        twist_msg.linear.y = cmd_vy
        twist_msg.linear.z = cmd_vz
        twist_msg.angular.z = cmd_wz

        twist_pub.publish(twist_msg)
        time.sleep(0.02)

# ----------------- 5. Reset Simulation -----------------
def reset_simulation():
    global target_altitude, cmd_vx, cmd_vy, cmd_wz, motors_active, world_name
    motors_active = True
    target_altitude = DEFAULT_ALTITUDE
    cmd_vx = 0.0
    cmd_vy = 0.0
    cmd_wz = 0.0
    subprocess.run([
        "gz", "service", "-s", f"/world/{world_name}/control",
        "--reqtype", "gz.msgs.WorldControl",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "2000",
        "--req", "reset: { all: true }, pause: false"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ----------------- 6. Integrated Flight & Live Sensor HUD -----------------
def format_dist(d):
    return f"{d:4.2f}m" if not math.isinf(d) else " CLEAR"

def print_hud():
    sys.stdout.write("\033[H\033[J")
    print("=" * 72)
    print("      🚁 NIDAR AI DRONE - INTEGRATED FLIGHT & SENSOR HUD")
    print("=" * 72)

    # Flight Status Section
    bar_len = 22
    filled = int(min(1.0, max(0.0, current_altitude / MAX_CEILING)) * bar_len)
    alt_bar = "█" * filled + "░" * (bar_len - filled)

    if not motors_active:
        flight_status = "🔴 MOTORS DISARMED"
    elif abs(current_altitude - target_altitude) < 0.12 and abs(cmd_vx) < 0.1 and abs(cmd_vy) < 0.1:
        flight_status = "🟢 HOVER LOCKED AT SET ALTITUDE"
    elif cmd_vx != 0 or cmd_vy != 0 or cmd_wz != 0:
        flight_status = "🔵 ACTIVE STEERING (1000Hz STABILIZATION)"
    elif current_altitude < target_altitude:
        flight_status = "🟡 CLIMBING TO TARGET"
    else:
        flight_status = "🟡 DESCENDING TO TARGET"

    print(f"  Flight Mode   : {flight_status}")
    print(f"  Live Altitude : [{alt_bar}] {current_altitude:4.2f}m / {MAX_CEILING:.1f}m MAX CEILING")
    print(f"  Target Set    : {target_altitude:4.2f}m  | Climb Speed: {vertical_velocity:+4.2f} m/s")
    print(f"  Fwd/Back (Vx) : {cmd_vx:+4.2f} m/s | Strafe (Vy) : {cmd_vy:+4.2f} m/s | Turn (Wz): {cmd_wz:+4.2f} rad/s")
    print("-" * 72)

    # Live Sensor Telemetry Section
    lidar_st = f"🟢 ONLINE ({lidar_rays_count} rays)" if lidar_connected else "⚪ CONNECTING..."
    stereo_st = f"🟢 {stereo_res} @ {stereo_fps:3.1f}fps" if stereo_connected else "⚪ CONNECTING..."
    cam5_st = f"🟢 {camera_5mp_res} @ {camera_5mp_fps:3.1f}fps" if camera_5mp_connected else "⚪ CONNECTING..."

    print("  LIVE SENSOR TELEMETRY:")
    print(f"    • LiDAR (360° LaserScan)     : {lidar_st}")
    print(f"    • IMX219-83 Stereo Depth Cam : {stereo_st}")
    print(f"    • IMX335 5MP WDR Camera (2K) : {cam5_st}")
    print("-" * 72)

    # Proximity & Forward Obstacle Warning Radar (5.0m detection radius)
    print("  FORWARD RADAR & OBSTACLE DETECTION (5.0m MAX RADIUS):")
    if obstacle_detected:
        dist = closest_obstacle_dist
        angle = closest_obstacle_angle_deg
        # Determine warning urgency level based on updated thresholds
        if dist <= CRITICAL_ALERT_DIST:
            warn_icon = "🚨 CRITICAL PROXIMITY ALERT! (< 1.5m)"
        elif dist <= WARNING_ALERT_DIST:
            warn_icon = "⚠️  WARNING: OBSTACLE NEARBY! (<= 3.0m)"
        else:
            warn_icon = "📢 CAUTION: OBSTACLE IN DETECTION ZONE (3.0m - 5.0m)"

        # Visual proximity distance gauge
        gauge_len = 20
        dist_fill = int(min(1.0, max(0.0, dist / OBS_ALERT_MAX_DIST)) * gauge_len)
        gauge = "▓" * dist_fill + "░" * (gauge_len - dist_fill)

        bearing_str = f"{abs(angle):.1f}° {'Left' if angle > 0 else 'Right'}" if abs(angle) > 2.0 else "Dead Center"
        print(f"    {warn_icon}")
        print(f"    Target Distance: [{gauge}] {dist:.2f} METERS ({bearing_str})")
    else:
        print("    🟢 FORWARD PATH CLEAR: No obstacles detected within 5.0m radius")

    # Directional sector breakdown
    f_l = format_dist(sector_ranges["left"])
    f_fl = format_dist(sector_ranges["front_left"])
    f_f = format_dist(sector_ranges["front"])
    f_fr = format_dist(sector_ranges["front_right"])
    f_r = format_dist(sector_ranges["right"])
    print(f"    Surround Ranges: [Left: {f_l}] [F-Left: {f_fl}] [Front: {f_f}] [F-Right: {f_fr}] [Right: {f_r}]")

    print("-" * 72)
    print("  Controls:")
    print("    W / S        : Target Altitude Up / Down (0.5m to 2.3m MAX)")
    print("    ↑ / ↓ (Up/Dn): Fly Forward / Backward")
    print("    ← / → (Lt/Rt): Strafe Left / Right")
    print("    A / D        : Turn Body Left / Right (Yaw)")
    print("    SPACE        : Auto-Level & Safe Indoor Hover at 1.5m")
    print("    R            : Reset Drone to Entry/Exit Helipad")
    print("    X            : Land / Stop")
    print("    Q            : Exit Controller")
    print("=" * 72)
    if current_altitude >= MAX_CEILING - 0.05:
        print("  ⚠️  CEILING LOCK: Maximum 2.3m height limit strictly active (under 8ft net)!")
    sys.stdout.flush()

# ----------------- 7. Non-blocking Keyboard Input -----------------
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.08)
        if rlist:
            ch1 = sys.stdin.read(1)
            if ch1 == '\x1b':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A': return 'UP'
                    elif ch3 == 'B': return 'DOWN'
                    elif ch3 == 'C': return 'RIGHT'
                    elif ch3 == 'D': return 'LEFT'
            return ch1
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main():
    global target_altitude, cmd_vx, cmd_vy, cmd_wz, last_key_time
    global motors_active, running

    detect_world_name()

    th_pose = threading.Thread(target=pose_reader_thread, daemon=True)
    th_ctrl = threading.Thread(target=flight_control_loop, daemon=True)
    th_pose.start()
    th_ctrl.start()

    time.sleep(0.2)
    print_hud()

    last_hud_refresh = time.time()
    try:
        while running:
            key = get_key()

            if key in ('q', 'Q', '\x03'):
                running = False
                break
            elif key in ('w', 'W'):
                motors_active = True
                target_altitude = min(MAX_CEILING, round(target_altitude + 0.25, 2))
                last_key_time = time.time()
            elif key in ('s', 'S'):
                target_altitude = max(MIN_ALTITUDE, round(target_altitude - 0.25, 2))
                last_key_time = time.time()
            elif key == 'UP':
                motors_active = True
                cmd_vx = 1.0     # 1.0 m/s forward
                last_key_time = time.time()
            elif key == 'DOWN':
                motors_active = True
                cmd_vx = -1.0    # 1.0 m/s backward
                last_key_time = time.time()
            elif key == 'LEFT':
                motors_active = True
                cmd_vy = 0.8     # Strafe left
                last_key_time = time.time()
            elif key == 'RIGHT':
                motors_active = True
                cmd_vy = -0.8    # Strafe right
                last_key_time = time.time()
            elif key in ('a', 'A'):
                motors_active = True
                cmd_wz = 0.7     # Yaw left
                last_key_time = time.time()
            elif key in ('d', 'D'):
                motors_active = True
                cmd_wz = -0.7    # Yaw right
                last_key_time = time.time()
            elif key == ' ':
                motors_active = True
                target_altitude = DEFAULT_ALTITUDE
                cmd_vx = 0.0
                cmd_vy = 0.0
                cmd_wz = 0.0
                last_key_time = time.time()
            elif key in ('x', 'X'):
                motors_active = False
                cmd_vx = 0.0
                cmd_vy = 0.0
                cmd_wz = 0.0
                target_altitude = 0.0
            elif key in ('r', 'R'):
                reset_simulation()

            if time.time() - last_hud_refresh > 0.12:
                print_hud()
                last_hud_refresh = time.time()

    finally:
        running = False
        twist_msg.linear.x = 0.0
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.z = 0.0
        twist_pub.publish(twist_msg)
        sys.stdout.write("\033[H\033[J")
        print("Teleop closed. Drone stopped.")
        os._exit(0)

if __name__ == '__main__':
    main()

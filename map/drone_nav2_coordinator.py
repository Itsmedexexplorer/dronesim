#!/usr/bin/env python3
"""
Drone Nav2 Flight Coordinator
=============================
Bridges Nav2 Navigation Stack with MAVROS 3D Drone Autopilot:
  • Executes Pre-flight Heartbeat, Motor Arming, and OFFBOARD mode switch.
  • Performs vertical climb to 1.5m Target Altitude.
  • Subscribes to Nav2's 2D /cmd_vel (Twist) output.
  • Injects real-time closed-loop Vertical PID altitude hold (vz).
  • Publishes 3D PositionTarget setpoints to /mavros/setpoint_raw/local at 30Hz.
  • Handles Mission State: TAKEOFF -> NAV2_ACTIVE -> RTH -> LANDING.
"""

import os
import sys
import time
import math
import signal
import threading
import numpy as np

def _clean_exit_handler(sig, frame):
    os._exit(0)

signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode

TARGET_ALTITUDE = 1.50  # meters AGL
HELIPAD_X = -5.80
HELIPAD_Y = -8.80


class DroneNav2Coordinator(Node):
    def __init__(self):
        super().__init__('drone_nav2_coordinator')
        self.get_logger().info('Initializing Drone Nav2 Flight Coordinator...')

        # State flags
        self.state_connected = False
        self.state_armed = False
        self.state_mode = "MANUAL"
        self.mission_state = "TAKEOFF"  # TAKEOFF, NAV2_ACTIVE, LANDING, DISARMED
        self.running = True
        self.has_left_helipad = False
        self.return_triggered = False

        # Drone pose
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_yaw = 0.0
        self.pose_lock = threading.Lock()

        # Nav2 commanded velocities
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.last_cmd_time = time.time()
        self.cmd_lock = threading.Lock()

        # ROS 2 Subscriptions
        self.sub_state = self.create_subscription(State, '/mavros/state', self.cb_state, 10)
        self.sub_pose  = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.cb_pose, 10)

        # In ROS 2 Jazzy, Nav2 controller_server publishes geometry_msgs/msg/Twist
        self.sub_cmd_vel1 = self.create_subscription(Twist, '/cmd_vel_nav', self.cb_cmd_vel, 10)
        self.sub_cmd_vel2 = self.create_subscription(Twist, '/cmd_vel', self.cb_cmd_vel, 10)
        self.sub_cmd_vel3 = self.create_subscription(Twist, '/cmd_vel_smoothed', self.cb_cmd_vel, 10)

        # Autonomous Landing Trigger
        self.sub_land = self.create_subscription(Bool, '/nidar/land_command', self.cb_land, 10)
        self.sub_relaunch = self.create_subscription(Bool, '/nidar/relaunch_command', self.cb_relaunch, 10)

        # Reactive LiDAR Collision Avoidance Shield
        self.repulsive_vx = 0.0
        self.repulsive_vy = 0.0
        self.min_obstacle_dist = 99.0
        self.shield_lock = threading.Lock()
        self._last_shield_log = 0.0
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.cb_scan, 10)

        self._last_log_t = 0.0

        # ROS 2 Publishers
        self.pub_setpoint = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.pub_complete = self.create_publisher(Bool, '/nidar/mission_complete', 10)

        # MAVROS Service Clients
        self.cli_arm = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.cli_mode = self.create_client(SetMode, '/mavros/set_mode')

        # Background mission loop
        self.thread = threading.Thread(target=self.flight_loop, daemon=True)
        self.thread.start()

    def cb_state(self, msg: State):
        self.state_connected = msg.connected
        self.state_armed = msg.armed
        self.state_mode = msg.mode

    def cb_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self.pose_lock:
            self.drone_x = msg.pose.position.x
            self.drone_y = msg.pose.position.y
            self.drone_z = msg.pose.position.z
            self.drone_yaw = yaw

    def cb_cmd_vel(self, msg: Twist):
        vx = float(msg.linear.x)
        vy = float(msg.linear.y)
        wz = float(msg.angular.z)
        with self.cmd_lock:
            self.cmd_vx = vx
            self.cmd_vy = vy
            self.cmd_wz = wz
            self.last_cmd_time = time.time()
        now = time.time()
        if (now - self._last_log_t) > 1.0:
            self.get_logger().info(f'📡 Nav2 cmd_vel [Twist]: vx={vx:+.2f}, vy={vy:+.2f}, wz={wz:+.2f}')
            self._last_log_t = now

    def cb_land(self, msg: Bool):
        if msg.data:
            self.return_triggered = True
            with self.pose_lock:
                dist_pad = math.hypot(self.drone_x - HELIPAD_X, self.drone_y - HELIPAD_Y)
                drone_y = self.drone_y
                drone_x = self.drone_x
            if self.has_left_helipad and (dist_pad < 0.60 or (drone_y < -8.2 and abs(drone_x - HELIPAD_X) < 0.60)):
                self.get_logger().info(f'🛬 Over Helipad (dist={dist_pad:.2f}m, y={drone_y:.2f}m)! Engaging vertical touchdown...')
                self.mission_state = "LANDING"
            else:
                self.get_logger().info(f'🛸 Return-to-Helipad command received ({dist_pad:.2f}m from pad). Keeping Nav2 active for corridor return...')

    def cb_relaunch(self, msg: Bool):
        if msg.data:
            self.get_logger().info('🚀 [NAV2 COORDINATOR] Relaunch signal received! Powering up for new takeoff...')
            self.has_left_helipad = False
            self.return_triggered = False
            self.mission_state = "TAKEOFF"
            self.pub_complete.publish(Bool(data=False))
            self.set_flight_mode("OFFBOARD")
            self.arm_vehicle(True)

    def cb_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= 6.0)
        if not np.any(valid):
            with self.shield_lock:
                self.repulsive_vx = 0.0
                self.repulsive_vy = 0.0
                self.min_obstacle_dist = 99.0
        with self.pose_lock:
            cx, cy, cyaw = self.drone_x, self.drone_y, self.drone_yaw

        r = ranges[valid]
        a = angles[valid]

        # Ignore any points in the 2.0m open South Gate doorway and Helipad chute
        wx = cx + r * np.cos(cyaw + a)
        wy = cy + r * np.sin(cyaw + a)
        in_chute = (wx >= -6.75) & (wx <= -4.85) & (wy >= -9.80) & (wy <= -7.20)
        not_gate = ~in_chute

        r = r[not_gate]
        a = a[not_gate]

        if len(r) == 0:
            with self.shield_lock:
                self.repulsive_vx = 0.0
                self.repulsive_vy = 0.0
                self.min_obstacle_dist = 99.0
            return

        # In base_link / body frame: x forward, y left
        ox = r * np.cos(a)
        oy = r * np.sin(a)

        COLLISION_THRESH = 0.45
        close_mask = r < COLLISION_THRESH
        min_dist = float(np.min(r))

        rep_vx = 0.0
        rep_vy = 0.0

        if np.any(close_mask):
            close_r = r[close_mask]
            close_ox = ox[close_mask]
            close_oy = oy[close_mask]

            # Repulsive vector points AWAY from obstacle (-ox, -oy)
            weights = (1.0 / np.clip(close_r, 0.15, COLLISION_THRESH)) - (1.0 / COLLISION_THRESH)
            push_x = -np.sum((close_ox / close_r) * weights)
            push_y = -np.sum((close_oy / close_r) * weights)

            rep_vx = float(np.clip(push_x * 0.16, -0.40, 0.40))
            rep_vy = float(np.clip(push_y * 0.16, -0.40, 0.40))

        with self.shield_lock:
            self.repulsive_vx = rep_vx
            self.repulsive_vy = rep_vy
            self.min_obstacle_dist = min_dist


    def send_target_vel(self, vx, vy, vz, wz):
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
        msg.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
                         PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                         PositionTarget.IGNORE_YAW)
        msg.velocity.x = float(vx)
        msg.velocity.y = float(vy)
        msg.velocity.z = float(vz)
        msg.yaw_rate   = float(wz)
        self.pub_setpoint.publish(msg)

    def arm_vehicle(self, arm=True):
        for _ in range(5):
            if not self.cli_arm.wait_for_service(timeout_sec=1.0):
                continue
            req = CommandBool.Request()
            req.value = arm
            future = self.cli_arm.call_async(req)
            start_t = time.time()
            while not future.done() and (time.time() - start_t) < 2.0:
                time.sleep(0.02)
            res = future.result()
            if res is not None and res.success:
                return True
            time.sleep(0.2)
        return False

    def set_flight_mode(self, mode_str="OFFBOARD"):
        for _ in range(5):
            if not self.cli_mode.wait_for_service(timeout_sec=1.0):
                continue
            req = SetMode.Request()
            req.custom_mode = mode_str
            future = self.cli_mode.call_async(req)
            start_t = time.time()
            while not future.done() and (time.time() - start_t) < 2.0:
                time.sleep(0.02)
            res = future.result()
            if res is not None and res.mode_sent:
                return True
            time.sleep(0.2)
        return False

    def flight_loop(self):
        self.get_logger().info('Flight Loop: Waiting for MAVROS FCU connection...')
        while self.running and not self.state_connected:
            time.sleep(0.1)

        self.get_logger().info('Flight Loop: Waiting for initial drone telemetry...')
        while self.running:
            with self.pose_lock:
                if self.drone_z != 0.0 or self.drone_x != 0.0 or self.drone_y != 0.0:
                    break
            time.sleep(0.1)

        # 1. Warmup setpoints before requesting OFFBOARD
        self.get_logger().info('Sending 25 warmup heartbeats to MAVROS...')
        for _ in range(25):
            self.send_target_vel(0.0, 0.0, 0.0, 0.0)
            time.sleep(0.04)

        # 2. Arm and switch to OFFBOARD mode
        self.get_logger().info('Arming vehicle & engaging OFFBOARD flight mode...')
        self.set_flight_mode("OFFBOARD")
        self.arm_vehicle(True)

        last_climb_log = time.time()
        last_nav_log = time.time()

        while self.running:
            with self.pose_lock:
                cz = self.drone_z

            # ── STATE 1: TAKEOFF CLIMB ──
            if self.mission_state == "TAKEOFF":
                alt_err = TARGET_ALTITUDE - cz
                if cz < 1.40:
                    vz = float(np.clip(1.8 * alt_err, 0.60, 1.20))
                    self.send_target_vel(0.0, 0.0, vz, 0.0)
                    now_t = time.time()
                    if now_t - last_climb_log > 0.8:
                        self.get_logger().info(f'🛫 Takeoff climb: alt={cz:.2f}m / {TARGET_ALTITUDE:.2f}m (vz={vz:.2f} m/s)')
                        last_climb_log = now_t
                else:
                    self.get_logger().info(f'🛫 Takeoff complete at {cz:.2f}m! Enabling Nav2 Velocity Tracking...')
                    self.mission_state = "NAV2_ACTIVE"
                time.sleep(0.033)
                continue

            # ── STATE 2: NAV2 VELOCITY TRACKING & ALTITUDE HOLD + COLLISION SHIELD ──
            elif self.mission_state == "NAV2_ACTIVE":
                now_t = time.time()
                with self.cmd_lock:
                    # Timeout safety: If Nav2 hasn't sent cmd_vel in 0.8s, hover in place
                    if (now_t - self.last_cmd_time) < 0.8:
                        vx = self.cmd_vx
                        vy = self.cmd_vy
                        wz = self.cmd_wz
                    else:
                        vx = 0.0
                        vy = 0.0
                        wz = 0.0

                # Read reactive collision shield
                with self.shield_lock:
                    rep_x = self.repulsive_vx
                    rep_y = self.repulsive_vy
                    obs_dist = self.min_obstacle_dist

                # Fuse Nav2 velocity with reactive collision shield
                safe_vx = vx + rep_x
                safe_vy = vy + rep_y

                with self.pose_lock:
                    cx, cy, cz, cyaw = self.drone_x, self.drone_y, self.drone_z, self.drone_yaw

                # Emergency brake and wall pushback if critically close (< 0.30m)
                if obs_dist < 0.30:
                    safe_vx = float(np.clip(rep_x, -0.35, 0.35))
                    safe_vy = float(np.clip(rep_y, -0.35, 0.35))
                    if now_t - self._last_shield_log > 1.0:
                        self.get_logger().warning(
                            f'🛡️ [COLLISION SHIELD] Wall at {obs_dist:.2f}m! Repulsive pushback: '
                            f'vx={safe_vx:+.2f}, vy={safe_vy:+.2f}'
                        )
                        self._last_shield_log = now_t
                else:
                    safe_vx = float(np.clip(safe_vx, -0.40, 0.40))
                    safe_vy = float(np.clip(safe_vy, -0.40, 0.40))

                # Closed-loop Altitude PID
                alt_err = TARGET_ALTITUDE - cz
                vz = float(np.clip(2.0 * alt_err, -0.60, 0.60))

                self.send_target_vel(safe_vx, safe_vy, vz, wz)

                if (abs(safe_vx) > 0.02 or abs(safe_vy) > 0.02 or abs(wz) > 0.05) and (now_t - last_nav_log > 1.5):
                    self.get_logger().info(f'🛸 Flying Nav2 trajectory: vx={safe_vx:+.2f}, vy={safe_vy:+.2f}, wz={wz:+.2f} | alt={cz:.2f}m | wall_dist={obs_dist:.2f}m')
                    last_nav_log = now_t

                # Track if drone has flown outside the helipad area
                if not self.has_left_helipad:
                    d_from_pad = math.hypot(cx - HELIPAD_X, cy - HELIPAD_Y)
                    if d_from_pad > 1.8 or cy > -7.5:
                        self.has_left_helipad = True
                        self.get_logger().info('🚀 [NAV2 COORDINATOR] Drone has cleared the helipad and entered mission arena!')

                # Check if returning drone has arrived over Helipad outside South Gate
                d_pad = math.hypot(cx - HELIPAD_X, cy - HELIPAD_Y)
                if self.has_left_helipad and self.return_triggered and (d_pad < 0.55 or (cy < -8.2 and abs(cx - HELIPAD_X) < 0.55)) and cz > 0.3:
                    self.get_logger().info(f'🎯 Arrived over Helipad on return (dist={d_pad:.2f}m, y={cy:.2f}m)! Commencing vertical touchdown...')
                    self.mission_state = "LANDING"
                    continue

                time.sleep(0.033)
                continue

            # ── STATE 2.5: RETURN DIRECTLY TO HELIPAD ──
            elif self.mission_state == "RETURN_TO_PAD":
                with self.pose_lock:
                    dx, dy, cz, cyaw = self.drone_x, self.drone_y, self.drone_z, self.drone_yaw
                dist_pad = math.hypot(dx - HELIPAD_X, dy - HELIPAD_Y)
                if dist_pad < 0.35:
                    self.get_logger().info(f'🎯 Perfectly centered over helipad (dist={dist_pad:.2f}m)! Commencing vertical touchdown...')
                    self.mission_state = "LANDING"
                    continue

                # Closed-loop P-controller towards helipad in world frame
                err_wx = HELIPAD_X - dx
                err_wy = HELIPAD_Y - dy
                v_wx = float(np.clip(0.8 * err_wx, -0.35, 0.35))
                v_wy = float(np.clip(0.8 * err_wy, -0.35, 0.35))

                # Rotate world velocity into drone body frame (x forward, y left)
                body_vx = math.cos(cyaw) * v_wx + math.sin(cyaw) * v_wy
                body_vy = -math.sin(cyaw) * v_wx + math.cos(cyaw) * v_wy

                # Read reactive collision shield
                with self.shield_lock:
                    rep_x = self.repulsive_vx
                    rep_y = self.repulsive_vy
                    obs_dist = self.min_obstacle_dist

                # Fuse guidance with collision shield
                if obs_dist < 0.30:
                    safe_vx = float(np.clip(rep_x, -0.35, 0.35))
                    safe_vy = float(np.clip(rep_y, -0.35, 0.35))
                else:
                    safe_vx = float(np.clip(body_vx + rep_x, -0.35, 0.35))
                    safe_vy = float(np.clip(body_vy + rep_y, -0.35, 0.35))

                # Maintain flight altitude at 1.50m
                alt_err = TARGET_ALTITUDE - cz
                vz = float(np.clip(2.0 * alt_err, -0.50, 0.50))
                self.send_target_vel(safe_vx, safe_vy, vz, 0.0)

                now_t = time.time()
                if now_t - last_nav_log > 1.0:
                    self.get_logger().info(f'🎯 Precision Transit to Helipad: pos=({dx:+.2f}, {dy:+.2f}) -> pad=({HELIPAD_X:+.2f}, {HELIPAD_Y:+.2f}) | dist={dist_pad:.2f}m')
                    last_nav_log = now_t

                time.sleep(0.033)
                continue

            # ── STATE 3: PRECISION TOUCHDOWN & DISARM ──
            elif self.mission_state == "LANDING":
                if cz > 0.15:
                    self.send_target_vel(0.0, 0.0, -0.25, 0.0)
                    now_t = time.time()
                    if now_t - last_climb_log > 1.0:
                        self.get_logger().info(f'🛬 Precision descent: alt={cz:.2f}m (vz=-0.25 m/s)')
                        last_climb_log = now_t
                else:
                    self.send_target_vel(0.0, 0.0, 0.0, 0.0)
                    self.arm_vehicle(False)
                    self.set_flight_mode("AUTO.LAND")
                    self.pub_complete.publish(Bool(data=True))
                    self.get_logger().info('🏁 [MISSION COMPLETE] Drone safely landed on helipad! Motors disarmed.')
                    self.mission_state = "DISARMED"
                time.sleep(0.033)
                continue

            elif self.mission_state == "DISARMED":
                self.send_target_vel(0.0, 0.0, 0.0, 0.0)
                time.sleep(0.05)
                continue


def main(args=None):
    rclpy.init(args=args)
    node = DroneNav2Coordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

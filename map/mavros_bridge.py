#!/usr/bin/env python3
"""
MAVROS ROS 2 Bridge Node
Provides standard MAVROS topics and services:
  • Topics Published:
      /mavros/state                     (mavros_msgs/msg/State)
      /mavros/local_position/pose       (geometry_msgs/msg/PoseStamped)
      /mavros/local_position/velocity_local (geometry_msgs/msg/TwistStamped)
  • Services Provided:
      /mavros/cmd/arming                (mavros_msgs/srv/CommandBool)
      /mavros/set_mode                  (mavros_msgs/srv/SetMode)
  • Topics Subscribed:
      /mavros/setpoint_raw/local        (mavros_msgs/msg/PositionTarget)
      /mavros/setpoint_velocity/cmd_vel (geometry_msgs/msg/TwistStamped)
      /mavros/setpoint_position/local   (geometry_msgs/msg/PoseStamped)
"""

import os
import re
import sys
import time
import math
import signal
import subprocess
import threading

def _clean_exit_handler(sig, frame):
    os._exit(0)

signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

# ROS 2 & MAVROS message types
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode

# Gazebo Transport
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.twist_pb2 import Twist as GzTwist
    from gz.msgs10.odometry_pb2 import Odometry as GzOdometry
    from gz.msgs10.boolean_pb2 import Boolean as GzBoolean
except ImportError:
    from gz.transport import Node as GzNode
    from gz.msgs.twist_pb2 import Twist as GzTwist
    from gz.msgs.odometry_pb2 import Odometry as GzOdometry
    from gz.msgs.boolean_pb2 import Boolean as GzBoolean

GAZEBO_TWIST_TOPIC = '/X3/gazebo/command/twist'
GAZEBO_ENABLE_TOPIC = '/X3/enable'


class MavrosBridgeNode(Node):
    def __init__(self):
        super().__init__('mavros_bridge_node')
        self.get_logger().info('Initializing MAVROS ROS 2 Bridge...')

        # Internal state
        self.connected = True
        self.armed = False
        self.mode = "MANUAL"
        self.drone_x = -5.80
        self.drone_y = -8.80
        self.drone_z = 0.05
        self.drone_vx = 0.0
        self.drone_vy = 0.0
        self.drone_vz = 0.0
        self.drone_qw = 0.7071
        self.drone_qx = 0.0
        self.drone_qy = 0.0
        self.drone_qz = 0.7071
        self.drone_yaw = 1.5708
        self.first_pose_received = False
        self.pose_lock = threading.Lock()
        self.running = True

        # Last setpoint command
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_vz = 0.0
        self.cmd_wz = 0.0
        self.cmd_lock = threading.Lock()
        self.last_cmd_time = time.time()

        # ── ROS 2 Publishers ──
        self.pub_state = self.create_publisher(State, '/mavros/state', 10)
        self.pub_pose  = self.create_publisher(PoseStamped, '/mavros/local_position/pose', 10)
        self.pub_vel   = self.create_publisher(TwistStamped, '/mavros/local_position/velocity_local', 10)

        # ── ROS 2 Services ──
        self.srv_arm = self.create_service(CommandBool, '/mavros/cmd/arming', self.handle_arming)
        self.srv_mode = self.create_service(SetMode, '/mavros/set_mode', self.handle_set_mode)

        # ── ROS 2 Subscribers ──
        self.sub_setpoint_raw = self.create_subscription(
            PositionTarget, '/mavros/setpoint_raw/local', self.handle_setpoint_raw, 10
        )
        self.sub_setpoint_vel = self.create_subscription(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', self.handle_setpoint_vel, 10
        )
        self.sub_setpoint_pos = self.create_subscription(
            PoseStamped, '/mavros/setpoint_position/local', self.handle_setpoint_pos, 10
        )

        # ── Gazebo Node Integration ──
        self.gz_node = GzNode()
        self.gz_twist_pub = self.gz_node.advertise(GAZEBO_TWIST_TOPIC, GzTwist)
        self.gz_enable_pub = self.gz_node.advertise(GAZEBO_ENABLE_TOPIC, GzBoolean)
        self.get_logger().info(f'📢 Actuating Gazebo on: {GAZEBO_TWIST_TOPIC} & {GAZEBO_ENABLE_TOPIC}')

        # Initial unpause of Gazebo simulation
        self.unpause_gazebo()

        # Background thread to stream real-time Gazebo ground-truth pose
        self.pose_thread = threading.Thread(target=self._pose_reader_worker, daemon=True)
        self.pose_thread.start()

        # High-frequency timers
        self.timer_state = self.create_timer(0.05, self.publish_state)  # 20 Hz state
        self.timer_pose  = self.create_timer(0.02, self.publish_pose)   # 50 Hz pose
        self.timer_act   = self.create_timer(0.02, self.actuate_gazebo) # 50 Hz actuator

        self.get_logger().info('✅ MAVROS ROS 2 Bridge Ready! Topics active under /mavros/...')

    def unpause_gazebo(self):
        try:
            subprocess.run([
                "gz", "service", "-s", "/world/nidar_world/control",
                "--reqtype", "gz.msgs.WorldControl",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "1500",
                "--req", "pause: false"
            ], capture_output=True, timeout=2)
        except Exception:
            pass

    def _pose_reader_worker(self):
        """Streams ground truth 3D position and orientation directly from Gazebo."""
        world_name = "nidar_world"
        try:
            res = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True, timeout=2)
            if "/world/nidar_world/pose/info" in res.stdout:
                world_name = "nidar_world"
            elif "/world/quadcopter/pose/info" in res.stdout:
                world_name = "quadcopter"
        except Exception:
            pass

        pose_topic = f"/world/{world_name}/pose/info"
        self.get_logger().info(f'📡 Streaming drone pose from: {pose_topic}')

        while self.running:
            try:
                proc = subprocess.Popen(
                    ["gz", "topic", "-t", pose_topic, "-e"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
                )
            except Exception as e:
                self.get_logger().warning(f'Pose reader failed to launch: {e}')
                time.sleep(1.0)
                continue

            curr_ent = None
            in_pos = in_orient = False
            tx = ty = tz = 0.0
            tqx = tqy = tqz = 0.0
            tqw = 1.0
            last_z = 0.05
            last_t = time.time()

            while self.running:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.strip()

                m = re.match(r'name:\s*"([^"]+)"', line)
                if m:
                    curr_ent = m.group(1).lower()
                    in_pos = in_orient = False

                if curr_ent != "x3":
                    continue

                if line == "position {":    in_pos = True
                if line == "orientation {": in_orient = True

                if in_pos:
                    for tag in ('x', 'y', 'z'):
                        m2 = re.match(rf'{tag}:\s*([\-\d.eE+]+)', line)
                        if m2:
                            val = float(m2.group(1))
                            if tag == 'x': tx = val
                            elif tag == 'y': ty = val
                            elif tag == 'z': tz = val

                if in_orient:
                    for tag in ('x', 'y', 'z', 'w'):
                        m2 = re.match(rf'{tag}:\s*([\-\d.eE+]+)', line)
                        if m2:
                            val = float(m2.group(1))
                            if tag == 'x': tqx = val
                            elif tag == 'y': tqy = val
                            elif tag == 'z': tqz = val
                            elif tag == 'w': tqw = val

                if line == "}" and in_pos:
                    in_pos = False
                if line == "}" and in_orient:
                    in_orient = False
                    siny = 2.0 * (tqw * tqz + tqx * tqy)
                    cosy = 1.0 - 2.0 * (tqy**2 + tqz**2)
                    yaw = math.atan2(siny, cosy)

                    now = time.time()
                    dt = now - last_t
                    vz = (tz - last_z) / dt if dt > 0.01 else 0.0
                    last_z, last_t = tz, now

                    with self.pose_lock:
                        self.drone_x = tx
                        self.drone_y = ty
                        self.drone_z = tz
                        self.drone_qw = tqw
                        self.drone_qx = tqx
                        self.drone_qy = tqy
                        self.drone_qz = tqz
                        self.drone_yaw = yaw
                        self.drone_vz = vz
                        self.first_pose_received = True

            proc.kill()
            time.sleep(0.5)

    def publish_state(self):
        msg = State()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.connected = self.connected
        msg.armed = self.armed
        msg.guided = (self.mode == "OFFBOARD" or self.mode == "GUIDED")
        msg.mode = self.mode
        msg.system_status = 4 if self.armed else 3
        self.pub_state.publish(msg)

    def publish_pose(self):
        with self.pose_lock:
            if not self.first_pose_received:
                return
            x, y, z = self.drone_x, self.drone_y, self.drone_z
            qw, qx, qy, qz = self.drone_qw, self.drone_qx, self.drone_qy, self.drone_qz
            vx, vy, vz = self.drone_vx, self.drone_vy, self.drone_vz

        now = self.get_clock().now().to_msg()

        # Pose
        p_msg = PoseStamped()
        p_msg.header.stamp = now
        p_msg.header.frame_id = "map"
        p_msg.pose.position.x = float(x)
        p_msg.pose.position.y = float(y)
        p_msg.pose.position.z = float(z)
        p_msg.pose.orientation.w = float(qw)
        p_msg.pose.orientation.x = float(qx)
        p_msg.pose.orientation.y = float(qy)
        p_msg.pose.orientation.z = float(qz)
        self.pub_pose.publish(p_msg)

        # Velocity
        v_msg = TwistStamped()
        v_msg.header.stamp = now
        v_msg.header.frame_id = "base_link"
        v_msg.twist.linear.x = float(vx)
        v_msg.twist.linear.y = float(vy)
        v_msg.twist.linear.z = float(vz)
        self.pub_vel.publish(v_msg)

    def handle_arming(self, request, response):
        self.armed = request.value
        response.success = True
        response.result = 0
        state_str = "ARMED" if self.armed else "DISARMED"
        self.get_logger().info(f'⚙️ [MAVROS] Motors {state_str}')
        self.unpause_gazebo()
        en_msg = GzBoolean()
        en_msg.data = self.armed
        self.gz_enable_pub.publish(en_msg)
        return response

    def handle_set_mode(self, request, response):
        req_mode = request.custom_mode.upper() if request.custom_mode else "MANUAL"
        self.mode = req_mode
        response.mode_sent = True
        self.get_logger().info(f'⚙️ [MAVROS] Flight Mode set to: {self.mode}')
        return response

    def handle_setpoint_raw(self, msg: PositionTarget):
        with self.cmd_lock:
            self.cmd_vx = msg.velocity.x
            self.cmd_vy = msg.velocity.y
            self.cmd_vz = msg.velocity.z
            self.cmd_wz = msg.yaw_rate
            self.last_cmd_time = time.time()

    def handle_setpoint_vel(self, msg: TwistStamped):
        with self.cmd_lock:
            self.cmd_vx = msg.twist.linear.x
            self.cmd_vy = msg.twist.linear.y
            self.cmd_vz = msg.twist.linear.z
            self.cmd_wz = msg.twist.angular.z
            self.last_cmd_time = time.time()

    def handle_setpoint_pos(self, msg: PoseStamped):
        with self.pose_lock:
            cx, cy, cz = self.drone_x, self.drone_y, self.drone_z
            yaw = self.drone_yaw

        tx = msg.pose.position.x
        ty = msg.pose.position.y
        tz = msg.pose.position.z

        dx = tx - cx
        dy = ty - cy
        dz = tz - cz

        vx_b = math.cos(yaw) * dx + math.sin(yaw) * dy
        vy_b = -math.sin(yaw) * dx + math.cos(yaw) * dy

        target_yaw = math.atan2(dy, dx)
        yaw_err = (target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

        with self.cmd_lock:
            self.cmd_vx = max(-0.85, min(0.85, 0.8 * vx_b))
            self.cmd_vy = max(-0.50, min(0.50, 0.8 * vy_b))
            self.cmd_vz = max(-0.50, min(0.50, 1.2 * dz))
            self.cmd_wz = max(-1.0, min(1.0, 1.4 * yaw_err))
            self.last_cmd_time = time.time()

    def actuate_gazebo(self):
        if not self.armed or self.mode not in ("OFFBOARD", "GUIDED", "AUTO.RTL", "AUTO.LAND"):
            gz_msg = GzTwist()
            self.gz_twist_pub.publish(gz_msg)
            return

        # Ensure Gazebo multicopter controller is continuously enabled
        en_msg = GzBoolean()
        en_msg.data = True
        self.gz_enable_pub.publish(en_msg)

        now = time.time()
        with self.cmd_lock:
            if now - self.last_cmd_time > 0.5:
                vx, vy, vz, wz = 0.0, 0.0, 0.0, 0.0
            else:
                vx, vy, vz, wz = self.cmd_vx, self.cmd_vy, self.cmd_vz, self.cmd_wz

        gz_msg = GzTwist()
        gz_msg.linear.x = float(vx)
        gz_msg.linear.y = float(vy)
        gz_msg.linear.z = float(vz)
        gz_msg.angular.z = float(wz)
        self.gz_twist_pub.publish(gz_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MavrosBridgeNode()
    try:
        rclpy.spin(node)
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

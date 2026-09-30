#!/usr/bin/env python3
"""
Drone TF and Odometry Bridge for Nav2
====================================
Broadcasts the complete transform tree and standard odometry required by Nav2:
  • TF: map -> odom
  • TF: odom -> base_link
  • TF: base_link -> base_footprint (static)
  • TF: base_link -> lidar_link (static)
  • Topic: /odom (nav_msgs/msg/Odometry)
  • Topic: /scan (sensor_msgs/msg/LaserScan) bridging Gazebo LiDAR
"""

import os
import sys
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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import tf2_ros

try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.laserscan_pb2 import LaserScan as GzLaserScan
except ImportError:
    from gz.transport import Node as GzNode
    from gz.msgs.laserscan_pb2 import LaserScan as GzLaserScan


class DroneTfOdomPublisher(Node):
    def __init__(self):
        super().__init__('drone_tf_odom_publisher')
        self.get_logger().info('Starting Drone TF & Odometry Bridge for Nav2...')

        # State storage
        self.latest_pose = None
        self.latest_twist = None
        self.lock = threading.Lock()

        # Publishers
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)
        self.pub_scan = self.create_publisher(LaserScan, '/scan', 10)
        self.pub_lidar_viz = self.create_publisher(LaserScan, '/nidar/lidar_scan', 10)
        self.pub_drone_marker = self.create_publisher(Marker, '/nidar/drone_marker', 10)

        # TF Broadcasters
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        now_msg = self.get_clock().now().to_msg()

        # Broadcast static base_link -> lidar_link
        static_lidar = TransformStamped()
        static_lidar.header.stamp = now_msg
        static_lidar.header.frame_id = 'base_link'
        static_lidar.child_frame_id = 'lidar_link'
        static_lidar.transform.translation.x = 0.0
        static_lidar.transform.translation.y = 0.0
        static_lidar.transform.translation.z = 0.05
        static_lidar.transform.rotation.w = 1.0

        # Broadcast static base_link -> base_footprint
        static_bf = TransformStamped()
        static_bf.header.stamp = now_msg
        static_bf.header.frame_id = 'base_link'
        static_bf.child_frame_id = 'base_footprint'
        static_bf.transform.translation.x = 0.0
        static_bf.transform.translation.y = 0.0
        static_bf.transform.translation.z = -0.05
        static_bf.transform.rotation.w = 1.0

        # Broadcast static base_link -> camera_link (imx219_stereo RGB-D position)
        static_cam = TransformStamped()
        static_cam.header.stamp = now_msg
        static_cam.header.frame_id = 'base_link'
        static_cam.child_frame_id = 'camera_link'
        static_cam.transform.translation.x = 0.17
        static_cam.transform.translation.y = 0.0
        static_cam.transform.translation.z = 0.025
        static_cam.transform.rotation.w = 1.0

        # Broadcast static camera_link -> camera_optical_frame (standard ROS REP-103 optical convention)
        static_cam_opt = TransformStamped()
        static_cam_opt.header.stamp = now_msg
        static_cam_opt.header.frame_id = 'camera_link'
        static_cam_opt.child_frame_id = 'camera_optical_frame'
        static_cam_opt.transform.translation.x = 0.0
        static_cam_opt.transform.translation.y = 0.0
        static_cam_opt.transform.translation.z = 0.0
        static_cam_opt.transform.rotation.x = 0.5
        static_cam_opt.transform.rotation.y = -0.5
        static_cam_opt.transform.rotation.z = 0.5
        static_cam_opt.transform.rotation.w = -0.5

        self.tf_static_broadcaster.sendTransform([static_lidar, static_bf, static_cam, static_cam_opt])

        # Subscriptions
        self.sub_pose = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.cb_pose, 10
        )
        self.sub_vel = self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_local', self.cb_vel, 10
        )

        # Gazebo LiDAR Subscriber (subscribing to both /nidar/lidar and /lidar)
        self.gz_node = GzNode()
        self.gz_node.subscribe(GzLaserScan, '/nidar/lidar', self.cb_gz_lidar)
        self.gz_node.subscribe(GzLaserScan, '/lidar', self.cb_gz_lidar)

        # High frequency TF & Odom timer (50Hz)
        self.timer = self.create_timer(0.02, self.publish_tf_and_odom)

    def cb_pose(self, msg: PoseStamped):
        with self.lock:
            self.latest_pose = msg

    def cb_vel(self, msg: TwistStamped):
        with self.lock:
            self.latest_twist = msg

    def cb_gz_lidar(self, msg: GzLaserScan):
        ranges = list(msg.ranges)
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'lidar_link'
        scan_msg.angle_min = float(msg.angle_min)
        scan_msg.angle_max = float(msg.angle_min + len(ranges) * msg.angle_step)
        scan_msg.angle_increment = float(msg.angle_step)
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.05
        scan_msg.range_min = float(msg.range_min)
        scan_msg.range_max = float(min(msg.range_max, 16.0))
        scan_msg.ranges = [float(r) if (not math.isinf(r) and not math.isnan(r)) else float('inf') for r in ranges]
        self.pub_scan.publish(scan_msg)
        self.pub_lidar_viz.publish(scan_msg)

    def _build_drone_marker(self, now_msg):
        # 1. Complete Solid Filled Triangle
        m = Marker()
        m.header.stamp = now_msg
        m.header.frame_id = 'base_link'
        m.ns = 'drone'
        m.id = 0
        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        # High-visibility Electric Cyan solid body
        m.color.r = 0.0
        m.color.g = 0.90
        m.color.b = 1.0
        m.color.a = 0.92
        m.pose.orientation.w = 1.0

        # Complete 3-Sided Triangle: 0.48m rear base, 0.62m total length to forward apex
        apex = Point(x=0.38, y=0.0, z=0.04)
        rear_left = Point(x=-0.24, y=0.24, z=0.04)
        rear_right = Point(x=-0.24, y=-0.24, z=0.04)

        m.points = [
            apex, rear_left, rear_right,
            apex, rear_right, rear_left
        ]
        return m

    def _build_drone_border_marker(self, now_msg):
        # 2. Complete Triangle Thick High-Contrast Border (3 Straight Sides)
        m = Marker()
        m.header.stamp = now_msg
        m.header.frame_id = 'base_link'
        m.ns = 'drone'
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.04  # Bold 4cm border
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0
        m.pose.orientation.w = 1.0

        apex = Point(x=0.38, y=0.0, z=0.045)
        rear_left = Point(x=-0.24, y=0.24, z=0.045)
        rear_right = Point(x=-0.24, y=-0.24, z=0.045)

        m.points = [apex, rear_left, rear_right, apex]
        return m

    def publish_tf_and_odom(self):
        with self.lock:
            pose_msg = self.latest_pose
            twist_msg = self.latest_twist

        now_msg = self.get_clock().now().to_msg()

        # 1. Always Broadcast TF: map -> odom
        t_map_odom = TransformStamped()
        t_map_odom.header.stamp = now_msg
        t_map_odom.header.frame_id = 'map'
        t_map_odom.child_frame_id = 'odom'
        t_map_odom.transform.translation.x = 0.0
        t_map_odom.transform.translation.y = 0.0
        t_map_odom.transform.translation.z = 0.0
        t_map_odom.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t_map_odom)

        if pose_msg is None:
            # Broadcast origin transform so Nav2 costmaps activate immediately at startup
            t_odom_base = TransformStamped()
            t_odom_base.header.stamp = now_msg
            t_odom_base.header.frame_id = 'odom'
            t_odom_base.child_frame_id = 'base_link'
            t_odom_base.transform.translation.x = 0.0
            t_odom_base.transform.translation.y = 0.0
            t_odom_base.transform.translation.z = 0.0
            t_odom_base.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t_odom_base)
            return

        # 2. Broadcast TF: odom -> base_link
        t_odom_base = TransformStamped()
        t_odom_base.header.stamp = now_msg
        t_odom_base.header.frame_id = 'odom'
        t_odom_base.child_frame_id = 'base_link'
        t_odom_base.transform.translation.x = float(pose_msg.pose.position.x)
        t_odom_base.transform.translation.y = float(pose_msg.pose.position.y)
        t_odom_base.transform.translation.z = float(pose_msg.pose.position.z)
        t_odom_base.transform.rotation = pose_msg.pose.orientation
        self.tf_broadcaster.sendTransform(t_odom_base)

        # 3. Publish Odometry on /odom
        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose = pose_msg.pose
        if twist_msg is not None:
            odom.twist.twist = twist_msg.twist
        self.pub_odom.publish(odom)

        # 4. Publish Complete Solid Triangle Drone Model on /nidar/drone_marker
        self.pub_drone_marker.publish(self._build_drone_marker(now_msg))
        self.pub_drone_marker.publish(self._build_drone_border_marker(now_msg))


def main(args=None):
    rclpy.init(args=args)
    node = DroneTfOdomPublisher()
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

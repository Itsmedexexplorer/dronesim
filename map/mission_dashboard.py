#!/usr/bin/env python3
"""
NIDAR Mission Live HUD & Statistics Dashboard
==============================================
Calculates and publishes real-time mission telemetry:
  • Explored Area (%) & m²
  • Unexplored Area (%) & m²
  • Survivors Located (Count / 6)
  • Mission Elapsed Time (MM:SS)
  • Flight Distance Traveled (meters)
  • Live 3D Floating HUD Billboard in RViz (/nidar/mission_hud)
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
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, Bool
from visualization_msgs.msg import Marker, MarkerArray

class MissionDashboard(Node):
    def __init__(self):
        super().__init__('mission_dashboard')
        self.get_logger().info('🚀 Starting NIDAR Mission Live HUD & Statistics Node...')

        self.start_time = time.time()
        self.completion_time = None
        self.is_completed = False

        self.survivor_count = 0
        self.total_survivors = 6
        self.explored_pct = 0.0
        self.unexplored_pct = 100.0
        self.explored_m2 = 0.0
        self.unexplored_m2 = 225.0  # 15m x 15m indoor compound = 225 m²

        self.cur_x = -5.80
        self.cur_y = -8.80
        self.cur_z = 0.0
        self.dist_traveled = 0.0
        self.last_pose = None
        self.lock = threading.Lock()

        # Subscribers
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.cb_map, map_qos)
        self.sub_survivors = self.create_subscription(Int32, '/nidar/survivor_count', self.cb_survivors, 10)
        self.sub_pose = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.cb_pose, 10)
        self.sub_complete = self.create_subscription(Bool, '/nidar/mission_complete', self.cb_complete, 10)
        self.sub_relaunch = self.create_subscription(Bool, '/nidar/relaunch_command', self.cb_relaunch, 10)

        # Publisher for RViz 3D HUD Billboard
        hud_qos = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.pub_hud = self.create_publisher(MarkerArray, '/nidar/mission_hud', hud_qos)

        # Update HUD at 2 Hz
        self.timer_hud = self.create_timer(0.5, self.update_and_publish_hud)
        # Terminal summary log every 4 seconds
        self.timer_log = self.create_timer(4.0, self.log_terminal_summary)

    def cb_survivors(self, msg: Int32):
        with self.lock:
            self.survivor_count = msg.data

    def cb_complete(self, msg: Bool):
        with self.lock:
            if msg.data and not self.is_completed:
                self.is_completed = True
                self.completion_time = time.time()
                elapsed = self.completion_time - self.start_time
                m, s = divmod(int(elapsed), 60)
                self.get_logger().info(f'🏁 [MISSION COMPLETED] Total Time Taken: {m:02d}m {s:02d}s | Survivors: {self.survivor_count}')
            elif not msg.data and self.is_completed:
                self.is_completed = False
                self.completion_time = None

    def cb_relaunch(self, msg: Bool):
        if msg.data:
            with self.lock:
                self.is_completed = False
                self.completion_time = None
                self.start_time = time.time()
                self.get_logger().info('🚀 [MISSION HUD] Drone relaunched! Resuming mission timer & HUD tracking.')

    def cb_pose(self, msg: PoseStamped):
        with self.lock:
            px = msg.pose.position.x
            py = msg.pose.position.y
            pz = msg.pose.position.z
            self.cur_x, self.cur_y, self.cur_z = px, py, pz

            if self.last_pose is not None:
                d = math.hypot(px - self.last_pose[0], py - self.last_pose[1])
                if 0.02 < d < 2.0:  # Valid movement step
                    self.dist_traveled += d
            self.last_pose = (px, py)

    def cb_map(self, msg: OccupancyGrid):
        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        data = np.array(msg.data, dtype=np.int8).reshape((h, w))

        # Crop to the 15m x 15m indoor arena (-7.5 to 7.5 in x and y)
        c0 = max(0, int((-7.5 - ox) / res))
        c1 = min(w, int((7.5 - ox) / res))
        r0 = max(0, int((-7.5 - oy) / res))
        r1 = min(h, int((7.5 - oy) / res))

        arena_data = data[r0:r1, c0:c1]
        total_cells = arena_data.size
        if total_cells == 0:
            return

        free_cells = np.sum(arena_data == 0)
        wall_cells = np.sum(arena_data == 100)
        unknown_cells = np.sum(arena_data == -1)

        explored_cells = free_cells + wall_cells
        cell_area = res * res

        with self.lock:
            self.explored_pct = (explored_cells / float(total_cells)) * 100.0
            self.unexplored_pct = (unknown_cells / float(total_cells)) * 100.0
            self.explored_m2 = explored_cells * cell_area
            self.unexplored_m2 = unknown_cells * cell_area

    def update_and_publish_hud(self):
        with self.lock:
            if self.is_completed and self.completion_time:
                elapsed = self.completion_time - self.start_time
                status_str = "🏁 MISSION COMPLETED"
            else:
                elapsed = time.time() - self.start_time
                status_str = "🔍 ACTIVE EXPLORATION & RESCUE"

            mins, secs = divmod(int(elapsed), 60)
            exp_pct = self.explored_pct
            unexp_pct = self.unexplored_pct
            exp_m2 = self.explored_m2
            unexp_m2 = self.unexplored_m2
            surv = self.survivor_count
            tot_surv = self.total_survivors
            dist = self.dist_traveled
            cz = self.cur_z

        now_msg = self.get_clock().now().to_msg()
        ma = MarkerArray()

        # 1. Dark semi-transparent background billboard panel
        bg = Marker()
        bg.header.frame_id = 'map'
        bg.header.stamp = now_msg
        bg.ns = 'mission_hud_bg'
        bg.id = 0
        bg.type = Marker.CUBE
        bg.action = Marker.ADD
        bg.pose.position.x = 0.0
        bg.pose.position.y = 9.80  # Placed right along the North Wall
        bg.pose.position.z = 4.20
        bg.pose.orientation.w = 1.0
        bg.scale.x = 14.5
        bg.scale.y = 0.20
        bg.scale.z = 4.0
        bg.color.r = 0.05
        bg.color.g = 0.08
        bg.color.b = 0.12
        bg.color.a = 0.85
        ma.markers.append(bg)

        # 2. Cyan border accent line for the billboard
        border = Marker()
        border.header.frame_id = 'map'
        border.header.stamp = now_msg
        border.ns = 'mission_hud_border'
        border.id = 1
        border.type = Marker.CUBE
        border.action = Marker.ADD
        border.pose.position.x = 0.0
        border.pose.position.y = 9.75
        border.pose.position.z = 6.20
        border.pose.orientation.w = 1.0
        border.scale.x = 14.6
        border.scale.y = 0.22
        border.scale.z = 0.12
        border.color.r = 0.0
        border.color.g = 0.85
        border.color.b = 1.0
        border.color.a = 1.0
        ma.markers.append(border)

        # 3. High-Contrast Mission Telemetry Text
        txt = Marker()
        txt.header.frame_id = 'map'
        txt.header.stamp = now_msg
        txt.ns = 'mission_hud_text'
        txt.id = 2
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = 0.0
        txt.pose.position.y = 9.50
        txt.pose.position.z = 4.20
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.45  # Readable font scale
        txt.color.r = 1.0
        txt.color.g = 1.0
        txt.color.b = 1.0
        txt.color.a = 1.0

        hud_content = (
            f"╔═════════════════════════════════════════════════════════════╗\n"
            f"        NIDAR JEV AUTONOMOUS MISSION DASHBOARD        \n"
            f"╠═════════════════════════════════════════════════════════════╣\n"
            f"  ⏱️ Time Elapsed:     {mins:02d}m {secs:02d}s\n"
            f"  🗺️ Explored Area:    {exp_pct:5.1f}%  ({exp_m2:5.1f} m²)\n"
            f"  ❓ Unexplored Area:  {unexp_pct:5.1f}%  ({unexp_m2:5.1f} m²)\n"
            f"  👥 Survivors Found:  {surv} / {tot_surv}  HUMANS TAGGED\n"
            f"  🛸 Flight Traveled:  {dist:5.1f} m  (Altitude: {cz:4.2f}m)\n"
            f"  📡 Status:           {status_str}\n"
            f"╚═════════════════════════════════════════════════════════════╝"
        )
        txt.text = hud_content
        ma.markers.append(txt)

        self.pub_hud.publish(ma)

    def log_terminal_summary(self):
        with self.lock:
            elapsed = (self.completion_time if self.is_completed else time.time()) - self.start_time
            m, s = divmod(int(elapsed), 60)
            self.get_logger().info(
                f'📊 [MISSION HUD] Time: {m:02d}:{s:02d} | '
                f'Explored: {self.explored_pct:.1f}% ({self.explored_m2:.1f}m²) | '
                f'Unexplored: {self.unexplored_pct:.1f}% | '
                f'Survivors: {self.survivor_count}/{self.total_survivors} | '
                f'Traveled: {self.dist_traveled:.1f}m'
            )

def main(args=None):
    rclpy.init(args=args)
    node = MissionDashboard()
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

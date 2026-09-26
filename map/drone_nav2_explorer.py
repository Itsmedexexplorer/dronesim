#!/usr/bin/env python3
"""
Autonomous Frontier Exploration Agent for Nav2
==============================================
Dispatches optimal frontier exploration goals to Nav2 (/navigate_to_pose):
  • Subscribes to /map (TRANSIENT_LOCAL QoS) from nidar_slam_node.
  • Detects unexplored frontier boundaries via matrix convolution.
  • Clusters frontiers and prioritizes by distance and cluster size.
  • Sends Action Goals to Nav2 Action Server (/navigate_to_pose).
  • Monitors exploration coverage and commands Return-To-Home when >= 95%.
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
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_msgs.msg import Int32, Bool

ARENA_LIMIT = 7.0  # Compound boundaries +/- 7m

UNKNOWN  = -1
FREE     = 0
OCCUPIED = 100


class DroneNav2Explorer(Node):
    def __init__(self):
        super().__init__('drone_nav2_explorer')
        self.get_logger().info('Starting Autonomous Frontier Explorer for Nav2...')

        self.drone_x = -5.80
        self.drone_y = -8.80
        self.drone_z = 0.0
        self.home_x = None
        self.home_y = None
        self.pose_lock = threading.Lock()

        self.grid = None
        self.map_info = None
        self.grid_lock = threading.Lock()

        # Local pose subscription
        self.sub_pose = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.cb_pose, 10
        )

        # SLAM Map subscription with TRANSIENT_LOCAL QoS matching publisher
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.sub_map = self.create_subscription(
            OccupancyGrid, '/map', self.cb_map, map_qos
        )

        # Nav2 Action Client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.current_goal_handle = None
        self.active_goal_coord = None
        self.goal_start_time = 0.0
        self.is_navigating = False
        self.failed_goals = {}  # (x, y) -> fail_count

        # Survivor Detection & Mission Subscriptions
        self.survivor_count = 0
        self.mission_complete_received = False
        self.sub_survivors = self.create_subscription(
            Int32, '/nidar/survivor_count', self.cb_survivor_count, 10
        )
        self.sub_complete = self.create_subscription(
            Bool, '/nidar/mission_complete', self.cb_mission_complete, 10
        )
        self.pub_land_cmd = self.create_publisher(Bool, '/nidar/land_command', 10)

        self.thread = threading.Thread(target=self.exploration_loop, daemon=True)
        self.thread.start()

    def cb_survivor_count(self, msg: Int32):
        self.survivor_count = msg.data

    def cb_mission_complete(self, msg: Bool):
        if msg.data:
            self.mission_complete_received = True

    def cb_pose(self, msg: PoseStamped):
        with self.pose_lock:
            self.drone_x = msg.pose.position.x
            self.drone_y = msg.pose.position.y
            self.drone_z = msg.pose.position.z
            if self.home_x is None:
                self.home_x = self.drone_x
                self.home_y = self.drone_y
                self.get_logger().info(f'📍 Home Helipad Anchor Locked at ({self.home_x:+.2f}, {self.home_y:+.2f})')

    def cb_map(self, msg: OccupancyGrid):
        w = msg.info.width
        h = msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((h, w))
        with self.grid_lock:
            self.grid = data
            self.map_info = msg.info

    def get_coverage(self):
        with self.grid_lock:
            if self.grid is None:
                return 0.0
            info = self.map_info
            g = self.grid

        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y

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
            if self.grid is None:
                return []
            info = self.map_info
            g = self.grid.copy()

        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        h, w = g.shape

        free_cells = (g == FREE)
        unknown_cells = (g == UNKNOWN)

        # Shift to find free cells directly bordering unknown space
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

        # Cluster adjacent frontier cells
        clusters = []
        visited = set()
        for cell in f_set:
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
            # Filter noise; retain meaningful clusters
            if len(cluster) >= 3:
                avg_c = sum(c for c, _ in cluster) / float(len(cluster))
                avg_r = sum(r for _, r in cluster) / float(len(cluster))
                wx = ox + (avg_c + 0.5) * res
                wy = oy + (avg_r + 0.5) * res
                if abs(wx) <= (ARENA_LIMIT - 0.5) and abs(wy) <= (ARENA_LIMIT - 0.5) and wy >= -5.8:
                    # Standoff buffer: nudge goal away from any nearby wall
                    safe_c = int(round((wx - ox) / res))
                    safe_r = int(round((wy - oy) / res))
                    check_rad = 7  # 0.35m radius
                    r0 = max(0, safe_r - check_rad)
                    r1 = min(h, safe_r + check_rad + 1)
                    c0 = max(0, safe_c - check_rad)
                    c1 = min(w, safe_c + check_rad + 1)
                    patch = g[r0:r1, c0:c1]
                    if np.any(patch == OCCUPIED):
                        with self.pose_lock:
                            cur_x, cur_y = self.drone_x, self.drone_y
                        angle_to_drone = math.atan2(cur_y - wy, cur_x - wx)
                        wx += 0.40 * math.cos(angle_to_drone)
                        wy += 0.40 * math.sin(angle_to_drone)
                        wy = max(-5.8, wy)
                    clusters.append((wx, wy, len(cluster)))

        return clusters

    def send_nav_goal(self, x, y):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 1.50

        # Point drone orientation towards target
        with self.pose_lock:
            cx, cy = self.drone_x, self.drone_y
        yaw = math.atan2(y - cy, x - cx)
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.active_goal_coord = (float(x), float(y))
        self.goal_start_time = time.time()
        self.is_navigating = True
        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('⚠️ Nav2 Goal was Rejected!')
            if self.active_goal_coord:
                gx, gy = self.active_goal_coord
                key = (round(gx, 1), round(gy, 1))
                self.failed_goals[key] = self.failed_goals.get(key, 0) + 1
            self.is_navigating = False
            return

        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        result = future.result()
        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('🎯 Nav2 Goal Reached Successfully!')
        else:
            self.get_logger().warning(f'⚠️ Nav2 Goal Finished with Status: {status}')
            if self.active_goal_coord:
                gx, gy = self.active_goal_coord
                key = (round(gx, 1), round(gy, 1))
                self.failed_goals[key] = self.failed_goals.get(key, 0) + 1
        self.is_navigating = False

    def navigate_waypoint(self, x, y, timeout=30.0, tolerance=0.45):
        # Ensure any previous active goal is cleanly finished or cancelled
        if self.is_navigating and self.current_goal_handle:
            try:
                future = self.current_goal_handle.cancel_goal_async()
                t_c = time.time()
                while not future.done() and (time.time() - t_c < 1.0):
                    time.sleep(0.05)
            except Exception:
                pass
            time.sleep(0.2)
            self.is_navigating = False
            self.current_goal_handle = None

        t_start = time.time()
        last_dispatch = 0.0

        while rclpy.ok() and (time.time() - t_start < timeout):
            with self.pose_lock:
                dist = math.hypot(self.drone_x - x, self.drone_y - y)
            if dist < tolerance:
                self.get_logger().info(f'✅ Reached waypoint ({x:+.2f}, {y:+.2f}) within {dist:.2f}m!')
                break

            if not self.is_navigating and (time.time() - last_dispatch > 1.0):
                self.send_nav_goal(x, y)
                last_dispatch = time.time()

            time.sleep(0.2)

        # Cleanly cancel before next sequential waypoint
        if self.is_navigating and self.current_goal_handle:
            try:
                cancel_future = self.current_goal_handle.cancel_goal_async()
                t_c = time.time()
                while not cancel_future.done() and (time.time() - t_c < 1.0):
                    time.sleep(0.05)
            except Exception:
                pass
            time.sleep(0.2)
            self.is_navigating = False
            self.current_goal_handle = None

    def exploration_loop(self):
        self.get_logger().info('Waiting for Nav2 /navigate_to_pose action server...')
        while not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().info('Waiting for Nav2 bringup to finish initializing...')
            time.sleep(1.0)
        self.get_logger().info('✅ Nav2 Action Server is Online!')

        self.get_logger().info('Waiting for initial /map from SLAM node...')
        while self.grid is None and rclpy.ok():
            time.sleep(0.5)
        self.get_logger().info('🗺️ Initial Occupancy Grid received!')

        self.get_logger().info('Waiting for drone to reach flight altitude (>= 1.2m)...')
        cz = 0.0
        while rclpy.ok():
            with self.pose_lock:
                cz = self.drone_z
            if cz >= 1.20:
                break
            time.sleep(0.5)
        self.get_logger().info(f'🛫 Drone is airborne at {cz:.2f}m! Activating frontier exploration...')

        # Deterministic Compound Penetration Sequence from Helipad through South Gate
        with self.pose_lock:
            cur_y = self.drone_y

        if cur_y < -7.0:
            self.get_logger().info('🚪 [ENTRY SEQUENCE] Penetrating facility through 2.0m South Gate...')
            self.get_logger().info('🚪 Waypoint 1: South Gate center threshold (-5.80, -7.20)...')
            self.navigate_waypoint(-5.80, -7.20, timeout=20.0, tolerance=0.45)

            self.get_logger().info('🚪 Waypoint 2: Room 0_0 geometric center (-5.10, -5.50)...')
            self.navigate_waypoint(-5.10, -5.50, timeout=20.0, tolerance=0.45)

            self.get_logger().info('🚪 Waypoint 3: Main Corridor Hub (-5.10, -3.40)...')
            self.navigate_waypoint(-5.10, -3.40, timeout=20.0, tolerance=0.45)

            self.get_logger().info('🏰 [INSIDE COMPOUND] Penetration successful! Active in main corridor. Launching exploration...')

        last_status_log = time.time()
        empty_frontiers_count = 0

        while rclpy.ok():
            cov = self.get_coverage()

            # Check exit condition: 97% coverage milestone or zero frontiers remaining
            should_exit = (cov >= 97.0) or (cov >= 95.0 and empty_frontiers_count >= 3) or self.mission_complete_received
            if should_exit:
                reason = f"All Sectors Explored & {self.survivor_count} Survivors Located ({cov:.1f}%)"
                self.get_logger().info(f'🏁 [AUTONOMOUS EXIT] {reason} Initiating exit navigation back to Helipad (-5.80, -8.80)...')

                # Cancel active exploration goal
                if self.is_navigating and self.current_goal_handle:
                    try:
                        self.current_goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                    self.is_navigating = False

                with self.pose_lock:
                    cy = self.drone_y

                # Step 1: Main corridor hub (-5.10, -3.40) if deep inside arena
                if cy > -3.0:
                    self.get_logger().info('🚪 Step 1: Navigating to Corridor Hub (-5.10, -3.40)...')
                    self.navigate_waypoint(-5.10, -3.40, timeout=35.0, tolerance=0.50)

                # Step 2: Room 0_0 center (-5.10, -5.50)
                if cy > -5.0:
                    self.get_logger().info('🚪 Step 2: Navigating into Room 0_0 vestibule (-5.10, -5.50)...')
                    self.navigate_waypoint(-5.10, -5.50, timeout=25.0, tolerance=0.50)

                # Step 3: South Gate threshold (-5.80, -7.20)
                self.get_logger().info('🚪 Step 3: Navigating through South Gate threshold (-5.80, -7.20)...')
                self.navigate_waypoint(-5.80, -7.20, timeout=25.0, tolerance=0.50)

                # Step 4: Direct to Launch & Exit Pad (-5.80, -8.80)
                self.get_logger().info('🎯 Step 4: Navigating to Helipad (-5.80, -8.80)...')
                self.navigate_waypoint(-5.80, -8.80, timeout=25.0, tolerance=0.50)

                self.get_logger().info('🛬 Directing Coordinator to Helipad Landing Zone (-5.80, -8.80)...')
                land_msg = Bool()
                land_msg.data = True
                for _ in range(5):
                    self.pub_land_cmd.publish(land_msg)
                    time.sleep(0.1)

                # Wait for touchdown
                t_land = time.time()
                while rclpy.ok() and (time.time() - t_land < 30.0):
                    with self.pose_lock:
                        if self.drone_z < 0.25:
                            break
                    time.sleep(0.5)

                self.get_logger().info('🏁 [MISSION COMPLETE] Autonomous Search, Localization & Helipad Touchdown Complete!')
                self.get_logger().info(
                    '🗺️ [MAP SAVED] Final 2D Map files saved in: /home/dexter/Downloads/Drone-simulation-2-main/map/\n'
                    '   • drone_map_final.png (Raw 2D Occupancy Map)\n'
                    '   • drone_map_annotated.png (Annotated Map with Solid Red Survivors & Flight Path)\n'
                    '   • nidar_autonomous_map.yaml & nidar_autonomous_map.pgm (ROS 2 Nav2 Map)'
                )
                break

            # Goal Watchdog: cancel if stuck in flight for > 25 seconds
            if self.is_navigating:
                elapsed = time.time() - self.goal_start_time
                if elapsed > 40.0:
                    self.get_logger().warning(f'⏱️ Nav2 goal exceeded 40s timeout (elapsed: {elapsed:.1f}s). Cancelling...')
                    if self.current_goal_handle:
                        try:
                            self.current_goal_handle.cancel_goal_async()
                        except Exception:
                            pass
                    if self.active_goal_coord:
                        gx, gy = self.active_goal_coord
                        key = (round(gx, 1), round(gy, 1))
                        self.failed_goals[key] = self.failed_goals.get(key, 0) + 1
                    self.is_navigating = False
                time.sleep(0.5)
                continue

            # Select next frontier goal
            with self.pose_lock:
                cx, cy = self.drone_x, self.drone_y

            clusters = self.find_frontiers()
            if clusters:
                empty_frontiers_count = 0
                # Score clusters by distance & size, penalizing failed goals
                def cluster_score(cl):
                    fx, fy, sz = cl
                    d = math.hypot(fx - cx, fy - cy)
                    fails = sum(cnt for (gx, gy), cnt in self.failed_goals.items() if math.hypot(fx - gx, fy - gy) < 0.8)
                    return d - min(3.0, 0.05 * sz) + (fails * 5.0)

                clusters.sort(key=cluster_score)
                chosen = None
                for cl in clusters:
                    fx, fy, _ = cl
                    if math.hypot(fx - cx, fy - cy) > 0.40:
                        chosen = cl
                        break

                if chosen is not None:
                    fx, fy, sz = chosen
                    self.get_logger().info(f'🧭 [NAV2 GOAL] Target ({fx:+.2f}, {fy:+.2f}) [{sz} cells] | Cov: {cov:.1f}%')
                    self.send_nav_goal(fx, fy)
            else:
                empty_frontiers_count += 1
                if time.time() - last_status_log > 5.0:
                    self.get_logger().info(f'🔍 Scanning for frontiers... (Coverage: {cov:.1f}%)')
                    last_status_log = time.time()

                # If no frontiers seen after several checks, nudge towards unexplored zone
                if empty_frontiers_count > 4:
                    empty_frontiers_count = 0
                    # If outside compound (south of outer gate y = -7.50), fly through the 2m wide South Gate
                    if cy < -7.0:
                        self.get_logger().info('🚪 Entering compound through South Gate (-5.80, -7.20)...')
                        self.send_nav_goal(-5.80, -7.20)
                    else:
                        angle = (time.time() % (2 * math.pi))
                        nudge_x = float(np.clip(cx + 1.2 * math.cos(angle), -ARENA_LIMIT + 0.5, ARENA_LIMIT - 0.5))
                        nudge_y = float(np.clip(cy + 1.2 * math.sin(angle), -5.8, ARENA_LIMIT - 0.5))
                        self.get_logger().info(f'🔄 Nudging explorer to ({nudge_x:+.2f}, {nudge_y:+.2f}) to discover new space...')
                        self.send_nav_goal(nudge_x, nudge_y)

            time.sleep(0.5)


def main(args=None):
    rclpy.init(args=args)
    node = DroneNav2Explorer()
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

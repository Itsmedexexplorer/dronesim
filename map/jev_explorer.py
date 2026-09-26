#!/usr/bin/env python3

import os
import sys
import time
import math
import signal
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_msgs.msg import Int32, Bool, String

ARENA_LIMIT = 7.1  # Comprehensive 15x15m indoor compound boundary (+/-7.5m outer walls)
UNKNOWN  = -1
FREE     = 0
OCCUPIED = 100

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
        self.sub_survivors = self.create_subscription(Int32, '/nidar/survivor_count', self.cb_survivor_count, 10)
        self.sub_complete = self.create_subscription(Bool, '/nidar/mission_complete', self.cb_mission_complete, 10)
        self.sub_land = self.create_subscription(Bool, '/nidar/land_command', self.cb_land_cmd, 10)
        self.sub_relaunch = self.create_subscription(Bool, '/nidar/relaunch_command', self.cb_relaunch_cmd, 10)
        self.relaunch_requested = False
        
        self.pub_land_cmd = self.create_publisher(Bool, '/nidar/land_command', 10)
        self.pub_complete = self.create_publisher(Bool, '/nidar/mission_complete', 10)
        self.pub_goal = self.create_publisher(PoseStamped, '/nidar/current_goal', 10)
        self.pub_status = self.create_publisher(String, '/nidar/jev_status', 10)

        self.active_goal_handle = None
        self.goal_status = None
        self.is_acting = False

        self.thread = threading.Thread(target=self.jev_core_loop, daemon=True)
        self.thread.start()

    def cb_survivor_count(self, msg: Int32): self.survivor_count = msg.data
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
            self.cancel_current_action()

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)
    
    def cb_pose(self, msg: PoseStamped):
        with self.pose_lock:
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
                    # Buffer away from nearby walls into free space
                    safe_c, safe_r = int(round((wx - ox) / res)), int(round((wy - oy) / res))
                    check_rad = 6
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
                        wx += 0.32 * math.cos(away_angle)
                        wy += 0.32 * math.sin(away_angle)
                    wx = float(np.clip(wx, -6.85, 6.85))
                    wy = float(np.clip(wy, -6.85, 6.85))
                    clusters.append((wx, wy, len(cluster)))
        return clusters

    def _act_navigate(self, x, y):
        self.get_logger().info(f'🧠 [JEV] ACT: Dispatching Nav2 Goal to ({x:.2f}, {y:.2f})')
        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = 1.50
        with self.pose_lock: cx, cy = self.drone_x, self.drone_y
        yaw = math.atan2(y - cy, x - cx)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        # Publish current goal for GCS visualizer
        self.pub_goal.publish(goal.pose)

        self.is_acting = True
        self.goal_status = "PENDING"
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warning('[JEV] ACT FAILED: Nav2 rejected the goal.')
            self.goal_status = "REJECTED"
            self.is_acting = False
            return
        self.active_goal_handle = handle
        res_future = handle.get_result_async()
        res_future.add_done_callback(self._goal_result_cb)

    def _goal_result_cb(self, future):
        res = future.result()
        if res.status == GoalStatus.STATUS_SUCCEEDED:
            self.goal_status = "SUCCEEDED"
        else:
            self.goal_status = f"FAILED_{res.status}"
        self.is_acting = False

    def cancel_current_action(self):
        if self.is_acting and self.active_goal_handle:
            self.get_logger().info('[JEV] Canceling current action...')
            self.active_goal_handle.cancel_goal_async()
            time.sleep(0.5)
        self.is_acting = False

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

        consecutive_failures = 0

        while rclpy.ok():
            time.sleep(0.8)
            
            # --- 1. OBSERVE ---
            with self.pose_lock:
                pos = (self.drone_x, self.drone_y)
            cov = self.get_coverage()
            frontiers = self.find_frontiers()
            self.memory['coverage'] = cov
            self.memory['frontiers'] = frontiers
            self.memory['position'] = pos
            self.publish_status(f"OBSERVE: Frontiers={len(frontiers)} | Cov={cov:.1f}%")
            
            self.get_logger().info(f'\n🧠 [JEV] --- COGNITIVE LOOP ---')
            self.get_logger().info(f'🧠 [JEV] OBSERVE: Pos=({pos[0]:.2f}, {pos[1]:.2f}), Coverage={cov:.1f}%, Frontiers={len(frontiers)}')
            
            # --- 2. QUESTION ---
            # Strictly require >= 95.0% arena exploration before returning (do NOT stop just because survivors are found!)
            return_needed = self.return_requested or (cov >= 95.0) or (cov >= 94.0 and len(frontiers) == 0)
            self.publish_status(f"QUESTION: Checking Return Condition (Cov: {cov:.1f}% / 95.0%)")
            
            if return_needed:
                self.get_logger().info(f'🎯 [JEV] DECIDE: Return to Base triggered (Coverage={cov:.1f}%, Survivors={self.survivor_count}/6, Manual={self.return_requested}).')
                
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

            # --- 3. THINK ---
            self.publish_status("THINK: Evaluating Frontier Heuristics & Obstacle Safety")
            best_action = None
            if len(frontiers) > 0:
                valid_options = []
                for fx, fy, sz in frontiers:
                    dist = math.hypot(fx - pos[0], fy - pos[1])
                    # Skip frontiers immediately under the drone footprint (< 0.35m)
                    if dist < 0.35:
                        continue

                    # Soft penalty for areas with recent action failures
                    fails = sum(1 for gx, gy in self.memory['failed_actions'] if math.hypot(fx-gx, fy-gy) < 1.0)
                    
                    # Soft penalty for areas already visited
                    visits = sum(1 for vx, vy in self.memory['visited_locations'] if math.hypot(fx-vx, fy-vy) < 1.0)

                    # Multi-objective heuristic score: distance, cluster size, failure & visit counts
                    score = dist - (0.18 * sz) + (fails * 8.0) + (visits * 12.0)
                    valid_options.append({'x': fx, 'y': fy, 'score': score, 'sz': sz})
                
                if valid_options:
                    valid_options.sort(key=lambda x: x['score'])
                    best_action = valid_options[0]
                    self.get_logger().info(f'🧠 [JEV] THINK: Selected optimal frontier at ({best_action["x"]:.2f}, {best_action["y"]:.2f}) [score={best_action["score"]:.2f}, size={best_action["sz"]}].')
            
            if best_action is None:
                # Exploratory wander behavior
                angle = (time.time() % (2 * math.pi))
                nx = float(np.clip(pos[0] + 1.5 * math.cos(angle), -6.2, 6.2))
                ny = float(np.clip(pos[1] + 1.5 * math.sin(angle), -6.2, 6.2))
                self.get_logger().info(f'🧠 [JEV] THINK: No frontiers nearby. Performing exploratory wander to ({nx:.2f}, {ny:.2f}).')
                best_action = {'x': nx, 'y': ny, 'sz': 0}

            # --- 4. DECIDE ---
            target_x, target_y = best_action['x'], best_action['y']
            self.publish_status(f"DECIDE: Selected target waypoint ({target_x:.2f}, {target_y:.2f})")
            self.get_logger().info(f'🧠 [JEV] DECIDE: Navigating to ({target_x:.2f}, {target_y:.2f}).')
            
            # --- 5. ACT ---
            self.publish_status(f"ACT: Navigating corridor to ({target_x:.2f}, {target_y:.2f})")
            self.memory['expected_outcome'] = (target_x, target_y)
            self._act_navigate(target_x, target_y)
            
            act_start_time = time.time()
            last_dist = math.hypot(target_x - pos[0], target_y - pos[1])
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
                if abs(dist - last_dist) < 0.08:
                    stuck_time += 0.4
                else:
                    stuck_time = 0.0
                    last_dist = dist
                    
                if dist < 0.40:
                    self.get_logger().info(f'🧠 [JEV] VERIFY: Reached frontier proximity ({dist:.2f}m). Smooth transition.')
                    self.cancel_current_action()
                    self.goal_status = "SUCCEEDED"
                    break
                    
                if stuck_time > 12.0 or (time.time() - act_start_time) > 40.0:
                    self.get_logger().warning(f'🧠 [JEV] VERIFY: Action taking too long ({stuck_time:.1f}s stalled). Aborting.')
                    self.cancel_current_action()
                    self.goal_status = "TIMEOUT"
                    break

            # --- 6. VERIFY & 7. REASSESS ---
            self.publish_status(f"VERIFY: Waypoint execution outcome: {self.goal_status}")
            self.get_logger().info(f'🧠 [JEV] VERIFY: Action finished with status: {self.goal_status}')
            
            if self.goal_status == "SUCCEEDED":
                self.get_logger().info(f'🧠 [JEV] REASSESS: Action succeeded. Updating visited memory.')
                self.memory['visited_locations'].append((target_x, target_y))
                consecutive_failures = 0
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

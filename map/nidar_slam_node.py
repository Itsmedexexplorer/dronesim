#!/usr/bin/env python3
"""
Nidar 2D SLAM & Occupancy Grid Mapping Node for Nav2
===================================================
Receives 360° LiDAR from Gazebo (/lidar) and drone pose (/mavros/local_position/pose).
Performs Bresenham raycasting to generate real-time 2D Occupancy Grid on /map.
Publishes nav_msgs/OccupancyGrid with TRANSIENT_LOCAL QoS for Nav2 Global Costmap.
"""

import os
import sys
import time
import math
import signal
import threading
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_active_node = None

def _clean_exit_handler(sig, frame):
    global _active_node
    if _active_node is not None:
        try:
            _active_node.save_map()
        except Exception:
            pass
    os._exit(0)

signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Bool

try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.laserscan_pb2 import LaserScan as GzLaserScan
except ImportError:
    from gz.transport import Node as GzNode
    from gz.msgs.laserscan_pb2 import LaserScan as GzLaserScan

# File save locations
SAVE_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVE_RAW   = os.path.join(SAVE_DIR, "drone_map_final.png")
SAVE_ANNOT = os.path.join(SAVE_DIR, "drone_map_annotated.png")
SAVE_PGM   = os.path.join(SAVE_DIR, "nidar_autonomous_map.pgm")
SAVE_YAML  = os.path.join(SAVE_DIR, "nidar_autonomous_map.yaml")

# Grid constants
GRID_RES   = 0.05       # 5cm cell resolution
GRID_SIZE  = 30.0       # 30m x 30m
GRID_N     = int(GRID_SIZE / GRID_RES) # 600x600 cells
ORIGIN_X   = -GRID_SIZE / 2.0
ORIGIN_Y   = -GRID_SIZE / 2.0

UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2


def world_to_cell(wx, wy):
    c = int(math.floor((wx - ORIGIN_X) / GRID_RES))
    r = int(math.floor((wy - ORIGIN_Y) / GRID_RES))
    if 0 <= c < GRID_N and 0 <= r < GRID_N:
        return c, r
    return None


def bresenham_cells(start, end):
    """Return every integer grid cell crossed by a ray, including both ends."""
    x0, y0 = start
    x1, y1 = end
    cells = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return cells
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy


def clip_free_ray_at_wall(prob_grid, start, end, occupied_threshold=65):
    """Clip free-space evidence before the first already established wall.

    This prevents a noisy or delayed long-range scan from carving a false mapped
    corridor through an obstacle that previous scans have consistently observed.
    Returns ``(endpoint, blocked)``.
    """
    cells = bresenham_cells(start, end)
    for index, (x, y) in enumerate(cells[1:], start=1):
        if prob_grid[y, x] > occupied_threshold:
            return cells[max(0, index - 1)], True
    return end, False


class NidarSlamNode(Node):
    def __init__(self):
        super().__init__('nidar_slam_node')
        global _active_node
        _active_node = self

        self.get_logger().info('Starting Nidar 2D SLAM & Occupancy Grid Node for Nav2...')

        # Dynamic Evidence / Probability Grid (50 = unknown, 0 = free, 100 = occupied)
        self.prob_grid = np.full((GRID_N, GRID_N), 50, dtype=np.int16)
        self.grid_lock = threading.Lock()

        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_roll = 0.0
        self.drone_pitch = 0.0
        self.drone_yaw = 0.0
        self.pose_lock = threading.Lock()

        # Conservative inverse-sensor model. A maximum-range return is useful
        # free-space evidence, but becomes unreliable at long range in narrow
        # doorways. Keep that evidence local and use a thin ray footprint.
        self.declare_parameter('lidar_topic', '/nidar/lidar')
        self.declare_parameter('trusted_free_range_m', 6.0)
        self.declare_parameter('free_ray_thickness_cells', 2)
        self.lidar_topic = str(self.get_parameter('lidar_topic').value)
        self.trusted_free_range_m = float(
            self.get_parameter('trusted_free_range_m').value)
        self.free_ray_thickness_cells = max(
            1, int(self.get_parameter('free_ray_thickness_cells').value))

        # Flight Trajectory & Survivor Tracking for Map Export
        self.flight_path = []
        self.confirmed_survivors = []
        self.stamped_survivors = set()
        self._last_save_time = time.time()

        # ROS 2 Subscribers
        self.sub_pose = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.cb_pose, 10
        )

        # Nav2 compatible map QoS
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.pub_map = self.create_publisher(OccupancyGrid, '/map', map_qos)

        # One canonical Gazebo scan source. Subscribing to aliases duplicated
        # evidence and made the same distant ray become "mapped" too quickly.
        self.gz_node = GzNode()
        self.gz_node.subscribe(GzLaserScan, self.lidar_topic, self.cb_lidar)
        self.get_logger().info(
            f'LiDAR mapping source: {self.lidar_topic}; trusted free range '
            f'{self.trusted_free_range_m:.1f} m; ray width '
            f'{self.free_ray_thickness_cells} cells')

        # Survivor Map Tagging Subscriber: logs detected survivors
        survivor_qos = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.sub_survivors = self.create_subscription(
            MarkerArray, '/nidar/survivor_markers', self.cb_survivor_markers, survivor_qos
        )

        # Completion & Landing Subscribers (triggers final map save)
        self.sub_complete = self.create_subscription(
            Bool, '/nidar/mission_complete', self.cb_save_trigger, 10
        )
        self.sub_land = self.create_subscription(
            Bool, '/nidar/land_command', self.cb_save_trigger, 10
        )

        # Map publishing timer (5Hz is ideal for Nav2 global costmap updates)
        self.timer_map = self.create_timer(0.20, self.publish_map)

        self._last_save_time = 0.0
        self._save_in_progress = False

    def cb_save_trigger(self, msg: Bool):
        now = time.time()
        if msg.data and (now - self._last_save_time > 20.0) and not self._save_in_progress:
            self._last_save_time = now
            self.get_logger().info('🏁 [AUTONOMOUS COMPLETION] Saving final 2D map files in background...')
            threading.Thread(target=self._async_save_worker, daemon=True).start()

    def _async_save_worker(self):
        self._save_in_progress = True
        try:
            self.save_map(silent=False)
        finally:
            self._save_in_progress = False

    def auto_save_map(self):
        # Save snapshot if flight path has developed
        now = time.time()
        if len(self.flight_path) > 10 and (now - self._last_save_time > 30.0) and not self._save_in_progress:
            self._last_save_time = now
            threading.Thread(target=self._async_save_worker, daemon=True).start()

    def cb_survivor_markers(self, msg: MarkerArray):
        with self.grid_lock:
            for m in msg.markers:
                if m.ns == 'survivors' and m.id % 10 == 0:  # Ground disc marker
                    pos_key = m.id // 10
                    for existing in self.confirmed_survivors:
                        if existing['id'] == pos_key:
                            existing.update(x=m.pose.position.x, y=m.pose.position.y)
                    if pos_key not in self.stamped_survivors:
                        self.stamped_survivors.add(pos_key)
                        sid = m.id // 10
                        self.confirmed_survivors.append({
                            'id': sid,
                            'x': m.pose.position.x,
                            'y': m.pose.position.y
                        })
                        self.get_logger().info(
                            f'🗺️ [2D MAP] Confirmed Solid Red Survivor #{sid} Marker at '
                            f'({m.pose.position.x:+.2f}, {m.pose.position.y:+.2f}) on /nidar/survivor_markers'
                        )

    def cb_pose(self, msg: PoseStamped):
        q = msg.pose.orientation

        # Roll (x-axis)
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        with self.pose_lock:
            self.drone_x = msg.pose.position.x
            self.drone_y = msg.pose.position.y
            self.drone_roll = roll
            self.drone_pitch = pitch
            self.drone_yaw = yaw
            if len(self.flight_path) == 0 or math.hypot(msg.pose.position.x - self.flight_path[-1][0], msg.pose.position.y - self.flight_path[-1][1]) > 0.15:
                self.flight_path.append((msg.pose.position.x, msg.pose.position.y))

    def cb_lidar(self, msg: GzLaserScan):
        ranges = list(msg.ranges)
        with self.pose_lock:
            cx, cy = self.drone_x, self.drone_y
            yaw = self.drone_yaw
            roll = self.drone_roll
            pitch = self.drone_pitch

        # Skip scan if drone is tilted (> 4.5 deg / 0.08 rad) to avoid projecting floor/ceiling hits into the 2D map
        if abs(roll) > 0.08 or abs(pitch) > 0.08:
            return

        p_drone = world_to_cell(cx, cy)
        if not p_drone:
            return

        angle_min = msg.angle_min
        angle_step = msg.angle_step
        max_r = min(msg.range_max, 10.0)

        with self.grid_lock:
            # Create a blank mask for cv2 drawing to optimize the raycasting
            # We just use OpenCV line drawing!
            c_drone = world_to_cell(cx, cy)
            if not c_drone: return
            x0, y0 = c_drone

            # Actually, to do probabilistic updates in bulk very fast:
            # Draw all free lines on a mask
            free_mask = np.zeros_like(self.prob_grid, dtype=np.uint8)
            wall_mask = np.zeros_like(self.prob_grid, dtype=np.uint8)

            for i, r in enumerate(ranges):
                if math.isinf(r) or math.isnan(r) or r <= msg.range_min:
                    continue
                ray_ang = (yaw + angle_min + i * angle_step + math.pi) % (2 * math.pi) - math.pi
                hit_wall = (r < max_r - 0.10)
                dist_ray = min(r, max_r)

                # A no-hit / maximum-range beam must not paint the whole arena.
                # It is still useful, but only inside a configurable trusted
                # radius around the actual drone pose.
                if not hit_wall:
                    dist_ray = min(dist_ray, self.trusted_free_range_m)

                # Draw free space only up to 0.15m before the wall to prevent erasing walls!
                free_dist = max(0.0, dist_ray - 0.15) if hit_wall else dist_ray
                fwx = cx + free_dist * math.cos(ray_ang)
                fwy = cy + free_dist * math.sin(ray_ang)
                c_free = world_to_cell(fwx, fwy)

                blocked_by_known_wall = False
                if c_free:
                    c_free, blocked_by_known_wall = clip_free_ray_at_wall(
                        self.prob_grid, (x0, y0), c_free)
                    cv2.line(
                        free_mask, (x0, y0), c_free, 1,
                        self.free_ray_thickness_cells)

                if hit_wall and not blocked_by_known_wall:
                    wx = cx + dist_ray * math.cos(ray_ang)
                    wy = cy + dist_ray * math.sin(ray_ang)
                    c_end = world_to_cell(wx, wy)
                    if c_end:
                        x1, y1 = c_end
                        cv2.circle(wall_mask, (x1, y1), 0, 30, -1)

            # Apply masks
            self.prob_grid = np.clip(self.prob_grid.astype(np.int16) - free_mask, 0, 100).astype(np.uint8)
            self.prob_grid = np.clip(self.prob_grid.astype(np.int16) + wall_mask, 0, 100).astype(np.uint8)

    def publish_map(self):
        with self.grid_lock:
            prob_copy = self.prob_grid.copy()

        # Free space must come from sensor evidence, including the entry gate.
        # Never erase a real obstruction just because a nominal route crosses it.

        now_msg = self.get_clock().now().to_msg()
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

        # Nav2 standard: -1 unknown (40..65), 0 free (<40), 100 occupied (>65)
        flat_data = np.full(GRID_N * GRID_N, -1, dtype=np.int8)
        raw = prob_copy.ravel()
        flat_data[raw < 40] = 0
        flat_data[raw > 65] = 100
        map_msg.data = flat_data.tolist()

        self.pub_map.publish(map_msg)

    def save_map(self, silent=False):
        try:
            with self.grid_lock:
                prob_copy = self.prob_grid.copy()
                survivors = list(self.confirmed_survivors)
            with self.pose_lock:
                path = list(self.flight_path)
                last_x = self.drone_x
                last_y = self.drone_y
                last_yaw = self.drone_yaw

            # Build the export solely from accumulated sensor evidence. Older
            # versions painted the helipad chute free during export, which made
            # an unvisited region look mapped and broke map provenance.
            rgb = np.full((GRID_N, GRID_N, 3), 160, dtype=np.uint8)
            rgb[prob_copy < 40] = [240, 240, 240]
            rgb[prob_copy > 65] = [25, 25, 25]

            # Paint confirmed humans in SOLID RED on raw map (no black dot!)
            for s in survivors:
                sx, sy = s['x'], s['y']
                for dx in np.arange(-0.45, 0.45 + 0.02, GRID_RES):
                    for dy in np.arange(-0.45, 0.45 + 0.02, GRID_RES):
                        if dx * dx + dy * dy <= 0.20:
                            cell = world_to_cell(sx + dx, sy + dy)
                            if cell:
                                rgb[cell[1], cell[0]] = [255, 0, 0]  # SOLID RED

            # Save Raw Map (drone_map_final.png)
            fig_r, ax_r = plt.subplots(figsize=(8, 8), dpi=150)
            ax_r.imshow(rgb, origin="lower")
            ax_r.axis("off")
            fig_r.tight_layout(pad=0)
            fig_r.savefig(SAVE_RAW, bbox_inches="tight", pad_inches=0)
            plt.close(fig_r)

            # 3. Save Annotated Map (drone_map_annotated.png)
            half = GRID_SIZE / 2.0
            fig_a, ax_a = plt.subplots(figsize=(12, 12), dpi=150)
            ax_a.imshow(rgb, origin="lower", extent=[-half, half, -half, half])

            # Coordinate Grid Lines
            for v in np.arange(-half, half + 1.0, 1.0):
                ax_a.axhline(v, color="gray", lw=0.3, alpha=0.35)
                ax_a.axvline(v, color="gray", lw=0.3, alpha=0.35)

            # Flight Path (Cyan/Blue)
            if len(path) > 1:
                px = [p[0] for p in path]
                py = [p[1] for p in path]
                ax_a.plot(px, py, color='#00aaff', lw=1.5, alpha=0.85, label='Flight Path')

            # Helipad Launch / Touchdown (-5.80, -8.80)
            ax_a.plot(-5.80, -8.80, 'gH', markersize=14, markeredgecolor='black', label='Helipad (Start & Touchdown)')

            # Solid Red Survivor Markers
            for s in survivors:
                sid, sx, sy = s['id'], s['x'], s['y']
                circle = plt.Circle((sx, sy), 0.45, color='#ff0000', alpha=0.95, zorder=7)
                ax_a.add_patch(circle)
                halo = plt.Circle((sx, sy), 0.70, color='#ff0000', fill=False, lw=2.0, linestyle='--', zorder=7)
                ax_a.add_patch(halo)
                ax_a.text(sx, sy + 0.60, f'SURVIVOR #{sid} [HUMAN]', color='#cc0000', fontsize=9, fontweight='bold', ha='center', zorder=8)

            # Drone Symbol: Directional Triangle + Square Frame for instant clarity
            sq_w = 0.48
            sq_half = sq_w / 2.0
            rect = plt.Rectangle((last_x - sq_half, last_y - sq_half), sq_w, sq_w, fill=False, edgecolor='#00bbee', lw=2.2, zorder=9, label='Drone Square Frame (0.48m)')
            ax_a.add_patch(rect)

            tip_x = last_x + 0.38 * math.cos(last_yaw)
            tip_y = last_y + 0.38 * math.sin(last_yaw)
            wl_x = last_x + 0.22 * math.cos(last_yaw + 2.4)
            wl_y = last_y + 0.22 * math.sin(last_yaw + 2.4)
            wr_x = last_x + 0.22 * math.cos(last_yaw - 2.4)
            wr_y = last_y + 0.22 * math.sin(last_yaw - 2.4)
            tri = plt.Polygon([[tip_x, tip_y], [wl_x, wl_y], [last_x, last_y], [wr_x, wr_y]], closed=True, facecolor='#00ffff', edgecolor='#0077aa', lw=1.8, zorder=10, label='Drone Heading (Triangle)')
            ax_a.add_patch(tri)

            # Stats Calculation
            total_cells = GRID_N * GRID_N
            free_cells = int(np.sum(prob_copy < 40))
            cov_pct = (free_cells / float(total_cells)) * 100.0
            free_m2 = free_cells * (GRID_RES ** 2)

            ax_a.set_xlabel("X (meters, ENU frame)", fontsize=11)
            ax_a.set_ylabel("Y (meters, ENU frame)", fontsize=11)
            ax_a.set_title(
                f"NIDAR Autonomous Drone — Final 2D Search & Occupancy Map\n"
                f"Survivors Located: {len(survivors)}  |  Explored Free Area: {free_m2:.1f} m²  |  "
                f"Coverage: {cov_pct:.1f}%  |  Saved: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                fontsize=11, fontweight="bold"
            )

            legend_handles = [
                mpatches.Patch(color=[0.10, 0.10, 0.10], label="Obstacle / Wall"),
                mpatches.Patch(color=[0.94, 0.94, 0.94], label="Free Space"),
                mpatches.Patch(color=[0.63, 0.63, 0.63], label="Unexplored"),
                mpatches.Patch(color='#00aaff', label="Flight Path"),
                mpatches.Patch(color='#00cc44', label="Helipad (Start / Touchdown)"),
                mpatches.Patch(color='#ff0000', label=f"Survivors Tagged ({len(survivors)})"),
            ]
            ax_a.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.92)

            fig_a.tight_layout()
            fig_a.savefig(SAVE_ANNOT, bbox_inches="tight")
            plt.close(fig_a)

            # 4. Save Standard ROS 2 Nav2 Map (.pgm and .yaml)
            pgm_data = np.full((GRID_N, GRID_N), 205, dtype=np.uint8)
            pgm_data[prob_copy < 40] = 254
            pgm_data[prob_copy > 65] = 0

            with open(SAVE_PGM, 'wb') as f:
                f.write(f"P5\n{GRID_N} {GRID_N}\n255\n".encode())
                f.write(np.flipud(pgm_data).tobytes())

            yaml_content = (
                f"image: nidar_autonomous_map.pgm\n"
                f"mode: trinary\n"
                f"resolution: {GRID_RES}\n"
                f"origin: [{ORIGIN_X:.1f}, {ORIGIN_Y:.1f}, 0.0]\n"
                f"negate: 0\n"
                f"occupied_thresh: 0.65\n"
                f"free_thresh: 0.25\n"
            )
            with open(SAVE_YAML, 'w') as f:
                f.write(yaml_content)

            if not silent:
                self.get_logger().info(
                    f'✅ [MAPS SAVED] 2D Map files successfully saved:\n'
                    f'   1. Raw 2D Map:       {SAVE_RAW}\n'
                    f'   2. Annotated 2D Map: {SAVE_ANNOT}\n'
                    f'   3. ROS 2 Map PGM:    {SAVE_PGM}\n'
                    f'   4. ROS 2 Map YAML:   {SAVE_YAML}'
                )
        except Exception as e:
            self.get_logger().error(f'Failed to save map files: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = NidarSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_map()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

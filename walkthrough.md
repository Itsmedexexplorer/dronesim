# MAVROS ROS 2 Autonomous Drone Navigation Architecture

## 1. Overview of the MAVROS Framework Migration

To provide future hardware compatibility with real autopilots (Pixhawk, Cube, PX4, ArduPilot), the autonomous exploration framework has been upgraded to run on **standard MAVROS ROS 2 topics and services**.

```
[Gazebo Simulation (X3 Multi-rotor)]
       ▲
       │  (Actuates /X3/gazebo/command/twist, reads /world/nidar_world/pose/info)
       ▼
[mavros_bridge.py]  (MAVROS ROS 2 Bridge Node)
       ▲
       │  Standard ROS 2 MAVROS Interfaces:
       │    • /mavros/state                     (mavros_msgs/msg/State)
       │    • /mavros/local_position/pose       (geometry_msgs/msg/PoseStamped)
       │    • /mavros/local_position/velocity   (geometry_msgs/msg/TwistStamped)
       │    • /mavros/cmd/arming                (mavros_msgs/srv/CommandBool)
       │    • /mavros/set_mode                  (mavros_msgs/srv/SetMode)
       │    • /mavros/setpoint_raw/local        (mavros_msgs/msg/PositionTarget)
       ▼
[nidar_mavros_navigator.py]  (Autonomous Offboard Navigator Node)
       ├── 2D Occupancy Grid Mapping & Raycasting (/nidar/lidar)
       ├── Frontier Detection & Cluster Centroids
       ├── A* Collision-Free Path Planning with Safety Dilation
       ├── Active 360° LiDAR Obstacle Repulsion
       └── Automated Mission State Machine:
             1. Warmup (Send setpoints before OFFBOARD mode)
             2. Arming & Mode Switch (OFFBOARD)
             3. Takeoff Climb (0.40 - 0.70 m/s upward to 1.5m)
             4. Systematic Corridor & Room Exploration
             5. Return-To-Home along explored highway
             6. Automated Landing on Helipad
```

---

## 2. Fixes Applied: Resolving Non-Takeoff Issue

1. **Gazebo Twist Topic Corrected**:
   - `mavros_bridge.py` was initially advertising `/X3/cmd_vel`. However, Gazebo's `MulticopterVelocityControl` plugin in `nidar_drone_world.sdf` listens on:
     ```
     /X3/gazebo/command/twist
     ```
   - Updated the bridge to publish directly to `/X3/gazebo/command/twist`.
2. **Real-Time Gazebo Ground-Truth Pose Stream**:
   - Added a dedicated background reader in `mavros_bridge.py` that listens to `/world/nidar_world/pose/info` and `/world/quadcopter/pose/info`.
   - Streams $(x, y, z, \text{yaw})$ to `/mavros/local_position/pose` at 50Hz.
3. **Takeoff Vertical Thrust**:
   - Updated the takeoff state in `nidar_mavros_navigator.py` to command upward velocity ($v_z \in [0.40, 0.70]\text{ m/s}$) until altitude reaches $\ge 1.35\text{m}$.
4. **Correct LiDAR Topic**:
   - Pointed LiDAR subscription to `/nidar/lidar` matching the world SDF.

---

## 3. Standard MAVROS Topics & Services Implemented

| Topic / Service Name | Message / Service Type | Direction | Purpose |
| :--- | :--- | :--- | :--- |
| `/mavros/state` | `mavros_msgs/msg/State` | Published | Reports FCU connection, arming state, and flight mode (`OFFBOARD`, `MANUAL`) |
| `/mavros/local_position/pose` | `geometry_msgs/msg/PoseStamped` | Published | High-frequency 3D position and orientation quaternion |
| `/mavros/local_position/velocity_local` | `geometry_msgs/msg/TwistStamped` | Published | Current body/local linear and angular velocities |
| `/mavros/cmd/arming` | `mavros_msgs/srv/CommandBool` | Service | Service to arm or disarm drone motors |
| `/mavros/set_mode` | `mavros_msgs/srv/SetMode` | Service | Service to engage `OFFBOARD` autonomous flight mode |
| `/mavros/setpoint_raw/local` | `mavros_msgs/msg/PositionTarget` | Subscribed | Accepts body/local velocity and yaw-rate setpoints |
| `/mavros/setpoint_position/local` | `geometry_msgs/msg/PoseStamped` | Subscribed | Accepts 3D target coordinates |

---

## 4. Launching the System

### Terminal 1: Start Gazebo Simulation
```bash
gz sim -v 4 -r /home/dexter/Downloads/Drone-simulation-2-main/worlds/nidar_drone_world.sdf
```

### Terminal 2: Launch MAVROS Autonomous Navigation
```bash
cd /home/dexter/Downloads/Drone-simulation-2-main
./run_mavros_autonomous.sh
```

---

## 5. Fixes for Flight Path Twisting & Corridor Ping-Ponging

From analysis of the screen recording (`Screen Recording 2026-09-19 115959.mp4`), two specific behavioral anomalies were identified and resolved:

### 1. Eliminating Yaw Twisting ("Twisting path again and again")
- **Root Cause**: The drone checked waypoint reach using a discrete radius (`dist <= 0.38m`). If the drone cut a corner or was slightly displaced sideways by obstacle deflection (`dist = 0.40m`), the waypoint remained active *behind* the drone. The heading error jumped to $140^\circ\text{--}180^\circ$, cutting forward speed (`abs(yaw_err) > 0.35` forced $v_x = 0$) and causing the drone to stop and spin backwards.
- **Fix - Lookahead Segment Projection**:
  - Implemented continuous segment projection:
    $$\text{proj} = \frac{(\mathbf{p}_{\text{drone}} - \mathbf{w}_i) \cdot (\mathbf{w}_{i+1} - \mathbf{w}_i)}{\|\mathbf{w}_{i+1} - \mathbf{w}_i\|^2}$$
  - If $\text{proj} \ge 0.70$ or the drone is closer to waypoint $i+1$ than waypoint $i$, the tracker immediately advances to the forward waypoint without turning back.
  - Added a **$4.5^\circ$ ($0.08\text{ rad}$) yaw deadband** to eliminate high-frequency heading chatter.
  - Allowed **smooth coordinated turning**; forward velocity is scaled by $\max(0.20, \cos(\text{yaw\_err}))$ rather than abruptly cut to 0.

### 2. Eliminating Wall Bouncing & Corridor Trapping
- **Root Cause**: In corridors ($1.0\text{--}1.2\text{m}$ wide), the threshold `min_side < 0.45` triggered bang-bang sideways pushes ($v_y = \pm 0.22\text{ m/s}$), causing the drone to ping-pong laterally between parallel walls. When it slowed down, the watchdog aborted after only $1.5\text{s}$, and re-selected the same corridor dead-end.
- **Fix - Proportional Corridor Centering**:
  - When lateral walls are detected within corridor range ($d_l < 1.15\text{m}$ and $d_r < 1.15\text{m}$), the lateral velocity is set proportionally to the difference:
    $$v_y = \text{clip}\big(0.32 \times (d_l - d_r),\, -0.18,\, +0.18\big)$$
    At the centerline ($d_l = d_r$), $v_y = 0$. Small displacements generate smooth restoring nudges toward the center.
  - Extended stuck watchdog from $1.5\text{s}$ to $3.2\text{s}$.
  - When stuck recovery activates, a heavy $+150$ penalty is painted onto the `visited_heatmap` around $(c_x, c_y)$, so the frontier cluster cost function penalizes the dead-end and selects unexplored open corridors.

---

## 6. Fix for `ExternalShutdownException` on Terminal Termination

### Root Cause
When pressing `Ctrl+C` or sending `SIGINT`/`SIGTERM` to `run_mavros_autonomous.sh`, ROS 2 (`rclpy`) terminates its internal executor wait loop by raising:
```python
rclpy.executors.ExternalShutdownException
```
Because `mavros_bridge.py` and `nidar_mavros_navigator.py` only caught `KeyboardInterrupt`, Python considered `ExternalShutdownException` an unhandled error and dumped the traceback to the terminal. Furthermore, calling `rclpy.shutdown()` when already shut down externally caused secondary errors.

### Fix
1. In both `mavros_bridge.py` and `nidar_mavros_navigator.py`:
   ```python
   from rclpy.executors import ExternalShutdownException

   try:
       rclpy.spin(node) # or executor.spin()
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
   ```
2. In `run_mavros_autonomous.sh`, unset the trap inside `cleanup()` (`trap - INT TERM EXIT`) to prevent duplicate shutdown messages.

---

## 7. Real-Time Map Visualization (RViz2 vs Foxglove)

### Tool Comparison: Which is Preferred?

| Feature | RViz2 (Recommended) | Foxglove Studio |
| :--- | :--- | :--- |
| **Setup Required** | **Zero Setup**: Pre-installed natively with ROS 2 in WSL | Requires installing Foxglove app & WebSocket bridge |
| **Data Protocol** | Native ROS 2 DDS (`nav_msgs`, `sensor_msgs`, `tf2`) | Requires `foxglove_bridge` or ROS 2 native connection |
| **Performance** | Native OpenGL via WSLg with zero latency | High GPU usage (Electron/browser) |
| **Map Rendering** | Native 2D Occupancy Grid (`/map`), TF frames, paths | 2D/3D panels with customizable widgets |
| **Conclusion** | **Winner for immediate out-of-the-box visualization** | Great for cloud/remote web monitoring |

### Published ROS 2 Topics for Live Visualization

| Topic | Type | Frame | Purpose |
| :--- | :--- | :--- | :--- |
| `/map` | `nav_msgs/msg/OccupancyGrid` | `map` | 2D Occupancy Grid (5cm resolution, 600x600 cells) |
| `/tf` | `tf2_msgs/msg/TFMessage` | `map` $\to$ `base_link` | 3D drone position & yaw rotation |
| `/nidar/path` | `nav_msgs/msg/Path` | `map` | Real-time planned A* frontier/highway path |
| `/nidar/lidar_scan` | `sensor_msgs/msg/LaserScan` | `base_link` | 360° LiDAR rays reflecting off walls |
| `/nidar/frontiers` | `visualization_msgs/msg/MarkerArray` | `map` | Yellow sphere markers of unexplored frontier targets |
| `/mavros/local_position/pose` | `geometry_msgs/msg/PoseStamped` | `map` | 3D local position & orientation arrow |

---

---

## 9. Fix for Open Corridor Pacing & Decision Hesitation

### Root Causes Identified
From analyzing your RViz map screenshot (`media_1789803772712.png`), the drone explored the main hallways and stopped at an intersection near the bottom-right room, unable to enter rooms:
1. **A\* Goal Obstacle Overlap (No Goal Snapping)**:
   - When frontier clusters form inside rooms, their centroids often lie close to interior walls or in downsampled blocks containing wall cells (`occ_mask[g_r, g_c] == True`).
   - A* previously required the exact goal cell to be un-dilated. When the goal cell touched a wall block, A* considered the goal an impassable obstacle and aborted (`found = False`), returning an empty path for every frontier cluster.
2. **Doorway Over-Dilation**:
   - 3 rounds of 8-connectivity dilation on a 10cm grid inflated walls by $\approx 0.30\text{--}0.42\text{m}$ on each side. Two opposing doorposts ($0.70\text{m}$ door opening) overlapped and closed the doorway in the A* graph, preventing the planner from finding routes into rooms.
3. **The 0.40m Proximity Skip & Open Corridor Wandering**:
   - When the drone navigated to a doorway, the cluster just inside was $\le 0.40\text{m}$ away.
   - `if math.hypot(fx - cx, fy - cy) > 0.40:` skipped this frontier cluster because it was within 0.40m, but the LiDAR hadn't yet pierced inside.
   - With all paths failing, the loop dropped to a blind fallback: `if min_fwd >= 0.65: send_vel(0.35, 0, 0, 0)`. The drone cruised down whatever open corridor it faced, bounced off the far wall, turned, and paced back and forth in the open corridor endlessly.
4. **Unbounded Heatmap Penalty**:
   - `visited_heatmap` incremented at 50Hz without an upper bound, adding excessive virtual distance penalty ($+50\text{m}$) to frequently traversed intersections.

---

### Fixes Applied

1. **Spiral Goal Snapping in A\***:
   If a frontier centroid is inside a wall dilation zone, A* performs an outward spiral search (up to 14 cells / $1.4\text{m}$) to snap the target to the nearest free, navigable cell directly in front of the frontier.
2. **1-Step Hard Dilation + Soft Cost Layer**:
   Hard obstacle buffer reduced to 1 round ($10\text{cm}$), keeping all doorways open. A 2nd ring adds a soft $+1.5$ step cost, guiding the drone to corridor centers when open while permitting door traversal.
3. **Frontier Sweep for Close Clusters ($\le 0.45\text{m}$)**:
   When adjacent to a frontier ($\le 0.45\text{m}$), the drone rotates directly to face the frontier cluster centroid, instantly casting 360° LiDAR rays into the room to clear unexplored space.
4. **Global Unexplored Sector Fallback**:
   If discrete cluster paths ever fail, the drone computes the median position of ALL remaining `UNKNOWN` cells across the arena and routes toward the least-explored quadrant. If completely surrounded, it rotates 360° in place to uncover room entrances.
5. **Heatmap Penalty Bounded**:
   Capped `visited_heatmap` increments at 40 and limited the maximum cost penalty to 5.0m, preventing valid rooms from being blackballed.

---

## 10. Verification

All tests in WSL passed with clean A* doorway traversal:
- Commit: `cbccbc9` on `main`

---

## 11. How to Run with Live Real-Time Mapping

### Option A: Autonomous Flight with Integrated RViz2 (One Command)
```bash
cd /home/dexter/Downloads/Drone-simulation-2-main
./run_mavros_autonomous.sh --rviz
```

### Option B: Run in Separate Terminals

**Terminal 1: Gazebo Simulation**
```bash
gz sim -v 4 -r /home/dexter/Downloads/Drone-simulation-2-main/worlds/nidar_drone_world.sdf
```

**Terminal 2: MAVROS Autonomous Navigation**
```bash
cd /home/dexter/Downloads/Drone-simulation-2-main
./run_mavros_autonomous.sh
```

**Terminal 3: Live RViz2 SLAM Visualizer**
```bash
cd /home/dexter/Downloads/Drone-simulation-2-main
./launch_rviz.sh
```

---

## 12. Stuck Recovery Constraints & Clean Terminal Shutdown

### 1. Root Cause: Corner Oscillations Defeating Old Watchdogs
- When the drone encountered an obstacle or sharp corner, the forward waypoint command drove it forward while `apply_lidar_obstacle_avoidance` pushed it back ($v_x = -0.18\text{ m/s}$).
- This formed a **limit-cycle oscillation** of $\pm 0.15\text{m}$.
- The previous watchdog condition `if math.hypot(cx - last_prog_pos[0], cy - last_prog_pos[1]) > 0.15` was continuously reset by the jitter, preventing escape triggers.
- Even if triggered, the old recovery lasted only 0.36s ($10\text{cm}$ travel) and immediately replanned to the identical corner.

### 2. The Solution: Sliding-Window Watchdog + 3-Stage Recovery + Blacklist Constraints

1. **Sliding-Window Displacement Watchdog**:
   - Maintains a historical trajectory buffer: `self.pose_history = deque(maxlen=250)` storing `(timestamp, x, y)`.
   - On every iteration, inspects positions recorded $3.2\text{--}5.5\text{s}$ prior. If net displacement is $< 0.35\text{m}$, the drone is confirmed stuck regardless of high-frequency jitter.

2. **Stage 1: Dynamic Blacklist Constraint**:
   - The stuck coordinates $(c_x, c_y)$ are added to `self.stuck_blacklist = [(bx, by, expiry_time)]` with an active timeout of 50 seconds.
   - Any frontier cluster within $1.3\text{m}$ of $(b_x, b_y)$ is rejected from selection.
   - Any planned A* path passing within $0.6\text{m}$ of $(b_x, b_y)$ is discarded, forcing the drone into a completely different corridor or room.
   - The local heatmap is penalized by $+25$ in a 7x7 cell region.

3. **Stage 2: Decisive Active Reverse (1.5 seconds / 45 steps)**:
   - Actively commands $v_x \le -0.28\text{ m/s}$ with lateral wall deflection, pulling the drone $\approx 0.5\text{m}$ back away from the corner into open space.

4. **Stage 3: 360° LiDAR Clearing Sweep (2.5 seconds / 75 steps)**:
   - Rotates at $\omega_z = 1.4\text{ rad/s}$ to sweep LiDAR rays into side openings, populate newly visible free space, and identify alternate frontier directions.

5. **Clean Terminal Shutdown on Ctrl+C**:
   - Registered `signal.signal(signal.SIGINT, _clean_exit_handler)` and `SIGTERM` in both `nidar_mavros_navigator.py` and `mavros_bridge.py` using `os._exit(0)`.
   - Updated `run_mavros_autonomous.sh` `cleanup()` trap to terminate child processes with zero Python tracebacks or lingering background nodes.

---

## 13. Room Exit / Doorway Centering Fix & Nav2 Architecture Comparison

### 1. Root Cause: Why the Drone Was Unable to Exit the Room
From analyzing the telemetry logs and uploaded screenshot (`media_1789888521666.png`):
- **Doorway Posts Mistaken for Head-On Walls**:
  - Doorway openings are $\approx 0.70\text{--}0.80\text{m}$ wide.
  - The previous collision detector computed `min_fwd = min(d_f, d_fl, d_fr)` using polar sectors extending to $\pm 75^\circ$ ($150^\circ$ hemisphere).
  - When the drone faced the open doorway, the door posts on either side at $\pm 35^\circ$ were within $0.35\text{m}$.
  - The detector saw `min_fwd = 0.35m < 0.38m` and triggered an emergency reverse (`vx = -0.18`) and sideways slide, actively repelling the drone away from the door every time it attempted to exit.
- **Path Blacklisting Blocking Escapes**:
  - `passes_blacklisted` rejected any path passing within $0.6\text{m}$ of the stuck location.
  - Since the doorway was adjacent to the stuck position inside the room, ALL escape paths through the door were disqualified, forcing the drone to remain trapped.

### 2. The Solution: Cartesian Flight Corridor & Dual-Wall Doorway Centering
1. **Cartesian Flight Corridor**:
   - Replaced the $150^\circ$ polar cone with a true **Cartesian rectangular corridor** ($x_b > 0$, $|y_b| \le 0.20\text{m}$).
   - Lateral door posts ($|y_b| \ge 0.25\text{m}$) are no longer classified as head-on collisions, leaving `fwd_clearance` open ($> 1.0\text{m}$) through the doorway.
2. **Doorway & Narrow-Passage Centering Controller**:
   - When both side posts are detected within $0.80\text{m}$ ($d_l, d_r \le 0.80\text{m}$) while forward corridor is open:
     $$v_y = \text{clip}(0.35 \times (d_l - d_r), -0.20, 0.20)$$
     $$v_x = \max(0.28, v_x)$$
   - Guides the drone straight down the centerline of the door at a steady $0.28\text{ m/s}$.
3. **Blacklist Scope Confined to Frontier Goals**:
   - Blacklisting is strictly restricted to rejecting dead-end frontier goals, ensuring the drone is never prohibited from flying along a path out of a room.

---

### 3. Can We Use Nav2 for the Drone?
**Yes!** In Section 14 below, we have now fully installed and deployed the official ROS 2 Navigation2 stack for the drone.

---

## 14. Official ROS 2 Navigation2 (Nav2) Stack Integration

The complete official ROS 2 Navigation2 stack has been installed and integrated into the simulation repository, establishing full future compatibility with standard robotics tooling, controllers, costmaps, and planners.

### Architecture

```
[Gazebo Simulation (X3 Multi-rotor)]
       ▲
       │  (Actuates /X3/gazebo/command/twist, reads /world/nidar_world/pose/info)
       ▼
[mavros_bridge.py]
       ▲
       │  /mavros/local_position/pose & /mavros/setpoint_raw/local
       ▼
[drone_tf_odom_publisher.py]
       ├── TF Tree: map -> odom -> base_link -> lidar_link
       ├── /odom (nav_msgs/Odometry)
       └── /scan (sensor_msgs/LaserScan, Gazebo bridge)
       ▼
[nidar_slam_node.py]
       └── /map (nav_msgs/OccupancyGrid, 5cm resolution, TRANSIENT_LOCAL)
       ▼
[ROS 2 Nav2 Navigation Stack] (nav2_bringup navigation_launch.py)
       ├── Global Costmap 2D (Static map + 0.28m door inflation layer)
       ├── Local Costmap 2D (4m x 4m rolling obstacle window from /scan)
       ├── Planner Server (SmacPlanner2D / NavfnPlanner)
       ├── Controller Server (MPPI / Regulated Pure Pursuit)
       ├── Behavior Tree Navigator (navigate_to_pose action server)
       └── Recoveries (Spin, BackUp, Wait)
       ▼
[drone_nav2_coordinator.py]
       ├── Manages Offboard State: Warmup -> Arm -> Takeoff to Z=1.5m
       ├── Injects closed-loop Altitude Hold PID (vz = 1.4 * (1.5 - z))
       └── Translates Nav2 2D /cmd_vel into 3D MAVROS PositionTarget
       ▲
       │ (Nav2 Goals /navigate_to_pose)
[drone_nav2_explorer.py]
       ├── Frontier Cluster Extraction on /map
       ├── Goal Dispatcher to Nav2 Action Server
       └── Target Coverage Monitor (>= 95% triggers Return-To-Home & Land)
```

### New Nav2 Components Created

| File | Path | Description |
| :--- | :--- | :--- |
| **TF & Odom Bridge** | [`map/drone_tf_odom_publisher.py`](file:///home/dexter/Downloads/Drone-simulation-2-main/map/drone_tf_odom_publisher.py) | Broadcasts full `map -> odom -> base_link -> lidar_link` TF tree, `/odom`, and `/scan` |
| **SLAM Map Node** | [`map/nidar_slam_node.py`](file:///home/dexter/Downloads/Drone-simulation-2-main/map/nidar_slam_node.py) | Real-time 2D SLAM node publishing `/map` with Transient Local QoS |
| **Nav2 Parameters** | [`config/nav2_drone_params.yaml`](file:///home/dexter/Downloads/Drone-simulation-2-main/config/nav2_drone_params.yaml) | Tuned Nav2 parameters ($r = 0.22\text{m}$ footprint, $0.28\text{m}$ doorway inflation) |
| **Flight Coordinator** | [`map/drone_nav2_coordinator.py`](file:///home/dexter/Downloads/Drone-simulation-2-main/map/drone_nav2_coordinator.py) | Executes takeoff, $1.5\text{m}$ altitude PID, and translates `/cmd_vel` to MAVROS |
| **Nav2 Explorer** | [`map/drone_nav2_explorer.py`](file:///home/dexter/Downloads/Drone-simulation-2-main/map/drone_nav2_explorer.py) | Dispatches autonomous frontier goals via Nav2 `navigate_to_pose` action |
| **Nav2 Launcher** | [`run_nav2_autonomous.sh`](file:///home/dexter/Downloads/Drone-simulation-2-main/run_nav2_autonomous.sh) | Single-command launcher for all Nav2, MAVROS, SLAM, and coordinator processes |

---

### How to Run

#### Option 1: Run Full Nav2 Autonomous Stack
```bash
# Terminal 1: Gazebo Simulation
gz sim -v 4 -r /home/dexter/Downloads/Drone-simulation-2-main/worlds/nidar_drone_world.sdf

# Terminal 2: ROS 2 Nav2 Autonomous Drone Navigation
cd /home/dexter/Downloads/Drone-simulation-2-main
./run_nav2_autonomous.sh --rviz
```

#### Option 2: Run Standalone Python Navigator (Fallback)
```bash
cd /home/dexter/Downloads/Drone-simulation-2-main
./run_mavros_autonomous.sh --rviz
```

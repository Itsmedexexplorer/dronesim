# 🚁 NIDAR AirMouse — Autonomous Indoor Search, Mapping & Survivor Tagging AI Drone

[![ROS 2 Lyrical](https://img.shields.io/badge/ROS_2-Lyrical-blue.svg)](https://docs.ros.org/)
[![Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic_10.x-orange.svg)](https://gazebosim.org/)
[![Nav2 Navigation](https://img.shields.io/badge/Nav2-DWB_Local_Planner-green.svg)](https://navigation.ros.org/)
[![MAVROS 2](https://img.shields.io/badge/MAVROS-Offboard_Control-purple.svg)](https://github.com/mavlink/mavros)
[![OpenCV Vision](https://img.shields.io/badge/OpenCV-Survivor_Detection-red.svg)](https://opencv.org/)

An enterprise-grade, fully autonomous GPS-denied quadcopter system engineered for the **NIDAR AirMouse Challenge Brief (Track 1 — Drone Innovation: Search, Mapping & Survivor Localization in Covered Damaged Facilities)**.

---

## 🌟 Mission Overview & Core Capabilities

The NIDAR AirMouse drone executes end-to-end autonomous search and rescue missions inside a complex, covered, GPS-denied $15\text{ m} \times 15\text{ m}$ indoor maze:
1. **Autonomous Takeoff & Staging**: Lifts off from the designated Launch Helipad at $(X=-5.80, Y=-8.80)$, climbs to cruise altitude ($1.50\text{ m}$), and penetrates the compound through the 2.0-meter South Gate.
2. **Real-Time 2D SLAM Mapping (`/map`)**: Processes 360° LiDAR rays via Bresenham raycasting at 5 cm resolution, continuously constructing an occupancy grid of rooms, corridors, and debris.
3. **Autonomous Frontier Exploration**: Dynamically clusters boundary frontiers, plans optimal exploration paths using ROS 2 Nav2, and pushes targets away from obstacles with a $0.40\text{ m}$ standoff buffer.
4. **Active 360° LiDAR Reactive Collision Shield**: Continuous 30 Hz proximity monitor computes an inverse-distance repulsive velocity field. If any wall or corner approaches $< 0.35\text{ m}$, the coordinator triggers an instant **hard brake and pushback**, completely preventing collisions.
5. **Human / Survivor Detection & 2D Map Tagging**: Onboard computer vision node (`nidar_survivor_detector.py`) inspects the 5MP WDR camera feed (`/nidar/camera`) in real-time, segments International Rescue Orange suits and fluorescent yellow safety vests, projects pixel centroids to map coordinates, and tags them with persistent 3D pins and floating text labels (`Survivor #N (X, Y)`).
6. **Permanent 2D Map Etching**: Confirmed survivor coordinates are automatically stamped as occupied obstacles directly onto the `/map` SLAM grid.
7. **Evidence-Bounded Search & Precision Touchdown**: The explorer has no survivor-count or time cap. After 10 minutes it prioritizes remaining unknown LiDAR gaps, then audits mapped blocks and camera viewpoints. It returns only after search confidence remains above 97%, or when the operator requests RTH.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Sim["Gazebo Sim (Harmonic)"]
        Physics["1000 Hz Drone Physics (MulticopterVelocityControl)"]
        Sensors["Sensors: 360° LiDAR, IMX335 5MP WDR Camera, IMX219 Stereo"]
        World["Covered Compound (15x15m, 8ft Net, 6 Standing Mannequins)"]
    end

    subgraph Bridges["Bridge & Hardware Abstraction Layer"]
        MB["mavros_bridge.py\n(/mavros/local_position/pose, /mavros/state)"]
        TF["drone_tf_odom_publisher.py\n(TF: map->odom->base_link, /scan)"]
    end

    subgraph Navigation["ROS 2 Nav2 Navigation Stack"]
        Costmap["Global & Local Costmaps (r=0.28m, inf=0.60m)"]
        Planner["Navfn Global Planner + DWB Local Controller"]
        BT["BT Navigator (/navigate_to_pose)"]
    end

    subgraph AutonomousCore["Autonomous Mission & Vision Stack"]
        SLAM["nidar_slam_node.py\n(2D Occupancy Grid on /map + Survivor Stamping)"]
        Vision["nidar_survivor_detector.py\n(HSV Color Segmentation, Ray Casting, 3D RViz Pins)"]
        Explorer["drone_nav2_explorer.py\n(Frontier Exploration, Goal Standoff, Exit Sequencer)"]
        Coordinator["drone_nav2_coordinator.py\n(Reactive LiDAR Collision Shield, Z=1.5m PID, Landing)"]
    end

    subgraph GCS["Ground Control & Visualization"]
        RViz["RViz2 / GCS Dashboard\n(/map, /scan, /nidar/survivor_markers, /nidar/path)"]
    end

    Sim --> Bridges
    Bridges --> Navigation
    Bridges --> AutonomousCore
    AutonomousCore --> Navigation
    Navigation --> Coordinator
    Coordinator --> MB
    MB --> Sim
    AutonomousCore --> RViz
    Navigation --> RViz
```

---

## 🎯 Autonomous Survivor Search & Localization (6 Survivors)

The facility contains **6 realistic standing survivor mannequins** ($1.75\text{ m}$ tall, $0.48\text{ m}$ diameter) equipped with high-visibility International Rescue Orange suits and fluorescent yellow reflective safety vests, placed in unknown rooms throughout the grid.

- **Blind Autonomous Search**: The drone operates with zero prior knowledge of survivor locations or count and discovers as many survivors as its onboard sensors can confirm.
- **Multimodal Localization**:
  - **360° LiDAR**: Since the mannequins stand $1.75\text{ m}$ tall, the drone's horizontal LiDAR at flight altitude ($1.50\text{ m}$) strikes their torso and head directly, drawing them geometrically as obstacles on the 2D SLAM grid (`/map`).
  - **Onboard Vision**: The 5MP WDR camera detects the vibrant rescue orange and fluorescent yellow colors at eye level, estimating range and relative bearing.
- **Real-Time Tagging**: Once localized, each survivor is tagged on the map with persistent 3D pins and floating coordinate labels (`Survivor #N [X, Y]`), and permanently stamped into the 2D occupancy grid.

### Survivor confirmation safety

The default `target_mode=simulation` uses an orange/yellow signature for the cylindrical Gazebo mannequins. `target_mode=person` enables SSD/YOLO for human imagery. The simulation signature is a heuristic, not a trained mannequin detector, and its score is not a calibrated probability. Appearance alone never creates a map marker.

- RGB proposals are evaluated against the aligned 640×480 RGB-D stream, which also supplies the HUD. RGB/depth sensor timestamps must agree within 75 ms.
- A candidate must have repeated distinct close visual observations with a freestanding RGB-D depth discontinuity. LiDAR projections can request inspection but cannot commit markers.
- Candidates farther than 2.0 m publish `/nidar/inspection_target` and are not added to the survivor map.
- `jev_explorer.py` selects a collision-checked 1.15–1.60 m stand-off pose, approaches the candidate, and holds briefly for close-range confirmation.
- Confirmed coordinates use the median of close observations and compensate depth from the visible torso surface to the object centre.

Run the perception regression and integration tests with:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m unittest -v test_perception_safety.py test_ssd_object_detection.py
python3 test_tri_modal_pipeline.py
```

---

## 📡 Sensor Suite & ROS 2 Topic Architecture

| Hardware Component | Sensor Type | Primary Topic | Rate | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Slamtec 360° LiDAR** | `gpu_lidar` | `/scan` & `/nidar/lidar` | 10 Hz | 2D SLAM mapping, Nav2 costmaps, reactive collision shield |
| **Waveshare IMX335 5MP WDR** | `camera` | `/nidar/camera` | 30 FPS | Primary survivor color segmentation & ray-cast localization |
| **Waveshare IMX219-83 Stereo** | `rgbd_camera` | `/nidar/stereo_camera/image` | 30 FPS | Auxiliary forward vision & 3D depth perception |
| **MAVROS 3D Drone Pose** | `geometry_msgs/PoseStamped` | `/mavros/local_position/pose` | 50 Hz | High-precision visual-inertial odometry & TF broadcast |
| **MAVROS Velocity Actuation**| `mavros_msgs/PositionTarget`| `/mavros/setpoint_raw/local` | 30 Hz | Body-frame velocity vector injection ($v_x, v_y, v_z, \omega_z$) |
| **2D SLAM Occupancy Grid** | `nav_msgs/OccupancyGrid` | `/map` | 5 Hz | Real-time 2D arena floor plan with survivor stamps |
| **Tagged Survivor Markers** | `visualization_msgs/MarkerArray` | `/nidar/survivor_markers` | 2 Hz | 3D colored pins and floating coordinate text in RViz |
| **Survivor Count** | `std_msgs/Int32` | `/nidar/survivor_count` | Event | Open-ended total of uniquely confirmed survivors |
| **Mission Completion** | `std_msgs/Bool` | `/nidar/mission_complete` | Event | Triggers return-to-home and gate exit sequence |
| **Emergency Landing Trigger** | `std_msgs/Bool` | `/nidar/land_command` | Event | Triggers coordinator precision touchdown over helipad |

---

## 📁 Repository Structure

```
├── config/
│   ├── nav2_drone_params.yaml    # Tuned Nav2 parameters (DWB controller, critics, costmaps)
│   ├── nidar_map.rviz            # Comprehensive RViz2 configuration (Map, TF, Path, Survivor Pins)
│   └── slam_toolbox_params.yaml  # Auxiliary SLAM Toolbox configuration
├── map/
│   ├── drone_nav2_coordinator.py # Active LiDAR collision shield, Z=1.5m PID, MAVROS velocity bridge
│   ├── drone_nav2_explorer.py    # Frontier exploration planner, standoff buffer, autonomous exit sequencer
│   ├── drone_tf_odom_publisher.py# TF tree broadcaster (map -> odom -> base_link -> lidar_link)
│   ├── mavros_bridge.py          # MAVROS state, pose, and offboard control bridge
│   ├── nidar_slam_node.py        # 2D Occupancy Grid SLAM node with survivor stamping
│   ├── nidar_survivor_detector.py# Real-time HSV color segmentation & 2D/3D map tagging node
│   ├── nidar_command_center.py   # Standalone manual HUD command center
│   └── wait_for_nav2.py          # Synchronous readiness check for Nav2 Action Server
├── worlds/
│   ├── nidar_drone_world.sdf     # 15x15m covered compound with 6 standing survivors & obstacles
│   └── quadcopter_stable.sdf     # Base Gazebo world template
├── run_sim.sh                    # Launches Gazebo Harmonic simulation world
├── run_nav2_autonomous.sh        # Single-command launcher for full autonomous stack + RViz
├── teleop_drone.py               # Lightweight keyboard teleoperation utility
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- **OS**: Ubuntu 22.04 / 24.04 (native or WSL 2)
- **Simulator**: [Gazebo Harmonic](https://gazebosim.org/) (`gz-sim 10.x`)
- **ROS Version**: ROS 2 (Lyrical / Humble) with Nav2 & MAVROS packages installed:
  ```bash
  sudo apt update
  sudo apt install -y ros-lyrical-navigation2 ros-lyrical-nav2-bringup ros-lyrical-mavros ros-lyrical-visualization-msgs
  ```

---

### Step 1: Launch the Simulation World
In your first terminal:
```bash
cd ~/drone-sim
./run_sim.sh
```
*Gazebo Harmonic will launch with the $15\text{ m} \times 15\text{ m}$ covered compound and all 6 standing survivor mannequins. Click the **Play ▶** button at the bottom-left if paused.*

---

### Step 2: Launch the Full Autonomous Mission
In a second terminal:
```bash
cd ~/drone-sim
./run_nav2_autonomous.sh --rviz
```

The script orchestrates the entire operational pipeline automatically:
1. Starts the `mavros_bridge` and locks local position telemetry.
2. Broadcasts the TF transform tree (`map` $\to$ `odom` $\to$ `base_link` $\to$ `lidar_link`).
3. Starts the 2D SLAM node generating `/map` at 5 cm resolution.
4. Brings up the ROS 2 Nav2 stack (DWB local planner and Navfn global planner).
5. Opens RViz2 loaded with the preconfigured NIDAR monitoring layout.
6. Launches `nidar_survivor_detector.py` for real-time camera inspection.
7. Launches `drone_nav2_explorer.py` to begin frontier exploration.
8. Initiates autonomous takeoff to $1.50\text{ m}$ and engages the active collision shield.

---

## 🛡️ Collision Avoidance & Safety Features

* **Tuned DWB Critics**: `BaseObstacle.scale: 8.0` and `ObstacleFootprint.scale: 8.0` strongly penalize paths near corners, preventing clipping.
* **Corridor Centering**: Inflation radii of $0.55\text{ m}$ (local) and $0.60\text{ m}$ (global) guide trajectories along corridor centers.
* **Active 360° Reactive Collision Shield**:
  - Distance $< 0.52\text{ m}$: Calculates real-time repulsive pushback away from walls.
  - Distance $< 0.35\text{ m}$: Overrides Nav2 with an emergency brake and active reverse impulse (`🛡️ [COLLISION SHIELD] Wall at ...m! Repulsive pushback`).
* **Standoff Buffer**: Frontier goals are verified against the occupancy grid; targets within $0.35\text{ m}$ of a wall are automatically nudged $0.40\text{ m}$ into clear space.

---

## 🏁 Mission Completion & Autonomous Exit

Ten minutes is a strategy switch, not a timeout:

1. From 0–10 minutes, normal frontier mapping remains primary.
2. At 90% coverage or 10 minutes, remaining unknown/black LiDAR gaps become the
   highest priority. The drone approaches from safe free space and faces the
   unknown centroid before its normal 360° scan.
3. It then audits mapped 2 m regions that are not visible from a successful
   360-degree RGB-D sweep. Coverage uses camera range plus occupancy-grid
   line-of-sight, so one sweep can cover adjacent open regions but never sees
   through a wall or unknown space.
4. Search confidence combines LiDAR coverage (70%), line-of-sight camera
   coverage (20%), and resolved inspections (10%). It remains capped below 97%
   while a queued inspection or meaningful frontier is unresolved; tiny
   residual edge noise does not keep a completed mission alive indefinitely.
5. Confidence must remain above 97% for 15 seconds before autonomous RTH.

During return:
1. The explorer cancels frontier tasks and transitions to `EXITING_MAZE`.
2. It plans a path through the South Gate and flies to the Launch Helipad.
3. It emits `/nidar/land_command`, causing a soft touchdown, motor cutoff, and disarm:
   ```
   🏁 [MISSION COMPLETE] Drone safely landed on helipad! Motors disarmed.
   ```

---

## 📜 License
Developed for the NIDAR AirMouse Challenge Brief. Distributed under the MIT License.

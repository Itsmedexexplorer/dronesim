#!/usr/bin/env bash
# ==============================================================================
# Nidar Autonomous Drone - ROS 2 Nav2 Navigation Stack Launcher
# ==============================================================================

set -e

export PATH="/opt/ros/jazzy/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

# NVIDIA CUDA & WSL2 GPU Hardware Acceleration (RTX 3050 offload)
if [ -d "/usr/lib/wsl/lib" ]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
    export LIBGL_ALWAYS_SOFTWARE=0
    export MESA_D3D12_DEFAULT_ADAPTER_NAME="NVIDIA"
    export __NV_PRIME_RENDER_OFFLOAD=1
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
fi

# Source ROS 2 base (provides all Nav2 and MAVROS packages)
source /opt/ros/jazzy/setup.bash

LAUNCH_RVIZ=false
LAUNCH_GCS=false
for arg in "$@"; do
    if [[ "$arg" == "--rviz" ]]; then LAUNCH_RVIZ=true; fi
    if [[ "$arg" == "--gcs" ]]; then LAUNCH_GCS=true; fi
done

echo "=========================================================="
echo "   NIDAR AUTONOMOUS DRONE - ROS 2 NAV2 ARCHITECTURE       "
echo "=========================================================="

# Clean up any stale background processes from earlier runs
pkill -9 -f jev_explorer.py 2>/dev/null || true
pkill -9 -f drone_nav2_coordinator.py 2>/dev/null || true
pkill -9 -f nidar_vision_tracker_node.py 2>/dev/null || true
pkill -9 -f nidar_ssd_detector.py 2>/dev/null || true
pkill -9 -f nidar_survivor_detector.py 2>/dev/null || true
pkill -9 -f drone_tf_odom_publisher.py 2>/dev/null || true
pkill -9 -f nidar_slam_node.py 2>/dev/null || true
pkill -9 -f slam_toolbox 2>/dev/null || true
pkill -9 -f mavros_bridge.py 2>/dev/null || true
pkill -9 -f nav2_bringup 2>/dev/null || true
pkill -9 -f collision_monitor 2>/dev/null || true
pkill -9 -f controller_server 2>/dev/null || true
pkill -9 -f planner_server 2>/dev/null || true
pkill -9 -f bt_navigator 2>/dev/null || true
pkill -9 -f rviz2 2>/dev/null || true
pkill -9 -f nidar_gcs.py 2>/dev/null || true
pkill -9 -f mission_dashboard.py 2>/dev/null || true
sleep 1

PIDS=()

cleanup() {
    trap - INT TERM EXIT
    echo ""
    echo "Shutting down all Nav2 & MAVROS processes cleanly..."
    for pid in "${PIDS[@]}"; do
        kill -9 $pid 2>/dev/null || true
    done
    pkill -9 -f jev_explorer.py 2>/dev/null || true
    pkill -9 -f drone_nav2_coordinator.py 2>/dev/null || true
    pkill -9 -f nidar_vision_tracker_node.py 2>/dev/null || true
    pkill -9 -f nidar_survivor_detector.py 2>/dev/null || true
    pkill -9 -f drone_tf_odom_publisher.py 2>/dev/null || true
    pkill -9 -f nidar_slam_node.py 2>/dev/null || true
    pkill -9 -f slam_toolbox 2>/dev/null || true
    pkill -9 -f mavros_bridge.py 2>/dev/null || true
    pkill -9 -f nav2_bringup 2>/dev/null || true
    pkill -9 -f collision_monitor 2>/dev/null || true
    pkill -9 -f controller_server 2>/dev/null || true
    pkill -9 -f planner_server 2>/dev/null || true
    pkill -9 -f bt_navigator 2>/dev/null || true
    pkill -9 -f rviz2 2>/dev/null || true
    pkill -9 -f nidar_gcs.py 2>/dev/null || true
    pkill -9 -f mission_dashboard.py 2>/dev/null || true
    echo "All processes terminated cleanly."
    exit 0
}

trap cleanup INT TERM EXIT

# 1. Start MAVROS Bridge
echo "  1. Starting MAVROS Bridge Node..."
python3 /home/dexter/Downloads/Drone-simulation-2-main/map/mavros_bridge.py &
PIDS+=($!)
sleep 2

# 2. Start Drone TF & Odometry Bridge
echo "  2. Starting Drone TF & Odometry Bridge (/odom, /scan, TF tree)..."
python3 /home/dexter/Downloads/Drone-simulation-2-main/map/drone_tf_odom_publisher.py &
PIDS+=($!)
sleep 1

# 3. Start 2D SLAM Node (/map)
echo "  3. Starting 2D SLAM Node (nidar_slam_node.py /map)..."
python3 /home/dexter/Downloads/Drone-simulation-2-main/map/nidar_slam_node.py &
PIDS+=($!)
sleep 1

# 4. Start Nav2 Navigation Stack (DWB Controller + NavfnPlanner)
echo "  4. Launching ROS 2 Nav2 Stack (DWB Local Planner, NavfnPlanner)..."
ros2 launch nav2_bringup navigation_launch.py \
    params_file:=/home/dexter/Downloads/Drone-simulation-2-main/config/nav2_drone_params.yaml \
    use_sim_time:=False \
    autostart:=True \
    use_composition:=False >/tmp/nav2.log 2>&1 &
PIDS+=($!)

# Actively wait for Action Server to become online
python3 /home/dexter/Downloads/Drone-simulation-2-main/map/wait_for_nav2.py

# 5. Optional RViz2 Visualizer
if [ "$LAUNCH_RVIZ" = true ]; then
    echo "  • Launching RViz2 Visualizer in background..."
    mkdir -p "$HOME/.rviz2"
    export QT_QPA_PLATFORM=xcb
    export OGRE_RTT_MODE=Copy
    export LIBGL_ALWAYS_INDIRECT=0
    rviz2 -d /home/dexter/Downloads/Drone-simulation-2-main/config/nidar_map.rviz >/dev/null 2>&1 &
    PIDS+=($!)
    sleep 1
fi

# 6. Start Autonomous Survivor Detector & Map Tagger (Vision Tracker with Spatial Duplicate Suppression)
echo "  5. Starting Autonomous Survivor Perception & Spatial Tracker Node..."
python3 /home/dexter/Downloads/Drone-simulation-2-main/map/nidar_vision_tracker_node.py &
PIDS+=($!)
sleep 1

# 7. Start Autonomous Frontier Explorer
echo "  6. Starting Autonomous Frontier Explorer Agent..."
python3 /home/dexter/Downloads/Drone-simulation-2-main/map/color_map_publisher.py &
python3 /home/dexter/Downloads/Drone-simulation-2-main/map/jev_explorer.py &
PIDS+=($!)
sleep 1

# 8. Start Mission Live HUD & Statistics Dashboard
echo "  7. Starting Mission Live HUD & Statistics Dashboard..."
python3 /home/dexter/Downloads/Drone-simulation-2-main/map/mission_dashboard.py &
PIDS+=($!)
sleep 1

# 9. Optional Ground Control Station (GCS) GUI
if [ "$LAUNCH_GCS" = true ]; then
    echo "  • Launching Mission Ground Control Station (GCS) Dashboard..."
    export QT_QPA_PLATFORM=xcb
    python3 /home/dexter/Downloads/Drone-simulation-2-main/map/nidar_gcs.py >/tmp/nidar_gcs.log 2>&1 &
    PIDS+=($!)
    sleep 1
fi

# 10. Start Drone Nav2 Flight Coordinator in Foreground
echo "  8. Starting Drone Nav2 Flight Coordinator (Takeoff, Z=1.5m PID, MAVROS)..."
python3 /home/dexter/Downloads/Drone-simulation-2-main/map/drone_nav2_coordinator.py

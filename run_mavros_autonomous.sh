#!/usr/bin/env bash
# ==============================================================================
# Nidar Drone MAVROS Autonomous Navigation Launcher
# ==============================================================================

set -e

export PATH="/opt/ros/jazzy/bin:/home/priyansh/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

# Source ROS 2 base (provides native MAVROS and Geographic messages)
source /opt/ros/jazzy/setup.bash

LAUNCH_RVIZ=false
if [[ "$1" == "--rviz" ]]; then
    LAUNCH_RVIZ=true
fi

echo "=========================================================="
echo "   NIDAR AUTONOMOUS DRONE - MAVROS ROS 2 ARCHITECTURE    "
echo "=========================================================="
echo "  1. Starting MAVROS Bridge Node (/mavros topics)..."

python3 /home/dexter/Downloads/Drone-simulation-2-main/map/mavros_bridge.py &
BRIDGE_PID=$!
RVIZ_PID=""

cleanup() {
    trap - INT TERM EXIT
    echo ""
    echo "Shutting down MAVROS processes cleanly..."
    if [ -n "$RVIZ_PID" ]; then
        kill -9 $RVIZ_PID 2>/dev/null || true
    fi
    kill -9 $BRIDGE_PID 2>/dev/null || true
    pkill -9 -f nidar_mavros_navigator.py 2>/dev/null || true
    pkill -9 -f mavros_bridge.py 2>/dev/null || true
    echo "MAVROS processes terminated cleanly."
    exit 0
}

trap cleanup INT TERM EXIT

sleep 2

if [ "$LAUNCH_RVIZ" = true ]; then
    echo "  • Launching RViz2 Visualizer in background..."
    mkdir -p "$HOME/.rviz2"
    export QT_QPA_PLATFORM=xcb
    export OGRE_RTT_MODE=Copy
    export LIBGL_ALWAYS_INDIRECT=0
    rviz2 -d /home/dexter/Downloads/Drone-simulation-2-main/config/nidar_map.rviz >/dev/null 2>&1 &
    RVIZ_PID=$!
    sleep 2
fi

echo "  2. Starting MAVROS Autonomous Navigator Node..."
python3 /home/dexter/Downloads/Drone-simulation-2-main/map/nidar_mavros_navigator.py

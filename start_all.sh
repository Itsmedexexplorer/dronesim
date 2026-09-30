#!/usr/bin/env bash
# ==============================================================================
# NIDAR Autonomous Drone - Unified One-Click Launch Script
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

cleanup_all() {
    echo ""
    echo "Terminating all NIDAR processes..."
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f jev_explorer.py 2>/dev/null || true
    pkill -9 -f drone_nav2_coordinator.py 2>/dev/null || true
    pkill -9 -f nidar_vision_tracker_node.py 2>/dev/null || true
    pkill -9 -f nidar_ssd_detector.py 2>/dev/null || true
    pkill -9 -f nidar_survivor_detector.py 2>/dev/null || true
    pkill -9 -f drone_tf_odom_publisher.py 2>/dev/null || true
    pkill -9 -f nidar_slam_node.py 2>/dev/null || true
    pkill -9 -f mavros_bridge.py 2>/dev/null || true
    pkill -9 -f nav2_bringup 2>/dev/null || true
    pkill -9 -f nidar_gcs.py 2>/dev/null || true
    pkill -9 -f mission_dashboard.py 2>/dev/null || true
    echo "Done."
    exit 0
}

trap cleanup_all INT TERM EXIT

echo "=========================================================="
echo "   🚁 NIDAR AIRMOUSE - UNIFIED AUTONOMOUS MISSION LAUNCH   "
echo "=========================================================="

# 1. Start Gazebo simulation in background
echo "1. Starting Gazebo Simulation World..."
./run_sim.sh &
GZ_PID=$!

# Wait for Gazebo to initialize physics and topics
echo "Waiting 4s for Gazebo to initialize..."
sleep 4

# 2. Launch full autonomous stack + Apple-designed GCS
echo "2. Launching Autonomous Nav2 Stack, JEV Explorer & GCS..."
./run_nav2_autonomous.sh --gcs

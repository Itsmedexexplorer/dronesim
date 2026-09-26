#!/usr/bin/env bash
# ==============================================================================
# Nidar Autonomous Drone - Gazebo Simulation World Launcher
# ==============================================================================

set -e

WORLD_FILE="/home/dexter/Downloads/Drone-simulation-2-main/worlds/nidar_drone_world.sdf"

# Clean up any lingering gz sim processes
pkill -9 -f "gz sim" 2>/dev/null || true

echo "=========================================================="
echo "   NIDAR AUTONOMOUS DRONE - GAZEBO SIMULATION ENGINE      "
echo "=========================================================="
echo "🌍 Loading World: $WORLD_FILE"
# Ensure Gazebo physics is unpaused and Multicopter controller enabled
(
    for i in $(seq 1 8); do
        sleep 1
        gz service -s /world/nidar_world/control --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean --timeout 1500 --req 'pause: false' 2>/dev/null || true
        gz topic -t /X3/enable --msgtype gz.msgs.Boolean -p 'data: true' 2>/dev/null || true
    done
) &

gz sim -v 4 -r "$WORLD_FILE"

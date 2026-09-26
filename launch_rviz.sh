#!/usr/bin/env bash
# ==============================================================================
# Nidar Drone Real-Time Map Visualizer (RViz2)
# ==============================================================================

set -e

export PATH="/opt/ros/jazzy/bin:/home/priyansh/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

source /opt/ros/jazzy/setup.bash

# Ensure ~/.rviz2 directory exists
mkdir -p "$HOME/.rviz2"

# WSLg Compatibility: Force X11/xcb mode for Qt & Ogre
# (Fixes "rviz::RenderSystem: error creating render window: RenderingAPIException")
export QT_QPA_PLATFORM=xcb
export OGRE_RTT_MODE=Copy
export LIBGL_ALWAYS_INDIRECT=0

echo "=========================================================="
echo "      NIDAR DRONE 2D SLAM REAL-TIME MAP VISUALIZER        "
echo "=========================================================="
echo "  • Fixed Frame:      map"
echo "  • Occupancy Grid:   /map"
echo "  • Drone Pose:       /mavros/local_position/pose"
echo "  • Planned Path:     /nidar/path"
echo "  • 360° LiDAR Scan:  /nidar/lidar_scan"
echo "  • Frontier Targets: /nidar/frontiers"
echo "  • 3D Drone Model:   /nidar/drone_marker"
echo "=========================================================="

rviz2 -d /home/dexter/Downloads/Drone-simulation-2-main/config/nidar_map.rviz "$@"

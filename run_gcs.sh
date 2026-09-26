#!/usr/bin/env bash
# ==============================================================================
# Nidar Autonomous Drone - Ground Control Station (GCS) Launcher
# ==============================================================================

set -e

source /opt/ros/jazzy/setup.bash
export QT_QPA_PLATFORM=xcb

echo "=========================================================="
echo "   NIDAR AUTONOMOUS DRONE - GROUND CONTROL STATION (GCS)  "
echo "=========================================================="
echo "🚀 Launching PyQt5 Mission Control Dashboard..."

python3 /home/dexter/Downloads/Drone-simulation-2-main/map/nidar_gcs.py

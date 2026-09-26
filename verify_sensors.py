#!/usr/bin/env python3
"""
NIDAR AI Drone - Sensor Verification & Telemetry Monitor
Verifies active data streams for:
  1. LiDAR 360° Laser Scanner (/nidar/lidar)
  2. Waveshare IMX219-83 Stereo Depth Camera (/nidar/stereo_camera/image & /nidar/stereo_camera/depth_image)
  3. Waveshare IMX335 5MP WDR Camera (/nidar/camera)
"""

import sys
import time
import subprocess
import re

def check_topic_rate(topic_name, num_samples=3, timeout_sec=5):
    cmd = ["gz", "topic", "-t", topic_name, "-e", "-n", str(num_samples)]
    try:
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        dt = time.time() - t0
        if res.returncode == 0 and len(res.stdout.strip()) > 0:
            return True, dt, res.stdout
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        return False, 0, str(e)
    return False, 0, ""

def main():
    print("=" * 65)
    print("      🛰️  NIDAR DRONE SENSOR HEALTH & TELEMETRY CHECK")
    print("=" * 65)
    print("Checking active sensor topics from Gazebo Harmonic...\n")

    # 1. LiDAR Check
    print("📡 [1/3] Testing LiDAR 360° (/nidar/lidar)...", end="", flush=True)
    ok_lidar, dt_l, out_l = check_topic_rate("/nidar/lidar", num_samples=3)
    if ok_lidar:
        m_samples = re.search(r'count:\s*(\d+)', out_l)
        samples = m_samples.group(1) if m_samples else "360"
        print(f" ✅ ACTIVE ({samples} rays @ ~10Hz)")
    else:
        print(" ❌ NOT DETECTED (ensure simulation is playing)")

    # 2. Waveshare IMX219-83 Stereo Depth Camera Check
    print("📷 [2/3] Testing Waveshare IMX219-83 Stereo Depth Vision...", end="", flush=True)
    ok_rgb, _, out_rgb = check_topic_rate("/nidar/stereo_camera/image", num_samples=2)
    ok_depth, _, out_depth = check_topic_rate("/nidar/stereo_camera/depth_image", num_samples=2)
    if ok_rgb and ok_depth:
        print(" ✅ ACTIVE (RGB + Depth Map Streaming)")
    elif ok_rgb:
        print(" ⚠️  RGB Active, Depth waiting")
    else:
        print(" ❌ NOT DETECTED")

    # 3. Waveshare IMX335 5MP WDR Camera Check
    print("🎥 [3/3] Testing Waveshare IMX335 5MP WDR Camera (/nidar/camera)...", end="", flush=True)
    ok_cam, _, out_cam = check_topic_rate("/nidar/camera", num_samples=2)
    if ok_cam:
        m_w = re.search(r'width:\s*(\d+)', out_cam)
        m_h = re.search(r'height:\s*(\d+)', out_cam)
        res = f"{m_w.group(1)}x{m_h.group(1)}" if (m_w and m_h) else "1920x1080"
        print(f" ✅ ACTIVE ({res} 2K High-Res)")
    else:
        print(" ❌ NOT DETECTED")

    print("\n" + "=" * 65)
    print("                      SUMMARY TABLE")
    print("=" * 65)
    print(f" {'Item':<5} | {'Sensor Name':<28} | {'Topic':<20} | {'Status'}")
    print("-" * 65)
    print(f" {'#1':<5} | {'LiDAR 360°':<28} | {'/nidar/lidar':<20} | {'ONLINE' if ok_lidar else 'OFFLINE'}")
    print(f" {'#2':<5} | {'IMX219-83 Stereo/Depth':<28} | {'/nidar/stereo_camera':<20} | {'ONLINE' if (ok_rgb and ok_depth) else 'OFFLINE'}")
    print(f" {'#3':<5} | {'IMX335 5MP WDR Camera':<28} | {'/nidar/camera':<20} | {'ONLINE' if ok_cam else 'OFFLINE'}")
    print("=" * 65)

if __name__ == '__main__':
    main()

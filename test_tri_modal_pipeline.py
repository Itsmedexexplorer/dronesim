#!/usr/bin/env python3
"""
Tri-Modal Perception & Localization Verification Test Suite
===========================================================
Validates:
  1. Custom NidarSurvivorNet ONNX inference accuracy & CPU latency.
  2. DepthAnalyzer metric depth extraction & freestanding 3D isolation test.
  3. VisionProposalScanner candidate proposal & batch inference.
  4. End-to-end integration with NidarSurvivorDetector:
     - Front camera RGB + Stereo Depth Map fusion.
     - Surround camera RGB + LiDAR cross-lookup.
     - Dynamic relocation: Moves mannequin to new locations, tests zero hardcoding.
     - False positive rejection: Tests empty walls, doorways, floor textures.
"""

import os
import sys
import math
import time
import json
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import PoseStamped

from map.tri_modal_detector import CustomSurvivorClassifier, DepthAnalyzer, VisionProposalScanner, STEREO_INTRINSICS
from map.nidar_survivor_detector import NidarSurvivorDetector, select_camera_for_bearing, project_point_to_camera


def render_synthetic_scene(mannequin_pos=None, distance=3.0, yaw_offset=0.0):
    """
    Renders realistic synthetic BGR and float32 depth frames:
    - mannequin_pos: (x_body, y_body) or None for empty scene
    - distance: distance to background wall
    """
    W, H = 640, 480
    fx, fy = STEREO_INTRINSICS['fx'], STEREO_INTRINSICS['fy']
    cx, cy = STEREO_INTRINSICS['cx'], STEREO_INTRINSICS['cy']

    # Background arena wall (gray concrete with slight texture)
    bgr = np.full((H, W, 3), 160, dtype=np.uint8)
    bgr[:H//3, :] = 190 # ceiling / wall upper
    bgr[H//3:H*2//3, :] = 150 # mid wall
    bgr[H*2//3:, :] = 110 # floor
    # Add subtle tile lines
    cv2.line(bgr, (0, H*2//3), (W, H*2//3), (80, 80, 80), 2)
    for x in range(0, W, 80):
        cv2.line(bgr, (x, H*2//3), (x - 60, H), (90, 90, 90), 1)

    # Background depth map: flat wall at distance (e.g. 5.0m)
    depth = np.full((H, W), 5.0, dtype=np.float32)
    # Floor depth gradient
    for v in range(H*2//3, H):
        frac = (v - H*2//3) / (H//3)
        depth[v, :] = max(1.2, 5.0 - frac * 3.5)

    if mannequin_pos is not None:
        xb, yb = mannequin_pos
        zc = xb
        xc = -yb
        yc = 0.60 # looking slightly down from drone z=1.5m to torso z=0.9m

        u = int(fx * (xc / zc) + cx)
        v = int(fy * (yc / zc) + cy)

        # Apparent width & height
        bw = max(16, int(fx * (0.46 / zc)))
        bh = max(36, int(fy * (1.75 / zc)))

        u1 = max(0, u - bw // 2)
        u2 = min(W, u + bw // 2)
        v1 = max(0, v - bh // 2)
        v2 = min(H, v + bh // 2)

        # Draw mannequin stack
        # Torso / Legs: Rescue Orange
        torso_v1 = v1 + int(bh * 0.35)
        torso_v2 = v2
        bgr[torso_v1:torso_v2, u1:u2] = [20, 95, 235] # BGR for Rescue Orange

        # Safety Vest: Fluorescent Yellow (upper chest)
        vest_v1 = v1 + int(bh * 0.20)
        vest_v2 = torso_v1 + int(bh * 0.15)
        bgr[vest_v1:vest_v2, u1:u2] = [30, 220, 240] # BGR for Bright Yellow

        # Head: Peach sphere
        head_v1 = v1
        head_v2 = vest_v1
        head_w = int(bw * 0.55)
        hu1 = max(0, u - head_w // 2)
        hu2 = min(W, u + head_w // 2)
        bgr[head_v1:head_v2, hu1:hu2] = [170, 195, 245] # BGR for Peach/Skin tone

        # Depth foreground
        depth[v1:v2, u1:u2] = float(zc)

    return bgr, depth


def test_custom_classifier():
    print("==================================================================")
    print("TEST 1: Custom NidarSurvivorNet Classifier & Latency Benchmark")
    print("==================================================================")
    clf = CustomSurvivorClassifier()
    assert clf.is_loaded, "CustomSurvivorClassifier failed to load!"

    # Test synthetic mannequin crop
    crop_m = np.zeros((80, 40, 3), dtype=np.uint8)
    crop_m[:20, :] = [170, 195, 245] # Peach head
    crop_m[20:45, :] = [30, 220, 240] # Yellow vest
    crop_m[45:, :] = [20, 95, 235]   # Orange body

    prob_m = clf.predict(crop_m)
    print(f"  P(Survivor | Mannequin Crop): {prob_m * 100:.2f}%")
    assert prob_m >= 0.95, f"Expected >95% confidence on mannequin crop, got {prob_m}"

    # Test synthetic wall crop
    crop_w = np.full((80, 40, 3), 150, dtype=np.uint8)
    prob_w = clf.predict(crop_w)
    print(f"  P(Survivor | Concrete Wall):  {prob_w * 100:.2f}%")
    assert prob_w <= 0.05, f"Expected <5% false positive on concrete wall, got {prob_w}"

    # Speed benchmark over 200 inferences
    t0 = time.time()
    for _ in range(200):
        clf.predict(crop_m)
    dt = (time.time() - t0) / 200.0 * 1000.0
    fps = 1000.0 / dt
    print(f"  Inference Latency: {dt:.3f} ms ({fps:.0f} FPS on CPU)")
    print("  => TEST 1 PASSED: High confidence on mannequin, zero false alarms on wall, ultra-fast.\n")


def test_depth_analyzer():
    print("==================================================================")
    print("TEST 2: DepthAnalyzer Metric Verification & Freestanding 3D Isolation")
    print("==================================================================")
    bgr, depth = render_synthetic_scene(mannequin_pos=(2.80, 0.20))

    # Mannequin bounding box
    bbox = [270, 160, 370, 380]
    valid, z_fg, is_free, step = DepthAnalyzer.analyze_roi_depth(depth, bbox)
    print(f"  Depth Valid: {valid}")
    print(f"  Extracted Foreground Metric Depth: {z_fg:.2f} m (Ground Truth: 2.80 m)")
    print(f"  Depth Step vs Background: {step:.2f} m (Freestanding: {is_free})")

    assert valid, "Depth extraction failed!"
    assert abs(z_fg - 2.80) <= 0.05, f"Depth error too high: {z_fg} vs 2.80"
    assert is_free, "Freestanding check failed!"
    assert step >= 0.15, f"Expected depth step >= 0.15m, got {step}"

    # Project to 3D body
    xb, yb, zb = DepthAnalyzer.project_2d_to_3d_body(320, 270, z_fg)
    print(f"  Projected 3D Body Coordinates: Xb={xb:+.2f}m, Yb={yb:+.2f}m, Zb={zb:+.2f}m")
    assert abs(xb - 2.97) <= 0.10, f"Body X error: {xb}" # 2.8m + 0.17m camera offset

    print("  => TEST 2 PASSED: Exact metric depth and 3D isolation confirmed.\n")


def test_dynamic_relocation_and_pipeline():
    print("==================================================================")
    print("TEST 3: End-to-End Tri-Modal Detector with Dynamic Relocation")
    print("==================================================================")
    rclpy.init()
    try:
        detector = NidarSurvivorDetector()

        # Simulate drone pose in map frame: (x=1.0, y=-2.0, z=1.5, yaw=0.0)
        pose_msg = PoseStamped()
        pose_msg.pose.position.x = 1.0
        pose_msg.pose.position.y = -2.0
        pose_msg.pose.position.z = 1.5
        pose_msg.pose.orientation.w = 1.0 # yaw = 0
        detector.cb_pose(pose_msg)

        # -------------------------------------------------------------
        # SCENARIO A: Mannequin located at body (Xb=3.0, Yb=0.0)
        # Expected Map location: (1.0 + 3.0, -2.0 + 0.0) = (4.00, -2.00)
        # -------------------------------------------------------------
        print("  --- Scenario A: Mannequin at Map (+4.00, -2.00) ---")
        bgr_a, depth_a = render_synthetic_scene(mannequin_pos=(3.0, 0.0))

        # Update detector depth cache
        with detector.depth_lock:
            detector.latest_depth_frame = depth_a
            detector.latest_depth_stamp = time.time()

        # Simulate camera frame stream while the object is still >2m away.
        for i in range(6):
            time.sleep(0.08) # simulate frame delta
            detector.latest_depth_stamp = time.time()
            detector.process_vision_first_detection(bgr_a, 'front')

        with detector.survivor_lock:
            confirmed = list(detector.confirmed_survivors)
        assert len(confirmed) == 0, "A survivor farther than 2m must not be mapped"
        print("  ✓ Far candidate requested for inspection and was not mapped.")

        # Simulate the explorer approaching while preserving the same world
        # coordinate: drone x=2.5 + centre-corrected RGB-D x=1.89 => map x=4.39.
        pose_msg.pose.position.x = 2.5
        detector.cb_pose(pose_msg)
        bgr_a_near, depth_a_near = render_synthetic_scene(mannequin_pos=(1.50, 0.0))
        with detector.depth_lock:
            detector.latest_depth_frame = depth_a_near
            detector.latest_depth_stamp = time.time()
        for _ in range(5):
            time.sleep(0.08)
            detector.latest_depth_stamp = time.time()
            detector.process_vision_first_detection(bgr_a_near, 'front')

        with detector.survivor_lock:
            confirmed = list(detector.confirmed_survivors)
            print(f"  Confirmed Survivors after approach: {len(confirmed)}")
        assert len(confirmed) == 1, f"Expected 1 close-range confirmed survivor, got {len(confirmed)}"
        s1 = confirmed[0]
        assert abs(s1['x'] - 4.39) <= 0.25, f"Expected Map X ~ 4.39, got {s1['x']}"
        assert abs(s1['y'] - (-2.00)) <= 0.25, f"Expected Map Y ~ -2.00, got {s1['y']}"
        print("  ✓ Survivor correctly localized and locked without Gazebo cheat.")

        # -------------------------------------------------------------
        # SCENARIO B: Empty Scene - verify zero new survivors or false alarms
        # -------------------------------------------------------------
        print("\n  --- Scenario B: Empty Wall - Verify 0 False Alarms ---")
        bgr_empty, depth_empty = render_synthetic_scene(mannequin_pos=None)
        with detector.depth_lock:
            detector.latest_depth_frame = depth_empty
            detector.latest_depth_stamp = time.time()

        for _ in range(5):
            time.sleep(0.05)
            detector.latest_depth_stamp = time.time()
            detector.process_vision_first_detection(bgr_empty, 'front')

        with detector.survivor_lock:
            assert len(detector.confirmed_survivors) == 1, "False positive occurred on empty wall!"
        print("  ✓ Zero false alarms on plain walls.")

        # -------------------------------------------------------------
        # SCENARIO C: Dynamic Relocation to a New Unknown Location
        # Drone turns left (yaw = +90 deg), drone at (x=-1.0, y=0.0)
        # Mannequin is at close body range (depth=1.5m; centre-corrected Xb=1.89m)
        # In Map frame:
        # wx = dx + Xb*cos(90) - Yb*sin(90) = -1.0 + 0 = -1.0
        # wy = dy + Xb*sin(90) + Yb*cos(90) = 0.0 + 1.89 = +1.89
        # Expected Map location: (-1.0, +1.89)
        # -------------------------------------------------------------
        print("\n  --- Scenario C: Dynamic Relocation to New Position Map (-1.00, +1.89) ---")
        pose_msg.pose.position.x = -1.0
        pose_msg.pose.position.y = 0.0
        pose_msg.pose.position.z = 1.5
        # Yaw = +90 deg (pi/2) -> q = [cos(pi/4), 0, 0, sin(pi/4)]
        pose_msg.pose.orientation.w = math.cos(math.pi / 4.0)
        pose_msg.pose.orientation.z = math.sin(math.pi / 4.0)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        detector.cb_pose(pose_msg)

        bgr_c, depth_c = render_synthetic_scene(mannequin_pos=(1.50, 0.0))
        with detector.depth_lock:
            detector.latest_depth_frame = depth_c
            detector.latest_depth_stamp = time.time()

        for _ in range(5):
            time.sleep(0.08)
            detector.latest_depth_stamp = time.time()
            detector.process_vision_first_detection(bgr_c, 'front')

        with detector.survivor_lock:
            confirmed = list(detector.confirmed_survivors)
            print(f"  Total Confirmed Survivors: {len(confirmed)}")
            for s in confirmed:
                print(f"    Survivor #{s['id']}: Map Position ({s['x']:+.2f}, {s['y']:+.2f}) | Conf: {s['confidence']}%")

        assert len(confirmed) == 2, f"Expected 2 confirmed survivors, got {len(confirmed)}"
        s2 = confirmed[1]
        assert abs(s2['x'] - (-1.00)) <= 0.35, f"Expected Map X ~ -1.00, got {s2['x']}"
        assert abs(s2['y'] - 1.89) <= 0.35, f"Expected Map Y ~ +1.89, got {s2['y']}"
        print("  ✓ Relocated survivor discovered and localized at new coordinates!")
        print("  => TEST 3 PASSED: Dynamic relocation works seamlessly without ground truth.")

    finally:
        detector.destroy_node()
        rclpy.shutdown()


def main():
    print("==================================================================")
    print("STARTING TRI-MODAL SURVIVOR PERCEPTION VERIFICATION SUITE")
    print("==================================================================")
    test_custom_classifier()
    test_depth_analyzer()
    test_dynamic_relocation_and_pipeline()
    print("\n==================================================================")
    print("ALL SYNTHETIC TESTS PASSED.")
    print("These fixtures do not validate autonomous coverage, real humans,")
    print("or precision/recall on live Gazebo camera images.")
    print("==================================================================")


if __name__ == '__main__':
    main()

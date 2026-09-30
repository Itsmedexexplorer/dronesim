#!/usr/bin/env python3
"""
NIDAR Survivor Perception & Localization Offline Validation Test Tool
=====================================================================
Ground Truth is loaded from the SDF ONLY in this evaluation script.
The runtime perception pipeline NEVER imports or references this file.

Calculates:
  • Ground Truth Position vs Detected Map Position
  • Euclidean Localization Error (m)
  • False Positives & False Negatives
  • Detection Latency
  • Produces formatted verification table
"""

import os
import sys
import re
import time
import math
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String

def load_ground_truth_from_sdf(sdf_path):
    """Parses mannequin poses directly from world SDF for test evaluation only."""
    survivors = {}
    if not os.path.exists(sdf_path):
        print(f"Error: SDF file not found at {sdf_path}")
        return survivors

    with open(sdf_path, 'r') as f:
        content = f.read()

    # Search for <model name="survivor_X"> ... <pose>X Y Z R P Y</pose>
    pattern = r'<model\s+name=["\'](survivor_\d+)["\']>(?:(?!</model>).)*?<pose>([^<]+)</pose>'
    matches = re.findall(pattern, content, re.DOTALL)
    for name, pose_str in matches:
        parts = [float(p) for p in pose_str.strip().split()]
        if len(parts) >= 2:
            survivors[name] = {
                'name': name,
                'gt_x': parts[0],
                'gt_y': parts[1],
                'gt_z': parts[2] if len(parts) > 2 else 0.0,
                'gt_yaw': parts[5] if len(parts) > 5 else 0.0
            }
    return survivors


class SurvivorValidator(Node):
    def __init__(self, ground_truth):
        super().__init__('test_survivor_localization')
        self.ground_truth = ground_truth
        self.confirmed_detections = []
        self.start_time = time.time()
        self.first_detection_times = {}

        self.sub_detailed = self.create_subscription(
            String, '/nidar/survivors_detailed', self.cb_detailed, 10
        )
        self.sub_count = self.create_subscription(
            Int32, '/nidar/survivor_count', self.cb_count, 10
        )
        self.get_logger().info(f'Loaded {len(self.ground_truth)} Ground-Truth Mannequins from SDF.')

    def cb_count(self, msg: Int32):
        pass

    def cb_detailed(self, msg: String):
        try:
            dets = json.loads(msg.data)
            now = time.time()
            for d in dets:
                sid = d.get('id')
                if sid not in self.first_detection_times:
                    self.first_detection_times[sid] = now - self.start_time
            self.confirmed_detections = dets
        except Exception:
            pass

    def evaluate(self):
        """Matches detected survivors against ground truth."""
        gt_keys = list(self.ground_truth.keys())
        matched_gt = set()
        results = []

        for d in self.confirmed_detections:
            sid = d.get('id')
            dx = float(d.get('map_x', d.get('x', 0.0)))
            dy = float(d.get('map_y', d.get('y', 0.0)))
            conf = d.get('confidence', 0)
            latency = self.first_detection_times.get(sid, 0.0)

            # Find closest GT mannequin
            best_gt = None
            min_dist = float('inf')
            for name, gt in self.ground_truth.items():
                dist = math.hypot(dx - gt['gt_x'], dy - gt['gt_y'])
                if dist < min_dist:
                    min_dist = dist
                    best_gt = name

            is_true_positive = (min_dist <= 0.75)
            if is_true_positive and best_gt not in matched_gt:
                matched_gt.add(best_gt)
                classification = "TRUE POSITIVE"
            elif is_true_positive and best_gt in matched_gt:
                classification = "DUPLICATE"
            else:
                classification = "FALSE POSITIVE (Wall/Noise)"

            gt_pos = f"({self.ground_truth[best_gt]['gt_x']:+.2f}, {self.ground_truth[best_gt]['gt_y']:+.2f})" if best_gt else "N/A"
            results.append({
                'id': sid,
                'detected_pos': f"({dx:+.2f}, {dy:+.2f})",
                'matched_gt': best_gt or 'None',
                'gt_pos': gt_pos,
                'error_m': min_dist if best_gt else 99.9,
                'conf': conf,
                'latency_s': latency,
                'classification': classification
            })

        # Check for False Negatives (unfound GT mannequins)
        unmatched = [name for name in gt_keys if name not in matched_gt]
        return results, unmatched

    def print_report(self):
        results, unmatched = self.evaluate()
        print("\n" + "=" * 90)
        print("         NIDAR SURVIVOR LOCALIZATION VALIDATION REPORT (GROUND TRUTH VS DETECTED)         ")
        print("=" * 90)
        print(f"{'ID':<4} | {'Detected (X, Y)':<18} | {'Matched GT':<12} | {'GT (X, Y)':<18} | {'Error (m)':<9} | {'Conf':<5} | {'Classification':<15}")
        print("-" * 90)

        for r in results:
            err_str = f"{r['error_m']:.2f}m" if r['error_m'] < 50.0 else "N/A"
            print(f"#{r['id']:<3} | {r['detected_pos']:<18} | {r['matched_gt']:<12} | {r['gt_pos']:<18} | {err_str:<9} | {r['conf']:<4}% | {r['classification']:<15}")

        print("-" * 90)
        tp = sum(1 for r in results if r['classification'] == "TRUE POSITIVE")
        fp = sum(1 for r in results if "FALSE POSITIVE" in r['classification'])
        dup = sum(1 for r in results if r['classification'] == "DUPLICATE")
        fn = len(unmatched)

        print(f"SUMMARY: True Positives={tp} | False Positives={fp} | Duplicates={dup} | Missed GT={fn}")
        if unmatched:
            print(f"Missed Mannequins: {', '.join(unmatched)}")
        print("=" * 90 + "\n")


def main():
    sdf_path = "/home/dexter/Downloads/Drone-simulation-2-main/worlds/nidar_drone_world.sdf"
    if len(sys.argv) > 1:
        sdf_path = sys.argv[1]

    gt = load_ground_truth_from_sdf(sdf_path)
    if not gt:
        print("No ground truth survivors found in SDF!")
        sys.exit(1)

    rclpy.init()
    node = SurvivorValidator(gt)
    print("Listening to /nidar/survivors_detailed... (Press Ctrl+C to print validation report)")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

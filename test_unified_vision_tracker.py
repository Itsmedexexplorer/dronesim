#!/usr/bin/env python3
"""
================================================================================
UNIT & INTEGRATION TEST SUITE: NIDAR UNIFIED VISION TRACKER
================================================================================
Validates:
  1. MannequinPerceptionEngine on High-Vis Color Stack Mannequins
  2. MannequinPerceptionEngine on Gray / Plastic Monotone Mannequins (Silhouette)
  3. SpatialSurvivorTracker 3D Euclidean Association (Distance Threshold = 1.0m)
  4. Spatial Duplicate Suppression & Rolling Update Filter (Refining Precision)
  5. Returning / Corridor Revisitation (Zero Duplicate Creation)
  6. Multi-Survivor Discrimination across the Global Map
================================================================================
"""

import os
import sys
import time
import math
import unittest
import numpy as np
import cv2

repo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_dir)

from map.nidar_vision_tracker_node import (
    MannequinPerceptionEngine, SpatialSurvivorTracker, UniqueSurvivor,
    is_close_range_detection, is_registration_quality,
    is_compact_map_object,
)


class TestUnifiedVisionTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 75)
        print("    STARTING NIDAR VISION & SPATIAL TRACKER VERIFICATION SUITE")
        print("=" * 75)
        cls.perception = MannequinPerceptionEngine(fx=363.0, fy=363.0, cx=320.0, cy=240.0)
        cls.tracker = SpatialSurvivorTracker(distance_threshold=1.0, min_confirm_hits=3, rolling_alpha=0.15)

    def setUp(self):
        self.tracker.clear()

    # --------------------------------------------------------------------------
    # Helper: Synthetic Frame Generators
    # --------------------------------------------------------------------------
    def _create_color_mannequin_image(self, u=320, v=240, bw=70, bh=210):
        """Generates synthetic camera image with high-vis rescue mannequin."""
        img = np.full((480, 640, 3), 160, dtype=np.uint8)  # Gray arena wall
        u1, u2 = u - bw // 2, u + bw // 2
        v1, v2 = v - bh // 2, v + bh // 2

        # 1. Rescue Orange Body (Lower 65%)
        torso_v1 = v1 + int(bh * 0.35)
        img[torso_v1:v2, u1:u2] = [20, 95, 235]  # BGR

        # 2. Fluorescent Yellow Safety Vest (Middle 15%)
        vest_v1 = v1 + int(bh * 0.20)
        img[vest_v1:torso_v1, u1:u2] = [30, 220, 240]

        # 3. Peach Head Sphere (Upper 20%, narrower)
        head_w = int(bw * 0.55)
        hu1 = u - head_w // 2
        hu2 = u + head_w // 2
        img[v1:vest_v1, hu1:hu2] = [170, 195, 245]

        # Metric depth map (mannequin at 2.50m, background at 5.0m)
        depth = np.full((480, 640), 5.0, dtype=np.float32)
        depth[v1:v2, u1:u2] = 2.50
        return img, depth

    def _create_gray_plastic_mannequin_image(self, u=320, v=240, bw=65, bh=195):
        """Generates synthetic camera image with gray/plastic monotone mannequin."""
        img = np.full((480, 640, 3), 190, dtype=np.uint8)  # Light gray wall
        u1, u2 = u - bw // 2, u + bw // 2
        v1, v2 = v - bh // 2, v + bh // 2

        # Dark gray plastic body with upright human proportions
        # Head (narrower upper)
        head_w = int(bw * 0.50)
        hu1 = u - head_w // 2
        hu2 = u + head_w // 2
        img[v1:v1 + int(bh * 0.25), hu1:hu2] = [50, 50, 50]

        # Torso & Legs (wider body)
        img[v1 + int(bh * 0.25):v2, u1:u2] = [55, 55, 55]

        depth = np.full((480, 640), 4.5, dtype=np.float32)
        depth[v1:v2, u1:u2] = 2.20
        return img, depth

    # --------------------------------------------------------------------------
    # Tests
    # --------------------------------------------------------------------------
    def test_01_color_stack_mannequin_detection(self):
        """Verify perception isolates high-vis mannequin and extracts 3D coordinates."""
        print("\n[TEST 1] Testing MannequinPerceptionEngine on High-Vis Color Stack...")
        bgr, depth = self._create_color_mannequin_image(u=320, v=240, bw=70, bh=210)
        dets = self.perception.detect(bgr, depth_img=depth, mode='hybrid')

        self.assertGreaterEqual(len(dets), 1, "Perception failed to detect color-stack mannequin!")
        best_det = max(dets, key=lambda d: d['confidence'])
        xc, yc, zc = best_det['cam_point']

        print(f"  • Detected candidates: {len(dets)}")
        print(f"  • Best confidence    : {best_det['confidence']*100:.1f}%")
        print(f"  • Detection method   : {best_det['method']}")
        print(f"  • Camera 3D coords   : Xc={xc:+.2f}m, Yc={yc:+.2f}m, Zc={zc:.2f}m")

        self.assertGreater(best_det['confidence'], 0.70)
        self.assertAlmostEqual(zc, 2.50, delta=0.15)
        self.assertAlmostEqual(xc, 0.00, delta=0.20)
        print("✅ TEST 1 PASSED: High-vis mannequin correctly isolated with metric 3D position.")

    def test_02_gray_plastic_silhouette_detection(self):
        """Verify perception isolates monotone gray/plastic mannequin via silhouette geometry."""
        print("\n[TEST 2] Testing MannequinPerceptionEngine on Gray Plastic Silhouette...")
        bgr, depth = self._create_gray_plastic_mannequin_image(u=320, v=240, bw=65, bh=195)
        dets = self.perception.detect(bgr, depth_img=depth, mode='silhouette')

        self.assertGreaterEqual(len(dets), 1, "Perception failed to detect plastic/gray mannequin silhouette!")
        best_det = max(dets, key=lambda d: d['confidence'])
        xc, yc, zc = best_det['cam_point']

        print(f"  • Detected candidates: {len(dets)}")
        print(f"  • Best confidence    : {best_det['confidence']*100:.1f}%")
        print(f"  • Detection method   : {best_det['method']}")
        print(f"  • Camera 3D coords   : Xc={xc:+.2f}m, Yc={yc:+.2f}m, Zc={zc:.2f}m")

        self.assertGreater(best_det['confidence'], 0.50)
        self.assertAlmostEqual(zc, 2.20, delta=0.15)
        print("✅ TEST 2 PASSED: Gray plastic mannequin detected via silhouette geometry.")

    def test_03_spatial_duplicate_suppression_and_rolling_update(self):
        """
        Verify SpatialSurvivorTracker suppresses duplicates and applies rolling update
        as robot hovers/lingers near a target with localization noise.
        """
        print("\n[TEST 3] Testing Spatial Duplicate Suppression & Rolling Update Filter...")
        true_pos = np.array([3.50, 1.20, 0.90])
        np.random.seed(42)

        # Simulate 15 consecutive detections with Gaussian localization noise (+/- 0.20m)
        now = time.time()
        for i in range(15):
            noise = np.random.normal(0.0, 0.12, size=3)
            measured_pos = true_pos + noise
            conf = 0.88 + float(np.random.uniform(-0.05, 0.05))

            confirmed, newly_confirmed = self.tracker.update(
                [(measured_pos, conf, (0.0, 0.0, 2.5))],
                timestamp=now + i * 0.1
            )

        print(f"  • Total confirmed unique survivors in master list: {len(confirmed)}")
        self.assertEqual(len(confirmed), 1, f"Expected exactly 1 survivor, got {len(confirmed)} (DUPLICATES CREATED!)")

        survivor = confirmed[0]
        final_error = float(np.linalg.norm(survivor.pos - true_pos))
        print(f"  • Survivor #1 hits recorded: {survivor.hits}")
        print(f"  • Estimated position       : ({survivor.pos[0]:.3f}, {survivor.pos[1]:.3f}, {survivor.pos[2]:.3f})")
        print(f"  • True centroid            : ({true_pos[0]:.3f}, {true_pos[1]:.3f}, {true_pos[2]:.3f})")
        print(f"  • Filter error vs true     : {final_error*1000.0:.1f} mm")

        self.assertEqual(survivor.hits, 15)
        self.assertLess(final_error, 0.08, "Rolling average failed to converge to true position!")
        print("✅ TEST 3 PASSED: Zero duplicates created; rolling update refined coordinate precision.")

    def test_04_corridor_revisitation_no_duplicate(self):
        """
        Verify that when the drone visits a second target down a corridor,
        and then REVISITS the first target, NO duplicates are created.
        """
        print("\n[TEST 4] Testing Corridor Navigation & Path Revisitation...")
        pos_A = np.array([3.50, 1.20, 0.90])    # Room A
        pos_B = np.array([-2.40, -1.80, 0.90])  # Room B (distance > 5m)

        now = time.time()

        # Step 1: Observe Target A (4 frames)
        for i in range(4):
            self.tracker.update([(pos_A + np.random.normal(0, 0.05, 3), 0.92, (0, 0, 2.5))], now + i * 0.1)

        confirmed, _ = self.tracker.update([], now + 0.5)
        self.assertEqual(len(confirmed), 1, "Failed to confirm Target A!")
        print(f"  • Step 1 (Room A): Confirmed count = {len(confirmed)}")

        # Step 2: Fly to Room B, observe Target B (4 frames)
        for i in range(4):
            self.tracker.update([(pos_B + np.random.normal(0, 0.05, 3), 0.89, (0, 0, 2.8))], now + 2.0 + i * 0.1)

        confirmed, _ = self.tracker.update([], now + 2.5)
        self.assertEqual(len(confirmed), 2, "Failed to confirm Target B!")
        print(f"  • Step 2 (Room B): Confirmed count = {len(confirmed)}")

        # Step 3: Fly back through corridor and REVISIT Target A (5 frames)
        for i in range(5):
            self.tracker.update([(pos_A + np.random.normal(0, 0.08, 3), 0.94, (0, 0, 2.2))], now + 5.0 + i * 0.1)

        confirmed, newly_confirmed = self.tracker.update([], now + 5.6)
        print(f"  • Step 3 (Revisiting Room A): Confirmed count = {len(confirmed)}")
        print(f"  • Newly confirmed count on return: {len(newly_confirmed)}")

        self.assertEqual(len(confirmed), 2, f"Corridor revisitation created duplicate! Count is {len(confirmed)}")
        self.assertEqual(len(newly_confirmed), 0, "Revisitation incorrectly created new confirmed survivor!")

        # Verify ID continuity
        sA = next(s for s in confirmed if s.id == 1)
        self.assertEqual(sA.hits, 9, "Survivor A hits not accumulated on revisitation!")
        print("✅ TEST 4 PASSED: Returning to target accumulated observations with zero duplicates.")

    def test_05_distance_threshold_boundary(self):
        """Verify distance thresholding properly separates distinct nearby objects."""
        print("\n[TEST 5] Testing Distance Threshold Boundary (Threshold = 1.0m)...")
        base_pos = np.array([0.0, 0.0, 0.90])
        close_pos = np.array([0.65, 0.0, 0.90])  # Distance 0.65m <= 1.0m (SAME SURVIVOR)
        far_pos = np.array([1.45, 0.0, 0.90])    # Distance 1.45m > 1.0m (DIFFERENT SURVIVOR)

        now = time.time()
        # Seed base survivor (3 hits)
        for _ in range(3):
            self.tracker.update([(base_pos, 0.90, (0, 0, 2))], now)

        confirmed, _ = self.tracker.update([], now)
        self.assertEqual(len(confirmed), 1)

        # Feed detection at 0.65m -> should match existing
        self.tracker.update([(close_pos, 0.90, (0, 0, 2))], now + 0.1)
        confirmed, _ = self.tracker.update([], now + 0.1)
        self.assertEqual(len(confirmed), 1, "Detection within 0.65m should NOT create new survivor!")

        # Feed detection at 1.45m -> should be tracked as distinct survivor
        for _ in range(3):
            self.tracker.update([(far_pos, 0.90, (0, 0, 2))], now + 0.2)
        confirmed, _ = self.tracker.update([], now + 0.2)
        self.assertEqual(len(confirmed), 2, "Detection at 1.45m (>1.0m) MUST create new survivor!")

        print("✅ TEST 5 PASSED: Distance thresholding accurately segments unique survivors.")

    def test_06_textured_human_ssd_detection(self):
        """Verify perception isolates realistic textured human models via MobileNet-SSD."""
        print("\n[TEST 6] Testing MannequinPerceptionEngine on Realistic Textured Humans (SSD)...")
        randy_path = '/home/dexter/.gz/fuel/fuel.gazebosim.org/openrobotics/models/rescue randy/3/thumbnails/1.png'
        person_path = '/home/dexter/.gz/fuel/fuel.gazebosim.org/openrobotics/models/standing person/3/thumbnails/1.png'

        for label, path in [("Rescue Randy", randy_path), ("Standing Person", person_path)]:
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            self.assertIsNotNone(img, f"Failed to load image at {path}")
            h, w = img.shape[:2]
            dummy_depth = np.full((h, w), 2.5, dtype=np.float32)

            dets = self.perception.detect(img, depth_img=dummy_depth, mode='hybrid')
            self.assertGreaterEqual(len(dets), 1, f"Failed to detect {label} with neural/hybrid detector!")
            best = max(dets, key=lambda d: d['confidence'])
            print(f"  • {label}: Detected {len(dets)} candidate(s) | Method={best['method']} | Conf={best['confidence']*100:.1f}%")
            self.assertGreaterEqual(best['confidence'], 0.50)

        print("✅ TEST 6 PASSED: Realistic textured human models detected with high neural confidence.")

    def test_07_same_frame_proposals_count_once(self):
        """SSD/color/silhouette proposals in one image must be one observation."""
        now = time.time()
        detections = [
            (np.array([2.00, 1.00, 0.90]), 0.91, (0.0, 0.0, 2.0)),
            (np.array([2.08, 1.04, 1.05]), 0.82, (0.1, 0.0, 2.1)),
            (np.array([1.95, 0.96, 0.75]), 0.76, (-0.1, 0.0, 1.9)),
        ]
        confirmed, _ = self.tracker.update(detections, now)
        self.assertEqual(len(confirmed), 0, "One image incorrectly confirmed a survivor")
        self.assertEqual(len(self.tracker.candidate_tracks), 1)
        self.assertEqual(self.tracker.candidate_tracks[0].hits, 1)

        self.tracker.update(detections, now + 0.1)
        confirmed, _ = self.tracker.update(detections, now + 0.2)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].hits, 3)

    def test_08_revisit_height_noise_does_not_duplicate(self):
        """Large perspective-induced Z changes still match the same map anchor."""
        now = time.time()
        base = np.array([3.0, -2.0, 0.8])
        for i in range(3):
            self.tracker.update([(base, 0.9, (0, 0, 2))], now + i * 0.1)

        revisit = [
            np.array([3.18, -1.92, 1.65]),
            np.array([2.88, -2.10, 0.25]),
            np.array([3.10, -2.05, 1.35]),
        ]
        for i, pos in enumerate(revisit):
            confirmed, _ = self.tracker.update(
                [(pos, 0.88, (0, 0, 2))], now + 5.0 + i * 0.1)
        self.assertEqual(len(confirmed), 1, "Revisit Z noise created a duplicate anchor")

    def test_09_far_detection_is_approach_only(self):
        """A distant YOLO hit must not be eligible for permanent registration."""
        self.assertFalse(is_close_range_detection((0.2, 0.0, 4.0), 2.6))
        self.assertTrue(is_close_range_detection((0.2, 0.0, 1.5), 2.6))
        self.assertFalse(is_close_range_detection((0.0, 0.0, float('nan')), 2.6))

    def test_10_registry_restore_preserves_identity(self):
        """A perception-node restart must retain IDs and suppress repeat marking."""
        now = time.time()
        pos = np.array([4.0, -1.0, 0.9])
        for i in range(3):
            confirmed, _ = self.tracker.update(
                [(pos, 0.92, (0.0, 0.0, 1.5))], now + i * 0.1)
        self.assertEqual(len(confirmed), 1)

        restored_tracker = SpatialSurvivorTracker(
            distance_threshold=1.0, min_confirm_hits=3, rolling_alpha=0.15)
        restored = restored_tracker.restore(self.tracker.snapshot(), now + 2.0)
        self.assertEqual(restored, 1)
        for i in range(4):
            confirmed, newly = restored_tracker.update(
                [(pos + np.array([0.15, -0.10, 0.3]), 0.90, (0.0, 0.0, 1.6))],
                now + 3.0 + i * 0.1)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].id, 1)
        self.assertEqual(newly, [])

    def test_11_registration_rejects_low_height_and_missing_depth(self):
        """Floor/wall false positives must never become permanent survivors."""
        valid = {
            'cam_point': (0.1, 0.0, 1.5), 'confidence': 0.80,
            'depth_verified': True, 'freestanding': True,
        }
        self.assertTrue(is_registration_quality(valid, np.array([1.0, 2.0, 1.15]), 2.6))
        self.assertFalse(is_registration_quality(valid, np.array([1.0, 2.0, 0.58]), 2.6))
        no_depth = dict(valid, depth_verified=False)
        self.assertFalse(is_registration_quality(no_depth, np.array([1.0, 2.0, 1.15]), 2.6))
        wall = dict(valid, freestanding=False)
        self.assertFalse(is_registration_quality(wall, np.array([1.0, 2.0, 1.15]), 2.6))

    def test_12_occupancy_context_rejects_walls(self):
        """Wall segments/ends fail while a compact survivor obstacle passes."""
        resolution = 0.05
        grid = np.zeros((120, 120), dtype=np.int16)

        # Compact 0.45m-diameter occupied body around world (2.0, 2.0).
        cr = cc = 40
        yy, xx = np.ogrid[:120, :120]
        grid[(yy - cr) ** 2 + (xx - cc) ** 2 <= 4 ** 2] = 100
        self.assertTrue(is_compact_map_object(
            grid, resolution, 0.0, 0.0, 2.0, 2.0))

        # A 0.5m door-edge panel, matching the last run's false-positive shape.
        wall_grid = np.zeros_like(grid)
        wall_grid[80, 30:41] = 100
        self.assertFalse(is_compact_map_object(
            wall_grid, resolution, 0.0, 0.0, 2.0, 4.0))

        # Long corridor wall.
        long_wall = np.zeros_like(grid)
        long_wall[20:70, 54] = 100
        self.assertFalse(is_compact_map_object(
            long_wall, resolution, 0.0, 0.0, 2.7, 2.0))


def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUnifiedVisionTracker)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if res.wasSuccessful():
        print("\n" + "=" * 75)
        print("   ALL TESTS PASSED SUCCESSFULLY! ZERO ERRORS, ZERO DUPLICATES.")
        print("=" * 75)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

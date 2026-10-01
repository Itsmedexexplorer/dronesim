#!/usr/bin/env python3
"""Regression tests for RGB-D alignment and survivor confirmation safety."""

import threading
import time
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from map.tri_modal_detector import DepthAnalyzer, VisionProposalScanner
from map.nidar_survivor_detector import NidarSurvivorDetector
from map.jev_explorer import (
    JevDecisionLayer, calculate_search_confidence, grid_line_is_clear)
from map.nidar_slam_node import NidarSlamNode, clip_free_ray_at_wall
from map.nidar_vision_tracker_node import rgbd_frames_are_synchronized


class _Image:
    def __init__(self, array):
        self.height, self.width = array.shape[:2]
        self.data = array.tobytes()


class _Logger:
    def info(self, _message):
        pass


def _candidate(t, distance, stamp):
    return {
        'wx': 1.50, 'wy': -0.25, 'dist': distance, 't': t,
        'fused_score': 90.0, 'vis_detected': True, 'vis_stamp': stamp,
        'geometry_verified': False, 'depth_verified': True,
        'freestanding': True, 'within_confirm_range': distance <= 2.0,
    }


class TestPerceptionSafety(unittest.TestCase):
    def test_free_space_ray_stops_before_an_established_wall(self):
        grid = np.full((20, 20), 50, dtype=np.uint8)
        grid[10, 8] = 90
        endpoint, blocked = clip_free_ray_at_wall(grid, (2, 10), (15, 10))
        self.assertTrue(blocked)
        self.assertEqual(endpoint, (7, 10))

    def test_free_space_ray_reaches_endpoint_when_unobstructed(self):
        grid = np.full((20, 20), 50, dtype=np.uint8)
        endpoint, blocked = clip_free_ray_at_wall(grid, (2, 2), (12, 7))
        self.assertFalse(blocked)
        self.assertEqual(endpoint, (12, 7))

    def test_search_confidence_cannot_pass_with_unresolved_evidence(self):
        complete = calculate_search_confidence(
            99.0, 100.0, 0, 0, True, False, False)
        self.assertGreater(complete, 97.0)
        self.assertLess(calculate_search_confidence(
            99.0, 100.0, 1, 0, True, False, False), 97.0)
        self.assertLess(calculate_search_confidence(
            99.0, 100.0, 0, 1, True, False, False), 97.0)
        self.assertLess(calculate_search_confidence(
            99.0, 100.0, 0, 0, True, True, False), 97.0)
        self.assertLess(calculate_search_confidence(
            99.0, 100.0, 0, 0, True, False, True), 97.0)
        self.assertLessEqual(calculate_search_confidence(
            99.0, 100.0, 0, 0, False, False, False), 85.0)

    def test_high_map_and_camera_evidence_can_reach_completion_trust(self):
        score = calculate_search_confidence(
            97.8, 94.0, 0, 0, True, False, False)
        self.assertGreater(score, 97.0)

    def test_camera_visibility_is_blocked_by_walls_and_unknown_space(self):
        grid = np.zeros((12, 12), dtype=np.int8)
        self.assertTrue(grid_line_is_clear(grid, (1, 6), (10, 6)))
        grid[6, 5] = 100
        self.assertFalse(grid_line_is_clear(grid, (1, 6), (10, 6)))
        grid[6, 5] = -1
        self.assertFalse(grid_line_is_clear(grid, (1, 6), (10, 6)))

    def test_rgb_and_depth_from_different_headings_are_not_fused(self):
        self.assertTrue(rgbd_frames_are_synchronized(10.000, 10.045, 0.08))
        self.assertFalse(rgbd_frames_are_synchronized(10.000, 10.125, 0.08))
        self.assertFalse(rgbd_frames_are_synchronized(10.000, None, 0.08))

    def test_inspection_queue_preserves_multiple_prospects(self):
        explorer = object.__new__(JevDecisionLayer)
        explorer.pose_lock = threading.Lock()
        explorer.inspection_lock = threading.Lock()
        explorer.drone_x = explorer.drone_y = 0.0
        explorer.pending_inspections = {
            (3.0, 0.0): (3.0, 0.0, (3.0, 0.0), time.time()),
            (1.0, 0.0): (1.0, 0.0, (1.0, 0.0), time.time()),
        }
        explorer.inspection_cooldowns = {}
        chosen = explorer._next_inspection()
        self.assertEqual(chosen[2], (1.0, 0.0))
        self.assertIn((3.0, 0.0), explorer.pending_inspections)

    def test_block_audit_selects_an_unscanned_free_block(self):
        explorer = object.__new__(JevDecisionLayer)
        explorer.grid_lock = threading.Lock()
        explorer.pose_lock = threading.Lock()
        explorer.grid = np.zeros((40, 40), dtype=np.int8)
        explorer.map_info = SimpleNamespace(
            resolution=0.4, width=40, height=40,
            origin=SimpleNamespace(position=SimpleNamespace(x=-8.0, y=-8.0)))
        explorer.drone_x = explorer.drone_y = -5.0
        explorer.visual_scan_locations = [(-5.0, -5.0)]
        explorer.audit_block_size_m = 2.0
        explorer.blocked_targets = []
        goal = explorer.find_block_audit_goal()
        self.assertIsNotNone(goal)
        scanned_key = (int((-5.0 + 7.1) // 2.0), int((-5.0 + 7.1) // 2.0))
        goal_key = (int((goal[0] + 7.1) // 2.0), int((goal[1] + 7.1) // 2.0))
        self.assertNotEqual(goal_key, scanned_key)

    def test_reobservations_preserve_anchor_and_update_confidence(self):
        detector = object.__new__(NidarSurvivorDetector)
        anchor = dict(id=1, x=1.5, y=-0.25, width_m=0.5, confidence=70)
        detector.confirmed_survivors = [anchor]
        detector.candidate_tracks = []
        detector.survivor_lock = threading.Lock()
        detector._do_publish = lambda: None
        for i in range(30):
            candidate = _candidate(i * 0.2, 1.5, i * 0.2)
            candidate.update(wx=1.5 + 0.2 * np.sin(i), width_m=0.5)
            detector.register_candidate_observation(candidate)
        self.assertEqual(len(detector.confirmed_survivors), 1)
        self.assertEqual(detector.candidate_tracks, [])
        self.assertEqual(anchor['confidence'], 90)
        self.assertEqual((anchor['x'], anchor['y']), (1.5, -0.25))
        candidate.update(width_m=2.0)
        detector.register_candidate_observation(candidate)
        self.assertEqual(detector.candidate_tracks, [])

    def test_map_marker_updates_by_id_not_position(self):
        mapper = object.__new__(NidarSlamNode)
        mapper.grid_lock = threading.Lock()
        mapper.confirmed_survivors = []
        mapper.stamped_survivors = set()
        mapper.get_logger = lambda: _Logger()
        marker = SimpleNamespace(ns='survivors', id=20,
                                 pose=SimpleNamespace(position=SimpleNamespace(x=1., y=2.)))
        mapper.cb_survivor_markers(SimpleNamespace(markers=[marker]))
        marker.pose.position.x = 1.2
        mapper.cb_survivor_markers(SimpleNamespace(markers=[marker]))
        marker.id = 25  # Not a ground-disc marker.
        mapper.cb_survivor_markers(SimpleNamespace(markers=[marker]))
        self.assertEqual(mapper.confirmed_survivors, [dict(id=2, x=1.2, y=2.)])

    def test_obstruction_requires_three_scans_then_cancels(self):
        explorer = object.__new__(JevDecisionLayer)
        explorer.is_acting = True
        explorer.cancel_pending = False
        explorer.route = [(0., 0.), (0.5, 0.), (1., 0.)]
        explorer.pose_lock = threading.Lock()
        explorer.drone_x = explorer.drone_y = explorer.drone_yaw = 0.
        explorer.blocked_scan_hits = 0
        explorer.blocked_scan_threshold = 3
        explorer.blocked_targets = []
        explorer.current_target = (3., 0.)
        cancelled = []
        explorer.cancel_current_action = lambda: cancelled.append(True)
        scan = SimpleNamespace(ranges=[0.5], range_min=0.1, range_max=10.,
                               angle_min=0., angle_increment=0.1)
        for _ in range(2):
            explorer.cb_obstacles(scan)
        self.assertEqual(cancelled, [])
        explorer.cb_obstacles(scan)
        self.assertEqual(cancelled, [True])
        self.assertEqual(explorer.goal_status, 'BLOCKED')
        self.assertGreater(explorer.blocked_targets[0][2], time.monotonic())

    def test_old_navigation_result_cannot_finish_new_goal(self):
        explorer = object.__new__(JevDecisionLayer)
        explorer.goal_generation = 5
        explorer.cancel_pending = False
        explorer.is_acting = True
        explorer.goal_status = 'PENDING'
        explorer._goal_result_cb(None, 3)
        self.assertTrue(explorer.is_acting)
        self.assertEqual(explorer.goal_status, 'PENDING')

    def test_gazebo_rgb_is_converted_to_opencv_bgr(self):
        rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        detector = object.__new__(NidarSurvivorDetector)
        decoded = detector._decode_gz_image(_Image(rgb))
        self.assertEqual(decoded[0, 0].tolist(), [0, 0, 255])
        self.assertEqual(decoded[0, 1].tolist(), [0, 255, 0])

    def test_depth_gap_accepts_isolated_object_and_rejects_flat_wall(self):
        isolated = np.full((100, 100), 5.0, dtype=np.float32)
        isolated[25:80, 40:60] = 1.50
        valid, distance, freestanding, gap = DepthAnalyzer.analyze_roi_depth(
            isolated, [80, 25, 120, 80], source_size=(200, 100))
        self.assertTrue(valid)
        self.assertAlmostEqual(distance, 1.50, places=2)
        self.assertTrue(freestanding)
        self.assertGreater(gap, 3.0)

        wall = np.full((100, 100), 1.50, dtype=np.float32)
        valid, _, freestanding, gap = DepthAnalyzer.analyze_roi_depth(
            wall, [40, 25, 60, 80])
        self.assertTrue(valid)
        self.assertFalse(freestanding)
        self.assertAlmostEqual(gap, 0.0, places=2)

    def test_mannequin_signature_operates_without_custom_onnx(self):
        image = np.full((240, 320, 3), 190, dtype=np.uint8)
        cv2.rectangle(image, (145, 95), (175, 180), (0, 89, 255), -1)
        cv2.rectangle(image, (142, 70), (178, 105), (38, 242, 242), -1)
        detections = VisionProposalScanner(None).scan_frame(image, conf_thresh=0.65)
        self.assertGreaterEqual(len(detections), 1)
        self.assertEqual(detections[0]['detector'], 'mannequin_signature')

    def test_far_candidate_is_not_mapped_until_close_depth_confirmation(self):
        detector = object.__new__(NidarSurvivorDetector)
        detector.confirmed_survivors = []
        detector.candidate_tracks = []
        detector.next_track_id = 1
        detector.survivor_lock = threading.Lock()
        detector._do_publish = lambda: None
        detector.get_logger = lambda: _Logger()

        start = time.time() - 1.0
        for i in range(4):
            detector.register_candidate_observation(_candidate(start + i * 0.10, 3.0, start + i * 0.10))
        self.assertEqual(detector.confirmed_survivors, [])

        for i in range(5):
            detector.register_candidate_observation(_candidate(start + 0.6 + i * 0.10, 1.50, start + 0.6 + i * 0.10))
        self.assertEqual(len(detector.confirmed_survivors), 1)
        survivor = detector.confirmed_survivors[0]
        self.assertLessEqual(survivor['confirmation_range_m'], 2.0)
        self.assertEqual(survivor['evidence'], 'vision+depth')

    def test_single_colour_objects_rejected_even_with_confident_toy_model(self):
        class OverconfidentModel:
            is_loaded = True
            def predict_batch(self, crops):
                return [1.0] * len(crops)
        scanner = VisionProposalScanner(OverconfidentModel())
        for colour in ((0, 89, 255), (38, 242, 242)):
            image = np.full((240, 320, 3), 190, dtype=np.uint8)
            cv2.rectangle(image, (100, 50), (200, 200), colour, -1)
            self.assertEqual(scanner.scan_frame(image), [])

    def test_lidar_projection_cannot_commit_survivor(self):
        detector = object.__new__(NidarSurvivorDetector)
        # Rejection must happen before any tracker or publisher access.
        for i in range(20):
            candidate = _candidate(i * 0.2, 1.0, i * 0.2)
            candidate['detector'] = 'lidar_visual_fusion'
            detector.register_candidate_observation(candidate)

    def test_inspection_goal_keeps_safe_standoff(self):
        explorer = object.__new__(JevDecisionLayer)
        explorer.pose_lock = threading.Lock()
        explorer.grid_lock = threading.Lock()
        explorer.drone_x = 0.0
        explorer.drone_y = 0.0
        explorer.grid = np.zeros((120, 120), dtype=np.int8)
        explorer.map_info = SimpleNamespace(
            width=120, height=120, resolution=0.1,
            origin=SimpleNamespace(position=SimpleNamespace(x=-6.0, y=-6.0)))
        target = (2.5, 0.0)
        goal = explorer._inspection_goal(*target)
        self.assertIsNotNone(goal)
        self.assertGreaterEqual(np.hypot(goal[0]-target[0], goal[1]-target[1]), 1.14)
        self.assertLessEqual(np.hypot(goal[0]-target[0], goal[1]-target[1]), 1.61)


if __name__ == '__main__':
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""
Unit test to verify that NidarVisionTrackerNode correctly transforms 3D detections
and publishes MarkerArray, survivor_count, and survivors_detailed.
"""

import os
import sys
import time
import json
import unittest
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Int32, String

repo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_dir)

from map.nidar_vision_tracker_node import NidarVisionTrackerNode, UniqueSurvivor


class TestMarkerGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = NidarVisionTrackerNode()
        # Mock drone pose at map (2.0, 1.0, 1.5), heading East (yaw = 0.0)
        pose_msg = PoseStamped()
        pose_msg.pose.position.x = 2.0
        pose_msg.pose.position.y = 1.0
        pose_msg.pose.position.z = 1.5
        pose_msg.pose.orientation.w = 1.0
        self.node.cb_pose(pose_msg)

    def tearDown(self):
        self.node.destroy_node()

    def test_01_kinematic_transform_accuracy(self):
        """Verify 3D camera coordinates map accurately to global map coordinates via kinematics."""
        # Camera point: Xc=0.0m (center), Yc=0.0m, Zc=2.0m (2m forward)
        pt_map = self.node._transform_camera_to_map(0.0, 0.0, 2.0)
        self.assertIsNotNone(pt_map, "Transform returned None despite valid drone pose!")

        # Expected: Xmap = 2.0 + (0.17 + 2.0) = 4.17m, Ymap = 1.0 + 0.0 = 1.0m, Zmap = 1.5 + 0.025 = 1.525m
        self.assertAlmostEqual(pt_map[0], 4.17, delta=0.02)
        self.assertAlmostEqual(pt_map[1], 1.00, delta=0.02)
        print(f"✅ Kinematic transform verified: Camera (0, 0, 2) -> Map ({pt_map[0]:.2f}, {pt_map[1]:.2f}, {pt_map[2]:.2f})")

    def test_02_marker_publishing_on_detection(self):
        """Verify that when a survivor is observed, MarkerArray is generated with correct IDs."""
        survivor = UniqueSurvivor(
            sid=1,
            pos_map=np.array([4.17, 1.00, 1.52]),
            confidence=0.92,
            timestamp=time.time(),
            first_cam_point=(0.0, 0.0, 2.0)
        )
        survivor.confirmed = True
        survivor.hits = 3

        markers_received = []
        def mrk_cb(msg):
            markers_received.append(msg)

        sub_mrk = self.node.create_subscription(
            MarkerArray, '/nidar/survivor_markers', mrk_cb, 10
        )

        detailed_received = []
        def det_cb(msg):
            detailed_received.append(msg)

        sub_det = self.node.create_subscription(
            String, '/nidar/survivors_detailed', det_cb, 10
        )

        count_received = []
        def cnt_cb(msg):
            count_received.append(msg)

        sub_cnt = self.node.create_subscription(
            Int32, '/nidar/survivor_count', cnt_cb, 10
        )

        # Allow DDS matching
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.02)
            time.sleep(0.02)

        # Trigger publishing state
        self.node._publish_ros_state([survivor])

        # Spin node to receive published messages
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.02)
            time.sleep(0.02)

        # Verify MarkerArray
        self.assertGreaterEqual(len(markers_received), 1, "Failed to publish to /nidar/survivor_markers!")
        ma = markers_received[0]
        marker_ids = [m.id for m in ma.markers]

        # Must have Disc (10), Body Cube (11), and Text Label (12)
        self.assertIn(10, marker_ids, "Disc marker (id 10) missing!")
        self.assertIn(11, marker_ids, "Body cube marker (id 11) missing!")
        self.assertIn(12, marker_ids, "Text billboard marker (id 12) missing!")

        # Verify Detailed JSON
        self.assertGreaterEqual(len(detailed_received), 1, "Failed to publish to /nidar/survivors_detailed!")
        data = json.loads(detailed_received[0].data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['id'], 1)
        self.assertEqual(data[0]['status'], 'CONFIRMED')
        self.assertAlmostEqual(data[0]['x'], 4.17, delta=0.05)
        self.assertAlmostEqual(data[0]['y'], 1.00, delta=0.05)

        # Verify Count
        self.assertGreaterEqual(len(count_received), 1, "Failed to publish to /nidar/survivor_count!")
        self.assertEqual(count_received[0].data, 1)

        print("✅ Marker publishing verified: Disc, Body Cube, and Text Label published with 100% fidelity.")


if __name__ == '__main__':
    unittest.main()

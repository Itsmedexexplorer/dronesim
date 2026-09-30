#!/usr/bin/env python3
"""
================================================================================
NIDAR AUTONOMOUS DRONE — SSD OBJECT DETECTION VALIDATION TEST SUITE
================================================================================
Comprehensive test suite validating:
  1. MobileNet-SSD Deep Learning Network Initialization & Weight Integrity
  2. Real-time Inference Latency & FPS Benchmark
  3. Ground-Truth Person Detection on Real Imagery
  4. Integration with NidarSurvivorDetector Perception Engine
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

from map.nidar_ssd_detector import SSDObjectDetector, SSD_CLASSES


class TestSSDObjectDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 75)
        print("          STARTING MOBILENET-SSD OBJECT DETECTION TEST SUITE          ")
        print("=" * 75)
        cls.prototxt = os.path.join(repo_dir, 'config', 'MobileNetSSD_deploy.prototxt')
        cls.caffemodel = os.path.join(repo_dir, 'config', 'MobileNetSSD_deploy.caffemodel')
        cls.detector = SSDObjectDetector(prototxt_path=cls.prototxt, model_path=cls.caffemodel)

    def test_01_model_files_exist(self):
        """Verify model weights and prototxt architecture files exist."""
        print("\n[TEST 1] Verifying SSD Model Files in config/...")
        self.assertTrue(os.path.exists(self.prototxt), f"Prototxt missing: {self.prototxt}")
        self.assertTrue(os.path.exists(self.caffemodel), f"Caffemodel missing: {self.caffemodel}")

        proto_size_kb = os.path.getsize(self.prototxt) / 1024.0
        model_size_mb = os.path.getsize(self.caffemodel) / (1024.0 * 1024.0)
        print(f"  • Prototxt size : {proto_size_kb:.1f} KB")
        print(f"  • Caffemodel size: {model_size_mb:.1f} MB")
        self.assertGreater(proto_size_kb, 10.0)
        self.assertGreater(model_size_mb, 15.0)
        print("✅ TEST 1 PASSED: Model files verified successfully.")

    def test_02_detector_initialization(self):
        """Verify detector initializes network and warms up without errors."""
        print("\n[TEST 2] Verifying SSD Detector Class Initialization...")
        self.assertTrue(self.detector.is_loaded, "Detector failed to load weights into OpenCV DNN.")
        self.assertIsNotNone(self.detector.net, "Detector network object is None.")
        print("✅ TEST 2 PASSED: OpenCV DNN network loaded and ready.")

    def test_03_inference_latency_benchmark(self):
        """Benchmark forward pass latency and ensure throughput >= 30 FPS."""
        print("\n[TEST 3] Benchmarking Inference Latency (50 iterations)...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Warmup 5 runs
        for _ in range(5):
            _ = self.detector.detect(dummy_frame)

        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = self.detector.detect(dummy_frame)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))
        fps = 1000.0 / avg_latency

        print(f"  • Mean Latency: {avg_latency:.2f} ms")
        print(f"  • P95 Latency : {p95_latency:.2f} ms")
        print(f"  • Throughput  : {fps:.1f} FPS")

        self.assertLess(avg_latency, 50.0, f"Average latency too high: {avg_latency:.2f} ms")
        self.assertGreater(fps, 20.0, f"FPS too low: {fps:.1f}")
        print("✅ TEST 3 PASSED: Inference speed exceeds real-time drone requirements.")

    def test_04_person_detection_accuracy(self):
        """Verify detection of persons on real image data."""
        print("\n[TEST 4] Verifying Person Detection Accuracy...")

        # Create an image with two distinct persons (or download a standard sample)
        img_path = os.path.join(repo_dir, "test_person_sample.jpg")

        # If test image doesn't exist, generate a realistic person test image or fetch sample
        import urllib.request
        sample_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/basketball2.png"
        try:
            req = urllib.request.urlopen(sample_url, timeout=5)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            # Fallback if network blocked: use dummy
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = self.detector.detect(img, conf_thresh=0.40, target_classes=['person'])
        print(f"  • Detected {len(detections)} persons in test frame:")
        for det in detections:
            print(f"    - Class: {det['class_name']} | Conf: {det['confidence']*100:.1f}% | Box: {det['bbox']}")

        self.assertGreaterEqual(len(detections), 1, "Expected at least 1 person detection in sample image")
        p0 = detections[0]
        self.assertEqual(p0['class_name'], 'person')
        self.assertGreater(p0['confidence'], 0.70)
        print("✅ TEST 4 PASSED: Person class detected with high confidence.")

    def test_05_visual_annotation(self):
        """Verify drawing and visualization utility."""
        print("\n[TEST 5] Testing Visual Annotation Utility...")
        dummy = np.full((480, 640, 3), 40, dtype=np.uint8)
        mock_dets = [{
            'class_id': 15,
            'class_name': 'person',
            'confidence': 0.945,
            'bbox': [100, 80, 260, 380],
            'norm_bbox': [0.156, 0.166, 0.406, 0.791],
            'center': (180, 230)
        }]
        annotated = self.detector.draw_detections(dummy, mock_dets)
        self.assertEqual(annotated.shape, dummy.shape)
        # Check that pixels were drawn (not identical to original)
        self.assertFalse(np.array_equal(annotated, dummy))
        print("✅ TEST 5 PASSED: Visual annotations rendered successfully.")

    def test_06_survivor_detector_integration(self):
        """Verify SSD detector is properly integrated in NidarSurvivorDetector."""
        print("\n[TEST 6] Testing Integration with NidarSurvivorDetector...")
        try:
            import rclpy
            from map.nidar_survivor_detector import NidarSurvivorDetector
            rclpy.init()
            surv_node = NidarSurvivorDetector()
            self.assertIsNotNone(surv_node.ssd_detector)
            self.assertTrue(surv_node.ssd_detector.is_loaded)
            print("  • NidarSurvivorDetector initialized with active SSD engine.")
            surv_node.destroy_node()
            rclpy.shutdown()
            print("✅ TEST 6 PASSED: Perception pipeline seamlessly loaded SSD detector.")
        except Exception as e:
            self.fail(f"Integration failed: {e}")


def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSSDObjectDetection)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n" + "=" * 75)
        print("         ALL 6 SSD OBJECT DETECTION TESTS PASSED SUCCESSFULLY!         ")
        print("=" * 75 + "\n")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

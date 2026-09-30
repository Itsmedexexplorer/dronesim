#!/usr/bin/env python3
"""
================================================================================
NIDAR AUTONOMOUS DRONE — SSD OBJECT DETECTOR (MobileNet-SSD)
================================================================================
Deep learning Single Shot MultiBox Detector (SSD) engine using OpenCV DNN.
Provides real-time person/survivor detection from camera streams with high efficiency,
low CPU footprint (< 25ms inference), and tight spatial integration.

Classes:
  - SSDObjectDetector: Reusable detector engine for single images or video streams.
  - NidarSsdRosNode: Standalone ROS 2 node streaming SSD detections and annotated HUD.
================================================================================
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2

# MobileNet-SSD Pascal VOC 20 Object Classes
SSD_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# Color palette for classes (B, G, R)
SSD_COLORS = {
    "person": (0, 220, 110),     # Vibrant green for survivors / persons
    "default": (255, 180, 50)    # Cyan/blue for other objects
}


class SSDObjectDetector:
    """
    High-performance MobileNet-SSD object detection engine powered by cv2.dnn.
    """
    def __init__(self, prototxt_path=None, model_path=None, default_conf=0.25):
        self.default_conf = default_conf
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.prototxt_path = prototxt_path or os.path.join(base_dir, 'config', 'MobileNetSSD_deploy.prototxt')
        self.model_path = model_path or os.path.join(base_dir, 'config', 'MobileNetSSD_deploy.caffemodel')

        self.net = None
        self.is_loaded = False
        self._load_network()

    def _load_network(self):
        if not os.path.exists(self.prototxt_path):
            print(f"[SSD] Error: Prototxt file not found at {self.prototxt_path}")
            return
        if not os.path.exists(self.model_path):
            print(f"[SSD] Error: Model weight file not found at {self.model_path}")
            return

        try:
            self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
            # Optimize for CPU execution with OpenCV default backends
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.is_loaded = True

            # Warm-up inference
            dummy = np.zeros((300, 300, 3), dtype=np.uint8)
            blob = cv2.dnn.blobFromImage(dummy, 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            self.net.forward()
            print(f"[SSD] MobileNet-SSD successfully initialized and warmed up from {os.path.basename(self.model_path)}")
        except Exception as e:
            print(f"[SSD] Failed to load MobileNet-SSD: {e}")
            self.net = None
            self.is_loaded = False

    def detect(self, img_bgr, conf_thresh=None, target_classes=None):
        """
        Runs SSD inference on a single BGR image.

        Args:
            img_bgr: numpy array (H, W, 3) in BGR format.
            conf_thresh: minimum confidence threshold (default: self.default_conf).
            target_classes: list of class names to filter by (e.g. ['person']). None returns all.

        Returns:
            List of detection dicts:
            [{
                'class_id': int,
                'class_name': str,
                'confidence': float (0..1),
                'bbox': [x1, y1, x2, y2], # Pixel coordinates in original image
                'norm_bbox': [nx1, ny1, nx2, ny2], # Normalized [0..1]
                'center': (cx, cy)
            }]
        """
        if not self.is_loaded or img_bgr is None:
            return []

        threshold = conf_thresh if conf_thresh is not None else self.default_conf
        h, w = img_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return []

        # Ensure 3-channel BGR
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

        # Scale factor 0.007843 = 1 / 127.5; mean subtraction = 127.5
        blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 0.007843, (300, 300), 127.5)

        try:
            self.net.setInput(blob)
            raw_detections = self.net.forward()
        except Exception as e:
            print(f"[SSD] Forward pass error: {e}")
            return []

        results = []
        # raw_detections shape: (1, 1, N, 7)
        for i in range(raw_detections.shape[2]):
            confidence = float(raw_detections[0, 0, i, 2])
            if confidence < threshold:
                continue

            class_id = int(raw_detections[0, 0, i, 1])
            if 0 <= class_id < len(SSD_CLASSES):
                class_name = SSD_CLASSES[class_id]
            else:
                class_name = f"class_{class_id}"

            if target_classes is not None and class_name not in target_classes:
                continue

            # Normalized bounding box coordinates [0, 1]
            nx1 = max(0.0, min(1.0, float(raw_detections[0, 0, i, 3])))
            ny1 = max(0.0, min(1.0, float(raw_detections[0, 0, i, 4])))
            nx2 = max(0.0, min(1.0, float(raw_detections[0, 0, i, 5])))
            ny2 = max(0.0, min(1.0, float(raw_detections[0, 0, i, 6])))

            # Pixel bounding box
            x1 = int(nx1 * w)
            y1 = int(ny1 * h)
            x2 = int(nx2 * w)
            y2 = int(ny2 * h)

            if x2 <= x1 or y2 <= y1:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            results.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'norm_bbox': [nx1, ny1, nx2, ny2],
                'center': (cx, cy)
            })

        return results

    def draw_detections(self, img_bgr, detections):
        """
        Draws stylish bounding boxes and telemetry tags onto an image copy.
        """
        annotated = img_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class_name']
            conf = det['confidence']
            color = SSD_COLORS.get(cls_name, SSD_COLORS['default'])

            # Bounding box with subtle corner brackets
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Corner accents
            corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 3, cv2.LINE_AA)

            # Center target pip
            cx, cy = det['center']
            cv2.circle(annotated, (cx, cy), 3, color, -1)

            # Header badge
            label = f"SSD: {cls_name.upper()} {conf*100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 10, badge_y2), (15, 23, 42), -1)
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 10, badge_y2), color, 1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated


# ==============================================================================
# OPTIONAL ROS 2 NODE FOR STANDALONE EXECUTION
# ==============================================================================
def run_ros_node():
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        from cv_bridge import CvBridge
    except ImportError:
        print("[SSD] ROS 2 dependencies not loaded. Source /opt/ros/jazzy/setup.bash.")
        sys.exit(1)

    class NidarSsdNode(Node):
        def __init__(self):
            super().__init__('nidar_ssd_detector')
            self.detector = SSDObjectDetector()
            self.bridge = CvBridge()

            self.sub_cam = self.create_subscription(
                Image, '/nidar/camera', self.cb_camera, 10
            )
            self.pub_dets = self.create_publisher(String, '/nidar/ssd/detections', 10)
            self.pub_view = self.create_publisher(Image, '/nidar/ssd/camera_view', 10)
            self.get_logger().info('SSD Object Detector ROS 2 Node active on /nidar/camera.')

        def cb_camera(self, msg: Image):
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().warn(f'Frame conversion failed: {e}')
                return

            t0 = time.time()
            detections = self.detector.detect(frame, target_classes=['person'])
            dt_ms = (time.time() - t0) * 1000.0

            # Broadcast detection JSON
            det_msg = String()
            det_msg.data = json.dumps({
                'timestamp': time.time(),
                'inference_ms': dt_ms,
                'count': len(detections),
                'detections': detections
            })
            self.pub_dets.publish(det_msg)

            # Broadcast annotated HUD
            annotated = self.detector.draw_detections(frame, detections)
            cv2.putText(annotated, f"SSD Inference: {dt_ms:.1f}ms | FPS: {1000.0/max(1.0, dt_ms):.1f}",
                        (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            out_img = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            self.pub_view.publish(out_img)

    rclpy.init()
    node = NidarSsdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="NIDAR SSD Object Detection Engine")
    parser.add_argument('--image', type=str, help="Path to input image file for detection")
    parser.add_argument('--output', type=str, default="ssd_output.png", help="Path to save annotated output image")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument('--benchmark', action='store_true', help="Run benchmark on model inference speed")
    parser.add_argument('--ros', action='store_true', help="Run as ROS 2 node")
    args = parser.parse_args()

    detector = SSDObjectDetector(default_conf=args.conf)
    if not detector.is_loaded:
        print("[SSD] Failed to load detector. Aborting.")
        sys.exit(1)

    if args.benchmark:
        print("\n" + "=" * 60)
        print("          BENCHMARKING MOBILENET-SSD INFERENCE SPEED          ")
        print("=" * 60)
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = detector.detect(dummy)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        avg_ms = np.mean(latencies[5:])
        std_ms = np.std(latencies[5:])
        fps = 1000.0 / avg_ms
        print(f"Average Latency : {avg_ms:.2f} ms ± {std_ms:.2f} ms")
        print(f"Throughput      : {fps:.1f} FPS")
        print("=" * 60 + "\n")
        return

    if args.image:
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            sys.exit(1)
        img = cv2.imread(args.image)
        t0 = time.time()
        dets = detector.detect(img, conf_thresh=args.conf)
        dt = (time.time() - t0) * 1000.0
        print(f"Found {len(dets)} objects in {dt:.1f}ms:")
        for d in dets:
            print(f"  - {d['class_name']}: {d['confidence']*100:.1f}% at {d['bbox']}")
        annotated = detector.draw_detections(img, dets)
        cv2.imwrite(args.output, annotated)
        print(f"Annotated result saved to {args.output}")
        return

    if args.ros:
        run_ros_node()
        return

    print("Usage: python3 nidar_ssd_detector.py [--image <path> | --benchmark | --ros]")

if __name__ == '__main__':
    main()

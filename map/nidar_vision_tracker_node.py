#!/usr/bin/env python3
"""
================================================================================
NIDAR AUTONOMOUS DRONE — PRODUCTION-READY SURVIVOR PERCEPTION & SPATIAL TRACKER
================================================================================
Author: Robotics & Computer Vision Engineering Team
Package: airmouse_controller / nidar_perception

This node solves the core search & rescue autonomy challenges:
  1. PERCEPTION FAILURE & FALSE POSITIVE ELIMINATION (NO RANDOM MARKING):
     - Multi-stage perception engine combining:
         * MobileNet-SSD (Deep Learning VOC 'person' class 15 with human textures).
         * OpenCV HOG Pedestrian Detector.
         * Multi-Band HSV Color Masking (Peach head + Fluorescent yellow vest + Orange suit).
         * Geometric Silhouette & Contour Extraction with MANDATORY depth-step verification.
     - Freestanding 3D Depth Verification:
         * Compares foreground median depth against left/right background halo (depth step >= 0.20m).
         * Flat walls, doorframes, ceiling beams, and shadows have zero depth step and are REJECTED.
     - Metric range gating: High-confidence confirmation strictly required within 2.6m.

  2. 360° LIDAR PROSPECTING & ACTIVE CAMERA INSPECTION:
     - 360° LiDAR (/scan) scans continuously for freestanding cylindrical obstacles (radius 0.15m-0.35m).
     - When an unconfirmed prospect is detected at range (2.0m-5.0m):
         * Checks if already within distance_threshold of ANY confirmed survivor. If yes -> IGNORE.
         * If new -> Publishes /nidar/inspection_target so jev_explorer.py approaches to safe standoff (1.35m) facing the target.
     - When the drone arrives facing the candidate at close range (<=2.6m):
         * Camera feed locks in, verifies human features, and registers the survivor.

  3. ZERO-DUPLICATE SPATIAL TRACKER (GLOBAL CONTEXT & ROLLING UPDATE):
     - Transforms local camera detections into the global map frame using tf2_ros.
     - Maintains an authoritative master list of unique, confirmed survivors.
     - 2D/3D Euclidean distance thresholding (configurable, default 1.80 metres).
     - If a detection is within the threshold of an existing survivor:
         * ZERO DUPLICATES are created.
         * An exponential rolling update filter refines the survivor's coordinates.
     - Once confirmed, the survivor's position is permanently locked in the exclusion registry:
         * /nidar/survivors_detailed notifies jev_explorer.py to blacklist the position with 1.5m radius.
         * The drone NEVER re-marks or gets trapped near that survivor, smoothly moving on to scan other rooms.

All frame IDs, topic names, distance thresholds, and filter parameters are fully
configurable via ROS 2 node parameters.
================================================================================
"""

import os
import sys
import time
import math
import json
import threading
from collections import deque
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2

# ROS 2 Core Libraries
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, qos_profile_sensor_data

# ROS 2 Message Types
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Int32, String, Bool
from cv_bridge import CvBridge

# TF2 Transform Libraries
import tf2_ros
import tf2_geometry_msgs

# Optional Gazebo Transport Fallback (for simulation environments without ros_gz bridge)
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.image_pb2 import Image as GzImage
except ImportError:
    try:
        from gz.transport import Node as GzNode
        from gz.msgs.image_pb2 import Image as GzImage
    except ImportError:
        GzNode = None
        GzImage = None


# ==============================================================================
# SECTION 1: MANNEQUIN PERCEPTION ENGINE
# ==============================================================================

class MannequinPerceptionEngine:
    """
    Robust computer vision pipeline to detect both realistic textured humans
    and synthetic Gazebo mannequins, while strictly rejecting flat walls, doorframes, and shadows.
    """
    def __init__(self, fx: float = 363.0, fy: float = 363.0, cx: float = 320.0, cy: float = 240.0,
                 custom_model_path: Optional[str] = None):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Optional ONNX Runtime session for lightweight custom classifier
        self.onnx_session = None
        self.input_name = None
        self.output_name = None
        if custom_model_path and os.path.exists(custom_model_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.onnx_session = ort.InferenceSession(custom_model_path, opts, providers=['CPUExecutionProvider'])
                self.input_name = self.onnx_session.get_inputs()[0].name
                self.output_name = self.onnx_session.get_outputs()[0].name
            except Exception:
                self.onnx_session = None

        # Pre-trained MobileNet-SSD Caffe detector (detects VOC 'person' class 15 with human textures)
        self.ssd_net = None
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proto_p = os.path.join(base_dir, 'config', 'MobileNetSSD_deploy.prototxt')
        model_p = os.path.join(base_dir, 'config', 'MobileNetSSD_deploy.caffemodel')
        if os.path.exists(proto_p) and os.path.exists(model_p):
            try:
                self.ssd_net = cv2.dnn.readNetFromCaffe(proto_p, model_p)
            except Exception:
                self.ssd_net = None

        # OpenCV HOG pedestrian detector fallback
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception:
            self.hog = None

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """Updates camera intrinsic parameters from CameraInfo."""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def _detect_ssd_person(self, bgr_img: np.ndarray) -> List[Dict]:
        """Runs MobileNet-SSD to detect realistic textured humans and clothing."""
        if self.ssd_net is None or bgr_img is None:
            return []
        h, w = bgr_img.shape[:2]
        try:
            blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img, (300, 300)), 0.007843, (300, 300), 127.5)
            self.ssd_net.setInput(blob)
            detections = self.ssd_net.forward()
            results = []
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                class_id = int(detections[0, 0, i, 1])
                # class_id 15 is 'person' in VOC
                if class_id == 15 and conf > 0.35:
                    bx1 = int(detections[0, 0, i, 3] * w)
                    by1 = int(detections[0, 0, i, 4] * h)
                    bx2 = int(detections[0, 0, i, 5] * w)
                    by2 = int(detections[0, 0, i, 6] * h)
                    bx1 = max(0, min(w - 1, bx1))
                    by1 = max(0, min(h - 1, by1))
                    bx2 = max(0, min(w - 1, bx2))
                    by2 = max(0, min(h - 1, by2))
                    bw = bx2 - bx1
                    bh = by2 - by1
                    if bw >= 12 and bh >= 20:
                        results.append({
                            'bbox': [bx1, by1, bx2, by2],
                            'center': (bx1 + bw // 2, by1 + bh // 2),
                            'confidence': min(0.99, conf),
                            'method': 'mobilenet_ssd'
                        })
            return results
        except Exception:
            return []

    def _detect_hog_person(self, bgr_img: np.ndarray) -> List[Dict]:
        """Runs OpenCV HOG pedestrian detector as fallback."""
        if self.hog is None or bgr_img is None:
            return []
        h, w = bgr_img.shape[:2]
        try:
            scale = 1.0
            if w > 480:
                scale = 480.0 / w
                small = cv2.resize(bgr_img, (480, int(h * scale)))
            else:
                small = bgr_img
            rects, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(4, 4), scale=1.05)
            results = []
            for (rx, ry, rw, rh), weight in zip(rects, weights):
                if weight > 0.15:
                    bx1 = max(0, int(rx / scale))
                    by1 = max(0, int(ry / scale))
                    bx2 = min(w - 1, int((rx + rw) / scale))
                    by2 = min(h - 1, int((ry + rh) / scale))
                    results.append({
                        'bbox': [bx1, by1, bx2, by2],
                        'center': (bx1 + (bx2 - bx1) // 2, by1 + (by2 - by1) // 2),
                        'confidence': min(0.95, float(weight) / 2.0 + 0.5),
                        'method': 'hog_pedestrian'
                    })
            return results
        except Exception:
            return []

    def _verify_depth_freestanding(self, depth_img: np.ndarray, bbox: List[int],
                                   bgr_shape: Optional[Tuple[int, int]] = None) -> Tuple[bool, float, bool, float]:
        """
        Extracts foreground depth and verifies whether the object is freestanding in 3D
        vs the surrounding background wall (depth step >= 0.18m).
        Supports resolution differences between BGR and depth images via automated scaling.
        Returns: (is_valid, z_foreground, is_freestanding, depth_step)
        """
        if depth_img is None:
            return False, 0.0, True, 0.0

        dh, dw = depth_img.shape[:2]
        x1, y1, x2, y2 = bbox

        # Auto-scale bounding box if BGR resolution differs from depth resolution
        if bgr_shape is not None and (bgr_shape[0] != dh or bgr_shape[1] != dw):
            ih, iw = bgr_shape[:2]
            scale_x = float(dw) / max(1.0, float(iw))
            scale_y = float(dh) / max(1.0, float(ih))
            x1 = int(round(x1 * scale_x))
            x2 = int(round(x2 * scale_x))
            y1 = int(round(y1 * scale_y))
            y2 = int(round(y2 * scale_y))

        x1 = max(0, min(dw - 1, x1))
        x2 = max(x1 + 1, min(dw, x2))
        y1 = max(0, min(dh - 1, y1))
        y2 = max(y1 + 1, min(dh, y2))
        bw = x2 - x1
        bh = y2 - y1

        # Central 40% region of the bounding box
        pad_u = int(bw * 0.25)
        pad_v = int(bh * 0.20)
        roi_d = depth_img[max(0, y1 + pad_v):min(dh, y2 - pad_v), max(0, x1 + pad_u):min(dw, x2 - pad_u)]
        valid_mask = np.isfinite(roi_d) & (roi_d > 0.35) & (roi_d < 9.0)
        valid_vals = roi_d[valid_mask]

        if len(valid_vals) < 6:
            return False, 0.0, False, 0.0

        # Foreground depth: median of lowest 40% distance values inside ROI
        near_cut = float(np.percentile(valid_vals, 40))
        near_vals = valid_vals[valid_vals <= near_cut]
        z_fg = float(np.median(near_vals))

        # Background halo sampling on left, right, and top
        halo_pad = max(6, int(bw * 0.35))
        bg_left = depth_img[y1:y2, max(0, x1 - halo_pad):x1].flatten()
        bg_right = depth_img[y1:y2, x2:min(dw, x2 + halo_pad)].flatten()
        bg_top = depth_img[max(0, y1 - halo_pad):y1, x1:x2].flatten()
        bg_all = np.concatenate([bg_left, bg_right, bg_top]) if (bg_left.size or bg_right.size or bg_top.size) else np.array([])

        open_air_count = int(np.sum(np.isinf(bg_all) | np.isnan(bg_all) | (bg_all > 8.0))) if bg_all.size else 0
        finite_bg = bg_all[np.isfinite(bg_all) & (bg_all > 0.35)] if bg_all.size else np.array([])

        if open_air_count >= 10:
            z_bg = 10.0
            depth_step = float(z_bg - z_fg)
            is_freestanding = True
        elif len(finite_bg) >= 8:
            z_bg = float(np.median(finite_bg))
            depth_step = float(z_bg - z_fg)
            is_freestanding = (depth_step >= 0.18)
        else:
            depth_step = 0.0
            is_freestanding = (z_fg <= 2.80)

        return True, z_fg, is_freestanding, depth_step

    def detect(self, bgr_img: np.ndarray, depth_img: Optional[np.ndarray] = None,
               mode: str = 'hybrid') -> List[Dict]:
        """
        Detects mannequin candidates in the BGR image and computes their 3D coordinates.
        Strictly rejects flat walls, doorframes, and shadows.
        """
        if bgr_img is None:
            return []

        h, w = bgr_img.shape[:2]
        candidates = []

        # ----------------------------------------------------------------------
        # Strategy A: Deep Learning (MobileNet-SSD) / HOG Pedestrian Detection
        # ----------------------------------------------------------------------
        if mode in ('person', 'neural', 'hybrid'):
            ssd_dets = self._detect_ssd_person(bgr_img)
            candidates.extend(ssd_dets)
            # 'person' is the production mode for realistic human models:
            # MobileNet-SSD person class only. HOG/silhouette wall-edge
            # fallbacks remain available in legacy neural/hybrid modes.
            if not ssd_dets and mode != 'person':
                hog_dets = self._detect_hog_person(bgr_img)
                candidates.extend(hog_dets)

        # ----------------------------------------------------------------------
        # Strategy B: Multi-Band HSV Color Stack Detection (Vests / Suits)
        # ----------------------------------------------------------------------
        if mode in ('color', 'hybrid'):
            color_candidates = self._detect_color_stack(bgr_img)
            for cc in color_candidates:
                cbx = cc['bbox']
                overlap = False
                for c in candidates:
                    sbx = c['bbox']
                    ix1 = max(cbx[0], sbx[0])
                    iy1 = max(cbx[1], sbx[1])
                    ix2 = min(cbx[2], sbx[2])
                    iy2 = min(cbx[3], sbx[3])
                    if ix1 < ix2 and iy1 < iy2:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        c_area = (cbx[2] - cbx[0]) * (cbx[3] - cbx[1])
                        if inter / float(c_area) > 0.35:
                            overlap = True
                            break
                if not overlap:
                    candidates.append(cc)

        # ----------------------------------------------------------------------
        # Strategy C: Geometric Silhouette & Contour Extraction
        # ----------------------------------------------------------------------
        if mode in ('silhouette', 'hybrid'):
            silh_candidates = self._detect_silhouette_contours(bgr_img)
            for sc in silh_candidates:
                sbx = sc['bbox']
                overlap = False
                for c in candidates:
                    cbx = c['bbox']
                    ix1 = max(sbx[0], cbx[0])
                    iy1 = max(sbx[1], cbx[1])
                    ix2 = min(sbx[2], cbx[2])
                    iy2 = min(sbx[3], cbx[3])
                    if ix1 < ix2 and iy1 < iy2:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        s_area = (sbx[2] - sbx[0]) * (sbx[3] - sbx[1])
                        if inter_area / float(s_area) > 0.35:
                            overlap = True
                            break
                if not overlap:
                    candidates.append(sc)

        # ----------------------------------------------------------------------
        # Strategy D: 3D Metric Coordinate Extraction & Freestanding Wall Rejection
        # ----------------------------------------------------------------------
        valid_detections = []
        for cand in candidates:
            bbox = cand['bbox']
            u, v = cand['center']
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1

            # 1. Depth extraction & Freestanding Check
            zc = None
            is_freestanding = True
            depth_step = 0.0
            depth_verified = False

            if depth_img is not None:
                d_ok, d_val, d_free, d_step = self._verify_depth_freestanding(depth_img, bbox, bgr_shape=(h, w))
                if d_ok:
                    depth_verified = True
                    zc = d_val
                    is_freestanding = d_free
                    depth_step = d_step

                    # CRITICAL FILTER: If an untextured silhouette contour is NOT freestanding
                    # (depth step < 0.18m), it is a flat wall corner or door edge! REJECT IT!
                    if not is_freestanding and cand['method'] == 'silhouette':
                        continue

            scale_f = float(h) / 480.0 if h > 0 else 1.0
            eff_fy = self.fy * scale_f
            eff_fx = self.fx * (float(w) / 640.0 if w > 0 else 1.0)
            eff_cx = self.cx * (float(w) / 640.0 if w > 0 else 1.0)
            eff_cy = self.cy * scale_f

            # Fallback to pinhole geometry ONLY if high-confidence neural/color candidate
            if zc is None or not (0.35 <= zc <= 9.0):
                if cand['confidence'] >= 0.65 and bh > 18:
                    zc = float((eff_fy * 1.75) / bh)
                else:
                    continue  # Unverified noisy contour without depth -> REJECT

            # Unproject from pixel (u, v) + depth Zc to 3D camera optical coordinates
            xc = float((u - eff_cx) * zc / eff_fx)
            yc = float((v - eff_cy) * zc / eff_fy)

            # Optional Neural Model Verification Crop
            if self.onnx_session is not None:
                crop = bgr_img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.shape[0] >= 12 and crop.shape[1] >= 8:
                    net_conf = self._infer_onnx(crop)
                    if net_conf >= 0.50:
                        cand['confidence'] = max(cand['confidence'], net_conf)
                        cand['method'] += '+neural'

            cand['cam_point'] = (xc, yc, zc)
            cand['depth_step'] = depth_step
            cand['freestanding'] = is_freestanding
            cand['depth_verified'] = depth_verified
            valid_detections.append(cand)

        return valid_detections

    def _detect_color_stack(self, bgr_img: np.ndarray) -> List[Dict]:
        """Detects mannequins via stacked vertical color masks (Head, Vest, Body)."""
        h, w = bgr_img.shape[:2]
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        # Orange Body: Hue [4, 24], Sat [75, 255], Val [55, 255]
        mask_orange = cv2.inRange(hsv, np.array([4, 75, 55]), np.array([24, 255, 255]))
        # Fluorescent Yellow Vest: Hue [24, 40], Sat [75, 255], Val [70, 255]
        mask_yellow = cv2.inRange(hsv, np.array([24, 75, 70]), np.array([40, 255, 255]))
        # Peach Head: Hue [0, 24], Sat [30, 160], Val [140, 255]
        mask_peach = cv2.inRange(hsv, np.array([0, 30, 140]), np.array([24, 160, 255]))

        combined = cv2.bitwise_or(mask_orange, mask_yellow)
        combined = cv2.bitwise_or(combined, mask_peach)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 60:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bh / max(1.0, float(bw))

            if not (1.3 <= aspect_ratio <= 5.2):
                continue

            roi_orange = mask_orange[by:by+bh, bx:bx+bw]
            roi_yellow = mask_yellow[by:by+bh, bx:bx+bw]
            n_orange = cv2.countNonZero(roi_orange)
            n_yellow = cv2.countNonZero(roi_yellow)

            if (n_orange + n_yellow) < 25:
                continue

            density = (n_orange + n_yellow) / float(bw * bh + 1e-5)
            ar_score = 1.0 - abs(aspect_ratio - 2.8) / 2.8
            conf = float(np.clip(0.50 * density * 2.5 + 0.30 * max(0.0, ar_score) + 0.20, 0.40, 0.98))

            candidates.append({
                'bbox': [bx, by, bx + bw, by + bh],
                'center': (int(bx + bw / 2), int(by + bh / 2)),
                'confidence': conf,
                'method': 'color_stack'
            })

        return candidates

    def _detect_silhouette_contours(self, bgr_img: np.ndarray) -> List[Dict]:
        """Detects gray, plastic, or monotone mannequins via edge & silhouette geometry."""
        h, w = bgr_img.shape[:2]
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 15, 4)

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, v_kernel)
        clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 120 or area > (h * w * 0.50):
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bh / max(1.0, float(bw))

            if not (1.5 <= aspect_ratio <= 4.5):
                continue

            bbox_area = bw * bh
            solidity = area / float(bbox_area + 1e-5)
            if not (0.35 <= solidity <= 0.90):
                continue

            head_h = max(2, int(bh * 0.25))
            head_roi = clean[by:by+head_h, bx:bx+bw]
            torso_roi = clean[by+head_h:by+int(bh*0.75), bx:bx+bw]

            head_w_px = np.count_nonzero(head_roi.any(axis=0))
            torso_w_px = np.count_nonzero(torso_roi.any(axis=0))

            if torso_w_px < 6:
                continue

            taper_ratio = head_w_px / max(1.0, float(torso_w_px))
            if not (0.25 <= taper_ratio <= 0.95):
                continue

            conf = float(np.clip(0.40 * solidity + 0.35 * (1.0 - abs(taper_ratio - 0.65)) + 0.15, 0.40, 0.88))
            candidates.append({
                'bbox': [bx, by, bx + bw, by + bh],
                'center': (int(bx + bw / 2), int(by + bh / 2)),
                'confidence': conf,
                'method': 'silhouette'
            })

        return candidates

    def _infer_onnx(self, crop_bgr: np.ndarray) -> float:
        """Infers custom classifier probability P(survivor) for candidate crop."""
        try:
            resized = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_LINEAR)
            blob = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
            out = self.onnx_session.run([self.output_name], {self.input_name: blob})
            logits = out[0][0]
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)
            return float(probs[1])
        except Exception:
            return 0.0


# ==============================================================================
# SECTION 2: SPATIAL SURVIVOR TRACKER (ZERO-DUPLICATE GLOBAL REGISTRY)
# ==============================================================================

class UniqueSurvivor:
    """Represents a unique, confirmed survivor in the global map frame."""
    def __init__(self, sid: int, pos_map: np.ndarray, confidence: float, timestamp: float,
                 first_cam_point: Tuple[float, float, float]):
        self.id = sid
        self.pos = np.asarray(pos_map, dtype=np.float64)
        self.anchor = self.pos.copy()
        self.confidence = float(confidence)
        self.hits = 1
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.confirmed = False
        self.first_cam_point = first_cam_point
        self.last_cam_point = first_cam_point
        self.position_history = deque([self.pos.copy()], maxlen=30)

    def add_observation(self, pos_map: np.ndarray, confidence: float, timestamp: float,
                        cam_point: Tuple[float, float, float], alpha: float = 0.15):
        """Refines position using an exponential moving average and median smoothing."""
        p = np.asarray(pos_map, dtype=np.float64)
        self.position_history.append(p.copy())
        robust_center = np.median(np.asarray(self.position_history), axis=0)
        self.pos = (1.0 - alpha) * self.pos + alpha * robust_center
        self.anchor = robust_center
        self.hits += 1
        self.confidence = 0.85 * self.confidence + 0.15 * confidence
        self.last_seen = timestamp
        self.last_cam_point = cam_point

    def to_dict(self) -> Dict:
        gx = round(float(self.pos[0]), 2)
        gy = round(float(self.pos[1]), 2)
        gz = round(float(self.pos[2]), 2)
        col = "W" if gx < -2.5 else ("C" if gx < 2.5 else "E")
        row = "S" if gy < -2.5 else ("M" if gy < 2.5 else "N")
        grid_x = int(np.clip(math.floor((gx + 7.5) / 2.5), 0, 5))
        grid_y = int(np.clip(math.floor((gy + 7.5) / 2.5), 0, 5))
        grid_lbl = f"{row}{col}-{grid_x}{grid_y}"

        return {
            'id': self.id,
            'status': 'CONFIRMED' if self.confirmed else 'CANDIDATE',
            'x': gx,
            'y': gy,
            'map_x': gx,
            'map_y': gy,
            'map_z': gz,
            'map_pos_fixed': True,
            'grid': grid_lbl,
            'confidence': int(round(self.confidence * 100.0)),
            'observations': self.hits,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen
        }


class SpatialSurvivorTracker:
    """
    Robust spatial tracking engine that maintains a master global list of unique survivors,
    transforms camera detections to map coordinates, suppresses duplicates, and refines
    existing survivor coordinate precision using a rolling update filter.
    """
    def __init__(self, distance_threshold: float = 1.80, min_confirm_hits: int = 3,
                 rolling_alpha: float = 0.15, max_history_weight: int = 25,
                 vertical_threshold: float = 1.5):
        """
        Args:
            distance_threshold: 2D/3D Euclidean distance (meters) to match existing survivors.
            min_confirm_hits: Minimum multi-frame detections before promoting candidate to confirmed.
            rolling_alpha: Minimum smoothing factor for exponential moving average.
            max_history_weight: Maximum observation count used to compute 1/N averaging.
        """
        self.distance_threshold = distance_threshold
        self.min_confirm_hits = min_confirm_hits
        self.rolling_alpha = rolling_alpha
        self.max_history_weight = max_history_weight
        self.vertical_threshold = vertical_threshold

        # Master Global Lists
        self.confirmed_survivors: List[UniqueSurvivor] = []
        self.candidate_tracks: List[UniqueSurvivor] = []
        self.next_survivor_id = 1
        self.lock = threading.Lock()

    def _distance(self, pos: np.ndarray, track: UniqueSurvivor) -> float:
        """Computes 2D ground plane distance with tolerant height gating."""
        if abs(float(pos[2] - track.anchor[2])) > self.vertical_threshold:
            return float('inf')
        return float(np.linalg.norm(np.asarray(pos[:2]) - track.anchor[:2]))

    def _coalesce_frame(self, detections):
        """Collapses overlapping proposals from the same image into one hit."""
        clusters = []
        for pos, conf, cam_pt in sorted(detections, key=lambda d: d[1], reverse=True):
            pos = np.asarray(pos, dtype=np.float64)
            match = next((c for c in clusters
                          if np.linalg.norm(pos[:2] - c[0][:2]) <= self.distance_threshold * 0.55), None)
            if match is None:
                clusters.append([pos, float(conf), cam_pt])
            else:
                total = match[1] + max(float(conf), 1e-3)
                match[0] = (match[0] * match[1] + pos * float(conf)) / total
                match[1] = max(match[1], float(conf))
                match[2] = cam_pt
        return [(c[0], c[1], c[2]) for c in clusters]

    def _merge_confirmed(self):
        """Enforces that no two confirmed survivors ever exist within distance_threshold of each other."""
        i = 0
        while i < len(self.confirmed_survivors):
            a = self.confirmed_survivors[i]
            j = i + 1
            while j < len(self.confirmed_survivors):
                b = self.confirmed_survivors[j]
                if self._distance(b.anchor, a) <= self.distance_threshold:
                    for p in b.position_history:
                        a.position_history.append(np.asarray(p).copy())
                    a.pos = np.median(np.asarray(a.position_history), axis=0)
                    a.anchor = a.pos.copy()
                    a.hits += b.hits
                    a.confidence = max(a.confidence, b.confidence)
                    a.last_seen = max(a.last_seen, b.last_seen)
                    del self.confirmed_survivors[j]
                else:
                    j += 1
            i += 1

    def update(self, detected_points_map: List[Tuple[np.ndarray, float, Tuple[float, float, float]]],
               timestamp: float) -> Tuple[List[UniqueSurvivor], List[UniqueSurvivor]]:
        """
        Processes new detections in global MAP frame coordinates.
        Returns: (master_confirmed_survivors, newly_confirmed_survivors)
        """
        newly_confirmed = []

        with self.lock:
            for pos_map, conf, cam_pt in self._coalesce_frame(detected_points_map):
                # --------------------------------------------------------------
                # STEP 1: Check against master confirmed survivors
                # --------------------------------------------------------------
                matched_confirmed = False
                min_dist_conf = float('inf')
                best_conf_target = None

                for survivor in self.confirmed_survivors:
                    d = self._distance(pos_map, survivor)
                    if d < min_dist_conf:
                        min_dist_conf = d
                        best_conf_target = survivor

                if best_conf_target is not None and min_dist_conf <= self.distance_threshold:
                    # MATCH FOUND: STRICTLY ZERO DUPLICATES!
                    matched_confirmed = True
                    best_conf_target.add_observation(
                        pos_map, conf, timestamp, cam_pt, self.rolling_alpha)
                    continue

                # --------------------------------------------------------------
                # STEP 2: Check against active candidate tracks
                # --------------------------------------------------------------
                matched_candidate = False
                min_dist_cand = float('inf')
                best_cand_target = None

                for cand in self.candidate_tracks:
                    d = self._distance(pos_map, cand)
                    if d < min_dist_cand:
                        min_dist_cand = d
                        best_cand_target = cand

                if best_cand_target is not None and min_dist_cand <= self.distance_threshold:
                    matched_candidate = True
                    best_cand_target.add_observation(
                        pos_map, conf, timestamp, cam_pt, self.rolling_alpha)

                    # Check confirmation threshold
                    # Detector confidence is classification evidence, not a
                    # substitute for independent spatial observations.
                    should_confirm = best_cand_target.hits >= self.min_confirm_hits
                    if should_confirm:
                        existing = min(self.confirmed_survivors,
                                       key=lambda s: self._distance(best_cand_target.anchor, s),
                                       default=None)
                        if (existing is not None and
                                self._distance(best_cand_target.anchor, existing) <= self.distance_threshold):
                            for p in best_cand_target.position_history:
                                existing.add_observation(p, conf, timestamp, cam_pt, self.rolling_alpha)
                            self.candidate_tracks.remove(best_cand_target)
                            continue
                        best_cand_target.confirmed = True
                        best_cand_target.id = self.next_survivor_id
                        self.next_survivor_id += 1
                        self.confirmed_survivors.append(best_cand_target)
                        newly_confirmed.append(best_cand_target)
                        self.candidate_tracks.remove(best_cand_target)
                else:
                    # ----------------------------------------------------------
                    # STEP 3: Create a new candidate track
                    # ----------------------------------------------------------
                    new_candidate = UniqueSurvivor(
                        sid=0,
                        pos_map=pos_map,
                        confidence=conf,
                        timestamp=timestamp,
                        first_cam_point=cam_pt
                    )
                    if self.min_confirm_hits <= 1:
                        new_candidate.confirmed = True
                        new_candidate.id = self.next_survivor_id
                        self.next_survivor_id += 1
                        self.confirmed_survivors.append(new_candidate)
                        newly_confirmed.append(new_candidate)
                    else:
                        self.candidate_tracks.append(new_candidate)

            # Clean stale candidate tracks that were not re-observed within 15 seconds
            self.candidate_tracks = [
                c for c in self.candidate_tracks if (timestamp - c.last_seen) < 15.0
            ]

            # Merge any confirmed survivors that drift within distance_threshold
            self._merge_confirmed()

            return list(self.confirmed_survivors), newly_confirmed

    def clear(self):
        """Clears all tracker state (e.g. on relaunch command)."""
        with self.lock:
            self.confirmed_survivors.clear()
            self.candidate_tracks.clear()
            self.next_survivor_id = 1

    def snapshot(self) -> List[Dict]:
        """Return the durable, map-frame identity registry."""
        with self.lock:
            return [s.to_dict() for s in self.confirmed_survivors]

    def restore(self, records: List[Dict], timestamp: Optional[float] = None) -> int:
        """Restore confirmed identities after a perception-node restart."""
        restored = 0
        now = time.time() if timestamp is None else float(timestamp)
        with self.lock:
            self.confirmed_survivors.clear()
            self.candidate_tracks.clear()
            max_id = 0
            for record in records:
                try:
                    sid = int(record['id'])
                    pos = np.array([
                        float(record['map_x'] if 'map_x' in record else record['x']),
                        float(record['map_y'] if 'map_y' in record else record['y']),
                        float(record.get('map_z', 0.9)),
                    ], dtype=np.float64)
                    if not np.all(np.isfinite(pos)):
                        continue
                    survivor = UniqueSurvivor(
                        sid=sid, pos_map=pos,
                        confidence=float(record.get('confidence', 80)) / 100.0,
                        timestamp=float(record.get('first_seen', now)),
                        first_cam_point=(0.0, 0.0, 0.0))
                    survivor.confirmed = True
                    survivor.hits = max(1, int(record.get('observations', 1)))
                    survivor.last_seen = float(record.get('last_seen', now))
                    self.confirmed_survivors.append(survivor)
                    max_id = max(max_id, sid)
                    restored += 1
                except (KeyError, TypeError, ValueError):
                    continue
            self._merge_confirmed()
            self.next_survivor_id = max(max_id + 1, 1)
        return restored


def is_close_range_detection(cam_point: Tuple[float, float, float],
                             confirm_range: float) -> bool:
    """Only close, finite RGB-D observations may enter the identity registry."""
    xc, _, zc = cam_point
    distance = math.hypot(float(xc), float(zc))
    return bool(np.isfinite(distance) and 0.20 <= zc and distance <= confirm_range)


def rgbd_frames_are_synchronized(image_stamp_s: Optional[float],
                                 depth_stamp_s: Optional[float],
                                 tolerance_s: float) -> bool:
    """True only when RGB and depth describe the same camera instant."""
    if image_stamp_s is None or depth_stamp_s is None:
        return False
    return abs(float(image_stamp_s) - float(depth_stamp_s)) <= float(tolerance_s)


def is_registration_quality(cand: Dict, pt_map: np.ndarray, confirm_range: float,
                            min_confidence: float = 0.60,
                            min_centroid_height: float = 0.75,
                            max_centroid_height: float = 1.80) -> bool:
    """Gate permanent identities using classification and 3-D evidence."""
    if not is_close_range_detection(cand['cam_point'], confirm_range):
        return False
    if float(cand.get('confidence', 0.0)) < min_confidence:
        return False
    if not cand.get('depth_verified', False) or not cand.get('freestanding', False):
        return False
    z_map = float(pt_map[2])
    return bool(np.isfinite(z_map) and min_centroid_height <= z_map <= max_centroid_height)


def is_compact_map_object(grid: np.ndarray, resolution: float,
                          origin_x: float, origin_y: float,
                          world_x: float, world_y: float,
                          window_radius_m: float = 0.80,
                          seed_radius_m: float = 0.40,
                          max_component_radius_m: float = 0.60) -> bool:
    """Reject wall-connected detections using only the live occupancy map.

    A survivor collision is a compact occupied island. A wall/door edge belongs
    to an occupied component that extends well beyond the candidate. This test
    deliberately uses no world file or known survivor coordinates.
    """
    if grid is None or resolution <= 0.0 or grid.ndim != 2:
        return False
    row = int(math.floor((world_y - origin_y) / resolution))
    col = int(math.floor((world_x - origin_x) / resolution))
    radius_px = max(2, int(math.ceil(window_radius_m / resolution)))
    seed_px = max(1, int(math.ceil(seed_radius_m / resolution)))
    h, w = grid.shape
    if not (0 <= row < h and 0 <= col < w):
        return False

    r0, r1 = max(0, row - radius_px), min(h, row + radius_px + 1)
    c0, c1 = max(0, col - radius_px), min(w, col + radius_px + 1)
    occupied = grid[r0:r1, c0:c1] >= 50
    if occupied.size == 0:
        return False

    local_row, local_col = row - r0, col - c0
    occ_rc = np.argwhere(occupied)
    distances = np.hypot(occ_rc[:, 0] - local_row, occ_rc[:, 1] - local_col)
    seed_candidates = occ_rc[distances <= seed_px]
    if len(seed_candidates) == 0:
        return False
    seed_dist = np.hypot(seed_candidates[:, 0] - local_row,
                         seed_candidates[:, 1] - local_col)
    seed = tuple(seed_candidates[int(np.argmin(seed_dist))])

    # Flood-fill the occupied component attached to the candidate.
    stack = [seed]
    visited = {seed}
    while stack:
        rr, cc = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = rr + dr, cc + dc
                key = (nr, nc)
                if (0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1]
                        and occupied[nr, nc] and key not in visited):
                    visited.add(key)
                    stack.append(key)

    component = np.asarray(list(visited), dtype=np.float64)
    if len(component) < 2:
        return False
    radial_extent = float(np.max(np.hypot(
        component[:, 0] - seed[0], component[:, 1] - seed[1]))) * resolution
    if radial_extent > max_component_radius_m:
        return False

    # A compact body should not form a long, nearly one-dimensional segment.
    if len(component) >= 6:
        cov = np.cov(component, rowvar=False)
        eig = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))
        if eig[1] > 1e-6:
            linearity = float((eig[1] - eig[0]) / (eig[1] + eig[0] + 1e-9))
            span = float(np.max(np.ptp(component, axis=0))) * resolution
            if linearity > 0.82 and span > 0.40:
                return False
    return True


# ==============================================================================
# SECTION 3: UNIFIED ROS 2 PYTHON NODE
# ==============================================================================

class NidarVisionTrackerNode(Node):
    """
    Unified ROS 2 Node combining MannequinPerceptionEngine, 360° LiDAR Prospector,
    Active Visual Inspection, and SpatialSurvivorTracker.
    """
    def __init__(self):
        super().__init__('nidar_vision_tracker_node')
        self.get_logger().info('Initializing NIDAR Production-Ready Survivor Perception & Spatial Tracker...')

        # ----------------------------------------------------------------------
        # 1. Declare Configurable ROS 2 Parameters
        # ----------------------------------------------------------------------
        self.declare_parameter('distance_threshold', 1.80)     # 1.8m Euclidean deduplication
        self.declare_parameter('confirm_range', 2.60)          # Confirmation requires <= 2.6m
        self.declare_parameter('inspection_standoff', 1.35)    # Safe distance from unconfirmed object
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_frame', 'camera_optical_frame')
        self.declare_parameter('image_topic', '/nidar/stereo_camera/image')
        self.declare_parameter('depth_topic', '/nidar/stereo_camera/depth_image')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('min_confirm_hits', 3)
        self.declare_parameter('min_registration_confidence', 0.60)
        self.declare_parameter('min_survivor_centroid_height', 0.75)
        self.declare_parameter('max_survivor_centroid_height', 1.80)
        self.declare_parameter('registry_path', '/tmp/nidar_survivor_registry.json')
        self.declare_parameter('detector_mode', 'person')
        self.declare_parameter('rolling_alpha', 0.15)
        self.declare_parameter('camera_fx', 363.0)
        self.declare_parameter('camera_fy', 363.0)
        self.declare_parameter('camera_cx', 320.0)
        self.declare_parameter('camera_cy', 240.0)
        self.declare_parameter('rgbd_sync_tolerance_s', 0.08)
        self.declare_parameter('processing_rate_hz', 5.0)

        # Retrieve parameter values
        self.dist_thresh = float(self.get_parameter('distance_threshold').value)
        self.confirm_range = float(self.get_parameter('confirm_range').value)
        self.standoff = float(self.get_parameter('inspection_standoff').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.cam_frame = str(self.get_parameter('camera_frame').value)
        self.img_topic = str(self.get_parameter('image_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.map_topic = str(self.get_parameter('map_topic').value)
        self.min_hits = int(self.get_parameter('min_confirm_hits').value)
        self.min_registration_confidence = float(
            self.get_parameter('min_registration_confidence').value)
        self.min_survivor_centroid_height = float(
            self.get_parameter('min_survivor_centroid_height').value)
        self.max_survivor_centroid_height = float(
            self.get_parameter('max_survivor_centroid_height').value)
        self.registry_path = str(self.get_parameter('registry_path').value)
        self.detector_mode = str(self.get_parameter('detector_mode').value)
        self.rolling_alpha = float(self.get_parameter('rolling_alpha').value)
        self.rgbd_sync_tolerance_s = max(0.01, float(
            self.get_parameter('rgbd_sync_tolerance_s').value))
        self.processing_rate_hz = max(2.0, float(
            self.get_parameter('processing_rate_hz').value))
        fx = float(self.get_parameter('camera_fx').value)
        fy = float(self.get_parameter('camera_fy').value)
        cx = float(self.get_parameter('camera_cx').value)
        cy = float(self.get_parameter('camera_cy').value)

        self.get_logger().info(
            f'Configuration loaded: MapFrame={self.map_frame} | CamFrame={self.cam_frame} | '
            f'DistanceThreshold={self.dist_thresh:.2f}m | ConfirmRange={self.confirm_range:.2f}m | '
            f'MinHits={self.min_hits} | Mode={self.detector_mode}'
        )

        # ----------------------------------------------------------------------
        # 2. Instantiate Perception Engine & Spatial Tracker
        # ----------------------------------------------------------------------
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'nidar_survivor_custom_detector.onnx'
        )
        self.perception = MannequinPerceptionEngine(
            fx=fx, fy=fy, cx=cx, cy=cy, custom_model_path=model_path
        )
        self.tracker = SpatialSurvivorTracker(
            distance_threshold=self.dist_thresh,
            min_confirm_hits=self.min_hits,
            rolling_alpha=self.rolling_alpha
        )
        self.last_sim_stamp = None
        self.pending_registry_records = None
        self.pending_registry_sim_stamp = None
        self._load_registry()

        # ----------------------------------------------------------------------
        # 3. TF2 Buffer & Listener for Coordinate Transformations
        # ----------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Drone Odometry / MAVROS pose state for guaranteed kinematics fallback
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_yaw = 0.0
        self.drone_pitch = 0.0
        self.drone_roll = 0.0
        self.first_pose_received = False
        self.pose_lock = threading.Lock()

        # Image Caches & Timers
        self.cv_bridge = CvBridge()
        self.latest_bgr_img = None
        self.latest_depth_img = None
        self.latest_image_stamp = None
        self.latest_image_stamp_s = None
        self.latest_depth_stamp_s = None
        self.latest_capture_pose = None
        self.latest_image_token = 0
        self.last_processed_token = -1
        self.img_lock = threading.Lock()
        self.inspection_cooldowns = {}
        self.last_scan_time = 0.0
        self.last_registry_save = 0.0
        self.map_lock = threading.Lock()
        self.occupancy_grid = None
        self.map_resolution = 0.0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0

        # ----------------------------------------------------------------------
        # 4. ROS 2 Publishers
        # ----------------------------------------------------------------------
        marker_qos = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.pub_markers = self.create_publisher(MarkerArray, '/nidar/survivor_markers', marker_qos)
        self.pub_count = self.create_publisher(Int32, '/nidar/survivor_count', 10)
        self.pub_detailed = self.create_publisher(String, '/nidar/survivors_detailed', 10)
        self.pub_hud = self.create_publisher(Image, '/nidar/camera_view', 10)
        self.pub_inspection = self.create_publisher(PoseStamped, '/nidar/inspection_target', 10)

        # ----------------------------------------------------------------------
        # 5. ROS 2 Subscriptions
        # ----------------------------------------------------------------------
        self.sub_cam_info = self.create_subscription(
            CameraInfo, '/nidar/camera_info', self.cb_cam_info, 10
        )
        self.sub_img = self.create_subscription(
            Image, self.img_topic, self.cb_ros_image, qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.cb_ros_depth, qos_profile_sensor_data
        )
        self.sub_scan = self.create_subscription(
            LaserScan, self.scan_topic, self.cb_scan, qos_profile_sensor_data
        )
        self.sub_map = self.create_subscription(
            OccupancyGrid, self.map_topic, self.cb_map, 10
        )
        self.sub_relaunch = self.create_subscription(
            Bool, '/nidar/relaunch_command', self.cb_relaunch, 10
        )
        self.sub_pose = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.cb_pose, 10
        )

        # Gazebo Transport Fallback (handles native Gazebo image topics)
        if GzNode is not None:
            try:
                self.gz_node = GzNode()
                self.gz_node.subscribe(GzImage, '/nidar/stereo_camera/image', self.cb_gz_stereo_image)
                self.gz_node.subscribe(GzImage, '/nidar/stereo_camera/depth_image', self.cb_gz_depth)
                self.gz_node.subscribe(GzImage, '/nidar/camera', self.cb_gz_wide_image)
                self.get_logger().info('Gazebo Transport fallback bridges initialized (stereo RGB-D + wide).')
            except Exception as e:
                self.get_logger().warn(f'Gazebo Transport bridge initialization skipped: {e}')

        # Five synchronized inferences per second are sufficient for the
        # three-frame confirmation gate and leave Nav2 enough CPU to maintain
        # its controller rate.  This remains configurable for faster hardware.
        self.timer = self.create_timer(
            1.0 / self.processing_rate_hz, self.process_detection_cycle)

    # --------------------------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------------------------
    def cb_pose(self, msg: PoseStamped):
        """Maintains instantaneous drone pose & heading for rock-solid coordinate projection."""
        q = msg.pose.orientation
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        with self.pose_lock:
            self.drone_x = float(msg.pose.position.x)
            self.drone_y = float(msg.pose.position.y)
            self.drone_z = float(msg.pose.position.z)
            self.drone_yaw = yaw
            self.drone_pitch = pitch
            self.drone_roll = roll
            self.first_pose_received = True

    def cb_map(self, msg: OccupancyGrid):
        """Cache the live SLAM grid for wall-connected false-positive vetoing."""
        try:
            grid = np.asarray(msg.data, dtype=np.int16).reshape(
                (msg.info.height, msg.info.width))
        except (ValueError, TypeError):
            return
        with self.map_lock:
            self.occupancy_grid = grid
            self.map_resolution = float(msg.info.resolution)
            self.map_origin_x = float(msg.info.origin.position.x)
            self.map_origin_y = float(msg.info.origin.position.y)

    def _map_context_allows_registration(self, pt_map: np.ndarray) -> bool:
        with self.map_lock:
            grid = None if self.occupancy_grid is None else self.occupancy_grid.copy()
            resolution = self.map_resolution
            origin_x = self.map_origin_x
            origin_y = self.map_origin_y
        return is_compact_map_object(
            grid, resolution, origin_x, origin_y,
            float(pt_map[0]), float(pt_map[1]))

    def cb_cam_info(self, msg: CameraInfo):
        """Dynamically updates camera intrinsics matrix if available."""
        if len(msg.k) >= 9 and msg.k[0] > 0.0:
            self.perception.update_intrinsics(
                fx=float(msg.k[0]), fy=float(msg.k[4]), cx=float(msg.k[2]), cy=float(msg.k[5])
            )

    def cb_ros_image(self, msg: Image):
        try:
            bgr = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            capture_pose = self._capture_pose_snapshot()
            with self.img_lock:
                self.latest_bgr_img = bgr
                self.latest_image_stamp = msg.header.stamp
                self.latest_image_stamp_s = (
                    float(msg.header.stamp.sec) +
                    float(msg.header.stamp.nanosec) * 1e-9)
                self.latest_capture_pose = capture_pose
                self.latest_image_token += 1
        except Exception:
            pass

    def cb_ros_depth(self, msg: Image):
        try:
            if msg.encoding in ('32FC1', '32f'):
                depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            elif msg.encoding in ('16UC1', '16u'):
                depth_16u = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                depth = depth_16u.astype(np.float32) / 1000.0
            else:
                depth = self.cv_bridge.imgmsg_to_cv2(msg)
            with self.img_lock:
                self.latest_depth_img = depth
                self.latest_depth_stamp_s = (
                    float(msg.header.stamp.sec) +
                    float(msg.header.stamp.nanosec) * 1e-9)
        except Exception:
            pass

    def _decode_gz_image(self, msg: GzImage):
        """Helper to decode raw Gazebo Transport Image bytes to OpenCV BGR."""
        total = msg.height * msg.width
        if total <= 0 or len(msg.data) < total:
            return None
        channels = len(msg.data) // total
        try:
            if channels == 4:
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 4))
                return cv2.cvtColor(raw[:, :, :3], cv2.COLOR_RGB2BGR)
            elif channels == 3:
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        except Exception:
            return None
        return None

    def cb_gz_stereo_image(self, msg: GzImage):
        """Decodes 640x480 RGB member of the RGB-D sensor."""
        bgr = self._decode_gz_image(msg)
        if bgr is not None:
            sim_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nsec) * 1e-9
            self._observe_sim_stamp(sim_stamp)
            capture_pose = self._capture_pose_snapshot()
            with self.img_lock:
                self.latest_bgr_img = bgr
                self.latest_image_stamp = rclpy.time.Time(
                    seconds=msg.header.stamp.sec,
                    nanoseconds=msg.header.stamp.nsec
                ).to_msg()
                self.latest_image_stamp_s = sim_stamp
                self.latest_capture_pose = capture_pose
                self.latest_image_token += 1

    def cb_gz_wide_image(self, msg: GzImage):
        """Decodes 1280x720 wide camera as secondary fallback."""
        bgr = self._decode_gz_image(msg)
        if bgr is not None:
            capture_pose = self._capture_pose_snapshot()
            with self.img_lock:
                if self.latest_bgr_img is None:
                    sim_stamp = (float(msg.header.stamp.sec) +
                                 float(msg.header.stamp.nsec) * 1e-9)
                    self.latest_bgr_img = bgr
                    self.latest_image_stamp = rclpy.time.Time(
                        seconds=msg.header.stamp.sec,
                        nanoseconds=msg.header.stamp.nsec
                    ).to_msg()
                    self.latest_image_stamp_s = sim_stamp
                    self.latest_capture_pose = capture_pose
                    self.latest_image_token += 1

    def _capture_pose_snapshot(self):
        """Freeze vehicle pose when a frame arrives, independent of clock domain."""
        with self.pose_lock:
            if not self.first_pose_received:
                return None
            return (self.drone_x, self.drone_y, self.drone_z, self.drone_yaw)

    def cb_gz_image(self, msg: GzImage):
        self.cb_gz_stereo_image(msg)

    def cb_gz_depth(self, msg: GzImage):
        """Decodes raw Gazebo Transport Depth Image message."""
        total = msg.height * msg.width
        if total <= 0 or len(msg.data) < total:
            return
        try:
            if len(msg.data) == total * 4:
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            elif len(msg.data) == total * 2:
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)).astype(np.float32) / 1000.0
            else:
                return
            with self.img_lock:
                self.latest_depth_img = depth
                self.latest_depth_stamp_s = (
                    float(msg.header.stamp.sec) +
                    float(msg.header.stamp.nsec) * 1e-9)
        except Exception:
            pass

    def cb_relaunch(self, msg: Bool):
        if msg.data:
            self.tracker.clear()
            self.inspection_cooldowns.clear()
            try:
                if self.registry_path and os.path.exists(self.registry_path):
                    os.unlink(self.registry_path)
            except OSError as exc:
                self.get_logger().warning(f'Could not clear survivor registry: {exc}')
            self.get_logger().info('🔄 [RELAUNCH] Cleared all master survivor tracks.')
            self._publish_ros_state([])

    def _load_registry(self):
        """Stage identities until Gazebo proves this is the same map session."""
        if not self.registry_path or not os.path.exists(self.registry_path):
            return
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            records = payload.get('survivors', payload) if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                raise ValueError('registry must contain a survivor list')
            saved_sim_stamp = payload.get('sim_stamp') if isinstance(payload, dict) else None
            if saved_sim_stamp is None:
                self.get_logger().warning(
                    'Ignoring legacy survivor registry without a simulation-session stamp')
                return
            self.pending_registry_records = records
            self.pending_registry_sim_stamp = float(saved_sim_stamp)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f'Ignoring invalid survivor registry: {exc}')

    def _observe_sim_stamp(self, sim_stamp: float):
        """Restore only within one Gazebo run; clear identities when time resets."""
        if not np.isfinite(sim_stamp):
            return
        if self.last_sim_stamp is not None and sim_stamp < self.last_sim_stamp - 2.0:
            self.tracker.clear()
            self.pending_registry_records = None
            self.pending_registry_sim_stamp = None
            self.get_logger().warning(
                'Gazebo clock reset detected; cleared survivor identities from the old map session')
        self.last_sim_stamp = sim_stamp

        if self.pending_registry_records is not None:
            # Same run advances monotonically. A smaller clock means a new world.
            if sim_stamp + 2.0 >= self.pending_registry_sim_stamp:
                restored = self.tracker.restore(self.pending_registry_records)
                self.get_logger().info(
                    f'🗃️ Restored {restored} survivor identities for the current Gazebo session')
            else:
                self.get_logger().warning(
                    'Discarded survivor registry from a previous Gazebo/map session')
            self.pending_registry_records = None
            self.pending_registry_sim_stamp = None

    def _save_registry(self, force: bool = False):
        """Atomically persist identities so a node restart cannot duplicate them."""
        now = time.time()
        if not self.registry_path or (not force and now - self.last_registry_save < 1.0):
            return
        tmp_path = self.registry_path + '.tmp'
        try:
            parent = os.path.dirname(self.registry_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            payload = {
                'version': 1,
                'map_frame': self.map_frame,
                'sim_stamp': self.last_sim_stamp,
                'saved_at': now,
                'survivors': self.tracker.snapshot(),
            }
            with open(tmp_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.registry_path)
            self.last_registry_save = now
        except OSError as exc:
            self.get_logger().warning(f'Could not persist survivor registry: {exc}')

    # --------------------------------------------------------------------------
    # 360° LiDAR Prospecting & Active Visual Inspection
    # --------------------------------------------------------------------------
    def cb_scan(self, msg: LaserScan):
        """
        Wide-area 360° LiDAR prospector.
        Finds human-sized cylinder clusters and requests active camera inspection.
        """
        now = time.time()
        if (now - self.last_scan_time) < 0.35:
            return
        self.last_scan_time = now

        # Extract robot pose from TF
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, 'base_link', rclpy.time.Time())
            rx = t.transform.translation.x
            ry = t.transform.translation.y
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            ryaw = math.atan2(siny, cosy)
        except Exception:
            return

        ranges = np.array(msg.ranges, dtype=np.float32)
        valid = np.isfinite(ranges) & (ranges > 0.40) & (ranges < 6.0)
        indices = np.where(valid)[0]
        if len(indices) < 8:
            return

        diffs = np.diff(indices)
        splits = np.where(diffs > 2)[0] + 1
        clusters = np.split(indices, splits)

        for c_idx in clusters:
            if len(c_idx) < 4 or len(c_idx) > 50:
                continue
            c_ranges = ranges[c_idx]
            if (np.max(c_ranges) - np.min(c_ranges)) > 0.45:
                continue

            angles = msg.angle_min + c_idx * msg.angle_increment
            pts_x = c_ranges * np.cos(angles)
            pts_y = c_ranges * np.sin(angles)
            pts = np.column_stack([pts_x, pts_y])

            # Linearity check: if linearity > 0.92, it is a flat wall!
            cov = np.cov(pts, rowvar=False)
            if cov.shape == (2, 2):
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = np.sort(np.maximum(eigvals, 0.0))
                if (eigvals[0] + eigvals[1]) > 1e-6:
                    linearity = (eigvals[1] - eigvals[0]) / (eigvals[1] + eigvals[0])
                    if linearity > 0.92:
                        continue

            cx = float(np.mean(pts_x))
            cy = float(np.mean(pts_y))
            dist = math.hypot(cx, cy)
            width = float(np.linalg.norm(pts[0] - pts[-1]))
            if not (0.15 <= width <= 0.70):
                continue

            # Transform prospect to map frame
            wx = rx + cx * math.cos(ryaw) - cy * math.sin(ryaw)
            wy = ry + cx * math.sin(ryaw) + cy * math.cos(ryaw)

            # Check if this prospect is within distance_threshold of ANY confirmed survivor
            with self.tracker.lock:
                already_confirmed = any(
                    np.linalg.norm(np.array([wx, wy]) - s.anchor[:2]) <= self.dist_thresh
                    for s in self.tracker.confirmed_survivors
                )
            if already_confirmed:
                continue  # ALREADY LOCKED & CONFIRMED: NEVER RE-INSPECT!

            # If far (> 2.0m), request camera inspection approach via /nidar/inspection_target
            if dist > 2.0:
                cooldown_key = (round(wx * 2.0) / 2.0, round(wy * 2.0) / 2.0)
                if (now - self.inspection_cooldowns.get(cooldown_key, 0.0)) >= 4.0:
                    self.inspection_cooldowns[cooldown_key] = now
                    msg_inspect = PoseStamped()
                    msg_inspect.header.frame_id = self.map_frame
                    msg_inspect.header.stamp = self.get_clock().now().to_msg()
                    msg_inspect.pose.position.x = float(wx)
                    msg_inspect.pose.position.y = float(wy)
                    msg_inspect.pose.position.z = float(dist)
                    msg_inspect.pose.orientation.w = 1.0
                    self.pub_inspection.publish(msg_inspect)
                    self.get_logger().info(
                        f"🔎 [ACTIVE INSPECTION] 360° LiDAR prospect at ({wx:+.2f}, {wy:+.2f}), "
                        f"{dist:.2f}m away. Requesting approach to confirm."
                    )

    # --------------------------------------------------------------------------
    # Core Vision Processing Loop
    # --------------------------------------------------------------------------
    def process_detection_cycle(self):
        """
        Executes perception, transforms coordinates via TF2, updates spatial tracker,
        and broadcasts authoritative ROS 2 topics.
        """
        now = time.time()
        with self.img_lock:
            bgr = None if self.latest_bgr_img is None else self.latest_bgr_img.copy()
            depth = None if self.latest_depth_img is None else self.latest_depth_img.copy()
            image_stamp = self.latest_image_stamp
            image_stamp_s = self.latest_image_stamp_s
            depth_stamp_s = self.latest_depth_stamp_s
            capture_pose = self.latest_capture_pose
            image_token = self.latest_image_token

        if bgr is None or image_token == self.last_processed_token:
            return
        # Never apply depth from a previous heading to the current RGB frame.
        # During an in-place camera sweep that mismatch looks like a person in
        # RGB but like a wall/floor in depth—the exact intermittent "shadow"
        # rejection observed late in a mission.  Do not consume the RGB token;
        # its matching depth callback may arrive before the next timer tick.
        if depth is None or not rgbd_frames_are_synchronized(
                image_stamp_s, depth_stamp_s, self.rgbd_sync_tolerance_s):
            return
        self.last_processed_token = image_token

        # 1. Run Computer Vision Perception Engine
        candidates = self.perception.detect(bgr, depth_img=depth, mode=self.detector_mode)

        # 2. Transform Camera Frame Detections into Global MAP Frame
        map_points = []
        for cand in candidates:
            xc, yc, zc = cand['cam_point']
            conf = cand['confidence']

            pt_map = self._transform_camera_to_map(
                xc, yc, zc, image_stamp, capture_pose=capture_pose)
            if pt_map is None:
                continue

            dist = math.hypot(xc, zc)

            # A distant person is an approach cue, never a permanent identity.
            # Registration happens only after close RGB-D re-observation.
            if is_registration_quality(
                    cand, pt_map, self.confirm_range,
                    min_confidence=self.min_registration_confidence,
                    min_centroid_height=self.min_survivor_centroid_height,
                    max_centroid_height=self.max_survivor_centroid_height) and \
                    self._map_context_allows_registration(pt_map):
                map_points.append((pt_map, conf, (xc, yc, zc)))

            # If far candidate (> 2.6m), trigger active inspection approach if not yet confirmed
            if dist > self.confirm_range:
                with self.tracker.lock:
                    already_known = any(
                        np.linalg.norm(pt_map[:2] - s.anchor[:2]) <= self.dist_thresh
                        for s in self.tracker.confirmed_survivors
                    )
                if not already_known:
                    cooldown_key = (round(pt_map[0] * 2.0) / 2.0, round(pt_map[1] * 2.0) / 2.0)
                    if (now - self.inspection_cooldowns.get(cooldown_key, 0.0)) >= 4.0:
                        self.inspection_cooldowns[cooldown_key] = now
                        msg_inspect = PoseStamped()
                        msg_inspect.header.frame_id = self.map_frame
                        msg_inspect.header.stamp = self.get_clock().now().to_msg()
                        msg_inspect.pose.position.x = float(pt_map[0])
                        msg_inspect.pose.position.y = float(pt_map[1])
                        msg_inspect.pose.position.z = float(dist)
                        msg_inspect.pose.orientation.w = 1.0
                        self.pub_inspection.publish(msg_inspect)
                        self.get_logger().info(
                            f"🔎 [ACTIVE INSPECTION] Far visual candidate at ({pt_map[0]:+.2f}, {pt_map[1]:+.2f}), "
                            f"{dist:.2f}m away. Requesting approach."
                        )

        # 3. Update Spatial Tracker with Rolling Average & Duplicate Suppression
        confirmed_list, newly_confirmed = self.tracker.update(map_points, timestamp=now)

        # Log newly confirmed unique survivors
        for new_s in newly_confirmed:
            self.get_logger().info(
                f'🚨 [SPATIAL TRACKER] 🎯 UNIQUE SURVIVOR CONFIRMED: ID #{new_s.id} | '
                f'Map: ({new_s.pos[0]:+.2f}, {new_s.pos[1]:+.2f}, {new_s.pos[2]:+.2f}) | '
                f'Confidence: {new_s.confidence*100:.0f}% | Master Total: {len(confirmed_list)}'
            )

        if newly_confirmed or confirmed_list:
            self._save_registry(force=bool(newly_confirmed))

        # 4. Broadcast Authoritative ROS 2 State & Visualizations
        self._publish_ros_state(confirmed_list)
        self._render_and_publish_hud(bgr, candidates, confirmed_list)

    def _transform_camera_to_map(self, xc: float, yc: float, zc: float,
                                 image_stamp=None, capture_pose=None) -> Optional[np.ndarray]:
        """
        Transform the observation at image time. A stale/latest pose is not safe
        for permanent map identity because drone motion can shift one person by
        metres between capture and processing.
        """
        # Timestamped ROS/Gazebo images must use their matching transform.
        try:
            pt_stamped = PointStamped()
            pt_stamped.header.frame_id = self.cam_frame
            pt_stamped.header.stamp = (image_stamp if image_stamp is not None
                                       else rclpy.time.Time().to_msg())
            pt_stamped.point.x = float(xc)
            pt_stamped.point.y = float(yc)
            pt_stamped.point.z = float(zc)

            transformed = self.tf_buffer.transform(
                pt_stamped, self.map_frame, timeout=rclpy.duration.Duration(seconds=0.05)
            )
            return np.array([
                transformed.point.x,
                transformed.point.y,
                transformed.point.z
            ], dtype=np.float64)
        except Exception:
            pass

        # Gazebo transport commonly stamps frames in simulation time while ROS
        # TF uses wall time. Use the pose frozen in the image callback—not the
        # newer processing-time pose—when the TF clock domains cannot match.
        if capture_pose is not None:
            dx, dy, dz, yaw = capture_pose
        else:
            with self.pose_lock:
                if not self.first_pose_received:
                    return None
                dx = self.drone_x
                dy = self.drone_y
                dz = self.drone_z
                yaw = self.drone_yaw

        # Camera optical frame: Z forward, X right, Y down
        # Drone body frame: X forward, Y left, Z up. Camera offset: (0.17, 0.0, 0.025)
        x_body = 0.17 + float(zc)
        y_body = 0.0 - float(xc)
        z_body = 0.025 - float(yc)

        # Rotate body to map by drone yaw
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        map_x = dx + (x_body * cos_y - y_body * sin_y)
        map_y = dy + (x_body * sin_y + y_body * cos_y)
        map_z = dz + z_body

        return np.array([map_x, map_y, map_z], dtype=np.float64)

    # --------------------------------------------------------------------------
    # Publishing & Visualization
    # --------------------------------------------------------------------------
    def _publish_ros_state(self, confirmed_survivors: List[UniqueSurvivor]):
        """Publishes Int32 count, JSON details, and 3D MarkerArray."""
        # 1. /nidar/survivor_count
        cnt_msg = Int32()
        cnt_msg.data = len(confirmed_survivors)
        self.pub_count.publish(cnt_msg)

        # 2. /nidar/survivors_detailed (authoritative JSON array)
        detailed_list = [s.to_dict() for s in confirmed_survivors]
        det_msg = String()
        det_msg.data = json.dumps(detailed_list)
        self.pub_detailed.publish(det_msg)

        # 3. /nidar/survivor_markers (3D RViz & GCS Visualization)
        marker_arr = MarkerArray()
        now_msg = self.get_clock().now().to_msg()

        # RViz retains markers by (namespace, id). An empty/shorter array does
        # not erase old IDs, so explicitly clear ghosts after restart, registry
        # merge, or mission reset before publishing the authoritative state.
        clear_marker = Marker()
        clear_marker.header.frame_id = self.map_frame
        clear_marker.header.stamp = now_msg
        clear_marker.action = Marker.DELETEALL
        marker_arr.markers.append(clear_marker)

        for s in confirmed_survivors:
            sid = s.id
            sx, sy, sz = s.pos

            # Ground ring disc (Marker.CYLINDER, type 0)
            m_disc = Marker()
            m_disc.header.frame_id = self.map_frame
            m_disc.header.stamp = now_msg
            m_disc.ns = 'survivors'
            m_disc.id = sid * 10 + 0
            m_disc.type = Marker.CYLINDER
            m_disc.action = Marker.ADD
            m_disc.pose.position.x = float(sx)
            m_disc.pose.position.y = float(sy)
            m_disc.pose.position.z = 0.08
            m_disc.pose.orientation.w = 1.0
            m_disc.scale.x = 1.00
            m_disc.scale.y = 1.00
            m_disc.scale.z = 0.15
            m_disc.color.r = 1.0
            m_disc.color.g = 0.0
            m_disc.color.b = 0.0
            m_disc.color.a = 0.90
            marker_arr.markers.append(m_disc)

            # Torso marker (Marker.CUBE, type 1)
            m_body = Marker()
            m_body.header.frame_id = self.map_frame
            m_body.header.stamp = now_msg
            m_body.ns = 'survivors'
            m_body.id = sid * 10 + 1
            m_body.type = Marker.CUBE
            m_body.action = Marker.ADD
            m_body.pose.position.x = float(sx)
            m_body.pose.position.y = float(sy)
            m_body.pose.position.z = 0.90
            m_body.pose.orientation.w = 1.0
            m_body.scale.x = 0.50
            m_body.scale.y = 0.50
            m_body.scale.z = 1.70
            m_body.color.r = 1.0
            m_body.color.g = 0.40
            m_body.color.b = 0.0
            m_body.color.a = 0.95
            marker_arr.markers.append(m_body)

            # Floating 3D Text Label (Marker.TEXT_VIEW_FACING, type 2)
            m_text = Marker()
            m_text.header.frame_id = self.map_frame
            m_text.header.stamp = now_msg
            m_text.ns = 'survivors_text'
            m_text.id = sid * 10 + 2
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x = float(sx)
            m_text.pose.position.y = float(sy)
            m_text.pose.position.z = 2.10
            m_text.pose.orientation.w = 1.0
            m_text.scale.z = 0.35
            m_text.color.r = 1.0
            m_text.color.g = 1.0
            m_text.color.b = 1.0
            m_text.color.a = 1.0
            m_text.text = f"SURVIVOR #{sid}\n({sx:+.2f}, {sy:+.2f})\nConf: {s.confidence*100:.0f}%\nHits: {s.hits}"
            marker_arr.markers.append(m_text)

        # Also publish active candidates as translucent warning markers
        with self.tracker.lock:
            active_cands = list(self.tracker.candidate_tracks)

        for i, c in enumerate(active_cands):
            cid = i + 1
            cx, cy, cz = c.pos
            m_cand = Marker()
            m_cand.header.frame_id = self.map_frame
            m_cand.header.stamp = now_msg
            m_cand.ns = 'candidate_prospects'
            m_cand.id = 500 + cid
            m_cand.type = Marker.CYLINDER
            m_cand.action = Marker.ADD
            m_cand.pose.position.x = float(cx)
            m_cand.pose.position.y = float(cy)
            m_cand.pose.position.z = 0.04
            m_cand.pose.orientation.w = 1.0
            m_cand.scale.x = 0.80
            m_cand.scale.y = 0.80
            m_cand.scale.z = 0.08
            m_cand.color.r = 1.0
            m_cand.color.g = 0.80
            m_cand.color.b = 0.0
            m_cand.color.a = 0.50
            marker_arr.markers.append(m_cand)

        self.pub_markers.publish(marker_arr)

    def _render_and_publish_hud(self, bgr_img: np.ndarray, candidates: List[Dict],
                                confirmed_survivors: List[UniqueSurvivor]):
        """Renders real-time HUD view with bounding boxes and survivor IDs."""
        annotated = bgr_img.copy()
        h, w = annotated.shape[:2]

        title = f"🛡️ NIDAR SPATIAL TRACKER | UNIQUE SURVIVORS: {len(confirmed_survivors)}"
        cv2.rectangle(annotated, (12, 10), (450, 36), (15, 23, 42), -1)
        cv2.rectangle(annotated, (12, 10), (450, 36), (16, 185, 129), 1)
        cv2.putText(annotated, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (52, 211, 153), 1, cv2.LINE_AA)

        for cand in candidates:
            x1, y1, x2, y2 = cand['bbox']
            conf = cand['confidence']
            xc, yc, zc = cand['cam_point']
            method = cand['method']

            color = (0, 255, 110) if 'color' in method else ((255, 180, 0) if 'ssd' in method else (0, 210, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            lbl = f"{method[:10].upper()} {conf*100:.0f}% | {zc:.1f}m"
            cv2.putText(annotated, lbl, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        try:
            ros_img = self.cv_bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.header.frame_id = self.cam_frame
            self.pub_hud.publish(ros_img)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = NidarVisionTrackerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

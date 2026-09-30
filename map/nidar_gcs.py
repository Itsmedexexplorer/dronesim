#!/usr/bin/env python3
"""
NIDAR Spatial Ground Control Station (GCS)
===========================================
Apple Vision Pro Inspired Spatial-Computing Architecture:
  • Map-First Hero Composition:
      - 2D Tactical SLAM Map is the primary visual centerpiece
      - Deep obsidian canvas (#07080c)
      - Luminous silver-grey explored free space (#f1f5f9)
      - Solid onyx obstacle boundaries (#020617)
      - Translucent cyan flight history & emerald Nav2 planned path
      - Fixed persistent map-frame survivor markers (never drift with drone)
  • Floating Translucent Spatial Panels:
      - Top Capsule Island: Mission timer, coverage %, survivor counter, action dock
      - Left Telemetry Card: Altitude, heading compass, coordinates, 360° LiDAR radar, rotor dynamics
      - Right Survivor Intel Card: Dedicated cards formatted with:
            SURVIVOR #ID
            Status: CONFIRMED
            Map position: fixed (X, Y)
            Grid: Sector
            Confidence: XX%
      - Bottom Floating Shelf:
            • Spatial FPV Camera with HUD & Landing Standby Screen
            • JEV State Machine: OBSERVE → QUESTION → THINK → DECIDE → ACT → VERIFY
"""

import os
import sys
import time
import math
import json
import signal
import threading
import numpy as np
import cv2

# Clean exit handlers
def _clean_exit_handler(sig, frame):
    QApplication.quit()
    os._exit(0)

signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QProgressBar,
    QScrollArea, QSplitter, QFileDialog, QGraphicsDropShadowEffect,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF, QRect, pyqtSignal, QSize
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont,
    QPolygonF, QPainterPath
)

# ROS 2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Int32, Bool, String
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS & STYLESHEETS (APPLE VISION PRO AESTHETIC)
# ─────────────────────────────────────────────────────────────────────────────
SPATIAL_BG = "#07080c"
SPATIAL_FONT = "Noto Sans"          # metrically close, installed SF-style fallback
SPATIAL_MONO = "DejaVu Sans Mono"
GLASS_BG = "rgba(24, 26, 34, 0.82)"
GLASS_BORDER = "1px solid rgba(255, 255, 255, 0.13)"
GLASS_BORDER_ACTIVE = "1px solid rgba(56, 189, 248, 0.40)"
GLASS_RADIUS = "22px"

MAIN_STYLESHEET = f"""
QMainWindow {{
    background-color: {SPATIAL_BG};
}}
QWidget {{
    color: #f8fafc;
    font-family: "SF Pro Display", "SF Pro Text", "Noto Sans", "Segoe UI", sans-serif;
    font-size: 10pt;
    selection-background-color: rgba(10, 132, 255, 0.55);
}}
QToolTip {{
    color: #ffffff;
    background-color: rgba(38, 38, 42, 0.96);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 8px;
    padding: 6px 9px;
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: rgba(255, 255, 255, 0.03);
    width: 6px;
    border-radius: 3px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: rgba(255, 255, 255, 0.20);
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: rgba(56, 189, 248, 0.50);
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""

GLASS_CARD_STYLE = f"""
QFrame {{
    background-color: {GLASS_BG};
    border: {GLASS_BORDER};
    border-radius: {GLASS_RADIUS};
}}
"""

def make_shadow(radius=24, offset_y=8, opacity=120):
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(radius)
    shadow.setOffset(0, offset_y)
    shadow.setColor(QColor(0, 0, 0, opacity))
    return shadow

class ClickableSpatialCard(QFrame):
    clicked = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 TELEMETRY BACKEND
# ─────────────────────────────────────────────────────────────────────────────
class GcsRosBackend(Node):
    def __init__(self):
        super().__init__('nidar_spatial_gcs_backend')
        self.lock = threading.Lock()
        self.bridge = CvBridge()

        # Telemetry
        self.camera_frame = None
        self.lidar_ranges = None
        self.lidar_angle_min = -math.pi
        self.lidar_angle_inc = 0.0174

        self.drone_x = -5.80
        self.drone_y = -8.80
        self.drone_z = 0.00
        self.drone_roll = 0.0
        self.drone_pitch = 0.0
        self.drone_yaw = 1.5708
        self.climb_rate = 0.0
        self.speed = 0.0
        self.pose_speed = 0.0
        self.last_pose_t = None
        self.last_px = None
        self.last_py = None

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_vz = 0.0
        self.cmd_wz = 0.0

        self.explored_pct = 0.0
        self.unexplored_pct = 100.0
        self.explored_m2 = 0.0
        self.survivors_count = 0
        self.survivors_list = []
        self.survivors_detailed = []
        self.mission_complete = False
        self.start_time = None
        self.completion_time = None

        # 2D Map
        self.map_grid = None
        self.map_w = 0
        self.map_h = 0
        self.map_res = 0.05
        self.map_ox = -15.0
        self.map_oy = -15.0

        # Nav2 Path & Goal
        self.nav_path = []
        self.current_goal = None
        self.flight_trail = []
        self.jev_status = "OBSERVE: Initializing subsystems..."

        # Subscriptions
        self.sub_cam = self.create_subscription(Image, '/nidar/camera_view', self.cb_cam, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.cb_scan, 10)
        self.sub_pose = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.cb_pose, 10)
        self.sub_vel = self.create_subscription(TwistStamped, '/mavros/local_position/velocity_local', self.cb_vel, 10)
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel_nav', self.cb_cmd, 10)

        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=QoSReliabilityPolicy.RELIABLE)
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.cb_map, map_qos)
        self.sub_plan = self.create_subscription(Path, '/plan', self.cb_plan, 10)
        self.sub_goal = self.create_subscription(PoseStamped, '/nidar/current_goal', self.cb_goal, 10)

        self.sub_surv_cnt = self.create_subscription(Int32, '/nidar/survivor_count', self.cb_surv_cnt, 10)
        surv_qos = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=QoSReliabilityPolicy.RELIABLE)
        self.sub_surv_mrk = self.create_subscription(MarkerArray, '/nidar/survivor_markers', self.cb_surv_mrk, surv_qos)
        self.sub_surv_det = self.create_subscription(String, '/nidar/survivors_detailed', self.cb_surv_det, 10)
        self.sub_complete = self.create_subscription(Bool, '/nidar/mission_complete', self.cb_complete, 10)
        self.sub_jev_stat = self.create_subscription(String, '/nidar/jev_status', self.cb_jev_status, 10)

        # Publishers
        self.pub_land = self.create_publisher(Bool, '/nidar/land_command', 10)
        self.pub_relaunch = self.create_publisher(Bool, '/nidar/relaunch_command', 10)

    def cb_cam(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.camera_frame = cv_img
                if self.start_time is None and self.drone_z > 0.3:
                    self.start_time = time.time()
        except Exception:
            pass

    def cb_scan(self, msg: LaserScan):
        with self.lock:
            self.lidar_ranges = list(msg.ranges)
            self.lidar_angle_min = msg.angle_min
            self.lidar_angle_inc = msg.angle_increment

    def cb_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr, cosr)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)

        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z

        now = time.time()
        with self.lock:
            if self.last_pose_t is not None and self.last_px is not None:
                dt = now - self.last_pose_t
                if 0.005 <= dt <= 0.5:
                    d_pos = math.hypot(px - self.last_px, py - self.last_py)
                    instant_spd = d_pos / dt
                    self.pose_speed = 0.30 * instant_spd + 0.70 * self.pose_speed
                elif dt > 0.5:
                    self.pose_speed = 0.0
            self.last_pose_t = now
            self.last_px = px
            self.last_py = py

            self.drone_x = px
            self.drone_y = py
            self.drone_z = pz
            self.drone_roll = roll
            self.drone_pitch = pitch
            self.drone_yaw = yaw

            if self.start_time is None and self.drone_z > 0.35:
                self.start_time = now

            if len(self.flight_trail) == 0 or math.hypot(px - self.flight_trail[-1][0], py - self.flight_trail[-1][1]) > 0.15:
                self.flight_trail.append((px, py))

    def cb_vel(self, msg: TwistStamped):
        with self.lock:
            self.climb_rate = msg.twist.linear.z
            v_spd = math.hypot(msg.twist.linear.x, msg.twist.linear.y)
            if v_spd > 0.01:
                self.speed = v_spd
            else:
                self.speed = self.pose_speed

    def cb_cmd(self, msg: Twist):
        with self.lock:
            self.cmd_vx = msg.linear.x
            self.cmd_vy = msg.linear.y
            self.cmd_vz = msg.linear.z
            self.cmd_wz = msg.angular.z

    def cb_map(self, msg: OccupancyGrid):
        w, h = msg.info.width, msg.info.height
        res = msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        data = np.array(msg.data, dtype=np.int8).reshape((h, w))

        c0 = max(0, int((-7.5 - ox) / res))
        c1 = min(w, int((7.5 - ox) / res))
        r0 = max(0, int((-7.5 - oy) / res))
        r1 = min(h, int((7.5 - oy) / res))
        arena = data[r0:r1, c0:c1]

        with self.lock:
            self.map_grid = data
            self.map_w = w
            self.map_h = h
            self.map_res = res
            self.map_ox = ox
            self.map_oy = oy

            if arena.size > 0:
                free = np.sum(arena == 0)
                wall = np.sum(arena == 100)
                unexp = np.sum(arena == -1)
                total = arena.size
                self.explored_pct = ((free + wall) / float(total)) * 100.0
                self.unexplored_pct = (unexp / float(total)) * 100.0
                self.explored_m2 = (free + wall) * (res * res)

    def cb_plan(self, msg: Path):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self.lock:
            self.nav_path = pts

    def cb_goal(self, msg: PoseStamped):
        with self.lock:
            self.current_goal = (msg.pose.position.x, msg.pose.position.y)

    def cb_surv_cnt(self, msg: Int32):
        with self.lock:
            self.survivors_count = msg.data

    def cb_surv_det(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self.lock:
                self.survivors_detailed = data
        except Exception:
            pass

    def cb_surv_mrk(self, msg: MarkerArray):
        survivors = []
        for m in msg.markers:
            # Type 1 is body cube
            if m.ns == 'survivors' and (m.id % 10 == 1 or m.id % 5 == 1):
                sid = (m.id // 10) if (m.id >= 10) else ((m.id - 1) // 5 + 1)
                survivors.append({
                    'id': sid,
                    'x': m.pose.position.x,
                    'y': m.pose.position.y
                })
        with self.lock:
            self.survivors_list = survivors

    def cb_complete(self, msg: Bool):
        with self.lock:
            if msg.data and not self.mission_complete:
                self.mission_complete = True
                self.completion_time = time.time()
            elif not msg.data:
                self.mission_complete = False
                self.completion_time = None

    def cb_jev_status(self, msg: String):
        with self.lock:
            self.jev_status = msg.data

    def send_land_cmd(self):
        msg = Bool(data=True)
        self.pub_land.publish(msg)

    def send_relaunch_cmd(self):
        msg = Bool(data=True)
        self.pub_relaunch.publish(msg)
        with self.lock:
            self.mission_complete = False
            self.completion_time = None
            self.start_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# APPLE SPATIAL CAMERA VIEWER WIDGET (ZERO-INFLATION & FLICKER-FREE)
# ─────────────────────────────────────────────────────────────────────────────
class CameraViewerWidget(QWidget):
    """
    Apple Vision Pro Inspired Hardware-Accelerated Camera Viewer.
    - Zero layout inflation: paints directly in paintEvent with QPainter.
    - Anti-flicker double-buffered rendering with SmoothPixmapTransform.
    - Aspect ratio preserved via centered letterbox/pillarbox geometry.
    - Standby / Touchdown Verified HUD card dynamically drawn in vector graphics.
    - Hero HUD pill overlay in Camera Focus mode.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(180, 110)
        self._pixmap = None
        self._is_standby = False
        self._standby_info = {}
        self._hero_hud_data = None

    def sizeHint(self):
        return QSize(340, 195)

    def minimumSizeHint(self):
        return QSize(180, 110)

    def set_pixmap(self, pixmap: QPixmap, hero_hud_data=None):
        self._is_standby = False
        self._pixmap = pixmap
        self._hero_hud_data = hero_hud_data
        self.update()

    def set_standby(self, info: dict):
        self._is_standby = True
        self._standby_info = info
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        w = self.width()
        h = self.height()

        # Deep OLED Canvas
        painter.fillRect(0, 0, w, h, QColor(4, 5, 8))

        if self._is_standby:
            self._render_standby(painter, w, h)
            return

        if self._pixmap is None or self._pixmap.isNull():
            # Elegant Apple-style placeholder
            painter.setPen(QColor(100, 116, 139))
            painter.setFont(QFont(SPATIAL_FONT, 9, QFont.DemiBold))
            painter.drawText(QRect(0, 0, w, h), Qt.AlignCenter, "WAITING FOR CAMERA STREAM...")
            return

        pw = self._pixmap.width()
        ph = self._pixmap.height()
        if pw <= 0 or ph <= 0:
            return

        scale = min(w / float(pw), h / float(ph))
        dw = int(pw * scale)
        dh = int(ph * scale)
        dx = (w - dw) // 2
        dy = (h - dh) // 2

        target_rect = QRect(dx, dy, dw, dh)
        painter.drawPixmap(target_rect, self._pixmap)

        # Subtle hairline frame border
        painter.setPen(QPen(QColor(255, 255, 255, 25), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(target_rect)

        # In Camera Hero mode, render vector-sharp telemetry HUD pill at the top
        if self._hero_hud_data and w > 420:
            self._render_hero_hud(painter, dx, dy, dw, dh)

    def _render_hero_hud(self, painter: QPainter, fx, fy, fw, fh):
        hud = self._hero_hud_data
        alt = hud.get('alt', 0.0)
        vz = hud.get('vz', 0.0)
        spd = hud.get('spd', 0.0)
        hdg = hud.get('hdg', 0)

        hud_text = f"ALT: {alt:4.2f}m   Vz: {vz:+4.2f}m/s   SPD: {spd:4.2f}m/s   HDG: {hdg:03d}°   🛡️ EIS: ACTIVE"

        pill_w = min(460, fw - 40)
        pill_h = 28
        pill_x = fx + (fw - pill_w) // 2
        pill_y = fy + 10

        # Frosted glass capsule
        painter.setPen(QPen(QColor(56, 189, 248, 120), 1))
        painter.setBrush(QBrush(QColor(15, 23, 42, 210)))
        painter.drawRoundedRect(pill_x, pill_y, pill_w, pill_h, 14, 14)

        painter.setPen(QColor(248, 250, 252))
        painter.setFont(QFont(SPATIAL_MONO, 8, QFont.Bold))
        painter.drawText(QRect(pill_x, pill_y, pill_w, pill_h), Qt.AlignCenter, hud_text)

    def _render_standby(self, painter: QPainter, w, h):
        cov = self._standby_info.get('cov', 100.0)
        surv = self._standby_info.get('surv', 6)

        card_w = min(420, max(260, w - 24))
        card_h = min(170, max(120, h - 20))
        cx = (w - card_w) // 2
        cy = (h - card_h) // 2

        # Frosted background card
        painter.setPen(QPen(QColor(255, 255, 255, 25), 1))
        painter.setBrush(QBrush(QColor(14, 18, 28, 230)))
        painter.drawRoundedRect(cx, cy, card_w, card_h, 16, 16)

        # Status Pill: CAMERA STANDBY
        pill_w = min(280, card_w - 40)
        pill_h = 24
        px = cx + (card_w - pill_w) // 2
        py = cy + 14
        painter.setPen(QPen(QColor(239, 68, 68, 140), 1))
        painter.setBrush(QBrush(QColor(69, 10, 10, 180)))
        painter.drawRoundedRect(px, py, pill_w, pill_h, 12, 12)

        painter.setPen(QColor(254, 202, 202))
        painter.setFont(QFont(SPATIAL_FONT, 8, QFont.Bold))
        painter.drawText(QRect(px, py, pill_w, pill_h), Qt.AlignCenter, "🔴 STANDBY — SENSORS POWERED DOWN")

        # Title: MISSION ACCOMPLISHED
        painter.setPen(QColor(52, 211, 153))
        painter.setFont(QFont(SPATIAL_FONT, 12, QFont.Bold))
        painter.drawText(QRect(cx, cy + 44, card_w, 26), Qt.AlignCenter, "🏆 MISSION ACCOMPLISHED")

        # Stats
        painter.setFont(QFont(SPATIAL_MONO, 9, QFont.Bold))
        painter.setPen(QColor(226, 232, 240))
        painter.drawText(QRect(cx, cy + 76, card_w, 22), Qt.AlignCenter, f"Coverage: {cov:.1f}%  |  Survivors: {surv}")

        painter.setFont(QFont(SPATIAL_FONT, 8))
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(QRect(cx, cy + 102, card_w, 20), Qt.AlignCenter, "Touchdown Verified on Base Helipad")


# ─────────────────────────────────────────────────────────────────────────────
# 2D TACTICAL ARENA MAP (HERO ELEMENT)
# ─────────────────────────────────────────────────────────────────────────────
class ArenaMapWidget(QWidget):
    """
    Primary Map Canvas with Apple Vision Pro spatial styling:
      - Deep Obsidian Background (#07080c)
      - Unexplored: Midnight Slate (#0f172a) with subtle coordinate grid
      - Explored Free Space: Bright Silver (#f1f5f9)
      - Walls & Blocks: Solid Deep Onyx (#020617)
      - Glowing Cyan Flight Trail & Emerald Nav2 Spline
      - Fixed Persistent Survivor Markers (never drift with drone)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 150)
        self.setMouseTracking(True)
        self.cursor_world = None

        self.min_wx = -8.5
        self.max_wx =  8.5
        self.min_wy = -10.5
        self.max_wy =  8.5

        self.cached_qimg = None
        self.drone_pos = (-5.80, -8.80)
        self.drone_yaw = 1.5708
        self.drone_alt = 0.0
        self.trail = []
        self.nav_path = []
        self.current_goal = None
        self.survivors = []

    def mouseMoveEvent(self, event):
        pos = event.pos()
        w, h = self.width(), self.height()
        scale_x = w / (self.max_wx - self.min_wx)
        scale_y = h / (self.max_wy - self.min_wy)
        scale = min(scale_x, scale_y) * 0.95
        ox = (w - (self.max_wx - self.min_wx) * scale) / 2.0
        oy = (h - (self.max_wy - self.min_wy) * scale) / 2.0

        wx = self.min_wx + (pos.x() - ox) / scale
        wy = self.max_wy - (pos.y() - oy) / scale
        self.cursor_world = (wx, wy)
        self.update()

    def update_map_data(self, grid, mw, mh, res, ox, oy, dx, dy, dz, dyaw, trail, nav_path, goal, survivors):
        self.drone_pos = (dx, dy)
        self.drone_yaw = dyaw
        self.drone_alt = dz
        self.trail = trail or []
        self.nav_path = nav_path or []
        self.current_goal = goal
        self.survivors = survivors or []

        if grid is not None and mw > 0 and mh > 0:
            c0 = max(0, int((self.min_wx - ox) / res))
            c1 = min(mw, int((self.max_wx - ox) / res))
            r0 = max(0, int((self.min_wy - oy) / res))
            r1 = min(mh, int((self.max_wy - oy) / res))

            crop = grid[r0:r1, c0:c1]
            ch, cw = crop.shape
            if ch > 0 and cw > 0:
                rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
                # Unexplored (-1) = Midnight Slate (15, 23, 42)
                rgb[crop == -1] = [15, 23, 42]
                # Explored Free (0) = Luminous Silver (241, 245, 249)
                rgb[crop == 0]  = [241, 245, 249]
                # Walls (100) = Solid Onyx (2, 6, 23)
                rgb[crop == 100]= [2, 6, 23]

                rgb_flipped = np.ascontiguousarray(np.flipud(rgb))
                qimg = QImage(rgb_flipped.data, cw, ch, cw * 3, QImage.Format_RGB888)
                self.cached_qimg = qimg

        self.update()

    def world_to_screen(self, wx, wy, scale, ox, oy):
        sx = ox + (wx - self.min_wx) * scale
        sy = oy + (self.max_wy - wy) * scale
        return sx, sy

    def paintEvent(self, event):
        painter = QPainter(self)
        self.render_tactical_map(painter, self.width(), self.height(), is_export=False)

    def export_map_pixmap(self, info=None):
        pix = QPixmap(1400, 1400)
        pix.fill(QColor(7, 8, 12))
        painter = QPainter(pix)
        self.render_tactical_map(painter, 1400, 1400, is_export=True, info=info)
        painter.end()
        return pix

    def render_tactical_map(self, painter, w, h, is_export=False, info=None):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(0, 0, w, h, QColor(7, 8, 12))

        top_y = 110 if is_export else 0
        draw_h = h - top_y

        scale_x = w / (self.max_wx - self.min_wx)
        scale_y = draw_h / (self.max_wy - self.min_wy)
        scale = min(scale_x, scale_y) * 0.94
        ox = (w - (self.max_wx - self.min_wx) * scale) / 2.0
        oy = top_y + (draw_h - (self.max_wy - self.min_wy) * scale) / 2.0

        # Export Banner Header
        if is_export:
            painter.fillRect(0, 0, w, 105, QColor(15, 23, 42))
            painter.setPen(QPen(QColor(30, 41, 59), 2))
            painter.drawLine(0, 105, w, 105)

            painter.setFont(QFont(SPATIAL_FONT, 16, QFont.Bold))
            painter.setPen(QPen(QColor(56, 189, 248)))
            painter.drawText(30, 42, "NIDAR SPATIAL OS — 2D SLAM TACTICAL MAP")

            painter.setFont(QFont(SPATIAL_FONT, 11))
            painter.setPen(QPen(QColor(226, 232, 240)))
            cov_val = info.get('cov', 0.0) if info else 0.0
            time_val = info.get('time', 'N/A') if info else 'N/A'
            surv_val = info.get('surv', 0) if info else 0
            painter.drawText(30, 78, f"Explored: {cov_val:.1f}%   |   Mission Time: {time_val}   |   Confirmed Survivors: {surv_val}")

            if info and info.get('survivors'):
                painter.setFont(QFont(SPATIAL_FONT, 9))
                painter.setPen(QPen(QColor(251, 146, 60)))
                surv_lines = [f"#{s.get('id', i+1)}: ({s.get('x', s.get('map_x', 0.0)):+.2f}, {s.get('y', s.get('map_y', 0.0)):+.2f})" for i, s in enumerate(info['survivors'])]
                painter.drawText(w - 440, 40, "Fixed Survivor Locations:")
                painter.drawText(w - 440, 64, "  " + "   ".join(surv_lines[:3]))
                if len(surv_lines) > 3:
                    painter.drawText(w - 440, 86, "  " + "   ".join(surv_lines[3:]))

        # 1. Background Grid Lines
        painter.setPen(QPen(QColor(255, 255, 255, 12), 1, Qt.DotLine))
        for gx in range(-8, 9, 2):
            p0x, p0y = self.world_to_screen(gx, self.min_wy, scale, ox, oy)
            p1x, p1y = self.world_to_screen(gx, self.max_wy, scale, ox, oy)
            painter.drawLine(int(p0x), int(p0y), int(p1x), int(p1y))
        for gy in range(-10, 9, 2):
            p0x, p0y = self.world_to_screen(self.min_wx, gy, scale, ox, oy)
            p1x, p1y = self.world_to_screen(self.max_wx, gy, scale, ox, oy)
            painter.drawLine(int(p0x), int(p0y), int(p1x), int(p1y))

        # 2. Draw Occupancy Grid
        if self.cached_qimg is not None:
            sx0, sy0 = self.world_to_screen(self.min_wx, self.max_wy, scale, ox, oy)
            sx1, sy1 = self.world_to_screen(self.max_wx, self.min_wy, scale, ox, oy)
            target_rect = QRectF(sx0, sy0, sx1 - sx0, sy1 - sy0)
            painter.drawImage(target_rect, self.cached_qimg)
        else:
            painter.setPen(QPen(QColor(100, 116, 139), 1))
            painter.drawText(w // 2 - 80, h // 2, "Awaiting SLAM Map Stream...")

        # 3. Base Helipad (-5.80, -8.75)
        hx, hy = self.world_to_screen(-5.80, -8.75, scale, ox, oy)
        pad_size = 1.9 * scale
        painter.setPen(QPen(QColor(16, 185, 129, 200), 2))
        painter.setBrush(QBrush(QColor(16, 185, 129, 35)))
        painter.drawRoundedRect(QRectF(hx - pad_size/2, hy - pad_size/2, pad_size, pad_size), 8, 8)
        painter.drawEllipse(QPointF(hx, hy), pad_size * 0.38, pad_size * 0.38)

        painter.setFont(QFont(SPATIAL_FONT, 11, QFont.Bold))
        painter.setPen(QPen(QColor(16, 185, 129)))
        painter.drawText(QRectF(hx - 15, hy - 15, 30, 30), Qt.AlignCenter, "H")

        # 4. Flight Trail
        if len(self.trail) > 1:
            painter.setPen(QPen(QColor(6, 182, 212, 140), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            poly = QPolygonF([QPointF(*self.world_to_screen(tx, ty, scale, ox, oy)) for tx, ty in self.trail[-250:]])
            painter.drawPolyline(poly)

        # 5. Nav2 Planned Path
        if len(self.nav_path) > 1:
            painter.setPen(QPen(QColor(16, 185, 129, 220), 2))
            painter.setBrush(Qt.NoBrush)
            path_poly = QPolygonF([QPointF(*self.world_to_screen(px, py, scale, ox, oy)) for px, py in self.nav_path])
            painter.drawPolyline(path_poly)

        # 6. Current Goal
        if self.current_goal is not None:
            gx, gy = self.world_to_screen(self.current_goal[0], self.current_goal[1], scale, ox, oy)
            painter.setPen(QPen(QColor(56, 189, 248), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(gx, gy), 7, 7)
            painter.drawLine(int(gx - 11), int(gy), int(gx + 11), int(gy))
            painter.drawLine(int(gx), int(gy - 11), int(gx), int(gy + 11))

        # 7. FIXED CONFIRMED SURVIVORS (Never inherit drone movement)
        for s in self.survivors:
            sx_val = s.get('x', s.get('map_x', 0.0))
            sy_val = s.get('y', s.get('map_y', 0.0))
            sid = s.get('id', 1)
            sx, sy = self.world_to_screen(sx_val, sy_val, scale, ox, oy)

            # Outer Soft Amber Halo
            painter.setPen(QPen(QColor(249, 115, 22, 160), 2))
            painter.setBrush(QBrush(QColor(249, 115, 22, 50)))
            painter.drawEllipse(QPointF(sx, sy), 14, 14)

            # Center Pinpoint
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QBrush(QColor(249, 115, 22)))
            painter.drawEllipse(QPointF(sx, sy), 5, 5)

            if w >= 440 or is_export:
                # Spatial Pill Tag: SURVIVOR #ID [FIXED]
                tag_text = f"SURVIVOR #{sid} [FIXED]"
                painter.setFont(QFont(SPATIAL_FONT, 8, QFont.Bold))
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(int(sx + 12), int(sy - 2), tag_text)

                coord_text = f"({sx_val:+.2f}, {sy_val:+.2f})"
                painter.setFont(QFont(SPATIAL_MONO, 7))
                painter.setPen(QPen(QColor(56, 189, 248)))
                painter.drawText(int(sx + 12), int(sy + 10), coord_text)
            else:
                painter.setFont(QFont(SPATIAL_FONT, 7, QFont.Bold))
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(int(sx + 8), int(sy + 4), f"S#{sid}")

        # 8. Live Drone with Heading Orientation
        dx, dy = self.world_to_screen(self.drone_pos[0], self.drone_pos[1], scale, ox, oy)

        # Pulse Ring
        painter.setPen(QPen(QColor(6, 182, 212, 120), 2))
        painter.setBrush(QBrush(QColor(6, 182, 212, 35)))
        painter.drawEllipse(QPointF(dx, dy), 16, 16)

        # Heading Triangle
        painter.save()
        painter.translate(dx, dy)
        yaw_deg = math.degrees(self.drone_yaw)
        painter.rotate(-yaw_deg)

        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setBrush(QBrush(QColor(34, 211, 238))) # Glowing Cyan
        tri = QPolygonF([
            QPointF(14, 0),
            QPointF(-8, -7),
            QPointF(-4, 0),
            QPointF(-8, 7)
        ])
        painter.drawPolygon(tri)
        painter.restore()

        # Legend Capsule (Top-Left) - render only if wide enough
        if w >= 450 or is_export:
            leg_y = top_y + 14
            painter.setPen(QPen(QColor(255, 255, 255, 18), 1))
            painter.setBrush(QBrush(QColor(15, 23, 42, 210)))
            painter.drawRoundedRect(QRectF(14, leg_y, 165, 82), 10, 10)

            items = [
                (QColor(15, 23, 42), "Unexplored Area"),
                (QColor(241, 245, 249), "Explored Space"),
                (QColor(2, 6, 23), "Walls / Obstacles"),
                (QColor(249, 115, 22), "Fixed Survivor"),
            ]
            for idx, (col, lbl) in enumerate(items):
                iy = leg_y + 16 + idx * 17
                painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
                painter.setBrush(QBrush(col))
                painter.drawRoundedRect(QRectF(24, iy - 6, 9, 9), 2, 2)
                painter.setPen(QPen(QColor(203, 213, 225)))
                painter.setFont(QFont(SPATIAL_FONT, 8))
                painter.drawText(40, iy + 2, lbl)

        # Scale Bar
        if w >= 380 or is_export:
            bar_len = int(2.0 * scale)
            bx, by = 16, h - 18
            painter.setPen(QPen(QColor(148, 163, 184), 2))
            painter.drawLine(bx, by, bx + bar_len, by)
            painter.drawLine(bx, by - 3, bx, by + 3)
            painter.drawLine(bx + bar_len, by - 3, bx + bar_len, by + 3)
            painter.setFont(QFont(SPATIAL_FONT, 7, QFont.Bold))
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(bx + 4, by - 5, "2.0 m")

            if not is_export and self.cursor_world:
                cwx, cwy = self.cursor_world
                painter.setFont(QFont(SPATIAL_MONO, 8))
                painter.setPen(QPen(QColor(100, 116, 139)))
                painter.drawText(bx + bar_len + 16, by + 2, f"CURSOR: X={cwx:+5.2f}m, Y={cwy:+5.2f}m")


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL LIDAR RADAR WIDGET
# ─────────────────────────────────────────────────────────────────────────────
class SpatialLidarRadarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ranges = []
        self.angle_min = -math.pi
        self.angle_inc = 0.0174
        self.min_dist = 99.0
        self.setMinimumSize(180, 160)
        self.setMaximumHeight(180)

    def update_scan(self, ranges, angle_min, angle_inc):
        self.ranges = ranges or []
        self.angle_min = angle_min
        self.angle_inc = angle_inc
        valid = [r for r in self.ranges if 0.05 < r < 8.0]
        self.min_dist = min(valid) if valid else 99.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(cx, cy) - 10

        # Concentric Distance Rings
        max_range = 4.5
        scale = radius / max_range

        for m in (1.0, 2.5, 4.0):
            r_px = int(m * scale)
            painter.setPen(QPen(QColor(56, 189, 248, 30), 1, Qt.DotLine))
            painter.drawEllipse(cx - r_px, cy - r_px, r_px * 2, r_px * 2)

        # Crosshairs
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
        painter.drawLine(cx, cy - radius, cx, cy + radius)
        painter.drawLine(cx - radius, cy, cx + radius, cy)

        # Center Drone Indicator
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(34, 211, 238)))
        painter.drawEllipse(cx - 3, cy - 3, 6, 6)

        # LiDAR Point Cloud
        if self.ranges:
            painter.setPen(QPen(QColor(34, 211, 238, 180), 2))
            for i, r in enumerate(self.ranges):
                if 0.10 < r < max_range:
                    ang = self.angle_min + i * self.angle_inc
                    px = cx - int(r * math.sin(ang) * scale)
                    py = cy - int(r * math.cos(ang) * scale)
                    painter.drawPoint(px, py)


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL PROPELLER DYNAMICS WIDGET
# ─────────────────────────────────────────────────────────────────────────────
class SpatialPropellerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 110)
        self.setMaximumHeight(120)
        self.motor_rpms = [0, 0, 0, 0]
        self.spin_angle = 0.0

    def update_motors(self, vx, vy, vz, wz, alt):
        if alt > 0.15:
            base = 6500.0
            self.motor_rpms[0] = int(base + vz * 1200 - vx * 600 + vy * 600 - wz * 800)
            self.motor_rpms[1] = int(base + vz * 1200 - vx * 600 - vy * 600 + wz * 800)
            self.motor_rpms[2] = int(base + vz * 1200 + vx * 600 + vy * 600 + wz * 800)
            self.motor_rpms[3] = int(base + vz * 1200 + vx * 600 - vy * 600 - wz * 800)
            self.spin_angle = (self.spin_angle + 28) % 360
        else:
            self.motor_rpms = [0, 0, 0, 0]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        arm = min(cx, cy) - 14

        # Cross Frame
        painter.setPen(QPen(QColor(148, 163, 184, 50), 2))
        painter.drawLine(cx - arm, cy - arm, cx + arm, cy + arm)
        painter.drawLine(cx - arm, cy + arm, cx + arm, cy - arm)

        # Center Body
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(30, 41, 59)))
        painter.drawEllipse(cx - 7, cy - 7, 14, 14)

        # 4 Motors
        positions = [
            (cx + arm, cy - arm),  # M1 Front-Right
            (cx - arm, cy + arm),  # M2 Rear-Left
            (cx + arm, cy + arm),  # M3 Rear-Right
            (cx - arm, cy - arm),  # M4 Front-Left
        ]
        for i, (mx, my) in enumerate(positions):
            active = self.motor_rpms[i] > 0
            ring_col = QColor(16, 185, 129, 180) if active else QColor(100, 116, 139, 60)
            painter.setPen(QPen(ring_col, 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(int(mx - 10), int(my - 10), 20, 20)

            # Motor Hub
            painter.setBrush(QBrush(ring_col))
            painter.drawEllipse(int(mx - 3), int(my - 3), 6, 6)


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL SURVIVOR CARD (INDIVIDUAL FIXED PERSISTENT INTEL)
# ─────────────────────────────────────────────────────────────────────────────
class SpatialSurvivorCard(QFrame):
    """
    Renders the exact user-specified survivor format:
      SURVIVOR #ID
      Status: CONFIRMED
      Map position: fixed (X, Y)
      Grid: Sector
      Confidence: XX%
    """
    def __init__(self, survivor_data, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 10px;
                margin-bottom: 6px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        sid = survivor_data.get('id', 1)
        sx = survivor_data.get('map_x', survivor_data.get('x', 0.0))
        sy = survivor_data.get('map_y', survivor_data.get('y', 0.0))
        grid = survivor_data.get('grid', 'SW-01')
        conf = survivor_data.get('confidence', 98)

        # 1. SURVIVOR #{id}
        lbl_title = QLabel(f"SURVIVOR #{sid}")
        lbl_title.setFont(QFont(SPATIAL_FONT, 11, QFont.Bold))
        lbl_title.setStyleSheet("color: #f8fafc;")
        layout.addWidget(lbl_title)

        # 2. Status: CONFIRMED
        status_row = QHBoxLayout()
        lbl_st_lbl = QLabel("Status:")
        lbl_st_lbl.setFont(QFont(SPATIAL_FONT, 9))
        lbl_st_lbl.setStyleSheet("color: #94a3b8;")
        lbl_status = QLabel("CONFIRMED")
        lbl_status.setFont(QFont(SPATIAL_FONT, 8, QFont.Bold))
        lbl_status.setStyleSheet("""
            background-color: rgba(16, 185, 129, 0.16);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.35);
            border-radius: 6px;
            padding: 2px 7px;
        """)
        status_row.addWidget(lbl_st_lbl)
        status_row.addWidget(lbl_status)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        # Map Position Callout (Fixed Map Coordinates)
        lbl_pos = QLabel(f"Map position: fixed ({sx:+.2f}m, {sy:+.2f}m)")
        lbl_pos.setFont(QFont(SPATIAL_MONO, 9))
        lbl_pos.setStyleSheet("color: #38bdf8;")
        layout.addWidget(lbl_pos)

        # Grid Sector Callout
        lbl_grid = QLabel(f"Grid: {grid}")
        lbl_grid.setFont(QFont(SPATIAL_FONT, 9))
        lbl_grid.setStyleSheet("color: #cbd5e1;")
        layout.addWidget(lbl_grid)

        # Confidence Bar
        conf_row = QHBoxLayout()
        lbl_conf = QLabel(f"Confidence: {conf}%")
        lbl_conf.setFont(QFont(SPATIAL_FONT, 9, QFont.Bold))
        lbl_conf.setStyleSheet("color: #34d399;")

        bar = QProgressBar()
        bar.setFixedHeight(5)
        bar.setRange(0, 100)
        bar.setValue(int(conf))
        bar.setTextVisible(False)
        bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(255, 255, 255, 0.08);
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #38bdf8);
                border-radius: 2px;
            }
        """)
        conf_row.addWidget(lbl_conf)
        conf_row.addWidget(bar, 1)
        layout.addLayout(conf_row)


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL JEV COGNITIVE STATE MACHINE WIDGET
# ─────────────────────────────────────────────────────────────────────────────
class SpatialJevWidget(QFrame):
    """
    Renders JEV as a concise 6-state pipeline:
      OBSERVE → QUESTION → THINK → DECIDE → ACT → VERIFY
    Active state is highlighted; zero chain-of-thought is exposed.
    """
    STEPS = ["OBSERVE", "QUESTION", "THINK", "DECIDE", "ACT", "VERIFY"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {GLASS_BG};
                border: {GLASS_BORDER};
                border-radius: {GLASS_RADIUS};
            }}
        """)
        self.setGraphicsEffect(make_shadow(20, 4, 100))

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(16, 12, 16, 12)
        vbox.setSpacing(6)

        hdr_row = QHBoxLayout()
        lbl_title = QLabel("JEV COGNITIVE STATE MACHINE")
        lbl_title.setFont(QFont(SPATIAL_FONT, 8, QFont.Bold))
        lbl_title.setStyleSheet("color: #38bdf8; letter-spacing: 1px;")
        hdr_row.addWidget(lbl_title)
        hdr_row.addStretch(1)
        vbox.addLayout(hdr_row)

        # 6-Step Pipeline Pills
        self.pills_layout = QHBoxLayout()
        self.pills_layout.setSpacing(6)
        self.pill_labels = {}

        for i, step in enumerate(self.STEPS):
            lbl = QLabel(step)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFont(QFont(SPATIAL_FONT, 8, QFont.Bold))
            lbl.setFixedHeight(26)
            self.set_pill_style(lbl, active=False)
            self.pill_labels[step] = lbl
            self.pills_layout.addWidget(lbl)

            if i < len(self.STEPS) - 1:
                arr = QLabel("→")
                arr.setFont(QFont(SPATIAL_FONT, 9))
                arr.setStyleSheet("color: rgba(255, 255, 255, 0.25);")
                self.pills_layout.addWidget(arr)

        vbox.addLayout(self.pills_layout)

        # Executive Objective (Concise high-level goal, no chain-of-thought)
        self.lbl_objective = QLabel("Objective: Autonomous Frontier Exploration & Compound Mapping")
        self.lbl_objective.setFont(QFont(SPATIAL_FONT, 9))
        self.lbl_objective.setStyleSheet("color: #94a3b8;")
        vbox.addWidget(self.lbl_objective)

    def set_pill_style(self, lbl, active=False):
        if active:
            lbl.setStyleSheet("""
                background-color: rgba(16, 185, 129, 0.25);
                color: #ffffff;
                border: 1px solid #10b981;
                border-radius: 8px;
                padding: 2px 10px;
            """)
        else:
            lbl.setStyleSheet("""
                background-color: rgba(255, 255, 255, 0.04);
                color: rgba(255, 255, 255, 0.35);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 8px;
                padding: 2px 10px;
            """)

    def update_status(self, raw_status):
        upper_text = raw_status.upper()
        active_step = "ACT"  # Default sensible active state

        for s in self.STEPS:
            if s in upper_text:
                active_step = s
                break

        if "NAVIGAT" in upper_text or "FRONTIER" in upper_text:
            active_step = "ACT"
        elif "TOUCHDOWN" in upper_text or "REACHED" in upper_text or "LAND" in upper_text:
            active_step = "VERIFY"

        for s, lbl in self.pill_labels.items():
            self.set_pill_style(lbl, active=(s == active_step))

        # Filter and summarize concisely for executive objective (no chain-of-thought)
        clean_text = raw_status.replace("DECIDE ➔ ACT:", "").replace("🧠 [JEV]", "").strip()
        if "ACT: Navigating" in raw_status or "frontier" in raw_status.lower():
            self.lbl_objective.setText("Objective: Navigating Corridor to Frontier Target")
        elif "RETURNING" in upper_text:
            self.lbl_objective.setText("Objective: Returning to Base Helipad (South Gate Exit)")
        elif "TOUCHDOWN" in upper_text:
            self.lbl_objective.setText("Objective: Precision Landing on Helipad & Motor Disarm")
        elif "STANDBY" in upper_text:
            self.lbl_objective.setText("Objective: Takeoff & Climb to 1.50m Cruise Altitude")
        else:
            self.lbl_objective.setText(f"Objective: {clean_text[:65]}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SPATIAL GCS WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class NidarGcsWindow(QMainWindow):
    def __init__(self, backend: GcsRosBackend):
        super().__init__()
        self.backend = backend
        self.setWindowTitle("NIDAR Spatial Ground Control Station (GCS)")
        self.resize(1500, 920)
        self.setStyleSheet(MAIN_STYLESHEET)

        self.view_mode = "map"  # "map" (Map Hero) or "camera" (Camera Hero)
        self._auto_export_done = False
        self._last_surv_sig = None

        self.init_spatial_ui()

        # Telemetry Timer at 25 Hz
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_telemetry_ui)
        self.timer.start(40)

    def init_spatial_ui(self):
        root_widget = QWidget(self)
        self.setCentralWidget(root_widget)
        root_vbox = QVBoxLayout(root_widget)
        root_vbox.setContentsMargins(16, 14, 16, 14)
        root_vbox.setSpacing(12)

        # ── 1. TOP SPATIAL FLOATING CAPSULE BAR ──
        top_capsule = QFrame()
        top_capsule.setStyleSheet(f"""
            QFrame {{
                background-color: {GLASS_BG};
                border: {GLASS_BORDER};
                border-radius: 20px;
            }}
        """)
        top_capsule.setGraphicsEffect(make_shadow(24, 4, 120))
        top_hb = QHBoxLayout(top_capsule)
        top_hb.setContentsMargins(18, 10, 18, 10)
        top_hb.setSpacing(14)

        # Brand Badge with Pulsing LED
        brand_layout = QHBoxLayout()
        self.lbl_led = QLabel("●")
        self.lbl_led.setFont(QFont(SPATIAL_FONT, 12))
        self.lbl_led.setStyleSheet("color: #10b981;")

        lbl_brand = QLabel("NIDAR SPATIAL")
        lbl_brand.setFont(QFont(SPATIAL_FONT, 12, QFont.Bold))
        lbl_brand.setStyleSheet("color: #f8fafc; letter-spacing: 0.5px;")

        self.lbl_flight_mode = QLabel("AUTONOMOUS")
        self.lbl_flight_mode.setFont(QFont(SPATIAL_FONT, 8, QFont.Bold))
        self.lbl_flight_mode.setStyleSheet("""
            background-color: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 6px;
            padding: 2px 8px;
        """)
        brand_layout.addWidget(self.lbl_led)
        brand_layout.addWidget(lbl_brand)
        brand_layout.addWidget(self.lbl_flight_mode)
        top_hb.addLayout(brand_layout)

        top_hb.addStretch(1)

        # Central Stopwatch Timer
        timer_box = QHBoxLayout()
        lbl_timer_ic = QLabel("⏱")
        lbl_timer_ic.setFont(QFont(SPATIAL_FONT, 12))
        self.lbl_big_timer = QLabel("00:00.0")
        self.lbl_big_timer.setFont(QFont(SPATIAL_MONO, 16, QFont.Bold))
        self.lbl_big_timer.setStyleSheet("""
            color: #34d399;
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 3px 14px;
        """)
        timer_box.addWidget(lbl_timer_ic)
        timer_box.addWidget(self.lbl_big_timer)
        top_hb.addLayout(timer_box)

        # Exploration Coverage Pill
        cov_box = QHBoxLayout()
        lbl_cov_ic = QLabel("🗺")
        self.lbl_cov_val = QLabel("0.0% Mapped")
        self.lbl_cov_val.setFont(QFont(SPATIAL_FONT, 11, QFont.Bold))
        self.lbl_cov_val.setStyleSheet("color: #38bdf8;")

        self.bar_cov = QProgressBar()
        self.bar_cov.setFixedWidth(130)
        self.bar_cov.setFixedHeight(8)
        self.bar_cov.setRange(0, 100)
        self.bar_cov.setValue(0)
        self.bar_cov.setTextVisible(False)
        self.bar_cov.setStyleSheet("""
            QProgressBar {
                background-color: rgba(255, 255, 255, 0.08);
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #38bdf8);
                border-radius: 4px;
            }
        """)
        cov_box.addWidget(lbl_cov_ic)
        cov_box.addWidget(self.lbl_cov_val)
        cov_box.addWidget(self.bar_cov)
        top_hb.addLayout(cov_box)

        # Survivor Tagged Count Pill
        self.lbl_surv_pill = QLabel("👥 0 Survivors")
        self.lbl_surv_pill.setFont(QFont(SPATIAL_FONT, 10, QFont.Bold))
        self.lbl_surv_pill.setStyleSheet("""
            color: #fb923c;
            background-color: rgba(249, 115, 22, 0.12);
            border: 1px solid rgba(249, 115, 22, 0.30);
            border-radius: 8px;
            padding: 4px 10px;
        """)
        top_hb.addWidget(self.lbl_surv_pill)

        # Apple Spatial Segmented View Switcher: [ 🗺️ Map Focus ] [ ⇄ Rotate View ] [ 📹 Camera Focus ]
        seg_container = QFrame()
        seg_container.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 12px;
                padding: 2px;
            }
        """)
        seg_layout = QHBoxLayout(seg_container)
        seg_layout.setContentsMargins(3, 3, 3, 3)
        seg_layout.setSpacing(2)

        self.btn_seg_map = QPushButton("🗺️ Map Focus")
        self.btn_seg_map.setCursor(Qt.PointingHandCursor)
        self.btn_seg_map.clicked.connect(lambda: self.set_view_mode('map'))

        self.btn_seg_rotate = QPushButton("⇄ Rotate View")
        self.btn_seg_rotate.setCursor(Qt.PointingHandCursor)
        self.btn_seg_rotate.clicked.connect(self.toggle_view_mode)

        self.btn_seg_cam = QPushButton("📹 Camera Focus")
        self.btn_seg_cam.setCursor(Qt.PointingHandCursor)
        self.btn_seg_cam.clicked.connect(lambda: self.set_view_mode('camera'))

        seg_layout.addWidget(self.btn_seg_map)
        seg_layout.addWidget(self.btn_seg_rotate)
        seg_layout.addWidget(self.btn_seg_cam)
        top_hb.addWidget(seg_container)

        top_hb.addStretch(1)

        # Action Buttons (Spatial Glass Styling)
        btn_style = """
            QPushButton {
                background-color: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 10px;
                color: #f1f5f9;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.14);
                border-color: rgba(56, 189, 248, 0.45);
            }
            QPushButton:pressed {
                background-color: rgba(56, 189, 248, 0.20);
            }
        """

        self.btn_download_map = QPushButton("💾 Export Map")
        self.btn_download_map.setToolTip("Save a high-resolution mission map snapshot")
        self.btn_download_map.setStyleSheet(btn_style)
        self.btn_download_map.clicked.connect(self.on_download_map_clicked)
        top_hb.addWidget(self.btn_download_map)

        self.btn_relaunch = QPushButton("🚀 Relaunch")
        self.btn_relaunch.setToolTip("Clear mission state and begin a new autonomous search")
        self.btn_relaunch.setStyleSheet("""
            QPushButton {
                background-color: rgba(16, 185, 129, 0.18);
                border: 1px solid rgba(16, 185, 129, 0.40);
                border-radius: 10px;
                color: #34d399;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(16, 185, 129, 0.30);
            }
        """)
        self.btn_relaunch.clicked.connect(self.on_relaunch_clicked)
        top_hb.addWidget(self.btn_relaunch)

        self.btn_land = QPushButton("🛬 Land / RTH")
        self.btn_land.setToolTip("Abort exploration, return to the helipad, and land")
        self.btn_land.setStyleSheet("""
            QPushButton {
                background-color: rgba(239, 68, 68, 0.16);
                border: 1px solid rgba(239, 68, 68, 0.35);
                border-radius: 10px;
                color: #f87171;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(239, 68, 68, 0.28);
            }
        """)
        self.btn_land.clicked.connect(self.on_land_clicked)
        top_hb.addWidget(self.btn_land)

        root_vbox.addWidget(top_capsule)

        # ── 2. MAIN SPATIAL WORKSPACE (MAP-FIRST HERO COMPOSITION) ──
        workspace = QHBoxLayout()
        workspace.setSpacing(12)

        # LEFT FLOATING SPATIAL PANEL: TELEMETRY & SENSORS
        self.panel_left = QFrame()
        self.panel_left.setFixedWidth(270)
        self.panel_left.setStyleSheet(GLASS_CARD_STYLE)
        self.panel_left.setGraphicsEffect(make_shadow(24, 6, 120))
        left_vbox = QVBoxLayout(self.panel_left)
        left_vbox.setContentsMargins(14, 14, 14, 14)
        left_vbox.setSpacing(10)

        lbl_telem_hdr = QLabel("FLIGHT TELEMETRY")
        lbl_telem_hdr.setFont(QFont(SPATIAL_FONT, 8, QFont.Bold))
        lbl_telem_hdr.setStyleSheet("color: #38bdf8; letter-spacing: 1px;")
        left_vbox.addWidget(lbl_telem_hdr)

        # Altitude Card
        alt_card = QFrame()
        alt_card.setStyleSheet("background: rgba(255,255,255,0.04); border-radius: 10px; padding: 6px;")
        alt_v = QVBoxLayout(alt_card)
        alt_v.setContentsMargins(8, 6, 8, 6)
        alt_v.setSpacing(2)
        lbl_alt_sub = QLabel("ALTITUDE")
        lbl_alt_sub.setFont(QFont(SPATIAL_FONT, 7, QFont.Bold))
        lbl_alt_sub.setStyleSheet("color: #94a3b8;")
        self.lbl_altitude = QLabel("0.00 m")
        self.lbl_altitude.setFont(QFont(SPATIAL_MONO, 15, QFont.Bold))
        self.lbl_altitude.setStyleSheet("color: #34d399;")
        self.lbl_climb = QLabel("Vz: +0.00 m/s  (Target: 1.50m)")
        self.lbl_climb.setFont(QFont(SPATIAL_MONO, 8))
        self.lbl_climb.setStyleSheet("color: #94a3b8;")
        alt_v.addWidget(lbl_alt_sub)
        alt_v.addWidget(self.lbl_altitude)
        alt_v.addWidget(self.lbl_climb)
        left_vbox.addWidget(alt_card)

        # Heading & Speed
        hdg_card = QFrame()
        hdg_card.setStyleSheet("background: rgba(255,255,255,0.04); border-radius: 10px; padding: 6px;")
        hdg_v = QVBoxLayout(hdg_card)
        hdg_v.setContentsMargins(8, 6, 8, 6)
        hdg_v.setSpacing(2)
        self.lbl_heading = QLabel("🧭 HDG: 090° [EAST]")
        self.lbl_heading.setFont(QFont(SPATIAL_MONO, 10, QFont.Bold))
        self.lbl_heading.setStyleSheet("color: #fbbf24;")
        self.lbl_speed = QLabel("🚀 SPEED: 0.00 m/s")
        self.lbl_speed.setFont(QFont(SPATIAL_MONO, 9))
        self.lbl_speed.setStyleSheet("color: #38bdf8;")
        self.lbl_coords = QLabel("📍 X: -5.80m, Y: -8.80m")
        self.lbl_coords.setFont(QFont(SPATIAL_MONO, 9))
        self.lbl_coords.setStyleSheet("color: #cbd5e1;")
        hdg_v.addWidget(self.lbl_heading)
        hdg_v.addWidget(self.lbl_speed)
        hdg_v.addWidget(self.lbl_coords)
        left_vbox.addWidget(hdg_card)

        # 360° LiDAR Radar Scope
        lbl_radar_sub = QLabel("SPATIAL 360° LIDAR RADAR")
        lbl_radar_sub.setFont(QFont(SPATIAL_FONT, 8, QFont.Bold))
        lbl_radar_sub.setStyleSheet("color: #38bdf8; letter-spacing: 0.5px;")
        left_vbox.addWidget(lbl_radar_sub)
        self.radar_widget = SpatialLidarRadarWidget()
        left_vbox.addWidget(self.radar_widget)

        # Rotor Dynamics
        lbl_prop_sub = QLabel("ROTOR DYNAMICS")
        lbl_prop_sub.setFont(QFont(SPATIAL_FONT, 8, QFont.Bold))
        lbl_prop_sub.setStyleSheet("color: #38bdf8; letter-spacing: 0.5px;")
        left_vbox.addWidget(lbl_prop_sub)
        self.prop_widget = SpatialPropellerWidget()
        left_vbox.addWidget(self.prop_widget)
        left_vbox.addStretch(1)

        workspace.addWidget(self.panel_left)

        # CENTER HERO ELEMENT (SWAPPABLE CONTAINER)
        self.hero_frame = QFrame()
        self.hero_frame.setStyleSheet(GLASS_CARD_STYLE)
        self.hero_frame.setGraphicsEffect(make_shadow(28, 8, 140))
        self.hero_vbox = QVBoxLayout(self.hero_frame)
        self.hero_vbox.setContentsMargins(10, 10, 10, 10)
        self.hero_vbox.setSpacing(6)

        # Hero Header Bar
        hero_hdr = QHBoxLayout()
        self.lbl_hero_title = QLabel("2D TACTICAL SLAM ARENA MAP")
        self.lbl_hero_title.setFont(QFont(SPATIAL_FONT, 9, QFont.Bold))
        self.lbl_hero_title.setStyleSheet("color: #38bdf8; letter-spacing: 0.8px;")
        hero_hdr.addWidget(self.lbl_hero_title)
        hero_hdr.addStretch(1)

        self.lbl_hero_badge = QLabel("MAP FOCUS")
        self.lbl_hero_badge.setFont(QFont(SPATIAL_FONT, 7, QFont.Bold))
        self.lbl_hero_badge.setStyleSheet("""
            background-color: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 5px;
            padding: 2px 7px;
        """)
        hero_hdr.addWidget(self.lbl_hero_badge)
        self.hero_vbox.addLayout(hero_hdr)

        self.hero_content_layout = QVBoxLayout()
        self.hero_content_layout.setContentsMargins(0, 0, 0, 0)
        self.hero_vbox.addLayout(self.hero_content_layout, 1)

        workspace.addWidget(self.hero_frame, 1)

        # RIGHT FLOATING SPATIAL PANEL: CONFIRMED SURVIVOR INTEL
        self.panel_right = QFrame()
        self.panel_right.setFixedWidth(290)
        self.panel_right.setStyleSheet(GLASS_CARD_STYLE)
        self.panel_right.setGraphicsEffect(make_shadow(24, 6, 120))
        right_vbox = QVBoxLayout(self.panel_right)
        right_vbox.setContentsMargins(14, 14, 14, 14)
        right_vbox.setSpacing(8)

        right_hdr = QHBoxLayout()
        lbl_surv_hdr = QLabel("SURVIVOR INTEL")
        lbl_surv_hdr.setFont(QFont(SPATIAL_FONT, 8, QFont.Bold))
        lbl_surv_hdr.setStyleSheet("color: #fb923c; letter-spacing: 1px;")
        right_hdr.addWidget(lbl_surv_hdr)
        right_hdr.addStretch(1)

        self.lbl_surv_badge = QLabel("0 CONFIRMED")
        self.lbl_surv_badge.setFont(QFont(SPATIAL_FONT, 8, QFont.Bold))
        self.lbl_surv_badge.setStyleSheet("color: #34d399;")
        right_hdr.addWidget(self.lbl_surv_badge)
        right_vbox.addLayout(right_hdr)

        # Scrollable container for individual SpatialSurvivorCards
        self.surv_scroll = QScrollArea()
        self.surv_scroll.setWidgetResizable(True)
        self.surv_container = QWidget()
        self.surv_vbox = QVBoxLayout(self.surv_container)
        self.surv_vbox.setContentsMargins(0, 0, 0, 0)
        self.surv_vbox.setSpacing(6)

        self.lbl_no_surv = QLabel("Scanning arena for human survivors...\\n\\nPersistent MAP-frame coordinates will anchor here upon multi-frame confirmation.")
        self.lbl_no_surv.setWordWrap(True)
        self.lbl_no_surv.setFont(QFont(SPATIAL_FONT, 9))
        self.lbl_no_surv.setStyleSheet("color: #64748b; padding: 12px;")
        self.surv_vbox.addWidget(self.lbl_no_surv)
        self.surv_vbox.addStretch(1)

        self.surv_scroll.setWidget(self.surv_container)
        right_vbox.addWidget(self.surv_scroll, 1)

        workspace.addWidget(self.panel_right)
        root_vbox.addLayout(workspace, 1)

        # ── 3. BOTTOM FLOATING SPATIAL SHELF: SWAPPABLE PIP & JEV PIPELINE ──
        bottom_shelf = QHBoxLayout()
        bottom_shelf.setSpacing(12)

        # Clickable Secondary PiP Card (Tap to Rotate / Swap)
        self.pip_card = ClickableSpatialCard()
        self.pip_card.setFixedWidth(340)
        self.pip_card.setFixedHeight(195)
        self.pip_card.setStyleSheet(GLASS_CARD_STYLE)
        self.pip_card.setGraphicsEffect(make_shadow(20, 4, 100))
        self.pip_card.clicked.connect(self.toggle_view_mode)

        pip_vbox = QVBoxLayout(self.pip_card)
        pip_vbox.setContentsMargins(8, 8, 8, 8)
        pip_vbox.setSpacing(4)

        pip_hdr = QHBoxLayout()
        self.lbl_pip_title = QLabel("FPV DRONE CAMERA  ⛶")
        self.lbl_pip_title.setFont(QFont(SPATIAL_FONT, 8, QFont.Bold))
        self.lbl_pip_title.setStyleSheet("color: #38bdf8; letter-spacing: 0.5px;")
        pip_hdr.addWidget(self.lbl_pip_title)
        pip_hdr.addStretch(1)

        self.lbl_eis_badge = QLabel("🛡️ EIS")
        self.lbl_eis_badge.setFont(QFont(SPATIAL_FONT, 7, QFont.Bold))
        self.lbl_eis_badge.setStyleSheet("""
            background-color: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.35);
            border-radius: 4px;
            padding: 1px 6px;
        """)
        pip_hdr.addWidget(self.lbl_eis_badge)

        self.lbl_pip_live = QLabel("LIVE 1080p")
        self.lbl_pip_live.setFont(QFont(SPATIAL_FONT, 7, QFont.Bold))
        self.lbl_pip_live.setStyleSheet("color: #10b981;")
        pip_hdr.addWidget(self.lbl_pip_live)
        pip_vbox.addLayout(pip_hdr)

        self.pip_content_layout = QVBoxLayout()
        self.pip_content_layout.setContentsMargins(0, 0, 0, 0)
        pip_vbox.addLayout(self.pip_content_layout, 1)

        bottom_shelf.addWidget(self.pip_card)

        # Floating JEV Cognitive State Machine Card
        self.jev_widget = SpatialJevWidget()
        bottom_shelf.addWidget(self.jev_widget, 1)

        root_vbox.addLayout(bottom_shelf)

        # Create Swappable Components
        self.map_widget = ArenaMapWidget()
        self.map_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.camera_widget = CameraViewerWidget()
        self.camera_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Initial layout: Map in Hero, Camera in PiP
        self.hero_content_layout.addWidget(self.map_widget)
        self.pip_content_layout.addWidget(self.camera_widget)
        self.update_segmented_buttons()

    def update_segmented_buttons(self):
        active_map = (self.view_mode == 'map')
        active_style = """
            QPushButton {
                background-color: rgba(56, 189, 248, 0.28);
                border: 1px solid rgba(56, 189, 248, 0.65);
                border-radius: 9px;
                color: #ffffff;
                font-size: 11px;
                font-weight: 700;
                padding: 5px 12px;
            }
        """
        active_cam_style = """
            QPushButton {
                background-color: rgba(16, 185, 129, 0.28);
                border: 1px solid rgba(16, 185, 129, 0.65);
                border-radius: 9px;
                color: #ffffff;
                font-size: 11px;
                font-weight: 700;
                padding: 5px 12px;
            }
        """
        inactive_style = """
            QPushButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 9px;
                color: #94a3b8;
                font-size: 11px;
                font-weight: 500;
                padding: 5px 12px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.08);
                color: #f1f5f9;
            }
        """
        rotate_style = """
            QPushButton {
                background-color: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 9px;
                color: #f8fafc;
                font-size: 11px;
                font-weight: 600;
                padding: 5px 12px;
            }
            QPushButton:hover {
                background-color: rgba(56, 189, 248, 0.18);
                border-color: rgba(56, 189, 248, 0.45);
            }
        """
        self.btn_seg_map.setStyleSheet(active_style if active_map else inactive_style)
        self.btn_seg_cam.setStyleSheet(active_cam_style if not active_map else inactive_style)
        self.btn_seg_rotate.setStyleSheet(rotate_style)

    def set_view_mode(self, mode):
        if mode == self.view_mode:
            return
        self.view_mode = mode

        # Detach widgets
        self.hero_content_layout.removeWidget(self.map_widget)
        self.hero_content_layout.removeWidget(self.camera_widget)
        self.pip_content_layout.removeWidget(self.map_widget)
        self.pip_content_layout.removeWidget(self.camera_widget)

        if self.view_mode == "map":
            self.hero_content_layout.addWidget(self.map_widget)
            self.pip_content_layout.addWidget(self.camera_widget)
            self.lbl_hero_title.setText("2D TACTICAL SLAM ARENA MAP")
            self.lbl_hero_title.setStyleSheet("color: #38bdf8; letter-spacing: 0.8px;")
            self.lbl_hero_badge.setText("MAP FOCUS")
            self.lbl_hero_badge.setStyleSheet("background-color: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 5px; padding: 2px 7px;")
            self.lbl_pip_title.setText("FPV DRONE CAMERA  ⛶")
            self.lbl_pip_title.setStyleSheet("color: #34d399; letter-spacing: 0.5px;")
            self.lbl_pip_live.setText("LIVE 1080p")
            self.lbl_pip_live.setStyleSheet("color: #10b981;")
        else:
            self.hero_content_layout.addWidget(self.camera_widget)
            self.pip_content_layout.addWidget(self.map_widget)
            self.lbl_hero_title.setText("FPV DRONE CAMERA (LIVE 1080p HUD & AI DETECTIONS)")
            self.lbl_hero_title.setStyleSheet("color: #34d399; letter-spacing: 0.8px;")
            self.lbl_hero_badge.setText("CAMERA FOCUS")
            self.lbl_hero_badge.setStyleSheet("background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 5px; padding: 2px 7px;")
            self.lbl_pip_title.setText("2D TACTICAL SLAM MAP  ⛶")
            self.lbl_pip_title.setStyleSheet("color: #38bdf8; letter-spacing: 0.5px;")
            self.lbl_pip_live.setText(f"{self.backend.explored_pct:.0f}% MAPPED")
            self.lbl_pip_live.setStyleSheet("color: #38bdf8;")

        self.map_widget.show()
        self.camera_widget.show()
        self.update_segmented_buttons()

    def toggle_view_mode(self):
        new_mode = "camera" if self.view_mode == "map" else "map"
        self.set_view_mode(new_mode)

    def on_download_map_clicked(self):
        default_name = f"/home/dexter/Downloads/Drone-simulation-2-main/map_exports/nidar_tactical_map_{int(time.time())}.png"
        fn, _ = QFileDialog.getSaveFileName(self, "Export High-Res 2D SLAM LiDAR Map", default_name, "PNG Image (*.png)")
        if fn:
            if not fn.lower().endswith('.png'):
                fn += '.png'
            info = {
                'cov': self.backend.explored_pct,
                'time': self.lbl_big_timer.text(),
                'surv': self.backend.survivors_count,
                'survivors': list(self.backend.survivors_detailed or self.backend.survivors_list)
            }
            pix = self.map_widget.export_map_pixmap(info)
            saved = pix.save(fn, "PNG")
            self.btn_download_map.setText("✓ Map Saved" if saved else "Export Failed")
            QTimer.singleShot(1800, lambda: self.btn_download_map.setText("💾 Export Map"))

    def on_relaunch_clicked(self):
        self._auto_export_done = False
        self._last_surv_sig = None
        self.backend.send_relaunch_cmd()
        self.btn_land.setEnabled(True)
        self.btn_land.setText("🛩 Land / RTH")
        self.lbl_flight_mode.setText("RELAUNCHING")
        self.lbl_flight_mode.setStyleSheet("background-color: rgba(16, 185, 129, 0.2); color: #34d399; border-radius: 6px; padding: 2px 8px;")
        self.lbl_big_timer.setStyleSheet("""
            color: #34d399;
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 3px 14px;
        """)

    def on_land_clicked(self):
        self.backend.send_land_cmd()
        self.lbl_flight_mode.setText("RETURNING HOME")
        self.btn_land.setEnabled(False)
        self.btn_land.setText("Return Requested")
        QTimer.singleShot(3000, self._restore_land_button)

    def _restore_land_button(self):
        """Restore the command affordance while preserving the flight status."""
        self.btn_land.setEnabled(True)
        self.btn_land.setText("🛩 Land / RTH")

    def update_telemetry_ui(self):
        with self.backend.lock:
            # 1. Mission Stopwatch Timer
            if self.backend.start_time:
                if self.backend.mission_complete and self.backend.completion_time:
                    dur = self.backend.completion_time - self.backend.start_time
                    prefix = "🏁 FINAL: "
                    self.lbl_big_timer.setStyleSheet("""
                        color: #fbbf24;
                        background-color: rgba(245, 158, 11, 0.15);
                        border: 1px solid rgba(245, 158, 11, 0.40);
                        border-radius: 10px;
                        padding: 3px 14px;
                    """)
                    self.lbl_flight_mode.setText("TOUCHDOWN / LANDED")
                    self.lbl_flight_mode.setStyleSheet("background-color: rgba(59, 130, 246, 0.2); color: #93c5fd; border-radius: 6px; padding: 2px 8px;")
                else:
                    dur = time.time() - self.backend.start_time
                    prefix = ""
                m = int(dur // 60)
                s = int(dur % 60)
                ms = int((dur - int(dur)) * 10)
                self.lbl_big_timer.setText(f"{prefix}{m:02d}:{s:02d}.{ms}")

            # 2. Coverage
            exp_pct = self.backend.explored_pct
            self.bar_cov.setValue(int(exp_pct))
            self.lbl_cov_val.setText(f"{exp_pct:4.1f}% Mapped")

            # 3. Flight Telemetry
            alt = self.backend.drone_z
            vz = self.backend.climb_rate
            self.lbl_altitude.setText(f"{alt:.2f} m")
            self.lbl_climb.setText(f"Vz: {vz:+.2f} m/s  (Target: 1.50m)")

            # 4. Heading & Compass
            yaw_deg = int((math.degrees(self.backend.drone_yaw) + 360) % 360)
            dirs = ["NORTH", "NE", "EAST", "SE", "SOUTH", "SW", "WEST", "NW"]
            dir_str = dirs[int((yaw_deg + 22.5) // 45) % 8]
            self.lbl_heading.setText(f"🧭 HDG: {yaw_deg:03d}° [{dir_str}]")

            # 5. Position & Speed
            px, py = self.backend.drone_x, self.backend.drone_y
            spd = max(self.backend.speed, self.backend.pose_speed)
            self.lbl_coords.setText(f"📍 X: {px:+5.2f}m, Y: {py:+5.2f}m")
            self.lbl_speed.setText(f"🚀 SPEED: {spd:4.2f} m/s")

            # 6. Survivors Intel & Count
            cnt = self.backend.survivors_count
            self.lbl_surv_pill.setText(f"👥 {cnt} Survivors")
            self.lbl_surv_badge.setText(f"{cnt} CONFIRMED")

            # 7. JEV State Machine Feed
            self.jev_widget.update_status(self.backend.jev_status)

            # 8. Propeller & Radar
            self.prop_widget.update_motors(
                self.backend.cmd_vx, self.backend.cmd_vy, self.backend.cmd_vz, self.backend.cmd_wz, alt
            )
            self.radar_widget.update_scan(
                self.backend.lidar_ranges, self.backend.lidar_angle_min, self.backend.lidar_angle_inc
            )

            # 9. Map Update (Using detailed fixed survivors if available)
            active_survivors = self.backend.survivors_detailed if self.backend.survivors_detailed else self.backend.survivors_list
            self.map_widget.update_map_data(
                self.backend.map_grid, self.backend.map_w, self.backend.map_h,
                self.backend.map_res, self.backend.map_ox, self.backend.map_oy,
                px, py, alt, self.backend.drone_yaw,
                self.backend.flight_trail, self.backend.nav_path,
                self.backend.current_goal, active_survivors
            )

            # 10. Rebuild the roster only when its content changes.  takeAt()
            # removes both widgets and stretch items, preventing invisible
            # spacers from accumulating after repeated tracker updates.
            sig = [(s.get('id', i+1),
                    round(float(s.get('map_x', s.get('x', 0.0))), 2),
                    round(float(s.get('map_y', s.get('y', 0.0))), 2),
                    s.get('confidence', 98), s.get('grid', ''))
                   for i, s in enumerate(active_survivors)]
            if getattr(self, '_last_surv_sig', None) != sig:
                self._last_surv_sig = sig
                while self.surv_vbox.count():
                    item = self.surv_vbox.takeAt(0)
                    widget = item.widget()
                    if widget is not None and widget is not self.lbl_no_surv:
                        widget.deleteLater()

                if active_survivors:
                    self.lbl_no_surv.hide()
                    for survivor in active_survivors:
                        self.surv_vbox.addWidget(SpatialSurvivorCard(survivor))
                else:
                    self.lbl_no_surv.show()
                    self.surv_vbox.addWidget(self.lbl_no_surv)
                self.surv_vbox.addStretch(1)

            raw_frame = self.backend.camera_frame
            roll = self.backend.drone_roll
            pitch = self.backend.drone_pitch
            is_mission_done = self.backend.mission_complete

        # Auto-export snapshot on mission complete
        if is_mission_done and not self._auto_export_done:
            self._auto_export_done = True
            auto_path = "/home/dexter/Downloads/Drone-simulation-2-main/map_exports/nidar_lidar_map_latest.png"
            info = {
                'cov': exp_pct,
                'time': self.lbl_big_timer.text(),
                'surv': cnt,
                'survivors': list(active_survivors)
            }
            pix = self.map_widget.export_map_pixmap(info)
            pix.save(auto_path, "PNG")

        # Camera Display: Live Feed or Landing Standby
        if is_mission_done:
            self.camera_widget.set_standby({'cov': exp_pct, 'surv': cnt})
            if self.view_mode == "map":
                self.lbl_pip_live.setText("STANDBY")
                self.lbl_pip_live.setStyleSheet("color: #ef4444;")

        elif raw_frame is not None:
            if self.view_mode == "map":
                self.lbl_pip_live.setText("LIVE 1080p")
                self.lbl_pip_live.setStyleSheet("color: #10b981;")

            rgb_img = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            qh, qw, qch = rgb_img.shape
            q_img = QImage(rgb_img.data, qw, qh, qch * qw, QImage.Format_RGB888)
            pix = QPixmap.fromImage(q_img)

            hero_hud = None
            if self.view_mode == "camera":
                yaw_deg = int((math.degrees(self.backend.drone_yaw) + 360) % 360)
                hero_hud = {'alt': alt, 'vz': vz, 'spd': spd, 'hdg': yaw_deg}

            self.camera_widget.set_pixmap(pix, hero_hud)


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
def _spin_backend(backend):
    """Run ROS callbacks without printing a traceback during normal UI exit."""
    try:
        rclpy.spin(backend)
    except Exception:
        if rclpy.ok():
            raise


def main():
    rclpy.init()
    backend = GcsRosBackend()

    ros_thread = threading.Thread(target=_spin_backend, args=(backend,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    window = NidarGcsWindow(backend)
    window.show()

    exit_code = app.exec_()
    backend.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()

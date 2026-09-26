import re

with open('map/nidar_survivor_detector.py', 'r') as f:
    content = f.read()

# Add imports
content = content.replace('from sensor_msgs.msg import LaserScan', 'from sensor_msgs.msg import LaserScan, Image\nfrom cv_bridge import CvBridge')

# Add publisher
init_code = """
        self.pub_complete = self.create_publisher(Bool, '/nidar/mission_complete', 10)
        self.pub_camera = self.create_publisher(Image, '/nidar/camera_view', 10)
        self.cv_bridge = CvBridge()
"""
content = content.replace("self.pub_complete = self.create_publisher(Bool, '/nidar/mission_complete', 10)", init_code.strip())

# Change marker to ORANGE BLOCK
marker_old = """
            # Marker B: 3D Body Cylinder (Solid Red)
            m_pin = Marker()
            m_pin.header.frame_id = 'map'
            m_pin.header.stamp = now_msg
            m_pin.ns = 'survivors'
            m_pin.id = sid * 5 + 1
            m_pin.type = Marker.CYLINDER
            m_pin.action = Marker.ADD
            m_pin.pose.position.x = float(sx)
            m_pin.pose.position.y = float(sy)
            m_pin.pose.position.z = 0.90
            m_pin.pose.orientation.w = 1.0
            m_pin.scale.x = 0.60
            m_pin.scale.y = 0.60
            m_pin.scale.z = 1.80
            m_pin.color.r = 1.0
            m_pin.color.g = 0.0
            m_pin.color.b = 0.0
            m_pin.color.a = 1.0
            marker_array.markers.append(m_pin)
"""

marker_new = """
            # Marker B: 3D Body Block (Solid Orange)
            m_pin = Marker()
            m_pin.header.frame_id = 'map'
            m_pin.header.stamp = now_msg
            m_pin.ns = 'survivors'
            m_pin.id = sid * 5 + 1
            m_pin.type = Marker.CUBE
            m_pin.action = Marker.ADD
            m_pin.pose.position.x = float(sx)
            m_pin.pose.position.y = float(sy)
            m_pin.pose.position.z = 0.90
            m_pin.pose.orientation.w = 1.0
            m_pin.scale.x = 1.00
            m_pin.scale.y = 1.00
            m_pin.scale.z = 1.80
            m_pin.color.r = 1.0
            m_pin.color.g = 0.5
            m_pin.color.b = 0.0
            m_pin.color.a = 1.0
            marker_array.markers.append(m_pin)
"""
if "m_pin.type = Marker.CYLINDER" in content:
    content = content.replace(marker_old.strip(), marker_new.strip())

with open('map/nidar_survivor_detector.py', 'w') as f:
    f.write(content)

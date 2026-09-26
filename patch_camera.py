with open('map/nidar_survivor_detector.py', 'r') as f:
    content = f.read()

cb_old = """
    def cb_camera(self, msg: GzImage):
        # Throttle processing to ~12 FPS
        now = time.time()
        if (now - self.last_img_time) < 0.08:
            return
        self.last_img_time = now
        self.img_count += 1
        if self.img_count == 1 or (now - self._last_log_t > 15.0):
            self.get_logger().info(f'📷 [CAMERA ACTIVE] Processing camera frames ({msg.width}x{msg.height}, frame #{self.img_count})')
            self._last_log_t = now
"""

cb_new = """
    def cb_camera(self, msg: GzImage):
        # Publish to ROS 2
        try:
            ros_img = Image()
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.header.frame_id = "camera_link"
            ros_img.height = msg.height
            ros_img.width = msg.width
            ros_img.encoding = "rgb8"
            ros_img.is_bigendian = False
            ros_img.step = msg.width * 3
            ros_img.data = msg.data
            self.pub_camera.publish(ros_img)
        except Exception as e:
            pass

        # Throttle processing to ~12 FPS
        now = time.time()
        if (now - self.last_img_time) < 0.08:
            return
        self.last_img_time = now
        self.img_count += 1
        if self.img_count == 1 or (now - self._last_log_t > 15.0):
            self.get_logger().info(f'📷 [CAMERA ACTIVE] Processing camera frames ({msg.width}x{msg.height}, frame #{self.img_count})')
            self._last_log_t = now
"""
content = content.replace(cb_old.strip(), cb_new.strip())

with open('map/nidar_survivor_detector.py', 'w') as f:
    f.write(content)

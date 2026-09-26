with open('map/nidar_survivor_detector.py', 'r') as f:
    content = f.read()

marker_logic = """
            m_text.pose.orientation.w = 1.0
            m_text.scale.z = 0.50
            m_text.color.r = 1.0
            m_text.color.g = 1.0
            m_text.color.b = 1.0
            m_text.color.a = 1.0
            m_text.text = f'HUMAN #{sid}'
            marker_array.markers.append(m_text)
"""

new_marker_logic = """
            m_text.pose.orientation.w = 1.0
            m_text.scale.z = 0.50
            m_text.color.r = 1.0
            m_text.color.g = 1.0
            m_text.color.b = 1.0
            m_text.color.a = 1.0
            m_text.text = f'HUMAN #{sid}'
            marker_array.markers.append(m_text)

        # Huge Floating Text for Total Survivors Found
        m_total = Marker()
        m_total.header.frame_id = 'map'
        m_total.header.stamp = now_msg
        m_total.ns = 'survivors_total'
        m_total.id = 9999
        m_total.type = Marker.TEXT_VIEW_FACING
        m_total.action = Marker.ADD
        m_total.pose.position.x = 0.0
        m_total.pose.position.y = 0.0
        m_total.pose.position.z = 8.0
        m_total.pose.orientation.w = 1.0
        m_total.scale.z = 2.0  # Huge text!
        m_total.color.r = 1.0
        m_total.color.g = 0.5
        m_total.color.b = 0.0
        m_total.color.a = 1.0
        m_total.text = f"SURVIVORS FOUND: {len(self.confirmed_survivors)} / 6"
        marker_array.markers.append(m_total)
"""

if "m_text.text = f'HUMAN #{sid}'" in content:
    content = content.replace(marker_logic.strip(), new_marker_logic.strip())
    with open('map/nidar_survivor_detector.py', 'w') as f:
        f.write(content)
        print("Patched text marker successfully!")
else:
    print("Failed to patch text marker.")

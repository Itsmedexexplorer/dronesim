#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, GridCells
from geometry_msgs.msg import Point
import numpy as np

class ColorMapPublisher(Node):
    def __init__(self):
        super().__init__('color_map_publisher')
        
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.sub = self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)
        
        self.pub_free = self.create_publisher(GridCells, '/map_cells_free', map_qos)
        self.pub_walls = self.create_publisher(GridCells, '/map_cells_walls', map_qos)
        self.pub_unknown = self.create_publisher(GridCells, '/map_cells_unknown', map_qos)
        
        self.get_logger().info('Color Map Publisher started. Publishing GridCells.')

    def map_cb(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        
        c0 = max(0, int((-9.0 - ox) / res))
        c1 = min(width, int((9.0 - ox) / res))
        r0 = max(0, int((-9.0 - oy) / res))
        r1 = min(height, int((9.0 - oy) / res))
        
        sub_data = data[r0:r1, c0:c1]
        
        gc_free = GridCells()
        gc_free.header = msg.header
        gc_free.cell_width = res
        gc_free.cell_height = res
        
        gc_walls = GridCells()
        gc_walls.header = msg.header
        gc_walls.cell_width = res
        gc_walls.cell_height = res

        gc_unknown = GridCells()
        gc_unknown.header = msg.header
        gc_unknown.cell_width = res
        gc_unknown.cell_height = res
        
        # Fast vectorized extraction of cells
        # Free cells
        rf, cf = np.where(sub_data == 0)
        if len(rf) > 0:
            xf = ox + (cf + c0 + 0.5) * res
            yf = oy + (rf + r0 + 0.5) * res
            gc_free.cells = [Point(x=float(x), y=float(y), z=-0.05) for x, y in zip(xf, yf)]

        # Wall cells
        rw, cw = np.where(sub_data == 100)
        if len(rw) > 0:
            xw = ox + (cw + c0 + 0.5) * res
            yw = oy + (rw + r0 + 0.5) * res
            gc_walls.cells = [Point(x=float(x), y=float(y), z=-0.05) for x, y in zip(xw, yw)]

        # Unknown cells
        ru, cu = np.where(sub_data == -1)
        if len(ru) > 0:
            xu = ox + (cu + c0 + 0.5) * res
            yu = oy + (ru + r0 + 0.5) * res
            gc_unknown.cells = [Point(x=float(x), y=float(y), z=-0.05) for x, y in zip(xu, yu)]

        self.pub_free.publish(gc_free)
        self.pub_walls.publish(gc_walls)
        self.pub_unknown.publish(gc_unknown)

def main(args=None):
    rclpy.init(args=args)
    node = ColorMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

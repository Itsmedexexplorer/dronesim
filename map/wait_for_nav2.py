#!/usr/bin/env python3
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

def main():
    rclpy.init()
    node = Node("nav2_waiter")
    client = ActionClient(node, NavigateToPose, "navigate_to_pose")
    print("Waiting for Nav2 /navigate_to_pose Action Server to become online...")
    ready = False
    start_t = time.time()
    for i in range(30):
        if client.wait_for_server(timeout_sec=1.0):
            ready = True
            elapsed = time.time() - start_t
            print(f"✅ Nav2 /navigate_to_pose Action Server is ONLINE! ({elapsed:.1f}s)")
            break
        print(f"  • Still initializing Nav2 stack... ({i+1}/30s)")
        time.sleep(0.5)

    rclpy.shutdown()
    sys.exit(0 if ready else 1)

if __name__ == "__main__":
    main()

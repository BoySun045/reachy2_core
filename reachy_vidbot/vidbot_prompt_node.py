#!/usr/bin/env python3
"""
Terminal prompt node for VidBot.

Asks the user for an object name and instruction, then publishes a trigger
message to /vidbot/trigger so the VidBot ROS node starts inference.
"""
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VidBotPromptNode(Node):
    def __init__(self):
        super().__init__("vidbot_prompt")
        self.trigger_pub = self.create_publisher(String, "/vidbot/trigger", 10)

    def prompt_and_publish(self):
        print()
        obj = input("Object: ").strip()
        instruction = input("Instruction: ").strip()

        if not obj or not instruction:
            self.get_logger().warn("Empty input, skipping")
            return

        msg = String()
        msg.data = json.dumps({"object": obj, "instruction": instruction})
        self.trigger_pub.publish(msg)
        self.get_logger().info(
            f'Triggered vidbot: object="{obj}", instruction="{instruction}"'
        )


def main(args=None):
    rclpy.init(args=args)
    node = VidBotPromptNode()

    # Brief spin so the publisher discovery handshake completes
    rclpy.spin_once(node, timeout_sec=0.5)

    try:
        while rclpy.ok():
            node.prompt_and_publish()
    except (KeyboardInterrupt, EOFError):
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

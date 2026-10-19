"""Quick camera connectivity check — subscribes to R1Pro camera topics and
reports which ones are publishing frames.  Exits 0 if all cameras OK, 1 if any
are missing.

Usage:
    python scripts/check_cameras.py [--timeout 3]
"""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


CAMERAS = {
    "left_wrist": {
        "topic": "/hdas/camera_wrist_left/color/image_raw",
        "compressed": False,
    },
    "right_wrist": {
        "topic": "/hdas/camera_wrist_right/color/image_raw",
        "compressed": False,
    },
    "zed_head": {
        "topic": "/hdas/camera_head/left_raw/image_raw_color/compressed",
        "compressed": True,
    },
}


class CameraChecker(Node):
    def __init__(self):
        super().__init__("camera_checker")
        self.received = {name: False for name in CAMERAS}

        for name, cfg in CAMERAS.items():
            msg_type = CompressedImage if cfg["compressed"] else Image
            cb = self._make_cb(name)
            self.create_subscription(msg_type, cfg["topic"], cb, qos_profile_sensor_data)

    def _make_cb(self, name: str):
        def _cb(msg):
            self.received[name] = True
        return _cb

    def all_ok(self) -> bool:
        return all(self.received.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=3.0, help="Seconds to wait for frames")
    args = parser.parse_args()

    rclpy.init()
    node = CameraChecker()

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and not node.all_ok():
        rclpy.spin_once(node, timeout_sec=0.1)

    # Print results
    all_ok = True
    for name in CAMERAS:
        status = "\033[0;32mOK\033[0m" if node.received[name] else "\033[0;31mNO FRAMES\033[0m"
        topic = CAMERAS[name]["topic"]
        print(f"  {name:15s}  {status}  ({topic})")
        if not node.received[name]:
            all_ok = False

    node.destroy_node()
    rclpy.shutdown()

    if not all_ok:
        print("\n\033[0;31mSome cameras are not publishing. Check connections/drivers.\033[0m")
        sys.exit(1)
    else:
        print("\n\033[0;32mAll cameras OK.\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()

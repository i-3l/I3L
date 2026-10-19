#!/usr/bin/env python3
"""
Script to view camera images from ROS topics with and without cropping.
Helps decide on the appropriate crop settings.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.bridge = CvBridge()

        # Storage for frames
        self.frames = {
            'left_wrist': None,
            'right_wrist': None,
            'zed': None,
        }

        # Storage for camera info
        self.camera_info = {
            'left_wrist': None,
            'right_wrist': None,
            'zed': None,
        }

        # Camera image topics
        self.topics = {
            'left_wrist': '/hdas/camera_wrist_left/color/image_raw',
            'right_wrist': '/hdas/camera_wrist_right/color/image_raw',
            'zed': '/hdas/camera_head/left_raw/image_raw_color/compressed',
        }

        # Camera info topics (ZED uses calibration topic)
        self.info_topics = {
            'left_wrist': '/hdas/camera_wrist_left/color/camera_info',
            'right_wrist': '/hdas/camera_wrist_right/color/camera_info',
            'zed': '/calib/head_left/camera_info',
        }

        # Create image subscriptions
        for name, topic in self.topics.items():
            # Use CompressedImage for compressed topics
            if topic.endswith('/compressed'):
                self.create_subscription(
                    CompressedImage,
                    topic,
                    lambda msg, n=name: self.compressed_image_callback(msg, n),
                    10
                )
            else:
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, n=name: self.image_callback(msg, n),
                    10
                )
            self.get_logger().info(f'Subscribed to {topic}')

        # Create camera info subscriptions
        for name, topic in self.info_topics.items():
            self.create_subscription(
                CameraInfo,
                topic,
                lambda msg, n=name: self.camera_info_callback(msg, n),
                10
            )
            self.get_logger().info(f'Subscribed to {topic}')

    def image_callback(self, msg, camera_name):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.frames[camera_name] = frame
        except Exception as e:
            self.get_logger().warning(f'Failed to convert {camera_name}: {e}')

    def compressed_image_callback(self, msg, camera_name):
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames[camera_name] = frame
            else:
                self.get_logger().warning(f'Failed to decode compressed {camera_name}')
        except Exception as e:
            self.get_logger().warning(f'Failed to convert compressed {camera_name}: {e}')

    def camera_info_callback(self, msg, camera_name):
        self.camera_info[camera_name] = msg

    def all_frames_received(self):
        return all(f is not None for f in self.frames.values())

    def all_camera_info_received(self):
        return all(info is not None for info in self.camera_info.values())

    def print_camera_parameters(self):
        """Print all camera parameters to console."""
        print("\n" + "=" * 80)
        print("CAMERA PARAMETERS")
        print("=" * 80)

        for name in ['left_wrist', 'right_wrist', 'zed']:
            info = self.camera_info[name]
            frame = self.frames[name]

            print(f"\n{'─' * 40}")
            print(f"  {name.upper()}")
            print(f"{'─' * 40}")

            if info is None:
                print("  Camera info not available")
                continue

            # Basic info
            print(f"  Resolution: {info.width} x {info.height}")
            if frame is not None:
                print(f"  Frame shape: {frame.shape}")

            # Distortion model and coefficients
            print(f"  Distortion model: {info.distortion_model}")
            d = list(info.d)
            print(f"  Distortion coeffs (D): {d}")
            if len(d) >= 5:
                print(f"    k1={d[0]:.6f}, k2={d[1]:.6f}, p1={d[2]:.6f}, p2={d[3]:.6f}, k3={d[4]:.6f}")

            # Intrinsic matrix K (3x3)
            k = np.array(info.k).reshape(3, 3)
            fx, fy = k[0, 0], k[1, 1]
            cx, cy = k[0, 2], k[1, 2]
            print(f"  Intrinsic Matrix (K):")
            print(f"    fx={fx:.2f}, fy={fy:.2f}")
            print(f"    cx={cx:.2f}, cy={cy:.2f}")

            # Calculate FOV
            if info.width > 0 and info.height > 0:
                fov_h = 2 * math.atan(info.width / (2 * fx)) * 180 / math.pi
                fov_v = 2 * math.atan(info.height / (2 * fy)) * 180 / math.pi
                fov_diag = 2 * math.atan(
                    math.sqrt(info.width**2 + info.height**2) / (2 * math.sqrt(fx * fy))
                ) * 180 / math.pi
                print(f"  Field of View:")
                print(f"    Horizontal: {fov_h:.1f} deg")
                print(f"    Vertical:   {fov_v:.1f} deg")
                print(f"    Diagonal:   {fov_diag:.1f} deg")

            # Rectification matrix R (3x3)
            r = np.array(info.r).reshape(3, 3)
            if not np.allclose(r, np.eye(3)):
                print(f"  Rectification Matrix (R):")
                for row in r:
                    print(f"    [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")

            # Projection matrix P (3x4)
            p = np.array(info.p).reshape(3, 4)
            print(f"  Projection Matrix (P):")
            print(f"    fx'={p[0, 0]:.2f}, fy'={p[1, 1]:.2f}")
            print(f"    cx'={p[0, 2]:.2f}, cy'={p[1, 2]:.2f}")
            print(f"    Tx={p[0, 3]:.4f}, Ty={p[1, 3]:.4f}")

            # Binning
            if info.binning_x > 0 or info.binning_y > 0:
                print(f"  Binning: {info.binning_x} x {info.binning_y}")

            # ROI
            roi = info.roi
            if roi.width > 0 and roi.height > 0:
                print(f"  ROI: x={roi.x_offset}, y={roi.y_offset}, w={roi.width}, h={roi.height}")
                print(f"    do_rectify: {roi.do_rectify}")

        print("\n" + "=" * 80 + "\n")


def center_crop(frame, target_w, target_h, x_offset=0):
    """Center crop an image to target size with optional x offset."""
    h, w = frame.shape[:2]
    start_y = max(0, (h - target_h) // 2)
    start_x = max(0, (w - target_w) // 2)
    start_x = min(start_x + x_offset, w - target_w)
    return frame[start_y:start_y + target_h, start_x:start_x + target_w], start_x, start_y


def main():
    rclpy.init()
    node = CameraViewer()

    print("Waiting for camera frames and camera info...")

    # Spin until we have all frames and camera info
    while not node.all_frames_received() or not node.all_camera_info_received():
        rclpy.spin_once(node, timeout_sec=0.1)

    print("All frames and camera info received!")

    # Print camera parameters to console
    node.print_camera_parameters()

    # Target crop size
    target_size = (240, 240)

    # Create figure with 3 rows x 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    fig.suptitle('Camera Images: Original (top) vs Center Crop (middle) vs Squish (bottom)', fontsize=14)

    camera_names = ['left_wrist', 'right_wrist', 'zed']

    # Offset for wrist cameras (20px right), zed stays centered
    offsets = {'left_wrist': 20, 'right_wrist': 20, 'zed': 0}

    for col, name in enumerate(camera_names):
        frame = node.frames[name]
        h, w = frame.shape[:2]
        x_offset = offsets[name]

        # Original image
        axes[0, col].imshow(frame)
        axes[0, col].set_title(f'{name}\nOriginal: {w}x{h}')
        axes[0, col].axis('off')

        # Cropped image with offset
        crop_h, crop_w = target_size
        cropped, start_x, start_y = center_crop(frame, crop_w, crop_h, x_offset)

        # Draw crop rectangle on original
        rect = plt.Rectangle((start_x, start_y), crop_w, crop_h,
                              linewidth=2, edgecolor='red', facecolor='none')
        axes[0, col].add_patch(rect)

        # Show cropped image
        axes[1, col].imshow(cropped)
        offset_str = f' (offset +{x_offset}px)' if x_offset else ''
        axes[1, col].set_title(f'Crop: {crop_w}x{crop_h}{offset_str}')
        axes[1, col].axis('off')

        # Squish view (resize without crop)
        squished = cv2.resize(frame, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
        axes[2, col].imshow(squished)
        axes[2, col].set_title(f'Squish: {w}x{h} -> {crop_w}x{crop_h}')
        axes[2, col].axis('off')

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'camera_crop_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {save_path}")

    plt.show()

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

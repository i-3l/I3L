#!/usr/bin/env python3
"""Live viewer for ZED cameras on the Franka workstation.

Usage:
    python scripts/view_zed_cameras.py                    # View all connected ZED cameras
    python scripts/view_zed_cameras.py /dev/zed_external  # View specific device
"""

import cv2
import glob
import os
import sys


def open_camera(device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    return cap


def process_frame(frame):
    """Extract left image from side-by-side stereo."""
    h, w = frame.shape[:2]
    if w > h * 1.5:
        frame = frame[:, : w // 2]
    return frame


def main():
    devices = sys.argv[1:] if len(sys.argv) > 1 else []

    # Auto-detect if no devices specified
    if not devices:
        for sym in ["/dev/zed_wrist", "/dev/zed_external"]:
            if os.path.exists(sym):
                devices.append(sym)
        if not devices:
            devices = sorted(glob.glob("/dev/video*"))

    if not devices:
        print("No camera devices found.")
        sys.exit(1)

    caps = {}
    for dev in devices:
        cap = open_camera(dev)
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                left = process_frame(frame)
                lh, lw = left.shape[:2]
                print(f"  {dev}: {w}x{h} raw -> {lw}x{lh} left image")
                caps[dev] = cap
            else:
                print(f"  {dev}: opened but no frame")
                cap.release()
        else:
            print(f"  {dev}: failed to open")

    if not caps:
        print("No cameras could be opened.")
        sys.exit(1)

    print(f"\nShowing {len(caps)} camera(s). Press q to quit.")

    while True:
        for dev, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                frame = process_frame(frame)
                h, w = frame.shape[:2]
                if h > 480:
                    scale = 480 / h
                    frame = cv2.resize(frame, (int(w * scale), 480))
                name = dev.replace("/dev/", "")
                cv2.imshow(name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

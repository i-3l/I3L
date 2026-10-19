#!/usr/bin/env python3
"""Stitch base and residual per-episode videos side-by-side."""

import argparse
import cv2
import numpy as np
from pathlib import Path


def stitch_videos(left_path, right_path, out_path, left_label="Base Policy", right_label="Residual Policy"):
    cap_l = cv2.VideoCapture(str(left_path))
    cap_r = cv2.VideoCapture(str(right_path))

    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30
    w_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))

    h = max(h_l, h_r)
    w_total = w_l + w_r + 4  # 4px divider

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_total, h))

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l and not ret_r:
            break
        if not ret_l:
            frame_l = np.zeros((h_l, w_l, 3), dtype=np.uint8)
        if not ret_r:
            frame_r = np.zeros((h_r, w_r, 3), dtype=np.uint8)

        # Resize to same height
        if frame_l.shape[0] != h:
            frame_l = cv2.resize(frame_l, (w_l, h))
        if frame_r.shape[0] != h:
            frame_r = cv2.resize(frame_r, (w_r, h))

        divider = np.ones((h, 4, 3), dtype=np.uint8) * 128
        combined = np.hstack([frame_l, divider, frame_r])

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, left_label, (10, 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, left_label, (10, 25), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        rx = w_l + 14
        cv2.putText(combined, right_label, (rx, 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, right_label, (rx, 25), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(combined)

    cap_l.release()
    cap_r.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/eval_matched")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    d = Path(args.dir)
    for i in range(args.episodes):
        left = d / f"base_ep{i}.mp4"
        right = d / f"residual_ep{i}.mp4"
        out = d / f"comparison_ep{i}.mp4"

        if not left.exists():
            print(f"Skipping ep {i}: {left} not found")
            continue
        if not right.exists():
            print(f"Skipping ep {i}: {right} not found")
            continue

        stitch_videos(left, right, out)
        print(f"Created {out}")

    print("Done!")


if __name__ == "__main__":
    main()

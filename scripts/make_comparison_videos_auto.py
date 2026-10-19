#!/usr/bin/env python3
"""Stitch base and residual per-episode videos side-by-side."""

import argparse
import cv2
import json
import numpy as np
from pathlib import Path


def stitch_pair(left_path, right_path, out_path, left_label, right_label):
    cap_l = cv2.VideoCapture(str(left_path))
    cap_r = cv2.VideoCapture(str(right_path))

    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30
    w_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))

    h = max(h_l, h_r)
    divider_w = 4
    w_total = w_l + w_r + divider_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_total, h))

    frame_count = 0
    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l and not ret_r:
            break
        if not ret_l:
            frame_l = np.zeros((h_l, w_l, 3), dtype=np.uint8)
        if not ret_r:
            frame_r = np.zeros((h_r, w_r, 3), dtype=np.uint8)

        if frame_l.shape[0] != h:
            frame_l = cv2.resize(frame_l, (w_l, h))
        if frame_r.shape[0] != h:
            frame_r = cv2.resize(frame_r, (w_r, h))

        divider = np.ones((h, divider_w, 3), dtype=np.uint8) * 128
        combined = np.hstack([frame_l, divider, frame_r])

        font = cv2.FONT_HERSHEY_SIMPLEX
        for text, x in [(left_label, 10), (right_label, w_l + divider_w + 10)]:
            cv2.putText(combined, text, (x, 25), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, text, (x, 25), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(combined)
        frame_count += 1

    cap_l.release()
    cap_r.release()
    writer.release()
    return frame_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--residual-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    residual_dir = Path(args.residual_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.episodes):
        left = base_dir / f"ep{i}.mp4"
        right = residual_dir / f"ep{i}.mp4"
        out = output_dir / f"comparison_ep{i}.mp4"

        if not left.exists():
            print(f"  Skip ep{i}: {left} not found")
            continue
        if not right.exists():
            print(f"  Skip ep{i}: {right} not found")
            continue

        # Load per-episode results for labels
        left_label = "Base Policy"
        right_label = "Residual Policy"
        left_res = base_dir / f"ep{i}_results.json"
        right_res = residual_dir / f"ep{i}_results.json"
        if left_res.exists():
            sr = json.load(open(left_res))["SR"]
            left_label += f" ({'OK' if sr > 0 else 'FAIL'})"
        if right_res.exists():
            sr = json.load(open(right_res))["SR"]
            right_label += f" ({'OK' if sr > 0 else 'FAIL'})"

        frames = stitch_pair(left, right, out, left_label, right_label)
        print(f"  Created {out} ({frames} frames)")

    print("Done!")


if __name__ == "__main__":
    main()

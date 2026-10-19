#!/usr/bin/env python3
"""
Convert HDF5 camera frames to video files.

Usage:
    python scripts/data/hdf5_to_video.py /path/to/data.hdf5
    python scripts/data/hdf5_to_video.py /path/to/data.hdf5 --demo demo_0
    python scripts/data/hdf5_to_video.py /path/to/data.hdf5 --fps 30 --output /path/to/output
"""

import argparse
import cv2
import h5py
import numpy as np
from pathlib import Path


def create_video(frames: np.ndarray, output_path: str, fps: int = 30) -> None:
    """Create a video from a numpy frames array."""
    if len(frames) == 0:
        print(f"  Skipping {output_path}: no frames")
        return

    h, w = frames.shape[1:3]

    # Write MP4 output with a codec OpenCV commonly supports.
    output_path = str(output_path).rsplit('.', 1)[0] + '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        print(f"  Error: Could not open video writer")
        return

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

    writer.release()
    print(f"  Saved: {output_path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")


def extract_camera_videos(hdf5_path: str, output_dir: str = None, demo: str = None, fps: int = 30) -> None:
    """Extract camera frames from HDF5 and save as videos."""
    hdf5_path = Path(hdf5_path)
    if output_dir is None:
        output_dir = hdf5_path.parent / f"{hdf5_path.stem}_videos"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f:
            print(f"Error: No 'data' group in {hdf5_path}")
            return

        demos = [demo] if demo else sorted(f['data'].keys())

        for demo_name in demos:
            if demo_name not in f['data']:
                print(f"Warning: {demo_name} not found, skipping")
                continue

            print(f"\nProcessing {demo_name}...")
            obs = f['data'][demo_name]['obs']

            # Find camera keys (contain 'rgb' in the name)
            camera_keys = [k for k in obs.keys() if 'rgb' in k.lower()]

            if not camera_keys:
                print(f"  No camera data found in {demo_name}")
                continue

            for camera_key in camera_keys:
                frames = obs[camera_key][:]

                # Create a short name for the camera
                # e.g., "robot::robot:left_realsense_link:Camera:0::rgb" -> "left_realsense"
                if 'left_realsense' in camera_key:
                    short_name = 'left_realsense'
                elif 'right_realsense' in camera_key:
                    short_name = 'right_realsense'
                elif 'zed_wrist' in camera_key:
                    short_name = 'zed_wrist'
                elif 'zed_external' in camera_key:
                    short_name = 'zed_external'
                elif 'zed' in camera_key:
                    short_name = 'zed'
                else:
                    # Fallback: use part of the key
                    short_name = camera_key.replace('::', '_').replace(':', '_')

                output_path = output_dir / f"{demo_name}_{short_name}.mp4"
                create_video(frames, str(output_path), fps=fps)


def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 camera frames to videos')
    parser.add_argument('hdf5_path', type=str, help='Path to HDF5 file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory')
    parser.add_argument('--demo', '-d', type=str, default=None, help='Specific demo to process (e.g., demo_0)')
    parser.add_argument('--fps', type=int, default=30, help='Output video FPS (default: 30)')

    args = parser.parse_args()

    print(f"Converting {args.hdf5_path} to videos...")
    extract_camera_videos(args.hdf5_path, args.output, args.demo, args.fps)
    print("\nDone!")


if __name__ == '__main__':
    main()

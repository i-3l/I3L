#!/usr/bin/env python3
"""
Visualize joint positions and velocities as timeseries from HDF5 data.

Usage:
    python scripts/vis/visualize_hdf5_data.py -i data/dataset.hdf5
    python scripts/vis/visualize_hdf5_data.py -i data/dataset.hdf5 -d 0 -o plots/
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional


# R1Pro joint layout (23 DOF)
JOINT_GROUPS = {
    "base": {"start": 0, "end": 3, "names": ["x", "y", "rz"]},
    "torso": {"start": 3, "end": 7, "names": ["t0", "t1", "t2", "t3"]},
    "left_arm": {"start": 7, "end": 14, "names": ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]},
    "left_gripper": {"start": 14, "end": 15, "names": ["L_grip"]},
    "right_arm": {"start": 15, "end": 22, "names": ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]},
    "right_gripper": {"start": 22, "end": 23, "names": ["R_grip"]},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize HDF5 joint data as timeseries")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("-d", "--demo", type=int, default=0, help="Demo index (default: 0)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--freq", type=float, default=30.0, help="Control frequency Hz (default: 30)")
    parser.add_argument("--show-actions", action="store_true", help="Overlay action commands")
    return parser.parse_args()


def load_demo(filepath: str, demo_idx: int) -> Dict[str, np.ndarray]:
    """Load position, velocity, and action data from a demo."""
    with h5py.File(filepath, "r") as f:
        demo = f["data"][f"demo_{demo_idx}"]
        proprio = demo["obs/robot::proprio"][:]
        n_joints = proprio.shape[1] // 2

        data = {
            "positions": proprio[:, :n_joints],
            "velocities": proprio[:, n_joints:],
        }

        if "action" in demo:
            data["actions"] = demo["action"][:]

        return data


def plot_timeseries(data: Dict[str, np.ndarray], demo_idx: int, freq: float,
                    show_actions: bool = False) -> Dict[str, plt.Figure]:
    """Create timeseries plots for each joint group."""
    positions = data["positions"]
    velocities = data["velocities"]
    actions = data.get("actions") if show_actions else None

    T = positions.shape[0]
    time = np.arange(T) / freq

    figures = {}

    for group_name, group in JOINT_GROUPS.items():
        start, end = group["start"], group["end"]
        names = group["names"]
        n_joints = end - start

        fig, axes = plt.subplots(n_joints, 2, figsize=(14, 2.5 * n_joints), sharex=True)
        if n_joints == 1:
            axes = axes.reshape(1, -1)

        for i, name in enumerate(names):
            joint_idx = start + i

            # Position timeseries
            ax = axes[i, 0]
            ax.plot(time, positions[:, joint_idx], 'b-', linewidth=1.2, label='Position')
            if actions is not None:
                ax.plot(time, actions[:, joint_idx], 'r--', linewidth=1, alpha=0.7, label='Action')
            ax.set_ylabel(f"{name}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

            # Velocity timeseries
            ax = axes[i, 1]
            ax.plot(time, velocities[:, joint_idx], 'g-', linewidth=1.2)
            ax.set_ylabel(f"{name}")
            ax.grid(True, alpha=0.3)

        axes[0, 0].set_title("Joint Position (rad)", fontsize=11)
        axes[0, 1].set_title("Joint Velocity (rad/s)", fontsize=11)
        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")

        fig.suptitle(f"Demo {demo_idx} - {group_name.upper()} ({T} steps, {T/freq:.1f}s)",
                     fontsize=12, fontweight='bold')
        fig.tight_layout()

        figures[group_name] = fig

    return figures


def print_info(filepath: str, demo_idx: int):
    """Print data info."""
    with h5py.File(filepath, "r") as f:
        n_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])
        demo = f["data"][f"demo_{demo_idx}"]
        proprio = demo["obs/robot::proprio"]

        print(f"\nFile: {filepath}")
        print(f"Demos: {n_demos}")
        print(f"Demo {demo_idx}: {proprio.shape[0]} steps, proprio shape {proprio.shape}")
        print("\nJoint layout (23 DOF):")
        for name, g in JOINT_GROUPS.items():
            print(f"  {name:14s}: indices {g['start']:2d}-{g['end']-1:2d}  {g['names']}")
        print()


def main():
    args = parse_args()

    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        return 1

    print_info(args.input, args.demo)

    data = load_demo(args.input, args.demo)
    figures = plot_timeseries(data, args.demo, args.freq, args.show_actions)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            path = out_dir / f"demo{args.demo}_{name}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
            plt.close(fig)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    exit(main())

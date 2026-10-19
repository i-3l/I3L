#!/usr/bin/env python3
"""
Plot joint angle traces from HDF5 demo data.

Each joint gets its own row. Shows both proprio (observed) and action (commanded).
By default overlays all demos on a single plot; use --demo for a specific one.

Usage:
    python scripts/data/plot_joint_traces.py /path/to/data.hdf5
    python scripts/data/plot_joint_traces.py /path/to/data.hdf5 --demo demo_0
    python scripts/data/plot_joint_traces.py /path/to/data.hdf5 --per-demo
"""

import argparse
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


from iiil.plotting.constants import FRANKA_JOINT_NAMES

# Distinct colors for overlaying many demos
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def load_demo(f, demo_name):
    """Load action and proprio arrays from an open HDF5 file."""
    demo = f[f"data/{demo_name}"]
    actions = demo["action"][:]
    proprio = None
    for k in demo["obs"].keys():
        if "proprio" in k.lower():
            proprio = demo["obs"][k][:]
            break
    return actions, proprio


def plot_all_demos(hdf5_path: str, demo_names: list, output_dir: Path, fps: int = 30):
    """Overlay all demos on a single plot, one row per joint."""
    with h5py.File(hdf5_path, "r") as f:
        all_data = []
        for name in demo_names:
            actions, proprio = load_demo(f, name)
            all_data.append((name, actions, proprio))

    n_joints = all_data[0][1].shape[1]
    fig, axes = plt.subplots(n_joints, 1, figsize=(20, 3 * n_joints), sharex=False)
    if n_joints == 1:
        axes = [axes]

    for demo_idx, (name, actions, proprio) in enumerate(all_data):
        color = COLORS[demo_idx % len(COLORS)]
        t = np.arange(actions.shape[0]) / fps
        for j in range(n_joints):
            ax = axes[j]
            ax.plot(t, actions[:, j], color=color, linewidth=0.8, alpha=0.6)

    for j in range(n_joints):
        ax = axes[j]
        label = FRANKA_JOINT_NAMES[j] if j < len(FRANKA_JOINT_NAMES) else f"Joint {j}"
        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    stem = Path(hdf5_path).stem
    fig.suptitle(f"{stem} — {len(demo_names)} demos (action)", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = output_dir / f"{stem}_all_joint_traces.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_single_demo(hdf5_path: str, demo_name: str, output_dir: Path, fps: int = 30):
    """Plot a single demo with action and proprio."""
    with h5py.File(hdf5_path, "r") as f:
        actions, proprio = load_demo(f, demo_name)

    n_joints = actions.shape[1]
    t_action = np.arange(actions.shape[0]) / fps

    fig, axes = plt.subplots(n_joints, 1, figsize=(20, 3 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    for j in range(n_joints):
        ax = axes[j]
        ax.plot(t_action, actions[:, j], label="action", color="tab:blue", linewidth=1.2)
        if proprio is not None and j < proprio.shape[1]:
            t_proprio = np.arange(proprio.shape[0]) / fps
            ax.plot(t_proprio, proprio[:, j], label="proprio", color="tab:orange", linewidth=1.2, alpha=0.7)

        label = FRANKA_JOINT_NAMES[j] if j < len(FRANKA_JOINT_NAMES) else f"Joint {j}"
        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        if j == 0:
            ax.legend(loc="upper right", fontsize=10)

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.suptitle(f"{Path(hdf5_path).stem} / {demo_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = output_dir / f"{demo_name}_joint_traces.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot joint angle traces from HDF5")
    parser.add_argument("hdf5_path", type=str, help="Path to HDF5 file")
    parser.add_argument("--demo", "-d", type=str, default=None, help="Specific demo to plot (e.g. demo_0)")
    parser.add_argument("--per-demo", action="store_true", help="Generate separate plot per demo instead of overlay")
    parser.add_argument("--fps", type=int, default=30, help="Control frequency (default: 30)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_path)
    output_dir = Path(args.output) if args.output else hdf5_path.parent / f"{hdf5_path.stem}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))

    if args.demo:
        # Single demo with action + proprio
        print(f"Plotting {args.demo} from {hdf5_path}")
        plot_single_demo(str(hdf5_path), args.demo, output_dir, args.fps)
    elif args.per_demo:
        # Separate plot per demo
        print(f"Plotting {len(demos)} demos from {hdf5_path}")
        for demo_name in demos:
            print(f"\nProcessing {demo_name}...")
            plot_single_demo(str(hdf5_path), demo_name, output_dir, args.fps)
    else:
        # All demos overlaid on one plot (default)
        print(f"Plotting {len(demos)} demos overlaid from {hdf5_path}")
        plot_all_demos(str(hdf5_path), demos, output_dir, args.fps)

    print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze per-joint tracking error statistics from GELLO debug logs.

Loads an HDF5 debug log, filters to tracking-active samples, and computes
per-joint statistics on signed_position_error and position_error (abs).
Optionally breaks down errors by takeover state and generates plots.

Usage:
    # Analyze latest session
    python scripts/analyze_tracking_errors.py

    # Analyze specific file
    python scripts/analyze_tracking_errors.py --log-file path/to/session.h5

    # Save plot
    python scripts/analyze_tracking_errors.py --plot --output plot.png
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from iiil.gello.debug.debug_logger import load_debug_log


def find_latest_log() -> Path:
    """Find the most recent debug log file."""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        raise FileNotFoundError(
            "No outputs directory found. Run this script from the project root."
        )

    log_files = sorted(outputs_dir.glob("*/*/logs/debug/session_*.h5"))
    if not log_files:
        raise FileNotFoundError(
            f"No debug log files found in {outputs_dir}/*/*/logs/debug/\n"
            "Make sure debug logging is enabled in your config."
        )

    latest = log_files[-1]
    return latest


def compute_stats(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-joint statistics along axis 0.

    Args:
        arr: Array of shape (N, dof).

    Returns:
        Dict with mean, max, min, std, rms per joint.
    """
    return {
        "Mean": np.mean(arr, axis=0),
        "Max": np.max(arr, axis=0),
        "Min": np.min(arr, axis=0),
        "Std": np.std(arr, axis=0),
        "RMS": np.sqrt(np.mean(arr ** 2, axis=0)),
    }


def print_stats_table(title: str, stats: Dict[str, np.ndarray], dof: int) -> None:
    """Print a formatted table of per-joint statistics."""
    print(f"\n{title}")
    print("-" * len(title))

    # Header row
    joint_headers = [f"J{i}" for i in range(dof)]
    header = f"{'Stat':<6}" + "".join(f"{h:>10}" for h in joint_headers)
    print(header)
    print("-" * len(header))

    for stat_name, values in stats.items():
        row = f"{stat_name:<6}" + "".join(f"{v:>10.4f}" for v in values)
        print(row)


def _shade_regions(ax, time, tracking_mask, takeover_data):
    """Add tracking and takeover shading to an axis."""
    ylim = ax.get_ylim()
    ax.fill_between(
        time, ylim[0], ylim[1],
        where=tracking_mask, alpha=0.08, color="green",
    )
    if takeover_data is not None:
        ax.fill_between(
            time, ylim[0], ylim[1],
            where=takeover_data.astype(bool),
            alpha=0.15, color="red",
        )
    ax.set_ylim(ylim)


def plot_tracking_errors(
    data: Dict[str, np.ndarray],
    tracking_mask: np.ndarray,
    dof: int,
    output_dir: str,
) -> None:
    """Generate goal-vs-measured plots.

    Produces:
    - all_joints.png: single figure with all joints, goal (dotted) and
      measured (solid) traces per joint.
    - joint_XX.png: one figure per joint with the same two traces.

    Requires goal_position and current_position signals in the data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    time = data["timestamp"] - data["timestamp"][0]

    has_takeover = "takeover_state" in data
    takeover = data["takeover_state"] if has_takeover else None

    goal = data["goal_position"]
    measured = data["current_position"]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ---- All-joints overview ----
    fig, ax = plt.subplots(figsize=(16, 6))
    for j in range(dof):
        c = colors[j % len(colors)]
        ax.plot(time, measured[:, j], color=c, alpha=0.7,
                label=f"J{j} measured")
        ax.plot(time, goal[:, j], color=c, alpha=0.5,
                linestyle="--", linewidth=1.2)
    ax.plot([], [], color="gray", linestyle="--", linewidth=1.2,
            label="goal (dashed)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Angle (rad)")
    ax.set_title("All Joints — Goal vs Measured")
    ax.legend(loc="upper right", ncol=4, fontsize="small")
    ax.grid(True, alpha=0.3)
    _shade_regions(ax, time, tracking_mask, takeover)
    fig.tight_layout()
    p = out / "all_joints.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved {p}")

    # ---- Individual joint plots ----
    for j in range(dof):
        fig, ax = plt.subplots(figsize=(14, 4))
        c = colors[j % len(colors)]
        ax.plot(time, measured[:, j], color=c, alpha=0.8,
                label="measured", linewidth=1.0)
        ax.plot(time, goal[:, j], color=c, alpha=0.5,
                linestyle="--", linewidth=1.2, label="goal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint Angle (rad)")
        ax.set_title(f"Joint {j} — Goal vs Measured")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        _shade_regions(ax, time, tracking_mask, takeover)
        fig.tight_layout()
        p = out / f"joint_{j:02d}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  Saved {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze per-joint tracking errors from GELLO debug logs",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Path to HDF5 log file. If not provided, uses latest session.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate goal-vs-measured plots (all-joints + per-joint).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save plots. Defaults to <log-file-stem>_plots/.",
    )
    args = parser.parse_args()

    # Resolve log file
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_path = find_latest_log()
    print(f"Loading: {log_path}")

    data = load_debug_log(str(log_path))

    # Metadata
    metadata = data["metadata"]
    dof = int(metadata["dof"])
    n_total = len(data["timestamp"])
    duration = data["timestamp"][-1] - data["timestamp"][0]

    print(f"\nSession: {log_path.name}")
    print(f"DOF: {dof}  |  Samples: {n_total}  |  Duration: {duration:.1f}s  |  Rate: {n_total / max(duration, 1e-6):.1f} Hz")

    # Filter to tracking-active samples
    is_tracking = data["is_tracking"].astype(bool)
    n_tracking = int(np.sum(is_tracking))

    if n_tracking == 0:
        print("\nNo tracking-active samples found in this session.")
        return

    print(f"Tracking samples: {n_tracking} / {n_total} ({100 * n_tracking / n_total:.1f}%)")

    # --- Absolute position error (always available) ---
    pos_err = data["position_error"][is_tracking]
    abs_stats = compute_stats(pos_err)
    print_stats_table("Absolute Position Error (rad) — tracking only", abs_stats, dof)

    # --- Signed position error (new signal, may be absent in old logs) ---
    if "signed_position_error" in data:
        signed_err = data["signed_position_error"][is_tracking]
        signed_stats = compute_stats(signed_err)
        print_stats_table("Signed Position Error (rad) — tracking only", signed_stats, dof)

    # --- Takeover breakdown ---
    has_takeover = "takeover_state" in data
    if has_takeover:
        takeover = data["takeover_state"].astype(bool)
        takeover_and_tracking = takeover & is_tracking
        no_takeover_and_tracking = ~takeover & is_tracking
        n_takeover = int(np.sum(takeover_and_tracking))
        n_no_takeover = int(np.sum(no_takeover_and_tracking))

        print(f"\nTakeover breakdown (tracking samples only):")
        print(f"  Takeover active:   {n_takeover}")
        print(f"  Takeover inactive: {n_no_takeover}")

        if n_takeover > 0:
            err_during = data["position_error"][takeover_and_tracking]
            stats_during = compute_stats(err_during)
            print_stats_table(
                "Abs Position Error DURING takeover", stats_during, dof
            )

        if n_no_takeover > 0:
            err_outside = data["position_error"][no_takeover_and_tracking]
            stats_outside = compute_stats(err_outside)
            print_stats_table(
                "Abs Position Error WITHOUT takeover", stats_outside, dof
            )

    # --- Plotting ---
    if args.plot or args.output_dir:
        if "goal_position" not in data or "current_position" not in data:
            print("\nPlotting requires goal_position and current_position signals.")
            print("Re-run data collection with the updated debug logger.")
        else:
            if args.output_dir:
                plot_dir = args.output_dir
            else:
                scripts_dir = Path(__file__).parent
                stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                plot_dir = str(scripts_dir / "tracking_plots" / stamp)
            print(f"\nGenerating plots in: {plot_dir}")
            plot_tracking_errors(data, is_tracking, dof, output_dir=plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

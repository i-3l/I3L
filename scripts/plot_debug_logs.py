#!/usr/bin/env python3
"""
Offline plotter for GELLO debug logs.

Loads HDF5 log files and generates publication-quality plots with the same
layout as the real-time monitor.

Usage:
    # Plot latest session
    python scripts/plot_debug_logs.py

    # Plot specific file
    python scripts/plot_debug_logs.py --log-file outputs/2025-12-13/.../session_*.h5

    # Plot time range
    python scripts/plot_debug_logs.py --start-time 10.0 --end-time 30.0

    # Save to PDF
    python scripts/plot_debug_logs.py --format pdf --output-dir plots/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from iiil.gello.debug.debug_logger import load_debug_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GELLO debug logs offline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot latest session
  %(prog)s

  # Plot specific file
  %(prog)s --log-file outputs/2025-12-13/14-39-17/logs/debug/session_*.h5

  # Plot time range and save as PDF
  %(prog)s --start-time 10.0 --end-time 30.0 --format pdf
        """
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to HDF5 log file. If not provided, uses latest session.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="Start time in seconds (default: from beginning)",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        help="End time in seconds (default: until end)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots (default: plots)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "both"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster output (default: 300)",
    )
    parser.add_argument(
        "--y-limit-currents",
        type=float,
        nargs=2,
        default=[-150, 150],
        metavar=("MIN", "MAX"),
        help="Y-axis limits for current plots in mA (default: -150 150)",
    )
    return parser.parse_args()


def find_latest_log() -> Path:
    """Find the most recent debug log file."""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        raise FileNotFoundError(
            "No outputs directory found. Run this script from the project root."
        )

    # Find all HDF5 files
    log_files = sorted(outputs_dir.glob("*/*/logs/debug/session_*.h5"))
    if not log_files:
        raise FileNotFoundError(
            f"No debug log files found in {outputs_dir}/*/*/logs/debug/\n"
            "Make sure debug logging is enabled in your config."
        )

    return log_files[-1]  # Most recent


def load_and_filter_data(
    filepath: Path,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Load HDF5 data and filter by time range."""
    print(f"Loading: {filepath}")
    data = load_debug_log(str(filepath))

    # Get metadata
    metadata = data["metadata"]
    dof = metadata["dof"]
    duration = data['timestamp'][-1] - data['timestamp'][0]

    print(f"  DOF: {dof}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Samples: {len(data['timestamp'])}")

    # Filter by time
    if start_time is not None or end_time is not None:
        t0 = data['timestamp'][0]
        rel_times = data['timestamp'] - t0

        mask = np.ones(len(rel_times), dtype=bool)
        if start_time is not None:
            mask &= rel_times >= start_time
        if end_time is not None:
            mask &= rel_times <= end_time

        # Apply mask to all signals
        filtered_data = {}
        for key, value in data.items():
            if key == "metadata":
                filtered_data[key] = value
            else:
                filtered_data[key] = value[mask]

        print(f"  Filtered to: {np.sum(mask)} samples ({start_time or 0:.1f}s - {end_time or duration:.1f}s)")
        return filtered_data

    return data


def plot_layout(
    data: Dict[str, np.ndarray],
    y_limits: Dict[str, Tuple[float, float]],
) -> plt.Figure:
    """Create per-motor grid layout.

    Layout:
    - Left column: Measured, Tracking, Gravity currents for each motor
    - Right column: Expected current and Error for each motor
    """
    dof = data["metadata"]["dof"]

    # Relative time in seconds
    times = data["timestamp"] - data["timestamp"][0]

    fig, axes = plt.subplots(dof, 2, figsize=(16, 3 * dof), sharex=True)

    # Handle single motor case (axes won't be 2D)
    if dof == 1:
        axes = axes.reshape(1, -1)

    for motor_idx in range(dof):
        # Left: Currents (measured, tracking, gravity)
        ax_curr = axes[motor_idx, 0]

        if 'measured_currents' in data:
            ax_curr.plot(times, data['measured_currents'][:, motor_idx],
                        '-', color='blue', label='Measured', linewidth=1.5)
        if 'tracking_currents' in data:
            ax_curr.plot(times, data['tracking_currents'][:, motor_idx],
                        '-', color='green', label='Tracking', linewidth=1.3, alpha=0.8)
        if 'gravity_currents' in data:
            ax_curr.plot(times, data['gravity_currents'][:, motor_idx],
                        '-', color='orange', label='Gravity', linewidth=1.3, alpha=0.8)

        ax_curr.set_ylabel(f'Motor {motor_idx}\nCurrent (mA)', fontsize=10)
        ax_curr.grid(True, alpha=0.3)
        ax_curr.legend(loc='upper right', fontsize=8)
        ax_curr.set_ylim(y_limits['currents'])

        # Right: Expected vs Error
        ax_err = axes[motor_idx, 1]

        if 'expected_currents' in data:
            ax_err.plot(times, data['expected_currents'][:, motor_idx],
                       '-', color='purple', label='Expected', linewidth=1.3)
        if 'current_diff' in data:
            ax_err.plot(times, data['current_diff'][:, motor_idx],
                       '-', color='red', label='Error', linewidth=1.3, alpha=0.8)
        if 'filtered_current_diff' in data:
            ax_err.plot(times, data['filtered_current_diff'][:, motor_idx],
                       '-', color='hotpink', label='Filtered Error', linewidth=1)

        ax_err.set_ylabel('Current (mA)', fontsize=10)
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(loc='upper right', fontsize=8)
        ax_err.set_ylim(y_limits['currents'])

    # Labels
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')

    # Title with metadata
    import datetime
    start_time_ts = data["metadata"].get("start_time", None)
    if start_time_ts:
        start_time_str = datetime.datetime.fromtimestamp(start_time_ts).strftime('%Y-%m-%d %H:%M:%S')
    else:
        start_time_str = "unknown"

    fig.suptitle(
        f'GELLO Debug Log - Per Motor View\nSession: {start_time_str}',
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()

    return fig


def main():
    args = parse_args()

    # Find log file
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        try:
            log_file = find_latest_log()
            print(f"Using latest log: {log_file}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    # Load data
    try:
        data = load_and_filter_data(log_file, args.start_time, args.end_time)
    except Exception as e:
        print(f"Error loading log file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Y-axis limits
    y_limits = {
        'currents': tuple(args.y_limit_currents),
    }

    # Create plot
    print(f"\nGenerating per-motor layout...")
    fig = plot_layout(data, y_limits)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    session_name = log_file.stem  # e.g., "session_2025-12-13_14-40-14"
    base_name = session_name

    if args.format in ["png", "both"]:
        output_path = output_dir / f"{base_name}.png"
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if args.format in ["pdf", "both"]:
        output_path = output_dir / f"{base_name}.pdf"
        fig.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")

    print("\nDone!")
    plt.close(fig)


if __name__ == "__main__":
    main()

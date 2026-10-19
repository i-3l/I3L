#!/usr/bin/env python3
"""
Example script for analyzing Gello debug logs.

This demonstrates how to load and analyze HDF5 debug logs created by the
debug logging system.

Usage:
    python examples/analyze_debug_logs.py logs/debug/session_2024-12-12_15-30-00.h5
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from iiil.gello.debug.debug_logger import load_debug_log


def analyze_tracking_performance(data):
    """Analyze tracking performance metrics."""
    print("\n" + "=" * 60)
    print("TRACKING PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Filter to tracking mode only
    is_tracking = data["is_tracking"]
    tracking_indices = np.where(is_tracking)[0]

    if len(tracking_indices) == 0:
        print("⚠️  No tracking data found in this session")
        return

    # Position error statistics
    pos_err = data["position_error"][tracking_indices]
    print(f"\n📍 Position Error (rad):")
    print(f"   Mean: {np.mean(pos_err, axis=0)}")
    print(f"   Max:  {np.max(pos_err, axis=0)}")
    print(f"   RMS:  {np.sqrt(np.mean(pos_err**2, axis=0))}")

    # Current difference statistics
    curr_diff = data["current_diff"][tracking_indices]
    print(f"\n⚡ Current Difference (mA):")
    print(f"   Mean: {np.mean(curr_diff, axis=0)}")
    print(f"   Std:  {np.std(curr_diff, axis=0)}")
    print(f"   RMS:  {np.sqrt(np.mean(curr_diff**2, axis=0))}")


def analyze_current_components(data):
    """Analyze current component breakdown."""
    print("\n" + "=" * 60)
    print("CURRENT COMPONENT ANALYSIS")
    print("=" * 60)

    # Average current magnitudes
    measured = np.abs(data["measured_currents"])
    tracking = np.abs(data["tracking_currents"])
    gravity = np.abs(data["gravity_currents"])
    friction = np.abs(data["friction_currents"])

    print(f"\n📊 Average Current Magnitudes (mA):")
    print(f"   Measured: {np.mean(measured, axis=0)}")
    print(f"   Tracking: {np.mean(tracking, axis=0)}")
    print(f"   Gravity:  {np.mean(gravity, axis=0)}")
    print(f"   Friction: {np.mean(friction, axis=0)}")

    # Current component ratios
    total_expected = tracking + gravity + friction
    tracking_ratio = tracking / (total_expected + 1e-6)
    gravity_ratio = gravity / (total_expected + 1e-6)

    print(f"\n📈 Component Ratios (avg %):")
    print(f"   Tracking: {np.mean(tracking_ratio, axis=0) * 100:.1f}%")
    print(f"   Gravity:  {np.mean(gravity_ratio, axis=0) * 100:.1f}%")


def plot_time_series(data, output_path=None):
    """Plot time-series data."""
    print(f"\n📈 Generating time-series plots...")

    # Convert to relative time
    time = data["timestamp"] - data["timestamp"][0]
    dof = data["metadata"]["dof"]

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Measured vs Expected Currents (Joint 0 only for clarity)
    ax = axes[0]
    ax.plot(time, data["measured_currents"][:, 0], "b-", label="Measured", alpha=0.7)
    ax.plot(
        time, data["expected_currents"][:, 0], "r--", label="Expected", alpha=0.7
    )
    ax.set_ylabel("Current (mA)")
    ax.set_title("Joint 0: Measured vs Expected Current")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Current Components (Joint 0)
    ax = axes[1]
    ax.plot(time, data["tracking_currents"][:, 0], "g-", label="Tracking", alpha=0.7)
    ax.plot(time, data["gravity_currents"][:, 0], "orange", label="Gravity", alpha=0.7)
    ax.plot(time, data["friction_currents"][:, 0], "purple", label="Friction", alpha=0.7)
    ax.set_ylabel("Current (mA)")
    ax.set_title("Joint 0: Current Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Position Error (all joints)
    ax = axes[2]
    for joint_idx in range(min(dof, 6)):  # Limit to 6 for readability
        ax.plot(
            time,
            data["position_error"][:, joint_idx],
            label=f"J{joint_idx}",
            alpha=0.7,
        )
    ax.set_ylabel("Position Error (rad)")
    ax.set_title("Position Tracking Error")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 4: Current Difference (all joints)
    ax = axes[3]
    for joint_idx in range(min(dof, 6)):
        ax.plot(
            time, data["current_diff"][:, joint_idx], label=f"J{joint_idx}", alpha=0.7
        )
    ax.set_ylabel("Current Diff (mA)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Current Difference (Measured - Expected)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    # Highlight tracking mode
    is_tracking = data["is_tracking"]
    for ax in axes:
        ylim = ax.get_ylim()
        ax.fill_between(
            time, ylim[0], ylim[1], where=is_tracking, alpha=0.1, color="green", label="Tracking"
        )
        ax.set_ylim(ylim)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"✓ Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_debug_logs.py <path_to_h5_file>")
        print("\nExample:")
        print("  python analyze_debug_logs.py logs/debug/session_2024-12-12_15-30-00.h5")
        return

    log_file = sys.argv[1]
    print(f"\n🔍 Loading debug log: {log_file}")

    # Load data
    try:
        data = load_debug_log(log_file)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Print metadata
    metadata = data["metadata"]
    duration = data["timestamp"][-1] - data["timestamp"][0]
    n_samples = len(data["timestamp"])

    print("\n" + "=" * 60)
    print("SESSION METADATA")
    print("=" * 60)
    print(f"DOF:              {metadata['dof']}")
    print(f"Control Freq:     {metadata['control_frequency']} Hz")
    print(f"Duration:         {duration:.1f} seconds")
    print(f"Samples:          {n_samples}")
    print(f"Actual Rate:      {n_samples / duration:.1f} Hz")
    print(f"Motor Types:      {metadata.get('motor_types', 'N/A')}")

    # Analyze performance
    analyze_tracking_performance(data)
    analyze_current_components(data)

    # Plot results
    output_path = Path(log_file).with_suffix(".png")
    plot_time_series(data, output_path)

    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

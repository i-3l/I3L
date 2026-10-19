#!/usr/bin/env python3
"""Standalone matplotlib plotter - launched as a separate subprocess.

This script runs in a completely fresh Python interpreter, avoiding
any conflicts with Isaac Sim's bundled libraries.
"""

import argparse
import pickle
import time
import numpy as np


def create_layout(dof, y_limits):
    """Create per-motor grid layout (dof rows × 2 columns).

    Layout:
    - Left column: Measured, Tracking, Gravity currents for each motor
    - Right column: Expected current and Error for each motor
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(dof, 2, figsize=(16, 3 * dof), sharex=True)

    # Handle single motor case (axes won't be 2D)
    if dof == 1:
        axes = axes.reshape(1, -1)

    lines = {}

    for motor_idx in range(dof):
        # Left column: Currents (measured, tracking, gravity)
        ax_currents = axes[motor_idx, 0]

        lines[f'motor_{motor_idx}_measured'] = ax_currents.plot([], [], '-',
            color='blue', label='Measured', linewidth=1.5)[0]
        lines[f'motor_{motor_idx}_tracking'] = ax_currents.plot([], [], '-',
            color='green', label='Tracking', linewidth=1.3, alpha=0.8)[0]
        lines[f'motor_{motor_idx}_gravity'] = ax_currents.plot([], [], '-',
            color='orange', label='Gravity', linewidth=1.3, alpha=0.8)[0]

        ax_currents.set_ylabel(f'Motor {motor_idx}\nCurrent (mA)', fontsize=10)
        ax_currents.grid(True, alpha=0.3)
        ax_currents.legend(loc='upper right', fontsize=8)
        ax_currents.set_ylim(y_limits.get('currents', [-150, 150]))

        # Right column: Expected vs Error
        ax_error = axes[motor_idx, 1]

        lines[f'motor_{motor_idx}_expected'] = ax_error.plot([], [], '-',
            color='purple', label='Expected', linewidth=1.3)[0]
        lines[f'motor_{motor_idx}_error'] = ax_error.plot([], [], '-',
            color='red', label='Error', linewidth=1.3, alpha=0.8)[0]
        lines[f'motor_{motor_idx}_filtered_error'] = ax_error.plot([], [], '-',
            color='hotpink', label='Filtered Error', linewidth=3.0, alpha=1.0)[0]

        ax_error.set_ylabel('Current (mA)', fontsize=10)
        ax_error.grid(True, alpha=0.3)
        ax_error.legend(loc='upper right', fontsize=8)
        ax_error.set_ylim(y_limits.get('currents', [-150, 150]))

    # X-axis labels
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')

    fig.suptitle('Gello Debug Monitor - Per Motor View', fontsize=14, fontweight='bold')
    fig.tight_layout()

    return fig, axes, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', required=True, help='Path to shared data file')
    parser.add_argument('--update-rate', type=float, default=10.0)
    parser.add_argument('--window-duration', type=float, default=10.0)
    parser.add_argument('--dof', type=int, required=True)
    parser.add_argument('--y-limits', type=str, default='{}', help='JSON string of y-limits')
    args = parser.parse_args()

    # Import matplotlib AFTER argparse (fresh environment)
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import json

    # Parse y-limits from JSON string
    try:
        y_limits = json.loads(args.y_limits) if args.y_limits else {}
    except json.JSONDecodeError:
        print(f"[StandalonePlotter] Warning: Failed to parse y-limits, using defaults")
        y_limits = {"currents": [-150, 150]}

    # Ensure we have default limits for currents
    if "currents" not in y_limits:
        y_limits["currents"] = [-150, 150]

    print(f"[StandalonePlotter] Starting (PID: {os.getpid()})")
    print(f"[StandalonePlotter] Backend: {matplotlib.get_backend()}")
    print(f"[StandalonePlotter] Data file: {args.data_file}")
    print(f"[StandalonePlotter] DOF: {args.dof}")
    print(f"[StandalonePlotter] Y-limits: {y_limits}")

    # Anchor the x-axis to the first timestamp so it does not reset each update
    base_time = None

    # Create per-motor layout
    fig, axes, lines = create_layout(args.dof, y_limits)

    plt.ion()
    plt.show(block=False)

    print(f"[StandalonePlotter] Window opened successfully!")
    print(f"[StandalonePlotter] Updating at {args.update_rate} Hz")

    # Update loop
    interval = 1.0 / args.update_rate

    try:
        while True:
            loop_start = time.time()

            # Load data from file
            try:
                with open(args.data_file, 'rb') as f:
                    data = pickle.load(f)

                if 'timestamp' in data and len(data['timestamp']) > 0:
                    times = np.array(data['timestamp'], dtype=float)
                    if base_time is None:
                        base_time = times[0]
                    times = times - base_time  # Keep time continuous across updates
                    n_samples = len(times)

                    # Update per-motor layout
                    for motor_idx in range(args.dof):
                        # Update currents subplot (left column)
                        if 'measured_currents' in data and len(data['measured_currents']) == n_samples:
                            y_data = data['measured_currents'][:, motor_idx]
                            lines[f'motor_{motor_idx}_measured'].set_data(times, y_data)

                        if 'tracking_currents' in data and len(data['tracking_currents']) == n_samples:
                            y_data = data['tracking_currents'][:, motor_idx]
                            lines[f'motor_{motor_idx}_tracking'].set_data(times, y_data)

                        if 'gravity_currents' in data and len(data['gravity_currents']) == n_samples:
                            y_data = data['gravity_currents'][:, motor_idx]
                            lines[f'motor_{motor_idx}_gravity'].set_data(times, y_data)

                        # Update error subplot (right column)
                        if 'expected_currents' in data and len(data['expected_currents']) == n_samples:
                            y_data = data['expected_currents'][:, motor_idx]
                            lines[f'motor_{motor_idx}_expected'].set_data(times, y_data)

                        if 'current_diff' in data and len(data['current_diff']) == n_samples:
                            y_data = data['current_diff'][:, motor_idx]
                            lines[f'motor_{motor_idx}_error'].set_data(times, y_data)

                        if 'filtered_current_diff' in data and len(data['filtered_current_diff']) == n_samples:
                            y_data = data['filtered_current_diff'][:, motor_idx]
                            lines[f'motor_{motor_idx}_filtered_error'].set_data(times, y_data)

                        # Update x-axis limits for this motor's subplots
                        if n_samples > 0:
                            x_min = max(0, times[-1] - args.window_duration)
                            axes[motor_idx, 0].set_xlim(x_min, times[-1])
                            axes[motor_idx, 1].set_xlim(x_min, times[-1])

                    # Redraw
                    fig.canvas.draw()
                    fig.canvas.flush_events()

            except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                pass  # Data file not ready yet or being written

            # Check if window was closed
            if not plt.fignum_exists(fig.number):
                print("[StandalonePlotter] Window closed by user")
                break

            # Sleep to maintain update rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("[StandalonePlotter] Interrupted")
    finally:
        plt.close('all')
        print("[StandalonePlotter] Shutdown complete")


if __name__ == '__main__':
    import os
    main()

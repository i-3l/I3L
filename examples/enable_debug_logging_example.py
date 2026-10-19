#!/usr/bin/env python3
"""
Example: Programmatically enable debug logging on a Gello controller.

This shows two ways to enable debug logging:
1. Via YAML configuration (recommended)
2. Programmatically with Python code
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_1_via_yaml():
    """Enable debug logging via YAML config (recommended approach)."""
    print("="*60)
    print("Example 1: Enable via YAML Configuration")
    print("="*60)

    print("""
Simply edit your gello config file (e.g., gello_a1.yaml):

    debug_logging:
      enable: true           # Enable the system
      log_to_file: true      # Save to HDF5
      log_directory: "logs/debug"

      plotting:
        enable: true         # Show real-time plots
        update_rate: 10      # Hz
        window_duration: 10.0  # seconds

Then use your controller normally:

    from iiil.gello.controller_factory import A1GelloControllerFactory

    # Load controller (debug logging auto-enabled from config)
    controller = A1GelloControllerFactory.get_gello_controller_instance(
        gello_config_name="gello_a1"
    )

    # Start control
    controller.start_control()

    # ... use controller normally ...

    # Close (automatically flushes debug data)
    controller.close()

Debug logs will be saved to: logs/debug/session_YYYY-MM-DD_HH-MM-SS.h5
""")


def example_2_programmatic():
    """Enable debug logging programmatically."""
    print("="*60)
    print("Example 2: Enable Programmatically")
    print("="*60)

    print("""
If you need to configure debug logging in code:

    from iiil.gello.controller_factory import A1GelloControllerFactory
    from iiil.gello.debug import DebugLoggingConfig, PlottingConfig

    # Create controller (without debug logging initially)
    controller = A1GelloControllerFactory.get_gello_controller_instance(
        gello_config_name="gello_a1"
    )

    # Configure debug logging
    debug_config = DebugLoggingConfig(
        enable=True,
        log_to_file=True,
        log_directory="my_experiment_logs",
        buffer_size=10000,
        flush_interval=1000,
        signals=[
            "measured_currents",
            "expected_currents",
            "current_diff",
            "position_error",
        ],
        plotting=PlottingConfig(
            enable=True,
            update_rate=10,
            window_duration=10.0,
            plot_groups={
                "currents": ["measured_currents", "expected_currents"],
                "errors": ["position_error", "current_diff"],
            }
        )
    )

    # Enable debug logging
    controller.enable_debug_logging(debug_config)

    # Start control
    controller.start_control()

    # ... use controller ...

    # Close (flushes data)
    controller.close()
""")


def example_3_minimal():
    """Minimal example - logging only, no plotting."""
    print("="*60)
    print("Example 3: Minimal (Logging Only, No Plots)")
    print("="*60)

    print("""
For maximum performance, disable plotting and just log to file:

    from iiil.gello.debug import DebugLoggingConfig, PlottingConfig

    debug_config = DebugLoggingConfig(
        enable=True,
        log_to_file=True,
        plotting=PlottingConfig(enable=False)  # No real-time plots
    )

    controller.enable_debug_logging(debug_config)

This adds < 0.1% overhead to the control loop.
""")


def example_4_analysis():
    """Example of analyzing logged data."""
    print("="*60)
    print("Example 4: Analyze Logged Data")
    print("="*60)

    print("""
After collecting data, analyze it offline:

    from iiil.gello.debug.debug_logger import load_debug_log
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load HDF5 file
    data = load_debug_log("logs/debug/session_2024-12-12_15-30-00.h5")

    # Convert to pandas DataFrame
    df = pd.DataFrame({
        'time': data['timestamp'] - data['timestamp'][0],
        'current_diff_j0': data['current_diff'][:, 0],
        'current_diff_j1': data['current_diff'][:, 1],
        'pos_error_j0': data['position_error'][:, 0],
        'tracking': data['is_tracking'],
    })

    # Filter to tracking mode only
    df_tracking = df[df['tracking']]

    # Calculate RMS error
    rms_error = (df_tracking['pos_error_j0']**2).mean()**0.5
    print(f"RMS tracking error (joint 0): {rms_error:.4f} rad")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(df['time'], df['current_diff_j0'])
    axes[0].set_ylabel('Current Diff (mA)')
    axes[0].set_title('Joint 0: Current Difference')

    axes[1].plot(df['time'], df['pos_error_j0'])
    axes[1].set_ylabel('Position Error (rad)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Joint 0: Position Tracking Error')

    plt.tight_layout()
    plt.savefig('tracking_analysis.png')

Or use the provided analysis script:

    python examples/analyze_debug_logs.py logs/debug/session_*.h5

This will generate comprehensive plots and statistics.
""")


def main():
    """Show all examples."""
    print("\n" + "="*60)
    print("GELLO DEBUG LOGGING - USAGE EXAMPLES")
    print("="*60 + "\n")

    example_1_via_yaml()
    print()

    example_2_programmatic()
    print()

    example_3_minimal()
    print()

    example_4_analysis()

    print("\n" + "="*60)
    print("For more information, see:")
    print("  - iiil/gello/debug/README.md")
    print("  - DEBUG_SYSTEM_SUMMARY.md")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

# Gello Debug Logging and Visualization

This module provides scalable real-time logging and visualization for Gello teleoperation debug data.

Debug messages and structured logging are disabled in the supplied hardware
configs. Enable `debug_logging.enable` for recording diagnostics or
`debug_messages.enable` for console diagnostics when needed.

## Features

- **100 Hz data logging** - Captures all control loop diagnostics at full frequency
- **HDF5 file output** - Efficient compressed storage with one file per session
- **Real-time matplotlib plots** - Optional live visualization at 10 Hz
- **Minimal overhead** - < 0.4% impact on 100 Hz control loop
- **Graceful degradation** - Automatically disables if dependencies unavailable

## Quick Start

### Enable Logging (No Visualization)

Edit `iiil/configs/gello/gello_a1.yaml`:

```yaml
debug_logging:
  enable: true
  log_to_file: true
  plotting:
    enable: false
```

Run your teleoperation script normally. Debug data will be logged to `logs/debug/session_YYYY-MM-DD_HH-MM-SS.h5`.

### Enable Real-Time Visualization

```yaml
debug_logging:
  enable: true
  log_to_file: true
  plotting:
    enable: true
    update_rate: 10  # Hz
```

This will open a matplotlib window showing live plots of:
- Measured vs. expected currents
- Current components (tracking, gravity, friction)
- Position errors
- Current differences

## Logged Signals

The system captures these signals at 100 Hz:

- `measured_currents` - Actual motor currents (mA)
- `tracking_currents` - Expected current from tracking torque
- `gravity_currents` - Expected current from gravity compensation
- `friction_currents` - Expected current from friction compensation
- `expected_currents` - Sum of all expected currents
- `current_diff` - Measured - expected (disturbance indicator)
- `position_error` - Absolute error from goal position
- `velocities` - Joint velocities
- `is_tracking` - Boolean flag for ENV_TO_GELLO vs GELLO_TO_ENV mode

## Offline Analysis

Load and analyze logged data:

```python
from iiil.gello.debug.debug_logger import load_debug_log
import pandas as pd
import matplotlib.pyplot as plt

# Load HDF5 file
data = load_debug_log("logs/debug/session_2024-12-12_15-30-00.h5")

# Print metadata
print(f"DOF: {data['metadata']['dof']}")
print(f"Duration: {data['timestamp'][-1] - data['timestamp'][0]:.1f} seconds")
print(f"Samples: {len(data['timestamp'])}")

# Convert to pandas DataFrame
df = pd.DataFrame({
    'time': data['timestamp'] - data['timestamp'][0],
    'current_diff_j0': data['current_diff'][:, 0],
    'current_diff_j1': data['current_diff'][:, 1],
    'pos_error_j0': data['position_error'][:, 0],
})

# Plot tracking performance
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Current difference (disturbance indicator)
axes[0].plot(df['time'], df['current_diff_j0'], label='Joint 0')
axes[0].plot(df['time'], df['current_diff_j1'], label='Joint 1')
axes[0].set_ylabel('Current Difference (mA)')
axes[0].legend()
axes[0].grid(True)

# Position error
axes[1].plot(df['time'], df['pos_error_j0'])
axes[1].set_ylabel('Position Error (rad)')
axes[1].set_xlabel('Time (s)')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

## Configuration Options

### DebugLoggingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable debug logging system |
| `log_to_file` | bool | `true` | Write data to HDF5 file |
| `log_directory` | str | `"logs/debug"` | Directory for log files |
| `buffer_size` | int | `10000` | Circular buffer size (samples) |
| `flush_interval` | int | `1000` | Flush to disk every N samples |
| `signals` | list[str] | (all) | Which signals to capture |
| `plotting` | PlottingConfig | - | Real-time plotting config |

### PlottingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable real-time visualization |
| `backend` | str | `"matplotlib"` | Plotting backend |
| `update_rate` | float | `10.0` | Plot update frequency (Hz) |
| `window_duration` | float | `10.0` | Time window to display (seconds) |
| `plot_groups` | dict | (see yaml) | Subplot organization |

## Performance

The debug logger is designed for minimal impact on the control loop:

| Operation | Time | Impact |
|-----------|------|--------|
| Buffer write (per sample) | 1-5 μs | 0.05% of 10ms loop |
| Data copy to plotter | 2-10 μs | 0.1% |
| **Total overhead** | **~15-35 μs** | **< 0.4%** |
| HDF5 flush (background) | 10-50ms | 0% (separate thread) |
| Matplotlib plotting | 50-100ms | 0% (separate thread) |

## Architecture

```
Control Loop (100 Hz)
    ↓
DebugDataLogger.log_sample()
    ↓
    ├─→ CircularBuffer (in-memory)    [< 5 μs]
    │
    └─→ Background Flush Thread
        └─→ HDF5 File                 [separate thread]

    └─→ Matplotlib Thread
        └─→ Real-time Plots           [separate thread, 10 Hz]
```

## HDF5 File Structure

```
session_2024-12-12_15-30-00.h5
├── /metadata
│   ├── start_time (attr)
│   ├── dof (attr)
│   ├── control_frequency (attr)
│   └── motor_types (dataset)
└── /data
    ├── timestamp [N,]
    ├── measured_currents [N, dof]
    ├── tracking_currents [N, dof]
    ├── gravity_currents [N, dof]
    ├── friction_currents [N, dof]
    ├── expected_currents [N, dof]
    ├── current_diff [N, dof]
    ├── position_error [N, dof]
    ├── velocities [N, dof]
    └── is_tracking [N,] (bool)
```

## Dependencies

**Required for logging:**
- `h5py` - HDF5 file I/O

**Required for plotting:**
- `matplotlib` - Real-time visualization

Both will gracefully degrade if unavailable (similar to the existing LivePlotter).

## Programmatic Usage

You can also enable debug logging programmatically:

```python
from iiil.gello.debug.debug_config import DebugLoggingConfig, PlottingConfig

# Create config
debug_config = DebugLoggingConfig(
    enable=True,
    log_to_file=True,
    log_directory="my_custom_logs",
    plotting=PlottingConfig(
        enable=True,
        update_rate=10,
    )
)

# Enable on controller
controller.enable_debug_logging(debug_config)

# ... run control loop ...

# Cleanup (automatically called on controller.close())
controller._debug_logger.close()
```

## Troubleshooting

### No plots appearing

1. Check that `plotting.enable = true` in config
2. Verify matplotlib is installed: `pip install matplotlib`
3. Check for errors in console output

### HDF5 file not created

1. Check that `enable = true` in config
2. Verify h5py is installed: `pip install h5py`
3. Ensure log directory exists and is writable

### Performance issues

1. Reduce plot update rate: `plotting.update_rate = 5`
2. Disable plotting: `plotting.enable = false`
3. Reduce buffer size: `buffer_size = 5000`

## Examples

Debug logging is set up by the controller factory in
[iiil/gello/factory/controller_builder.py](../factory/controller_builder.py). See also
`examples/enable_debug_logging_example.py`.

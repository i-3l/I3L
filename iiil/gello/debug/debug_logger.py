"""Main debug data logger with HDF5 file output and optional plotting."""

import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from iiil.gello.configs.motor_type import MotorType
from iiil.gello.debug.debug_buffer import CircularBuffer
from iiil.gello.debug.debug_config import DebugLoggingConfig


class DebugDataLogger:
    """Coordinates debug data logging and real-time plotting.

    Manages:
    - Circular buffer for high-frequency data capture
    - Background HDF5 file writing
    - Optional real-time matplotlib visualization
    """

    def __init__(
        self,
        config: DebugLoggingConfig,
        dof: int,
        motor_types: List[MotorType],
        logger=None,
    ):
        """Initialize debug logger.

        Args:
            config: Debug logging configuration
            dof: Degrees of freedom (number of joints)
            motor_types: List of motor type objects for metadata
            logger: Optional logger for warnings/errors
        """
        self.config = config
        self.dof = dof
        self.motor_types = motor_types
        self.logger = logger

        # Separate circular buffers:
        # - plot_buffer feeds the real-time plotter and is never cleared
        # - log_buffer is flushed to disk and cleared to avoid duplicate writes
        self.plot_buffer = CircularBuffer(config.buffer_size, dof, config.signals)
        self.log_buffer = CircularBuffer(config.buffer_size, dof, config.signals)
        # Backwards-compatible alias (plotter reads from self.buffer)
        self.buffer = self.plot_buffer

        # Sample counter
        self.sample_count = 0
        self.start_time = time.time()

        # HDF5 file management
        self.h5_file = None
        self.h5_path = None
        self.flush_thread = None
        self.flush_lock = threading.Lock()

        if config.log_to_file:
            self._init_hdf5_file()

        # Optional plotting
        self.plotter = None
        if config.plotting.enable:
            self._init_plotting()

    def _init_hdf5_file(self) -> None:
        """Initialize HDF5 file for logging."""
        try:
            import h5py
        except ImportError as e:
            if self.logger:
                self.logger.warning(f"HDF5 logging disabled (h5py import failed: {e})")
            self.config.log_to_file = False
            return

        # Create log directory
        log_dir = Path(self.config.log_directory)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.h5_path = log_dir / f"session_{timestamp}.h5"

        # Create HDF5 file
        self.h5_file = h5py.File(self.h5_path, "w")

        # Create metadata group
        meta = self.h5_file.create_group("metadata")
        meta.attrs["start_time"] = self.start_time
        meta.attrs["dof"] = self.dof
        meta.attrs["control_frequency"] = 100.0  # Hz
        meta.attrs["buffer_size"] = self.config.buffer_size
        meta.attrs["flush_interval"] = self.config.flush_interval

        # Store motor types
        motor_names = [mt.value.motor_name for mt in self.motor_types]
        meta.create_dataset(
            "motor_types", data=np.array(motor_names, dtype="S")
        )

        # Create data group with resizable datasets
        data = self.h5_file.create_group("data")
        data.create_dataset(
            "timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=True,
            compression="gzip",
        )
        data.create_dataset(
            "is_tracking",
            shape=(0,),
            maxshape=(None,),
            dtype=np.bool_,
            chunks=True,
            compression="gzip",
        )
        data.create_dataset(
            "takeover_state",
            shape=(0,),
            maxshape=(None,),
            dtype=np.bool_,
            chunks=True,
            compression="gzip",
        )

        for signal in self.config.signals:
            data.create_dataset(
                signal,
                shape=(0, self.dof),
                maxshape=(None, self.dof),
                dtype=np.float32,
                chunks=True,
                compression="gzip",
            )

        if self.logger:
            self.logger.info(f"Debug logging to: {self.h5_path}")

    def _init_plotting(self) -> None:
        """Initialize real-time plotting."""
        try:
            from iiil.gello.debug.debug_plotter import DebugPlotter

            self.plotter = DebugPlotter(
                self.config.plotting, self.dof, self.buffer, logger=self.logger
            )
            self.plotter.start()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Real-time plotting disabled: {e}")
            self.config.plotting.enable = False

    def log_sample(self, **data) -> None:
        """Log a single data sample (called from 100 Hz control loop).

        This method is optimized for minimal overhead (< 5 μs).

        Args:
            **data: Keyword arguments for each signal value
                timestamp: float
                measured_currents: np.ndarray[dof]
                tracking_currents: np.ndarray[dof]
                gravity_currents: np.ndarray[dof]
                friction_currents: np.ndarray[dof]
                expected_currents: np.ndarray[dof]
                current_diff: np.ndarray[dof]
                position_error: np.ndarray[dof]
                velocities: np.ndarray[dof]
                goal_position: np.ndarray[dof]
                current_position: np.ndarray[dof]
                signed_position_error: np.ndarray[dof]
                is_tracking: bool
                takeover_state: bool
        """
        # Write to circular buffer (very fast, < 5 μs)
        self.plot_buffer.append(**data)
        self.log_buffer.append(**data)
        self.sample_count += 1

        # Trigger background flush periodically
        if (
            self.config.log_to_file
            and self.sample_count % self.config.flush_interval == 0
        ):
            self._schedule_flush()

    def _schedule_flush(self) -> None:
        """Schedule a background flush to HDF5 file (non-blocking)."""
        if self.flush_thread is not None and self.flush_thread.is_alive():
            # Previous flush still running, skip this one
            return

        self.flush_thread = threading.Thread(target=self._flush_to_disk, daemon=True)
        self.flush_thread.start()

    def _flush_to_disk(self) -> None:
        """Flush circular buffer to HDF5 file (runs in background thread)."""
        if self.h5_file is None:
            return

        with self.flush_lock:
            try:
                # Get all buffered data
                data = self.log_buffer.get_all()

                if len(data["timestamp"]) == 0:
                    return  # No data to flush

                # Append to HDF5 datasets
                h5_data = self.h5_file["data"]
                current_size = h5_data["timestamp"].shape[0]
                new_size = current_size + len(data["timestamp"])

                # Resize datasets
                h5_data["timestamp"].resize((new_size,))
                h5_data["is_tracking"].resize((new_size,))
                h5_data["takeover_state"].resize((new_size,))
                for signal in self.config.signals:
                    if signal in data:
                        h5_data[signal].resize((new_size, self.dof))

                # Write data
                h5_data["timestamp"][current_size:new_size] = data["timestamp"]
                h5_data["is_tracking"][current_size:new_size] = data["is_tracking"]
                h5_data["takeover_state"][current_size:new_size] = data["takeover_state"]
                for signal in self.config.signals:
                    if signal in data:
                        h5_data[signal][current_size:new_size] = data[signal]

                # Flush to disk
                self.h5_file.flush()

                # Clear logging buffer after successful flush (plot buffer remains intact)
                self.log_buffer.clear()

                if self.logger:
                    self.logger.debug(
                        f"Flushed {len(data['timestamp'])} samples to {self.h5_path}"
                    )

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error flushing to HDF5: {e}")

    def close(self) -> None:
        """Close logger and flush any remaining data."""
        # Stop plotter
        if self.plotter is not None:
            self.plotter.stop()

        # Final flush
        if self.config.log_to_file and self.h5_file is not None:
            self._flush_to_disk()

            # Wait for flush thread to complete
            if self.flush_thread is not None and self.flush_thread.is_alive():
                self.flush_thread.join(timeout=5.0)

            # Close HDF5 file
            self.h5_file.close()
            if self.logger:
                self.logger.info(f"Debug log saved to: {self.h5_path}")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass


def load_debug_log(filepath: str) -> Dict[str, np.ndarray]:
    """Load debug log from HDF5 file.

    Args:
        filepath: Path to HDF5 log file

    Returns:
        Dictionary mapping signal names to numpy arrays
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required to load debug logs")

    with h5py.File(filepath, "r") as f:
        data = {}
        h5_data = f["data"]

        # Load all datasets
        for key in h5_data.keys():
            data[key] = h5_data[key][:]

        # Load metadata as attributes
        data["metadata"] = dict(f["metadata"].attrs)

        # Decode motor types
        if "motor_types" in f["metadata"]:
            data["metadata"]["motor_types"] = [
                mt.decode("utf-8") for mt in f["metadata"]["motor_types"][:]
            ]

    return data

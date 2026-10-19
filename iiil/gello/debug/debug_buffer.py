"""Circular buffer for high-frequency debug data storage."""

from typing import Dict, List, Optional
import numpy as np


class CircularBuffer:
    """Lock-free circular buffer for debug data.

    Optimized for high-frequency writes (100 Hz) with minimal overhead.
    Pre-allocates numpy arrays to avoid dynamic memory allocation.
    """

    def __init__(self, buffer_size: int, dof: int, signals: List[str]):
        """Initialize circular buffer.

        Args:
            buffer_size: Maximum number of samples to store
            dof: Degrees of freedom (number of joints)
            signals: List of signal names to track
        """
        self.buffer_size = buffer_size
        self.dof = dof
        self.signals = signals
        self.index = 0
        self.full = False

        # Pre-allocate numpy arrays for each signal
        self.buffers: Dict[str, np.ndarray] = {}

        # Scalar signals
        self.buffers["timestamp"] = np.zeros(buffer_size, dtype=np.float64)
        self.buffers["is_tracking"] = np.zeros(buffer_size, dtype=np.bool_)
        self.buffers["takeover_state"] = np.zeros(buffer_size, dtype=np.bool_)

        # Vector signals (per-joint)
        for signal in signals:
            if signal in [
                "measured_currents",
                "tracking_currents",
                "gravity_currents",
                "friction_currents",
                "expected_currents",
                "current_diff",
                "filtered_current_diff",
                "position_error",
                "velocities",
                "goal_velocity",  # Feedforward goal velocity (from differentiation)
                "goal_position",
                "current_position",
                "signed_position_error",
            ]:
                self.buffers[signal] = np.zeros((buffer_size, dof), dtype=np.float32)

    def append(self, **data) -> None:
        """Append a data sample to the buffer.

        This operation is designed to be very fast (< 5 μs) for use in
        the 100 Hz control loop.

        Args:
            **data: Keyword arguments for each signal value
        """
        idx = self.index

        # Write to pre-allocated arrays (single index assignment)
        for key, value in data.items():
            if key in self.buffers:
                self.buffers[key][idx] = value

        # Update circular index
        self.index = (self.index + 1) % self.buffer_size
        if self.index == 0:
            self.full = True

    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all buffered data in chronological order.

        Returns:
            Dictionary mapping signal names to numpy arrays
        """
        if not self.full:
            # Buffer not full yet, return data up to current index
            return {
                key: arr[: self.index].copy() for key, arr in self.buffers.items()
            }
        else:
            # Buffer is full, need to roll to maintain chronological order
            # Data from index to end, then from 0 to index
            result = {}
            for key, arr in self.buffers.items():
                if arr.ndim == 1:
                    result[key] = np.concatenate([arr[self.index :], arr[: self.index]])
                else:  # ndim == 2
                    result[key] = np.vstack([arr[self.index :], arr[: self.index]])
            return result

    def get_recent(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Get the most recent n samples.

        Args:
            n_samples: Number of recent samples to retrieve

        Returns:
            Dictionary mapping signal names to numpy arrays
        """
        n_samples = min(n_samples, self.buffer_size)

        if not self.full:
            # Buffer not full, just take last n_samples
            start_idx = max(0, self.index - n_samples)
            return {
                key: arr[start_idx : self.index].copy()
                for key, arr in self.buffers.items()
            }
        else:
            # Buffer is full, need to wrap around
            if n_samples >= self.buffer_size:
                return self.get_all()

            start_idx = (self.index - n_samples) % self.buffer_size
            result = {}
            for key, arr in self.buffers.items():
                if start_idx < self.index:
                    # No wrap-around
                    result[key] = arr[start_idx : self.index].copy()
                else:
                    # Wrap-around case
                    if arr.ndim == 1:
                        result[key] = np.concatenate(
                            [arr[start_idx:], arr[: self.index]]
                        )
                    else:  # ndim == 2
                        result[key] = np.vstack([arr[start_idx:], arr[: self.index]])
            return result

    def get_count(self) -> int:
        """Get the number of samples currently in the buffer."""
        return self.buffer_size if self.full else self.index

    def clear(self) -> None:
        """Clear the buffer (reset to empty state)."""
        self.index = 0
        self.full = False
        # No need to zero arrays, just reset pointers

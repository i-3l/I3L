"""
Takeover detection for Gello bilateral control.

Detects when a human takes over control during ENV_TO_GELLO mode
and when they release control for auto-return.
"""

import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union
from iiil.gello.configs.motor_type import MotorType

logger = logging.getLogger(__name__)


@dataclass
class TakeoverConfig:
    """Simple takeover detection configuration."""
    # Global enable
    enable: bool = True

    # Current filtering (low-pass filter to reduce noise)
    current_filter_ema_enable: bool = True  # EMA filter enable/disable
    current_filter_alpha: float = 0.3  # EMA smoothing factor (0-1)
    current_spike_median_enable: bool = False  # Apply median filter to remove isolated spikes
    current_spike_median_window: int = 5  # Median filter window size (samples)
    current_spike_clamp_enable: bool = False   # Clamp per-sample current jumps
    current_spike_clamp_max: float = 50.0      # mA - max allowed current change per sample (10ms at 100Hz)
    # Butterworth low-pass filter (better noise rejection than EMA)
    current_butterworth_enable: bool = False  # Enable 2nd-order Butterworth filter
    current_butterworth_cutoff: float = 5.0   # Cutoff frequency in Hz (at 100Hz sampling)
    current_butterworth_order: int = 2        # Filter order (2 recommended)

    # Position filtering (NEW)
    position_filter_ema_enable: bool = False  # EMA filter enable/disable
    position_filter_alpha: float = 0.3  # EMA smoothing factor (0-1)
    position_spike_median_enable: bool = False  # Apply median filter to remove isolated spikes
    position_spike_median_window: int = 5  # Median filter window size (samples)
    position_spike_clamp_enable: bool = False  # Clamp per-sample position jumps
    position_spike_clamp_max: float = 0.1  # rad - max allowed position change per sample (10ms)

    # Velocity filtering (NEW)
    velocity_filter_ema_enable: bool = False  # EMA filter enable/disable
    velocity_filter_alpha: float = 0.3  # EMA smoothing factor (0-1)
    velocity_spike_median_enable: bool = False  # Apply median filter to remove isolated spikes
    velocity_spike_median_window: int = 5  # Median filter window size (samples)
    velocity_spike_clamp_enable: bool = False  # Clamp per-sample velocity jumps
    velocity_spike_clamp_max: float = 1.0  # rad/s - max allowed velocity change per sample (10ms)

    # Detection method enables
    use_current: bool = True
    use_position: bool = True
    use_velocity: bool = True
    use_momentum: bool = False  # Momentum observer-based detection
    use_touch: bool = False     # Capacitive touch sensor detection

    # Touch sensor release cooldown (seconds after touch release before returning to policy)
    touch_release_cooldown: float = 0.5

    # Thresholds (can be scalar or per-joint array)
    current_threshold: Union[float, np.ndarray] = 150.0  # mA excess current (scalar or per-joint)
    position_threshold: Union[float, np.ndarray] = 0.5   # rad error (scalar or per-joint)
    velocity_threshold: float = 0.3   # rad/s
    momentum_threshold: Union[float, np.ndarray] = 0.3   # Nm per joint (scalar or per-joint)

    # Temporal filtering (frames at 100Hz)
    window_size: int = 10  # 100ms

    # Auto-return
    enable_auto_return: bool = True
    auto_return_timeout: float = 3.0  # seconds
    release_velocity: float = 0.05    # rad/s
    release_current_derivative: float = 10.0  # mA/s - current must be stable (low rate of change)
    takeover_grace_period: float = 0.2  # seconds to suppress detection after resuming tracking


class TakeoverDetector:
    """Detects when human takes over or releases control."""

    def __init__(self, config: TakeoverConfig, dof: int, motor_types: List[MotorType]):
        self.config = config
        self.dof = dof
        self.motor_types = motor_types
        self.current_threshold = self._normalize_threshold(config.current_threshold)
        self.position_threshold = self._normalize_position_threshold(config.position_threshold)
        self.momentum_threshold = self._normalize_momentum_threshold(config.momentum_threshold)
        self.config.position_threshold = self.position_threshold
        self._suppress_until: float = 0.0

        # Runtime enable/disable (separate from config.enable)
        # This allows experiment lifecycle to control detection without modifying config
        self._runtime_enabled: bool = False

        # Current filtering state
        self.filtered_current: Optional[np.ndarray] = None  # Filtered current measurements
        self.filtered_current_diff: Optional[np.ndarray] = None  # Last computed filtered difference
        self.current_median_buffer: Optional[np.ndarray] = None  # Rolling buffer for median filter
        self.current_median_idx: int = 0  # Circular buffer index for median filter
        self.current_prev_value: Optional[np.ndarray] = None  # Previous current for spike clamping

        # Butterworth filter state (for current)
        self._butter_b: Optional[np.ndarray] = None  # Numerator coefficients
        self._butter_a: Optional[np.ndarray] = None  # Denominator coefficients
        self._butter_x_hist: Optional[np.ndarray] = None  # Input history [order, dof]
        self._butter_y_hist: Optional[np.ndarray] = None  # Output history [order, dof]
        if config.current_butterworth_enable:
            self._init_butterworth_filter(
                config.current_butterworth_cutoff,
                config.current_butterworth_order,
                fs=100.0  # 100 Hz sampling rate
            )

        # Position filtering state
        self.filtered_position: Optional[np.ndarray] = None  # Filtered position measurements
        self.position_median_buffer: Optional[np.ndarray] = None  # Rolling buffer for median filter
        self.position_median_idx: int = 0  # Circular buffer index for median filter
        self.position_prev_value: Optional[np.ndarray] = None  # Previous position for spike clamping

        # Velocity filtering state
        self.filtered_velocity: Optional[np.ndarray] = None  # Filtered velocity measurements
        self.velocity_median_buffer: Optional[np.ndarray] = None  # Rolling buffer for median filter
        self.velocity_median_idx: int = 0  # Circular buffer index for median filter
        self.velocity_prev_value: Optional[np.ndarray] = None  # Previous velocity for spike clamping

        # Momentum filtering state (reuses current filter settings)
        self.filtered_momentum: Optional[np.ndarray] = None
        self.momentum_median_buffer: Optional[np.ndarray] = None
        self.momentum_median_idx: int = 0
        self.momentum_prev_value: Optional[np.ndarray] = None

        # Buffers for temporal filtering
        self.buffers = {
            'current': np.zeros((config.window_size, dof)),
            'position': np.zeros((config.window_size, dof)),
            'velocity': np.zeros((config.window_size, dof)),
            'momentum': np.zeros((config.window_size, dof)),
        }
        self.buffer_idx = 0

        # Auto-return state
        self.release_start_time: Optional[float] = None

        # Release detection: track current history for derivative calculation
        self.release_current_history = np.zeros((5, dof))  # Last 5 samples (50ms at 100Hz)
        self.release_history_idx = 0
        self.prev_current: Optional[np.ndarray] = None

        # Tracking for cleaner printouts (only print on state changes)
        self._last_takeover_state: bool = False
        self._last_release_state: bool = False
        self._diag_counter: int = 0
        self._event_callback = print

        # Cached release stats for dashboard display
        self._release_vel_max: float = 0.0
        self._release_curr_std: float = 0.0

        # Touch sensor state (injected after construction via set_touch_sensor)
        self._touch_sensor = None  # Optional TouchSensorReader
        self._touch_released_time: Optional[float] = None  # When touch was last released

    def set_event_callback(self, fn):
        """Set callback for dashboard event logging."""
        self._event_callback = fn

    def set_touch_sensor(self, sensor) -> None:
        """Inject the touch sensor reader (created by BilateralController)."""
        self._touch_sensor = sensor

    def enable(self):
        """Enable takeover detection (called when experiment starts)."""
        self._runtime_enabled = True
        self.reset()  # Clear stale state from previous episodes

    def disable(self):
        """Disable takeover detection (called when experiment ends)."""
        self._runtime_enabled = False
        self.reset()  # Clear state

    def _is_touch_only_mode(self) -> bool:
        """Check if detector is in touch-only mode (no analog methods active)."""
        return (
            self.config.use_touch
            and not self.config.use_current
            and not self.config.use_position
            and not self.config.use_velocity
            and not self.config.use_momentum
        )

    def detect_takeover(
        self,
        goal_pos: np.ndarray,
        curr_pos: np.ndarray,
        curr_vel: np.ndarray,
        curr_current: np.ndarray,
        tracking_torques: np.ndarray,
        gravity_torques: np.ndarray,
        friction_torques: np.ndarray,
        external_torques: Optional[np.ndarray] = None,
    ) -> bool:
        """Check if human is taking over."""
        if not self._runtime_enabled or not self.config.enable:
            return False

        # Optional grace period to avoid immediate retrigger after auto-return/resume
        if time.time() < self._suppress_until:
            return False

        # Touch sensor check (instant binary signal, no filtering needed)
        touch_active = False
        if self.config.use_touch and self._touch_sensor is not None:
            touch_active = self._touch_sensor.is_touched

        # Touch-only mode: skip all analog detection methods
        if self._is_touch_only_mode():
            result = touch_active
            if result and not self._last_takeover_state:
                logger.debug(">>> TAKEOVER DETECTED (touch)")
            elif not result and self._last_takeover_state:
                # Touch released -- don't log here, release is handled by detect_release
                pass
            self._last_takeover_state = result
            return result

        # --- Analog detection methods (existing logic) ---

        # Filter position and velocity raw measurements
        filtered_position = self._apply_filter_pipeline(curr_pos, 'position')
        filtered_velocity = self._apply_filter_pipeline(curr_vel, 'velocity')

        # Update buffers using filtered values
        if self.config.use_current:
            total_torque = tracking_torques + gravity_torques + friction_torques
            expected_current = self._torques_to_current(total_torque)

            # Compute raw error FIRST, then filter the ERROR signal
            # This is key: filtering raw current then subtracting expected doesn't work
            # because expected_current can change sharply (from tracking torques)
            raw_current_diff = curr_current - expected_current

            # Apply filter pipeline to the ERROR signal itself
            self.filtered_current_diff = self._apply_filter_pipeline(raw_current_diff, 'current')

            self.buffers['current'][self.buffer_idx] = np.abs(self.filtered_current_diff)

        # Always update position buffer so the dashboard can display tracking
        # errors regardless of which detection method is active.
        self.buffers['position'][self.buffer_idx] = np.abs(goal_pos - filtered_position)

        if self.config.use_velocity:
            self.buffers['velocity'][self.buffer_idx] = np.abs(filtered_velocity)

        if self.config.use_momentum and external_torques is not None:
            filtered_ext = self._apply_filter_pipeline(external_torques, 'momentum')
            self.buffers['momentum'][self.buffer_idx] = np.abs(filtered_ext)

        self.buffer_idx = (self.buffer_idx + 1) % self.config.window_size

        # Check thresholds (sustained over window)
        triggered_joints = []  # List of (method, joint_idx, value, threshold)

        # Touch detection in mixed mode (OR with other methods)
        if touch_active:
            triggered_joints.append(('touch', -1, 1.0, 0.5))

        if self.config.use_current:
            avg = np.mean(self.buffers['current'], axis=0)
            for j in range(self.dof):
                if avg[j] > self.current_threshold[j]:
                    triggered_joints.append(('current', j, avg[j], self.current_threshold[j]))

        if self.config.use_position:
            avg = np.mean(self.buffers['position'], axis=0)
            for j in range(self.dof):
                if avg[j] > self.position_threshold[j]:
                    triggered_joints.append(('position', j, avg[j], self.position_threshold[j]))

        if self.config.use_velocity:
            avg = np.mean(self.buffers['velocity'], axis=0)
            for j in range(self.dof):
                if avg[j] > self.config.velocity_threshold:
                    triggered_joints.append(('velocity', j, avg[j], self.config.velocity_threshold))

        if self.config.use_momentum and external_torques is not None:
            avg = np.mean(self.buffers['momentum'], axis=0)
            for j in range(self.dof):
                if avg[j] > self.momentum_threshold[j]:
                    triggered_joints.append(('momentum', j, avg[j], self.momentum_threshold[j]))

        # Return True if ANY method detects takeover
        result = len(triggered_joints) > 0

        # Only log on state transitions
        if result and not self._last_takeover_state:
            # Takeover just started - build detailed info for all joints
            triggered_set = {(m, j) for m, j, v, t in triggered_joints}

            trigger_source = " (touch)" if ('touch', -1) in triggered_set else ""
            lines = [f">>> TAKEOVER DETECTED{trigger_source}"]
            if self.config.use_current:
                avg_curr = np.mean(self.buffers['current'], axis=0)
                joint_strs = []
                for j in range(self.dof):
                    marker = " <<<" if ('current', j) in triggered_set else ""
                    joint_strs.append(f"J{j}={avg_curr[j]:.1f}/{self.current_threshold[j]:.0f}{marker}")
                lines.append(f"  Current: {', '.join(joint_strs)}")
            if self.config.use_position:
                avg_pos = np.mean(self.buffers['position'], axis=0)
                joint_strs = []
                for j in range(self.dof):
                    marker = " <<<" if ('position', j) in triggered_set else ""
                    joint_strs.append(f"J{j}={avg_pos[j]:.3f}/{self.position_threshold[j]:.2f}{marker}")
                lines.append(f"  Position: {', '.join(joint_strs)}")
            if self.config.use_velocity:
                avg_vel = np.mean(self.buffers['velocity'], axis=0)
                joint_strs = []
                for j in range(self.dof):
                    marker = " <<<" if ('velocity', j) in triggered_set else ""
                    joint_strs.append(f"J{j}={avg_vel[j]:.3f}/{self.config.velocity_threshold:.2f}{marker}")
                lines.append(f"  Velocity: {', '.join(joint_strs)}")
            if self.config.use_momentum:
                avg_mom = np.mean(self.buffers['momentum'], axis=0)
                joint_strs = []
                for j in range(self.dof):
                    marker = " <<<" if ('momentum', j) in triggered_set else ""
                    joint_strs.append(f"J{j}={avg_mom[j]:.3f}/{self.momentum_threshold[j]:.2f}{marker}")
                lines.append(f"  Momentum: {', '.join(joint_strs)}")
            logger.debug("\n".join(lines))
        self._last_takeover_state = result
        return result

    def detect_release(self, curr_vel: np.ndarray, curr_current: np.ndarray) -> bool:
        """Check if human has released control.

        Uses velocity stability and current derivative (rate of change) instead of absolute current.
        This works even with gravity compensation active, since steady-state gravity currents
        have zero derivative.

        When touch sensor is the only detection method (touch-only mode), uses a simple
        cooldown timer instead of velocity/current checks.

        NOTE: Uses RAW values intentionally. The filter pipeline is stateful (EMA, Butterworth)
        and is already called by detect_takeover() in the same tick. Calling it again here with
        different inputs (raw current vs current_diff) would corrupt filter state. The std dev
        window used here provides sufficient noise rejection on its own.
        """
        if not self._runtime_enabled or not self.config.enable_auto_return:
            return False

        # Touch-based release detection
        if self.config.use_touch and self._touch_sensor is not None:
            touch_active = self._touch_sensor.is_touched

            if self._is_touch_only_mode():
                # Touch-only mode: simple cooldown timer
                if not touch_active:
                    if self._touch_released_time is None:
                        self._touch_released_time = time.time()
                    elapsed = time.time() - self._touch_released_time
                    if elapsed >= self.config.touch_release_cooldown:
                        self._touch_released_time = None
                        return True
                else:
                    # Touch still active, reset cooldown timer
                    self._touch_released_time = None
                return False

            # Mixed mode: if touch is still active, definitely not released
            if touch_active:
                self._touch_released_time = None
                self.release_start_time = None
                return False
            # Touch released in mixed mode -- fall through to existing checks

        # Use raw values — do NOT call _apply_filter_pipeline here.
        # detect_takeover already advanced the filter state this tick.
        filtered_current = curr_current
        filtered_velocity = curr_vel

        # Update current history for derivative calculation (using filtered values)
        self.release_current_history[self.release_history_idx] = filtered_current
        self.release_history_idx = (self.release_history_idx + 1) % len(self.release_current_history)

        # Check velocity: must be below threshold (joints not moving)
        vel_abs = np.abs(filtered_velocity)
        vel_quiet = np.all(vel_abs < self.config.release_velocity)

        # Check current derivative: must be stable (low rate of change)
        # Use standard deviation across recent history as proxy for derivative magnitude
        current_std = np.std(self.release_current_history, axis=0)
        current_stable = np.all(current_std < self.config.release_current_derivative)

        # Cache stats for dashboard display
        self._release_vel_max = float(np.max(vel_abs))
        self._release_curr_std = float(np.max(current_std))

        # Both conditions must be met
        is_quiet = vel_quiet and current_stable

        if is_quiet:
            if self.release_start_time is None:
                self.release_start_time = time.time()

            elapsed = time.time() - self.release_start_time

            if elapsed >= self.config.auto_return_timeout:
                self.release_start_time = None
                return True
        else:
            self.release_start_time = None

        return False

    def reset(self):
        """Clear detection state and start a grace period to avoid stale triggers."""
        for key in self.buffers:
            self.buffers[key].fill(0.0)
        self.buffer_idx = 0
        self.release_current_history.fill(0.0)
        self.release_history_idx = 0
        self.release_start_time = None
        self._suppress_until = time.time() + self.config.takeover_grace_period
        self._last_takeover_state = False
        self._last_release_state = False
        self._touch_released_time = None

        # Reset current filter state
        self.filtered_current = None
        self.filtered_current_diff = None
        self.current_median_buffer = None
        self.current_median_idx = 0
        self.current_prev_value = None
        self._reset_butterworth_filter()

        # Reset position filter state
        self.filtered_position = None
        self.position_median_buffer = None
        self.position_median_idx = 0
        self.position_prev_value = None

        # Reset velocity filter state
        self.filtered_velocity = None
        self.velocity_median_buffer = None
        self.velocity_median_idx = 0
        self.velocity_prev_value = None

        # Reset momentum filter state
        self.filtered_momentum = None
        self.momentum_median_buffer = None
        self.momentum_median_idx = 0
        self.momentum_prev_value = None

    def _normalize_threshold(self, threshold: Union[float, np.ndarray]) -> np.ndarray:
        """Ensure current threshold is a per-joint array."""
        if isinstance(threshold, (int, float)):
            return np.full(self.dof, threshold, dtype=np.float32)

        arr = np.asarray(threshold, dtype=np.float32)
        if arr.shape != (self.dof,):
            raise ValueError(f"current_threshold must have {self.dof} elements, got shape {arr.shape}")
        return arr

    def _normalize_position_threshold(self, threshold: Union[float, np.ndarray]) -> np.ndarray:
        """Ensure position threshold is a per-joint array."""
        if isinstance(threshold, (int, float)):
            return np.full(self.dof, threshold, dtype=np.float32)

        arr = np.asarray(threshold, dtype=np.float32)
        if arr.shape != (self.dof,):
            raise ValueError(f"position_threshold must have {self.dof} elements, got shape {arr.shape}")
        return arr

    def _normalize_momentum_threshold(self, threshold: Union[float, np.ndarray]) -> np.ndarray:
        """Ensure momentum threshold is a per-joint array."""
        if isinstance(threshold, (int, float)):
            return np.full(self.dof, threshold, dtype=np.float32)

        arr = np.asarray(threshold, dtype=np.float32)
        if arr.shape != (self.dof,):
            raise ValueError(f"momentum_threshold must have {self.dof} elements, got shape {arr.shape}")
        return arr

    def _apply_filter_pipeline(
        self,
        raw_value: np.ndarray,
        signal_type: str,  # 'current', 'position', or 'velocity'
    ) -> np.ndarray:
        """Apply configurable 3-stage filter pipeline to any signal.

        Stages (all optional, configured per signal type):
        1. Median filter - Removes isolated spikes
        2. Spike clamp - Limits per-sample change rate
        3. EMA filter - Smooths remaining noise

        Args:
            raw_value: Raw measurements from sensors
            signal_type: Type of signal ('current', 'position', or 'velocity')

        Returns:
            Filtered measurements
        """
        # Get signal-specific config
        if signal_type == 'current':
            ema_enable = self.config.current_filter_ema_enable
            alpha = self.config.current_filter_alpha
            median_enable = self.config.current_spike_median_enable
            median_window = self.config.current_spike_median_window
            clamp_enable = self.config.current_spike_clamp_enable
            clamp_max = self.config.current_spike_clamp_max
            butterworth_enable = self.config.current_butterworth_enable
            median_buffer_attr = 'current_median_buffer'
            median_idx_attr = 'current_median_idx'
            prev_value_attr = 'current_prev_value'
            filtered_attr = 'filtered_current'
        elif signal_type == 'position':
            ema_enable = self.config.position_filter_ema_enable
            alpha = self.config.position_filter_alpha
            median_enable = self.config.position_spike_median_enable
            median_window = self.config.position_spike_median_window
            clamp_enable = self.config.position_spike_clamp_enable
            clamp_max = self.config.position_spike_clamp_max
            butterworth_enable = False  # Not implemented for position yet
            median_buffer_attr = 'position_median_buffer'
            median_idx_attr = 'position_median_idx'
            prev_value_attr = 'position_prev_value'
            filtered_attr = 'filtered_position'
        elif signal_type == 'velocity':
            ema_enable = self.config.velocity_filter_ema_enable
            alpha = self.config.velocity_filter_alpha
            median_enable = self.config.velocity_spike_median_enable
            median_window = self.config.velocity_spike_median_window
            clamp_enable = self.config.velocity_spike_clamp_enable
            clamp_max = self.config.velocity_spike_clamp_max
            butterworth_enable = False  # Not implemented for velocity yet
            median_buffer_attr = 'velocity_median_buffer'
            median_idx_attr = 'velocity_median_idx'
            prev_value_attr = 'velocity_prev_value'
            filtered_attr = 'filtered_velocity'
        elif signal_type == 'momentum':
            # Momentum reuses current filter settings (same noise characteristics)
            ema_enable = self.config.current_filter_ema_enable
            alpha = self.config.current_filter_alpha
            median_enable = self.config.current_spike_median_enable
            median_window = self.config.current_spike_median_window
            clamp_enable = self.config.current_spike_clamp_enable
            clamp_max = self.config.current_spike_clamp_max
            butterworth_enable = False  # Observer output is already smooth
            median_buffer_attr = 'momentum_median_buffer'
            median_idx_attr = 'momentum_median_idx'
            prev_value_attr = 'momentum_prev_value'
            filtered_attr = 'filtered_momentum'
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        processed = raw_value.copy()

        # Stage 1: Median filter (optional)
        if median_enable:
            median_buffer = getattr(self, median_buffer_attr)
            median_idx = getattr(self, median_idx_attr)

            # Initialize buffer on first call
            if median_buffer is None:
                median_buffer = np.zeros((median_window, self.dof), dtype=np.float32)
                median_buffer[:] = raw_value
                setattr(self, median_buffer_attr, median_buffer)
                median_idx = 0

            # Add new sample to circular buffer
            median_buffer[median_idx] = raw_value
            median_idx = (median_idx + 1) % median_window
            setattr(self, median_idx_attr, median_idx)

            # Compute median
            processed = np.median(median_buffer, axis=0).astype(np.float32)

        # Stage 2: Spike clamp (optional)
        if clamp_enable:
            prev_value = getattr(self, prev_value_attr)

            if prev_value is not None:
                delta = processed - prev_value
                delta_clamped = np.clip(delta, -clamp_max, clamp_max)
                processed = prev_value + delta_clamped

            setattr(self, prev_value_attr, processed.copy())

        # Stage 3: Butterworth low-pass filter (optional, current only)
        # Provides better noise rejection than EMA with less phase delay
        if butterworth_enable:
            processed = self._apply_butterworth_filter(processed)

        # Stage 4: EMA filter (optional)
        if ema_enable:
            filtered = getattr(self, filtered_attr)

            if filtered is None:
                setattr(self, filtered_attr, processed.copy())
                return processed

            filtered = alpha * processed + (1.0 - alpha) * filtered
            setattr(self, filtered_attr, filtered)
            return filtered

        return processed

    def _init_butterworth_filter(self, cutoff: float, order: int, fs: float = 100.0) -> None:
        """Initialize Butterworth low-pass filter coefficients.

        Uses bilinear transform to convert analog Butterworth to digital filter.
        Currently supports order=2 (recommended for real-time use).

        Args:
            cutoff: Cutoff frequency in Hz
            order: Filter order (2 recommended)
            fs: Sampling frequency in Hz (default 100 Hz)
        """
        if order != 2:
            print(f"[TakeoverDetector] Warning: Only order=2 Butterworth supported, got {order}")
            order = 2

        # Pre-warp cutoff frequency for bilinear transform
        wc = np.tan(np.pi * cutoff / fs)
        wc2 = wc * wc
        sqrt2 = np.sqrt(2.0)

        # 2nd-order Butterworth coefficients via bilinear transform
        # H(s) = 1 / (s^2 + sqrt(2)*s + 1) transformed to H(z)
        norm = 1.0 + sqrt2 * wc + wc2

        self._butter_b = np.array([
            wc2 / norm,           # b0
            2.0 * wc2 / norm,     # b1
            wc2 / norm,           # b2
        ], dtype=np.float64)

        self._butter_a = np.array([
            1.0,                                    # a0 (always 1, not used)
            2.0 * (wc2 - 1.0) / norm,              # a1
            (1.0 - sqrt2 * wc + wc2) / norm,       # a2
        ], dtype=np.float64)

        # Initialize history buffers [order, dof] - start with zeros
        self._butter_x_hist = np.zeros((order, self.dof), dtype=np.float64)
        self._butter_y_hist = np.zeros((order, self.dof), dtype=np.float64)

        print(f"[TakeoverDetector] Butterworth filter initialized: "
              f"cutoff={cutoff}Hz, order={order}, fs={fs}Hz")
        print(f"[TakeoverDetector] Filter coefficients: b={self._butter_b}, a={self._butter_a}")

    def _apply_butterworth_filter(self, x: np.ndarray) -> np.ndarray:
        """Apply 2nd-order Butterworth filter to input signal.

        Uses Direct Form I difference equation:
        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

        Args:
            x: Input signal array of shape (dof,)

        Returns:
            Filtered signal array of shape (dof,)
        """
        if self._butter_b is None:
            return x  # Filter not initialized

        b = self._butter_b
        a = self._butter_a

        # Compute output: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        y = (b[0] * x +
             b[1] * self._butter_x_hist[0] +
             b[2] * self._butter_x_hist[1] -
             a[1] * self._butter_y_hist[0] -
             a[2] * self._butter_y_hist[1])

        # Update history (shift and insert new values)
        self._butter_x_hist[1] = self._butter_x_hist[0]
        self._butter_x_hist[0] = x.copy()
        self._butter_y_hist[1] = self._butter_y_hist[0]
        self._butter_y_hist[0] = y.copy()

        return y.astype(np.float32)

    def _reset_butterworth_filter(self) -> None:
        """Reset Butterworth filter state (call on mode changes)."""
        if self._butter_x_hist is not None:
            self._butter_x_hist.fill(0.0)
        if self._butter_y_hist is not None:
            self._butter_y_hist.fill(0.0)

    def _torques_to_current(self, torques: np.ndarray) -> np.ndarray:
        """Convert torques (Nm) to expected motor currents in mA."""
        currents_mA = np.zeros(self.dof, dtype=np.float32)
        for i in range(self.dof):
            motor = self.motor_types[i].value
            current_units = motor.convert_torque_to_current(torques[i])
            currents_mA[i] = current_units * motor.current_unit_to_mA
        return currents_mA

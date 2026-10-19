"""
Goal tracking controller for Gello.

Implements PID control with optional velocity feedforward to make Gello
track desired joint positions.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GoalTrackingConfig:
    """PID tracking configuration with optional feedforward."""

    enable: bool = True
    kp: np.ndarray = None  # Proportional gains per joint
    ki: np.ndarray = None  # Integral gains per joint
    kd: np.ndarray = None  # Derivative gains per joint

    # Anti-windup settings
    integral_limit: float = 1.0  # Max integral contribution (radians)
    reset_integral_on_goal_change: bool = True  # Reset integral when goal changes significantly

    # Feedforward control
    feedforward_enable: bool = False  # Master enable/disable for feedforward
    kff: np.ndarray = None  # Velocity feedforward gains per joint
    velocity_filter_alpha: float = 0.3  # EMA smoothing for goal velocity (0-1, lower = smoother)

    def __post_init__(self):
        """Convert lists to numpy arrays if needed."""
        if self.kp is not None and not isinstance(self.kp, np.ndarray):
            self.kp = np.array(self.kp, dtype=np.float32)
        if self.ki is not None and not isinstance(self.ki, np.ndarray):
            self.ki = np.array(self.ki, dtype=np.float32)
        if self.kd is not None and not isinstance(self.kd, np.ndarray):
            self.kd = np.array(self.kd, dtype=np.float32)
        if self.kff is not None and not isinstance(self.kff, np.ndarray):
            self.kff = np.array(self.kff, dtype=np.float32)


class GoalTracker:
    """
    PID controller with optional velocity feedforward for tracking goal joint positions.

    Computes tracking torques to make Gello follow a desired trajectory.
    Includes integral action to eliminate steady-state error from
    gravity/friction disturbances, and optional velocity feedforward to
    reduce tracking lag.
    """

    def __init__(self, config: GoalTrackingConfig, dof: int):
        """
        Initialize goal tracker.

        Args:
            config: Goal tracking configuration
            dof: Degrees of freedom
        """
        self.config = config
        self.dof = dof

        # Set default gains if not provided
        if config.kp is None:
            config.kp = np.ones(dof) * 20.0
        if config.ki is None:
            config.ki = np.zeros(dof)  # Default: no integral (backwards compatible)
        if config.kd is None:
            config.kd = np.ones(dof) * 2.0

        # Expand gains for bilateral systems (7 values → 14)
        config.kp = self._expand_gains(config.kp, dof, "Kp")
        config.ki = self._expand_gains(config.ki, dof, "Ki")
        config.kd = self._expand_gains(config.kd, dof, "Kd")

        # Expand feedforward gains if provided
        if config.kff is not None:
            config.kff = self._expand_gains(config.kff, dof, "Kff")

        # Initialize state variables
        self._init_state()

    def _expand_gains(self, gains: np.ndarray, dof: int, name: str) -> np.ndarray:
        """
        Expand gains array for bilateral systems.

        If gains length is half of DOF, duplicate for both arms.

        Args:
            gains: Gain array
            dof: Target degrees of freedom
            name: Name for error messages

        Returns:
            Expanded gains array of shape (dof,)
        """
        if len(gains) == dof:
            return gains
        elif len(gains) * 2 == dof:
            # Bilateral system: duplicate gains for both arms
            return np.concatenate([gains, gains])
        else:
            raise ValueError(
                f"{name} must have {dof} elements or {dof // 2} elements "
                f"(for bilateral auto-duplication), got {len(gains)}"
            )

    def _init_state(self):
        """Initialize internal state variables. Called after gains are expanded."""
        self._goal_position: Optional[np.ndarray] = None
        self._tracking_enabled = False

        # Integral state
        self._integral_error = np.zeros(self.dof, dtype=np.float32)
        self._last_goal: Optional[np.ndarray] = None
        self._dt = 0.01  # 100Hz control loop

        # Feedforward state
        self._prev_goal_for_velocity: Optional[np.ndarray] = None
        self._prev_goal_time: Optional[float] = None
        self._filtered_goal_velocity = np.zeros(self.dof, dtype=np.float32)

    @property
    def is_tracking(self) -> bool:
        """Check if currently tracking a goal."""
        return self._tracking_enabled and self._goal_position is not None

    def set_goal(self, goal_position: np.ndarray):
        """
        Set a new goal position.

        If reset_integral_on_goal_change is enabled and the goal has changed
        significantly (>0.1 rad), the integral state is reset to prevent
        sudden torque spikes.

        Also computes goal velocity via differentiation for feedforward control.

        Args:
            goal_position: Desired joint positions (radians)
        """
        if len(goal_position) != self.dof:
            raise ValueError(f"Goal must have {self.dof} elements, got {len(goal_position)}")

        goal_position = np.array(goal_position, dtype=np.float32)
        current_time = time.time()

        # Compute goal velocity for feedforward (differentiate goal position)
        if self.config.feedforward_enable:
            if self._prev_goal_for_velocity is not None and self._prev_goal_time is not None:
                dt = current_time - self._prev_goal_time
                if dt > 0.001:  # Avoid division by very small dt
                    raw_velocity = (goal_position - self._prev_goal_for_velocity) / dt
                    # EMA filter to smooth velocity
                    alpha = self.config.velocity_filter_alpha
                    self._filtered_goal_velocity = (
                        alpha * raw_velocity + (1.0 - alpha) * self._filtered_goal_velocity
                    )

            self._prev_goal_for_velocity = goal_position.copy()
            self._prev_goal_time = current_time

        # Reset integral when goal changes significantly
        if self.config.reset_integral_on_goal_change:
            if self._last_goal is not None:
                goal_change = np.linalg.norm(goal_position - self._last_goal)
                if goal_change > 0.1:  # ~6 degrees
                    self._integral_error = np.zeros(self.dof, dtype=np.float32)

        self._last_goal = goal_position.copy()
        self._goal_position = goal_position

    def reset_integral(self):
        """
        Reset integral state.

        Call this when tracking stops (e.g., during takeover) or when
        restarting tracking from a new position.
        """
        self._integral_error = np.zeros(self.dof, dtype=np.float32)

    def reset_feedforward(self):
        """
        Reset feedforward state.

        Call this when tracking stops or restarts to avoid stale velocity
        estimates causing torque spikes.
        """
        self._prev_goal_for_velocity = None
        self._prev_goal_time = None
        self._filtered_goal_velocity = np.zeros(self.dof, dtype=np.float32)

    def start_tracking(self):
        """Enable goal tracking."""
        if not self.config.enable:
            return
        self._tracking_enabled = True

    def stop_tracking(self):
        """Disable goal tracking."""
        self._tracking_enabled = False
        # Reset state when tracking stops
        self.reset_integral()
        self.reset_feedforward()

    def set_gains(self, kp=None, ki=None, kd=None, kff=None, reset_integral=True):
        """Update PID and feedforward gains at runtime. Takes effect on next control loop iteration."""
        if kp is not None:
            kp = np.array(kp, dtype=np.float32)
            self.config.kp = self._expand_gains(kp, self.dof, "Kp")
        if ki is not None:
            ki = np.array(ki, dtype=np.float32)
            self.config.ki = self._expand_gains(ki, self.dof, "Ki")
        if kd is not None:
            kd = np.array(kd, dtype=np.float32)
            self.config.kd = self._expand_gains(kd, self.dof, "Kd")
        if kff is not None:
            kff = np.array(kff, dtype=np.float32)
            self.config.kff = self._expand_gains(kff, self.dof, "Kff")
        if reset_integral:
            self.reset_integral()
            self.reset_feedforward()

    @property
    def goal_velocity(self) -> np.ndarray:
        """Get the current filtered goal velocity (for debugging/logging)."""
        return self._filtered_goal_velocity.copy()

    def compute_tracking_torque(
        self,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Compute PID tracking torque with optional velocity feedforward.

        Args:
            current_position: Current joint positions (radians)
            current_velocity: Current joint velocities (rad/s)

        Returns:
            Tracking torques (N·m)
        """
        if not self.is_tracking:
            return np.zeros(self.dof, dtype=np.float32)

        # Compute position error
        position_error = self._goal_position - current_position

        # Update integral (with anti-windup clamping)
        self._integral_error += position_error * self._dt
        self._integral_error = np.clip(
            self._integral_error,
            -self.config.integral_limit,
            self.config.integral_limit
        )

        # PID control: tau_fb = Kp * error + Ki * integral - Kd * velocity
        kp_term = self.config.kp * position_error
        ki_term = self.config.ki * self._integral_error
        kd_term = self.config.kd * current_velocity
        tau_fb = kp_term + ki_term - kd_term

        # Velocity feedforward: tau_ff = Kff * goal_velocity
        tau_ff = np.zeros(self.dof, dtype=np.float32)
        if self.config.feedforward_enable and self.config.kff is not None:
            tau_ff = self.config.kff * self._filtered_goal_velocity

        return tau_fb + tau_ff

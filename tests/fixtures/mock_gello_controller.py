#!/usr/bin/env python3
"""
Mock Gello controller for testing without hardware.

Provides realistic mock implementations that can be used in unit and integration tests.
"""

import numpy as np
from typing import Tuple, Optional
from unittest.mock import Mock
from iiil.gello.controllers.base_controller import BilateralGelloController


class MockGelloController(BilateralGelloController):
    """
    Mock implementation of BilateralGelloController for testing.

    This mock controller simulates a real controller without requiring hardware.
    It maintains internal state and responds to commands realistically.
    """

    def __init__(self, dof: int = 7, simulate_bilateral: bool = False):
        """
        Initialize mock controller.

        Args:
            dof: Degrees of freedom per arm
            simulate_bilateral: If True, simulate bilateral setup (dof * 2 total joints)
        """
        self._dof_per_arm = dof
        self._simulate_bilateral = simulate_bilateral
        self._dof = (dof * 2) if simulate_bilateral else dof

        # Internal state
        self._positions = np.zeros(self._dof, dtype=np.float32)
        self._velocities = np.zeros(self._dof, dtype=np.float32)
        self._currents = np.zeros(self._dof, dtype=np.float32)
        self._valid_joints = np.ones(self._dof, dtype=bool)

        # Goal tracking state
        self._goal = None
        self._is_following_goal = False
        self._control_started = False

        # Simulated movement parameters
        self._position_noise = 0.001  # Small noise for realism
        self._goal_convergence_rate = 0.1  # How fast to converge to goal

    @property
    def dof(self) -> int:
        """Get total DOF."""
        return self._dof

    @property
    def q(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint positions."""
        # Add small noise for realism
        noisy_positions = self._positions + np.random.randn(self._dof) * self._position_noise

        # If following goal, simulate movement toward goal
        if self._is_following_goal and self._goal is not None:
            # Move positions toward goal
            error = self._goal - self._positions
            self._positions += error * self._goal_convergence_rate
            self._velocities = error * self._goal_convergence_rate / 0.01  # Approximate velocity

        return noisy_positions.copy(), self._valid_joints.copy()

    @property
    def v(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint velocities."""
        return self._velocities.copy(), self._valid_joints.copy()

    @property
    def current(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint currents."""
        return self._currents.copy(), self._valid_joints.copy()

    @property
    def motor_groups(self):
        """Return empty motor group metadata for interface compatibility."""
        return ()

    @property
    def position_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mock trajectory is the current position."""
        return self.q

    @property
    def velocity_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mock trajectory velocity is the current velocity."""
        return self.v

    def set_new_goal(self, goal_joint_angles: np.ndarray):
        """Set new goal position."""
        # Pad or truncate goal to match DOF
        if len(goal_joint_angles) < self._dof:
            self._goal = np.zeros(self._dof, dtype=np.float32)
            self._goal[:len(goal_joint_angles)] = goal_joint_angles
        else:
            self._goal = goal_joint_angles[:self._dof].copy()

    def start_control(self):
        """Start control loop."""
        self._control_started = True

    def stop_goal_following(self):
        """Stop following goal."""
        self._is_following_goal = False

    def resume_goal_following(self):
        """Resume following goal."""
        self._is_following_goal = True

    def has_reached_goal(self, block: bool = False, tolerance: float = 0.1) -> bool:
        """Check if goal reached."""
        if self._goal is None:
            return True

        error = np.abs(self._positions - self._goal)
        return np.all(error < tolerance)

    def close(self, disable_torque: bool = True):
        """Close controller."""
        self._control_started = False
        self._is_following_goal = False

    def ensure_torque_enabled(self):
        """Ensure torque is enabled (no-op for mock)."""
        pass

    def set_stop_tracking_on_takeover(self, stop: bool):
        """Set whether to stop tracking on takeover (no-op for mock)."""
        pass

    def set_takeover_callback(self, callback):
        """Set takeover callback (no-op for mock)."""
        pass

    def set_auto_return_callback(self, callback):
        """Set auto-return callback (no-op for mock)."""
        pass

    def set_position(self, positions: np.ndarray):
        """Set current position (for testing)."""
        self._positions = positions.copy()

    def set_velocity(self, velocities: np.ndarray):
        """Set current velocity (for testing)."""
        self._velocities = velocities.copy()

    def set_current(self, currents: np.ndarray):
        """Set current readings (for testing)."""
        self._currents = currents.copy()

    def set_joint_validity(self, valid_joints: np.ndarray):
        """Set which joints are valid (for testing)."""
        self._valid_joints = valid_joints.copy()


def create_mock_controller_a1() -> MockGelloController:
    """Create mock A1 controller (6 DOF)."""
    return MockGelloController(dof=6, simulate_bilateral=False)


def create_mock_controller_r1pro() -> MockGelloController:
    """Create mock R1Pro single arm controller (7 DOF)."""
    return MockGelloController(dof=7, simulate_bilateral=False)


def create_mock_controller_r1pro_bilateral() -> MockGelloController:
    """Create mock R1Pro bilateral controller (14 joints total, 7 DOF per arm)."""
    return MockGelloController(dof=7, simulate_bilateral=True)


def create_mock_controller_with_state(
    dof: int,
    positions: Optional[np.ndarray] = None,
    velocities: Optional[np.ndarray] = None,
    currents: Optional[np.ndarray] = None,
) -> MockGelloController:
    """
    Create mock controller with specific state.

    Args:
        dof: Degrees of freedom
        positions: Initial positions (default: zeros)
        velocities: Initial velocities (default: zeros)
        currents: Initial currents (default: zeros)

    Returns:
        MockGelloController with specified state
    """
    controller = MockGelloController(dof=dof, simulate_bilateral=False)

    if positions is not None:
        controller.set_position(positions)
    if velocities is not None:
        controller.set_velocity(velocities)
    if currents is not None:
        controller.set_current(currents)

    return controller


# Factory for easy controller creation in tests
class MockControllerFactory:
    """Factory for creating mock controllers in tests."""

    _instance = None

    @staticmethod
    def get_gello_controller_instance(
        gello_config_name: str = "gello_a1",
        port_overrides=None
    ) -> MockGelloController:
        """
        Get mock controller instance matching the real factory API.

        Args:
            gello_config_name: Config name (gello_a1, gello_r1pro, etc.)
            port_overrides: Ignored for mock

        Returns:
            MockGelloController instance
        """
        # Parse config name to determine DOF
        if "a1" in gello_config_name:
            return create_mock_controller_a1()
        elif "r1pro" in gello_config_name:
            return create_mock_controller_r1pro_bilateral()
        else:
            # Default to A1
            return create_mock_controller_a1()

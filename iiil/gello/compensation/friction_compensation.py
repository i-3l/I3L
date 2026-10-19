"""
Static friction compensation module.

Provides plug-and-play friction compensation for GELLO arms.
"""

import numpy as np
from typing import Optional
from iiil.gello.compensation.compensation_config import FrictionCompensationConfig


class FrictionCompensator:
    """
    Static friction compensation using oscillating directions.

    This compensator helps overcome static friction (stiction) by applying
    small oscillating torques when joint velocities are near zero.
    """

    def __init__(self, config: FrictionCompensationConfig, dof: int):
        """
        Initialize friction compensator.

        Args:
            config: Friction compensation configuration
            dof: Number of degrees of freedom
        """
        self.config = config
        self.dof = dof
        self.enabled = config.enable

        self.speed_threshold = config.speed_threshold
        self.gains = config.get_gains(dof)

        # Internal state: oscillation direction for each joint
        self._oscillation_direction = np.ones(dof, dtype=np.float32)

    def compute_torque(
        self,
        velocities: np.ndarray,
        gravity_torques: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute friction compensation torque.

        When joint velocity is below threshold, applies an oscillating torque
        proportional to the gravity torque magnitude.

        Args:
            velocities: Joint velocities in rad/s (shape: (dof,))
            gravity_torques: Gravity compensation torques in Nm (shape: (dof,))
                            If None, uses constant magnitude for oscillation

        Returns:
            Friction compensation torques in Nm (shape: (dof,))

        Example:
            >>> compensator = FrictionCompensator(config, dof=6)
            >>> v = np.array([0.1, 0.05, 0.8, 0.2, 0.0, 0.1])
            >>> tau_g = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            >>> tau_f = compensator.compute_torque(v, tau_g)
        """
        if not self.enabled:
            return np.zeros(self.dof, dtype=np.float32)

        # Validate input dimensions
        if len(velocities) != self.dof:
            raise ValueError(f"Expected {self.dof} velocities, got {len(velocities)}")

        tau_friction = np.zeros(self.dof, dtype=np.float32)

        # Identify joints moving slowly (below threshold)
        slow_joints = np.abs(velocities) < self.speed_threshold

        if not np.any(slow_joints):
            return tau_friction

        if gravity_torques is not None:
            if len(gravity_torques) != self.dof:
                raise ValueError(f"Expected {self.dof} gravity torques, got {len(gravity_torques)}")

            # Apply oscillating torque proportional to gravity magnitude
            tau_friction[slow_joints] = (
                self.gains[slow_joints] *
                self._oscillation_direction[slow_joints] *
                np.abs(gravity_torques[slow_joints])
            )
        else:
            # Use constant magnitude if no gravity torques provided
            tau_friction[slow_joints] = (
                self.gains[slow_joints] *
                self._oscillation_direction[slow_joints]
            )

        # Flip oscillation direction for next time
        self._oscillation_direction[slow_joints] *= -1

        return tau_friction

    def reset_oscillations(self):
        """Reset oscillation directions to initial state."""
        self._oscillation_direction = np.ones(self.dof, dtype=np.float32)

    def update_gains(self, gains: np.ndarray):
        """
        Update friction compensation gains.

        Args:
            gains: New gains (shape: (dof,))
        """
        if len(gains) != self.dof:
            raise ValueError(f"Expected {self.dof} gains, got {len(gains)}")

        self.gains = np.array(gains, dtype=np.float32)

    def update_speed_threshold(self, threshold: float):
        """
        Update velocity threshold for friction compensation.

        Args:
            threshold: New threshold in rad/s
        """
        self.speed_threshold = float(threshold)

    def enable(self):
        """Enable friction compensation."""
        self.enabled = True

    def disable(self):
        """Disable friction compensation."""
        self.enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if friction compensation is enabled."""
        return self.enabled


def create_friction_compensator(
    dof: int,
    speed_threshold: float = 0.9,
    gain: float = 0.3,
    per_joint_gains: Optional[np.ndarray] = None,
    enable: bool = True
) -> FrictionCompensator:
    """
    Convenience function to create a friction compensator.

    Args:
        dof: Number of degrees of freedom
        speed_threshold: Velocity threshold in rad/s (default: 0.9)
        gain: Global gain multiplier (default: 0.3)
        per_joint_gains: Optional per-joint gains (overrides global gain)
        enable: Whether to enable compensation (default: True)

    Returns:
        Configured FrictionCompensator
    """
    config = FrictionCompensationConfig(
        enable=enable,
        speed_threshold=speed_threshold,
        gain=gain,
        per_joint_gains=per_joint_gains
    )

    return FrictionCompensator(config, dof)

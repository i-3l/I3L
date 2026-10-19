"""
Anti-friction compensation module.

Provides velocity-based anti-friction compensation for GELLO arms to counteract
back-EMF resistance when backdriving motors in current control mode.
"""

import numpy as np
from typing import Optional
from iiil.gello.compensation.compensation_config import AntiFrictionCompensationConfig


class AntiFrictionCompensator:
    """
    Anti-friction compensation for motor backdriving resistance.

    When motors are in current control mode and you command a current (e.g., for
    gravity compensation), backdriving the motor generates back EMF that the
    current controller fights against, causing stiffness.

    This compensator applies torque in the direction of motion to counteract
    this resistance:

        τ_anti_friction = viscous_gain × velocity + coulomb_gain × sign(velocity)

    - Viscous term: Compensates back EMF (proportional to velocity)
    - Coulomb term: Compensates static/kinetic friction (constant in direction of motion)
    """

    def __init__(self, config: AntiFrictionCompensationConfig, dof: int):
        """
        Initialize anti-friction compensator.

        Args:
            config: Anti-friction compensation configuration
            dof: Number of degrees of freedom
        """
        self.config = config
        self.dof = dof
        self.enabled = config.enable

        self.viscous_gains = config.get_viscous_gains(dof)
        self.coulomb_gains = config.get_coulomb_gains(dof)
        self.velocity_deadband = config.velocity_deadband

    def compute_torque(self, velocities: np.ndarray) -> np.ndarray:
        """
        Compute anti-friction compensation torque.

        Applies torque in the direction of motion to assist backdriving:
        - When velocity > 0: positive torque
        - When velocity < 0: negative torque

        Args:
            velocities: Joint velocities in rad/s (shape: (dof,))

        Returns:
            Anti-friction compensation torques in Nm (shape: (dof,))

        Example:
            >>> compensator = AntiFrictionCompensator(config, dof=6)
            >>> v = np.array([0.5, -0.3, 0.1, -0.8, 0.0, 0.2])
            >>> tau = compensator.compute_torque(v)
            >>> # tau will have same sign as v (assisting motion)
        """
        if not self.enabled:
            return np.zeros(self.dof, dtype=np.float32)

        # Validate input dimensions
        if len(velocities) != self.dof:
            raise ValueError(f"Expected {self.dof} velocities, got {len(velocities)}")

        tau = np.zeros(self.dof, dtype=np.float32)

        # Apply deadband to prevent oscillation at low velocities
        active = np.abs(velocities) > self.velocity_deadband

        if not np.any(active):
            return tau

        # Viscous term: sqrt scaling (boosts small velocities, compresses large)
        tau[active] += self.viscous_gains[active] * np.sqrt(np.abs(velocities[active])) * np.sign(velocities[active])

        # Coulomb term: constant magnitude in direction of velocity
        tau[active] += self.coulomb_gains[active] * np.sign(velocities[active])

        return tau

    def update_viscous_gains(self, gains: np.ndarray):
        """
        Update viscous compensation gains.

        Args:
            gains: New viscous gains (shape: (dof,))
        """
        if len(gains) != self.dof:
            raise ValueError(f"Expected {self.dof} gains, got {len(gains)}")

        self.viscous_gains = np.array(gains, dtype=np.float32)

    def update_coulomb_gains(self, gains: np.ndarray):
        """
        Update coulomb compensation gains.

        Args:
            gains: New coulomb gains (shape: (dof,))
        """
        if len(gains) != self.dof:
            raise ValueError(f"Expected {self.dof} gains, got {len(gains)}")

        self.coulomb_gains = np.array(gains, dtype=np.float32)

    def update_velocity_deadband(self, deadband: float):
        """
        Update velocity deadband threshold.

        Args:
            deadband: New deadband in rad/s
        """
        self.velocity_deadband = float(deadband)

    def enable(self):
        """Enable anti-friction compensation."""
        self.enabled = True

    def disable(self):
        """Disable anti-friction compensation."""
        self.enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if anti-friction compensation is enabled."""
        return self.enabled


def create_anti_friction_compensator(
    dof: int,
    viscous_gain: float = 0.05,
    coulomb_gain: float = 0.01,
    velocity_deadband: float = 0.05,
    per_joint_viscous_gains: Optional[np.ndarray] = None,
    per_joint_coulomb_gains: Optional[np.ndarray] = None,
    enable: bool = True
) -> AntiFrictionCompensator:
    """
    Convenience function to create an anti-friction compensator.

    Args:
        dof: Number of degrees of freedom
        viscous_gain: Global viscous gain in Nm/(rad/s) (default: 0.05)
        coulomb_gain: Global coulomb gain in Nm (default: 0.01)
        velocity_deadband: Deadband in rad/s (default: 0.05)
        per_joint_viscous_gains: Optional per-joint viscous gains
        per_joint_coulomb_gains: Optional per-joint coulomb gains
        enable: Whether to enable compensation (default: True)

    Returns:
        Configured AntiFrictionCompensator
    """
    config = AntiFrictionCompensationConfig(
        enable=enable,
        viscous_gain=viscous_gain,
        coulomb_gain=coulomb_gain,
        velocity_deadband=velocity_deadband,
        per_joint_viscous_gains=per_joint_viscous_gains,
        per_joint_coulomb_gains=per_joint_coulomb_gains
    )

    return AntiFrictionCompensator(config, dof)

"""
Gravity compensation module using Pinocchio.

Provides plug-and-play gravity compensation for GELLO arms.
"""

import numpy as np
import pinocchio as pin
from typing import Optional
from iiil.gello.compensation.compensation_config import GravityCompensationConfig


class GravityCompensator:
    """
    Gravity compensation using Pinocchio dynamics.

    This module computes gravity compensation torques for a robot arm
    based on its URDF model.
    """

    def __init__(self, config: GravityCompensationConfig, dof: int):
        """
        Initialize gravity compensator.

        Args:
            config: Gravity compensation configuration
            dof: Number of degrees of freedom
        """
        self.config = config
        self.dof = dof
        self.enabled = config.enable

        if self.enabled:
            if not config.urdf_path:
                raise ValueError("URDF path must be provided for gravity compensation")

            # Load URDF model
            self.model = pin.buildModelFromUrdf(config.urdf_path)

            # Override gravity vector if specified (default is [0, 0, -9.81] i.e. +z up)
            if config.gravity_vector is not None:
                self.model.gravity.linear = np.array(config.gravity_vector, dtype=np.float64)
                print(f"Gravity vector set to {config.gravity_vector}")

            self.data = self.model.createData()
            
            # Check if URDF DOF matches system DOF
            urdf_dof = self.model.nv
            self._is_bilateral_single_arm_urdf = (urdf_dof * 2 == dof)
            
            if urdf_dof != dof and not self._is_bilateral_single_arm_urdf:
                raise ValueError(
                    f"URDF DOF ({urdf_dof}) does not match system DOF ({dof}). "
                    f"For bilateral systems, URDF should have {dof} DOF or {dof//2} DOF (single arm)."
                )
            
            if self._is_bilateral_single_arm_urdf:
                print(f"Gravity compensation initialized with single-arm LEFT URDF ({urdf_dof} DOF) for bilateral system ({dof} DOF)")
            else:
                print(f"Gravity compensation initialized with URDF: {config.urdf_path}")

            # Get multipliers
            self.multipliers = config.get_multipliers(dof)
        else:
            self.model = None
            self.data = None
            self.multipliers = None
            self._is_bilateral_single_arm_urdf = False

    def compute_torque(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute gravity compensation torque.

        Uses Pinocchio's RNEA (Recursive Newton-Euler Algorithm) with zero
        accelerations to compute pure gravity torques.

        Args:
            positions: Joint positions in radians (shape: (dof,))
            velocities: Joint velocities in rad/s (shape: (dof,), optional)
                       If None, assumes zero velocities

        Returns:
            Gravity compensation torques in Nm (shape: (dof,))

        Example:
            >>> compensator = GravityCompensator(config, dof=6)
            >>> q = np.zeros(6)
            >>> tau_g = compensator.compute_torque(q)
        """
        if not self.enabled:
            return np.zeros(self.dof, dtype=np.float32)

        if velocities is None:
            velocities = np.zeros(self.dof, dtype=np.float32)

        # Validate input dimensions
        if len(positions) != self.dof:
            raise ValueError(f"Expected {self.dof} positions, got {len(positions)}")
        if len(velocities) != self.dof:
            raise ValueError(f"Expected {self.dof} velocities, got {len(velocities)}")

        # Handle bilateral system with single-arm URDF
        if self._is_bilateral_single_arm_urdf:
            arm_dof = self.model.nv

            # Split positions and velocities into right and left arms
            q_right = positions[:arm_dof]
            q_left = positions[arm_dof:]
            v_right = velocities[:arm_dof] if velocities is not None else np.zeros(arm_dof, dtype=np.float32)
            v_left = velocities[arm_dof:] if velocities is not None else np.zeros(arm_dof, dtype=np.float32)

            # For right arm, mirror the joint positions to use the left-arm URDF model
            # The URDF represents the left arm, so we:
            # 1. Negate right arm joint positions (mirror to left arm frame)
            # 2. Compute gravity torques using left arm model
            # 3. Negate torques back (right arm torques have opposite sign)
            # This works because the right arm is a kinematic mirror of the left arm
            q_right_mirrored = -q_right  # Mirror joint positions to left arm frame
            tau_g_right_mirrored = pin.rnea(
                self.model,
                self.data,
                q_right_mirrored,
                np.zeros_like(v_right),
                np.zeros_like(v_right)
            )
            # Mirror the torques back to right arm frame
            tau_g_right = -tau_g_right_mirrored

            # Compute gravity torque for left arm directly (matches URDF)
            tau_g_left = pin.rnea(
                self.model,
                self.data,
                q_left,
                np.zeros_like(v_left),
                np.zeros_like(v_left)
            )

            # Concatenate torques (right arm first, then left arm)
            tau_g = np.concatenate([tau_g_right, tau_g_left])
        else:
            # Standard case: URDF DOF matches system DOF
            # Compute gravity torque using RNEA with zero accelerations
            # tau = RNEA(q, v, a=0) gives us pure gravity + coriolis terms
            # For gravity only, we pass v=0, a=0
            tau_g = pin.rnea(
                self.model,
                self.data,
                positions,
                np.zeros_like(velocities),  # Zero velocity for pure gravity
                np.zeros_like(velocities)   # Zero acceleration
            )

        # Apply per-joint multipliers
        tau_g = tau_g * self.multipliers

        return tau_g.astype(np.float32)

    def update_multipliers(self, multipliers: np.ndarray):
        """
        Update gravity compensation multipliers.

        Useful for online tuning of compensation strength.

        Args:
            multipliers: New multipliers (shape: (dof,))
        """
        if len(multipliers) != self.dof:
            raise ValueError(f"Expected {self.dof} multipliers, got {len(multipliers)}")

        self.multipliers = np.array(multipliers, dtype=np.float32)

    def enable(self):
        """Enable gravity compensation."""
        if self.model is None:
            raise RuntimeError("Cannot enable: compensator not initialized with URDF")
        self.enabled = True

    def disable(self):
        """Disable gravity compensation."""
        self.enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if gravity compensation is enabled."""
        return self.enabled


def create_gravity_compensator(
    urdf_path: str,
    dof: int,
    multipliers: Optional[np.ndarray] = None,
    enable: bool = True
) -> GravityCompensator:
    """
    Convenience function to create a gravity compensator.

    Args:
        urdf_path: Path to URDF file
        dof: Number of degrees of freedom
        multipliers: Optional per-joint multipliers (default: all 1.0)
        enable: Whether to enable compensation (default: True)

    Returns:
        Configured GravityCompensator
    """
    if multipliers is None:
        multipliers = np.ones(dof, dtype=np.float32)

    config = GravityCompensationConfig(
        enable=enable,
        urdf_path=urdf_path,
        multipliers=multipliers
    )

    return GravityCompensator(config, dof)

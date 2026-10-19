"""
Configuration classes for compensation modules.

Provides clean, modular configuration for gravity and friction compensation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, Sequence


@dataclass
class GravityCompensationConfig:
    """
    Configuration for gravity compensation.

    Attributes:
        enable: Whether to enable gravity compensation
        urdf_path: Path to URDF file for dynamics model
        multipliers: Per-joint gravity compensation multipliers (0.0 to 1.0+)
                     Allows tuning compensation strength per joint
    """
    enable: bool = False
    urdf_path: str = ""
    multipliers: Union[float, Sequence[float], np.ndarray] = 1.0
    gravity_vector: Optional[Sequence[float]] = None  # Default: [0, 0, -9.81] (+z up)

    def __post_init__(self):
        """Validate and normalize multipliers to numpy array."""
        if isinstance(self.multipliers, (int, float)):
            # Will be expanded to correct size when DOF is known
            self._multipliers_scalar = self.multipliers
        else:
            self.multipliers = np.array(self.multipliers, dtype=np.float32)
        if self.gravity_vector is not None:
            self.gravity_vector = np.array(self.gravity_vector, dtype=np.float64)

    def get_multipliers(self, dof: int) -> np.ndarray:
        """
        Get multipliers array for the specified DOF.

        For bilateral systems, if multipliers length is half the DOF,
        the multipliers are duplicated for both arms.

        Args:
            dof: Number of degrees of freedom

        Returns:
            Multipliers as numpy array of shape (dof,)
        """
        if hasattr(self, '_multipliers_scalar'):
            return np.full(dof, self._multipliers_scalar, dtype=np.float32)
        elif isinstance(self.multipliers, np.ndarray):
            if len(self.multipliers) == dof:
                return self.multipliers
            elif len(self.multipliers) * 2 == dof:
                # Bilateral system: duplicate multipliers for both arms
                return np.concatenate([self.multipliers, self.multipliers])
            else:
                raise ValueError(
                    f"Multipliers length ({len(self.multipliers)}) "
                    f"does not match DOF ({dof}) or half DOF ({dof // 2})"
                )
        else:
            multipliers_array = np.array(self.multipliers, dtype=np.float32)
            if len(multipliers_array) == dof:
                return multipliers_array
            elif len(multipliers_array) * 2 == dof:
                # Bilateral system: duplicate multipliers for both arms
                return np.concatenate([multipliers_array, multipliers_array])
            else:
                raise ValueError(
                    f"Multipliers length ({len(multipliers_array)}) "
                    f"does not match DOF ({dof}) or half DOF ({dof // 2})"
                )


@dataclass
class FrictionCompensationConfig:
    """
    Configuration for static friction compensation.

    Static friction compensation helps overcome stiction by oscillating
    motor directions when velocity is near zero.

    Attributes:
        enable: Whether to enable friction compensation
        speed_threshold: Velocity threshold below which friction comp activates (rad/s)
        gain: Gain multiplier for friction compensation torque
        per_joint_gains: Optional per-joint gains (overrides global gain)
    """
    enable: bool = False
    speed_threshold: float = 0.9
    gain: float = 0.3
    per_joint_gains: Optional[Union[Sequence[float], np.ndarray]] = None

    def __post_init__(self):
        """Validate and normalize per-joint gains."""
        if self.per_joint_gains is not None:
            self.per_joint_gains = np.array(self.per_joint_gains, dtype=np.float32)

    def get_gains(self, dof: int) -> np.ndarray:
        """
        Get gains array for the specified DOF.

        Args:
            dof: Number of degrees of freedom

        Returns:
            Gains as numpy array of shape (dof,)
        """
        if self.per_joint_gains is not None:
            if len(self.per_joint_gains) != dof:
                raise ValueError(
                    f"Per-joint gains length ({len(self.per_joint_gains)}) "
                    f"does not match DOF ({dof})"
                )
            return self.per_joint_gains
        else:
            return np.full(dof, self.gain, dtype=np.float32)


@dataclass
class AntiFrictionCompensationConfig:
    """
    Configuration for anti-friction (back-EMF) compensation.

    Anti-friction compensation helps overcome motor resistance when backdriving
    by applying velocity-proportional torque in the direction of motion.

    τ_anti_friction = viscous_gain × velocity + coulomb_gain × sign(velocity)

    Attributes:
        enable: Whether to enable anti-friction compensation
        viscous_gain: Gain for velocity-proportional term (Nm/(rad/s))
        coulomb_gain: Gain for constant friction term (Nm) - sign(velocity) term
        velocity_deadband: Velocity below which compensation is zero (prevents oscillation)
        per_joint_viscous_gains: Optional per-joint viscous gains (overrides global gain)
        per_joint_coulomb_gains: Optional per-joint coulomb gains (overrides global gain)
    """
    enable: bool = False
    viscous_gain: float = 0.05
    coulomb_gain: float = 0.01
    velocity_deadband: float = 0.05  # rad/s
    per_joint_viscous_gains: Optional[Union[Sequence[float], np.ndarray]] = None
    per_joint_coulomb_gains: Optional[Union[Sequence[float], np.ndarray]] = None

    def __post_init__(self):
        """Validate and normalize per-joint gains."""
        if self.per_joint_viscous_gains is not None:
            self.per_joint_viscous_gains = np.array(self.per_joint_viscous_gains, dtype=np.float32)
        if self.per_joint_coulomb_gains is not None:
            self.per_joint_coulomb_gains = np.array(self.per_joint_coulomb_gains, dtype=np.float32)

    def get_viscous_gains(self, dof: int) -> np.ndarray:
        """
        Get viscous gains array for the specified DOF.

        Args:
            dof: Number of degrees of freedom

        Returns:
            Viscous gains as numpy array of shape (dof,)
        """
        if self.per_joint_viscous_gains is not None:
            if len(self.per_joint_viscous_gains) == dof:
                return self.per_joint_viscous_gains
            elif len(self.per_joint_viscous_gains) * 2 == dof:
                # Bilateral system: duplicate gains for both arms
                return np.concatenate([self.per_joint_viscous_gains, self.per_joint_viscous_gains])
            else:
                raise ValueError(
                    f"Per-joint viscous gains length ({len(self.per_joint_viscous_gains)}) "
                    f"does not match DOF ({dof}) or half DOF ({dof // 2})"
                )
        return np.full(dof, self.viscous_gain, dtype=np.float32)

    def get_coulomb_gains(self, dof: int) -> np.ndarray:
        """
        Get coulomb gains array for the specified DOF.

        Args:
            dof: Number of degrees of freedom

        Returns:
            Coulomb gains as numpy array of shape (dof,)
        """
        if self.per_joint_coulomb_gains is not None:
            if len(self.per_joint_coulomb_gains) == dof:
                return self.per_joint_coulomb_gains
            elif len(self.per_joint_coulomb_gains) * 2 == dof:
                # Bilateral system: duplicate gains for both arms
                return np.concatenate([self.per_joint_coulomb_gains, self.per_joint_coulomb_gains])
            else:
                raise ValueError(
                    f"Per-joint coulomb gains length ({len(self.per_joint_coulomb_gains)}) "
                    f"does not match DOF ({dof}) or half DOF ({dof // 2})"
                )
        return np.full(dof, self.coulomb_gain, dtype=np.float32)


@dataclass
class CompensationConfig:
    """
    Combined configuration for all compensation modules.

    Attributes:
        gravity: Gravity compensation configuration
        friction: Friction compensation configuration
        anti_friction: Anti-friction (back-EMF) compensation configuration
    """
    gravity: GravityCompensationConfig = field(default_factory=GravityCompensationConfig)
    friction: FrictionCompensationConfig = field(default_factory=FrictionCompensationConfig)
    anti_friction: AntiFrictionCompensationConfig = field(default_factory=AntiFrictionCompensationConfig)

    @classmethod
    def create_default(cls, dof: int = 6) -> 'CompensationConfig':
        """
        Create a default configuration with sensible defaults.

        Args:
            dof: Number of degrees of freedom

        Returns:
            CompensationConfig with default settings
        """
        return cls(
            gravity=GravityCompensationConfig(
                enable=False,
                urdf_path="",
                multipliers=0.7
            ),
            friction=FrictionCompensationConfig(
                enable=False,
                speed_threshold=0.9,
                gain=0.3
            ),
            anti_friction=AntiFrictionCompensationConfig(
                enable=False,
                viscous_gain=0.05,
                coulomb_gain=0.01,
                velocity_deadband=0.05
            )
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'CompensationConfig':
        """
        Create configuration from dictionary (e.g., loaded from YAML).

        Args:
            config_dict: Dictionary with 'gravity', 'friction', and/or 'anti_friction' keys

        Returns:
            CompensationConfig object
        """
        gravity_config = config_dict.get('gravity', {})
        friction_config = config_dict.get('friction', {})
        anti_friction_config = config_dict.get('anti_friction', {})

        return cls(
            gravity=GravityCompensationConfig(**gravity_config) if gravity_config else GravityCompensationConfig(),
            friction=FrictionCompensationConfig(**friction_config) if friction_config else FrictionCompensationConfig(),
            anti_friction=AntiFrictionCompensationConfig(**anti_friction_config) if anti_friction_config else AntiFrictionCompensationConfig()
        )

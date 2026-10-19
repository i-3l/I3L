"""Compensation configuration builder for Gello controllers."""

import os
import numpy as np
from iiil.gello.compensation.compensation_config import (
    CompensationConfig,
    GravityCompensationConfig,
    FrictionCompensationConfig,
    AntiFrictionCompensationConfig,
)
from iiil.utils.misc_utils import get_package_root_path


def build_compensation_config(config) -> CompensationConfig:
    """
    Build compensation configuration from Gello config.

    Args:
        config: Gello configuration object with compensation.gravity, compensation.friction,
                and optionally compensation.anti_friction

    Returns:
        CompensationConfig with gravity, friction, and anti-friction settings

    Raises:
        AttributeError: If config doesn't have 'compensation' field
    """
    if not hasattr(config, "compensation"):
        raise AttributeError(
            "Config must have 'compensation' field with gravity and friction sections. "
            "Legacy format is no longer supported."
        )

    comp_config = config.compensation

    base_root = get_package_root_path()
    urdf_rel = comp_config.gravity.urdf_path

    # Resolve URDF path (may be null when gravity compensation is disabled)
    if urdf_rel is not None:
        candidate_path = os.path.join(base_root, urdf_rel)
        if not os.path.exists(candidate_path):
            candidate_path = os.path.join(os.path.dirname(base_root), urdf_rel)
    else:
        candidate_path = None

    gravity_vector = getattr(comp_config.gravity, "gravity_vector", None)

    gravity_config = GravityCompensationConfig(
        enable=comp_config.gravity.enable,
        urdf_path=candidate_path,
        multipliers=np.array(comp_config.gravity.multipliers),
        gravity_vector=gravity_vector,
    )

    friction_config = FrictionCompensationConfig(
        enable=comp_config.friction.enable,
        speed_threshold=comp_config.friction.get("speed_threshold", 0.9),
        gain=comp_config.friction.get("gain", 0.3),
    )

    # Build anti-friction config (optional section)
    anti_friction_dict = getattr(comp_config, "anti_friction", None)
    if anti_friction_dict is not None:
        # Handle per-joint gains if provided
        per_joint_viscous = anti_friction_dict.get("per_joint_viscous_gains", None)
        per_joint_coulomb = anti_friction_dict.get("per_joint_coulomb_gains", None)

        anti_friction_config = AntiFrictionCompensationConfig(
            enable=anti_friction_dict.get("enable", False),
            viscous_gain=anti_friction_dict.get("viscous_gain", 0.05),
            coulomb_gain=anti_friction_dict.get("coulomb_gain", 0.01),
            velocity_deadband=anti_friction_dict.get("velocity_deadband", 0.05),
            per_joint_viscous_gains=np.array(per_joint_viscous) if per_joint_viscous is not None else None,
            per_joint_coulomb_gains=np.array(per_joint_coulomb) if per_joint_coulomb is not None else None,
        )
    else:
        # Default: disabled
        anti_friction_config = AntiFrictionCompensationConfig(enable=False)

    return CompensationConfig(
        gravity=gravity_config,
        friction=friction_config,
        anti_friction=anti_friction_config
    )

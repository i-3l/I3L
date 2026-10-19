"""Controller instance builder for Gello."""

import numpy as np
from typing import List
from iiil.gello.configs.gello_motor_group import GelloMotorGroup
from iiil.gello.compensation.compensation_config import CompensationConfig
from iiil.gello.utils.joint_utils import infer_dof_from_motor_groups


def build_controller(
    control_type: str,
    motor_groups: List[GelloMotorGroup],
    compensation_config: CompensationConfig,
    config,
):
    """
    Build a controller instance based on type.

    Args:
        control_type: Type of bilateral control ("bilateral" is the only supported type)
        motor_groups: List of configured motor groups
        compensation_config: Compensation configuration
        config: Original Gello configuration with additional parameters

    Returns:
        Configured controller instance

    Raises:
        ValueError: If control_type is not supported
    """
    if control_type != "bilateral":
        raise ValueError(
            f"Unsupported controller type: {control_type}. "
            f"Only 'bilateral' is supported."
        )

    return _build_bilateral(motor_groups, compensation_config, config)


def _build_bilateral(
    motor_groups: List[GelloMotorGroup],
    compensation_config: CompensationConfig,
    config,
):
    """
    Build BilateralController.

    Uses current-based control with joint offsets and goal tracking.
    Supports GELLO_TO_ENV (user control) and ENV_TO_GELLO (goal tracking) modes.
    """
    from iiil.gello.controllers.bilateral_controller import BilateralController
    from iiil.gello.factory.goal_tracking_builder import (
        build_goal_tracking_config,
        build_takeover_config,
        build_momentum_observer_config,
        build_touch_sensor_config,
    )

    dof = infer_dof_from_motor_groups(motor_groups)

    # Get joint offsets
    joint_offsets = config.get("joint_offsets", None)
    if joint_offsets is None:
        joint_offsets = np.zeros(dof)
    else:
        joint_offsets = np.array(joint_offsets)

    # Get joint limits (for proper angle wrapping with large offsets)
    joint_limits_cfg = config.get("joint_limits", None)
    if joint_limits_cfg is not None:
        joint_limits = (
            np.array(joint_limits_cfg.lower, dtype=np.float32),
            np.array(joint_limits_cfg.upper, dtype=np.float32),
        )
    else:
        joint_limits = None

    # Get control parameters
    control_frequency = config.get("control_frequency", 100.0)
    debug = config.get("debug", False)

    # Build goal tracking config (for ENV_TO_GELLO mode)
    goal_tracking_config = build_goal_tracking_config(config, dof)

    # Build takeover detection config
    takeover_config = build_takeover_config(config)

    # Build momentum observer config
    momentum_observer_config = build_momentum_observer_config(config)

    # Build touch sensor config
    touch_sensor_config = build_touch_sensor_config(config)

    # Get GELLO_TO_ENV mode config (controls torque behavior in user-control mode)
    gello_to_env_config = config.get("gello_to_env", None)
    if gello_to_env_config is not None:
        gello_to_env_config = dict(gello_to_env_config)  # Convert OmegaConf to dict
    else:
        gello_to_env_config = {"torque_enabled": False}  # Default: free float

    # Create controller
    controller = BilateralController(
        motor_groups=motor_groups,
        compensation_config=compensation_config,
        goal_tracking_config=goal_tracking_config,
        takeover_config=takeover_config,
        momentum_observer_config=momentum_observer_config,
        gello_to_env_config=gello_to_env_config,
        joint_offsets=joint_offsets,
        joint_limits=joint_limits,
        control_frequency=control_frequency,
        debug=debug,
        touch_sensor_config=touch_sensor_config,
    )

    # Configure debug messages (console output filtering)
    debug_msg_config = config.get("debug_messages", None)
    if debug_msg_config is not None:
        controller.configure_debug_messages(debug_msg_config)

    # Enable debug logging if configured
    debug_logging_config = config.get("debug_logging", None)
    if debug_logging_config is not None and debug_logging_config.get("enable", False):
        controller.enable_debug_logging(debug_logging_config)

    return controller

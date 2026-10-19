"""Motor group builder for Gello controllers."""

from typing import List, Optional
from iiil.gello.configs.gello_motor_group import GelloMotorGroup
from iiil.gello.configs.motor_type import MotorType


def build_motor_groups(config, port_overrides: Optional[list] = None) -> List[GelloMotorGroup]:
    """
    Build motor groups from configuration.

    Args:
        config: Gello configuration object with motor_groups field
        port_overrides: Optional list of port paths to override config ports

    Returns:
        List of configured GelloMotorGroup objects

    Raises:
        AttributeError: If config doesn't have 'motor_groups' field
    """
    if not hasattr(config, "motor_groups"):
        raise AttributeError(
            "Config must have 'motor_groups' field. "
            "Legacy format (gello_motor_groups) is no longer supported."
        )

    motor_group_configs = config.motor_groups

    motor_groups = []
    for idx, motor_group_config in enumerate(motor_group_configs):
        # Use port override if available, otherwise use config port
        device_port = motor_group_config.device_port
        if port_overrides and idx < len(port_overrides):
            device_port = port_overrides[idx]

        motor_groups.append(
            GelloMotorGroup(
                motor_ids=motor_group_config.motor_ids,
                motor_type=MotorType.from_string(motor_group_config.motor_type),
                robot_joint_indices=motor_group_config.robot_joint_indices,
                arm_joint_signs=motor_group_config.arm_joint_signs,
                device_port=device_port,
                baudrate=motor_group_config.baudrate,
                current_limit=motor_group_config.get("current_limit", None),
                max_joint_velocity=motor_group_config.get("max_joint_velocity", 1.0),
                profile_velocity=motor_group_config.get("profile_velocity", None),
                profile_acceleration=motor_group_config.get("profile_acceleration", None),
                position_deadband=motor_group_config.get("position_deadband", 0),
            )
        )

    return motor_groups

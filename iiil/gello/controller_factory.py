"""
Controller factory module - separated to avoid circular imports.

This module is separate from gello_utils.py to break the circular import chain.
It provides a singleton factory for creating Gello controller instances.
"""

from typing import Optional
from iiil.gello.factory import (
    load_gello_config,
    build_motor_groups,
    build_compensation_config,
    build_controller,
)


class A1GelloControllerFactory:
    """
    Singleton factory for creating and maintaining A1 Gello controller instances.

    This factory ensures only one controller instance exists at a time and handles
    configuration changes by recreating the instance when needed.
    """

    _controller_instance: Optional[object] = None  # Avoid importing BilateralGelloController at module level
    _current_config_name: Optional[str] = None
    _current_task_overlay: Optional[str] = None

    @staticmethod
    def get_gello_controller_instance(gello_config_name: str = "gello_a1", port_overrides: Optional[list] = None, task_overlay: Optional[str] = None):
        """
        Get or create the singleton Gello controller instance.

        If the config name or task overlay changes, the existing controller is
        closed and a new one is created.

        Args:
            gello_config_name: Name of config file without .yaml extension
                             (e.g., "gello_a1" or "gello_r1pro")
            port_overrides: Optional list of port paths to override config ports
            task_overlay: Optional per-task overlay name (e.g., "autotune").
                         Loaded from {config_name}/{task_overlay}.yaml and
                         deep-merged on top of the base config.

        Returns:
            Configured BilateralGelloController instance
        """
        # Reset instance if config or overlay changed
        if (
            A1GelloControllerFactory._controller_instance is not None
            and (
                A1GelloControllerFactory._current_config_name != gello_config_name
                or A1GelloControllerFactory._current_task_overlay != task_overlay
            )
        ):
            A1GelloControllerFactory._controller_instance.close()
            A1GelloControllerFactory._controller_instance = None

        # Create new instance if needed
        if A1GelloControllerFactory._controller_instance is None:
            A1GelloControllerFactory._current_config_name = gello_config_name
            A1GelloControllerFactory._current_task_overlay = task_overlay
            A1GelloControllerFactory._controller_instance = _create_controller(gello_config_name, port_overrides, task_overlay)

        return A1GelloControllerFactory._controller_instance


def _create_controller(config_name: str, port_overrides: Optional[list] = None, task_overlay: Optional[str] = None):
    """
    Create a new controller instance from configuration.

    This is a helper function that orchestrates the controller creation process.

    Args:
        config_name: Name of the Gello configuration
        port_overrides: Optional list of port paths to override config ports
        task_overlay: Optional per-task overlay name for config merging

    Returns:
        Newly created controller instance
    """
    # Load configuration (with optional overlay merge)
    config = load_gello_config(config_name, task_overlay=task_overlay)

    _validate_motor_counts(config_name, config)

    # Build components
    motor_groups = build_motor_groups(config, port_overrides)
    compensation_config = build_compensation_config(config)

    # Create controller
    control_type = config.bilateral_control_type
    controller = build_controller(control_type, motor_groups, compensation_config, config)

    return controller


def _validate_motor_counts(config_name, config) -> None:
    """
    Quick sanity check: ensure total joint count in motor_groups matches expected
    for the selected Gello configuration.

    Args:
        config_name: Name of the Gello configuration (e.g., gello_a1, gello_r1pro)
        config: Loaded config (OmegaConf)

    Raises:
        ValueError: If the number of configured joints does not match expectations.
    """
    # Expected total joint count per configuration
    expected_total = {
        "gello_a1": 6,      # Single A1 arm (6 DOF)
        "gello_r1pro": 14,  # Bilateral R1Pro (7 DOF per arm)
        "gello_franka": 7,  # Single Franka arm (7 DOF)
    }.get(config_name, None)

    if expected_total is None:
        return  # Unknown config; skip validation

    motor_groups = getattr(config, "motor_groups", None)
    if not motor_groups:
        raise ValueError(f"Gello config '{config_name}' has no motor_groups defined.")

    total_joints = 0
    for group in motor_groups:
        # Prefer robot_joint_indices length; fall back to motor_ids
        joint_indices = getattr(group, "robot_joint_indices", None)
        if joint_indices is not None:
            total_joints += len(joint_indices)
        else:
            motor_ids = getattr(group, "motor_ids", [])
            total_joints += len(motor_ids)

    if total_joints != expected_total:
        raise ValueError(
            f"Gello config '{config_name}' has {total_joints} configured joints, "
            f"expected {expected_total} for this hardware. Please fix the config."
        )

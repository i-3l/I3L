"""
Joint utilities for validation, limits, and transformations.

Provides functions for joint limit checking, validation, and common joint operations.
"""

import numpy as np
from typing import Tuple, Optional


def validate_joint_limits(
    joint_angles: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[bool, np.ndarray]:
    """
    Validate that joint angles are within specified limits.

    Args:
        joint_angles: Joint angles to validate (radians)
        lower_limits: Lower joint limits (radians)
        upper_limits: Upper joint limits (radians)
        tolerance: Tolerance for limit checking (default: 1e-6)

    Returns:
        Tuple of (all_valid, violations_mask)
        - all_valid: True if all joints are within limits
        - violations_mask: Boolean array indicating which joints violate limits

    Example:
        >>> q = np.array([0.5, 1.5, -0.5])
        >>> lower = np.array([0.0, 0.0, 0.0])
        >>> upper = np.array([1.0, 2.0, 1.0])
        >>> validate_joint_limits(q, lower, upper)
        (False, array([False, False, True]))
    """
    violations = (joint_angles < lower_limits - tolerance) | (joint_angles > upper_limits + tolerance)
    all_valid = not np.any(violations)
    return all_valid, violations


def clip_to_limits(
    joint_angles: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray
) -> np.ndarray:
    """
    Clip joint angles to specified limits.

    Args:
        joint_angles: Joint angles to clip (radians)
        lower_limits: Lower joint limits (radians)
        upper_limits: Upper joint limits (radians)

    Returns:
        Clipped joint angles

    Example:
        >>> q = np.array([0.5, 1.5, -0.5])
        >>> lower = np.array([0.0, 0.0, 0.0])
        >>> upper = np.array([1.0, 2.0, 1.0])
        >>> clip_to_limits(q, lower, upper)
        array([0.5, 1.5, 0.0])
    """
    return np.clip(joint_angles, lower_limits, upper_limits)


def wrap_to_joint_centers(
    joint_angles: np.ndarray,
    joint_centers: np.ndarray,
) -> np.ndarray:
    """
    Wrap joint angles to within ±π of each joint's center.

    Handles encoder wraparound for joints whose encoder-to-radian conversion
    can produce values outside [-π, π]. Uses modular arithmetic to find the
    equivalent angle closest to the joint center.

    This matches FrankaJoylo's normalization:
        center + mod(angle - center + π, 2π) - π

    Args:
        joint_angles: Joint angles in radians (may be outside [-π, π])
        joint_centers: Center of each joint's operating range in radians

    Returns:
        Wrapped joint angles, each within ±π of its center
    """
    two_pi = 2.0 * np.pi
    return joint_centers + np.mod(joint_angles - joint_centers + np.pi, two_pi) - np.pi


def check_joint_distance_to_goal(
    current_positions: np.ndarray,
    goal_positions: np.ndarray,
    tolerance: float = 0.1,
    valid_joints: Optional[np.ndarray] = None
) -> bool:
    """
    Check if current positions are within tolerance of goal.

    Args:
        current_positions: Current joint positions (radians)
        goal_positions: Goal joint positions (radians)
        tolerance: Maximum allowed distance in radians (default: 0.1)
        valid_joints: Boolean mask of valid joints (default: all True)

    Returns:
        True if all valid joints are within tolerance of goal

    Example:
        >>> current = np.array([0.05, 0.08, 0.12])
        >>> goal = np.array([0.0, 0.0, 0.0])
        >>> check_joint_distance_to_goal(current, goal, tolerance=0.1)
        False
    """
    if valid_joints is None:
        valid_joints = np.ones(len(current_positions), dtype=bool)

    # Only check valid joints
    if not np.any(valid_joints):
        return True  # No valid joints, consider as reached

    joint_diff = np.abs(current_positions[valid_joints] - goal_positions[valid_joints])
    return np.all(joint_diff <= tolerance)


def compute_joint_velocities(
    positions_current: np.ndarray,
    positions_previous: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Compute joint velocities from finite differences.

    Args:
        positions_current: Current joint positions (radians)
        positions_previous: Previous joint positions (radians)
        dt: Time difference (seconds)

    Returns:
        Joint velocities (rad/s)

    Example:
        >>> current = np.array([1.0, 2.0, 3.0])
        >>> previous = np.array([0.9, 1.9, 2.9])
        >>> compute_joint_velocities(current, previous, 0.1)
        array([1.0, 1.0, 1.0])
    """
    if dt <= 0:
        return np.zeros_like(positions_current)

    return (positions_current - positions_previous) / dt


def compute_joint_accelerations(
    velocities_current: np.ndarray,
    velocities_previous: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Compute joint accelerations from finite differences.

    Args:
        velocities_current: Current joint velocities (rad/s)
        velocities_previous: Previous joint velocities (rad/s)
        dt: Time difference (seconds)

    Returns:
        Joint accelerations (rad/s²)
    """
    if dt <= 0:
        return np.zeros_like(velocities_current)

    return (velocities_current - velocities_previous) / dt


def create_default_joint_limits(dof: int, limit_value: float = np.pi) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create default symmetric joint limits for a given DOF.

    Args:
        dof: Number of degrees of freedom
        limit_value: Limit magnitude (default: π radians)

    Returns:
        Tuple of (lower_limits, upper_limits)

    Example:
        >>> create_default_joint_limits(3)
        (array([-3.14159265, -3.14159265, -3.14159265]),
         array([3.14159265, 3.14159265, 3.14159265]))
    """
    lower_limits = np.full(dof, -limit_value, dtype=np.float32)
    upper_limits = np.full(dof, limit_value, dtype=np.float32)
    return lower_limits, upper_limits


def infer_dof_from_motor_groups(motor_groups) -> int:
    """
    Infer the total degrees of freedom from motor groups.

    Args:
        motor_groups: Sequence of GelloMotorGroup objects

    Returns:
        Total number of degrees of freedom

    Example:
        >>> # Assuming motor_groups with robot_joint_indices [0,1,2] and [3,4,5]
        >>> infer_dof_from_motor_groups(motor_groups)
        6
    """
    if not motor_groups:
        return 0

    max_joint_index = max(
        max(group.robot_joint_indices) for group in motor_groups
    )
    return max_joint_index + 1


def expand_goal_to_full_dof(
    goal_joint_angles: np.ndarray,
    current_positions: np.ndarray,
    total_dof: int,
) -> np.ndarray:
    """
    Expand a possibly shorter goal vector to match the controller DOF.

    Args:
        goal_joint_angles: Desired joint angles (can be <= total_dof)
        current_positions: Current joint positions used to fill unspecified joints
        total_dof: Target number of joints for the controller

    Returns:
        np.ndarray: Goal vector with length == total_dof

    Raises:
        ValueError: If goal length exceeds total_dof or current_positions length mismatches total_dof
    """
    goal = np.asarray(goal_joint_angles, dtype=np.float32)
    if len(goal) > total_dof:
        raise ValueError(
            f"Goal has {len(goal)} joints but controller requires {total_dof}"
        )

    if len(current_positions) != total_dof:
        raise ValueError(
            f"Current positions length {len(current_positions)} does not match controller DOF {total_dof}"
        )

    if len(goal) == total_dof:
        return goal.copy()

    full_goal = current_positions.copy()
    full_goal[: len(goal)] = goal
    return full_goal

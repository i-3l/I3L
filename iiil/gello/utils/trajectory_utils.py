"""
Trajectory generation and interpolation utilities for GELLO system.

Provides functions for velocity-limited trajectory interpolation.
"""

import numpy as np
import time
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrajectoryState:
    """
    State of a trajectory interpolation.

    Attributes:
        start_position: Starting joint positions
        goal_position: Goal joint positions
        start_time: Time when trajectory started
        duration: Total duration of the trajectory in seconds
    """
    start_position: np.ndarray
    goal_position: np.ndarray
    start_time: float
    duration: float


def compute_trajectory_duration(
    start_position: np.ndarray,
    goal_position: np.ndarray,
    max_velocities: np.ndarray,
    valid_joints: Optional[np.ndarray] = None
) -> float:
    """
    Compute the time needed to reach goal with velocity limits.

    The trajectory duration is determined by the slowest joint.

    Args:
        start_position: Current joint positions (radians)
        goal_position: Desired joint positions (radians)
        max_velocities: Maximum velocity for each joint (rad/s)
        valid_joints: Boolean mask of valid joints (default: all True)

    Returns:
        Duration in seconds needed to complete trajectory

    Example:
        >>> start = np.array([0, 0, 0])
        >>> goal = np.array([1.57, 0.5, 0.3])
        >>> max_vel = np.array([1.0, 1.0, 1.0])
        >>> compute_trajectory_duration(start, goal, max_vel)
        1.57
    """
    if valid_joints is None:
        valid_joints = np.ones(len(start_position), dtype=bool)

    # Calculate distance for each joint
    joint_distances = np.abs(goal_position - start_position)

    # Only consider valid joints
    valid_distances = joint_distances[valid_joints]
    valid_max_velocities = max_velocities[valid_joints]

    if len(valid_distances) == 0 or not np.any(valid_max_velocities > 0):
        return 0.0

    # Calculate time needed for each joint
    joint_times = np.zeros_like(valid_distances)
    non_zero_vel_mask = valid_max_velocities > 0
    joint_times[non_zero_vel_mask] = (
        valid_distances[non_zero_vel_mask] / valid_max_velocities[non_zero_vel_mask]
    )

    # Return the maximum time (slowest joint determines duration)
    return float(np.max(joint_times))


def interpolate_trajectory(
    trajectory_state: TrajectoryState,
    current_time: Optional[float] = None
) -> np.ndarray:
    """
    Compute interpolated position along a linear trajectory.

    Args:
        trajectory_state: Trajectory state containing start, goal, time info
        current_time: Current time (default: time.time())

    Returns:
        Interpolated joint positions

    Example:
        >>> state = TrajectoryState(
        ...     start_position=np.array([0, 0]),
        ...     goal_position=np.array([1, 1]),
        ...     start_time=0.0,
        ...     duration=1.0
        ... )
        >>> interpolate_trajectory(state, current_time=0.5)
        array([0.5, 0.5])
    """
    if current_time is None:
        current_time = time.time()

    elapsed_time = current_time - trajectory_state.start_time

    # Check if trajectory is complete
    if elapsed_time >= trajectory_state.duration:
        return trajectory_state.goal_position.copy()

    # Linear interpolation
    if trajectory_state.duration > 0:
        t = elapsed_time / trajectory_state.duration
    else:
        t = 1.0

    # Clamp to [0, 1]
    t = np.clip(t, 0.0, 1.0)

    # Interpolate: p(t) = start + t * (goal - start)
    interpolated = (
        trajectory_state.start_position +
        t * (trajectory_state.goal_position - trajectory_state.start_position)
    )

    return interpolated


def create_trajectory(
    start_position: np.ndarray,
    goal_position: np.ndarray,
    max_velocities: np.ndarray,
    valid_joints: Optional[np.ndarray] = None,
    start_time: Optional[float] = None
) -> TrajectoryState:
    """
    Create a new trajectory state from start to goal.

    Args:
        start_position: Starting joint positions
        goal_position: Goal joint positions
        max_velocities: Maximum velocities for each joint
        valid_joints: Boolean mask of valid joints (default: all True)
        start_time: Start time (default: time.time())

    Returns:
        TrajectoryState object ready for interpolation
    """
    if start_time is None:
        start_time = time.time()

    duration = compute_trajectory_duration(
        start_position, goal_position, max_velocities, valid_joints
    )

    return TrajectoryState(
        start_position=start_position.copy(),
        goal_position=goal_position.copy(),
        start_time=start_time,
        duration=duration
    )


def is_trajectory_complete(
    trajectory_state: TrajectoryState,
    current_time: Optional[float] = None
) -> bool:
    """
    Check if a trajectory has completed.

    Args:
        trajectory_state: Trajectory state to check
        current_time: Current time (default: time.time())

    Returns:
        True if trajectory is complete, False otherwise
    """
    if current_time is None:
        current_time = time.time()

    elapsed_time = current_time - trajectory_state.start_time
    return elapsed_time >= trajectory_state.duration

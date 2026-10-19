"""
Base controller interface for bilateral GELLO control.

Defines the abstract interface that all GELLO controllers must implement.
"""

import numpy as np
from abc import ABC, abstractmethod
from iiil.gello.configs.gello_motor_group import GelloMotorGroup
from typing import Tuple, Sequence


class BilateralGelloController(ABC):
    """
    Abstract base class for bilateral GELLO controllers.

    Bilateral control means the robot can command the GELLO device to follow
    a trajectory while the user can also interact with it.
    """

    # Default joint limits for the robot arm (can be overridden by subclasses)
    JOINT_LIMIT_LOW = np.array([-np.pi / 2, -0.1, -3.3161, -2.8798, -1.6581, -2.8798])
    JOINT_LIMIT_HIGH = np.array([np.pi / 2, 3.2289, 0.01, 2.8798, 1.6581, 2.8798])

    @abstractmethod
    def stop_goal_following(self):
        """Stops making the physical GELLO move towards the goal position."""
        pass

    @abstractmethod
    def resume_goal_following(self):
        """Resumes making the physical GELLO move towards the provided goal position."""
        pass

    @abstractmethod
    def set_new_goal(self, goal_joint_angles: np.ndarray):
        """
        Set a new goal for the controller to follow.

        Args:
            goal_joint_angles: The new goal joint angles in radians.
        """
        pass

    @abstractmethod
    def start_control(self):
        """Start the control loop."""
        pass

    @abstractmethod
    def ensure_torque_enabled(self):
        """
        Re-enable torque / control mode for all motors without restarting control.
        Useful after comm glitches or resets.
        """
        pass

    @abstractmethod
    def close(self, disable_torque: bool = True):
        """
        Close the controller and release resources.

        Args:
            disable_torque: If True (default), torque is turned off during shutdown.
        """
        pass

    @property
    @abstractmethod
    def q(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the joint angles in radians and a mask of valid joints.

        Returns:
            Tuple of (joint_angles, valid_joints_mask)
        """
        pass

    @property
    @abstractmethod
    def v(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns joint velocities in rad/s and a mask of valid joints.

        Note: If some joints are not connected, the returned joint velocity
        at that index will be 0.

        Returns:
            Tuple of (joint_velocities, valid_joints_mask)
        """
        pass

    @property
    @abstractmethod
    def current(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the joint currents in mA and a mask of valid joints.

        Returns:
            Tuple of (joint_currents, valid_joints_mask)
        """
        pass

    @property
    @abstractmethod
    def dof(self) -> int:
        """
        Get total degrees of freedom (DOF) for this controller.

        Returns:
            int: Total number of joints controlled by this controller.

        Examples:
            - Single arm A1 (6 DOF): returns 6
            - Single arm R1Pro (7 DOF): returns 7
            - Bilateral R1Pro (2 arms, 7 DOF each): returns 14
        """
        pass

    @abstractmethod
    def has_reached_goal(self, block: bool = False, tolerance: float = 0.1) -> bool:
        """
        Check if the arm has reached the goal position.

        Args:
            block: If True, block until the goal is reached. If False, return immediately.
            tolerance: Maximum allowed distance from goal in radians for each joint.

        Returns:
            True if the arm is within tolerance of the goal, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def motor_groups(self) -> Sequence[GelloMotorGroup]:
        """
        Returns the motor groups of the controller.

        Returns:
            Sequence of GelloMotorGroup objects
        """
        pass

    @property
    @abstractmethod
    def position_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the joint position trajectory in radians.

        This is the underlying immediate goal position of the robot arm.

        Returns:
            Tuple of (position_trajectory, valid_joints_mask)
        """
        pass

    @property
    @abstractmethod
    def velocity_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the joint velocity trajectory in rad/s.

        This is the underlying immediate goal velocity of the robot arm.

        Returns:
            Tuple of (velocity_trajectory, valid_joints_mask)
        """
        pass

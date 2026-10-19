"""
Gello movement utilities - smooth movement and parking.

Extracted from GelloPolicy to enable reuse and testing.
"""

import time
import numpy as np
import logging

from iiil.gello.utils.joint_utils import expand_goal_to_full_dof

logger = logging.getLogger(__name__)


class GelloMovementHelper:
    """
    Helper class for Gello movement operations.

    Provides:
    - Smooth interpolation to target positions
    - Parking at home position
    - Convergence waiting
    """

    def __init__(
        self,
        gello_controller,
        convergence_tolerance: float = 0.1,
        max_wait_time: float = 10.0,
    ):
        """
        Initialize movement helper.

        Args:
            gello_controller: BilateralController instance
            convergence_tolerance: Position tolerance for convergence (rad)
            max_wait_time: Maximum wait time for convergence (seconds)
        """
        self._controller = gello_controller
        self._tolerance = convergence_tolerance
        self._max_wait = max_wait_time

    def move_smoothly(
        self,
        target_joints: np.ndarray,
        duration: float = 3.0,
        dt: float = 0.01,
    ) -> None:
        """
        Move Gello to target using linear interpolation.

        Blocks until movement complete.

        Args:
            target_joints: Target joint positions
            duration: Movement duration in seconds
            dt: Update interval in seconds
        """
        current_joints, _ = self._controller.q
        start_time = time.time()

        while time.time() - start_time < duration:
            alpha = (time.time() - start_time) / duration
            alpha = np.clip(alpha, 0.0, 1.0)

            interp_joints = current_joints * (1 - alpha) + target_joints * alpha
            self._controller.set_new_goal(interp_joints)
            time.sleep(dt)

        # Set final goal
        self._controller.set_new_goal(target_joints)

    def park(self, goal: np.ndarray) -> bool:
        """
        Drive Gello to goal and wait for convergence.

        Args:
            goal: Target position (partial or full DOF)

        Returns:
            True if converged, False if timeout or error
        """
        if goal is None:
            return True

        try:
            self._controller.start_control()
            self._controller.resume_goal_following()

            curr_pos, _ = self._controller.q
            full_goal = expand_goal_to_full_dof(goal, curr_pos, self._controller.dof)
            self._controller.set_new_goal(full_goal)

            return self._controller.has_reached_goal(
                block=True,
                tolerance=self._tolerance,
            )
        except Exception as e:
            logger.warning(f"Failed to park Gello: {e}")
            return False

    def wait_for_convergence(
        self,
        timeout: float = None,
        check_interval: float = 0.05,
    ) -> bool:
        """
        Wait for Gello to reach current goal.

        Args:
            timeout: Override default timeout (seconds)
            check_interval: How often to check convergence (seconds)

        Returns:
            True if converged, False if timeout
        """
        timeout = timeout or self._max_wait
        start_time = time.time()

        logger.info("Waiting for Gello joint angles to converge to robot goal...")
        self._controller.resume_goal_following()

        while time.time() - start_time < timeout:
            if self._controller.has_reached_goal(
                block=False,
                tolerance=self._tolerance,
            ):
                elapsed = time.time() - start_time
                logger.info(f"Converged in {elapsed:.2f}s")
                return True
            time.sleep(check_interval)

        logger.warning(f"Convergence timeout after {timeout:.2f}s. Proceeding anyway.")
        return False


__all__ = ["GelloMovementHelper"]

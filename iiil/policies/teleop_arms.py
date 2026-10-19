"""
Unified arm controller that handles any DOF count without branching.

This module provides DOF-agnostic arm control using slice-based operations.
Works seamlessly with 6, 7, 14, or any N DOF configurations.
"""
from typing import List
import torch
import numpy as np
import logging

from iiil.policies.teleop_types import ArmMapping

logger = logging.getLogger(__name__)


class ArmController:
    """
    Unified controller for a single arm that works for any DOF (6, 7, N).

    Key insight: Use slices from ArmMapping instead of DOF counts.
    This eliminates all branching based on DOF numbers.
    """

    def __init__(self, mapping: ArmMapping):
        """
        Initialize arm controller with a mapping.

        Args:
            mapping: ArmMapping defining robot ↔ gello joint correspondence
        """
        self.mapping = mapping

    def gello_to_robot_action(
        self,
        gello_joints: np.ndarray,
        gello_valid: np.ndarray,
        robot_proprio: torch.Tensor,
        action_template: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map gello joints to robot action space.

        For invalid gello joints, fallback to robot proprioception to avoid jumps.

        Args:
            gello_joints: Full gello joint positions (all arms)
            gello_valid: Bool mask indicating which gello joints are valid
            robot_proprio: Current robot proprioception
            action_template: Template action to populate

        Returns:
            Action tensor with this arm's slice populated from gello
        """
        action = action_template.clone()

        # Extract this arm's slice from gello
        gello_arm_joints = gello_joints[self.mapping.gello_slice]
        gello_arm_valid = gello_valid[self.mapping.gello_slice]

        # Map to robot action space
        robot_slice = self.mapping.robot_slice
        count = min(len(gello_arm_joints), robot_slice.stop - robot_slice.start)

        if count > 0:
            target_start = robot_slice.start
            target_end = target_start + count

            # Convert to torch
            gello_arm_joints_t = torch.from_numpy(gello_arm_joints[:count]).to(torch.float32)
            gello_arm_valid_t = torch.from_numpy(gello_arm_valid[:count])

            # Log data transfer
            num_valid_joints = gello_arm_valid_t.sum().item()
            logger.debug(f"        Transferring {int(num_valid_joints)}/{count} valid gello joints → robot action")
            if num_valid_joints < count:
                logger.debug(f"        Using robot proprio fallback for {count - int(num_valid_joints)} invalid joints")

            # Use gello for valid joints
            action[target_start:target_end] = torch.where(
                gello_arm_valid_t,
                gello_arm_joints_t,
                robot_proprio[target_start:target_end]  # Fallback to proprio
            )

        return action

    def robot_to_gello_goal(
        self,
        robot_proprio: torch.Tensor,
        gello_goal_array: np.ndarray,
    ) -> np.ndarray:
        """
        Map robot joints to gello goal space.

        Used in ENV_TO_GELLO mode to make gello follow robot.

        Args:
            robot_proprio: Current robot joint positions
            gello_goal_array: Gello goal array to populate (modified in-place)

        Returns:
            Modified gello goal array
        """
        robot_joints = robot_proprio[self.mapping.robot_slice]
        count = min(self.mapping.dof, len(robot_joints))

        if count > 0:
            gello_start = self.mapping.gello_slice.start
            robot_values = robot_joints[:count].cpu().numpy()
            gello_goal_array[gello_start:gello_start + count] = robot_values

            logger.debug(f"        Transferring {count} robot joints → gello goal")
            logger.debug(f"        Robot joints: [{robot_values[0]:.3f}, {robot_values[1]:.3f}, ...]")

        return gello_goal_array

    def policy_to_gello_goal(
        self,
        policy_action: torch.Tensor,
        gello_goal_array: np.ndarray,
    ) -> np.ndarray:
        """
        Map policy action (robot joint space) to gello goal space.

        Used in admittance mode to send policy predictions to GELLO.
        Similar to robot_to_gello_goal() but uses policy_action instead of proprio.

        Args:
            policy_action: Policy predicted action in robot joint space
            gello_goal_array: Gello goal array to populate (modified in-place)

        Returns:
            Modified gello goal array
        """
        # Extract robot joints from policy action
        policy_joints = policy_action[self.mapping.robot_slice]
        count = min(self.mapping.dof, len(policy_joints))

        if count > 0:
            gello_start = self.mapping.gello_slice.start
            policy_values = policy_joints[:count].cpu().numpy()
            gello_goal_array[gello_start:gello_start + count] = policy_values

            logger.debug(f"        Transferring {count} policy joints → gello goal")
            logger.debug(f"        Policy joints: [{policy_values[0]:.3f}, {policy_values[1]:.3f}, ...]")

        return gello_goal_array


class UnifiedArmController:
    """
    Manages multiple arms with unified DOF-agnostic interface.

    Automatically handles 1-arm, 2-arm, or N-arm configurations.
    Single-arm vs dual-arm is implicit based on number of ArmMapping objects.
    """

    def __init__(self, arm_controllers: List[ArmController]):
        """
        Initialize unified arm controller.

        Args:
            arm_controllers: List of ArmController instances (one per arm)
        """
        self.arms = arm_controllers
        logger.info(f"UnifiedArmController initialized with {len(self.arms)} arm(s)")
        for i, arm in enumerate(self.arms):
            logger.info(f"  Arm {i}: {arm.mapping.name} ({arm.mapping.dof} DOF)")

    def apply_gello_to_env(
        self,
        gello_joints: np.ndarray,
        gello_valid: np.ndarray,
        robot_proprio: torch.Tensor,
        action_template: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map gello state to robot action for all arms.

        Processes each arm sequentially, populating different slices of the action.

        Args:
            gello_joints: Full gello joint positions
            gello_valid: Bool mask for valid gello joints
            robot_proprio: Current robot proprioception
            action_template: Template action to start from

        Returns:
            Complete action with all arms populated
        """
        action = action_template.clone()
        logger.debug(f"    Processing {len(self.arms)} arm(s) for gello→env mapping...")

        for i, arm in enumerate(self.arms):
            logger.debug(f"      Arm {i} ({arm.mapping.name}): mapping gello[{arm.mapping.gello_slice}] → robot[{arm.mapping.robot_slice}]")
            action = arm.gello_to_robot_action(
                gello_joints, gello_valid, robot_proprio, action
            )

        return action

    def apply_env_to_gello(
        self,
        robot_proprio: torch.Tensor,
        gello_current_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Map robot state to gello goal for all arms.

        Used in ENV_TO_GELLO mode to sync gello with robot movements.

        Args:
            robot_proprio: Current robot joint positions
            gello_current_pos: Current gello positions (used as base)

        Returns:
            Complete gello goal with all arms populated
        """
        gello_goal = np.array(gello_current_pos, copy=True)
        logger.debug(f"    Processing {len(self.arms)} arm(s) for env→gello mapping...")

        for i, arm in enumerate(self.arms):
            logger.debug(f"      Arm {i} ({arm.mapping.name}): mapping robot[{arm.mapping.robot_slice}] → gello[{arm.mapping.gello_slice}]")
            gello_goal = arm.robot_to_gello_goal(robot_proprio, gello_goal)

        return gello_goal

    def apply_policy_to_gello(
        self,
        policy_action: torch.Tensor,
        gello_current_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Map policy action to gello goal for all arms.

        Used in admittance mode to send policy predictions to GELLO
        as low-stiffness goals.

        Args:
            policy_action: Policy predicted action in robot space
            gello_current_pos: Current gello positions (used as base)

        Returns:
            Complete gello goal with all arms populated from policy
        """
        gello_goal = np.array(gello_current_pos, copy=True)
        logger.debug(f"    Processing {len(self.arms)} arm(s) for policy→gello mapping...")

        for i, arm in enumerate(self.arms):
            logger.debug(f"      Arm {i} ({arm.mapping.name}): mapping policy[{arm.mapping.robot_slice}] → gello[{arm.mapping.gello_slice}]")
            gello_goal = arm.policy_to_gello_goal(policy_action, gello_goal)

        return gello_goal


__all__ = ["ArmController", "UnifiedArmController"]

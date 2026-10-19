"""
Pipeline orchestration for teleop control flow.

Connects hooks, modes, and arms into a clean, explicit control flow.
Replaces the implicit control flow in GelloPolicy.predict().
"""
import time
import logging

import numpy as np
import torch
from typing import Optional, TYPE_CHECKING

from iiil.policies.teleop_hooks import TeleopHookRegistry, GelloState, EnvState
from iiil.policies.teleop_modes import ControlModeManager
from iiil.policies.teleop_arms import UnifiedArmController

if TYPE_CHECKING:
    from iiil.policies.teleop_gating import AdmittanceGate

logger = logging.getLogger(__name__)


class TeleopPipeline:
    """
    Core teleop pipeline orchestrating hooks, modes, and arm control.

    This class provides the main control flow:
    1. Read and process state through hooks
    2. Sync env→gello or policy→gello depending on mode
    3. Create gello-derived action
    4. Apply gate if configured, otherwise let mode blend actions
    5. Return final action

    This replaces the implicit control flow previously scattered across
    GelloPolicy.predict() and helper methods.
    """

    def __init__(
        self,
        hook_registry: TeleopHookRegistry,
        mode_manager: ControlModeManager,
        arm_controller: UnifiedArmController,
        gello_controller,
        verbose: bool = False,
    ):
        """
        Initialize the teleop pipeline.

        Args:
            hook_registry: Registry for gello/env state hooks
            mode_manager: Manager for control modes
            arm_controller: Unified arm controller for all arms
            gello_controller: BilateralController hardware interface
            verbose: Enable verbose logging of pipeline steps
        """
        self.hooks = hook_registry
        self.modes = mode_manager
        self.arms = arm_controller
        self._gello_controller = gello_controller
        self._movement_helper = None  # Set externally via set_movement_helper()
        self._first_policy_sync = True
        self._blend_steps_remaining = 0  # Non-blocking blend counter
        self._blend_start_pos = None
        self._blend_target = None
        self._blend_total_steps = 0
        self._step_count = 0
        self._verbose = verbose

        # Optional gate for admittance mode (set externally)
        self._gate: Optional["AdmittanceGate"] = None

        # Gate state tracking for logging
        self._last_gate_state: Optional[str] = None  # "policy" or "gello" or None
        self._last_gate_decision: Optional[bool] = None  # is_correcting

        logger.info("=" * 80)
        logger.info("TeleopPipeline initialized")
        logger.info(f"  Active mode: {mode_manager.active_mode_name}")
        logger.info(f"  Number of arms: {len(arm_controller.arms)}")
        logger.info(f"  Arm mappings: {[(arm.mapping.name, arm.mapping.dof) for arm in arm_controller.arms]}")
        logger.info("=" * 80)

    def set_movement_helper(self, movement_helper) -> None:
        """Set the movement helper for smooth initial sync."""
        self._movement_helper = movement_helper

    def reset_first_sync(self) -> None:
        """Reset the first-sync flag so the next policy→gello sync is smooth."""
        self._first_policy_sync = True

    def set_gate(self, gate: "AdmittanceGate") -> None:
        """
        Set the admittance gate for this pipeline.

        When a gate is set, it will be used to decide between
        gello_action and policy_action in admittance mode.

        Args:
            gate: AdmittanceGate instance
        """
        self._gate = gate
        logger.info("[Pipeline] Admittance gate configured")

    def create_env_state(self, obs: dict[str, torch.Tensor]) -> EnvState:
        """
        Extract environment state from observation.

        Args:
            obs: Observation dictionary from environment

        Returns:
            EnvState snapshot
        """
        robot_proprio = obs.get("robot::proprio", torch.zeros(14))

        return EnvState(
            robot_proprio=robot_proprio,
            timestamp=time.time(),
        )

    def sync_env_to_gello(self, env_state: EnvState, gello_curr_pos: np.ndarray) -> None:
        """
        Sync robot position to gello goal.

        Only syncs if the active mode requires it (e.g., ENV_TO_GELLO mode).

        Args:
            env_state: Current environment state
            gello_curr_pos: Current gello position (from read_state)
        """
        if not self.modes.should_sync_env_to_gello():
            return

        logger.debug("    → Syncing robot position to gello goal (ENV_TO_GELLO active)")

        # Map robot proprio → gello goal
        gello_goal = self.arms.apply_env_to_gello(
            env_state.robot_proprio,
            gello_curr_pos,
        )

        # Validate the sync
        logger.debug(f"    → Gello current: [{gello_curr_pos[0]:.3f}, {gello_curr_pos[1]:.3f}, ...]")
        logger.debug(f"    → Gello goal:    [{gello_goal[0]:.3f}, {gello_goal[1]:.3f}, ...]")
        logger.debug(f"    → Setting new gello goal and resuming PD control")

        self._gello_controller.set_new_goal(gello_goal)
        self._gello_controller.resume_goal_following()

    def sync_policy_to_gello(self, policy_action: torch.Tensor, gello_curr_pos: np.ndarray) -> None:
        """
        Sync policy action to GELLO goal (inverse of sync_env_to_gello).

        Sends policy's predicted action to GELLO as a goal,
        which GELLO tracks with low stiffness (configurable Kp).

        Args:
            policy_action: Action predicted by base policy
            gello_curr_pos: Current gello position (from read_state)
        """
        if not self.modes.should_sync_policy_to_gello():
            return

        logger.debug("    → Syncing policy action to gello goal (admittance mode)")

        # Map policy action (robot space) → gello goal (gello space)
        gello_goal = self.arms.apply_policy_to_gello(
            policy_action,
            gello_curr_pos,
        )

        logger.debug(f"    → Policy action: [{policy_action[0:3].cpu().numpy()}...]")
        logger.debug(f"    → Gello goal:    [{gello_goal[0:3]}...]")

        if self._first_policy_sync:
            # Start non-blocking blend from current position to policy target
            # ~90 steps at 30Hz = 3 seconds of smooth interpolation
            logger.info("    → First policy→gello sync: starting smooth blend to policy target")
            self._blend_start_pos = np.array(gello_curr_pos, copy=True)
            self._blend_target = gello_goal
            self._blend_total_steps = 90
            self._blend_steps_remaining = self._blend_total_steps
            self._first_policy_sync = False
            # Set initial interpolated goal (alpha=0 → current position)
            self._gello_controller.set_new_goal(self._blend_start_pos)
            self._gello_controller.resume_goal_following()
        elif self._blend_steps_remaining > 0:
            # Continue blending toward the latest policy target
            self._blend_target = gello_goal
            alpha = 1.0 - (self._blend_steps_remaining / self._blend_total_steps)
            blended = self._blend_start_pos * (1 - alpha) + self._blend_target * alpha
            self._blend_steps_remaining -= 1
            self._gello_controller.set_new_goal(blended)
            self._gello_controller.resume_goal_following()
        else:
            logger.debug(f"    → Setting new gello goal with low-stiffness PD control")
            self._gello_controller.set_new_goal(gello_goal)
            self._gello_controller.resume_goal_following()

    def compute_action(
        self,
        obs: dict[str, torch.Tensor],
        policy_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Main pipeline: hooks → modes → action blending.

        This is the core method called from GelloPolicy.predict().

        Pipeline flow:
        1. Read env state → process through hooks
        2. Sync policy→gello OR env→gello depending on mode
        3. Read gello state and create gello-derived action
        4. Apply gate if configured, otherwise let mode compute action
        5. Return final action

        Args:
            obs: Observation dictionary from environment
            policy_action: Action from base policy

        Returns:
            Final blended action to send to environment
        """
        self._step_count += 1

        # Log at INFO level every 50 steps to show pipeline is working (only in verbose mode)
        if self._verbose and self._step_count % 50 == 1:
            logger.info(f"[Pipeline Step {self._step_count}] Active mode: {self.modes.active_mode_name}")

        logger.debug(f"[Step {self._step_count}] Pipeline starting | Mode: {self.modes.active_mode_name}")

        # 1. Get gello state (cached from control loop - NO USB read)
        logger.debug("  [1/4] Getting gello hardware state (cached)...")
        positions, velocities, _, valid = self._gello_controller.read_state(use_cache=True)

        gello_state = GelloState(
            joint_positions=positions,
            joint_velocities=velocities,
            valid_joints=valid,
            timestamp=time.time(),
        )

        num_gello_hooks = len(self.hooks._gello_hooks)
        if num_gello_hooks > 0:
            logger.debug(f"  [1/4] Processing through {num_gello_hooks} gello hooks...")
            gello_state = self.hooks.emit_gello_update(gello_state)

        # 2. Read and process env state through hooks
        logger.debug("  [2/4] Reading environment state...")
        env_state = self.create_env_state(obs)

        num_env_hooks = len(self.hooks._env_hooks)
        if num_env_hooks > 0:
            logger.debug(f"  [2/4] Processing through {num_env_hooks} env hooks...")
            env_state = self.hooks.emit_env_update(env_state)

        # 3. Sync policy→gello OR env→gello depending on mode
        logger.debug("  [3/4] Checking if policy→gello or env→gello sync needed...")
        if self.modes.should_sync_policy_to_gello():
            self.sync_policy_to_gello(policy_action, gello_state.joint_positions)
        elif self.modes.should_sync_env_to_gello():
            self.sync_env_to_gello(env_state, gello_state.joint_positions)
        else:
            logger.debug("    → No sync needed (GELLO_TO_ENV mode active)")

        # 4. Create gello-derived action
        logger.debug("  [4/4] Creating gello-derived action from hardware...")
        gello_action = self.arms.apply_gello_to_env(
            gello_state.joint_positions,
            gello_state.valid_joints,
            env_state.robot_proprio,
            policy_action,  # Use as template
        )

        # 5. Apply gate if configured, otherwise let mode compute action
        if self._gate is not None and self.modes.active_mode_name == 'admittance':
            # Use gate to decide between gello_action and policy_action
            final_action, is_correcting = self._gate.process(gello_action, policy_action)
            self._last_gate_state = self._gate.last_gate_state
            self._last_gate_decision = self._gate.last_gate_decision

            if is_correcting:
                logger.debug("  → Gate: Human correcting, using GELLO action")
            else:
                logger.debug("  → Gate: No correction, using policy action")

            logger.debug(f"[Step {self._step_count}] Pipeline complete (gated)\n")
            return final_action

        # No gate - let mode compute action
        self._last_gate_state = None
        self._last_gate_decision = None

        if self.modes.active_mode is None:
            logger.warning("  No active mode! Falling back to policy action")
            return policy_action

        logger.debug(f"  → Mode '{self.modes.active_mode_name}' computing final action...")
        final_action = self.modes.active_mode.compute_action(
            gello_state, env_state, gello_action, policy_action
        )

        logger.debug(f"[Step {self._step_count}] Pipeline complete → returning '{self.modes.active_mode_name}' action\n")
        return final_action


__all__ = ["TeleopPipeline"]

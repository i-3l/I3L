"""
GelloPolicy - Simplified orchestration layer for Gello teleoperation.

All complexity is delegated to specialized modules:
- GelloPolicyBuilder: Construction and configuration
- RobotTypeStrategy: Robot-specific behavior
- TeleopPipeline: Control flow orchestration
- InputHandler: Input processing
- GelloMovementHelper: Movement utilities
- DataCollectionState: State tracking
"""

import logging
from typing import Dict, Optional
import torch
import numpy as np

from omegaconf import DictConfig

from iiil.policies.policy_base import BasePolicy, PolicyMode
from iiil.policies.teleop_types import GelloControlDirection
from iiil.policies.teleop_builder import (
    GelloPolicyBuilder,
    GelloPolicyComponents,
    ImmediateComponents,
    create_base_policy,
)
from iiil.utils.audio import speak

logger = logging.getLogger(__name__)


class GelloPolicy(BasePolicy):
    """
    Gello teleop policy with clean separation of concerns.

    Supports three control modes:
    - GELLO_TO_ENV: User controls robot via Gello (pure teleoperation)
    - ENV_TO_GELLO: Policy controls robot, Gello follows
    - ADMITTANCE: Policy commands Gello with low stiffness, human can correct
    """
    # Enable trial control (X=start, A=accept, Y=reject)
    supports_trial_control: bool = True

    def __init__(self, config: DictConfig) -> None:
        from time import perf_counter
        super().__init__(config)

        self._config = config
        self._builder = GelloPolicyBuilder(config)

        # Build immediate components (don't need robot yet)
        self._immediate: ImmediateComponents = self._builder.build_immediate()

        # Deferred components (built in load)
        self._components: Optional[GelloPolicyComponents] = None
        self._base_policy: Optional[BasePolicy] = None

        # Create base policy early (but don't load it yet)
        self._base_policy = create_base_policy(self._immediate.base_policy_type)

        # Control direction state
        self._control_direction = self._immediate.initial_direction
        self._pending_env_to_gello_sync = False

        # input state
        self._input_state = None

        # Trial state for data collection
        self._trial_active = False

        # Debounce for episode end buttons (A/Y)
        # Period is set from environment frequency via set_debounce_period()
        self._debounce_period = 0.0  # No debounce until set from frequency
        self._last_episode_end_time = 0.0

        # Gripper takeover state: hold policy gripper on admittance entry,
        # switch to JoyCon only when user actually presses a gripper button.
        self._gripper_was_correcting: bool = False
        self._gripper_override: Dict[str, float] = {}  # per-side policy gripper to hold
        self._last_joycon_gripper: Dict[str, float] = {}  # track joycon values for change detection

        # Event callback for dashboard (fallback to print)
        self._event_callback = print


    def load(self, **kwargs) -> None:
        """
        Build deferred components and load base policy.

        This is where the robot is available, so we can build
        robot-specific components.
        """
        self.robot = kwargs["robot"]

        # Load base policy first
        self._base_policy.load(**kwargs)

        # Build complete component set
        self._components = self._builder.build_complete(
            immediate=self._immediate,
            robot=self.robot,
            base_policy=self._base_policy,
        )

        # Set up callbacks for non-admittance modes
        self._setup_takeover_callbacks()

        # Initialize control direction
        self._set_control_direction(self._control_direction, source="init")

        # Set initial arm controller mode
        self._components.robot_strategy.set_controller_mode(
            self.robot,
            self._components.base_policy_control_type
        )

        # Set up gate callbacks if gate is enabled
        if self._components.gate is not None:
            self._components.gate.set_on_release_callback(self._on_human_correction_end)
            self._components.gate.set_on_takeover_start_callback(self._on_human_correction_start)

    def predict(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Main prediction - delegates to pipeline.

        Flow:
        1. Read input state FIRST (so button checks work in main.py)
        2. Handle input-driven mode switching
        3. Start control loop if needed
        4. Get policy action based on mode
        5. Apply gripper from input
        6. Run pipeline to get final blended action
        7. Track data collection state
        """
        from time import perf_counter
        _profile = getattr(self, '_profile_predict', False)
        if _profile:
            _t = {'start': perf_counter()}

        c = self._components

        # Read input state FIRST - this must happen before any button checks
        # in main.py step() which are called after predict() returns
        self._input_state = c.input_handler.read_input_state()

        if _profile:
            _t['after_input'] = perf_counter()

        # Handle input-driven mode switching
        new_dir, changed = c.input_handler.update_control_direction(self._control_direction)
        if changed:
            self._set_control_direction(new_dir, source="joycon")

        # Start control if needed
        if not c.runtime_state.started_control:
            c.gello_controller.start_control()
            c.runtime_state.started_control = True
            # If starting in GELLO_TO_ENV mode, disable torque for backdrivability
            # (start_control enables torque, but GELLO_TO_ENV needs it disabled)
            if self._control_direction == GelloControlDirection.GELLO_TO_ENV:
                c.gello_controller.stop_goal_following()

        # Handle legacy reset flag
        if c.runtime_state.should_reset_gello_position:
            c.movement_helper.wait_for_convergence()
            if self._control_direction == GelloControlDirection.GELLO_TO_ENV:
                c.gello_controller.stop_goal_following()
            c.runtime_state.should_reset_gello_position = False

        # Read locomotion command EARLY so we can use it for input override detection
        locomotion_cmd = None
        if c.locomotion_handler is not None and c.locomotion_handler.connected:
            locomotion_cmd = c.locomotion_handler.get_command(obs)

        # Detect any JoyCon interaction (buttons + joystick movement) and set input override
        if c.gate is not None:
            any_input = False
            if self._input_state is not None:
                any_input = self._input_state.any_control_input
            if locomotion_cmd is not None:
                any_input = any_input or locomotion_cmd.has_active_input
            c.gate.set_input_override(any_input)

        # Get policy action based on mode
        policy_action = self._get_policy_action(obs, **kwargs)

        if _profile:
            _t['after_policy'] = perf_counter()

        # Handle pending sync after mode switch
        if self._pending_env_to_gello_sync and c.mode_manager.should_sync_env_to_gello():
            env_state = c.pipeline.create_env_state(obs)
            gello_pos, _, _, _ = c.pipeline._gello_controller.read_state(use_cache=True)
            c.pipeline.sync_env_to_gello(env_state, gello_pos)
            self._pending_env_to_gello_sync = False

        # Run pipeline to get arm action from Gello
        final_action = c.pipeline.compute_action(obs, policy_action)

        if _profile:
            _t['after_pipeline'] = perf_counter()

        # Apply locomotion (base, torso) - gripper handled separately below
        # In admittance mode when not correcting, use policy's base/torso instead of JoyCon
        use_policy_locomotion = (
            c.gate is not None and
            c.mode_manager.active_mode_name == "admittance" and
            not c.gate.is_currently_correcting()
        )

        joycon_gripper = None
        if locomotion_cmd is not None:
            # Extract gripper, let locomotion only apply base/torso
            joycon_gripper = locomotion_cmd.gripper_cmds
            locomotion_cmd.gripper_cmds = None
            if not use_policy_locomotion:
                # Only apply JoyCon locomotion when human is correcting or not in admittance
                final_action = c.locomotion_handler.apply_command_to_action(
                    final_action, locomotion_cmd, c.action_slices
                )

        # When policy controls locomotion, copy base/torso from policy_action
        if use_policy_locomotion:
            if "base" in c.action_slices:
                final_action[c.action_slices["base"]] = policy_action[c.action_slices["base"]]
            if "torso" in c.action_slices:
                final_action[c.action_slices["torso"]] = policy_action[c.action_slices["torso"]]

        # When no locomotion handler controls base/torso, zero them out.
        # The action template (from get_no_op_action) contains chassis *positions*,
        # but base_cmd is interpreted as *velocity* — so non-zero values cause movement.
        if not use_policy_locomotion and locomotion_cmd is None:
            if "base" in c.action_slices:
                final_action[c.action_slices["base"]] = 0.0
            if "torso" in c.action_slices:
                final_action[c.action_slices["torso"]] = 0.0

        if _profile:
            _t['after_locomotion'] = perf_counter()

        # Apply gripper based on mode and correction state
        is_correcting = c.gate is not None and c.gate.is_currently_correcting()
        in_admittance = (
            c.gate is not None and
            c.mode_manager.active_mode_name == "admittance"
        )

        if in_admittance and not is_correcting:
            # Policy controls gripper
            for gs in c.gripper_slices:
                final_action[gs] = policy_action[gs]
            self._gripper_was_correcting = False
            self._gripper_override.clear()
            self._last_joycon_gripper.clear()

        elif in_admittance and is_correcting:
            if joycon_gripper:
                if not self._gripper_was_correcting:
                    # Takeover entry: capture policy gripper as override, record joycon baseline
                    for side, sk in [("left", "left_gripper"), ("right", "right_gripper"), ("right", "gripper")]:
                        if side in joycon_gripper and sk in c.action_slices:
                            self._gripper_override[side] = policy_action[c.action_slices[sk]].item()
                            self._last_joycon_gripper[side] = joycon_gripper[side]
                    self._gripper_was_correcting = True

                # Detect joycon gripper change → user pressed button → drop override for that side.
                # A press with no recorded baseline (side absent last frame) also counts:
                # momentary triggers (Franka/A1) report a side only while held, so a fresh
                # press after release is user intent even if its value matches the old baseline.
                for side in list(self._gripper_override.keys()):
                    if side in joycon_gripper:
                        if (side not in self._last_joycon_gripper
                                or abs(joycon_gripper[side] - self._last_joycon_gripper[side]) > 0.01):
                            del self._gripper_override[side]

                # Update joycon tracking
                for side in joycon_gripper:
                    self._last_joycon_gripper[side] = joycon_gripper[side]

                # Apply: override (policy hold) or joycon (user took control)
                for side, sk in [("left", "left_gripper"), ("right", "right_gripper"), ("right", "gripper")]:
                    if side in joycon_gripper and sk in c.action_slices:
                        if side in self._gripper_override:
                            final_action[c.action_slices[sk]] = self._gripper_override[side]
                        else:
                            final_action[c.action_slices[sk]] = joycon_gripper[side]
            else:
                # Correcting without JoyCon gripper: hold policy gripper
                for gs in c.gripper_slices:
                    final_action[gs] = policy_action[gs]
                # Momentary triggers report nothing while released: forget the
                # baseline so the next press is a rising edge that drops the
                # override instead of being swallowed by a stale baseline match.
                self._last_joycon_gripper.clear()
            self._gripper_was_correcting = True

        elif joycon_gripper:
            # Not in admittance: direct JoyCon gripper
            if "left" in joycon_gripper and "left_gripper" in c.action_slices:
                final_action[c.action_slices["left_gripper"]] = joycon_gripper["left"]
            if "right" in joycon_gripper and "right_gripper" in c.action_slices:
                final_action[c.action_slices["right_gripper"]] = joycon_gripper["right"]
            # Single gripper (Franka/A1): map "right" → "gripper"
            if "right" in joycon_gripper and "gripper" in c.action_slices:
                final_action[c.action_slices["gripper"]] = joycon_gripper["right"]

        # Track data collection state
        if c.mode_manager.active_mode_name == "admittance":
            c.data_state.update_policy_target(policy_action)
            c.data_state.update_gello_actual(final_action)
            c.data_state.update_gate_state(
                c.pipeline._last_gate_state,
                c.pipeline._last_gate_decision,
            )

        self._update_controller_mode()
        c.data_state.update_input_state(self._input_state)

        if _profile:
            _t['end'] = perf_counter()
            if not hasattr(self, '_predict_count'):
                self._predict_count = 0
            self._predict_count += 1
            if self._predict_count % 50 == 0:
                input_t = (_t['after_input'] - _t['start']) * 1000
                policy_t = (_t['after_policy'] - _t['after_input']) * 1000
                pipeline_t = (_t['after_pipeline'] - _t['after_policy']) * 1000
                loco_t = (_t['after_locomotion'] - _t['after_pipeline']) * 1000
                rest_t = (_t['end'] - _t['after_locomotion']) * 1000
                total_t = (_t['end'] - _t['start']) * 1000
                logger.info(f"  [PREDICT] total={total_t:.1f}ms | input={input_t:.1f}ms | policy={policy_t:.1f}ms | pipeline={pipeline_t:.1f}ms | locomotion={loco_t:.1f}ms | rest={rest_t:.1f}ms")

        return final_action
    
    def check_block_ended(self) -> bool:
        """
        Check if Plus button was pressed to end entire data collection block.
        If pressed, moves robot and Gello to home position before returning.
        Returns:
            bool: True if the block has ended, False otherwise
        """
        if self._input_state and self._input_state.termination_pressed:
            # Move robot to home position before ending block
            if hasattr(self, 'robot') and hasattr(self.robot, 'move_to_home'):
                self._event_callback("[EXIT] Moving robot to home position...")
                self.robot.move_to_home()
                self._event_callback("[EXIT] Robot at home position.")
            # Move Gello to matching home position
            if hasattr(self, 'move_gello_to_home'):
                self._event_callback("[EXIT] Moving Gello to home position...")
                self.move_gello_to_home()
                self._event_callback("[EXIT] Gello at home position.")
            return True
        return False

    def check_episode_ended(self, success: bool = True) -> bool:
        """
        Check if A (accept) or Y (reject) was pressed to end episode.

        Only triggers when:
        1. Trial is active (recording in progress)
        2. Debounce period has passed since last episode end

        Args:
            success (bool): True to check for accept (A), False to check for reject (Y)
        Returns:
            bool: True if the episode has ended, False otherwise
        """
        from time import perf_counter

        if not self._input_state:
            return False

        # Only allow A/Y to end episode when trial is active
        if not self._trial_active:
            return False

        # Check button state
        if success:
            button_pressed = self._input_state.accept_pressed
        else:
            button_pressed = self._input_state.reject_pressed

        if not button_pressed:
            return False

        # Apply debounce - ignore if within debounce period
        current_time = perf_counter()
        if current_time - self._last_episode_end_time < self._debounce_period:
            return False

        # Record time of this episode end
        self._last_episode_end_time = current_time
        return True

    def check_start_trial_pressed(self) -> bool:
        """Check if X button is pressed."""
        if not self._input_state:
            return False
        return self._input_state.start_trial_pressed

    def check_home_requested(self) -> bool:
        """Check if Home button is pressed."""
        if not self._input_state:
            return False
        return self._input_state.home_pressed

    def refresh_input_state(self) -> None:
        """Refresh input state from hardware. Call before checking buttons."""
        if self._components:
            self._input_state = self._components.input_handler.read_input_state()

    def is_trial_active(self) -> bool:
        """
        Check if trial recording is currently active.

        If JoyCon is not available, always returns True (fallback to always record).
        This ensures data collection works even without button control.
        """
        # If JoyCon not available, always record (fallback behavior)
        if self._components and not self._components.input_handler.available:
            return True
        return self._trial_active

    def start_trial(self) -> None:
        """Mark trial as started (called when X pressed)."""
        self._trial_active = True
        callback = self._event_callback
        border = "=" * 70
        callback(f"\033[1m\033[92m{border}\033[0m\n\033[1m\033[92m  RECORDING STARTED\033[0m\n\033[1m\033[92m  Press A (accept) or Y (reject) to end trial\033[0m\n\033[1m\033[92m{border}\033[0m")
        speak("recording started")

    def stop_trial(self, success: bool = None) -> None:
        """
        Mark trial as stopped (called on episode end).

        Args:
            success: True if accepted (A button), False if rejected (Y button), None if unknown
        """
        self._trial_active = False
        callback = self._event_callback
        border = "=" * 70
        if success is True:
            callback(f"\033[1m\033[92m{border}\033[0m\n\033[1m\033[92m  RECORDING ACCEPTED\033[0m\n\033[1m\033[92m  Trial saved successfully\033[0m\n\033[1m\033[92m{border}\033[0m")
            speak("recording saved")
        elif success is False:
            callback(f"\033[1m\033[91m{border}\033[0m\n\033[1m\033[91m  RECORDING REJECTED\033[0m\n\033[1m\033[91m  Trial discarded\033[0m\n\033[1m\033[91m{border}\033[0m")
            speak("recording discarded")
        else:
            callback(f"\033[1m\033[93m{border}\033[0m\n\033[1m\033[93m  RECORDING STOPPED\033[0m\n\033[1m\033[93m{border}\033[0m")

    def move_gello_to_home(self) -> None:
        """Move Gello to home position (called when Home button pressed)."""
        c = self._components

        # Compute gello home position
        current_joints, _ = c.gello_controller.q
        robot_home = self.robot.get_home_position() if hasattr(self.robot, 'get_home_position') else torch.zeros(self.robot.n_dof)
        robot_proprio = torch.cat([robot_home, torch.zeros_like(robot_home)])
        gello_home = c.pipeline.arms.apply_env_to_gello(robot_proprio, current_joints)

        # Set goal to current position FIRST to prevent snap on mode switch
        c.gello_controller.set_new_goal(current_joints)
        if hasattr(c.gello_controller, 'ensure_torque_enabled'):
            c.gello_controller.ensure_torque_enabled()
        c.gello_controller.resume_goal_following()
        # Now safe to switch to position control — goal matches current pose
        if hasattr(c.gello_controller, 'switch_to_position_control'):
            c.gello_controller.switch_to_position_control()

        # Move Gello smoothly to home position (blocks until done)
        c.movement_helper.move_smoothly(gello_home)

        if self._control_direction == GelloControlDirection.GELLO_TO_ENV:
            # Switch back to current control for backdrivable teleop
            if hasattr(c.gello_controller, 'switch_to_current_control'):
                c.gello_controller.switch_to_current_control()
            c.gello_controller.stop_goal_following()
            c.mode_manager.switch_mode("gello_to_env")
        else:
            # Admittance: restore the exact state that a takeover-release produces
            # so the Gello follows the policy on the next tick.
            c.gello_controller.set_new_goal(gello_home)
            if c.gate is not None:
                c.gate.clear_paused_policy_action()
                if c.gate._stiffness_switching_enabled:
                    c.gate._apply_gains(c.gate._normal_gains)
                if c.gate.config.use_position_control:
                    c.gello_controller.switch_to_position_control()
            c.mode_manager.switch_mode(self._control_direction.value)
            self._base_policy.reset()

        # Reset takeover detector AFTER move so the motion doesn't trigger correction
        if hasattr(c.gello_controller, '_takeover_detector'):
            c.gello_controller._takeover_detector.reset()
        if c.gate is not None:
            c.gate._was_correcting = False
            c.gate._input_override = False
            c.gate._interpolating = False
            c.gate._interp_start_action = None
            c.gate._last_gate_state = "policy"
            c.gate._last_gate_decision = False
        self._gripper_was_correcting = False
        self._gripper_override.clear()
        self._last_joycon_gripper.clear()

        self._event_callback("[HOME] Gello ready for control.")

    def _get_policy_action(self, obs, **kwargs) -> torch.Tensor:
        """Get action from base policy based on current mode."""
        c = self._components
        mode = c.mode_manager.active_mode_name

        if mode == "gello_to_env":
            # User in direct control - skip expensive replanning
            return self.robot.get_no_op_action()

        if mode == "admittance" and c.gate is not None:
            # Check if human is correcting and we should freeze trajectory
            should_freeze = (
                c.gate.is_currently_correcting()
                and not c.gate.config.predict_during_correction
            )
            if should_freeze:
                paused = c.gate.get_paused_policy_action()
                if paused is not None:
                    return paused
                return self.robot.get_no_op_action()

        # Normal case: call base policy
        return c.base_policy.predict(obs=obs, **kwargs)

    def _set_control_direction(self, direction: GelloControlDirection, source: str = "") -> None:
        """Centralized mode switching."""
        c = self._components

        prev_direction = self._control_direction
        if prev_direction == direction and c.mode_manager.active_mode_name == direction.value:
            return

        self._control_direction = direction
        c.mode_manager.switch_mode(direction.value)

        # Configure tracking behavior based on mode
        stop_on_takeover = direction != GelloControlDirection.ADMITTANCE
        c.gello_controller.set_stop_tracking_on_takeover(stop_on_takeover)

        # Handle ENV_TO_GELLO switch
        if direction == GelloControlDirection.ENV_TO_GELLO:
            if prev_direction != direction:
                self._reset_base_policy_for_follow_mode()
            self._pending_env_to_gello_sync = True
        else:
            self._pending_env_to_gello_sync = False

        if source:
            logger.debug(f"[Mode] Control direction set to {direction.value} (source={source})")

    def _update_controller_mode(self) -> None:
        """Update robot controller using strategy pattern."""
        c = self._components
        mode = c.mode_manager.active_mode_name

        if mode in ("gello_to_env", "admittance"):
            c.robot_strategy.set_controller_mode(self.robot, "joint")
        else:
            c.robot_strategy.set_controller_mode(
                self.robot,
                c.base_policy_control_type
            )

    def _setup_takeover_callbacks(self) -> None:
        """Set up takeover callbacks for non-admittance modes."""
        c = self._components

        # Skip takeover callbacks entirely when takeover is disabled by policy config
        takeover_override = self._config.get("takeover_enable", None)
        if takeover_override is False:
            logger.info("Takeover callbacks disabled (policy.takeover_enable=false)")
            if hasattr(c.gello_controller, '_takeover_detector'):
                c.gello_controller._takeover_detector.disable()
            return

        if self._control_direction != GelloControlDirection.ADMITTANCE:
            def on_takeover():
                logger.debug("[Takeover] Human takeover detected, switching to GELLO_TO_ENV mode")
                self._set_control_direction(GelloControlDirection.GELLO_TO_ENV, source="takeover")

            def on_auto_return():
                logger.debug("[AutoReturn] Human released control, switching to ENV_TO_GELLO mode")
                self._set_control_direction(GelloControlDirection.ENV_TO_GELLO, source="auto_return")

            c.gello_controller.set_takeover_callback(on_takeover)
            c.gello_controller.set_auto_return_callback(on_auto_return)
        else:
            logger.info("Takeover detection disabled for admittance mode")

    def _reset_base_policy_for_follow_mode(self) -> None:
        """Drop any queued plan when returning to follow mode."""
        try:
            self._base_policy.reset()
        except Exception as exc:
            logger.warning(f"Failed to reset base policy on mode switch: {exc}")

    def _on_human_correction_end(self) -> None:
        """Called when human stops correcting in admittance mode."""
        try:
            logger.debug("[GelloPolicy] Human released control, resetting base policy for replan")
            self._base_policy.reset()
        except Exception as exc:
            logger.warning(f"Failed to reset base policy on correction end: {exc}")

    def _on_human_correction_start(self) -> None:
        """Called when human starts correcting in admittance mode."""
        c = self._components
        if c.gate is None:
            return
        if not hasattr(self._base_policy, 'action_buffer'):
            return

        try:
            if self._base_policy.action_buffer.shape[0] > 0:
                paused_action = self._base_policy.action_buffer[0].clone()
                c.gate.set_paused_policy_action(paused_action)
                logger.debug("[GelloPolicy] Captured paused trajectory point at takeover")
            else:
                logger.warning("[GelloPolicy] No trajectory point to capture (action_buffer empty)")
        except Exception as exc:
            logger.warning(f"Failed to capture trajectory point on correction start: {exc}")

    def set_mode(self, policy_mode: PolicyMode) -> None:
        """Set policy mode (LEADING or FOLLOWING)."""
        if policy_mode == PolicyMode.LEADING:
            self._set_control_direction(GelloControlDirection.GELLO_TO_ENV, source="policy_mode")
        elif policy_mode == PolicyMode.FOLLOWING:
            self._set_control_direction(GelloControlDirection.ENV_TO_GELLO, source="policy_mode")
        else:
            raise ValueError(f"Invalid policy mode: {policy_mode}")

    def on_env_reset(self, obs: dict[str, torch.Tensor]) -> None:
        """Handle environment reset."""
        c = self._components
        c.runtime_state.ready_for_trial = False

        # Initialize input state so first step() can check buttons
        self._input_state = c.input_handler.read_input_state()

        # Enable/disable takeover detection for new episode based on policy config
        takeover_override = self._config.get("takeover_enable", None)
        if hasattr(c.gello_controller, '_takeover_detector'):
            if takeover_override is False:
                c.gello_controller._takeover_detector.disable()
            elif takeover_override is True:
                c.gello_controller._takeover_detector.config.enable = True
                c.gello_controller._takeover_detector.enable()
            else:
                c.gello_controller._takeover_detector.enable()

        # Switch to appropriate mode
        c.mode_manager.switch_mode(self._control_direction.value)

        # Start control if needed
        if not c.runtime_state.started_control:
            c.gello_controller.start_control()
            c.runtime_state.started_control = True
            # If starting in GELLO_TO_ENV mode, disable torque for backdrivability
            if self._control_direction == GelloControlDirection.GELLO_TO_ENV:
                c.gello_controller.stop_goal_following()

        # Only ensure torque enabled for modes that need active control
        if self._control_direction != GelloControlDirection.GELLO_TO_ENV:
            if hasattr(c.gello_controller, "ensure_torque_enabled"):
                c.gello_controller.ensure_torque_enabled()

        # Sync Gello to robot position for ENV_TO_GELLO and ADMITTANCE modes
        if self._control_direction in (GelloControlDirection.ENV_TO_GELLO, GelloControlDirection.ADMITTANCE):
            if "robot::proprio" in obs:
                gello_curr_pos, _ = c.gello_controller.q
                # Map robot's current position to gello space (handles non-zero homes like Franka)
                gello_goal = c.pipeline.arms.apply_env_to_gello(
                    obs["robot::proprio"], gello_curr_pos
                )
                c.runtime_state.last_reset_goal = gello_goal
                c.movement_helper.move_smoothly(gello_goal, duration=5.0)
                c.runtime_state.should_reset_gello_position = False

        # GELLO_TO_ENV: move Gello to home so user starts at the right position
        if self._control_direction == GelloControlDirection.GELLO_TO_ENV:
            self.move_gello_to_home()

        c.pipeline.reset_first_sync()
        c.runtime_state.ready_for_trial = True

        # Log status based on JoyCon availability
        if not c.input_handler.available:
            logger.info("JoyCon not available - recording all data (no trial control)")

    def ready_for_trial(self) -> bool:
        """Whether device has finished aligning after env reset."""
        return self._components.runtime_state.ready_for_trial

    def set_event_callback(self, fn) -> None:
        """Set callback for dashboard event logging. Propagates to subcomponents."""
        self._event_callback = fn
        c = self._components
        if c is not None:
            if c.gate is not None and hasattr(c.gate, 'set_event_callback'):
                c.gate.set_event_callback(fn)
            if hasattr(c.gello_controller, 'set_event_callback'):
                c.gello_controller.set_event_callback(fn)

    def set_debounce_period(self, period: float) -> None:
        """
        Set the debounce period for A/Y buttons.

        Args:
            period: Debounce period in seconds (typically 1/frequency)
        """
        self._debounce_period = period

    def reset(self) -> None:
        """Reset policy state."""
        c = self._components

        # Disable takeover detection between experiments
        if hasattr(c.gello_controller, '_takeover_detector'):
            c.gello_controller._takeover_detector.disable()

        self._base_policy.reset()
        c.data_state.reset()
        c.runtime_state.should_reset_gello_position = False
        c.runtime_state.ready_for_trial = False
        self._input_state = None
        self._trial_active = False  # Reset trial state
        self._last_episode_end_time = 0.0  # Reset debounce timer
        self._gripper_was_correcting = False
        self._gripper_override.clear()
        self._last_joycon_gripper.clear()

    def close(self) -> None:
        """Clean up resources."""
        c = self._components

        # Disable takeover detection
        if hasattr(c.gello_controller, '_takeover_detector'):
            c.gello_controller._takeover_detector.disable()

        self._base_policy.close()

        # Clean up locomotion handler
        if c.locomotion_handler is not None:
            c.locomotion_handler.close()

        # Zero torque and disable motors on shutdown
        try:
            c.gello_controller.close(disable_torque=True)
        except TypeError:
            c.gello_controller.close()

    def get_current_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return state for data collection wrapper."""
        c = self._components

        base_state = None
        if hasattr(self._base_policy, 'get_current_state'):
            base_state = self._base_policy.get_current_state()

        return c.data_state.to_dict(
            mode_name=c.mode_manager.active_mode_name,
            include_base_policy=base_state,
        )

    # Expose joycon for compatibility
    @property
    def joycon(self):
        """Access JoyCon controller for compatibility."""
        return self._components.input_handler.joycon if self._components else self._immediate.input_handler.joycon

    # Expose attributes for compatibility with tests and external code
    @property
    def _num_arm_joints(self) -> int:
        """Number of arm joints (for compatibility)."""
        return self._immediate.gello_arm_dof

    @property
    def _base_policy_control_type(self) -> str:
        """Base policy control type (for compatibility)."""
        if self._components is not None:
            return self._components.base_policy_control_type
        return "joint"  # Default

    @property
    def _arm_controller(self):
        """Arm controller reference (for compatibility)."""
        if self._components is not None:
            return self._components.robot_strategy.get_arm_controller(self.robot)
        return None

    @_arm_controller.setter
    def _arm_controller(self, value):
        """Allow setting arm controller for tests."""
        # This is a compatibility shim for tests that set _arm_controller directly
        pass


__all__ = ["GelloPolicy"]

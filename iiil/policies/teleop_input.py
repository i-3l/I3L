"""
Input handling for teleop policies.

Abstracts JoyCon, gripper, and other input sources into a unified interface.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
import torch
import logging

from iiil.policies.teleop_types import GelloControlDirection
from iiil.policies.teleop_switches import JoyconSwitchState, update_control_direction_from_inputs

logger = logging.getLogger(__name__)


@dataclass
class InputState:
    """Current state of all inputs for data collection control."""
    gripper_close: bool = False
    gripper_open: bool = False
    mode_switch_pressed: bool = False    # Minus (-) button - toggle GELLO↔ENV
    start_trial_pressed: bool = False    # X button - start trial
    accept_pressed: bool = False         # A button - accept + save trial
    reject_pressed: bool = False         # Y button - reject + discard trial
    termination_pressed: bool = False    # Plus (+) button - end block
    home_pressed: bool = False           # Home button - move to home position

    @property
    def any_button_pressed(self) -> bool:
        """True if any JoyCon button or trigger is pressed."""
        return (
            self.gripper_close or self.gripper_open
            or self.mode_switch_pressed or self.start_trial_pressed
            or self.accept_pressed or self.reject_pressed
            or self.termination_pressed or self.home_pressed
        )

    @property
    def any_control_input(self) -> bool:
        """True if any physical control input is active (gripper triggers).

        Excludes trial-management buttons (X, A, Y, Plus, Home, Minus)
        which should not trigger takeover detection.
        """
        return self.gripper_close or self.gripper_open


class InputHandler:
    """
    Unified input handling for teleop policies.

    Abstracts JoyCon specifics and provides clean interface for:
    - Mode switching (X button)
    - Future input sources

    Note: Gripper control is now handled by LocomotionHandler.
    """

    def __init__(self, joycon_required: bool = True):
        """
        Initialize input handler.

        JoyCon connection is deferred until connect() is called,
        because robot_type is not known at init time.

        Args:
            joycon_required: If True, raise error when JoyCon not connected.
                           If False, disable input gracefully.
        """
        self._joycon_required = joycon_required
        self._joycon: Optional[Any] = None
        self._available = False
        self._switch_state = JoyconSwitchState()

    def connect(self, robot_type: str) -> None:
        """
        Connect to JoyCon for the given robot type.

        Called by GelloPolicyBuilder.build_complete() when robot_type is known.
        Uses the shared JoyCon singleton from joycon_utils.

        Args:
            robot_type: Robot type string (e.g., "A1", "R1Pro")
        """
        try:
            from iiil.utils.joycon_utils import get_joycon_interface
            self._joycon = get_joycon_interface(robot_type, required=self._joycon_required)
            self._available = self._joycon.connected

            if not self._available and self._joycon_required:
                raise RuntimeError("JoyCon is required but could not be initialized.")
            if not self._available:
                logger.warning("JoyCon inputs disabled; button switching will be ignored.")
        except ImportError as e:
            if self._joycon_required:
                raise RuntimeError("joycon_utils module required but not found") from e
            logger.warning("JoyCon module not available - input disabled")
        except Exception as e:
            if self._joycon_required:
                raise
            logger.warning(f"JoyCon connection failed: {e}")

    @property
    def available(self) -> bool:
        """Whether input device is connected and healthy."""
        if not self._available or self._joycon is None:
            return False
        return self._joycon.connected and self._joycon.is_healthy

    @property
    def joycon(self):
        """Access underlying JoyCon controller (for compatibility)."""
        return self._joycon

    def read_input_state(self) -> InputState:
        """
        Read current input state.

        Returns:
            InputState with current button/trigger states
        """
        if not self._available:
            return InputState()

        return InputState(
            gripper_close=self._joycon.trigger_r,
            gripper_open=self._joycon.trigger_zr,
            mode_switch_pressed=self._joycon.button_minus,   # Minus for mode switch
            start_trial_pressed=self._joycon.button_x,       # X for start trial
            accept_pressed=self._joycon.button_a,            # A for accept (save)
            reject_pressed=self._joycon.button_y,            # Y for reject (discard)
            termination_pressed=self._joycon.button_plus,    # Plus to end data collection
            home_pressed=self._joycon.button_home or self._joycon.button_b,  # Home or B to move to home
        )

    def update_control_direction(
        self,
        current: GelloControlDirection,
    ) -> Tuple[GelloControlDirection, bool]:
        """
        Check for mode switch input.

        Args:
            current: Current control direction

        Returns:
            Tuple of (new_direction, changed)
        """
        new_direction, self._switch_state = update_control_direction_from_inputs(
            current,
            self._joycon,
            self._switch_state,
            self._available,
        )

        return new_direction, new_direction != current

    def apply_gripper_input(
        self,
        action: torch.Tensor,
        gripper_slices: List[slice],
    ) -> torch.Tensor:
        """
        Apply gripper input to action tensor.

        Note: Gripper control is now handled by LocomotionHandler.
        This method is kept for backward compatibility but does nothing.

        Args:
            action: Action tensor to modify
            gripper_slices: List of slices for gripper dimensions

        Returns:
            Unmodified action tensor
        """
        # Gripper now handled by LocomotionHandler
        return action


__all__ = ["InputState", "InputHandler"]

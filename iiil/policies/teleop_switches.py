from dataclasses import dataclass
from typing import Tuple

from iiil.policies.teleop_types import GelloControlDirection


@dataclass
class JoyconSwitchState:
    """State tracker for JoyCon button presses."""
    button_minus: bool = False  # Minus button for mode switching


def update_control_direction_from_inputs(
    current_direction: GelloControlDirection,
    joycon,
    state: JoyconSwitchState,
    joycon_available: bool,
) -> Tuple[GelloControlDirection, JoyconSwitchState]:
    """
    Update control direction based on JoyCon buttons.

    Simplified to 2 modes:
    - Minus (-) button toggles between GELLO_TO_ENV and ENV_TO_GELLO

    Args:
        current_direction: Current control direction
        joycon: JoyCon controller instance
        state: Button state tracker
        joycon_available: Whether JoyCon is connected

    Returns:
        Tuple of (new_direction, updated_state)
    """
    if not joycon_available:
        return current_direction, state

    new_direction = current_direction

    # Minus button: toggle between the two modes
    latest_minus = joycon.button_minus
    if latest_minus and latest_minus != state.button_minus:
        if current_direction == GelloControlDirection.ENV_TO_GELLO:
            new_direction = GelloControlDirection.GELLO_TO_ENV
            print("[MODE] Switched to GELLO_TO_ENV mode", flush=True)
        elif current_direction == GelloControlDirection.GELLO_TO_ENV:
            new_direction = GelloControlDirection.ENV_TO_GELLO
            print("[MODE] Switched to ENV_TO_GELLO mode", flush=True)

    state.button_minus = latest_minus

    return new_direction, state


__all__ = ["JoyconSwitchState", "update_control_direction_from_inputs"]

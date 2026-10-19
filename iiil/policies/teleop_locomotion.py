"""
Locomotion handling for teleop policies.

Provides a modular, robot-agnostic interface for base, torso, and gripper control
via external input devices (JoyCons, etc.).

Architecture:
- BaseLocomotionHandler: Abstract interface
- NoOpLocomotionHandler: For robots without locomotion (A1)
- R1ProLocomotionHandler: JoyCon-based control for R1Pro

This follows the same pattern as RobotTypeStrategy - each robot type
can have its own locomotion handler.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class LocomotionCommand:
    """
    Container for locomotion commands.

    All fields are optional - handlers only set what they control.
    """
    base_cmd: Optional[np.ndarray] = None      # (3,) for x, y, yaw
    torso_cmd: Optional[np.ndarray] = None     # Robot-specific DOF
    gripper_cmds: Optional[Dict[str, float]] = None  # {"left": float, "right": float}
    has_active_input: bool = False  # True if any joystick/d-pad is active

    @property
    def has_base(self) -> bool:
        return self.base_cmd is not None

    @property
    def has_torso(self) -> bool:
        return self.torso_cmd is not None

    @property
    def has_grippers(self) -> bool:
        return self.gripper_cmds is not None


class BaseLocomotionHandler(ABC):
    """
    Abstract base class for locomotion handling.

    Each robot type can have its own implementation that handles
    base movement, torso control, and gripper input from external devices.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Handler identifier."""
        pass

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the input device is connected."""
        pass

    @abstractmethod
    def get_command(self, obs: Dict[str, torch.Tensor]) -> LocomotionCommand:
        """
        Get locomotion commands from input device.

        Args:
            obs: Current observation dict (may contain proprio for torso feedback)

        Returns:
            LocomotionCommand with any applicable commands
        """
        pass

    @abstractmethod
    def apply_to_action(
        self,
        action: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        action_slices: Dict[str, slice],
    ) -> torch.Tensor:
        """
        Apply locomotion commands to action tensor.

        Args:
            action: Action tensor to modify
            obs: Current observation dict
            action_slices: Dict mapping component names to slices

        Returns:
            Modified action tensor
        """
        pass

    def apply_command_to_action(
        self,
        action: torch.Tensor,
        cmd: LocomotionCommand,
        action_slices: Dict[str, slice],
    ) -> torch.Tensor:
        """
        Apply a pre-computed command to action tensor.

        This avoids re-reading input devices when the command is already available.
        Default implementation calls apply_to_action; subclasses can override for efficiency.

        Args:
            action: Action tensor to modify
            cmd: Pre-computed locomotion command
            action_slices: Dict mapping component names to slices

        Returns:
            Modified action tensor
        """
        # Default implementation - subclasses should override for efficiency
        return self.apply_to_action(action, {}, action_slices)

    def close(self) -> None:
        """Clean up resources."""
        pass


class NoOpLocomotionHandler(BaseLocomotionHandler):
    """
    No-op handler for robots without external locomotion control.

    Used for robots where base/torso/gripper are not controlled via JoyCon.
    """

    def __init__(self, required: bool = True, **kwargs):
        """Accept same args as other handlers for factory compatibility."""
        pass  # No-op

    @property
    def name(self) -> str:
        return "noop"

    @property
    def connected(self) -> bool:
        return True  # Always "connected" since it's a no-op

    def get_command(self, obs: Dict[str, torch.Tensor]) -> LocomotionCommand:
        return LocomotionCommand()  # Empty command

    def apply_to_action(
        self,
        action: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        action_slices: Dict[str, slice],
    ) -> torch.Tensor:
        return action  # No modification

    def apply_command_to_action(
        self,
        action: torch.Tensor,
        cmd: LocomotionCommand,
        action_slices: Dict[str, slice],
    ) -> torch.Tensor:
        return action  # No modification


class A1LocomotionHandler(BaseLocomotionHandler):
    """
    A1 locomotion handler - JoyCon gripper control only.

    A1 has no mobile base or torso, so this handler only manages
    gripper control via JoyCon triggers (R = close, ZR = open).

    Uses the shared JoyCon singleton from joycon_utils.
    """

    GRIPPER_SLICE = "gripper"  # A1 has single gripper

    def __init__(self, required: bool = True, **kwargs):
        """
        Initialize A1 locomotion handler.

        Args:
            required: If True, raise error when JoyCon not found
            **kwargs: Passed to JoyCon interface (for compatibility)
        """
        self._interface = None
        self._connected = False

        try:
            from iiil.utils.joycon_utils import get_joycon_interface
            self._interface = get_joycon_interface("A1", required=required)
            self._connected = self._interface.connected

            if self._connected:
                logger.info("A1 locomotion handler initialized (JoyCon connected)")
            else:
                logger.warning("A1 locomotion handler: JoyCon not connected")

        except ImportError as e:
            if required:
                raise RuntimeError("joycon_utils module required but not found") from e
            logger.warning("A1 locomotion handler: JoyCon module not available")
        except Exception as e:
            if required:
                raise
            logger.warning(f"A1 locomotion handler initialization failed: {e}")

    @property
    def name(self) -> str:
        return "a1_joycon"

    @property
    def connected(self) -> bool:
        return self._connected

    def get_command(self, obs: Dict[str, torch.Tensor]) -> LocomotionCommand:
        """Get gripper command from JoyCon."""
        if not self._connected or self._interface is None:
            return LocomotionCommand()

        # Check health - return empty command on disconnect
        if not self._interface.is_healthy:
            if self._interface.is_reconnecting:
                logger.debug("Joy-Con reconnecting, returning empty command")
            return LocomotionCommand()

        try:
            gripper = self._interface.get_gripper_command()
        except Exception as e:
            logger.warning(f"JoyCon read failed: {e}")
            self._connected = False
            return LocomotionCommand()

        # A1 has single gripper, map to "right" for consistency with gripper_slices
        gripper_val = gripper.get("gripper", 0.0)
        if gripper_val == 0.0:
            return LocomotionCommand()  # No button pressed

        return LocomotionCommand(
            gripper_cmds={"right": gripper_val}
        )

    def apply_to_action(
        self,
        action: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        action_slices: Dict[str, slice],
    ) -> torch.Tensor:
        """Apply JoyCon gripper command to A1 action tensor."""
        if not self._connected:
            return action

        cmd = self.get_command(obs)
        return self.apply_command_to_action(action, cmd, action_slices)

    def apply_command_to_action(
        self,
        action: torch.Tensor,
        cmd: LocomotionCommand,
        action_slices: Dict[str, slice],
    ) -> torch.Tensor:
        """
        Apply pre-computed gripper command to A1 action tensor.

        Only modifies action if a gripper button is pressed (value != 0).
        """
        if not self._connected:
            return action

        if cmd.has_grippers and self.GRIPPER_SLICE in action_slices:
            val = cmd.gripper_cmds.get("right", 0.0)
            if val != 0.0:  # Only apply if button pressed
                action[action_slices[self.GRIPPER_SLICE]] = val

        return action

    def close(self) -> None:
        """Clean up - don't close shared singleton."""
        # Don't close the shared JoyCon interface
        self._connected = False


class R1ProLocomotionHandler(BaseLocomotionHandler):
    """
    JoyCon-based locomotion handler for R1Pro robot.

    Controls:
    - Base: Left joystick (XY translation or yaw rotation)
    - Torso: Right joystick (pitch/yaw) + D-pad (stand/squat presets)
    - Grippers: ZL (left) / ZR (right)

    Matches brs-ctrl's R1ProJoyConInterface functionality.
    """

    # Action slice names for R1Pro
    BASE_SLICE = "base"
    TORSO_SLICE = "torso"
    LEFT_GRIPPER_SLICE = "left_gripper"
    RIGHT_GRIPPER_SLICE = "right_gripper"

    def __init__(self, required: bool = True, **kwargs):
        """
        Initialize R1Pro locomotion handler.

        Args:
            required: If True, raise error when JoyCons not found
            **kwargs: Passed to R1ProJoyConInterface
        """
        self._interface = None
        self._connected = False

        try:
            from iiil.utils.joycon_utils import get_joycon_interface
            self._interface = get_joycon_interface("R1Pro", required=required)
            self._connected = self._interface.connected

            if self._connected:
                logger.info("R1Pro locomotion handler initialized (JoyCon connected)")
            else:
                logger.warning("R1Pro locomotion handler: JoyCon not connected")

        except ImportError as e:
            if required:
                raise RuntimeError("joycon_utils module required but not found") from e
            logger.warning("R1Pro locomotion handler: JoyCon module not available")
        except Exception as e:
            if required:
                raise
            logger.warning(f"R1Pro locomotion handler initialization failed: {e}")

    @property
    def name(self) -> str:
        return "r1pro_joycon"

    @property
    def connected(self) -> bool:
        return self._connected

    def get_command(self, obs: Dict[str, torch.Tensor]) -> LocomotionCommand:
        """Get locomotion commands from JoyCon."""
        if not self._connected or self._interface is None:
            return LocomotionCommand()

        # Check health - hold position on disconnect
        if not self._interface.is_healthy:
            if self._interface.is_reconnecting:
                logger.debug("Joy-Con reconnecting, holding position")
            return self._create_hold_position_command(obs)

        # Extract current torso position from proprio
        robot_proprio = obs.get("robot::proprio", None)
        if robot_proprio is None:
            logger.debug("No robot::proprio in obs, using zero torso")
            current_torso = np.zeros(4)
        else:
            # R1Pro proprio: [base(3), torso(4), ...]
            proprio_np = robot_proprio.cpu().numpy()
            if len(proprio_np) < 7:
                logger.warning(
                    f"robot::proprio too short ({len(proprio_np)} < 7), "
                    "using zero torso. Expected R1Pro format: [base(3), torso(4), ...]"
                )
                current_torso = np.zeros(4)
            else:
                current_torso = proprio_np[3:7]

        # Get commands from JoyCon interface with error handling for disconnection
        try:
            joycon_cmd = self._interface.act(current_torso)
        except Exception as e:
            logger.warning(f"JoyCon read failed (possibly disconnected): {e}")
            self._connected = False
            return LocomotionCommand()

        torso_cmd_arr = np.array(joycon_cmd["torso_cmd"], dtype=np.float32)

        # Convert gripper from brs-ctrl [0.1, 1.0] to IIIL [-1, 1]
        gripper_cmds = None
        if "gripper_cmd" in joycon_cmd:
            brs_grippers = joycon_cmd["gripper_cmd"]
            gripper_cmds = {
                "left": brs_grippers["left"] * 2.0 - 1.0,
                "right": brs_grippers["right"] * 2.0 - 1.0,
            }

        # Detect joystick/d-pad activity
        base_cmd_arr = np.array(joycon_cmd["mobile_base_cmd"], dtype=np.float32)
        base_active = np.any(np.abs(base_cmd_arr) > 0.01)
        torso_active = np.any(np.abs(torso_cmd_arr - current_torso) > 0.01)

        return LocomotionCommand(
            base_cmd=base_cmd_arr,
            torso_cmd=torso_cmd_arr,
            gripper_cmds=gripper_cmds,
            has_active_input=bool(base_active or torso_active),
        )

    def _create_hold_position_command(self, obs: Dict[str, torch.Tensor]) -> LocomotionCommand:
        """Create command to hold current position when Joy-Con disconnects."""
        robot_proprio = obs.get("robot::proprio", None)
        if robot_proprio is None:
            return LocomotionCommand(
                base_cmd=np.zeros(3, dtype=np.float32),
                torso_cmd=np.zeros(4, dtype=np.float32),
            )

        proprio_np = robot_proprio.cpu().numpy()
        current_torso = proprio_np[3:7] if len(proprio_np) >= 7 else np.zeros(4)
        return LocomotionCommand(
            base_cmd=np.zeros(3, dtype=np.float32),  # No base movement
            torso_cmd=current_torso.astype(np.float32),  # Hold torso position
            gripper_cmds=None,  # Keep current gripper state
        )

    def apply_to_action(
        self,
        action: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        action_slices: Dict[str, slice],
    ) -> torch.Tensor:
        """Apply JoyCon commands to R1Pro action tensor."""
        if not self._connected:
            return action

        cmd = self.get_command(obs)
        return self.apply_command_to_action(action, cmd, action_slices)

    def apply_command_to_action(
        self,
        action: torch.Tensor,
        cmd: LocomotionCommand,
        action_slices: Dict[str, slice],
    ) -> torch.Tensor:
        """
        Apply pre-computed command to R1Pro action tensor.

        This is the efficient path when command is already available.
        """
        if not self._connected:
            return action

        # Apply base command
        if cmd.has_base and self.BASE_SLICE in action_slices:
            action[action_slices[self.BASE_SLICE]] = torch.from_numpy(cmd.base_cmd)

        # Apply torso command
        if cmd.has_torso and self.TORSO_SLICE in action_slices:
            action[action_slices[self.TORSO_SLICE]] = torch.from_numpy(cmd.torso_cmd)

        # Apply gripper commands
        if cmd.has_grippers:
            if self.LEFT_GRIPPER_SLICE in action_slices:
                action[action_slices[self.LEFT_GRIPPER_SLICE]] = cmd.gripper_cmds["left"]
            if self.RIGHT_GRIPPER_SLICE in action_slices:
                action[action_slices[self.RIGHT_GRIPPER_SLICE]] = cmd.gripper_cmds["right"]

        return action

    def close(self) -> None:
        """Clean up JoyCon connections."""
        if self._interface is not None:
            self._interface.close()
            self._interface = None
        self._connected = False


class FrankaLocomotionHandler(BaseLocomotionHandler):
    """
    Franka locomotion handler - JoyCon gripper control only.

    Franka Panda has no mobile base or torso, so this handler only manages
    gripper control via JoyCon triggers (R = close, ZR = open).

    Identical to A1LocomotionHandler but uses "Franka" robot type for
    JoyCon interface initialization.
    """

    GRIPPER_SLICE = "gripper"

    def __init__(self, required: bool = True, **kwargs):
        self._interface = None
        self._connected = False

        try:
            from iiil.utils.joycon_utils import get_joycon_interface
            # Use "A1" interface type (single JoyCon, single gripper)
            self._interface = get_joycon_interface("A1", required=required)
            self._connected = self._interface.connected

            if self._connected:
                logger.info("Franka locomotion handler initialized (JoyCon connected)")
            else:
                logger.warning("Franka locomotion handler: JoyCon not connected")

        except ImportError as e:
            if required:
                raise RuntimeError("joycon_utils module required but not found") from e
            logger.warning("Franka locomotion handler: JoyCon module not available")
        except Exception as e:
            if required:
                raise
            logger.warning(f"Franka locomotion handler initialization failed: {e}")

    @property
    def name(self) -> str:
        return "franka_joycon"

    @property
    def connected(self) -> bool:
        return self._connected

    def get_command(self, obs: Dict[str, torch.Tensor]) -> LocomotionCommand:
        """Get gripper command from JoyCon."""
        if not self._connected or self._interface is None:
            return LocomotionCommand()

        if not self._interface.is_healthy:
            if self._interface.is_reconnecting:
                logger.debug("Joy-Con reconnecting, returning empty command")
            return LocomotionCommand()

        try:
            gripper = self._interface.get_gripper_command()
        except Exception as e:
            logger.warning(f"JoyCon read failed: {e}")
            self._connected = False
            return LocomotionCommand()

        gripper_val = gripper.get("gripper", 0.0)
        if gripper_val == 0.0:
            return LocomotionCommand()

        return LocomotionCommand(
            gripper_cmds={"right": gripper_val}
        )

    def apply_to_action(
        self,
        action: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        action_slices: Dict[str, slice],
    ) -> torch.Tensor:
        if not self._connected:
            return action
        cmd = self.get_command(obs)
        return self.apply_command_to_action(action, cmd, action_slices)

    def apply_command_to_action(
        self,
        action: torch.Tensor,
        cmd: LocomotionCommand,
        action_slices: Dict[str, slice],
    ) -> torch.Tensor:
        if not self._connected:
            return action

        if cmd.has_grippers and self.GRIPPER_SLICE in action_slices:
            val = cmd.gripper_cmds.get("right", 0.0)
            if val != 0.0:
                action[action_slices[self.GRIPPER_SLICE]] = val

        return action

    def close(self) -> None:
        self._connected = False


# Registry of locomotion handlers by robot type
_LOCOMOTION_HANDLERS: Dict[str, type] = {
    "A1": A1LocomotionHandler,
    "R1Pro": R1ProLocomotionHandler,
    "Franka": FrankaLocomotionHandler,
}


def get_locomotion_handler(
    robot_type: str,
    required: bool = True,
    **kwargs,
) -> BaseLocomotionHandler:
    """
    Factory function to get appropriate locomotion handler.

    Args:
        robot_type: Robot type string (e.g., "A1", "R1Pro")
        required: If True, raise error when input device not found
        **kwargs: Passed to handler constructor

    Returns:
        Appropriate locomotion handler instance
    """
    if robot_type not in _LOCOMOTION_HANDLERS:
        logger.info(f"No locomotion handler for robot type '{robot_type}', using NoOp")
        return NoOpLocomotionHandler()

    handler_class = _LOCOMOTION_HANDLERS[robot_type]
    return handler_class(required=required, **kwargs)


def register_locomotion_handler(robot_type: str, handler_class: type) -> None:
    """
    Register a new locomotion handler for a robot type.

    Args:
        robot_type: Robot type string
        handler_class: Handler class (must inherit from BaseLocomotionHandler)
    """
    if not issubclass(handler_class, BaseLocomotionHandler):
        raise TypeError(f"Handler must inherit from BaseLocomotionHandler")
    _LOCOMOTION_HANDLERS[robot_type] = handler_class
    logger.info(f"Registered locomotion handler '{handler_class.__name__}' for '{robot_type}'")


__all__ = [
    "LocomotionCommand",
    "BaseLocomotionHandler",
    "NoOpLocomotionHandler",
    "A1LocomotionHandler",
    "R1ProLocomotionHandler",
    "FrankaLocomotionHandler",
    "get_locomotion_handler",
    "register_locomotion_handler",
]

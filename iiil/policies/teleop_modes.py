"""
Pluggable control modes for teleop pipeline.

Control modes define how gello and policy actions are blended.
New modes can be added by subclassing ControlMode and registering with ControlModeManager.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
import logging

from iiil.policies.teleop_hooks import GelloState, EnvState

logger = logging.getLogger(__name__)


class ControlMode(ABC):
    """
    Abstract base class for control modes.

    A control mode defines:
    1. How to blend gello and policy actions
    2. Whether to sync environment state to gello
    3. Lifecycle hooks when entering/exiting the mode
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this control mode."""
        pass

    @abstractmethod
    def compute_action(
        self,
        gello_state: GelloState,
        env_state: EnvState,
        gello_action: torch.Tensor,
        policy_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend gello and policy actions according to this mode's strategy.

        Args:
            gello_state: Current gello hardware state
            env_state: Current environment/robot state
            gello_action: Action derived from gello positions
            policy_action: Action from the base policy

        Returns:
            Final blended action to send to the environment
        """
        pass

    @abstractmethod
    def on_enter(self, gello_controller) -> None:
        """
        Called when switching TO this mode.

        Args:
            gello_controller: BilateralController instance for hardware control
        """
        pass

    @abstractmethod
    def on_exit(self, gello_controller) -> None:
        """
        Called when switching FROM this mode.

        Args:
            gello_controller: BilateralController instance for hardware control
        """
        pass

    def should_sync_env_to_gello(self) -> bool:
        """
        Whether this mode requires syncing robot pose to gello goal.

        Returns:
            True if robot state should be synced to gello, False otherwise
        """
        return False


class GelloToEnvMode(ControlMode):
    """
    GELLO_TO_ENV mode - pure teleoperation where user controls robot via GELLO.

    In this mode:
    - GELLO floats free (no goal tracking)
    - User moves GELLO directly
    - Robot follows GELLO's actual position with 1:1 tracking
    - No policy involvement
    """

    @property
    def name(self) -> str:
        return "gello_to_env"

    def compute_action(
        self,
        gello_state: GelloState,
        env_state: EnvState,
        gello_action: torch.Tensor,
        policy_action: torch.Tensor,
    ) -> torch.Tensor:
        """Robot follows GELLO directly."""
        return gello_action

    def on_enter(self, gello_controller) -> None:
        """Stop goal tracking - GELLO floats free for human control."""
        logger.info(f"[MODE SWITCH] Entering {self.name} mode")
        logger.info("  → Switching to current control (gravity comp for backdriving)")
        gello_controller.switch_to_current_control()
        logger.info("  → Stopping GELLO goal tracking (free float for user control)")
        gello_controller.stop_goal_following()

    def on_exit(self, gello_controller) -> None:
        """Nothing to clean up."""
        logger.info(f"[MODE SWITCH] Exiting {self.name} mode")

    def should_sync_env_to_gello(self) -> bool:
        """No syncing - user moves GELLO freely."""
        return False

    def should_sync_policy_to_gello(self) -> bool:
        """No policy sync."""
        return False


class EnvToGelloMode(ControlMode):
    """
    ENV_TO_GELLO mode - policy/robot drives GELLO to follow.

    In this mode:
    - Robot follows policy directly
    - GELLO tracks robot position (goal tracking active)
    - Useful for demonstrating policy execution to user
    """

    def __init__(self, use_position_control: bool = True):
        self._use_position_control = use_position_control

    @property
    def name(self) -> str:
        return "env_to_gello"

    def compute_action(
        self,
        gello_state: GelloState,
        env_state: EnvState,
        gello_action: torch.Tensor,
        policy_action: torch.Tensor,
    ) -> torch.Tensor:
        """Robot follows policy directly."""
        logger.debug(f"      → {self.name}: robot follows policy action")
        return policy_action

    def on_enter(self, gello_controller) -> None:
        """Start goal tracking - GELLO follows robot position."""
        logger.info(f"[MODE SWITCH] Entering {self.name} mode")
        if self._use_position_control:
            logger.info("  → Switching to position control (hardware PID for stiff tracking)")
            gello_controller.switch_to_position_control()
        logger.info("  → Starting GELLO goal tracking (follows robot position)")

        # Set initial goal to current position
        curr_positions, _ = gello_controller.q
        gello_controller.set_new_goal(curr_positions)
        gello_controller.resume_goal_following()

        # Ensure tracking is enabled
        if hasattr(gello_controller, "_goal_tracker"):
            gello_controller._goal_tracker.start_tracking()
        if hasattr(gello_controller, "_should_follow_goal"):
            gello_controller._should_follow_goal = True

    def on_exit(self, gello_controller) -> None:
        """Stop tracking when exiting."""
        logger.info(f"[MODE SWITCH] Exiting {self.name} mode")
        gello_controller.stop_goal_following()

    def should_sync_env_to_gello(self) -> bool:
        """Sync robot position → GELLO goal."""
        return True

    def should_sync_policy_to_gello(self) -> bool:
        """Use env sync, not policy sync."""
        return False


class AdmittanceMode(ControlMode):
    """
    Admittance mode - policy commands GELLO, robot follows GELLO's actual position.

    In this mode:
    - Base policy generates target actions
    - Policy actions are sent to GELLO as goal (with low stiffness)
    - Human can freely push/guide GELLO
    - Robot follows GELLO's actual position with 1:1 tracking
    - No mode switching on human intervention
    """

    def __init__(self, use_position_control: bool = True):
        self._use_position_control = use_position_control

    @property
    def name(self) -> str:
        return "admittance"

    def compute_action(
        self,
        gello_state: GelloState,
        env_state: EnvState,
        gello_action: torch.Tensor,
        policy_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Use gello_action (GELLO's actual position after human correction).
        Policy action was already sent to GELLO as goal in the pipeline.
        """
        logger.debug(f"      → {self.name}: robot follows gello actual position")
        return gello_action

    def on_enter(self, gello_controller) -> None:
        """Start goal tracking when entering this mode."""
        logger.info(f"[MODE SWITCH] Entering {self.name} mode")
        if self._use_position_control:
            logger.info("  → Switching to position control (hardware PID for policy tracking)")
            gello_controller.switch_to_position_control()
        logger.info("  → Starting GELLO goal tracking with low stiffness")

        # Set initial goal to current position
        curr_positions, _ = gello_controller.q
        gello_controller.set_new_goal(curr_positions)
        gello_controller.resume_goal_following()

        # Ensure tracking is enabled
        if hasattr(gello_controller, "_goal_tracker"):
            gello_controller._goal_tracker.start_tracking()
        if hasattr(gello_controller, "_should_follow_goal"):
            gello_controller._should_follow_goal = True

    def on_exit(self, gello_controller) -> None:
        """Stop tracking when exiting."""
        logger.info(f"[MODE SWITCH] Exiting {self.name} mode")
        gello_controller.stop_goal_following()

    def should_sync_policy_to_gello(self) -> bool:
        """
        New method: whether to sync policy action to GELLO goal.
        Returns True for this mode.
        """
        return True


class ControlModeManager:
    """
    Manages control mode registration and switching.

    Handles:
    - Registering custom control modes
    - Switching between modes with proper lifecycle hooks
    - Tracking active mode
    """

    def __init__(self, gello_controller):
        """
        Initialize the mode manager.

        Args:
            gello_controller: BilateralController instance for hardware control
        """
        self._modes: Dict[str, ControlMode] = {}
        self._active_mode: Optional[ControlMode] = None
        self._gello_controller = gello_controller

    def register_mode(self, mode: ControlMode) -> None:
        """
        Register a control mode.

        Args:
            mode: ControlMode instance to register

        Raises:
            ValueError: If a mode with the same name is already registered
        """
        if mode.name in self._modes:
            raise ValueError(f"Mode '{mode.name}' is already registered")
        self._modes[mode.name] = mode
        logger.info(f"Registered control mode: {mode.name}")

    def switch_mode(self, mode_name: str) -> None:
        """
        Switch to a different control mode.

        Args:
            mode_name: Name of the mode to switch to

        Raises:
            ValueError: If mode_name is not a registered mode
        """
        if mode_name not in self._modes:
            raise ValueError(f"Unknown mode: {mode_name}. Available modes: {list(self._modes.keys())}")

        new_mode = self._modes[mode_name]

        # Skip if already in this mode
        if self._active_mode == new_mode:
            logger.debug(f"Already in {mode_name} mode, skipping switch")
            return

        # Exit old mode, enter new mode
        old_mode_name = self._active_mode.name if self._active_mode else "None"
        logger.info("=" * 80)
        logger.info(f"MODE TRANSITION: {old_mode_name} → {mode_name}")
        logger.info("=" * 80)

        if self._active_mode is not None:
            self._active_mode.on_exit(self._gello_controller)

        new_mode.on_enter(self._gello_controller)
        self._active_mode = new_mode

    @property
    def active_mode(self) -> Optional[ControlMode]:
        """Get the currently active control mode."""
        return self._active_mode

    @property
    def active_mode_name(self) -> Optional[str]:
        """Get the name of the currently active control mode."""
        return self._active_mode.name if self._active_mode else None

    def should_sync_env_to_gello(self) -> bool:
        """
        Check if current mode requires syncing robot to gello.

        Returns:
            True if syncing is needed, False otherwise
        """
        if self._active_mode is None:
            return False
        return self._active_mode.should_sync_env_to_gello()

    def should_sync_policy_to_gello(self) -> bool:
        """
        Check if current mode requires syncing policy action to GELLO goal.

        This is the inverse of should_sync_env_to_gello() - it sends
        policy predictions to GELLO instead of robot state.

        Returns:
            True if policy→gello syncing is needed, False otherwise
        """
        if self._active_mode is None:
            return False
        # Check if mode has this method (new modes only)
        if hasattr(self._active_mode, 'should_sync_policy_to_gello'):
            return self._active_mode.should_sync_policy_to_gello()
        return False


__all__ = ["ControlMode", "GelloToEnvMode", "EnvToGelloMode", "AdmittanceMode", "ControlModeManager"]

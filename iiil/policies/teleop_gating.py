"""
Optional gating for admittance mode - detects human corrections.

This module provides configurable gating that uses the existing TakeoverDetector
from BilateralController to detect when a human is correcting the robot.

When gating is enabled:
- During correction: Robot follows GELLO actual position (human's correction)
- Not correcting: Robot follows policy action directly

When gating is disabled (default):
- Robot always follows GELLO actual position (pure admittance)
"""

from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import time
import torch
import logging

from iiil.utils.audio import speak

logger = logging.getLogger(__name__)


@dataclass
class GatingConfig:
    """Configuration for admittance gating."""
    enabled: bool = False  # Default: gating disabled (robot always follows GELLO)
    replan_on_release: bool = True  # Reset policy when human releases control
    predict_during_correction: bool = False  # If True, policy keeps predicting during correction (not frozen)
    takeover_transition_duration: float = 0.0  # Seconds to blend from policy to gello on takeover (0.0 = instant snap)
    use_position_control: bool = True  # When false, skip position/current mode switching (stay in current control)


class AdmittanceGate:
    """
    Detects human corrections and gates between policy/gello actions.

    Uses existing TakeoverDetector from BilateralController - preserves
    all existing thresholds and detection logic.
    """

    def __init__(self, config: GatingConfig, gello_controller):
        """
        Initialize the admittance gate.

        Args:
            config: Gating configuration
            gello_controller: BilateralController instance for hardware access
        """
        self.config = config
        self._gello_controller = gello_controller
        self._was_correcting = False
        self._on_release_callback: Optional[Callable] = None
        self._on_takeover_start_callback: Optional[Callable] = None
        self._paused_policy_action: Optional[torch.Tensor] = None
        self._last_gate_state: Optional[str] = None
        self._last_gate_decision: Optional[bool] = None
        self._input_override: bool = False
        self._normal_gains: Optional[dict] = None
        self._takeover_gains: Optional[dict] = None
        self._stiffness_switching_enabled: bool = False
        self._event_callback = print
        # Smooth interpolation state for takeover entry
        self._interpolating: bool = False
        self._interp_start_time: float = 0.0
        self._interp_start_action: Optional[torch.Tensor] = None

    def set_event_callback(self, fn) -> None:
        """Set callback for dashboard event logging."""
        self._event_callback = fn

    def set_on_release_callback(self, callback: Callable) -> None:
        """
        Register callback for when human releases control.

        Args:
            callback: Function to call when human stops correcting
        """
        self._on_release_callback = callback

    def set_on_takeover_start_callback(self, callback: Callable) -> None:
        """
        Register callback for when human starts correcting.

        Args:
            callback: Function to call when human starts correcting
        """
        self._on_takeover_start_callback = callback

    def is_currently_correcting(self) -> bool:
        """
        Check if human is currently correcting (without side effects).

        Returns:
            True if human is currently correcting, False otherwise
        """
        return self._was_correcting

    def set_paused_policy_action(self, action: torch.Tensor) -> None:
        """
        Store the policy action to use during takeover.

        Args:
            action: The trajectory point to freeze at
        """
        self._paused_policy_action = action.clone().detach()

    def get_paused_policy_action(self) -> Optional[torch.Tensor]:
        """
        Get the stored paused policy action.

        Returns:
            The paused policy action, or None if not set
        """
        return self._paused_policy_action

    def clear_paused_policy_action(self) -> None:
        """Clear the stored paused policy action."""
        self._paused_policy_action = None

    def set_input_override(self, any_input_active: bool) -> None:
        """Set external input override for takeover. True = any JoyCon interaction detected."""
        self._input_override = any_input_active

    def set_stiffness_configs(self, normal_gains, takeover_gains):
        """Configure gain switching between normal tracking and takeover."""
        self._normal_gains = normal_gains
        self._takeover_gains = takeover_gains
        self._stiffness_switching_enabled = (takeover_gains is not None and normal_gains is not None)
        callback = getattr(self, '_event_callback', print)
        if self._stiffness_switching_enabled:
            callback(f"\033[32m[Gate] Stiffness switching enabled — normal: {normal_gains}, takeover: {takeover_gains}\033[0m")
        else:
            callback(f"\033[31m[Gate] Stiffness switching DISABLED — normal={normal_gains is not None}, takeover={takeover_gains is not None}\033[0m")

    def _apply_gains(self, gains):
        if gains is None:
            return
        self._gello_controller.set_tracking_gains(
            kp=gains.get("kp"), ki=gains.get("ki"), kd=gains.get("kd"),
            kff=gains.get("kff"),
            reset_integral=True,
        )

    def _snap_goal_to_current(self):
        """Set GELLO goal to its current position to prevent snap-back on release."""
        try:
            current_pos, _ = self._gello_controller.q
            self._gello_controller.set_new_goal(current_pos)
            logger.info("[Gate] Snapped GELLO goal to current position")
        except Exception as exc:
            logger.warning(f"[Gate] Failed to snap goal to current: {exc}")

    def detect_correction(self) -> bool:
        """
        Check if human is correcting using existing takeover detector.

        Reads the last detection result from the 100Hz control loop rather
        than calling detect_takeover() again, which would corrupt the
        detector's internal filter/buffer state (EMA, Butterworth, rolling
        window) by advancing them at an irregular ~20Hz on top of the
        100Hz loop.

        Returns:
            True if human is actively correcting, False otherwise
        """
        if self._input_override:
            return True
        return self._gello_controller.is_takeover_detected

    def process(
        self,
        gello_action: torch.Tensor,
        policy_action: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """
        Process actions through gate.

        Args:
            gello_action: Action from GELLO actual position
            policy_action: Action from base policy

        Returns:
            Tuple of (final_action, is_correcting)
        """
        if not self.config.enabled:
            self._last_gate_state = None
            self._last_gate_decision = None
            return gello_action, False  # Always use gello when gating disabled

        is_correcting = self.detect_correction()

        # Store for logging
        self._last_gate_decision = is_correcting
        self._last_gate_state = "gello" if is_correcting else "policy"

        # Handle state transitions: not correcting -> correcting (START)
        if not self._was_correcting and is_correcting:
            logger.debug("[Gate] Human started correction - capturing paused point")
            speak("takeover started")
            if self._on_takeover_start_callback:
                self._on_takeover_start_callback()
            # Apply soft gains BEFORE switching mode so the first current-mode
            # iteration uses takeover gains (not stiff gains + stale integral).
            if self._stiffness_switching_enabled:
                logger.debug(f"[Gate] Applied gains: takeover (soft)")
                self._apply_gains(self._takeover_gains)
            # Switch to current control for compliance during human correction
            if self.config.use_position_control:
                logger.debug("[Gate] Switching to current control (compliance for correction)")
                self._gello_controller.switch_to_current_control()
            # Start smooth interpolation if configured
            if self.config.takeover_transition_duration > 0.0:
                self._interpolating = True
                self._interp_start_action = None
                self._interp_start_time = 0.0
                logger.debug(
                    f"[Gate] Smooth takeover transition enabled: "
                    f"{self.config.takeover_transition_duration:.2f}s interpolation"
                )

        # Handle state transitions: correcting -> not correcting (END)
        if self._was_correcting and not is_correcting:
            logger.debug("[Gate] Human released control")
            speak("takeover ended")
            self.clear_paused_policy_action()
            self._interpolating = False
            self._interp_start_action = None
            if self._on_release_callback and self.config.replan_on_release:
                self._on_release_callback()
            # Snap GELLO goal to current position BEFORE restoring stiff gains.
            # Without this, GELLO would track the stale policy goal with full
            # stiffness, causing the arms to snap back to the pre-takeover position.
            self._snap_goal_to_current()
            if self._stiffness_switching_enabled:
                logger.debug(f"[Gate] Applied gains: normal (stiff)")
                self._apply_gains(self._normal_gains)
            # Switch back to position control for stiff policy tracking
            if self.config.use_position_control:
                logger.debug("[Gate] Switching to position control (hardware PID for tracking)")
                self._gello_controller.switch_to_position_control()

        # On the release step, use gello_action (current position) instead of
        # the stale policy_action that was computed before the policy was reset.
        just_released = self._was_correcting and not is_correcting
        self._was_correcting = is_correcting

        if is_correcting:
            if self._interpolating:
                # First frame: capture policy_action as interpolation start
                if self._interp_start_action is None:
                    self._interp_start_action = policy_action.clone()
                    self._interp_start_time = time.time()
                    logger.debug("[Gate] Takeover interpolation started")
                    return self._interp_start_action, True

                elapsed = time.time() - self._interp_start_time
                alpha = min(elapsed / self.config.takeover_transition_duration, 1.0)

                if alpha >= 1.0:
                    self._interpolating = False
                    self._interp_start_action = None
                    logger.debug("[Gate] Takeover interpolation complete, now tracking gello directly")
                    return gello_action, True

                blended = self._interp_start_action * (1.0 - alpha) + gello_action * alpha
                logger.debug(
                    f"[Gate] Takeover interpolating "
                    f"(alpha={alpha:.3f}, {elapsed:.3f}s / {self.config.takeover_transition_duration:.3f}s)"
                )
                return blended, True
            return gello_action, True
        elif just_released:
            return gello_action, False
        else:
            return policy_action, False

    @property
    def last_gate_state(self) -> Optional[str]:
        """Get the last gate state for logging."""
        return self._last_gate_state

    @property
    def last_gate_decision(self) -> Optional[bool]:
        """Get the last gate decision for logging."""
        return self._last_gate_decision


__all__ = ["GatingConfig", "AdmittanceGate"]

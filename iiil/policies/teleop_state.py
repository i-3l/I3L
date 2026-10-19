"""
Data collection state tracking for teleop policies.

Tracks policy targets, Gello actuals, and correction state for DAgger training.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time
import torch


@dataclass
class DataCollectionState:
    """
    Tracks data collection state for admittance mode.

    Stores policy predictions vs Gello actuals for DAgger training.
    HDF5 output uses training-pipeline field names: base_action, oracle_action, is_oracle_active.
    """

    # Latest values
    policy_target: Optional[torch.Tensor] = None
    gello_actual: Optional[torch.Tensor] = None
    timestamp: Optional[float] = None
    input_state: Optional[torch.Tensor] = None

    # Gate state (if gating enabled)
    gate_state: Optional[str] = None  # "policy" or "gello"
    gate_decision: Optional[bool] = None  # is_correcting

    def update_policy_target(self, action: torch.Tensor) -> None:
        """Record policy's predicted action."""
        self.policy_target = action.clone().detach()
        self.timestamp = time.time()

    def update_gello_actual(self, action: torch.Tensor) -> None:
        """Record Gello's actual position (after human correction)."""
        self.gello_actual = action.clone().detach()

    def update_gate_state(self, state: Optional[str], decision: Optional[bool]) -> None:
        """Record gate state for logging."""
        self.gate_state = state
        self.gate_decision = decision

    def update_input_state(self, input_state) -> None:
        """Record input state for logging."""
        self.input_state = torch.tensor([input_state.start_trial_pressed], dtype=torch.bool)

    def to_dict(self, mode_name: str = None, include_base_policy: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Convert to dictionary for HDF5 logging.

        Output keys match training pipeline expectations:
        - base_action (from policy_target)
        - oracle_action (from gello_actual)
        - is_oracle_active (from gate_decision)

        Args:
            mode_name: Current mode name to include
            include_base_policy: Optional base policy state to include

        Returns:
            State dictionary or None if no data
        """
        if self.policy_target is None:
            return {"input_state": self.input_state}

        is_oracle_active = bool(self.gate_decision) if self.gate_decision is not None else False

        state = {
            "base_action": self.policy_target,
            "oracle_action": self.gello_actual,
            "is_oracle_active": torch.tensor(is_oracle_active),
            "timestamp": torch.tensor(self.timestamp) if self.timestamp else None,
            "input_state": self.input_state,
        }

        if include_base_policy:
            state["base_policy"] = include_base_policy

        return state

    def reset(self) -> None:
        """Clear all state."""
        self.policy_target = None
        self.gello_actual = None
        self.timestamp = None
        self.gate_state = None
        self.gate_decision = None


__all__ = ["DataCollectionState"]

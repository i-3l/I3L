"""
Config validation and defaults for teleop policies.

Provides clean config parsing with no legacy support.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class GatingConfig:
    """Gating configuration for admittance mode."""
    enabled: bool = False  # Default: no gating (robot always follows GELLO)
    replan_on_release: bool = True  # Reset policy when human releases control
    predict_during_correction: bool = False  # If True, policy keeps predicting during correction (not frozen)
    takeover_transition_duration: float = 0.0  # Seconds to blend from policy to gello on takeover (0.0 = instant snap)
    use_position_control: bool = True  # When false, skip position/current mode switching (stay in current control)


def parse_gating_config(raw_config: Dict[str, Any]) -> GatingConfig:
    """
    Parse gating config from policy config.

    Expected config format:
        admittance:
          gating:
            enabled: true/false
            replan_on_release: true/false
            predict_during_correction: true/false
            takeover_transition_duration: 0.0

    Args:
        raw_config: Raw policy configuration dictionary

    Returns:
        Parsed GatingConfig with defaults applied
    """
    config = GatingConfig()

    if "admittance" in raw_config:
        adm = raw_config["admittance"]
        config.use_position_control = adm.get("use_position_control", True)
        if "gating" in adm:
            gating = adm["gating"]
            config.enabled = gating.get("enabled", False)
            config.replan_on_release = gating.get("replan_on_release", True)
            config.predict_during_correction = gating.get("predict_during_correction", False)
            config.takeover_transition_duration = gating.get("takeover_transition_duration", 0.0)

    return config


__all__ = ["GatingConfig", "parse_gating_config"]

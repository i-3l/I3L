"""Configuration dataclasses for debug logging and plotting."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class PlottingConfig:
    """Configuration for real-time plotting.

    Uses per-motor layout: one row per motor with 2 subplots.
    - Left column: Measured, Tracking, Gravity currents
    - Right column: Expected current and Error
    """

    enable: bool = False
    backend: str = "matplotlib"  # "matplotlib" or "none"
    update_rate: float = 10.0  # Hz
    window_duration: float = 10.0  # seconds

    # Y-axis limits by signal type (fixed limits prevent scale jumping)
    y_limits: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "currents": [-150, 150],  # mA
            "position_error": [-0.5, 0.5],  # radians
            "velocities": [-2.0, 2.0],  # rad/s
        }
    )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PlottingConfig":
        """Create PlottingConfig from dictionary."""
        if config_dict is None:
            return cls()
        return cls(**config_dict)


@dataclass
class DebugLoggingConfig:
    """Configuration for debug data logging and visualization."""

    enable: bool = False
    log_to_file: bool = True
    log_directory: str = "logs/debug"
    buffer_size: int = 10000  # samples
    flush_interval: int = 1000  # samples

    # Signals to log
    signals: List[str] = field(
        default_factory=lambda: [
            "measured_currents",
            "tracking_currents",
            "gravity_currents",
            "friction_currents",
            "expected_currents",
            "current_diff",
            "position_error",
            "velocities",
            "goal_position",
            "current_position",
            "signed_position_error",
        ]
    )

    # Plotting configuration
    plotting: PlottingConfig = field(default_factory=PlottingConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DebugLoggingConfig":
        """Create DebugLoggingConfig from dictionary."""
        if config_dict is None:
            return cls()

        # Handle nested plotting config
        plotting_dict = config_dict.pop("plotting", None)
        plotting_config = PlottingConfig.from_dict(plotting_dict)

        return cls(plotting=plotting_config, **config_dict)

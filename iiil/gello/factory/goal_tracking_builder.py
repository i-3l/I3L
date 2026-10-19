"""Goal tracking, takeover, touch sensor, and momentum observer configuration builders for Gello controllers."""

import numpy as np
from typing import Optional
from iiil.gello.control.goal_tracker import GoalTrackingConfig
from iiil.gello.control.takeover_detector import TakeoverConfig
from iiil.gello.control.momentum_observer import MomentumObserverConfig
from iiil.gello.control.touch_sensor import TouchSensorConfig


def build_goal_tracking_config(config, dof: int) -> GoalTrackingConfig:
    """
    Build goal tracking configuration from Gello config.

    Args:
        config: Gello configuration object
        dof: Degrees of freedom

    Returns:
        GoalTrackingConfig with PID gains and optional feedforward settings
    """
    if not hasattr(config, "goal_tracking"):
        # No goal tracking config, return disabled
        return GoalTrackingConfig(enable=False)

    goal_config = config.goal_tracking

    # Get PID gains
    kp = goal_config.get("kp", None)
    ki = goal_config.get("ki", None)
    kd = goal_config.get("kd", None)

    # Convert to numpy arrays and validate
    # Allow either full DOF or half DOF (for bilateral auto-duplication)
    def _validate_gains(gains, name):
        if gains is None:
            return None
        arr = np.array(gains, dtype=np.float32)
        if len(arr) != dof and len(arr) * 2 != dof:
            raise ValueError(
                f"{name} must have {dof} elements or {dof // 2} elements "
                f"(for bilateral auto-duplication), got {len(arr)}"
            )
        return arr

    kp = _validate_gains(kp, "Kp")
    ki = _validate_gains(ki, "Ki")
    kd = _validate_gains(kd, "Kd")

    # Get enable flag
    enable = goal_config.get("enable", True)

    # Get anti-windup settings
    integral_limit = goal_config.get("integral_limit", 1.0)
    reset_integral_on_goal_change = goal_config.get("reset_integral_on_goal_change", True)

    # Get feedforward settings
    ff_config = goal_config.get("feedforward", {})
    feedforward_enable = ff_config.get("enable", False)
    kff = _validate_gains(ff_config.get("kff", None), "kff")
    velocity_filter_alpha = ff_config.get("velocity_filter_alpha", 0.3)

    return GoalTrackingConfig(
        enable=enable,
        kp=kp,
        ki=ki,
        kd=kd,
        integral_limit=integral_limit,
        reset_integral_on_goal_change=reset_integral_on_goal_change,
        feedforward_enable=feedforward_enable,
        kff=kff,
        velocity_filter_alpha=velocity_filter_alpha,
    )


def build_takeover_config(config) -> TakeoverConfig:
    """
    Build takeover detection configuration from Gello config.

    Args:
        config: Gello configuration object

    Returns:
        TakeoverConfig with detection settings
    """
    if not hasattr(config, "takeover"):
        # No takeover config, return defaults
        return TakeoverConfig()

    takeover_config = config.takeover
    position_threshold = takeover_config.get("position_threshold", 0.5)
    if isinstance(position_threshold, (list, tuple, np.ndarray)):
        position_threshold = np.array(position_threshold, dtype=np.float32)
    current_threshold = takeover_config.get("current_threshold", 150.0)
    if isinstance(current_threshold, (list, tuple, np.ndarray)):
        current_threshold = np.array(current_threshold, dtype=np.float32)
    momentum_threshold = takeover_config.get("momentum_threshold", 0.3)
    if isinstance(momentum_threshold, (list, tuple, np.ndarray)):
        momentum_threshold = np.array(momentum_threshold, dtype=np.float32)

    # Get filter configs
    current_filter_config = takeover_config.get("current_filter", {})
    position_filter_config = takeover_config.get("position_filter", {})
    velocity_filter_config = takeover_config.get("velocity_filter", {})

    return TakeoverConfig(
        enable=takeover_config.get("enable", True),

        # Current filter
        current_filter_ema_enable=current_filter_config.get("ema_enable", True),
        current_filter_alpha=current_filter_config.get("alpha", 0.3),
        current_spike_median_enable=current_filter_config.get("spike_median_enable", False),
        current_spike_median_window=current_filter_config.get("spike_median_window", 5),
        current_spike_clamp_enable=current_filter_config.get("spike_clamp_enable", False),
        current_spike_clamp_max=current_filter_config.get("spike_clamp_max", 50.0),
        # Butterworth filter (better noise rejection)
        current_butterworth_enable=current_filter_config.get("butterworth_enable", False),
        current_butterworth_cutoff=current_filter_config.get("butterworth_cutoff", 5.0),
        current_butterworth_order=current_filter_config.get("butterworth_order", 2),

        # Position filter
        position_filter_ema_enable=position_filter_config.get("ema_enable", False),
        position_filter_alpha=position_filter_config.get("alpha", 0.3),
        position_spike_median_enable=position_filter_config.get("spike_median_enable", False),
        position_spike_median_window=position_filter_config.get("spike_median_window", 5),
        position_spike_clamp_enable=position_filter_config.get("spike_clamp_enable", False),
        position_spike_clamp_max=position_filter_config.get("spike_clamp_max", 0.1),

        # Velocity filter
        velocity_filter_ema_enable=velocity_filter_config.get("ema_enable", False),
        velocity_filter_alpha=velocity_filter_config.get("alpha", 0.3),
        velocity_spike_median_enable=velocity_filter_config.get("spike_median_enable", False),
        velocity_spike_median_window=velocity_filter_config.get("spike_median_window", 5),
        velocity_spike_clamp_enable=velocity_filter_config.get("spike_clamp_enable", False),
        velocity_spike_clamp_max=velocity_filter_config.get("spike_clamp_max", 1.0),

        # Detection methods
        use_current=takeover_config.get("use_current", True),
        use_position=takeover_config.get("use_position", True),
        use_velocity=takeover_config.get("use_velocity", True),
        use_momentum=takeover_config.get("use_momentum", False),
        use_touch=takeover_config.get("use_touch", False),

        # Touch sensor release
        touch_release_cooldown=takeover_config.get("touch_release_cooldown", 0.5),

        # Thresholds
        current_threshold=current_threshold,
        position_threshold=position_threshold,
        velocity_threshold=takeover_config.get("velocity_threshold", 0.3),
        momentum_threshold=momentum_threshold,

        # Temporal filtering
        window_size=takeover_config.get("window_size", 10),

        # Auto-return
        enable_auto_return=takeover_config.get("enable_auto_return", True),
        auto_return_timeout=takeover_config.get("auto_return_timeout", 3.0),
        release_velocity=takeover_config.get("release_velocity", 0.05),
        release_current_derivative=takeover_config.get("release_current_derivative", 10.0),
        takeover_grace_period=takeover_config.get("takeover_grace_period", 0.2),
    )


def build_momentum_observer_config(config) -> MomentumObserverConfig:
    """
    Build momentum observer configuration from Gello config.

    Args:
        config: Gello configuration object

    Returns:
        MomentumObserverConfig with observer settings
    """
    if not hasattr(config, "momentum_observer"):
        return MomentumObserverConfig()

    mo_config = config.momentum_observer
    return MomentumObserverConfig(
        enable=mo_config.get("enable", False),
        gain=mo_config.get("gain", 30.0),
        include_coriolis=mo_config.get("include_coriolis", True),
        leak_rate=mo_config.get("leak_rate", 0.01),
    )


def build_touch_sensor_config(config) -> Optional[TouchSensorConfig]:
    """
    Build touch sensor configuration from Gello config.

    Args:
        config: Gello configuration object

    Returns:
        TouchSensorConfig or None if not configured
    """
    if not hasattr(config, "touch_sensor"):
        return None

    ts = config.touch_sensor
    return TouchSensorConfig(
        enable=ts.get("enable", False),
        port=ts.get("port", "/dev/tty_touch_sensor"),
        baudrate=ts.get("baudrate", 115200),
        num_channels=ts.get("num_channels", 1),
        reconnect_interval=ts.get("reconnect_interval", 2.0),
        stale_timeout=ts.get("stale_timeout", 0.5),
    )

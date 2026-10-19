"""IIIL Plotting Library.

Usage::

    from iiil.plotting import load_episodes, plot_joint_traces

    episodes = load_episodes("path/to/data.hdf5")
    plot_joint_traces(episodes, "output/traces.png")

All figure functions accept ``list[EpisodeData]`` and an output path,
and return the saved ``Path``.
"""

# Data loading
from .data import EpisodeData, load_episodes, sorted_demo_names

# Figure-level API (one call → one saved figure)
from .figures import (
    plot_action_delta_traces,
    plot_detector_metrics,
    plot_joint_traces,
    plot_joint_traces_gradient,
    plot_mahalanobis,
    plot_residual_delta_multi,
    plot_residual_delta_single,
    plot_residual_multi,
    plot_residual_single,
    plot_state_comparison,
)

# Panel-level API (draw into an existing Axes)
from .panels import (
    draw_joint_traces_colored,
    draw_joint_traces_gradient,
    draw_multi_rollout_delta,
    draw_multi_rollout_residual,
    draw_residual_delta,
    draw_residual_trace,
    draw_signal_traces,
    draw_takeover_panel,
)

# Signals / utilities
from .signals import (
    compute_deviation_norm,
    compute_takeover_probability,
    interpolate_signal,
    make_time_grid,
    moving_average,
)

# Style
from .style import (
    BASE_COLOR,
    DEVIATION_CMAP,
    MAHAL_COLOR,
    ORACLE_TRACE_COLOR,
    POLICY_COLOR,
    RESIDUAL_COLOR,
    TAKEOVER_COLOR,
    make_figure,
    save_figure,
    style_axis,
    style_top_axis,
    truncated_cmap,
)

# Constants
from .constants import (
    A1_JOINT_NAMES,
    FRANKA_JOINT_NAMES,
    METRIC_NAMES,
    REAL_R1PRO_JOINT_NAMES,
    extract_joint_traces,
    infer_joint_layout,
)

__all__ = [
    # Data
    "EpisodeData",
    "load_episodes",
    "sorted_demo_names",
    # Figures
    "plot_action_delta_traces",
    "plot_detector_metrics",
    "plot_joint_traces",
    "plot_joint_traces_gradient",
    "plot_mahalanobis",
    "plot_residual_delta_multi",
    "plot_residual_delta_single",
    "plot_residual_multi",
    "plot_residual_single",
    "plot_state_comparison",
    # Panels
    "draw_joint_traces_colored",
    "draw_joint_traces_gradient",
    "draw_multi_rollout_delta",
    "draw_multi_rollout_residual",
    "draw_residual_delta",
    "draw_residual_trace",
    "draw_signal_traces",
    "draw_takeover_panel",
    # Signals
    "compute_deviation_norm",
    "compute_takeover_probability",
    "interpolate_signal",
    "make_time_grid",
    "moving_average",
    # Style
    "BASE_COLOR",
    "DEVIATION_CMAP",
    "MAHAL_COLOR",
    "ORACLE_TRACE_COLOR",
    "POLICY_COLOR",
    "RESIDUAL_COLOR",
    "TAKEOVER_COLOR",
    "make_figure",
    "save_figure",
    "style_axis",
    "style_top_axis",
    "truncated_cmap",
    # Constants
    "A1_JOINT_NAMES",
    "FRANKA_JOINT_NAMES",
    "METRIC_NAMES",
    "REAL_R1PRO_JOINT_NAMES",
    "extract_joint_traces",
    "infer_joint_layout",
]

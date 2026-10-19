"""High-level figure functions.

Each function creates a complete figure from episode data and returns the
saved path.  They all accept the same ``list[EpisodeData]`` produced by
:func:`iiil.plotting.data.load_episodes`.

Typical usage::

    from iiil.plotting.data import load_episodes
    from iiil.plotting.figures import plot_joint_traces

    episodes = load_episodes("path/to/data.hdf5")
    plot_joint_traces(episodes, "output/joint_traces.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

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
from .signals import (
    _get,
    compute_deviation_norm,
    compute_takeover_probability,
    interpolate_signal,
    make_time_grid,
)
from .style import (
    BASE_COLOR,
    DEVIATION_CMAP,
    MAHAL_COLOR,
    METRIC_COLORS,
    ORACLE_TRACE_COLOR,
    POLICY_COLOR,
    RESIDUAL_COLOR,
    TAKEOVER_COLOR,
    make_figure,
    save_figure,
    style_axis,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Joint traces — blue/orange policy vs takeover
# ═══════════════════════════════════════════════════════════════════════════

def plot_joint_traces(
    episodes: list,
    output_path: str | Path,
    *,
    title: str | None = None,
    joint_names: list[str] | None = None,
    dpi: int = 220,
) -> Path:
    """Takeover probability (top) + per-joint blue/orange traces.

    This is the canonical "orange & blue traces" plot.
    """
    joint_names = joint_names or _get(episodes[0], "joint_names")
    n_joints = len(joint_names)
    fig, axes = make_figure(n_joints + 1)
    grid = make_time_grid(episodes)
    takeover_prob = compute_takeover_probability(episodes, grid)

    title = title or f"Joint Traces ({len(episodes)} episodes)"
    draw_takeover_panel(axes[0], grid, takeover_prob, title=title)

    for j, name in enumerate(joint_names):
        draw_joint_traces_colored(axes[j + 1], episodes, j, joint_name=name)

    axes[-1].set_xlabel("Timestamp (s)")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Joint traces — deviation-gradient coloring
# ═══════════════════════════════════════════════════════════════════════════

def plot_joint_traces_gradient(
    episodes: list,
    output_path: str | Path,
    *,
    title: str | None = None,
    joint_names: list[str] | None = None,
    series_attr: str = "joints",
    ylabel_prefix: str = "",
    use_symlog: bool = False,
    dpi: int = 220,
) -> Path:
    """Takeover probability (top) + per-joint traces colored by deviation magnitude."""
    eps_with_delta = [ep for ep in episodes if _get(ep, "action_delta_mag") is not None]
    if not eps_with_delta:
        raise ValueError("Episodes must have action_delta_mag for gradient coloring")

    joint_names = joint_names or _get(episodes[0], "joint_names")
    n_joints = len(joint_names)
    fig, axes = make_figure(n_joints + 1)
    grid = make_time_grid(episodes)
    takeover_prob = compute_takeover_probability(episodes, grid)

    vmin, vmax = compute_deviation_norm(eps_with_delta)
    norm = Normalize(vmin=vmin, vmax=vmax)

    title = title or f"Joint Traces — Deviation Gradient ({len(episodes)} eps)"
    draw_takeover_panel(axes[0], grid, takeover_prob, title=title)

    for j, name in enumerate(joint_names):
        draw_joint_traces_gradient(
            axes[j + 1], eps_with_delta, j, norm,
            joint_name=f"{ylabel_prefix}{name}",
            series_attr=series_attr,
            use_symlog=use_symlog,
        )

    axes[-1].set_xlabel("Timestamp (s)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DEVIATION_CMAP)
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01).set_label("Deviation From Base Policy")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.05, hspace=0.16)
    return save_figure(fig, output_path, dpi=dpi)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Action delta traces (gradient coloring)
# ═══════════════════════════════════════════════════════════════════════════

def plot_action_delta_traces(
    episodes: list,
    output_path: str | Path,
    *,
    title: str | None = None,
    joint_names: list[str] | None = None,
    dpi: int = 220,
) -> Path:
    """Action delta (corrected − base) per joint with deviation-gradient coloring."""
    return plot_joint_traces_gradient(
        episodes, output_path,
        title=title or f"Action Delta — Deviation Gradient ({len(episodes)} eps)",
        joint_names=joint_names,
        series_attr="action_delta",
        ylabel_prefix="Δ",
        use_symlog=True,
        dpi=dpi,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Detector metric traces (cosine, L1, etc.)
# ═══════════════════════════════════════════════════════════════════════════

def plot_detector_metrics(
    episodes: list,
    output_path: str | Path,
    *,
    metric_names: list[str] | None = None,
    interrupt_threshold: list[float] | None = None,
    release_threshold: list[float] | None = None,
    title: str | None = None,
    dpi: int = 220,
) -> Path | None:
    """Takeover probability (top) + per-metric score traces with thresholds.

    Episodes must have ``detector_scores`` with shape ``(T, n_metrics)``.
    """
    from .constants import METRIC_NAMES as DEFAULT_METRIC_NAMES

    eps = [ep for ep in episodes if _get(ep, "detector_scores") is not None]
    if not eps:
        return None

    scores_0 = _get(eps[0], "detector_scores")
    n_metrics = scores_0.shape[1] if scores_0.ndim > 1 else 1
    metric_names = (metric_names or DEFAULT_METRIC_NAMES)[:n_metrics]

    fig, axes = make_figure(
        n_metrics + 1,
        panel_height=2.5,
    )
    # Override height_ratios for metric panels
    fig.clear()
    fig, axes = plt.subplots(
        n_metrics + 1, 1,
        figsize=(14, 2.5 * (n_metrics + 1)),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0] + [1.2] * n_metrics},
    )
    axes = list(axes)

    grid = make_time_grid(eps)
    takeover_prob = compute_takeover_probability(eps, grid)

    title = title or f"Detector Metrics ({len(eps)} episodes)"
    draw_takeover_panel(axes[0], grid, takeover_prob, title=title)

    colors = METRIC_COLORS[:n_metrics]
    for mi, name in enumerate(metric_names):
        mean_sig = _interpolate_metric_column(eps, grid, mi)

        thresholds = {}
        if interrupt_threshold and mi < len(interrupt_threshold):
            thresholds["Interrupt"] = (interrupt_threshold[mi], "red", "--")
        if release_threshold and mi < len(release_threshold):
            thresholds["Release"] = (release_threshold[mi], "blue", "--")

        draw_signal_traces(
            axes[mi + 1], eps, grid, mean_sig,
            signal_key="detector_scores",
            signal_index=mi,
            label=name,
            color=colors[mi],
            thresholds=thresholds,
            takeover_prob=takeover_prob,
        )
        if mi == 0:
            axes[mi + 1].legend(loc="upper right", fontsize=8, frameon=True)

    axes[-1].set_xlabel("Timestamp (s)")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def _interpolate_metric_column(
    episodes: list,
    grid: np.ndarray,
    col: int,
    smooth_window: int = 9,
) -> np.ndarray:
    from .signals import moving_average

    interp_vals = []
    interp_masks = []
    for ep in episodes:
        t = _get(ep, "timestamps")
        scores = _get(ep, "detector_scores")
        sig = scores[:, col] if scores.ndim > 1 else scores.reshape(-1)
        interp_vals.append(np.interp(grid, t, sig))
        interp_masks.append(grid <= t[-1])
    vals = np.stack(interp_vals)
    masks = np.stack(interp_masks)
    mean = np.sum(vals * masks, axis=0) / np.clip(np.sum(masks, axis=0), 1, None)
    return moving_average(mean, smooth_window)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Mahalanobis distance trace
# ═══════════════════════════════════════════════════════════════════════════

def plot_mahalanobis(
    episodes: list,
    output_path: str | Path,
    *,
    interrupt_threshold: float | None = None,
    release_threshold: float | None = None,
    title: str | None = None,
    dpi: int = 220,
) -> Path | None:
    """Takeover probability (top) + Mahalanobis distance traces (bottom)."""
    eps = [ep for ep in episodes if _get(ep, "mahal_distance") is not None]
    if not eps:
        return None

    all_dists = np.concatenate([_get(ep, "mahal_distance") for ep in eps])
    max_thresh = max(t for t in [interrupt_threshold, release_threshold] if t is not None) if (interrupt_threshold or release_threshold) else 0.0
    y_cap = max(
        max_thresh * 3.0,
        float(np.percentile(all_dists[all_dists < np.percentile(all_dists, 99)], 99)),
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                              gridspec_kw={"height_ratios": [1.0, 1.5]})
    axes = list(axes)

    grid = make_time_grid(eps)
    takeover_prob = compute_takeover_probability(eps, grid)

    title = title or f"Mahalanobis Distance ({len(eps)} episodes)"
    draw_takeover_panel(axes[0], grid, takeover_prob, title=title)

    mean_mahal = interpolate_signal(eps, grid, "mahal_distance", clip_range=(0, y_cap))

    thresholds = {}
    if release_threshold is not None:
        thresholds["Release thresh."] = (release_threshold, "blue", "--")
    if interrupt_threshold is not None:
        thresholds["Interrupt thresh."] = (interrupt_threshold, "red", "--")

    draw_signal_traces(
        axes[1], eps, grid, mean_mahal,
        signal_key="mahal_distance",
        label="Mahalanobis\nDistance",
        color=MAHAL_COLOR,
        thresholds=thresholds,
        takeover_prob=takeover_prob,
    )
    axes[1].set_ylim(-0.5, y_cap * 1.05)
    axes[1].legend(loc="upper right", fontsize=9, frameon=True)
    axes[-1].set_xlabel("Timestamp (s)")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Residual policy — single rollout
# ═══════════════════════════════════════════════════════════════════════════

def plot_residual_single(
    episode,
    output_path: str | Path,
    *,
    title: str | None = None,
    joint_names: list[str] | None = None,
    show_oracle: bool = True,
    dpi: int = 180,
) -> Path:
    """Base vs corrected action for one episode, one row per joint.

    Optionally overlays oracle action if available.
    """
    base = _get(episode, "base_action")
    actions = _get(episode, "actions")
    if base is None or actions is None:
        raise ValueError("Episode must have both base_action and actions")

    oracle = _get(episode, "oracle_action") if show_oracle else None
    correction = _get(episode, "correction_mask")
    joint_names = joint_names or _get(episode, "joint_names")
    n_dims = base.shape[1]
    t = np.arange(n_dims, dtype=np.float64) if _get(episode, "timestamps") is None else _get(episode, "timestamps")
    # Use action dim, not joint dim — they may differ
    n_action = base.shape[1]

    fig, axes = plt.subplots(
        n_action, 1,
        figsize=(14, max(2.2 * n_action, 8)),
        sharex=True,
        constrained_layout=True,
    )
    if n_action == 1:
        axes = [axes]

    timesteps = _get(episode, "timestamps")
    for j in range(n_action):
        name = joint_names[j] if j < len(joint_names) else f"J{j + 1}"
        draw_residual_trace(
            axes[j], timesteps, base, actions, j,
            intervention=correction.astype(float) if correction is not None else None,
            oracle_action=oracle,
            joint_name=name,
            show_legend=(j == 0),
        )

    axes[-1].set_xlabel("Timestamp (s)")
    fig.suptitle(title or f"Residual Policy — {_get(episode, 'name')}")
    return save_figure(fig, output_path, dpi=dpi)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Residual policy — single rollout deltas
# ═══════════════════════════════════════════════════════════════════════════

def plot_residual_delta_single(
    episode,
    output_path: str | Path,
    *,
    title: str | None = None,
    joint_names: list[str] | None = None,
    show_oracle: bool = True,
    dpi: int = 180,
) -> Path:
    """Δ(corrected - base) for one episode, optionally with Δ(oracle - base)."""
    base = _get(episode, "base_action")
    actions = _get(episode, "actions")
    if base is None or actions is None:
        raise ValueError("Episode must have both base_action and actions")

    oracle = _get(episode, "oracle_action") if show_oracle else None
    correction = _get(episode, "correction_mask")
    joint_names = joint_names or _get(episode, "joint_names")
    n_action = base.shape[1]

    fig, axes = plt.subplots(
        n_action, 1,
        figsize=(14, max(2.2 * n_action, 8)),
        sharex=True,
        constrained_layout=True,
    )
    if n_action == 1:
        axes = [axes]

    timesteps = _get(episode, "timestamps")
    for j in range(n_action):
        name = joint_names[j] if j < len(joint_names) else f"J{j + 1}"
        draw_residual_delta(
            axes[j], timesteps, base, actions, j,
            oracle_action=oracle,
            intervention=correction.astype(float) if correction is not None else None,
            joint_name=name,
            show_legend=(j == 0),
        )

    axes[-1].set_xlabel("Timestamp (s)")
    fig.suptitle(title or f"Residual Δ — {_get(episode, 'name')}")
    return save_figure(fig, output_path, dpi=dpi)


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Residual policy — multi-rollout overlay (absolute)
# ═══════════════════════════════════════════════════════════════════════════

def plot_residual_multi(
    episodes: list,
    output_path: str | Path,
    *,
    title: str | None = None,
    joint_names: list[str] | None = None,
    show_oracle: bool = True,
    dpi: int = 220,
) -> Path:
    """Overlay base/corrected/oracle from N rollouts.

    Top row: takeover probability.  Remaining rows: one per joint.
    """
    eps = [ep for ep in episodes if _get(ep, "base_action") is not None]
    if not eps:
        raise ValueError("Episodes must have base_action for residual plots")

    joint_names = joint_names or _get(eps[0], "joint_names")
    n_action = _get(eps[0], "base_action").shape[1]
    names = [joint_names[j] if j < len(joint_names) else f"J{j + 1}" for j in range(n_action)]

    fig, axes = make_figure(n_action + 1)
    grid = make_time_grid(eps)
    takeover_prob = compute_takeover_probability(eps, grid)

    title = title or f"Residual Multi-Rollout ({len(eps)} eps)"
    draw_takeover_panel(axes[0], grid, takeover_prob, title=title)

    for j, name in enumerate(names):
        draw_multi_rollout_residual(
            axes[j + 1], eps, j,
            joint_name=name,
            show_oracle=show_oracle,
        )

    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=BASE_COLOR, label="Base policy"),
        mlines.Line2D([], [], color=RESIDUAL_COLOR, label="Corrected"),
    ]
    if show_oracle:
        handles.append(mlines.Line2D([], [], color=ORACLE_TRACE_COLOR, linestyle="--", label="Oracle"))
    axes[1].legend(handles=handles, loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Timestamp (s)")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


# ═══════════════════════════════════════════════════════════════════════════
# 9.  Residual policy — multi-rollout overlay (deltas)
# ═══════════════════════════════════════════════════════════════════════════

def plot_residual_delta_multi(
    episodes: list,
    output_path: str | Path,
    *,
    title: str | None = None,
    joint_names: list[str] | None = None,
    show_oracle: bool = True,
    dpi: int = 220,
) -> Path:
    """Overlay Δ(corrected − base) and Δ(oracle − base) from N rollouts."""
    eps = [ep for ep in episodes if _get(ep, "base_action") is not None]
    if not eps:
        raise ValueError("Episodes must have base_action for residual delta plots")

    joint_names = joint_names or _get(eps[0], "joint_names")
    n_action = _get(eps[0], "base_action").shape[1]
    names = [joint_names[j] if j < len(joint_names) else f"J{j + 1}" for j in range(n_action)]

    fig, axes = make_figure(n_action + 1)
    grid = make_time_grid(eps)
    takeover_prob = compute_takeover_probability(eps, grid)

    title = title or f"Residual Δ Multi-Rollout ({len(eps)} eps)"
    draw_takeover_panel(axes[0], grid, takeover_prob, title=title)

    for j, name in enumerate(names):
        draw_multi_rollout_delta(
            axes[j + 1], eps, j,
            joint_name=name,
            show_oracle=show_oracle,
        )

    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=RESIDUAL_COLOR, label="Residual Δ"),
    ]
    if show_oracle:
        handles.append(mlines.Line2D([], [], color=ORACLE_TRACE_COLOR, linestyle="--", label="Oracle Δ"))
    axes[1].legend(handles=handles, loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Timestamp (s)")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


# ═══════════════════════════════════════════════════════════════════════════
# 10. State comparison — human demos vs correction data
# ═══════════════════════════════════════════════════════════════════════════

def plot_state_comparison(
    human_demos: list,
    correction_demos: list,
    output_path: str | Path,
    *,
    title: str | None = None,
    joint_names: list[str] | None = None,
    dpi: int = 220,
) -> Path:
    """Compare joint-state distributions between human demos and correction data."""
    all_eps = human_demos + correction_demos
    joint_names = joint_names or _get(all_eps[0], "joint_names")
    n_joints = len(joint_names)

    fig, axes = make_figure(n_joints + 1)
    grid = make_time_grid(all_eps)

    # Top panel: state occupancy
    top_ax = axes[0]
    if human_demos:
        human_vals = []
        human_masks = []
        for ep in human_demos:
            t = _get(ep, "timestamps")
            human_vals.append(np.interp(grid, t, _get(ep, "human_demo_mask").astype(float)))
            human_masks.append(grid <= t[-1])
        hv = np.stack(human_vals)
        hm = np.stack(human_masks)
        from .signals import moving_average
        human_prob = moving_average(np.sum(hv * hm, axis=0) / np.clip(np.sum(hm, axis=0), 1, None), 11)
        top_ax.plot(grid, human_prob, color="#1f78b4", linewidth=2.4, label="Original demos")

    if correction_demos:
        corr_vals = []
        corr_masks = []
        for ep in correction_demos:
            t = _get(ep, "timestamps")
            corr_vals.append(np.interp(grid, t, _get(ep, "correction_mask").astype(float)))
            corr_masks.append(grid <= t[-1])
        cv = np.stack(corr_vals)
        cm = np.stack(corr_masks)
        from .signals import moving_average
        corr_prob = moving_average(np.sum(cv * cm, axis=0) / np.clip(np.sum(cm, axis=0), 1, None), 11)
        top_ax.plot(grid, corr_prob, color="#d95f02", linewidth=2.4, label="Correction state")

    top_ax.set_ylim(-0.02, 1.02)
    top_ax.set_ylabel("State\nOccup.")
    top_ax.set_title(title or "State Comparison")
    top_ax.grid(alpha=0.2)
    top_ax.legend(loc="upper right", frameon=True)

    for j, name in enumerate(joint_names):
        ax = axes[j + 1]
        for ep in human_demos:
            y = np.where(_get(ep, "human_demo_mask"), _get(ep, "joints")[:, j], np.nan)
            ax.plot(_get(ep, "timestamps"), y, color="#2b8cbe", alpha=0.20, linewidth=0.9)
        for ep in correction_demos:
            y = np.where(_get(ep, "correction_mask"), _get(ep, "joints")[:, j], np.nan)
            ax.plot(_get(ep, "timestamps"), y, color="#e6550d", alpha=0.30, linewidth=1.0)
        ax.set_ylabel(name)
        style_axis(ax)

    axes[-1].set_xlabel("Timestamp (s)")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)

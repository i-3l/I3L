"""Low-level drawing primitives for individual subplot panels.

Every function here takes an ``Axes`` and draws into it.  They never
create figures, save files, or manage layouts — that is the job of the
figure-level modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from .signals import _get
from .style import (
    CORRECTED_ACTION_COLOR,
    BASE_COLOR,
    DEVIATION_CMAP,
    INTERVENTION_BG_COLOR,
    ORACLE_TRACE_COLOR,
    POLICY_COLOR,
    RESIDUAL_COLOR,
    TAKEOVER_COLOR,
    TAKEOVER_FILL_COLOR,
    TAKEOVER_LINE_COLOR,
    style_axis,
    style_top_axis,
    truncated_cmap,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# ───────────────────────────────────────────────────────────────────────────
# Takeover probability panel (top row)
# ───────────────────────────────────────────────────────────────────────────

def draw_takeover_panel(
    ax: Axes,
    grid: np.ndarray,
    takeover_prob: np.ndarray,
    *,
    title: str | None = None,
    ylabel: str = "Takeover\nProb.",
    color: str = TAKEOVER_LINE_COLOR,
    fill_color: str = TAKEOVER_FILL_COLOR,
    fill_alpha: float = 0.55,
) -> None:
    ax.plot(grid, takeover_prob, color=color, linewidth=3)
    ax.fill_between(grid, 0, takeover_prob, color=fill_color, alpha=fill_alpha)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    style_top_axis(ax)


# ───────────────────────────────────────────────────────────────────────────
# Joint trace panels — policy/takeover coloring
# ───────────────────────────────────────────────────────────────────────────

def draw_joint_traces_colored(
    ax: Axes,
    episodes: list,
    joint_idx: int,
    *,
    joint_name: str | None = None,
    policy_color: str = POLICY_COLOR,
    takeover_color: str = TAKEOVER_COLOR,
    policy_alpha: float = 0.15,
    takeover_alpha: float = 0.2,
) -> None:
    """Draw per-episode joint traces split into policy (blue) and takeover (orange)."""
    for ep in episodes:
        t = _get(ep, "timestamps")
        y = _get(ep, "joints")[:, joint_idx]
        active = _get(ep, "correction_mask")
        ax.plot(t, np.where(active, np.nan, y), color=policy_color, alpha=policy_alpha, linewidth=0.8)
        ax.plot(t, np.where(active, y, np.nan), color=takeover_color, alpha=takeover_alpha, linewidth=1.0)
    if joint_name:
        ax.set_ylabel(joint_name)
    style_axis(ax)


# ───────────────────────────────────────────────────────────────────────────
# Joint trace panels — gradient coloring by deviation magnitude
# ───────────────────────────────────────────────────────────────────────────

def _add_gradient_segments(
    ax: Axes,
    t: np.ndarray,
    y: np.ndarray,
    active: np.ndarray,
    values: np.ndarray,
    norm: Normalize,
    cmap,
) -> None:
    if len(t) < 2:
        return
    inactive = np.where(active, np.nan, y)
    ax.plot(t, inactive, color="#d9d9d9", alpha=0.14, linewidth=0.7)

    segments = []
    segment_values = []
    for i in range(len(t) - 1):
        if active[i] or active[i + 1]:
            segments.append([[t[i], y[i]], [t[i + 1], y[i + 1]]])
            segment_values.append(float(0.5 * (values[i] + values[i + 1])))
    if not segments:
        return
    lc = LineCollection(np.asarray(segments), cmap=cmap, norm=norm, linewidths=1.5, alpha=0.9)
    lc.set_array(np.asarray(segment_values))
    ax.add_collection(lc)


def draw_joint_traces_gradient(
    ax: Axes,
    episodes: list,
    joint_idx: int,
    norm: Normalize,
    *,
    joint_name: str | None = None,
    series_attr: str = "joints",
    cmap=None,
    use_symlog: bool = False,
) -> None:
    """Draw joint traces colored by action-deviation magnitude (viridis gradient)."""
    if cmap is None:
        cmap = DEVIATION_CMAP
    for ep in episodes:
        t = _get(ep, "timestamps")
        y = _get(ep, series_attr)[:, joint_idx]
        active = _get(ep, "correction_mask")
        mag = _get(ep, "action_delta_mag")
        _add_gradient_segments(ax, t, y, active, mag, norm, cmap)
    if joint_name:
        ax.set_ylabel(joint_name)
    if use_symlog:
        ax.set_yscale("symlog", linthresh=0.01)
    style_axis(ax)


# ───────────────────────────────────────────────────────────────────────────
# Scalar signal panel (cosine, L1, Mahalanobis, etc.)
# ───────────────────────────────────────────────────────────────────────────

def draw_signal_traces(
    ax: Axes,
    episodes: list,
    grid: np.ndarray,
    mean_signal: np.ndarray,
    *,
    signal_key: str,
    signal_index: int | None = None,
    label: str = "",
    color: str = "#e41a1c",
    trace_alpha: float = 0.06,
    trace_lw: float = 0.6,
    mean_lw: float = 2.5,
    thresholds: dict[str, tuple[float, str, str]] | None = None,
    takeover_prob: np.ndarray | None = None,
    takeover_shade_above: float = 0.5,
) -> None:
    """Draw individual episode signal traces + bold mean + optional threshold lines.

    *thresholds*: ``{"Interrupt": (value, "red", "--"), "Release": (value, "blue", "--")}``
    """
    for ep in episodes:
        t = _get(ep, "timestamps")
        sig = _get(ep, signal_key)
        if signal_index is not None:
            sig = sig[:, signal_index]
        sig = sig.reshape(-1)
        ax.plot(t, sig, color=color, alpha=trace_alpha, linewidth=trace_lw)

    ax.plot(grid, mean_signal, color=color, linewidth=mean_lw, label=f"Mean {label}" if label else None)

    if thresholds:
        for name, (val, clr, ls) in thresholds.items():
            ax.axhline(val, color=clr, linestyle=ls, linewidth=1.5, alpha=0.8, label=f"{name} ({val:.2g})")

    if takeover_prob is not None:
        ylims = ax.get_ylim()
        ax.fill_between(
            grid, ylims[0], ylims[1],
            where=takeover_prob > takeover_shade_above,
            color=TAKEOVER_COLOR, alpha=0.08,
        )

    if label:
        ax.set_ylabel(label)
    style_axis(ax)


# ───────────────────────────────────────────────────────────────────────────
# Residual policy panels — single rollout
# ───────────────────────────────────────────────────────────────────────────

def draw_residual_trace(
    ax: Axes,
    timesteps: np.ndarray,
    base_action: np.ndarray,
    corrected_action: np.ndarray,
    joint_idx: int,
    *,
    intervention: np.ndarray | None = None,
    oracle_action: np.ndarray | None = None,
    joint_name: str | None = None,
    base_color: str = BASE_COLOR,
    corrected_color: str = RESIDUAL_COLOR,
    oracle_color: str = ORACLE_TRACE_COLOR,
    show_legend: bool = False,
) -> None:
    """Draw base vs corrected action for a single rollout on one joint."""
    ax.plot(timesteps, base_action[:, joint_idx], color=base_color, linewidth=1.3, label="Base policy")
    ax.plot(timesteps, corrected_action[:, joint_idx], color=corrected_color, linewidth=1.4, label="Corrected")

    if oracle_action is not None:
        ax.plot(timesteps, oracle_action[:, joint_idx], color=oracle_color, linewidth=1.1,
                alpha=0.7, linestyle="--", label="Oracle")

    if intervention is not None:
        _shade_intervention(ax, intervention)

    if joint_name:
        ax.set_ylabel(joint_name)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8)
    style_axis(ax, grid_alpha=0.2)


def draw_residual_delta(
    ax: Axes,
    timesteps: np.ndarray,
    base_action: np.ndarray,
    corrected_action: np.ndarray,
    joint_idx: int,
    *,
    oracle_action: np.ndarray | None = None,
    intervention: np.ndarray | None = None,
    joint_name: str | None = None,
    show_legend: bool = False,
) -> None:
    """Draw Δ(corrected - base) and optionally Δ(oracle - base)."""
    delta = corrected_action[:, joint_idx] - base_action[:, joint_idx]
    ax.plot(timesteps, delta, color=RESIDUAL_COLOR, linewidth=1.4, label="Residual Δ")
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)

    if oracle_action is not None:
        oracle_delta = oracle_action[:, joint_idx] - base_action[:, joint_idx]
        ax.plot(timesteps, oracle_delta, color=ORACLE_TRACE_COLOR, linewidth=1.1,
                alpha=0.7, linestyle="--", label="Oracle Δ")

    if intervention is not None:
        _shade_intervention(ax, intervention)

    if joint_name:
        ax.set_ylabel(f"Δ {joint_name}")
    if show_legend:
        ax.legend(loc="upper right", fontsize=8)
    style_axis(ax, grid_alpha=0.2)


# ───────────────────────────────────────────────────────────────────────────
# Multi-rollout residual panels
# ───────────────────────────────────────────────────────────────────────────

def draw_multi_rollout_residual(
    ax: Axes,
    episodes: list,
    joint_idx: int,
    *,
    joint_name: str | None = None,
    base_color: str = BASE_COLOR,
    corrected_color: str = RESIDUAL_COLOR,
    oracle_color: str = ORACLE_TRACE_COLOR,
    alpha: float = 0.15,
    show_oracle: bool = True,
) -> None:
    """Overlay base/corrected/oracle from many episodes on one joint panel."""
    for ep in episodes:
        t = _get(ep, "timestamps")
        base = _get(ep, "base_action")
        actions = _get(ep, "actions")
        ax.plot(t, base[:, joint_idx], color=base_color, alpha=alpha, linewidth=0.8)
        ax.plot(t, actions[:, joint_idx], color=corrected_color, alpha=alpha, linewidth=0.9)

        if show_oracle:
            oracle = _get(ep, "oracle_action")
            if oracle is not None:
                ax.plot(t, oracle[:, joint_idx], color=oracle_color, alpha=alpha * 0.7,
                        linewidth=0.7, linestyle="--")

    if joint_name:
        ax.set_ylabel(joint_name)
    style_axis(ax)


def draw_multi_rollout_delta(
    ax: Axes,
    episodes: list,
    joint_idx: int,
    *,
    joint_name: str | None = None,
    show_oracle: bool = True,
    alpha: float = 0.15,
) -> None:
    """Overlay Δ(corrected-base) and Δ(oracle-base) from many episodes."""
    for ep in episodes:
        t = _get(ep, "timestamps")
        base = _get(ep, "base_action")
        actions = _get(ep, "actions")
        delta = actions[:, joint_idx] - base[:, joint_idx]
        ax.plot(t, delta, color=RESIDUAL_COLOR, alpha=alpha, linewidth=0.9)

        if show_oracle:
            oracle = _get(ep, "oracle_action")
            if oracle is not None:
                oracle_delta = oracle[:, joint_idx] - base[:, joint_idx]
                ax.plot(t, oracle_delta, color=ORACLE_TRACE_COLOR, alpha=alpha * 0.7,
                        linewidth=0.7, linestyle="--")

    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
    if joint_name:
        ax.set_ylabel(f"Δ {joint_name}")
    style_axis(ax)


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _shade_intervention(ax: Axes, intervention: np.ndarray) -> None:
    active = intervention.reshape(-1) >= 0.5
    if not np.any(active):
        return
    xlim = ax.get_xlim()
    x = np.arange(len(active))
    start = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            ax.axvspan(x[start], x[i - 1], color=INTERVENTION_BG_COLOR, alpha=0.22, lw=0)
            start = None
    if start is not None:
        ax.axvspan(x[start], x[len(active) - 1], color=INTERVENTION_BG_COLOR, alpha=0.22, lw=0)
    ax.set_xlim(xlim)

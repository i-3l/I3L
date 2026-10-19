"""Matplotlib setup, figure styling, and save helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

POLICY_COLOR = "#1f77b4"
TAKEOVER_COLOR = "#ff7f0e"
MAHAL_COLOR = "#2ca02c"
BASE_POLICY_COLOR = "#9aa0a6"
CORRECTED_ACTION_COLOR = "#d94841"
ORACLE_COLOR = "#e6550d"
HUMAN_DEMO_COLOR = "#2b8cbe"
INTERVENTION_BG_COLOR = "#f4c7c3"
METRIC_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
TAKEOVER_FILL_COLOR = "#fdd0a2"
TAKEOVER_LINE_COLOR = "#8c2d04"

# Darker palette for single-rollout plots
RESIDUAL_COLOR = "#d94841"
BASE_COLOR = "#9aa0a6"
ORACLE_TRACE_COLOR = "#4daf4a"


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

def truncated_cmap(name: str, lo: float = 0.18, hi: float = 0.98) -> LinearSegmentedColormap:
    base = matplotlib.colormaps.get_cmap(name)
    return LinearSegmentedColormap.from_list(f"{name}_trunc", base(np.linspace(lo, hi, 256)))


DEVIATION_CMAP = truncated_cmap("viridis", 0.22, 0.98)


# ---------------------------------------------------------------------------
# Default figure dimensions
# ---------------------------------------------------------------------------

DEFAULT_FIG_WIDTH = 14
PANEL_HEIGHT = 2.1
TOP_PANEL_HEIGHT_RATIO = 1.25
DEFAULT_DPI = 220
DEFAULT_GRID_POINTS = 300


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def style_axis(ax: Axes, grid_alpha: float = 0.15) -> None:
    ax.grid(True, alpha=grid_alpha)


def style_top_axis(ax: Axes, grid_alpha: float = 0.2) -> None:
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=grid_alpha)


def save_figure(
    fig: Figure,
    path: str | Path,
    dpi: int = DEFAULT_DPI,
    close: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def make_figure(
    n_rows: int,
    *,
    width: float = DEFAULT_FIG_WIDTH,
    panel_height: float = PANEL_HEIGHT,
    top_ratio: float = TOP_PANEL_HEIGHT_RATIO,
    sharex: bool = True,
) -> tuple[Figure, list[Axes]]:
    height_ratios = [top_ratio] + [1.0] * (n_rows - 1)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(width, panel_height * n_rows),
        sharex=sharex,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_rows == 1:
        axes = [axes]
    return fig, list(axes)

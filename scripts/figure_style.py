"""Unified figure style for I3L publication figures.

Usage:
    from figure_style import apply_style, COLORS, get_palette

    apply_style()  # Call once before creating figures

    fig, ax = plt.subplots()
    ax.bar(x, y, color=COLORS['primary'])
    # or
    palette = get_palette(4)  # Get 4 colors from the gradient
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler


# === Color Palette ===
# Warm gradient: burnt orange → amber → cream
# With dark grounding color and accent

COLORS = {
    # Primary orange gradient (dark to light)
    "burnt_orange": "#C45B28",
    "deep_orange": "#D4722C",
    "orange": "#E8873C",
    "amber": "#F2A553",
    "gold": "#F5BE6E",
    "light_gold": "#F9D89A",
    "cream": "#FDF0D5",

    # Grounding / neutral
    "charcoal": "#2D2D2D",
    "dark_gray": "#4A4A4A",
    "medium_gray": "#7A7A7A",
    "light_gray": "#BFBFBF",
    "off_white": "#F7F5F2",

    # Accent (complementary for contrast)
    "navy": "#2C3E50",
    "teal": "#1A7A6D",
    "muted_blue": "#5B8FA8",

    # Semantic
    "success": "#5B9E4D",
    "failure": "#C45B28",
    "neutral": "#7A7A7A",

    # Primary shorthand
    "primary": "#D4722C",
    "secondary": "#F2A553",
    "tertiary": "#F5BE6E",
    "quaternary": "#F9D89A",
}

# Ordered palette for sequential data (e.g., rounds 1-4)
SEQUENTIAL_PALETTE = [
    COLORS["burnt_orange"],
    COLORS["deep_orange"],
    COLORS["amber"],
    COLORS["light_gold"],
]

# Categorical palette for distinct categories
CATEGORICAL_PALETTE = [
    COLORS["burnt_orange"],
    COLORS["amber"],
    COLORS["teal"],
    COLORS["muted_blue"],
    COLORS["navy"],
    COLORS["gold"],
]


def get_palette(n: int, style: str = "sequential") -> list[str]:
    """Get n colors from a palette.

    Args:
        n: Number of colors needed.
        style: 'sequential' for ordered gradient, 'categorical' for distinct colors.
    """
    if style == "sequential":
        if n <= len(SEQUENTIAL_PALETTE):
            return SEQUENTIAL_PALETTE[:n]
        # Interpolate from the gradient
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "warm", [COLORS["burnt_orange"], COLORS["amber"], COLORS["cream"]]
        )
        return [mpl.colors.to_hex(cmap(i / max(1, n - 1))) for i in range(n)]
    else:
        return (CATEGORICAL_PALETTE * ((n // len(CATEGORICAL_PALETTE)) + 1))[:n]


def apply_style():
    """Apply publication-quality figure style. Call once at the start of your script."""

    plt.rcParams.update({
        # Figure
        "figure.facecolor": "white",
        "figure.edgecolor": "none",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,

        # Font — use serif for academic papers
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,

        # Use TeX-style math rendering
        "mathtext.fontset": "cm",

        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["dark_gray"],
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 4,
        "axes.titlepad": 8,
        "axes.prop_cycle": cycler("color", CATEGORICAL_PALETTE),

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,

        # Grid (when enabled)
        "grid.color": COLORS["light_gray"],
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": COLORS["light_gray"],
        "legend.fancybox": False,

        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,

        # Bars
        "patch.edgecolor": COLORS["dark_gray"],
        "patch.linewidth": 0.4,

        # Layout
        "figure.constrained_layout.use": True,
    })


# Standard figure sizes for academic papers (in inches)
# Single column: ~3.25", double column: ~6.75", full page width: ~7"
FIGSIZE = {
    "single_col": (3.25, 2.5),
    "single_col_tall": (3.25, 3.5),
    "double_col": (6.75, 2.5),
    "double_col_tall": (6.75, 4.0),
    "full_width": (7.0, 3.0),
    "full_page": (7.0, 8.0),
}


def figure(size: str = "double_col", **kwargs):
    """Create a figure with standard sizing.

    Args:
        size: Key from FIGSIZE or a (width, height) tuple.
    """
    if isinstance(size, str):
        figsize = FIGSIZE[size]
    else:
        figsize = size
    return plt.figure(figsize=figsize, **kwargs)

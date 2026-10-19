"""Signal processing helpers shared across all figures."""

from __future__ import annotations

import numpy as np

from .style import DEFAULT_GRID_POINTS


def moving_average(values: np.ndarray, window: int = 9) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values
    window = min(window, values.size)
    kernel = np.ones(window, dtype=np.float64) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def compute_takeover_probability(
    episodes: list,
    grid: np.ndarray,
    *,
    mask_attr: str = "correction_mask",
    smooth_window: int = 11,
) -> np.ndarray:
    """Aggregate per-episode boolean masks into a takeover probability curve.

    Each episode is expected to have `timestamps` (1-D) and a boolean array
    at the attribute (or dict key) given by *mask_attr*.
    """
    if not episodes:
        return np.zeros_like(grid)

    values = []
    masks = []
    for ep in episodes:
        t = _get(ep, "timestamps")
        active = _get(ep, mask_attr).astype(float)
        values.append(np.interp(grid, t, active))
        masks.append(grid <= t[-1])

    values_arr = np.stack(values)
    masks_arr = np.stack(masks)
    prob = np.sum(values_arr * masks_arr, axis=0) / np.clip(np.sum(masks_arr, axis=0), 1, None)
    return moving_average(prob, smooth_window)


def interpolate_signal(
    episodes: list,
    grid: np.ndarray,
    signal_key: str,
    *,
    smooth_window: int = 9,
    clip_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Interpolate a scalar per-timestep signal across episodes and return the mean."""
    interp_vals = []
    interp_masks = []
    for ep in episodes:
        t = _get(ep, "timestamps")
        sig = _get(ep, signal_key).reshape(-1).astype(np.float64)
        if clip_range is not None:
            sig = np.clip(sig, clip_range[0], clip_range[1])
        interp_vals.append(np.interp(grid, t, sig))
        interp_masks.append(grid <= t[-1])

    vals = np.stack(interp_vals)
    masks = np.stack(interp_masks)
    mean = np.sum(vals * masks, axis=0) / np.clip(np.sum(masks, axis=0), 1, None)
    return moving_average(mean, smooth_window)


def make_time_grid(episodes: list, n_points: int = DEFAULT_GRID_POINTS) -> np.ndarray:
    max_t = max(float(_get(ep, "timestamps")[-1]) for ep in episodes)
    return np.linspace(0.0, max_t, n_points)


def compute_deviation_norm(
    episodes: list,
    *,
    percentile_lo: float = 15.0,
    percentile_hi: float = 97.0,
) -> tuple[float, float]:
    """Return (vmin, vmax) for action deviation coloring."""
    all_mag = np.concatenate([_get(ep, "action_delta_mag") for ep in episodes])
    lo = max(0.0, float(np.percentile(all_mag, percentile_lo)))
    hi = float(np.percentile(all_mag, percentile_hi))
    return lo, max(hi, lo + 1e-6)


# ---------------------------------------------------------------------------
# Attribute access helper — works for dataclasses and dicts
# ---------------------------------------------------------------------------

def _get(obj, key):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)

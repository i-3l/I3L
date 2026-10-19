#!/usr/bin/env python3
"""
Filter and visualize LeRobot correction datasets.

Generates joint trace plots with a correction probability subplot on top,
showing intervention density across time for each episode overlaid.

Usage:
    python scripts/filter_data.py plot \
        data/books-base-int-1-25 \
        data/books-hg-int-2-25 \
        data/books-hg-int-3-25

    python scripts/filter_data.py plot data/books-base-int-1-25 --episode 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np


def load_lerobot_dataset(dataset_path: Path) -> dict:
    """Load all episodes from a LeRobot dataset directory."""
    import json
    import pandas as pd

    data_dir = dataset_path / "data" / "chunk-000"
    meta_path = dataset_path / "meta" / "info.json"

    with open(meta_path) as f:
        info = json.load(f)

    episodes = []
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        episodes.append({
            "actions": np.stack(df["action"].values),
            "obs_state": np.stack(df["observation.state"].values),
            "int_state": df["int_state"].values.astype(np.uint8),
            "frame_index": df["frame_index"].values,
            "episode_index": int(df["episode_index"].iloc[0]),
        })

    return {"episodes": episodes, "info": info, "name": dataset_path.name}


def compute_correction_probability(int_states: np.ndarray, window: int = 30) -> np.ndarray:
    """Compute smoothed correction probability (fraction of frames in correction state)."""
    is_correction = (int_states == 2).astype(np.float32)
    kernel = np.ones(window) / window
    smoothed = np.convolve(is_correction, kernel, mode="same")
    return smoothed


def plot_dataset_traces(
    dataset_path: Path,
    output_dir: Optional[Path] = None,
    episode_idx: Optional[int] = None,
    joint_dims: Optional[list[int]] = None,
    fps: int = 30,
):
    """Plot joint traces with correction probability on top for a dataset."""
    data = load_lerobot_dataset(dataset_path)
    episodes = data["episodes"]
    ds_name = data["name"]

    if output_dir is None:
        output_dir = dataset_path / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_action_dims = episodes[0]["actions"].shape[1]
    if joint_dims is None:
        joint_dims = list(range(min(n_action_dims, 14)))

    if episode_idx is not None:
        _plot_single_episode(episodes[episode_idx], ds_name, joint_dims, fps, output_dir)
    else:
        _plot_all_episodes(episodes, ds_name, joint_dims, fps, output_dir)


def _plot_all_episodes(
    episodes: list[dict],
    ds_name: str,
    joint_dims: list[int],
    fps: int,
    output_dir: Path,
):
    """Overlay all episodes with correction probability on top."""
    n_joints = len(joint_dims)
    n_rows = 1 + n_joints  # correction prob + joints

    fig, axes = plt.subplots(n_rows, 1, figsize=(20, 2.5 * n_rows), sharex=False)
    ax_corr = axes[0]

    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(episodes))))

    # Correction probability panel
    for i, ep in enumerate(episodes):
        int_states = ep["int_state"]
        t = np.arange(len(int_states)) / fps
        corr_prob = compute_correction_probability(int_states, window=30)
        ax_corr.plot(t, corr_prob, color=colors[i % len(colors)], alpha=0.5, linewidth=0.8)

    ax_corr.set_ylabel("Correction\nProbability", fontsize=10, fontweight="bold")
    ax_corr.set_ylim(-0.05, 1.05)
    ax_corr.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax_corr.grid(True, alpha=0.2)
    ax_corr.set_title(f"{ds_name} — {len(episodes)} episodes", fontsize=13, fontweight="bold")

    # Add colored background bands showing mean correction regions
    _add_mean_correction_band(ax_corr, episodes, fps)

    # Joint action traces
    for j_idx, dim in enumerate(joint_dims):
        ax = axes[1 + j_idx]
        for i, ep in enumerate(episodes):
            actions = ep["actions"][:, dim]
            int_states = ep["int_state"]
            t = np.arange(len(actions)) / fps
            _plot_colored_trace(ax, t, actions, int_states, colors[i % len(colors)], alpha=0.5)

        ax.set_ylabel(f"Joint {dim}", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    # Legend
    legend_patches = [
        mpatches.Patch(color="green", alpha=0.3, label="Autonomous (state 3)"),
        mpatches.Patch(color="orange", alpha=0.3, label="Pre-intervention (state 1)"),
        mpatches.Patch(color="red", alpha=0.3, label="Correction (state 2)"),
    ]
    ax_corr.legend(handles=legend_patches, loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = output_dir / f"{ds_name}_traces.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_single_episode(
    episode: dict,
    ds_name: str,
    joint_dims: list[int],
    fps: int,
    output_dir: Path,
):
    """Plot a single episode with correction probability on top."""
    n_joints = len(joint_dims)
    n_rows = 1 + n_joints

    fig, axes = plt.subplots(n_rows, 1, figsize=(20, 2.5 * n_rows), sharex=True)
    ax_corr = axes[0]

    int_states = episode["int_state"]
    ep_idx = episode["episode_index"]
    t = np.arange(len(int_states)) / fps

    # Correction probability
    corr_prob = compute_correction_probability(int_states, window=30)
    ax_corr.plot(t, corr_prob, color="red", linewidth=1.5)
    ax_corr.fill_between(t, 0, corr_prob, alpha=0.2, color="red")
    ax_corr.set_ylabel("Correction\nProbability", fontsize=10, fontweight="bold")
    ax_corr.set_ylim(-0.05, 1.05)
    ax_corr.grid(True, alpha=0.2)
    ax_corr.set_title(
        f"{ds_name} — Episode {ep_idx} ({len(int_states)} frames)",
        fontsize=13, fontweight="bold",
    )

    # Background shading for intervention regions
    _shade_intervention_regions(ax_corr, int_states, fps)

    # Joint action traces with colored segments
    for j_idx, dim in enumerate(joint_dims):
        ax = axes[1 + j_idx]
        actions = episode["actions"][:, dim]
        _plot_colored_trace_single(ax, t, actions, int_states)
        _shade_intervention_regions(ax, int_states, fps)
        ax.set_ylabel(f"Joint {dim}", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    legend_patches = [
        mpatches.Patch(color="green", alpha=0.3, label="Autonomous (state 3)"),
        mpatches.Patch(color="orange", alpha=0.3, label="Pre-intervention (state 1)"),
        mpatches.Patch(color="red", alpha=0.3, label="Correction (state 2)"),
    ]
    ax_corr.legend(handles=legend_patches, loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = output_dir / f"{ds_name}_episode_{ep_idx:03d}_traces.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _shade_intervention_regions(ax, int_states: np.ndarray, fps: int):
    """Add background shading for intervention state regions."""
    t = np.arange(len(int_states)) / fps
    state_colors = {1: "orange", 2: "red"}
    for state, color in state_colors.items():
        mask = int_states == state
        changes = np.diff(mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])
        for s, e in zip(starts, ends):
            ax.axvspan(t[s], t[min(e, len(t) - 1)], alpha=0.15, color=color, linewidth=0)


def _add_mean_correction_band(ax, episodes: list[dict], fps: int):
    """Add a shaded band showing mean+std correction probability across episodes."""
    max_len = max(len(ep["int_state"]) for ep in episodes)
    all_probs = np.full((len(episodes), max_len), np.nan)
    for i, ep in enumerate(episodes):
        prob = compute_correction_probability(ep["int_state"], window=30)
        all_probs[i, :len(prob)] = prob

    t = np.arange(max_len) / fps
    mean_prob = np.nanmean(all_probs, axis=0)
    ax.fill_between(t, 0, mean_prob, alpha=0.15, color="red", label="_mean_correction")


def _plot_colored_trace(ax, t, values, int_states, base_color, alpha=0.5):
    """Plot a trace with segments colored by intervention state."""
    # For overlay mode, use base color but modulate intensity by state
    state_alphas = {3: alpha * 0.5, 1: alpha * 0.8, 2: alpha}
    prev_state = int_states[0]
    seg_start = 0

    for i in range(1, len(int_states)):
        if int_states[i] != prev_state or i == len(int_states) - 1:
            end = i + 1 if i == len(int_states) - 1 else i
            seg_alpha = state_alphas.get(prev_state, alpha * 0.5)
            ax.plot(
                t[seg_start:end], values[seg_start:end],
                color=base_color, alpha=seg_alpha, linewidth=0.6,
            )
            seg_start = i
            prev_state = int_states[i]


def _plot_colored_trace_single(ax, t, values, int_states):
    """Plot a single trace colored by intervention state."""
    state_colors = {0: "blue", 1: "orange", 2: "red", 3: "green"}
    prev_state = int_states[0]
    seg_start = 0

    for i in range(1, len(int_states)):
        if int_states[i] != prev_state or i == len(int_states) - 1:
            end = i + 1 if i == len(int_states) - 1 else i
            color = state_colors.get(prev_state, "gray")
            ax.plot(t[seg_start:end], values[seg_start:end], color=color, linewidth=1.0)
            seg_start = i
            prev_state = int_states[i]


def plot_multi_dataset(
    dataset_paths: list[Path],
    output_dir: Optional[Path] = None,
    joint_dims: Optional[list[int]] = None,
    fps: int = 30,
):
    """Plot all datasets side by side (one column per dataset)."""
    datasets = [load_lerobot_dataset(p) for p in dataset_paths]
    n_datasets = len(datasets)

    n_action_dims = datasets[0]["episodes"][0]["actions"].shape[1]
    if joint_dims is None:
        joint_dims = list(range(min(n_action_dims, 8)))

    n_joints = len(joint_dims)
    n_rows = 1 + n_joints

    fig, axes = plt.subplots(
        n_rows, n_datasets,
        figsize=(8 * n_datasets, 2.2 * n_rows),
        sharex=False, sharey="row",
    )
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)

    for col, data in enumerate(datasets):
        episodes = data["episodes"]
        ds_name = data["name"]
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(episodes))))

        # Top row: correction probability
        ax = axes[0, col]
        for i, ep in enumerate(episodes):
            int_states = ep["int_state"]
            t = np.arange(len(int_states)) / fps
            corr_prob = compute_correction_probability(int_states, window=30)
            ax.plot(t, corr_prob, color=colors[i % len(colors)], alpha=0.5, linewidth=0.8)

        _add_mean_correction_band(ax, episodes, fps)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.2)
        ax.set_title(f"{ds_name}\n({len(episodes)} eps)", fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Correction\nProbability", fontsize=9, fontweight="bold")

        # Joint rows
        for j_idx, dim in enumerate(joint_dims):
            ax = axes[1 + j_idx, col]
            for i, ep in enumerate(episodes):
                actions = ep["actions"][:, dim]
                int_states = ep["int_state"]
                t = np.arange(len(actions)) / fps
                _plot_colored_trace(ax, t, actions, int_states, colors[i % len(colors)], alpha=0.5)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(f"Joint {dim}", fontsize=8, fontweight="bold")

        axes[-1, col].set_xlabel("Time (s)", fontsize=9)

    # Legend on first column
    legend_patches = [
        mpatches.Patch(color="green", alpha=0.3, label="Autonomous"),
        mpatches.Patch(color="orange", alpha=0.3, label="Pre-intervention"),
        mpatches.Patch(color="red", alpha=0.3, label="Correction"),
    ]
    axes[0, 0].legend(handles=legend_patches, loc="upper right", fontsize=7)

    fig.suptitle("Books Task — Correction Data Across Iterations", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_dir is None:
        output_dir = dataset_paths[0].parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "books_multi_dataset_traces.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Filter and visualize LeRobot correction datasets")
    subparsers = parser.add_subparsers(dest="command")

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate trace plots with correction probability")
    plot_parser.add_argument(
        "datasets", nargs="+", type=Path,
        help="Path(s) to LeRobot dataset directories",
    )
    plot_parser.add_argument("--episode", "-e", type=int, default=None, help="Plot single episode")
    plot_parser.add_argument("--joints", "-j", nargs="+", type=int, default=None, help="Joint dims to plot")
    plot_parser.add_argument("--fps", type=int, default=30, help="Control frequency (default: 30)")
    plot_parser.add_argument("--output", "-o", type=Path, default=None, help="Output directory")

    args = parser.parse_args()

    if args.command == "plot":
        if len(args.datasets) > 1 and args.episode is None:
            plot_multi_dataset(args.datasets, args.output, args.joints, args.fps)
        else:
            for ds_path in args.datasets:
                plot_dataset_traces(ds_path, args.output, args.episode, args.joints, args.fps)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

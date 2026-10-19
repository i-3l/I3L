#!/usr/bin/env python3
"""Plot per-joint action deltas from a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

R1PRO_ACTION_NAMES = [
    "base_x",
    "base_y",
    "base_yaw",
    "torso_0",
    "torso_1",
    "torso_2",
    "torso_3",
    "left_arm_0",
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "left_arm_4",
    "left_arm_5",
    "left_arm_6",
    "left_gripper",
    "right_arm_0",
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "right_arm_4",
    "right_arm_5",
    "right_arm_6",
    "right_gripper",
]

INT_STATE_LABELS = {
    0: "human",
    1: "pre-int",
    2: "intervention",
    3: "policy",
}

INT_STATE_COLORS = {
    0: "#5ac85a",
    1: "#ffd43b",
    2: "#ff3b30",
    3: "#b0b0b0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot abs(action - observation.state[:action_dim]) for each joint in each "
            "episode of a LeRobot dataset."
        )
    )
    parser.add_argument("repo_id", help="Dataset repo id. Bare names use --default-namespace.")
    parser.add_argument(
        "--default-namespace",
        default="wensi-ai",
        help="Namespace used for bare dataset names. Default: wensi-ai.",
    )
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=None,
        help="Local LeRobot root. Defaults to HF_LEROBOT_HOME or ~/.cache/huggingface/lerobot.",
    )
    parser.add_argument("--download", action="store_true", help="Download missing dataset from HF.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "annotation" / "action_delta_magnitude",
        help="Output root directory.",
    )
    parser.add_argument(
        "--state-start",
        type=int,
        default=0,
        help="Start index in observation.state to compare against action. Default: 0.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override dataset FPS for x-axis.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only plot the first N episodes. Summary still follows this selection.",
    )
    parser.add_argument(
        "--no-per-episode-plots",
        action="store_true",
        help="Only write the aggregate heatmap and summary files.",
    )
    return parser.parse_args()


def default_lerobot_home() -> Path:
    env_home = os.environ.get("HF_LEROBOT_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".cache" / "huggingface" / "lerobot"


def normalize_repo_id(repo_id: str, default_namespace: str) -> str:
    if "/" in repo_id:
        return repo_id
    return f"{default_namespace}/{repo_id}"


def resolve_dataset_path(repo_id: str, lerobot_home: Path | None, download: bool) -> Path:
    if repo_id.startswith("/") or repo_id.startswith("."):
        raise ValueError(f"Expected a Hugging Face repo id, got path-like value: {repo_id}")

    home = (lerobot_home or default_lerobot_home()).expanduser().resolve()
    dataset_path = home.joinpath(*Path(repo_id).parts)
    if dataset_path.exists():
        return dataset_path
    if not download:
        raise FileNotFoundError(
            f"Dataset '{repo_id}' was not found at {dataset_path}. Use --download to fetch it."
        )

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Downloading requires huggingface_hub.") from exc

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} to {dataset_path}")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dataset_path)
    return dataset_path


def episode_index(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return 10**9


def load_info(dataset_path: Path) -> dict[str, Any]:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stack_array_column(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in series])


def int_state_values(series: pd.Series) -> np.ndarray:
    values = []
    for value in series:
        arr = np.asarray(value).reshape(-1)
        values.append(int(arr[0]) if arr.size else -1)
    return np.asarray(values, dtype=np.int32)


def load_episode_delta(parquet_path: Path, state_start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    columns = ["action", "observation.state"]
    try:
        df = pd.read_parquet(parquet_path, columns=columns + ["int_state"])
    except (KeyError, ValueError):
        df = pd.read_parquet(parquet_path, columns=columns)
    action = stack_array_column(df["action"])
    state = stack_array_column(df["observation.state"])
    state_end = state_start + action.shape[1]
    if state.shape[1] < state_end:
        raise ValueError(
            f"{parquet_path} has observation.state dim {state.shape[1]}, "
            f"cannot slice [{state_start}:{state_end}] for action dim {action.shape[1]}."
        )
    proprio = state[:, state_start:state_end]
    states = int_state_values(df["int_state"]) if "int_state" in df.columns else None
    return action - proprio, np.abs(action - proprio), states


def joint_names(action_dim: int) -> list[str]:
    if action_dim == len(R1PRO_ACTION_NAMES):
        return R1PRO_ACTION_NAMES
    return [f"joint_{idx:02d}" for idx in range(action_dim)]


def plot_episode(
    out_path: Path,
    episode_name: str,
    magnitude: np.ndarray,
    int_state: np.ndarray | None,
    names: list[str],
    fps: float | None,
) -> None:
    frames, action_dim = magnitude.shape
    x = np.arange(frames) / fps if fps else np.arange(frames)
    xlabel = "Time (s)" if fps else "Frame"

    fig_height = max(8.0, action_dim * 1.05)
    fig, axes = plt.subplots(action_dim, 1, figsize=(18, fig_height), sharex=True)
    if action_dim == 1:
        axes = [axes]

    shade_int_state_regions(axes, int_state, fps)
    for joint_idx, ax in enumerate(axes):
        ax.plot(x, magnitude[:, joint_idx], color="tab:blue", linewidth=0.8)
        ax.set_ylabel(names[joint_idx], fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=8)

    axes[-1].set_xlabel(xlabel)
    add_int_state_legend(axes[0], int_state)
    fig.suptitle(f"{episode_name}: |action - proprio|", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def iter_runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    if values.size == 0:
        return []
    runs = []
    start = 0
    current = int(values[0])
    for idx in range(1, values.size):
        value = int(values[idx])
        if value != current:
            runs.append((start, idx, current))
            start = idx
            current = value
    runs.append((start, values.size, current))
    return runs


def shade_int_state_regions(axes: list[plt.Axes], int_state: np.ndarray | None, fps: float | None) -> None:
    if int_state is None:
        return
    for start, end, state in iter_runs(int_state):
        color = INT_STATE_COLORS.get(state)
        if color is None:
            continue
        x_start = start / fps if fps else start
        x_end = end / fps if fps else end
        alpha = 0.28 if state == 2 else 0.12
        for ax in axes:
            ax.axvspan(x_start, x_end, color=color, alpha=alpha, linewidth=0)


def add_int_state_legend(ax: plt.Axes, int_state: np.ndarray | None) -> None:
    if int_state is None:
        return
    present_states = sorted(int(state) for state in np.unique(int_state))
    patches = [
        Patch(
            facecolor=INT_STATE_COLORS[state],
            alpha=0.28 if state == 2 else 0.12,
            label=f"{state}: {INT_STATE_LABELS[state]}",
        )
        for state in present_states
        if state in INT_STATE_COLORS
    ]
    if patches:
        ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=min(len(patches), 4))


def plot_heatmap(out_path: Path, rows: list[dict[str, Any]], names: list[str]) -> None:
    mean_abs = np.asarray([row["mean_abs_delta_per_joint"] for row in rows], dtype=np.float32)
    fig_width = max(12.0, len(names) * 0.55)
    fig_height = max(6.0, len(rows) * 0.32)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(mean_abs, aspect="auto", cmap="magma")
    ax.set_title("Mean |action - proprio| Per Episode")
    ax.set_xlabel("Joint")
    ax.set_ylabel("Episode")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["episode"] for row in rows], fontsize=8)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean absolute delta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    repo_id = normalize_repo_id(args.repo_id, args.default_namespace)
    dataset_path = resolve_dataset_path(repo_id, args.lerobot_home, args.download)
    info = load_info(dataset_path)
    fps = float(args.fps) if args.fps else (float(info["fps"]) if info.get("fps") else None)

    parquets = sorted((dataset_path / "data").glob("chunk-*/episode_*.parquet"), key=episode_index)
    if args.max_episodes is not None:
        parquets = parquets[: args.max_episodes]
    if not parquets:
        raise FileNotFoundError(f"No LeRobot parquet episodes found under {dataset_path / 'data'}")

    output_dir = args.output_dir.expanduser().resolve() / repo_id.replace("/", "__")
    episodes_dir = output_dir / "episodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_per_episode_plots:
        episodes_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    names = None
    for parquet_path in parquets:
        signed_delta, magnitude, int_state = load_episode_delta(parquet_path, args.state_start)
        if names is None:
            names = joint_names(magnitude.shape[1])
        episode_name = parquet_path.stem
        row = {
            "episode": episode_name,
            "num_frames": int(magnitude.shape[0]),
            "mean_abs_delta": float(magnitude.mean()),
            "max_abs_delta": float(magnitude.max()),
            "mean_abs_delta_per_joint": magnitude.mean(axis=0).astype(float).tolist(),
            "max_abs_delta_per_joint": magnitude.max(axis=0).astype(float).tolist(),
            "mean_signed_delta_per_joint": signed_delta.mean(axis=0).astype(float).tolist(),
        }
        if int_state is not None:
            row["int_state_counts"] = {
                INT_STATE_LABELS.get(state, str(state)): int(np.sum(int_state == state))
                for state in sorted(int(value) for value in np.unique(int_state))
            }
        rows.append(row)
        if not args.no_per_episode_plots:
            plot_episode(
                episodes_dir / f"{episode_name}_action_delta_magnitude.png",
                episode_name,
                magnitude,
                int_state,
                names,
                fps,
            )
        print(f"{episode_name}: mean_abs_delta={row['mean_abs_delta']:.6f} max_abs_delta={row['max_abs_delta']:.6f}")

    assert names is not None
    summary = {
        "repo_id": repo_id,
        "dataset_path": str(dataset_path),
        "fps": fps,
        "state_start": args.state_start,
        "delta_definition": f"action - observation.state[{args.state_start}:{args.state_start}+action_dim]",
        "joint_names": names,
        "num_episodes": len(rows),
        "episodes": rows,
    }
    write_json(output_dir / "action_delta_magnitude_summary.json", summary)

    csv_rows = []
    for row in rows:
        csv_row = {
            "episode": row["episode"],
            "num_frames": row["num_frames"],
            "mean_abs_delta": row["mean_abs_delta"],
            "max_abs_delta": row["max_abs_delta"],
        }
        for joint_idx, name in enumerate(names):
            csv_row[f"mean_abs_{name}"] = row["mean_abs_delta_per_joint"][joint_idx]
            csv_row[f"max_abs_{name}"] = row["max_abs_delta_per_joint"][joint_idx]
        csv_rows.append(csv_row)
    pd.DataFrame(csv_rows).to_csv(output_dir / "action_delta_magnitude_summary.csv", index=False)
    plot_heatmap(output_dir / "mean_action_delta_magnitude_heatmap.png", rows, names)

    print(f"\nSaved plots and summaries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

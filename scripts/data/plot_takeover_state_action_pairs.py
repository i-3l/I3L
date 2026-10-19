#!/usr/bin/env python3
"""Plot human takeover burden directly from LeRobot int_state labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_SPECS = [
    ("books-new", "Base Int 1", "wensi-ai/books-new-base-int-1-25"),
    ("books-new", "HG Int 2", "wensi-ai/books-new-hg-int-2-25"),
    ("books-new", "HG Int 3", "wensi-ai/books-new-hg-int-3-25"),
    ("books-new", "HG Int 4", "wensi-ai/books-new-hg-int-4-25"),
    ("bread", "Base Int 1", "wensi-ai/bread-base-int-1-25"),
    ("bread", "HG Int 2", "wensi-ai/bread-hg-int-2-25"),
    ("bread", "HG Int 3", "wensi-ai/bread-hg-int-3-25"),
    ("bread", "HG Int 4", "wensi-ai/bread-hg-int-4-25"),
    ("open-radio", "Base Int 1", "wensi-ai/or-base-int-1-25"),
    ("open-radio", "HG Int 2", "wensi-ai/or-hg-int-2-25"),
    ("open-radio", "HG Int 3", "wensi-ai/or-hg-int-3-25"),
    ("open-radio", "HG Int 4", "wensi-ai/or-hg-int-4-25"),
    ("popcorn", "Base Int 1", "wensi-ai/popcorn-base-int-1-25"),
    ("popcorn", "HG Int 2", "wensi-ai/popcorn-hg-int-2-25"),
    ("popcorn", "HG Int 3", "wensi-ai/popcorn-hg-int-3-25"),
    ("popcorn", "HG Int 4", "wensi-ai/popcorn-hg-int-4-25"),
]

ITERATION_LABELS = ["Base Int 1", "HG Int 2", "HG Int 3", "HG Int 4"]
TASK_LABELS = {
    "books-new": "Books",
    "bread": "Bread",
    "open-radio": "Open radio",
    "popcorn": "Popcorn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot human takeover burden from first-N LeRobot episodes. "
            "Human takeover state-action pairs are frames with int_state == 2."
        )
    )
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "lerobot",
        help="Local LeRobot dataset root. Default: ~/.cache/huggingface/lerobot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "annotation" / "takeover_state_action_pairs",
        help="Directory for the plot and metrics files.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=25,
        help="Use episode_000000 through episode_000024 by default.",
    )
    parser.add_argument(
        "--min-intervention-length",
        type=int,
        default=15,
        help="Ignore contiguous int_state==2 runs shorter than this many frames. Use 0 for raw counts.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing datasets instead of failing.",
    )
    return parser.parse_args()


def dataset_path(lerobot_home: Path, repo_id: str) -> Path:
    return lerobot_home.expanduser().joinpath(*Path(repo_id).parts)


def episode_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return 10**9


def episode_paths(root: Path) -> dict[int, Path]:
    paths = sorted((root / "data").glob("chunk-*/episode_*.parquet"), key=episode_number)
    return {episode_number(path): path for path in paths}


def int_state_values(path: Path) -> np.ndarray:
    df = pd.read_parquet(path, columns=["int_state"])
    values = []
    for value in df["int_state"]:
        arr = np.asarray(value).reshape(-1)
        values.append(int(arr[0]) if arr.size else -1)
    return np.asarray(values, dtype=np.int32)


def filter_short_interventions(states: np.ndarray, min_frames: int) -> np.ndarray:
    if min_frames <= 1 or states.size == 0:
        return states
    filtered = states.copy()
    mask = states == 2
    start = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            if idx - start < min_frames:
                filtered[start:idx] = -1
            start = None
    if start is not None and states.size - start < min_frames:
        filtered[start:] = -1
    return filtered


def summarize_dataset(
    lerobot_home: Path,
    task: str,
    iteration_label: str,
    repo_id: str,
    max_episodes: int,
    min_intervention_length: int,
) -> dict[str, Any]:
    root = dataset_path(lerobot_home, repo_id)
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found at {root}")

    by_index = episode_paths(root)
    missing = [idx for idx in range(max_episodes) if idx not in by_index]
    if missing:
        preview = ", ".join(f"episode_{idx:06d}" for idx in missing[:10])
        raise ValueError(f"{repo_id} missing required first episodes: {preview}")

    total_pairs = 0
    total_frames = 0
    episode_rows = []
    for idx in range(max_episodes):
        states = filter_short_interventions(
            int_state_values(by_index[idx]),
            min_intervention_length,
        )
        takeover_frames = int(np.sum(states == 2))
        num_frames = int(states.size)
        total_pairs += takeover_frames
        total_frames += num_frames
        episode_rows.append(
            {
                "episode": f"episode_{idx:06d}",
                "num_frames": num_frames,
                "num_takeover_state_action_pairs": takeover_frames,
                "takeover_percent": 100.0 * takeover_frames / num_frames if num_frames else 0.0,
            }
        )

    return {
        "task": TASK_LABELS.get(task, task),
        "task_key": task,
        "iteration_label": iteration_label,
        "repo_id": repo_id,
        "dataset_path": str(root),
        "num_episodes": max_episodes,
        "num_frames": total_frames,
        "num_takeover_state_action_pairs": total_pairs,
        "takeover_percent": 100.0 * total_pairs / total_frames if total_frames else 0.0,
        "min_intervention_length": min_intervention_length,
        "episodes": episode_rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "task_key",
        "iteration_label",
        "repo_id",
        "dataset_path",
        "num_episodes",
        "num_frames",
        "num_takeover_state_action_pairs",
        "takeover_percent",
        "min_intervention_length",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fieldnames} for row in rows])


def plot_metric(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    x = np.arange(len(ITERATION_LABELS))
    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for task_idx, task in enumerate(tasks):
        task_rows = [row for row in rows if row["task"] == task]
        by_label = {row["iteration_label"]: row for row in task_rows}
        y = [by_label[label][metric] if label in by_label else np.nan for label in ITERATION_LABELS]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.4,
            markersize=6,
            label=task,
            color=colors[task_idx % len(colors)],
        )

    ax.set_title(title)
    ax.set_xticks(x, ITERATION_LABELS)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Task", loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    rows = []
    errors = []
    for task, iteration_label, repo_id in DATASET_SPECS:
        try:
            row = summarize_dataset(
                args.lerobot_home,
                task,
                iteration_label,
                repo_id,
                args.max_episodes,
                args.min_intervention_length,
            )
        except Exception as exc:
            error = {"repo_id": repo_id, "error": str(exc)}
            errors.append(error)
            print(f"{repo_id}: ERROR {exc}")
            if not args.allow_missing:
                continue
        else:
            rows.append(row)
            print(
                f"{repo_id}: takeover_state_action_pairs="
                f"{row['num_takeover_state_action_pairs']} "
                f"takeover_percent={row['takeover_percent']:.2f}%"
            )

    if not rows:
        raise RuntimeError("No LeRobot datasets were summarized; cannot plot.")

    write_json(output_dir / "takeover_state_action_pairs.json", {"rows": rows, "errors": errors})
    write_csv(output_dir / "takeover_state_action_pairs.csv", rows)
    plot_metric(
        rows,
        output_dir / "takeover_percent.png",
        metric="takeover_percent",
        title="Human Takeover Burden",
        ylabel="episode frames in takeover (%)",
    )
    plot_metric(
        rows,
        output_dir / "takeover_state_action_pairs.png",
        metric="num_takeover_state_action_pairs",
        title="Human Takeover State-Action Pairs",
        ylabel="state-action pairs",
    )

    print(f"\nSaved percent plot to {output_dir / 'takeover_percent.png'}")
    print(f"Saved state-action pair plot to {output_dir / 'takeover_state_action_pairs.png'}")
    print(f"Saved metrics to {output_dir / 'takeover_state_action_pairs.json'}")
    print(f"Saved CSV to {output_dir / 'takeover_state_action_pairs.csv'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot which R1 Pro body parts are active during human interventions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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
    ("radio", "Base Int 1", "wensi-ai/radio-base-int-1-25"),
    ("radio", "HG Int 2", "wensi-ai/radio-hg-int-2-25"),
    ("radio", "HG Int 3", "wensi-ai/radio-hg-int-3-25"),
    ("radio", "HG Int 4", "wensi-ai/radio-hg-int-4-25"),
    ("popcorn", "Base Int 1", "wensi-ai/popcorn-base-int-1-25"),
    ("popcorn", "HG Int 2", "wensi-ai/popcorn-hg-int-2-25"),
    ("popcorn", "HG Int 3", "wensi-ai/popcorn-hg-int-3-25"),
    ("popcorn", "HG Int 4", "wensi-ai/popcorn-hg-int-4-25"),
]

TASK_LABELS = {
    "books-new": "Boxing Books up for Storage",
    "bread": "Prepare Make Ahead Breakfast Bowls",
    "radio": "Turning on Radio",
    "popcorn": "Make Microwave Popcorn",
}
PLOT_TASK_LABELS = {
    "books-new": "Boxing Books\nup for Storage",
    "bread": "Prepare Make-Ahead\nBreakfast Bowls",
    "radio": "Turning on\nRadio",
    "popcorn": "Make Microwave\nPopcorn",
}

# R1 Pro action layout: 3 base + 4 torso + 7 arm + 1 gripper per side.
BODY_PARTS = {
    "base": slice(0, 3),
    "torso": slice(3, 7),
    "left_arm": slice(7, 15),
    "right_arm": slice(15, 23),
}
ARM_JOINTS = {
    "left_arm": slice(7, 14),
    "right_arm": slice(15, 22),
}
GRIPPER_INDICES = {
    "left_arm": 14,
    "right_arm": 22,
}
BODY_LABELS = {
    "base": "Base",
    "torso": "Torso",
    "left_arm": "Left arm",
    "right_arm": "Right arm",
}
BODY_COLORS = {
    "base": "#B85C00",
    "torso": "#D2AA54",
    "left_arm": "#7E9878",
    "right_arm": "#356D73",
}
PLOT_TASK_ORDER = ["books-new", "popcorn", "bread", "radio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each task, plot the percentage of int_state==2 frames where each "
            "R1 Pro body part has at least one action-state delta above a threshold. "
            "Body parts are counted independently, so percentages may sum above 100%."
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
        default=REPO_ROOT / "outputs" / "annotation" / "intervention_body_part_usage",
        help="Directory for the plot and metrics files.",
    )
    parser.add_argument(
        "--base-threshold",
        type=float,
        default=0.003,
        help="Minimum absolute base delta-position action for an active base. Default: 0.003.",
    )
    parser.add_argument(
        "--joint-threshold",
        type=float,
        default=0.02,
        help="Minimum absolute arm joint delta for an active arm. Default: 0.02.",
    )
    parser.add_argument(
        "--torso-threshold",
        type=float,
        default=0.01,
        help="Minimum absolute torso joint delta for an active torso. Default: 0.01.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.75,
        help="Minimum absolute binary gripper action-state delta. Default: 0.75.",
    )
    parser.add_argument(
        "--state-start",
        type=int,
        default=0,
        help="Index in observation.state corresponding to action[0]. Default: 0.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional maximum number of episodes per dataset; default uses all episodes.",
    )
    parser.add_argument(
        "--min-intervention-length",
        type=int,
        default=1,
        help="Ignore contiguous intervention runs shorter than this many frames. Default: 1.",
    )
    parser.add_argument(
        "--min-active-frames",
        type=int,
        default=3,
        help="Ignore per-body activation runs shorter than this many consecutive frames. Default: 3.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing datasets instead of failing.",
    )
    return parser.parse_args()


def episode_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return 10**9


def intervention_mask(states: np.ndarray, min_frames: int) -> np.ndarray:
    mask = states == 2
    return filter_short_runs(mask, min_frames)


def filter_short_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    if min_frames <= 1:
        return mask

    filtered = mask.copy()
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start < min_frames:
                filtered[start:index] = False
            start = None
    if start is not None and mask.size - start < min_frames:
        filtered[start:] = False
    return filtered


def body_part_activity(
    delta_magnitude: np.ndarray,
    thresholds: dict[str, float],
    min_active_frames: int = 1,
) -> dict[str, np.ndarray]:
    if delta_magnitude.ndim != 2 or delta_magnitude.shape[1] < 23:
        raise ValueError(
            f"Expected R1 Pro action deltas with at least 23 columns; got {delta_magnitude.shape}."
        )
    active = {
        "base": np.any(
            delta_magnitude[:, BODY_PARTS["base"]] > thresholds["base"], axis=1
        ),
        "torso": np.any(
            delta_magnitude[:, BODY_PARTS["torso"]] > thresholds["torso"], axis=1
        ),
    }
    for arm, joint_indices in ARM_JOINTS.items():
        joint_active = np.any(delta_magnitude[:, joint_indices] > thresholds["joints"], axis=1)
        gripper_active = delta_magnitude[:, GRIPPER_INDICES[arm]] > thresholds["grippers"]
        active[arm] = joint_active | gripper_active
    return {
        body_part: filter_short_runs(mask, min_active_frames)
        for body_part, mask in active.items()
    }


def body_part_counts(
    delta_magnitude: np.ndarray,
    thresholds: dict[str, float],
    min_active_frames: int = 1,
) -> dict[str, int]:
    return {
        body_part: int(np.sum(mask))
        for body_part, mask in body_part_activity(
            delta_magnitude, thresholds, min_active_frames
        ).items()
    }


def segment_presence_counts(
    activity: dict[str, np.ndarray], intervention: np.ndarray
) -> tuple[dict[str, int], int]:
    counts = {body_part: 0 for body_part in BODY_PARTS}
    segments = 0
    padded = np.concatenate(([False], intervention, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    for start, end in zip(edges[::2], edges[1::2]):
        segments += 1
        for body_part, mask in activity.items():
            counts[body_part] += int(np.any(mask[start:end]))
    return counts, segments


def activation_event_counts(activity: dict[str, np.ndarray]) -> dict[str, int]:
    counts = {}
    for body_part, mask in activity.items():
        padded = np.concatenate(([False], mask, [False])).astype(np.int8)
        counts[body_part] = int(np.sum(np.diff(padded) == 1))
    return counts


def action_magnitude(action: np.ndarray, state: np.ndarray, state_start: int) -> np.ndarray:
    """Return direct base commands and action-state deltas for position-controlled joints."""
    state_end = state_start + action.shape[1]
    magnitude = np.abs(action).copy()
    magnitude[:, 3:] = np.abs(action[:, 3:] - state[:, state_start + 3 : state_end])
    return magnitude


def summarize_dataset(
    root: Path,
    task_key: str,
    iteration_label: str,
    repo_id: str,
    thresholds: dict[str, float],
    state_start: int,
    max_episodes: int | None,
    min_intervention_length: int,
    min_active_frames: int,
) -> dict[str, Any]:
    paths = sorted((root / "data").glob("chunk-*/episode_*.parquet"), key=episode_number)
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'data'}")

    counts = {body_part: 0 for body_part in BODY_PARTS}
    segment_counts = {body_part: 0 for body_part in BODY_PARTS}
    event_counts = {body_part: 0 for body_part in BODY_PARTS}
    intervention_frames = 0
    intervention_segments = 0
    total_frames = 0
    for path in paths:
        df = pd.read_parquet(path, columns=["action", "observation.state", "int_state"])
        action = np.stack(df["action"].map(np.asarray))
        state = np.stack(df["observation.state"].map(np.asarray))
        states = np.asarray(
            [np.asarray(value).reshape(-1)[0] for value in df["int_state"]], dtype=np.int32
        )
        if action.shape[1] != 23:
            raise ValueError(f"{path} has action dimension {action.shape[1]}; expected 23.")
        if state.shape[1] != 46:
            raise ValueError(f"{path} has state dimension {state.shape[1]}; expected 46.")
        state_end = state_start + action.shape[1]
        if state.shape[1] < state_end:
            raise ValueError(
                f"{path} has state dimension {state.shape[1]}, smaller than required {state_end}."
        )

        mask = intervention_mask(states, min_intervention_length)
        magnitude = action_magnitude(action, state, state_start)
        magnitude[~mask] = 0.0
        activity = body_part_activity(magnitude, thresholds, min_active_frames)
        episode_counts = {
            body_part: int(np.sum(active)) for body_part, active in activity.items()
        }
        episode_segment_counts, episode_segments = segment_presence_counts(activity, mask)
        episode_event_counts = activation_event_counts(activity)
        for body_part, count in episode_counts.items():
            counts[body_part] += count
            segment_counts[body_part] += episode_segment_counts[body_part]
            event_counts[body_part] += episode_event_counts[body_part]
        intervention_frames += int(np.sum(mask))
        intervention_segments += episode_segments
        total_frames += len(df)

    percentages = {
        body_part: 100.0 * count / intervention_frames if intervention_frames else 0.0
        for body_part, count in counts.items()
    }
    segment_percentages = {
        body_part: 100.0 * count / intervention_segments if intervention_segments else 0.0
        for body_part, count in segment_counts.items()
    }
    total_events = sum(event_counts.values())
    event_percentages = {
        body_part: 100.0 * count / total_events if total_events else 0.0
        for body_part, count in event_counts.items()
    }
    return {
        "task": TASK_LABELS.get(task_key, task_key),
        "task_key": task_key,
        "iteration_label": iteration_label,
        "repo_id": repo_id,
        "dataset_path": str(root),
        "num_episodes": len(paths),
        "num_frames": total_frames,
        "num_intervention_frames": intervention_frames,
        "num_intervention_segments": intervention_segments,
        "active_frame_counts": counts,
        "active_frame_percentages": percentages,
        "active_segment_counts": segment_counts,
        "active_segment_percentages": segment_percentages,
        "activation_event_counts": event_counts,
        "activation_event_percentages": event_percentages,
    }


def aggregate_tasks(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        grouped[row["task_key"]].append(row)

    task_rows = []
    for task_key, rows in grouped.items():
        denominator = sum(row["num_intervention_frames"] for row in rows)
        segment_denominator = sum(row["num_intervention_segments"] for row in rows)
        counts = {
            body_part: sum(row["active_frame_counts"][body_part] for row in rows)
            for body_part in BODY_PARTS
        }
        percentages = {
            body_part: 100.0 * count / denominator if denominator else 0.0
            for body_part, count in counts.items()
        }
        segment_counts = {
            body_part: sum(row["active_segment_counts"][body_part] for row in rows)
            for body_part in BODY_PARTS
        }
        segment_percentages = {
            body_part: 100.0 * count / segment_denominator if segment_denominator else 0.0
            for body_part, count in segment_counts.items()
        }
        event_counts = {
            body_part: sum(row["activation_event_counts"][body_part] for row in rows)
            for body_part in BODY_PARTS
        }
        total_events = sum(event_counts.values())
        event_percentages = {
            body_part: 100.0 * count / total_events if total_events else 0.0
            for body_part, count in event_counts.items()
        }
        task_rows.append(
            {
                "task": TASK_LABELS.get(task_key, task_key),
                "task_key": task_key,
                "num_datasets": len(rows),
                "num_episodes": sum(row["num_episodes"] for row in rows),
                "num_intervention_frames": denominator,
                "num_intervention_segments": segment_denominator,
                "active_frame_counts": counts,
                "active_frame_percentages": percentages,
                "active_segment_counts": segment_counts,
                "active_segment_percentages": segment_percentages,
                "activation_event_counts": event_counts,
                "activation_event_percentages": event_percentages,
            }
        )
    return task_rows


def write_csv(
    path: Path, rows: list[dict[str, Any]], thresholds: dict[str, float]
) -> None:
    fields = [
        "task",
        "task_key",
        "num_datasets",
        "num_episodes",
        "num_intervention_frames",
        "num_intervention_segments",
        "body_part",
        "threshold",
        "gripper_threshold",
        "active_frames",
        "frame_percentage",
        "active_segments",
        "segment_percentage",
        "activation_events",
        "event_percentage",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for body_part in BODY_PARTS:
                writer.writerow(
                    {
                        **{field: row[field] for field in fields[:6]},
                        "body_part": body_part,
                        "threshold": body_threshold(body_part, thresholds),
                        "gripper_threshold": (
                            thresholds["grippers"] if body_part in ARM_JOINTS else ""
                        ),
                        "active_frames": row["active_frame_counts"][body_part],
                        "frame_percentage": row["active_frame_percentages"][body_part],
                        "active_segments": row["active_segment_counts"][body_part],
                        "segment_percentage": row["active_segment_percentages"][body_part],
                        "activation_events": row["activation_event_counts"][body_part],
                        "event_percentage": row["activation_event_percentages"][body_part],
                    }
                )


def body_threshold(body_part: str, thresholds: dict[str, float]) -> float:
    if body_part == "base":
        return thresholds["base"]
    if body_part == "torso":
        return thresholds["torso"]
    return thresholds["joints"]


def plot_usage(
    rows: list[dict[str, Any]],
    output_path: Path,
    aggregation: str,
) -> None:
    if aggregation not in {"events", "segments"}:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    rows = sorted(rows, key=lambda row: PLOT_TASK_ORDER.index(row["task_key"]))
    body_parts = list(BODY_PARTS)
    count_key = (
        "activation_event_counts" if aggregation == "events" else "active_segment_counts"
    )
    task_labels = [PLOT_TASK_LABELS[row["task_key"]] for row in rows]
    y = np.arange(len(rows))
    text_colors = {
        "base": "white",
        "torso": "#29383D",
        "left_arm": "#29383D",
        "right_arm": "white",
    }
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Roboto"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.5,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.45, 1.42), constrained_layout=True)
        left = np.zeros(len(rows))
        for body_part in body_parts:
            percentages = []
            for row in rows:
                counts = np.array([row[count_key][part] for part in body_parts], dtype=float)
                percentages.append(100.0 * row[count_key][body_part] / counts.sum())
            percentages_array = np.asarray(percentages)
            ax.barh(
                y,
                percentages_array,
                left=left,
                height=0.70,
                color=BODY_COLORS[body_part],
                edgecolor="white",
                linewidth=0.7,
                label=BODY_LABELS[body_part],
            )
            for row_idx, percentage in enumerate(percentages_array):
                if percentage >= 6.0:
                    ax.text(
                        left[row_idx] + percentage / 2.0,
                        row_idx,
                        f"{percentage:.1f}",
                        ha="center",
                        va="center",
                        color=text_colors[body_part],
                        fontsize=6.2,
                        fontweight="bold",
                    )
                elif percentage > 0.0:
                    center = left[row_idx] + percentage / 2.0
                    ax.text(
                        center,
                        row_idx - 0.45,
                        f"{percentage:.1f}",
                        ha="center",
                        va="center",
                        color="#29383D",
                        fontsize=5.8,
                        fontweight="bold",
                    )
            left += percentages_array

        ax.set_yticks(y, task_labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Share of body-part activations (%)", labelpad=2)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            borderaxespad=0.15,
            ncols=4,
            frameon=False,
            handlelength=1.1,
            columnspacing=0.8,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.025)
        fig.savefig(output_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.025)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    if (
        args.base_threshold < 0
        or args.torso_threshold < 0
        or args.joint_threshold < 0
        or args.gripper_threshold < 0
    ):
        raise ValueError("Thresholds must be non-negative.")
    if args.state_start < 0:
        raise ValueError("--state-start must be non-negative.")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive.")
    if args.min_active_frames <= 0:
        raise ValueError("--min-active-frames must be positive.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {
        "base": args.base_threshold,
        "torso": args.torso_threshold,
        "joints": args.joint_threshold,
        "grippers": args.gripper_threshold,
    }
    dataset_rows = []
    errors = []
    for task_key, iteration_label, repo_id in DATASET_SPECS:
        root = args.lerobot_home.expanduser().joinpath(*Path(repo_id).parts)
        try:
            row = summarize_dataset(
                root,
                task_key,
                iteration_label,
                repo_id,
                thresholds,
                args.state_start,
                args.max_episodes,
                args.min_intervention_length,
                args.min_active_frames,
            )
        except Exception as exc:
            errors.append({"repo_id": repo_id, "error": str(exc)})
            print(f"{repo_id}: ERROR {exc}")
        else:
            dataset_rows.append(row)
            print(
                f"{repo_id}: intervention_segments={row['num_intervention_segments']} "
                + " ".join(
                    f"{part}={row['active_segment_percentages'][part]:.1f}%"
                    for part in BODY_PARTS
                )
            )

    if errors and not args.allow_missing:
        raise RuntimeError(
            f"Failed to read {len(errors)} dataset(s). Re-run with --allow-missing to skip them."
        )
    if not dataset_rows:
        raise RuntimeError("No LeRobot datasets were summarized; cannot plot.")

    task_rows = aggregate_tasks(dataset_rows)
    payload = {
        "primary_plot_metric": "segment_presence",
        "magnitude_definition": (
            "Base uses |action| because it is already a delta-position command; torso, arms, "
            "and grippers use |action - observation.state|."
        ),
        "metric_definitions": {
            "frame": "Number of qualifying active frames per body part.",
            "segment_presence": (
                "Each body part receives at most one vote per contiguous int_state==2 segment."
            ),
            "activation_event": (
                "Each qualifying consecutive activation run counts once, including multiple "
                "runs within one intervention segment."
            ),
        },
        "thresholds": thresholds,
        "state_start": args.state_start,
        "min_intervention_length": args.min_intervention_length,
        "min_active_frames": args.min_active_frames,
        "body_part_action_indices": {
            body_part: list(range(indices.start, indices.stop))
            for body_part, indices in BODY_PARTS.items()
        },
        "gripper_action_indices": GRIPPER_INDICES,
        "tasks": task_rows,
        "datasets": dataset_rows,
        "errors": errors,
    }
    with (output_dir / "intervention_body_part_usage.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    write_csv(output_dir / "intervention_body_part_usage.csv", task_rows, thresholds)
    plot_usage(
        task_rows,
        output_dir / "intervention_body_part_usage.png",
        "segments",
    )
    plot_usage(
        task_rows,
        output_dir / "intervention_body_part_usage_events.png",
        "events",
    )

    print(f"\nSaved plot and metrics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

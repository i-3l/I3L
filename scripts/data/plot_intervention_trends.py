#!/usr/bin/env python3
"""Plot intervention trend metrics across dataset iterations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from summarize_intervention_periods import (
    REPO_ROOT,
    normalize_repo_id,
    output_path_for_repo,
    resolve_dataset_path,
    summarize_dataset,
    write_json,
)


TASK_REPO_PREFIXES = {
    "open radio": "or",
    "open_radio": "or",
    "open-radio": "or",
}

TASK_DISPLAY_NAMES = {
    "books-new": "Books",
    "bread": "Bread",
    "radio": "Radio",
    "open radio": "Open radio",
    "open_radio": "Open radio",
    "open-radio": "Open radio",
    "popcorn": "Popcorn",
    "mug": "Mug",
    "pnp": "Strawberries",
    "peg": "Peg",
}

FULL_TASK_DISPLAY_NAMES = {
    "Books": "Boxing Books up for Storage",
    "Bread": "Prepare Make-Ahead Breakfast Bowls",
    "Radio": "Turning on Radio",
    "Open radio": "Open Fridge and Pick Up Radio",
    "Popcorn": "Make Microwave Popcorn",
    "Mug": "Mug Hanging",
    "Strawberries": "Pick-and-Place Strawberries",
    "Peg": "Peg Insertion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot takeover trends across base/hg intervention dataset iterations."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["bread", "radio", "books", "open_radio"],
        help="Task labels or repo prefixes. Default: bread radio books open_radio.",
    )
    parser.add_argument(
        "--iterations",
        nargs="+",
        default=["base-int-1-25", "hg-int-2-25", "hg-int-3-25"],
        help="Dataset suffixes in trend order.",
    )
    parser.add_argument(
        "--iteration-labels",
        nargs="+",
        default=["Base 1", "HG 2", "HG 3"],
        help="Labels shown on the x axis. Must match --iterations length.",
    )
    parser.add_argument(
        "--x-axis-labels",
        nargs="+",
        default=None,
        help="Optional display labels for the x axis. Must match --iterations length.",
    )
    parser.add_argument(
        "--final-success-rates",
        nargs="+",
        type=float,
        default=None,
        help="Optional final evaluation success rate for each task, in --tasks order.",
    )
    parser.add_argument(
        "--default-namespace",
        default="wensi-ai",
        help="Namespace used for generated repo ids. Default: wensi-ai.",
    )
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=None,
        help="Local LeRobot dataset root. Defaults to HF_LEROBOT_HOME or ~/.cache/huggingface/lerobot.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing datasets from Hugging Face.",
    )
    parser.add_argument(
        "--min-intervention-length",
        type=int,
        default=15,
        help="Ignore intervention periods shorter than this many frames. Default: 15.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "annotation" / "intervention_trends",
        help="Directory for summaries and plots.",
    )
    return parser.parse_args()


def dataset_repo_id(task: str, iteration: str, namespace: str) -> str:
    repo_prefix = TASK_REPO_PREFIXES.get(task, task)
    return normalize_repo_id(f"{repo_prefix}-{iteration}", namespace)


def display_task_name(task: str) -> str:
    return TASK_DISPLAY_NAMES.get(task, task.replace("_", " "))


def final_metric_row(task: str, success_rate: float, min_intervention_length: int) -> dict[str, Any]:
    return {
        "task": display_task_name(task),
        "iteration": "final",
        "iteration_label": "Final",
        "repo_id": "synthetic/final",
        "num_episodes": 0,
        "num_frames": 0,
        "num_takeovers": 0,
        "mean_takeovers_per_episode": 0.0,
        "median_takeovers_per_episode": 0.0,
        "intervention_frames": 0,
        "total_takeover_duration_sec": 0.0,
        "mean_takeover_duration_sec": 0.0,
        "mean_total_takeover_duration_per_episode_sec": 0.0,
        "median_total_takeover_duration_per_episode_sec": 0.0,
        "mean_takeover_percent_per_episode": 0.0,
        "median_takeover_percent_per_episode": 0.0,
        "task_success_rate_percent": success_rate,
        "min_intervention_length": min_intervention_length,
    }


def episode_takeover_percentages(summary: dict[str, Any]) -> list[float]:
    percentages = []
    for episode in summary["episodes"]:
        num_frames = episode["num_frames"]
        if num_frames <= 0:
            continue
        percentages.append(100.0 * episode["intervention_frames"] / num_frames)
    return percentages


def episode_takeover_durations(summary: dict[str, Any]) -> list[float]:
    fps = summary.get("fps")
    if not fps:
        return []
    return [episode["intervention_frames"] / fps for episode in summary["episodes"]]


def episode_takeover_counts(summary: dict[str, Any]) -> list[int]:
    return [episode["num_intervention_periods"] for episode in summary["episodes"]]


def task_success_rate(summary: dict[str, Any]) -> float:
    episodes = summary["episodes"]
    if not episodes:
        return 0.0
    successes = [
        episode.get("raw_intervention_frames", episode["intervention_frames"]) == 0
        for episode in episodes
    ]
    return 100.0 * float(np.mean(successes))


def metric_row(task: str, iteration: str, label: str, summary: dict[str, Any]) -> dict[str, Any]:
    lengths_frames = summary["intervention_lengths_frames"]
    fps = summary.get("fps")
    total_duration_sec = None
    mean_duration_sec = None
    if fps:
        total_duration_sec = summary["intervention_frames"] / fps
        if lengths_frames:
            mean_duration_sec = float(np.mean(lengths_frames) / fps)
        else:
            mean_duration_sec = 0.0

    percentages = episode_takeover_percentages(summary)
    episode_durations = episode_takeover_durations(summary)
    episode_counts = episode_takeover_counts(summary)
    return {
        "task": task,
        "iteration": iteration,
        "iteration_label": label,
        "repo_id": summary["repo_id"],
        "num_episodes": summary["num_episodes"],
        "num_frames": summary["num_frames"],
        "num_takeovers": summary["num_intervention_periods"],
        "mean_takeovers_per_episode": float(np.mean(episode_counts)) if episode_counts else 0.0,
        "median_takeovers_per_episode": float(np.median(episode_counts)) if episode_counts else 0.0,
        "intervention_frames": summary["intervention_frames"],
        "total_takeover_duration_sec": total_duration_sec,
        "mean_takeover_duration_sec": mean_duration_sec,
        "mean_total_takeover_duration_per_episode_sec": (
            float(np.mean(episode_durations)) if episode_durations else 0.0
        ),
        "median_total_takeover_duration_per_episode_sec": (
            float(np.median(episode_durations)) if episode_durations else 0.0
        ),
        "mean_takeover_percent_per_episode": float(np.mean(percentages)) if percentages else 0.0,
        "median_takeover_percent_per_episode": float(np.median(percentages)) if percentages else 0.0,
        "task_success_rate_percent": task_success_rate(summary),
        "min_intervention_length": summary["min_intervention_length"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "iteration",
        "iteration_label",
        "repo_id",
        "num_episodes",
        "num_frames",
        "num_takeovers",
        "mean_takeovers_per_episode",
        "median_takeovers_per_episode",
        "intervention_frames",
        "total_takeover_duration_sec",
        "mean_takeover_duration_sec",
        "mean_total_takeover_duration_per_episode_sec",
        "median_total_takeover_duration_per_episode_sec",
        "mean_takeover_percent_per_episode",
        "median_takeover_percent_per_episode",
        "task_success_rate_percent",
        "min_intervention_length",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_trends(
    rows: list[dict[str, Any]],
    iteration_labels: list[str],
    output_path: Path,
    x_axis_labels: list[str] | None = None,
    demo_baseline: float | None = None,
    horizontal: bool = False,
) -> None:
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    x = np.arange(len(iteration_labels))
    tick_labels = x_axis_labels or iteration_labels
    takeover_indices = [
        idx
        for idx, label in enumerate(iteration_labels)
        if any(
            row["iteration_label"] == label and row.get("iteration") != "final"
            for row in rows
        )
    ]
    takeover_x = x[takeover_indices]
    takeover_labels = [tick_labels[idx] for idx in takeover_indices]
    takeover_iteration_labels = [iteration_labels[idx] for idx in takeover_indices]
    metrics = [
        ("mean_takeovers_per_episode", "Takeover count", "Count / episode"),
        ("mean_takeover_percent_per_episode", "Takeover percentage", "Frames under takeover (%)"),
        (
            "mean_total_takeover_duration_per_episode_sec",
            "Takeover duration",
            "Seconds / episode",
        ),
    ]

    colors = ["#405B8C", "#66824B", "#A64D60", "#8870A3", "#D2AA54", "#705044", "#8FA7CE", "#BD8CA6"]
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Roboto"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
    }
    with plt.rc_context(style):
        fig, axes_grid = plt.subplots(
            1 if horizontal else 2, 4 if horizontal else 2,
            figsize=(7.1, 1.8) if horizontal else (3.45, 3.35),
            constrained_layout=True,
        )
        fig.get_layout_engine().set(h_pad=0.015, hspace=0.0)
        axes = axes_grid.ravel()
        task_offsets = np.linspace(-0.09, 0.09, len(tasks))
        compact_takeover_labels = [label.replace("Round ", "R") for label in takeover_labels]
        compact_tick_labels = [label.replace("Round ", "R") for label in tick_labels]

        for panel_idx, (ax, (metric, title, ylabel)) in enumerate(
            zip(axes[: len(metrics)], metrics, strict=True)
        ):
            values_by_round = [
                [
                    row[metric]
                    for row in rows
                    if row["iteration_label"] == label and row.get("iteration") != "final"
                ]
                for label in takeover_iteration_labels
            ]
            means = [float(np.mean(values)) for values in values_by_round]

            for task_idx, task in enumerate(tasks):
                task_rows = {
                    row["iteration_label"]: row
                    for row in rows
                    if row["task"] == task and row.get("iteration") != "final"
                }
                task_values = [task_rows[label][metric] for label in takeover_iteration_labels]
                ax.scatter(
                    takeover_x + task_offsets[task_idx],
                    task_values,
                    s=7,
                    color=colors[task_idx],
                    alpha=0.62,
                    edgecolors="none",
                    zorder=2,
                )

            ax.plot(
                takeover_x,
                means,
                marker="o",
                linewidth=1.7,
                markersize=3.2,
                color="#29383D",
                zorder=3,
            )
            ax.set_title(f"({chr(ord('a') + panel_idx)}) {title}", loc="left")
            ax.set_xticks(takeover_x, compact_takeover_labels)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)

        sr_ax = axes[-1]
        sr_ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="#F2F2F2", zorder=0)
        task_success = []
        for task_idx, task in enumerate(tasks):
            task_rows = [row for row in rows if row["task"] == task]
            by_label = {row["iteration_label"]: row for row in task_rows}
            sr_y = np.array(
                [
                    by_label[label].get("task_success_rate_percent", np.nan)
                    if label in by_label
                    else np.nan
                    for label in iteration_labels
                ]
            )
            task_success.append(sr_y)
        task_success_array = np.asarray(task_success)
        success_by_round = np.nanmean(task_success_array, axis=0)
        bar_colors = ["#69979A"] * (len(x) - 1) + ["#B85C00"]
        sr_ax.bar(x, success_by_round, width=0.62, color=bar_colors, edgecolor="none", zorder=2)
        for task_idx, sr_y in enumerate(task_success_array):
            sr_ax.scatter(
                x + task_offsets[task_idx],
                sr_y,
                s=7,
                color=colors[task_idx],
                alpha=1.0,
                edgecolors="none",
                zorder=3,
            )
        sr_ax.axvline(x[-1] - 0.5, color="#999999", linewidth=0.8, linestyle=":")
        sr_ax.set_title("(d) Success rate", loc="left")
        sr_ax.set_xticks(x, compact_tick_labels)
        sr_ax.set_ylabel("Success (%)")
        sr_ax.set_ylim(0, 105)
        if demo_baseline is not None:
            sr_ax.axhline(demo_baseline, color="#777777", linewidth=0.7,
                          linestyle=(0, (4, 2)), zorder=4)
            sr_ax.text(0.02, demo_baseline + 2.5, "Demo-only",
                       transform=sr_ax.get_yaxis_transform(), ha="left", va="bottom",
                       fontsize=5.0, color="#666666", zorder=5)
        sr_ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
        sr_ax.spines[["top", "right"]].set_visible(False)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=3.2,
                markerfacecolor=colors[task_idx],
                markeredgecolor="none",
                label=FULL_TASK_DISPLAY_NAMES.get(task, task),
            )
            for task_idx, task in enumerate(tasks)
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18 if horizontal else -0.15),
            ncols=4 if horizontal else 2,
            frameon=False,
            fontsize=5.3,
            handletextpad=0.25,
            columnspacing=0.8,
            labelspacing=0.3,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.025)
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    if len(args.iterations) != len(args.iteration_labels):
        raise ValueError("--iterations and --iteration-labels must have the same length.")
    if args.x_axis_labels is not None and len(args.iterations) != len(args.x_axis_labels):
        raise ValueError("--iterations and --x-axis-labels must have the same length.")
    if args.final_success_rates is not None and len(args.tasks) != len(args.final_success_rates):
        raise ValueError("--tasks and --final-success-rates must have the same length.")

    output_dir = args.output_dir.expanduser().resolve()
    summary_dir = output_dir / "summaries"
    rows = []
    errors = []

    for task in args.tasks:
        for iteration, label in zip(args.iterations, args.iteration_labels, strict=True):
            repo_id = dataset_repo_id(task, iteration, args.default_namespace)
            summary_path = output_path_for_repo(summary_dir, repo_id)
            try:
                dataset_path = resolve_dataset_path(repo_id, args.lerobot_home, args.download)
                summary = summarize_dataset(repo_id, dataset_path, args.min_intervention_length)
            except Exception as exc:
                error_payload = {
                    "task": task,
                    "iteration": iteration,
                    "repo_id": repo_id,
                    "error": str(exc),
                }
                write_json(summary_path, error_payload)
                errors.append(error_payload)
                print(f"{repo_id}: ERROR {exc}")
                continue

            write_json(summary_path, summary)
            row = metric_row(display_task_name(task), iteration, label, summary)
            rows.append(row)
            print(
                f"{repo_id}: takeovers={row['num_takeovers']} "
                f"mean_duration={row['mean_takeover_duration_sec']:.3f}s "
                f"mean_episode_takeover={row['mean_takeover_percent_per_episode']:.2f}%"
            )

    iteration_labels = list(args.iteration_labels)
    x_axis_labels = list(args.x_axis_labels) if args.x_axis_labels is not None else None
    if args.final_success_rates is not None:
        rows.extend(
            final_metric_row(task, success_rate, args.min_intervention_length)
            for task, success_rate in zip(args.tasks, args.final_success_rates, strict=True)
        )
        iteration_labels.append("Final")
        if x_axis_labels is not None:
            x_axis_labels.append("Final")

    if errors:
        write_json(output_dir / "errors.json", {"errors": errors})
    if not rows:
        raise RuntimeError("No datasets were summarized; cannot plot trends.")

    metrics_json = output_dir / "trend_metrics.json"
    metrics_csv = output_dir / "trend_metrics.csv"
    plot_path = output_dir / "intervention_trends.png"
    write_json(metrics_json, {"rows": rows, "errors": errors})
    write_csv(metrics_csv, rows)
    plot_trends(rows, iteration_labels, plot_path, x_axis_labels)

    print(f"\nSaved plot to {plot_path}")
    print(f"Saved metrics to {metrics_json}")
    print(f"Saved CSV to {metrics_csv}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

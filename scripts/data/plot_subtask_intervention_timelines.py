#!/usr/bin/env python3
"""Reconstruct Figure 3 intervention timelines from frame-level rollout data.

Remove intervention state runs shorter than 15 frames, pool frame counts in 50
whole-episode progress bins, and smooth the per-bin rate with Gaussian sigma=3.
These settings reproduce the archived Bread, Books, and Popcorn vector curves.
Use --reference-geometry to verify against extracted original PDF coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


REPO_ROOT = Path(__file__).resolve().parents[2]
ROUNDS = ["Round 1", "Round 2", "Round 3", "Round 4"]
# Displayed colors sampled from the original Figma plot with its saturation filter.
COLORS = ["#FFDB93", "#FFCF96", "#FA9501", "#C95700"]
LINESTYLES = [":", "-.", "--", "-"]

TASKS = {
    "bread": {
        "title": "Open Drawer and Serve Bread",
        "boundary_round": 4,
        "y_max": 42,
        "datasets": [
            "bread-base-int-1-25",
            "bread-hg-int-2-25",
            "bread-hg-int-3-25",
            "bread-hg-int-4-25",
        ],
    },
    "open_radio": {
        "title": "Open Fridge and Pick Up Radio",
        "boundary_round": 4,
        "y_max": 40,
        "datasets": [
            "or-base-int-1-25",
            "or-hg-int-2-25",
            "or-hg-int-3-25",
            "or-hg-int-4-25",
        ],
    },
    "radio": {
        "title": "Turning on Radio",
        "boundary_round": 4,
        "y_max": 50,
        "datasets": [
            "radio-base-int-1-25",
            "radio-hg-int-2-25",
            "radio-hg-int-3-25",
            "radio-hg-int-4-25",
        ],
    },
    "books": {
        "title": "Boxing Books for Storage",
        "boundary_round": 2,
        "y_max": 22,
        "datasets": [
            "books-new-base-int-1-25",
            "books-new-hg-int-2-25",
            "books-new-hg-int-3-25",
            "books-new-hg-int-4-25",
        ],
    },
    "popcorn": {
        "title": "Make Microwave Popcorn",
        "boundary_round": 2,
        "y_max": 47,
        "datasets": [
            "popcorn-base-int-1-25",
            "popcorn-hg-int-2-25",
            "popcorn-hg-int-3-25",
            "popcorn-hg-int-4-25",
        ],
    },
}
for key, title in [("mug", "Mug Hanging"), ("pnp", "Pick-and-Place Strawberries"), ("peg", "Peg Insertion")]:
    TASKS[key] = {
        "title": title,
        "boundary_round": 4,
        "y_max": None,
        "datasets": [f"{key}-{suffix}" for suffix in
                     ["base-int-1-25", "hg-int-2-25", "hg-int-3-25", "hg-int-4-25"]],
    }
DEFAULT_TASKS = ["bread", "open_radio", "books", "radio", "popcorn", "mug", "pnp", "peg"]
LABELS = {
    "pick up bowl": "Pick bowl", "pick up basket": "Pick basket",
    "pick up book 1": "Pick book 1", "pick up book 2": "Pick book 2",
    "pick up popcorn": "Pick popcorn",
    "navigate to table": "Navigate", "pick up radio": "Pick radio",
    "navigate to radio": "Navigate to radio", "poke button": "Press button",
    "pick up red strawberry": "Pick red", "place red strawberry": "Place red",
    "pick up purple strawberry": "Pick purple", "place purple strawberry": "Place purple",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 3 rates over normalized whole-episode progress."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASKS),
        default=DEFAULT_TASKS,
        help="Tasks to plot. Default: Figure 3 tasks whose frame data are available locally.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "annotation" / "wensi-ai",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "annotation" / "figure3_reproduced",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Normalized bins across each full episode. Default: 50.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=3.0,
        help="Gaussian smoothing sigma in normalized bins. Default: 3.0.",
    )
    parser.add_argument("--reference-geometry", type=Path,
                        help="archived_curve_geometry.json from reproduce_figure3_archived_curves.py")
    parser.add_argument("--archived-strawberry", type=Path,
                        help="Append the archived Strawberry curves from a geometry JSON; no local rollout data.")
    parser.add_argument("--png-only", action="store_true", help="Export PNGs without PDF copies.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG export resolution.")
    parser.add_argument("--y-max", type=float, help="Fix the upper y-axis limit for every task.")
    parser.add_argument("--no-bare-ylabel", action="store_true",
                        help="Omit the y-axis title on individual PNGs, retaining numeric ticks.")
    parser.add_argument("--align-bare-axes", action="store_true",
                        help="Use fixed canvas margins for aligned individual panels.")
    parser.add_argument("--hide-bare-yticks", action="store_true",
                        help="Make individual y tick labels transparent without changing layout.")
    parser.add_argument("--single-line-bare-xlabels", action="store_true",
                        help="Keep each subtask label on one visible line, preserving reserved margins.")
    return parser.parse_args()


def load_intervention_mask(dataset: Path, episode_index: int, num_frames: int,
                           demo_name: str | None = None) -> np.ndarray:
    if demo_name is not None:
        with h5py.File(dataset, "r") as source:
            states = np.asarray(source[f"data/{demo_name}/policy/int_state"]).reshape(-1)
        if len(states) != num_frames:
            raise ValueError(f"Annotation/data length mismatch: {dataset}: {demo_name}")
        return filter_interventions(states)
    parquet = dataset / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet.exists():
        matches = list((dataset / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Could not locate episode {episode_index} in {dataset}")
        parquet = matches[0]
    states = pd.read_parquet(parquet, columns=["int_state"])["int_state"].to_numpy()
    states = np.asarray(
        [int(np.asarray(value).reshape(-1)[0]) for value in states[:num_frames]],
        dtype=np.int32,
    )
    return filter_interventions(states)


def filter_interventions(states: np.ndarray) -> np.ndarray:
    mask = states == 2
    boundaries = np.r_[0, np.flatnonzero(np.diff(states) != 0) + 1, len(states)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 15:
            mask[start:end] = False
    return mask


def load_timeline(
    annotation_path: Path,
    bins: int,
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    subtasks = payload["subtasks"]
    dataset = Path(payload.get("dataset", payload.get("hdf5_path"))).expanduser()
    if "dataset" in payload and not dataset.exists():
        dataset = Path.home() / ".cache" / "huggingface" / "lerobot" / payload["repo_id"]
    intervention_counts = np.zeros(bins, dtype=float)
    frame_counts = np.zeros(bins, dtype=float)
    episode_boundaries = []
    summary_rows = []

    for name, episode in payload["episodes"].items():
        mask = load_intervention_mask(dataset, int(episode["episode_index"]), int(episode["num_frames"]),
                                      name.split("/")[-1] if "hdf5_path" in payload else None)
        bin_indices = np.clip(
            (np.arange(mask.size, dtype=float) / mask.size * bins).astype(int),
            0,
            bins - 1,
        )
        frame_counts += np.bincount(bin_indices, minlength=bins)
        intervention_counts += np.bincount(bin_indices[mask], minlength=bins)
        episode_boundaries.append(
            [
                100.0 * float(annotation["end_frame"]) / mask.size
                for annotation in episode["annotations"][:-1]
            ]
        )

    mean_timeline = 100.0 * np.divide(
        intervention_counts,
        frame_counts,
        out=np.zeros_like(intervention_counts),
        where=frame_counts > 0,
    )
    for index, subtask in enumerate(subtasks):
        intervention_frames = sum(
            int(episode["annotations"][index]["intervention_frames"] or 0)
            for episode in payload["episodes"].values()
        )
        total_frames = sum(
            int(episode["annotations"][index]["total_frames"] or 0)
            for episode in payload["episodes"].values()
        )
        summary_rows.append(
            {
                "subtask": subtask,
                "episodes": len(payload["episodes"]),
                "intervention_frame_rate_percent": 100.0 * intervention_frames / total_frames,
            }
        )
    return subtasks, mean_timeline, np.mean(episode_boundaries, axis=0), summary_rows


def style_axis(
    ax: plt.Axes,
    subtasks: list[str],
    boundaries: np.ndarray,
    show_ylabel: bool,
    y_max: float,
) -> None:
    for boundary in boundaries:
        ax.axvline(boundary, color="#CFCFCF", linewidth=0.7, zorder=0)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.6)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, y_max)
    edges = np.concatenate(([0.0], boundaries, [100.0]))
    labels = [LABELS.get(s, s[0].upper() + s[1:]) for s in subtasks]
    labels = [label.replace(" ", "\n", 1) if width < 15 else label
              for label, width in zip(labels, np.diff(edges), strict=True)]
    ax.set_xticks(0.5 * (edges[:-1] + edges[1:]), labels)
    ax.set_yticks(np.arange(0, y_max + 1e-9, 10))
    ax.tick_params(axis="x", labelsize=7, length=0, colors="#777777")
    ax.tick_params(axis="y", labelsize=7, length=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("Intervention rate (%)", fontsize=8)
    else:
        ax.set_yticklabels([])


def plot_task(
    ax: plt.Axes,
    subtasks: list[str],
    boundaries: np.ndarray,
    timelines: list[np.ndarray],
    smoothing: float,
    show_ylabel: bool,
    y_max: float,
    already_smoothed: bool = False,
) -> None:
    x = np.linspace(0, 100, timelines[0].size)
    for round_index, timeline in enumerate(timelines):
        curve = timeline if already_smoothed else gaussian_filter1d(timeline, sigma=smoothing)
        ax.plot(
            x,
            curve,
            color=COLORS[round_index],
            linestyle=LINESTYLES[round_index],
            linewidth=1.2,
            label=ROUNDS[round_index],
            zorder=2 + round_index,
        )
        ax.fill_between(
            x,
            0,
            curve,
            color=COLORS[round_index] if round_index < 3 else "#FFD5B1",
            alpha=0.08 if round_index < 3 else 1.0,
            linewidth=0,
        )
    style_axis(ax, subtasks, boundaries, show_ylabel, y_max)


def save_figure(fig: plt.Figure, path: Path, transparent: bool = False, png_only: bool = False,
                extra_artists: list | None = None, dpi: int = 300) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", transparent=transparent,
                bbox_extra_artists=extra_artists)
    if not png_only:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", transparent=transparent,
                    bbox_extra_artists=extra_artists)


def main() -> int:
    args = parse_args()
    if args.smoothing <= 0 or args.bins <= 1:
        raise ValueError("--smoothing must be positive and --bins must exceed one")
    if args.y_max is not None and args.y_max <= 0:
        raise ValueError("--y-max must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_data: dict[str, dict[str, Any]] = {}
    summary_rows = []
    timeline_rows = []

    for task_key in args.tasks:
        spec = TASKS[task_key]
        timelines = []
        boundaries = []
        subtasks = None
        for round_label, dataset_name in zip(ROUNDS, spec["datasets"], strict=True):
            annotation_path = args.annotation_root / dataset_name / "subtask_annotations.json"
            round_subtasks, timeline, round_boundaries, rows = load_timeline(
                annotation_path, args.bins
            )
            if subtasks is not None and round_subtasks != subtasks:
                raise ValueError(f"Subtasks differ across rounds for {task_key}")
            subtasks = round_subtasks
            timelines.append(timeline)
            boundaries.append(round_boundaries)
            summary_rows.extend({"task": task_key, "round": round_label, **row} for row in rows)
            smoothed = gaussian_filter1d(timeline, sigma=args.smoothing)
            for index, rate in enumerate(timeline):
                timeline_rows.append(
                    {
                        "task": task_key,
                        "round": round_label,
                        "episode_progress_percent": 100.0 * index / (timeline.size - 1),
                        "intervention_rate_percent": rate,
                        "smoothed_intervention_rate_percent": smoothed[index],
                    }
                )
        task_data[task_key] = {
            "subtasks": subtasks,
            # Reference round identified by matching archived divider coordinates.
            "boundaries": boundaries[spec["boundary_round"] - 1],
            "timelines": timelines,
        }
        if spec["y_max"] is None:
            peak = max(float(gaussian_filter1d(t, sigma=args.smoothing).max()) for t in timelines)
            spec["y_max"] = max(5, 5 * math.ceil(peak * 1.05 / 5))

    if args.reference_geometry:
        references = json.loads(args.reference_geometry.read_text())["tasks"]
        verification = {}
        for task_key, data in task_data.items():
            if task_key not in references:
                continue
            reference = references[task_key]
            errors = []
            for timeline, curve in zip(data["timelines"], reference["curves"], strict=True):
                predicted = gaussian_filter1d(timeline, sigma=args.smoothing)
                x = np.linspace(0, 100, timeline.size)
                np.testing.assert_allclose(x, curve["progress_percent"], rtol=0, atol=1e-5)
                np.testing.assert_allclose(predicted, curve["rate_percent"], rtol=0, atol=1e-5)
                errors.append(float(np.max(np.abs(predicted - curve["rate_percent"]))))
            np.testing.assert_allclose(data["boundaries"], reference["boundaries"], rtol=0, atol=1e-5)
            verification[task_key] = {"max_curve_error_percentage_points_by_round": errors,
                                      "source_sha256": reference["source_sha256"]}
        (output_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        print(json.dumps(verification, indent=2))

    if args.archived_strawberry:
        archived = json.loads(args.archived_strawberry.read_text())["tasks"]["a1_pnp_hard"]
        task_data["a1_pnp_hard"] = {
            "subtasks": archived["labels"],
            "boundaries": archived["boundaries"],
            "timelines": [np.asarray(curve["rate_percent"]) for curve in archived["curves"]],
            "already_smoothed": True,
        }
        TASKS["a1_pnp_hard"] = {"title": "Pick and Place Strawberry", "y_max": archived["y_max"]}

    if args.y_max is not None:
        for task_key in task_data:
            TASKS[task_key]["y_max"] = args.y_max

    provenance = {
        "bins": args.bins, "gaussian_sigma_bins": args.smoothing,
        "fixed_y_max": args.y_max,
        "minimum_intervention_frames": 15,
        "tasks": {
            key: {"source": "archived PDF geometry" if data.get("already_smoothed") else
                  ("local rollout HDF5" if key in ("mug", "pnp", "peg") else "local rollout Parquet"),
                  "datasets": TASKS[key].get("datasets", []),
                  "boundary_round": TASKS[key].get("boundary_round")}
            for key, data in task_data.items()
        },
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    for filename, rows in (
        ("intervention_rates_by_subtask.csv", summary_rows),
        ("intervention_timeline_values.csv", timeline_rows),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    style = {"font.family": "sans-serif", "font.sans-serif": ["Roboto"]}
    with plt.rc_context(style):
        plot_tasks = list(task_data)
        num_columns = 1 if len(plot_tasks) == 3 else min(2, len(plot_tasks))
        num_rows = math.ceil(len(plot_tasks) / num_columns)
        fig, axes = plt.subplots(
            num_rows,
            num_columns,
            figsize=((7.0 if num_columns == 1 else 5.0 * num_columns), 2.4 * num_rows),
            squeeze=False,
        )
        flat_axes = axes.ravel()
        for index, (ax, task_key) in enumerate(zip(flat_axes, plot_tasks)):
            data = task_data[task_key]
            plot_task(
                ax,
                data["subtasks"],
                data["boundaries"],
                data["timelines"],
                args.smoothing,
                True,
                TASKS[task_key]["y_max"],
                data.get("already_smoothed", False),
            )
            ax.set_title(TASKS[task_key]["title"], loc="left", fontsize=10, fontweight="bold")
        for ax in flat_axes[len(plot_tasks) :]:
            ax.set_visible(False)
        handles, labels = flat_axes[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), h_pad=1.5)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncols=4,
            frameon=False,
            fontsize=8,
        )
        save_figure(fig, output_dir / "intervention_timeline_preview", png_only=args.png_only, dpi=args.dpi)
        plt.close(fig)

        for task_key in plot_tasks:
            data = task_data[task_key]
            fig, ax = plt.subplots(figsize=(6.0, 1.55), constrained_layout=not args.align_bare_axes)
            if args.align_bare_axes:
                fig.subplots_adjust(left=0.045, right=0.995, bottom=0.22, top=0.80)
            plot_task(
                ax,
                data["subtasks"],
                data["boundaries"],
                data["timelines"],
                args.smoothing,
                True,
                TASKS[task_key]["y_max"],
                data.get("already_smoothed", False),
            )
            if args.no_bare_ylabel:
                ax.set_ylabel("")
            if args.hide_bare_yticks:
                # Retain label extents so constrained layout and tight export stay identical.
                for label in ax.get_yticklabels():
                    label.set_alpha(0)
            if args.single_line_bare_xlabels:
                for label in ax.get_xticklabels():
                    if "\n" in label.get_text():
                        text = label.get_text().replace("\n", " ")
                        position = (100, label.get_position()[1]) if text == "Close microwave" else label.get_position()
                        ax.text(*position, text,
                                transform=label.get_transform(), fontproperties=label.get_fontproperties(),
                                color=label.get_color(), ha="right" if text == "Close microwave" else label.get_ha(), va=label.get_va(),
                                in_layout=False)
                        label.set_alpha(0)
            if args.align_bare_axes:
                path = output_dir / f"timeline_{task_key}_bare"
                fig.savefig(path.with_suffix(".png"), dpi=args.dpi, transparent=True)
                if not args.png_only:
                    fig.savefig(path.with_suffix(".pdf"), transparent=True)
            else:
                save_figure(fig, output_dir / f"timeline_{task_key}_bare", transparent=True, png_only=args.png_only,
                            extra_artists=list(ax.texts) if args.single_line_bare_xlabels else None, dpi=args.dpi)
            plt.close(fig)

    print(f"Saved frame-level intervention timelines to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

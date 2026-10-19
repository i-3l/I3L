#!/usr/bin/env python3
"""Plot takeover metrics on the union of annotated subtask intervals.

JSONs select episodes and inclusive frame ranges. LeRobot/HDF5 int_state flags
deduplicate shared boundary frames and takeovers crossing subtask boundaries.
Annotation success flags are aggregated separately for the Figure 5 SR panel.
Outputs are a preview and metrics, not paper replacements.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import h5py
import numpy as np
from matplotlib.lines import Line2D

from summarize_intervention_periods import int_state_values, intervention_periods


ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    "books-new": "Boxing Books up for Storage",
    "bread": "Open Drawer and Serve Bread",
    "radio": "Turning on Radio",
    "or": "Open Fridge and Pick Up Radio",
    "popcorn": "Make Microwave Popcorn",
    "mug": "Mug Hanging",
    "pnp": "Pick-and-Place Strawberries",
    "peg": "Peg Insertion",
}
ROUNDS = ["base-int-1-25", "hg-int-2-25", "hg-int-3-25", "hg-int-4-25"]


def episode_metrics(states, annotations, fps, min_frames):
    """Count unique frames/runs; never sum overlapping subtask summaries."""
    selected = np.zeros(len(states), dtype=bool)
    for annotation in annotations:
        start, end = annotation["start_frame"], annotation["end_frame"]
        if start is None or end is None or not 0 <= start <= end < len(states):
            raise ValueError("Incomplete or out-of-range subtask annotation")
        selected[start:end + 1] = True
    if not selected.any() or fps <= 0:
        raise ValueError("Empty annotation coverage or invalid FPS")
    # Apply the existing 15-frame filter before clipping to annotated ranges.
    active = np.zeros(len(states), dtype=bool)
    for period in intervention_periods(states, min_frames):
        active[period["start_frame"]:period["end_frame"] + 1] = True
    active &= selected
    runs = intervention_periods(np.where(active, 2, 3), 1)
    return {
        "num_frames": len(states),
        "annotated_frames": int(selected.sum()),
        "intervention_frames": int(active.sum()),
        "takeovers": len(runs),
        "takeover_percent": 100 * float(active.sum()) / selected.sum(),
        "duration_seconds": float(active.sum()) / fps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path,
                        default=ROOT / "outputs/annotation/wensi-ai")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/annotation/annotation_takeover_trends")
    parser.add_argument("--min-frames", type=int, default=15)
    args = parser.parse_args()
    if args.min_frames < 1:
        parser.error("--min-frames must be positive")
    rows = []
    for task, title in TASKS.items():
        for round_index, suffix in enumerate(ROUNDS, 1):
            path = args.annotation_root / f"{task}-{suffix}" / "subtask_annotations.json"
            payload = json.loads(path.read_text())
            if payload.get("frame_interval") != "inclusive":
                raise ValueError(f"Unsupported frame interval: {path}")
            dataset = Path(payload.get("dataset", payload.get("hdf5_path"))).expanduser()
            if "dataset" in payload and not dataset.exists():
                dataset = Path.home() / ".cache/huggingface/lerobot" / payload["repo_id"]
            episodes = []
            for name, episode in payload["episodes"].items():
                if "hdf5_path" in payload:
                    with h5py.File(dataset, "r") as source:
                        states = np.asarray(source[f"data/{name.split('/')[-1]}/policy/int_state"]).reshape(-1)
                else:
                    matches = list((dataset / "data").glob(
                        f"chunk-*/episode_{episode['episode_index']:06d}.parquet"))
                    if len(matches) != 1:
                        raise ValueError(f"Expected one Parquet for {path}: {name}")
                    states = int_state_values(matches[0])
                if len(states) != episode["num_frames"]:
                    raise ValueError(f"Annotation/data length mismatch: {path}: {name}")
                metrics = episode_metrics(states, episode["annotations"],
                                          payload["fps"], args.min_frames)
                success_flags = [a.get("success") for a in episode["annotations"]]
                if not success_flags or any(type(flag) is not bool for flag in success_flags):
                    raise ValueError(f"Missing annotation success flags: {path}: {name}")
                episodes.append({"episode": name, "annotation_success": all(success_flags), **metrics})
            if not episodes:
                raise ValueError(f"No annotated episodes: {path}")
            row = {
                "task": title, "round": round_index,
                "annotation": str(path.resolve()),
                "annotation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "dataset": str(dataset), "episodes": episodes,
                "annotation_sr": 100 * float(np.mean([e["annotation_success"] for e in episodes])),
                **{key: float(np.mean([e[key] for e in episodes]))
                   for key in ("takeovers", "takeover_percent", "duration_seconds")},
            }
            rows.append(row)
            print(f"{task} R{round_index}: {len(episodes)} episodes; "
                  f"takeovers={row['takeovers']:.3f}, "
                  f"frames={row['takeover_percent']:.3f}%, "
                  f"duration={row['duration_seconds']:.3f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"source": "annotation interval union + LeRobot/HDF5 int_state",
              "minimum_run_frames": args.min_frames, "rows": rows}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    colors = ["#5B8DB8", "#67A982", "#D9826B", "#8E7DB8", "#E5AA52", "#8B6F61", "#579A9B", "#B27C9C"]
    with plt.rc_context({"font.family": "serif", "font.size": 8}):
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65))
        for ax, key, title, ylabel in zip(
                axes, ("takeovers", "takeover_percent", "duration_seconds"),
                ("(a) Takeovers", "(b) Takeover frames", "(c) Takeover duration"),
                ("per annotated episode", "annotated frames (%)", "seconds / annotated episode")):
            values = np.array([[r[key] for r in rows if r["task"] == task]
                               for task in TASKS.values()])
            for index, series in enumerate(values):
                ax.scatter(np.arange(1, 5) + np.linspace(-.09, .09, len(TASKS))[index],
                           series, color=colors[index], alpha=.68, s=13, zorder=3)
            ax.plot(range(1, 5), values.mean(axis=0), "o-", color="#264C63", zorder=4)
            ax.set(title=title, ylabel=ylabel, xticks=range(1, 5),
                   xticklabels=["R1", "R2", "R3", "R4"])
            ax.grid(axis="y", color="#D9D9D9", linewidth=.6)
            ax.spines[["top", "right"]].set_visible(False)
        handles = [Line2D([], [], marker="o", linestyle="none", color=color,
                          label=title, markersize=4)
                   for color, title in zip(colors, TASKS.values())]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                   fontsize=7)
        fig.tight_layout(rect=(0, .23, 1, 1))
        for extension in ("png", "pdf"):
            fig.savefig(args.output_dir / f"takeover_trends.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render Figure 5 in the original style using annotation-guided takeover metrics.

Existing five-task SR values are retained (Open Radio final is 90%). Mug
R1--R4 SR is the fraction of episodes with every annotation marked successful;
these flags mean intervention-free annotation intervals. Strawberries uses the
same annotation SR rule, as does Peg. Final SR is 92% for Mug and Peg, 92.8% for Strawberries.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from plot_intervention_trends import FULL_TASK_DISPLAY_NAMES, final_metric_row, plot_trends


ROOT = Path(__file__).resolve().parents[2]
METRICS = {
    "mean_takeovers_per_episode": "takeovers",
    "mean_takeover_percent_per_episode": "takeover_percent",
    "mean_total_takeover_duration_per_episode_sec": "duration_seconds",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizontal", action="store_true")
    args = parser.parse_args()
    rows = json.loads((ROOT / "outputs/annotation/intervention_trends/trend_metrics.json").read_text())["rows"]
    new_rows = json.loads((ROOT / "outputs/annotation/annotation_takeover_trends/metrics.json").read_text())["rows"]
    labels = list(dict.fromkeys(row["iteration_label"] for row in rows))
    for row in rows:
        if row["iteration"] == "final":
            if row["task"] == "Open radio":
                row["task_success_rate_percent"] = 90.0
            continue
        annotation_name = ("Open Drawer and Serve Bread" if row["task"] == "Bread"
                           else FULL_TASK_DISPLAY_NAMES[row["task"]])
        annotation = next(r for r in new_rows
                          if r["task"] == annotation_name
                          and r["round"] == labels.index(row["iteration_label"]) + 1)
        for plotted, computed in METRICS.items():
            row[plotted] = annotation[computed]
    for task, short_name, final_sr in [("mug", "Mug", 92.0), ("pnp", "Strawberries", 92.8), ("peg", "Peg", 92.0)]:
        task_rows = [r for r in new_rows if r["task"] == FULL_TASK_DISPLAY_NAMES[short_name]]
        assert [r["round"] for r in task_rows] == [1, 2, 3, 4]
        for annotation in task_rows:
            rows.append({
                "task": short_name, "iteration": f"hg-int-{annotation['round']}-25",
                "iteration_label": labels[annotation["round"] - 1],
                "task_success_rate_percent": annotation["annotation_sr"],
                **{plotted: annotation[computed] for plotted, computed in METRICS.items()},
            })
        rows.append(final_metric_row(task, final_sr, 15))
        print(short_name, "SR", [r["annotation_sr"] for r in task_rows], "Final", final_sr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plot_metrics.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    plot_trends(rows, labels, args.output_dir / "intervention_trends.png",
                ["Round 1", "Round 2", "Round 3", "Round 4", "Final"],
                demo_baseline=float(np.mean([55.6, 30.4, 68, 48, 76, 52, 80, 56])),
                horizontal=args.horizontal)
    for key in METRICS:
        means = [np.mean([r[key] for r in rows if r["iteration_label"] == label])
                 for label in labels[:-1]]
        print(key, "means", means, "R1-R4 decrease (%)", 100 * (1 - means[-1] / means[0]))
    print("Eight-task SR", [np.mean([r["task_success_rate_percent"] for r in rows
                                  if r["iteration_label"] == label]) for label in labels])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recover Figure 3 curve geometry from the archived vector PDFs.

This reproduces published drawing coordinates, not the raw-rollout estimator.
Source: corgiTrax/2026_arxiv_IIIL_paper, commit 537204f8dda3714e2b4b1e98d2a086174c70bf81.
The parser deliberately supports only these pinned Matplotlib PDF assets.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zlib

import matplotlib.pyplot as plt
import numpy as np


COMMIT = "537204f8dda3714e2b4b1e98d2a086174c70bf81"
TASKS = {
    "bread": "Open Drawer and Serve Bread",
    "books": "Boxing Books for Storage",
    "popcorn": "Make Microwave Popcorn",
    "a1_pnp_hard": "Pick and Place Strawberry",
}
NUMBER = r"-?\d+(?:\.\d+)?"
POINT = re.compile(rf"({NUMBER}) ({NUMBER}) [ml]")


def extract(pdf, grid_step):
    stream = re.search(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.S).group(1)
    content = zlib.decompress(stream).decode("latin1")
    clip = re.search(rf"({NUMBER}) ({NUMBER}) ({NUMBER}) ({NUMBER}) re W n", content)
    x0, y0, width, height = map(float, clip.groups())
    curves = []
    boundaries = []
    grid_y = []
    for block in content.split("\nS\n")[:-1]:
        # Last path before the stroke operator; earlier paths may be filled areas.
        start = list(re.finditer(rf"^{NUMBER} {NUMBER} m$", block, re.M))[-1].start()
        points = np.array(POINT.findall(block[start:]), dtype=float)
        if len(points) == 50 and np.all(np.diff(points[:, 0]) > 0):
            rgb = re.findall(rf"({NUMBER})\s+({NUMBER})\s+({NUMBER}) RG", block)[-1]
            curves.append({"points_pdf": points.tolist(), "color": list(map(float, rgb))})
        elif len(points) == 2:
            if np.isclose(points[0, 0], points[1, 0]):
                boundaries.append(float(points[0, 0]))
            elif np.isclose(points[0, 1], points[1, 1]):
                grid_y.append(float(points[0, 1]))
    assert len(curves) == 4, f"Expected four 50-point paths, got {len(curves)}"
    # Tick interval verified against each corresponding labeled source PDF.
    spacing = float(np.median(np.diff(sorted(set(grid_y)))))
    for curve in curves:
        points = np.array(curve["points_pdf"])
        curve["progress_percent"] = (100 * (points[:, 0] - x0) / width).tolist()
        curve["rate_percent"] = (grid_step * (points[:, 1] - y0) / spacing).tolist()
    labels = re.findall(r"\[ \(([^()]*)\) \] TJ", content)
    return {
        "clip_pdf": [x0, y0, width, height],
        "points_per_percentage": spacing / grid_step,
        "grid_step": grid_step,
        "y_max": height / spacing * grid_step,
        "boundaries": [100 * (x - x0) / width for x in boundaries],
        "labels": labels,
        "curves": curves,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/annotation/figure3_exact"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source_repo": "https://github.com/corgiTrax/2026_arxiv_IIIL_paper", "commit": COMMIT, "tasks": {}}
    rows = []
    for task in TASKS:
        for extension in ("pdf", "png"):
            name = f"fig_timeline_{task}_bare.{extension}"
            data = subprocess.check_output([
                "git", "-C", str(args.source_repo), "show", f"{COMMIT}:figures/timelines/{name}"
            ])
            target = args.output_dir / name
            target.write_bytes(data)
            assert target.read_bytes() == data
            if extension == "pdf":
                geometry = extract(data, 5 if task == "books" else 10)
                geometry["source_sha256"] = hashlib.sha256(data).hexdigest()
        manifest["tasks"][task] = geometry
        for index, curve in enumerate(geometry["curves"], 1):
            rows.extend({"task": task, "round": index, "episode_progress_percent": x, "intervention_rate_percent": y}
                        for x, y in zip(curve["progress_percent"], curve["rate_percent"], strict=True))
    (args.output_dir / "archived_curve_geometry.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.output_dir / "archived_curve_values.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "path.simplify": False})
    fig, axes = plt.subplots(len(TASKS), 1, figsize=(9, 2.2 * len(TASKS)))
    for ax, (task, title) in zip(axes, TASKS.items(), strict=True):
        geometry = manifest["tasks"][task]
        for boundary in geometry["boundaries"]:
            ax.axvline(boundary, color="#bfbfbf", linewidth=0.5)
        for index, curve in enumerate(geometry["curves"]):
            x, y = curve["progress_percent"], curve["rate_percent"]
            ax.fill_between(x, y, color=curve["color"], alpha=0.08 if index < 3 else 0.18)
            ax.plot(x, y, color=curve["color"], linestyle=[":", "-.", "--", "-"][index],
                    linewidth=1.2, label=f"Round {index + 1}")
            # Round-trip coordinates must match the PDF within its six-decimal precision.
            recovered = np.column_stack((np.array(x) * geometry["clip_pdf"][2] / 100 + geometry["clip_pdf"][0],
                                         np.array(y) * geometry["points_per_percentage"] + geometry["clip_pdf"][1]))
            assert np.max(np.abs(recovered - curve["points_pdf"])) < 1e-6
        edges = [0, *geometry["boundaries"], 100]
        ax.set_xticks((np.array(edges[:-1]) + edges[1:]) / 2, geometry["labels"])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, geometry["y_max"])
        ax.set_yticks(np.arange(0, geometry["y_max"], geometry["grid_step"]))
        ax.set_ylabel("Intervention rate (%)")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#e7e7e7", linewidth=0.5)
        ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
        ax.tick_params(length=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=1.8)
    for extension in ("pdf", "png"):
        fig.savefig(args.output_dir / f"recovered_curve_preview.{extension}", dpi=300)
    plt.close(fig)
    print(f"Recovered {len(rows)} original vector points; round-trip error < 1e-6 PDF points. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()

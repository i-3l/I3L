#!/usr/bin/env python3
"""Generate graphs from eval.sh output directory."""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_repeats(policy_dir):
    srs = []
    for f in sorted(policy_dir.glob("repeat_*.json")):
        with open(f) as fh:
            srs.append(json.load(fh)["SR"])
    return srs


def load_episode_results(video_dir):
    results = []
    for f in sorted(video_dir.glob("ep*_results.json")):
        with open(f) as fh:
            data = json.load(fh)
        ep_idx = int(f.stem.split("_")[0].replace("ep", ""))
        results.append({"ep": ep_idx, "sr": data["SR"], "video": str(f).replace("_results.json", ".mp4")})
    return results


def plot_sr_comparison(output_dir, base_srs, residual_srs=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    base_mean = np.mean(base_srs)
    base_std = np.std(base_srs) if len(base_srs) > 1 else 0

    names = ["Base Policy"]
    means = [base_mean]
    stds = [base_std]
    colors = ["#4C72B0"]

    if residual_srs:
        res_mean = np.mean(residual_srs)
        res_std = np.std(residual_srs) if len(residual_srs) > 1 else 0
        names.append("Residual Policy")
        means.append(res_mean)
        stds.append(res_std)
        colors.append("#55A868" if res_mean > base_mean else "#C44E52")

    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.5, width=0.5)

    for bar, m, s in zip(bars, means, stds):
        label = f"{m:.1%}"
        if s > 0:
            label += f" ± {s:.1%}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                label, ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Evaluation: Success Rate Comparison", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=base_mean, color="#4C72B0", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = output_dir / "sr_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def plot_per_episode(output_dir, base_eps, residual_eps=None):
    fig, ax = plt.subplots(figsize=(10, 4))

    base_results = [e["sr"] for e in base_eps]
    ax.bar([e["ep"] - 0.2 for e in base_eps], base_results, width=0.35,
           color="#4C72B0", label="Base", edgecolor="black", linewidth=0.3)

    if residual_eps:
        res_results = [e["sr"] for e in residual_eps]
        ax.bar([e["ep"] + 0.2 for e in residual_eps], res_results, width=0.35,
               color="#55A868", label="Residual", edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Success (0/1)", fontsize=11)
    ax.set_title("Per-Episode Success", fontsize=13)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Success"])
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = output_dir / "per_episode.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def plot_repeat_spread(output_dir, base_srs, residual_srs=None):
    if len(base_srs) <= 1 and (residual_srs is None or len(residual_srs) <= 1):
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    data = [base_srs]
    labels = ["Base"]
    colors = ["#4C72B0"]
    if residual_srs and len(residual_srs) > 0:
        data.append(residual_srs)
        labels.append("Residual")
        colors.append("#55A868")

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.4)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, d in enumerate(data):
        ax.scatter([i + 1] * len(d), d, color="black", zorder=3, s=30, alpha=0.7)

    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_title("Success Rate Distribution Across Repeats", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = output_dir / "repeat_spread.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    base_dir = output_dir / "base"
    residual_dir = output_dir / "residual"

    base_srs = load_repeats(base_dir) if base_dir.exists() else []
    residual_srs = load_repeats(residual_dir) if residual_dir.exists() else []

    if not base_srs:
        print("No base results found")
        return

    print(f"  Base SR: {base_srs}")
    if residual_srs:
        print(f"  Residual SR: {residual_srs}")

    # SR comparison bar chart
    plot_sr_comparison(output_dir, base_srs, residual_srs or None)

    # Repeat spread (box plot)
    plot_repeat_spread(output_dir, base_srs, residual_srs or None)

    # Per-episode results from video recordings
    base_eps = load_episode_results(base_dir / "videos") if (base_dir / "videos").exists() else []
    residual_eps = load_episode_results(residual_dir / "videos") if (residual_dir / "videos").exists() else []
    if base_eps:
        plot_per_episode(output_dir, base_eps, residual_eps or None)

    # Write summary JSON
    summary = {
        "base": {
            "success_rates": base_srs,
            "mean": float(np.mean(base_srs)),
            "std": float(np.std(base_srs)) if len(base_srs) > 1 else 0.0,
        }
    }
    if residual_srs:
        summary["residual"] = {
            "success_rates": residual_srs,
            "mean": float(np.mean(residual_srs)),
            "std": float(np.std(residual_srs)) if len(residual_srs) > 1 else 0.0,
        }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

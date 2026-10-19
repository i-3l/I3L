#!/usr/bin/env python3
"""Analyze subtask annotations and compute success/intervention statistics.

Reads subtask_annotations.json files and corresponding HDF5 data to compute
intervention rates, success rates, and takeover distributions per subtask.

Outputs:
  - CSV with per-episode, per-subtask statistics
  - Summary table printed to stdout
  - Takeover distribution plot (PNG)

Usage:
    python scripts/data/analyze_subtask_annotations.py \
        --annotations data/a1_pnp_hard/corrections_train/subtask_annotations.json \
                      data/a1_pnp_hard/correction2_hgdagger/corrections_train/subtask_annotations.json \
                      data/a1_pnp_hard/correction3_hgdagger/corrections_train/subtask_annotations.json \
                      data/a1_pnp_hard/correction4_hgdagger/subtask_annotations.json \
        --labels "Round 1" "Round 2" "Round 3" "Round 4" \
        --output outputs/pnp_hard_subtask_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import apply_style, COLORS, get_palette, FIGSIZE  # noqa: E402
apply_style()


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze subtask annotations.")
    parser.add_argument(
        "--annotations", nargs="+", type=Path, required=True,
        help="Paths to subtask_annotations.json files.",
    )
    parser.add_argument(
        "--labels", nargs="+", type=str, default=None,
        help="Labels for each annotation file (e.g., 'Round 1' 'Round 2').",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/pnp_hard_subtask_analysis"),
        help="Output directory for CSV and plots.",
    )
    return parser.parse_args()


def load_int_state_from_hdf5(hdf5_path: Path, demo_key: str, num_frames: int) -> np.ndarray | None:
    """Load intervention state from HDF5."""
    try:
        with h5py.File(hdf5_path, "r") as f:
            policy_group = f.get(f"data/{demo_key}/policy")
            if policy_group is None:
                return None
            int_state_ds = policy_group.get("int_state")
            if int_state_ds is None:
                return None
            raw = int_state_ds[:]
            if raw.ndim == 2 and raw.shape[1] == 1:
                raw = raw.reshape(-1)
            return raw[:num_frames].astype(np.int32)
    except (OSError, KeyError, ValueError) as exc:
        print(f"Warning: could not read {demo_key} from {hdf5_path}: {exc}", file=sys.stderr)
        return None


def compute_subtask_metrics(
    annotation: dict,
    int_state: np.ndarray | None,
) -> dict:
    """Compute intervention metrics for a single subtask annotation."""
    start = annotation.get("start_frame")
    end = annotation.get("end_frame")

    if start is None or end is None:
        return {
            "total_frames": None,
            "observed_frames": None,
            "intervention_frames": None,
            "on_policy_frames": None,
            "num_interventions": None,
            "intervention_occurred": None,
            "success": None,
            "intervention_rate": None,
        }

    start, end = int(start), int(end)
    total_frames = end - start + 1

    if int_state is None or int_state.size == 0:
        return {
            "total_frames": total_frames,
            "observed_frames": None,
            "intervention_frames": None,
            "on_policy_frames": None,
            "num_interventions": None,
            "intervention_occurred": None,
            "success": None,
            "intervention_rate": None,
        }

    state_end = min(end, int_state.size - 1)
    if start > state_end:
        return {
            "total_frames": total_frames,
            "observed_frames": 0,
            "intervention_frames": None,
            "on_policy_frames": None,
            "num_interventions": None,
            "intervention_occurred": None,
            "success": None,
            "intervention_rate": None,
        }

    states = int_state[start: state_end + 1]
    intervention_mask = states == 2
    intervention_frames = int(np.sum(intervention_mask))
    on_policy_frames = int(np.sum(states == 3))

    # Count intervention runs
    num_interventions = 0
    if intervention_frames > 0:
        starts = intervention_mask & np.concatenate([np.array([True]), ~intervention_mask[:-1]])
        num_interventions = int(np.sum(starts))

    intervention_occurred = intervention_frames > 0
    intervention_rate = intervention_frames / max(1, len(states))

    return {
        "total_frames": total_frames,
        "observed_frames": len(states),
        "intervention_frames": intervention_frames,
        "on_policy_frames": on_policy_frames,
        "num_interventions": num_interventions,
        "intervention_occurred": intervention_occurred,
        "success": not intervention_occurred,
        "intervention_rate": intervention_rate,
    }


def analyze_annotations(annotation_path: Path, label: str) -> list[dict]:
    """Analyze all episodes in one annotation file."""
    with open(annotation_path) as f:
        data = json.load(f)

    hdf5_path = Path(data.get("hdf5_path", ""))
    if not hdf5_path.exists():
        # Try relative to annotation file
        hdf5_path = annotation_path.parent / hdf5_path.name
    if not hdf5_path.exists():
        # Try finding any hdf5 in the same directory
        candidates = list(annotation_path.parent.glob("*.hdf5"))
        if candidates:
            hdf5_path = candidates[0]
    if not hdf5_path.exists():
        print(f"Warning: no HDF5 data found for {annotation_path}", file=sys.stderr)

    rows = []
    for ep_name, ep_data in data.get("episodes", {}).items():
        num_frames = ep_data.get("num_frames", 0)
        demo_key = ep_name.split("/")[-1] if "/" in ep_name else ep_name

        # Load int_state
        int_state = None
        if hdf5_path.exists():
            int_state = load_int_state_from_hdf5(hdf5_path, demo_key, num_frames)

        for annotation in ep_data.get("annotations", []):
            metrics = compute_subtask_metrics(annotation, int_state)
            rows.append({
                "round": label,
                "episode": ep_name,
                "subtask_index": annotation["subtask_index"],
                "subtask": annotation["subtask"],
                "start_frame": annotation.get("start_frame"),
                "end_frame": annotation.get("end_frame"),
                **metrics,
            })

    return rows


def print_summary_table(all_rows: list[dict], subtask_names: list[str]) -> str:
    """Print and return a summary table."""
    rounds = sorted(set(r["round"] for r in all_rows))

    lines = []
    header = f"{'Round':<12} {'Subtask':<28} {'Success%':>9} {'Interv%':>9} {'#Eps':>6} {'Avg IntFrames':>14}"
    lines.append(header)
    lines.append("-" * len(header))

    for round_label in rounds:
        round_rows = [r for r in all_rows if r["round"] == round_label]

        for subtask in subtask_names:
            st_rows = [r for r in round_rows if r["subtask"] == subtask and r["success"] is not None]
            if not st_rows:
                continue
            n = len(st_rows)
            success_rate = sum(1 for r in st_rows if r["success"]) / n * 100
            avg_int_rate = np.mean([r["intervention_rate"] for r in st_rows if r["intervention_rate"] is not None]) * 100
            avg_int_frames = np.mean([r["intervention_frames"] for r in st_rows if r["intervention_frames"] is not None])
            lines.append(f"{round_label:<12} {subtask:<28} {success_rate:>8.1f}% {avg_int_rate:>8.1f}% {n:>6} {avg_int_frames:>14.1f}")

        # Overall for this round
        valid_rows = [r for r in round_rows if r["success"] is not None]
        if valid_rows:
            n = len(valid_rows)
            overall_success = sum(1 for r in valid_rows if r["success"]) / n * 100
            observed_frames = sum(r["observed_frames"] for r in valid_rows)
            intervention_frames = sum(r["intervention_frames"] for r in valid_rows)
            overall_int_rate = intervention_frames / observed_frames * 100
            lines.append(f"{round_label:<12} {'OVERALL':<28} {overall_success:>8.1f}% {overall_int_rate:>8.1f}% {n:>6}")
        lines.append("")

    table = "\n".join(lines)
    print(table)
    return table


def plot_takeover_distribution(all_rows: list[dict], output_dir: Path) -> None:
    """Plot subtask success rate per round as grouped bars."""
    rounds = sorted(set(r["round"] for r in all_rows))
    subtask_names = []
    for r in all_rows:
        if r["subtask"] not in subtask_names:
            subtask_names.append(r["subtask"])
    subtask_names.sort(key=lambda s: next((r["subtask_index"] for r in all_rows if r["subtask"] == s), 0))

    fig, ax = plt.subplots(figsize=FIGSIZE["double_col_tall"])

    x = np.arange(len(subtask_names))
    n_rounds = len(rounds)
    width = 0.7 / n_rounds
    palette = get_palette(n_rounds)

    for i, round_label in enumerate(rounds):
        round_rows = [r for r in all_rows if r["round"] == round_label]
        success_rates = []
        for subtask in subtask_names:
            st_rows = [r for r in round_rows if r["subtask"] == subtask and r["success"] is not None]
            if st_rows:
                success_rates.append(sum(1 for r in st_rows if r["success"]) / len(st_rows) * 100)
            else:
                success_rates.append(0)

        offset = (i - (n_rounds - 1) / 2) * width
        bars = ax.bar(x + offset, success_rates, width, label=round_label, color=palette[i], edgecolor="white", linewidth=0.3)

    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel("Subtask")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("strawberry", "straw.").replace("pick up ", "pick\n").replace("place ", "place\n") for s in subtask_names], fontsize=8)
    ax.set_ylim(0, 109)
    ax.axhline(y=100, color=COLORS["light_gray"], linestyle="--", linewidth=0.5, zorder=0)
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("Subtask Success Rate by Correction Round")

    plt.savefig(output_dir / "subtask_success_by_round.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "subtask_success_by_round.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_dir / 'subtask_success_by_round.png'}")


def plot_intervention_timeline(all_rows: list[dict], output_dir: Path) -> None:
    """Plot intervention rate vs subtask progression for each round."""
    rounds = sorted(set(r["round"] for r in all_rows))
    subtask_names = []
    for r in all_rows:
        if r["subtask"] not in subtask_names:
            subtask_names.append(r["subtask"])
    subtask_names.sort(key=lambda s: next((r["subtask_index"] for r in all_rows if r["subtask"] == s), 0))

    fig, ax = plt.subplots(figsize=FIGSIZE["double_col_tall"])

    x = np.arange(len(subtask_names))
    n_rounds = len(rounds)
    width = 0.7 / n_rounds
    palette = get_palette(n_rounds)

    for i, round_label in enumerate(rounds):
        round_rows = [r for r in all_rows if r["round"] == round_label]
        int_rates = []
        for subtask in subtask_names:
            st_rows = [r for r in round_rows if r["subtask"] == subtask and r["intervention_rate"] is not None]
            if st_rows:
                int_rates.append(np.mean([r["intervention_rate"] for r in st_rows]) * 100)
            else:
                int_rates.append(0)

        offset = (i - (n_rounds - 1) / 2) * width
        ax.bar(x + offset, int_rates, width, label=round_label, color=palette[i], edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Subtask")
    ax.set_ylabel("Intervention Rate (%)")
    ax.set_title("Intervention Rate by Subtask and Correction Round")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("strawberry", "straw.").replace("pick up ", "pick\n").replace("place ", "place\n") for s in subtask_names], fontsize=8)
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0)

    plt.savefig(output_dir / "intervention_rate_by_subtask.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "intervention_rate_by_subtask.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_dir / 'intervention_rate_by_subtask.png'}")


def main() -> int:
    args = parse_args()

    labels = args.labels or [f"Round {i+1}" for i in range(len(args.annotations))]
    if len(labels) != len(args.annotations):
        raise ValueError("Number of --labels must match number of --annotations files.")

    all_rows = []
    for annotation_path, label in zip(args.annotations, labels):
        if not annotation_path.exists():
            print(f"Warning: {annotation_path} not found, skipping.", file=sys.stderr)
            continue
        rows = analyze_annotations(annotation_path, label)
        all_rows.extend(rows)
        print(f"Loaded {label}: {len(rows)} subtask entries from {annotation_path}")

    if not all_rows:
        print("No data to analyze.", file=sys.stderr)
        return 1

    subtask_names = []
    for r in all_rows:
        if r["subtask"] not in subtask_names:
            subtask_names.append(r["subtask"])
    subtask_names.sort(key=lambda s: next((r["subtask_index"] for r in all_rows if r["subtask"] == s), 0))

    # Output directory
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = output_dir / "subtask_analysis.csv"
    fieldnames = ["round", "episode", "subtask_index", "subtask", "start_frame", "end_frame",
                  "total_frames", "observed_frames", "intervention_frames", "on_policy_frames", "num_interventions",
                  "intervention_occurred", "success", "intervention_rate"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved CSV: {csv_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUBTASK ANALYSIS SUMMARY")
    print("=" * 80 + "\n")
    summary = print_summary_table(all_rows, subtask_names)

    # Save summary as text
    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary)

    # Plots
    plot_takeover_distribution(all_rows, output_dir)
    plot_intervention_timeline(all_rows, output_dir)

    print(f"\nAll outputs saved to: {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

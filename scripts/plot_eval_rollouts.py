#!/usr/bin/env python3
"""Generate per-rollout joint-level graphs comparing policy actions vs CuRobo oracle.

For each rollout, produces a figure with:
  - One row per joint (6 joints for A1)
  - Policy action vs CuRobo oracle action timeseries
  - Cosine distance and L1 distance timeseries with threshold lines

Expects HDF5 files produced by eval_pnp_hard.sh (policy=residual_eval + mode=collection),
which contain per-step base_action, oracle_action, and detector scores.

Usage:
  python scripts/plot_eval_rollouts.py --output-dir data/eval_pnp_hard/<timestamp>
  python scripts/plot_eval_rollouts.py --hdf5 path/to/rollouts.hdf5 --graphs-dir graphs/
"""

import argparse
import json
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

JOINT_NAMES = ["Joint 0", "Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Gripper"]
C_POLICY = "#4C72B0"
C_ORACLE = "#DD8452"
C_COSINE = "#8172B2"
C_L1 = "#C44E52"

COSINE_THRESHOLD = 0.720
L1_THRESHOLD = 0.480


def load_rollouts(hdf5_path):
    rollouts = []
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            return rollouts
        keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for k in keys:
            g = f["data"][k]
            entry = {"action": np.array(g["action"])}
            if "policy/base_action" in g:
                entry["base_action"] = np.array(g["policy/base_action"])
            if "policy/oracle_action" in g:
                entry["oracle_action"] = np.array(g["policy/oracle_action"])
            if "policy/detector_interrupt_scores" in g:
                entry["detector_scores"] = np.array(g["policy/detector_interrupt_scores"])
            if "reward" in g:
                entry["reward"] = np.array(g["reward"])
            rollouts.append(entry)
    return rollouts


def compute_metrics(base_action, oracle_action):
    """Compute per-step cosine distance and L1 distance."""
    T = len(base_action)
    cosine_dist = np.zeros(T)
    l1_dist = np.zeros(T)
    for t in range(T):
        a = base_action[t].flatten()
        b = oracle_action[t].flatten()
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm > 0 and b_norm > 0:
            cos_sim = np.dot(a, b) / (a_norm * b_norm)
            cosine_dist[t] = 1.0 - (cos_sim + 1.0) / 2.0
        else:
            cosine_dist[t] = 0.5
        l1_dist[t] = np.sum(np.abs(a - b))
    return cosine_dist, l1_dist


def plot_rollout(rollout, idx, out_path, success=None):
    base = rollout.get("base_action", rollout["action"])
    oracle = rollout.get("oracle_action")
    n_joints = min(base.shape[1], 7)
    has_oracle = oracle is not None

    n_rows = n_joints + (1 if has_oracle else 0)
    height_ratios = [1] * n_joints + ([1.4] if has_oracle else [])
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.5 * n_rows),
                             sharex=True, gridspec_kw={"height_ratios": height_ratios})
    if n_rows == 1:
        axes = [axes]
    T = len(base)

    # Joint rows
    for j in range(n_joints):
        ax = axes[j]
        if has_oracle:
            ax.plot(oracle[:, j], color=C_ORACLE, alpha=0.85, linewidth=1.8, label="CuRobo (oracle)")
        ax.plot(base[:, j], color=C_POLICY, alpha=0.85, linewidth=1.2, label="Policy")
        name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"Joint {j}"
        ax.set_ylabel(name, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend(fontsize=9, loc="upper right", ncol=2)

    # Metric row (cosine + L1 with thresholds)
    if has_oracle:
        ax = axes[n_joints]
        if "detector_scores" in rollout and rollout["detector_scores"].shape[0] == T:
            scores = rollout["detector_scores"]
            cosine_dist = scores[:, 0] if scores.ndim == 2 and scores.shape[1] >= 1 else np.zeros(T)
            l1_dist = scores[:, 1] if scores.ndim == 2 and scores.shape[1] >= 2 else np.zeros(T)
        else:
            cosine_dist, l1_dist = compute_metrics(base, oracle)

        # Shade regions where cosine exceeds threshold
        above_thresh = cosine_dist > COSINE_THRESHOLD
        in_region = False
        for t_idx in range(T):
            if above_thresh[t_idx] and not in_region:
                start = t_idx
                in_region = True
            elif not above_thresh[t_idx] and in_region:
                ax.axvspan(start, t_idx, alpha=0.1, color="red", linewidth=0)
                in_region = False
        if in_region:
            ax.axvspan(start, T, alpha=0.1, color="red", linewidth=0)

        ax.plot(cosine_dist, color=C_COSINE, alpha=0.9, linewidth=1.2, label="Cosine dist")
        ax.axhline(COSINE_THRESHOLD, color=C_COSINE, linestyle="--", alpha=0.7, linewidth=1.5,
                   label=f"Cosine thresh ({COSINE_THRESHOLD})")
        ax.set_ylabel("Cosine dist", fontsize=10, color=C_COSINE, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

        ax2 = ax.twinx()
        ax2.plot(l1_dist, color=C_L1, alpha=0.9, linewidth=1.2, label="L1 dist")
        ax2.axhline(L1_THRESHOLD, color=C_L1, linestyle="--", alpha=0.7, linewidth=1.5,
                    label=f"L1 thresh ({L1_THRESHOLD})")
        ax2.set_ylabel("L1 dist", fontsize=10, color=C_L1, fontweight="bold")
        ax2.tick_params(labelsize=8)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right", ncol=4)

    axes[-1].set_xlabel("Timestep", fontsize=10)

    status = ""
    if success is not None:
        status = " (SUCCESS)" if success else " (FAIL)"
    fig.suptitle(f"Rollout {idx}{status}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_summary(rollouts, out_path, policy_name=""):
    """Plot summary: avg cosine/L1 per rollout as bar chart."""
    avg_cosine = []
    avg_l1 = []
    successes = []
    for r in rollouts:
        base = r.get("base_action", r["action"])
        oracle = r.get("oracle_action")
        if oracle is None:
            continue
        if "detector_scores" in r and r["detector_scores"].shape[0] == len(base):
            scores = r["detector_scores"]
            if scores.ndim == 2 and scores.shape[1] >= 2:
                avg_cosine.append(scores[:, 0].mean())
                avg_l1.append(scores[:, 1].mean())
            else:
                c, l = compute_metrics(base, oracle)
                avg_cosine.append(c.mean())
                avg_l1.append(l.mean())
        else:
            c, l = compute_metrics(base, oracle)
            avg_cosine.append(c.mean())
            avg_l1.append(l.mean())
        reward = r.get("reward")
        if reward is not None:
            successes.append(reward.sum() > 0)
        else:
            successes.append(None)

    if not avg_cosine:
        return

    N = len(avg_cosine)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, N * 0.4), 6), sharex=True)

    x = np.arange(N)
    colors = []
    for s in successes:
        if s is True:
            colors.append("#55A868")
        elif s is False:
            colors.append("#C44E52")
        else:
            colors.append("#7f7f7f")

    ax1.bar(x, avg_cosine, color=colors, alpha=0.7)
    ax1.axhline(COSINE_THRESHOLD, color=C_COSINE, linestyle="--", alpha=0.7, label=f"Threshold ({COSINE_THRESHOLD})")
    ax1.set_ylabel("Avg Cosine Distance")
    ax1.legend(fontsize=8)
    ax1.set_title(f"{policy_name} — Per-Rollout Metrics vs CuRobo Oracle")

    ax2.bar(x, avg_l1, color=colors, alpha=0.7)
    ax2.axhline(L1_THRESHOLD, color=C_L1, linestyle="--", alpha=0.7, label=f"Threshold ({L1_THRESHOLD})")
    ax2.set_ylabel("Avg L1 Distance")
    ax2.set_xlabel("Rollout Index")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def process_policy_dir(policy_dir):
    """Generate graphs for a single policy directory."""
    policy_dir = Path(policy_dir)
    hdf5_path = policy_dir / "rollouts.hdf5"
    if not hdf5_path.exists():
        print(f"  Skipping {policy_dir.name}: no rollouts.hdf5")
        return

    graphs_dir = policy_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    policy_name = ""
    name_file = policy_dir / "policy_name.txt"
    if name_file.exists():
        policy_name = name_file.read_text().strip()

    print(f"  Processing {policy_dir.name} ({policy_name})...")
    rollouts = load_rollouts(str(hdf5_path))
    if not rollouts:
        print(f"    No rollouts found in HDF5")
        return

    # Per-rollout graphs
    for i, r in enumerate(rollouts):
        reward = r.get("reward")
        success = reward.sum() > 0 if reward is not None else None
        out = graphs_dir / f"rollout_{i:03d}.png"
        plot_rollout(r, i, str(out), success=success)

    # Summary graph
    plot_summary(rollouts, str(graphs_dir / "summary.png"), policy_name=policy_name)
    print(f"    Generated {len(rollouts)} rollout graphs + summary → {graphs_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate per-rollout eval graphs")
    parser.add_argument("--output-dir", type=str, help="Top-level eval output directory (scans for policy_* subdirs)")
    parser.add_argument("--hdf5", type=str, help="Single HDF5 file to plot")
    parser.add_argument("--graphs-dir", type=str, help="Output directory for graphs (used with --hdf5)")
    args = parser.parse_args()

    if args.hdf5:
        graphs_dir = Path(args.graphs_dir) if args.graphs_dir else Path(args.hdf5).parent / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        policy_name = Path(args.hdf5).parent.name
        rollouts = load_rollouts(args.hdf5)
        if not rollouts:
            print("No rollouts found")
            return
        for i, r in enumerate(rollouts):
            reward = r.get("reward")
            success = reward.sum() > 0 if reward is not None else None
            plot_rollout(r, i, str(graphs_dir / f"rollout_{i:03d}.png"), success=success)
        plot_summary(rollouts, str(graphs_dir / "summary.png"), policy_name=policy_name)
        print(f"Generated {len(rollouts)} rollout graphs + summary → {graphs_dir}")
    elif args.output_dir:
        output_dir = Path(args.output_dir)
        policy_dirs = sorted(output_dir.glob("policy_*"))
        if not policy_dirs:
            # Maybe it's a single policy dir
            if (output_dir / "rollouts.hdf5").exists():
                process_policy_dir(output_dir)
            else:
                print(f"No policy_* directories found in {output_dir}")
            return
        for pd in policy_dirs:
            process_policy_dir(pd)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

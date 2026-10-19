#!/usr/bin/env python3
"""
Plot residual effectiveness using per-step action traces from ResidualPolicyWrapper.

Requires --traces-path from eval_simple_residual.py (action_traces.json).
Also requires --hdf5 from eval rollout (has oracle_action from CuRobo).

Graphs:
  4b: Mean |base+residual - oracle| vs |base - oracle| per joint, shaded ±std
  1b: Single rollout with success/failure label (base, oracle, executed)
  5:  Single rollout of raw residual output per joint
"""

import argparse
import json
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

JOINT_NAMES = [f"Joint {i}" for i in range(6)]
C_BASE = "#4C72B0"
C_ORACLE = "#DD8452"
C_EXECUTED = "#55A868"
C_RESIDUAL_RAW = "#C44E52"


def load_traces_and_oracle(traces_path, hdf5_path):
    """Load action traces (from wrapper) and oracle actions (from HDF5).

    Returns list of episode dicts with keys:
      base (T,7), residual (T,7), final (T,7), gate (T,), oracle (T,7), success (bool)
    """
    with open(traces_path) as f:
        traces = json.load(f)

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
        episodes = []
        for i, (trace, dk) in enumerate(zip(traces, demo_keys)):
            grp = f["data"][dk]
            oracle = grp["policy/oracle_action"][:, :7].copy()
            reward = grp["reward"][:].copy()
            success = bool(reward.sum() > 0)

            base = np.array(trace["base"])[:, :7]
            residual = np.array(trace["residual"])[:, :7]
            final = np.array(trace["final"])[:, :7]
            gate = np.array(trace["gate"])

            # Detector scores (cosine + L1 disagreement between base and oracle)
            if "policy/detector_interrupt_scores" in grp:
                scores = grp["policy/detector_interrupt_scores"][:]
                cosine_scores = scores[:, 0].copy()
                l1_scores = scores[:, 1].copy()
            else:
                cosine_scores = None
                l1_scores = None

            T = min(len(base), len(oracle))
            ep = {
                "base": base[:T],
                "residual": residual[:T],
                "final": final[:T],
                "gate": gate[:T],
                "oracle": oracle[:T],
                "success": success,
                "idx": i,
            }
            if cosine_scores is not None:
                ep["cosine_scores"] = cosine_scores[:T]
                ep["l1_scores"] = l1_scores[:T]
            episodes.append(ep)
    return episodes


def plot_graph4b(episodes, output_dir):
    """Mean |base+residual - oracle| vs |base - oracle| per joint, shaded ±std."""
    n_joints = 6
    max_T = max(len(ep["base"]) for ep in episodes)

    err_base_all = []
    err_exec_all = []
    for ep in episodes:
        T = len(ep["base"])
        eb = np.abs(ep["base"][:, :n_joints] - ep["oracle"][:, :n_joints])
        ee = np.abs(ep["final"][:, :n_joints] - ep["oracle"][:, :n_joints])
        # Pad to max_T
        eb_pad = np.full((max_T, n_joints), np.nan)
        ee_pad = np.full((max_T, n_joints), np.nan)
        eb_pad[:T] = eb
        ee_pad[:T] = ee
        err_base_all.append(eb_pad)
        err_exec_all.append(ee_pad)

    err_base_all = np.array(err_base_all)   # (N, max_T, 6)
    err_exec_all = np.array(err_exec_all)   # (N, max_T, 6)

    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 18), sharex=True)
    ts = np.arange(max_T)

    for j in range(n_joints):
        ax = axes[j]
        bm = np.nanmean(err_base_all[:, :, j], axis=0)
        bs = np.nanstd(err_base_all[:, :, j], axis=0)
        em = np.nanmean(err_exec_all[:, :, j], axis=0)
        es = np.nanstd(err_exec_all[:, :, j], axis=0)

        ax.plot(ts, bm, color=C_BASE, linewidth=1.5, label="|base − oracle|")
        ax.fill_between(ts, np.maximum(bm - bs, 0), bm + bs, color=C_BASE, alpha=0.15)
        ax.plot(ts, em, color=C_EXECUTED, linewidth=1.5, label="|base+residual − oracle|")
        ax.fill_between(ts, np.maximum(em - es, 0), em + es, color=C_EXECUTED, alpha=0.15)

        ax.set_ylabel(f"{JOINT_NAMES[j]}\n|error| (rad)", fontsize=10)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=10, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Timestep", fontsize=11)

    # Aggregate stats for title
    all_base_mse = np.array([np.mean((ep["base"][:, :n_joints] - ep["oracle"][:, :n_joints])**2) for ep in episodes])
    all_exec_mse = np.array([np.mean((ep["final"][:, :n_joints] - ep["oracle"][:, :n_joints])**2) for ep in episodes])
    n_helped = (all_exec_mse < all_base_mse).sum()
    pct_improvement = (all_base_mse.mean() - all_exec_mse.mean()) / all_base_mse.mean() * 100

    fig.suptitle(
        f"Graph 4b: Mean Error vs Oracle ({len(episodes)} episodes)\n"
        f"MSE(base,oracle)={all_base_mse.mean():.6f}  MSE(exec,oracle)={all_exec_mse.mean():.6f}  "
        f"improvement={pct_improvement:+.1f}%  helped={n_helped}/{len(episodes)} eps",
        fontsize=12,
    )
    plt.tight_layout()
    out = output_dir / "graph4b_error_vs_oracle_mean.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Graph 4b: {out}")


def shade_gate(ax, gate):
    """Shade timesteps where gate fired (red)."""
    in_gate = False
    start = 0
    for t in range(len(gate)):
        if gate[t] == 1 and not in_gate:
            start = t
            in_gate = True
        elif gate[t] == 0 and in_gate:
            ax.axvspan(start, t, alpha=0.12, color="red", linewidth=0)
            in_gate = False
    if in_gate:
        ax.axvspan(start, len(gate), alpha=0.12, color="red", linewidth=0)


def compute_oracle_need(ep):
    """Derive binary 'oracle intervention needed' mask from detector scores or MSE.

    Uses cosine/L1 detector scores if available (threshold = median of nonzero scores).
    Falls back to per-step base-vs-oracle MSE > median.
    """
    if "cosine_scores" in ep and ep["cosine_scores"] is not None:
        cos = ep["cosine_scores"]
        l1 = ep["l1_scores"]
        cos_nonzero = cos[cos > 0]
        l1_nonzero = l1[l1 > 0]
        cos_thresh = np.median(cos_nonzero) if len(cos_nonzero) > 0 else 0.1
        l1_thresh = np.median(l1_nonzero) if len(l1_nonzero) > 0 else 0.1
        return (cos > cos_thresh) | (l1 > l1_thresh)
    base = ep["base"][:, :6]
    oracle = ep["oracle"][:, :6]
    mse = ((base - oracle) ** 2).mean(axis=1)
    return mse > np.median(mse)


def shade_oracle_need(ax, mask):
    """Shade timesteps where oracle intervention was needed (green)."""
    in_need = False
    start = 0
    for t in range(len(mask)):
        if mask[t] and not in_need:
            start = t
            in_need = True
        elif not mask[t] and in_need:
            ax.axvspan(start, t, alpha=0.10, color="#2E7D32", linewidth=0)
            in_need = False
    if in_need:
        ax.axvspan(start, len(mask), alpha=0.10, color="#2E7D32", linewidth=0)


def plot_graph1b(ep, output_dir, suffix=""):
    """Single eval rollout: base, oracle, base+residual per joint. Shows success/failure.
    Red shading = gate fired, green shading = oracle says intervention needed."""
    n_joints = 6
    oracle_need = compute_oracle_need(ep)
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 16), sharex=True)
    T = len(ep["base"])

    for j in range(n_joints):
        ax = axes[j]
        shade_oracle_need(ax, oracle_need)
        shade_gate(ax, ep["gate"])
        ax.plot(ep["base"][:, j], color=C_BASE, linewidth=1.2, alpha=0.8, label="Base")
        ax.plot(ep["oracle"][:, j], color=C_ORACLE, linewidth=1.2, alpha=0.8, label="Oracle (CuRobo)")
        ax.plot(ep["final"][:, j], color=C_EXECUTED, linewidth=1.5, alpha=0.9, label="Base + Residual")
        ax.set_ylabel(f"{JOINT_NAMES[j]}\n(rad)", fontsize=10)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=9, loc="upper right", ncol=3)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    status = "SUCCESS" if ep["success"] else "FAILURE"
    color = "#2E7D32" if ep["success"] else "#C62828"
    n_gate = int(ep["gate"].sum())
    n_need = int(oracle_need.sum())
    mse_base = np.mean((ep["base"][:, :n_joints] - ep["oracle"][:, :n_joints])**2)
    mse_exec = np.mean((ep["final"][:, :n_joints] - ep["oracle"][:, :n_joints])**2)
    fig.suptitle(
        f"Graph 1: Eval Rollout ep{ep['idx']} — [{status}]\n"
        f"Gate fired: {n_gate}/{T} (red) | Oracle need: {n_need}/{T} (green) | "
        f"MSE base→oracle: {mse_base:.5f}  exec→oracle: {mse_exec:.5f}",
        fontsize=12, color=color,
    )
    plt.tight_layout()
    out = output_dir / f"graph1_rollout{suffix}_ep{ep['idx']:02d}.png"
    plt.savefig(out, dpi=150)
    plt.close()


def plot_graph5(ep, output_dir, suffix=""):
    """Single rollout: raw residual output per joint. Shows gate decisions (red) and oracle need (green)."""
    n_joints = 6
    oracle_need = compute_oracle_need(ep)
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 16), sharex=True)
    T = len(ep["residual"])

    for j in range(n_joints):
        ax = axes[j]
        shade_oracle_need(ax, oracle_need)
        shade_gate(ax, ep["gate"])
        ax.plot(ep["residual"][:, j], color=C_RESIDUAL_RAW, linewidth=1.2, alpha=0.9, label="Residual output")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel(f"{JOINT_NAMES[j]}\nresidual (norm)", fontsize=10)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Timestep", fontsize=11)
    status = "SUCCESS" if ep["success"] else "FAILURE"
    color = "#2E7D32" if ep["success"] else "#C62828"
    n_need = int(oracle_need.sum())
    n_gate = int(ep["gate"].sum())
    mag_gate_on = np.abs(ep["residual"][ep["gate"] == 1]).mean() if ep["gate"].sum() > 0 else 0
    mag_gate_off = np.abs(ep["residual"][ep["gate"] == 0]).mean() if (ep["gate"] == 0).sum() > 0 else 0
    fig.suptitle(
        f"Graph 5: Residual Output — ep{ep['idx']} [{status}]\n"
        f"Red=gate ON ({n_gate}/{T}) | Green=oracle need ({n_need}/{T}) | "
        f"|residual| ON={mag_gate_on:.4f}, OFF={mag_gate_off:.4f}",
        fontsize=12, color=color,
    )
    plt.tight_layout()
    out = output_dir / f"graph5_residual{suffix}_ep{ep['idx']:02d}.png"
    plt.savefig(out, dpi=150)
    plt.close()


def plot_graph2(episodes, output_dir):
    """Graph 2: Mean +/- std of base vs base+residual across all eval episodes."""
    n_joints = 6
    max_T = max(len(ep["base"]) for ep in episodes)

    base_all = []
    exec_all = []
    for ep in episodes:
        T = len(ep["base"])
        b_pad = np.full((max_T, n_joints), np.nan)
        e_pad = np.full((max_T, n_joints), np.nan)
        b_pad[:T] = ep["base"][:, :n_joints]
        e_pad[:T] = ep["final"][:, :n_joints]
        base_all.append(b_pad)
        exec_all.append(e_pad)

    base_all = np.array(base_all)
    exec_all = np.array(exec_all)
    ts = np.arange(max_T)

    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 18), sharex=True)
    for j in range(n_joints):
        ax = axes[j]
        bm = np.nanmean(base_all[:, :, j], axis=0)
        bs = np.nanstd(base_all[:, :, j], axis=0)
        em = np.nanmean(exec_all[:, :, j], axis=0)
        es = np.nanstd(exec_all[:, :, j], axis=0)

        ax.plot(ts, bm, color=C_BASE, linewidth=1.5, label="Base")
        ax.fill_between(ts, bm - bs, bm + bs, color=C_BASE, alpha=0.15)
        ax.plot(ts, em, color=C_EXECUTED, linewidth=1.5, label="Base + Residual")
        ax.fill_between(ts, em - es, em + es, color=C_EXECUTED, alpha=0.15)

        ax.set_ylabel(f"{JOINT_NAMES[j]}\n(rad)", fontsize=10)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=10, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    fig.suptitle(
        f"Graph 2: Mean +/- Std — Base vs Base+Residual ({len(episodes)} eval episodes)",
        fontsize=13,
    )
    plt.tight_layout()
    out = output_dir / "graph2_mean_std.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Graph 2: {out}")


def plot_graph3(episodes, output_dir):
    """Graph 3: All eval traces overlaid — base (blue) vs base+residual (green)."""
    n_joints = 6
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 18), sharex=True)

    for j in range(n_joints):
        ax = axes[j]
        for ep in episodes:
            ax.plot(ep["base"][:, j], color=C_BASE, alpha=0.08, linewidth=0.5)
            ax.plot(ep["final"][:, j], color=C_EXECUTED, alpha=0.08, linewidth=0.5)
        ax.plot([], color=C_BASE, linewidth=2, label="Base")
        ax.plot([], color=C_EXECUTED, linewidth=2, label="Base + Residual")
        ax.set_ylabel(f"{JOINT_NAMES[j]}\n(rad)", fontsize=10)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=10, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    fig.suptitle(
        f"Graph 3: All Eval Traces — Base vs Base+Residual ({len(episodes)} episodes)",
        fontsize=13,
    )
    plt.tight_layout()
    out = output_dir / "graph3_all_traces.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Graph 3: {out}")


def plot_graph6(ep, output_dir, suffix=""):
    """Detector scores (cosine, L1) + gate decision + oracle need over time."""
    if "cosine_scores" not in ep or ep["cosine_scores"] is None:
        return
    oracle_need = compute_oracle_need(ep)
    T = len(ep["gate"])
    ts = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Row 1: cosine + L1 scores
    ax = axes[0]
    shade_oracle_need(ax, oracle_need)
    shade_gate(ax, ep["gate"])
    ax.plot(ts, ep["cosine_scores"], color="#4C72B0", linewidth=1.2, label="Cosine score")
    ax.plot(ts, ep["l1_scores"], color="#DD8452", linewidth=1.2, label="L1 score")
    ax.set_ylabel("Detector Score")
    ax.legend(fontsize=9, loc="upper right", ncol=2)

    # Row 2: gate probability (if gate exists, use binary gate as proxy)
    ax = axes[1]
    shade_oracle_need(ax, oracle_need)
    ax.fill_between(ts, 0, ep["gate"], color="#C44E52", alpha=0.4, step="mid", label="Gate ON")
    ax.set_ylabel("Gate Decision")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=9, loc="upper right")

    # Row 3: per-step base-oracle MSE
    ax = axes[2]
    shade_oracle_need(ax, oracle_need)
    shade_gate(ax, ep["gate"])
    base_oracle_mse = ((ep["base"][:, :6] - ep["oracle"][:, :6]) ** 2).mean(axis=1)
    exec_oracle_mse = ((ep["final"][:, :6] - ep["oracle"][:, :6]) ** 2).mean(axis=1)
    ax.plot(ts, base_oracle_mse, color="#4C72B0", linewidth=1.0, alpha=0.8, label="MSE(base, oracle)")
    ax.plot(ts, exec_oracle_mse, color="#55A868", linewidth=1.0, alpha=0.8, label="MSE(exec, oracle)")
    ax.set_ylabel("MSE vs Oracle")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=9, loc="upper right", ncol=2)

    status = "SUCCESS" if ep["success"] else "FAILURE"
    color = "#2E7D32" if ep["success"] else "#C62828"
    fig.suptitle(
        f"Graph 6: Detector Scores + Gate — ep{ep['idx']} [{status}]\n"
        f"Red=gate ON | Green=oracle need (cosine/L1 > median)",
        fontsize=12, color=color,
    )
    plt.tight_layout()
    out = output_dir / f"graph6_detector{suffix}_ep{ep['idx']:02d}.png"
    plt.savefig(out, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True, help="action_traces.json from eval server")
    parser.add_argument("--hdf5", required=True, help="eval rollout HDF5 (has oracle_action)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rollouts", type=int, default=10, help="Max rollouts for graph 1b/5/6")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_traces_and_oracle(args.traces, args.hdf5)
    print(f"Loaded {len(episodes)} episodes ({sum(e['success'] for e in episodes)} success, "
          f"{sum(not e['success'] for e in episodes)} failure)")

    # Graph 2: mean +/- std across all eval episodes
    plot_graph2(episodes, output_dir)

    # Graph 3: all traces overlaid
    plot_graph3(episodes, output_dir)

    # Graph 4: mean error vs oracle across all episodes
    plot_graph4b(episodes, output_dir)

    # Graph 1 and 5: per-episode rollouts (cap at max_rollouts)
    rollouts_dir = output_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    # Pick a mix of success and failure episodes
    successes = [e for e in episodes if e["success"]]
    failures = [e for e in episodes if not e["success"]]
    selected = failures[:min(len(failures), args.max_rollouts // 2)]
    remaining = args.max_rollouts - len(selected)
    selected += successes[:remaining]

    for ep in selected:
        plot_graph1b(ep, rollouts_dir)
        plot_graph5(ep, rollouts_dir)
        plot_graph6(ep, rollouts_dir)

    print(f"  Generated {len(selected)} rollout graphs (graph1 + graph5 + graph6) in {rollouts_dir}")
    print("  Done!")


if __name__ == "__main__":
    main()

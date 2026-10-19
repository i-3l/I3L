#!/usr/bin/env python3
"""
Generate 4 residual policy trace graphs:

1. Single rollout: 6 rows (joints), 3 traces (oracle, residual, base)
2. Mean ± std: aggregate of 50 eval rollouts per policy
3. All traces overlaid: all eval rollouts on one graph
4. Error magnitude: |oracle - residual| vs |oracle - base| per joint
"""

import argparse
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

JOINT_NAMES = [f"Joint {i}" for i in range(6)]
C_BASE = "#4C72B0"
C_ORACLE = "#DD8452"
C_RESIDUAL = "#55A868"


def load_actions(hdf5_path):
    """Load per-episode action arrays from HDF5."""
    episodes = []
    with h5py.File(hdf5_path, "r") as f:
        for k in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1])):
            episodes.append(f["data"][k]["action"][:, :6].copy())
    return episodes


def load_correction_demo(hdf5_path, demo_idx=None):
    """Load a single correction demo with base/oracle/executed actions."""
    with h5py.File(hdf5_path, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        if demo_idx is None:
            # Pick demo with most intervention steps
            best_idx, best_count = 0, 0
            for i, k in enumerate(keys):
                n = f["data"][k]["policy/is_oracle_active"][:].sum()
                if n > best_count:
                    best_idx, best_count = i, n
            demo_idx = best_idx
        k = keys[demo_idx]
        g = f["data"][k]
        return {
            "base": g["policy/base_action"][:, :6].copy(),
            "oracle": g["policy/oracle_action"][:, :6].copy(),
            "executed": g["action"][:, :6].copy(),
            "is_oracle_active": g["policy/is_oracle_active"][:].flatten().astype(bool),
            "demo_idx": demo_idx,
        }


def shade_interventions(ax, mask):
    in_int = False
    start = 0
    for t in range(len(mask)):
        if mask[t] and not in_int:
            start = t
            in_int = True
        elif not mask[t] and in_int:
            ax.axvspan(start, t, alpha=0.12, color="red", linewidth=0)
            in_int = False
    if in_int:
        ax.axvspan(start, len(mask), alpha=0.12, color="red", linewidth=0)


# ==================== GRAPH 1: Single rollout ====================

def plot_single_rollout(demo, output_dir, suffix=""):
    """6 rows (joints), 3 traces: oracle (orange), base (blue), executed (green)."""
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    T = len(demo["base"])

    for j in range(6):
        ax = axes[j]
        shade_interventions(ax, demo["is_oracle_active"])
        ax.plot(demo["base"][:, j], color=C_BASE, linewidth=1.2, alpha=0.8, label="Base Policy")
        ax.plot(demo["oracle"][:, j], color=C_ORACLE, linewidth=1.2, alpha=0.8, label="Oracle (CuRobo)")
        ax.plot(demo["executed"][:, j], color=C_RESIDUAL, linewidth=1.5, alpha=0.9, label="Executed")
        ax.set_ylabel(f"{JOINT_NAMES[j]}\n(rad)", fontsize=10)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=9, loc="upper right", ncol=3)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    fig.suptitle(f"Single Rollout — Correction Demo {demo['demo_idx']}\n(shaded = oracle active)", fontsize=13)
    plt.tight_layout()

    out = output_dir / f"graph1_single_rollout{suffix}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Graph 1: {out}")


# ==================== GRAPH 2: Mean ± std ====================

def plot_mean_std(base_eps, residual_eps, output_dir):
    """6 rows, mean trace + shaded std band, base vs residual."""
    min_len = min(
        min(len(e) for e in base_eps),
        min(len(e) for e in residual_eps),
    )
    base_arr = np.stack([e[:min_len] for e in base_eps])      # (N, T, 6)
    res_arr = np.stack([e[:min_len] for e in residual_eps])    # (N, T, 6)
    ts = np.arange(min_len)

    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)

    for j in range(6):
        ax = axes[j]
        bm, bs = base_arr[:, :, j].mean(0), base_arr[:, :, j].std(0)
        rm, rs = res_arr[:, :, j].mean(0), res_arr[:, :, j].std(0)

        ax.plot(ts, bm, color=C_BASE, linewidth=1.5, label="Base Policy")
        ax.fill_between(ts, bm - bs, bm + bs, color=C_BASE, alpha=0.15)
        ax.plot(ts, rm, color=C_RESIDUAL, linewidth=1.5, label="Residual Policy")
        ax.fill_between(ts, rm - rs, rm + rs, color=C_RESIDUAL, alpha=0.15)

        ax.set_ylabel(f"{JOINT_NAMES[j]}\n(rad)", fontsize=10)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=9, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    n_base, n_res = len(base_eps), len(residual_eps)
    fig.suptitle(f"Mean ± Std — Base ({n_base} eps) vs Residual ({n_res} eps), truncated to {min_len} steps", fontsize=13)
    plt.tight_layout()

    out = output_dir / "graph2_mean_std.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Graph 2: {out}")


# ==================== GRAPH 3: All traces overlaid ====================

def plot_all_traces(base_eps, residual_eps, output_dir):
    """6 rows, all eval rollouts as thin transparent lines."""
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)

    for j in range(6):
        ax = axes[j]
        for e in base_eps:
            ax.plot(e[:, j], color=C_BASE, alpha=0.08, linewidth=0.5)
        for e in residual_eps:
            ax.plot(e[:, j], color=C_RESIDUAL, alpha=0.08, linewidth=0.5)

        # Legend entries
        ax.plot([], color=C_BASE, linewidth=2, label="Base Policy")
        ax.plot([], color=C_RESIDUAL, linewidth=2, label="Residual Policy")

        ax.set_ylabel(f"{JOINT_NAMES[j]}\n(rad)", fontsize=10)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=9, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    fig.suptitle(f"All Eval Traces — Base ({len(base_eps)} eps) vs Residual ({len(residual_eps)} eps)", fontsize=13)
    plt.tight_layout()

    out = output_dir / "graph3_all_traces.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Graph 3: {out}")


# ==================== GRAPH 4: Error magnitude ====================

def plot_error_magnitude(demo, output_dir):
    """6 rows, |oracle - residual| vs |oracle - base| per joint."""
    err_base = np.abs(demo["oracle"] - demo["base"])
    err_residual = np.abs(demo["oracle"] - demo["executed"])

    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)

    for j in range(6):
        ax = axes[j]
        shade_interventions(ax, demo["is_oracle_active"])
        ax.plot(err_base[:, j], color=C_BASE, linewidth=1.2, alpha=0.8, label="|oracle − base|")
        ax.plot(err_residual[:, j], color=C_RESIDUAL, linewidth=1.2, alpha=0.8, label="|oracle − executed|")

        ax.set_ylabel(f"{JOINT_NAMES[j]}\n(rad)", fontsize=10)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=9, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    fig.suptitle(f"Error vs Oracle — Demo {demo['demo_idx']}\n(shaded = oracle active)", fontsize=13)
    plt.tight_layout()

    out = output_dir / "graph4_error_magnitude.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Graph 4: {out}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correction-hdf5", default="data/corrections_n060.hdf5")
    parser.add_argument("--base-hdf5", default=None)
    parser.add_argument("--residual-hdf5", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Graph 1 & 4: from correction data (has base + oracle + executed)
    # Generate one graph1 per demo + one graph4 for the demo with most interventions
    correction_path = Path(args.correction_hdf5)
    if correction_path.exists():
        import h5py
        with h5py.File(str(correction_path), "r") as f:
            n_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])
        rollouts_dir = output_dir / "rollouts"
        rollouts_dir.mkdir(parents=True, exist_ok=True)
        best_demo_idx, best_count = 0, 0
        for i in range(n_demos):
            demo = load_correction_demo(str(correction_path), demo_idx=i)
            plot_single_rollout(demo, rollouts_dir, suffix=f"_demo{i:02d}")
            n_oracle = demo["is_oracle_active"].sum()
            if n_oracle > best_count:
                best_demo_idx, best_count = i, n_oracle
        # Graph 4: error magnitude for demo with most interventions
        demo = load_correction_demo(str(correction_path), demo_idx=best_demo_idx)
        plot_error_magnitude(demo, output_dir)
    else:
        print(f"  Skipping graphs 1 & 4: {correction_path} not found")

    # Graphs 2 & 3: from eval rollout HDF5s (base vs residual)
    if args.base_hdf5 and args.residual_hdf5:
        base_path = Path(args.base_hdf5)
        res_path = Path(args.residual_hdf5)
        if base_path.exists() and res_path.exists():
            base_eps = load_actions(str(base_path))
            res_eps = load_actions(str(res_path))
            print(f"  Loaded {len(base_eps)} base, {len(res_eps)} residual rollouts")
            plot_mean_std(base_eps, res_eps, output_dir)
            plot_all_traces(base_eps, res_eps, output_dir)
        else:
            if not base_path.exists():
                print(f"  Skipping graphs 2 & 3: {base_path} not found")
            if not res_path.exists():
                print(f"  Skipping graphs 2 & 3: {res_path} not found")
    else:
        print("  Skipping graphs 2 & 3: --base-hdf5 and --residual-hdf5 not provided")

    print("  Done!")


if __name__ == "__main__":
    main()

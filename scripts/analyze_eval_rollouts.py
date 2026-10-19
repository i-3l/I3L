#!/usr/bin/env python3
"""
Analyze eval rollout HDF5s that were collected with policy=residual_eval
(SharedAutonomyPolicy with CuRobo oracle running in background).

Each demo has:
  - action: what the robot actually executed (residual-augmented base)
  - policy/base_action: what the residual policy output (= base + residual when gate fires)
  - policy/oracle_action: what CuRobo would do (ground truth)
  - policy/is_oracle_active: always False (we disabled switching)
  - policy/detector_interrupt_scores: cosine + L1 scores

Reports per-step and aggregate metrics comparing residual output vs oracle.
"""

import argparse
import h5py
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_rollout_hdf5(hdf5_path):
    results = []
    with h5py.File(hdf5_path, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for k in keys:
            g = f["data"][k]
            executed = g["action"][:, :6]
            T = len(executed)

            has_policy = "policy" in g
            if not has_policy:
                results.append({"demo": k, "T": T, "has_oracle": False})
                continue

            base_action = g["policy/base_action"][:, :6] if "policy/base_action" in g else None
            oracle_action = g["policy/oracle_action"][:, :6] if "policy/oracle_action" in g else None

            if oracle_action is None or base_action is None:
                results.append({"demo": k, "T": T, "has_oracle": False})
                continue

            # executed = what SA policy output = base_action (since we disabled switching)
            # base_action = what the residual-augmented IL policy predicted
            # oracle_action = what CuRobo predicted

            # MSE: executed vs oracle (how far is our policy from ground truth?)
            mse_executed_oracle = ((executed - oracle_action) ** 2).mean(axis=1)
            # MSE: would need pure base action for comparison, but we don't have it
            # We DO have detector scores which indicate disagreement

            # Detector scores if available
            has_scores = "policy/detector_interrupt_scores" in g
            if has_scores:
                scores = g["policy/detector_interrupt_scores"][:]
                cosine_scores = scores[:, 0]
                l1_scores = scores[:, 1]
            else:
                cosine_scores = np.zeros(T)
                l1_scores = np.zeros(T)

            # Per-joint MSE
            per_joint_mse = ((executed - oracle_action) ** 2).mean(axis=0)

            # Action magnitude
            residual_mag = np.abs(executed - oracle_action).mean(axis=1)

            results.append({
                "demo": k,
                "T": T,
                "has_oracle": True,
                "mse_executed_oracle_mean": float(mse_executed_oracle.mean()),
                "mse_executed_oracle_std": float(mse_executed_oracle.std()),
                "mse_per_joint": per_joint_mse.tolist(),
                "cosine_score_mean": float(cosine_scores.mean()),
                "cosine_score_max": float(cosine_scores.max()),
                "l1_score_mean": float(l1_scores.mean()),
                "l1_score_max": float(l1_scores.max()),
                "action_diff_mag_mean": float(residual_mag.mean()),
                "action_diff_mag_p95": float(np.percentile(residual_mag, 95)),
            })
    return results


def plot_eval_metrics(results, output_dir):
    oracle_results = [r for r in results if r.get("has_oracle")]
    if not oracle_results:
        print("  No oracle data found in eval HDF5 — skipping metric plots")
        return

    # MSE per episode
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. MSE per episode
    ax = axes[0, 0]
    mses = [r["mse_executed_oracle_mean"] for r in oracle_results]
    ax.bar(range(len(mses)), mses, color="#C44E52", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE(executed, oracle)")
    ax.set_title("Per-Episode MSE vs Oracle")

    # 2. Detector scores per episode
    ax = axes[0, 1]
    cosines = [r["cosine_score_mean"] for r in oracle_results]
    l1s = [r["l1_score_mean"] for r in oracle_results]
    x = range(len(cosines))
    ax.bar([i - 0.2 for i in x], cosines, width=0.35, label="Cosine", color="#4C72B0")
    ax.bar([i + 0.2 for i in x], l1s, width=0.35, label="L1", color="#DD8452")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Detector Score")
    ax.set_title("Mean Detector Scores per Episode")
    ax.legend()

    # 3. Per-joint MSE (averaged across episodes)
    ax = axes[1, 0]
    joint_mses = np.mean([r["mse_per_joint"] for r in oracle_results], axis=0)
    ax.bar(range(6), joint_mses, color="#55A868")
    ax.set_xlabel("Joint")
    ax.set_ylabel("Mean MSE vs Oracle")
    ax.set_title("Per-Joint MSE (averaged)")
    ax.set_xticks(range(6))

    # 4. Action difference magnitude distribution
    ax = axes[1, 1]
    mags = [r["action_diff_mag_mean"] for r in oracle_results]
    p95s = [r["action_diff_mag_p95"] for r in oracle_results]
    ax.bar([i - 0.2 for i in x], mags, width=0.35, label="Mean", color="#4C72B0")
    ax.bar([i + 0.2 for i in x], p95s, width=0.35, label="P95", color="#C44E52")
    ax.set_xlabel("Episode")
    ax.set_ylabel("|executed - oracle|")
    ax.set_title("Action Difference vs Oracle")
    ax.legend()

    plt.suptitle("Eval Rollout Analysis (with CuRobo Oracle)", fontsize=14)
    plt.tight_layout()
    out = output_dir / "eval_oracle_metrics.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, help="Eval rollout HDF5 with oracle data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {args.hdf5}...")
    results = analyze_rollout_hdf5(args.hdf5)

    oracle_results = [r for r in results if r.get("has_oracle")]
    print(f"  {len(results)} episodes, {len(oracle_results)} with oracle data")

    if oracle_results:
        print()
        print(f"{'Demo':<10} {'T':<5} {'MSE(exec,oracle)':<18} {'Cosine':<9} {'L1':<9} {'|diff| mean':<12}")
        print("-" * 70)
        for r in oracle_results:
            print(f"{r['demo']:<10} {r['T']:<5} {r['mse_executed_oracle_mean']:<18.5f} {r['cosine_score_mean']:<9.3f} {r['l1_score_mean']:<9.3f} {r['action_diff_mag_mean']:<12.4f}")

        avg = lambda key: np.mean([r[key] for r in oracle_results])
        print("-" * 70)
        print(f"{'AVERAGE':<10} {'':5} {avg('mse_executed_oracle_mean'):<18.5f} {avg('cosine_score_mean'):<9.3f} {avg('l1_score_mean'):<9.3f} {avg('action_diff_mag_mean'):<12.4f}")

        plot_eval_metrics(oracle_results, output_dir)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"per_episode": results}, f, indent=2)
        print(f"  Saved {args.output_json}")


if __name__ == "__main__":
    main()

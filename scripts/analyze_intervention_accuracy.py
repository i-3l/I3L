#!/usr/bin/env python3
"""
Principled intervention accuracy analysis.

Compares the residual policy's gate decisions against ground-truth intervention
need derived from the CuRobo oracle trajectory during eval rollouts.

Ground truth: at each timestep, the oracle (CuRobo) and base policy both predict
an action. The detector computes cosine and L1 disagreement scores between them.
When the scores are high, the base policy is wrong and intervention is needed.

This script:
1. Loads eval HDF5 (collected with policy=residual_eval, which has oracle data)
2. Loads the residual checkpoint and runs the gate on the same observations
3. Derives ground-truth "intervention needed" labels from oracle-vs-base disagreement
4. Reports gate accuracy/precision/recall at multiple detector thresholds
5. Computes ROC/AUC for the gate's intervention probability
6. Computes MSE(executed, oracle) split by gate decision — does gating help?
7. Generates plots: ROC, threshold sweep, per-episode breakdown

Usage:
  python scripts/analyze_intervention_accuracy.py \
    --eval-hdf5 data/eval_auto/<run>/residual/rollouts_repeat01.hdf5 \
    --residual-ckpt <ckpt_path> \
    --base-ckpt <base_ckpt_path> \
    --arch simple_residual_a1_v13 \
    --output-dir data/eval_auto/<run>
"""

import argparse
import h5py
import json
import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../il_lib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_residual_policy(arch, base_ckpt, residual_ckpt):
    """Load the residual policy from checkpoint."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, ListConfig
    from il_lib.utils.config_utils import register_omegaconf_resolvers
    from il_lib.utils.training_utils import load_torch

    import iiil as _iiil
    extra = [f"file://{_iiil.__path__[0]}/configs"]
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../il_lib/il_lib/configs"))

    hydra_args = [
        f"arch={arch}", "robot=a1_iiil", "task=pnp/base", "data_dir=/tmp/fake",
        f"module.base_policy_ckpt_path={base_ckpt}",
        f"hydra.searchpath=[{','.join(extra)}]",
    ]

    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="base_config", overrides=hydra_args)
        register_omegaconf_resolvers()
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)

        raw = cfg.module.get("base_policy_overrides", None)
        existing = list(OmegaConf.to_container(raw, resolve=True)) if isinstance(raw, ListConfig) else (list(raw) if raw else [])
        if not any(o.startswith("robot=") for o in existing):
            existing.append("robot=a1_iiil")
        if not any(o.startswith("task=") for o in existing):
            existing.append("task=pnp/base")
        cfg.module.base_policy_overrides = existing

        policy = instantiate(cfg.module, _recursive_=False)
        ckpt = load_torch(residual_ckpt, map_location="cpu")
        policy.load_state_dict(ckpt["state_dict"], strict=False)
        policy = policy.to("cuda")
        policy.eval()
    return policy


def run_gate_on_eval_data(policy, hdf5_path):
    """
    Run the residual gate on eval rollout data step-by-step.
    Returns per-step: gate_prob, gate_decision, residual_action, and the raw HDF5 data.
    """
    episodes = []
    with h5py.File(hdf5_path, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for k in keys:
            g = f["data"][k]
            if "policy" not in g:
                continue

            executed = g["action"][:]
            T = len(executed)

            base_action = g["policy/base_action"][:] if "policy/base_action" in g else None
            oracle_action = g["policy/oracle_action"][:] if "policy/oracle_action" in g else None
            if base_action is None or oracle_action is None:
                continue

            has_scores = "policy/detector_interrupt_scores" in g
            if has_scores:
                scores = g["policy/detector_interrupt_scores"][:]
                cosine_scores = scores[:, 0]
                l1_scores = scores[:, 1]
            else:
                cosine_scores = np.zeros(T)
                l1_scores = np.zeros(T)

            qpos_arm = torch.tensor(g["obs/robot::proprio"][:, :6], dtype=torch.float32)
            qpos_gripper = torch.tensor(g["obs/robot::proprio"][:, 6:7], dtype=torch.float32)
            eef_pos = torch.tensor(g["obs/robot::proprio"][:, 40:43], dtype=torch.float32)
            eef_quat = torch.tensor(g["obs/robot::proprio"][:, 43:47], dtype=torch.float32)

            has_task = "obs/task::object_pose" in g
            if has_task:
                task_obs = torch.tensor(g["obs/task::object_pose"][:], dtype=torch.float32)
            else:
                task_obs = None

            gate_probs = []
            gate_decisions = []
            pred_residuals = []

            for t in range(T):
                obs = {
                    "qpos": {"arm": qpos_arm[t:t+1].unsqueeze(0).cuda(), "gripper": qpos_gripper[t:t+1].unsqueeze(0).cuda()},
                    "eef": {"0_pos": eef_pos[t:t+1].unsqueeze(0).cuda(), "0_quat": eef_quat[t:t+1].unsqueeze(0).cuda()},
                    "base_action": torch.tensor(base_action[t:t+1, :], dtype=torch.float32).unsqueeze(0).cuda(),
                }
                if task_obs is not None:
                    obs["task"] = task_obs[t:t+1].unsqueeze(0).cuda()

                with torch.no_grad():
                    res, gate, gate_w = policy.act_bundle(obs)
                pred_residuals.append(res.squeeze().cpu().numpy())
                gate_probs.append(float(gate_w.squeeze().cpu()))
                gate_decisions.append(int(gate.squeeze().cpu().item()))

            episodes.append({
                "demo": k,
                "T": T,
                "executed": executed,
                "base_action": base_action,
                "oracle_action": oracle_action,
                "cosine_scores": cosine_scores,
                "l1_scores": l1_scores,
                "gate_probs": np.array(gate_probs),
                "gate_decisions": np.array(gate_decisions),
                "pred_residuals": np.stack(pred_residuals),
            })

    return episodes


def compute_ground_truth_labels(episodes, cosine_thresh, l1_thresh):
    """
    Derive binary ground-truth "intervention needed" labels from detector scores.
    A step needs intervention if cosine > cosine_thresh OR l1 > l1_thresh.
    """
    for ep in episodes:
        ep["gt_intervention"] = (
            (ep["cosine_scores"] > cosine_thresh) | (ep["l1_scores"] > l1_thresh)
        ).astype(int)
    return episodes


def compute_metrics_at_threshold(episodes, cosine_thresh, l1_thresh):
    """Compute gate accuracy/precision/recall against detector-derived ground truth."""
    episodes = compute_ground_truth_labels(episodes, cosine_thresh, l1_thresh)

    all_gt = np.concatenate([ep["gt_intervention"] for ep in episodes])
    all_pred = np.concatenate([ep["gate_decisions"] for ep in episodes])

    tp = ((all_pred == 1) & (all_gt == 1)).sum()
    fp = ((all_pred == 1) & (all_gt == 0)).sum()
    fn = ((all_pred == 0) & (all_gt == 1)).sum()
    tn = ((all_pred == 0) & (all_gt == 0)).sum()

    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    gt_rate = all_gt.mean()

    return {
        "cosine_thresh": cosine_thresh,
        "l1_thresh": l1_thresh,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "gt_intervention_rate": float(gt_rate),
        "pred_intervention_rate": float(all_pred.mean()),
    }


def compute_roc(episodes):
    """
    Compute ROC curve for the gate probability against oracle-disagreement ground truth.
    Uses a range of oracle-MSE thresholds to define "intervention needed", then sweeps
    gate probability thresholds.
    """
    all_gate_probs = np.concatenate([ep["gate_probs"] for ep in episodes])
    all_base = np.concatenate([ep["base_action"][:, :6] for ep in episodes])
    all_oracle = np.concatenate([ep["oracle_action"][:, :6] for ep in episodes])

    oracle_mse = ((all_base - all_oracle) ** 2).mean(axis=1)
    mse_median = np.median(oracle_mse)
    gt_labels = (oracle_mse > mse_median).astype(int)

    thresholds = np.linspace(0, 1, 101)
    tprs, fprs = [], []
    for t in thresholds:
        pred = (all_gate_probs >= t).astype(int)
        tp = ((pred == 1) & (gt_labels == 1)).sum()
        fp = ((pred == 1) & (gt_labels == 0)).sum()
        fn = ((pred == 0) & (gt_labels == 1)).sum()
        tn = ((pred == 0) & (gt_labels == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tprs.append(tpr)
        fprs.append(fpr)

    auc = -np.trapz(tprs, fprs)
    return np.array(fprs), np.array(tprs), thresholds, float(auc), float(mse_median)


def compute_mse_by_gate(episodes):
    """
    Split executed-vs-oracle MSE by gate decision.
    Shows whether gating actually helps: MSE should be lower when gate fires.
    """
    results = {"gate_on": [], "gate_off": [], "all": []}
    for ep in episodes:
        executed = ep["executed"][:, :6]
        oracle = ep["oracle_action"][:, :6]
        base = ep["base_action"][:, :6]
        gate = ep["gate_decisions"]

        mse_exec_oracle = ((executed - oracle) ** 2).mean(axis=1)
        mse_base_oracle = ((base - oracle) ** 2).mean(axis=1)

        for t in range(ep["T"]):
            entry = {
                "mse_executed_oracle": float(mse_exec_oracle[t]),
                "mse_base_oracle": float(mse_base_oracle[t]),
            }
            results["all"].append(entry)
            if gate[t] == 1:
                results["gate_on"].append(entry)
            else:
                results["gate_off"].append(entry)

    def agg(lst, key):
        if not lst:
            return 0.0
        return float(np.mean([x[key] for x in lst]))

    return {
        "gate_on_mse_exec_oracle": agg(results["gate_on"], "mse_executed_oracle"),
        "gate_on_mse_base_oracle": agg(results["gate_on"], "mse_base_oracle"),
        "gate_off_mse_exec_oracle": agg(results["gate_off"], "mse_executed_oracle"),
        "gate_off_mse_base_oracle": agg(results["gate_off"], "mse_base_oracle"),
        "all_mse_exec_oracle": agg(results["all"], "mse_executed_oracle"),
        "all_mse_base_oracle": agg(results["all"], "mse_base_oracle"),
        "n_gate_on": len(results["gate_on"]),
        "n_gate_off": len(results["gate_off"]),
    }


def compute_per_episode_stats(episodes):
    """Per-episode breakdown of gate behavior vs oracle disagreement."""
    stats = []
    for ep in episodes:
        base = ep["base_action"][:, :6]
        oracle = ep["oracle_action"][:, :6]
        executed = ep["executed"][:, :6]
        gate = ep["gate_decisions"]

        oracle_mse = ((base - oracle) ** 2).mean(axis=1)
        exec_mse = ((executed - oracle) ** 2).mean(axis=1)

        high_disagree = oracle_mse > np.median(oracle_mse) if len(oracle_mse) > 0 else np.zeros(len(oracle_mse), dtype=bool)

        stats.append({
            "demo": ep["demo"],
            "T": ep["T"],
            "gate_rate": float(gate.mean()),
            "oracle_mse_mean": float(oracle_mse.mean()),
            "exec_mse_mean": float(exec_mse.mean()),
            "mse_improvement": float(1 - exec_mse.mean() / max(oracle_mse.mean(), 1e-8)),
            "gate_fires_on_high_disagree": float(gate[high_disagree].mean()) if high_disagree.sum() > 0 else 0.0,
            "gate_fires_on_low_disagree": float(gate[~high_disagree].mean()) if (~high_disagree).sum() > 0 else 0.0,
            "cosine_mean": float(ep["cosine_scores"].mean()),
            "l1_mean": float(ep["l1_scores"].mean()),
        })
    return stats


def plot_results(episodes, roc_data, threshold_sweep, per_ep_stats, mse_by_gate, output_dir):
    """Generate all analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. ROC curve
    ax = axes[0, 0]
    fprs, tprs, _, auc, _ = roc_data
    ax.plot(fprs, tprs, "b-", linewidth=2, label=f"Gate (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC: Gate vs Oracle Disagreement")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 2. Threshold sweep — F1 / accuracy vs cosine threshold
    ax = axes[0, 1]
    cos_threshs = [s["cosine_thresh"] for s in threshold_sweep]
    f1s = [s["f1"] for s in threshold_sweep]
    accs = [s["accuracy"] for s in threshold_sweep]
    precs = [s["precision"] for s in threshold_sweep]
    recs = [s["recall"] for s in threshold_sweep]
    ax.plot(cos_threshs, f1s, "b-", label="F1")
    ax.plot(cos_threshs, accs, "g-", label="Accuracy")
    ax.plot(cos_threshs, precs, "r--", label="Precision", alpha=0.7)
    ax.plot(cos_threshs, recs, "m--", label="Recall", alpha=0.7)
    ax.set_xlabel("Cosine Threshold (GT)")
    ax.set_ylabel("Score")
    ax.set_title("Gate Metrics vs GT Threshold")
    ax.legend(fontsize=8)

    # 3. MSE by gate decision
    ax = axes[0, 2]
    labels = ["Gate ON", "Gate OFF"]
    exec_mses = [mse_by_gate["gate_on_mse_exec_oracle"], mse_by_gate["gate_off_mse_exec_oracle"]]
    base_mses = [mse_by_gate["gate_on_mse_base_oracle"], mse_by_gate["gate_off_mse_base_oracle"]]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, exec_mses, w, label="Executed vs Oracle", color="#4C72B0")
    ax.bar(x + w/2, base_mses, w, label="Base vs Oracle", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MSE")
    ax.set_title("MSE by Gate Decision")
    ax.legend(fontsize=8)

    # 4. Per-episode gate rate vs oracle disagreement
    ax = axes[1, 0]
    gate_rates = [s["gate_rate"] for s in per_ep_stats]
    oracle_mses = [s["oracle_mse_mean"] for s in per_ep_stats]
    ax.scatter(oracle_mses, gate_rates, alpha=0.7, s=40)
    ax.set_xlabel("Mean Oracle-Base MSE")
    ax.set_ylabel("Gate Rate")
    ax.set_title("Gate Rate vs Oracle Disagreement")

    # 5. Gate selectivity: fires on high vs low disagreement
    ax = axes[1, 1]
    high = [s["gate_fires_on_high_disagree"] for s in per_ep_stats]
    low = [s["gate_fires_on_low_disagree"] for s in per_ep_stats]
    x = np.arange(len(per_ep_stats))
    ax.bar(x - 0.2, high, 0.35, label="High disagree", color="#55A868")
    ax.bar(x + 0.2, low, 0.35, label="Low disagree", color="#C44E52")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Gate Fire Rate")
    ax.set_title("Gate Selectivity")
    ax.legend(fontsize=8)

    # 6. Gate probability histogram with oracle disagreement overlay
    ax = axes[1, 2]
    all_probs = np.concatenate([ep["gate_probs"] for ep in episodes])
    all_base = np.concatenate([ep["base_action"][:, :6] for ep in episodes])
    all_oracle = np.concatenate([ep["oracle_action"][:, :6] for ep in episodes])
    all_mse = ((all_base - all_oracle) ** 2).mean(axis=1)
    median_mse = np.median(all_mse)
    high_mask = all_mse > median_mse
    ax.hist(all_probs[high_mask], bins=50, alpha=0.6, label="High disagree", color="#C44E52", density=True)
    ax.hist(all_probs[~high_mask], bins=50, alpha=0.6, label="Low disagree", color="#55A868", density=True)
    ax.set_xlabel("Gate Probability")
    ax.set_ylabel("Density")
    ax.set_title("Gate Prob Distribution by Oracle Need")
    ax.legend(fontsize=8)

    plt.suptitle("Intervention Accuracy Analysis", fontsize=14, y=1.01)
    plt.tight_layout()
    out = output_dir / "intervention_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Principled intervention accuracy analysis")
    parser.add_argument("--eval-hdf5", required=True, help="Eval rollout HDF5 with oracle data")
    parser.add_argument("--residual-ckpt", required=True)
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--arch", default="simple_residual_a1_v13")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading residual policy (arch={args.arch})...")
    policy = load_residual_policy(args.arch, args.base_ckpt, args.residual_ckpt)

    has_gate = getattr(policy, '_use_intervention_head', False)
    print(f"  Gate enabled: {has_gate}")

    print(f"Running gate on eval data: {args.eval_hdf5}...")
    episodes = run_gate_on_eval_data(policy, args.eval_hdf5)
    print(f"  {len(episodes)} episodes loaded")

    if not episodes:
        print("ERROR: No episodes with oracle data found")
        return

    total_steps = sum(ep["T"] for ep in episodes)
    total_gate_on = sum(ep["gate_decisions"].sum() for ep in episodes)
    print(f"  Total steps: {total_steps}, gate fires: {total_gate_on} ({total_gate_on/total_steps:.1%})")

    # --- Threshold sweep ---
    print("\n--- Threshold Sweep (gate vs detector ground truth) ---")
    threshold_sweep = []
    cosine_thresholds = np.arange(0.05, 0.80, 0.05)
    for ct in cosine_thresholds:
        m = compute_metrics_at_threshold(episodes, cosine_thresh=ct, l1_thresh=ct)
        threshold_sweep.append(m)
    # Find best F1
    best = max(threshold_sweep, key=lambda x: x["f1"])
    print(f"  Best F1={best['f1']:.3f} at cosine_thresh={best['cosine_thresh']:.2f}")
    print(f"    Acc={best['accuracy']:.3f} Prec={best['precision']:.3f} Rec={best['recall']:.3f}")
    print(f"    GT intervention rate={best['gt_intervention_rate']:.1%}, Pred={best['pred_intervention_rate']:.1%}")

    # --- ROC ---
    print("\n--- ROC Analysis (gate prob vs oracle MSE) ---")
    fprs, tprs, roc_thresholds, auc, mse_median = compute_roc(episodes)
    print(f"  AUC={auc:.3f} (oracle MSE threshold={mse_median:.5f})")

    # --- MSE by gate decision ---
    print("\n--- MSE Split by Gate Decision ---")
    mse_by_gate = compute_mse_by_gate(episodes)
    print(f"  Gate ON  ({mse_by_gate['n_gate_on']:>5} steps): exec→oracle MSE={mse_by_gate['gate_on_mse_exec_oracle']:.5f}, base→oracle={mse_by_gate['gate_on_mse_base_oracle']:.5f}")
    print(f"  Gate OFF ({mse_by_gate['n_gate_off']:>5} steps): exec→oracle MSE={mse_by_gate['gate_off_mse_exec_oracle']:.5f}, base→oracle={mse_by_gate['gate_off_mse_base_oracle']:.5f}")
    if mse_by_gate['gate_on_mse_base_oracle'] > 0:
        improvement = (1 - mse_by_gate['gate_on_mse_exec_oracle'] / mse_by_gate['gate_on_mse_base_oracle']) * 100
        print(f"  Gate ON improvement: {improvement:+.1f}% (exec vs base, relative to oracle)")

    # --- Per-episode stats ---
    per_ep_stats = compute_per_episode_stats(episodes)

    print("\n--- Per-Episode Breakdown ---")
    print(f"{'Demo':<10} {'T':<5} {'GateRate':<10} {'BaseMSE':<10} {'ExecMSE':<10} {'Improv':<9} {'HiDisag':<9} {'LoDisag':<9}")
    print("-" * 82)
    for s in per_ep_stats:
        print(f"{s['demo']:<10} {s['T']:<5} {s['gate_rate']:<10.3f} {s['oracle_mse_mean']:<10.5f} {s['exec_mse_mean']:<10.5f} {s['mse_improvement']:<9.1%} {s['gate_fires_on_high_disagree']:<9.3f} {s['gate_fires_on_low_disagree']:<9.3f}")

    # --- Plots ---
    print("\nGenerating plots...")
    roc_data = (fprs, tprs, roc_thresholds, auc, mse_median)
    plot_results(episodes, roc_data, threshold_sweep, per_ep_stats, mse_by_gate, output_dir)

    # --- Save JSON ---
    output_json = args.output_json or str(output_dir / "intervention_accuracy.json")
    summary = {
        "config": {
            "arch": args.arch,
            "eval_hdf5": args.eval_hdf5,
            "has_gate": has_gate,
            "total_steps": total_steps,
            "total_episodes": len(episodes),
        },
        "gate_summary": {
            "gate_fire_rate": float(total_gate_on / total_steps),
        },
        "roc": {
            "auc": auc,
            "oracle_mse_threshold": mse_median,
        },
        "best_threshold_match": best,
        "mse_by_gate": mse_by_gate,
        "threshold_sweep": threshold_sweep,
        "per_episode": per_ep_stats,
    }
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {output_json}")


if __name__ == "__main__":
    main()

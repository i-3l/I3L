#!/usr/bin/env python3
"""
Offline analysis: run trained residual model on correction data states
where we have oracle ground truth.

Reports:
- MSE between (base + residual) and oracle on intervention steps
- MSE between base and oracle on intervention steps (baseline)
- Gate accuracy vs ground truth int_state
- Intervention precision/recall
- Per-joint error breakdown
"""

import argparse
import h5py
import json
import torch
import numpy as np
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, ListConfig
from il_lib.utils.config_utils import register_omegaconf_resolvers
from il_lib.utils.training_utils import load_torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../il_lib"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correction-hdf5", default="data/corrections_n060.hdf5")
    parser.add_argument("--residual-ckpt", required=True)
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--arch", default="simple_residual_a1_v13")
    args = parser.parse_args()

    import iiil as _iiil
    extra = [f"file://{_iiil.__path__[0]}/configs"]
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../il_lib/il_lib/configs"))

    hydra_args = [
        f"arch={args.arch}", "robot=a1_iiil", "task=pnp/base", "data_dir=/tmp/fake",
        f"module.base_policy_ckpt_path={args.base_ckpt}",
        f"hydra.searchpath=[{','.join(extra)}]",
    ]

    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="base_config", overrides=hydra_args)
        register_omegaconf_resolvers()
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)

        # Forward overrides for base policy loading
        raw = cfg.module.get("base_policy_overrides", None)
        existing = list(OmegaConf.to_container(raw, resolve=True)) if isinstance(raw, ListConfig) else (list(raw) if raw else [])
        if not any(o.startswith("robot=") for o in existing):
            existing.append("robot=a1_iiil")
        if not any(o.startswith("task=") for o in existing):
            existing.append("task=pnp/base")
        cfg.module.base_policy_overrides = existing

        policy = instantiate(cfg.module, _recursive_=False)
        ckpt = load_torch(args.residual_ckpt, map_location="cpu")
        policy.load_state_dict(ckpt["state_dict"], strict=False)
        policy = policy.to("cuda")
        policy.eval()

    # Load correction data
    print(f"Loading {args.correction_hdf5}...")
    with h5py.File(args.correction_hdf5, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))

        all_results = []
        for k in keys:
            g = f["data"][k]
            base_action = torch.tensor(g["policy/base_action"][:], dtype=torch.float32)
            oracle_action = torch.tensor(g["policy/oracle_action"][:], dtype=torch.float32)
            is_oracle = g["policy/is_oracle_active"][:].flatten().astype(bool)

            # Run residual model on each step
            qpos_arm = torch.tensor(g["obs/robot::proprio"][:, :6], dtype=torch.float32)
            qpos_gripper = torch.tensor(g["obs/robot::proprio"][:, 6:7], dtype=torch.float32)
            task_obs = torch.tensor(g["obs/task::object_pose"][:], dtype=torch.float32)
            eef_pos = torch.tensor(g["obs/robot::proprio"][:, 40:43], dtype=torch.float32)
            eef_quat = torch.tensor(g["obs/robot::proprio"][:, 43:47], dtype=torch.float32)

            T = len(base_action)
            pred_residuals = []
            pred_gates = []

            for t in range(T):
                obs = {
                    "qpos": {"arm": qpos_arm[t:t+1].unsqueeze(0).cuda(), "gripper": qpos_gripper[t:t+1].unsqueeze(0).cuda()},
                    "eef": {"0_pos": eef_pos[t:t+1].unsqueeze(0).cuda(), "0_quat": eef_quat[t:t+1].unsqueeze(0).cuda()},
                    "task": task_obs[t:t+1].unsqueeze(0).cuda(),
                    "base_action": base_action[t:t+1, :].unsqueeze(0).cuda(),
                }
                with torch.no_grad():
                    res, gate = policy.act(obs)
                pred_residuals.append(res.squeeze().cpu())
                pred_gates.append(int(gate.squeeze().cpu().item() >= 0.5) if gate is not None else 1)

            pred_residuals = torch.stack(pred_residuals)  # (T, 7)
            pred_gates = np.array(pred_gates)

            # Compute metrics
            base_arm = base_action[:, :6].numpy()
            oracle_arm = oracle_action[:, :6].numpy()
            residual_arm = pred_residuals[:, :6].numpy()
            corrected_arm = base_arm + residual_arm

            # MSE: base vs oracle, corrected vs oracle
            mse_base = ((base_arm - oracle_arm) ** 2).mean(axis=1)
            mse_corrected = ((corrected_arm - oracle_arm) ** 2).mean(axis=1)

            # Gate accuracy
            gate_correct = (pred_gates == is_oracle.astype(int))

            all_results.append({
                "demo": k,
                "T": T,
                "n_intervention": int(is_oracle.sum()),
                "gate_fires": int(pred_gates.sum()),
                "gate_accuracy": float(gate_correct.mean()),
                "gate_precision": float(gate_correct[pred_gates == 1].mean()) if pred_gates.sum() > 0 else 0,
                "gate_recall": float(pred_gates[is_oracle].mean()) if is_oracle.sum() > 0 else 0,
                # MSE on intervention steps
                "mse_base_on_int": float(mse_base[is_oracle].mean()) if is_oracle.sum() > 0 else 0,
                "mse_corrected_on_int": float(mse_corrected[is_oracle].mean()) if is_oracle.sum() > 0 else 0,
                # MSE on all steps
                "mse_base_all": float(mse_base.mean()),
                "mse_corrected_all": float(mse_corrected.mean()),
                # Residual magnitude
                "residual_mag_int": float(np.abs(residual_arm[is_oracle]).mean()) if is_oracle.sum() > 0 else 0,
                "residual_mag_non": float(np.abs(residual_arm[~is_oracle]).mean()) if (~is_oracle).sum() > 0 else 0,
                # Did correction help?
                "correction_helped_pct": float((mse_corrected[is_oracle] < mse_base[is_oracle]).mean()) if is_oracle.sum() > 0 else 0,
            })

    # Aggregate
    print()
    print("=" * 90)
    print("OFFLINE ANALYSIS: Residual on Correction Data")
    print("=" * 90)
    print()
    print(f"{'Demo':<10} {'T':<5} {'Int':<5} {'GateFires':<10} {'GateAcc':<9} {'Prec':<7} {'Recall':<8} {'MSE(base)':<11} {'MSE(corr)':<11} {'Helped%':<9}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['demo']:<10} {r['T']:<5} {r['n_intervention']:<5} {r['gate_fires']:<10} {r['gate_accuracy']:<9.3f} {r['gate_precision']:<7.3f} {r['gate_recall']:<8.3f} {r['mse_base_on_int']:<11.5f} {r['mse_corrected_on_int']:<11.5f} {r['correction_helped_pct']:<9.1%}")

    # Averages
    print("-" * 90)
    avg = lambda key: np.mean([r[key] for r in all_results])
    print(f"{'AVERAGE':<10} {'':5} {'':5} {'':10} {avg('gate_accuracy'):<9.3f} {avg('gate_precision'):<7.3f} {avg('gate_recall'):<8.3f} {avg('mse_base_on_int'):<11.5f} {avg('mse_corrected_on_int'):<11.5f} {avg('correction_helped_pct'):<9.1%}")

    print()
    print(f"Average gate accuracy:      {avg('gate_accuracy'):.1%}")
    print(f"Average gate precision:     {avg('gate_precision'):.1%}")
    print(f"Average gate recall:        {avg('gate_recall'):.1%}")
    print(f"MSE base→oracle (int):      {avg('mse_base_on_int'):.5f}")
    print(f"MSE corrected→oracle (int): {avg('mse_corrected_on_int'):.5f}")
    improvement = (1 - avg('mse_corrected_on_int') / max(avg('mse_base_on_int'), 1e-8)) * 100
    print(f"MSE improvement:            {improvement:+.1f}%")
    print(f"Correction helped:          {avg('correction_helped_pct'):.1%} of intervention steps")
    print(f"Residual mag (int):         {avg('residual_mag_int'):.4f}")
    print(f"Residual mag (non-int):     {avg('residual_mag_non'):.4f}")
    print(f"Ratio (int/non):            {avg('residual_mag_int') / max(avg('residual_mag_non'), 1e-8):.2f}x")

    if args.output_json:
        summary = {
            "per_demo": all_results,
            "aggregate": {
                "gate_accuracy": float(avg("gate_accuracy")),
                "gate_precision": float(avg("gate_precision")),
                "gate_recall": float(avg("gate_recall")),
                "mse_base_on_int": float(avg("mse_base_on_int")),
                "mse_corrected_on_int": float(avg("mse_corrected_on_int")),
                "mse_improvement_pct": float(improvement),
                "correction_helped_pct": float(avg("correction_helped_pct")),
                "residual_mag_int": float(avg("residual_mag_int")),
                "residual_mag_non": float(avg("residual_mag_non")),
            }
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()

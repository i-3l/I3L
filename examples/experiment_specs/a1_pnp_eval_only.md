title = "A1 PnP eval-only rerun"
name = "a1_pnp_eval_only"
root_dir = "/path/to/existing/outputs/experiments/a1_pnp_iterative"
robot = "a1_iiil"
base_task = "pnp/base"
demo_counts = [20]
rounds = 2
methods = ["demos", "cotrain", "hgdagger", "sirius"]
phases = ["eval"]
eval_repeats = 5
use_wandb = true
tags = ["a1", "sim", "eval-only"]

---
Goal:
Rerun evaluation for an already-trained experiment without recollecting or
retraining.

Before running:
- Replace `root_dir` with the real experiment output root
- Make sure the corresponding checkpoints already exist under `train_runs/`

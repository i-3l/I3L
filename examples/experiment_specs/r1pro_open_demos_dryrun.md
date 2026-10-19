title = "R1Pro open-task demos-only dry run"
name = "r1pro_open_demos_dryrun"
robot = "r1pro_iiil"
collect_env = "sim_r1pro_joint"
eval_env = "sim_r1pro_joint"
base_task = "open/base"
collection_task = "open/dc"
base_arch = "wbvima_r1pro"
residual_arch = "residual_r1pro"
demo_counts = [20]
rounds = 0
methods = ["demos"]
dry_run = true
use_wandb = true
tags = ["r1pro", "sim", "dry-run", "demos-only"]

---
Goal:
Validate the command expansion and defaults for an R1Pro setup before attempting
real collection or training.

This is intentionally `dry_run = true` because the correction path is still less
generic for non-A1 robots.

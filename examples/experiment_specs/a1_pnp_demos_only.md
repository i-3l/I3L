title = "A1 PnP demos-only baseline"
name = "a1_pnp_demos_only"
robot = "a1_iiil"
base_task = "pnp/base"
demo_counts = [20, 40]
rounds = 0
methods = ["demos"]
train_max_steps = 50000
eval_repeats = 3
use_wandb = true
tags = ["a1", "sim", "baseline", "demos-only"]

---
Goal:
Establish the demos-only baseline before adding correction rounds.

Compare:
- 20 demos vs 40 demos
- whether the policy is already good enough that correction rounds are not worth it

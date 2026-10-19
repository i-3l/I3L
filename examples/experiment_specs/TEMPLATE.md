# Copy this file and replace the values you care about.
# Keys match long CLI flags, but use underscores instead of dashes.

title = "Short human-readable title"
name = "experiment_name"
robot = "a1_iiil"
base_task = "pnp/base"
demo_counts = [20]
rounds = 1
methods = ["demos", "cotrain", "hgdagger"]
iteration_method = "cotrain"
correction_episodes = 20
train_max_steps = 50000
eval_repeats = 3
use_wandb = true
tags = ["tag1", "tag2"]

# Optional examples:
# root_dir = "/absolute/path/to/output_root"
# base_demo_source = "/absolute/path/to/existing_demos.hdf5"
# phases = ["eval"]
# base_train_override = ["trainer.precision=16"]
# eval_override = ["task.some_flag=true"]

---
Goal:
Write what this run is trying to measure.

Questions:
- What should improve?
- What baseline should this be compared against?
- What failure mode should I inspect first if it underperforms?

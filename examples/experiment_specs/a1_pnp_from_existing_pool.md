title = "A1 PnP from an existing demo pool"
name = "a1_pnp_existing_pool"
robot = "a1_iiil"
base_task = "pnp/base"
base_demo_source = "/path/to/seed_demos_master.hdf5"
demo_counts = [20, 40]
rounds = 1
correction_episodes = 20
methods = ["demos", "cotrain", "hgdagger", "sirius"]
iteration_method = "cotrain"
train_max_steps = 50000
eval_repeats = 3
use_wandb = true
tags = ["a1", "sim", "existing-demos"]

---
Goal:
Skip recollecting seed demos and reuse an already-curated demo pool.

Before running:
- Replace `base_demo_source` with a real path
- Run once with `--dry-run` to confirm the path and slice sizes

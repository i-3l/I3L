title = "A1 PnP iterative cotrain baseline"
name = "a1_pnp_iterative"
robot = "a1_iiil"
base_task = "pnp/base"
demo_counts = [20]
rounds = 2
correction_episodes = 20
methods = ["demos", "cotrain", "hgdagger", "sirius"]
iteration_method = "cotrain"
train_max_steps = 50000
eval_repeats = 3
use_wandb = true
tags = ["a1", "sim", "iterative", "cotrain"]
base_train_override = ["trainer.precision=16"]

---
Goal:
Compare whether cumulative correction rounds keep improving over the demos-only
baseline and whether `cotrain` should be the deployed policy for the next round.

Questions:
- Does round 2 improve over round 1?
- How does `hgdagger` compare to `cotrain` on the same cumulative data?
- Are failures concentrated in a small set of rollout modes?

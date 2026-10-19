title = "A1 PnP iterative collection while still deploying demos"
name = "a1_pnp_iterative_deploy_demos"
robot = "a1_iiil"
base_task = "pnp/base"
demo_counts = [20]
rounds = 2
correction_episodes = 20
methods = ["demos", "cotrain", "hgdagger"]
iteration_method = "demos"
train_max_steps = 50000
eval_repeats = 3
use_wandb = true
tags = ["a1", "sim", "iterative", "deploy-demos"]

---
Goal:
Collect multiple correction rounds while keeping the deployed policy fixed to the
original demos-only checkpoint.

This isolates the effect of adding more correction data from the effect of
changing the deployed policy between rounds.

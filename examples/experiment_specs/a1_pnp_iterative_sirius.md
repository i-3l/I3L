title = "A1 PnP iterative sirius deployment"
name = "a1_pnp_iterative_sirius"
robot = "a1_iiil"
base_task = "pnp/base"
demo_counts = [20]
rounds = 2
correction_episodes = 20
methods = ["demos", "sirius", "hgdagger"]
iteration_method = "sirius"
train_max_steps = 50000
eval_repeats = 3
use_wandb = true
tags = ["a1", "sim", "iterative", "sirius"]

---
Goal:
Compare an iterative setup that deploys `sirius` between rounds instead of
`cotrain`.

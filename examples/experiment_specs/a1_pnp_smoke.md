title = "A1 PnP smoke test"
name = "a1_pnp_smoke"
robot = "a1_iiil"
base_task = "pnp/base"
demo_counts = [5]
rounds = 1
correction_episodes = 5
methods = ["demos", "cotrain"]
iteration_method = "cotrain"
train_max_steps = 2000
eval_repeats = 1
use_wandb = true
tags = ["a1", "sim", "smoke"]

---
Goal:
Quickly validate that collection, training, and eval still connect end to end.

This is a cheap sanity-check card, not a meaningful benchmark.

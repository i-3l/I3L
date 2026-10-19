title = "A1 open-task iterative cotrain"
name = "a1_open_iterative"
robot = "a1_iiil"
base_task = "open/base"
collection_task = "open/dc"
demo_counts = [20]
rounds = 2
correction_episodes = 20
methods = ["demos", "cotrain", "hgdagger"]
iteration_method = "cotrain"
train_max_steps = 50000
eval_repeats = 3
use_wandb = true
tags = ["a1", "sim", "open-task", "iterative"]

---
Goal:
Run the same iterative pipeline shape on the open-door task instead of pick and
place.

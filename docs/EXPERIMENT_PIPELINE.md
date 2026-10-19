# Simulation Experiment Pipeline

Use [experiment.sh](../experiment.sh) to collect data, train policies, and
evaluate them in simulation. Install the dependencies in the
[README](../README.md#installation) first. For real-robot collection, use the
[Franka](FRANKA_EXPERIMENT_GUIDE.md) or [R1 Pro](R1PRO.md) guide instead.

This is a configurable experiment helper, not a preset that reproduces every
paper setting. Check the task, data budget, training settings, and selected
checkpoints before comparing a run with the paper.

## Start with a dry run

Run `./experiment.sh` for the interactive wizard, or inspect an example card:

```bash
./experiment.sh cards
./experiment.sh --spec examples/experiment_specs/a1_pnp_iterative.md --dry-run
```

The wizard can create, inspect, and launch cards stored in
[experiment_cards](../experiment_cards). A dry run prints the commands without
running collection or training.

For a small two-round A1 experiment:

```bash
./experiment.sh \
  --name a1_pnp_iter \
  --robot a1_iiil \
  --base-task pnp/base \
  --demo-counts 20 \
  --rounds 2 \
  --correction-episodes 20 \
  --methods demos,hgdagger \
  --iteration-method hgdagger \
  --train-max-steps 50000 \
  --eval-repeats 3 \
  --no-use-wandb \
  --dry-run
```

Remove `--dry-run` to execute after checking the commands. These are example
budgets; the paper's simulated tasks are `pnp_hard` and `open_radio`.

## Collection and training rounds

For each value in `--demo-counts`, round 0 trains a demonstrations-only base
policy. Each subsequent round collects corrections against a served policy,
combines the original demonstrations with all corrections collected so far,
then trains and evaluates the requested methods.

`--iteration-method` selects which method supplies the policy for the next
round. Set it explicitly when comparing collection schedules. Without an
override, the preference is `cotrain`, `hgdagger`, `sirius`, then `demos`.
Selecting `demos` keeps collecting against the original base policy.

The helper calls `run_task.py` for collection and evaluation,
`scripts/data/process_data.py` for intervention labels, and the sibling
`il_lib/train.py` and `il_lib/serve.py` for training and serving.

| Method | Training data |
|---|---|
| `demos` | Original demonstrations only |
| `cotrain` | Demonstrations and all collected rollout frames |
| `hgdagger` | Demonstrations and active corrections (`hg_daggar` sampler) |
| `sirius` | Collected data with SIRIUS sampling |
| `residual` | Residual architecture conditioned on the base checkpoint |

Base methods use `--base-arch`; residual training uses `--residual-arch`.
The pipeline supports residual training only with `--rounds 1`.

## Residual configurations

The pipeline defaults to `residual_a1` for A1 and `residual_r1pro` for R1 Pro.
These use image-based residual models. The retained
`simple_residual_a1_v12` and `simple_residual_a1_v13` configs are separate
task-state experiments, not interchangeable replacements for those baselines:

- **v12:** first-stage action-head training without an intervention gate.
- **v13:** second-stage gate training initialized from a v12 checkpoint, using
  `+module.stage2_ckpt_path=<V12_CKPT>`.

Both simple configs use task-state features, proprioception, and the base
action. The pipeline does not automatically orchestrate their two-stage
training. `eval.sh` and the residual analysis scripts default to v13; choose
`--arch` to match the checkpoint being inspected.

## Reuse data or run selected phases

Reuse an existing HDF5 file or directory with `--base-demo-source`:

```bash
./experiment.sh --name a1_existing \
  --base-demo-source /path/to/demos_or_dir \
  --base-task pnp/base --demo-counts 20,40 --dry-run
```

Use the same experiment name or root when resuming an existing run.

| Phase | Action |
|---|---|
| `seed_data` | Collect or locate the initial demonstrations |
| `base_train` | Train the demonstrations-only policy |
| `correction_data` | Collect and process each round's corrections |
| `correction_train` | Train the requested correction methods |
| `eval` | Evaluate available checkpoints |
| `full` | Run all phases |

For example, with an existing experiment card:

```bash
./experiment.sh --spec /path/to/card.md --phases eval
./experiment.sh --spec /path/to/card.md \
  --skip-collection --phases correction_train,eval
```

## Experiment cards and overrides

Cards contain `key = value` settings, an optional `---` separator, and notes:

```text
title = "A1 baseline"
name = "a1_baseline"
base_task = "pnp/base"
demo_counts = [20]
rounds = 0
methods = ["demos"]
use_wandb = false

---
Train and evaluate a demonstrations-only baseline.
```

Keys correspond to long CLI options with underscores instead of dashes.
Values may be strings, numbers, booleans, or lists. `title` and `tags` are
metadata only. CLI flags override card settings:

```bash
./experiment.sh --spec /path/to/card.md --train-max-steps 10000 --eval-repeats 1
```

The [example cards](../examples/experiment_specs) cover demos-only training,
iterative collection, fixed-base-policy collection, reuse of existing demos,
evaluation-only runs, smoke tests, and an R1 Pro dry run.

Frequently used options:

| Purpose | Options |
|---|---|
| Output location | `--name`, `--root-dir` |
| Robot and task | `--robot`, `--base-task`, `--collection-task` |
| Simulation environments | `--collect-env`, `--eval-env` |
| Collection policies | `--base-demo-policy`, `--correction-policy` |
| Collection budget | `--demo-counts`, `--rounds`, `--correction-episodes` |
| Architectures | `--base-arch`, `--residual-arch` |
| Execution | `--phases`, `--skip-collection`, `--dry-run` |

Forward Hydra overrides with `--correction-collect-override`,
`--base-train-override`, `--reweight-train-override`,
`--residual-train-override`, or `--eval-override`. Repeat the option for
multiple overrides:

```bash
./experiment.sh --spec /path/to/card.md \
  --base-train-override trainer.precision=16 \
  --base-train-override bs=64 --dry-run
```

Use `--help` for the full option list. `--force-overwrite` reruns steps and
recreates generated artifacts; use it only when replacing those outputs is intended.

## Outputs and logging

Outputs default to `outputs/experiments/<name>/`:

| Path | Contents |
|---|---|
| `datasets/seed_master/` | Initial demonstration pool |
| `datasets/human/nXXX/` | Selected demonstration subset |
| `datasets/corrections/nXXX/round_YY/` | Corrections from one round |
| `datasets/merged/nXXX/round_YY/` | Links to demonstrations and corrections through that round |
| `train_runs/` | Training outputs |
| `eval/nXXX/round_YY/<method>/` | Evaluation JSON files and summaries |
| `manifest.tsv` | Method-to-checkpoint mapping |
| `experiment_config.json` | Resolved pipeline settings |
| `RUN.md`, `experiment_spec.<ext>` | Launch command, notes, and source card |

The parent output directory also contains `experiment_registry.tsv`.
Run records can contain local paths and notes; review them before sharing.
They are generated outputs, not files to include in the source release.

W&B is enabled by default. Pass `--no-use-wandb` to keep training logging
local; otherwise configure your account before launch. The helper does not
check W&B login in advance.

Commands use the active environment unless you set `--collect-env-name`,
`--process-env-name`, `--train-env-name`, `--serve-env-name`, or
`--eval-env-name`.

## Robot support and evaluation checks

A1 is the default path: `a1_iiil`, `sim_a1_joint`, `dp_a1`,
`residual_a1`, with `curobo` demonstrations and `sa` corrections.
R1 Pro requires explicit settings, and the shared-autonomy collection path
is A1-oriented. Inspect the R1 Pro example with `--dry-run` before using it;
do not assume it provides an end-to-end R1 Pro reproduction.

Two details matter when interpreting outputs:

- The helper selects the lowest loss encoded in checkpoint filenames, using
  modification time to break ties. This is not necessarily the last training
  checkpoint; verify `manifest.tsv` against your intended evaluation protocol.
- The results-file check requires an `SR` field but does not verify that all
  requested rollouts completed. Check process logs and completion counts before
  treating interrupted runs as valid evaluations.

Start with a small demonstrations-only run, verify its data and evaluation
outputs, then add correction rounds.

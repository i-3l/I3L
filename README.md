<div align="center">

# I3L

### Iterative Interactive Imitation Learning for Whole-Body Mobile Manipulation

**Whole-body corrections. Iterative policy refinement.**

[Project Website](https://i-3l.github.io/) ·
[Paper](https://arxiv.org/abs/2610.17748) ·
[Models & Datasets](https://huggingface.co/I3L) ·
[Hardware](https://i-3l.github.io/#hardware)

<a href="https://i-3l.github.io/">
  <img src="https://i-3l.github.io/static/images/teaser.jpg" alt="I3L: whole-body correction with JoyLo+, iterative retraining, and tasks across robot embodiments." width="100%">
</a>

</div>

I3L is a hardware and learning system for iterative whole-body correction in
mobile manipulation. A policy runs autonomously while motorized JoyLo+ leader
arms track its commands; when the policy is about to fail, the operator touches
the arms to take over, corrects the robot, and releases control. Corrections are
aggregated with prior demonstrations to retrain the policy, which is redeployed
for the next round.

Explore robot videos, interactive results, and the JoyLo+ hardware on the
[project website](https://i-3l.github.io/).

## Getting Started

- Start here for installation and a simulation example.
- [Simulation experiments](docs/EXPERIMENT_PIPELINE.md): collection rounds,
  training methods, experiment cards, and evaluation checks.
- [Franka setup](docs/FRANKA.md), [collection schedule](docs/FRANKA_EXPERIMENT_GUIDE.md),
  and [correction sessions](docs/CORRECTION_EXPERIMENT_GUIDE.md).
- [R1 Pro setup and operation](docs/R1PRO.md).
- [JoyLo+ configuration](iiil/configs/gello/README.md) and
  [optional debug logging](iiil/gello/debug/README.md).

## What's Included

This repository contains:

- **Teleoperation and correction:** JoyLo+ / GELLO bilateral control (gravity and
  friction compensation, PID goal tracking, touch- and current-based takeover
  detection), JoyCon input, and admittance-mode correction.
- **Robots and tasks:** simulated A1, R1 Pro, and Franka in OmniGibson; real
  Franka Panda and Galaxea R1 Pro interfaces; simulated pick-and-place, open,
  pour, slice, and radio tasks.
- **Data collection:** HDF5 recording of observations, actions, and intervention
  state, plus processing tools (intervention labeling, merging, video export,
  LeRobot conversion, subtask annotation).
- **Learning:** dataset classes and sampling strategies (HG-DAgger, SIRIUS,
  co-training) and architecture configs used with
  [il_lib](https://github.com/wensi-ai/il_lib) for training and serving.
- **Experiment pipeline:** multi-round collect → train → evaluate orchestration
  in simulation.

## Installation

I3L runs on Linux with an NVIDIA GPU and depends on two companion repositories
installed from source:

| Repository | Branch | Commit |
|---|---|---|
| [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) (OmniGibson) | `i3l` | `5009d69f346bc8f0fafeb13cf836069a8ed7f00e` |
| [il_lib](https://github.com/wensi-ai/il_lib) | `iiil` | `ed04e5983f9d7b9e39167353fbc0b5d33c75f842` |

**1. Clone the repositories** side by side:

```bash
git clone https://github.com/i-3l/I3L.git
git clone -b i3l https://github.com/StanfordVL/BEHAVIOR-1K.git
git clone -b iiil https://github.com/wensi-ai/il_lib.git
```

**2. Set up BEHAVIOR-1K.** This creates a conda environment named `behavior`:

```bash
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --joylo --eval
conda activate behavior
```

Add `--dataset` if you also want the BEHAVIOR assets dataset.

**3. Install cuRobo support** (used for motion planning and the scripted oracle):

```bash
conda install "cuda-compiler=12.4" "cuda-toolkit=12.4" -c conda-forge
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDA_PATH="$CUDA_HOME"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
pip install -e OmniGibson[primitives]
```

**4. Install I3L, then il_lib** (in this order):

```bash
cd ../I3L && pip install -e .
cd ../il_lib && pip install -e .
```

From the I3L directory, optional extras are `pip install -e ".[lerobot]"` for LeRobot datasets,
`".[detectors]"` for the sklearn interrupt detector, `".[mujoco]"` for the
MuJoCo Franka simulator, and `".[dev]"` for tests.

Data, outputs, and logs are written under the repository root by default. Set
`IIIL_BASE_PATH` to use a different root.

The table above records dependency commits; cloning a branch alone does not
pin that version. Check out the listed commits when you need those exact
versions.

## Quick Start

All tasks run through `run_task.py` with Hydra configs:

```bash
python run_task.py env=<env> task=<task> policy=<policy> [overrides...]
```

```bash
# Scripted cuRobo policy on simulated A1 pick-and-place
python run_task.py env=sim_a1_joint task=pnp/base policy=curobo

# JoyLo teleoperation in simulation (requires JoyLo hardware)
python run_task.py env=sim_a1_joint task=pnp/base policy=gello
python run_task.py env=sim_r1pro_joint task=open/base policy=gello policy.gello_config=gello_r1pro

# Inspect the resolved config
python run_task.py env=sim_a1_joint task=pnp/base policy=curobo --cfg job
```

### Configuration Options

| Group | Options |
|---|---|
| `env` | `sim_a1_joint`, `sim_a1_ik`, `sim_r1pro_joint`, `sim_franka_joint`, `real_franka_joint`, `real_r1pro_joint` |
| `task` | `pnp`, `pnp_hard`, `open`, `open_radio`, `radio`, `pour`, `slice` (each with `/base` and `/dc` variants); `real/base`, `real/dc`, `real/playback`; `gello/gello_benchmark` |
| `policy` | `curobo`, `gello`, `il`, `sa`, `spacemouse`, `playback`, `waypoint`, `random_position`, `noise`, `sine`, `dummy`, `residual_eval`, `binary_tarjan` |
| `robot` | `a1_iiil`, `a1_7dof_iiil`, `r1pro_iiil`, `real_r1pro_iiil`, `franka_iiil` (defaults follow `env`) |
| `policy.gello_config` | `gello_a1`, `gello_r1pro`, `gello_franka` (defaults follow the robot) |

The paper's simulated tasks are `pnp_hard` (A1 Pick-and-Place Strawberries) and
`open_radio` (R1 Pro Open Fridge and Pick Up Radio).

### JoyLo Control Modes

The `gello` policy supports three control directions
(`policy.control_direction`):

| Mode | Behavior |
|---|---|
| `gello_to_env` | The operator drives the robot directly. |
| `env_to_gello` | A base policy drives the robot; the JoyLo arms follow. |
| `admittance` | The base policy drives the JoyLo arms with low stiffness; the operator can push against them to correct the robot. |

With gating enabled (`policy.admittance.gating.enabled=true`), the robot follows
the policy until a takeover is detected and follows the JoyLo arms during the
correction. See `iiil/configs/policy/gello.yaml` and
`iiil/configs/gello/README.md` for tuning.

## Data Collection

Use a `*/dc` task with `mode=collection`:

```bash
# Human demonstrations via JoyLo
python run_task.py env=sim_a1_joint task=pnp_hard/dc policy=gello \
    mode=collection hdf_path=data/demos.hdf5

# Corrections against a served policy (shared autonomy)
python run_task.py env=sim_a1_joint task=pnp_hard/dc policy=sa \
    mode=collection hdf_path=data/corrections.hdf5
```

During collection, the JoyCon controls each trial: **X** starts recording,
**A** ends the trial as a success (saved), **Y** ends it as a failure
(discarded), and **B/Home** moves the robot home and starts recording if no
trial is active. Clear the home path first. By default only successful
episodes are saved; set `log_only_success=false` to keep all of them.

Each episode is stored under `data/demo_<i>/` with `obs/`, `action`, `reward`,
`terminated`, `truncated`, `time/`, and `policy/` (intervention flags plus base
and oracle actions).

### Processing

```bash
python scripts/data/process_data.py -i data/corrections.hdf5   # add int_state labels
python scripts/data/process_data.py -i data/demos.hdf5 -d     # label full demonstrations
python scripts/data/merge_hdf5.py a.hdf5 b.hdf5 -o merged.hdf5    # merge sessions
python scripts/data/chunk_dataset.py -i data/demos.hdf5 -o data/demos_20.hdf5 -n 20   # keep the first N demos
python scripts/data/hdf5_to_video.py data/demos.hdf5             # export camera video
python scripts/data/hdf2lerobot.py -i data/demos.hdf5 -o data/lerobot -r A1 -t pnp_hard   # convert to LeRobot (-r A1 or R1Pro)
```

`process_data.py` labels every frame with an intervention state:
`0` human demonstration, `1` pre-intervention, `2` active intervention,
`3` autonomous execution.

Processing modifies the HDF5 file in place; preserve the original recordings.

## Training and Evaluation

Training and serving use il_lib, which picks up I3L's architecture, robot, and
task configs automatically:

```bash
cd ../il_lib

# Diffusion policy on demonstrations
python train.py arch=dp_a1 robot=a1_iiil task=pnp_hard/base data_dir=/path/to/data

# HG-DAgger round: demonstrations + corrections, intervention-filtered sampling
python train.py arch=dp_a1 robot=a1_iiil task=pnp_hard/base data_dir=/path/to/merged \
    data.reweight_strategy=hg_daggar

# Serve a checkpoint over websocket (port 8000)
python serve.py arch=dp_a1 robot=a1_iiil task=pnp_hard/base ckpt_path=/path/to/ckpt.pth
```

`data.reweight_strategy` selects the correction-learning method: `hg_daggar`
(HG-DAgger: demonstrations and interventions only), `sirius` (SIRIUS
reweighting), or `cotrain` (all data at its original ratio). Weights & Biases logging is on
by default; pass `use_wandb=false` to disable it.

Evaluate a served policy in simulation:

```bash
# From the I3L repository
cd /path/to/I3L
python run_task.py env=sim_a1_joint task=pnp_hard/base policy=il

# Multi-checkpoint evaluation on A1 Pick-and-Place Strawberries
./eval_pnp_hard.sh --episodes 50 --seed 0 /path/to/ckpt1.pth /path/to/ckpt2.pth
```

## Experiment Pipeline

`experiment.sh` runs the full simulated loop: seed demonstrations, base
training, correction collection against the deployed policy, retraining for
each method, and evaluation.

```bash
./experiment.sh                                          # interactive menu
./experiment.sh --spec examples/experiment_specs/a1_pnp_iterative.md
./experiment.sh --dry-run --name a1_pnp_debug --robot a1_iiil --base-task pnp/base
```

See [docs/EXPERIMENT_PIPELINE.md](docs/EXPERIMENT_PIPELINE.md) for all options
and [examples/experiment_specs](examples/experiment_specs) for experiment cards.

The example cards are starting points, not exact paper-reproduction presets.
The pipeline guide also distinguishes the image-based residual baselines from
the experimental v12/v13 task-state models.

## Real Robots

- **Franka Panda + JoyLo+:** [docs/FRANKA.md](docs/FRANKA.md) (setup),
  [docs/FRANKA_EXPERIMENT_GUIDE.md](docs/FRANKA_EXPERIMENT_GUIDE.md) (collection
  schedule), and
  [docs/CORRECTION_EXPERIMENT_GUIDE.md](docs/CORRECTION_EXPERIMENT_GUIDE.md)
  (operator walkthrough for a correction session).
- **Galaxea R1 Pro + JoyLo+:** [docs/R1PRO.md](docs/R1PRO.md).

`./run.sh` provides a menu for the common real-robot modes. Hardware helpers:
`scripts/setup_gello_ports.py` (USB port symlinks), `scripts/read_gello_joints.py`
(read JoyLo joints), `connect.sh` (JoyCon pairing), `align.sh` (scene
alignment), and `flash.sh` (touch-sensor firmware).

## Repository Layout

```
run_task.py            Entry point (Hydra)
iiil/
  configs/             Hydra configs: env, task, policy, robot, gello, arch
  policies/            Policies, including GelloPolicy and its teleop pipeline
  gello/               JoyLo hardware stack: drivers, control, compensation, touch sensor
  detectors/           Interrupt/release detectors for shared autonomy
  datas/               Training datasets and samplers
  tasks/, wrappers/    OmniGibson tasks, robot and environment wrappers
scripts/               Data processing, evaluation, visualization, hardware utilities
docs/                  Robot setup and experiment guides
examples/              Experiment cards and debug-logging examples
assets/                JoyLo URDFs and meshes
tests/                 Unit and integration tests
```

## Tests

```bash
pytest tests/unit
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

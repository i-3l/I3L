# Franka Correction Session

During correction collection, the previous round's policy controls the robot.
Touch and move JoyLo+ when a correction is needed, then release it when the
policy can continue. Record the completed episode for the next training round.

Complete the hardware setup in [FRANKA.md](FRANKA.md) first. The
[Franka Experiment Guide](FRANKA_EXPERIMENT_GUIDE.md) defines the collection
schedule, button mappings, and safety rules.

## Prepare the session

Have the E-stop within reach, power on JoyLo+, and pair the right JoyCon:

```bash
./connect.sh
```

Assign an operator ID and correction round before recording. Use a separate
file for each session, for example `hang_corr1_op1.hdf5` or
`hang_corr1_op1_2.hdf5` for a second session.

For Mug Hanging, align the scene with:

```bash
./align.sh
```

Select `hang`. Match the rack to the reference overlay; place the mug at a
varied position inside the green region. Recheck alignment if the rack moves.

Run the pre-flight check and resolve any failed checks before proceeding:

```bash
./setup_franka_robot.sh
```

With the workcell clear, unlock the joints and activate FCI in Franka Desk.
Check that the robot is ready before starting collection.

## Start the policy and collector

In a separate terminal, serve the previous round's checkpoint (the base
policy for round 1):

```bash
cd /path/to/il_lib
python serve.py arch=dp_franka robot=franka_iiil task=real/base \
    ckpt_path=/absolute/path/to/checkpoint.pth
```

Wait for the server on port 8000. Then, from I3L:

```bash
./run.sh 11
```

Enter the session's HDF5 filename when prompted. The collector runs pre-flight
checks, captures the scene, and connects to the policy. Robot control is live
from launch, including before recording starts.

## Collect one round

Use the [button table](FRANKA_EXPERIMENT_GUIDE.md#buttons). Normally, press
**B** to home and begin recording; **X** starts recording without homing.
Do not press B during an active trial.

- Intervene when failure seems likely, and resume autonomy when the robot can
  continue. An episode may include several corrections.
- Press **A** when the task succeeds, including when corrections were needed.
  Press **Y** when the attempt fails. With the default success-only recording,
  failed attempts are discarded.
- If an episode was saved by mistake, record its zero-based trial number for
  review before training. Keep a record of any exclusions.
- Check terminal feedback after button presses. If a press was missed, confirm
  the current state before retrying.
- Between trials, use the approved safe reset procedure. Recording being off
  does not mean the robot is disabled.

Collect the number of saved episodes specified by the experiment plan
(typically 25 per correction round). Extra episodes are not automatically
assigned to validation; the split is made during data preparation.

## Finish and prepare the data

After ending the last trial, clear the home path and press **+** to finish
collection and home the robot. Stop the policy server after the collector
exits. Follow the lab's shutdown procedure and lock the joints when finished.

To combine sessions and reserve episodes beyond the first 25 for validation:

```bash
python scripts/data/merge_hdf5.py hang_corr1_op1.hdf5 hang_corr1_op1_2.hdf5 \
    -o corr1_train/expert.hdf5 --train-count 25 --val-output corr1_val/expert.hdf5
python scripts/data/process_data.py -i corr1_train/expert.hdf5
```

If a validation file was created, process it as well:

```bash
python scripts/data/process_data.py -i corr1_val/expert.hdf5
```

The merge tool assigns episodes in input order; it does not choose episodes
by policy performance. Keep the original recordings: processing modifies
HDF5 files in place.

Train on the original demonstrations plus all correction rounds collected so
far with `data.reweight_strategy=hg_daggar`; see
[Training and Evaluation](../README.md#training-and-evaluation). Serve the new
checkpoint before starting the next round.

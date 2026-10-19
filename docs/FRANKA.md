# Franka Panda + Joylo Integration

Guide for running IIIL with a real Franka Panda arm using a 7-DOF Joylo controller (Dynamixel XL430 + XL330 motors) and deoxys for robot control.

For day-to-day data collection, use the concise
[Franka Experiment Guide](FRANKA_EXPERIMENT_GUIDE.md).

## Architecture

```
Workstation                        NUC                     Franka
┌────────────────────────┐   ┌──────────────────┐   ┌──────────────┐
│  IIIL (run_task.py)    │   │  auto_arm.sh     │   │  Control Box │
│  ├── GelloPolicy       │   │  franka-interface │   │  + Arm       │
│  │   └── Bilateral     │   │  (libfranka)     │   │              │
│  │       Controller    │   └──────────────────┘   └──────────────┘
│  │       (Joylo 100Hz) │        via ZMQ                via ethernet
│  └── RealFranka        │
│      └── DeoxysFranka  │──ZMQ──┘
│         Interface      │
│                        │
│  Touch sensor (ESP32)  │──USB serial
│  Joylo motors          │──USB serial x2
│  JoyCon                │──Bluetooth
└────────────────────────┘
```

Key design: IIIL's `BilateralController` replaces FrankaJoylo's `JoyloSystem` for Joylo hardware control. This gives us current reading, takeover detection (via touch sensor), compensation, PID tracking, and the full teleop/admittance/data-collection pipeline. Only `DeoxysFrankaInterface` is carried over from FrankaJoylo for Franka arm control.

## One-time setup

Complete the [main installation](../README.md#installation), then install the
Franka-specific dependencies below in the same `behavior` environment.
The commands assume you have obtained FrankaJoylo and deoxys_control.

**1. Install FrankaJoylo and deoxys:**

```bash
cd /path/to/FrankaJoylo && python -m pip install -e .
cd /path/to/deoxys_control/deoxys && python -m pip install -e .
```

**2. System dependencies and udev rules:**
```bash
sudo apt install libhidapi-dev

# Joylo USB port symlinks
cd /path/to/FrankaJoylo && ./setup.sh

# Touch sensor symlink
sudo cp /path/to/I3L/scripts/99-touch-sensor.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Start a session

**1. Network:** Workstation ethernet must be `172.16.0.1/24` on the interface connected to the switch.

**2. Hardware startup:**
```bash
# Power on control box + NUC
# Franka Desk (http://172.16.0.2/desk) → unlock joints → activate FCI
# SSH to NUC:
ssh <nuc-user>@<nuc-ip>
tmux
cd deoxys_control/deoxys
./auto_scripts/auto_arm.sh config/<your-robot>.yml
# Leave running, plug in Joylo USB + touch sensor, pair JoyCon
```

**3. Verify and calibrate:**
```bash
conda activate behavior

# Check imports
python -c "from franka_joylo.deoxys_franka import DeoxysFrankaInterface; print('OK')"
python -c "from iiil.wrappers.robots import RealFranka; print('OK')"

# Verify joint offsets (Franka does NOT move, Joylo snaps to match)
cd /path/to/I3L
python run_task.py env=real_franka_joint task=real/base policy=gello \
    policy.gello_config=gello_franka policy.control_direction=env_to_gello
```

**4. Run:**
```bash
./run.sh  # Option 10 for DC, 11 for correction
```

The `config.json` in FrankaJoylo is already tuned and checked in, so `setup.sh` is mainly for the udev rules — it won't overwrite existing calibration.

---

## Detailed Reference

### Calibration

#### Joint offset verification

The offsets in `iiil/configs/gello/gello_franka.yaml` were translated from FrankaJoylo's `config.json`. Verify on hardware by running tracking mode (Franka does NOT move):

```bash
python run_task.py env=real_franka_joint task=real/base policy=gello \
    policy.gello_config=gello_franka policy.control_direction=env_to_gello
```

The Joylo should snap to match the Franka's current pose. If joints are offset, adjust `joint_offsets` in `gello_franka.yaml`.

#### Takeover threshold tuning

The default config uses touch-only takeover detection (`use_touch: true`, `use_current: false`). If you later want current-based detection, tune thresholds with:

```bash
python scripts/auto_tune_takeover.py --config gello_franka
```

### Running

#### Quick start (run.sh)

```bash
cd /path/to/I3L
./run.sh
# Select:
#   10 — Franka Data Collection (gello_to_env teleop + HDF5 recording)
#   11 — Franka Policy Correction (IL + admittance + touch takeover + HDF5)
```

#### Direct commands

**Teleop (robot WILL move — hold the E-stop):**

```bash
python run_task.py env=real_franka_joint task=real/base policy=gello \
    policy.gello_config=gello_franka policy.control_direction=gello_to_env \
    policy.require_joycon=false
```

**Data collection:**

```bash
python run_task.py env=real_franka_joint task=real/dc policy=gello \
    policy.gello_config=gello_franka policy.base_policy=dummy \
    policy.control_direction=gello_to_env policy.require_joycon=false \
    mode=collection hdf_path=data/franka_demos.hdf5
```

**Policy correction (admittance + touch takeover):**

```bash
python run_task.py env=real_franka_joint task=real/base policy=gello \
    policy.gello_config=gello_franka policy.base_policy=il \
    policy.require_joycon=false policy.takeover_enable=true \
    mode=collection enforce_frequency=true hdf_path=data/franka_corrections.hdf5
```

#### JoyCon controls during data collection

JoyCon is optional. If paired, these buttons are available:

| Button | Action |
|--------|--------|
| X | Start recording trial |
| A | End trial as **success** (saved) |
| Y | End trial as **failure** (discarded) |
| B or Home | Move home, then start recording if no trial is active |
| Plus (+) | Finish collection and go home |
| R / ZR | Close / open the gripper |

Do not press B/Home during recording. Homing is not collision-planned: clear
its path first. Recording being off does not disable robot control. Follow the
[safety checklist](FRANKA_EXPERIMENT_GUIDE.md#safety), and use the physical
E-stop for unsafe motion.

## Configuration Reference

### Key config files

| File | Purpose |
|------|---------|
| `iiil/configs/env/real_franka_joint.yaml` | Environment config (env_type, frequency) |
| `iiil/configs/robot/franka_iiil.yaml` | Robot config (type, action_dim, action_keys) |
| `iiil/configs/gello/gello_franka.yaml` | Gello hardware config (motors, offsets, takeover, touch sensor) |

### Motor groups

| Group | Motor Type | Motor IDs | Franka Joints | USB Port |
|-------|-----------|-----------|---------------|----------|
| 12V | XL430-W250-T | 0, 1, 2 | 0, 1, 3 | `/dev/franka_large` |
| 5V | XL330-M288-T | 0, 1, 2, 3 | 2, 4, 5, 6 | `/dev/franka_small` |

### Joint offset formula

Offsets were translated from FrankaJoylo's calibration using:

```
iiil_offset = fj_offset + (2048 - DXL_ZERO_OFFSETS) * fj_sign * pi/2048
```

### Tunable parameters in gello_franka.yaml

| Section | Parameter | Default | Notes |
|---------|-----------|---------|-------|
| `touch_sensor` | `enable` | `true` | Requires ESP32-S3 + MPR121 |
| `takeover` | `use_touch` | `true` | Touch-only mode (clean binary signal) |
| `takeover` | `use_current` | `false` | Enable for mixed-mode detection |
| `compensation.anti_friction` | `viscous_gain` | `0.01` | Increase if Joylo is hard to backdrive |
| `goal_tracking` | `kp` | `[0.3, ..., 0.1]` | Tracking stiffness (admittance mode) |
| `gello_to_env` | `torque_enabled` | `false` | Free-spinning (Joylo is lightweight) |

## Troubleshooting

- **"Timed out waiting for Franka joint state"** — deoxys on the NUC isn't running. SSH in and start `auto_arm.sh`.
- **"[TxRxResult] There is no status packet!"** — Dynamixel motor not responding. Check USB cables, run `python scripts/read_gello_joints.py` to scan.
- **JoyCon `ImportError: Unable to load libhidapi`** — install `sudo apt install libhidapi-dev`.
- **Touch sensor not detected** — check `/dev/tty_touch_sensor` exists. If not, replug the ESP32-S3 and check udev rules.
- **Joints move in wrong direction** — flip the sign in `arm_joint_signs` for the affected motor group in `gello_franka.yaml`.
- **Joylo doesn't match Franka pose in tracking** — adjust `joint_offsets` in `gello_franka.yaml`. Use sim teleop in FrankaJoylo to cross-check.

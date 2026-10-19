# JoyLo+ Hardware Stack

This package connects the JoyLo+ leader arms to I3L's teleoperation and
correction policies. It handles motor communication, policy-target tracking,
gravity and friction compensation, and takeover sensing.

For demonstrations and the hardware overview, see the
[I3L project website](https://i-3l.github.io/#hardware).

## Start here

1. Follow the [Franka](../../docs/FRANKA.md) or
   [R1 Pro](../../docs/R1PRO.md) setup guide for the robot.
2. Select and verify the [hardware configuration](../configs/gello/README.md):
   motor IDs, ports, joint mapping, signs, offsets, and limits.
3. Set up the [capacitive touch sensor](capacitive_sensor/README.md).
4. Use the robot guide's launch commands. Keep the workcell clear and the
   physical E-stop accessible; the leader arms can move during tracking.

The software retains the name `gello` in paths and configuration keys.

## How the components fit together

```text
GelloPolicy and teleoperation pipeline
  -> controller_factory.py / factory/
  -> controllers/bilateral_controller.py
       -> drivers/       motor communication and operating modes
       -> control/       goal tracking, takeover detection, touch input
       -> compensation/  gravity and friction compensation
       -> debug/         optional diagnostic recording and plots
```

The current implementation is `BilateralController`, derived from
`BilateralGelloController`. The factory supports
`bilateral_control_type: bilateral`; older concurrent, parallel, and
`SimpleBilateralController` examples do not apply to this version.

The controller supports goal tracking as well as hand-guided motion. The
policy layer selects the control direction and coordinates robot actions,
JoyCon inputs, recording, and takeover transitions.

| Policy control direction | Purpose |
|---|---|
| `gello_to_env` | The operator drives the robot with the leaders |
| `env_to_gello` | The leaders follow the robot/policy targets |
| `admittance` | Policy tracking with compliant operator correction |

Use `policy.control_direction` to select a mode. Takeover gating, policy
resumption, and transition settings live in
[the policy config](../configs/policy/gello.yaml); hardware sensing and gains
live in [the robot-specific configs](../configs/gello).

## Hardware and calibration

The supplied configs cover a six-joint A1 leader, two seven-joint R1 Pro
leaders, and a seven-joint Franka leader. Their motor types, bus layouts,
signs, and offsets differ. Do not transfer calibration between them without
checking the physical hardware.

Joint mapping and encoder conversion are handled by the motor groups and
[encoder utilities](utils/encoder_utils.py). Gravity compensation uses the
checked-in URDFs and per-joint multipliers. Verify the URDF frame and joint
signs before tuning compensation; incorrect signs can produce unsafe motion.

For a new assembly, start from the closest complete hardware config and
follow the [configuration guide](../configs/gello/README.md). Do not use a
partial example as a ready-to-run calibration.

## Diagnostics

Console debug messages and structured recording are disabled in the supplied
configs. Enable them only when needed; see the
[debug guide](debug/README.md) for configuration and file formats.

Useful checks, run from the repository root:

```bash
python scripts/read_gello_joints.py --list-ports
python scripts/read_gello_joints.py --config gello_franka
```

Check port assignments, motor IDs, and baud rates before tuning gains. For
touch-sensor problems, use the
[sensor troubleshooting guide](capacitive_sensor/README.md#troubleshooting).

## Source reference

- [Factory](controller_factory.py): creates a controller from a hardware config.
- [Config loader](factory/config_loader.py): loads YAML and optional task overlays.
- [Controller](controllers/bilateral_controller.py): control loop and operating modes.
- [Goal tracking](control/goal_tracker.py) and
  [takeover detection](control/takeover_detector.py).
- [Compensation](compensation): gravity, friction, and backdriving assistance.
- [Motor drivers](drivers): low-level Dynamixel communication.

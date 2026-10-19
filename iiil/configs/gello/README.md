# JoyLo+ Hardware Configuration

Use a complete robot-specific YAML file as the starting point. These configs
include calibration for the associated leader assembly; verify the settings
against your hardware before enabling motion.

For robot startup and operator controls, see the
[Franka](../../../docs/FRANKA.md) and [R1 Pro](../../../docs/R1PRO.md) guides.
For the software structure, see the [hardware stack](../../gello/README.md).

## Choose a configuration

| Config | Leader joints | Motor groups |
|---|---|---|
| [gello_a1.yaml](gello_a1.yaml) | 6 | Two groups on two buses |
| [gello_r1pro.yaml](gello_r1pro.yaml) | 14: right 0–6, left 7–13 | Four groups sharing two buses |
| [gello_franka.yaml](gello_franka.yaml) | 7 | Two groups on two buses |

Select one with `policy.gello_config=gello_franka`, for example.
The factory currently supports only `bilateral_control_type: bilateral`.
It supports policy-target tracking and compliant correction, not just
current-only teleoperation.

### Motor mapping in the supplied configs

Joint indices are zero-based. Each row lists motor IDs and their corresponding
joint indices in the same order. These are software settings, not a substitute
for checking the installed motor labels.

| Config / group | Motor type | Motor IDs | Joint indices | Port |
|---|---|---|---|---|
| A1 large | XM430-W350-T | 0, 1, 2 | 0, 1, 2 | `/dev/tty_large_motor` |
| A1 small | XC330-M288-T | 3, 4, 5 | 3, 4, 5 | `/dev/tty_small_motor` |
| R1 Pro right large | XM430-W350-T | 0, 1, 2, 3 | 0, 1, 2, 3 | `/dev/tty_large_motor` |
| R1 Pro right small | XC330-M288-T | 0, 1, 2 | 4, 5, 6 | `/dev/tty_small_motor` |
| R1 Pro left large | XM430-W350-T | 4, 5, 6, 7 | 7, 8, 9, 10 | `/dev/tty_large_motor` |
| R1 Pro left small | XC330-M288-T | 3, 4, 5 | 11, 12, 13 | `/dev/tty_small_motor` |
| Franka large | XL430-W250-T | 0, 1, 2 | 0, 1, 3 | `/dev/franka_large` |
| Franka small | XC330-M288-T | 0, 1, 2, 3 | 2, 4, 5, 6 | `/dev/franka_small` |

The supplied configs use a 1,000,000-baud motor connection and a nominal
100 Hz leader control loop. Check the actual hardware before changing either.

## Configure USB ports

From the repository root:

```bash
python scripts/read_gello_joints.py --list-ports
python scripts/setup_gello_ports.py
```

The setup helper guides you through the motor buses and creates the named
symlinks used by the selected config. Creating links under `/dev` may require
administrator privileges; review any suggested commands before running them.

You can instead edit each group's `device_port`, or override ports for the
joint-reading utility:

```bash
python scripts/read_gello_joints.py --config gello_a1 \
    --ports /dev/ttyUSB0 /dev/ttyUSB1
```

Port overrides follow motor-group order. R1 Pro has four groups on two shared
buses, so do not assume that every group needs a separate USB adapter.

## Settings to verify

| Setting | What it controls |
|---|---|
| `motor_groups[].motor_ids` | IDs present on the group's bus |
| `motor_type` | Driver/register interpretation for that servo model |
| `robot_joint_indices` | Placement in the leader joint vector |
| `arm_joint_signs` | Joint direction multipliers |
| `device_port`, `baudrate` | Serial connection |
| `joint_offsets` | Calibrated angular offsets in radians |
| `joint_limits` | Valid joint ranges, where specified |
| `home_position` | Configured leader home pose, where specified |
| `max_joint_velocity` | Per-group velocity setting in rad/s |
| `current_limit` | Motor current-limit register setting; check driver units |
| `profile_velocity`, `profile_acceleration` | Motor-specific profile register settings |

Do not interpret every current-limit number as milliamps. For example, the
R1 Pro config documents raw Dynamixel units. Check the driver and the motor
manual before changing limits.

### Calibration

The encoder conversion uses a center of 2048 ticks and applies joint signs,
offsets, and limits. A center encoder value is not a universal robot home pose.
Retain the supplied calibration as a reference and verify each joint's
direction and range on your assembly.

Franka's config documents its conversion from the FrankaJoylo calibration.
Its explicit joint limits matter for angle wrapping; do not replace them
with a generic range. Follow the
[Franka calibration procedure](../../../docs/FRANKA.md#calibration).

Incorrect joint signs, offsets, or compensation can cause unexpected motion.
Keep the workspace clear, use the physical E-stop when needed, and verify
mapping before running a policy.

## Tracking, compensation, and takeover

| Section | Purpose |
|---|---|
| `goal_tracking` | Goal-tracking gains and convergence settings |
| `compensation.gravity` | URDF-based gravity compensation and joint multipliers |
| `compensation.friction` | Static-friction compensation |
| `compensation.anti_friction` | Assistance when backdriving the leaders |
| `touch_sensor` | Sensor port and touch/release settings |
| `takeover` | Touch/current detection and release behavior |
| `gello_to_env.torque_enabled` | Whether direct teleoperation uses motor torque |

Use the existing config for gain dimensions and available fields.
Gravity-vector direction, joint signs, and URDF coordinates must agree.
Compensation and torque settings differ between assemblies.

Control direction and takeover gating are policy settings in
[policy/gello.yaml](../policy/gello.yaml). Hardware configs may also provide
`admittance_stiffness` and `takeover_stiffness`; explicit policy overrides
take precedence. See the [touch-sensor guide](../../gello/capacitive_sensor/README.md)
for wiring, firmware, and sensitivity tuning.

## Task overlays and custom assemblies

Optional overlays live in `<config_name>/<overlay>.yaml` and are merged over
the base hardware config. Select one with `policy.gello_task_overlay`.
If an overlay is missing, the loader warns and uses the base config.

For a custom assembly:

1. Copy the closest YAML file within this directory.
2. Update ports, motor types, IDs, mapping, signs, offsets, and limits together.
3. Check the URDF and compensation frame before enabling compensation.
4. Read joint positions and verify the mapping before testing tracking or takeover.
5. Validate operation using the robot-specific safety procedure.

Do not copy gains or assume a calibration is safe solely because the joint
count matches.

## Debugging

Diagnostics are opt-in:

```yaml
debug_messages:
  enable: true
debug_logging:
  enable: true
```

Console message categories have their own switches. Structured logging can
write local diagnostic files; see the [debug guide](../../gello/debug/README.md).

If motors do not respond, check power, USB permissions, port assignments,
motor IDs, and baud rates. For incorrect motion, stop and verify joint mapping,
signs, offsets, and limits before tuning gains. For sensor failures, check the
sensor port and firmware rather than disabling takeover safeguards.

Motor register details are available in the
[ROBOTIS manuals](https://emanual.robotis.com/).

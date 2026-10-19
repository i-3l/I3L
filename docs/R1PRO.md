# R1 Pro Teleoperation Setup

This guide covers setting up and operating the Galaxea R1 Pro with the JoyLo+
interface. Install I3L and its dependencies first (see the [README](../README.md#installation)).

The R1 Pro low-level interface comes from the `brs_ctrl` package, which must be
installed in the same environment.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Test JoyLo in Simulation](#test-joylo-in-simulation)
- [Robot Controller](#robot-controller)
- [Cameras](#cameras)
- [Teleoperation and Data Collection](#teleoperation-and-data-collection)

---

## Prerequisites

Before starting teleoperation:

1. **JoyCons**: Pair both JoyCons over Bluetooth.
2. **JoyLo pose**: Place the JoyLo arms in their initial pose.

---

## Test JoyLo in Simulation

Verify JoyLo control in simulation before connecting to the real robot:

```bash
python run_task.py env=sim_r1pro_joint task=open/base policy=gello \
    policy.control_direction=gello_to_env policy.require_joycon=false
```

All joints should follow the JoyLo arms. JoyLo uses delta joint position tracking.

---

## Robot Controller

### Network Configuration

The workstation talks to the robot over Ethernet using a ROS 2 FastDDS
discovery server. The examples below use `10.0.0.30` for the robot and
`10.0.0.10` for the workstation; adapt them to your network.

1. **On the robot** (log in with a keyboard and monitor if SSH fails):
   ```bash
   sudo ip addr add 10.0.0.30/24 dev eth1
   ```

2. **On the workstation**:
   ```bash
   sudo ip addr add 10.0.0.10/24 dev <ethernet-interface>
   ```

3. **Log in to the robot**:
   ```bash
   ssh <robot-user>@10.0.0.30
   ```

If your robot uses a different address, set `ROS_DISCOVERY_SERVER` before
running `run.sh` or `ros.sh` (default `10.0.0.30:11811`), and update the address
in `scripts/real_robot/super_client_configuration_file.xml`.

### Start the Robot Controller

On the robot:

```bash
cd ~/galaxea/install/startup_config/share/startup_config/script/
./robot_startup.sh boot ../sessions.d/ATCStandard/R1PROBody.d/
```

Verify with `ros2 topic list`.

### Verify the Workstation Connection

On the workstation:

```bash
source ros.sh
ros2 topic list
```

You should see the robot's topics.

### Shutdown

1. Reset the robot:
   ```bash
   source ~/galaxea/install/setup.bash
   python3 ~/galaxea/install/mobiman/share/mobiman/scripts/robotOpenbox/R1Pro/r1pro_test_open_box.py
   ```

2. Stop the controller:
   ```bash
   cd ~/galaxea/install/startup_config/share/startup_config/script/
   ./robot_startup.sh kill
   ```

---

## Cameras

The R1 Pro uses two wrist RealSense cameras and a head ZED 2. Camera streams
are published over ROS 2 from a colcon workspace on the robot (`~/IIIL_ws`
below). Streams only use bandwidth when the workstation subscribes to them.

Start the cameras on the robot:

```bash
source ~/IIIL_ws/install/setup.bash
ros2 launch ~/IIIL_ws/src/camera_launcher/launch/camera_launch.launch.py
```

You should see `/zed/*`, `/left_wrist*`, and `/right_wrist*` topics. You can
also view the streams in `rviz2`.

Camera settings live in the robot-side workspace:

- ZED 2: `zed-ros2-wrapper/zed_wrapper/config/{common_stereo,zed2}.yaml`
- Resolution and launch arguments: `camera_launcher/launch/camera_launch.launch.py`

Rebuild after changing them:

```bash
cd ~/IIIL_ws && colcon build
```

---

## Teleoperation and Data Collection

`./run.sh` provides a menu for the common R1 Pro modes (data collection, policy
rollout, playback, evaluation, and policy correction). The equivalent direct
commands are below. Replace `data/<name>.hdf5` with your output file.

**Data collection (JoyLo teleoperation):**

```bash
python run_task.py env=real_r1pro_joint task=real/dc policy=gello \
    policy.control_direction=gello_to_env policy.require_joycon=true \
    hdf_path=data/<name>.hdf5 overwrite=true
```

**Policy rollout without JoyLo** (serve a policy first; see the README):

```bash
python run_task.py env=real_r1pro_joint task=real/dc policy=dummy \
    hdf_path=data/<name>.hdf5 overwrite=true
```

**Policy correction (policy drives JoyLo; take over by touch):**

```bash
python run_task.py env=real_r1pro_joint task=real/dc policy=gello \
    policy.base_policy=il hdf_path=data/<name>.hdf5 overwrite=true
```

**Replay recorded demonstrations:**

```bash
# All demos, once
python run_task.py env=real_r1pro_joint task=real/base policy=playback \
    policy.model.hdf_path=data/<name>.hdf5

# Specific demos
python run_task.py ... policy.model.demo_ids=[0,2,5]

# Repeat all demos three times
python run_task.py ... policy.model.repeat=3

# Demos 1 and 3, twice each (total episodes = len(demo_ids) x repeat)
python run_task.py ... policy.model.demo_ids=[1,3] policy.model.repeat=2
```

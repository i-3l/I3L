import numpy as np
import torch
import omnigibson.utils.transform_utils as T
from omnigibson.controllers import MultiFingerGripperController, HolonomicBaseJointController
from omnigibson.controllers.controller_view import ControllerView
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.robots.robot import Robot


class SimA1(Robot):
    def __init__(self, **kwargs):
        kwargs.setdefault("model", "a1")
        super().__init__(**kwargs)

    @property
    def model_name(self) -> str:
        return "A1"

    def move_to_home(self) -> None:
        """
        Move robot to home/default joint positions.
        In simulation, this can be done instantly by setting joint positions directly.
        Called when Home button is pressed.
        """
        # Use the robot's configured reset position, or zeros if not set
        if hasattr(self, '_reset_joint_pos') and self._reset_joint_pos is not None:
            home_pos = self._reset_joint_pos
        else:
            home_pos = torch.zeros(self.n_dof)
        self.set_joint_positions(home_pos, drive=True)

    def absolute_base_action_to_delta(self, action: torch.Tensor) -> torch.Tensor:
        """
        Converts the base action from absolute (w.r.t. root link) to delta (w.r.t. base footprint link).
        Since A1 is a non-holonomic robot, this function is a no-op.
        Args:
            action (torch.Tensor): action computed by the policy
        Returns:
            torch.Tensor: action to be applied to the robot
        """
        return action

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Function to perform any processing of the action before applying it to the robot.
        Args:
            action (torch.Tensor): action computed by the policy 
        Returns:
            torch.Tensor: action to be applied to the robot
        """
        assert self.action_dim == action.shape[0], f"Action dimension mismatch: {self.action_dim} != {action.shape[0]}"
        return action

    def get_no_op_action(self) -> torch.Tensor:
        action = torch.zeros(self.action_dim)
        for name, (group_key, controller_idx) in self.controllers.items():
            partial_action = ControllerView.compute_no_op_action(group_key, controller_idx)
            action[self.controller_action_idx[name]] = partial_action
        return action
    
    def q_to_action(self, q: torch.Tensor) -> torch.Tensor:
        input_is_1d = q.ndim == 1
        if input_is_1d:
            q = q.unsqueeze(0)
        action = []
        for _, (group_key, _) in self.controllers.items():
            command = q[:, ControllerView.get_dof_idx(group_key)]
            if ControllerView.is_controller_type(group_key, MultiFingerGripperController):
                command = command[:, :1]
            command = ControllerView.reverse_preprocess_command(group_key, command)
            action.append(command)
        action = torch.cat(action, dim=1)
        assert action.shape[1] == self.action_dim, (
            f"Action should have dimension {self.action_dim}, got {action.shape[1]}"
        )
        if input_is_1d:
            action = action.squeeze(0)
        return action


class SimR1Pro(Robot):
    def __init__(self, **kwargs):
        kwargs.setdefault("model", "r1pro")
        super().__init__(**kwargs)

    @property
    def model_name(self) -> str:
        return "R1Pro"
    
    @property
    def grasping_mode(self):
        """
        Grasping mode of this robot. Is one of AG_MODES

        Returns:
            str: Grasping mode for this robot
        """
        assert self.is_manipulation
        return self._grasping_mode
    
    @grasping_mode.setter
    def grasping_mode(self, mode: str) -> None:
        """
        Grasping mode of this robot. Is one of AG_MODES

        Returns:
            str: Grasping mode for this robot
        """
        self._grasping_mode = mode

    def move_to_home(self) -> None:
        """
        Move robot to home/default joint positions.
        In simulation, this can be done instantly by setting joint positions directly.
        Called when Home button is pressed.
        """
        # Use the robot's configured reset position, or zeros if not set
        if hasattr(self, '_reset_joint_pos') and self._reset_joint_pos is not None:
            home_pos = self._reset_joint_pos
        else:
            home_pos = torch.zeros(self.n_dof)
        self.set_joint_positions(home_pos, drive=True)

    def get_no_op_action(self) -> torch.Tensor:
        action = torch.zeros(self.action_dim)
        for name, (group_key, controller_idx) in self.controllers.items():
            partial_action = ControllerView.compute_no_op_action(group_key, controller_idx)
            action[self.controller_action_idx[name]] = partial_action
        return action
    
    def absolute_base_action_to_delta(self, action: torch.Tensor) -> torch.Tensor:
        """
        Converts the base action from absolute (w.r.t. root link) to delta (w.r.t. base footprint link).
        Args:
            action (torch.Tensor): action computed by the policy
        Returns:
            torch.Tensor: action to be applied to the robot
        """
        # For a holonomic base joint controller, the command should be in the robot local frame
        # For orientation, we need to convert the command to a delta angle
        command = action[self.base_action_idx]
        base_joint_pos = self.get_joint_positions()[self.base_idx]
        cur_rz_joint_pos = base_joint_pos[5]
        delta_q = wrap_angle(command[2] - cur_rz_joint_pos)

        # For translation, we need to convert the command to the robot local frame
        body_pos = base_joint_pos[:3]
        body_quat = T.mat2quat(T.euler_intrinsic2mat(base_joint_pos[3:6]))
        canonical_pos = torch.tensor([command[0], command[1], body_pos[2]], dtype=torch.float32)
        local_pos = T.relative_pose_transform(canonical_pos, torch.tensor([0.0, 0.0, 0.0, 1.0]), body_pos, body_quat)[0]
        action[self.base_action_idx] = torch.tensor([local_pos[0], local_pos[1], delta_q])
        return action

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Function to perform any processing of the action before applying it to the robot.
        Args:
            action (torch.Tensor): action computed by the policy 
        Returns:
            torch.Tensor: action to be applied to the robot
        """       
        assert self.action_dim == action.shape[0], f"Action dimension mismatch: {self.action_dim} != {action.shape[0]}"
        return action

    def q_to_action(self, q: torch.Tensor) -> torch.Tensor:
        input_is_1d = q.ndim == 1
        if input_is_1d:
            q = q.unsqueeze(0)
        action = []
        for _, (group_key, _) in self.controllers.items():
            command = q[:, ControllerView.get_dof_idx(group_key)]
            if ControllerView.is_controller_type(group_key, MultiFingerGripperController):
                command = command[:, :1]
            command = ControllerView.reverse_preprocess_command(group_key, command)
            action.append(command)
        action = torch.cat(action, dim=1)
        assert action.shape[1] == self.action_dim, (
            f"Action should have dimension {self.action_dim}, got {action.shape[1]}"
        )
        if input_is_1d:
            action = action.squeeze(0)
        return action
    
    def reset(self, *args, **kwargs):
        """
        Override the default reset to also reset the base position to the default.
        This is important for sim-to-real transfer, as the real robot will always start at the default position.
        """
        self.links["base_link"].mass = 150.0
        return super().reset(*args, **kwargs)


class SimFranka(Robot):
    """Simulated Franka Panda using MuJoCo via FrankaJoylo's SimFrankaInterface.

    Exercises the same I3L pipeline (GelloPolicy → TeleopPipeline) as RealFranka
    but with a MuJoCo physics backend and viewer, so you can verify Joylo mapping
    and teleop behaviour without the real robot or deoxys.
    """

    @property
    def model_name(self) -> str:
        return "Franka"

    def __init__(self, name: str = "franka") -> None:
        super().__init__(name=name, model="franka")
        self.resetted = False
        self._prev_joint_pos = None
        self._prev_time = None

        import mujoco
        import mujoco.viewer
        import threading

        from franka_joylo.sim import SimFrankaInterface

        self._sim = SimFrankaInterface.create()
        self.interface = self._sim
        self._model = self._sim.model
        self._data = self._sim.data
        self._gripper_state = -1.0

        # Launch MuJoCo viewer in background thread (skip if no display)
        self._viewer_stop = threading.Event()
        self._has_viewer = False
        import os
        if os.environ.get("DISPLAY"):
            self._has_viewer = True
            self._viewer_thread = threading.Thread(target=self._run_viewer, daemon=True)
            self._viewer_thread.start()
        else:
            # Headless: just step physics in background
            self._viewer_thread = threading.Thread(target=self._run_headless, daemon=True)
            self._viewer_thread.start()

    def _run_viewer(self):
        import mujoco
        import mujoco.viewer
        import time

        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            while viewer.is_running() and not self._viewer_stop.is_set():
                mujoco.mj_step(self._model, self._data)
                viewer.sync()
                elapsed = self._model.opt.timestep
                time.sleep(max(0, elapsed))

    def _run_headless(self):
        import mujoco
        import time

        while not self._viewer_stop.is_set():
            mujoco.mj_step(self._model, self._data)
            time.sleep(max(0, self._model.opt.timestep))

    def apply_action(self, action: torch.Tensor) -> None:
        action_np = action.detach().cpu().numpy()
        arm_cmd = action_np[0:7]
        gripper_raw = float(action_np[7])
        self._gripper_state = gripper_raw
        self.interface.send_joint_positions(arm_cmd)

    def get_joint_positions(self) -> torch.Tensor:
        arm_pos = self.interface.read_joint_positions()
        positions = np.concatenate([arm_pos, [self._gripper_state]])
        return torch.from_numpy(positions).float()

    def get_joint_velocities(self) -> torch.Tensor:
        import time as _time
        current_pos = self.interface.read_joint_positions()
        current_time = _time.perf_counter()
        if self._prev_joint_pos is not None and self._prev_time is not None:
            dt = current_time - self._prev_time
            arm_vel = (current_pos - self._prev_joint_pos) / dt if dt > 0 else np.zeros(7)
        else:
            arm_vel = np.zeros(7)
        self._prev_joint_pos = current_pos.copy()
        self._prev_time = current_time
        velocities = np.concatenate([arm_vel, [0.0]])
        return torch.from_numpy(velocities).float()

    def get_joint_efforts(self) -> torch.Tensor:
        return torch.zeros(8)

    @property
    def action_dim(self) -> int:
        return 8

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        assert self.action_dim == action.shape[0], \
            f"Action dimension mismatch: {self.action_dim} != {action.shape[0]}"
        return action

    def get_no_op_action(self) -> torch.Tensor:
        return self.get_joint_positions()

    def reset(self) -> None:
        pass

    def get_home_position(self) -> torch.Tensor:
        """Return Franka home joint positions (7 arm + 1 gripper)."""
        return torch.tensor([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4, -1.0], dtype=torch.float32)

    def move_to_home(self) -> None:
        self._move_to_position(self.get_home_position().numpy())

    def _move_to_position(self, target: np.ndarray, steps: int = 50) -> None:
        import time
        current = self.get_joint_positions().numpy()
        for i in range(steps):
            alpha = (i + 1) / steps
            interpolated = current * (1 - alpha) + target * alpha
            self.apply_action(torch.from_numpy(interpolated).float())
            time.sleep(0.02)

    def close(self) -> None:
        self._viewer_stop.set()

    @property
    def controllers(self) -> dict:
        return {}


class RealFranka(Robot):
    """Real Franka Panda robot interface using deoxys for control."""

    @property
    def model_name(self) -> str:
        return "Franka"

    def __init__(self, name: str = "franka", gripper_open_action: float = -1.0) -> None:
        super().__init__(name=name, model="franka")
        self.resetted = False
        self._prev_joint_pos = None
        self._prev_time = None

        # Cap how far "open" travels. The Hand's finger speed is firmware
        # capped (~0.1 m/s measured regardless of commanded speed), so opening
        # to less than the full 80 mm is the only way to open faster: deoxys
        # maps action -> width 0.08*|action|, e.g. -0.75 opens to 60 mm in
        # ~25% less time. -1.0 keeps full travel.
        self._gripper_open_action = float(gripper_open_action)

        from franka_joylo.deoxys_franka import DeoxysFrankaInterface

        # Stiffer than franka_joylo defaults (kp=100, kd=10) so the arm tracks
        # the commanded gello position with less lag. Kd scaled ~sqrt(kp ratio).
        self.interface = DeoxysFrankaInterface(
            control_freq=100,
            stiff_kp=[200.0] * 7,
            stiff_kd=[14.0] * 7,
        )

        # Wait for initial joint state from deoxys
        import time
        timeout = 5.0
        start_time = time.time()
        while self.interface.robot_interface.state_buffer_size == 0:
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    "Timed out waiting for Franka joint state. "
                    "Is deoxys running on the NUC (auto_arm.sh)?"
                )

        # Default gripper state: open (-1.0)
        self._gripper_state = -1.0

    def apply_action(self, action: torch.Tensor) -> None:
        """
        Send action to Franka via deoxys.

        Action tensor format (8 DOF):
          - arm: indices 0-6 (7 DOF)
          - gripper: index 7 (1 DOF, -1=open, 1=close)
        """
        action_np = action.detach().cpu().numpy()
        arm_cmd = action_np[0:7]
        gripper_raw = float(action_np[7])

        # Negative = open: clamp travel to the configured open width.
        if gripper_raw < 0.0:
            gripper_raw = max(gripper_raw, self._gripper_open_action)

        # Store gripper state for readback
        self._gripper_state = gripper_raw

        # Set gripper action on deoxys interface and send arm command
        self.interface._gripper_action = gripper_raw
        self.interface.send_joint_positions(arm_cmd)

    def get_joint_positions(self) -> torch.Tensor:
        """
        Read joint positions from deoxys.

        Returns tensor: [arm(7), gripper(1)]
        """
        arm_pos = self.interface.read_joint_positions()  # (7,)

        positions = np.concatenate([
            arm_pos,
            [self._gripper_state],
        ])

        return torch.from_numpy(positions).float()

    def get_joint_velocities(self) -> torch.Tensor:
        """
        Estimate joint velocities via numerical differentiation.

        Returns tensor: [arm(7), gripper(1)]
        """
        import time as _time
        current_pos = self.interface.read_joint_positions()
        current_time = _time.perf_counter()

        if self._prev_joint_pos is not None and self._prev_time is not None:
            dt = current_time - self._prev_time
            if dt > 0:
                arm_vel = (current_pos - self._prev_joint_pos) / dt
            else:
                arm_vel = np.zeros(7)
        else:
            arm_vel = np.zeros(7)

        self._prev_joint_pos = current_pos.copy()
        self._prev_time = current_time

        velocities = np.concatenate([arm_vel, [0.0]])  # gripper velocity not tracked
        return torch.from_numpy(velocities).float()

    def get_joint_efforts(self) -> torch.Tensor:
        """Read joint efforts/torques (not exposed by deoxys)."""
        return torch.zeros(8)

    @property
    def action_dim(self) -> int:
        return 8  # 7 arm + 1 gripper

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        assert self.action_dim == action.shape[0], \
            f"Action dimension mismatch: {self.action_dim} != {action.shape[0]}"
        return action

    def get_no_op_action(self) -> torch.Tensor:
        return self.get_joint_positions()

    def reset(self) -> None:
        """No-op for real robot. Use move_to_home() explicitly."""
        pass

    def get_home_position(self) -> torch.Tensor:
        """Return Franka home joint positions (7 arm + 1 gripper)."""
        return torch.tensor([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4, -1.0], dtype=torch.float32)

    def move_to_home(self) -> None:
        """Move Franka to home position with smooth interpolation."""
        home_position = self.get_home_position().numpy()
        self._move_to_position(home_position)

    def _move_to_position(self, target: np.ndarray, steps: int = 50) -> None:
        """Smoothly interpolate to target position."""
        import time
        current = self.get_joint_positions().numpy()
        for i in range(steps):
            alpha = (i + 1) / steps
            interpolated = current * (1 - alpha) + target * alpha
            # The gripper is a discrete open/close command, not a joint
            # position: interpolating it sweeps through dozens of distinct
            # partial-width commands, each executed as a separate blocking
            # gripper call on the NUC. Command the target state directly.
            interpolated[7] = target[7]
            self.apply_action(torch.from_numpy(interpolated).float())
            time.sleep(0.02)  # 50Hz interpolation

    def close(self) -> None:
        """Clean up deoxys interface."""
        try:
            self.interface.close()
        except Exception:
            pass

    @property
    def controllers(self) -> dict:
        """Stub for compatibility - real robots don't use OmniGibson controllers."""
        return {}


class RealR1Pro(Robot):
    """Real R1Pro robot interface using brs-ctrl for ROS2 communication."""

    @property
    def model_name(self) -> str:
        return "R1Pro"

    def __init__(self, name: str = "r1pro") -> None:
        super().__init__(name=name, model="r1pro")
        self.resetted = False
        self._rclpy = None

        # Import brs-ctrl (ROS2)
        import rclpy
        from brs_ctrl.robot_interface import R1ProInterface
        from brs_ctrl.robot_interface.grippers.galaxea_g1 import GalaxeaR1ProGripper

        self._rclpy = rclpy

        # Initialize ROS2 if not already done
        if not rclpy.ok():
            rclpy.init()

        # Create robot interface with grippers
        # Note: Increase step_interval to reduce latency from trajectory interpolation
        # Default 0.4 rad causes multi-step interpolation with 10ms sleep per step
        # Setting to 2.0 rad allows ~115° per step, making most commands single-step (~10ms)
        self.interface = R1ProInterface(
            left_gripper=GalaxeaR1ProGripper(
                left_or_right="left",
                gripper_close_stroke=0.0,
                gripper_open_stroke=100.0,
            ),
            right_gripper=GalaxeaR1ProGripper(
                left_or_right="right",
                gripper_close_stroke=0.0,
                gripper_open_stroke=100.0,
            ),
            arm_joint_control_step_interval=3.0,  # Reduce latency: allow larger steps
            torso_joint_control_step_interval=1.0,  # Reduce latency for torso too
            log_clipping_warnings=False,  # Suppress spam from joint limit clipping
        )

        # Wait for initial joint state
        self._wait_for_joint_state()

    def _wait_for_joint_state(self, timeout: float = 5.0) -> None:
        """Wait until robot joint state is available."""
        import time
        start_time = time.time()
        while self.interface.last_joint_position is None:
            self._rclpy.spin_once(self.interface, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for robot joint state. Is the robot running?")

    def apply_action(self, action: torch.Tensor) -> None:
        """
        Send action to robot via ROS2.

        Action tensor format (R1Pro, 23 DOF):
          - base:          indices 0-2   (3 DOF)
          - torso:         indices 3-6   (4 DOF)
          - left_arm:      indices 7-13  (7 DOF)
          - left_gripper:  index 14      (1 DOF)
          - right_arm:     indices 15-21 (7 DOF)
          - right_gripper: index 22      (1 DOF)
        """
        action_np = action.detach().cpu().numpy()

        # Parse action tensor
        base_cmd = action_np[0:3]
        torso_cmd = action_np[3:7]
        left_arm = action_np[7:14]
        gripper_left_raw = float(action_np[14])
        right_arm = action_np[15:22]
        gripper_right_raw = float(action_np[22])

        # Convert gripper from IIIL range [-1, 1] to brs-ctrl range [0, 1]
        # IIIL: 1=closed, -1=open
        # brs-ctrl: 1=closed, 0=open
        # Formula: brs_ctrl = (iiil + 1) / 2
        # NOTE: Must convert to Python float, not numpy.float64, because brs-ctrl's
        # GalaxeaR1ProGripper.act() uses isinstance(action, (float, int)) assertion
        gripper_left = float(np.clip((gripper_left_raw + 1.0) / 2.0, 0.0, 1.0))
        gripper_right = float(np.clip((gripper_right_raw + 1.0) / 2.0, 0.0, 1.0))

        # Send to robot via brs-ctrl
        self.interface.control(
            arm_cmd={"left": left_arm, "right": right_arm},
            gripper_cmd={"left": gripper_left, "right": gripper_right},
            torso_cmd=torso_cmd,
            base_cmd=base_cmd,
        )
        # Quick non-blocking spin to flush outgoing messages (reduces command latency)
        self._rclpy.spin_once(self.interface, timeout_sec=0.01)

    def get_joint_positions(self) -> torch.Tensor:
        """
        Read joint positions from ROS2.

        Returns tensor in same order as action:
          [base(3), torso(4), left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]

        Note: ROS2 spin handled in env._spin_ros_once() to avoid redundant calls.
        """
        pos = self.interface.last_joint_position
        gripper = self.interface.last_gripper_state
        chassis_pos = self.interface.last_chassis_position

        if pos is None:
            return torch.zeros(23)

        # Get chassis position from /hdas/feedback_chassis (3 values)
        if chassis_pos is not None:
            base_pos = chassis_pos.flatten()
        else:
            base_pos = np.zeros(3)

        # Extract gripper positions and convert from stroke (0-100) to IIIL range (-1, 1)
        # stroke = close + (1 - brs_action) * (open - close)
        # With close=0, open=100: brs_action = 1 - stroke/100
        # Then convert brs_action [0,1] to iiil_action [-1,1]: iiil = 2*brs - 1
        # Combined: iiil = 2*(1 - stroke/100) - 1 = 1 - stroke/50
        if gripper is not None:
            left_grip_state = gripper.get("left_gripper")
            right_grip_state = gripper.get("right_gripper")

            if left_grip_state is not None and "gripper_position" in left_grip_state:
                left_stroke = np.atleast_1d(left_grip_state["gripper_position"])[-1]
                # Convert stroke (0-100) → IIIL (-1, 1): stroke=0 → 1 (closed), stroke=100 → -1 (open)
                left_grip_iiil = np.clip(1.0 - left_stroke / 50.0, -1.0, 1.0)
            else:
                left_grip_iiil = -1.0  # Default to open

            if right_grip_state is not None and "gripper_position" in right_grip_state:
                right_stroke = np.atleast_1d(right_grip_state["gripper_position"])[-1]
                right_grip_iiil = np.clip(1.0 - right_stroke / 50.0, -1.0, 1.0)
            else:
                right_grip_iiil = -1.0  # Default to open
        else:
            left_grip_iiil = -1.0  # Default to open
            right_grip_iiil = -1.0

        # Build position vector in action format
        positions = np.concatenate([
            base_pos,                           # base: 3
            pos["torso"].flatten(),             # torso: 4
            pos["left_arm"].flatten(),          # left_arm: 7
            [left_grip_iiil],                   # left_gripper: 1
            pos["right_arm"].flatten(),         # right_arm: 7
            [right_grip_iiil],                  # right_gripper: 1
        ])

        return torch.from_numpy(positions).float()

    def get_joint_velocities(self) -> torch.Tensor:
        """
        Read joint velocities from ROS2.

        Returns tensor in same order as action:
          [base(3), torso(4), left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]

        Note: ROS2 spin handled in env._spin_ros_once() to avoid redundant calls.
        """
        vel = self.interface.last_joint_velocity
        chassis_vel = self.interface.last_chassis_velocity

        if vel is None:
            return torch.zeros(23)

        # Get chassis velocity from /hdas/feedback_chassis (first 3 of 6 values)
        if chassis_vel is not None:
            base_vel = chassis_vel.flatten()
        else:
            base_vel = np.zeros(3)

        # Build velocity vector in action format
        # Gripper velocities are not tracked, use zeros
        velocities = np.concatenate([
            base_vel,                           # base: 3
            vel["torso"].flatten(),             # torso: 4
            vel["left_arm"].flatten(),          # left_arm: 7
            [0.0],                              # left_gripper: 1 (not tracked)
            vel["right_arm"].flatten(),         # right_arm: 7
            [0.0],                              # right_gripper: 1 (not tracked)
        ])

        return torch.from_numpy(velocities).float()

    def get_joint_efforts(self) -> torch.Tensor:
        """Read joint efforts/torques from ROS2."""
        # Not exposed by brs-ctrl interface
        return torch.zeros(23)

    @property
    def action_dim(self) -> int:
        """Total action dimension for R1Pro."""
        return 23

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Process action before applying to robot.

        Args:
            action: Action tensor from policy

        Returns:
            Processed action tensor
        """
        assert self.action_dim == action.shape[0], f"Action dimension mismatch: {self.action_dim} != {action.shape[0]}"
        return action

    def get_no_op_action(self) -> torch.Tensor:
        """Return a no-op action that holds current position."""
        return self.get_joint_positions()

    def reset(self) -> None:
        """
        Override OmniGibson's reset().

        By default this is a no-op for the real robot. Some rollout modes opt in
        to physically homing the robot during env.reset().
        """
        if getattr(self, "_reset_to_home", False):
            self.move_to_home()

    def set_reset_to_home(self, enabled: bool) -> None:
        """Configure whether reset() should physically move to home."""
        self._reset_to_home = enabled

    def move_to_home(self) -> None:
        """
        Move robot to home position with smooth interpolation.
        Called when Home button is pressed.
        """
        # Get home position (neutral - all joints at 0)
        home_position = self._get_home_position()

        # Move robot smoothly to home position
        self._move_to_position(home_position)

    def _get_home_position(self) -> np.ndarray:
        """Get the home/default joint position for R1Pro."""
        # Neutral position: all joints at 0
        # Format: [base(3), torso(4), left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
        return np.zeros(23)

        # The following is for squat + arm forward
        # return torch.cat([
        #     torch.zeros(3, dtype=torch.float32),    # base
        #     torch.tensor([-0.63, 1.65, 1.05, 0.00]),  # torso
        #     torch.tensor([0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0, 0.0]),   # arm + gripper
        #     torch.tensor([0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0, 0.0]),   # arm + gripper
        # ])

    def _move_to_position(self, target: np.ndarray, steps: int = 50) -> None:
        """Smoothly interpolate to target position."""
        import time

        current = self.get_joint_positions().numpy()

        for i in range(steps):
            alpha = (i + 1) / steps
            interpolated = current * (1 - alpha) + target * alpha
            self.apply_action(torch.from_numpy(interpolated).float())
            time.sleep(0.02)  # 50Hz interpolation

    def tuck(self) -> None:
        """Move robot to tucked/safe position."""
        # TODO: Implement tuck behavior
        pass

    def close(self) -> None:
        """Clean up robot interface."""
        try:
            self.interface.close()
        except Exception:
            pass

    @property
    def controllers(self) -> dict:
        """
        Stub controllers property for compatibility with teleop strategy.

        Real robots don't use OmniGibson's controller system.
        Returns empty dict so set_controller_mode becomes a no-op.
        """
        return {}

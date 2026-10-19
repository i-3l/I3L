import h5py
import json
import numpy as np
import omnigibson as og
import os
import torch
from omnigibson.envs.hdf5_data_wrapper import HDF5DataWrapper
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.utils.ui_utils import create_module_logger
from pathlib import Path
from time import perf_counter
from typing import Dict

logger = create_module_logger("Env Wrapper")


class SimDataCollectionWrapper(HDF5DataWrapper):
    # Flag for trial-based recording (X to start, A/Y to end)
    supports_trial_recording: bool = True

    def __init__(self, *args, manual_save_discard=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._episode_start_time = None
        self._manual_save_discard = manual_save_discard

    def add_metadata(self, group, name, data):
        """
        OmniGibson metadata occasionally contains objects that h5py can't serialize
        directly as attributes. Normalize nested values to plain JSON-friendly types
        before persisting them.
        """

        def _sanitize(value):
            if isinstance(value, dict):
                return {str(k): _sanitize(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [_sanitize(v) for v in value]
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if torch.is_tensor(value):
                if value.numel() == 1:
                    return value.item()
                return value.detach().cpu().tolist()
            if isinstance(value, (str, int, float, bool, bytes)) or value is None:
                return value
            return str(value)

        sanitized = _sanitize(data)
        if isinstance(sanitized, (dict, list)):
            group.attrs[name] = json.dumps(sanitized)
        elif sanitized is None:
            group.attrs[name] = "null"
        else:
            group.attrs[name] = sanitized

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        # Store action, obs, reward, terminated, truncated, info
        step_data = dict()
        step_data["obs"] = self._process_obs(obs=obs, info=info)
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["terminated"] = terminated
        step_data["truncated"] = truncated
        # Timestamp in ms since episode start (stored under time/ group)
        if self._episode_start_time is not None:
            step_data["time"] = {
                "timestamp_ms": torch.tensor([(perf_counter() - self._episode_start_time) * 1000.0])
            }
        policy_state = self.env.task.policy.get_current_state()
        if policy_state is not None:
            step_data["policy"] = policy_state
        return step_data

    def flush_current_traj(self):
        """
        IIIL Change: add policy as nested key in obs
        """
        num_steps = len(self.current_traj_history) - 1
        if num_steps <= 0:
            self.current_traj_history = []
            return

        if self._manual_save_discard:
            success_str = "\033[92mSUCCESS\033[0m" if self.env.task.success else "\033[91mFAILED\033[0m"
            print(f"\n\033[1m  Episode ended: {success_str} ({num_steps} steps)\033[0m", flush=True)
            print(f"\033[1m  Press [s] to save, [d] to discard:\033[0m ", end="", flush=True)
            import sys, tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while True:
                    ch = sys.stdin.read(1)
                    if ch in ('s', 'S'):
                        print("s\n", flush=True)
                        should_save = True
                        break
                    elif ch in ('d', 'D'):
                        print("d\n", flush=True)
                        should_save = False
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        else:
            should_save = self.should_save_current_episode

        if should_save:
            traj_grp_name = f"demo_{self.traj_count}"
            traj_grp = self._process_traj_to_hdf5(self.current_traj_history, traj_grp_name, nested_keys=["obs", "policy", "time"])
            print(f"\033[1m\033[94m  📁 Saved as: {traj_grp_name} ({num_steps} steps)\033[0m", flush=True)
            self.traj_count += 1
            self._postprocess_traj_group(traj_grp)

            if isinstance(self.env, (RealEnvironment, RealFrankaEnvironment)):
                self.flush_current_file()
            elif self.traj_count % self.flush_every_n_traj == 0:
                self.flush_current_file()
        else:
            print(f"\033[1m\033[93m  🗑️  Discarded ({num_steps} steps)\033[0m", flush=True)
            self.step_count -= len(self.current_traj_history)

        # Clear trajectory and transition buffers
        self.current_traj_history = []

    def reset(self):
        """
        IIIL Change: add initial time to traj history
        """
        obs, info = super().reset()
        # Start timestamp at 0 for episode
        self._episode_start_time = perf_counter()
        self.current_traj_history[0]["time"] = {"timestamp_ms": torch.tensor([0.0])}
        return obs, info
    

class RealEnvironment(og.Environment):
    def __init__(self, config: Dict):
        self._task = None
        self._robots = None
        self.frequency = config["env"]["frequency"]
        self.step_time = 1 / self.frequency
        self._timer = perf_counter()
        # dummy values 
        self._scene_graph_builder = None
        self._automatic_reset = False
        # Variables reset at the beginning of each episode
        self._current_step = 0
        self._current_episode = 0
        
        # ROS2 specific imports
        import rclpy
        self.rclpy = rclpy
        # Setup ROS2 camera subscriptions
        self._setup_cameras(config)

    def _setup_cameras(self, config: Dict):
        """Setup ROS2 camera subscriptions."""
        # Initialize ROS2 if not already done
        if not self.rclpy.ok():
            self.rclpy.init()

        # Create a ROS2 node for camera subscriptions
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        self._ros_node = self.rclpy.create_node('real_env_cameras')
        self._bridge = CvBridge()

        # Camera resolution from config (default 240x240)
        self._camera_resolution = tuple(config.get("camera_resolution", [240, 240]))

        # Storage for latest frames
        self._latest_frames = {
            'robot:left_realsense_link:Camera:0': None,
            'robot:right_realsense_link:Camera:0': None,
            'robot:zed_link:Camera:0': None,
        }
        if config.get("head_stereo", False):
            self._latest_frames['robot:zed_link:Camera:1'] = None
        
        # Camera topics from config with defaults
        camera_config = config.get("cameras", {})
        self._camera_topics = {
            'robot:left_realsense_link:Camera:0': camera_config.get('left_topic', '/hdas/camera_wrist_left/color/image_raw'),
            'robot:right_realsense_link:Camera:0': camera_config.get('right_topic', '/hdas/camera_wrist_right/color/image_raw'),
            'robot:zed_link:Camera:0': camera_config.get('zed_topic', '/hdas/camera_head/left_raw/image_raw_color/compressed'),
        }
        if config.get("head_stereo", False):
            self._camera_topics['robot:zed_link:Camera:1'] = camera_config.get('zed_right_topic', '/hdas/camera_head/right_raw/image_raw_color/compressed')
        
        # Create subscriptions with sensor-appropriate QoS
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage

        self._camera_subs = []
        for name, topic in self._camera_topics.items():
            # Use CompressedImage for topics ending with /compressed
            if topic.endswith('/compressed'):
                callback = self._make_compressed_camera_callback(name)
                msg_type = CompressedImage
            else:
                callback = self._make_camera_callback(name)
                msg_type = Image
            sub = self._ros_node.create_subscription(
                msg_type,
                topic,
                callback,
                qos_profile_sensor_data
            )
            self._camera_subs.append(sub)
            logger.info(f"Subscribed to camera '{name}' on topic: {topic} (type: {msg_type.__name__})")

        # Debug: print subscription info
        print(f"[DEBUG] Created {len(self._camera_subs)} camera subscriptions on node '{self._ros_node.get_name()}'")
        for sub in self._camera_subs:
            print(f"[DEBUG]   - {sub.topic_name}")

        # Warmup: spin for a bit to allow DDS discovery and receive first frames
        import time
        print("[DEBUG] Warming up camera subscriptions (1 second)...")
        start = time.time()
        while time.time() - start < 1.0:
            self.rclpy.spin_once(self._ros_node, timeout_sec=0.05)
        # Report which cameras have frames
        for name, frame in self._latest_frames.items():
            status = "OK" if frame is not None else "NO FRAME"
            print(f"[DEBUG] Camera '{name}': {status}")
    
    def _make_camera_callback(self, camera_name: str):
        """Create a callback for a specific camera (raw Image messages)."""

        def callback(msg) -> None:
            try:
                # Try RGB first, fall back to BGR if needed
                try:
                    frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                except Exception:
                    frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    # Convert BGR to RGB
                    import cv2
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._latest_frames[camera_name] = frame  # Copy to avoid cv_bridge buffer reuse
            except Exception as e:
                logger.warning(f"Failed to convert image from '{camera_name}': {e}")
        return callback

    def _make_compressed_camera_callback(self, camera_name: str):
        """Create a callback for compressed image topics."""
        import cv2
        import numpy as np

        def callback(msg) -> None:
            try:
                # Decode compressed image
                np_arr = np.frombuffer(msg.data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._latest_frames[camera_name] = frame
                else:
                    logger.warning(f"Failed to decode compressed image from '{camera_name}'")
            except Exception as e:
                logger.warning(f"Failed to convert compressed image from '{camera_name}': {e}")
        return callback
    
    def _spin_ros_once(self):
        """Process ROS2 callbacks to update camera frames and robot state."""
        # Spin camera node multiple times to process all pending camera messages
        # spin_once only processes ONE callback, so we need multiple calls for multiple cameras
        if hasattr(self, '_ros_node'):
            for _ in range(len(self._latest_frames) + 2):  # +2 for safety margin
                self.rclpy.spin_once(self._ros_node, timeout_sec=0.001)
        # Spin robot interface node multiple times to process all pending state callbacks
        # R1ProInterface has 4+ callbacks: torso, left_arm, right_arm, chassis, (cameras if enabled)
        # At 100Hz feedback, multiple callbacks queue up - must process ALL for fresh data
        if self._robots and hasattr(self._robots[0], 'interface'):
            for _ in range(8):  # Process up to 8 pending callbacks (covers all state topics)
                self.rclpy.spin_once(self._robots[0].interface, timeout_sec=0.001)

    def render(self):
        pass

    def step(self, action, **kwargs):
        """
        Execute one environment step on the real robot.

        Args:
            action: Action tensor to send to the robot

        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
        """
        _profile = getattr(self, '_profile_steps', False)
        if _profile:
            _t = {}
            _t['start'] = perf_counter()

        # Send action to robot
        if self._robots and hasattr(self._robots[0], 'apply_action'):
            if _profile:
                _t['apply_start'] = perf_counter()
            self._robots[0].apply_action(action)
            if _profile:
                _t['apply_end'] = perf_counter()

        # Timing control - track loop time but don't block
        # (Sleep removed to reduce latency - frequency control handled externally if needed)
        loop_time = perf_counter() - self._timer
        if _profile:
            _t['before_sleep'] = perf_counter()
            _t['after_sleep'] = perf_counter()  # No sleep, just record timestamp
        self._timer = perf_counter()

        # Run final post-processing
        if _profile:
            _t['post_step_start'] = perf_counter()
        result = self._post_step(action)
        if _profile:
            _t['post_step_end'] = perf_counter()
            # Print env.step breakdown
            apply_t = (_t.get('apply_end', _t['start']) - _t.get('apply_start', _t['start'])) * 1000
            sleep_t = (_t['after_sleep'] - _t['before_sleep']) * 1000
            post_t = (_t['post_step_end'] - _t['post_step_start']) * 1000
            total = (_t['post_step_end'] - _t['start']) * 1000
            logger.info(f"  [ENV.STEP] total={total:.1f}ms | apply_action={apply_t:.1f}ms | sleep={sleep_t:.1f}ms | post_step={post_t:.1f}ms")
        return result
    
    @property
    def task(self):
        return self._task
    
    @task.setter
    def task(self, task):
        self._task = task

    @property
    def robots(self):
        return self._robots
    
    @robots.setter
    def robots(self, robots):
        self._robots = robots

    def _reset_variables(self):
        """Reset internal episode variables."""
        self._current_step = 0
        self._current_episode += 1
        self._timer = perf_counter()

    def reset(self, get_obs=True, **kwargs):
        # Reset the task
        if self._task is not None:
            self.task.reset(self)
        # Reset internal variables
        self._reset_variables()

        if get_obs:
            return self.get_obs()

    def get_obs(self):
        """
        Get the current environment observation.

        Returns:
            2-tuple:
                dict: Keyword-mapped observations, which are possibly nested
                dict: Additional information about the observations
        """
        _profile = getattr(self, '_profile_steps', False)
        if _profile:
            _t = {'start': perf_counter()}

        # Spin ROS2 to get latest camera frames
        if _profile:
            _t['spin_start'] = perf_counter()
        self._spin_ros_once()
        if _profile:
            _t['spin_end'] = perf_counter()

        obs = dict()
        info = dict()

        # Add task observations
        if self._task is not None:
            obs["task"] = self._task.get_obs(env=self)

        # Add robot proprioception (joint positions and velocities)
        if self._robots:
            robot = self._robots[0]
            if _profile:
                _t['joint_pos_start'] = perf_counter()
            joint_pos = robot.get_joint_positions()
            if _profile:
                _t['joint_pos_end'] = perf_counter()
                _t['joint_vel_start'] = perf_counter()
            joint_vel = robot.get_joint_velocities()
            if _profile:
                _t['joint_vel_end'] = perf_counter()
            # Match the format expected by teleop pipeline: "robot::proprio"
            obs["robot::proprio"] = torch.cat([joint_pos, joint_vel])
            if _profile:
                spin_t = (_t['spin_end'] - _t['spin_start']) * 1000
                pos_t = (_t['joint_pos_end'] - _t['joint_pos_start']) * 1000
                vel_t = (_t['joint_vel_end'] - _t['joint_vel_start']) * 1000
                logger.info(f"    [GET_OBS] spin_ros={spin_t:.1f}ms | get_joint_pos={pos_t:.1f}ms | get_joint_vel={vel_t:.1f}ms")

        # Add camera observations
        for camera_name, frame in self._latest_frames.items():
            if frame is not None:
                # convert to torch uint8 tensor (copy to avoid shared memory with ROS callback)
                frame_tensor = torch.from_numpy(frame)  # (H, W, C)
                obs[f"robot::{camera_name}::rgb"] = frame_tensor
            else:
                logger.warning(f"No frame received yet for camera: {camera_name}")

        return obs, info

    def close(self):
        """Clean up resources."""
        if self._robots:
            for robot in self._robots:
                if hasattr(robot, 'tuck'):
                    robot.tuck()
        
        # Cleanup ROS2 resources
        if hasattr(self, '_ros_node'):
            self._ros_node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


class RealFrankaEnvironment:
    """Real Franka environment - no ROS2 dependency.

    Uses deoxys for robot control. ZED cameras captured via OpenCV in
    background threads (see iiil/utils/camera_capture.py).
    """

    def __init__(self, config):
        self._task = None
        self._robots = None
        self.frequency = config["env"]["frequency"]
        self.step_time = 1 / self.frequency
        self._timer = perf_counter()
        # Dummy values for compatibility with OmniGibson env interface
        self._scene_graph_builder = None
        self._automatic_reset = False
        self._current_step = 0
        self._current_episode = 0

        # Camera setup (optional — gracefully handles missing cameras)
        self._cameras = {}
        camera_config = config.get("cameras", {})
        if camera_config:
            from iiil.utils.camera_capture import ZedCameraCapture
            for cam_key, cam_cfg in camera_config.items():
                name = cam_cfg["name"]
                device = cam_cfg["device"]
                resolution = tuple(cam_cfg.get("resolution", [240, 240]))
                capture_resolution = cam_cfg.get("capture_resolution", None)
                if capture_resolution is not None:
                    capture_resolution = tuple(capture_resolution)
                capture_fps = cam_cfg.get("capture_fps", None)
                crop_size = cam_cfg.get("crop_size", None)
                crop_offset = tuple(cam_cfg.get("crop_offset", [0, 0]))
                v4l2_controls = cam_cfg.get("v4l2", None)
                if v4l2_controls is not None:
                    v4l2_controls = dict(v4l2_controls)
                try:
                    capture = ZedCameraCapture(
                        device_path=device,
                        name=name,
                        resolution=resolution,
                        capture_resolution=capture_resolution,
                        capture_fps=capture_fps,
                        crop_size=crop_size,
                        crop_offset=crop_offset,
                        v4l2_controls=v4l2_controls,
                    )
                    self._cameras[name] = capture
                    logger.info(f"Camera '{cam_key}' ({name}) opened on {device}")
                except RuntimeError as e:
                    logger.warning(f"Skipping camera '{cam_key}': {e}")

    def render(self):
        pass

    def step(self, action, **kwargs):
        _profile = getattr(self, '_profile_steps', False)
        if _profile:
            _t = {'start': perf_counter()}

        # Send action to robot
        if self._robots and hasattr(self._robots[0], 'apply_action'):
            self._robots[0].apply_action(action)

        loop_time = perf_counter() - self._timer
        self._timer = perf_counter()

        result = self._post_step(action)

        if _profile:
            total = (perf_counter() - _t['start']) * 1000
            logger.info(f"  [ENV.STEP] total={total:.1f}ms")
        return result

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, task):
        self._task = task

    @property
    def robots(self):
        return self._robots

    @robots.setter
    def robots(self, robots):
        self._robots = robots

    def _reset_variables(self):
        self._current_step = 0
        self._current_episode += 1
        self._timer = perf_counter()

    def reset(self, get_obs=True, **kwargs):
        if self._task is not None:
            self.task.reset(self)
        self._reset_variables()
        if get_obs:
            return self.get_obs()

    def get_obs(self):
        obs = dict()
        info = dict()

        # Add task observations
        if self._task is not None:
            obs["task"] = self._task.get_obs(env=self)

        # Add robot proprioception
        if self._robots:
            robot = self._robots[0]
            joint_pos = robot.get_joint_positions()
            joint_vel = robot.get_joint_velocities()
            obs["robot::proprio"] = torch.cat([joint_pos, joint_vel])

        # Add camera observations
        for camera_name, capture in self._cameras.items():
            frame = capture.get_frame()
            if frame is not None:
                obs[f"robot::{camera_name}::rgb"] = torch.from_numpy(frame)
            else:
                logger.warning(f"No frame available from camera: {camera_name}")

        return obs, info

    @property
    def episode_steps(self):
        """Compatibility with OmniGibson termination conditions."""
        return self._current_step

    @property
    def _n_successful_episodes(self):
        return 0

    @_n_successful_episodes.setter
    def _n_successful_episodes(self, value):
        pass

    def _post_step(self, action):
        self._current_step += 1
        obs, info = self.get_obs()

        # Evaluate termination conditions from task (same as og.Environment)
        # Use tensors (not raw bools) so HDF5 wrapper can th.stack() them
        terminated = torch.tensor(False)
        truncated = torch.tensor(False)
        reward = 0.0
        if self._task is not None and hasattr(self._task, '_termination_conditions'):
            done, done_info = self._task._step_termination(self, action)
            terminated = torch.tensor(bool(done))
            info["done"] = done_info
            # Update task success state (og.Environment does this in task.step(),
            # but we call _step_termination directly)
            self._task._success = done_info.get("success", False)
            self._task._done = done
            # BlockEnd (Plus button) should end the entire program, not just reset
            if done and done_info.get("termination_conditions", {}).get("timeout", {}).get("done", False):
                truncated = torch.tensor(True)

        return obs, reward, terminated, truncated, info

    def close(self):
        # Close cameras
        for capture in self._cameras.values():
            capture.close()
        # Close robots
        if self._robots:
            for robot in self._robots:
                if hasattr(robot, 'close'):
                    robot.close()


class RealDataCollectionWrapper(SimDataCollectionWrapper):
    def __init__(
        self, env, output_path, compression=dict(), overwrite=True, only_successes=True, flush_every_n_traj=2
    ):
        """
        Args:
            env (Environment): The environment to wrap
            output_path (str): path to store hdf5 data file
            only_successes (bool): Whether to only save successful episodes
        """
        assert isinstance(
            env, (RealEnvironment, RealFrankaEnvironment)
        ), "Expected wrapped @env to be a RealEnvironment or RealFrankaEnvironment!"
        self.traj_count = 0
        self.step_count = 0
        self.only_successes = only_successes
        self.flush_every_n_traj = flush_every_n_traj
        self.current_obs = None
        self.compression = compression
        self._episode_start_time = None
        self._manual_save_discard = False
        self._trial_step_count = 0  # Step counter within current trial

        # Get target frequency from env for consistent timestamp computation
        self._target_frequency = getattr(env, 'frequency', 30.0)
        self._target_period_ms = 1000.0 / self._target_frequency

        self.current_traj_history = []

        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        logger.info(f"\nWriting Real robot dataset hdf5 to: {output_path}\n")
        logger.info(f"Target frequency: {self._target_frequency} Hz ({self._target_period_ms:.1f}ms per step)\n")
        self.hdf5_file = h5py.File(output_path, "w" if overwrite else "a")
        if "data" not in set(self.hdf5_file.keys()):
            data_grp = self.hdf5_file.create_group("data")
        else:
            data_grp = self.hdf5_file["data"]
        self.add_metadata(group=data_grp, name="env_args", data={"env_name": "OmniGibson", "type": 4})
        # Store target frequency as metadata for training/replay
        data_grp.attrs["target_frequency_hz"] = self._target_frequency

        # call EnvironmentWrapper constructor
        EnvironmentWrapper.__init__(self, env)

        # Print initial message (actual button guidance printed after policy loads)

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        """
        Parse step data with ideal timestamps based on target frequency.

        Records both under time/ group:
        - time/timestamp_ms: Ideal timestamp based on step count * target period
        - time/actual_timestamp_ms: Actual wall-clock time since trial start

        This ensures consistent temporal spacing for training regardless of
        actual processing time variations.
        """
        step_data = dict()
        step_data["obs"] = self._process_obs(obs=obs, info=info)
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["terminated"] = terminated
        step_data["truncated"] = truncated

        # Compute timestamps (stored under time/ group)
        if self._episode_start_time is not None:
            # Ideal timestamp: evenly spaced based on target frequency
            self._trial_step_count += 1
            step_data["time"] = {
                "timestamp_ms": torch.tensor([self._trial_step_count * self._target_period_ms]),
                # Actual timestamp: wall-clock time for debugging/analysis
                "actual_timestamp_ms": torch.tensor([(perf_counter() - self._episode_start_time) * 1000.0])
            }

        # Add policy state if available
        policy_state = self.env.task.policy.get_current_state()
        if policy_state is not None:
            step_data["policy"] = policy_state

        return step_data

    def start_trial_recording(self, current_obs: dict, info: dict = None) -> None:
        """
        Start recording a new trial with the current observation as initial obs.

        Called when X button is pressed to start trial. This clears any stale
        data from reset and begins fresh with the current robot state.

        Args:
            current_obs: Current observation when trial starts
            info: Optional info dict
        """
        # Clear any stale trajectory data
        self.current_traj_history = []
        # Reset step counter for ideal timestamp computation
        self._trial_step_count = 0
        # Start timestamp at 0 for trial
        self._episode_start_time = perf_counter()
        # Add current observation as the initial obs for this trial
        info = info if info is not None else {}
        self.current_traj_history.append({
            "obs": self._process_obs(current_obs.copy(), info),
            "time": {
                "timestamp_ms": torch.tensor([0.0]),  # Ideal timestamp (step 0)
                "actual_timestamp_ms": torch.tensor([0.0]),  # Actual wall-clock time
            }
        })

    def step(self, action, n_render_iterations=1):
        """
        Run the environment step() function with conditional data recording.

        Data is only recorded when trial is active (X button pressed to start).

        Args:
            action: action to take in environment
            n_render_iterations: Number of rendering iterations

        Returns:
            5-tuple: (obs, reward, terminated, truncated, info)
        """
        # Check if policy supports trial control and if trial is active
        should_record = True
        if hasattr(self.env, 'task') and hasattr(self.env.task, 'policy'):
            policy = self.env.task.policy
            if policy.supports_trial_control:
                should_record = policy.is_trial_active()

        # Make sure actions are always flattened tensors
        if isinstance(action, dict):
            action = torch.cat([act for act in action.values()])

        next_obs, reward, terminated, truncated, info = self.env.step(action, n_render_iterations=n_render_iterations)

        # Only record if trial is active
        if should_record:
            self.step_count += 1
            self._record_step_trajectory(action, next_obs, reward, terminated, truncated, info)

        return next_obs, reward, terminated, truncated, info

    def _record_step_trajectory(self, action, obs, reward, terminated, truncated, info):
        """
        Record the current step data to the trajectory history.
        """
        step_data = self._parse_step_data(action, obs, reward, terminated, truncated, info)
        self.current_traj_history.append(step_data)

    def _process_obs(self, obs, info):
        """
        Modifies @obs inplace for any relevant post-processing

        Args:
            obs (dict): Keyword-mapped relevant observations from the immediate env step
            info (dict): Keyword-mapped relevant information from the immediate env step
        """
        # Remove task from obs
        obs.pop("task", None)
        # Clone all tensors to ensure complete memory isolation
        # This prevents any shared memory issues from cv_bridge or numpy views
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                obs[key] = value.clone()
        return obs


class RealDataPlaybackWrapper(HDF5DataWrapper):
    def __init__(self, env, input_path, only_successes=True):
        """
        Args:
            env (Environment): The environment to wrap
            input_path (str): path to stored hdf5 data file
            only_successes (bool): Whether to only save successful episodes
        """
        assert isinstance(
            env, RealEnvironment
        ), "Expected wrapped @env to be a subclass of RealEnvironment class!"
        self.traj_count = 0
        self.step_count = 0
        self.only_successes = only_successes
        self.current_obs = None

        self.current_traj_history = []

        Path(os.path.dirname(input_path)).mkdir(parents=True, exist_ok=True)
        logger.info(f"\nWriting Real robot dataset hdf5 to: {input_path}\n")
        self.hdf5_file = h5py.File(input_path, "r")

        # call EnvironmentWrapper constructor
        EnvironmentWrapper.__init__(self, env)

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        # Store dumped state, reward, terminated, truncated
        step_data = dict()
        step_data["actions"] = action
        step_data["obs"] = obs["task"]
        step_data["reward"] = reward
        step_data["terminated"] = terminated
        step_data["truncated"] = truncated

        # Add ground truth intent
        step_data["intent"] = self.env.task.get_current_intent()

        return step_data
    

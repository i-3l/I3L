import cv2
import iiil.tasks   # This is needed to register tasks
import iiil.wrappers.robots  # This is needed to register robots
import json
import logging
import numpy as np
import omnigibson as og
import os
import sys
import torch
import traceback
from av.container import Container
from av.stream import Stream
from iiil.policies import policy_lib
from iiil.policies.policy_base import BasePolicy
from iiil.utils.camera_stream_viewer import CameraStreamViewer
from iiil.utils.dashboard import Dashboard, DashboardState
from iiil.utils.eval_io import close_video_writer, write_results_json
from iiil.utils.misc_utils import BASE_PATHS
from iiil.utils.omni_paths import prepare_omni_registry_paths
from iiil.wrappers.envs import SimDataCollectionWrapper, RealEnvironment, RealFrankaEnvironment, RealDataCollectionWrapper
from iiil.wrappers.robots import RealR1Pro, RealFranka, SimFranka, SimA1, SimR1Pro
from omnigibson.transition_rules import (
    ToggleableMachineRule,
    MixingToolRule,
    CookingSystemRule,
)
from omegaconf import DictConfig, OmegaConf
from omnigibson.learning.utils.obs_utils import write_video
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES, flatten_obs_dict, PROPRIOCEPTION_INDICES
from omnigibson.macros import gm
from omnigibson.robots.robot import Robot
from omnigibson.tasks import REGISTERED_TASKS
from omnigibson.utils.ui_utils import create_module_logger
from pathlib import Path
from signal import signal, SIGINT
from typing import Union, Tuple


# set global variables to boost performance
def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


gm.ENABLE_FLATCACHE = _env_flag("IIIL_ENABLE_FLATCACHE", True)
gm.USE_GPU_DYNAMICS = _env_flag("IIIL_USE_GPU_DYNAMICS", False)
gm.ENABLE_TRANSITION_RULES = _env_flag("IIIL_ENABLE_TRANSITION_RULES", True)
for rule in [ToggleableMachineRule, MixingToolRule, CookingSystemRule]:
    rule.ENABLED = False

# create module logger
logger = create_module_logger("IIILTask")
logger.setLevel(logging.INFO)
# suppress trimesh warnings (only got xx/xx samples!)
logging.getLogger("trimesh").setLevel(logging.ERROR)


class IIILTask:
    def __init__(self, cfg: DictConfig) -> None:
        self.config = cfg
        # Enable step profiling via config or env var
        self._profile_steps = getattr(cfg, 'profile_steps', False) or os.environ.get('IIIL_PROFILE_STEPS', '').lower() in ('1', 'true')
        if self._profile_steps:
            logger.info("[PROFILING] Step profiling enabled - will log timing every 50 steps")

        # Enable loop timer profiling (writes CSV on shutdown)
        self._profile_csv = getattr(cfg, 'profile_csv', False) or os.environ.get('IIIL_PROFILE_CSV', '').lower() in ('1', 'true')
        self._step_timer = None
        if self._profile_csv:
            from iiil.utils.loop_timer import LoopTimer
            self._step_timer = LoopTimer("main_step", print_interval_s=5.0)
            logger.info("[PROFILING] CSV step profiling enabled")

        # Verbose logging via config or env var
        self._verbose = getattr(cfg, 'verbose', False) or os.environ.get('IIIL_VERBOSE', '').lower() in ('1', 'true')

        # Rate limiting configuration
        # enforce_frequency: if True, sleep to maintain target Hz; if False, run as fast as possible
        self._enforce_frequency = getattr(cfg, 'enforce_frequency', False)
        self._step_start_time = None
        # get base path
        self.base_path = None
        for p in BASE_PATHS:
            if Path(p).exists():
                self.base_path = p
                break
        assert self.base_path is not None, "No valid base path found"

        self.total_time = 0
        self.robot_action = dict()
        # load env, where env type, identified from yaml name, must be one of "sim" or "real"
        self.env = self.load_env()
        # Propagate profiling flag to env (for env.step timing)
        if hasattr(self.env, 'env'):
            # Wrapped env (data collection wrapper)
            self.env.env._profile_steps = self._profile_steps
        else:
            self.env._profile_steps = self._profile_steps
        # load robot and policy
        self.robot = self.load_robot()
        if hasattr(self.robot, "set_reset_to_home"):
            self.robot.set_reset_to_home(bool(getattr(cfg, "reset_robot_on_env_reset", False)))
        self.policy = self.load_policy()
        # Propagate profiling flag to policy
        self.policy._profile_predict = self._profile_steps
        # We additionally pass in robot and policy to task
        self.env.task.robot = self.robot
        self.env.task.policy = self.policy
        # Initialize trial counters before the first step() / dashboard update
        self.env.task.n_trials = 0
        self.env.task.n_success_trials = 0
        self.obs = self._preprocess_obs(self.env.reset()[0])
        self.policy.on_env_reset(self.obs)
        # manually reset environment episode number
        self.env._current_episode = 0
        self.env._n_successful_episodes = 0
        self._video_writer = None

        # --- Dashboard ---
        dashboard_enabled = getattr(cfg, 'dashboard', True)
        robot_type = str(getattr(self.config, 'robot', {}).get('type', ''))
        self._robot_type = robot_type
        is_gello = str(getattr(self.config, 'policy', {}).get('type', '')) == 'gello'
        extra_rows = 0
        if is_gello:
            if 'R1Pro' in robot_type or 'r1pro' in robot_type:
                extra_rows = 2
            else:
                extra_rows = 1  # A1 or other single-arm
        self.dashboard = Dashboard(enabled=dashboard_enabled, extra_rows=extra_rows)
        # Capture task name once from config
        self._task_name = getattr(getattr(cfg, 'task', None), 'name', '') or ''
        # Propagate event callback to policy so components can log through dashboard
        if hasattr(self.policy, 'set_event_callback'):
            self.policy.set_event_callback(self.dashboard.log_event)

        # This preview is deliberately scoped to real Franka collection. It
        # consumes the already-cropped observations instead of reopening the
        # camera devices.
        self.camera_viewer = None
        self._franka_counted_demos = 0
        self._franka_takeover_demos = 0
        if self.env_type == "real_franka" and self.config.mode == "collection":
            viewer_cfg = self.config.env.cfg.get("camera_viewer", {})
            self.camera_viewer = CameraStreamViewer(
                enabled=bool(viewer_cfg.get("enabled", False)),
                show_takeover_count=bool(
                    viewer_cfg.get("show_takeover_count", False)
                ),
            )
            self._update_franka_camera_viewer()

    @property
    def video_writer(self) -> Tuple[Container, Stream]:
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            (container, stream) = self._video_writer
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)
            # Close the container
            container.close()
        self._video_writer = video_writer

    def load_env(self) -> Union[og.Environment, RealEnvironment]:
        """
        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        self.env_type = self.config.env.env_type
        if self.env_type == "sim":
            prepare_omni_registry_paths()
            config = OmegaConf.to_container(self.config.env.cfg)
            # Update observation modalities
            config["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES[self.config.robot.type].keys())
            env = og.Environment(configs=config)
            # check mode
            if self.config.mode is not None:
                if self.config.mode == "collection":
                    # don't join if hdf_path is already absolute
                    if not os.path.isabs(self.config.hdf_path):
                        hdf_path = os.path.join(self.base_path, self.config.hdf_path)
                    else:
                        hdf_path = self.config.hdf_path
                    if os.path.exists(hdf_path) and not self.config.overwrite:
                        raise FileExistsError(f"HDF5 file {hdf_path} already exists! Please set `overwrite` to True to overwrite it.")
                    env = SimDataCollectionWrapper(env, hdf_path, only_successes=self.config.log_only_success, manual_save_discard=self.config.get("manual_save_discard", True))
                else:
                    raise ValueError(f"Invalid mode {self.config.mode}")
        elif self.env_type == "real":
            env = RealEnvironment(config=self.config.env.cfg)
            env.task = REGISTERED_TASKS[self.config.task.type](**self.config.task)
            if self.config.mode == "collection":
                # don't join if hdf_path is already absolute
                if not os.path.isabs(self.config.hdf_path): 
                    hdf_path = os.path.join(self.base_path, self.config.hdf_path)
                else:
                    hdf_path = self.config.hdf_path
                if os.path.exists(hdf_path) and not self.config.overwrite:
                    raise FileExistsError(f"HDF5 file {hdf_path} already exists! Please set `overwrite` to True to overwrite it.")
                env = RealDataCollectionWrapper(env, hdf_path, only_successes=self.config.log_only_success)
            elif self.config.mode is not None:
                raise ValueError(f"Invalid mode {self.config.mode}")
        elif self.env_type in ("real_franka", "sim_franka"):
            env = RealFrankaEnvironment(config=self.config.env.cfg)
            env.task = REGISTERED_TASKS[self.config.task.type](**self.config.task)
            if self.config.mode == "collection":
                # don't join if hdf_path is already absolute
                if not os.path.isabs(self.config.hdf_path):
                    hdf_path = os.path.join(self.base_path, self.config.hdf_path)
                else:
                    hdf_path = self.config.hdf_path
                if os.path.exists(hdf_path) and not self.config.overwrite:
                    raise FileExistsError(f"HDF5 file {hdf_path} already exists! Please set `overwrite` to True to overwrite it.")
                env = RealDataCollectionWrapper(env, hdf_path, only_successes=self.config.log_only_success)
            elif self.config.mode is not None:
                raise ValueError(f"Invalid mode {self.config.mode}")
        else:
            raise ValueError(f"Invalid environment type {self.env_type}")
        return env

    def load_robot(self) -> Robot:
        if self.env_type == "sim":
            robot = self.env.scene.object_registry("name", "robot")
            if robot.model == "a1":
                robot.__class__ = SimA1
            elif robot.model == "r1pro":
                robot.__class__ = SimR1Pro
        elif self.env_type == "real":
            # Select robot class based on robot type in config
            robot_type = self.config.robot.type
            if "R1Pro" in robot_type:
                robot = RealR1Pro(name="robot")
            else:
                raise ValueError(f"Invalid robot type {robot_type}")
            self.env.robots = [robot]
            # need to load task here because it's not handled by env
            self.env.task.load(self.env)
        elif self.env_type == "real_franka":
            robot = RealFranka(
                name="robot",
                gripper_open_action=float(self.config.env.cfg.get("gripper_open_action", -1.0)),
            )
            self.env.robots = [robot]
            self.env.task.load(self.env)
        elif self.env_type == "sim_franka":
            robot = SimFranka(name="robot")
            self.env.robots = [robot]
            self.env.task.load(self.env)
        else:
            raise ValueError(f"Invalid environment type {self.env_type}")
        return robot

    def load_policy(self) -> BasePolicy:
        self._auto_detect_gello_overlay()
        policy = policy_lib[self.config.policy.type](config=self.config.policy)
        policy.load(robot=self.robot)

        # Set debounce period from environment frequency
        env = self.env.env if hasattr(self.env, 'env') else self.env
        if hasattr(env, 'frequency') and hasattr(policy, 'set_debounce_period'):
            debounce_period = 1.0 / env.frequency
            policy.set_debounce_period(debounce_period)
            logger.info(f"Set policy debounce period to {debounce_period*1000:.1f}ms (frequency={env.frequency}Hz)")

        # Enable control loop profiling on Gello controller when CSV profiling is active
        if self._profile_csv and hasattr(policy, '_components'):
            ctrl = getattr(policy._components, 'gello_controller', None)
            if ctrl is not None and hasattr(ctrl, 'enable_loop_profiling'):
                ctrl.enable_loop_profiling("logs/profile_control_loop.csv")

        return policy

    def _auto_detect_gello_overlay(self) -> None:
        """
        Auto-detect gello_task_overlay from the task type when not explicitly set.

        Derives overlay name from task.type by stripping the 'Task' suffix and
        lowercasing (e.g., OpenTask -> open, AutotuneTask -> autotune).
        Only sets the overlay if the corresponding file exists on disk.
        """
        from omegaconf import OmegaConf
        from iiil.utils.misc_utils import get_config_path

        policy_cfg = self.config.policy
        if policy_cfg.get("gello_task_overlay", None) is not None:
            return  # Explicit override, don't auto-detect

        if policy_cfg.get("type") != "gello":
            return  # Only relevant for gello policy

        task_type = self.config.task.get("type", None)
        if task_type is None:
            return

        # Derive overlay name: "OpenTask" -> "open", "AutotuneTask" -> "autotune"
        overlay_name = task_type.removesuffix("Task").lower()

        # Resolve gello config name: explicit > robot-type mapping > default
        gello_config_name = policy_cfg.get("gello_config", None)
        if gello_config_name is None:
            robot_type = self.config.get("robot", {}).get("type", None)
            gello_config_name = {
                "R1Pro": "gello_r1pro",
                "A1": "gello_a1",
                "Franka": "gello_franka",
            }.get(robot_type, "gello_r1pro")

        overlay_path = os.path.join(
            get_config_path(), "gello", gello_config_name, f"{overlay_name}.yaml"
        )

        if os.path.exists(overlay_path):
            OmegaConf.update(self.config, "policy.gello_task_overlay", overlay_name)
            logger.info(
                f"Auto-detected gello task overlay '{overlay_name}' from task type '{task_type}'"
            )

    def step(self) -> None:
        """
        Single step of the task with optional trial control.
        X button starts trial, A/Y buttons end trial (via termination conditions).
        Home button moves robot and Gello to home position.
        """
        from time import perf_counter, sleep
        _profile = getattr(self, '_profile_steps', False)
        _timer = self._step_timer  # None if CSV profiling disabled

        # Rate limiting: record step start time
        step_start = perf_counter()
        if _timer:
            _timer.mark("start")
        if _profile:
            _t = {'start': step_start}

        # Check if Home button pressed to move to home position
        if self.policy.supports_trial_control:
            # Refresh input state before checking buttons (ensures fresh read)
            if hasattr(self.policy, 'refresh_input_state'):
                self.policy.refresh_input_state()

            if self.policy.check_home_requested():
                self._move_to_home()
                # Refresh obs so policy sees the robot at home
                self.obs = self._preprocess_obs(self.env.get_obs()[0])
                # Auto-start trial (recording) and resume policy mode
                if not self.policy.is_trial_active():
                    self.policy.start_trial()
                    if getattr(self.env, 'supports_trial_recording', False):
                        self.env.start_trial_recording(self.obs)
                self._update_franka_camera_viewer()
                return False, False  # Continue without ending episode

            # Check if X button pressed to start trial
            if self.policy.check_start_trial_pressed() and not self.policy.is_trial_active():
                self.policy.start_trial()
                if getattr(self.env, 'supports_trial_recording', False):
                    self.env.start_trial_recording(self.obs)

        if _timer:
            _timer.mark("input")
        if _profile:
            _t['after_input'] = perf_counter()

        task_info = self.env.task.task_info
        if task_info == {}:
            self.robot_action = self.robot.get_no_op_action().cpu()
        else:
            action = self.policy.predict(
                obs=self.obs,
                task_info=task_info,
            )
            self.robot_action = self.robot.process_action(action.flatten().cpu())

        if _timer:
            _timer.mark("predict")
        if _profile:
            _t['after_predict'] = perf_counter()

        self.obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)

        if _timer:
            _timer.mark("env_step")
        if _profile:
            _t['after_env_step'] = perf_counter()

        self.obs = self._preprocess_obs(self.obs)

        if _timer:
            _timer.mark("postprocess")
            _timer.end_iteration()
        if _profile:
            _t['end'] = perf_counter()
            # Print timing every 50 steps
            if not hasattr(self, '_step_count'):
                self._step_count = 0
            self._step_count += 1
            if self._step_count % 50 == 0:
                input_t = (_t['after_input'] - _t['start']) * 1000
                predict_t = (_t['after_predict'] - _t['after_input']) * 1000
                env_step_t = (_t['after_env_step'] - _t['after_predict']) * 1000
                postprocess_t = (_t['end'] - _t['after_env_step']) * 1000
                total_t = (_t['end'] - _t['start']) * 1000
                logger.info(f"[STEP {self._step_count}] total={total_t:.1f}ms | input={input_t:.1f}ms | predict={predict_t:.1f}ms | env.step={env_step_t:.1f}ms | postprocess={postprocess_t:.1f}ms")

        # Update dashboard
        self.dashboard.update(self._gather_dashboard_state())

        # Check for consecutive planning failures (e.g. CuRobo stuck)
        base = getattr(self.policy, '_base_policy', self.policy)
        if getattr(base, 'needs_reset', False):
            self.dashboard.log_event("\033[1;31m>>> AUTO-RESET: Too many consecutive planning failures — resetting episode\033[0m")
            self.reset_episode()
            return False, False

        if terminated:
            self.env.task.n_trials += 1
            success = info["done"]["success"] if "done" in info and "success" in info["done"] else None
            if success:
                self.env.task.n_success_trials += 1
                self.env._n_successful_episodes += 1
            if self.policy.supports_trial_control:
                self.policy.stop_trial(success=success)
            # Flush trajectory data immediately so it survives Ctrl+C during home movement
            if hasattr(self.env, 'flush_current_traj') and len(self.env.current_traj_history) > 0:
                self.env.flush_current_traj()

        # Refresh after a possible flush so saved counts change immediately.
        self._update_franka_camera_viewer()

        # Rate limiting: sleep to maintain target frequency if enabled
        if self._enforce_frequency:
            # Get frequency from env (handle wrapped env case)
            env = self.env.env if hasattr(self.env, 'env') else self.env
            if hasattr(env, 'frequency'):
                target_period = 1.0 / env.frequency
                elapsed = perf_counter() - step_start
                sleep_time = target_period - elapsed
                if sleep_time > 0:
                    sleep(sleep_time)
                if _profile and self._step_count % 50 == 0:
                    actual_period = perf_counter() - step_start
                    logger.info(f"  [RATE] target={target_period*1000:.1f}ms | elapsed={elapsed*1000:.1f}ms | slept={max(0,sleep_time)*1000:.1f}ms | actual={actual_period*1000:.1f}ms")

        return terminated, truncated

    def _update_franka_camera_viewer(self) -> None:
        """Update the real-Franka-only camera window and durable demo counts."""
        if self.camera_viewer is None:
            return

        total_saved = int(getattr(self.env, "traj_count", 0))
        hdf5_file = getattr(self.env, "hdf5_file", None)
        while hdf5_file is not None and self._franka_counted_demos < total_saved:
            demo_name = f"demo_{self._franka_counted_demos}"
            try:
                demo = hdf5_file["data"][demo_name]
                active = demo.get("policy/is_oracle_active")
                if active is not None and bool(np.any(active[()])):
                    self._franka_takeover_demos += 1
            except (KeyError, OSError, ValueError):
                # A flush is normally complete before this runs. If not, retry
                # this demo on the next control-loop iteration.
                break
            self._franka_counted_demos += 1

        self.camera_viewer.update(
            self.obs,
            total_saved=total_saved,
            takeover_saved=self._franka_takeover_demos,
            recording=bool(getattr(self.policy, "_trial_active", False)),
        )

    def _move_to_home(self) -> None:
        """
        Move robot AND Gello to home position.
        Called when Home button is pressed.
        """
        self.dashboard.log_event("[HOME] Moving robot and Gello to home position...")

        # Move robot to home position
        if hasattr(self.robot, 'move_to_home'):
            self.robot.move_to_home()

        # Move Gello to home position
        if hasattr(self.policy, 'move_gello_to_home'):
            self.policy.move_gello_to_home()

        self.dashboard.log_event("[HOME] Done.")

    def reset(self) -> None:
        """
        Reset the environment and policy, and clear aggregated metrics.
        """
        self.reset_episode()
        self.env.task.n_success_trials, self.env.task.n_trials = 0, 0

    def reset_episode(self) -> None:
        """
        Reset environment + policy and cache the initial observation.
        Useful for rolling into the next episode without wiping metrics.
        """
        # env.reset() triggers task._reset_agent() which calls policy.reset()
        # So we don't need to call policy.reset() directly here
        self.obs = self._preprocess_obs(self.env.reset()[0])
        # Give policies a hook to re-home hardware using the fresh observation
        self.policy.on_env_reset(self.obs)

    def _preprocess_obs(self, obs: dict) -> dict:
        """
        Preprocess the observation dictionary before passing it to the policy.
        """
        obs = flatten_obs_dict(obs)
        return obs

    def _gather_dashboard_state(self) -> DashboardState:
        """Collect current runtime values into a DashboardState snapshot."""
        state = DashboardState()

        # Trial info
        state.trial_num = self.env.task.n_trials
        state.trial_total = int(getattr(
            getattr(getattr(self.config, 'task', None), 'termination_config', None),
            'total_episode', 0
        ) or 0)
        state.trial_successes = self.env.task.n_success_trials
        state.task_name = self._task_name

        # Demo number from data collection wrapper
        env = self.env
        if hasattr(env, 'traj_count'):
            state.demo_num = env.traj_count
        elif hasattr(env, 'env') and hasattr(env.env, 'traj_count'):
            state.demo_num = env.env.traj_count

        # Recording state
        state.recording = getattr(self.policy, '_trial_active', False)

        # Gate state and takeover info (only available with GelloPolicy)
        components = getattr(self.policy, '_components', None)
        if components is not None:
            gate = getattr(components, 'gate', None)
            if gate is not None:
                is_correcting = gate.is_currently_correcting()
                state.gate_state = "TAKEOVER" if is_correcting else "POLICY"
                state.takeover_detected = is_correcting

            # Control mode
            mode_mgr = getattr(components, 'mode_manager', None)
            if mode_mgr is not None:
                state.control_mode = mode_mgr.active_mode_name or ""

            # Peak joint diagnostic and release state from takeover detector
            gello_ctrl = getattr(components, 'gello_controller', None)
            if gello_ctrl is not None:
                detector = getattr(gello_ctrl, '_takeover_detector', None)
                if detector is not None and hasattr(detector, 'buffers'):
                    try:
                        avg = np.mean(detector.buffers['current'], axis=0)
                        peak_j = int(np.argmax(avg))
                        state.peak_joint_name = f"J{peak_j}"
                        state.peak_value = float(avg[peak_j])
                        state.peak_threshold = float(detector.current_threshold[peak_j])
                    except Exception:
                        pass

                    # Tracking error (position buffer)
                    try:
                        state.tracking_error = list(np.mean(detector.buffers['position'], axis=0))
                        state.position_threshold = list(detector.position_threshold)
                    except Exception:
                        pass

                    # Release timer state
                    state.release_timer_active = detector.release_start_time is not None
                    if state.release_timer_active:
                        import time as _time
                        state.release_elapsed = _time.time() - detector.release_start_time
                    state.release_timeout = detector.config.auto_return_timeout
                    state.release_vel_max = detector._release_vel_max
                    state.release_vel_threshold = detector.config.release_velocity
                    state.release_curr_std = detector._release_curr_std
                    state.release_curr_threshold = detector.config.release_current_derivative

        state.robot_type = self._robot_type
        state.robot_dof = len(state.tracking_error) if state.tracking_error else 0

        return state

    def _write_video(self) -> None:
        if self.config.robot.type == "R1Pro":
            # concatenate obs
            left_wrist_rgb = cv2.resize(
                self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(),
                (224, 224),
            )
            right_wrist_rgb = cv2.resize(
                self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(),
                (224, 224),
            )
            head_rgb = cv2.resize(
                self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy(),
                (224, 224),
            )
            frame = np.expand_dims(np.hstack([left_wrist_rgb, head_rgb, right_wrist_rgb]), 0)
        elif self.config.robot.type == "A1":
            wrist_rgb = cv2.resize(
                self.obs[ROBOT_CAMERA_NAMES["A1"]["wrist"] + "::rgb"].numpy(),
                (224, 224),
            )
            external_rgb = cv2.resize(
                self.obs[ROBOT_CAMERA_NAMES["A1"]["external"] + "::rgb"].numpy(),
                (224, 224),
            )
            frame = np.expand_dims(np.hstack([wrist_rgb, external_rgb]), 0)
        write_video(
            frame,
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

        
    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.dashboard.clear()
        if self.camera_viewer is not None:
            self.camera_viewer.close()
        # save data if applicable
        if self.config.mode == "collection":
            self.env.save_data()
        close_video_writer(self._video_writer)
        self._video_writer = None
        n_trials = getattr(self.env.task, 'n_trials', 0)
        n_success = getattr(self.env.task, 'n_success_trials', 0)
        if self.config.results_json_path is not None:
            write_results_json(self.config.results_json_path, n_trials, n_success)
        # print stats
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {n_success}")
        logger.info(f"Total trials: {n_trials}")
        if n_trials > 0:
            logger.info(f"Success rate: {n_success / n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        # Save step profiling CSV
        if self._step_timer is not None:
            os.makedirs("logs", exist_ok=True)
            self._step_timer.save_csv("logs/profile_main_step.csv")
            self._step_timer = None

        self.policy.close()
        self.env.close()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        og.shutdown()
        sys.exit(0)

"""
Bilateral controller using only current control.

A clean implementation for bilateral control supporting two modes:
- GELLO_TO_ENV: User controls Gello device → reads position → sends to robot
- ENV_TO_GELLO: Robot sends target → Gello follows using PD control through torques

Always uses current-based control with gravity and friction compensation.
Goal tracking is done via PD torque control (not position commands).
"""

import logging
import numpy as np
import time
from threading import Event, Thread, Lock
from typing import Sequence, Optional, Tuple, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from iiil.gello.controllers.base_controller import BilateralGelloController
from iiil.gello.configs.gello_motor_group import GelloMotorGroup
from iiil.gello.drivers.unified_driver import UnifiedDXLDriver
from iiil.gello.compensation.compensation_config import CompensationConfig
from iiil.gello.compensation.gravity_compensation import GravityCompensator
from iiil.gello.compensation.friction_compensation import FrictionCompensator
from iiil.gello.compensation.anti_friction_compensation import AntiFrictionCompensator
from iiil.gello.control.goal_tracker import GoalTracker, GoalTrackingConfig
from iiil.gello.control.takeover_detector import TakeoverDetector, TakeoverConfig
from iiil.gello.control.touch_sensor import TouchSensorReader, TouchSensorConfig
from iiil.gello.control.momentum_observer import MomentumObserver, MomentumObserverConfig
from iiil.gello.utils.encoder_utils import absolute_encoder_to_radians, radians_to_absolute_encoder
from iiil.gello.utils.trajectory_utils import create_trajectory, interpolate_trajectory, TrajectoryState
from iiil.gello.utils.joint_utils import (
    infer_dof_from_motor_groups,
    check_joint_distance_to_goal,
    clip_to_limits,
    create_default_joint_limits,
    expand_goal_to_full_dof,
    wrap_to_joint_centers,
)

@dataclass
class DriverInfo:
    driver: UnifiedDXLDriver
    robot_indices: np.ndarray
    joint_signs: np.ndarray  # Signs for torque conversion (robot space → motor space)

class BilateralController(BilateralGelloController):
    """
    Bilateral controller using only current control.

    This controller is DOF-agnostic and supports two modes:
    1. GELLO_TO_ENV: User controls Gello, position is read and sent to robot
    2. ENV_TO_GELLO: Robot sends target position, Gello follows via PD torque control

    Always uses current-based control with gravity/friction compensation.
    """

    def __init__(
        self,
        *,
        motor_groups: Sequence[GelloMotorGroup],
        compensation_config: Optional[CompensationConfig] = None,
        goal_tracking_config: Optional[GoalTrackingConfig] = None,
        takeover_config: Optional[TakeoverConfig] = None,
        momentum_observer_config: Optional[MomentumObserverConfig] = None,
        gello_to_env_config: Optional[Dict] = None,
        joint_offsets: Optional[np.ndarray] = None,
        joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        control_frequency: float = 30.0,
        debug: bool = False,
        touch_sensor_config: Optional[TouchSensorConfig] = None,
    ):
        """
        Initialize bilateral controller.

        Args:
            motor_groups: List of motor group configurations
            compensation_config: Compensation configuration (gravity/friction/anti-friction)
            goal_tracking_config: Goal tracking configuration for ENV_TO_GELLO mode
            takeover_config: Takeover detection configuration
            momentum_observer_config: Momentum observer configuration for external torque estimation
            gello_to_env_config: GELLO_TO_ENV mode configuration with:
                - torque_enabled: bool - if True, apply gravity+anti-friction in GELLO_TO_ENV mode
            joint_offsets: Additional joint offsets in radians (default: zeros)
            joint_limits: Tuple of (lower, upper) joint limit arrays. When provided,
                         enables proper angle wrapping for encoders with large offsets.
            control_frequency: Target control loop frequency in Hz
            debug: Whether to print debug information
        """
        # Initialize stop event early to prevent AttributeError in __del__
        self._stop_event = Event()
        self._control_thread = None
        self._has_started = False

        # Infer DOF from motor groups
        self._dof = infer_dof_from_motor_groups(motor_groups)
        self._motor_groups = motor_groups

        # Joint limits and wrapping
        if joint_limits is not None:
            self.JOINT_LIMIT_LOW, self.JOINT_LIMIT_HIGH = joint_limits
            self._joint_centers = (self.JOINT_LIMIT_LOW + self.JOINT_LIMIT_HIGH) / 2.0
            self._use_angle_wrapping = True
            print(f"Joint limits configured (wrapping enabled): "
                  f"centers=[{', '.join(f'{c:.2f}' for c in self._joint_centers)}]")
        else:
            self.JOINT_LIMIT_LOW, self.JOINT_LIMIT_HIGH = create_default_joint_limits(self._dof)
            self._joint_centers = None
            self._use_angle_wrapping = False

        # Joint direction signs
        self._joint_signs = np.ones(self._dof, dtype=np.float32)
        for group in motor_groups:
            self._joint_signs[group.robot_joint_indices] = group.arm_joint_signs

        # Joint offsets (additional to encoder zero at 180°)
        self._joint_offsets = joint_offsets if joint_offsets is not None else np.zeros(self._dof)

        # Initialize drivers (grouping by port to avoid conflicts)
        self._drivers: List[UnifiedDXLDriver] = []
        self._driver_infos: List[DriverInfo] = []
        
        # Group motor configs by port
        port_groups: Dict[str, List[GelloMotorGroup]] = {}
        for group in motor_groups:
            if group.device_port not in port_groups:
                port_groups[group.device_port] = []
            port_groups[group.device_port].append(group)
        
        # Create one driver per port
        for port, groups in port_groups.items():
            # Aggregate IDs, indices, and signs
            all_ids = []
            all_indices = []
            all_signs = []

            # Use the first group's config for port-level settings
            first_group = groups[0]
            baudrate = first_group.baudrate
            motor_type = first_group.motor_type

            for group in groups:
                # Validation
                if group.baudrate != baudrate:
                    print(f"Warning: Baudrate mismatch on port {port}. Using {baudrate}.")

                all_ids.extend(group.motor_ids)
                all_indices.extend(group.robot_joint_indices)
                all_signs.extend(group.arm_joint_signs)

            # Create driver — pass current_limit and profile params from first group.
            driver = UnifiedDXLDriver(
                ids=all_ids,
                port=port,
                motor_type=motor_type,
                baudrate=baudrate,
                initial_mode="current",
                current_limit=first_group.current_limit,
                profile_velocity=first_group.profile_velocity,
                profile_acceleration=first_group.profile_acceleration,
                position_deadband=first_group.position_deadband,
            )

            self._drivers.append(driver)
            self._driver_infos.append(DriverInfo(
                driver=driver,
                robot_indices=np.array(all_indices, dtype=int),
                joint_signs=np.array(all_signs, dtype=np.float32)
            ))

        # Initialize compensation modules
        if compensation_config is None:
            compensation_config = CompensationConfig.create_default(self._dof)

        self._gravity_compensator = GravityCompensator(
            compensation_config.gravity, self._dof
        )
        self._friction_compensator = FrictionCompensator(
            compensation_config.friction, self._dof
        )
        self._anti_friction_compensator = AntiFrictionCompensator(
            compensation_config.anti_friction, self._dof
        )

        # GELLO_TO_ENV mode configuration
        # When torque_enabled=True: apply gravity + anti-friction compensation
        # When torque_enabled=False: disable torque for free float (legacy behavior)
        if gello_to_env_config is None:
            gello_to_env_config = {}
        self._gello_to_env_torque_enabled = gello_to_env_config.get("torque_enabled", False)

        # Initialize goal tracking for ENV_TO_GELLO mode
        if goal_tracking_config is None:
            goal_tracking_config = GoalTrackingConfig(enable=True)
        self._goal_tracker = GoalTracker(goal_tracking_config, self._dof)

        # Initialize takeover detector
        if takeover_config is None:
            takeover_config = TakeoverConfig()
        motor_types_per_joint = self._get_motor_types_per_joint()
        self._takeover_detector = TakeoverDetector(takeover_config, self._dof, motor_types_per_joint)

        # Initialize touch sensor (if configured)
        self._touch_sensor: Optional[TouchSensorReader] = None
        if touch_sensor_config is not None and touch_sensor_config.enable:
            self._touch_sensor = TouchSensorReader(touch_sensor_config)
            self._takeover_detector.set_touch_sensor(self._touch_sensor)

        # Initialize momentum observer (shares Pinocchio model with gravity compensator)
        if momentum_observer_config is None:
            momentum_observer_config = MomentumObserverConfig()
        if momentum_observer_config.enable and self._gravity_compensator.model is not None:
            self._momentum_observer = MomentumObserver(
                config=momentum_observer_config,
                pinocchio_model=self._gravity_compensator.model,
                dof=self._dof,
                is_bilateral_single_arm=self._gravity_compensator._is_bilateral_single_arm_urdf,
                dt=self._control_period,
            )
        else:
            self._momentum_observer = MomentumObserver(
                config=MomentumObserverConfig(enable=False),
                pinocchio_model=None,
                dof=self._dof,
                is_bilateral_single_arm=False,
            )

        self._takeover_callback = None
        self._auto_return_callback = None
        self._event_callback = print

        # Admittance mode flag: whether to stop tracking on takeover
        # Default True = stop tracking (legacy behavior for ENV_TO_GELLO mode)
        # Set to False for admittance mode where we want physical blend to continue
        self._stop_tracking_on_takeover = True

        # Goal tracking state (for trajectory interpolation and convergence checking)
        self._goal = None
        self._trajectory_state: Optional[TrajectoryState] = None
        self._should_follow_goal = False
        self._is_torque_mode = False
        self._goal_lock = Lock()

        # Control state
        self._control_frequency = control_frequency
        self._control_period = 1.0 / control_frequency
        self._debug = debug

        # Debug message filtering (console output)
        self._debug_msg = {
            'enable': False,
            'control_loop': False,
            'tracking': False,
            'user_control': False,
            'takeover': True,
            'release': True,
            'goal_reached': False,
        }

        # Debug logging (structured data capture and visualization)
        self._debug_logger = None

        # Loop profiling (manual per-section timing)
        self._loop_timer = None
        self._loop_timer_output_path = None

        # Thread management
        self._control_thread = Thread(target=self._control_loop, daemon=True)

        # Get max velocities from motor groups
        self._max_velocities = np.zeros(self._dof, dtype=np.float32)
        for group in motor_groups:
            self._max_velocities[group.robot_joint_indices] = group.max_joint_velocity

        # State for admittance gating (accessed from pipeline)
        self._last_measured_current = np.zeros(self._dof, dtype=np.float32)
        self._last_tracking_torque = np.zeros(self._dof, dtype=np.float32)
        self._last_gravity_torque = np.zeros(self._dof, dtype=np.float32)
        self._last_friction_torque = np.zeros(self._dof, dtype=np.float32)
        self._last_anti_friction_torque = np.zeros(self._dof, dtype=np.float32)

        # Cached state from control loop (for fast reads from main thread)
        self._cached_positions = np.zeros(self._dof, dtype=np.float32)
        self._cached_velocities = np.zeros(self._dof, dtype=np.float32)
        self._cached_currents = np.zeros(self._dof, dtype=np.float32)
        self._cached_valid = np.ones(self._dof, dtype=bool)
        self._cache_lock = Lock()

    def start_control(self):
        """Start the control loop."""
        if self._has_started:
            return

        # Enable torque for all motors
        for driver in self._drivers:
            driver.set_torque_mode(True)

        # Start touch sensor reader thread (if configured)
        if self._touch_sensor is not None:
            self._touch_sensor.start()

        self._has_started = True
        self._control_thread.start()

    def ensure_torque_enabled(self):
        """
        Best-effort torque re-enable without restarting the control thread.
        Useful if comm errors temporarily disable torque.
        """
        for driver in self._drivers:
            try:
                driver.set_torque_mode(True)
            except Exception as e:
                print(f"Failed to re-enable torque for driver {driver}: {e}")

    def _set_all_torque(self, enable: bool):
        """Enable or disable torque on all motor drivers."""
        for driver in self._drivers:
            try:
                driver.set_torque_mode(enable)
            except Exception as e:
                print(f"Failed to set torque mode: {e}")

    def stop_goal_following(self):
        """Stop following goal (switch to GELLO_TO_ENV mode - user control)."""
        self._goal_tracker.stop_tracking()
        with self._goal_lock:
            self._should_follow_goal = False
        # Disable torque only if gello_to_env.torque_enabled is False
        # When torque_enabled=True, we keep torque on for gravity + anti-friction
        if not self._gello_to_env_torque_enabled:
            self._set_all_torque(False)
            self._is_torque_mode = False

    def resume_goal_following(self):
        """Resume following goal (switch to ENV_TO_GELLO mode - robot commands Gello)."""
        # Re-enable torque before tracking
        if not self._is_torque_mode:
            self._set_all_torque(True)
            self._is_torque_mode = True
        self._goal_tracker.start_tracking()
        with self._goal_lock:
            self._should_follow_goal = True

    def set_new_goal(self, goal_joint_angles: np.ndarray):
        """
        Set a new goal position for the controller to follow.

        In ENV_TO_GELLO mode, Gello will track this goal using PD torque control.

        Args:
            goal_joint_angles: Target joint positions in radians
        """
        if len(goal_joint_angles) > self._dof:
            raise ValueError(f"Expected at most {self._dof} joint angles, got {len(goal_joint_angles)}")

        current_pos, valid_joints = self.q
        full_goal = expand_goal_to_full_dof(goal_joint_angles, current_pos, self._dof)

        # Set goal in goal tracker
        self._goal_tracker.set_goal(full_goal)
        # Ensure tracking is active for new goals (can be stopped by takeover)
        self._goal_tracker.start_tracking()

        # Create trajectory for smooth interpolation and convergence checking
        with self._goal_lock:
            self._trajectory_state = create_trajectory(
                start_position=current_pos,
                goal_position=full_goal,
                max_velocities=self._max_velocities,
                valid_joints=valid_joints
            )
            self._goal = full_goal.copy()
            self._should_follow_goal = True

    def set_takeover_callback(self, callback):
        """
        Set callback for takeover detection.

        Args:
            callback: Function to call when user takes over control (signature: callback() -> None)
        """
        self._takeover_callback = callback

    def set_auto_return_callback(self, callback):
        """
        Set callback for auto-return (release detection).

        Args:
            callback: Function to call when user releases control and auto-return triggers (signature: callback() -> None)
        """
        self._auto_return_callback = callback

    def set_event_callback(self, fn) -> None:
        """Set callback for dashboard event logging. Propagates to takeover detector."""
        self._event_callback = fn
        if hasattr(self, '_takeover_detector'):
            self._takeover_detector.set_event_callback(fn)

    def set_stop_tracking_on_takeover(self, stop: bool) -> None:
        """
        Configure whether takeover detection stops goal tracking.

        In admittance mode, we want to keep tracking so the physical blend
        continues (policy goal + human force deflection). Set to False.

        In ENV_TO_GELLO mode with mode switching, we want to stop tracking
        when human takes over. Set to True (default).

        Args:
            stop: If True, stop goal tracking when takeover detected.
                  If False, continue tracking (for admittance physical blend).
        """
        self._stop_tracking_on_takeover = stop

    def set_tracking_gains(self, kp=None, ki=None, kd=None, kff=None, reset_integral=True):
        """Update goal tracking PID and feedforward gains at runtime."""
        self._goal_tracker.set_gains(kp=kp, ki=ki, kd=kd, kff=kff, reset_integral=reset_integral)

    def switch_to_position_control(self):
        """Switch all motor drivers to Dynamixel position control mode.

        Reads current encoder positions and passes them as the initial goal
        so motors don't jump to a stale GoalPosition register value.
        """
        raw_positions, _, _, _ = self._read_all_motor_state()
        for info in self._driver_infos:
            driver = info.driver
            indices = info.robot_indices
            driver.switch_to_position_mode(
                initial_goal_encoder=raw_positions[indices]
            )

    def switch_to_current_control(self):
        """Switch all motor drivers to Dynamixel current (torque) control mode."""
        for info in self._driver_infos:
            info.driver.switch_to_current_mode()

    @property
    def is_takeover_detected(self) -> bool:
        """Last takeover detection result from the 100Hz control loop."""
        return self._takeover_detector._last_takeover_state

    @property
    def estimated_external_torque(self) -> Optional[np.ndarray]:
        """Last estimated external torque from momentum observer (Nm per joint)."""
        return self._momentum_observer.external_torque

    def configure_debug_messages(self, debug_msg_config):
        """
        Configure which debug messages are printed to console.

        Args:
            debug_msg_config: Dict with debug message configuration
                {
                    'enable': bool,  # Master switch
                    'control_loop': bool,  # Control loop frequency
                    'tracking': bool,  # Tracking mode currents/errors
                    'user_control': bool,  # User control mode
                    'takeover': bool,  # Takeover detection
                    'release': bool,  # Release/auto-return
                    'goal_reached': bool,  # Goal convergence
                }
        """
        if debug_msg_config:
            self._debug_msg.update(debug_msg_config)

    def enable_debug_logging(self, debug_config):
        """
        Enable structured debug logging with optional real-time visualization.

        Args:
            debug_config: DebugLoggingConfig or dict with debug logging configuration
        """
        try:
            from iiil.gello.debug.debug_config import DebugLoggingConfig
            from iiil.gello.debug.debug_logger import DebugDataLogger

            # Convert dict to config if needed
            if isinstance(debug_config, dict):
                debug_config = DebugLoggingConfig.from_dict(debug_config)

            # Initialize debug logger
            motor_types_per_joint = self._get_motor_types_per_joint()

            self._debug_logger = DebugDataLogger(
                debug_config, self._dof, motor_types_per_joint
            )

            logger.info("Structured debug logging enabled")

        except Exception:
            logger.exception("Failed to enable structured debug logging")
            self._debug_logger = None

    def enable_loop_profiling(self, output_path: str = "logs/profile_control_loop.csv") -> None:
        """Enable per-section timing of the control loop thread.

        Timing data is accumulated in memory and saved to CSV when close() is called.

        Args:
            output_path: Path for the output CSV file.
        """
        from iiil.utils.loop_timer import LoopTimer
        self._loop_timer = LoopTimer("control_loop", print_interval_s=5.0)
        self._loop_timer_output_path = output_path
        print(f"[BilateralController] Loop profiling enabled → {output_path}")

    def has_reached_goal(self, block: bool = False, tolerance: float = 0.1) -> bool:
        """
        Check if Gello has reached the goal position.

        Args:
            block: If True, block until goal is reached
            tolerance: Distance tolerance in radians

        Returns:
            True if goal is reached (within tolerance)
        """
        if self._goal is None:
            return True

        if block:
            while not self._check_reached_goal(tolerance):
                if self._stop_event.is_set():
                    return False
                time.sleep(0.01)
            return True
        else:
            return self._check_reached_goal(tolerance)

    def _check_reached_goal(self, tolerance: float) -> bool:
        """Helper to check if goal is reached."""
        if self._goal is None:
            return True

        current_pos, valid_joints = self.q
        return check_joint_distance_to_goal(
            current_pos, self._goal, tolerance, valid_joints
        )

    def _get_motor_types_per_joint(self) -> List:
        """Get motor type for each joint."""
        motor_types = [None] * self._dof
        for group in self._motor_groups:
            # group.motor_type is already a MotorType enum
            for idx in group.robot_joint_indices:
                motor_types[idx] = group.motor_type
        return motor_types

    def close(self, disable_torque: bool = True):
        """Stop control loop and close all drivers."""
        print("Closing bilateral controller")
        self._goal_tracker.stop_tracking()
        with self._goal_lock:
            self._should_follow_goal = False

        self._stop_event.set()
        if self._control_thread is not None and self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)
            if self._control_thread.is_alive():
                print("Control thread did not exit cleanly within timeout")

        # Stop touch sensor reader thread
        if self._touch_sensor is not None:
            self._touch_sensor.stop()

        # Save loop profiling data
        if self._loop_timer is not None and self._loop_timer_output_path is not None:
            import os
            os.makedirs(os.path.dirname(self._loop_timer_output_path) or ".", exist_ok=True)
            self._loop_timer.save_csv(self._loop_timer_output_path)
            self._loop_timer = None

        # Close debug logger (flush any remaining data)
        if self._debug_logger is not None:
            self._debug_logger.close()
            self._debug_logger = None

        # Best effort to zero torque and disable before shutting down drivers
        if disable_torque:
            self._zero_and_disable_motors()

        for driver in self._drivers:
            driver.close(disable_torque=disable_torque)

    def _zero_and_disable_motors(self):
        """
        Send zero torque and disable torque on all drivers.
        This is a best-effort safety step for teardown / CTRL-C.
        """
        if not hasattr(self, "_drivers"):
            return

        for driver in self._drivers:
            torque_on = False
            try:
                torque_on = driver.torque_enabled()
            except Exception as e:
                print(f"Failed to read torque state for driver {driver}: {e}")

            # Avoid turning on torque during shutdown if we never started control
            if not torque_on and not self._has_started:
                try:
                    driver.set_torque_mode(False)
                except Exception as e:
                    print(f"Failed to ensure torque is off for driver {driver}: {e}")
                continue

            try:
                if not torque_on:
                    driver.set_torque_mode(True)
            except Exception as e:
                print(f"Failed to enable torque for shutdown on driver {driver}: {e}")

            try:
                zero_cmd = np.zeros(len(driver.motor_ids), dtype=np.float32)
                driver.set_controls(zero_cmd)
            except Exception as e:
                print(f"Failed to send zero torque to driver {driver}: {e}")

            try:
                driver.set_torque_mode(False)
            except Exception as e:
                print(f"Failed to disable torque for driver {driver}: {e}")

    def _control_loop(self):
        """Main control loop running in separate thread."""
        last_print_time = time.time()
        loop_count = 0
        _timer = self._loop_timer  # local ref for speed (None if profiling disabled)
        first_iteration = True

        while not self._stop_event.is_set():
            loop_start = time.time()
            if _timer:
                _timer.mark("start")

            # ===== SINGLE USB READ for position, velocity, current =====
            # This uses combined CVP read + Fast Sync Read for maximum efficiency
            raw_positions, motor_velocities, curr_currents, valid_joints = self._read_all_motor_state()
            if _timer:
                _timer.mark("usb_read")

            # Convert raw encoder positions to joint angles (with sign conversion for PID)
            curr_positions = absolute_encoder_to_radians(
                raw_positions,
                signs=self._joint_signs,
                offsets=self._joint_offsets
            )
            # Wrap angles to ±π of each joint center before clipping.
            # Without this, large offsets (e.g. Franka joints with DXL_ZERO_OFFSETS far
            # from 2048) can produce angles outside [-π, π] that get silently clipped
            # to wrong values. This matches FrankaJoylo's _dxl_to_rad normalization.
            if self._use_angle_wrapping:
                curr_positions = wrap_to_joint_centers(curr_positions, self._joint_centers)
            curr_positions = clip_to_limits(
                curr_positions,
                self.JOINT_LIMIT_LOW,
                self.JOINT_LIMIT_HIGH
            )

            # Startup diagnostic: print first encoder reading and converted positions
            if first_iteration:
                first_iteration = False
                raw_str = ', '.join(f'{int(v)}' for v in raw_positions)
                pos_str = ', '.join(f'{v:.4f}' for v in curr_positions)
                sign_str = ', '.join(f'{int(s)}' for s in self._joint_signs)
                off_str = ', '.join(f'{v:.4f}' for v in self._joint_offsets)
                print(f"\n=== Gello Startup Diagnostic ===")
                print(f"  Raw encoders:     [{raw_str}]")
                print(f"  Joint signs:      [{sign_str}]")
                print(f"  Joint offsets:    [{off_str}]")
                print(f"  Converted pos:    [{pos_str}]")
                if self._use_angle_wrapping:
                    ctr_str = ', '.join(f'{v:.4f}' for v in self._joint_centers)
                    print(f"  Joint centers:    [{ctr_str}]")
                print(f"  Angle wrapping:   {self._use_angle_wrapping}")
                print(f"================================\n")

            # Convert velocities to robot frame (with sign conversion)
            curr_velocities = motor_velocities * self._joint_signs

            # Cache for fast access from main thread (no USB read needed)
            with self._cache_lock:
                self._cached_positions[:] = curr_positions
                self._cached_velocities[:] = curr_velocities
                self._cached_currents[:] = curr_currents
                self._cached_valid[:] = valid_joints

            # URDF-frame positions for gravity compensation (without sign conversion)
            # Pinocchio expects positions matching the URDF's kinematic model
            curr_positions_urdf = absolute_encoder_to_radians(
                raw_positions,
                signs=np.ones(self._dof, dtype=np.float32),  # No sign flip
                offsets=self._joint_offsets
            )
            # Wrap URDF-frame positions to [-π, π] for correct gravity computation.
            # Same encoder wraparound issue as robot-frame positions.
            if self._use_angle_wrapping:
                curr_positions_urdf = wrap_to_joint_centers(
                    curr_positions_urdf, np.zeros(self._dof, dtype=np.float32)
                )

            # Compute gravity compensation using URDF-frame positions
            tau_gravity = self._gravity_compensator.compute_torque(
                curr_positions_urdf, curr_velocities
            )
            tau_friction = self._friction_compensator.compute_torque(
                curr_velocities, tau_gravity
            )
            # Compute anti-friction compensation (assists backdriving)
            # Uses robot-frame velocities since we want torque in direction of motion
            tau_anti_friction = self._anti_friction_compensator.compute_torque(
                curr_velocities
            )

            if _timer:
                _timer.mark("compensation")

            # Compute goal tracking torque (for ENV_TO_GELLO mode)
            tau_tracking = self._goal_tracker.compute_tracking_torque(
                curr_positions, curr_velocities
            )

            # Convert robot-frame torques to motor frame for current comparison.
            # tau_tracking is in robot frame (computed from sign-converted positions).
            # Measured currents and gravity/friction torques are in motor frame.
            # Motor frame = URDF frame (positive current = positive encoder movement).
            tau_tracking_motor = tau_tracking * self._joint_signs
            tau_anti_friction_motor = tau_anti_friction * self._joint_signs

            # Store for admittance gating access
            self._last_measured_current = curr_currents
            self._last_tracking_torque = tau_tracking
            self._last_gravity_torque = tau_gravity
            self._last_friction_torque = tau_friction
            self._last_anti_friction_torque = tau_anti_friction

            # Update momentum observer with current state
            # tau_cmd_motor is the total torque that was commanded to the motors
            # The observer relates current state to the forces that acted during the interval
            if self._goal_tracker.is_tracking:
                tau_cmd_motor = tau_gravity + tau_friction + tau_tracking_motor
            elif self._gello_to_env_torque_enabled:
                tau_cmd_motor = tau_gravity + tau_anti_friction_motor
            else:
                tau_cmd_motor = np.zeros(self._dof, dtype=np.float32)

            tau_ext = self._momentum_observer.update(
                positions_urdf=curr_positions_urdf,
                velocities_urdf=motor_velocities,
                tau_cmd_motor=tau_cmd_motor,
            )

            if _timer:
                _timer.mark("tracking")

            # Detect takeover (human pushing against goal tracking)
            if self._goal_tracker.is_tracking:
                # Structured debug logging (100 Hz when enabled)
                if self._debug_logger is not None:
                    pos_err = np.abs(self._goal_tracker._goal_position - curr_positions)
                    curr_tracking = self._takeover_detector._torques_to_current(tau_tracking_motor)
                    curr_gravity = self._takeover_detector._torques_to_current(tau_gravity)
                    curr_friction = self._takeover_detector._torques_to_current(tau_friction)
                    expected_currents = curr_tracking + curr_gravity + curr_friction
                    diff_currents = curr_currents - expected_currents

                    # Get filtered current diff from takeover detector (if available)
                    filtered_diff = (
                        self._takeover_detector.filtered_current_diff
                        if self._takeover_detector.filtered_current_diff is not None
                        else diff_currents
                    )

                    # Log to structured system at 100 Hz
                    self._debug_logger.log_sample(
                        timestamp=loop_start,
                        measured_currents=curr_currents,
                        tracking_currents=curr_tracking,
                        gravity_currents=curr_gravity,
                        friction_currents=curr_friction,
                        expected_currents=expected_currents,
                        current_diff=diff_currents,
                        filtered_current_diff=filtered_diff,
                        position_error=pos_err,
                        velocities=np.abs(curr_velocities),
                        goal_velocity=self._goal_tracker.goal_velocity,
                        external_torques=tau_ext,
                        is_tracking=True,
                        goal_position=self._goal_tracker._goal_position,
                        current_position=curr_positions,
                        signed_position_error=self._goal_tracker._goal_position - curr_positions,
                        takeover_state=self._takeover_detector._last_takeover_state,
                    )

                # Console debug print every second (filtered)
                if self._debug_msg['enable'] and self._debug_msg['tracking'] and loop_count % 100 == 0:
                    pos_err = np.abs(self._goal_tracker._goal_position - curr_positions)
                    print(f"[DEBUG] Tracking mode. Pos err: {pos_err}, max: {np.max(pos_err):.3f}")
                    curr_tracking = self._takeover_detector._torques_to_current(tau_tracking_motor)
                    curr_gravity = self._takeover_detector._torques_to_current(tau_gravity)
                    curr_friction = self._takeover_detector._torques_to_current(tau_friction)
                    expected_currents = curr_tracking + curr_gravity + curr_friction
                    diff_currents = curr_currents - expected_currents
                    print(
                        "[DEBUG] Currents (mA): "
                        f"measured={curr_currents}, "
                        f"tracking={curr_tracking}, "
                        f"gravity={curr_gravity}, "
                        f"friction={curr_friction}, "
                        f"expected_total={expected_currents}, "
                        f"diff={diff_currents}"
                    )

                was_takeover = self._takeover_detector._last_takeover_state

                takeover = self._takeover_detector.detect_takeover(
                    self._goal_tracker._goal_position,
                    curr_positions,
                    curr_velocities,
                    curr_currents,
                    tau_tracking_motor,
                    tau_gravity,
                    tau_friction,
                    external_torques=tau_ext,
                )

                # In admittance mode (tracking stays on during takeover), use
                # release detection to confirm the human has actually let go
                # instead of immediately flipping state when current drops.
                # Only when enable_auto_return is on; otherwise let state flip immediately.
                if (not self._stop_tracking_on_takeover
                        and self._takeover_detector.config.enable_auto_return
                        and was_takeover and not takeover):
                    # Takeover was active but current dropped — check release criteria
                    release = self._takeover_detector.detect_release(curr_velocities, curr_currents)
                    if not release:
                        # Not confirmed released yet — hold takeover state
                        self._takeover_detector._last_takeover_state = True
                    else:
                        logger.debug("<<< TAKEOVER ENDED (release confirmed)")
                elif was_takeover and not self._takeover_detector._last_takeover_state:
                    # Non-admittance mode: takeover ended immediately
                    logger.debug("<<< TAKEOVER ENDED")

                if takeover:
                    if self._debug_msg['enable'] and self._debug_msg['takeover']:
                        if self._stop_tracking_on_takeover:
                            print("[BilateralController] Takeover detected! Stopping tracking.")
                        else:
                            print("[BilateralController] Takeover detected! (admittance mode: tracking continues)")
                    if self._takeover_callback is not None:
                        self._takeover_callback()
                    # Only stop tracking if configured to do so
                    # In admittance mode, we keep tracking so the physical blend continues
                    # (policy goal + human force deflection)
                    if self._stop_tracking_on_takeover:
                        self._goal_tracker.stop_tracking()

            # Auto-return detection
            else:
                # Structured debug logging for non-tracking mode (100 Hz when enabled)
                if self._debug_logger is not None:
                    pos_err = np.zeros(self._dof, dtype=np.float32)  # No tracking, no error
                    curr_tracking = np.zeros(self._dof, dtype=np.float32)  # No tracking torque
                    curr_gravity = self._takeover_detector._torques_to_current(tau_gravity)
                    curr_friction = np.zeros(self._dof, dtype=np.float32)  # No friction in GELLO_TO_ENV
                    expected_currents = curr_gravity
                    diff_currents = curr_currents - expected_currents

                    # Get filtered current diff from takeover detector (if available)
                    filtered_diff = (
                        self._takeover_detector.filtered_current_diff
                        if self._takeover_detector.filtered_current_diff is not None
                        else diff_currents
                    )

                    self._debug_logger.log_sample(
                        timestamp=loop_start,
                        measured_currents=curr_currents,
                        tracking_currents=curr_tracking,
                        gravity_currents=curr_gravity,
                        friction_currents=curr_friction,
                        expected_currents=expected_currents,
                        current_diff=diff_currents,
                        filtered_current_diff=filtered_diff,
                        position_error=pos_err,
                        velocities=np.abs(curr_velocities),
                        goal_velocity=self._goal_tracker.goal_velocity,
                        external_torques=tau_ext,
                        is_tracking=False,
                        goal_position=np.zeros(self._dof, dtype=np.float32),
                        current_position=curr_positions,
                        signed_position_error=np.zeros(self._dof, dtype=np.float32),
                        takeover_state=self._takeover_detector._last_takeover_state,
                    )

                # Console debug print every second (filtered)
                if self._debug_msg['enable'] and self._debug_msg['user_control'] and loop_count % 100 == 0:
                    print(f"[DEBUG] NOT tracking (GELLO_TO_ENV mode)")

                release = self._takeover_detector.detect_release(curr_velocities, curr_currents)
                if release:
                    # Resume tracking - set initial goal to current position to avoid immediate re-trigger
                    # Policy will update this goal on its next iteration
                    if self._debug_msg['enable'] and self._debug_msg['release']:
                        print(f"[BilateralController] Release detected! Resuming ENV_TO_GELLO mode.")
                    # Clear takeover buffers and add a grace period to avoid stale detections
                    self._takeover_detector.reset()
                    self._goal_tracker.set_goal(curr_positions)
                    self._goal_tracker.start_tracking()
                    # Notify mode manager to switch back to ENV_TO_GELLO
                    if self._auto_return_callback is not None:
                        self._auto_return_callback()

            if _timer:
                _timer.mark("detect")

            # Sum all torques in motor frame based on mode:
            # 1. Tracking (ENV_TO_GELLO/ADMITTANCE): gravity + friction + tracking (NO anti-friction)
            # 2. GELLO_TO_ENV with torque enabled: gravity + anti-friction (no tracking)
            # 3. GELLO_TO_ENV with torque disabled: no commands (free float)
            #
            # Note: Anti-friction is ONLY applied in GELLO_TO_ENV mode to assist backdriving.
            # In tracking modes, the PID Kd term provides damping - anti-friction would interfere.
            if self._goal_tracker.is_tracking:
                # ENV_TO_GELLO or ADMITTANCE mode: gravity + friction + tracking (no anti-friction)
                tau_total_motor = tau_gravity + tau_friction + tau_tracking_motor
                # Precompute encoder goal from GoalTracker for position-mode drivers
                goal_encoder = radians_to_absolute_encoder(
                    self._goal_tracker._goal_position,
                    signs=self._joint_signs,
                    offsets=self._joint_offsets,
                )
                # Send commands to each driver atomically (mode check + dispatch
                # under the same lock to prevent race with _switch_to_mode).
                for info in self._driver_infos:
                    driver = info.driver
                    driver_indices = info.robot_indices
                    driver.set_controls_for_tracking(
                        tau_total_motor[driver_indices],
                        goal_encoder[driver_indices],
                    )
            elif self._gello_to_env_torque_enabled:
                # GELLO_TO_ENV with torque enabled: gravity + anti-friction only
                tau_total_motor = tau_gravity + tau_anti_friction_motor
                for info in self._driver_infos:
                    driver = info.driver
                    driver_indices = info.robot_indices
                    driver_torques = tau_total_motor[driver_indices]
                    driver.set_controls(driver_torques)
            # else: GELLO_TO_ENV with torque disabled - no commands (free float)

            if _timer:
                _timer.mark("write")

            # Control loop frequency debug (filtered)
            if self._debug_msg['enable'] and self._debug_msg['control_loop']:
                loop_count += 1
                if time.time() - last_print_time >= 1.0:
                    freq = loop_count / (time.time() - last_print_time)
                    print(f"Control loop frequency: {freq:.2f} Hz")
                    loop_count = 0
                    last_print_time = time.time()
            else:
                loop_count += 1

            if _timer:
                _timer.end_iteration()

            # Sleep to maintain control frequency
            loop_elapsed = time.time() - loop_start
            sleep_time = self._control_period - loop_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _read_all_motor_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read position, velocity, and current from all motor groups in ONE USB call.

        This is the most efficient way to read motor state - uses combined CVP read
        with Fast Sync Read (single packet for all data from all motors).

        Returns:
            Tuple of (raw_positions, velocities_rad_s, currents_mA, valid_joints)
            - raw_positions: encoder tick positions (not converted to radians)
            - velocities_rad_s: velocities in rad/s (motor frame, no sign conversion)
            - currents_mA: currents in mA
            - valid_joints: boolean mask of valid joints
        """
        raw_positions = np.zeros(self._dof, dtype=np.float32)
        velocities = np.zeros(self._dof, dtype=np.float32)
        currents = np.zeros(self._dof, dtype=np.float32)
        valid_joints = np.zeros(self._dof, dtype=bool)

        for info in self._driver_infos:
            driver = info.driver
            indices = info.robot_indices

            # Single USB call for all 3 values (uses combined CVP + Fast Sync Read)
            motor_state = driver.get_motor_state()

            # Raw encoder positions (for later conversion)
            raw_positions[indices] = motor_state.joint_angles

            # Convert velocities from motor units to rad/s (0.229 rev/min)
            motor_velocities = motor_state.joint_velocities * 0.229 * 2 * np.pi / 60
            velocities[indices] = motor_velocities

            # Currents in mA
            currents[indices] = motor_state.joint_currents

            valid_joints[indices] = True

        return raw_positions, velocities, currents, valid_joints

    def _read_joint_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read raw encoder positions from all motor groups.

        Note: For efficiency, prefer using _read_all_motor_state() in the control loop
        to read position, velocity, and current in one call.
        """
        raw_positions, _, _, valid_joints = self._read_all_motor_state()
        return raw_positions, valid_joints

    def _read_joint_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read joint velocities from all motor groups in robot frame.

        Note: For efficiency, prefer using _read_all_motor_state() in the control loop.
        """
        _, velocities, _, valid_joints = self._read_all_motor_state()
        # Apply sign conversion to match position frame (robot frame)
        robot_velocities = velocities * self._joint_signs
        return robot_velocities, valid_joints

    def _read_joint_currents(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read joint currents from all motor groups.

        Note: For efficiency, prefer using _read_all_motor_state() in the control loop.
        """
        _, _, currents, valid_joints = self._read_all_motor_state()
        return currents, valid_joints

    def _read_q(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read joint positions from hardware (blocking USB read)."""
        encoder_positions, valid_joints = self._read_joint_positions()

        # Convert from encoder to radians using absolute encoder utils
        joint_angles = absolute_encoder_to_radians(
            encoder_positions,
            signs=self._joint_signs,
            offsets=self._joint_offsets
        )

        # Clip to limits
        joint_angles = clip_to_limits(
            joint_angles,
            self.JOINT_LIMIT_LOW,
            self.JOINT_LIMIT_HIGH
        )

        return joint_angles, valid_joints

    def read_state(self, use_cache: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get position, velocity, and current state.

        By default, returns cached values from the control loop (no USB read).
        Set use_cache=False to force a fresh USB read.

        Args:
            use_cache: If True (default), return cached values from control loop.
                      If False, do a fresh USB read.

        Returns:
            Tuple of (positions, velocities, currents, valid_joints)
            - positions: joint positions in radians (robot frame, sign-converted)
            - velocities: joint velocities in rad/s (robot frame, sign-converted)
            - currents: joint currents in mA
            - valid_joints: boolean mask of valid joints
        """
        # Use cached values if control loop is running and cache requested
        if use_cache and self._control_thread is not None and self._control_thread.is_alive():
            with self._cache_lock:
                return (
                    self._cached_positions.copy(),
                    self._cached_velocities.copy(),
                    self._cached_currents.copy(),
                    self._cached_valid.copy(),
                )

        # Fresh USB read
        raw_positions, motor_velocities, currents, valid_joints = self._read_all_motor_state()

        positions = absolute_encoder_to_radians(
            raw_positions,
            signs=self._joint_signs,
            offsets=self._joint_offsets
        )
        if self._use_angle_wrapping:
            positions = wrap_to_joint_centers(positions, self._joint_centers)
        positions = clip_to_limits(
            positions,
            self.JOINT_LIMIT_LOW,
            self.JOINT_LIMIT_HIGH
        )
        velocities = motor_velocities * self._joint_signs

        return positions, velocities, currents, valid_joints

    @property
    def q(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint positions in radians (cached from control loop).

        For fresh reads, use read_state() instead.
        """
        if self._control_thread.is_alive():
            with self._cache_lock:
                return self._cached_positions.copy(), self._cached_valid.copy()
        return self._read_q()

    @property
    def q_urdf_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get joint positions in URDF frame (without sign conversion).

        This is needed for gravity compensation, which expects positions
        in the URDF's native frame, not sign-converted positions.

        Returns:
            Tuple of (positions_in_urdf_frame, valid_joints)
        """
        encoder_positions, valid_joints = self._read_joint_positions()

        # Convert to radians WITHOUT sign conversion - just encoder to radians + offset
        # URDF frame uses: (encoder - 2048) * π/2048 + offset
        joint_angles = absolute_encoder_to_radians(
            encoder_positions,
            signs=np.ones(self._dof, dtype=np.float32),  # No sign flip
            offsets=self._joint_offsets
        )

        return joint_angles, valid_joints

    def _read_v(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read joint velocities from hardware (blocking USB read)."""
        return self._read_joint_velocities()

    @property
    def v(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint velocities in rad/s (cached from control loop).

        For fresh reads, use read_state() instead.
        """
        if self._control_thread.is_alive():
            with self._cache_lock:
                return self._cached_velocities.copy(), self._cached_valid.copy()
        return self._read_v()

    @property
    def current(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint currents in mA (cached from control loop).

        For fresh reads, use read_state() instead.
        """
        if self._control_thread.is_alive():
            with self._cache_lock:
                return self._cached_currents.copy(), self._cached_valid.copy()
        return self._read_joint_currents()

    @property
    def motor_groups(self) -> Sequence[GelloMotorGroup]:
        """Get motor groups."""
        return self._motor_groups

    @property
    def dof(self) -> int:
        """
        Get total degrees of freedom (DOF) for this controller.

        Returns:
            int: Total number of joints controlled by this controller.
                 For bilateral setups, this is the total DOF across all arms.

        Examples:
            - A1 single arm (6 DOF): returns 6
            - R1Pro single arm (7 DOF): returns 7
            - R1Pro bilateral (2 arms, 7 DOF each): returns 14
        """
        return self._dof

    @property
    def position_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current trajectory position (interpolated goal).

        Returns interpolated goal position when tracking, otherwise current position.
        """
        with self._goal_lock:
            if self._trajectory_state is not None and self._should_follow_goal:
                traj_pos = interpolate_trajectory(self._trajectory_state)
                valid_joints = np.ones(self._dof, dtype=bool)
                return traj_pos, valid_joints
            else:
                return self.q

    @property
    def velocity_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trajectory velocity.

        For current-based control, this is not directly controlled, returns zeros.
        """
        return np.zeros(self._dof, dtype=np.float32), np.ones(self._dof, dtype=bool)

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_stop_event'):
            self.close()

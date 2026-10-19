"""
Unified JoyCon interface module.

Provides robot-agnostic JoyCon handling with:
- BaseJoyConInterface: Abstract base with common connection logic
- A1JoyConInterface: Right JoyCon only (gripper + mode switching)
- R1ProJoyConInterface: Both JoyCons (base + torso + grippers)

Factory function get_joycon_interface() returns a singleton shared across
InputHandler and LocomotionHandler.

Includes heartbeat/disconnection detection via failure tracking in _safe_get(),
with automatic reconnection in a background thread.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from enum import Enum
from threading import Thread, Lock, Event
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


class JoyConHealthStatus(Enum):
    """Joy-Con connection health status."""
    HEALTHY = "healthy"          # Connected and responsive
    DEGRADED = "degraded"        # Some failures but below threshold
    DISCONNECTED = "disconnected"  # Exceeded failure threshold
    RECONNECTING = "reconnecting"  # Currently attempting reconnection


class BaseJoyConInterface(ABC):
    """
    Abstract base class for JoyCon interfaces.

    Provides common connection logic and safe getter methods.
    Subclasses implement robot-specific behavior.
    """

    def __init__(
        self,
        required: bool = True,
        failure_threshold: int = 5,
        reconnect_interval: float = 2.0,
        max_reconnect_attempts: int = 10,
    ):
        """
        Initialize JoyCon interface.

        Args:
            required: If True, raise error when JoyCons not found.
                     If False, gracefully disable functionality.
            failure_threshold: Consecutive read failures to trigger disconnect.
            reconnect_interval: Seconds between reconnection attempts.
            max_reconnect_attempts: Max reconnection attempts (0 = infinite).
        """
        self.jc_left: Optional[Any] = None
        self.jc_right: Optional[Any] = None
        self.connected = False
        self._required = required

        # Health tracking configuration
        self._failure_threshold = failure_threshold
        self._reconnect_interval = reconnect_interval
        self._max_reconnect_attempts = max_reconnect_attempts

        # Health tracking state
        self._health_lock = Lock()
        self._health_status: Dict[str, Dict[str, Any]] = {}
        self._reconnect_thread: Optional[Thread] = None
        self._reconnect_stop_event = Event()

    def _connect_right(self) -> bool:
        """
        Connect to Right JoyCon.

        Returns:
            True if connection successful, False otherwise.

        Raises:
            RuntimeError: If required=True and connection fails.
        """
        try:
            from pyjoycon import JoyCon, get_R_id
            jc_id = get_R_id()
            if jc_id[0] is None:
                if self._required:
                    raise RuntimeError("Right JoyCon not found!")
                logger.warning("Right JoyCon not found - some features disabled")
                return False
            self.jc_right = JoyCon(*jc_id)
            return True
        except ImportError as e:
            if self._required:
                raise RuntimeError("pyjoycon is required but could not be imported") from e
            logger.warning("pyjoycon not available - JoyCon disabled")
            return False
        except Exception as e:
            if self._required:
                raise RuntimeError(f"Failed to connect to Right JoyCon: {e}") from e
            logger.warning(f"Right JoyCon connection failed: {e}")
            return False

    def _connect_left(self) -> bool:
        """
        Connect to Left JoyCon.

        Returns:
            True if connection successful, False otherwise.

        Raises:
            RuntimeError: If required=True and connection fails.
        """
        try:
            from pyjoycon import JoyCon, get_L_id
            jc_id = get_L_id()
            if jc_id[0] is None:
                if self._required:
                    raise RuntimeError("Left JoyCon not found!")
                logger.warning("Left JoyCon not found - some features disabled")
                return False
            self.jc_left = JoyCon(*jc_id)
            return True
        except ImportError as e:
            if self._required:
                raise RuntimeError("pyjoycon is required but could not be imported") from e
            logger.warning("pyjoycon not available - JoyCon disabled")
            return False
        except Exception as e:
            if self._required:
                raise RuntimeError(f"Failed to connect to Left JoyCon: {e}") from e
            logger.warning(f"Left JoyCon connection failed: {e}")
            return False

    def _safe_get(self, joycon, method_name: str, default=False, joycon_id: str = "right"):
        """
        Safely call a JoyCon method with failure tracking.

        Args:
            joycon: JoyCon instance (may be None)
            method_name: Method to call (e.g., "get_button_x")
            default: Value to return if call fails
            joycon_id: Identifier for health tracking ("left" or "right")

        Returns:
            Method result or default value
        """
        if joycon is None:
            return default

        # Skip reads for disconnected or reconnecting Joy-Cons
        with self._health_lock:
            health = self._health_status.get(joycon_id, {})
            status = health.get("status")
            if status in (JoyConHealthStatus.DISCONNECTED, JoyConHealthStatus.RECONNECTING):
                return default

        try:
            result = getattr(joycon, method_name)()
            self._record_success(joycon_id)
            return result
        except Exception as e:
            self._record_failure(joycon_id, e)
            return default

    def _record_success(self, joycon_id: str) -> None:
        """Record successful read, resetting failure counter."""
        with self._health_lock:
            if joycon_id not in self._health_status:
                self._health_status[joycon_id] = {
                    "status": JoyConHealthStatus.HEALTHY,
                    "failures": 0,
                    "reconnect_attempts": 0,
                }

            health = self._health_status[joycon_id]
            health["failures"] = 0

            # Restore healthy status if was unhealthy
            if health["status"] != JoyConHealthStatus.HEALTHY:
                was_disconnected = health["status"] in (
                    JoyConHealthStatus.DISCONNECTED,
                    JoyConHealthStatus.RECONNECTING,
                )
                health["status"] = JoyConHealthStatus.HEALTHY
                health["reconnect_attempts"] = 0
                if was_disconnected:
                    logger.info(f"Joy-Con {joycon_id} connection restored")

    def _record_failure(self, joycon_id: str, error: Exception) -> None:
        """Record failed read and check if threshold exceeded."""
        with self._health_lock:
            if joycon_id not in self._health_status:
                self._health_status[joycon_id] = {
                    "status": JoyConHealthStatus.HEALTHY,
                    "failures": 0,
                    "reconnect_attempts": 0,
                }

            health = self._health_status[joycon_id]
            health["failures"] += 1
            failures = health["failures"]

            # Check if we've exceeded the threshold
            if failures >= self._failure_threshold:
                # Only log and trigger reconnect if not already disconnected/reconnecting
                if health["status"] not in (
                    JoyConHealthStatus.DISCONNECTED,
                    JoyConHealthStatus.RECONNECTING,
                ):
                    logger.warning(
                        f"Joy-Con {joycon_id} disconnected after "
                        f"{failures} consecutive failures: {error}"
                    )
                    health["status"] = JoyConHealthStatus.DISCONNECTED
                    self._trigger_reconnect(joycon_id)
            elif failures > 1:
                health["status"] = JoyConHealthStatus.DEGRADED

    def _trigger_reconnect(self, joycon_id: str) -> None:
        """Start background reconnection thread if not already running."""
        # Only start one reconnect thread at a time - it will handle all disconnected Joy-Cons
        if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
            return

        self._reconnect_stop_event.clear()
        self._reconnect_thread = Thread(
            target=self._reconnect_loop,
            daemon=True,
            name="JoyCon-Reconnect",
        )
        self._reconnect_thread.start()

    def _reconnect_loop(self) -> None:
        """Background loop attempting to reconnect all disconnected Joy-Cons."""
        while not self._reconnect_stop_event.is_set():
            # Find all Joy-Cons that need reconnection
            joycons_to_reconnect = []
            with self._health_lock:
                for jc_id, health in self._health_status.items():
                    status = health.get("status")
                    if status in (JoyConHealthStatus.DISCONNECTED, JoyConHealthStatus.RECONNECTING):
                        # Check if max attempts exceeded
                        if (
                            self._max_reconnect_attempts > 0
                            and health["reconnect_attempts"] >= self._max_reconnect_attempts
                        ):
                            continue  # Skip this one, max attempts reached
                        joycons_to_reconnect.append(jc_id)

            # If nothing to reconnect, exit the loop
            if not joycons_to_reconnect:
                return

            # Attempt reconnection for each disconnected Joy-Con
            for jc_id in joycons_to_reconnect:
                with self._health_lock:
                    health = self._health_status.get(jc_id)
                    if health is None:
                        continue
                    health["status"] = JoyConHealthStatus.RECONNECTING
                    health["reconnect_attempts"] += 1
                    attempt = health["reconnect_attempts"]

                logger.info(f"Attempting Joy-Con {jc_id} reconnection (attempt {attempt})")

                try:
                    success = self._attempt_reconnect(jc_id)
                    if success:
                        with self._health_lock:
                            health = self._health_status.get(jc_id)
                            if health:
                                health["status"] = JoyConHealthStatus.HEALTHY
                                health["failures"] = 0
                                health["reconnect_attempts"] = 0
                        logger.info(f"Joy-Con {jc_id} reconnected successfully")
                except Exception as e:
                    logger.debug(f"Reconnect attempt for {jc_id} failed: {e}")

            # Wait before next attempt
            self._reconnect_stop_event.wait(self._reconnect_interval)

    def _attempt_reconnect(self, joycon_id: str) -> bool:
        """
        Attempt to reconnect a specific Joy-Con.

        Returns:
            True if reconnection successful, False otherwise.
        """
        try:
            from pyjoycon import JoyCon, get_R_id, get_L_id

            if joycon_id == "right":
                jc_id = get_R_id()
                if jc_id[0] is None:
                    return False
                self.jc_right = JoyCon(*jc_id)
                # Verify connection with a test read
                self.jc_right.get_status()
                return True
            elif joycon_id == "left":
                jc_id = get_L_id()
                if jc_id[0] is None:
                    return False
                self.jc_left = JoyCon(*jc_id)
                self.jc_left.get_status()
                return True
        except Exception:
            pass
        return False

    def _get_required_joycons(self) -> List[str]:
        """Return list of required Joy-Con IDs. Override in subclasses."""
        return ["right"]  # Default for A1

    @property
    def is_healthy(self) -> bool:
        """Whether all required Joy-Cons are responsive."""
        if not self.connected:
            return False
        required = self._get_required_joycons()
        with self._health_lock:
            for jc_id in required:
                health = self._health_status.get(jc_id, {})
                status = health.get("status", JoyConHealthStatus.HEALTHY)
                if status != JoyConHealthStatus.HEALTHY:
                    return False
        return True

    @property
    def is_reconnecting(self) -> bool:
        """Whether reconnection is in progress."""
        with self._health_lock:
            for health in self._health_status.values():
                if health.get("status") == JoyConHealthStatus.RECONNECTING:
                    return True
        return False

    def close(self):
        """Clean up JoyCon connections and stop reconnection thread."""
        # Stop reconnection thread
        self._reconnect_stop_event.set()
        if self._reconnect_thread is not None:
            self._reconnect_thread.join(timeout=2.0)
            self._reconnect_thread = None

        if self.jc_left is not None:
            try:
                del self.jc_left
            except Exception:
                pass
        if self.jc_right is not None:
            try:
                del self.jc_right
            except Exception:
                pass
        self.jc_left = None
        self.jc_right = None
        self.connected = False

    def __del__(self):
        self.close()

    @abstractmethod
    def get_gripper_command(self) -> Dict[str, float]:
        """
        Get gripper command(s) from JoyCon input.

        Returns:
            Dict with gripper values. Keys depend on robot type.
        """
        pass


class A1JoyConInterface(BaseJoyConInterface):
    """
    JoyCon interface for A1 robot.

    Uses Right JoyCon only for:
    - Gripper control (trigger_r = close, trigger_zr = open)
    - Mode switching (button_x)
    """

    def __init__(self, required: bool = True):
        super().__init__(required)
        self.connected = self._connect_right()

        if self.connected:
            # Check battery level
            try:
                battery = self.jc_right.get_battery_level()
                if battery <= 1:
                    logger.warning(f"Right JoyCon battery is low: {battery}")
            except Exception:
                pass
            logger.info("A1 JoyCon interface initialized (Right JoyCon connected)")

    def _get_required_joycons(self) -> List[str]:
        """A1 only requires right Joy-Con."""
        return ["right"]

    # Button properties for mode switching
    @property
    def button_x(self) -> bool:
        """X button state (start trial)."""
        return self._safe_get(self.jc_right, "get_button_x", joycon_id="right")

    @property
    def button_y(self) -> bool:
        """Y button state (reject/discard trial)."""
        return self._safe_get(self.jc_right, "get_button_y", joycon_id="right")

    @property
    def button_a(self) -> bool:
        """A button state (accept/save trial)."""
        return self._safe_get(self.jc_right, "get_button_a", joycon_id="right")

    @property
    def button_b(self) -> bool:
        """B button state."""
        return self._safe_get(self.jc_right, "get_button_b", joycon_id="right")

    @property
    def button_plus(self) -> bool:
        """Plus button state (end data collection block)."""
        return self._safe_get(self.jc_right, "get_button_plus", joycon_id="right")

    @property
    def button_home(self) -> bool:
        """Home button state (reset to home position)."""
        return self._safe_get(self.jc_right, "get_button_home", joycon_id="right")

    @property
    def button_minus(self) -> bool:
        """Minus button state (mode switch). Always False for A1 (Right JoyCon only)."""
        return False  # A1 only connects to Right JoyCon, Minus is on Left

    @property
    def trigger_r(self) -> bool:
        """R trigger button state (gripper close)."""
        return self._safe_get(self.jc_right, "get_button_r", joycon_id="right")

    @property
    def trigger_zr(self) -> bool:
        """ZR trigger button state (gripper open)."""
        return self._safe_get(self.jc_right, "get_button_zr", joycon_id="right")

    def get_gripper_command(self) -> Dict[str, float]:
        """
        Get single gripper command from triggers.

        Returns:
            {"gripper": value} where value is:
            - 1.0 if trigger_r pressed (close)
            - -1.0 if trigger_zr pressed (open)
            - 0.0 if neither pressed (no change)
        """
        if not self.connected:
            return {"gripper": 0.0}

        if self.trigger_r:
            return {"gripper": 1.0}
        elif self.trigger_zr:
            return {"gripper": -1.0}
        return {"gripper": 0.0}


class R1ProJoyConInterface(BaseJoyConInterface):
    """
    JoyCon interface for R1Pro robot.

    Uses both Left and Right JoyCons for:
    - Base control: Left joystick (XY translation or yaw rotation when pressed)
    - Torso control: Right joystick (pitch/yaw) + D-pad (stand/squat presets)
    - Grippers: ZL (left gripper), ZR (right gripper)

    Matches brs-ctrl's R1ProJoyConInterface functionality.
    """

    # R1Pro torso joint limits (from brs-ctrl)
    TORSO_JOINT_HIGH = np.array([1.8326, 2.5307, 1.5708, 3.0543])
    TORSO_JOINT_LOW = np.array([-1.1345, -2.7925, -1.8326, -3.0543])

    def __init__(
        self,
        *,
        calibration_file: Optional[str] = None,
        # Mobile base parameters
        mobile_base_x_move_max: float = 0.3,
        mobile_base_y_move_max: float = 0.3,
        mobile_base_yaw_rotate_max: float = 0.4,
        # Torso parameters
        torso_joint_max_delta: float = 0.1,
        torso_joints1_2_stand_q: np.ndarray = None,
        torso_joints1_2_squat_q: np.ndarray = None,
        # Gripper parameters
        gripper_toggle_mode: bool = True,
        # Common parameters
        joystick_ema_alpha: float = 0.9,
        joystick_idle_percentage: float = 0.2,
        # Whether to require both JoyCons
        required: bool = True,
    ):
        """
        Initialize R1Pro JoyCon interface.

        Args:
            calibration_file: Path to joystick calibration YAML
            mobile_base_x_move_max: Max base X velocity (m/s)
            mobile_base_y_move_max: Max base Y velocity (m/s)
            mobile_base_yaw_rotate_max: Max base yaw rate (rad/s)
            torso_joint_max_delta: Max torso joint delta per step (rad)
            torso_joints1_2_stand_q: Torso joints 0,1 for standing pose
            torso_joints1_2_squat_q: Torso joints 0,1 for squatting pose
            gripper_toggle_mode: If True, gripper toggles on button press
            joystick_ema_alpha: EMA smoothing factor (0-1)
            joystick_idle_percentage: Joystick deadzone as percentage
            required: If True, raise error when JoyCons not found
        """
        super().__init__(required)

        # Set defaults
        if torso_joints1_2_stand_q is None:
            torso_joints1_2_stand_q = np.array([0.0, 0.0])
        if torso_joints1_2_squat_q is None:
            torso_joints1_2_squat_q = np.array([1.74, -2.70])

        self._torso_joints1_2_stand_q = torso_joints1_2_stand_q
        self._torso_joints1_2_squat_q = torso_joints1_2_squat_q

        # Store parameters
        self._mobile_base_x_move_max = mobile_base_x_move_max
        self._mobile_base_y_move_max = mobile_base_y_move_max
        self._mobile_base_yaw_rotate_max = mobile_base_yaw_rotate_max
        self._torso_joint_max_delta = torso_joint_max_delta
        self._gripper_toggle_mode = gripper_toggle_mode
        self._joystick_ema_alpha = joystick_ema_alpha

        # Load calibration
        self.calibration_data = self._load_calibration(calibration_file, joystick_idle_percentage)

        # EMA state
        self._lh_unpressed_ema = None
        self._lh_pressed_ema = None
        self._lv_ema = None
        self._rh_ema = None
        self._rv_ema = None

        # Gripper toggle state
        self._left_gripper_current_action = 0.1
        self._right_gripper_current_action = 0.1
        self._left_gripper_prev_pressed = False
        self._right_gripper_prev_pressed = False

        # Connect to both JoyCons
        left_ok = self._connect_left()
        right_ok = self._connect_right()
        self.connected = left_ok and right_ok

        if self.connected:
            # Check battery levels
            try:
                jc_l_battery = self.jc_left.get_battery_level()
                jc_r_battery = self.jc_right.get_battery_level()
                if jc_l_battery <= 1:
                    logger.warning(f"Left JoyCon battery is low: {jc_l_battery}")
                if jc_r_battery <= 1:
                    logger.warning(f"Right JoyCon battery is low: {jc_r_battery}")
            except Exception:
                pass
            logger.info("R1Pro JoyCon interface initialized (both JoyCons connected)")

    def _get_required_joycons(self) -> List[str]:
        """R1Pro requires both left and right Joy-Cons."""
        return ["left", "right"]

    def _load_calibration(self, calibration_file: Optional[str], idle_percentage: float) -> dict:
        """Load joystick calibration or use defaults."""
        # Default calibration (center at ~2048, range 0-4095)
        default_cal = {
            "left": {
                "horizontal": {"limits": [0, 4095], "center_range": [1800, 2300]},
                "vertical": {"limits": [0, 4095], "center_range": [1800, 2300]},
            },
            "right": {
                "horizontal": {"limits": [0, 4095], "center_range": [1800, 2300]},
                "vertical": {"limits": [0, 4095], "center_range": [1800, 2300]},
            },
        }

        if calibration_file and os.path.exists(calibration_file):
            try:
                import yaml
                with open(calibration_file, "r") as f:
                    data = yaml.safe_load(f)
                    if "joystick" in data:
                        default_cal = data["joystick"]
                    logger.info(f"Loaded JoyCon calibration from {calibration_file}")
            except Exception as e:
                logger.warning(f"Failed to load calibration file: {e}, using defaults")

        # Expand center range based on idle percentage
        for side in ["left", "right"]:
            for axis in ["horizontal", "vertical"]:
                full_range = (
                    default_cal[side][axis]["limits"][1] -
                    default_cal[side][axis]["limits"][0]
                )
                expand = full_range * idle_percentage / 2
                default_cal[side][axis]["center_range"][0] -= expand
                default_cal[side][axis]["center_range"][1] += expand

        return default_cal

    # Button properties for data collection control
    @property
    def button_x(self) -> bool:
        """X button state (start trial)."""
        return self._safe_get(self.jc_right, "get_button_x", joycon_id="right")

    @property
    def button_y(self) -> bool:
        """Y button state (reject/discard trial)."""
        return self._safe_get(self.jc_right, "get_button_y", joycon_id="right")

    @property
    def button_a(self) -> bool:
        """A button state (accept/save trial)."""
        return self._safe_get(self.jc_right, "get_button_a", joycon_id="right")

    @property
    def button_b(self) -> bool:
        """B button state."""
        return self._safe_get(self.jc_right, "get_button_b", joycon_id="right")

    @property
    def button_plus(self) -> bool:
        """Plus button state (end data collection block)."""
        return self._safe_get(self.jc_right, "get_button_plus", joycon_id="right")

    @property
    def button_home(self) -> bool:
        """Home button state (reset to home position)."""
        return self._safe_get(self.jc_right, "get_button_home", joycon_id="right")

    @property
    def button_minus(self) -> bool:
        """Minus button state (mode switch GELLO↔ENV)."""
        return self._safe_get(self.jc_left, "get_button_minus", joycon_id="left")

    @property
    def trigger_r(self) -> bool:
        """R trigger button state."""
        return self._safe_get(self.jc_right, "get_button_r", joycon_id="right")

    @property
    def trigger_zr(self) -> bool:
        """ZR trigger button state (right gripper)."""
        return self._safe_get(self.jc_right, "get_button_zr", joycon_id="right")

    def act(self, curr_torso_q: np.ndarray) -> Dict[str, Any]:
        """
        Get control commands from JoyCon input.

        Mapping strategy (matches brs-ctrl exactly):
        - Mobile base: Left joystick unpressed -> XY translation
                      Left joystick pressed -> yaw rotation
        - Torso: Right joystick vertical -> forward/backward (pitch)
                Right joystick horizontal -> yaw rotation
                Left D-pad up -> stand preset
                Left D-pad down -> squat preset
        - Gripper: ZL -> left gripper, ZR -> right gripper

        Args:
            curr_torso_q: Current torso joint positions (4,)

        Returns:
            Dict with mobile_base_cmd, torso_cmd, gripper_cmd
        """
        if not self.connected:
            return {
                "mobile_base_cmd": np.zeros(3),
                "torso_controller": "joint_position",
                "torso_cmd": curr_torso_q.copy() if curr_torso_q is not None else np.zeros(4),
                "gripper_cmd": {"left": 0.1, "right": 0.1},
            }

        assert curr_torso_q.shape == self.TORSO_JOINT_HIGH.shape, \
            f"Expected torso shape {self.TORSO_JOINT_HIGH.shape}, got {curr_torso_q.shape}"

        # Read joystick values
        is_left_joystick_pressed = self._safe_get(self.jc_left, "get_button_l_stick", joycon_id="left")
        raw_lh = self._safe_get(self.jc_left, "get_stick_left_horizontal", 2048, joycon_id="left")
        raw_lv = self._safe_get(self.jc_left, "get_stick_left_vertical", 2048, joycon_id="left")

        lh_neutral = 0.5 * sum(self.calibration_data["left"]["horizontal"]["center_range"])
        lv_neutral = 0.5 * sum(self.calibration_data["left"]["vertical"]["center_range"])

        # Apply EMA smoothing based on joystick press state
        if is_left_joystick_pressed:
            self._lh_pressed_ema = (
                raw_lh if self._lh_pressed_ema is None
                else self._joystick_ema_alpha * raw_lh + (1 - self._joystick_ema_alpha) * self._lh_pressed_ema
            )
            self._lh_unpressed_ema = (
                lh_neutral if self._lh_unpressed_ema is None
                else self._joystick_ema_alpha * lh_neutral + (1 - self._joystick_ema_alpha) * self._lh_unpressed_ema
            )
            self._lv_ema = (
                lv_neutral if self._lv_ema is None
                else self._joystick_ema_alpha * lv_neutral + (1 - self._joystick_ema_alpha) * self._lv_ema
            )
        else:
            self._lh_pressed_ema = (
                lh_neutral if self._lh_pressed_ema is None
                else self._joystick_ema_alpha * lh_neutral + (1 - self._joystick_ema_alpha) * self._lh_pressed_ema
            )
            self._lh_unpressed_ema = (
                raw_lh if self._lh_unpressed_ema is None
                else self._joystick_ema_alpha * raw_lh + (1 - self._joystick_ema_alpha) * self._lh_unpressed_ema
            )
            self._lv_ema = (
                raw_lv if self._lv_ema is None
                else self._joystick_ema_alpha * raw_lv + (1 - self._joystick_ema_alpha) * self._lv_ema
            )

        # Right joystick
        raw_rh = self._safe_get(self.jc_right, "get_stick_right_horizontal", 2048, joycon_id="right")
        raw_rv = self._safe_get(self.jc_right, "get_stick_right_vertical", 2048, joycon_id="right")

        self._rh_ema = (
            raw_rh if self._rh_ema is None
            else self._joystick_ema_alpha * raw_rh + (1 - self._joystick_ema_alpha) * self._rh_ema
        )
        self._rv_ema = (
            raw_rv if self._rv_ema is None
            else self._joystick_ema_alpha * raw_rv + (1 - self._joystick_ema_alpha) * self._rv_ema
        )

        # Process mobile base movement
        base_displacement = self._process_base_command(is_left_joystick_pressed)

        # Process torso movement
        torso_cmd = self._process_torso_command(curr_torso_q)

        # Process gripper
        gripper_cmd = self._process_gripper_command()

        return {
            "mobile_base_cmd": np.array(base_displacement),
            "torso_controller": "joint_position",
            "torso_cmd": torso_cmd,
            "gripper_cmd": gripper_cmd,
        }

    def _process_base_command(self, is_pressed: bool) -> list:
        """Process mobile base command from left joystick."""
        lh_neutral_range = self.calibration_data["left"]["horizontal"]["center_range"]
        lh_limits = self.calibration_data["left"]["horizontal"]["limits"]
        lv_neutral_range = self.calibration_data["left"]["vertical"]["center_range"]
        lv_limits = self.calibration_data["left"]["vertical"]["limits"]

        if is_pressed:
            # Yaw rotation only
            lh = self._lh_pressed_ema
            if lh_neutral_range[0] <= lh <= lh_neutral_range[1]:
                return [0, 0, 0]

            lh = np.clip(lh, lh_limits[0], lh_limits[1])
            delta_lh = lh - 0.5 * sum(lh_neutral_range)
            delta_lh *= -1  # Horizontal direction is reversed
            delta_lh = np.sign(delta_lh) * np.clip(
                abs(delta_lh), a_max=(lh_limits[1] - lh_limits[0]) / 2, a_min=0
            )
            return [
                0,
                0,
                2.0 * delta_lh / (lh_limits[1] - lh_limits[0]) * self._mobile_base_yaw_rotate_max,
            ]
        else:
            # XY translation
            lh = self._lh_unpressed_ema
            lv = self._lv_ema
            base_displacement = [0, 0, 0]

            if lh > lh_neutral_range[1] or lh < lh_neutral_range[0]:
                lh = np.clip(lh, lh_limits[0], lh_limits[1])
                delta_lh = lh - 0.5 * sum(lh_neutral_range)
                delta_lh *= -1  # Horizontal direction is reversed
                delta_lh = np.sign(delta_lh) * np.clip(
                    abs(delta_lh), a_max=(lh_limits[1] - lh_limits[0]) / 2, a_min=0
                )
                base_displacement[1] = (
                    2.0 * delta_lh / (lh_limits[1] - lh_limits[0]) * self._mobile_base_x_move_max
                )

            if lv > lv_neutral_range[1] or lv < lv_neutral_range[0]:
                lv = np.clip(lv, lv_limits[0], lv_limits[1])
                delta_lv = lv - 0.5 * sum(lv_neutral_range)
                delta_lv = np.sign(delta_lv) * np.clip(
                    abs(delta_lv), a_max=(lv_limits[1] - lv_limits[0]) / 2, a_min=0
                )
                base_displacement[0] = (
                    2.0 * delta_lv / (lv_limits[1] - lv_limits[0]) * self._mobile_base_y_move_max
                )

            return base_displacement

    def _process_torso_command(self, curr_torso_q: np.ndarray) -> np.ndarray:
        """Process torso command from right joystick and D-pad."""
        rh_neutral_range = self.calibration_data["right"]["horizontal"]["center_range"]
        rh_limits = self.calibration_data["right"]["horizontal"]["limits"]
        rv_neutral_range = self.calibration_data["right"]["vertical"]["center_range"]
        rv_limits = self.calibration_data["right"]["vertical"]["limits"]

        rh = self._rh_ema
        rv = self._rv_ema

        torso_cmd = curr_torso_q.copy()

        # Right joystick horizontal -> torso yaw (joint 3)
        if rh > rh_neutral_range[1] or rh < rh_neutral_range[0]:
            rh = np.clip(rh, rh_limits[0], rh_limits[1])
            delta_rh = rh - 0.5 * sum(rh_neutral_range)
            delta_rh *= -1  # Horizontal direction is reversed
            delta_rh = np.sign(delta_rh) * np.clip(
                abs(delta_rh), a_max=(rh_limits[1] - rh_limits[0]) / 2, a_min=0
            )
            torso_yaw_rotation = (
                2.0 * delta_rh / (rh_limits[1] - rh_limits[0]) * self._torso_joint_max_delta
            )
            torso_cmd[3] += torso_yaw_rotation

        # Right joystick vertical -> torso pitch (joint 2)
        if rv > rv_neutral_range[1] or rv < rv_neutral_range[0]:
            rv = np.clip(rv, rv_limits[0], rv_limits[1])
            delta_rv = rv - 0.5 * sum(rv_neutral_range)
            delta_rv = np.sign(delta_rv) * np.clip(
                abs(delta_rv), a_max=(rv_limits[1] - rv_limits[0]) / 2, a_min=0
            )
            delta_rv *= -1  # Vertical direction is reversed
            # Asymmetric speed: faster forward (joystick down), slower backward (up)
            multiplier = 1.5 if delta_rv > 0 else 2.75
            torso_cmd[2] += (
                multiplier * delta_rv / (rv_limits[1] - rv_limits[0]) * self._torso_joint_max_delta
            )

        # D-pad presets for stand/squat (joints 0, 1)
        if self._safe_get(self.jc_left, "get_button_up", joycon_id="left"):
            diff = self._torso_joints1_2_stand_q - torso_cmd[:2]
            torso_cmd[0] += np.sign(diff[0]) * self._torso_joint_max_delta * 1.74 / 2.7
            torso_cmd[1] += np.sign(diff[1]) * self._torso_joint_max_delta
        elif self._safe_get(self.jc_left, "get_button_down", joycon_id="left"):
            diff = self._torso_joints1_2_squat_q - torso_cmd[:2]
            torso_cmd[0] += np.sign(diff[0]) * self._torso_joint_max_delta * 1.74 / 2.7
            torso_cmd[1] += np.sign(diff[1]) * self._torso_joint_max_delta

        # Clamp joints 0,1 to stand/squat range
        torso_cmd[0] = np.clip(
            torso_cmd[0],
            self._torso_joints1_2_stand_q[0],
            self._torso_joints1_2_squat_q[0],
        )
        torso_cmd[1] = np.clip(
            torso_cmd[1],
            self._torso_joints1_2_squat_q[1],
            self._torso_joints1_2_stand_q[1],
        )

        # Clamp all joints to limits
        torso_cmd = np.clip(torso_cmd, self.TORSO_JOINT_LOW, self.TORSO_JOINT_HIGH)

        return torso_cmd

    def _process_gripper_command(self) -> Dict[str, float]:
        """Process gripper command from ZL/ZR buttons."""
        left_pressed = self._safe_get(self.jc_left, "get_button_zl", joycon_id="left")
        right_pressed = self._safe_get(self.jc_right, "get_button_zr", joycon_id="right")

        if self._gripper_toggle_mode:
            # Toggle mode - edge-triggered on button press
            # Left gripper: toggle on rising edge (was not pressed, now pressed)
            if left_pressed and not self._left_gripper_prev_pressed:
                self._left_gripper_current_action = (
                    1.0 if self._left_gripper_current_action < 0.5 else 0.1
                )
            self._left_gripper_prev_pressed = left_pressed

            # Right gripper: toggle on rising edge
            if right_pressed and not self._right_gripper_prev_pressed:
                self._right_gripper_current_action = (
                    1.0 if self._right_gripper_current_action < 0.5 else 0.1
                )
            self._right_gripper_prev_pressed = right_pressed

            left_gripper = self._left_gripper_current_action
            right_gripper = self._right_gripper_current_action
        else:
            # Hold mode - hold button to close
            left_gripper = 1.0 if left_pressed else 0.1
            right_gripper = 1.0 if right_pressed else 0.1

        return {"left": left_gripper, "right": right_gripper}

    def get_gripper_command(self) -> Dict[str, float]:
        """
        Get dual gripper commands from ZL/ZR buttons.

        Returns:
            {"left": float, "right": float} with values in brs-ctrl range [0.1, 1.0]
        """
        if not self.connected:
            return {"left": 0.1, "right": 0.1}
        return self._process_gripper_command()


# =============================================================================
# Factory Function (Singleton Pattern)
# =============================================================================

# Global singleton - only one JoyCon interface active at a time
_active_joycon: Optional[BaseJoyConInterface] = None
_active_robot_type: Optional[str] = None


def get_joycon_interface(robot_type: str, required: bool = True) -> BaseJoyConInterface:
    """
    Factory to get shared JoyCon interface (singleton per session).

    This ensures InputHandler and LocomotionHandler share the same
    JoyCon connection instead of fighting for hardware access.

    Args:
        robot_type: Robot type string (e.g., "A1", "R1Pro")
        required: If True, raise error when JoyCons not found

    Returns:
        JoyCon interface instance (shared singleton)
    """
    global _active_joycon, _active_robot_type

    if _active_joycon is not None:
        if _active_robot_type != robot_type:
            logger.warning(
                f"JoyCon already initialized for {_active_robot_type}, "
                f"reusing for {robot_type}"
            )
        return _active_joycon

    # Create new instance based on robot type
    if robot_type == "R1Pro":
        _active_joycon = R1ProJoyConInterface(required=required)
    else:
        _active_joycon = A1JoyConInterface(required=required)

    _active_robot_type = robot_type
    logger.info(f"JoyCon interface created for {robot_type}")
    return _active_joycon


def clear_joycon_interface():
    """
    Clear the singleton JoyCon interface.

    Call this when cleaning up or in tests to reset state.
    """
    global _active_joycon, _active_robot_type
    if _active_joycon is not None:
        _active_joycon.close()
    _active_joycon = None
    _active_robot_type = None


__all__ = [
    "JoyConHealthStatus",
    "BaseJoyConInterface",
    "A1JoyConInterface",
    "R1ProJoyConInterface",
    "get_joycon_interface",
    "clear_joycon_interface",
]

from typing import Sequence, Union, Literal, Optional
import time
from threading import RLock

import numpy as np
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite

from iiil.gello.dxl.constants import (
    operating_modes,
    Comm,
    OperatingMode,
    GoalCurrent,
    GoalPosition,
    CurrentLimit,
    PresentPosition,
    PresentVelocity,
    DXL_LOBYTE,
    DXL_LOWORD,
    DXL_HIBYTE,
    DXL_HIWORD,
)
from iiil.gello.configs.motor_type import MotorType
from iiil.gello.drivers.base_driver import DXLBaseDriver, PresentCurrent, PresentCVP, PositionTrajectory, VelocityTrajectory


class UnifiedDXLDriver(DXLBaseDriver):
    """
    A unified driver that can switch between current and position control modes dynamically.
    This avoids creating multiple PortHandler and PacketHandler instances per port.
    """
    
    # Dynamixel control table addresses for profile registers
    _PROFILE_ACCELERATION_ADDR = 108
    _PROFILE_VELOCITY_ADDR = 112
    _PROFILE_REG_LEN = 4  # bytes

    def __init__(
        self,
        ids: Union[int, Sequence[int]],
        port: str,
        motor_type: MotorType,
        baudrate: int = 3000000,
        *,
        initial_mode: Literal["current", "position"] = "current",
        current_limit: Optional[float] = None,
        profile_velocity: Optional[int] = None,
        profile_acceleration: Optional[int] = None,
        position_deadband: int = 0,
    ):
        if isinstance(ids, int):
            ids = [ids]
        assert all(isinstance(i, int) for i in ids)

        self._ids = ids
        self._port_name = port
        self._motor_type = motor_type
        self._current_motor_state = None
        self._lock = RLock()
        self._closed = False  # keeps inherited close() idempotent
        self._current_limit = current_limit
        self._profile_velocity = profile_velocity
        self._profile_acceleration = profile_acceleration
        self._position_deadband = position_deadband
        self._last_written_goal: Optional[np.ndarray] = None

        # Initialize the port handler and packet handler
        self._port_handler = PortHandler(port)
        self._packet_handler = PacketHandler(2.0)
        
        # Create group sync write objects for both modes
        self._current_group_sync_write = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            GoalCurrent.ADDR.value,
            GoalCurrent.LEN.value,
        )
        
        self._position_group_sync_write = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            GoalPosition.ADDR.value,
            GoalPosition.LEN.value,
        )
        
        # For current-based position control
        if current_limit is not None:
            self._current_limit_group_sync_write = GroupSyncWrite(
                self._port_handler,
                self._packet_handler,
                GoalCurrent.ADDR.value,
                GoalCurrent.LEN.value,
            )

        # Initialize GroupSyncRead objects for different data types
        self._position_sync_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            PresentPosition.ADDR.value,
            PresentPosition.LEN.value,
        )
        self._velocity_sync_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            PresentVelocity.ADDR.value,
            PresentVelocity.LEN.value,
        )
        self._current_sync_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            PresentCurrent.ADDR.value,
            PresentCurrent.LEN.value,
        )
        self._position_trajectory_sync_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            PositionTrajectory.ADDR.value,
            PositionTrajectory.LEN.value,
        )
        self._velocity_trajectory_sync_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            VelocityTrajectory.ADDR.value,
            VelocityTrajectory.LEN.value,
        )
        # Combined read for current+velocity+position in one USB round trip
        self._cvp_sync_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            PresentCVP.ADDR.value,
            PresentCVP.LEN.value,
        )

        # Add parameters for each Dynamixel servo to all sync read objects
        for dxl_id in self._ids:
            self._position_sync_read.addParam(dxl_id)
            self._velocity_sync_read.addParam(dxl_id)
            self._current_sync_read.addParam(dxl_id)
            self._position_trajectory_sync_read.addParam(dxl_id)
            self._velocity_trajectory_sync_read.addParam(dxl_id)
            self._cvp_sync_read.addParam(dxl_id)

        # Open the port and set the baudrate
        if not self._port_handler.openPort():
            raise RuntimeError("Failed to open the port")

        if not self._port_handler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

        # Give the port a moment to settle after opening and setting baudrate
        time.sleep(0.2)

        # Clear any pre-existing hardware errors (overload, etc.) by rebooting
        self._clear_hardware_errors()

        # Initialize in the specified mode
        self._current_mode = initial_mode
        self._switch_to_mode(initial_mode)

        # Disable torque for each Dynamixel servo
        self._torque_enabled = False
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            print(f"port: {port}, {e}")

        # Cached motor state values (used by inherited get_motor_state)
        self._joint_positions = None
        self._joint_velocities = None
        self._joint_currents = None
        self._joint_position_trajectory = None
        self._joint_velocity_trajectory = None

    # Hardware Error Status register (address 70, 1 byte)
    _ADDR_HARDWARE_ERROR_STATUS = 70

    def _clear_hardware_errors(self):
        """Check each motor for hardware errors and reboot any that have them."""
        for dxl_id in self._ids:
            val, res, _ = self._packet_handler.read1ByteTxRx(
                self._port_handler, dxl_id, self._ADDR_HARDWARE_ERROR_STATUS,
            )
            if res == 0 and val != 0:
                print(f"[UnifiedDXLDriver] Motor {dxl_id} on {self._port_name} has hardware error "
                      f"(status=0x{val:02x}), rebooting...")
                self._packet_handler.reboot(self._port_handler, dxl_id)
                time.sleep(1.0)
                for _attempt in range(10):
                    val2, res2, _ = self._packet_handler.read1ByteTxRx(
                        self._port_handler, dxl_id, self._ADDR_HARDWARE_ERROR_STATUS,
                    )
                    if res2 == 0 and val2 == 0:
                        print(f"[UnifiedDXLDriver] Motor {dxl_id} on {self._port_name} recovered.")
                        break
                    time.sleep(0.5)
                else:
                    print(f"[UnifiedDXLDriver] WARNING: Motor {dxl_id} on {self._port_name} still has "
                          f"hardware error after reboot (status=0x{val:02x}). "
                          f"Check power supply and wiring.")

    def _switch_to_mode(self, mode: Literal["current", "position"], initial_goal_encoder=None):
        """Switch the operating mode of all motors.

        Args:
            mode: Target operating mode ("current" or "position").
            initial_goal_encoder: Optional encoder positions to write as the goal
                BEFORE re-enabling torque. This prevents the motor from jumping to a
                stale GoalPosition register value when switching to position mode.
        """
        assert mode in ["current", "position"]
        print(f"Switching to {mode} mode")

        if mode == "position" and self._current_limit is not None:
            # Use current-based position control
            operating_mode = "current_based_position"
        elif mode == "position":
            # Use extended position mode — supports goals outside [0, 4095].
            # Required for joints whose encoder range straddles the 0/4095
            # boundary (e.g. Franka joints with large offsets).  Standard
            # position mode (3) wraps goals to [0, 4095], causing
            # full-revolution jumps across the boundary.
            operating_mode = "extended_position"
        else:
            operating_mode = mode

        # Reset deadband state so the first write after a mode switch always goes through
        self._last_written_goal = None

        # Cycle torque mode to enable switching to the new mode
        with self._lock:
            # Clear hardware errors first (e.g. stale input voltage error from
            # a previous 5V rail sag) so the torque-disable write succeeds.
            self._clear_hardware_errors()

            self._set_torque_mode(False)

            # Change the operating mode
            self._broadcast_byte_to_motors_no_lock(OperatingMode.ADDR, operating_modes[operating_mode])

            if operating_mode == "current_based_position":
                self._set_current_limit(self._current_limit)

            # Write profile registers (position/current-based-position modes only)
            if mode == "position":
                self._write_profile_registers()

            # Write initial goal before enabling torque to prevent jump
            if mode == "position" and initial_goal_encoder is not None:
                self._set_position_controls(initial_goal_encoder, force_write=True)

            # In current mode, zero goal currents before enabling torque.
            # Changing operating mode resets Goal Current to Current Limit;
            # enabling torque at that value causes a current surge that sags
            # the 5V USB rail and trips the input voltage error on low-power
            # motors (XL330/XC330).
            if mode == "current":
                self._zero_goal_currents()

            self._set_torque_mode(True)

            self._current_mode = mode

    def _set_current_limit(self, current_limit: Optional[float]):
        """Set the current limit for position control."""
        if current_limit is not None:
            current_limits = np.array([current_limit] * len(self._ids), dtype=np.int32)
            for dxl_id, current in zip(self._ids, current_limits):
                current_value = int(current)
                param = [DXL_LOBYTE(current_value), DXL_HIBYTE(current_value)]
                dxl_addparam_result = self._current_limit_group_sync_write.addParam(dxl_id, param)
                if not dxl_addparam_result:
                    raise RuntimeError(
                        f"Failed to set goal current for Dynamixel with ID {dxl_id}"
                    )
            # Syncwrite goal current
            dxl_comm_result = self._current_limit_group_sync_write.txPacket()
            if dxl_comm_result != Comm.SUCCESS.value:
                raise RuntimeError(
                    f"Failed to syncwrite goal current. Returned msg: {dxl_comm_result}"
                )
            # Clear syncwrite parameter storage
            self._current_limit_group_sync_write.clearParam()

    def _write_profile_registers(self):
        """Write Profile Velocity and Profile Acceleration to all motors.

        Must be called with torque disabled (inside ``_switch_to_mode``).
        These registers accept 4-byte unsigned integers.
        """
        for addr, value in [
            (self._PROFILE_ACCELERATION_ADDR, self._profile_acceleration),
            (self._PROFILE_VELOCITY_ADDR, self._profile_velocity),
        ]:
            if value is None:
                continue
            value = int(value)
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packet_handler.write4ByteTxRx(
                    self._port_handler, dxl_id, addr, value
                )
                if dxl_comm_result != Comm.SUCCESS.value or dxl_error != 0:
                    print(
                        f"Warning: Failed to write profile register {addr} "
                        f"for motor {dxl_id}: comm={dxl_comm_result}, err={dxl_error}"
                    )

    def switch_to_current_mode(self):
        """Switch all motors to current control mode."""
        if self._current_mode != "current":
            self._switch_to_mode("current")

    def switch_to_position_mode(self, initial_goal_encoder=None):
        """Switch all motors to position control mode.

        Args:
            initial_goal_encoder: Optional encoder positions to write as the initial
                goal before enabling torque. Pass current encoder readings to ensure
                zero-jump entry into position mode.
        """
        if self._current_mode != "position":
            self._switch_to_mode("position", initial_goal_encoder=initial_goal_encoder)

    def _create_group_sync_write(self, operating_mode: str) -> GroupSyncWrite:
        """This method is not used in the unified driver."""
        pass

    def _set_controls(self, controls: Sequence[float]):
        """
        Set controls based on the current mode.
        
        Assumes lock is already acquired.
        """
        if self._current_mode == "current":
            self._set_current_controls(controls)
        elif self._current_mode == "position":
            self._set_position_controls(controls)

    def _set_current_controls(self, torques: Sequence[float]):
        """
        Convert torques to currents and send them to the motors.
        
        Assumes lock is already acquired.
        """
        # Convert torques to currents using the motor's torque coefficient
        joint_currents = self._motor_type.value.convert_torque_to_current(np.array(torques, dtype=np.float32))
        
        # Read current limits if not already done
        if not hasattr(self, '_current_limits'):
            self._current_limits = self._read_current_limits_no_lock()
        
        # Clip currents to motor limits
        joint_currents = np.clip(
            joint_currents, -self._current_limits, self._current_limits
        )

        for dxl_id, current in zip(self._ids, joint_currents):
            current_value = int(current)
            param = [DXL_LOBYTE(current_value), DXL_HIBYTE(current_value)]
            dxl_addparam_result = self._current_group_sync_write.addParam(dxl_id, param)
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set goal current for Dynamixel with ID {dxl_id}"
                )
        
        # Syncwrite goal current
        dxl_comm_result = self._current_group_sync_write.txPacket()
        if dxl_comm_result != Comm.SUCCESS.value:
            raise RuntimeError(
                f"Failed to syncwrite goal current. Returned msg: {dxl_comm_result}"
            )

        # Clear syncwrite parameter storage
        self._current_group_sync_write.clearParam()

    def _zero_goal_currents(self):
        """Write zero goal current to all motors (lock must be held).

        Called before enabling torque in current mode to prevent the inrush
        current surge that occurs when Goal Current still holds the previous
        Current Limit value after an operating mode change.
        """
        for dxl_id in self._ids:
            param = [0, 0]  # 2-byte zero
            self._current_group_sync_write.addParam(dxl_id, param)
        self._current_group_sync_write.txPacket()
        self._current_group_sync_write.clearParam()

    def _set_position_controls(self, joint_angles: Sequence[float], force_write: bool = False):
        """Set joint angles for position control (extended position mode).

        In extended position mode, goal values are not limited to [0, 4095].
        We unwrap the goal relative to the last-written goal so the motor
        always takes the shortest path (< half revolution).

        Assumes the lock is already acquired for communication.

        Args:
            joint_angles (Sequence[float]): Desired encoder positions for each motor
            force_write: If True, skip deadband check and always write to hardware.
        """
        joint_positions = np.asarray(joint_angles, dtype=np.int32)

        # Unwrap: adjust goal so it is within ±2048 of the last-written goal.
        # This prevents full-revolution jumps for joints operating near the
        # encoder 0/4095 boundary (e.g. Franka joints with large offsets).
        if self._last_written_goal is not None:
            diff = joint_positions - self._last_written_goal
            # Wrap diff to [-2048, 2047]
            diff = (diff + 2048) % 4096 - 2048
            joint_positions = self._last_written_goal + diff

        # Deadband: skip USB write when all joints are within N ticks of last-written goal
        if (not force_write
                and self._position_deadband > 0
                and self._last_written_goal is not None
                and np.all(np.abs(joint_positions - self._last_written_goal) <= self._position_deadband)):
            return

        for dxl_id, position in zip(self._ids, joint_positions):
            position_value = int(position)
            # Allocate goal position value into byte array
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position_value)),
                DXL_HIBYTE(DXL_LOWORD(position_value)),
                DXL_LOBYTE(DXL_HIWORD(position_value)),
                DXL_HIBYTE(DXL_HIWORD(position_value)),
            ]

            # Add goal position value to the Syncwrite parameter storage
            dxl_addparam_result = self._position_group_sync_write.addParam(
                dxl_id, param_goal_position
            )
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set joint angle for Dynamixel with ID {dxl_id}"
                )

        # Syncwrite goal position
        dxl_comm_result = self._position_group_sync_write.txPacket()
        if dxl_comm_result != Comm.SUCCESS.value:
            raise RuntimeError(
                f"Failed to syncwrite goal position. Returned msg: {dxl_comm_result}"
            )

        # Clear syncwrite parameter storage
        self._position_group_sync_write.clearParam()

        # Update last-written goal for deadband comparison
        self._last_written_goal = joint_positions.copy()

    def _read_current_limits(self) -> np.ndarray:
        """Read the current limits for all motors."""
        with self._lock:
            return self._read_current_limits_no_lock()

    def _read_current_limits_no_lock(self) -> np.ndarray:
        """Read the current limits for all motors.
        
        Assumes the lock is already acquired.
        """
        current_limit_group_sync_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            CurrentLimit.ADDR.value,
            CurrentLimit.LEN.value,
        )
        # Add parameters for each Dynamixel servo to the group sync read
        for dxl_id in self._ids:
            if not current_limit_group_sync_read.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )
        # read current limit
        while True:
            dxl_comm_result = current_limit_group_sync_read.txRxPacket()
            if dxl_comm_result == Comm.SUCCESS.value:
                break
            time.sleep(0.1)
        current_limits = []
        for i, dxl_id in enumerate(self._ids):
            if current_limit_group_sync_read.isAvailable(
                dxl_id, CurrentLimit.ADDR.value, CurrentLimit.LEN.value
            ):
                current_limit = current_limit_group_sync_read.getData(
                    dxl_id, CurrentLimit.ADDR.value, CurrentLimit.LEN.value
                )
                current_limit = CurrentLimit.to_numpy(current_limit)
                current_limits.append(current_limit)
        del current_limit_group_sync_read
        return np.array(current_limits, dtype=np.float32)

    def set_controls_for_tracking(self, torques: Sequence[float], goal_encoder: Sequence[float]):
        """Atomically check mode and send the appropriate data.

        Eliminates the race between the control loop reading ``current_mode``
        and ``_switch_to_mode`` changing it on another thread.  Both the mode
        check and the data dispatch happen under the same lock.
        """
        if len(torques) != len(self._ids) or len(goal_encoder) != len(self._ids):
            raise ValueError(
                "The length of torques and goal_encoder must match the number of servos"
            )
        with self._lock:
            if not self._torque_enabled:
                return
            if self._current_mode == "position":
                self._set_position_controls(goal_encoder)
            else:
                self._set_current_controls(torques)

    @property
    def current_mode(self) -> str:
        """Get the current control mode."""
        return self._current_mode

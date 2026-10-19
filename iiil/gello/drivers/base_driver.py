import numpy as np
import time
from dataclasses import dataclass
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from enum import Enum
from iiil.gello.configs.motor_type import MotorType
from threading import Lock, Event, Thread
from typing import Protocol, Sequence, Union, Literal

from iiil.gello.dxl.constants import (
    operating_modes,
    PresentPosition,
    PresentVelocity,
    Torque,
    Comm,
    OperatingMode,
)

class HomingOffset(Enum):
    ADDR = 20
    LEN = 4

class PresentCurrent(Enum):
    # Additional information to get the current used by the motor
    ADDR = 126
    LEN = 2

    @staticmethod
    def to_proper_current(current_16bit: int) -> int:
        """
        The current is read as a 16-bit integer 2's complement, but we want to convert it to a 32-bit integer.
        """
        current_16bit_signed = np.int16(current_16bit)
        current_32bit = np.int32(current_16bit_signed)
        return current_32bit


class PresentCVP(Enum):
    """Combined read for Current, Velocity, Position (addresses 126-135, 10 bytes contiguous)."""
    ADDR = 126
    LEN = 10  # 2 (current) + 4 (velocity) + 4 (position)

class VelocityTrajectory(Enum):
    ADDR = 136
    LEN = 4

class PositionTrajectory(Enum):
    ADDR = 140
    LEN = 4

@dataclass
class DXLMotorState:
    """
    Holds the state information of the underlying Dynamixel motors for a particular driver. 
    """

    # joint angles in radians
    joint_angles: np.ndarray

    # joint velocities in radians per second
    joint_velocities: np.ndarray

    # joint currents in mA
    joint_currents: np.ndarray

    # joint position trajectory in radians
    joint_position_trajectory: np.ndarray

    # joint velocity trajectory in radians per second
    joint_velocity_trajectory: np.ndarray


class DXLBaseDriverProtocol(Protocol):
    """
    Protocol for controlling Dynamixel servos.
    """
    
    def set_controls(self, controls: Sequence[float]):
        """Set the controls for the Dynamixel servos, which are different depending on the mode.
        For instance, there can be position control or current control.

        Args:
            controls (Sequence[float]): A list of controls to be used for controlling the servos.
        """
        pass

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        pass

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        pass

    def get_motor_state(self) -> DXLMotorState:
        """Get current motor state using combined CVP + Fast Sync Read.

        Position, velocity, and current are always read together in ONE USB call.

        Returns:
            DXLMotorState: Current motor state with all available data
        """
        pass

    def close(self):
        """Close the driver."""
        pass


class DXLBaseDriver(DXLBaseDriverProtocol):
    read_hz: int = 500

    def __init__(
        self,
        ids: Union[int, Sequence[int]],
        port: str,
        motor_type: MotorType,
        baudrate: int = 3000000,
        *,
        operating_mode: Literal[
            "current",
            "velocity",
            "position",
            "extended_position",
            "current_based_position",
            "pwm",
        ],
    ):
        if isinstance(ids, int):
            ids = [ids]
        assert all(isinstance(i, int) for i in ids)

        self._ids = ids
        self._port_name = port
        self._motor_type = motor_type
        self._current_motor_state = None
        self._lock = Lock()
        self._closed = False

        # Initialize the port handler, packet handler, and group sync read/write
        self._port_handler = PortHandler(port)
        self._packet_handler = PacketHandler(2.0)
        assert operating_mode in operating_modes
        
        # Create group sync write object if the subclass needs it
        group_sync_write = self._create_group_sync_write(operating_mode)
        if group_sync_write is not None:
            self._group_sync_write = group_sync_write

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

        # change the operating mode
        self._broadcast_byte_to_motors(OperatingMode.ADDR, operating_modes[operating_mode])

        # Disable torque for each Dynamixel servo
        self._torque_enabled = False
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            print(f"port: {port}, {e}")

        # Cached motor state values
        self._joint_positions = None
        self._joint_velocities = None
        self._joint_currents = None
        self._joint_position_trajectory = None
        self._joint_velocity_trajectory = None

    def _broadcast_byte_to_motors(self, address, value):
        """
        Handles sending a single byte command to all Dynamixel motors for changing settings.
        """
        with self._lock:
            self._broadcast_byte_to_motors_no_lock(address, value)

    def _broadcast_byte_to_motors_no_lock(self, address, value):
        """
        Handles sending a single byte command to all Dynamixel motors for changing settings.
        Assumes the lock is already acquired.
        """
        failed = []
        for dxl_id in self._ids:
            success, dxl_comm_result, dxl_error = self._write_byte_with_retry(
                dxl_id,
                address.value,
                value.value,
            )
            if not success:
                if dxl_comm_result != Comm.SUCCESS.value:
                    err = self._packet_handler.getTxRxResult(dxl_comm_result)
                else:
                    err = self._packet_handler.getRxPacketError(dxl_error)
                failed.append((dxl_id, err))

        if failed:
            failed_str = ", ".join([f"{dxl_id} ({err})" for dxl_id, err in failed])
            raise ValueError(
                f"Failed to write address {address.value} on port {self._port_name} for IDs: {failed_str}. "
                "Check power, baudrate, and motor IDs."
            )

    @property
    def motor_ids(self):
        return self._ids

    def _create_group_sync_write(self, operating_mode: str) -> GroupSyncWrite:
        """Create a GroupSyncWrite object for the given operating mode.
        
        This method can be overridden by subclasses that need to create specific
        GroupSyncWrite objects. The unified driver doesn't use this method.
        
        Returns:
            GroupSyncWrite object or None if not needed
        """
        return None

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_controls(self, controls: Sequence[float]):
        """Set the controls for the Dynamixel servos, which are different depending on the mode.
        For instance, there can be position control or current control.

        Args:
            controls (Sequence[float]): A list of controls to be used for controlling the servos.
        """
        if len(controls) != len(self._ids):
            raise ValueError(
                "The length of controls must match the number of servos"
            )
        with self._lock:
            if not self._torque_enabled:
                raise RuntimeError("Torque must be enabled to start control")
            self._set_controls(controls)

    def _set_controls(self, controls: Sequence[float]):
        """set controls according to different controller type"""
        raise NotImplementedError

    def set_torque_mode(self, enable: bool):
        with self._lock:
            self._set_torque_mode(enable)

    def _set_torque_mode(self, enable: bool):
        """
        Internally sets the torque mode, assuming that the lock is already acquired.
        Fails gracefully per-servo to avoid crashing when one motor drops a packet.
        """
        torque_value = Torque.ENABLE.value if enable else Torque.DISABLE.value
        failed = False
        for dxl_id in self._ids:
            success, dxl_comm_result, dxl_error = self._write_byte_with_retry(
                dxl_id,
                Torque.ADDR.value,
                torque_value,
            )
            if not success:
                error_msg = (
                    f"Failed to set torque mode for Dynamixel with ID {dxl_id}"
                )
                if dxl_comm_result != 0:
                    error_msg += self._packet_handler.getTxRxResult(dxl_comm_result)
                elif dxl_error != 0:
                    error_msg += self._packet_handler.getRxPacketError(dxl_error)
                print(error_msg)
                failed = True
        # Only mark enabled if all succeeded
        self._torque_enabled = enable and not failed

    def _write_byte_with_retry(self, dxl_id: int, address: int, value: int, *, retries: int = 2, retry_delay: float = 0.05):
        """
        Helper to write a single byte with a few retries to smooth over transient bus drops.

        Args:
            dxl_id: Dynamixel ID to write to
            address: Control table address
            value: Byte value to write
            retries: Number of additional attempts after the first failure
            retry_delay: Delay between retries in seconds

        Returns:
            (success, last_comm_result, last_error)
        """
        last_comm_result, last_error = Comm.SUCCESS.value, 0
        for attempt in range(retries + 1):
            last_comm_result, last_error = self._packet_handler.write1ByteTxRx(
                self._port_handler, dxl_id, address, value
            )
            if last_comm_result == Comm.SUCCESS.value and last_error == 0:
                return True, last_comm_result, last_error
            time.sleep(retry_delay)

        return False, last_comm_result, last_error


    def set_homing_offset(
        self, dxl_ids: Union[int, Sequence[int]], values: Union[int, Sequence[int]]
    ):
        if isinstance(dxl_ids, int):
            dxl_ids = [dxl_ids]
        if isinstance(values, int):
            values = [values] * len(dxl_ids)
        assert len(dxl_ids) == len(
            values
        ), f"dxl_ids and values must have the same length, got {len(dxl_ids)} and {len(values)}"
        with self._lock:
            for dxl_id, value in zip(dxl_ids, values):
                dxl_comm_result, dxl_error = self._packet_handler.write4ByteTxRx(
                    self._port_handler, dxl_id, HomingOffset.ADDR.value, value
                )
                if dxl_comm_result != Comm.SUCCESS.value or dxl_error != 0:
                    error_msg = (
                        f"Failed to set homing offset for Dynamixel with ID {dxl_id}"
                    )
                    if dxl_comm_result != 0:
                        error_msg += self._packet_handler.getTxRxResult(dxl_comm_result)
                    elif dxl_error != 0:
                        error_msg += self._packet_handler.getRxPacketError(dxl_error)
                    raise RuntimeError(error_msg)

    def _read_motor_data(self, group_sync_read, addr_const, len_const):
        """Read motor data using Fast Sync Read.

        Args:
            group_sync_read: The GroupSyncRead object to use
            addr_const: The address constant (e.g., PresentPosition.ADDR)
            len_const: The length constant (e.g., PresentPosition.LEN)

        Returns:
            tuple: (motor_data array, error_occurred)
        """
        _motor_data = np.zeros(len(self._ids), dtype=int)

        dxl_comm_result = group_sync_read.fastSyncRead()
        if dxl_comm_result != Comm.SUCCESS.value:
            print(f"Fast Sync Read failed: {self._packet_handler.getTxRxResult(dxl_comm_result)}")
            return _motor_data, True

        for i, dxl_id in enumerate(self._ids):
            if group_sync_read.isAvailable(dxl_id, addr_const.value, len_const.value):
                data = group_sync_read.getData(dxl_id, addr_const.value, len_const.value)
                _motor_data[i] = np.int32(np.uint32(data))
            else:
                return _motor_data, True

        return _motor_data, False

    def _read_cvp_combined(self):
        """Read current, velocity, and position in ONE USB round trip using Fast Sync Read.

        Uses Fast Sync Read (instruction 0x8A) which returns ONE combined status packet
        for all motors instead of N individual packets.

        Memory layout at address 126-135 (10 bytes):
          - bytes 0-1: Present Current (signed 16-bit)
          - bytes 2-5: Present Velocity (signed 32-bit)
          - bytes 6-9: Present Position (signed 32-bit)

        Returns:
            tuple: (currents, velocities, positions, error_occurred)
        """
        num_motors = len(self._ids)
        currents = np.zeros(num_motors, dtype=np.int32)
        velocities = np.zeros(num_motors, dtype=np.int32)
        positions = np.zeros(num_motors, dtype=np.int32)

        # Fast Sync Read - single combined status packet
        dxl_comm_result = self._cvp_sync_read.fastSyncRead()
        if dxl_comm_result != Comm.SUCCESS.value:
            print(f"[CVP] Fast Sync Read failed: {self._packet_handler.getTxRxResult(dxl_comm_result)}")
            return None, None, None, True

        for i, dxl_id in enumerate(self._ids):
            if not self._cvp_sync_read.isAvailable(dxl_id, PresentCVP.ADDR.value, PresentCVP.LEN.value):
                return None, None, None, True

            # Extract current (2 bytes at offset 0)
            currents[i] = self._cvp_sync_read.getData(dxl_id, PresentCurrent.ADDR.value, PresentCurrent.LEN.value)

            # Extract velocity (4 bytes at offset 2)
            raw_velocity = self._cvp_sync_read.getData(dxl_id, PresentVelocity.ADDR.value, PresentVelocity.LEN.value)
            velocities[i] = np.int32(np.uint32(raw_velocity))

            # Extract position (4 bytes at offset 6)
            raw_position = self._cvp_sync_read.getData(dxl_id, PresentPosition.ADDR.value, PresentPosition.LEN.value)
            positions[i] = np.int32(np.uint32(raw_position))

        return currents, velocities, positions, False

    def _read_all_motor_data(self):
        """Read all motor data using combined CVP + Fast Sync Read.

        Position, velocity, and current are always read together in ONE USB call
        using combined CVP + Fast Sync Read for optimal performance.

        Returns:
            tuple: (joint_positions, joint_velocities, joint_currents)
        """
        # Always use combined CVP + Fast Sync Read
        joint_currents, joint_velocities, joint_positions, error = self._read_cvp_combined()
        if error:
            joint_positions = None
            joint_velocities = None
            joint_currents = None

        return joint_positions, joint_velocities, joint_currents

    def get_motor_state(self) -> DXLMotorState:
        """
        Get the current motor state by reading from hardware.

        Position, velocity, and current are always read together in ONE USB call
        using combined CVP + Fast Sync Read for optimal performance.

        Returns:
            DXLMotorState: Current motor state with all available data
        """
        with self._lock:
            joint_positions, joint_velocities, joint_currents_16bit = self._read_all_motor_data()

            if joint_positions is not None:
                self._joint_positions = joint_positions
            if joint_velocities is not None:
                self._joint_velocities = joint_velocities
            if joint_currents_16bit is not None:
                joint_currents_proper = PresentCurrent.to_proper_current(joint_currents_16bit)
                self._joint_currents = joint_currents_proper * self._motor_type.value.current_unit_to_mA

        rtn = DXLMotorState(
            joint_angles=self._joint_positions,
            joint_velocities=self._joint_velocities,
            joint_currents=self._joint_currents,
            joint_position_trajectory=self._joint_position_trajectory,
            joint_velocity_trajectory=self._joint_velocity_trajectory,
        )
        return rtn

    def close(self, disable_torque: bool = True):
        # Make close idempotent
        if self._closed:
            return
        self._closed = True

        # Best-effort shutdown; ignore errors if the port is already closed
        if disable_torque:
            try:
                if self.torque_enabled():
                    self.set_torque_mode(False)
            except Exception:
                pass

        # Clear parameters for all sync objects
        try:
            self._position_sync_read.clearParam()
            self._velocity_sync_read.clearParam()
            self._current_sync_read.clearParam()
            self._position_trajectory_sync_read.clearParam()
            self._velocity_trajectory_sync_read.clearParam()
            self._cvp_sync_read.clearParam()
        except Exception:
            pass

        try:
            self._port_handler.closePort()
        except Exception:
            pass

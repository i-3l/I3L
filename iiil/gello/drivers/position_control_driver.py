import numpy as np
from dynamixel_sdk import GroupSyncWrite
from typing import Sequence
from iiil.gello.drivers.base_driver import DXLBaseDriver
from iiil.gello.configs.motor_type import MotorType

from iiil.gello.dxl.constants import (
    GoalPosition,
    DXL_LOBYTE,
    DXL_LOWORD,
    DXL_HIBYTE,
    DXL_HIWORD,
    Comm,
    GoalCurrent,
)


class DXLPositionControlDriver(DXLBaseDriver):
    def __init__(
        self,
        ids: Sequence[int],
        port: str,
        motor_type: MotorType,
        current_limit: float = None,
        baudrate: int = 3000000,
    ):
        self._operating_mode = "position"
        if current_limit is not None:
            self._operating_mode = "current_based_position"
            self._current_limit = current_limit

        super().__init__(
            ids=ids,
            port=port,
            motor_type=motor_type,
            baudrate=baudrate,
            operating_mode=self._operating_mode,
        )

        # Handle special case for position based motors
        # enabling torque in position control mode lets present position be in one turn
        self.set_torque_mode(True)
        self.set_torque_mode(False)

        if self._operating_mode == "current_based_position":
            self._current_limit_group_sync_write = GroupSyncWrite(
                self._port_handler,
                self._packet_handler,
                GoalCurrent.ADDR.value,
                GoalCurrent.LEN.value,
            )
        
    def _create_group_sync_write(self, operating_mode: str) -> GroupSyncWrite:
        assert operating_mode in ["position", "current_based_position"], """Use `operating_mode = "position"`"""
        group_sync_write = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            GoalPosition.ADDR.value,
            GoalPosition.LEN.value,
        )
        return group_sync_write

    def _set_controls(self, joint_angles: Sequence[float]):
        """Set joint angles for position control.

        Assumes the lock is already acquired for communication.
        
        Args:
            joint_angles (Sequence[float]): Desired joint angles in radians for each motor
        """
        joint_positions = joint_angles.astype(np.int32)
        
        # Validate position values are within valid range [0, 4095]
        assert np.all(joint_positions >= 0) and np.all(
            joint_positions < 4096
        ), "Joint positions must be within [0, 4096), current joint positions: " + str(joint_positions) + " and ids: " + str(self._ids)
        
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
            dxl_addparam_result = self._group_sync_write.addParam(
                dxl_id, param_goal_position
            )
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set joint angle for Dynamixel with ID {dxl_id}"
                )

        # Syncwrite goal position
        dxl_comm_result = self._group_sync_write.txPacket()
        if dxl_comm_result != Comm.SUCCESS.value:
            raise RuntimeError(
                f"Failed to syncwrite goal position. Returned msg: {dxl_comm_result}"
            )

        # Clear syncwrite parameter storage
        self._group_sync_write.clearParam()

        if self._operating_mode == "current_based_position":
            current_limits = np.array([self._current_limit] * len(self._ids), dtype=np.int32)
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
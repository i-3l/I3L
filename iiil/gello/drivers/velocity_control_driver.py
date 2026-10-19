import numpy as np
from iiil.gello.drivers.base_driver import DXLBaseDriver
from iiil.gello.configs.motor_type import MotorType
from dynamixel_sdk import GroupSyncWrite
from typing import Sequence
from enum import Enum
from iiil.gello.dxl.constants import (
    DXL_LOBYTE,
    DXL_LOWORD,
    DXL_HIBYTE,
    DXL_HIWORD,
    Comm,
)

class GoalVelocity(Enum):
    ADDR = 104
    LEN = 4

class DXLVelocityControlDriver(DXLBaseDriver):
    def __init__(
        self,
        ids: Sequence[int],
        port: str,
        motor_type: MotorType,
        baudrate: int = 3000000,
    ):
        super().__init__(
            ids=ids,
            port=port,
            motor_type=motor_type,
            baudrate=baudrate,
            operating_mode="velocity",
        )

    def _create_group_sync_write(self, operating_mode: str) -> GroupSyncWrite:
        assert operating_mode == "velocity", """Use `operating_mode = "velocity"`"""
        group_sync_write = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            GoalVelocity.ADDR.value,
            GoalVelocity.LEN.value,
        )
        return group_sync_write

    def _set_controls(self, joint_velocities: Sequence[float]):
        """Set joint velocities for velocity control.

        Assumes the lock is already acquired for communication.
        
        Args:
            joint_velocities (Sequence[float]): Desired joint velocities in rad/s for each motor
        """
        # Convert joint velocities from rad/s to DXL velocity units
        # DXL velocity unit: 0.229 rpm per unit
        # Convert rad/s to rpm: (rad/s) * 60 / (2*pi) = rpm
        # Then convert to DXL units: rpm / 0.229
        joint_velocities_dxl = np.array(joint_velocities, dtype=np.int32) * 60 / (2 * np.pi) / 0.229
        joint_velocities_dxl = joint_velocities_dxl.astype(np.int32)
        
        for dxl_id, velocity in zip(self._ids, joint_velocities_dxl):
            velocity_value = int(velocity)
            # Allocate goal velocity value into byte array
            param_goal_velocity = [
                DXL_LOBYTE(DXL_LOWORD(velocity_value)),
                DXL_HIBYTE(DXL_LOWORD(velocity_value)),
                DXL_LOBYTE(DXL_HIWORD(velocity_value)),
                DXL_HIBYTE(DXL_HIWORD(velocity_value)),
            ]

            # Add goal velocity value to the Syncwrite parameter storage
            dxl_addparam_result = self._group_sync_write.addParam(
                dxl_id, param_goal_velocity
            )
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set joint velocity for Dynamixel with ID {dxl_id}"
                )

        # Syncwrite goal velocity
        dxl_comm_result = self._group_sync_write.txPacket()
        if dxl_comm_result != Comm.SUCCESS.value:
            raise RuntimeError(
                f"Failed to syncwrite goal velocity. Returned msg: {dxl_comm_result}"
            )

        # Clear syncwrite parameter storage
        self._group_sync_write.clearParam() 
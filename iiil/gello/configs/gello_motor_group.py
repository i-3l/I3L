from dataclasses import dataclass
from iiil.gello.configs.motor_type import MotorType
from typing import Optional, Sequence, Union

@dataclass
class GelloMotorGroup:
    """
    Holds information about a group of motors, which may be connected to a single U2D2
    """
    # IDs of the motors in this group
    motor_ids: Sequence[int]

    # Robot motor type: only one motor type is supported per motor group
    motor_type: MotorType

    # Indices of the robot joints corresponding to each motor id
    robot_joint_indices: Sequence[int]

    # Direction that the GELLO needs to spin relative to the robot joint
    arm_joint_signs: Sequence[int]
    
    # The path to the device
    device_port: str
    baudrate: int = 3000000

    # Current limit for position control
    current_limit: Optional[float] = None

    # Dynamixel Profile Velocity (addr 112, 4 bytes) — limits position-mode speed.
    # Unit: 0.229 rev/min ≈ 0.024 rad/s.  0 = no limit (hardware default).
    profile_velocity: Optional[int] = None

    # Dynamixel Profile Acceleration (addr 108, 4 bytes) — limits position-mode accel.
    # Unit: 214.577 rev/min² ≈ 3.74 rad/s².  0 = no limit (hardware default).
    profile_acceleration: Optional[int] = None

    # Position write deadband in encoder ticks. When > 0, GoalPosition writes are
    # suppressed if the new goal is within this many ticks of the last-written goal.
    # 1 tick = 0.088° = 0.00153 rad. Default 0 = disabled (write every cycle).
    position_deadband: int = 0

    # Maximum joint velocity in rad/s
    max_joint_velocity: Union[float, Sequence[float]] = 1.0

    def __post_init__(self):
        # Validates that there is one robot joint idex per motor id
        assert len(self.motor_ids) == len(self.robot_joint_indices), (
            "The number of motor ids must match the number of robot joint indices."
        )

        # Makes sure that there is a 1-to-1 mapping between motor ids and max_joint_velocity values
        if isinstance(self.max_joint_velocity, float):
            self.max_joint_velocity = [self.max_joint_velocity] * len(self.motor_ids)

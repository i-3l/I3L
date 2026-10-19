from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class MotorProperties:
    """
    Dataclass to hold hardware specific motor properties
    """
    motor_name: str
    torque_to_current_coefficient: float
    current_unit_to_mA: float

    def convert_torque_to_current(self, torque: float) -> float:
        """
        Convert torque to current
        """
        return torque * self.torque_to_current_coefficient


class MotorType(Enum):
    """
    Enum for supported Dynamixel motor types.

    torque_to_current_coefficient:
        Maps torque [Nm] → Dynamixel current units
        current_units = torque [Nm] * torque_to_current_coefficient
    """

    # XM430-W350-T
    # Datasheet: 4.1 Nm @ 2.3 A
    # I/tau = 2.3 / 4.1 = 0.56098 A/Nm
    # Dynamixel current unit = 2.69 mA
    # => (0.56098 * 1000) / 2.69 ≈ 208.54 units/Nm
    XM430_W350_T = MotorProperties(
        motor_name="XM430-W350-T",
        torque_to_current_coefficient=(2.3 / 4.1) * 1000 / 2.69,
        current_unit_to_mA=2.69,
    )

    # XC330-M288-T
    # Datasheet: 1.1 Nm @ 1.2 A
    # I/tau = 1.2 / 1.1 = 1.0909 A/Nm
    # Dynamixel current unit = 1 mA
    # => (1.0909 * 1000) / 1 ≈ 1090.9 units/Nm
    XC330_M288_T = MotorProperties(
        motor_name="XC330-M288-T",
        torque_to_current_coefficient=(1.2 / 1.1) * 1000,
        current_unit_to_mA=1.0,
    )

    # XL430-W250-T (12V, used for Franka Joylo proximal joints)
    # Datasheet: 1.5 Nm @ 1.5 A
    # I/tau = 1.5 / 1.5 = 1.0 A/Nm
    # Dynamixel current unit = 2.69 mA (same as XM430 family)
    # => (1.0 * 1000) / 2.69 ≈ 371.7 units/Nm
    XL430_W250_T = MotorProperties(
        motor_name="XL430-W250-T",
        torque_to_current_coefficient=(1.5 / 1.5) * 1000 / 2.69,
        current_unit_to_mA=2.69,
    )

    # XL330-M288-T (5V, used for Franka Joylo distal joints)
    # Datasheet: 0.34 Nm @ 0.57 A
    # I/tau = 0.57 / 0.34 = 1.6765 A/Nm
    # Dynamixel current unit = 1 mA (same as XC330 family)
    # => (1.6765 * 1000) / 1 ≈ 1676.5 units/Nm
    XL330_M288_T = MotorProperties(
        motor_name="XL330-M288-T",
        torque_to_current_coefficient=(0.57 / 0.34) * 1000,
        current_unit_to_mA=1.0,
    )

    @classmethod
    def from_string(cls, motor_type: str):
        for member in cls:
            if motor_type == member.value.motor_name:
                return member
        raise ValueError(f"Invalid motor type: {motor_type}")
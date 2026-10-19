"""
GELLO (General Low-Cost) bilateral teleoperation system.

Modular architecture with plug-and-play compensation and utilities.
"""

# Core configuration classes
from iiil.gello.configs.gello_motor_group import GelloMotorGroup
from iiil.gello.configs.motor_type import MotorType, MotorProperties

# Controllers (base and implementations)
# NOTE: Controller imports removed to avoid circular import with gello_utils
# Import controllers directly from iiil.gello.controllers.* instead

# Compensation modules
from iiil.gello.compensation.compensation_config import (
    GravityCompensationConfig,
    FrictionCompensationConfig,
    CompensationConfig
)
from iiil.gello.compensation.gravity_compensation import GravityCompensator, create_gravity_compensator
from iiil.gello.compensation.friction_compensation import FrictionCompensator, create_friction_compensator

# Utility modules
# NOTE: Utils imports removed to avoid circular import with factory modules
# Import utils directly from iiil.gello.utils.* instead

__all__ = [
    # Core
    'GelloMotorGroup',
    'MotorType',
    'MotorProperties',

    # Compensation
    'GravityCompensationConfig',
    'FrictionCompensationConfig',
    'CompensationConfig',
    'GravityCompensator',
    'FrictionCompensator',
    'create_gravity_compensator',
    'create_friction_compensator',

]

"""
Compensation modules for GELLO system.
"""

from iiil.gello.compensation.compensation_config import (
    GravityCompensationConfig,
    FrictionCompensationConfig,
    AntiFrictionCompensationConfig,
    CompensationConfig
)
from iiil.gello.compensation.gravity_compensation import GravityCompensator, create_gravity_compensator
from iiil.gello.compensation.friction_compensation import FrictionCompensator, create_friction_compensator
from iiil.gello.compensation.anti_friction_compensation import AntiFrictionCompensator, create_anti_friction_compensator

__all__ = [
    'GravityCompensationConfig',
    'FrictionCompensationConfig',
    'AntiFrictionCompensationConfig',
    'CompensationConfig',
    'GravityCompensator',
    'FrictionCompensator',
    'AntiFrictionCompensator',
    'create_gravity_compensator',
    'create_friction_compensator',
    'create_anti_friction_compensator',
]

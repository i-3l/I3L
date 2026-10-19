from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GelloControlDirection(Enum):
	"""
	Direction of control between Gello and the environment.

	Three modes:
	- GELLO_TO_ENV: User controls gello, which commands the robot
	- ENV_TO_GELLO: Robot (via policy) commands gello to follow
	- ADMITTANCE: Policy commands GELLO (with low stiffness), robot follows GELLO's actual position
	"""

	GELLO_TO_ENV = "gello_to_env"
	ENV_TO_GELLO = "env_to_gello"
	ADMITTANCE = "admittance"

	@classmethod
	def from_string(cls, control_direction: str) -> "GelloControlDirection":
		if control_direction == cls.GELLO_TO_ENV.value:
			return cls.GELLO_TO_ENV
		if control_direction == cls.ENV_TO_GELLO.value:
			return cls.ENV_TO_GELLO
		if control_direction == cls.ADMITTANCE.value:
			return cls.ADMITTANCE
		raise ValueError(f"Invalid control direction: {control_direction}. Valid options: gello_to_env, env_to_gello, admittance")


@dataclass
class ArmMapping:
	"""Mapping between robot action slices and Gello joint slices."""

	name: str
	robot_slice: slice
	gello_slice: slice
	dof: int


@dataclass
class TeleopRuntimeState:
	"""Mutable runtime state for TeleopPolicy."""

	started_control: bool = False
	should_reset_gello_position: bool = True
	last_reset_goal: Optional[object] = None
	ready_for_trial: bool = False


__all__ = ["GelloControlDirection", "ArmMapping", "TeleopRuntimeState"]

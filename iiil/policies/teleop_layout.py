from typing import Dict, List, Tuple

from iiil.policies.teleop_types import ArmMapping


def build_action_layout(robot_cfg, default_arm_dof: int) -> Tuple[list, Dict, Dict, int]:
	"""
	Build ordered action layout from the robot config. Falls back to arm/gripper only.

	Returns:
		action_keys: ordered list of action keys
		action_key_dims: dict of key -> dimension
		action_slices: dict of key -> slice in the flat action vector
		action_dim: total length of the action vector
	"""
	action_key_dims = {}
	action_keys = []
	if robot_cfg is not None:
		try:
			if hasattr(robot_cfg, "action_key_dims"):
				action_key_dims = dict(robot_cfg.action_key_dims) if robot_cfg.action_key_dims else {}
			else:
				action_key_dims = robot_cfg.get("action_key_dims", {}) if robot_cfg else {}
			if hasattr(robot_cfg, "action_keys"):
				action_keys = list(robot_cfg.action_keys) if robot_cfg.action_keys else []
			else:
				action_keys = robot_cfg.get("action_keys", []) if robot_cfg else []
		except Exception:
			# Keep defaults below if access fails
			action_key_dims = {}
			action_keys = []

	if not action_key_dims:
		action_key_dims = {"arm": default_arm_dof, "gripper": 1}
	if not action_keys:
		action_keys = list(action_key_dims.keys())

	offset = 0
	action_slices = {}
	for key in action_keys:
		dim = int(action_key_dims.get(key, 0))
		action_slices[key] = slice(offset, offset + dim)
		offset += dim

	# Ensure base/torso are not treated as arms if present
	return action_keys, action_key_dims, action_slices, offset


def select_gello_slice_for_arm(arm_key: str, gello_arm_dof: int, gello_num_arms: int) -> slice:
	"""
	Choose which Gello arm slice to use for a given robot arm key.
	Right arm is the first slice on bilateral hardware.
	"""
	if gello_num_arms == 1:
		return slice(0, gello_arm_dof)
	if "right" in arm_key:
		return slice(0, gello_arm_dof)
	if "left" in arm_key:
		return slice(gello_arm_dof, gello_arm_dof * 2)
	return slice(0, gello_arm_dof)


def build_arm_mappings(
	arm_keys: List[str],
	action_slices: Dict[str, slice],
	gello_arm_dof: int,
	gello_num_arms: int,
) -> List[ArmMapping]:
	"""
	Create mappings from robot action slices to Gello joints for all arms.
	"""
	mappings: List[ArmMapping] = []
	for arm_key in arm_keys:
		if arm_key not in action_slices:
			continue
		
		# Skip non-arm keys (like base, torso, etc. if they accidentally match substring)
		# But we rely on 'arm_keys' list already being filtered by caller.
		# The issue is if the caller passes 'base' as an arm key? 
		# No, caller does: self._arm_keys = [k for k in self._action_keys if "arm" in k]
		
		robot_slice = action_slices[arm_key]
		gello_slice = select_gello_slice_for_arm(arm_key, gello_arm_dof, gello_num_arms)
		dof = min(robot_slice.stop - robot_slice.start, gello_slice.stop - gello_slice.start)
		mappings.append(
			ArmMapping(
				name=arm_key,
				robot_slice=slice(robot_slice.start, robot_slice.start + dof),
				gello_slice=slice(gello_slice.start, gello_slice.start + dof),
				dof=dof,
			)
		)

	if not mappings:
		# Fallback only if no specific arm keys were found
		# Check if generic "arm" key exists
		if "arm" in action_slices:
			robot_slice = action_slices["arm"]
			dof = min(robot_slice.stop - robot_slice.start, gello_arm_dof)
			mappings.append(
				ArmMapping(
					name="arm",
					robot_slice=slice(robot_slice.start, robot_slice.start + dof),
					gello_slice=slice(0, dof),
					dof=dof,
				)
			)
	return mappings


__all__ = ["build_action_layout", "select_gello_slice_for_arm", "build_arm_mappings"]

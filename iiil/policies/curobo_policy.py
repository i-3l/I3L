import torch
from iiil.policies.policy_base import BasePolicy, PolicyMode
from iiil.utils.misc_utils import ActionTypesSelection
from iiil.wrappers.robots import SimA1
from omegaconf import DictConfig
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.controllers import InverseKinematicsController
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils import transform_utils as T
from typing import List, Optional


logger = create_module_logger("CuroboPolicy")

n_attached_obj_calls = 0

class CuroboPolicy(BasePolicy):
	"""
	Policy for Curobo Motion Generator
	"""

	def __init__(self, config: DictConfig) -> None:
		super().__init__(config)
		self.cmg = None
		self.action_buffer = torch.tensor([])
		# For Holonomic robots, the action should always be deltas, but curobo will always return absolute base actions
		# so we need to convert them to deltas if necessary
		self.base_action_is_absolute = False
		self.plan_failed = False
		# max len of action buffer to store (i.e. how many steps before replanning)
		# if None, will execute the whole trajectory planned by Curobo
		self.action_buffer_len = config.action_buffer_len
		# max number of retries before failing the plan
		# if non-zero, will retry planning after executing the first 10 steps retured by the previous failed plan.
		# Often times the first 10 steps are enough to get the robot to a better position to plan a new trajectory
		self.max_retry_before_fail = config.max_retry_before_fail
		# Counter to keep track of how many times we have retried planning
		self._retry_counter = 0
		# Consecutive plan failure counter (for auto-reset)
		self._consecutive_failures = 0
		self.consecutive_failure_limit = 10
		position_delta_limits = config.get("base_position_delta_limits", None)
		self.base_position_delta_limits = None if position_delta_limits is None else tuple(position_delta_limits)

	def _convert_absolute_base_action(self, action: torch.Tensor) -> torch.Tensor:
		action = self.robot.absolute_base_action_to_delta(action)
		if self.base_position_delta_limits is not None:
			limits = action.new_tensor(self.base_position_delta_limits)
			action[self.robot.base_action_idx] = torch.clamp(
				action[self.robot.base_action_idx],
				min=-limits,
				max=limits,
			)
		return action

	def _is_ik_controller(self) -> bool:
		if not hasattr(self, "robot") or self.robot is None:
			return False
		for arm_name in self.robot.arm_names:
			controller_name = f"arm_{arm_name}"
			if controller_name in self.robot.controllers:
				controller = self.robot.controllers[controller_name]
				if isinstance(controller, InverseKinematicsController):
					return True
		return False

	def predict(self, obs: dict[str, torch.Tensor], task_info: dict, **kwargs) -> torch.Tensor:
		direct_action = task_info["intent"].get("direct_action", None)
		if direct_action is not None:
			self.action_buffer = torch.tensor([])
			self.base_action_is_absolute = False
			self._retry_counter = 0
			return torch.as_tensor(direct_action, dtype=torch.float32)

		is_ik_controller = self._is_ik_controller()
		if task_info["intent"]["replan"]:
			# If replan is True, we ignore the action buffer and plan a new trajectory based on the current observation and task info
			self.action_buffer = torch.tensor([])
		if self.action_buffer.shape[0] != 0:
			# pop the next action in action buffer
			action = self.action_buffer[0]
			if self.base_action_is_absolute:
				action = self._convert_absolute_base_action(action)
			self.action_buffer = self.action_buffer[1:]
		else:
			if task_info["intent"]["target_pos"] is not None:
				attached_obj = None if "attached_obj" not in task_info else task_info["attached_obj"]
				attached_obj_scale = None if "attached_obj_scale" not in task_info else task_info["attached_obj_scale"]
				ignore_collision_objects = task_info.get("ignore_collision_objects", None)
				if attached_obj is not None:
					global n_attached_obj_calls
					n_attached_obj_calls += 1
					# The reason we are doing this is because trimesh will segfault at the exact 750th call on raycast when
					# CuRoboMotionGenerator calls attach objects to robot. So we recreate the CMG instance here before that to avoid the segfault.
					if n_attached_obj_calls == 749:
						use_default_embodiment_only = True if isinstance(self.robot, SimA1) else False
						self.cmg = CuRoboMotionGenerator(
							robot=self.robot,
							batch_size=1,
							use_cuda_graph=False,
							use_default_embodiment_only=use_default_embodiment_only,
						)
						n_attached_obj_calls = 0
				ik_only = bool(task_info["intent"].get("ik_only", False))
				emb_sel = task_info["intent"]["action_type"]

				successes, traj_paths = self.plan_trajectory(
					target_pos=task_info["intent"]["target_pos"],
					target_quat=task_info["intent"]["target_quat"],
					emb_sel=emb_sel,
					motion_constraint=task_info["intent"].get("motion_constraint", None),
					attached_obj=attached_obj,
					attached_obj_scale=attached_obj_scale,
					ik_only=ik_only,
					ik_world_collision_check=not ik_only,
					ignore_collision_objects=ignore_collision_objects,
				)
				
				if not successes[0]:
					# current plan failed
					if self._retry_counter == self.max_retry_before_fail:
						# max retry reached, fail the plan
						self.plan_failed = True
					else:
						logger.warning(f"Retrying planning, retry count: {self._retry_counter}")
						self._retry_counter += 1
						if traj_paths[0] is not None:
							# we execute the first 10 steps of the trajectory and then retry planning
							action_buffer_len = 10
						else:
							# curobo failed to return any trajectory, so we do 10 no-op and retry planning
							self.action_buffer = self.robot.get_no_op_action().repeat(9, 1)
							self.base_action_is_absolute = False
							return self.action_buffer[0]
				else:
					# plan success, reset retry counter and failure state
					self._retry_counter = 0
					self._consecutive_failures = 0
					self.plan_failed = False
					action_buffer_len = self.action_buffer_len
				if self.plan_failed:
					action = self.robot.get_no_op_action()
					self.base_action_is_absolute = False
					self._consecutive_failures += 1
					if self._consecutive_failures >= self.consecutive_failure_limit:
						print(f"\033[1;31m>>> CUROBO: {self._consecutive_failures} consecutive planning failures — requesting reset\033[0m")
					else:
						logger.warning(f"Curobo failed to plan trajectory ({self._consecutive_failures}/{self.consecutive_failure_limit})")
					self._retry_counter = 0
					return action
				else:
					get_full_js = False if ik_only else True
					q_traj = self.cmg.path_to_joint_trajectory(
						path=traj_paths[0], 
						get_full_js=get_full_js, 
						emb_sel=emb_sel,
					).cpu().float()
					# Since ik only mode returns the goal joint pos only, we need to add the current joint pos to the traj
					# so that we can interpolate between the two
					if ik_only:
						q_traj = torch.stack([obs["robot::proprio"][:self.robot.n_joints], q_traj])
					max_inter_dist = 0.015 if self.robot.model == "r1pro" else 0.03
					q_traj = self.cmg.add_linearly_interpolated_waypoints(traj=q_traj, max_inter_dist=max_inter_dist)
					if is_ik_controller:
						arm_name = self.robot.default_arm
						arm_joint_names = self.cmg.mg[emb_sel].kinematics.joint_names
						arm_q_traj = q_traj[:, :len(arm_joint_names)]
						fk_state = self.cmg.mg[emb_sel].ik_solver.fk(arm_q_traj.to(self.cmg.mg[emb_sel].ik_solver.tensor_args.device))
						pos = fk_state.ee_position.cpu().float()
						quat_xyzw = fk_state.ee_quaternion[:, [1, 2, 3, 0]].cpu().float()
						pose_traj = torch.cat([pos, T.quat2axisangle(quat_xyzw)], dim=-1)
						current_pose = torch.cat([self.robot.get_relative_eef_position(arm_name), T.quat2axisangle(self.robot.get_relative_eef_orientation(arm_name))])
						pose_traj_with_start = torch.cat([current_pose.unsqueeze(0), pose_traj], dim=0)
						pose_absolute = torch.cat([pose_traj_with_start[1:, :3], pose_traj_with_start[1:, 3:6]], dim=1)
						self.action_buffer = torch.cat([pose_absolute, torch.zeros(pose_absolute.shape[0], 1, dtype=pose_absolute.dtype, device=pose_absolute.device)], dim=1)
						if action_buffer_len:
							self.action_buffer = self.action_buffer[:action_buffer_len]
						for arm_name in self.robot.arm_names:
							self.action_buffer[:, self.robot.gripper_action_idx[arm_name]] = task_info["intent"][f"gripper_{arm_name}"]
						action = self.action_buffer[0]
						self.action_buffer = self.action_buffer[1:]
						self.base_action_is_absolute = False
					else:
						self.action_buffer = self.robot.q_to_action(q_traj)
						if action_buffer_len:
							self.action_buffer = self.action_buffer[:action_buffer_len]
						for arm_name in self.robot.arm_names:
							self.action_buffer[:, self.robot.gripper_action_idx[arm_name]] = task_info["intent"][f"gripper_{arm_name}"]
						action = self._convert_absolute_base_action(self.action_buffer[0])
						self.action_buffer = self.action_buffer[1:]
						self.base_action_is_absolute = True
			else:
				action = self.robot.get_no_op_action()
				for arm_name in self.robot.arm_names:
					action[self.robot.gripper_action_idx[arm_name]] = task_info["intent"][f"gripper_{arm_name}"]
				# construct action buffer that repeats the gripper action 10 times
				self.action_buffer = torch.stack([action] * 10)
				self.base_action_is_absolute = False
		
		return action

	def load(self, **kwargs) -> None:
		self.robot = kwargs["robot"]
		# Create CuRobo instance
		# A1 only has default embodiment, so set to True here
		use_default_embodiment_only = True if isinstance(self.robot, SimA1) else False
		self.cmg = CuRoboMotionGenerator(
			robot=kwargs["robot"],
			batch_size=1,
			use_cuda_graph=True,
			use_default_embodiment_only=use_default_embodiment_only,
		)
		
	def reset(self) -> None:
		self.action_buffer_len = self.config.action_buffer_len
		self.action_buffer = torch.tensor([])
		self._retry_counter = 0
		self._consecutive_failures = 0
		self.plan_failed = False

	@property
	def needs_reset(self) -> bool:
		"""True when consecutive planning failures have hit the limit."""
		return self._consecutive_failures >= self.consecutive_failure_limit

	def close(self) -> None:
		pass

	def set_mode(self, policy_mode: PolicyMode) -> None:
		"""
		Set the mode of the policy. For CuroboPolicy, this does nothing.
		
		Args:
			policy_mode (PolicyMode): The mode to set for the policy
		"""
		# Default implementation does nothing
		pass

	def plan_trajectory(
		self, 
		target_pos, 
		target_quat, 
		emb_sel=CuRoboEmbodimentSelection.DEFAULT, 
		attached_obj=None, 
		attached_obj_scale=None,
		motion_constraint: Optional[List[float]] = None,
		ik_only=False,
		ik_world_collision_check=True,
		ignore_collision_objects=None,
	):
		skip_obstacle_update = False
		if ignore_collision_objects:
			self.cmg.update_obstacles(ignore_objects=ignore_collision_objects)
			skip_obstacle_update = True
		# Generate collision-free trajectories to the sampled eef poses (including self-collisions)
		successes, traj_paths = self.cmg.compute_trajectories(
			target_pos=target_pos,
			target_quat=target_quat,
			is_local=False,
			max_attempts=100,
			timeout=1.0,
			ik_fail_return=50,
			enable_finetune_trajopt=True,
			finetune_attempts=1,
			return_full_result=False,
			success_ratio=1.0,
			attached_obj=attached_obj,
			attached_obj_scale=attached_obj_scale,
			motion_constraint=motion_constraint,
			skip_obstacle_update=skip_obstacle_update,
			emb_sel=emb_sel,
			ik_only=ik_only,
			ik_world_collision_check=ik_world_collision_check
		)
		return successes, traj_paths
	
	def check_episode_ended(self, success: bool=True) -> bool:
		return self.plan_failed and not success

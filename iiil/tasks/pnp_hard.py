import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch
from collections import defaultdict
from iiil.tasks.pick_place import PickPlaceTask
from iiil.utils.math_utils import add_position_noise, add_quaternion_noise
from iiil.utils.misc_utils import ActionTypesSelection
from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.object_states import OnTop
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.python_utils import create_class_from_registry_and_config
from omnigibson.utils.ui_utils import create_module_logger
from typing import Any, Optional, Tuple


logger = create_module_logger("PickPlaceHardTask")
logger.setLevel("INFO")


class PickPlaceHardTask(PickPlaceTask):
    """
    Pick-place variant with two strawberries.

    The task always manipulates object_0 first, then object_1. The second
    strawberry is tinted green, and the grasp / place intent mirrors
    PickPlaceTask so both objects use the same relative grasp pose.
        """

    def __init__(
        self,
        show_trajectory: bool = True,
        n_objects: int = 2,
        randomize_initial_pose: bool = True,
        randomize_robot_pose_prob: float = 0.0,
        randomize_trajectory_prob: float = 0.0,
        termination_config=None,
        reward_config=None,
        is_target_fixed: bool = True,
        fixed_object_poses_path: str = None,
        save_object_poses_path: str = None,
    ):
        if n_objects != 2:
            logger.warning("PickPlaceHardTask always uses exactly two strawberries; ignoring n_objects=%s.", n_objects)
        super().__init__(
            show_trajectory=show_trajectory,
            n_objects=2,
            randomize_initial_pose=randomize_initial_pose,
            randomize_robot_pose_prob=randomize_robot_pose_prob,
            randomize_trajectory_prob=randomize_trajectory_prob,
            termination_config=termination_config,
            reward_config=reward_config,
            is_target_fixed=is_target_fixed,
            fixed_object_poses_path=fixed_object_poses_path,
            save_object_poses_path=save_object_poses_path,
        )

    @property
    def default_robot_joint_pos(self) -> Optional[torch.Tensor]:
        return None

    def _default_robot_joint_pos_for_robot(self, robot) -> Optional[torch.Tensor]:
        robot_model = str(getattr(robot, "model", "")).lower()
        if robot_model == "r1pro":
            return torch.cat([
                torch.zeros(6, dtype=torch.float32),
                torch.tensor([-0.63, 1.65, 1.05, 0.00]),
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.57, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                torch.ones(4, dtype=torch.float32) * 0.05,
            ])
        if robot_model == "a1":
            return torch.tensor([0.0, 1.877, -0.945, 1.570, 0.931, 0.0, 0.03, 0.03])
        return None

    def _load(self, env):
        for obj_config in self.object_config:
            obj = env.scene.object_registry("name", obj_config["name"])
            if obj is None:
                obj = create_class_from_registry_and_config(
                    cls_name=obj_config["type"],
                    cls_registry=REGISTERED_OBJECTS,
                    cfg=obj_config,
                    cls_type_descriptor="object",
                )
                env.scene.add_object(obj)

            obj_pos = torch.tensor([0.0, 0.0, 0.0]) if "position" not in obj_config else torch.tensor(obj_config["position"])
            obj_orn = torch.tensor([0.0, 0.0, 0.0, 1.0]) if "orientation" not in obj_config else torch.tensor(obj_config["orientation"])
            obj_pos, obj_orn = T.pose_transform(*env.scene.get_position_orientation(), obj_pos, obj_orn)
            obj.set_position_orientation(obj_pos, obj_orn)

        if og.sim.viewer_camera is not None:
            og.sim.viewer_camera.set_position_orientation(*self.viewer_camera_pose)

        default_joint_pos = self._default_robot_joint_pos_for_robot(env.robots[0])
        if default_joint_pos is not None:
            env.robots[0]._reset_joint_pos = default_joint_pos
        BehaviorTask._load(self, env)

    @property
    def bddl(self) -> str:
        return """
        (define (problem pick_and_place_hard-0)
            (:domain omnigibson)
            (:objects
                object.n.01_1 object.n.01_2 - object.n.01
                chopping_board.n.01_1 - chopping_board.n.01
                hamper.n.02_1 - hamper.n.02
                agent.n.01_1 - agent.n.01
            )
            (:init 
                (ontop object.n.01_1 chopping_board.n.01_1)
                (ontop object.n.01_2 chopping_board.n.01_1)
                (ontop hamper.n.02_1 chopping_board.n.01_1) 
            )
            (:goal
                (and
                    (ontop object.n.01_1 hamper.n.02_1)
                    (ontop object.n.01_2 hamper.n.02_1)
                )
            )
        )
        """

    @property
    def object_config(self):
        cfg = super().object_config
        cfg[0]["position"] = [0.45, -0.14, 0.043]
        cfg[1]["position"] = [0.45, 0.00, 0.043]
        return cfg

    def post_play_load(self, env) -> None:
        super().post_play_load(env)
        self._set_green_strawberry_material()

    def _set_green_strawberry_material(self) -> None:
        green = torch.tensor([0.24, 0.74, 0.24], dtype=torch.float32)
        for material in self.objects[1].materials:
            try:
                material.diffuse_tint = green
            except (AttributeError, NotImplementedError):
                pass
            try:
                material.diffuse_color_constant = green
            except (AttributeError, NotImplementedError):
                pass

    def _next_object_idx(self) -> int:
        if not self.objects[0].states[OnTop].get_value(self.target):
            return 0
        return 1

    def _uses_right_arm_api(self) -> bool:
        return self.robot is not None and "right" in getattr(self.robot, "eef_link_names", {})

    def _manipulation_arm(self) -> str:
        return "right" if self._uses_right_arm_api() else self.robot.default_arm

    def _eef_link_name(self) -> str:
        return self.robot.eef_link_names[self._manipulation_arm()]

    def _base_link_name(self) -> Optional[str]:
        return getattr(self.robot, "base_footprint_link_name", None) if self._uses_right_arm_api() else None

    def _eef_position(self) -> torch.Tensor:
        return self.robot.get_eef_position(self._manipulation_arm()) if self._uses_right_arm_api() else self.robot.get_eef_position()

    def _is_grasping(self, obj) -> bool:
        if self._uses_right_arm_api():
            return self.robot.is_grasping(arm=self._manipulation_arm(), candidate_obj=obj) == 1
        return self.robot.is_grasping(candidate_obj=obj) == 1

    def _set_gripper_intent(self, intent: dict[str, Any], value: float) -> None:
        if self._uses_right_arm_api():
            intent["gripper_left"] = 1.0
            intent["gripper_right"] = value
        else:
            intent["gripper_0"] = value

    def _target_action_type(self) -> ActionTypesSelection:
        return ActionTypesSelection.RIGHT_ARM if self._uses_right_arm_api() else ActionTypesSelection.DEFAULT

    def _add_base_target(self, intent: dict[str, Any]) -> None:
        base_link_name = self._base_link_name()
        if base_link_name is None:
            return
        init_container_pos = self.init_container.get_position_orientation()[0]
        base_pos = torch.tensor([init_container_pos[0] - 0.7, init_container_pos[1], 0.0])
        base_quat = T.euler2quat(torch.zeros(3))
        intent["target_pos"][base_link_name] = base_pos
        intent["target_quat"][base_link_name] = base_quat

    def step(self, env, action) -> Tuple[float, bool, dict]:
        reward, done, info = super(PickPlaceTask, self).step(env=env, action=action)
        subtask_id, current_intent = self.get_current_task_state_and_intent()
        info["object_idx"] = self.object_idx
        info["subtask_id"] = subtask_id
        info["intent"] = current_intent
        if subtask_id in [4, 5]:
            attached_link_name = "right_eef_link" if self._uses_right_arm_api() else "eef_link"
            info["attached_obj"] = {attached_link_name: self.objects[self.object_idx].root_link}
            info["attached_obj_scale"] = {attached_link_name: 0.9}
        self.task_info = info
        return reward, done, info

    def get_current_task_state_and_intent(self) -> Tuple[int, dict[str, Any]]:
        self.object_idx = self._next_object_idx()
        if self._randomize_trajectory_prob > 0.0 and np.random.rand() < self._randomize_trajectory_prob:
            pos_noise_std, quat_noise_std = 0.01, 0.03
        else:
            pos_noise_std, quat_noise_std = 0.0, 0.0

        intent = defaultdict(lambda: None)
        intent["replan"] = False
        intent["action_type"] = self._target_action_type()
        eef_link_name = self._eef_link_name()
        base_quat = torch.tensor([0.9239, 0.0, 0.3827, 0.0])
        current_object = self.objects[self.object_idx]

        if current_object.states[OnTop].get_value(self.target):
            self._in_abnormal_state = False
            subtask_id = 7
            self._set_gripper_intent(intent, 1.0)
        elif self._is_grasping(current_object):
            self._in_abnormal_state = False
            if torch.linalg.norm(self._eef_position()[:2] - self.target_pose[:2] - torch.tensor([-0.015, 0.0])) < 0.03:
                subtask_id = 6
                self._set_gripper_intent(intent, 1.0)
            elif not current_object.states[OnTop].get_value(self.init_container):
                subtask_id = 5
                intent["target_pos"] = {
                    eef_link_name: add_position_noise(torch.tensor([-0.015, 0.0, 0.08]) + self.target_pose[:3], noise_std=pos_noise_std)
                }
                intent["target_quat"] = {eef_link_name: add_quaternion_noise(base_quat, noise_std=quat_noise_std)}
                self._set_gripper_intent(intent, -1.0)
            else:
                subtask_id = 4
                intent["target_pos"] = {eef_link_name: torch.tensor([-0.015, 0.0, 0.05]) + self.object_pose[self.object_idx][:3]}
                intent["target_quat"] = {eef_link_name: base_quat}
                self._set_gripper_intent(intent, -1.0)
                intent["ik_only"] = True
        elif torch.linalg.norm(self._eef_position() - self.object_pose[self.object_idx][:3] - torch.tensor([-0.015, 0.0, 0.022])) < 0.02:
            self._in_abnormal_state = False
            subtask_id = 3
            self._set_gripper_intent(intent, -1.0)
        else:
            if torch.sum(self.robot.get_joint_positions()[-2:]) > 0.05:
                self._set_gripper_intent(intent, 1.0)
                if torch.linalg.norm(self._eef_position()[:2] - self.object_pose[self.object_idx][:2] - torch.tensor([-0.015, 0.0])) < 0.03:
                    subtask_id = 2
                    intent["target_pos"] = {eef_link_name: torch.tensor([-0.015, 0.0, 0.022]) + self.object_pose[self.object_idx][:3]}
                    intent["target_quat"] = {eef_link_name: base_quat}
                    intent["ik_only"] = True
                else:
                    subtask_id = 1
                    intent["target_pos"] = {
                        eef_link_name: add_position_noise(torch.tensor([-0.015, 0.0, 0.05]) + self.object_pose[self.object_idx][:3], noise_std=pos_noise_std)
                    }
                    intent["target_quat"] = {eef_link_name: add_quaternion_noise(base_quat, noise_std=quat_noise_std)}
                    self._add_base_target(intent)
            else:
                if not self._in_abnormal_state:
                    intent["replan"] = True
                self._in_abnormal_state = True
                subtask_id = 0
                self._set_gripper_intent(intent, 1.0)
        return subtask_id, intent

    def reset(self, env) -> None:
        self._in_abnormal_state = False
        super(PickPlaceTask, self).reset(env)
        if self.robot is not None:
            logger.info(f"Episode {env._current_episode + 1}.")
            if np.random.rand() < self._randomize_robot_pose_prob:
                og.sim.step()
                action = self.robot.get_no_op_action()
                action += np.random.uniform(-0.15, 0.15, size=action.shape)
                for _ in range(10):
                    env.step(action, False)
        if self.randomize_initial_pose:
            self._sample_new_target(env)
        self.task_info = dict()

    def _sample_new_target(self, env) -> None:
        super()._sample_new_target(env)
        self.object_idx = 0

    def _get_obs(self, env):
        self.object_pose = torch.stack([torch.cat(self.objects[i].get_position_orientation()) for i in range(self._n_objects)])
        rel_object_pose = self.object_pose.clone()
        if self.robot is not None and self._uses_right_arm_api():
            rel_object_pose[:, :3] -= self.robot.get_eef_position("right")
        else:
            rel_object_pose[:, :3] -= env.robots[0].get_eef_position()
        self.target_pose = torch.cat(self.target.get_position_orientation())
        obs = {
            "object_pose": torch.cat((torch.unsqueeze(self.target_pose[:3], dim=0), rel_object_pose[:, :3]), dim=0).flatten(),
        }
        return dict(), obs

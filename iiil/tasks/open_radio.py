import math
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch
from collections import defaultdict
from iiil.tasks.task_base import BaseTask
from iiil.utils.goal_utils import BlockEnd, EpisodeFail, PredicateGoal, Timeout
from iiil.utils.misc_utils import ActionTypesSelection
from omnigibson.macros import macros
from omnigibson.object_states import OnTop, Open, ToggledOn
from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.bddl_utils import BDDLEntity
from omnigibson.utils.constants import JointType
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import classproperty, create_class_from_registry_and_config
from omnigibson.utils.ui_utils import create_module_logger
from typing import Any, Optional, Tuple


logger = create_module_logger("OpenRadioTask")
logger.setLevel("INFO")

RADIO_BASE_TARGET_Y_THRESHOLD = 0.02
RADIO_BASE_TARGET_X_LOWER_MARGIN = 0.01
RETRACT_SKIP_TO_RADIO_Y_THRESHOLD = 0.15
RETRACT_SKIP_TO_RADIO_X_LOWER_MARGIN = 0.08
TORSO_RETRACT_MAX_STEP = 0.05
TORSO_RESET_THRESHOLD = 0.04

SUBTASK_LABELS = {
    0: "open fridge: open gripper",
    1: "open fridge: move to door pregrasp",
    2: "open fridge: reach door handle",
    3: "open fridge: grasp door handle",
    4: "open fridge: pull door open",
    6: "release fridge handle",
    7: "retract after opening fridge",
    8: "retract: reset torso",
    9: "radio: open gripper",
    10: "radio: navigate to table",
    11: "radio: reach top pregrasp",
    12: "radio: reach handle",
    13: "radio: grasp handle",
    14: "radio: pick up",
    15: "radio: success",
}


with macros.unlocked():
    macros.object_states.open_state.JOINT_THRESHOLD_BY_TYPE[JointType.JOINT_REVOLUTE] = 0.6


class OpenRadioTask(BaseTask, BehaviorTask):
    def __init__(
        self,
        show_trajectory: bool = True,
        randomize_robot_pose_prob: float = 0.0,
        randomize_initial_pose_prob: float = 0.0,
        termination_config=None,
        reward_config=None,
    ):
        self._show_trajectory = show_trajectory
        self._randomize_robot_pose_prob = randomize_robot_pose_prob
        self._randomize_initial_pose_prob = randomize_initial_pose_prob
        self._did_retract_after_open = False
        self.task_info = dict()

        super().__init__(
            self,
            predefined_problem=self.bddl,
            online_object_sampling=True,
            use_presampled_robot_pose=False,
            termination_config=termination_config,
            reward_config=reward_config,
        )

    @property
    def object_config(self):
        return [
            {
                "type": "DatasetObject",
                "name": "fridge",
                "category": "fridge",
                "model": "dszchb",
                "position": [1.0, 0.5, 0.86],
                "orientation": [0, 0, 1, 0],
                "fixed_base": True,
            },
            {
                "type": "DatasetObject",
                "name": "table",
                "category": "breakfast_table",
                "model": "rhjoby",
                "position": [1.0, -1.5, 0.8],
                "orientation": [0, 0, 0, 1],
                "bounding_box": [0.61, 0.76, 0.7],
                "fixed_base": True,
            },
            {
                "type": "DatasetObject",
                "name": "radio",
                "category": "radio",
                "position": [1.0, -1.5, 0.7],
                "orientation": [0, 0, 0, 1],
                "scale": 0.68,
                "model": "wxnicr",
            },
        ]

    @property
    def viewer_camera_pose(self):
        return [-1.1524, -2.5555, 2.2496], [0.5009, -0.1229, -0.2041, 0.8321]

    @property
    def default_robot_joint_pos(self) -> Optional[torch.Tensor]:
        return torch.cat([
            torch.zeros(6, dtype=torch.float32),
            torch.tensor([-0.63, 1.65, 1.05, 0.00]),
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.57, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.ones(4, dtype=torch.float32) * 0.05,
        ])

    @property
    def bddl(self) -> str:
        return """
        (define (problem open_fridge_pick_radio-0)
            (:domain omnigibson)
            (:objects
                agent.n.01_1 - agent.n.01
                electric_refrigerator.n.01_1 - electric_refrigerator.n.01
                breakfast_table.n.01_1 - breakfast_table.n.01
                radio_receiver.n.01_1 - radio_receiver.n.01
            )
            (:init 
                (not (open electric_refrigerator.n.01_1))
                (not (toggled_on radio_receiver.n.01_1))
                (ontop radio_receiver.n.01_1 breakfast_table.n.01_1)
            )
            (:goal
                (and
                    (open electric_refrigerator.n.01_1)
                    (toggled_on radio_receiver.n.01_1)
                )
            )
        )
        """

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
        if self.default_robot_joint_pos is not None:
            env.robots[0]._reset_joint_pos = self.default_robot_joint_pos
        BehaviorTask._load(self, env)

    def post_play_load(self, env) -> None:
        self.fridge = env.scene.object_registry("name", "fridge")
        self.table = env.scene.object_registry("name", "table")
        self.radio = env.scene.object_registry("name", "radio")
        for joint in self.fridge.joints.values():
            joint.friction = 0.2
        self.radio.links["base_link"].mass = 1.0

        env.task.object_scope["agent.n.01_1"] = BDDLEntity(bddl_inst="agent.n.01_1", entity=env.robots[0])
        env.task.object_scope["electric_refrigerator.n.01_1"] = BDDLEntity(
            bddl_inst="electric_refrigerator.n.01_1", entity=self.fridge
        )
        env.task.object_scope["breakfast_table.n.01_1"] = BDDLEntity(
            bddl_inst="breakfast_table.n.01_1", entity=self.table
        )
        env.task.object_scope["radio_receiver.n.01_1"] = BDDLEntity(
            bddl_inst="radio_receiver.n.01_1", entity=self.radio
        )
        self.fridge_pose = torch.zeros(7)
        self.fridge_door_pose = torch.zeros(7)
        self.table_pose = torch.zeros(7)
        self.radio_pose = torch.zeros(7)

        super().post_play_load(env)

    def step(self, env, action) -> Tuple[float, bool, dict]:
        reward, done, info = super().step(env=env, action=action)
        subtask_id, current_intent = self.get_current_task_state_and_intent()
        info["subtask_id"] = subtask_id
        info["intent"] = current_intent
        info["phase"] = self._current_phase()
        logger.debug(
            f"[open_radio step {getattr(env, 'episode_steps', '?')}] "
            f"phase={info['phase']} subtask={subtask_id} "
            f"({SUBTASK_LABELS.get(subtask_id, 'unknown')})"
        )

        if subtask_id in [3, 4]:
            info["attached_obj"] = {"right_eef_link": self.fridge.links["link_0"]}
            info["attached_obj_scale"] = {"right_eef_link": 0.6}
        elif subtask_id in [13, 14]:
            info["attached_obj"] = {"right_eef_link": self.radio.links["base_link"]}
            info["attached_obj_scale"] = {"right_eef_link": 0.1}

        self.task_info = info
        return reward, done, info

    def get_current_task_state_and_intent(self) -> Tuple[int, dict[str, Any]]:
        self._sync_grasping_mode_with_fridge()
        if not self.fridge.states[Open].get_value():
            return self._get_open_intent()
        if self.robot.is_grasping(arm="right", candidate_obj=self.fridge) == 1:
            return self._get_release_fridge_intent()
        if not self._did_retract_after_open:
            if not self._torso_at_reset():
                return self._get_retract_torso_intent()
            elif self._arms_retracted() or self._near_radio_base_target_for_retract_skip():
                self._did_retract_after_open = True
            else:
                return self._get_retract_intent()
        radio_subtask_id, intent = self._get_radio_intent()
        return radio_subtask_id + 10, intent

    def _current_phase(self) -> str:
        if not self.fridge.states[Open].get_value():
            return "open"
        if self.robot.is_grasping(arm="right", candidate_obj=self.fridge) == 1:
            return "release"
        if not self._did_retract_after_open:
            return "retract"
        return "radio"

    def _sync_grasping_mode_with_fridge(self) -> None:
        if self.robot is None:
            return
        target_mode = "sticky" if self.fridge.states[Open].get_value() else "assisted"
        if self.robot.grasping_mode != target_mode:
            self.robot.grasping_mode = target_mode

    def _right_arm_retract_pose_for_base(self, base_pos: torch.Tensor, base_quat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return T.pose_transform(
            base_pos,
            base_quat,
            torch.tensor([0.430, -0.252, 0.917]),
            T.euler2quat(torch.tensor([0.0, math.pi / 2, 0.0])),
        )

    def _reset_torso_joint_pos(self) -> torch.Tensor:
        return self.default_robot_joint_pos[6:10]

    def _current_torso_joint_pos(self) -> torch.Tensor:
        joint_pos = self.robot.get_joint_positions()
        if joint_pos.shape[0] >= 28:
            return joint_pos[6:10]
        return joint_pos[3:7]

    def _torso_at_reset(self) -> bool:
        return torch.max(torch.abs(self._current_torso_joint_pos() - self._reset_torso_joint_pos())) < TORSO_RESET_THRESHOLD

    def _get_retract_torso_intent(self) -> Tuple[int, dict[str, Any]]:
        intent = defaultdict(lambda: None)
        current_torso = self._current_torso_joint_pos()
        reset_torso = self._reset_torso_joint_pos()
        next_torso = current_torso + torch.clamp(
            reset_torso - current_torso,
            min=-TORSO_RETRACT_MAX_STEP,
            max=TORSO_RETRACT_MAX_STEP,
        )
        action = self.robot.get_no_op_action().clone()
        action[3:7] = next_torso
        intent["action_type"] = ActionTypesSelection.RIGHT_ARM
        intent["direct_action"] = action
        intent["gripper_left"] = 1.0
        intent["gripper_right"] = 1.0
        return 8, intent

    def _retract_targets(self) -> Tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        right_eef_link_name = self.robot.eef_link_names["right"]
        base_link_name = self.robot.base_footprint_link_name
        robot_pos, _ = self.robot.get_position_orientation()
        robot_pos[0] = -0.7
        base_quat = T.euler2quat(torch.tensor([0.0, 0.0, 0.0]))
        right_pos, right_quat = self._right_arm_retract_pose_for_base(robot_pos, base_quat)
        target_pos = {
            right_eef_link_name: right_pos,
            base_link_name: robot_pos,
        }
        target_quat = {
            right_eef_link_name: right_quat,
            base_link_name: base_quat,
        }
        return target_pos, target_quat

    def _arms_retracted(self) -> bool:
        target_pos, _ = self._retract_targets()
        right_eef_link_name = self.robot.eef_link_names["right"]
        return torch.linalg.norm(self.robot.get_eef_position("right") - target_pos[right_eef_link_name]) < 0.25

    def _get_retract_intent(self) -> Tuple[int, dict[str, Any]]:
        intent = defaultdict(lambda: None)
        intent["action_type"] = ActionTypesSelection.RIGHT_ARM
        intent["ik_only"] = True
        intent["target_pos"], intent["target_quat"] = self._retract_targets()
        intent["gripper_left"] = 1.0
        intent["gripper_right"] = 1.0
        return 7, intent

    def _get_release_fridge_intent(self) -> Tuple[int, dict[str, Any]]:
        intent = defaultdict(lambda: None)
        intent["action_type"] = ActionTypesSelection.BODY
        intent["gripper_left"] = 1.0
        intent["gripper_right"] = 1.0
        return 6, intent

    def _get_radio_base_target(self) -> Tuple[torch.Tensor, torch.Tensor]:
        table_local_pose = (
            torch.tensor([-0.7, 0.2, -0.6681]),
            T.euler2quat(torch.zeros(3)),
        )
        return T.pose_transform(
            torch.tensor([self.table_pose[0], self.radio_pose[1], self.table_pose[2]]),
            self.table_pose[3:],
            *table_local_pose,
        )

    def _at_radio_base_target(self) -> bool:
        return self._near_radio_base_target(
            y_threshold=RADIO_BASE_TARGET_Y_THRESHOLD,
            x_lower_margin=RADIO_BASE_TARGET_X_LOWER_MARGIN,
        )

    def _near_radio_base_target_for_retract_skip(self) -> bool:
        return self._near_radio_base_target(
            y_threshold=RETRACT_SKIP_TO_RADIO_Y_THRESHOLD,
            x_lower_margin=RETRACT_SKIP_TO_RADIO_X_LOWER_MARGIN,
        )

    def _near_radio_base_target(self, y_threshold: float, x_lower_margin: float) -> bool:
        base_pos, _ = self._get_radio_base_target()
        cur_robot_pos = self.robot.get_position_orientation()[0]
        return abs(cur_robot_pos[1] - base_pos[1]) < y_threshold and cur_robot_pos[0] > base_pos[0] - x_lower_margin

    def _radio_y_aligned(self) -> bool:
        base_pos, _ = self._get_radio_base_target()
        cur_robot_pos = self.robot.get_position_orientation()[0]
        return abs(cur_robot_pos[1] - base_pos[1]) < RADIO_BASE_TARGET_Y_THRESHOLD

    def _get_staged_radio_base_target(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base_pos, base_quat = self._get_radio_base_target()
        if not self._radio_y_aligned():
            cur_robot_pos = self.robot.get_position_orientation()[0]
            base_pos = base_pos.clone()
            base_pos[0] = cur_robot_pos[0]
        return base_pos, base_quat

    def _get_open_intent(self) -> Tuple[int, dict[str, Any]]:
        intent = defaultdict(lambda: None)
        intent["action_type"] = ActionTypesSelection.BODY
        right_eef_link_name = self.robot.eef_link_names["right"]
        base_link_name = self.robot.base_footprint_link_name
        if self.robot.is_grasping(arm="right", candidate_obj=self.fridge) == 1:
            subtask_id = 4
            fridge_door_open_local_pose = (
                torch.tensor([0.97, 0.35, 0.15]),
                T.euler2quat(torch.tensor([math.pi / 2, math.pi / 2, 0])),
            )
            right_hand_pos, right_hand_quat = T.pose_transform(
                self.fridge_pose[:3], self.fridge_pose[3:], *fridge_door_open_local_pose
            )
            intent["target_pos"] = {right_eef_link_name: right_hand_pos}
            intent["target_quat"] = {right_eef_link_name: right_hand_quat}
            intent["action_type"] = ActionTypesSelection.RIGHT_ARM
            intent["ik_only"] = True
            intent["gripper_left"] = 1.0
            intent["gripper_right"] = -1.0
        else:
            fridge_door_local_pose = (
                torch.tensor([0.06, -0.57, 0.00]),
                T.euler2quat(torch.tensor([0, math.pi / 2, math.pi])),
            )
            fridge_door_pregrasp_local_pose = (
                torch.tensor([0.11, -0.57, 0.00]),
                fridge_door_local_pose[1],
            )
            right_hand_pos, right_hand_quat = T.pose_transform(
                self.fridge_door_pose[:3], self.fridge_door_pose[3:], *fridge_door_local_pose
            )
            right_hand_pregrasp_pos, right_hand_pregrasp_quat = T.pose_transform(
                self.fridge_door_pose[:3], self.fridge_door_pose[3:], *fridge_door_pregrasp_local_pose
            )
            if torch.linalg.norm(self.robot.get_eef_position("right") - right_hand_pos) < 0.025:
                subtask_id = 3
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = -1.0
            else:
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = 1.0
                fridge_base_local_pose = (
                    torch.tensor([0.69, -0.7, -1.00]),
                    T.euler2quat(torch.tensor([0.0, 0.0, math.pi])),
                )
                base_pos, base_quat = T.pose_transform(
                    self.fridge_door_pose[:3], self.fridge_door_pose[3:], *fridge_base_local_pose
                )
                gripper_open = torch.sum(self.robot.get_joint_positions()[-2:]) > 0.06
                base_at_target = torch.linalg.norm(self.robot.get_position_orientation()[0] - base_pos) < 0.01
                pregrasp_at_target = torch.linalg.norm(self.robot.get_eef_position("right") - right_hand_pregrasp_pos) < 0.01
                if not gripper_open:
                    subtask_id = 0
                elif base_at_target and pregrasp_at_target:
                    subtask_id = 2
                    intent["action_type"] = ActionTypesSelection.RIGHT_ARM
                    intent["target_pos"] = {
                        right_eef_link_name: right_hand_pos,
                        base_link_name: base_pos,
                    }
                    intent["target_quat"] = {
                        right_eef_link_name: right_hand_quat,
                        base_link_name: base_quat,
                    }
                else:
                    subtask_id = 1
                    intent["action_type"] = ActionTypesSelection.RIGHT_ARM
                    intent["ik_only"] = True
                    intent["target_pos"] = {
                        right_eef_link_name: right_hand_pregrasp_pos,
                        base_link_name: base_pos,
                    }
                    intent["target_quat"] = {
                        right_eef_link_name: right_hand_pregrasp_quat,
                        base_link_name: base_quat,
                    }
        return subtask_id, intent

    def _get_radio_intent(self) -> Tuple[int, dict[str, Any]]:
        intent = defaultdict(lambda: None)
        intent["action_type"] = ActionTypesSelection.BODY
        right_eef_link_name = self.robot.eef_link_names["right"]
        left_eef_link_name = self.robot.eef_link_names["left"]
        base_link_name = self.robot.base_footprint_link_name
        gripper_open = torch.sum(self.robot.get_joint_positions()[-2:]) > 0.06
        if self.robot.is_grasping(arm="right", candidate_obj=self.radio) == 1:
            if self.radio.get_position_orientation()[0][2] > self.table_pose[2] + 0.2:
                subtask_id = 5
                self.radio.states[ToggledOn].set_value(True)
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = -1.0
            else:
                subtask_id = 4
                right_hand_pos, right_hand_quat = self.robot.get_eef_pose("right")
                left_hand_pos, left_hand_quat = self.robot.get_eef_pose("left")
                right_hand_pos[2] += 0.2
                intent["action_type"] = ActionTypesSelection.BODY
                intent["ik_only"] = True
                intent["target_pos"] = {
                    right_eef_link_name: right_hand_pos,
                    left_eef_link_name: left_hand_pos,
                }
                intent["target_quat"] = {
                    right_eef_link_name: right_hand_quat,
                    left_eef_link_name: left_hand_quat,
                }
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = -1.0
        else:
            target_handle_local_pose = (
                torch.tensor([-0.02, 0.06, 0.07]),
                T.euler2quat(torch.tensor([0, math.pi / 2, -math.pi / 2])),
            )
            target_top_local_pose = (
                torch.tensor([-0.02, 0.11, 0.12]),
                target_handle_local_pose[1],
            )
            right_hand_pos, right_hand_quat = T.pose_transform(
                self.radio_pose[:3], self.radio_pose[3:], *target_handle_local_pose
            )
            right_hand_top_pos, right_hand_top_quat = T.pose_transform(
                self.radio_pose[:3], self.radio_pose[3:], *target_top_local_pose
            )
            if not gripper_open:
                subtask_id = -1
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = 1.0
            elif torch.linalg.norm(self.robot.get_eef_position("right") - right_hand_pos) < 0.02:
                subtask_id = 3
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = -1.0
            else:
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = 1.0
                base_pos, base_quat = self._get_radio_base_target()
                if self._at_radio_base_target():
                    pregrasp_at_target = torch.linalg.norm(self.robot.get_eef_position("right") - right_hand_top_pos) < 0.04
                    subtask_id = 2 if pregrasp_at_target else 1
                    intent["action_type"] = ActionTypesSelection.BODY
                    left_hand_pos, left_hand_quat = self.robot.get_eef_pose("left")
                    left_hand_pos[0] = right_hand_pos[0] if pregrasp_at_target else right_hand_top_pos[0]
                    intent["target_pos"] = {
                        right_eef_link_name: right_hand_pos if pregrasp_at_target else right_hand_top_pos,
                        left_eef_link_name: left_hand_pos,
                    }
                    intent["target_quat"] = {
                        right_eef_link_name: right_hand_quat if pregrasp_at_target else right_hand_top_quat,
                        left_eef_link_name: left_hand_quat,
                    }
                else:
                    subtask_id = 0
                    base_pos, base_quat = self._get_staged_radio_base_target()
                    right_retract_pos, right_retract_quat = self._right_arm_retract_pose_for_base(base_pos, base_quat)
                    intent["action_type"] = ActionTypesSelection.RIGHT_ARM
                    intent["ik_only"] = True
                    intent["target_pos"] = {
                        right_eef_link_name: right_retract_pos,
                        base_link_name: base_pos,
                    }
                    intent["target_quat"] = {
                        right_eef_link_name: right_retract_quat,
                        base_link_name: base_quat,
                    }
        return subtask_id, intent

    def reset(self, env) -> None:
        self._did_retract_after_open = False
        super().reset(env)
        if self.robot is not None:
            logger.info(f"Episode {env._current_episode + 1}.")
            if np.random.rand() < self._randomize_robot_pose_prob:
                og.sim.step()
                robot_initial_pos = torch.tensor([-1.0, 0.0, 0.0]) + torch.cat(
                    ((torch.rand(2) - 0.5) * 0.3, torch.tensor([0.0]))
                )
                robot_initial_orn = T.euler2quat(torch.tensor([0.0, 0.0, np.random.uniform(-math.pi / 10, math.pi / 10)])) # 36 degrees total
                self.robot.set_position_orientation(robot_initial_pos, robot_initial_orn)
        self._sample_new_target(env)
        self.task_info = dict()
        self._sync_grasping_mode_with_fridge()

    def _sample_new_target(self, env) -> None:
        if np.random.rand() < self._randomize_initial_pose_prob:
            x_noise = np.random.uniform(-0.1, 0.1)
            fridge_y = 0.5 + np.random.uniform(-0.05, 0.05)
            table_y = -1.5 + np.random.uniform(-0.05, 0.05)
            fridge_pos = torch.tensor([1.0 + x_noise, fridge_y, 0.86])
            table_pos = torch.tensor([1.0 + x_noise, table_y, 0.67])
        else:
            fridge_pos = torch.tensor([1.0, 0.5, 0.86])
            table_pos = torch.tensor([1.0, -1.5, 0.67])

        self.fridge_pose = torch.cat((fridge_pos, T.euler2quat(torch.tensor([0.0, 0.0, -math.pi]))))
        self.fridge.set_position_orientation(self.fridge_pose[:3], self.fridge_pose[3:])
        self.fridge.set_joint_positions(torch.zeros(1))

        self.table_pose = torch.cat((table_pos, T.euler2quat(torch.zeros(3))))
        self.table.set_position_orientation(self.table_pose[:3], self.table_pose[3:])
        self.radio.states[OnTop].set_value(self.table, True)
        self.radio.set_position_orientation(orientation=[0, 0, 0.7071, 0.7071])
        self.radio.keep_still()
        self.radio.states[ToggledOn].set_value(False)

    def _load_non_low_dim_observation_space(self):
        obs_space = dict()
        obs_space["objects"] = self._build_obs_box_space(
            shape=(28,), low=-float("inf"), high=float("inf"), dtype=NumpyTypes.FLOAT32
        )
        return obs_space

    def _get_obs(self, env):
        self.fridge_pose = torch.cat(self.fridge.get_position_orientation())
        self.fridge_door_pose = torch.cat(self.fridge.links["link_0"].get_position_orientation())
        self.table_pose = torch.cat(self.table.get_position_orientation())
        self.radio_pose = torch.cat(self.radio.get_position_orientation())
        obs = {
            "objects": torch.cat([self.fridge_pose, self.fridge_door_pose, self.table_pose, self.radio_pose]),
        }
        return dict(), obs

    def _create_termination_conditions(self):
        terminations = dict()
        terminations["timeout"] = BlockEnd(
            total_episode=self._termination_config.get("total_episode"),
            total_successful_episode=self._termination_config.get("total_successful_episode"),
        )
        terminations["max_time"] = Timeout(episode_timeout_step=self._termination_config["episode_timeout_step"])
        terminations["predicate"] = PredicateGoal(
            goal_fcn=lambda: self.activity_goal_conditions,
            hold_step_threshold=self._termination_config["hold_step_threshold"],
        )
        terminations["task_fail"] = EpisodeFail()
        return terminations

    @classproperty
    def default_termination_config(cls):
        return {
            "hold_step_threshold": 15,
            "episode_timeout_step": 800,
            "total_episode": 40,
            "total_successful_episode": None,
        }

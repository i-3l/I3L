import math
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch
from collections import defaultdict
from iiil.utils.misc_utils import ActionTypesSelection
from iiil.utils.goal_utils import BlockEnd, Timeout, PredicateGoal, EpisodeFail
from iiil.tasks.task_base import BaseTask
from omnigibson.macros import macros
from omnigibson.object_states import Open
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.bddl_utils import BDDLEntity
from omnigibson.utils.constants import JointType
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.ui_utils import create_module_logger
from typing import Any, Tuple, Optional

logger = create_module_logger("OpenTask")
logger.setLevel("INFO")


with macros.unlocked():
    macros.object_states.open_state.JOINT_THRESHOLD_BY_TYPE[JointType.JOINT_REVOLUTE] = 0.6


class OpenTask(BaseTask, BehaviorTask):
    def __init__(
        self, 
        show_trajectory: bool=True,
        randomize_robot_pose_prob: float=0.0,
        randomize_initial_pose_prob: float=0.0,
        termination_config=None, 
        reward_config=None,
    ):
        self._show_trajectory = show_trajectory
        self._randomize_robot_pose_prob = randomize_robot_pose_prob
        self._randomize_initial_pose_prob = randomize_initial_pose_prob

        # extra info to be passed in to the step function
        self.task_info = dict()

        super().__init__(
            self, 
            predefined_problem=self.bddl,
            online_object_sampling=True,
            use_presampled_robot_pose=False,
            termination_config=termination_config,
            reward_config=reward_config
        )

    @property
    def object_config(self):
        cfg = [{
            "type": "DatasetObject",
            "name": "target",
            "category": "fridge",
            "model": "dszchb",
            "position": [1.0, 0.0, 0.86],
            "orientation": [0, 0, 1, 0],
            "fixed_base": True,
        }]
        return cfg
    
    @property
    def viewer_camera_pose(self):
        return [-1.1524, -2.5555,  2.2496], [0.5009, -0.1229, -0.2041,  0.8321]

    @property
    def default_robot_joint_pos(self) -> Optional[torch.Tensor]:
        """
        Default robot joint positions upon startup. 
        If return None, will use robot's default reset joint positions.
        
        Returns:
            Optional[torch.tensor]: optional default robot joint positions.
        """
        return torch.cat([
            torch.zeros(6, dtype=torch.float32),    # base
            torch.tensor([-0.63, 1.65, 1.05, 0.00]),  # torso
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.57, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # arm
            torch.ones(4, dtype=torch.float32) * 0.05    # grippers
        ])

    @property
    def bddl(self) -> str:
        return f"""
        (define (problem open_fridge-0)
            (:domain omnigibson)
            (:objects
                agent.n.01_1 - agent.n.01
                electric_refrigerator.n.01_1 - electric_refrigerator.n.01
            )
            (:init 
                (not 
                    (open electric_refrigerator.n.01_1)
                )
            )
            (:goal
                (and
                    (open electric_refrigerator.n.01_1)
                )
            )
        )
        """

    def _load(self, env):
        BaseTask._load(self, env)
        BehaviorTask._load(self, env)

    def post_play_load(self, env) -> None:
        self.target = env.scene.object_registry("name", "target")
        # decrease the friction of the target door joint
        for j in self.target.joints:
            self.target.joints[j].friction = 0.2
        # assign object scope
        env.task.object_scope["agent.n.01_1"] = BDDLEntity(bddl_inst="agent.n.01_1", entity=env.robots[0])
        env.task.object_scope["electric_refrigerator.n.01_1"] = BDDLEntity(bddl_inst="electric_refrigerator.n.01_1", entity=self.target)
        self.target_pose = torch.zeros(7)
        self.target_door_pose = torch.zeros(7)

        super().post_play_load(env)
    
    def step(self, env, action) -> Tuple[float, bool, dict]:
        # Run super method first
        reward, done, info = super().step(env=env, action=action)
        # update task info
        subtask_id, current_intent = self.get_current_task_state_and_intent()
        info["subtask_id"] = subtask_id
        info["intent"] = current_intent
        if subtask_id in [3, 4]:
            info["attached_obj"] = {"right_eef_link": self.target.links["link_0"]}
            info["attached_obj_scale"] = {"right_eef_link": 0.6}
        self.task_info = info
        return reward, done, info
    
    def get_current_task_state_and_intent(self) -> Tuple[int, dict[str, Any]]:
        """
        Perform a backward checking on the current state of the task 
        Subtasks:
            0: open gripper
            1: navigate to fridge and reach towards front of handle
            2: reach towards handle
            3: grasp handle
            4: pull open container
            5: task success
        Returns:
            Tuple[int, dict[str, Any]] subtask id and the intent for each part of the robot
        """
        intent = defaultdict(lambda: None)
        intent["action_type"] = ActionTypesSelection.BODY
        right_eef_link_name = self.robot.eef_link_names["right"]
        base_link_name = self.robot.base_footprint_link_name
        if self.robot.is_grasping(arm="right", candidate_obj=self.target) == 1:
            if self.target.states[Open].get_value():
                # in subtask 5, task success
                subtask_id = 5
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = -1.0
            else:
                # in subtask 4, open init container door
                subtask_id = 4
                fridge_door_open_local_pose = (
                    torch.tensor([0.97, 0.35, 0.15]),
                    T.euler2quat(torch.tensor([math.pi / 2, math.pi / 2, 0])),
                )
                right_hand_pos, right_hand_quat = T.pose_transform(
                    self.target_pose[:3], self.target_pose[3:], *fridge_door_open_local_pose
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
                self.target_door_pose[:3], self.target_door_pose[3:], *fridge_door_local_pose
            )
            right_hand_pregrasp_pos, right_hand_pregrasp_quat = T.pose_transform(
                self.target_door_pose[:3], self.target_door_pose[3:], *fridge_door_pregrasp_local_pose
            )
            if torch.linalg.norm(self.robot.get_eef_position("right") - right_hand_pos) < 0.025:
                # in subtask 3, grasp handle
                subtask_id = 3
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = -1.0
            else:
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = 1.0
                fridge_base_local_pose = (
                    torch.tensor([0.69, -0.7, -1.00]), 
                    T.euler2quat(torch.tensor([0.0, 0.0, math.pi]))
                )
                base_pos, base_quat = T.pose_transform(
                    self.target_door_pose[:3], self.target_door_pose[3:], *fridge_base_local_pose
                )
                gripper_open = torch.sum(self.robot.get_joint_positions()[-2:]) > 0.06
                base_at_target = torch.linalg.norm(self.robot.get_position_orientation()[0] - base_pos) < 0.01
                pregrasp_at_target = torch.linalg.norm(self.robot.get_eef_position("right") - right_hand_pregrasp_pos) < 0.01
                if not gripper_open:
                    # in subtask 0, open gripper
                    subtask_id = 0
                elif base_at_target and pregrasp_at_target:
                    # in subtask 2, reach towards handle
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
                    # in subtask 1, navigate towards fridge and move arm to pregrasp
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

    def reset(self, env) -> None:
        # super reset
        super().reset(env)
        if self.robot is not None:
            logger.info(f"Episode {env._current_episode + 1}.")
            # TODO: sample new robot pose
            if np.random.rand() < self._randomize_robot_pose_prob:
                og.sim.step()
                robot_initial_pos = torch.tensor([-1.0, 0.0, 0.0]) + torch.cat(((torch.rand(2) - 0.5) * 0.3, torch.tensor([0.0])))
                robot_initial_orn = T.euler2quat(torch.tensor([0.0, 0.0, np.random.uniform(-math.pi / 6, math.pi / 6)]))
                self.robot.set_position_orientation(robot_initial_pos, robot_initial_orn)
        # sample new target
        self._sample_new_target(env)
        # reset state
        self.task_info = dict()

    def _sample_new_target(self, env) -> None:
        """
        Sample the new poses for objects and targets for the next trial and store in state dict
        Note that we also sample distractor poses, as well as the target index
        """
        if np.random.rand() < self._randomize_initial_pose_prob:
            # randomly sample a new pose for the target within the range 
            new_pos = torch.tensor([1.0, 0.0, 0.86]) + torch.cat(((torch.rand(2) - 0.5) * 0.3, torch.tensor([0.0])))
            new_orn = torch.zeros(3)
            new_orn[2] = - math.pi
            self.target_pose = torch.cat((new_pos, T.euler2quat(new_orn)))
            self.target.set_position_orientation(self.target_pose[:3], self.target_pose[3:])
            # randomly sample a new joint position for the target door
            self.target.set_joint_positions(torch.rand(1) * math.pi / 6)

    def _load_non_low_dim_observation_space(self):
        """
        Load the non-low-dimensional observation space for the task.
        """
        obs_space = dict()
        obs_space["objects"] = self._build_obs_box_space(
            shape=(14,), low=-float("inf"), high=float("inf"), dtype=NumpyTypes.FLOAT32
        )
        return obs_space       

    def _get_obs(self, env):
        """
        Get the observation for the teleop task
            - target's (and the distractors') location
        Note that these are low dim observations
        """
        # update object and target pose
        self.target_pose = torch.cat(self.target.get_position_orientation())
        self.target_door_pose = torch.cat(self.target.links["link_0"].get_position_orientation())
        # construct low dim observation dict
        obs = {
            "objects": torch.cat([self.target_pose, self.target_door_pose]),
        }
        return dict(), obs
    
    def _create_termination_conditions(self):
        """
        Create the termination conditions for the teleop task
        """
        terminations = dict()
        terminations["timeout"] = BlockEnd(
            total_episode=self._termination_config.get("total_episode"),
            total_successful_episode=self._termination_config.get("total_successful_episode"),
        )
        terminations["max_time"] = Timeout(episode_timeout_step=self._termination_config["episode_timeout_step"])
        terminations["predicate"] = PredicateGoal(
            goal_fcn=lambda: self.activity_goal_conditions, hold_step_threshold=self._termination_config["hold_step_threshold"]
        )
        terminations["task_fail"] = EpisodeFail()
        return terminations

    @classproperty
    def default_termination_config(cls):
        return {
            "hold_step_threshold": 15,
            "episode_timeout_step": 400,
            "total_episode": 40,
            "total_successful_episode": None,
        }

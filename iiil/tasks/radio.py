import math
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch
from collections import defaultdict
from iiil.utils.misc_utils import ActionTypesSelection
from iiil.utils.goal_utils import BlockEnd, Timeout, PredicateGoal, EpisodeFail
from iiil.tasks.task_base import BaseTask
from omnigibson.object_states import OnTop, ToggledOn
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.bddl_utils import BDDLEntity
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.ui_utils import create_module_logger
from typing import Any, Tuple, Optional

logger = create_module_logger("RadioTask")
logger.setLevel("INFO")


class RadioTask(BaseTask, BehaviorTask):
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
        cfg = [
            {
                "type": "DatasetObject",
                "name": "table",
                "category": "breakfast_table",
                "model": "rhjoby",
                "position": [1.0, 0.0, 0.8],
                "orientation": [0, 0, 0, 1],
                "bounding_box": [0.61, 0.76, 0.7],
                "fixed_base": True,
            },
            {
                "type": "DatasetObject",
                "name": "target",
                "category": "radio",
                "position": [1.0, 0.0, 0.7],
                "orientation": [0, 0, 0, 1],
                "scale": 0.68,
                "model": "wxnicr",
            }
        ]
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
        (define (problem turn_on_radio-0)
            (:domain omnigibson)
            (:objects
                agent.n.01_1 - agent.n.01
                breakfast_table.n.01_1 - breakfast_table.n.01
                radio_receiver.n.01_1 - radio_receiver.n.01
            )
            (:init 
                (not (toggled_on radio_receiver.n.01_1))
                (ontop radio_receiver.n.01_1 breakfast_table.n.01_1)
            )
            (:goal
                (and 
                    (toggled_on ?radio_receiver.n.01_1)
                )
            )
        )
        """

    def _load(self, env):
        BaseTask._load(self, env)
        BehaviorTask._load(self, env)

    def post_play_load(self, env) -> None:
        self.table = env.scene.object_registry("name", "table")
        self.target = env.scene.object_registry("name", "target")
        self.target.links["base_link"].mass = 1.0
        # assign object scope
        env.task.object_scope["agent.n.01_1"] = BDDLEntity(bddl_inst="agent.n.01_1", entity=env.robots[0])
        env.task.object_scope["breakfast_table.n.01_1"] = BDDLEntity(bddl_inst="breakfast_table.n.01_1", entity=self.table)
        env.task.object_scope["radio_receiver.n.01_1"] = BDDLEntity(bddl_inst="radio_receiver.n.01_1", entity=self.target)
        self.target_pose = torch.zeros(7)

        super().post_play_load(env)

    def step(self, env, action) -> Tuple[float, bool, dict]:
        # Run super method first
        reward, done, info = super().step(env=env, action=action)
        # update task info
        subtask_id, current_intent = self.get_current_task_state_and_intent()
        info["subtask_id"] = subtask_id
        info["intent"] = current_intent
        if subtask_id in [3, 4]:
            info["attached_obj"] = {"right_eef_link": self.target.links["base_link"]}
            info["attached_obj_scale"] = {"right_eef_link": 0.1}
        self.task_info = info
        return reward, done, info
    
    def get_current_task_state_and_intent(self) -> Tuple[int, dict[str, Any]]:
        """
        Perform a backward checking on the current state of the task 
        Subtasks:
            0: navigate to table
            1: reach towards top of radio
            2: reach down towards radio
            3: grasp radio
            4: pick up radio
        Returns:
            Tuple[int, dict[str, Any]] subtask id and the intent for each part of the robot 
        """
        intent = defaultdict(lambda: None)
        intent["action_type"] = ActionTypesSelection.BODY
        right_eef_link_name = self.robot.eef_link_names["right"]
        left_eef_link_name = self.robot.eef_link_names["left"]
        base_link_name = self.robot.base_footprint_link_name
        if self.robot.is_grasping(arm="right", candidate_obj=self.target) == 1:
            if self.target.get_position_orientation()[0][2] > self.table_pose[2] + 0.2:
                # in subtask 5, task success
                subtask_id = 5
                self.target.states[ToggledOn].set_value(True)
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = -1.0
            else:
                # in subtask 4, raise radio up
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
                self.target_pose[:3], self.target_pose[3:], *target_handle_local_pose
            )
            right_hand_top_pos, right_hand_top_quat = T.pose_transform(
                self.target_pose[:3], self.target_pose[3:], *target_top_local_pose
            )
            if torch.linalg.norm(self.robot.get_eef_position("right") - right_hand_pos) < 0.01:
                # in subtask 3, grasp handle
                subtask_id = 3
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = -1.0
            else:
                intent["gripper_left"] = 1.0
                intent["gripper_right"] = 1.0
                table_local_pose = (
                    torch.tensor([-0.7, 0.2, -0.6681]), 
                    T.euler2quat(torch.zeros(3))
                )
                base_pos, base_quat = T.pose_transform(
                    torch.tensor([self.table_pose[0], self.target_pose[1], self.table_pose[2]]), self.table_pose[3:], *table_local_pose
                )
                cur_robot_pos = self.robot.get_position_orientation()[0]
                if abs(cur_robot_pos[1] - base_pos[1]) < 0.02 and cur_robot_pos[0] > base_pos[0] - 0.01:
                    pregrasp_at_target = torch.linalg.norm(self.robot.get_eef_position("right") - right_hand_top_pos) < 0.03
                    # in subtask 1, reach towards top of radio; in subtask 2, reach down towards radio handle
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
                    # in subtask 0, navigate towards table
                    subtask_id = 0
                    intent["action_type"] = ActionTypesSelection.BASE
                    intent["target_pos"] = {base_link_name: base_pos}
                    intent["target_quat"] = {base_link_name: base_quat}
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
        # assisted mode doesn't work with radio, so we switch to sticky mode for better grasping
        if self.robot is not None:
            self.robot.grasping_mode = "sticky"


    def _sample_new_target(self, env) -> None:
        """
        Sample the new poses for objects and targets for the next trial and store in state dict
        Note that we also sample distractor poses, as well as the target index
        """
        if np.random.rand() < self._randomize_initial_pose_prob:
            # randomly sample a new pose for the table within the range 
            new_pos = torch.tensor([1.0, 0.0, 0.67]) + torch.cat((torch.rand(2) * 0.5 - 0.25, torch.tensor([0.0])))
            self.table_pose = torch.cat((new_pos, T.euler2quat(torch.zeros(3))))
            self.table.set_position_orientation(self.table_pose[:3], self.table_pose[3:])
            # sample radio on top of the table, and set toggled off
            self.target.states[OnTop].set_value(self.table, True)
            # Now, update radio orientation to always face left
            self.target.set_position_orientation(orientation=[0, 0, 0.7071, 0.7071])
            self.target.keep_still()
            self.target.states[ToggledOn].set_value(False)

    def _load_non_low_dim_observation_space(self):
        """
        Load the non-low-dimensional observation space for the task.
        """
        obs_space = dict()
        obs_space["objects"] = self._build_obs_box_space(
            shape=(7,), low=-float("inf"), high=float("inf"), dtype=NumpyTypes.FLOAT32
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
        self.table_pose = torch.cat(self.table.get_position_orientation())
        # construct low dim observation dict
        obs = {
            "objects": self.target_pose
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

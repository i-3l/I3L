import math
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch
from collections import defaultdict
from iiil.tasks.task_base import BaseTask
from iiil.utils.misc_utils import ActionTypesSelection
from iiil.utils.goal_utils import BlockEnd, Timeout, PredicateGoal, EpisodeFail
from omnigibson.objects import DatasetObject
from omnigibson.object_states import OnTop
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.bddl_utils import BDDLEntity
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.ui_utils import create_module_logger
from typing import Any, Optional, Tuple

logger = create_module_logger("SliceTask")
logger.setLevel("INFO")


class SliceTask(BaseTask, BehaviorTask):
    def __init__(
        self, 
        show_trajectory: bool=True,
        n_objects: int=1,
        randomize_initial_pose: bool=True,
        termination_config=None, 
        reward_config=None,
        is_target_fixed: bool=False,
    ):
        self._object_distance_threshold = 0.2
        self._show_trajectory = show_trajectory
        self._n_objects = n_objects
        assert self._n_objects == 1, "Currently only support one object for slicing task"
        self._randomize_initial_pose = randomize_initial_pose
        self._is_target_fixed = is_target_fixed

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
        target_position = [0.65, 0.2, 0.02] if not self._is_target_fixed else [0.65, 0.2, 0.056]
        self.obj_cfg = {
            "type": "DatasetObject",
            "name": "object_0",
            "category": "strawberry",
            "model": "xcnzxh",
            "scale": [1.25] * 3,
            "position": [0.55, 0.0, 0.043],
        }
        cfg = [self.obj_cfg,
        {
            "type": "DatasetObject",
            "name": f"knife",
            "category": "parer",
            "model": "lwpdhi",
            "bounding_box": [0.2, 0.025, 0.015],
            "position": target_position,
            "fixed_base": self._is_target_fixed,
        },
        {
            "type": "DatasetObject",
            "name": f"init_container",
            "category": "chopping_board",
            "model": "psabiv",
            "scale": [0.8, 1.2, 3.5],
            "position": [0.475, 0.0, 0.0],
            "fixed_base": True,
        }]
        return cfg
    
    @property
    def viewer_camera_pose(self):
        return [0.03136392, 0.25843843, 0.94360132], [-0.16955264,  0.27033132,  0.80286883, -0.50356184]

    @property
    def bddl(self) -> str:
        return f"""
        (define (problem slice_strawberry-0)
            (:domain omnigibson)
            (:objects
                strawberry.n.01_1 - strawberry.n.01
                half__strawberry.n.01_1 half__strawberry.n.01_2 - half__strawberry.n.01
                chopping_board.n.01_1 - chopping_board.n.01
                knife.n.01_1 - knife.n.01
                agent.n.01_1 - agent.n.01
            )
            (:init 
                (ontop strawberry.n.01_1 chopping_board.n.01_1)
                (ontop knife.n.01_1 chopping_board.n.01_1)
                (future half__strawberry.n.01_1) 
                (future half__strawberry.n.01_2)
            )
            (:goal
                (and
                    (real ?half__strawberry.n.01_1)
                    (real ?half__strawberry.n.01_2)
                )
            )
        )
        """
    
    @property
    def default_robot_joint_pos(self) -> Optional[torch.Tensor]:
        """
        Default robot joint positions upon startup. 
        If return None, will use robot's default reset joint positions.
        
        Returns:
            Optional[torch.tensor]: optional default robot joint positions.
        """
        return torch.tensor([0.0, 1.877, -0.945, 1.570, 0.931, 0.0, 0.03, 0.03])

    def _load(self, env):
        BaseTask._load(self, env)
        BehaviorTask._load(self, env)

    def post_play_load(self, env) -> None:
        # get the object prims
        self.objects = [env.scene.object_registry("name", f"object_{i}") for i in range(self._n_objects)]
        with og.sim.stopped():
            for object in self.objects:
                object.links["base_link"].mass = 0.5

        self.knife = env.scene.object_registry("name", "knife")
        self.init_container = env.scene.object_registry("name", "init_container")
        # assign object scope
        env.task.object_scope["agent.n.01_1"] = BDDLEntity(bddl_inst="agent.n.01_1", entity=env.robots[0])
        env.task.object_scope["chopping_board.n.01_1"] = BDDLEntity(bddl_inst="chopping_board.n.01_1", entity=self.init_container)
        env.task.object_scope["knife.n.01_1"] = BDDLEntity(bddl_inst="knife.n.01_1", entity=self.knife)
        for i in range(self._n_objects):
            env.task.object_scope[f"strawberry.n.01_{i+1}"] = BDDLEntity(bddl_inst=f"strawberry.n.01_{i+1}", entity=self.objects[i])
        env.task.object_scope["half__strawberry.n.01_1"] = BDDLEntity(bddl_inst="half__strawberry.n.01_1", entity=None)
        env.task.object_scope["half__strawberry.n.01_2"] = BDDLEntity(bddl_inst="half__strawberry.n.01_2", entity=None)
        self.object_pose = torch.zeros((self._n_objects, 7))
        self.knife_pose = torch.zeros(7)

        super().post_play_load(env)
    
    def step(self, env, action) -> Tuple[float, bool, dict]:
        # Run super method first
        reward, done, info = super().step(env=env, action=action)
        # update task info
        subtask_id, current_intent = self.get_current_task_state_and_intent()
        info["object_idx"] = self.object_idx
        info["subtask_id"] = subtask_id
        info["intent"] = current_intent
        if subtask_id in [4, 5, 6, 7]:
            info["attached_obj"] = {"eef_link": self.knife.root_link}
            info["attached_obj_scale"] = {"eef_link": 0.95}
        self.task_info = info
        return reward, done, info
    
    def get_current_task_state_and_intent(self) -> Tuple[int, dict[str, Any]]:
        """
        Perform a backward checking on the current state of the task 
        Subtasks:
            0: open gripper
            1: reach towards top of knife
            2: reach towards knife
            3: grasp knife
            4: raise knife
            5: reach towards top of object
            6: cut object
            7: release knife
        Returns:
            Tuple[int, dict[str, Any]] subtask id and the intent for each part of the robot
        """
        intent = defaultdict(lambda: None)
        intent["action_type"] = ActionTypesSelection.DEFAULT
        eef_link_name = self.robot.eef_link_names[self.robot.default_arm]
        if torch.allclose(self.object_pose[self.object_idx], torch.zeros(7)):
            # passed subtasks 7, task complete
            subtask_id = 7
            intent["gripper_0"] = -1.0
        elif self.robot.is_grasping(candidate_obj=self.knife) == 1:
            if torch.linalg.norm(
                self.robot.get_eef_position() - self.object_pose[self.object_idx][:3] - torch.tensor([-0.04, -0.1, 0.1])
            ) < 0.01:
                # in subtask 6, cut object
                subtask_id = 6
                intent["target_pos"] = {eef_link_name: torch.tensor([-0.04, -0.1, 0.1]) + self.object_pose[self.object_idx][:3]}
                intent["target_quat"] = {eef_link_name: torch.tensor([0.2705981, 0.6532815, 0.2705981, 0.6532815])}
                intent["gripper_0"] = -1.0
                intent["action_type"] = ActionTypesSelection.ARM_IK
            # check whether knife is on top of init container anymore
            elif not self.knife.states[OnTop].get_value(self.init_container):
                # in subtask 5, reach towards object
                subtask_id = 5
                intent["target_pos"] = {eef_link_name: torch.tensor([-0.04, -0.1, 0.1]) + self.object_pose[self.object_idx][:3]}
                intent["target_quat"] = {eef_link_name: torch.tensor([0.5, 0.5, 0.5, 0.5])}
                intent["gripper_0"] = -1.0
            else:
                # in subtask 4, raise knife
                subtask_id = 4
                target_local_pose = (
                    torch.tensor([0.035, 0.0, 0.15]), 
                    T.euler2quat(torch.tensor([math.pi, 0, math.pi]))
                )
                eef_link_pos, eef_link_quat = T.pose_transform(
                    self.knife_pose[:3], self.knife_pose[3:], *target_local_pose
                )
                intent["target_pos"] = {eef_link_name: eef_link_pos}
                intent["target_quat"] = {eef_link_name: eef_link_quat}
                intent["gripper_0"] = -1.0
                intent["action_type"] = ActionTypesSelection.ARM_IK
        elif torch.sum(self.robot.get_joint_positions()[-2:]) > 0.05:
            # gripper width > 0.05, gripper is open
            target_local_pose = (
                torch.tensor([0.035, 0.0, 0.041]), 
                T.euler2quat(torch.tensor([math.pi, 0, math.pi]))
            )
            eef_link_pos, eef_link_quat = T.pose_transform(
                self.knife_pose[:3], self.knife_pose[3:], *target_local_pose
            )
            if torch.linalg.norm(self.robot.get_eef_position() - eef_link_pos) < 0.01:
                # in subtask 3, grasp knife
                subtask_id = 3
                intent["gripper_0"] = -1.0
            else:
                intent["gripper_0"] = 1.0
                if torch.linalg.norm(self.robot.get_eef_position()[:2] - eef_link_pos[:2]) < 0.01:
                    # in subtask 2, reach towards object
                    subtask_id = 2
                    intent["target_pos"] = {eef_link_name: eef_link_pos}
                    intent["target_quat"] = {eef_link_name: eef_link_quat}
                    intent["action_type"] = ActionTypesSelection.ARM_IK
                else:
                    # in subtask 1, reach towards top of target
                    subtask_id = 1
                    intent["target_pos"] = {eef_link_name: eef_link_pos + torch.tensor([0.0, 0.0, 0.03])}
                    intent["target_quat"] = {eef_link_name: eef_link_quat}
        else:
            # in subtask 0, open gripper
            subtask_id = 0
            intent["gripper_0"] = 1.0
        return subtask_id, intent

    def reset(self, env) -> None:
        half_objects = env.scene.object_registry("category", "half_strawberry", set()).copy()
        if len(half_objects) > 0:
            # remove all halfed objects, reimport the original object
            for half_object in half_objects:
                env.scene.remove_object(half_object)
            self.objects = [DatasetObject(**self.obj_cfg)]
            env.scene.add_object(self.objects[0])
            og.sim.step()
        # super reset
        super().reset(env)
        if self.robot is not None:
            logger.info(f"Episode {env._current_episode + 1}.")
            # TODO: sample new robot pose
        # sample new target
        if self._randomize_initial_pose:
            self._sample_new_target(env)
        # reset state
        self.task_info = dict()

    def _sample_new_target(self, env) -> None:
        """
        Sample the new poses for objects and targets for the next trial and store in state dict
        Note that we also sample distractor poses, as well as the target index
        """
        # sample new object 
        for i, manipulation_object in enumerate(self.objects):
            manipulation_object.highlighted = False
        self.object_idx = np.random.choice(self._n_objects)
        # self.objects[self.object_idx].highlighted = True

        # Resample the sheet height only when the target is not fixed.
        vertical_offset = np.random.uniform(0.00, 0.05) if not self._is_target_fixed else 0.025
        self.init_container.set_position_orientation(position=[0.475, 0.0, vertical_offset])

        self.knife.states[OnTop].set_value(self.init_container, True, use_trav_map=False)
        # randomly sample a rotation within range such that the knife is reachable by the robot
        ori = T.euler2quat(torch.tensor([0, 0, np.random.uniform(math.pi * 2, math.pi * 4) / 3]))
        self.knife.set_position_orientation(position=self.knife.get_position_orientation()[0], orientation=ori)
        for i in range(len(self.objects)):
            # Make sure that the object isn't too close to the other objects
            objects_to_check = self.objects[:i] + [self.knife]
            for _ in range(10):
                success = self.objects[i].states[OnTop].set_value(self.init_container, True, use_trav_map=False)
                for obj in objects_to_check:
                    if torch.linalg.norm(
                        self.objects[i].get_position_orientation()[0] - obj.get_position_orientation()[0]
                    ) < self._object_distance_threshold:
                        success = False
                if success:
                    break
            if not success:
                logger.warning("Bad poses sampled. Objects might be too close to each other.")
        # record the object and target poses
        self.object_pose = torch.stack([torch.cat(self.objects[i].get_position_orientation()) for i in range(self._n_objects)])
        self.knife_pose = torch.cat(self.knife.get_position_orientation())

    def _load_non_low_dim_observation_space(self):
        """
        Load the non-low-dimensional observation space for the task.
        """
        obs_space = dict()
        obs_space["object_pose"] = self._build_obs_box_space(
            shape=(7 * (self._n_objects + 1),), low=-float("inf"), high=float("inf"), dtype=NumpyTypes.FLOAT32
        )
        return obs_space       

    def _get_obs(self, env):
        """
        Get the observation for the teleop task
            - object's (and the distractors') location
            - target's (and the distractors') location
        Note that these are low dim observations
        """
        # update object and target pose
        try:
            self.object_pose = torch.stack([torch.cat(self.objects[i].get_position_orientation()) for i in range(self._n_objects)])
        except: 
            # if the object is not in the scene, set the pose to zero
            self.object_pose = torch.zeros((self._n_objects, 7))
        self.knife_pose = torch.cat(self.knife.get_position_orientation())

        # construct low dim observation dict
        obs = {
            "object_pose": torch.cat((
                torch.unsqueeze(self.knife_pose, dim=0), 
                self.object_pose,
            ), dim=0).flatten(),
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
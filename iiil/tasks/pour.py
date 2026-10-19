import math
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch
from collections import defaultdict
from iiil.tasks.task_base import BaseTask
from iiil.utils.misc_utils import ActionTypesSelection
from iiil.utils.goal_utils import BlockEnd, Timeout, PredicateGoal, EpisodeFail
from omnigibson.object_states import Contains, Filled, OnTop
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.bddl_utils import BDDLEntity
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.ui_utils import create_module_logger
from typing import Any, Optional, Tuple

logger = create_module_logger("PourTask")
logger.setLevel("INFO")


class PourTask(BaseTask, BehaviorTask):
    def __init__(
        self, 
        show_trajectory: bool=True,
        n_objects: int=1,
        randomize_initial_pose: bool=True,
        randomize_robot_pose_prob: float=0.0,
        randomize_initial_pose_prob: Optional[float]=None,
        termination_config=None, 
        reward_config=None,
        is_target_fixed: bool=False,
    ):
        self._object_distance_threshold = 0.15
        self._show_trajectory = show_trajectory
        self._n_objects = n_objects
        self._randomize_robot_pose_prob = randomize_robot_pose_prob
        self._randomize_initial_pose_prob = (
            float(randomize_initial_pose)
            if randomize_initial_pose_prob is None
            else randomize_initial_pose_prob
        )
        self._is_target_fixed = is_target_fixed
        self._particle_system = None

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
        target_position = [0.65, 0.0, 0.02] if not self._is_target_fixed else [0.65, 0.0, 0.056]
        cfg = [{
            "type": "DatasetObject",
            "name": f"object_0",
            "category": "coffee_cup",
            "model": "ykuftq",
            "bounding_box": [0.09, 0.06, 0.08],
            "position": [0.6, 0.0, 0.6],
        },
        {
            "type": "DatasetObject",
            "name": f"target",
            "category": "bowl",
            "model": "nkkhbn",
            "bounding_box": [0.09, 0.09, 0.02],
            "position": target_position,
            "fixed_base": self._is_target_fixed,
        },
        {
            "type": "DatasetObject",
            "name": f"init_container",
            "category": "chopping_board",
            "model": "psabiv",
            "scale": [0.6, 1.0, 3.5],
            "fixed_base": True,
        }]
        return cfg
    
    @property
    def viewer_camera_pose(self):
        return [0.03136392, 0.25843843, 0.94360132], [-0.16955264,  0.27033132,  0.80286883, -0.50356184]

    @property
    def bddl(self) -> str:
        return f"""
        (define (problem pour-0)
            (:domain omnigibson)
            (:objects
                object.n.01_1 - object.n.01
                chopping_board.n.01_1 - chopping_board.n.01
                bowl.n.01_1 - bowl.n.01
                agent.n.01_1 - agent.n.01
                coffee_bean.n.01_1 - coffee_bean.n.01
            )
            (:init 
                (ontop object.n.01_1 chopping_board.n.01_1)
                (ontop bowl.n.01_1 chopping_board.n.01_1) 
                (contains object.n.01_1 coffee_bean.n.01_1)
            )
            (:goal
                (and
                    (contains bowl.n.01_1 coffee_bean.n.01_1)
                    (not (contains object.n.01_1 coffee_bean.n.01_1))
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
        return None

    def _default_robot_joint_pos_for_robot(self, robot) -> Optional[torch.Tensor]:
        robot_model = str(getattr(robot, "model", "")).lower()
        if robot_model == "r1pro":
            return torch.cat([
                torch.zeros(6, dtype=torch.float32),    # base
                torch.tensor([-0.63, 1.65, 1.05, 0.00]),  # torso
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.57, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # arms
                torch.ones(4, dtype=torch.float32) * 0.05    # grippers
            ])
        if robot_model == "a1":
            return torch.tensor([0.0, 1.877, -0.945, 1.570, 0.931, 0.0, 0.03, 0.03])
        return None

    def _load(self, env):
        BaseTask._load(self, env)
        default_joint_pos = self._default_robot_joint_pos_for_robot(env.robots[0])
        if default_joint_pos is not None:
            env.robots[0]._reset_joint_pos = default_joint_pos
        BehaviorTask._load(self, env)

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

    def post_play_load(self, env) -> None:
        # get the object prims
        self.objects = [env.scene.object_registry("name", f"object_{i}") for i in range(self._n_objects)]
        with og.sim.stopped():
            for object in self.objects:
                object.links["base_link"].mass = 0.5

        self.target = env.scene.object_registry("name", "target")
        # change target mass
        self.target.links["base_link"].mass = 1.0
        self.init_container = env.scene.object_registry("name", "init_container")
        # assign object scope
        env.task.object_scope["agent.n.01_1"] = BDDLEntity(bddl_inst="agent.n.01_1", entity=env.robots[0])
        env.task.object_scope["chopping_board.n.01_1"] = BDDLEntity(bddl_inst="chopping_board.n.01_1", entity=self.init_container)
        env.task.object_scope["bowl.n.01_1"] = BDDLEntity(bddl_inst="bowl.n.01_1", entity=self.target)
        env.task.object_scope["coffee_bean.n.01_1"] = BDDLEntity(bddl_inst="coffee_bean.n.01_1", entity=None)
        for i in range(self._n_objects):
            env.task.object_scope[f"object.n.01_{i+1}"] = BDDLEntity(bddl_inst=f"object.n.01_{i+1}", entity=self.objects[i])
        self.object_pose = torch.zeros((self._n_objects, 7))
        self.target_pose = torch.zeros(7)

        super().post_play_load(env)
    
    def step(self, env, action) -> Tuple[float, bool, dict]:
        # Run super method first
        reward, done, info = super().step(env=env, action=action)
        # update task info
        subtask_id, current_intent = self.get_current_task_state_and_intent()
        info["object_idx"] = self.object_idx
        info["subtask_id"] = subtask_id
        info["intent"] = current_intent
        if subtask_id in [3, 4, 5, 6, 7]:
            attached_link_name = "right_eef_link" if self._uses_right_arm_api() else "eef_link"
            info["attached_obj"] = {attached_link_name: self.objects[self.object_idx].root_link}
            info["attached_obj_scale"] = {attached_link_name: 0.95}
        self.task_info = info
        return reward, done, info
    
    def get_current_task_state_and_intent(self) -> Tuple[int, dict[str, Any]]:
        """
        Perform a backward checking on the current state of the task 
        Subtasks:
            0: open gripper
            1: reach towards object
            2: grasp object
            3: raise object
            4: reach towards target
            5: rotate object
        Returns:
            Tuple[int, dict[str, Any]] subtask id and the intent for each part of the robot
        """
        intent = defaultdict(lambda: None)
        intent["action_type"] = self._target_action_type()
        eef_link_name = self._eef_link_name()
        if self.target.states[Contains].get_value(self._particle_system):
            # passed subtasks 7, task complete
            subtask_id = 7
            self._set_gripper_intent(intent, -1.0)
        elif self._is_grasping(self.objects[self.object_idx]):
            target_pose = self.target_pose[:3] + torch.tensor([-0.03, 0.0, 0.18])
            if torch.linalg.norm(
                self._eef_position() - target_pose
            ) < 0.015:
                # in subtask 6, rotate object
                subtask_id = 6
                intent["target_pos"] = {eef_link_name: target_pose}
                intent["target_quat"] = {eef_link_name: T.euler2quat(torch.tensor([math.pi, math.pi / 2, math.pi]))}
                self._set_gripper_intent(intent, -1.0)
                intent["motion_constraint"] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]    # restrict motion along y axis
            # check whether object is on top anymore
            elif not self.objects[self.object_idx].states[OnTop].get_value(self.init_container):
                # in subtask 5, reach towards target
                subtask_id = 5
                intent["target_pos"] = {eef_link_name: target_pose}
                intent["target_quat"] = {eef_link_name: T.euler2quat(torch.tensor([math.pi, -math.pi / 4, 0.0]))}
                self._set_gripper_intent(intent, -1.0)
            else:
                target_local_pose = (
                    torch.tensor([0.07, 0.0, 0.05]), 
                    T.euler2quat(torch.tensor([math.pi, -math.pi / 4, math.pi]))
                )
                eef_link_pos, eef_link_quat = T.pose_transform(
                    self.object_pose[self.object_idx, :3], self.object_pose[self.object_idx, 3:], *target_local_pose
                )
                # in subtask 4, raise object
                subtask_id = 4
                intent["target_pos"] = {eef_link_name: eef_link_pos}
                intent["target_quat"] = {eef_link_name: eef_link_quat}
                self._set_gripper_intent(intent, -1.0)
                intent["ik_only"] = True
        elif torch.sum(self.robot.get_joint_positions()[-2:]) > 0.05:
            # gripper width > 0.05, gripper is open
            target_local_pose = (
                torch.tensor([0.07, 0.0, 0.025]), 
                T.euler2quat(torch.tensor([math.pi, -math.pi / 4, math.pi]))
            )
            eef_link_pos, eef_link_quat = T.pose_transform(
                self.object_pose[self.object_idx, :3], self.object_pose[self.object_idx, 3:], *target_local_pose
            )
            if torch.linalg.norm(self._eef_position() - eef_link_pos) < 0.01:
                # in subtask 3, grasp object
                subtask_id = 3
                self._set_gripper_intent(intent, -1.0)
            else:
                self._set_gripper_intent(intent, 1.0)
                if torch.linalg.norm(self._eef_position()[:2] - eef_link_pos[:2]) < 0.01:
                    # in subtask 2, reach towards object
                    subtask_id = 2
                    intent["target_pos"] = {eef_link_name: eef_link_pos}
                    intent["target_quat"] = {eef_link_name: eef_link_quat}
                    intent["ik_only"] = True
                else:
                    # in subtask 1, reach towards top of target
                    subtask_id = 1
                    intent["target_pos"] = {eef_link_name: eef_link_pos + torch.tensor([0.0, 0.0, 0.05])}
                    intent["target_quat"] = {eef_link_name: eef_link_quat}
                    self._add_base_target(intent)
        else:
            # in subtask 0, open gripper
            subtask_id = 0
            self._set_gripper_intent(intent, 1.0)
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
        if np.random.rand() < self._randomize_initial_pose_prob:
            # sample new target and object poses
            self._sample_new_target(env)
        else:
            self._particle_system = env.scene.get_system("coffee_bean")
            self.object_pose = torch.stack([torch.cat(self.objects[i].get_position_orientation()) for i in range(self._n_objects)])
            self.target_pose = torch.cat(self.target.get_position_orientation())
        # reset state
        self.task_info = dict()

    def _sample_new_target(self, env) -> None:
        """
        Sample the new poses for objects and targets for the next trial and store in state dict
        Note that we also sample distractor poses, as well as the target index
        """
        self._particle_system = env.scene.get_system("coffee_bean")
        # sample new object 
        for i, manipulation_object in enumerate(self.objects):
            manipulation_object.highlighted = False
        self.object_idx = np.random.choice(self._n_objects)

        # Resample the sheet height only when the target is not fixed.
        vertical_offset = np.random.uniform(0.00, 0.05) if not self._is_target_fixed else 0.025
        self.init_container.set_position_orientation(position=[0.5, 0.0, vertical_offset])

        # Sample the target position if not fixed
        if not self._is_target_fixed: 
            self.target.states[OnTop].set_value(self.init_container, True)

        # Sample object states
        for i in range(self._n_objects):
            # Make sure that the object isn't too close to the other objects
            objects_to_check = self.objects[:i] + [self.target]
            for _ in range(10):
                success = self.objects[i].states[OnTop].set_value(self.init_container, True)
                for obj in objects_to_check:
                    if torch.linalg.norm(
                        self.objects[i].get_position_orientation()[0] - obj.get_position_orientation()[0]
                    ) < self._object_distance_threshold:
                        success = False
                if success:
                    break
            if not success:
                logger.warning("Bad poses sampled. Objects might be too close to each other.")
            # randomly sample a rotation within range such that the cup is reachable by the robot
            ori = T.euler2quat(torch.tensor([0, 0, np.random.uniform(math.pi * 5, math.pi * 7) / 6]))
            self.objects[i].set_position_orientation(position=self.objects[i].get_position_orientation()[0], orientation=ori)
        # Fill object with particles
        while not self.objects[self.object_idx].states[Contains].get_value(self._particle_system):
            self._particle_system.remove_all_particles()
            success = self.objects[self.object_idx].states[Filled].set_value(self._particle_system, True)
            og.sim.step()
        # record the object and target poses
        self.object_pose = torch.stack([torch.cat(self.objects[i].get_position_orientation()) for i in range(self._n_objects)])
        self.target_pose = torch.cat(self.target.get_position_orientation())

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
        self.object_pose = torch.stack([torch.cat(self.objects[i].get_position_orientation()) for i in range(self._n_objects)])
        self.target_pose = torch.cat(self.target.get_position_orientation())
        # construct low dim observation dict
        obs = {
            "object_pose": torch.cat((torch.unsqueeze(self.target_pose, dim=0), self.object_pose), dim=0).flatten(),
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

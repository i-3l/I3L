import numpy as np
import omnigibson as og
import torch
from collections import defaultdict
from iiil.tasks.task_base import BaseTask
from iiil.utils.math_utils import add_position_noise, add_quaternion_noise
from iiil.utils.misc_utils import ActionTypesSelection
from iiil.utils.goal_utils import BlockEnd, Timeout, PredicateGoal, EpisodeFail
from omnigibson.object_states import OnTop
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.bddl_utils import BDDLEntity
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.ui_utils import create_module_logger
from typing import Any, Optional, Tuple

logger = create_module_logger("PickPlaceTask")
logger.setLevel("INFO")


class PickPlaceTask(BaseTask, BehaviorTask):
    def __init__(
        self,
        show_trajectory: bool=True,
        n_objects: int=1,
        randomize_initial_pose: bool=True,
        randomize_robot_pose_prob: float=0.0,
        randomize_trajectory_prob: float=0.0,
        termination_config=None,
        reward_config=None,
        is_target_fixed: bool=True,
        fixed_object_poses_path: str=None,
        save_object_poses_path: str=None,
    ):
        self._object_distance_threshold = 0.15
        self._show_trajectory = show_trajectory
        self._n_objects = n_objects
        self._is_target_fixed = is_target_fixed
        self._randomize_robot_pose_prob = randomize_robot_pose_prob
        self._randomize_trajectory_prob = randomize_trajectory_prob
        self.randomize_initial_pose = randomize_initial_pose
        self._in_abnormal_state = False
        self.task_info = dict()
        self._fixed_object_poses = None
        if fixed_object_poses_path is not None:
            import json
            with open(fixed_object_poses_path) as f:
                self._fixed_object_poses = json.load(f)
        self._save_object_poses_path = save_object_poses_path
        self._saved_poses = {}
        self._sample_count = 0

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
        target_position = [0.45, 0.1, 0.083]
        cfg = [{
            "type": "DatasetObject",
            "name": f"object_{i}",
            "category": "strawberry",
            "model": "xcnzxh",
            "scale": [1.25] * 3,
            "position": [0.45, -0.1, 0.043],
        } for i in range(self._n_objects)]
        cfg.extend([
            {
                "type": "DatasetObject",
                "name": f"target",
                "category": "bowl",
                "model": "xpnlup",
                "bounding_box": [0.1, 0.1, 0.015],
                "position": target_position,
                "fixed_base": self._is_target_fixed,
            },
            {
                "type": "DatasetObject",
                "name": f"init_container",
                "category": "chopping_board",
                "model": "psabiv",
                "scale": [0.9, 1.0, 3.5],
                "position": [0.45, 0.0, 0.0],
                "fixed_base": True,
            },
        ])
        return cfg
    
    @property
    def viewer_camera_pose(self):
        return [0.03136392, 0.25843843, 0.94360132], [-0.16955264,  0.27033132,  0.80286883, -0.50356184]

    @property
    def bddl(self) -> str:
        obj_init_str = "\n".join([f"(ontop object.n.01_{i+1} chopping_board.n.01_1)" for i in range(self._n_objects)])
        obj_goal_str = "\n".join([f"(ontop object.n.01_{i+1} hamper.n.02_1)" for i in range(self._n_objects)])
        return f"""
        (define (problem pick_and_place-0)
            (:domain omnigibson)
            (:objects
                {" ".join([f"object.n.01_{i}" for i in range(1, self._n_objects + 1)]) + " - object.n.01"}
                chopping_board.n.01_1 - chopping_board.n.01
                hamper.n.02_1 - hamper.n.02
                agent.n.01_1 - agent.n.01
            )
            (:init 
                {obj_init_str}
                (ontop hamper.n.02_1 chopping_board.n.01_1) 
            )
            (:goal
                (and
                    {obj_goal_str}
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
        # return torch.zeros((8,), dtype=torch.float32)

    def _load(self, env):
        BaseTask._load(self, env)
        BehaviorTask._load(self, env)

    def post_play_load(self, env) -> None:
        # get the object prims
        self.objects = [env.scene.object_registry("name", f"object_{i}") for i in range(self._n_objects)]
        with og.sim.stopped():
            for object in self.objects:
                object.links["base_link"].mass = 0.5

        self.target = env.scene.object_registry("name", "target")
        self.init_container = env.scene.object_registry("name", "init_container")
        # assign object scope
        env.task.object_scope["agent.n.01_1"] = BDDLEntity(bddl_inst="agent.n.01_1", entity=env.robots[0])
        env.task.object_scope["chopping_board.n.01_1"] = BDDLEntity(bddl_inst="chopping_board.n.01_1", entity=self.init_container)
        env.task.object_scope["hamper.n.02_1"] = BDDLEntity(bddl_inst="hamper.n.02_1", entity=self.target)
        for i in range(self._n_objects):
            env.task.object_scope[f"object.n.01_{i+1}"] = BDDLEntity(bddl_inst=f"object.n.01_{i+1}", entity=self.objects[i])
        self.object_pose = torch.zeros((self._n_objects, 7))
        self.target_pose = torch.zeros(7)

        super().post_play_load(env)
    
    def step(self, env, action) -> Tuple[float, bool, dict]:
        # Run super method first
        reward, done, info = super().step(env=env, action=action)
        # Iterate through possible task states
        # update task info
        subtask_id, current_intent = self.get_current_task_state_and_intent()
        info["object_idx"] = self.object_idx
        info["subtask_id"] = subtask_id
        info["intent"] = current_intent
        if subtask_id in [4, 5]:
            info["attached_obj"] = {"eef_link": self.objects[self.object_idx].root_link}
            info["attached_obj_scale"] = {"eef_link": 0.9}
        self.task_info = info
        return reward, done, info
    
    def get_current_task_state_and_intent(self) -> Tuple[int, dict[str, Any]]:
        """
        Perform a backward checking on the current state of the task 
        Subtasks:
            0: open gripper
            1: reach towards top of object
            2: reach towards object
            3: grasp object
            4: raise object
            5: reach towards target
            6: release object
        Returns:
            Tuple[int, dict[str, Any]] subtask id and the intent for each part of the robot
        """
        # whether we want to randomize trajectory
        if self._randomize_trajectory_prob > 0.0 and np.random.rand() < self._randomize_trajectory_prob:
            pos_noise_std, quat_noise_std = 0.01, 0.03
        else:
            pos_noise_std, quat_noise_std = 0.0, 0.0
        intent = defaultdict(lambda: None)
        intent["replan"] = False
        intent["action_type"] = ActionTypesSelection.DEFAULT
        eef_link_name = self.robot.eef_link_names[self.robot.default_arm]
        base_quat = torch.tensor([0.9239, 0.0, 0.3827, 0.0])
        if self.objects[self.object_idx].states[OnTop].get_value(self.target):
            self._in_abnormal_state = False
            # passed subtasks 7, task complete
            subtask_id = 7
            intent["gripper_0"] = 1.0
        elif self.robot.is_grasping(candidate_obj=self.objects[self.object_idx]) == 1:
            self._in_abnormal_state = False
            if torch.linalg.norm(
                self.robot.get_eef_position()[:2] - self.target_pose[:2] - torch.tensor([-0.015, 0.0])
            ) < 0.03:
                # in subtask 6, release object
                subtask_id = 6
                intent["gripper_0"] = 1.0
            # check whether object is on top anymore
            elif not self.objects[self.object_idx].states[OnTop].get_value(self.init_container):
                # in subtask 5, reach towards target
                subtask_id = 5
                intent["target_pos"] = {eef_link_name: add_position_noise(torch.tensor([-0.015, 0.0, 0.08]) + self.target_pose[:3], noise_std=pos_noise_std)}
                intent["target_quat"] = {eef_link_name: add_quaternion_noise(base_quat, noise_std=quat_noise_std)}
                intent["gripper_0"] = -1.0
            else:
                # in subtask 4, raise object
                subtask_id = 4
                intent["target_pos"] = {eef_link_name: torch.tensor([-0.015, 0.0, 0.05]) + self.object_pose[self.object_idx][:3]}
                intent["target_quat"] = {eef_link_name: base_quat}
                intent["gripper_0"] = -1.0
                intent["ik_only"] = True
        elif torch.linalg.norm(
            self.robot.get_eef_position() - self.object_pose[self.object_idx][:3] - torch.tensor([-0.015, 0.0, 0.022])
        ) < 0.02:
            self._in_abnormal_state = False
            # in subtask 3, grasp object
            subtask_id = 3
            intent["gripper_0"] = -1.0
        else:
            # gripper width > 0.05, gripper is open
            if torch.sum(self.robot.get_joint_positions()[-2:]) > 0.05:
                intent["gripper_0"] = 1.0
                if torch.linalg.norm(
                    self.robot.get_eef_position()[:2] - self.object_pose[self.object_idx][:2] - torch.tensor([-0.015, 0.0])
                ) < 0.03:
                    # in subtask 2, reach towards object
                    subtask_id = 2
                    intent["target_pos"] = {eef_link_name: torch.tensor([-0.015, 0.0, 0.022]) + self.object_pose[self.object_idx][:3]}
                    intent["target_quat"] = {eef_link_name: base_quat}
                    intent["ik_only"] = True
                else:
                    # in subtask 1, reach towards top of object
                    subtask_id = 1
                    intent["target_pos"] = {eef_link_name: add_position_noise(torch.tensor([-0.015, 0.0, 0.05]) + self.object_pose[self.object_idx][:3], noise_std=pos_noise_std)}
                    intent["target_quat"] = {eef_link_name: add_quaternion_noise(base_quat, noise_std=quat_noise_std)}
            else:
                if not self._in_abnormal_state:
                    intent["replan"] = True
                self._in_abnormal_state = True
                # in subtask 0, open gripper
                subtask_id = 0
                intent["gripper_0"] = 1.0
        return subtask_id, intent

    def reset(self, env) -> None:
        self._in_abnormal_state = False
        # super reset
        super().reset(env)
        if self.robot is not None:
            logger.info(f"Episode {env._current_episode + 1}.")
             # sample new robot pose
            if np.random.rand() < self._randomize_robot_pose_prob:
                og.sim.step()
                action = self.robot.get_no_op_action()
                action += np.random.uniform(-0.15, 0.15, size=action.shape)
                for _ in range(10):
                    env.step(action, False)
        # sample new target
        if self.randomize_initial_pose:        
            self._sample_new_target(env)
        # reset state
        self.task_info = dict()

    def _sample_new_target(self, env) -> None:
        """
        Sample the new poses for objects and targets for the next trial and store in state dict
        Note that we also sample distractor poses, as well as the target index
        """
        is_real_episode = self.robot is not None
        if is_real_episode:
            episode_idx = self._sample_count
            self._sample_count += 1
        else:
            episode_idx = -1

        # If fixed poses are provided, place objects at exact recorded positions
        if self._fixed_object_poses is not None and is_real_episode:
            ep_key = str(episode_idx)
            if ep_key in self._fixed_object_poses:
                poses = self._fixed_object_poses[ep_key]
                self.object_idx = poses.get("object_idx", 0)
                for i, obj_pose in enumerate(poses["objects"]):
                    pos = torch.tensor(obj_pose["position"], dtype=torch.float32)
                    quat = torch.tensor(obj_pose["orientation"], dtype=torch.float32)
                    self.objects[i].set_position_orientation(position=pos, orientation=quat)
                if "target" in poses:
                    tp = poses["target"]
                    self.target.set_position_orientation(
                        position=torch.tensor(tp["position"], dtype=torch.float32),
                        orientation=torch.tensor(tp["orientation"], dtype=torch.float32),
                    )
                og.sim.step()
                self.object_pose = torch.stack([torch.cat(self.objects[i].get_position_orientation()) for i in range(self._n_objects)])
                self.target_pose = torch.cat(self.target.get_position_orientation())
                logger.info(f"Episode {episode_idx}: using fixed object poses")
                return

        candidate_idx = []
        # sample new object
        for i, manipulation_object in enumerate(self.objects):
            manipulation_object.highlighted = False
            if not manipulation_object.states[OnTop].get_value(self.target):
                candidate_idx.append(i)
        self.object_idx = np.random.choice(self._n_objects)

        vertical_offset = np.random.uniform(0.00, 0.05) if not self._is_target_fixed else 0.025
        self.init_container.set_position_orientation(position=[0.45, 0.0, vertical_offset])
        if not self._is_target_fixed:
            self.target.states[OnTop].set_value(self.init_container, True)

        all_valid = False
        for _global_attempt in range(100):
            all_valid = True
            for i in range(len(self.objects)):
                objects_to_check = self.objects[:i] + [self.target]
                obj_placed = False
                for _ in range(10):
                    success = self.objects[i].states[OnTop].set_value(self.init_container, True)
                    for obj in objects_to_check:
                        if torch.linalg.norm(
                            self.objects[i].get_position_orientation()[0] - obj.get_position_orientation()[0]
                        ) < self._object_distance_threshold:
                            success = False
                    if success:
                        obj_placed = True
                        break
                if not obj_placed:
                    all_valid = False
                    break
            if all_valid:
                break
        if not all_valid:
            logger.warning("Bad poses sampled after 100 global attempts. Objects might be too close to each other.")

        # record the object and target poses
        self.object_pose = torch.stack([torch.cat(self.objects[i].get_position_orientation()) for i in range(self._n_objects)])
        self.target_pose = torch.cat(self.target.get_position_orientation())

        if self._save_object_poses_path is not None and is_real_episode:
            ep_data = {"object_idx": int(self.object_idx), "objects": [], "target": {}}
            for i in range(len(self.objects)):
                pos, quat = self.objects[i].get_position_orientation()
                ep_data["objects"].append({
                    "position": pos.cpu().tolist(),
                    "orientation": quat.cpu().tolist(),
                })
            t_pos, t_quat = self.target.get_position_orientation()
            ep_data["target"] = {
                "position": t_pos.cpu().tolist(),
                "orientation": t_quat.cpu().tolist(),
            }
            self._saved_poses[str(episode_idx)] = ep_data
            import json
            with open(self._save_object_poses_path, "w") as f:
                json.dump(self._saved_poses, f, indent=2)

    def _load_non_low_dim_observation_space(self):
        """
        Load the non-low-dimensional observation space for the task.
        """
        obs_space = dict()
        obs_space["object_pose"] = self._build_obs_box_space(
            shape=(3 * (self._n_objects + 1),), low=-float("inf"), high=float("inf"), dtype=NumpyTypes.FLOAT32
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
        # record relative object pose by subtracting eef pose
        rel_object_pose = self.object_pose.clone()
        rel_object_pose[:, :3] -= env.robots[0].get_eef_position()
        self.target_pose = torch.cat(self.target.get_position_orientation())
        # construct low dim observation dict
        obs = {
            "object_pose": torch.cat((torch.unsqueeze(self.target_pose[:3], dim=0), rel_object_pose[:, :3]), dim=0).flatten(),
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

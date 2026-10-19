import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch
from gymnasium.spaces import Box
from omnigibson.objects import PrimitiveObject, REGISTERED_OBJECTS
from omnigibson.tasks import DummyTask
from omnigibson.utils.python_utils import create_class_from_registry_and_config
from typing import Any, Iterable, Optional, Tuple

TASK_STATE = {
    "IDLE": 0,
    "RUNNING": 1,
    "SUCCESS": 2,
    "FAIL": 3,
}


class BaseTask(DummyTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = None
        self.robot = None
        self.n_trials = 0
        self.n_success_trials = 0
        self.object_idx = 0

    @property
    def object_config(self) -> list:
        return list()
    
    @property
    def viewer_camera_pose(self) -> Tuple[Iterable[float], Iterable[float]]:
        return [-1.87, 1.14, 1.66], [-0.18, 0.382, 0.828, -0.37]

    @classmethod
    def verify_scene_and_task_config(cls, scene_cfg, task_cfg):
        pass

    @property
    def default_robot_joint_pos(self) -> Optional[torch.Tensor]:
        """
        Default robot joint positions upon startup. 
        If return None, will use robot's default reset joint positions.
        
        Returns:
            Optional[torch.tensor]: optional default robot joint positions.
        """
        return None

    def _load(self, env: og.Environment) -> None:
        for obj_config in self.object_config:
            obj = env.scene.object_registry("name", obj_config["name"])
            # Create object
            if obj is None:
                obj = create_class_from_registry_and_config(
                    cls_name=obj_config["type"],
                    cls_registry=REGISTERED_OBJECTS,
                    cfg=obj_config,
                    cls_type_descriptor="object",
                )
                # Import the object into the simulator and set the pose
                env.scene.add_object(obj)

            obj_pos = torch.tensor([0.0, 0.0, 0.0]) if "position" not in obj_config else torch.tensor(obj_config["position"])
            obj_orn = torch.tensor([0.0, 0.0, 0.0, 1.0]) if "orientation" not in obj_config else torch.tensor(obj_config["orientation"])
            obj_pos, obj_orn = T.pose_transform(*env.scene.get_position_orientation(), obj_pos, obj_orn)
            obj.set_position_orientation(obj_pos, obj_orn)
        # update viewer camera pose if possible
        if og.sim.viewer_camera is not None:
            og.sim.viewer_camera.set_position_orientation(*self.viewer_camera_pose)
        # set default robot joint positions
        if self.default_robot_joint_pos is not None:
            env.robots[0]._reset_joint_pos = self.default_robot_joint_pos
        # if show_trajectory is set, import visual markers
        print("Loading visual markers..." * 10)
        if self._show_trajectory:
            objs = [PrimitiveObject(
                name=f"marker_{i}", 
                primitive_type="Sphere", 
                color=[1, 0, 0], 
                radius=0.01, 
                visual_only=True
            ) for i in range(100)]
            og.sim.batch_add_objects(objs, scenes=[env.scene] * len(objs))

    def _load_non_low_dim_observation_space(self) -> dict[str, Box]:
        """
        Load the non-low-dimensional observation space for the task.
        """
        raise NotImplementedError("Subclass must implement this method")

    def get_current_task_state_and_intent(self) -> Tuple[int, dict[str, Any]]:
        """
        Perform a backward checking on the current state of the task 
        Returns:
            Tuple[int, dict[str, Any]] subtask id and the intent for each part of the robot
        """
        raise NotImplementedError("Subclass must implement this method")
   
    def _reset_agent(self, env: og.Environment) -> None:
        for robot in env.robots:
            robot.reset()
        if self.policy is not None:
            self.policy.reset()
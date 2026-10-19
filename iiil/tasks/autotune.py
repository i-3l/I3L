import omnigibson.utils.transform_utils as T
import torch
from collections import defaultdict
from iiil.utils.misc_utils import ActionTypesSelection
from iiil.utils.goal_utils import BlockEnd, Timeout, EpisodeFail, EpisodeSucceed
from iiil.tasks.task_base import BaseTask, TASK_STATE
from omnigibson.utils.python_utils import classproperty
from typing import Any, Tuple


# Conservative reachable workspace for each arm (robot-local frame)
RIGHT_WS = {"x": (0.2, 0.65), "y": (-0.35, 0.05), "z": (0.5, 1.15)}
LEFT_WS = {"x": (0.2, 0.65), "y": (-0.05, 0.35), "z": (0.5, 1.15)}

# Small set of useful wrist orientations (local frame, as quaternions)
# forward-facing, downward, angled-left, angled-right
WRIST_ORIENTATIONS = [
    T.euler2quat(torch.tensor([0.0, 3.14159, 0.0])),           # forward-facing (palm down)
    T.euler2quat(torch.tensor([0.0, 3.14159 / 2, 0.0])),      # angled down 45 deg
    T.euler2quat(torch.tensor([3.14159 / 4, 3.14159, 0.0])),   # tilted left
    T.euler2quat(torch.tensor([-3.14159 / 4, 3.14159, 0.0])),  # tilted right
]

WAYPOINT_HOLD_STEPS = 80


class AutotuneTask(BaseTask):
    """
    Task that generates random waypoints for both arms independently.
    Designed for takeover threshold tuning — provides good bimanual joint coverage
    without requiring any scene objects or BDDL goals.
    """
    def __init__(
        self,
        show_trajectory: bool = False,
        termination_config=None,
        reward_config=None,
        **kwargs
    ):
        super().__init__(
            termination_config=termination_config,
            reward_config=reward_config,
        )
        self._show_trajectory = show_trajectory
        self.task_info = dict()
        self.task_info["task_state"] = TASK_STATE["RUNNING"]
        self.task_info["intent"] = {
            "target_pos": None,
            "target_quat": None,
            "action_type": None,
            "gripper_left": 1.0,
            "gripper_right": 1.0,
        }

        self._step_count = 0
        self._steps_at_target = 0
        self._left_target = None   # (pos, quat) in world frame
        self._right_target = None  # (pos, quat) in world frame

    def _sample_waypoint(self, ws):
        """Sample a random waypoint within workspace bounds, transformed to world frame."""
        local_pos = torch.tensor([
            ws["x"][0] + (ws["x"][1] - ws["x"][0]) * torch.rand(1).item(),
            ws["y"][0] + (ws["y"][1] - ws["y"][0]) * torch.rand(1).item(),
            ws["z"][0] + (ws["z"][1] - ws["z"][0]) * torch.rand(1).item(),
        ])
        local_quat = WRIST_ORIENTATIONS[torch.randint(len(WRIST_ORIENTATIONS), (1,)).item()]
        world_pos, world_quat = T.pose_transform(
            *self.robot.get_position_orientation(), local_pos, local_quat
        )
        return world_pos, world_quat

    def _sample_all_waypoints(self):
        """Sample fresh waypoints for both arms."""
        self._left_target = self._sample_waypoint(LEFT_WS)
        self._right_target = self._sample_waypoint(RIGHT_WS)
        self._steps_at_target = 0

    def _load_non_low_dim_observation_space(self):
        return dict()

    def get_current_task_state_and_intent(self) -> Tuple[int, dict[str, Any]]:
        left_eef_name = self.robot.eef_link_names["left"]
        right_eef_name = self.robot.eef_link_names["right"]
        intent = defaultdict(lambda: None)
        intent["action_type"] = ActionTypesSelection.DEFAULT
        intent["target_pos"] = {
            left_eef_name: self._left_target[0],
            right_eef_name: self._right_target[0],
        }
        intent["target_quat"] = {
            left_eef_name: self._left_target[1],
            right_eef_name: self._right_target[1],
        }
        intent["gripper_left"] = 1.0
        intent["gripper_right"] = 1.0
        return TASK_STATE["RUNNING"], intent

    def step(self, env, action) -> Tuple[float, bool, dict]:
        reward, done, info = super().step(env=env, action=action)

        self._step_count += 1
        self._steps_at_target += 1

        if self._steps_at_target >= WAYPOINT_HOLD_STEPS:
            self._sample_all_waypoints()

        subtask_id, current_intent = self.get_current_task_state_and_intent()
        info["subtask_id"] = subtask_id
        info["intent"] = current_intent
        self.task_info = info
        return reward, done, info

    def _get_obs(self, env):
        return dict(), dict()

    def reset(self, env) -> None:
        super().reset(env)
        self._step_count = 0
        if self.robot is not None:
            self._sample_all_waypoints()
        self.task_info = dict()

    def _create_termination_conditions(self):
        terminations = dict()

        total_ep = self._termination_config.get("total_episode")
        total_success_ep = self._termination_config.get("total_successful_episode")
        if total_ep and total_ep > 0:
            terminations["timeout"] = BlockEnd(total_episode=total_ep, total_successful_episode=total_success_ep)
        else:
            terminations["timeout"] = BlockEnd(total_episode=999999, total_successful_episode=total_success_ep)

        timeout_steps = self._termination_config.get("episode_timeout_step")
        if timeout_steps and timeout_steps > 0:
            terminations["max_time"] = Timeout(episode_timeout_step=timeout_steps)

        terminations["task_success"] = EpisodeSucceed()
        terminations["task_fail"] = EpisodeFail()
        return terminations

    @classproperty
    def default_termination_config(cls):
        return {
            "hold_step_threshold": 15,
            "episode_timeout_step": 600,
            "total_episode": 9999,
            "total_successful_episode": None,
        }

    @property
    def object_config(self) -> list:
        return list()

    def load(self, env):
        self._loaded = True

    def _reset_scene(self, env):
        pass

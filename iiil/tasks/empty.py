from typing import Any, Tuple
from iiil.tasks.task_base import BaseTask
from iiil.utils.goal_utils import BlockEnd, Timeout, EpisodeFail, EpisodeSucceed
from omnigibson.utils.python_utils import classproperty


class EmptyTask(BaseTask):
    """
    Minimal task that uses BaseTask's default behavior.
    Returns None for default_robot_joint_pos, allowing the robot to use its default reset positions.
    This makes it compatible with any robot type (A1, R1Pro, etc.).
    Can be used for data collection in real robot setups where no specific task is needed.
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
            reward_config=reward_config
        )
        self._show_trajectory = show_trajectory
        self.task_info = dict()
        # Set task state to RUNNING so that the policy is queried
        self.task_info["task_state"] = 1
        self.task_info["intent"] = {
            "target_pos": None,
            "target_quat": None,
            "action_type": None,
            # Gripper keys needed by CuroboPolicy
            "gripper_left": 0.0,
            "gripper_right": 0.0,
            "gripper_0": 0.0,
        }

    def _load_non_low_dim_observation_space(self):
        """
        Load the non-low-dimensional observation space for the task.
        Returns empty dict since this is a minimal task with no task-specific observations.
        """
        return dict()

    def get_current_task_state_and_intent(self) -> Tuple[int, dict[str, Any]]:
        """
        Get the current task state and intent.
        Returns a minimal implementation for the empty task.
        """
        return 0, {
            "intent": {
                "target_pos": None,
                "target_quat": None,
                "action_type": None,
                # Gripper keys needed by CuroboPolicy
                "gripper_left": 0.0,
                "gripper_right": 0.0,
                "gripper_0": 0.0,
            }
        }

    def load(self, env):
        """
        Don't need to do anything here
        """
        self._loaded = True

    def _reset_scene(self, env):
        """
        Don't do any reset here
        """
        pass

    def _create_termination_conditions(self):
        """
        Create the termination conditions for the teleop task.

        Supports disabling conditions via config:
        - episode_timeout_step: null or 0 → no timeout
        - total_episode: null or 0 → no episode limit (only Plus button ends block)
        """
        terminations = dict()

        # Block end: Plus button OR episode limit (if set)
        total_ep = self._termination_config.get("total_episode")
        total_success_ep = self._termination_config.get("total_successful_episode")
        if total_ep and total_ep > 0:
            terminations["timeout"] = BlockEnd(total_episode=total_ep, total_successful_episode=total_success_ep)
        else:
            # No episode limit - only Plus button ends block
            terminations["timeout"] = BlockEnd(total_episode=999999, total_successful_episode=total_success_ep)

        # Episode timeout (if set)
        timeout_steps = self._termination_config.get("episode_timeout_step")
        if timeout_steps and timeout_steps > 0:
            terminations["max_time"] = Timeout(episode_timeout_step=timeout_steps)
        # If null/0, no timeout condition added - episode only ends via buttons

        # Button-based termination (always enabled)
        terminations["task_success"] = EpisodeSucceed()
        terminations["task_fail"] = EpisodeFail()
        return terminations

    @classproperty
    def default_termination_config(cls):
        return {
            "hold_step_threshold": 15,
            "episode_timeout_step": 400,  # Set to null or 0 to disable
            "total_episode": 40,          # Set to null or 0 to disable
            "total_successful_episode": None,
        }
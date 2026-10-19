from omnigibson.termination_conditions.termination_condition_base import SuccessCondition, FailureCondition
from omnigibson.termination_conditions.predicate_goal import PredicateGoal as _PredicateGoal


class Timeout(FailureCondition):
    """
    Timeout (failure condition)
    Episode terminates if max_step steps have passed

    Args:
        max_step (int): Maximum number of steps before timeout occurs
    """

    def __init__(self, episode_timeout_step=10):
        # Store internal vars
        self._episode_timeout_step = episode_timeout_step
        # Run super
        super().__init__()

    def _step(self, task, env, action):
        # Terminate if number of steps passed exceeds threshold
        return env.episode_steps >= self._episode_timeout_step
    

class BlockEnd(FailureCondition):
    """
    BlockEnd (failure condition)
    Episode terminates if block end condition is met
    """
    def __init__(self, total_episode: int = None, total_successful_episode: int = None):
        super().__init__()
        self._total_episode = total_episode
        self._total_successful_episode = total_successful_episode

    def _step(self, task, env, action):
        if task.policy.check_block_ended():
            return True
        if self._total_episode is not None and env._current_episode >= self._total_episode:
            return True
        if self._total_successful_episode is not None and task.n_success_trials >= self._total_successful_episode:
            return True
        return False
    

class EpisodeSucceed(SuccessCondition):
    """
    Episode succeeded (success condition)
    Used for manually end the episode when the task is completed
    """
    def __init__(self):
        super().__init__()

    def _step(self, task, env, action):
        return task.policy.check_episode_ended(success=True)
    

class EpisodeFail(FailureCondition):
    """
    Episode failed (failure condition)
    Used for manually end the episode when the task is beyond recovery
    """
    def __init__(self):
        super().__init__()

    def _step(self, task, env, action):
        return task.policy.check_episode_ended(success=False)
    

class PredicateGoal(_PredicateGoal):
    """
    Overwrite the original PredicateGoal (success condition) to include hold time counter
    """

    def __init__(self, goal_fcn, hold_step_threshold: int=10):
        # Run super
        super().__init__(goal_fcn)
        self._hold_step_threshold = hold_step_threshold
        self._hold_counter = 0

    def reset(self, task, env):
        # Run super first
        super().reset(task, env)
        # reset counter
        self._hold_counter = 0

    def _step(self, task, env, action):
        done = super()._step(task, env, action)
        if done:
            self._hold_counter += 1
        else:
            self._hold_counter = 0
        return done and self._hold_counter >= self._hold_step_threshold

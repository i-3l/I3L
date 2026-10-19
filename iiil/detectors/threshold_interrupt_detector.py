import torch
from iiil.detectors.interrupt_detector import InterruptDetector
from omegaconf import DictConfig
from omnigibson.utils.processing_utils import MovingAverageFilter
from typing import Any


class ThresholdInterruptDetector(InterruptDetector):
    """
    A threshold-based interrupt detector that detects interruptions based on
    end-effector position differences and current thresholds.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the ThresholdInterruptDetector.
        
        Args:
            config (DictConfig): Configuration dictionary containing all parameters.
        """
        # Read all values from config
        filter_width = config.get('filter_width', 10)
        
        # Current threshold parameters
        self._current_thresholds = torch.tensor(config.get('current_thresholds', [250, 250, 250, 150, 150, 150]))

        # Initialize filters
        self._current_filter = MovingAverageFilter(obs_dim=6, filter_width=config.current_filter_width)
        self._prev_is_interrupted = False
        self._prev_robot_arm_q = None

    def is_interrupted(self, 
                       robot_obs: dict[str, Any], 
                       policy_action: torch.Tensor, 
                       oracle_obs: dict[str, Any]) -> bool:
        """
        Determine if the current policy action should be interrupted.
        
        Args:
            robot_obs (dict): The observation dictionary for the robot
            policy_action (torch.Tensor): The action proposed by the policy 
            oracle_obs (dict): The current Gello observation dictionary
            
        Returns:
            bool: True if interruption is detected, False otherwise
        """
        robot_arm_q = robot_obs["joint_angles"]
        robot_eef_pos = robot_obs["eef_position"]
        
        gello_arm_q = oracle_obs["joint_angles"]
        gello_eef_pos = oracle_obs["eef_position"]
        gello_arm_v = oracle_obs["joint_velocities"]

        is_interrupted = False

        # Check current thresholds
        gello_current = oracle_obs["joint_currents"]
        gello_current_smoothed = torch.from_numpy(self._current_filter.estimate(gello_current))

        gravity_currents = oracle_obs["gravity_currents"]

        gello_position_trajectory = oracle_obs["position_trajectory"]
        gello_velocity_trajectory = oracle_obs["velocity_trajectory"]

        tracking_currents = torch.abs(gello_current_smoothed - gravity_currents)
        current_violations = tracking_currents > self._current_thresholds
        if current_violations.any():
            is_interrupted = True
            violated_joints = torch.where(current_violations)[0]
            print(f"Joints {violated_joints.tolist()} exceeded current threshold")

        # Check joint position differences if we have previous robot state
        if self._prev_robot_arm_q is not None:
            joint_diff = torch.abs(self._prev_robot_arm_q - gello_arm_q)
            position_violations = joint_diff > 0.04
            if position_violations.any():
                is_interrupted = True
                violated_joints = torch.where(position_violations)[0]
                print(f"Joints {violated_joints.tolist()} exceeded position threshold")

        self._prev_is_interrupted = is_interrupted
        self._prev_robot_arm_q = robot_arm_q

        # Log data if logger is available
        if self._logger is not None:
            self._logger.log_data({
                "robot_arm_q": robot_arm_q,
                "policy_action": policy_action,
                "gello_arm_q": gello_arm_q,
                "gello_arm_v": gello_arm_v,
                "robot_eef_pos": robot_eef_pos,
                "gello_eef_pos": gello_eef_pos,
                "is_interrupted": is_interrupted,
                "gello_current": gello_current,
                "gello_current_smoothed": gello_current_smoothed,
                "gravity_currents": gravity_currents,
                "gello_position_trajectory": gello_position_trajectory,
                "gello_velocity_trajectory": gello_velocity_trajectory,
            })

        return is_interrupted

    def reset(self):
        """
        Resets the interrupt detector.
        """
        self._current_filter.reset()
        self._prev_is_interrupted = False
        self._prev_robot_arm_q = None
        if self._logger is not None:
            self._logger.reset()

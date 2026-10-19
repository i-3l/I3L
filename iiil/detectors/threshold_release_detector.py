import torch
from iiil.detectors.release_detector import ReleaseDetector
from omnigibson.utils.processing_utils import MovingAverageFilter
from omegaconf import DictConfig
from typing import Any


class ThresholdReleaseDetector(ReleaseDetector):
    """
    A release detector that detects when the GELLO end effector doesn't move 
    for a certain number of timesteps below a certain threshold.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the ThresholdReleaseDetector.
        
        Args:
            config (DictConfig): Configuration dictionary containing all parameters.
        """
        # Read all values from config
        self._movement_threshold = config.get('movement_threshold', 0.01)  # Threshold for movement detection
        self._stationary_timesteps = config.get('stationary_timesteps', 50)  # Number of timesteps to be stationary
        filter_width = config.get('filter_width', 10)

        # Movement tracking
        self._eef_movement_filter = MovingAverageFilter(obs_dim=1, filter_width=filter_width)
        self._stationary_counter = 0
        self._prev_eef_pos = None
        self._prev_is_released = False

    def is_released(
        self, 
        robot_obs: dict[str, Any], 
        policy_action: torch.Tensor, 
        oracle_obs: dict[str, Any]
    ) -> bool:
        """
        Determine if the Oracle has released control.

        Args:
            robot_obs (dict): The current observation dictionary
            policy_action (torch.Tensor): The action proposed by the policy
            oracle_obs (dict): The current Oracle observation dictionary

        Returns:
            bool: True if the Oracle arm is released, False otherwise
        """
        oracle_eef_pos = oracle_obs["eef_position"]

        # Calculate movement from previous position
        if self._prev_eef_pos is not None:
            eef_movement = torch.norm(oracle_eef_pos - self._prev_eef_pos)
            avg_eef_movement = self._eef_movement_filter.estimate(eef_movement)
            
            # Check if movement is below threshold
            if avg_eef_movement < self._movement_threshold:
                self._stationary_counter += 1
            else:
                self._stationary_counter = 0
        else:
            eef_movement = torch.tensor(0.0)
            avg_eef_movement = torch.tensor(0.0)
            self._stationary_counter = 0
        
        # Update previous position
        self._prev_eef_pos = oracle_eef_pos.clone()
        
        # Determine if released based on stationary counter
        is_released = self._stationary_counter >= self._stationary_timesteps
        
        # Reset counter if we detect movement after being released
        if is_released and not self._prev_is_released:
            # Just transitioned to released state
            pass
        elif not is_released and self._prev_is_released:
            # Just transitioned from released state, reset counter
            self._stationary_counter = 0
        
        self._prev_is_released = is_released
    
        # Log data if logger is available
        if self._logger is not None:
            self._logger.log_data({
                "oracle_eef_pos": oracle_eef_pos,
                "eef_movement": eef_movement,
                "stationary_counter": self._stationary_counter,
                "is_released": is_released,
            })

        return is_released
    
    def reset(self):
        """
        Resets the release detector.
        """
        self._prev_eef_pos = None
        self._prev_is_released = False
        self._stationary_counter = 0
        self._eef_movement_filter.reset()
        if self._logger is not None:
            self._logger.reset()
import torch
import numpy as np
import omnigibson.utils.transform_utils as T
from collections import deque
from typing import Any, Dict, Optional
from omegaconf import DictConfig
from omnigibson.utils.processing_utils import MovingAverageFilter
from iiil.detectors.interrupt_detector import InterruptDetector
from iiil.detectors.release_detector import ReleaseDetector
from iiil.detectors.policy_metrics import metric_registry


class PolicyComparisonDetector(InterruptDetector, ReleaseDetector):
    """
    A combined interrupt and release detector that compares outputs between base and oracle policies.
    
    This detector maintains buffers of past N timesteps of both base and oracle policy actions,
    computes various similarity metrics between them, and triggers interrupts/releases based on
    threshold comparisons.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the PolicyComparisonDetector.
        
        Args:
            config (DictConfig): Configuration dictionary containing all parameters.
        """
        # Buffer parameters (inspired by reference implementation)
        self._buffer_size = config.buffer_size
        self._base_action_buffer = deque(maxlen=self._buffer_size)
        self._oracle_action_buffer = deque(maxlen=self._buffer_size)

        # Threshold parameters
        self._interrupt_threshold = np.asarray(config.interrupt_threshold, dtype=np.float32)
        self._release_threshold = np.asarray(config.release_threshold, dtype=np.float32)
        self._moving_threshold = config.moving_threshold  # Minimum action magnitude
        
        # Distance metric (inspired by reference)
        distance_metric_name = config.distance_metric
        self._distance_metrics = metric_registry.get_distance_metric(distance_metric_name)
        
        # Aggregation metric (inspired by reference)
        aggregate_metric_name = config.aggregate_metric
        self._aggregate_metric = metric_registry.get_aggregate_metric(aggregate_metric_name)
        
        # Moving average filter for smoothing scores
        self._use_moving_average = config.get('use_moving_average', True)
        if self._use_moving_average:
            self._score_filter = MovingAverageFilter(obs_dim=len(self._distance_metrics), filter_width=config.get("ma_window", 5))
        else:
            self._score_filter = None
            
        # State tracking
        self._current_interrupt_scores = np.zeros(len(self._distance_metrics))
        self._current_release_scores = np.zeros(len(self._distance_metrics))

    def is_interrupted(self, detector_obs: dict[str, Any]) -> bool:
        """
        Determine if the current policy action should be interrupted.
        
        Args:
            detector_obs (dict): The observation dictionary containing all necessary information for detection
            
        Returns:
            bool: True if interruption is detected, False otherwise
        """
        scores = self._compute_scores(detector_obs)
        if scores is None:
            return False

        self._current_interrupt_scores = scores
        return np.any(self._current_interrupt_scores > self._interrupt_threshold)

    def is_released(self, detector_obs: dict[str, Any]) -> bool:
        """
        Determine if the Oracle has released control.

        Args:
            detector_obs (dict): The observation dictionary containing all necessary information for detection

        Returns:
            bool: True if the Oracle arm is released, False otherwise
        """
        scores = self._compute_scores(detector_obs)
        if scores is None:
            return False

        self._current_release_scores = scores
        return np.all(self._current_release_scores < self._release_threshold)

    def _compute_scores(self, detector_obs: dict[str, Any]) -> Optional[np.ndarray]:
        policy_action, oracle_action = self._extract_compare_actions(detector_obs)

        # Check if actions are too small (inspired by reference moving_threshold)
        if policy_action.norm() < self._moving_threshold:
            return None

        oracle_action = oracle_action.clone()
        policy_action = policy_action.clone()
        oracle_action[oracle_action.abs() < self._moving_threshold] = 0.0
        policy_action[policy_action.abs() < self._moving_threshold] = 0.0

        self._update_buffers(policy_action, oracle_action)
        if len(self._base_action_buffer) < 1:
            return None

        base_action_current = self._aggregate_metric(self._base_action_buffer)
        oracle_action_current = self._aggregate_metric(self._oracle_action_buffer)
        scores = np.asarray(
            [metric(base_action_current, oracle_action_current) for metric in self._distance_metrics],
            dtype=np.float32,
        )
        if self._score_filter is not None:
            scores = np.asarray(self._score_filter.estimate(0, scores), dtype=np.float32)
        return scores

    def _extract_compare_actions(self, detector_obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        policy_obs = detector_obs["policy"]
        if "base_compare_action" in policy_obs and "oracle_compare_action" in policy_obs:
            return (
                policy_obs["base_compare_action"].detach().cpu().float().flatten(),
                policy_obs["oracle_compare_action"].detach().cpu().float().flatten(),
            )

        return self._extract_single_arm_eef_actions(detector_obs)

    def _extract_single_arm_eef_actions(self, detector_obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        oracle_action = torch.cat([
            detector_obs["policy"]["oracle_target_eef_pose"][0][0] - detector_obs["robot"]["eef_pose"][0][0],
            T.quat2euler(T.quat_distance(detector_obs["policy"]["oracle_target_eef_pose"][1][0], detector_obs["robot"]["eef_pose"][1][0])),
            detector_obs["policy"]["oracle_action"][-1:] - detector_obs["robot"]["gripper_states"],
        ])
        policy_action = torch.cat([
            detector_obs["policy"]["base_target_eef_pose"][0][0] - detector_obs["robot"]["eef_pose"][0][0],
            T.quat2euler(T.quat_distance(detector_obs["policy"]["base_target_eef_pose"][1][0], detector_obs["robot"]["eef_pose"][1][0])),
            detector_obs["policy"]["base_action"][-1:] - detector_obs["robot"]["gripper_states"],
        ])
        return policy_action, oracle_action

    def _update_buffers(self, base_action: torch.Tensor, oracle_action: torch.Tensor) -> None:
        """
        Update the action buffers with new actions.
        
        Args:
            base_action (torch.Tensor): Action from base policy
            oracle_action (torch.Tensor): Action from oracle policy
        """
        # Convert to numpy for easier processing
        base_action_np = base_action.detach().cpu().numpy()
        oracle_action_np = oracle_action.detach().cpu().numpy()
        
        self._base_action_buffer.append(base_action_np)
        self._oracle_action_buffer.append(oracle_action_np)

    def reset(self):
        """
        Resets the detector.
        """
        self._base_action_buffer.clear()
        self._oracle_action_buffer.clear()
        
        if self._score_filter is not None:
            self._score_filter.reset()

    def get_current_state(self) -> Optional[Dict[str, torch.Tensor]]:
        return {
            "detector_interrupt_scores": torch.as_tensor(self._current_interrupt_scores, dtype=torch.float32),
            "detector_release_scores": torch.as_tensor(self._current_release_scores, dtype=torch.float32),
        }

import joblib
import numpy as np
import torch
from iiil.detectors.interrupt_detector import InterruptDetector
from omegaconf import DictConfig
from typing import Any


class SklearnRegressionInterruptDetector(InterruptDetector):
    """
    An interrupt detector that uses a scikit-learn regression model to predict expected joint currents.
    If the measured current deviates from the predicted value by more than a threshold, interruption is triggered.
    """
    def __init__(self, config: DictConfig):
        """
        Initialize the SklearnRegressionInterruptDetector.
        Args:
            config (DictConfig): Configuration dictionary containing all parameters.
        """
        self._model_path = config.model_path
        self._current_diff_threshold = config.current_diff_threshold
        self._obs_features = config.obs_features
        self._prev_is_interrupted = False
        # Load the sklearn regression model
        self._model = joblib.load(self._model_path)

        self._prev_currents = None

    def _get_input_features(self, robot_obs: dict[str, Any], oracle_obs: dict[str, Any]) -> np.ndarray:
        # Get joint angle differences between robot and GELLO
        robot_joints = robot_obs["joint_angles"]
        gello_joints = oracle_obs["joint_angles"]
        if isinstance(robot_joints, torch.Tensor):
            robot_joints = robot_joints.cpu().numpy()
        if isinstance(gello_joints, torch.Tensor):
            gello_joints = gello_joints.cpu().numpy()
        joint_diffs = robot_joints - gello_joints

        # Get gravity currents
        gravity_currents = oracle_obs["gravity_currents"]
        if isinstance(gravity_currents, torch.Tensor):
            gravity_currents = gravity_currents.cpu().numpy()

        # Get previous currents
        prev_currents = self._prev_currents
        if isinstance(prev_currents, torch.Tensor):
            prev_currents = prev_currents.cpu().numpy()

        # Concatenate features
        features = np.concatenate([joint_diffs, gravity_currents, prev_currents])
        return features


    def is_interrupted(self, 
                       robot_obs: dict[str, Any], 
                       policy_action: torch.Tensor, 
                       oracle_obs: dict[str, Any]) -> bool:
        """
        Use the regression model to predict expected current, compare to measured, and trigger if difference is too large.
        """
        if self._prev_currents is None:
            self._prev_currents = oracle_obs["joint_currents"]
            return False

        # Prepare features as a flat numpy array
        features = self._get_input_features(robot_obs, oracle_obs)
        features = np.reshape(features, (1, -1))

        # Predict expected current
        predicted_current_delta = self._model.predict(features).flatten()  # shape (6,)
        measured_current = oracle_obs["joint_currents"]
        measured_current_delta = measured_current - self._prev_currents
        if isinstance(measured_current_delta, torch.Tensor):
            measured_current_delta = measured_current_delta.cpu().numpy()
        current_diff = np.abs(measured_current_delta - predicted_current_delta)
        is_interrupted = (current_diff > self._current_diff_threshold).any()
        self._prev_is_interrupted = is_interrupted

        # Log data if logger is available
        if self._logger is not None:
            self._logger.log_data({
                "features": torch.tensor(features.squeeze()),
                "predicted_current": torch.tensor(predicted_current_delta),
                "measured_current": torch.tensor(measured_current_delta),
                "current_diff": torch.tensor(current_diff),
                "is_interrupted": is_interrupted,
            })
        self._prev_currents = measured_current
        return is_interrupted

    def reset(self):
        self._prev_is_interrupted = False
        if self._logger is not None:
            self._logger.reset() 
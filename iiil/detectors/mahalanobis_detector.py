import numpy as np
import torch
from typing import Any, Dict, Optional
from omegaconf import DictConfig
from omnigibson.utils.processing_utils import MovingAverageFilter
from iiil.detectors.interrupt_detector import InterruptDetector
from iiil.detectors.release_detector import ReleaseDetector


class MahalanobisDetector(InterruptDetector, ReleaseDetector):
    """Combined interrupt + release detector using Mahalanobis distance.

    Above threshold = OOD = interrupt (policy failing).
    Below threshold for window_size consecutive checks = in-distribution = release.
    """

    def __init__(self, config: DictConfig):
        data = np.load(config.calibration_path)
        self._mean = data["mean"]
        self._precision = data["precision"]
        self._interrupt_threshold = float(config.interrupt_threshold)
        self._release_threshold = float(config.release_threshold)
        self._window_size = int(config.get("window_size", 5))

        self._use_moving_average = config.get("use_moving_average", True)
        if self._use_moving_average:
            ma_window = int(config.get("ma_window", 5))
            self._ma_filter = MovingAverageFilter(obs_dim=1, filter_width=ma_window)
        else:
            self._ma_filter = None

        self._consecutive_in_dist = 0
        self._consecutive_ood = 0
        self._interrupt_window = int(config.get("interrupt_window", 3))
        self._current_distance = 0.0

    def _compute_distance(self, embedding: np.ndarray) -> float:
        while embedding.ndim > 2:
            embedding = embedding.squeeze(0)
        if embedding.ndim == 2:
            embedding = embedding.mean(axis=0)
        delta = embedding - self._mean
        dist_sq = float(delta @ self._precision @ delta)
        return np.sqrt(max(dist_sq, 0.0))

    def _update_distance(self, detector_obs: dict[str, Any]) -> bool:
        embedding = detector_obs["policy"].get("obs_embedding", None)
        if embedding is None:
            return False

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        distance = self._compute_distance(embedding)

        if self._ma_filter is not None:
            distance = float(self._ma_filter.estimate(0, np.array([distance]))[0])

        self._current_distance = distance
        return True

    def is_interrupted(self, detector_obs: dict[str, Any]) -> bool:
        if not self._update_distance(detector_obs):
            return False

        if self._current_distance > self._interrupt_threshold:
            self._consecutive_ood += 1
        else:
            self._consecutive_ood = 0

        return self._consecutive_ood >= self._interrupt_window

    def is_released(self, detector_obs: dict[str, Any]) -> bool:
        if not self._update_distance(detector_obs):
            return False

        if self._current_distance < self._release_threshold:
            self._consecutive_in_dist += 1
        else:
            self._consecutive_in_dist = 0

        return self._consecutive_in_dist >= self._window_size

    def reset(self):
        self._consecutive_in_dist = 0
        self._consecutive_ood = 0
        self._current_distance = 0.0
        if self._ma_filter is not None:
            self._ma_filter.reset()

    def get_current_state(self) -> Optional[Dict[str, torch.Tensor]]:
        return {
            "mahalanobis_distance": torch.tensor([self._current_distance], dtype=torch.float32),
        }

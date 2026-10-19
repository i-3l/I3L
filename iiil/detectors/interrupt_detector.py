import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class InterruptDetector(ABC):
    """
    Abstract base class for interrupt detection.
    An interrupt detector determines if the current policy action should be interrupted
    based on the current observation, policy action, and Oracle observation.
    """

    @abstractmethod
    def is_interrupted(self, detector_obs: dict[str, Any]) -> bool:
        """
        Determine if the current policy action should be interrupted.

        Args:
            detector_obs (dict): The observation dictionary containing all necessary information for detection

        Returns:
            bool: True if the Oracle motion is being interrupted, False otherwise
        """
        raise NotImplementedError('Subclasses need to implement this')

    @abstractmethod
    def reset(self):
        """
        Resets the interrupt detector.
        """
        raise NotImplementedError('Subclasses need to implement this')

    @abstractmethod
    def get_current_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Generate state information for logging/visualization
        """
        raise NotImplementedError('Subclasses need to implement this')

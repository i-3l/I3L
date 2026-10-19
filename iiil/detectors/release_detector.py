import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ReleaseDetector(ABC):
    """
    Abstract base class for release detection.
    A release detector determines if the Oracle has released control based on the current Oracle observation.
    In the context the GELLO robot, this typically means the human has let go of the robot arm.
    """

    @abstractmethod
    def is_released(self, detector_obs: dict[str, Any]) -> bool:
        """
        Determine if the Oracle has released control.

        Args:
            detector_obs (dict): The observation dictionary containing all necessary information for detection
            oracle_obs (dict): The current Oracle observation dictionary

        Returns:
            bool: True if the Oracle arm is released, False otherwise
        """
        raise NotImplementedError('Subclasses need to implement this')
    
    @abstractmethod
    def reset(self):
        """
        Resets the release detector.
        """
        raise NotImplementedError('Subclasses need to implement this')

    @abstractmethod
    def get_current_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Generate state information for logging/visualization
        """
        raise NotImplementedError('Subclasses need to implement this')

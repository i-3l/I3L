import torch
from iiil.detectors.interrupt_detector import InterruptDetector
from iiil.detectors.release_detector import ReleaseDetector
from iiil.detectors.serial_voltage_reader import SerialVoltageReader
from omegaconf import DictConfig
from typing import Any, Dict, Optional


class VoltageBasedDetector(InterruptDetector, ReleaseDetector):
    """
    A unified voltage-based detector that can work as both an interrupt detector and release detector.
    
    The detector reads voltage values from a serial voltage reader and compares them against a threshold.
    The comparison direction (above/below threshold) and behavior (immediate vs consecutive timesteps)
    are configurable.
    """
    
    def __init__(self, config: DictConfig, voltage_reader: SerialVoltageReader, mode: str = "interrupt"):
        """
        Args:
            config (DictConfig): Configuration dictionary containing all parameters.
            voltage_reader (SerialVoltageReader): Object to read voltage values.
            mode (str): Either "interrupt" or "release" to determine the behavior.
        """
        self._threshold = config.voltage_threshold
        self._voltage_reader = voltage_reader
        self._mode = mode
        
        # For release mode, we need consecutive timestep counting
        if mode == "release":
            self._consecutive_timesteps = config.get('consecutive_timesteps', 10)
            self._counter = 0
            self._prev_is_released = False
        else:
            self._consecutive_timesteps = None
            self._counter = None
            self._prev_is_released = None

    def is_interrupted(self, robot_obs: dict[str, Any], policy_action: torch.Tensor, oracle_obs: dict[str, Any]) -> bool:
        """
        Determine if the current policy action should be interrupted.
        
        For interrupt mode: triggers when voltage is above threshold (indicating force/resistance)
        """
        if self._mode != "interrupt":
            raise ValueError("This detector is not configured for interrupt mode")
            
        value = self._voltage_reader.read_voltage()
        is_interrupted = value > self._threshold
        return is_interrupted

    def is_released(
        self, 
        robot_obs: dict[str, Any], 
        policy_action: torch.Tensor, 
        oracle_obs: dict[str, Any]
    ) -> bool:
        """
        Determine if the human has released the GELLO arm.
        
        For release mode: triggers when voltage is below threshold for consecutive timesteps
        """
        if self._mode != "release":
            raise ValueError("This detector is not configured for release mode")
            
        value = self._voltage_reader.read_voltage()
        if value < self._threshold:
            self._counter += 1
        else:
            self._counter = 0
        is_released = self._counter >= self._consecutive_timesteps
        self._prev_is_released = is_released
        return is_released

    def reset(self):
        """Resets the detector state."""
        if self._mode == "release":
            self._counter = 0
            self._prev_is_released = False 

    def get_current_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Generate state information for logging/visualization
        """
        state = {
            "voltage_threshold": torch.tensor(self._threshold),
            "current_voltage": torch.tensor(self._voltage_reader.read_voltage())
        }
        if self._mode == "release":
            state["consecutive_below_threshold"] = torch.tensor(self._counter)
            state["is_released"] = torch.tensor(self._prev_is_released)
        return state
"""
SpaceMouse teleop policy.

Wraps PySpaceMouse for 3DConnexion SpaceMouse control.
"""

import threading
from importlib import import_module
from omegaconf import DictConfig
import torch

from iiil.policies.policy_base import BasePolicy


class SpaceMousePolicy(BasePolicy):
    """
    Wraps around PySpaceMouse for using the 3DConnexion SpaceMouse for control.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.pyspacemouse = import_module('pyspacemouse')
        self.current_gripper = 1.0
        self.gripper_button_pressed = False
        self.raw_data = None
        self._thread = None

    def predict(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        if self.raw_data:
            action = torch.cat([
                torch.tensor([
                    self.raw_data.y, -self.raw_data.x, self.raw_data.z,
                    self.raw_data.roll, self.raw_data.pitch, -self.raw_data.yaw,
                ]),
                torch.zeros(1)
            ])
            if self.raw_data.buttons[1]:
                if not self.gripper_button_pressed:
                    self.gripper_button_pressed = True
                    self.current_gripper = -self.current_gripper
            else:
                self.gripper_button_pressed = False
            action[-1] = self.current_gripper
            self.previous_action = action
        else:
            action = torch.zeros(7)
        return action.unsqueeze(0).to(self.device)

    def load(self, **kwargs) -> None:
        # Start spacemouse
        assert self.pyspacemouse.open(), \
            "[SpaceMouse] Cannot connect to SpaceMouse!"
        self._thread = threading.Thread(target=self._update_internal_data, daemon=True)
        self._thread.start()

    def reset(self) -> None:
        self.current_gripper = 1.0
        self.gripper_button_pressed = False
        self.raw_data = None

    def close(self) -> None:
        if self._thread:
            self._thread.join(timeout=1.0)

    def check_episode_ended(self, success: bool = True) -> bool:
        return self.raw_data and self.raw_data.buttons[0]

    def _update_internal_data(self) -> None:
        """
        Thread that stores the spacemouse input to self.raw_data.
        """
        while True:
            self.raw_data = self.pyspacemouse.read()


__all__ = ["SpaceMousePolicy"]

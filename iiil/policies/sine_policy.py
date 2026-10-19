"""Sine wave policy for testing teleop pipelines without a trained model.

Oscillates each arm joint with a phase offset, producing smooth repeating
motion that is easy to visually verify.  Gripper stays open.
"""

import math
import time

import numpy as np
import torch
from omegaconf import DictConfig

from iiil.policies.policy_base import BasePolicy

# Franka home position (same as RealFranka.move_to_home)
_HOME = np.array([0.0, 0.0, 0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 4])


class SineWavePolicy(BasePolicy):
    """Deterministic joint-space sine wave — no model, no server."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.amplitude = getattr(config, "amplitude", 0.3)
        self.period = getattr(config, "period", 4.0)
        self.action_dim = config.action_dim
        self._t0 = time.time()

    def predict(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        t = time.time() - self._t0
        omega = 2 * math.pi / self.period
        offsets = np.array([
            self.amplitude * math.sin(omega * t + i * math.pi / 4)
            for i in range(7)
        ])
        arm = _HOME + offsets
        gripper = -1.0  # open
        action = np.concatenate([arm, [gripper]])
        # Pad if action_dim > 8 (e.g. for robots with base/torso)
        if self.action_dim > 8:
            action = np.concatenate([np.zeros(self.action_dim - 8), action])
        return torch.from_numpy(action).float()

    def load(self, **kwargs):
        self._t0 = time.time()

    def reset(self) -> None:
        self._t0 = time.time()

    def close(self) -> None:
        pass

    def check_episode_ended(self, success: bool = True) -> bool:
        return False

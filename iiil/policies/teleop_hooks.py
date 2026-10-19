"""
Hook system for teleop pipeline with preprocessing/postprocessing support.
"""
from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np
import torch


@dataclass
class GelloState:
    """Snapshot of Gello hardware state at a given time."""

    joint_positions: np.ndarray      # (dof,) joint angles in radians
    joint_velocities: np.ndarray     # (dof,) joint velocities in rad/s
    valid_joints: np.ndarray         # (dof,) bool mask for valid readings
    timestamp: float                 # Time when state was captured


@dataclass
class EnvState:
    """Snapshot of environment/robot state."""

    robot_proprio: torch.Tensor      # Full proprioception from observation
    timestamp: float                 # Time when state was captured


class TeleopHookRegistry:
    """
    Registry for teleop callbacks that can preprocess/postprocess state.

    Hooks are executed in priority order (lower priority values run first).
    Each hook receives the current state and returns a potentially modified state.

    Example usage:
        registry = TeleopHookRegistry()

        # Register a deadzone filter
        def deadzone_filter(state: GelloState) -> GelloState:
            threshold = 0.01  # rad/s
            velocities = state.joint_velocities.copy()
            velocities[np.abs(velocities) < threshold] = 0.0
            return dataclasses.replace(state, joint_velocities=velocities)

        registry.register_gello_hook(deadzone_filter, priority=10)
    """

    def __init__(self):
        self._gello_hooks: List[Tuple[int, Callable[[GelloState], GelloState]]] = []
        self._env_hooks: List[Tuple[int, Callable[[EnvState], EnvState]]] = []

    def register_gello_hook(self, hook: Callable[[GelloState], GelloState], priority: int = 0):
        """
        Register a callback for gello state updates.

        Args:
            hook: Function that takes GelloState and returns modified GelloState
            priority: Execution priority (lower values run first)
        """
        self._gello_hooks.append((priority, hook))
        self._gello_hooks.sort(key=lambda x: x[0])

    def register_env_hook(self, hook: Callable[[EnvState], EnvState], priority: int = 0):
        """
        Register a callback for environment state updates.

        Args:
            hook: Function that takes EnvState and returns modified EnvState
            priority: Execution priority (lower values run first)
        """
        self._env_hooks.append((priority, hook))
        self._env_hooks.sort(key=lambda x: x[0])

    def emit_gello_update(self, state: GelloState) -> GelloState:
        """
        Call all registered gello hooks in priority order.

        Args:
            state: Current gello state

        Returns:
            Modified gello state after all hooks have been applied
        """
        for _, hook in self._gello_hooks:
            state = hook(state)
        return state

    def emit_env_update(self, state: EnvState) -> EnvState:
        """
        Call all registered env hooks in priority order.

        Args:
            state: Current environment state

        Returns:
            Modified environment state after all hooks have been applied
        """
        for _, hook in self._env_hooks:
            state = hook(state)
        return state

    def clear_gello_hooks(self):
        """Remove all registered gello hooks."""
        self._gello_hooks.clear()

    def clear_env_hooks(self):
        """Remove all registered env hooks."""
        self._env_hooks.clear()

    def clear_all_hooks(self):
        """Remove all registered hooks."""
        self.clear_gello_hooks()
        self.clear_env_hooks()


__all__ = ["GelloState", "EnvState", "TeleopHookRegistry"]

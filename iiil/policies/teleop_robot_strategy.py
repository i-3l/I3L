"""
Strategy pattern for robot-type-specific behavior.

Eliminates repetitive A1 vs R1Pro branching throughout the codebase.
Each robot type has its own strategy that encapsulates:
- Action layout configuration
- Expected Gello config
- Arm controller access
- Controller mode setting
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class RobotTypeStrategy(ABC):
    """
    Abstract base for robot-type-specific behavior.

    Encapsulates all behavior that differs between robot types (A1, R1Pro, etc.).
    Adding a new robot type only requires creating a new strategy subclass.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Robot type identifier (e.g., 'A1', 'R1Pro')."""
        pass

    @property
    @abstractmethod
    def action_layout(self) -> Dict[str, Any]:
        """
        Return action key configuration for this robot type.

        Returns:
            Dict with:
                - 'action_keys': List of action key names in order
                - 'action_key_dims': Dict mapping key names to dimensions
        """
        pass

    @property
    @abstractmethod
    def expected_gello_config(self) -> str:
        """Expected Gello config name for this robot type."""
        pass

    @abstractmethod
    def get_arm_controller(self, robot) -> Any:
        """
        Get the primary arm controller from robot.

        Args:
            robot: Robot instance with controllers dict

        Returns:
            Primary arm controller instance
        """
        pass

    @abstractmethod
    def set_controller_mode(self, robot, mode: str) -> None:
        """
        Set controller mode for all arms on this robot.

        Args:
            robot: Robot instance
            mode: Controller mode ("joint" or policy control type)
        """
        pass

    @abstractmethod
    def get_arm_controller_names(self) -> List[str]:
        """Return list of arm controller names for this robot."""
        pass

    def validate_gello_config(self, gello_config_name: str) -> None:
        """
        Validate that Gello config matches this robot type.

        Args:
            gello_config_name: Name of loaded Gello config

        Raises:
            ValueError: If config doesn't match expected
        """
        if gello_config_name != self.expected_gello_config:
            raise ValueError(
                f"Gello config mismatch: Robot '{self.name}' requires '{self.expected_gello_config}', "
                f"but '{gello_config_name}' is loaded. This will cause DOF mismatches!"
            )


class A1Strategy(RobotTypeStrategy):
    """Strategy for A1 robot (single 6-DOF arm)."""

    @property
    def name(self) -> str:
        return "A1"

    @property
    def action_layout(self) -> Dict[str, Any]:
        return {
            "action_keys": ["arm", "gripper"],
            "action_key_dims": {"arm": 6, "gripper": 1}
        }

    @property
    def expected_gello_config(self) -> str:
        return "gello_a1"

    def get_arm_controller(self, robot) -> Any:
        """Get arm_0 controller for A1.

        Returns None for real robots that don't have OmniGibson controllers.
        """
        controllers = getattr(robot, 'controllers', {})
        if not controllers:
            # Real robot - no controllers dict
            return None
        if "arm_0" not in controllers:
            raise KeyError(f"A1 robot must have 'arm_0' controller, found: {list(controllers.keys())}")
        return controllers["arm_0"]

    def set_controller_mode(self, robot, mode: str) -> None:
        """Set controller mode for A1's single arm.

        No-op for real robots that don't have OmniGibson controllers.
        """
        controllers = getattr(robot, 'controllers', {})
        if not controllers:
            return
        if "arm_0" in controllers:
            controllers["arm_0"].controller_mode = mode

    def get_arm_controller_names(self) -> List[str]:
        return ["arm_0"]


class R1ProStrategy(RobotTypeStrategy):
    """Strategy for R1Pro robot (dual 7-DOF arms)."""

    @property
    def name(self) -> str:
        return "R1Pro"

    @property
    def action_layout(self) -> Dict[str, Any]:
        return {
            "action_keys": ["base", "torso", "left_arm", "left_gripper", "right_arm", "right_gripper"],
            "action_key_dims": {
                "base": 3,
                "torso": 4,
                "left_arm": 7,
                "left_gripper": 1,
                "right_arm": 7,
                "right_gripper": 1
            }
        }

    @property
    def expected_gello_config(self) -> str:
        return "gello_r1pro"

    def get_arm_controller(self, robot) -> Any:
        """Get primary arm controller for R1Pro (prefers left arm).

        Returns None for real robots that don't have OmniGibson controllers.
        """
        controllers = getattr(robot, 'controllers', {})
        if not controllers:
            # Real robot - no controllers dict
            return None
        if "arm_left" in controllers:
            return controllers["arm_left"]
        if "arm_right" in controllers:
            return controllers["arm_right"]
        raise KeyError(
            f"R1Pro robot must have 'arm_left' or 'arm_right' controller, "
            f"found: {list(controllers.keys())}"
        )

    def set_controller_mode(self, robot, mode: str) -> None:
        """Set controller mode for both R1Pro arms.

        No-op for real robots that don't have OmniGibson controllers.
        """
        controllers = getattr(robot, 'controllers', {})
        if not controllers:
            # Real robot - no controllers to set
            return
        for arm_name in ["arm_left", "arm_right"]:
            if arm_name in controllers:
                controllers[arm_name].controller_mode = mode

    def get_arm_controller_names(self) -> List[str]:
        return ["arm_left", "arm_right"]


class FrankaStrategy(RobotTypeStrategy):
    """Strategy for Franka Panda robot (single 7-DOF arm)."""

    @property
    def name(self) -> str:
        return "Franka"

    @property
    def action_layout(self) -> Dict[str, Any]:
        return {
            "action_keys": ["arm", "gripper"],
            "action_key_dims": {"arm": 7, "gripper": 1}
        }

    @property
    def expected_gello_config(self) -> str:
        return "gello_franka"

    def get_arm_controller(self, robot) -> Any:
        """Returns None for real Franka (no OmniGibson controllers)."""
        controllers = getattr(robot, 'controllers', {})
        if not controllers:
            return None
        if "arm_0" not in controllers:
            raise KeyError(f"Franka robot must have 'arm_0' controller, found: {list(controllers.keys())}")
        return controllers["arm_0"]

    def set_controller_mode(self, robot, mode: str) -> None:
        """No-op for real Franka (no OmniGibson controllers)."""
        controllers = getattr(robot, 'controllers', {})
        if not controllers:
            return
        if "arm_0" in controllers:
            controllers["arm_0"].controller_mode = mode

    def get_arm_controller_names(self) -> List[str]:
        return ["arm_0"]


# Registry of known robot strategies
_ROBOT_STRATEGIES: Dict[str, type] = {
    "A1": A1Strategy,
    "R1Pro": R1ProStrategy,
    "RealR1Pro": R1ProStrategy,
    "Franka": FrankaStrategy,
    "RealFranka": FrankaStrategy,
}


def get_robot_strategy(robot_type: str) -> RobotTypeStrategy:
    """
    Factory function to get the appropriate strategy.

    Args:
        robot_type: Robot type string (e.g., "A1", "R1Pro")

    Returns:
        Appropriate strategy instance
    """
    if robot_type not in _ROBOT_STRATEGIES:
        logger.warning(f"Unknown robot type '{robot_type}', defaulting to A1Strategy")
        return A1Strategy()

    return _ROBOT_STRATEGIES[robot_type]()


def get_robot_strategy_from_config(robot_cfg) -> Optional[RobotTypeStrategy]:
    """
    Get strategy from robot config object.

    Args:
        robot_cfg: Robot configuration (dict, DictConfig, or string)

    Returns:
        Strategy instance or None if robot type cannot be determined
    """
    robot_type = None

    if isinstance(robot_cfg, str):
        robot_type = robot_cfg
    elif isinstance(robot_cfg, dict):
        robot_type = robot_cfg.get("type")
    elif robot_cfg is not None:
        robot_type = getattr(robot_cfg, "type", None)

    if robot_type is None:
        return None

    return get_robot_strategy(robot_type)


__all__ = [
    "RobotTypeStrategy",
    "A1Strategy",
    "R1ProStrategy",
    "FrankaStrategy",
    "get_robot_strategy",
    "get_robot_strategy_from_config",
]

"""
Builder pattern for GelloPolicy construction.

Separates object construction from representation, making it easy to:
- Create different configurations
- Support new robot types
- Enable testing with mock dependencies
- Handle deferred initialization (init vs load split)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import logging
import numpy as np

from omegaconf import DictConfig

from iiil.gello.controller_factory import A1GelloControllerFactory
from iiil.policies.teleop_types import GelloControlDirection, TeleopRuntimeState
from iiil.policies.teleop_layout import build_action_layout, build_arm_mappings
from iiil.policies.teleop_hooks import TeleopHookRegistry
from iiil.policies.teleop_modes import ControlModeManager, GelloToEnvMode, EnvToGelloMode, AdmittanceMode
from iiil.policies.teleop_arms import UnifiedArmController, ArmController
from iiil.policies.teleop_pipeline import TeleopPipeline
from iiil.policies.teleop_config import parse_gating_config
from iiil.policies.teleop_gating import AdmittanceGate
from iiil.policies.teleop_input import InputHandler
from iiil.policies.teleop_movement import GelloMovementHelper
from iiil.policies.teleop_state import DataCollectionState
from iiil.policies.teleop_robot_strategy import (
    RobotTypeStrategy,
    get_robot_strategy,
    get_robot_strategy_from_config,
)
from iiil.policies.teleop_locomotion import (
    BaseLocomotionHandler,
    get_locomotion_handler,
)

logger = logging.getLogger(__name__)


@dataclass
class ImmediateComponents:
    """
    Components built during __init__ (before robot is available).

    These don't require the actual robot instance.
    """
    gello_controller: Any  # BilateralController
    input_handler: InputHandler
    hooks: TeleopHookRegistry
    runtime_state: TeleopRuntimeState
    data_state: DataCollectionState

    # Layout info (preliminary, may be rebuilt in load)
    gello_total_dof: int
    gello_num_arms: int
    gello_arm_dof: int

    # Config values
    initial_direction: GelloControlDirection
    base_policy_type: str
    convergence_tolerance: float
    max_convergence_wait_time: float


@dataclass
class DeferredComponents:
    """
    Components built during load() (when robot is available).

    These require the actual robot instance for DOF/layout info.
    """
    robot_strategy: RobotTypeStrategy
    arm_controller: UnifiedArmController
    arm_controller_ref: Any  # Robot's arm controller
    action_dim: int
    action_slices: Dict[str, slice]
    gripper_slices: List[slice]
    arm_mappings: List[Any]


@dataclass
class GelloPolicyComponents:
    """
    Container for all GelloPolicy dependencies.

    Combines immediate and deferred components into a single interface.
    """
    # Core controllers
    gello_controller: Any  # BilateralController
    mode_manager: ControlModeManager
    pipeline: TeleopPipeline
    arm_controller: UnifiedArmController

    # Input/output handling
    input_handler: InputHandler
    gate: Optional[AdmittanceGate]

    # Strategies and helpers
    robot_strategy: RobotTypeStrategy
    movement_helper: GelloMovementHelper

    # State tracking
    runtime_state: TeleopRuntimeState
    data_state: DataCollectionState

    # Base policy (created separately to handle load())
    base_policy: Any  # BasePolicy
    base_policy_control_type: str

    # Layout info
    action_dim: int
    action_slices: Dict[str, slice]
    gripper_slices: List[slice]

    # Locomotion handler for base/torso/gripper control (robot-type specific)
    locomotion_handler: BaseLocomotionHandler = None


class GelloPolicyBuilder:
    """
    Builder for GelloPolicy with deferred initialization support.

    Handles the complex construction of GelloPolicy by separating:
    - Immediate components: Built in __init__ without robot
    - Deferred components: Built in load() with robot info

    Usage:
        builder = GelloPolicyBuilder(config)
        immediate = builder.build_immediate()
        # ... later in load() ...
        components = builder.build_complete(immediate, robot, base_policy)
    """

    def __init__(self, config: DictConfig):
        """
        Initialize builder with policy config.

        Args:
            config: Policy configuration from Hydra
        """
        self._config = config
        self._robot_cfg = config.get("robot", None) or config.get("robot_type", None)

    def build_immediate(self) -> ImmediateComponents:
        """
        Build components that don't require robot instance.

        Called during GelloPolicy.__init__().

        Returns:
            ImmediateComponents container
        """
        # Build input handler
        require_joycon = self._config.get("require_joycon", True)
        input_handler = InputHandler(joycon_required=require_joycon)

        # Build Gello controller with auto-selection
        gello_config_name = self._resolve_gello_config()
        gello_task_overlay = self._config.get("gello_task_overlay", None)
        gello_controller = A1GelloControllerFactory.get_gello_controller_instance(
            gello_config_name=gello_config_name,
            task_overlay=gello_task_overlay,
        )

        # Determine DOF from Gello controller
        gello_total_dof = gello_controller.dof
        gello_num_arms = 2 if gello_total_dof > 7 else 1
        gello_arm_dof = gello_total_dof // gello_num_arms

        logger.info(
            f"Gello total DOF: {gello_total_dof}, per-arm DOF: {gello_arm_dof}, "
            f"num arms: {gello_num_arms}"
        )

        # Parse initial direction
        initial_direction = GelloControlDirection.from_string(
            self._config.control_direction
        )

        return ImmediateComponents(
            gello_controller=gello_controller,
            input_handler=input_handler,
            hooks=TeleopHookRegistry(),
            runtime_state=TeleopRuntimeState(),
            data_state=DataCollectionState(),
            gello_total_dof=gello_total_dof,
            gello_num_arms=gello_num_arms,
            gello_arm_dof=gello_arm_dof,
            initial_direction=initial_direction,
            base_policy_type=self._config.base_policy,
            convergence_tolerance=self._config.get('joint_convergence_tolerance', 0.1),
            max_convergence_wait_time=self._config.get('max_convergence_wait_time', 10.0),
        )

    def build_deferred(self, immediate: ImmediateComponents, robot) -> DeferredComponents:
        """
        Build components that require robot instance.

        Called during GelloPolicy.load().

        Args:
            immediate: Previously built immediate components
            robot: Robot instance from environment

        Returns:
            DeferredComponents container
        """
        # Get robot strategy
        robot_strategy = get_robot_strategy(robot.model_name)

        # Validate Gello config matches robot
        current_config_name = A1GelloControllerFactory._current_config_name
        robot_strategy.validate_gello_config(current_config_name)

        # Build action layout using robot strategy
        logger.info(f"Building action layout for robot: {robot_strategy.name}")
        layout_config = robot_strategy.action_layout
        action_keys, action_key_dims, action_slices, action_dim = build_action_layout(
            layout_config, immediate.gello_arm_dof
        )

        # Get arm keys and gripper slices
        arm_keys = [k for k in action_keys if "arm" in k]
        if not arm_keys:
            arm_keys = ["arm"]
        gripper_slices = [
            action_slices[k] for k in action_keys
            if "gripper" in k and k in action_slices
        ]

        # Build arm mappings
        arm_mappings = build_arm_mappings(
            arm_keys,
            action_slices,
            immediate.gello_arm_dof,
            immediate.gello_num_arms,
        )

        logger.info(f"  Arm mappings: {[(m.name, m.robot_slice) for m in arm_mappings]}")
        logger.info(f"  Action dim: {action_dim}, Keys: {action_keys}")

        # Build unified arm controller
        arm_controllers = [ArmController(mapping) for mapping in arm_mappings]
        unified_arm_controller = UnifiedArmController(arm_controllers)

        # Get robot's arm controller reference
        arm_controller_ref = robot_strategy.get_arm_controller(robot)

        return DeferredComponents(
            robot_strategy=robot_strategy,
            arm_controller=unified_arm_controller,
            arm_controller_ref=arm_controller_ref,
            action_dim=action_dim,
            action_slices=action_slices,
            gripper_slices=gripper_slices,
            arm_mappings=arm_mappings,
        )

    def build_complete(
        self,
        immediate: ImmediateComponents,
        robot,
        base_policy,
    ) -> GelloPolicyComponents:
        """
        Build complete component set with all dependencies.

        Args:
            immediate: Previously built immediate components
            robot: Robot instance from environment
            base_policy: Loaded base policy instance

        Returns:
            Complete GelloPolicyComponents container
        """
        # Build deferred components
        deferred = self.build_deferred(immediate, robot)

        # Connect InputHandler now that we know robot type
        # This uses the shared JoyCon singleton from joycon_utils
        immediate.input_handler.connect(deferred.robot_strategy.name)

        # Build mode manager
        gating_config = parse_gating_config(self._config)
        use_pos = gating_config.use_position_control
        mode_manager = ControlModeManager(immediate.gello_controller)
        mode_manager.register_mode(GelloToEnvMode())
        mode_manager.register_mode(EnvToGelloMode(use_position_control=use_pos))
        mode_manager.register_mode(AdmittanceMode(use_position_control=use_pos))

        # Build pipeline
        pipeline = TeleopPipeline(
            hook_registry=immediate.hooks,
            mode_manager=mode_manager,
            arm_controller=deferred.arm_controller,
            gello_controller=immediate.gello_controller,
            verbose=self._config.get("verbose", False),
        )

        # Build gate if enabled
        gate = self._build_gate(immediate.gello_controller)
        if gate is not None:
            pipeline.set_gate(gate)

        # Configure gain switching for dual-stiffness admittance mode
        if gate is not None:
            normal_gains, takeover_gains = self._parse_stiffness_configs(immediate.gello_controller)
            gate.set_stiffness_configs(normal_gains, takeover_gains)

        # Apply stiffness override if configured
        self._apply_stiffness_override(immediate.gello_controller)

        # Build movement helper
        movement_helper = GelloMovementHelper(
            gello_controller=immediate.gello_controller,
            convergence_tolerance=immediate.convergence_tolerance,
            max_wait_time=immediate.max_convergence_wait_time,
        )
        pipeline.set_movement_helper(movement_helper)

        # Build locomotion handler for base/torso/gripper control
        require_joycon = self._config.get("require_joycon", True)
        locomotion_handler = get_locomotion_handler(
            robot_type=deferred.robot_strategy.name,
            required=require_joycon,
        )
        logger.info(
            f"Locomotion handler: {locomotion_handler.name} "
            f"(connected={locomotion_handler.connected})"
        )

        return GelloPolicyComponents(
            gello_controller=immediate.gello_controller,
            mode_manager=mode_manager,
            pipeline=pipeline,
            arm_controller=deferred.arm_controller,
            input_handler=immediate.input_handler,
            gate=gate,
            robot_strategy=deferred.robot_strategy,
            movement_helper=movement_helper,
            runtime_state=immediate.runtime_state,
            data_state=immediate.data_state,
            base_policy=base_policy,
            base_policy_control_type=base_policy.config.control_type,
            action_dim=deferred.action_dim,
            action_slices=deferred.action_slices,
            gripper_slices=deferred.gripper_slices,
            locomotion_handler=locomotion_handler,
        )

    def _resolve_gello_config(self) -> str:
        """
        Resolve Gello config name from robot config or explicit setting.

        Returns:
            Gello config name string
        """
        gello_config_name = self._config.get("gello_config", None)

        # Try to auto-select from robot config
        if gello_config_name is None and self._robot_cfg is not None:
            strategy = get_robot_strategy_from_config(self._robot_cfg)
            if strategy is not None:
                gello_config_name = strategy.expected_gello_config
                logger.info(
                    f"Auto-selected gello config '{gello_config_name}' "
                    f"for robot type '{strategy.name}'"
                )

        # Default fallback
        if gello_config_name is None:
            gello_config_name = "gello_a1"
            logger.warning(
                f"No gello_config specified and robot type unknown, "
                f"defaulting to '{gello_config_name}'"
            )

        return gello_config_name

    def _build_gate(self, gello_controller) -> Optional[AdmittanceGate]:
        """
        Build admittance gate if enabled in config.

        Args:
            gello_controller: Gello controller instance

        Returns:
            AdmittanceGate or None if disabled
        """
        gating_config = parse_gating_config(self._config)

        if gating_config.enabled:
            logger.info("Admittance gating enabled")
            return AdmittanceGate(gating_config, gello_controller)
        else:
            logger.info("Admittance gating disabled (robot always follows GELLO)")
            return None

    def _parse_stiffness_configs(self, gello_controller):
        """
        Parse normal and takeover stiffness configs.

        Resolution order (first non-null wins):
          1. Policy config (gello.yaml) -- explicit per-run override
          2. Gello hardware config (gello_r1pro.yaml / gello_a1.yaml)
          3. Current goal_tracker gains (hardware defaults already loaded)

        Returns:
            Tuple of (normal_gains_dict, takeover_gains_dict) where each is
            {"kp": array, "ki": array, "kd": array} or None.
        """
        def _parse_gains(section):
            """Parse a stiffness config section into a gains dict, or None if all null."""
            if section is None:
                return None
            kp = section.get("kp", None)
            ki = section.get("ki", None)
            kd = section.get("kd", None)
            kff = section.get("kff", None)
            if kp is None and ki is None and kd is None and kff is None:
                return None
            gains = {}
            if kp is not None:
                gains["kp"] = np.array(kp, dtype=np.float32)
            if ki is not None:
                gains["ki"] = np.array(ki, dtype=np.float32)
            if kd is not None:
                gains["kd"] = np.array(kd, dtype=np.float32)
            if kff is not None:
                gains["kff"] = np.array(kff, dtype=np.float32)
            return gains

        # Try policy config first (explicit per-run override)
        normal_gains = _parse_gains(self._config.get("admittance_stiffness", None))
        takeover_gains = _parse_gains(self._config.get("takeover_stiffness", None))

        # Fall back to gello hardware config if policy config didn't specify
        if normal_gains is None or takeover_gains is None:
            gello_hw_config = self._load_gello_hardware_config()
            if gello_hw_config is not None:
                if normal_gains is None:
                    hw_admittance = gello_hw_config.get("admittance_stiffness", None)
                    normal_gains = _parse_gains(hw_admittance)
                    if normal_gains is not None:
                        logger.info("Loaded admittance_stiffness from gello hardware config")
                if takeover_gains is None:
                    hw_takeover = gello_hw_config.get("takeover_stiffness", None)
                    takeover_gains = _parse_gains(hw_takeover)
                    if takeover_gains is not None:
                        logger.info("Loaded takeover_stiffness from gello hardware config")

        # If takeover_stiffness is set but admittance_stiffness is not,
        # capture current hardware-default gains as the "normal" gains
        # so they can be restored after takeover ends.
        if takeover_gains is not None and normal_gains is None:
            if hasattr(gello_controller, "_goal_tracker"):
                tracker_cfg = gello_controller._goal_tracker.config
                normal_gains = {
                    "kp": tracker_cfg.kp.copy(),
                    "ki": tracker_cfg.ki.copy(),
                    "kd": tracker_cfg.kd.copy(),
                }
                if tracker_cfg.kff is not None:
                    normal_gains["kff"] = tracker_cfg.kff.copy()
                logger.info(
                    "Captured hardware-default gains as normal gains for stiffness switching"
                )

        return normal_gains, takeover_gains

    def _load_gello_hardware_config(self):
        """Load the gello hardware config using the factory's stored config name."""
        config_name = A1GelloControllerFactory._current_config_name
        if config_name is None:
            logger.debug("No hardware config selected for stiffness overrides")
            return None
        try:
            from iiil.gello.factory import load_gello_config
            return load_gello_config(config_name)
        except Exception as e:
            logger.warning(f"Failed to load gello hardware config '{config_name}': {e}")
            return None

    def _apply_stiffness_override(self, gello_controller) -> None:
        """
        Apply admittance stiffness override if specified.

        Resolution order: policy config > gello hardware config > defaults.

        Args:
            gello_controller: Gello controller instance
        """
        # Try policy config first
        stiffness_config = self._config.get("admittance_stiffness", None)
        source = "policy config"

        # Check if policy config has any non-null values
        has_policy_override = (
            stiffness_config is not None
            and any(stiffness_config.get(k) is not None for k in ("kp", "ki", "kd", "kff"))
        )

        # Fall back to gello hardware config
        if not has_policy_override:
            gello_hw_config = self._load_gello_hardware_config()
            if gello_hw_config is not None:
                stiffness_config = gello_hw_config.get("admittance_stiffness", None)
                source = "gello hardware config"
                has_policy_override = (
                    stiffness_config is not None
                    and any(stiffness_config.get(k) is not None for k in ("kp", "ki", "kd", "kff"))
                )

        if has_policy_override and hasattr(gello_controller, "set_tracking_gains"):
            logger.info(f"Overriding GELLO stiffness from {source}")
            kp = stiffness_config.get("kp", None)
            ki = stiffness_config.get("ki", None)
            kd = stiffness_config.get("kd", None)
            kff = stiffness_config.get("kff", None)
            gello_controller.set_tracking_gains(
                kp=np.array(kp, dtype=np.float32) if kp is not None else None,
                ki=np.array(ki, dtype=np.float32) if ki is not None else None,
                kd=np.array(kd, dtype=np.float32) if kd is not None else None,
                kff=np.array(kff, dtype=np.float32) if kff is not None else None,
                reset_integral=True,
            )
            if kp is not None:
                logger.info(f"  Set Kp: {kp}")
            if ki is not None:
                logger.info(f"  Set Ki: {ki}")
            if kd is not None:
                logger.info(f"  Set Kd: {kd}")


def create_base_policy(base_policy_type: str):
    """
    Create base policy instance.

    Separate from builder to handle circular import issues.

    Args:
        base_policy_type: Policy type name

    Returns:
        BasePolicy instance (not loaded)
    """
    # Import here due to circular dependency
    from iiil.policies import policy_lib
    from iiil.utils.policy_utils import get_policy_config

    base_policy_config = get_policy_config(base_policy_type)

    if base_policy_type not in policy_lib:
        raise ValueError(f"Base policy {base_policy_type} not found in policy library.")

    return policy_lib[base_policy_config.type](base_policy_config)


__all__ = [
    "ImmediateComponents",
    "DeferredComponents",
    "GelloPolicyComponents",
    "GelloPolicyBuilder",
    "create_base_policy",
]

import logging
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
import torch
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
from iiil.policies.policy_base import BasePolicy, PolicyMode
from iiil.utils.policy_utils import get_policy_config, get_interrupt_detector_config
from iiil.detectors.threshold_interrupt_detector import ThresholdInterruptDetector
from iiil.detectors.sklearn_interrupt_detector import SklearnRegressionInterruptDetector
from iiil.detectors.voltage_based_detector import VoltageBasedDetector
from iiil.detectors.policy_comparison_detector import PolicyComparisonDetector
from iiil.detectors.keyboard_interrupt_detector import KeyboardInterruptDetector
from iiil.detectors.threshold_release_detector import ThresholdReleaseDetector
from iiil.detectors.mahalanobis_release_detector import MahalanobisReleaseDetector
from iiil.detectors.mahalanobis_detector import MahalanobisDetector
from iiil.detectors.serial_voltage_reader import SerialVoltageReader
from iiil.utils.misc_utils import fk
from iiil.wrappers.robots import SimA1
from omegaconf import DictConfig, OmegaConf
from omnigibson.utils.gym_utils import recursively_generate_flat_dict
from omnigibson.utils.ui_utils import create_module_logger
from typing import Any, Dict, Optional, Tuple

logger = create_module_logger("HybridPolicy")
logger.setLevel(logging.INFO) 


class SharedAutonomyPolicy(BasePolicy):
    """
    A hybrid policy that runs shared autonomy with two policies (base and oracle) and switches between them
    based on interrupt and release detection.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        # cmg for fk computation
        self.cmg = None

        # Create the base and oracle policies
        self._base_policy = self._create_policy(config.base_policy)
        self._oracle_policy = self._create_policy(config.oracle_policy)

        # Set initial modes
        self._base_policy.set_mode(PolicyMode.LEADING)
        self._oracle_policy.set_mode(PolicyMode.FOLLOWING)

        # Initialize interrupt and release detectors
        self._interrupt_detector = None
        self._release_detector = None
        self._voltage_reader = None

        # Load detectors if specified
        self._setup_detectors(config)

        # Optional passive observer (logs metrics without controlling switching)
        self._passive_observer = None
        self._setup_passive_observer(config)

        # State tracking
        self._switch_check_interval = config.switch_check_interval
        self._switch_min_interval = config.switch_min_interval
        self._current_leading_policy = self._base_policy
        self._current_following_policy = self._oracle_policy
        self._is_oracle_active = False
        self._step_from_last_switch = 0
        self._prev_robot_obs = None
        self._current_state = None
        self._detector_component_weights = config.get("detector_component_weights", {})

    def _create_policy(self, policy_config: DictConfig) -> BasePolicy:
        """
        Loads and creates a policy by merging the base config with the provided config

        :param policy_config: The configuration for the policy to load
        """
        # Import here due to circular dependency outside
        from iiil.policies import policy_lib

        # Load the base config for this policy type
        base_config = get_policy_config(policy_config.type)
        
        # Merge the base config with the provided config
        # The provided config takes precedence over the base config
        merged_config = OmegaConf.merge(base_config, policy_config)
        
        if merged_config.type not in policy_lib:
            raise ValueError(f"Policy {merged_config.type} not found in policy library.")
        return policy_lib[merged_config.type](merged_config)

    def _setup_detectors(self, config: DictConfig) -> None:
        """
        Setup interrupt and release detectors based on configuration
        """
        interrupt_detector_type = config.interrupt_detector
        release_detector_type = config.release_detector
        interrupt_detector_override = config.get("interrupt_detector_config", None)
        release_detector_override = config.get("release_detector_config", None)
        shared_detector_types = ("policy_comparison_detector", "mahalanobis_detector", "keyboard_interrupt_detector")

        # Check if voltage reader is needed
        needs_voltage_reader = (
            interrupt_detector_type == "voltage_based_detector" or
            release_detector_type == "voltage_based_detector"
        )
        
        if needs_voltage_reader:
            # Use the unified voltage_based_detector config for voltage reader setup
            voltage_reader_config = get_interrupt_detector_config("voltage_based_detector")
            
            self._voltage_reader = SerialVoltageReader(
                serial_port=voltage_reader_config.serial_port,
                baudrate=voltage_reader_config.baudrate,
                timeout=voltage_reader_config.timeout
            )

        # Load interrupt detector
        if interrupt_detector_type is not None:
            interrupt_detector_config = get_interrupt_detector_config(interrupt_detector_type)
            if interrupt_detector_override is not None:
                interrupt_detector_config = OmegaConf.merge(
                    interrupt_detector_config,
                    interrupt_detector_override,
                )
            if (
                interrupt_detector_type == "policy_comparison_detector"
                and release_detector_override is not None
            ):
                interrupt_detector_config = OmegaConf.merge(
                    interrupt_detector_config,
                    release_detector_override,
                )
            if interrupt_detector_type == "threshold_interrupt_detector":
                self._interrupt_detector = ThresholdInterruptDetector(config=interrupt_detector_config)
            elif interrupt_detector_type == "sklearn_interrupt_detector":
                self._interrupt_detector = SklearnRegressionInterruptDetector(config=interrupt_detector_config)
            elif interrupt_detector_type == "voltage_based_detector":
                self._interrupt_detector = VoltageBasedDetector(
                    config=interrupt_detector_config,
                    voltage_reader=self._voltage_reader,
                    mode="interrupt"
                )
            elif interrupt_detector_type == "policy_comparison_detector":
                self._interrupt_detector = PolicyComparisonDetector(config=interrupt_detector_config)
            elif interrupt_detector_type == "mahalanobis_detector":
                self._interrupt_detector = MahalanobisDetector(config=interrupt_detector_config)
            elif interrupt_detector_type == "keyboard_interrupt_detector":
                self._interrupt_detector = KeyboardInterruptDetector(config=interrupt_detector_config)
            else:
                raise ValueError(f"Unknown interrupt detector type: {interrupt_detector_type}")

        # Load release detector
        if release_detector_type is not None and release_detector_type not in shared_detector_types:
            release_detector_config = get_interrupt_detector_config(release_detector_type)
            if release_detector_override is not None:
                release_detector_config = OmegaConf.merge(
                    release_detector_config,
                    release_detector_override,
                )
            if release_detector_type == "threshold_release_detector":
                self._release_detector = ThresholdReleaseDetector(config=release_detector_config)
            elif release_detector_type == "voltage_based_detector":
                self._release_detector = VoltageBasedDetector(
                    config=release_detector_config,
                    voltage_reader=self._voltage_reader,
                    mode="release"
                )
            elif release_detector_type == "mahalanobis_release_detector":
                self._release_detector = MahalanobisReleaseDetector(config=release_detector_config)
            else:
                raise ValueError(f"Unknown release detector type: {release_detector_type}")
        elif interrupt_detector_type in shared_detector_types and (
            release_detector_type is None or release_detector_type == interrupt_detector_type
        ):
            self._release_detector = self._interrupt_detector
        else:
            raise ValueError(
                "Release detector type must be specified if not using a shared interrupt/release detector."
            )

    def _setup_passive_observer(self, config: DictConfig) -> None:
        """Setup a passive observer that logs metrics each step without controlling switching."""
        observer_type = config.get("passive_observer", None)
        if observer_type is None:
            return
        observer_config_override = config.get("passive_observer_config", None)
        observer_config = get_interrupt_detector_config(observer_type)
        if observer_config_override is not None:
            observer_config = OmegaConf.merge(observer_config, observer_config_override)
        if observer_type == "mahalanobis_detector":
            self._passive_observer = MahalanobisDetector(config=observer_config)
        elif observer_type == "mahalanobis_release_detector":
            self._passive_observer = MahalanobisReleaseDetector(config=observer_config)
        else:
            raise ValueError(f"Unknown passive observer type: {observer_type}")
        logger.info(f"Passive observer enabled: {observer_type}")

    def predict(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Predict action from the currently active policy and handle switching logic

        Args:
            obs (dict[str, torch.Tensor]): observation dictionary from the environment
            **kwargs: additional arguments

        Returns:
            torch.Tensor: the predicted action from the active policy
        """
        # Get actions from both policies
        base_action = self._base_policy.predict(obs=obs, **kwargs)
        oracle_action = self._oracle_policy.predict(obs=obs, **kwargs)
        output_action = oracle_action if self._is_oracle_active else base_action
        self._current_state = {
            "base_action": base_action,
            "oracle_action": oracle_action,
            "is_oracle_active": torch.tensor([self._is_oracle_active]),
        }
        self._step_from_last_switch += 1
        is_keyboard_switch = isinstance(self._interrupt_detector, KeyboardInterruptDetector) or isinstance(
            self._release_detector, KeyboardInterruptDetector
        )
        if not is_keyboard_switch and self._prev_robot_obs is None:
            self._prev_robot_obs = self._prepare_obs_for_detector(obs, base_action, oracle_action)[0]
        
        # Handle interrupt and release detection (switch between base to oracle)
        if is_keyboard_switch or self._step_from_last_switch % self._switch_check_interval == 0:
            # Prepare observations for detectors
            if is_keyboard_switch:
                detector_obs = {}
            else:
                robot_obs, policy_obs = self._prepare_obs_for_detector(obs, base_action, oracle_action)
                detector_obs = {"robot": self._prev_robot_obs, "policy": policy_obs}
                self._prev_robot_obs = robot_obs
            if not self._is_oracle_active and self._interrupt_detector is not None:
                is_interrupted = self._interrupt_detector.is_interrupted(detector_obs)
                
                if (is_keyboard_switch or self._step_from_last_switch >= self._switch_min_interval) and is_interrupted:
                    self._switch_to_oracle()
                    logger.info("Interrupted! Switching to oracle policy.")

            elif self._is_oracle_active and self._release_detector is not None:
                is_released = self._release_detector.is_released(detector_obs)
                if (is_keyboard_switch or self._step_from_last_switch >= self._switch_min_interval) and is_released:
                    self._switch_to_base()
                    logger.info("Released! Switching to base policy.")

        # Merge detector state into policy state for HDF5 logging
        if self._interrupt_detector is not None and hasattr(self._interrupt_detector, 'get_current_state'):
            detector_state = self._interrupt_detector.get_current_state()
            if detector_state is not None:
                self._current_state.update(detector_state)
        if self._release_detector is not None and self._release_detector is not self._interrupt_detector:
            release_state = self._release_detector.get_current_state()
            if release_state is not None:
                self._current_state.update(release_state)

        # Feed passive observer and log its state
        if self._passive_observer is not None:
            embedding = getattr(self._base_policy, 'last_obs_embedding', None)
            if embedding is not None:
                passive_detector_obs = {"policy": {"obs_embedding": embedding}}
                self._passive_observer._update_distance(passive_detector_obs)
            observer_state = self._passive_observer.get_current_state()
            if observer_state is not None:
                self._current_state.update(observer_state)

        # Return action from the currently active policy
        return output_action

    def _switch_to_oracle(self) -> None:
        """
        Switch control to the oracle policy
        """
        self._is_oracle_active = True
        self._step_from_last_switch = 0
        self._current_leading_policy = self._oracle_policy
        self._current_following_policy = self._base_policy
        
        # Update policy modes
        self._oracle_policy.set_mode(PolicyMode.LEADING)
        self._base_policy.set_mode(PolicyMode.FOLLOWING)
        self._oracle_policy.reset()
        self._base_policy.reset()
        
        # Make action buffer length unlimited for oracle policy
        if hasattr(self._oracle_policy, "action_buffer_len"):
            self._oracle_policy.action_buffer_len = None
        # reset detector
        if self._interrupt_detector is not None:
            self._interrupt_detector.reset()

    def _switch_to_base(self) -> None:
        """
        Switch control to the base policy
        """
        self._is_oracle_active = False
        self._step_from_last_switch = 0
        self._current_leading_policy = self._base_policy
        self._current_following_policy = self._oracle_policy
        
        # Update policy modes
        self._base_policy.set_mode(PolicyMode.LEADING)
        self._oracle_policy.set_mode(PolicyMode.FOLLOWING)
        self._oracle_policy.reset()
        self._base_policy.reset()

        # reset detector
        if self._release_detector is not None:
            self._release_detector.reset()

    def _prepare_obs_for_detector(self, obs: dict[str, torch.Tensor], 
                                 base_action: torch.Tensor, 
                                 oracle_action: torch.Tensor) -> Tuple[dict[str, Any], dict[str, Any]]:
        """
        Prepare observations for the detectors

        Args:
            obs (dict[str, torch.Tensor]): observation dictionary from the environment
            base_action (torch.Tensor): action from base policy
            oracle_action (torch.Tensor): action from oracle policy

        Returns:
            Tuple[dict[str, Any], dict[str, Any]]: prepared robot and policy observations for detectors
        """
        if self._use_whole_body_detector_vector():
            n_joints = getattr(self.robot, "n_joints", base_action.shape[0])
            joint_angles = obs["robot::proprio"][:n_joints].cpu().float()
            base_compare_action, base_components = self._build_whole_body_compare_action(base_action, joint_angles)
            oracle_compare_action, oracle_components = self._build_whole_body_compare_action(oracle_action, joint_angles)
            robot_obs = {
                "joint_angles": joint_angles,
            }
            policy_obs = {
                "base_action": base_action,
                "oracle_action": oracle_action,
                "base_compare_action": base_compare_action,
                "oracle_compare_action": oracle_compare_action,
                "base_compare_components": base_components,
                "oracle_compare_components": oracle_components,
            }

            embedding = getattr(self._base_policy, 'last_obs_embedding', None)
            if embedding is not None:
                policy_obs["obs_embedding"] = embedding

            return robot_obs, policy_obs

        # Single-arm fallback used by A1 detectors.
        robot_obs = {
            "joint_angles": obs["robot::proprio"][:6].cpu(),
            "eef_pose": fk(self.cmg, obs["robot::proprio"][:6]),
            "gripper_states": torch.where(obs["robot::proprio"][6:8].cpu().mean(dim=0, keepdim=True) >= 0.024, 1.0, -1.0),
        }
        
        policy_obs = {
            "base_action": base_action,  # Add base action for policy comparison detector
            "base_target_eef_pose": fk(self.cmg, base_action[:6]),
            "oracle_action": oracle_action,  # Add oracle action for policy comparison detector
            "oracle_target_eef_pose": fk(self.cmg, oracle_action[:6]),
        }

        embedding = getattr(self._base_policy, 'last_obs_embedding', None)
        if embedding is not None:
            policy_obs["obs_embedding"] = embedding

        return robot_obs, policy_obs

    def _use_whole_body_detector_vector(self) -> bool:
        robot_model = str(getattr(self.robot, "model", "")).lower()
        robot_name = str(getattr(self.robot, "model_name", "")).lower()
        return robot_model == "r1pro" or robot_name == "r1pro"

    def _build_whole_body_compare_action(
        self,
        action: torch.Tensor,
        current_joint_angles: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Build a robot-controller ordered comparison vector for whole-body robots.

        R1Pro actions include base, torso, both arms, and both grippers. Base
        and grippers are compared as raw commands; torso commands are compared
        as joint deltas; arm commands are converted to EEF deltas from the
        current pose.
        """
        component_names = [
            "base",
            "trunk",
            "torso",
            "arm_left",
            "left_arm",
            "gripper_left",
            "left_gripper",
            "arm_right",
            "right_arm",
            "gripper_right",
            "right_gripper",
        ]
        seen_indices = set()
        components = {}
        vector_parts = []
        arm_eef_deltas = None

        for component_name in component_names:
            if not hasattr(self.robot, "controller_action_idx"):
                continue
            if component_name not in self.robot.controller_action_idx:
                continue

            idx = self.robot.controller_action_idx[component_name]
            if isinstance(idx, slice):
                index_key = tuple(range(idx.start, idx.stop, idx.step or 1))
            else:
                index_key = tuple(torch.as_tensor(idx).flatten().cpu().tolist())
            if index_key in seen_indices:
                continue
            seen_indices.add(index_key)

            canonical_name = self._canonical_detector_component_name(component_name)
            weight = float(self._get_detector_component_weight(component_name))
            if canonical_name in ("left_arm", "right_arm") and current_joint_angles is not None:
                if arm_eef_deltas is None:
                    arm_eef_deltas = self._compute_r1pro_arm_eef_deltas(action, current_joint_angles)
                flat_component = arm_eef_deltas.get(canonical_name)
                if flat_component is None:
                    flat_component = action[idx].detach().cpu().float().flatten()
            elif canonical_name == "torso" and current_joint_angles is not None:
                flat_component = (
                    action[idx].detach().cpu().float()
                    - current_joint_angles[idx].detach().cpu().float()
                ).flatten()
            else:
                flat_component = action[idx].detach().cpu().float().flatten()

            weighted_component = flat_component * weight
            components[component_name] = weighted_component
            vector_parts.append(weighted_component)

        if not vector_parts:
            return action.detach().cpu().float().flatten(), {"action": action.detach().cpu().float().flatten()}

        return torch.cat(vector_parts), components

    def _compute_r1pro_arm_eef_deltas(
        self,
        action: torch.Tensor,
        current_joint_angles: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.cmg is None or CuRoboEmbodimentSelection.DEFAULT not in self.cmg.mg:
            return {}

        target_joint_angles = current_joint_angles.detach().cpu().float().clone()
        action_cpu = action.detach().cpu().float()
        for component_name in ("arm_left", "left_arm", "arm_right", "right_arm"):
            if component_name not in self.robot.controller_action_idx:
                continue
            idx = self.robot.controller_action_idx[component_name]
            target_joint_angles[idx] = action_cpu[idx]

        current_state = self._compute_curobo_kinematics(current_joint_angles)
        target_state = self._compute_curobo_kinematics(target_joint_angles)
        if current_state is None or target_state is None:
            return {}

        deltas = {}
        for arm_name, component_name in (("left", "left_arm"), ("right", "right_arm")):
            link_name = self.robot.eef_link_names.get(arm_name)
            if link_name not in current_state.link_poses or link_name not in target_state.link_poses:
                continue

            current_pose = current_state.link_poses[link_name]
            target_pose = target_state.link_poses[link_name]
            current_pos = current_pose.position[0].detach().cpu().float()
            target_pos = target_pose.position[0].detach().cpu().float()
            current_quat = current_pose.quaternion[0, [1, 2, 3, 0]].detach().cpu().float()
            target_quat = target_pose.quaternion[0, [1, 2, 3, 0]].detach().cpu().float()
            deltas[component_name] = torch.cat([
                target_pos - current_pos,
                T.quat2euler(T.quat_distance(target_quat, current_quat)).detach().cpu().float(),
            ])

        return deltas

    def _compute_curobo_kinematics(self, joint_angles: torch.Tensor):
        emb_sel = CuRoboEmbodimentSelection.DEFAULT
        mg = self.cmg.mg[emb_sel]
        joint_state = lazy.curobo.types.state.JointState(
            position=self.cmg._tensor_args.to_device(joint_angles.detach().cpu().float()).unsqueeze(0),
            joint_names=self.cmg.robot_joint_names,
        )
        ordered_joint_state = joint_state.get_ordered_joint_state(mg.kinematics.joint_names)
        return mg.kinematics.compute_kinematics(ordered_joint_state)

    def _get_detector_component_weight(self, component_name: str) -> float:
        component_name = self._canonical_detector_component_name(component_name)
        if component_name in self._detector_component_weights:
            return self._detector_component_weights[component_name]
        return 1.0

    def _canonical_detector_component_name(self, component_name: str) -> str:
        aliases = {
            "trunk": "torso",
            "arm_left": "left_arm",
            "arm_right": "right_arm",
            "gripper_left": "left_gripper",
            "gripper_right": "right_gripper",
        }
        return aliases.get(component_name, component_name)

    def load(self, **kwargs) -> None:
        """
        Load both policies
        """
        self._base_policy.load(**kwargs)
        self._oracle_policy.load(**kwargs)
        self.robot = kwargs["robot"]
        if hasattr(self._oracle_policy, 'cmg'):
            self.cmg = self._oracle_policy.cmg
        elif hasattr(self._base_policy, 'cmg'):
            self.cmg = self._base_policy.cmg
        else:
            from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
            # Create CuRobo instance
            # A1 only has default embodiment, so set to True here
            use_default_embodiment_only = True if isinstance(self.robot, SimA1) else False
            self.cmg = CuRoboMotionGenerator(
                robot=kwargs["robot"],
                batch_size=1,
                use_cuda_graph=True,
                use_default_embodiment_only=use_default_embodiment_only,
            )

    def reset(self) -> None:
        """
        Reset both policies and detectors
        """
        self._base_policy.reset()
        self._oracle_policy.reset()
        
        if self._release_detector is not None:
            self._release_detector.reset()
        if self._interrupt_detector is not None:
            self._interrupt_detector.reset()
            
        # Reset to base policy being active
        self._is_oracle_active = False
        self._step_from_last_switch = 0
        self._current_leading_policy = self._base_policy
        self._current_following_policy = self._oracle_policy
        self._prev_robot_obs = None
        n_metrics = 2
        if self._interrupt_detector is not None and hasattr(self._interrupt_detector, '_distance_metrics'):
            n_metrics = len(self._interrupt_detector._distance_metrics)
        self._current_state = {
            "base_action": self.robot.get_no_op_action(),
            "oracle_action": self.robot.get_no_op_action(),
            "is_oracle_active": torch.tensor([self._is_oracle_active]),
            "detector_interrupt_scores": torch.zeros(n_metrics, dtype=torch.float32),
            "detector_release_scores": torch.zeros(n_metrics, dtype=torch.float32),
        }
        if isinstance(self._release_detector, (MahalanobisReleaseDetector, MahalanobisDetector)):
            self._current_state["mahalanobis_distance"] = torch.zeros(1, dtype=torch.float32)
        if self._passive_observer is not None:
            self._passive_observer.reset()
            self._current_state["mahalanobis_distance"] = torch.zeros(1, dtype=torch.float32)

    def close(self) -> None:
        """
        Close both policies and detectors
        """
        self._base_policy.close()
        self._oracle_policy.close()
        
        if self._release_detector is not None and hasattr(self._release_detector, '_logger') and self._release_detector._logger is not None:
            self._release_detector._logger.close()
        if self._interrupt_detector is not None and hasattr(self._interrupt_detector, '_logger') and self._interrupt_detector._logger is not None:
            self._interrupt_detector._logger.close()

    def set_mode(self, policy_mode: PolicyMode) -> None:
        """
        Set the mode for both policies
        
        Args:
            policy_mode (PolicyMode): The mode to set for both policies
        """
        raise NotImplementedError("SharedAutonomyPolicy does not support setting modes")

    def get_current_state(self) -> Optional[Dict[str, torch.Tensor]]:
        return recursively_generate_flat_dict(self._current_state)

    def check_block_ended(self) -> bool:
        return self._base_policy.check_block_ended() or self._oracle_policy.check_block_ended()

    def check_episode_ended(self, success: bool=True) -> bool:
        if self._keyboard_discard_requested():
            logger.info("Discard requested from keyboard. Ending episode as failure.")
            return True
        return self._base_policy.check_episode_ended(success=success) or self._oracle_policy.check_episode_ended(success=success)

    def _keyboard_discard_requested(self) -> bool:
        for detector in (self._interrupt_detector, self._release_detector):
            if detector is not None and hasattr(detector, "consume_discard_requested"):
                if detector.consume_discard_requested():
                    return True
        return False


class NoisyPolicy(BasePolicy):
    """
    A hybrid policy that runs two policies (base and noise) and deterministically switches between them
    based on a predefined time period.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        # Create the base and noise policies
        self._base_policy = self._create_policy(config.base_policy)

        # Initialize noise parameters
        self._noise_scale = config.noise_scale
        self._n_noise_step = config.n_noise_step
        self._noise_every_n_step = config.noise_every_n_step
        self._policy_counter = 0
        self._noise_counter = 0
        self._is_noise_active = False
        self._noise = None

    def _create_policy(self, policy_config: DictConfig) -> BasePolicy:
        """
        Loads and creates a policy by merging the base config with the provided config

        :param policy_config: The configuration for the policy to load
        """
        # Import here due to circular dependency outside
        from iiil.policies import policy_lib

        # Load the base config for this policy type
        base_config = get_policy_config(policy_config.type)
        
        # Merge the base config with the provided config
        # The provided config takes precedence over the base config
        merged_config = OmegaConf.merge(base_config, policy_config)
        
        if merged_config.type not in policy_lib:
            raise ValueError(f"Policy {merged_config.type} not found in policy library.")
        return policy_lib[merged_config.type](merged_config)

    def predict(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Predict action from the currently active policy and handle switching logic

        Args:
            obs (dict[str, torch.Tensor]): observation dictionary from the environment
            **kwargs: additional arguments

        Returns:
            torch.Tensor: the predicted action from the active policy
        """
        if self._is_noise_active:
            self._noise_counter += 1
            # get noise action
            output_action = self.robot.get_no_op_action() + self._noise
            if self._noise_counter >= self._n_noise_step:
                logger.info("[NoisyPolicy] Switching back to policy")
                self._is_noise_active = False
                self._noise_counter = 0
                self._policy_counter = 0
                self._base_policy.reset()
        else:
            self._policy_counter += 1
            # Get actions from both policies
            output_action = self._base_policy.predict(obs=obs, **kwargs)
            if self._policy_counter >= self._noise_every_n_step:
                logger.info("[NoisyPolicy] Adding noise!")
                self._is_noise_active = True
                self._policy_counter = 0
                self._noise_counter = 0
                self._noise = (2 * torch.rand(output_action.shape) - 1) * self._noise_scale
        # Return action from the currently active policy
        return output_action

    def load(self, **kwargs) -> None:
        """
        Load both policies
        """
        self._base_policy.load(**kwargs)
        self.robot = kwargs["robot"]

    def reset(self) -> None:
        """
        Reset both policies and noise counter
        """
        self._base_policy.reset()
        self._noise_counter = 0
        self._policy_counter = 0
        self._is_oracle_active = False

    def close(self) -> None:
        """
        Close base policy
        """
        self._base_policy.close()

    def check_block_ended(self):
        return self._base_policy.check_block_ended()

    def check_episode_ended(self, success: bool=True) -> bool:
        return self._base_policy.check_episode_ended(success=success)

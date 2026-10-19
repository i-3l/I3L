import numpy as np
import torch
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

import iiil.policies.gello_policy as gello_mod
import iiil.policies.teleop_builder as builder_mod
import iiil.policies.teleop_input as input_mod
from iiil.policies.policy_base import PolicyMode
from tests.fixtures.mock_gello_controller import MockGelloController


class _StaticFactory:
    """Simple factory that always returns the provided controller."""

    def __init__(self, controller, config_name="gello_a1"):
        self._controller = controller
        self._current_config_name = config_name

    def get_gello_controller_instance(self, gello_config_name="gello_a1", port_overrides=None):
        self._current_config_name = gello_config_name
        return self._controller


class _JoyconStub:
    """Minimal Joycon stub to bypass hardware."""

    trigger_r = False
    trigger_zr = False
    button_x = False
    connected = True

    def __init__(self, required=True):
        """Accept required argument for compatibility."""
        pass


def _make_config(control_direction: str, gello_config: str, robot_arm_dof: int):
    """Utility to assemble a minimal policy config."""
    return OmegaConf.create({
        "type": "gello",
        "control_type": "joint",
        "gello_config": gello_config,
        "base_policy": "curobo",
        "control_direction": control_direction,
        "bilateral_blend_factor": 0.5,
        "joint_convergence_tolerance": 0.05,
        "max_convergence_wait_time": 1.0,
        "robot_type": {
            "type": "A1",
            "action_key_dims": {
                "arm": robot_arm_dof,
                "gripper": 1,
            },
        },
    })


def _patch_base_policy():
    """
    Patch helper that adapts the base policy output length to the policy's inferred DOF.

    Returns:
        Context manager for patching gello_mod.create_base_policy (where it's used)
    """
    def _factory(base_policy_type):
        # Default to 6 DOF arm + 1 gripper for A1
        action_dim = 7
        return Mock(
            config=Mock(control_type="joint"),
            predict=Mock(return_value=torch.ones(action_dim)),
            load=Mock(),
            reset=Mock(),
            close=Mock(),
        )

    # Patch where the function is USED, not where it's DEFINED
    return patch.object(gello_mod, "create_base_policy", side_effect=_factory)


def _attach_stub_arm_controller(policy):
    """Attach a minimal stub arm controller so blending logic can run without hardware."""
    # Note: new GelloPolicy has _arm_controller as a property with a no-op setter
    # The actual controller mode is managed by robot_strategy
    pass


def _create_mock_robot(model_name: str = "A1", action_dim: int = 7):
    """Create a mock robot for testing."""
    mock_arm_controller = Mock()
    mock_arm_controller.controller_mode = "joint"

    robot = Mock()
    robot.model_name = model_name
    robot.controllers = {"arm_0": mock_arm_controller}
    if model_name == "R1Pro":
        robot.controllers = {
            "arm_left": mock_arm_controller,
            "arm_right": mock_arm_controller,
        }
    robot.get_no_op_action = Mock(return_value=torch.zeros(action_dim))
    return robot


class TestGelloPolicyModes(unittest.TestCase):
    def test_env_to_gello_follows_robot_for_a1(self):
        controller = MockGelloController(dof=6, simulate_bilateral=False)
        controller._position_noise = 0.0

        config = _make_config(control_direction="env_to_gello", gello_config="gello_a1", robot_arm_dof=6)

        with patch("iiil.policies.teleop_builder.A1GelloControllerFactory", new=_StaticFactory(controller)), \
             patch("iiil.policies.teleop_input.JoyconController", new=_JoyconStub), \
             patch("iiil.policies.gello_policy.CuRoboMotionGenerator"), \
             _patch_base_policy():

            policy = gello_mod.GelloPolicy(config)
            mock_robot = _create_mock_robot("A1", action_dim=7)
            policy.load(robot=mock_robot)
            _attach_stub_arm_controller(policy)

        obs = {"robot::proprio": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])}
        action = policy.predict(obs)

        # Robot leads: output should match base policy (ones)
        self.assertTrue(torch.allclose(action, torch.ones_like(action)))
        # Goal should target all controller joints
        self.assertEqual(len(controller._goal), controller.dof)
        self.assertTrue(np.allclose(controller._goal, obs["robot::proprio"][: controller.dof].numpy()))
        self.assertTrue(controller._is_following_goal)

    def test_gello_to_env_leads_and_pads_goal_for_r1pro_bilateral(self):
        controller = MockGelloController(dof=7, simulate_bilateral=True)  # 14 DOF total
        controller._position_noise = 0.0
        controller._goal_convergence_rate = 0.0
        controller.set_position(np.linspace(0.0, 1.3, controller.dof))

        config = _make_config(control_direction="gello_to_env", gello_config="gello_r1pro", robot_arm_dof=7)

        with patch("iiil.policies.teleop_builder.A1GelloControllerFactory", new=_StaticFactory(controller, "gello_r1pro")), \
             patch("iiil.policies.teleop_input.JoyconController", new=_JoyconStub), \
             patch("iiil.policies.gello_policy.CuRoboMotionGenerator"), \
             _patch_base_policy():

            policy = gello_mod.GelloPolicy(config)
            mock_robot = _create_mock_robot("R1Pro", action_dim=23)  # R1Pro has 23 DOF total
            policy.load(robot=mock_robot)
            _attach_stub_arm_controller(policy)

        robot_obs = torch.linspace(0.05, 0.75, 8)  # 7 arm joints + gripper
        obs = {"robot::proprio": robot_obs}

        action = policy.predict(obs)

        # Gello leads: action should mirror Gello joint state for the arm DOF
        expected_gello_action = torch.from_numpy(controller.q[0][:policy._num_arm_joints])
        self.assertTrue(torch.allclose(action[: policy._num_arm_joints], expected_gello_action.to(torch.float32)))
        # Gripper passes through from observation
        self.assertAlmostEqual(action[-1].item(), robot_obs[-1].item())

        # Goal should be padded to full bilateral DOF: first half from robot, second half from current positions
        self.assertEqual(len(controller._goal), controller.dof)
        self.assertTrue(np.allclose(controller._goal[: policy._num_arm_joints], robot_obs[: policy._num_arm_joints].numpy()))
        self.assertTrue(np.allclose(
            controller._goal[policy._num_arm_joints:],
            controller.q[0][policy._num_arm_joints:]
        ))
        # Since Gello is leader, goal following should be stopped after reset
        self.assertFalse(controller._is_following_goal)

    def test_switch_between_leader_and_follower(self):
        controller = MockGelloController(dof=6, simulate_bilateral=False)
        controller._position_noise = 0.0

        config = _make_config(control_direction="env_to_gello", gello_config="gello_a1", robot_arm_dof=6)

        with patch("iiil.policies.teleop_builder.A1GelloControllerFactory", new=_StaticFactory(controller)), \
             patch("iiil.policies.teleop_input.JoyconController", new=_JoyconStub), \
             patch("iiil.policies.gello_policy.CuRoboMotionGenerator"), \
             _patch_base_policy():

            policy = gello_mod.GelloPolicy(config)
            mock_robot = _create_mock_robot("A1", action_dim=7)
            policy.load(robot=mock_robot)
            _attach_stub_arm_controller(policy)

        obs = {"robot::proprio": torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0])}

        # Initial follower mode -> resume_goal_following after reset
        policy.predict(obs)
        self.assertTrue(controller._is_following_goal)

        # Switch to leader
        policy.set_mode(PolicyMode.LEADING)
        policy.predict(obs)
        self.assertFalse(controller._is_following_goal)

        # Switch back to follower
        policy.set_mode(PolicyMode.FOLLOWING)
        policy.predict(obs)
        self.assertTrue(controller._is_following_goal)

    def test_round_trip_action_flow_env_and_gello(self):
        controller = MockGelloController(dof=6, simulate_bilateral=False)
        controller._position_noise = 0.0
        controller._goal_convergence_rate = 0.0
        controller.set_position(np.linspace(0.1, 0.6, 6))

        config = _make_config(control_direction="env_to_gello", gello_config="gello_a1", robot_arm_dof=6)

        with patch("iiil.policies.teleop_builder.A1GelloControllerFactory", new=_StaticFactory(controller)), \
             patch("iiil.policies.teleop_input.JoyconController", new=_JoyconStub), \
             patch("iiil.policies.gello_policy.CuRoboMotionGenerator"), \
             _patch_base_policy():

            policy = gello_mod.GelloPolicy(config)
            mock_robot = _create_mock_robot("A1", action_dim=7)
            policy.load(robot=mock_robot)
            _attach_stub_arm_controller(policy)

        # Baseline observation (arm + gripper)
        obs = {"robot::proprio": torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2])}

        # Follower: robot leads, so output should mirror base policy (ones) and we follow the goal
        action_env_to_gello = policy.predict(obs)
        self.assertTrue(torch.all(action_env_to_gello == 1.0))
        self.assertTrue(controller._is_following_goal)

        # Switch to leader: Gello drives, so output should match device joints and we pause goal following
        policy.set_mode(PolicyMode.LEADING)
        action_gello_to_env = policy.predict(obs)
        expected_gello = torch.from_numpy(controller.q[0][: policy._num_arm_joints]).to(torch.float32)
        torch.testing.assert_close(action_gello_to_env[: policy._num_arm_joints], expected_gello)
        self.assertAlmostEqual(action_gello_to_env[-1].item(), obs["robot::proprio"][-1].item())
        self.assertFalse(controller._is_following_goal)

        # Switch back to follower: base policy action should be used again and goal following resumes
        policy.set_mode(PolicyMode.FOLLOWING)
        action_back_to_env_to_gello = policy.predict(obs)
        self.assertTrue(torch.all(action_back_to_env_to_gello == 1.0))
        self.assertTrue(controller._is_following_goal)


if __name__ == "__main__":
    unittest.main()

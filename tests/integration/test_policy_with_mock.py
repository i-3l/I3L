#!/usr/bin/env python3
"""
Integration tests for Gello policy with mocked controller.

Tests the policy end-to-end without requiring hardware,
using mock robot observations and mock Gello controller.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.fixtures import MockControllerFactory, create_mock_controller_r1pro_bilateral


class TestPolicyWithMockController(unittest.TestCase):
    """Integration tests for policy with mock controller."""

    def create_mock_config(self, gello_config="gello_r1pro", robot_arm_dof=6):
        """Create mock policy configuration."""
        config_dict = {
            'type': 'gello',
            'control_type': 'joint',
            'gello_config': gello_config,
            'base_policy': 'curobo',
            'control_direction': 'gello_to_env',
            'bilateral_blend_factor': 0.5,
            'joint_convergence_tolerance': 0.1,
            'max_convergence_wait_time': 10.0,
            'robot_type': {
                'type': 'A1',
                'action_key_dims': {
                    'arm': robot_arm_dof,
                    'gripper': 1
                }
            }
        }
        return OmegaConf.create(config_dict)

    def create_mock_observation(self, num_arm_joints=6):
        """Create mock robot observation."""
        return {
            'robot::proprio': torch.zeros(num_arm_joints + 1),  # arm + gripper
        }

    @patch('iiil.policies.teleop_policy.A1GelloControllerFactory')
    @patch('iiil.policies.teleop_policy.JoyconController')
    def test_policy_initialization_a1_robot(self, mock_joycon, mock_factory):
        """Test policy initialization with A1 robot."""
        # Setup mocks
        mock_factory.get_gello_controller_instance.return_value = create_mock_controller_r1pro_bilateral()
        mock_joycon.return_value = Mock()

        config = self.create_mock_config(gello_config="gello_r1pro", robot_arm_dof=6)

        # Mock base policy creation
        with patch.object(sys.modules['iiil.policies.teleop_policy'].GelloPolicy,
                         '_create_base_policy', return_value=Mock(config=Mock(control_type='joint'))):
            from iiil.policies.teleop_policy import GelloPolicy

            policy = GelloPolicy(config)

            # Verify DOF detection
            self.assertEqual(policy._gello_controller.dof, 14, "Gello should have 14 DOF (bilateral)")
            self.assertEqual(policy._gello_arm_dof, 7, "Gello arm should have 7 DOF")
            self.assertEqual(policy._robot_arm_dof, 6, "Robot should have 6 DOF")
            self.assertEqual(policy._num_arm_joints, 6, "Should use 6 joints (minimum)")

    @patch('iiil.policies.teleop_policy.A1GelloControllerFactory')
    @patch('iiil.policies.teleop_policy.JoyconController')
    def test_policy_initialization_r1pro_robot(self, mock_joycon, mock_factory):
        """Test policy initialization with R1Pro robot."""
        # Setup mocks
        mock_factory.get_gello_controller_instance.return_value = create_mock_controller_r1pro_bilateral()
        mock_joycon.return_value = Mock()

        config = self.create_mock_config(gello_config="gello_r1pro", robot_arm_dof=7)

        # Mock base policy creation
        with patch.object(sys.modules['iiil.policies.teleop_policy'].GelloPolicy,
                         '_create_base_policy', return_value=Mock(config=Mock(control_type='joint'))):
            from iiil.policies.teleop_policy import GelloPolicy

            policy = GelloPolicy(config)

            # Verify DOF detection
            self.assertEqual(policy._num_arm_joints, 7, "Should use all 7 joints")

    @patch('iiil.policies.teleop_policy.A1GelloControllerFactory')
    @patch('iiil.policies.teleop_policy.JoyconController')
    def test_policy_action_creation_a1(self, mock_joycon, mock_factory):
        """Test action creation with A1 robot."""
        # Setup mocks
        mock_controller = create_mock_controller_r1pro_bilateral()
        # Set realistic positions for first 6 joints
        positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mock_controller.set_position(positions)

        mock_factory.get_gello_controller_instance.return_value = mock_controller
        mock_joycon.return_value = Mock(trigger_r=False, trigger_zr=False)

        config = self.create_mock_config(gello_config="gello_r1pro", robot_arm_dof=6)

        with patch.object(sys.modules['iiil.policies.teleop_policy'].GelloPolicy,
                         '_create_base_policy', return_value=Mock(config=Mock(control_type='joint'))):
            from iiil.policies.teleop_policy import GelloPolicy

            policy = GelloPolicy(config)

            # Create mock observation
            obs = self.create_mock_observation(num_arm_joints=6)

            # Create Gello action
            gello_action = policy._create_gello_action(obs)

            # Verify action shape: 6 arm joints + 1 gripper = 7
            self.assertEqual(len(gello_action), 7)

            # Verify first 6 joints come from Gello
            gello_positions, _ = mock_controller.q
            for i in range(6):
                self.assertAlmostEqual(gello_action[i].item(), gello_positions[i], places=2)

    @patch('iiil.policies.teleop_policy.A1GelloControllerFactory')
    @patch('iiil.policies.teleop_policy.JoyconController')
    def test_policy_action_creation_r1pro(self, mock_joycon, mock_factory):
        """Test action creation with R1Pro robot."""
        # Setup mocks
        mock_controller = create_mock_controller_r1pro_bilateral()
        positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mock_controller.set_position(positions)

        mock_factory.get_gello_controller_instance.return_value = mock_controller
        mock_joycon.return_value = Mock(trigger_r=False, trigger_zr=False)

        config = self.create_mock_config(gello_config="gello_r1pro", robot_arm_dof=7)

        with patch.object(sys.modules['iiil.policies.teleop_policy'].GelloPolicy,
                         '_create_base_policy', return_value=Mock(config=Mock(control_type='joint'))):
            from iiil.policies.teleop_policy import GelloPolicy

            policy = GelloPolicy(config)

            # Create mock observation
            obs = self.create_mock_observation(num_arm_joints=7)

            # Create Gello action
            gello_action = policy._create_gello_action(obs)

            # Verify action shape: 7 arm joints + 1 gripper = 8
            self.assertEqual(len(gello_action), 8)

            # Verify all 7 joints come from Gello
            gello_positions, _ = mock_controller.q
            for i in range(7):
                self.assertAlmostEqual(gello_action[i].item(), gello_positions[i], places=2)

    @patch('iiil.policies.teleop_policy.A1GelloControllerFactory')
    @patch('iiil.policies.teleop_policy.JoyconController')
    def test_policy_gripper_control(self, mock_joycon, mock_factory):
        """Test gripper control through joycon."""
        # Setup mocks
        mock_controller = create_mock_controller_r1pro_bilateral()
        mock_factory.get_gello_controller_instance.return_value = mock_controller

        config = self.create_mock_config(gello_config="gello_r1pro", robot_arm_dof=6)

        with patch.object(sys.modules['iiil.policies.teleop_policy'].GelloPolicy,
                         '_create_base_policy', return_value=Mock(config=Mock(control_type='joint'))):
            from iiil.policies.teleop_policy import GelloPolicy

            # Test gripper close (trigger_r)
            mock_joycon.return_value = Mock(trigger_r=True, trigger_zr=False)
            policy = GelloPolicy(config)
            obs = self.create_mock_observation(num_arm_joints=6)
            action = policy._create_gello_action(obs)
            self.assertEqual(action[-1].item(), 1.0, "Gripper should close")

            # Test gripper open (trigger_zr)
            mock_joycon.return_value = Mock(trigger_r=False, trigger_zr=True)
            policy = GelloPolicy(config)
            obs = self.create_mock_observation(num_arm_joints=6)
            action = policy._create_gello_action(obs)
            self.assertEqual(action[-1].item(), -1.0, "Gripper should open")


class TestMockControllerBehavior(unittest.TestCase):
    """Test mock controller behaves realistically."""

    def test_mock_controller_basic_properties(self):
        """Test mock controller provides expected properties."""
        controller = create_mock_controller_r1pro_bilateral()

        # Test dof
        self.assertEqual(controller.dof, 14)

        # Test q returns correct shape
        positions, valid = controller.q
        self.assertEqual(len(positions), 14)
        self.assertEqual(len(valid), 14)
        self.assertTrue(np.all(valid))

    def test_mock_controller_goal_tracking(self):
        """Test mock controller tracks goals."""
        controller = create_mock_controller_r1pro_bilateral()

        # Set a goal
        goal = np.array([0.5] * 14)
        controller.set_new_goal(goal)
        controller.resume_goal_following()

        # Initial position should be zeros
        initial_pos, _ = controller.q
        self.assertTrue(np.allclose(initial_pos, 0, atol=0.01))

        # After multiple reads, should move toward goal
        for _ in range(20):
            pos, _ = controller.q

        # Should be closer to goal
        final_pos, _ = controller.q
        self.assertTrue(np.mean(np.abs(final_pos - goal)) < np.mean(np.abs(initial_pos - goal)))

    def test_mock_controller_state_setting(self):
        """Test setting controller state for testing."""
        controller = create_mock_controller_r1pro_bilateral()

        # Set specific positions
        test_positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] * 2)
        controller.set_position(test_positions)

        positions, _ = controller.q
        self.assertTrue(np.allclose(positions, test_positions, atol=0.01))


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""
End-to-end system tests for Gello + Policy integration.

Tests the complete flow from configuration loading to action generation
without requiring actual hardware.
"""

import unittest
import torch
import numpy as np
from unittest.mock import patch, Mock
from omegaconf import OmegaConf

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.fixtures import MockControllerFactory


class TestEndToEndSystem(unittest.TestCase):
    """End-to-end system tests."""

    def create_full_system_config(self, gello_config="gello_r1pro", robot_arm_dof=6):
        """Create complete system configuration."""
        return OmegaConf.create({
            'type': 'gello',
            'control_type': 'joint',
            'gello_config': gello_config,
            'base_policy': 'curobo',
            'control_direction': 'env_to_gello',
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
        })

    @patch('iiil.policies.teleop_policy.A1GelloControllerFactory', new=MockControllerFactory)
    @patch('iiil.policies.teleop_policy.JoyconController')
    def test_complete_workflow_a1_robot_r1pro_gello(self, mock_joycon):
        """Test complete workflow: A1 robot with R1Pro bilateral Gello."""
        mock_joycon.return_value = Mock(trigger_r=False, trigger_zr=False, button_x=False)

        config = self.create_full_system_config(gello_config="gello_r1pro", robot_arm_dof=6)

        with patch.object(sys.modules['iiil.policies.teleop_policy'].GelloPolicy,
                         '_create_base_policy', return_value=Mock(
                             config=Mock(control_type='joint'),
                             predict=Mock(return_value=torch.zeros(7))
                         )):
            from iiil.policies.teleop_policy import GelloPolicy

            # Initialize policy
            policy = GelloPolicy(config)

            # Verify correct DOF detection
            self.assertEqual(policy._gello_controller.dof, 14)
            self.assertEqual(policy._num_arm_joints, 6)

            # Simulate robot observation
            obs = {
                'robot::proprio': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0])  # 6 arm + 1 gripper
            }

            # Generate action
            action = policy._create_gello_action(obs)

            # Verify action has correct shape
            self.assertEqual(len(action), 7)  # 6 arm joints + 1 gripper

            # Verify action values are reasonable
            self.assertTrue(torch.all(torch.isfinite(action)))

    @patch('iiil.policies.teleop_policy.A1GelloControllerFactory', new=MockControllerFactory)
    @patch('iiil.policies.teleop_policy.JoyconController')
    def test_complete_workflow_r1pro_robot_r1pro_gello(self, mock_joycon):
        """Test complete workflow: R1Pro robot with R1Pro bilateral Gello."""
        mock_joycon.return_value = Mock(trigger_r=False, trigger_zr=False, button_x=False)

        config = self.create_full_system_config(gello_config="gello_r1pro", robot_arm_dof=7)

        with patch.object(sys.modules['iiil.policies.teleop_policy'].GelloPolicy,
                         '_create_base_policy', return_value=Mock(
                             config=Mock(control_type='joint'),
                             predict=Mock(return_value=torch.zeros(8))
                         )):
            from iiil.policies.teleop_policy import GelloPolicy

            # Initialize policy
            policy = GelloPolicy(config)

            # Verify correct DOF detection
            self.assertEqual(policy._gello_controller.dof, 14)
            self.assertEqual(policy._num_arm_joints, 7)

            # Simulate robot observation
            obs = {
                'robot::proprio': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0])  # 7 arm + 1 gripper
            }

            # Generate action
            action = policy._create_gello_action(obs)

            # Verify action has correct shape
            self.assertEqual(len(action), 8)  # 7 arm joints + 1 gripper

            # Verify action values are reasonable
            self.assertTrue(torch.all(torch.isfinite(action)))

    @patch('iiil.policies.teleop_policy.A1GelloControllerFactory', new=MockControllerFactory)
    @patch('iiil.policies.teleop_policy.JoyconController')
    def test_goal_setting_and_tracking(self, mock_joycon):
        """Test setting goals and tracking them."""
        mock_joycon.return_value = Mock(trigger_r=False, trigger_zr=False, button_x=False)

        config = self.create_full_system_config(gello_config="gello_r1pro", robot_arm_dof=6)

        with patch.object(sys.modules['iiil.policies.teleop_policy'].GelloPolicy,
                         '_create_base_policy', return_value=Mock(
                             config=Mock(control_type='joint'),
                             predict=Mock(return_value=torch.zeros(7))
                         )):
            from iiil.policies.teleop_policy import GelloPolicy

            policy = GelloPolicy(config)

            # Set a goal
            goal = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
            policy._gello_controller.set_new_goal(goal)

            # Verify goal is set
            self.assertIsNotNone(policy._gello_controller._goal)
            self.assertEqual(len(policy._gello_controller._goal), 14)  # Padded to full DOF

    @patch('iiil.policies.teleop_policy.A1GelloControllerFactory', new=MockControllerFactory)
    @patch('iiil.policies.teleop_policy.JoyconController')
    def test_control_loop_simulation(self, mock_joycon):
        """Test simulated control loop execution."""
        mock_joycon.return_value = Mock(trigger_r=False, trigger_zr=False, button_x=False)

        config = self.create_full_system_config(gello_config="gello_r1pro", robot_arm_dof=6)

        with patch.object(sys.modules['iiil.policies.teleop_policy'].GelloPolicy,
                         '_create_base_policy', return_value=Mock(
                             config=Mock(control_type='joint'),
                             predict=Mock(return_value=torch.zeros(7))
                         )):
            from iiil.policies.teleop_policy import GelloPolicy

            policy = GelloPolicy(config)

            # Simulate multiple control steps
            for step in range(10):
                obs = {
                    'robot::proprio': torch.randn(7)  # Random robot state
                }

                # Generate action
                action = policy._create_gello_action(obs)

                # Verify action is valid
                self.assertEqual(len(action), 7)
                self.assertTrue(torch.all(torch.isfinite(action)))

                # Set new goal based on action (simulating robot feedback)
                policy._gello_controller.set_new_goal(action[:6].numpy())

    @patch('iiil.policies.teleop_policy.A1GelloControllerFactory', new=MockControllerFactory)
    @patch('iiil.policies.teleop_policy.JoyconController')
    def test_graceful_degradation_with_invalid_joints(self, mock_joycon):
        """Test system handles invalid/disconnected joints gracefully."""
        mock_joycon.return_value = Mock(trigger_r=False, trigger_zr=False, button_x=False)

        config = self.create_full_system_config(gello_config="gello_r1pro", robot_arm_dof=6)

        with patch.object(sys.modules['iiil.policies.teleop_policy'].GelloPolicy,
                         '_create_base_policy', return_value=Mock(
                             config=Mock(control_type='joint'),
                             predict=Mock(return_value=torch.zeros(7))
                         )):
            from iiil.policies.teleop_policy import GelloPolicy

            policy = GelloPolicy(config)

            # Mark some joints as invalid
            valid_joints = np.ones(14, dtype=bool)
            valid_joints[5] = False  # Invalidate joint 5
            policy._gello_controller.set_joint_validity(valid_joints)

            obs = {
                'robot::proprio': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0])
            }

            # Should still generate valid action
            action = policy._create_gello_action(obs)

            # Joint 5 should use robot observation instead of Gello
            self.assertAlmostEqual(action[5].item(), obs['robot::proprio'][5].item())


class TestSystemRobustness(unittest.TestCase):
    """Test system robustness and edge cases."""

    @patch('iiil.policies.teleop_policy.A1GelloControllerFactory', new=MockControllerFactory)
    @patch('iiil.policies.teleop_policy.JoyconController')
    def test_handles_zero_positions(self, mock_joycon):
        """Test system handles all-zero positions."""
        mock_joycon.return_value = Mock(trigger_r=False, trigger_zr=False, button_x=False)

        config = OmegaConf.create({
            'type': 'gello',
            'control_type': 'joint',
            'gello_config': 'gello_r1pro',
            'base_policy': 'curobo',
            'control_direction': 'gello_to_env',
            'bilateral_blend_factor': 0.5,
            'robot_type': {
                'type': 'A1',
                'action_key_dims': {'arm': 6, 'gripper': 1}
            }
        })

        with patch.object(sys.modules['iiil.policies.teleop_policy'].GelloPolicy,
                         '_create_base_policy', return_value=Mock(
                             config=Mock(control_type='joint'),
                             predict=Mock(return_value=torch.zeros(7))
                         )):
            from iiil.policies.teleop_policy import GelloPolicy

            policy = GelloPolicy(config)
            obs = {'robot::proprio': torch.zeros(7)}

            action = policy._create_gello_action(obs)
            self.assertTrue(torch.all(torch.isfinite(action)))

    @patch('iiil.policies.teleop_policy.A1GelloControllerFactory', new=MockControllerFactory)
    @patch('iiil.policies.teleop_policy.JoyconController')
    def test_handles_extreme_positions(self, mock_joycon):
        """Test system handles positions near joint limits."""
        mock_joycon.return_value = Mock(trigger_r=False, trigger_zr=False, button_x=False)

        config = OmegaConf.create({
            'type': 'gello',
            'control_type': 'joint',
            'gello_config': 'gello_r1pro',
            'base_policy': 'curobo',
            'control_direction': 'gello_to_env',
            'bilateral_blend_factor': 0.5,
            'robot_type': {
                'type': 'A1',
                'action_key_dims': {'arm': 6, 'gripper': 1}
            }
        })

        with patch.object(sys.modules['iiil.policies.teleop_policy'].GelloPolicy,
                         '_create_base_policy', return_value=Mock(
                             config=Mock(control_type='joint'),
                             predict=Mock(return_value=torch.zeros(7))
                         )):
            from iiil.policies.teleop_policy import GelloPolicy

            policy = GelloPolicy(config)

            # Set extreme positions
            extreme_positions = np.array([np.pi, -np.pi, np.pi/2, -np.pi/2, 0, 0] * 2 + [0, 0])
            policy._gello_controller.set_position(extreme_positions)

            obs = {'robot::proprio': torch.zeros(7)}
            action = policy._create_gello_action(obs)

            # Should handle extreme positions without NaN/Inf
            self.assertTrue(torch.all(torch.isfinite(action)))


if __name__ == '__main__':
    unittest.main()

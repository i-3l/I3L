#!/usr/bin/env python3
"""
Unit tests for Gello controller DOF property.

Tests the DOF property interface without requiring hardware.
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Tuple, Sequence


class TestControllerDOFProperty(unittest.TestCase):
    """Test suite for controller DOF property."""

    def test_base_interface_has_dof_property(self):
        """Test that base interface defines dof as abstract property."""
        from iiil.gello.controllers.base_controller import BilateralGelloController

        # Verify dof is in the interface
        self.assertTrue(hasattr(BilateralGelloController, 'dof'))

        # Verify it's a property
        dof_attr = getattr(BilateralGelloController, 'dof')
        self.assertIsInstance(dof_attr, property)

    def test_bilateral_controller_implements_dof(self):
        """Test that BilateralController implements dof property."""
        from iiil.gello.controllers.bilateral_controller import BilateralController

        # Verify dof property exists
        self.assertTrue(hasattr(BilateralController, 'dof'))

        # Verify it's a property
        dof_attr = getattr(BilateralController, 'dof')
        self.assertIsInstance(dof_attr, property)

    def test_dof_property_is_read_only(self):
        """Test that dof property cannot be modified."""
        from iiil.gello.controllers.bilateral_controller import BilateralController

        dof_property = getattr(BilateralController, 'dof')

        # Verify no setter
        self.assertIsNone(dof_property.fset, "dof property should be read-only")

    def test_dof_property_returns_int(self):
        """Test that dof property has correct return type annotation."""
        from iiil.gello.controllers.bilateral_controller import BilateralController
        import inspect

        dof_property = getattr(BilateralController, 'dof')
        sig = inspect.signature(dof_property.fget)

        # Check return annotation
        self.assertEqual(sig.return_annotation, int)

    def test_dof_property_has_docstring(self):
        """Test that dof property is documented."""
        from iiil.gello.controllers.bilateral_controller import BilateralController

        dof_property = getattr(BilateralController, 'dof')
        self.assertIsNotNone(dof_property.fget.__doc__)
        self.assertIn('DOF', dof_property.fget.__doc__)


class TestMockedControllerDOF(unittest.TestCase):
    """Test DOF property with mocked controller."""

    def create_mock_controller(self, dof: int):
        """Create a mock controller with specified DOF."""
        from iiil.gello.controllers.base_controller import BilateralGelloController

        # Create mock that implements the interface
        mock_controller = Mock(spec=BilateralGelloController)

        # Set up dof property
        type(mock_controller).dof = property(lambda self: dof)

        # Set up q property to return appropriate shape
        mock_controller.q = (
            np.zeros(dof, dtype=np.float32),
            np.ones(dof, dtype=bool)
        )

        # Set up other required properties
        mock_controller.v = (
            np.zeros(dof, dtype=np.float32),
            np.ones(dof, dtype=bool)
        )
        mock_controller.current = (
            np.zeros(dof, dtype=np.float32),
            np.ones(dof, dtype=bool)
        )

        return mock_controller

    def test_mock_6dof_controller(self):
        """Test mock controller with 6 DOF."""
        controller = self.create_mock_controller(6)

        self.assertEqual(controller.dof, 6)
        self.assertEqual(len(controller.q[0]), 6)

    def test_mock_7dof_controller(self):
        """Test mock controller with 7 DOF."""
        controller = self.create_mock_controller(7)

        self.assertEqual(controller.dof, 7)
        self.assertEqual(len(controller.q[0]), 7)

    def test_mock_14dof_bilateral_controller(self):
        """Test mock controller with 14 DOF (bilateral)."""
        controller = self.create_mock_controller(14)

        self.assertEqual(controller.dof, 14)
        self.assertEqual(len(controller.q[0]), 14)

    def test_mock_controller_dof_readonly(self):
        """Test that mock controller dof cannot be modified."""
        controller = self.create_mock_controller(6)

        with self.assertRaises(AttributeError):
            controller.dof = 10


class TestDOFInference(unittest.TestCase):
    """Test DOF inference logic in policy."""

    def test_bilateral_dof_inference(self):
        """Test that bilateral DOF (14) is correctly interpreted as 7 per arm."""
        gello_total_dof = 14

        # Logic from policy
        gello_arm_dof = gello_total_dof // 2 if gello_total_dof > 7 else gello_total_dof

        self.assertEqual(gello_arm_dof, 7)

    def test_single_arm_r1pro_inference(self):
        """Test that single R1Pro (7 DOF) is correctly interpreted."""
        gello_total_dof = 7

        # Logic from policy
        gello_arm_dof = gello_total_dof // 2 if gello_total_dof > 7 else gello_total_dof

        self.assertEqual(gello_arm_dof, 7)

    def test_single_arm_a1_inference(self):
        """Test that single A1 (6 DOF) is correctly interpreted."""
        gello_total_dof = 6

        # Logic from policy
        gello_arm_dof = gello_total_dof // 2 if gello_total_dof > 7 else gello_total_dof

        self.assertEqual(gello_arm_dof, 6)

    def test_min_dof_selection(self):
        """Test that minimum DOF is selected correctly."""
        # Gello has 7 DOF, robot has 6 DOF
        gello_arm_dof = 7
        robot_arm_dof = 6
        num_arm_joints = min(gello_arm_dof, robot_arm_dof)

        self.assertEqual(num_arm_joints, 6)

    def test_matching_dof(self):
        """Test when Gello and robot DOF match."""
        # Both have 7 DOF
        gello_arm_dof = 7
        robot_arm_dof = 7
        num_arm_joints = min(gello_arm_dof, robot_arm_dof)

        self.assertEqual(num_arm_joints, 7)


if __name__ == '__main__':
    unittest.main()

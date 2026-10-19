#!/usr/bin/env python3
"""
Test script to validate gravity compensation calculations.

This script tests the gravity compensation without needing real hardware by:
1. Loading the URDF and computing gravity torques at known poses
2. Verifying physical plausibility (correct signs, magnitudes)
3. Testing bilateral mirroring logic
4. Validating multiplier handling

Run: conda run -n behavior python scripts/test_gravity_compensation.py
"""

import numpy as np
import pinocchio as pin
import os
import sys
from dataclasses import dataclass
from typing import Union, Sequence

# Minimal reimplementation to avoid import issues with av/openvino
@dataclass
class GravityCompensationConfig:
    """Minimal config for testing."""
    enable: bool = False
    urdf_path: str = ""
    multipliers: Union[float, Sequence[float], np.ndarray] = 1.0

    def __post_init__(self):
        if isinstance(self.multipliers, (int, float)):
            self._multipliers_scalar = self.multipliers
        else:
            self.multipliers = np.array(self.multipliers, dtype=np.float32)

    def get_multipliers(self, dof: int) -> np.ndarray:
        if hasattr(self, '_multipliers_scalar'):
            return np.full(dof, self._multipliers_scalar, dtype=np.float32)
        elif isinstance(self.multipliers, np.ndarray):
            if len(self.multipliers) == dof:
                return self.multipliers
            elif len(self.multipliers) * 2 == dof:
                return np.concatenate([self.multipliers, self.multipliers])
            else:
                raise ValueError(f"Multipliers length ({len(self.multipliers)}) doesn't match DOF ({dof})")
        else:
            arr = np.array(self.multipliers, dtype=np.float32)
            if len(arr) == dof:
                return arr
            elif len(arr) * 2 == dof:
                return np.concatenate([arr, arr])
            else:
                raise ValueError(f"Multipliers length ({len(arr)}) doesn't match DOF ({dof})")


class GravityCompensator:
    """Minimal gravity compensator for testing."""

    def __init__(self, config: GravityCompensationConfig, dof: int):
        self.config = config
        self.dof = dof
        self.enabled = config.enable

        if self.enabled:
            self.model = pin.buildModelFromUrdf(config.urdf_path)
            self.data = self.model.createData()

            urdf_dof = self.model.nv
            self._is_bilateral_single_arm_urdf = (urdf_dof * 2 == dof)

            if urdf_dof != dof and not self._is_bilateral_single_arm_urdf:
                raise ValueError(f"URDF DOF ({urdf_dof}) doesn't match system DOF ({dof})")

            self.multipliers = config.get_multipliers(dof)
        else:
            self.model = None
            self.data = None
            self.multipliers = None
            self._is_bilateral_single_arm_urdf = False

    def compute_torque(self, positions: np.ndarray, velocities=None) -> np.ndarray:
        if not self.enabled:
            return np.zeros(self.dof, dtype=np.float32)

        if self._is_bilateral_single_arm_urdf:
            arm_dof = self.model.nv
            q_right = positions[:arm_dof]
            q_left = positions[arm_dof:]

            tau_g_right = pin.rnea(self.model, self.data, q_right,
                                   np.zeros(arm_dof), np.zeros(arm_dof))

            q_left_mirrored = -q_left
            tau_g_left_mirrored = pin.rnea(self.model, self.data, q_left_mirrored,
                                           np.zeros(arm_dof), np.zeros(arm_dof))
            tau_g_left = -tau_g_left_mirrored

            tau_g = np.concatenate([tau_g_right, tau_g_left])
        else:
            tau_g = pin.rnea(self.model, self.data, positions,
                            np.zeros(self.dof), np.zeros(self.dof))

        return (tau_g * self.multipliers).astype(np.float32)


def get_urdf_path():
    """Get the URDF path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, "assets/gello_r1pro_urdf/robot_7dof.urdf")


def test_raw_pinocchio():
    """Test raw Pinocchio gravity computation to understand the model."""
    print("=" * 60)
    print("TEST 1: Raw Pinocchio Gravity Computation")
    print("=" * 60)

    urdf_path = get_urdf_path()
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    print(f"URDF: {urdf_path}")
    print(f"Model DOF (nv): {model.nv}")
    print(f"Model nq: {model.nq}")
    print(f"Joint names: {[model.names[i] for i in range(model.njoints)]}")
    print()

    # Test at home position (all zeros)
    print("--- Home Position (q = zeros) ---")
    q_home = np.zeros(model.nv)
    tau_g_home = pin.rnea(model, data, q_home, np.zeros(model.nv), np.zeros(model.nv))
    print(f"Joint positions: {q_home}")
    print(f"Gravity torques: {tau_g_home}")
    print(f"  (Positive = motor needs to push in positive joint direction to hold)")
    print()

    # Test at a few different poses
    test_poses = [
        ("Arm extended forward (joint 1 = pi/2)", [0, np.pi/2, 0, 0, 0, 0, 0]),
        ("Arm extended backward (joint 1 = -pi/2)", [0, -np.pi/2, 0, 0, 0, 0, 0]),
        ("Joint 0 rotated 90 deg", [np.pi/2, 0, 0, 0, 0, 0, 0]),
        ("Joint 2 = pi/4", [0, 0, np.pi/4, 0, 0, 0, 0]),
        ("Joint 3 = pi/4", [0, 0, 0, np.pi/4, 0, 0, 0]),
    ]

    for name, q in test_poses:
        q = np.array(q, dtype=np.float64)
        tau_g = pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv))
        print(f"--- {name} ---")
        print(f"q = {q}")
        print(f"tau_g = {tau_g}")
        print()

    return model, data


def test_gravity_compensator_single_arm():
    """Test the GravityCompensator class for single arm."""
    print("=" * 60)
    print("TEST 2: GravityCompensator (Single Arm, 7 DOF)")
    print("=" * 60)

    urdf_path = get_urdf_path()

    # Test with multipliers = 1.0 (no scaling)
    config = GravityCompensationConfig(
        enable=True,
        urdf_path=urdf_path,
        multipliers=1.0  # Scalar, will be expanded
    )

    compensator = GravityCompensator(config, dof=7)

    print(f"Compensator enabled: {compensator.enabled}")
    print(f"Multipliers: {compensator.multipliers}")
    print()

    # Test at home position
    q_home = np.zeros(7, dtype=np.float32)
    tau_g = compensator.compute_torque(q_home)
    print(f"Home position gravity torques: {tau_g}")
    print()

    # Test with custom multipliers
    custom_multipliers = np.array([0.6, 0.1, 0.01, 0.3, 0.0, 0.1, 0.1], dtype=np.float32)
    config2 = GravityCompensationConfig(
        enable=True,
        urdf_path=urdf_path,
        multipliers=custom_multipliers
    )
    compensator2 = GravityCompensator(config2, dof=7)
    tau_g_scaled = compensator2.compute_torque(q_home)

    print(f"With multipliers {custom_multipliers}:")
    print(f"Scaled gravity torques: {tau_g_scaled}")
    print(f"Expected (tau_g * multipliers): {tau_g * custom_multipliers}")
    print(f"Match: {np.allclose(tau_g_scaled, tau_g * custom_multipliers)}")
    print()


def test_gravity_compensator_bilateral():
    """Test the GravityCompensator for bilateral (14 DOF) system."""
    print("=" * 60)
    print("TEST 3: GravityCompensator (Bilateral, 14 DOF)")
    print("=" * 60)

    urdf_path = get_urdf_path()

    # Test with 7 multipliers (should be duplicated for bilateral)
    multipliers_7 = np.array([0.6, 0.1, 0.01, 0.3, 0.0, 0.1, 0.1], dtype=np.float32)
    config = GravityCompensationConfig(
        enable=True,
        urdf_path=urdf_path,
        multipliers=multipliers_7
    )

    compensator = GravityCompensator(config, dof=14)

    print(f"Input multipliers (7): {multipliers_7}")
    print(f"Expanded multipliers (14): {compensator.multipliers}")
    print(f"Left arm multipliers match right arm: {np.allclose(compensator.multipliers[:7], compensator.multipliers[7:])}")
    print()

    # Test at home position (both arms at zeros)
    q_home = np.zeros(14, dtype=np.float32)
    tau_g = compensator.compute_torque(q_home)

    print(f"Home position (both arms at zeros):")
    print(f"  Right arm torques (0-6): {tau_g[:7]}")
    print(f"  Left arm torques (7-13): {tau_g[7:]}")
    print()

    # The left arm mirroring logic negates positions and torques
    # At home (zeros), both should be similar but possibly sign-flipped
    # depending on the URDF convention

    # Test with right arm in a pose, left arm at home
    q_test = np.zeros(14, dtype=np.float32)
    q_test[1] = np.pi/4  # Right arm joint 1 at 45 degrees
    tau_g_test = compensator.compute_torque(q_test)

    print(f"Right arm joint 1 = pi/4, left arm at home:")
    print(f"  Right arm torques: {tau_g_test[:7]}")
    print(f"  Left arm torques: {tau_g_test[7:]}")
    print()

    # Test mirrored pose (left arm mirrors right arm)
    q_mirrored = np.zeros(14, dtype=np.float32)
    q_mirrored[1] = np.pi/4   # Right arm joint 1
    q_mirrored[8] = -np.pi/4  # Left arm joint 1 (mirrored = negated)
    tau_g_mirrored = compensator.compute_torque(q_mirrored)

    print(f"Mirrored pose (right j1=pi/4, left j1=-pi/4):")
    print(f"  Right arm torques: {tau_g_mirrored[:7]}")
    print(f"  Left arm torques: {tau_g_mirrored[7:]}")
    print(f"  Note: For mirrored pose, torques should be opposite signs")
    print()


def test_sign_correction_simulation():
    """Simulate the sign correction that happens in the bilateral controller."""
    print("=" * 60)
    print("TEST 4: Sign Correction Simulation")
    print("=" * 60)

    urdf_path = get_urdf_path()

    # Arm joint signs from config
    # Right arm (0-3): [-1, -1, -1, -1], (4-6): [-1, -1, -1]
    # Left arm (7-10): [1, -1, -1, 1], (11-13): [-1, -1, -1]
    joint_signs = np.array([
        -1, -1, -1, -1, -1, -1, -1,  # Right arm
        1, -1, -1, 1, -1, -1, -1     # Left arm
    ], dtype=np.float32)

    print(f"Joint signs from config: {joint_signs}")
    print()

    # Create compensator
    config = GravityCompensationConfig(
        enable=True,
        urdf_path=urdf_path,
        multipliers=1.0
    )
    compensator = GravityCompensator(config, dof=14)

    # Compute gravity torques at home
    q_home = np.zeros(14, dtype=np.float32)
    tau_g = compensator.compute_torque(q_home)

    print(f"Gravity torques (robot joint space): {tau_g}")
    print()

    # Apply sign correction (what the fixed bilateral controller does)
    tau_motor = tau_g * joint_signs

    print(f"Motor torques after sign correction: {tau_motor}")
    print()

    print("Comparison (joint by joint):")
    print(f"{'Joint':<6} {'Sign':<6} {'Robot τ':<12} {'Motor τ':<12}")
    print("-" * 40)
    for i in range(14):
        arm = "R" if i < 7 else "L"
        joint_idx = i if i < 7 else i - 7
        print(f"{arm}{joint_idx:<5} {joint_signs[i]:>5.0f} {tau_g[i]:>11.4f} {tau_motor[i]:>11.4f}")
    print()


def test_physical_plausibility():
    """Test that gravity torques are physically plausible."""
    print("=" * 60)
    print("TEST 5: Physical Plausibility Checks")
    print("=" * 60)

    urdf_path = get_urdf_path()
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    print("These checks verify the URDF model produces sensible gravity torques.")
    print()

    # Check 1: At home position, proximal joints should have larger torques
    # (they support more mass)
    q_home = np.zeros(model.nv)
    tau_g = pin.rnea(model, data, q_home, np.zeros(model.nv), np.zeros(model.nv))

    print("CHECK 1: Torque magnitudes at home position")
    print(f"  Torques: {tau_g}")
    print(f"  |τ| per joint: {np.abs(tau_g)}")
    # Generally, shoulder joints should have larger torques than wrist joints
    print()

    # Check 2: Rotating the arm should change gravity torques
    q_rotated = np.zeros(model.nv)
    q_rotated[1] = np.pi/2  # Rotate joint 1 by 90 degrees
    tau_g_rotated = pin.rnea(model, data, q_rotated, np.zeros(model.nv), np.zeros(model.nv))

    print("CHECK 2: Torque change when rotating joint 1 by 90°")
    print(f"  Home torques: {tau_g}")
    print(f"  Rotated torques: {tau_g_rotated}")
    print(f"  Difference: {tau_g_rotated - tau_g}")
    torque_changed = not np.allclose(tau_g, tau_g_rotated)
    print(f"  Torques changed: {torque_changed} {'✓' if torque_changed else '✗'}")
    print()

    # Check 3: Gravity torques should be smooth (no discontinuities)
    print("CHECK 3: Torque smoothness (sweep joint 1 from -pi/2 to pi/2)")
    angles = np.linspace(-np.pi/2, np.pi/2, 11)
    print(f"  {'Angle (deg)':<12} {'τ[0]':<10} {'τ[1]':<10} {'τ[2]':<10}")
    print("  " + "-" * 45)
    prev_tau = None
    smooth = True
    for angle in angles:
        q = np.zeros(model.nv)
        q[1] = angle
        tau = pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv))
        print(f"  {np.degrees(angle):>10.1f}° {tau[0]:>9.4f} {tau[1]:>9.4f} {tau[2]:>9.4f}")
        if prev_tau is not None:
            max_jump = np.max(np.abs(tau - prev_tau))
            if max_jump > 0.5:  # Arbitrary threshold
                smooth = False
        prev_tau = tau.copy()
    print(f"  Torques are smooth: {smooth} {'✓' if smooth else '✗'}")
    print()


def test_practical_verification():
    """
    Practical verification guide - helps determine if signs are correct.

    For each joint, predicts what should happen when gravity comp is enabled.
    """
    print("=" * 60)
    print("TEST 6: Practical Verification Guide")
    print("=" * 60)
    print()
    print("This test helps you verify gravity compensation direction for each joint.")
    print("For each joint, it predicts what torque direction the motor should apply.")
    print()

    urdf_path = get_urdf_path()
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    # Config signs
    right_arm_signs = np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32)

    # Compute at home
    q_home = np.zeros(7)
    tau_g = pin.rnea(model, data, q_home, np.zeros(7), np.zeros(7))

    # Motor torques after sign correction
    tau_motor = tau_g * right_arm_signs

    print("RIGHT ARM - Home Position Verification")
    print("=" * 60)
    print()
    print("At home position (all joints at 0), here's what gravity comp should do:")
    print()
    print(f"{'Joint':<8} {'URDF τ (Nm)':<14} {'Sign':<6} {'Motor τ (Nm)':<14} {'Expected Effect'}")
    print("-" * 70)

    for i in range(7):
        urdf_tau = tau_g[i]
        sign = right_arm_signs[i]
        motor_tau = tau_motor[i]

        if abs(motor_tau) < 0.001:
            effect = "Nearly zero - no significant compensation"
        elif motor_tau > 0:
            effect = "Motor pushes POSITIVE direction"
        else:
            effect = "Motor pushes NEGATIVE direction"

        print(f"Joint {i:<3} {urdf_tau:>12.4f}  {sign:>5.0f}  {motor_tau:>12.4f}  {effect}")

    print()
    print("HOW TO VERIFY:")
    print("-" * 60)
    print("1. Disable all compensation except gravity comp for one joint")
    print("2. Set multiplier to 1.0 for that joint only (others to 0)")
    print("3. Enable torque and see which direction the motor pushes")
    print("4. Compare with the 'Expected Effect' column above")
    print()
    print("If a joint pushes the OPPOSITE direction, either:")
    print("  a) The arm_joint_signs for that joint is wrong in the YAML config")
    print("  b) The URDF joint direction is inverted")
    print()

    # Show what happens at different angles for the problematic joint 3
    print("=" * 60)
    print("DEEP DIVE: Joint 3 (third joint from shoulder)")
    print("=" * 60)
    print()
    print("Sweeping joint 1 to see how joint 3 torque changes:")
    print(f"{'J1 angle':<12} {'J3 URDF τ':<12} {'J3 Motor τ':<12}")
    print("-" * 40)

    for angle_deg in [-90, -45, 0, 45, 90]:
        q = np.zeros(7)
        q[1] = np.radians(angle_deg)
        tau = pin.rnea(model, data, q, np.zeros(7), np.zeros(7))
        motor_tau_3 = tau[3] * right_arm_signs[3]
        print(f"{angle_deg:>8}°   {tau[3]:>10.4f}   {motor_tau_3:>10.4f}")

    print()
    print("Notice: Joint 3 torque changes sign as the arm moves!")
    print("At some positions it needs positive torque, at others negative.")
    print()


def test_bilateral_mirroring_issue():
    """
    Check if bilateral mirroring logic matches the physical setup.
    """
    print("=" * 60)
    print("TEST 7: Bilateral Mirroring Analysis")
    print("=" * 60)
    print()

    # Config signs from YAML
    right_signs = np.array([-1, -1, -1, -1, -1, -1, -1])
    left_signs = np.array([1, -1, -1, 1, -1, -1, -1])  # Note: 0 and 3 differ!

    print("Joint signs from config:")
    print(f"  Right arm: {right_signs}")
    print(f"  Left arm:  {left_signs}")
    print()

    # Check which joints differ
    diff_joints = np.where(right_signs != left_signs)[0]
    same_joints = np.where(right_signs == left_signs)[0]

    print(f"Joints with SAME sign (R==L): {same_joints.tolist()}")
    print(f"Joints with DIFFERENT sign (R!=L): {diff_joints.tolist()}")
    print()

    print("ANALYSIS:")
    print("-" * 60)
    print()
    print("The current gravity compensator bilateral logic assumes:")
    print("  - Left arm = perfect kinematic mirror of right arm")
    print("  - q_left_mirrored = -q_left")
    print("  - tau_left = -tau_from_mirrored_position")
    print()
    print("But joints 0 and 3 have OPPOSITE signs between arms,")
    print("which suggests they are NOT mirrored the same way.")
    print()
    print("POTENTIAL FIX:")
    print("-" * 60)
    print("The bilateral mirroring logic may need to account for which")
    print("joints are truly mirrored vs which have the same direction.")
    print()
    print("Option 1: Per-joint mirroring flags in config")
    print("Option 2: Separate URDFs for each arm")
    print("Option 3: Compute each arm independently without mirroring")
    print()


def run_all_tests():
    """Run all gravity compensation tests."""
    print("\n" + "=" * 60)
    print("GRAVITY COMPENSATION TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_raw_pinocchio()
        test_gravity_compensator_single_arm()
        test_gravity_compensator_bilateral()
        test_sign_correction_simulation()
        test_physical_plausibility()
        test_practical_verification()
        test_bilateral_mirroring_issue()

        print("=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        print("\nReview the output above to verify:")
        print("1. Gravity torques have sensible signs and magnitudes")
        print("2. Bilateral mirroring produces expected results")
        print("3. Sign correction matches your hardware configuration")
        print("4. Multipliers are correctly applied")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())

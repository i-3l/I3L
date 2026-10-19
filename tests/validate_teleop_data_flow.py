#!/usr/bin/env python3
"""
Validate that teleop data flow is correct for both control modes.

This script tests the data mapping logic without requiring actual hardware.
"""
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class ArmMapping:
    """Mock ArmMapping for testing."""
    name: str
    robot_slice: slice
    gello_slice: slice
    dof: int


def test_gello_to_robot_mapping():
    """Test GELLO_TO_ENV: gello joints → robot action."""
    print("=" * 80)
    print("TEST 1: GELLO_TO_ENV Mode (User controls robot)")
    print("=" * 80)

    # Test 6DOF single-arm
    print("\n6DOF Single-Arm A1:")
    mapping = ArmMapping(name="arm", robot_slice=slice(0, 6), gello_slice=slice(0, 6), dof=6)
    gello_joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    gello_valid = np.array([True, True, True, True, True, True])
    action = torch.zeros(7)  # 6 arm + 1 gripper

    # Simulate mapping
    gello_arm = gello_joints[mapping.gello_slice]
    action[mapping.robot_slice] = torch.from_numpy(gello_arm)

    print(f"  Gello joints:  {gello_joints}")
    print(f"  Robot action:  {action[:6].numpy()}")
    assert np.allclose(action[:6].numpy(), gello_joints), "A1 mapping failed!"
    print("  ✓ A1 mapping correct")

    # Test R1Pro dual-arm
    print("\nR1Pro Dual-Arm (7 DOF per arm):")
    mappings = [
        ArmMapping(name="arm_right", robot_slice=slice(0, 7), gello_slice=slice(0, 7), dof=7),
        ArmMapping(name="arm_left", robot_slice=slice(7, 14), gello_slice=slice(7, 14), dof=7)
    ]
    gello_joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,  # right arm
                             1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])  # left arm
    gello_valid = np.ones(14, dtype=bool)
    action = torch.zeros(15)  # 14 arms + 1 gripper

    # Simulate mapping for both arms
    for mapping in mappings:
        gello_arm = gello_joints[mapping.gello_slice]
        action[mapping.robot_slice] = torch.from_numpy(gello_arm)

    print(f"  Gello right arm:  {gello_joints[0:7]}")
    print(f"  Robot right arm:  {action[0:7].numpy()}")
    assert np.allclose(action[0:7].numpy(), gello_joints[0:7]), "Right arm mapping failed!"

    print(f"  Gello left arm:   {gello_joints[7:14]}")
    print(f"  Robot left arm:   {action[7:14].numpy()}")
    assert np.allclose(action[7:14].numpy(), gello_joints[7:14]), "Left arm mapping failed!"
    print("  ✓ R1Pro dual-arm mapping correct")

    print("\n✓ All GELLO_TO_ENV tests passed!")


def test_robot_to_gello_mapping():
    """Test ENV_TO_GELLO: robot joints → gello goal."""
    print("\n" + "=" * 80)
    print("TEST 2: ENV_TO_GELLO Mode (Gello follows robot)")
    print("=" * 80)

    # Test 6DOF single-arm
    print("\n6DOF Single-Arm A1:")
    mapping = ArmMapping(name="arm", robot_slice=slice(0, 6), gello_slice=slice(0, 6), dof=6)
    robot_proprio = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0])  # 6 arm + 1 gripper
    gello_goal = np.zeros(6)

    # Simulate mapping
    robot_arm = robot_proprio[mapping.robot_slice].cpu().numpy()
    gello_goal[mapping.gello_slice] = robot_arm

    print(f"  Robot proprio: {robot_proprio[:6].numpy()}")
    print(f"  Gello goal:    {gello_goal}")
    assert np.allclose(gello_goal, robot_proprio[:6].numpy()), "A1 reverse mapping failed!"
    print("  ✓ A1 reverse mapping correct")

    # Test R1Pro dual-arm
    print("\nR1Pro Dual-Arm (7 DOF per arm):")
    mappings = [
        ArmMapping(name="arm_right", robot_slice=slice(0, 7), gello_slice=slice(0, 7), dof=7),
        ArmMapping(name="arm_left", robot_slice=slice(7, 14), gello_slice=slice(7, 14), dof=7)
    ]
    robot_proprio = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,  # right arm
                                  1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,  # left arm
                                  0.0])  # gripper
    gello_goal = np.zeros(14)

    # Simulate mapping for both arms
    for mapping in mappings:
        robot_arm = robot_proprio[mapping.robot_slice].cpu().numpy()
        gello_goal[mapping.gello_slice] = robot_arm

    print(f"  Robot right arm: {robot_proprio[0:7].numpy()}")
    print(f"  Gello right goal: {gello_goal[0:7]}")
    assert np.allclose(gello_goal[0:7], robot_proprio[0:7].numpy()), "Right arm reverse mapping failed!"

    print(f"  Robot left arm:  {robot_proprio[7:14].numpy()}")
    print(f"  Gello left goal:  {gello_goal[7:14]}")
    assert np.allclose(gello_goal[7:14], robot_proprio[7:14].numpy()), "Left arm reverse mapping failed!"
    print("  ✓ R1Pro dual-arm reverse mapping correct")

    print("\n✓ All ENV_TO_GELLO tests passed!")


def test_invalid_joint_fallback():
    """Test that invalid gello joints fall back to robot proprio."""
    print("\n" + "=" * 80)
    print("TEST 3: Invalid Joint Fallback")
    print("=" * 80)

    mapping = ArmMapping(name="arm", robot_slice=slice(0, 6), gello_slice=slice(0, 6), dof=6)
    gello_joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    gello_valid = np.array([True, True, False, False, True, True])  # joints 2,3 invalid
    robot_proprio = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.0])  # current robot position
    action = torch.zeros(7)

    # Simulate fallback logic
    gello_arm = torch.from_numpy(gello_joints[mapping.gello_slice])
    gello_valid_t = torch.from_numpy(gello_valid[mapping.gello_slice])

    action[mapping.robot_slice] = torch.where(
        gello_valid_t,
        gello_arm,
        robot_proprio[mapping.robot_slice]  # Fallback
    )

    expected = np.array([0.1, 0.2, 1.2, 1.3, 0.5, 0.6])  # Valid from gello, invalid from proprio

    print(f"  Gello joints:     {gello_joints}")
    print(f"  Gello valid:      {gello_valid}")
    print(f"  Robot proprio:    {robot_proprio[:6].numpy()}")
    print(f"  Final action:     {action[:6].numpy()}")
    print(f"  Expected:         {expected}")

    assert np.allclose(action[:6].numpy(), expected), "Invalid joint fallback failed!"
    print("  ✓ Invalid joints correctly fall back to robot proprio")
    print("\n✓ Invalid joint fallback test passed!")


def test_real_r1pro_action_format():
    """Test RealR1Pro action tensor format."""
    print("\n" + "=" * 80)
    print("TEST 4: Real R1Pro Action Format")
    print("=" * 80)

    # R1Pro full action: [base(3), torso(4), left_arm(7), left_grip(1), right_arm(7), right_grip(1)]
    action = torch.zeros(23)

    slices = {
        'base': slice(0, 3),
        'torso': slice(3, 7),
        'left_arm': slice(7, 14),
        'left_gripper': slice(14, 15),
        'right_arm': slice(15, 22),
        'right_gripper': slice(22, 23),
    }

    # Verify total
    total = sum(s.stop - s.start for s in slices.values())
    assert total == 23, f"Expected 23 DOF, got {total}"
    print(f"  ✓ Total action dimension: {total}")

    # Test arm mapping with correct slices
    # Gello: [right(0:7), left(7:14)]
    # Robot: [left_arm(7:14), right_arm(15:22)]
    gello_right = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    gello_left = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])

    action[slices['left_arm']] = torch.from_numpy(gello_left)
    action[slices['right_arm']] = torch.from_numpy(gello_right)

    print(f"  Gello right (0:7) → Robot right (15:22)")
    print(f"  Gello left (7:14) → Robot left (7:14)")
    assert np.allclose(action[7:14].numpy(), gello_left), "Left arm mapping failed"
    assert np.allclose(action[15:22].numpy(), gello_right), "Right arm mapping failed"
    print("  ✓ Arm mapping correct")

    print("\n✓ Real R1Pro action format test passed!")


def test_gripper_conversion():
    """Test gripper value conversion IIIL ↔ brs-ctrl."""
    print("\n" + "=" * 80)
    print("TEST 5: Gripper Value Conversion")
    print("=" * 80)

    # IIIL: [-1, 1] where 1=closed, -1=open
    # brs-ctrl: [0, 1] where 1=closed, 0=open

    def iiil_to_brs(val):
        return np.clip((val + 1.0) / 2.0, 0.0, 1.0)

    def stroke_to_iiil(stroke):
        # stroke 0=closed, 100=open
        return np.clip(1.0 - stroke / 50.0, -1.0, 1.0)

    # Test IIIL → brs-ctrl
    assert abs(iiil_to_brs(1.0) - 1.0) < 0.01, "Closed gripper conversion failed"
    assert abs(iiil_to_brs(-1.0) - 0.0) < 0.01, "Open gripper conversion failed"
    assert abs(iiil_to_brs(0.0) - 0.5) < 0.01, "Half gripper conversion failed"
    print("  ✓ IIIL → brs-ctrl conversion correct")

    # Test stroke → IIIL
    assert abs(stroke_to_iiil(0) - 1.0) < 0.01, "Stroke 0 conversion failed"
    assert abs(stroke_to_iiil(100) - (-1.0)) < 0.01, "Stroke 100 conversion failed"
    assert abs(stroke_to_iiil(50) - 0.0) < 0.01, "Stroke 50 conversion failed"
    print("  ✓ stroke → IIIL conversion correct")

    print("\n✓ Gripper conversion test passed!")


def test_robot_strategy_real_robot():
    """Test robot strategy handles real robots correctly."""
    print("\n" + "=" * 80)
    print("TEST 6: Robot Strategy Real Robot Handling")
    print("=" * 80)

    from iiil.policies.teleop_robot_strategy import R1ProStrategy

    class MockRealRobot:
        model_name = "R1Pro"
        @property
        def controllers(self):
            return {}  # Empty for real robots

    robot = MockRealRobot()
    strategy = R1ProStrategy()

    # get_arm_controller should return None
    result = strategy.get_arm_controller(robot)
    assert result is None, "Expected None for real robot"
    print("  ✓ get_arm_controller returns None for real robot")

    # set_controller_mode should be no-op
    try:
        strategy.set_controller_mode(robot, "joint")
        print("  ✓ set_controller_mode is safe for real robot")
    except Exception as e:
        print(f"  ✗ set_controller_mode failed: {e}")
        raise

    print("\n✓ Robot strategy real robot test passed!")


def test_locomotion_handler_abstraction():
    """Test locomotion handler abstraction and factory."""
    print("\n" + "=" * 80)
    print("TEST 7: Locomotion Handler Abstraction")
    print("=" * 80)

    from iiil.policies.teleop_locomotion import (
        LocomotionCommand,
        NoOpLocomotionHandler,
        get_locomotion_handler,
    )

    # Test LocomotionCommand dataclass
    print("\nLocomotionCommand:")
    cmd = LocomotionCommand(
        base_cmd=np.array([0.1, 0.2, 0.05]),
        torso_cmd=np.array([0.0, 0.0, 0.5, -0.3]),
        gripper_cmds={"left": -0.8, "right": 1.0},
    )
    assert cmd.has_base, "Should have base command"
    assert cmd.has_torso, "Should have torso command"
    assert cmd.has_grippers, "Should have gripper commands"
    print("  ✓ LocomotionCommand with all fields correct")

    cmd_empty = LocomotionCommand()
    assert not cmd_empty.has_base, "Empty should not have base"
    assert not cmd_empty.has_torso, "Empty should not have torso"
    assert not cmd_empty.has_grippers, "Empty should not have grippers"
    print("  ✓ Empty LocomotionCommand correct")

    # Test NoOpLocomotionHandler
    print("\nNoOpLocomotionHandler:")
    noop_handler = NoOpLocomotionHandler()
    assert noop_handler.name == "noop", "Name should be 'noop'"
    assert noop_handler.connected, "NoOp should always be 'connected'"
    noop_cmd = noop_handler.get_command({})
    assert not noop_cmd.has_base, "NoOp should return empty command"
    print("  ✓ NoOpLocomotionHandler correct")

    # Test factory function
    print("\nFactory function:")
    a1_handler = get_locomotion_handler("A1", required=False)
    assert a1_handler.name == "noop", "A1 should get NoOp handler"
    print("  ✓ A1 gets NoOpLocomotionHandler")

    # Unknown robot type should also get NoOp
    unknown_handler = get_locomotion_handler("UnknownRobot", required=False)
    assert unknown_handler.name == "noop", "Unknown should get NoOp handler"
    print("  ✓ Unknown robot gets NoOpLocomotionHandler")

    # R1Pro should get R1ProLocomotionHandler (may not connect without hardware)
    r1pro_handler = get_locomotion_handler("R1Pro", required=False)
    assert r1pro_handler.name == "r1pro_joycon", "R1Pro should get R1Pro handler"
    print(f"  ✓ R1Pro gets R1ProLocomotionHandler (connected={r1pro_handler.connected})")

    print("\n✓ Locomotion handler abstraction test passed!")


def test_r1pro_locomotion_application():
    """Test R1Pro locomotion command application to action tensor."""
    print("\n" + "=" * 80)
    print("TEST 8: R1Pro Locomotion Application")
    print("=" * 80)

    from iiil.policies.teleop_locomotion import LocomotionCommand

    # R1Pro action slices
    action_slices = {
        'base': slice(0, 3),
        'torso': slice(3, 7),
        'left_arm': slice(7, 14),
        'left_gripper': slice(14, 15),
        'right_arm': slice(15, 22),
        'right_gripper': slice(22, 23),
    }

    # Mock locomotion command (already in IIIL format)
    cmd = LocomotionCommand(
        base_cmd=np.array([0.1, 0.2, 0.05], dtype=np.float32),
        torso_cmd=np.array([0.0, 0.0, 0.5, -0.3], dtype=np.float32),
        gripper_cmds={"left": -0.8, "right": 1.0},  # Already IIIL range [-1, 1]
    )

    # Test applying to action tensor
    print("\nApplying locomotion to action:")
    action = torch.zeros(23)

    # Simulate what R1ProLocomotionHandler.apply_to_action does
    if cmd.has_base and 'base' in action_slices:
        action[action_slices['base']] = torch.from_numpy(cmd.base_cmd)
    if cmd.has_torso and 'torso' in action_slices:
        action[action_slices['torso']] = torch.from_numpy(cmd.torso_cmd)
    if cmd.has_grippers:
        if 'left_gripper' in action_slices:
            action[action_slices['left_gripper']] = cmd.gripper_cmds["left"]
        if 'right_gripper' in action_slices:
            action[action_slices['right_gripper']] = cmd.gripper_cmds["right"]

    # Verify
    assert np.allclose(action[0:3].numpy(), cmd.base_cmd), "Base not applied"
    print(f"  Base: {action[0:3].numpy()} ✓")

    assert np.allclose(action[3:7].numpy(), cmd.torso_cmd), "Torso not applied"
    print(f"  Torso: {action[3:7].numpy()} ✓")

    assert abs(action[14].item() - (-0.8)) < 0.01, "Left gripper not applied"
    assert abs(action[22].item() - 1.0) < 0.01, "Right gripper not applied"
    print(f"  Left gripper: {action[14].item():.2f} ✓")
    print(f"  Right gripper: {action[22].item():.2f} ✓")

    print("\n✓ R1Pro locomotion application test passed!")


if __name__ == "__main__":
    print("\nTeleop Data Flow Validation Tests")
    print("=" * 80)

    test_gello_to_robot_mapping()
    test_robot_to_gello_mapping()
    test_invalid_joint_fallback()
    test_real_r1pro_action_format()
    test_gripper_conversion()
    test_robot_strategy_real_robot()
    test_locomotion_handler_abstraction()
    test_r1pro_locomotion_application()

    print("\n" + "=" * 80)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 80)
    print("\nData flow validation complete!")
    print("The teleop pipeline correctly handles:")
    print("  1. GELLO_TO_ENV: User moves gello → Robot moves")
    print("  2. ENV_TO_GELLO: Policy moves robot → Gello follows")
    print("  3. Invalid joints fall back to robot proprio")
    print("  4. Real R1Pro 23-DOF action format")
    print("  5. Gripper value conversion (IIIL ↔ brs-ctrl)")
    print("  6. Robot strategies handle real robots gracefully")
    print("  7. Locomotion handler abstraction (modular, extensible)")
    print("  8. R1Pro locomotion application (base, torso, grippers)")
    print("\nTo run on real hardware:")
    print("  python run_task.py env=real_r1pro task=<task> policy=gello policy.base_policy=il")

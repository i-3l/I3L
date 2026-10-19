import numpy as np
import time
from types import SimpleNamespace
from unittest.mock import patch

from iiil.gello.controller_factory import A1GelloControllerFactory
from iiil.gello.compensation.gravity_compensation import GravityCompensator
from iiil.gello.compensation.compensation_config import GravityCompensationConfig
from iiil.policies.policy_base import PolicyMode
import torch
from omegaconf import OmegaConf
import pytest


class DummyDXLDriver:
    """Hardware-free driver stub with the same surface as UnifiedDXLDriver."""

    def __init__(self, ids, port, motor_type, baudrate, multithread_read_joints=False, initial_mode="current", current_limit=None):
        self.current_mode = initial_mode
        self._num = len(ids)
        self._traj = np.zeros(self._num, dtype=np.float32)
        self.closed = False

    def set_torque_mode(self, enable: bool):
        return None

    def switch_to_current_mode(self):
        self.current_mode = "current"

    def switch_to_position_mode(self):
        self.current_mode = "position"

    def set_controls(self, controls):
        self._last_controls = np.array(controls)

    def get_motor_state(self, **kwargs):
        return SimpleNamespace(
            joint_angles=np.zeros(self._num, dtype=np.float32),
            joint_velocities=np.zeros(self._num, dtype=np.float32),
            joint_currents=np.zeros(self._num, dtype=np.float32),
            joint_position_trajectory=self._traj.copy(),
            joint_velocity_trajectory=np.zeros(self._num, dtype=np.float32),
        )

    def close(self):
        self.closed = True


def test_controller_factory_loads_a1_and_r1pro_with_single_arm_urdfs():
    """End-to-end config load covering A1 and mirrored-URDF R1Pro paths without touching hardware."""
    # Reset factory between instantiations
    A1GelloControllerFactory._controller_instance = None
    A1GelloControllerFactory._current_config_name = None

    controllers = []
    patches = [
        patch("iiil.gello.drivers.unified_driver.UnifiedDXLDriver", DummyDXLDriver),
        patch("iiil.gello.controllers.bilateral_controller.UnifiedDXLDriver", DummyDXLDriver),
        patch("iiil.gello.controllers.concurrent_gello_controller.UnifiedDXLDriver", DummyDXLDriver),
    ]
    with patches[0], patches[1], patches[2]:
        try:
            # A1 single arm
            ctl_6 = A1GelloControllerFactory.get_gello_controller_instance("gello_a1")
            assert ctl_6.dof == 6
            ctl_6.start_control()
            ctl_6.set_new_goal(np.zeros(ctl_6.dof))
            time.sleep(0.05)
            controllers.append(ctl_6)

            # R1Pro bilateral mirroring a single-arm URDF
            A1GelloControllerFactory._controller_instance = None
            ctl_7 = A1GelloControllerFactory.get_gello_controller_instance("gello_r1pro")
            assert ctl_7.dof == 14
            ctl_7.start_control()
            ctl_7.set_new_goal(np.zeros(ctl_7.dof))
            time.sleep(0.05)
            controllers.append(ctl_7)

            # Verify the gravity compensator behavior for mirrored URDF
            arm_urdf_dof = 7
            config = GravityCompensationConfig(
                enable=True,
                urdf_path="assets/gello_r1pro_urdf/robot_7dof.urdf",
                multipliers=np.ones(ctl_7.dof, dtype=np.float32),
            )
            compensator = GravityCompensator(config, dof=ctl_7.dof)
            assert compensator._is_bilateral_single_arm_urdf
            assert compensator.model.nv == arm_urdf_dof
        finally:
            for c in controllers:
                c.close()
            A1GelloControllerFactory._controller_instance = None
            A1GelloControllerFactory._current_config_name = None


def _make_policy_config(gello_config: str, control_direction: str = "env_to_gello"):
    return OmegaConf.create({
        "type": "gello",
        "control_type": "joint",
        "gello_config": gello_config,
        "base_policy": "curobo",
        "control_direction": control_direction,
        "bilateral_blend_factor": 0.5,
        "joint_convergence_tolerance": 0.05,
        "max_convergence_wait_time": 0.1,
        "robot_type": {
            "type": "A1",
            "action_key_dims": {"arm": 6, "gripper": 1},
        },
    })


def _stub_base_policy(self, base_policy_type):
    action_dim = self._num_arm_joints + 1
    return SimpleNamespace(
        config=SimpleNamespace(control_type="joint"),
        predict=lambda **_: torch.ones(action_dim),
        close=lambda: None,
        reset=lambda: None,
        load=lambda **kwargs: None,
    )


class _JoyconStub:
    trigger_r = False
    trigger_zr = False
    button_x = False


class _ArmControllerStub:
    def __init__(self, mode):
        self.controller_mode = mode


def _attach_stub_arm_controller(policy):
    policy._arm_controller = _ArmControllerStub(policy._base_policy_control_type)


def _patch_drivers():
    return [
        patch("iiil.gello.drivers.unified_driver.UnifiedDXLDriver", DummyDXLDriver),
        patch("iiil.gello.controllers.bilateral_controller.UnifiedDXLDriver", DummyDXLDriver),
        patch("iiil.gello.controllers.concurrent_gello_controller.UnifiedDXLDriver", DummyDXLDriver),
    ]


def test_sim_loop_with_policy_and_mock_driver_end_to_end():
    """Simulates a few policy steps with real controller build but mocked drivers (no hardware)."""
    from iiil.policies import teleop_policy

    patches = _patch_drivers()
    with patches[0], patches[1], patches[2], \
         patch.object(teleop_policy.GelloPolicy, "_create_base_policy", autospec=True, side_effect=_stub_base_policy), \
         patch("iiil.policies.teleop_policy.JoyconController", new=_JoyconStub):

        # Fresh controller instance
        A1GelloControllerFactory._controller_instance = None
        A1GelloControllerFactory._current_config_name = None

        policy = teleop_policy.GelloPolicy(_make_policy_config("gello_a1", control_direction="env_to_gello"))
        _attach_stub_arm_controller(policy)
        policy._state.should_reset_gello_position = False  # avoid blocking wait in tests

        obs = {"robot::proprio": torch.tensor([0.2, 0.1, -0.1, -0.2, 0.0, 0.05, 0.5])}

        # Follower: robot leads, base policy action should flow through
        action_env_to_gello = policy.predict(obs)
        assert action_env_to_gello.shape[0] == policy._num_arm_joints + 1
        assert policy._gello_controller._has_started
        assert policy._gello_controller._goal is not None

        # Switch to leader mode and ensure goal following pauses and Gello drives the action
        policy.set_mode(PolicyMode.LEADING)
        policy._handle_control_direction()
        action_gello_to_env = policy.predict(obs)
        assert action_gello_to_env.shape == action_env_to_gello.shape
        assert policy._control_direction.name.lower() == "gello_to_env"
        # Gello joint state should dominate in leader mode
        gello_q, _ = policy._gello_controller.q
        assert action_gello_to_env[0].item() == pytest.approx(gello_q[0], rel=1e-3, abs=1e-3)

        # Back to follower for completeness
        policy.set_mode(PolicyMode.FOLLOWING)
        policy._handle_control_direction()
        action_back = policy.predict(obs)
        assert policy._gello_controller._should_follow_goal
        assert action_back.shape == action_env_to_gello.shape

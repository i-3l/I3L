import numpy as np
from pathlib import Path

from iiil.gello.compensation.compensation_config import GravityCompensationConfig
from iiil.gello.compensation.gravity_compensation import GravityCompensator


def _project_root() -> Path:
    # .../IIIL/tests/unit -> parents[2] is repo root
    return Path(__file__).resolve().parents[2]


def test_bilateral_uses_single_arm_urdf_and_mirrors_torques():
    """
    Validates that a bilateral system (14 DOF) can run gravity compensation
    using a single-arm URDF and that the mirrored arm torques are sign-flipped.
    """
    urdf_path = _project_root() / "assets/gello_r1pro_urdf/robot_7dof.urdf"
    assert urdf_path.exists(), "Single-arm URDF missing"

    config = GravityCompensationConfig(
        enable=True,
        urdf_path=str(urdf_path),
        multipliers=np.ones(14, dtype=np.float32),
    )
    compensator = GravityCompensator(config, dof=14)

    assert compensator._is_bilateral_single_arm_urdf  # mirrored mode enabled
    assert compensator.model.nv == 7  # URDF itself is single arm

    # Choose an asymmetric pose and mirror it for the left arm
    q_right = np.array([0.2, -0.1, 0.3, -0.25, 0.15, -0.05, 0.05], dtype=np.float32)
    q = np.concatenate([q_right, -q_right])

    torques = compensator.compute_torque(q)

    # Torques should mirror with opposite sign when positions are mirrored
    np.testing.assert_allclose(torques[:7], -torques[7:], atol=1e-5)


def test_single_arm_urdf_path_is_used_for_r1pro_config():
    """Smoke test that the configured URDF path resolves to the single-arm file."""
    import omegaconf
    from iiil.utils.misc_utils import get_package_root_path
    import os

    cfg_path = _project_root() / "iiil/configs/gello/gello_r1pro.yaml"
    config = omegaconf.OmegaConf.load(cfg_path)
    pkg_root = Path(get_package_root_path())
    rel = config.compensation.gravity.urdf_path
    primary = (pkg_root / rel).resolve()
    fallback = (pkg_root.parent / rel).resolve()

    resolved_urdf = primary if primary.exists() else fallback
    assert resolved_urdf.exists()
    assert resolved_urdf == fallback  # we expect the repo-level assets path

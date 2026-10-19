"""Robot joint names, metric names, and layout inference."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Joint name tables
# ---------------------------------------------------------------------------

A1_JOINT_NAMES = [f"J{i + 1}" for i in range(6)] + ["Gripper"]

FRANKA_JOINT_NAMES = [
    "J0 (shoulder pan)",
    "J1 (shoulder lift)",
    "J2 (elbow twist)",
    "J3 (elbow flex)",
    "J4 (wrist twist)",
    "J5 (wrist flex)",
    "J6 (wrist roll)",
    "Gripper",
]

REAL_R1PRO_JOINT_NAMES = [
    "x", "y", "rz", "t0", "t1", "t2", "t3",
    "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L_grip",
    "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R_grip",
]

# ---------------------------------------------------------------------------
# Detector metric names
# ---------------------------------------------------------------------------

METRIC_NAMES = ["cosine", "L1"]

# ---------------------------------------------------------------------------
# Default slice conventions
# ---------------------------------------------------------------------------

DEFAULT_ARM_SLICE = slice(0, 6)
DEFAULT_GRIPPER_SLICE = slice(6, 8)


# ---------------------------------------------------------------------------
# Joint layout inference
# ---------------------------------------------------------------------------

def infer_joint_layout(
    state_dim: int,
    action_dim: int,
    arm_slice: slice = DEFAULT_ARM_SLICE,
    gripper_slice: slice = DEFAULT_GRIPPER_SLICE,
) -> tuple[list[str], slice, slice, bool]:
    """Return (joint_names, primary_slice, secondary_slice, use_primary_direct)."""
    if state_dim == 46 and action_dim == 23:
        return REAL_R1PRO_JOINT_NAMES, slice(0, 23), slice(0, 23), True
    if action_dim == 7:
        return A1_JOINT_NAMES, arm_slice, gripper_slice, False
    if state_dim >= action_dim > 0:
        names = [f"J{i + 1}" for i in range(action_dim)]
        return names, slice(0, action_dim), slice(0, action_dim), True
    names = [f"J{i + 1}" for i in range(state_dim)]
    return names, slice(0, state_dim), slice(0, state_dim), True


def extract_joint_traces(
    state: np.ndarray,
    action_dim: int,
    arm_slice: slice = DEFAULT_ARM_SLICE,
    gripper_slice: slice = DEFAULT_GRIPPER_SLICE,
) -> tuple[np.ndarray, list[str]]:
    """Extract joint traces and names from proprioceptive state array."""
    joint_names, primary_slice, secondary_slice, use_primary_direct = infer_joint_layout(
        state_dim=state.shape[1],
        action_dim=action_dim,
        arm_slice=arm_slice,
        gripper_slice=gripper_slice,
    )
    if use_primary_direct:
        return state[:, primary_slice], joint_names

    arm = state[:, arm_slice]
    gripper = state[:, gripper_slice]
    if gripper.size == 0:
        grip = np.zeros((len(state), 1), dtype=state.dtype)
    elif gripper.ndim == 1 or gripper.shape[1] == 1:
        grip = gripper.reshape(len(state), 1)
    else:
        grip = np.mean(gripper, axis=1, keepdims=True)
    return np.concatenate([arm, grip], axis=1), joint_names

"""Unified episode data model and HDF5 loader.

Loads all available signals from IIIL HDF5 files into a single dataclass.
Every optional field defaults to ``None`` so the same structure works for
human demos, correction rollouts, and detector-instrumented SA runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from .constants import extract_joint_traces, DEFAULT_ARM_SLICE, DEFAULT_GRIPPER_SLICE


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EpisodeData:
    """One episode from an HDF5 dataset.

    Required fields: timestamps, joints, correction_mask.
    Everything else is populated when available in the HDF5.
    """

    name: str
    source: str
    timestamps: np.ndarray  # (T,) seconds

    # Joint state
    joints: np.ndarray      # (T, n_joints)
    joint_names: list[str]

    # Action / policy
    actions: np.ndarray | None = None             # (T, action_dim)
    base_action: np.ndarray | None = None         # (T, action_dim)
    oracle_action: np.ndarray | None = None       # (T, action_dim)
    action_delta: np.ndarray | None = None        # (T, action_dim)
    action_delta_mag: np.ndarray | None = None    # (T,)

    # Boolean masks
    correction_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    human_demo_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    # Intervention state (0=human, 1=pre-int, 2=active, 3=normal)
    int_state: np.ndarray | None = None

    # Detector signals
    detector_scores: np.ndarray | None = None      # (T, n_metrics)
    mahal_distance: np.ndarray | None = None        # (T,)

    # Full proprioceptive state (kept for custom analyses)
    proprio: np.ndarray | None = None               # (T, state_dim)

    @property
    def has_correction(self) -> bool:
        return bool(np.any(self.correction_mask))

    @property
    def is_human_demo(self) -> bool:
        return bool(np.all(self.human_demo_mask))

    @property
    def n_joints(self) -> int:
        return self.joints.shape[1]

    @property
    def n_steps(self) -> int:
        return len(self.timestamps)


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def sorted_demo_names(data_group: h5py.Group) -> list[str]:
    def _key(name: str) -> tuple[int, str]:
        try:
            return (int(name.split("_")[-1]), name)
        except ValueError:
            return (10**9, name)
    return sorted(data_group.keys(), key=_key)


def sanitize_timestamps(
    timestamps: np.ndarray,
    fallback_fps: float | None = None,
) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        return ts
    if np.all(np.isfinite(ts)) and np.all(np.diff(ts) >= -1e-9):
        return ts - ts[0]
    if fallback_fps and fallback_fps > 0:
        return np.arange(len(ts), dtype=np.float64) / fallback_fps
    return np.arange(len(ts), dtype=np.float64)


# ---------------------------------------------------------------------------
# Single-demo extraction
# ---------------------------------------------------------------------------

def _load_demo_from_group(
    demo_grp: h5py.Group,
    source: str,
    arm_slice: slice,
    gripper_slice: slice,
    allow_approx_action_delta: bool,
) -> EpisodeData:
    # Timestamps
    if "time" in demo_grp and "timestamp_ms" in demo_grp["time"]:
        timestamps = demo_grp["time"]["timestamp_ms"][:].astype(np.float64).reshape(-1) / 1000.0
    else:
        timestamps = np.arange(len(demo_grp["action"]), dtype=np.float64)
    timestamps = sanitize_timestamps(timestamps)

    # State & actions
    proprio = demo_grp["obs"]["robot::proprio"][:]
    actions = demo_grp["action"][:]

    # Joint traces
    joints, joint_names = extract_joint_traces(
        proprio,
        action_dim=actions.shape[1],
        arm_slice=arm_slice,
        gripper_slice=gripper_slice,
    )

    # Intervention masks
    policy_grp = demo_grp.get("policy")
    int_state = None
    correction_mask = np.zeros(len(actions), dtype=bool)
    human_demo_mask = np.zeros(len(actions), dtype=bool)

    if policy_grp is not None and "int_state" in policy_grp:
        int_state = policy_grp["int_state"][:].reshape(-1).astype(np.int32)
        correction_mask = int_state == 2
        human_demo_mask = int_state == 0
    elif policy_grp is not None and "is_oracle_active" in policy_grp:
        oracle = policy_grp["is_oracle_active"][:].reshape(-1).astype(bool)
        correction_mask = oracle
        human_demo_mask = np.zeros_like(correction_mask)
    else:
        human_demo_mask = np.ones(len(actions), dtype=bool)

    # Base / oracle actions
    base_action = None
    oracle_action = None
    if policy_grp is not None:
        for key in ("base_action", "policy_action", "base_policy_action"):
            if key in policy_grp:
                base_action = policy_grp[key][:]
                break
        if "oracle_action" in policy_grp:
            oracle_action = policy_grp["oracle_action"][:]

    # Action delta
    if base_action is not None:
        action_delta = actions - base_action
    elif allow_approx_action_delta:
        action_delta = np.diff(actions, axis=0, prepend=actions[:1])
    else:
        action_delta = None

    action_delta_mag = np.linalg.norm(action_delta, axis=1) if action_delta is not None else None

    # Detector scores
    detector_scores = None
    if policy_grp is not None and "detector_interrupt_scores" in policy_grp:
        raw = policy_grp["detector_interrupt_scores"][:]
        if raw.shape[0] == len(timestamps):
            detector_scores = raw

    # Mahalanobis distance
    mahal_distance = None
    if policy_grp is not None and "mahalanobis_distance" in policy_grp:
        mahal_distance = policy_grp["mahalanobis_distance"][:].reshape(-1)

    return EpisodeData(
        name=demo_grp.name.split("/")[-1],
        source=source,
        timestamps=timestamps,
        joints=joints,
        joint_names=joint_names,
        actions=actions,
        base_action=base_action,
        oracle_action=oracle_action,
        action_delta=action_delta,
        action_delta_mag=action_delta_mag,
        correction_mask=correction_mask,
        human_demo_mask=human_demo_mask,
        int_state=int_state,
        detector_scores=detector_scores,
        mahal_distance=mahal_distance,
        proprio=proprio,
    )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_episodes(
    path: str | Path,
    *,
    arm_slice: slice = DEFAULT_ARM_SLICE,
    gripper_slice: slice = DEFAULT_GRIPPER_SLICE,
    allow_approx_action_delta: bool = True,
    max_episodes: int | None = None,
) -> list[EpisodeData]:
    """Load episodes from one or more HDF5 files.

    *path* can be a single ``.hdf5`` file or a directory (searched recursively).
    """
    path = Path(path)
    if path.is_file():
        hdf5_paths = [path]
    else:
        hdf5_paths = sorted(path.rglob("*.hdf5"))

    episodes: list[EpisodeData] = []
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as f:
            data = f.get("data")
            if data is None:
                continue
            for demo_name in sorted_demo_names(data):
                ep = _load_demo_from_group(
                    data[demo_name],
                    source=str(hdf5_path),
                    arm_slice=arm_slice,
                    gripper_slice=gripper_slice,
                    allow_approx_action_delta=allow_approx_action_delta,
                )
                if ep.timestamps.size >= 2:
                    episodes.append(ep)
                if max_episodes is not None and len(episodes) >= max_episodes:
                    return episodes
    return episodes

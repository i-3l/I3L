"""Datasets for binary tarjan correction policy training."""

import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from iiil.datas.dataset import IIILDataset

logger = logging.getLogger(__name__)

CONTEXT_STEPS = 16


class InterventionSegmentDataset(IIILDataset):
    """Dataset that only yields timesteps within intervention segments (int_state==2)
    plus a configurable context window around them.

    Actions targets are replaced with oracle_action for intervention timesteps,
    keeping the original (base) action for the context portion.
    """

    def _preload_demo(self, demo_key: Any) -> Dict[str, Any]:
        demo = super()._preload_demo(demo_key)

        if "int_state" not in demo.get("policy", {}):
            return demo

        int_state = demo["policy"]["int_state"]
        oracle_action = demo["policy"].get("oracle_action")
        if oracle_action is None:
            return demo

        # Replace action targets with oracle_action for int_state==2 timesteps
        # The actions dict has keys like 'arm', 'gripper' with shape (T, dim)
        # oracle_action is (T, 7) already normalized
        from iiil.datas.dataset import ACTION_QPOS_INDICES
        robot_type = self._robot_type

        for joint_key, indices in ACTION_QPOS_INDICES[robot_type].items():
            if joint_key in demo["actions"]:
                mask = (int_state == 2)
                if mask.any():
                    demo["actions"][joint_key][mask] = oracle_action[mask][:, indices]

        return demo

    def _compute_valid_indices(self) -> np.ndarray:
        """Compute indices that fall within intervention segments + context."""
        valid_indices = []

        for demo_id, demo in enumerate(self._all_demos):
            if "int_state" not in demo.get("policy", {}):
                continue

            int_state = demo["policy"]["int_state"].numpy()
            T = len(int_state)

            # Find contiguous intervention segments (int_state==2)
            is_intervention = (int_state == 2)
            if not is_intervention.any():
                continue

            # Find segment boundaries
            diff = np.diff(is_intervention.astype(int), prepend=0, append=0)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            # Expand each segment by context_steps and collect valid timesteps
            valid_mask = np.zeros(T, dtype=bool)
            for s, e in zip(starts, ends):
                seg_start = max(0, s - self._context_steps)
                seg_end = min(T, e + self._context_steps)
                valid_mask[seg_start:seg_end] = True

            # Collect valid (idx, demo_id) pairs
            for idx in np.where(valid_mask)[0]:
                valid_indices.append((idx, demo_id))

        return np.array(valid_indices) if valid_indices else np.empty((0, 2), dtype=int)

    def __init__(self, *args, context_steps: int = CONTEXT_STEPS, **kwargs):
        self._context_steps = context_steps
        super().__init__(*args, **kwargs)
        # After parent __init__ builds _demo_indices, filter to intervention segments
        filtered = self._compute_valid_indices()
        if len(filtered) > 0:
            # Intersect with existing _demo_indices (respects demo_slices)
            existing = set(map(tuple, self._demo_indices.tolist()))
            filtered = np.array([x for x in filtered if tuple(x) in existing])

        orig_len = len(self._demo_indices)
        self._demo_indices = filtered
        logger.info(
            f"InterventionSegmentDataset: {len(self._demo_indices)}/{orig_len} "
            f"timesteps retained (context={self._context_steps})"
        )


class UpweightedSpecialistDataset(IIILDataset):
    """Dataset combining HG-DAgger filtering with temporal upweighting.

    - Base demos (int_state==0): ALL timesteps kept, temporally upweighted
    - Correction demos: only int_state==0 and int_state==2 kept (HG-DAgger filtering),
      temporally upweighted. Action targets replaced with oracle_action for int_state==2.

    Temporal weighting uses a precomputed intervention density histogram.
    """

    def __init__(self, *args, intervention_density_path: str = "", **kwargs):
        self._intervention_density_path = intervention_density_path
        super().__init__(*args, **kwargs)

        # Load density weights
        if intervention_density_path and intervention_density_path != "":
            self._density_weights = np.load(intervention_density_path)
        else:
            self._density_weights = np.ones(50)
        self._n_bins = len(self._density_weights)

        # Filter correction demo timesteps: drop int_state==1 and int_state==3 (HG-DAgger)
        filtered_indices = []
        for i, (idx, demo_id) in enumerate(self._demo_indices):
            demo = self._all_demos[demo_id]
            int_state = demo.get("policy", {}).get("int_state")
            if int_state is None:
                # Base demo (no int_state) — keep all
                filtered_indices.append(i)
            elif int_state[idx] in (0, 2):
                # Keep human demo and oracle-active timesteps
                filtered_indices.append(i)
            # Drop int_state==1 (pre-intervention) and int_state==3 (on-policy)

        orig_len = len(self._demo_indices)
        self._demo_indices = self._demo_indices[filtered_indices]

        # Compute per-sample temporal weights
        self._sample_weights = np.ones(len(self._demo_indices), dtype=np.float64)
        for i, (idx, demo_id) in enumerate(self._demo_indices):
            demo = self._all_demos[demo_id]
            if isinstance(demo["actions"], dict):
                T = next(iter(demo["actions"].values())).shape[0]
            else:
                T = len(demo["actions"])
            normalized_t = idx / max(T - 1, 1)
            bin_idx = min(int(normalized_t * self._n_bins), self._n_bins - 1)
            self._sample_weights[i] = self._density_weights[bin_idx]

        # Normalize to probability distribution
        self._sample_probs = self._sample_weights / self._sample_weights.sum()

        logger.info(
            f"UpweightedSpecialistDataset: {len(self._demo_indices)}/{orig_len} samples "
            f"(after HG-DAgger filtering), "
            f"weight range [{self._sample_weights.min():.2f}, {self._sample_weights.max():.2f}], "
            f"effective ratio {self._sample_weights.max()/self._sample_weights.min():.1f}x"
        )

    def _preload_demo(self, demo_key: Any) -> Dict[str, Any]:
        demo = super()._preload_demo(demo_key)

        # For correction demos: replace actions with oracle_action during int_state==2
        if "int_state" not in demo.get("policy", {}):
            return demo

        int_state = demo["policy"]["int_state"]
        oracle_action = demo["policy"].get("oracle_action")
        if oracle_action is None:
            return demo

        from iiil.datas.dataset import ACTION_QPOS_INDICES
        robot_type = self._robot_type

        mask = (int_state == 2)
        if mask.any():
            for joint_key, indices in ACTION_QPOS_INDICES[robot_type].items():
                if joint_key in demo["actions"]:
                    demo["actions"][joint_key][mask] = oracle_action[mask][:, indices]

        return demo

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Weighted random sampling based on intervention density
        sampled_idx = np.random.choice(len(self._demo_indices), p=self._sample_probs)
        return super().__getitem__(sampled_idx)

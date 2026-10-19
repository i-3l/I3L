#!/usr/bin/env python3
"""
Script to augment HDF5 files with intervention state information.

This script does the following data processing step:

1. For real data, "center crop" the observations to 240p resolution.
2. Optionally remove any part of the data where robot is not moving (norm of action is smaller than a threshold).
3. Adds an 'int_state' field to HDF5 files based on oracle activity:
    - 0: Human demonstration (is_human_demo=True)
    - 1: Pre-intervention (oracle inactive but within pre_intervention_length of next intervention)
    - 2: Oracle active (intervention happening)
    - 3: Normal operation (oracle inactive and not near intervention)
"""

import argparse
import h5py
import numpy as np
from iiil.utils.obs_utils import crop_camera_obs
from pathlib import Path


def _is_target_resolution(obs_data: np.ndarray, target_res: tuple[int, int]) -> bool:
    """Return whether image observations already have the requested HxW."""
    if obs_data.ndim < 3:
        return False
    h, w = obs_data.shape[-3], obs_data.shape[-2]
    return (h, w) == target_res


def crop_real_obs(demo_group: h5py.Group, target_res: tuple[int, int] = (240, 240)) -> None:
    """
    Crop RGB observations in place and store back to HDF5.
    
    obs are in demo_group["obs"]["robot::robot:<camera>::rgb"]
    we want to center crop with offset along width:
        offsets = {'left_realsense_link': 20, 'right_realsense_link': 20, 'zed_link': 0}
    """
    if "obs" not in demo_group:
        raise ValueError("'obs' field not found in demo")
    
    obs_group = demo_group["obs"]
    
    # Find all camera obs keys (pattern: robot::robot:<camera>::rgb)
    for key in list(obs_group.keys()):
        if "rgb" in key:
            obs_data = obs_group[key][:]
            if _is_target_resolution(obs_data, target_res):
                print(f"    Skipping crop for {key}: already {target_res[0]}x{target_res[1]}")
                continue
            cropped = crop_camera_obs(obs_data, key, target_res=target_res)
            # Delete old dataset and write cropped data back to HDF5
            del obs_group[key]
            obs_group.create_dataset(key, data=cropped, dtype=obs_data.dtype)


def remove_no_motion(demo_group: h5py.Group, window: int=5, action_norm_threshold: float=0.003) -> None:
    """
    Remove timesteps where robot is not moving (action norm is below threshold).
    Uses a sliding window to smooth the decision. Filters all timestep-indexed 
    datasets in place and stores back to HDF5.
    
    Args:
        demo_group: HDF5 group representing a single demo
        window: Sliding window size for smoothing motion detection
        action_norm_threshold: Threshold for action norm to consider as motion
    """
    if "action" not in demo_group:
        raise ValueError("'action' field not found in demo")
    
    action_data = demo_group["action"][:]  # Shape: (seq_len, action_dim)
    delta_action = np.diff(action_data, axis=0, prepend=action_data[0:1, :])
    seq_len = len(action_data)
    
    # Compute action norms for each timestep
    action_norms = np.linalg.norm(delta_action, axis=-1)  # Shape: (seq_len,)
    
    # Apply sliding window to smooth: a timestep has motion if any step in window has significant motion
    motion_mask = np.zeros(seq_len, dtype=bool)
    for i in range(seq_len):
        window_start = max(0, i - window // 2)
        window_end = min(seq_len, i + window // 2 + 1)
        if np.max(action_norms[window_start:window_end]) > action_norm_threshold:
            motion_mask[i] = True
    
    # Get indices where motion is detected
    filtered_indices = np.where(motion_mask)[0]
    # print out number of timesteps removed
    print(f"    Removed {seq_len - len(filtered_indices)} / {seq_len} timesteps with no motion")
    
    # Filter top-level datasets
    for key in list(demo_group.keys()):
        if key != "obs" and key != "policy":
            try:
                data = demo_group[key][:]
                if len(data) == seq_len:  # Only filter if it's timestep-indexed
                    filtered_data = data[filtered_indices]
                    del demo_group[key]
                    demo_group.create_dataset(key, data=filtered_data, dtype=data.dtype)
            except Exception as e:
                pass  # Skip if filtering fails
    
    # Filter obs group datasets
    if "obs" in demo_group:
        obs_group = demo_group["obs"]
        for key in list(obs_group.keys()):
            try:
                data = obs_group[key][:]
                if len(data) == seq_len:  # Only filter if it's timestep-indexed
                    filtered_data = data[filtered_indices]
                    del obs_group[key]
                    obs_group.create_dataset(key, data=filtered_data, dtype=data.dtype)
            except Exception as e:
                pass  # Skip if filtering fails
    
    # Filter policy group datasets
    if "policy" in demo_group:
        policy_group = demo_group["policy"]
        for key in list(policy_group.keys()):
            try:
                data = policy_group[key][:]
                if len(data) == seq_len:  # Only filter if it's timestep-indexed
                    filtered_data = data[filtered_indices]
                    del policy_group[key]
                    policy_group.create_dataset(key, data=filtered_data, dtype=data.dtype)
            except Exception as e:
                pass  # Skip if filtering fails


def compute_int_state(is_oracle_active: np.ndarray, is_human_demo: bool, pre_intervention_length: int) -> np.ndarray:
    """
    Compute intervention state based on oracle activity.
    
    Args:
        is_oracle_active: Array of boolean values indicating oracle activity
        is_human_demo: Boolean flag indicating if this is a human demonstration
        pre_intervention_length: Number of steps before intervention to mark as pre-intervention
        
    Returns:
        Array of intervention states (0, 1, 2, or 3)
    """
    n = len(is_oracle_active)
    int_state = np.zeros((n, 1), dtype=np.uint8)
    
    # Case 0: Human demonstration
    if is_human_demo:
        return int_state  # All zeros
    
    # Process each timestep
    for i in range(n):
        if is_oracle_active[i]:
            # Case 2: Oracle is currently active
            int_state[i, 0] = 2
        else:
            # Oracle is not active - check if we're near the next intervention
            # Look ahead to find the next intervention
            near_intervention = False
            for j in range(min(i + pre_intervention_length, n - 1), i, -1):
                if is_oracle_active[j]:
                    near_intervention = True
                    break
            
            if near_intervention:
                # Case 1: Pre-intervention (within pre_intervention_length of next intervention)
                int_state[i, 0] = 1
            else:
                # Case 3: Normal operation (not near any intervention)
                int_state[i, 0] = 3
    
    return int_state


def augment_demo(demo_group, is_human_demo: bool, pre_intervention_length: int, demo_name: str=""):
    """
    Augment a single demo with intervention state information.
    
    Args:
        demo_group: HDF5 group representing a single demo
        is_human_demo: Boolean flag indicating if this is a human demonstration
        pre_intervention_length: Number of steps before intervention to mark as pre-intervention
        demo_name: Name of the demo for logging purposes
    """
    print(f"  Processing {demo_name}")
    
    # Check if required field exists
    if "policy" not in demo_group:
        if is_human_demo:
            print(f"    'policy' group not found - creating for human demo")
            demo_group.create_group("policy")
        else:
            raise ValueError(f"'policy' group not found in {demo_name}")
    
    # Handle is_oracle_active field
    if "is_oracle_active" not in demo_group["policy"]:
        if is_human_demo:
            # For human demos, create is_oracle_active with all True values
            if "action" not in demo_group:
                raise ValueError(f"'action' field not found in {demo_name} to determine sequence length")
            seq_length = len(demo_group["action"])
            is_oracle_active = np.ones(seq_length, dtype=bool)
            print(f"    'is_oracle_active' not found - creating with all True values for human demo")
        else:
            raise ValueError(f"'is_oracle_active' field not found in {demo_name}/policy")
    else:
        # Read existing is_oracle_active
        is_oracle_active = demo_group["policy"]["is_oracle_active"][:]
    
    print(f"    Found {len(is_oracle_active)} timesteps")
    print(f"    Oracle active for {np.sum(is_oracle_active)} timesteps")
    
    # Compute int_state
    int_state = compute_int_state(is_oracle_active, is_human_demo, pre_intervention_length)
    
    # Count distribution
    counts = {
        0: np.sum(int_state[:, 0] == 0),
        1: np.sum(int_state[:, 0] == 1),
        2: np.sum(int_state[:, 0] == 2),
        3: np.sum(int_state[:, 0] == 3),
    }
    print(f"    Intervention state distribution:")
    print(f"      State 0 (human demo): {counts[0]}")
    print(f"      State 1 (pre-intervention): {counts[1]}")
    print(f"      State 2 (oracle active): {counts[2]}")
    print(f"      State 3 (normal operation): {counts[3]}")
    
    # Write int_state to file (delete if exists)
    if "int_state" in demo_group["policy"]:
        print(f"    Deleting existing 'int_state' field")
        del demo_group["policy"]["int_state"]
    
    demo_group["policy"].create_dataset("int_state", data=int_state, dtype=np.uint8)
    print(f"    Successfully wrote 'int_state' field")
    
    # If human demo, copy action to base_action and oracle_action
    if is_human_demo:
        if "action" not in demo_group:
            raise ValueError(f"'action' field not found in {demo_name}")
        
        action_data = demo_group["action"][:]
        print(f"    Human demo detected - copying action data to base_action and oracle_action")
        
        # Create or overwrite base_action
        if "base_action" in demo_group["policy"]:
            print(f"      Deleting existing 'base_action' field")
            del demo_group["policy"]["base_action"]
        demo_group["policy"].create_dataset("base_action", data=action_data)
        
        # Create or overwrite oracle_action
        if "oracle_action" in demo_group["policy"]:
            print(f"      Deleting existing 'oracle_action' field")
            del demo_group["policy"]["oracle_action"]
        demo_group["policy"].create_dataset("oracle_action", data=action_data)
        
        print(f"    Successfully wrote 'base_action' and 'oracle_action' fields")


def process_hdf5_file(file_path: str, is_human_demo: bool, pre_intervention_length: int, crop_images: bool, remove_motion: bool) -> None:
    """Augment HDF5 file with intervention states and optionally crop/filter."""
    print(f"Processing file: {file_path}")
    
    with h5py.File(file_path, "r+") as f:
        if "data" not in f:
            raise ValueError(f"'data' group not found in {file_path}")
        
        demo_keys = [key for key in f["data"].keys() if key.startswith("demo_")]
        
        if not demo_keys:
            raise ValueError(f"No 'demo_xxx' groups found in {file_path}/data")
        
        print(f"Found {len(demo_keys)} demos to process")
        
        for demo_key in sorted(demo_keys):
            if crop_images:
                try:
                    crop_real_obs(f["data"][demo_key])
                    print(f"  Cropped images for {demo_key}")
                except Exception as e:
                    print(f"  Warning: Failed to crop images for {demo_key}: {e}")
            
            if remove_motion:
                try:
                    remove_no_motion(f["data"][demo_key])
                    print(f"  Removed no-motion timesteps for {demo_key}")
                except Exception as e:
                    print(f"  Warning: Failed to filter motion for {demo_key}: {e}")
            
            augment_demo(f["data"][demo_key], is_human_demo, pre_intervention_length, demo_key)


def main():
    parser = argparse.ArgumentParser(
        description="Augment HDF5 files with intervention state information"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the HDF5 file to process"
    )
    parser.add_argument(
        "-d",
        "--is-human-demo",
        action="store_true",
        help="Flag indicating this is a human demonstration (all states will be 0)"
    )
    parser.add_argument(
        "-l",
        "--pre-intervention-length",
        type=int,
        default=15,
        help="Number of steps before intervention to mark as pre-intervention (default: 15)"
    )
    parser.add_argument(
        "-c",
        "--crop-images",
        action="store_true",
        help="Center crop camera observations to 240p resolution"
    )
    parser.add_argument(
        "-r",
        "--remove-motion",
        action="store_true",
        help="Remove timesteps where robot is not moving (action norm below threshold)"
    )
    
    args = parser.parse_args()
    
    file_path = Path(args.input)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.suffix == ".hdf5":
        if not file_path.is_dir():
            raise ValueError(f"Input path must be an HDF5 file or a directory containing HDF5 files: {file_path}")
        for h5_file in file_path.glob("*.hdf5"):
            process_hdf5_file(h5_file, args.is_human_demo, args.pre_intervention_length, args.crop_images, args.remove_motion)
    else:
        process_hdf5_file(file_path, args.is_human_demo, args.pre_intervention_length, args.crop_images, args.remove_motion)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

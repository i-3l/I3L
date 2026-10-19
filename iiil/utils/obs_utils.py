import cv2
import numpy as np
import torch


def crop_camera_obs(obs_data: np.ndarray, obs_key: str, target_res=(240, 240)):
    offsets = {"left_realsense_link": 20, "right_realsense_link": 20, "zed_link": 0, "external_camera": 0, "wrist_camera": 0}
    # Find all camera obs keys (pattern: robot::robot:<camera>::rgb)
    batched = True
    is_torch = False
    if isinstance(obs_data, torch.Tensor):
        is_torch = True
        if obs_data.ndim == 3:
            batched = False
            obs_data = obs_data.unsqueeze(0)
        seq_len, h, w, c = obs_data.shape
    else:
        if obs_data.ndim == 3:
            batched = False
            obs_data = obs_data[np.newaxis, ...]  # Add seq_len dim if missing
        seq_len, h, w, c = obs_data.shape

    target_h, target_w = target_res
    if (h, w) == (target_h, target_w):
        return obs_data[0] if not batched else obs_data
    
    # Determine which camera this is
    camera_name = None
    for cam_key in offsets.keys():
        if cam_key in obs_key:
            camera_name = cam_key
            break
    
    if camera_name is None:
        return obs_data
    
    offset = offsets[camera_name]
    
    # Special handling for zed_link: crop to 360p then downsample to 240p
    if camera_name == "zed_link":
        # First center crop to 360x360
        intermediate_res = 360
        start_h = (h - intermediate_res) // 2
        start_w = (w - intermediate_res) // 2 + offset
        cropped_360 = obs_data[:, start_h:start_h + intermediate_res, start_w:start_w + intermediate_res, :]
        # Then downsample to 240x240
        if is_torch:
            # Convert torch dtype to numpy dtype for cv2 and np.zeros
            torch_to_np_dtype = {
                torch.uint8: np.uint8,
                torch.float32: np.float32,
                torch.float64: np.float64,
                torch.int32: np.int32,
                torch.int64: np.int64,
            }
            np_dtype = torch_to_np_dtype.get(obs_data.dtype, np.float32)
            cropped = np.zeros((seq_len, target_h, target_w, c), dtype=np_dtype)
            for i in range(seq_len):
                arr = cropped_360[i].cpu().numpy() if cropped_360[i].is_cuda else cropped_360[i].numpy()
                cropped[i] = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            cropped = torch.from_numpy(cropped).to(obs_data.dtype)
        else:
            cropped = np.zeros((seq_len, target_h, target_w, c), dtype=obs_data.dtype)
            for i in range(seq_len):
                cropped[i] = cv2.resize(cropped_360[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        # For other cameras, directly crop to target resolution
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2 + offset
        cropped = obs_data[:, start_h:start_h + target_h, start_w:start_w + target_w, :]
        if is_torch:
            cropped = cropped.clone()  # Ensure contiguous
    if not batched:
        cropped = cropped[0]  # Remove seq_len dim if it was added
    return cropped

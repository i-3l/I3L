import h5py
import numpy as np
import os
import random
import torch
import torch.distributed as dist
import torch.nn.functional as F
from copy import deepcopy
from iiil.utils.misc_utils import REWEIGHTING_STRATEGIES
from omegaconf import ListConfig, OmegaConf
from omnigibson.learning.datas.iterable_dataset import BehaviorIterableDataset
from omnigibson.learning.utils.array_tensor_utils import (
    any_concat,
    any_slice,
    any_stack,
    any_ones_like,
    get_batch_size,
    sequential_sum_balanced_partitioning,
)
from omnigibson.learning.utils.obs_utils import MIN_DEPTH, MAX_DEPTH
from omnigibson.learning.utils.eval_utils import (
    ACTION_QPOS_INDICES,
    JOINT_RANGE,
    PROPRIO_QPOS_INDICES,
    PROPRIOCEPTION_INDICES,
    EEF_POSITION_RANGE,
)
from omnigibson.utils.ui_utils import create_module_logger
from torch.utils.data import Dataset, get_worker_info
from tqdm import tqdm
from typing import Any, Dict, Generator, List, Optional

logger = create_module_logger("IIILDataset")
logger.setLevel("INFO")

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    LeRobotDataset = None


class IIILDataset(Dataset):
    """
    Dataset for IIIL imitation learning
    """
    
    @classmethod
    def get_all_demo_keys(cls, data_path: str, _: Optional[Any] = None) -> List[h5py.Dataset]:
        demo_keys = []
        assert os.path.exists(data_path), f"Data path does not exist!"
        for file_name in tqdm(sorted(os.listdir(data_path))):
            if file_name.endswith(".hdf5"):
                f = h5py.File(os.path.join(data_path, file_name), "r", swmr=True, libver='latest')
                for demo_name in f["data"]:
                    demo_keys.append(f["data"][demo_name])
        return demo_keys
    
    def __init__(
        self,
        *args,
        data_path: str,
        demo_keys: List[Any],
        robot_type: str = "R1Pro",
        obs_window_size: int,
        ctx_len: int,
        use_action_chunks: bool = False,
        action_prediction_horizon: Optional[int] = None,
        downsample_factor: int = 1,
        visual_obs_types: List[str],
        multi_view_cameras: Optional[Dict[str, Any]] = None,
        load_visual_obs_to_memory: bool = False,
        rank_based_partitioning: bool = False,
        use_task_info: bool = False,
        task_info_range: Optional[ListConfig] = None,
        seed: int = 42,
        shuffle: bool = True,
        action_history_len: int = 1,
        **kwargs,
    ) -> None:
        """
        Args:
            data_path (str): Path to dataset
            task_name (str): Name of the task
            downsample_factor (int): Downsample factor for the dataset
            obs_window_size (int): Number of observations to return in each data point
            ctx_len (int): Context length for action chunks
            use_action_chunks (bool): Whether to use action chunks
            visual_obs_types (List[str]): List of visual observation types to load
            multi_view_cameras (Dict[str, Dict[str, Any]]): Multi-view camera configurations
            load_visual_obs_to_memory (bool): Whether to load visual observations into memory
            rank_based_partitioning (bool): Whether to partition data based on rank
            use_task_info (bool): Whether to use task info in the dataset
            task_info_range (Optional[ListConfig]): Range for normalizing task info
            seed (int): Random seed for shuffling
            shuffle (bool): Whether to shuffle the dataset
        """
        super().__init__()
        self._data_path = data_path
        self._demo_keys = demo_keys
        self._robot_type = robot_type
        self._obs_window_size = obs_window_size
        self._ctx_len = ctx_len
        self._use_action_chunks = use_action_chunks
        self._action_prediction_horizon = action_prediction_horizon
        assert (
            self._action_prediction_horizon is not None if self._use_action_chunks else True
        ), "action_prediction_horizon must be provided if use_action_chunks is True!"
        self._downsample_factor = downsample_factor
        assert self._downsample_factor >= 1, "downsample_factor must be >= 1!"
        self._use_task_info = use_task_info
        self._task_info_range = (
            torch.tensor(OmegaConf.to_container(task_info_range)) if task_info_range is not None else None
        )
        self._seed = seed
        self._shuffle = shuffle
        self._load_visual_obs_to_memory = load_visual_obs_to_memory
        self._action_history_len = action_history_len

        assert set(visual_obs_types).issubset(
            {"rgb", "depth_linear", "seg_instance_id", "pcd"}
        ), "visual_obs_types must be a subset of {'rgb', 'depth_linear', 'seg_instance_id', 'pcd'}!"
        self._visual_obs_types = set(visual_obs_types)

        self._multi_view_cameras = multi_view_cameras

        # Pre-extract low-dim data for all demos
        self._all_demos = [self._preload_demo(demo_key) for demo_key in self._demo_keys]

        # get demo lengths (N_chunks)
        self._demo_lengths = []
        for demo in self._all_demos:
            L = get_batch_size(demo, strict=True)
            assert L >= self._obs_window_size >= 1
            self._demo_lengths.append(L - self._obs_window_size + 1)

        # We keep track of which slices of demos to use
        self._demo_slices = []

        # partition data based on rank if needed
        if rank_based_partitioning:
            global_worker_id, total_global_workers = self._get_global_worker_id()
            start_demo_id, start_demo_idx, end_demo_id, end_demo_idx = sequential_sum_balanced_partitioning(
                self._demo_lengths, total_global_workers, global_worker_id
            )
            for demo_idx, demo_length in enumerate(self._demo_lengths):
                if demo_idx < start_demo_id or demo_idx > end_demo_id:
                    self._demo_slices.append((0, 1 - self._obs_window_size))  # empty slices
                else:
                    start_idx = start_demo_idx if demo_idx == start_demo_id else 0
                    end_idx = end_demo_idx if demo_idx == end_demo_id else demo_length
                    self._demo_slices.append((start_idx, end_idx))
        else:
            # We use the full length of the demos
            for demo_length in self._demo_lengths:
                self._demo_slices.append((0, demo_length))

        if self._load_visual_obs_to_memory:
            # We first truncate the preloaded demos to only keep those within the demo_slices
            self._all_demos = [any_slice(
                demo, 
                np.s_[start_idx : end_idx + self._obs_window_size - 1]
            ) for demo, (start_idx, end_idx) in zip(self._all_demos, self._demo_slices)]
            demo_keys = []
            logger.info("Loading visual observations into memory...")
            for demo_id, df in enumerate(tqdm(self._demo_keys)):
                start_idx, end_idx = self._demo_slices[demo_id]
                demo_keys.append({
                    f"{camera['name']}::{obs_type}": df["obs"][f"{camera['name']}::{obs_type}"][start_idx : end_idx + self._obs_window_size - 1] 
                    for camera in self._multi_view_cameras.values() for obs_type in self._visual_obs_types
                })
            self._demo_keys = demo_keys
            self._demo_indices = np.concatenate([
                [(idx, demo_id) for idx in range(0, end_idx - start_idx)]
                for demo_id, (start_idx, end_idx) in enumerate(self._demo_slices)
                if end_idx > 0
            ])
        else:
            self._demo_keys = [df["obs"] for df in self._demo_keys]
            self._demo_indices = np.concatenate([
                [(idx, demo_id) for idx in range(start_idx, end_idx)]
                for demo_id, (start_idx, end_idx) in enumerate(self._demo_slices)
                if end_idx > 0
            ])
        logger.info(f"Loaded {len(self._demo_keys)} demos from {data_path}, total data points: {len(self._demo_indices)}")

    def __len__(self) -> int:
        return len(self._demo_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        idx, demo_id = self._demo_indices[index]
        data = dict()
        s = np.s_[idx : idx + self._obs_window_size]
        for k in self._all_demos[demo_id]:
            if k in ["actions", "policy"]:
                data[k] = any_slice(self._all_demos[demo_id][k], np.s_[idx : idx + self._ctx_len])
                if data[k] == {}:
                    continue
                action_chunk_size = get_batch_size(data[k], strict=True)
                pad_size = self._ctx_len - action_chunk_size
                # Construct action masks
                if self._use_action_chunks:
                    assert pad_size == 0, "pad_size should be 0 if use_action_chunks is True!"
                    if k == "actions":
                        data["masks"] = self._all_demos[demo_id]["action_masks"][idx : idx + self._ctx_len]
                else:
                    # pad action chunks to equal length of ctx_len
                    data[k] = any_concat(
                        [
                            data[k],
                        ]
                        + [any_ones_like(any_slice(data[k], np.s_[0:1]))] * pad_size,
                        dim=0,
                    )
                    if k == "actions":
                        data["masks"] = torch.cat(
                            [
                                torch.ones((action_chunk_size,), dtype=torch.bool),
                                torch.zeros((pad_size,), dtype=torch.bool),
                            ],
                            dim=0,
                        )
            elif k != "action_masks":
                data[k] = any_slice(self._all_demos[demo_id][k], s)
            else:
                # action_masks has already been processed
                pass
        if self._action_history_len > 1 and "policy" in self._all_demos[demo_id] and "base_action" in self._all_demos[demo_id]["policy"]:
            N = self._action_history_len
            all_base = self._all_demos[demo_id]["policy"]["base_action"]
            T_demo = all_base.shape[0]
            history = []
            for k in range(N):
                t = idx - (N - 1) + k
                t = max(0, min(t, T_demo - 1))
                history.append(all_base[t])
            data["base_action_history"] = torch.cat(history, dim=-1)

        df = self._demo_keys[demo_id]
        for obs_type in self._visual_obs_types:
            if obs_type == "pcd":
                raise KeyError("PCD not supported in sampling dataset!")
            else:
                for camera in self._multi_view_cameras.values():
                    obs = torch.from_numpy(
                        df[f"{camera['name']}::{obs_type}"][
                            idx * self._downsample_factor : (idx + self._obs_window_size)
                            * self._downsample_factor : self._downsample_factor
                        ]
                    )
                    if obs_type == "rgb":
                        data["obs"][f"{camera['name']}::{obs_type}"] = F.interpolate(
                            obs[..., :3].to(torch.uint8).movedim(-1, -3),
                            size=tuple(camera["resolution"]),
                            mode='nearest-exact'
                        )
                    elif obs_type == "depth_linear":
                        depth = F.interpolate(
                            obs.to(torch.float32).unsqueeze(0),
                            size=tuple(camera["resolution"]),
                            mode='nearest-exact'
                        ).squeeze(0)
                        depth = torch.clamp(depth, MIN_DEPTH, MAX_DEPTH)
                        data["obs"][f"{camera['name']}::{obs_type}"] = depth
        return data

    def _extract_low_dim_data(self, demo_key: Any) -> Dict[str, torch.Tensor]:
        proprio = torch.from_numpy(np.array(demo_key["obs"]["robot::proprio"][::self._downsample_factor], dtype=np.float32))
        action = torch.from_numpy(np.array(demo_key["action"][:], dtype=np.float32))
        ret = {
            "proprio": proprio,
            "action": action,
        }
        if self._use_task_info:
            for key in demo_key["obs"].keys():
                if key.startswith("task::"):
                    ret["task"] = torch.from_numpy(np.array(demo_key["obs"][key][::self._downsample_factor], dtype=np.float32))
                    break
        if "policy" in demo_key and "base_action" in demo_key["policy"]:
            ret["base_action"] = torch.from_numpy(np.array(demo_key["policy"]["base_action"][:], dtype=np.float32))
            ret["oracle_action"] = torch.from_numpy(np.array(demo_key["policy"]["oracle_action"][:], dtype=np.float32))
            ret["int_state"] = torch.from_numpy(np.array(demo_key["policy"]["int_state"][:], dtype=np.uint8))
        if "dp_embedding" in demo_key:
            ret["dp_embedding"] = torch.from_numpy(np.array(demo_key["dp_embedding"][:], dtype=np.float32))
        return ret
    
    def _preload_demo(self, demo_key: Any) -> Dict[str, Any]:
        """
        Preload a single demo into memory. Currently it loads action, proprio, and optionally task info.
        Args:
            demo_key (Any): Key of the demo to preload.
        Returns:
            demo (dict): Preloaded demo.
        """
        demo = dict()
        demo["obs"] = {"qpos": dict(), "eef": dict()}
        demo["policy"] = dict()
        # load low_dim data
        action_dict = dict()
        low_dim_data = self._extract_low_dim_data(demo_key)
        for key, data in low_dim_data.items():
            if key == "proprio":
                # normalize proprioception
                if "base_qvel" in PROPRIOCEPTION_INDICES[self._robot_type]:
                    demo["obs"]["odom"] = {
                        "base_velocity": 2
                        * (
                            data[..., PROPRIOCEPTION_INDICES[self._robot_type]["base_qvel"]]
                            - JOINT_RANGE[self._robot_type]["base"][0]
                        )
                        / (JOINT_RANGE[self._robot_type]["base"][1] - JOINT_RANGE[self._robot_type]["base"][0])
                        - 1.0
                    }
                for key in PROPRIO_QPOS_INDICES[self._robot_type]:
                    if "gripper" in key:
                        # rectify gripper actions to {-1, 1}
                        demo["obs"]["qpos"][key] = torch.mean(
                            data[..., PROPRIO_QPOS_INDICES[self._robot_type][key]], dim=-1, keepdim=True
                        )
                        demo["obs"]["qpos"][key] = torch.where(
                            demo["obs"]["qpos"][key]
                            > (JOINT_RANGE[self._robot_type][key][0] + JOINT_RANGE[self._robot_type][key][1]) * 0.8,
                            1.0,
                            -1.0,
                        )
                    else:
                        # normalize the qpos to [-1, 1]
                        demo["obs"]["qpos"][key] = (
                            2
                            * (
                                data[..., PROPRIO_QPOS_INDICES[self._robot_type][key]]
                                - JOINT_RANGE[self._robot_type][key][0]
                            )
                            / (JOINT_RANGE[self._robot_type][key][1] - JOINT_RANGE[self._robot_type][key][0])
                            - 1.0
                        )
                if self._robot_type in EEF_POSITION_RANGE:
                    for key in EEF_POSITION_RANGE[self._robot_type]:
                        demo["obs"]["eef"][f"{key}_pos"] = (
                            2
                            * (
                                data[..., PROPRIOCEPTION_INDICES[self._robot_type][f"eef_{key}_pos"]]
                                - EEF_POSITION_RANGE[self._robot_type][key][0]
                            )
                            / (EEF_POSITION_RANGE[self._robot_type][key][1] - EEF_POSITION_RANGE[self._robot_type][key][0])
                            - 1.0
                        )
                        # don't normalize the eef orientation
                        demo["obs"]["eef"][f"{key}_quat"] = data[
                            ..., PROPRIOCEPTION_INDICES[self._robot_type][f"eef_{key}_quat"]
                        ]
            elif key == "action":
                # Note that we need to take the action at the timestamp before the next observation
                # First pad the action array so that it is divisible by the downsample factor
                if data.shape[0] % self._downsample_factor != 0:
                    pad_size = self._downsample_factor - (data.shape[0] % self._downsample_factor)
                    # pad with the last action
                    data = torch.cat([data, data[-1:].repeat(pad_size, 1)], dim=0)
                # Now downsample the action array
                data = data[self._downsample_factor - 1 :: self._downsample_factor]
                for key, indices in ACTION_QPOS_INDICES[self._robot_type].items():
                    action_dict[key] = data[:, indices]
                    # action normalization
                    if "gripper" not in key:  # Gripper actions are already normalized to [-1, 1]
                        action_dict[key] = (
                            2
                            * (action_dict[key] - JOINT_RANGE[self._robot_type][key][0])
                            / (JOINT_RANGE[self._robot_type][key][1] - JOINT_RANGE[self._robot_type][key][0])
                            - 1.0
                        )
                if self._use_action_chunks:
                    # make actions from (T, A) to (T, L_pred_horizon, A)
                    # need to construct a mask
                    action_chunks = []
                    action_chunk_masks = []
                    action_structure = deepcopy(any_slice(action_dict, np.s_[0:1]))  # (1, A)
                    for t in range(get_batch_size(action_dict, strict=True)):
                        action_chunk = any_slice(action_dict, np.s_[t : t + self._action_prediction_horizon])
                        action_chunk_size = get_batch_size(action_chunk, strict=True)
                        pad_size = self._action_prediction_horizon - action_chunk_size
                        mask = any_concat(
                            [
                                torch.ones((action_chunk_size,), dtype=torch.bool),
                                torch.zeros((pad_size,), dtype=torch.bool),
                            ],
                            dim=0,
                        )  # (L_pred_horizon,)
                        action_chunk = any_concat(
                            [
                                action_chunk,
                            ]
                            + [any_ones_like(action_structure)] * pad_size,
                            dim=0,
                        )  # (L_pred_horizon, A)
                        action_chunks.append(action_chunk)
                        action_chunk_masks.append(mask)
                    action_chunks = any_stack(action_chunks, dim=0)  # (T, L_pred_horizon, A)
                    action_chunk_masks = torch.stack(action_chunk_masks, dim=0)  # (T, L_pred_horizon)
                    demo["actions"] = action_chunks
                    demo["action_masks"] = action_chunk_masks
                else:
                    demo["actions"] = action_dict
            elif key == "task":
                if self._task_info_range is not None:
                    # Normalize task info to [-1, 1]
                    demo["obs"]["task"] = (
                        2 * (data - self._task_info_range[0]) / (self._task_info_range[1] - self._task_info_range[0])
                        - 1.0
                    )
                else:
                    # If no range is provided, just use the raw data
                    demo["obs"]["task"] = data
            elif key == "int_state":
                # Downsample policy related data similar to action handling
                # First pad if necessary
                if data.shape[0] % self._downsample_factor != 0:
                    pad_size = self._downsample_factor - (data.shape[0] % self._downsample_factor)
                    # pad with the last value
                    data = torch.cat([data, data[-1:].repeat(pad_size)], dim=0)
                # Now downsample and ensure 1D shape [T]
                demo["policy"][key] = data[self._downsample_factor - 1 :: self._downsample_factor].squeeze()
            elif key in ["base_action", "oracle_action"]:
                # Downsample
                if data.shape[0] % self._downsample_factor != 0:
                    pad_size = self._downsample_factor - (data.shape[0] % self._downsample_factor)
                    data = torch.cat([data, data[-1:].repeat(pad_size, 1)], dim=0)
                data = data[self._downsample_factor - 1 :: self._downsample_factor]
                # Normalize to [-1, 1] using JOINT_RANGE (same as regular actions)
                for joint_key, indices in ACTION_QPOS_INDICES[self._robot_type].items():
                    if "gripper" not in joint_key:
                        data[:, indices] = (
                            2 * (data[:, indices] - JOINT_RANGE[self._robot_type][joint_key][0])
                            / (JOINT_RANGE[self._robot_type][joint_key][1] - JOINT_RANGE[self._robot_type][joint_key][0])
                            - 1.0
                        )
                demo["policy"][key] = data
            else:
                # For other keys, just store the data as is
                demo[key] = data
        return demo

    def _get_global_worker_id(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            num_workers = worker_info.num_workers if worker_info else 1
            global_worker_id = rank * num_workers + worker_id
            total_global_workers = world_size * num_workers
        else:
            global_worker_id = worker_id
            total_global_workers = worker_info.num_workers if worker_info else 1
        return global_worker_id, total_global_workers


class IIILIterableDataset(BehaviorIterableDataset):
    @classmethod
    def get_all_demo_keys(cls, data_path: str, task_name: str) -> List[h5py.Dataset]:
        demo_keys = []
        assert os.path.exists(data_path), f"Data path does not exist!"
        for file_name in tqdm(sorted(os.listdir(data_path))):
            if file_name.endswith(".hdf5"):
                f = h5py.File(os.path.join(data_path, file_name), "r", swmr=True, libver='latest')
                for demo_name in f["data"]:
                    demo_keys.append(f["data"][demo_name])
        return demo_keys

    def get_streamed_data(self, demo_ptr: int, start_idx: int, end_idx: int) -> Generator[Dict[str, Any], None, None]:
        df = self._demo_keys[demo_ptr]
        chunk_generator = self._chunk_demo(demo_ptr, start_idx, end_idx)
        # Initialize obs loaders
        obs_loaders = dict()
        for obs_type in self._visual_obs_types:
            if obs_type == "pcd":
                obs_loaders["pcd"] = self._h5_window_generator(df["obs"][f"robot::fused_pcd"], start_idx, end_idx)
            else:
                for camera_id, camera in self._multi_view_cameras.items():
                    obs_loaders[f"{camera_id}::{obs_type}"] = self._h5_window_generator(
                        df["obs"][f"{camera['name']}::{obs_type}"], start_idx, end_idx
                    )

        for _ in range(start_idx, end_idx):
            data, mask = next(chunk_generator)
            # load visual obs
            for obs_key in obs_loaders.keys():
                obs = next(obs_loaders[obs_key])
                if "::" in obs_key:
                    camera_id, obs_type = obs_key.split("::")
                    camera_name = self._multi_view_cameras[camera_id]["name"]
                    if obs_type == "rgb":
                        data["obs"][f"{camera_name}::{obs_type}"] = F.interpolate(
                            obs[..., :3].to(torch.uint8).movedim(-1, -3),
                            size=tuple(self._multi_view_cameras[camera_id]["resolution"]),
                            mode='nearest-exact'
                        )
                    elif obs_type == "depth_linear":
                        depth = F.interpolate(
                            obs.to(torch.float32).unsqueeze(0),
                            size=tuple(self._multi_view_cameras[camera_id]["resolution"]),
                            mode='nearest-exact'
                        ).squeeze(0)
                        depth = torch.clamp(depth, MIN_DEPTH, MAX_DEPTH)
                        data["obs"][f"{camera_name}::{obs_type}"] = depth
                else:
                    data["obs"][obs_key] = obs
            data["masks"] = mask
            yield data

    def _extract_low_dim_data(self, demo_key: Any) -> Dict[str, torch.Tensor]:
        proprio = torch.from_numpy(np.array(demo_key["obs"]["robot::proprio"][::self._downsample_factor], dtype=np.float32))
        action = torch.from_numpy(np.array(demo_key["action"][::self._downsample_factor], dtype=np.float32))
        ret = {
            "proprio": proprio,
            "action": action,
        }
        if self._use_task_info:
            for key in demo_key["obs"].keys():
                if key.startswith("task::"):
                    ret["task"] = torch.from_numpy(np.array(demo_key["obs"][key][::self._downsample_factor], dtype=np.float32))
                    break
        if "policy" in demo_key:
            for key in demo_key["policy"].keys():
                if key == "int_state":
                    ret[key] = torch.from_numpy(np.array(demo_key["policy"][key][::self._downsample_factor], dtype=np.float32))
                elif key in ["base_action", "oracle_action"]:
                    action_data = torch.from_numpy(np.array(demo_key["policy"][key][::self._downsample_factor], dtype=np.float32))
                    # Normalize to [-1, 1] using JOINT_RANGE (same as regular actions)
                    for joint_key, indices in ACTION_QPOS_INDICES[self._robot_type].items():
                        if "gripper" not in joint_key:
                            action_data[:, indices] = (
                                2 * (action_data[:, indices] - JOINT_RANGE[self._robot_type][joint_key][0])
                                / (JOINT_RANGE[self._robot_type][joint_key][1] - JOINT_RANGE[self._robot_type][joint_key][0])
                                - 1.0
                            )
                    ret[key] = action_data
                else:
                    ret[key] = torch.from_numpy(np.array(demo_key["policy"][key][::self._downsample_factor], dtype=np.float32))
        return ret
    
    def _get_task_id_by_name(self, task_name: str) -> int:
        """
        Dummy function for IIILDataset. Always returns 0.
        """
        return 0
    

class IIILSamplingDataset(IIILDataset):
    def __init__(self, reweight_strategy: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set take over data ratio
        self.reweight_strategy = reweight_strategy
        # Load sampling indices
        self._human_demo_data, self._pre_int_data, self._int_data, self._policy_data = [], [], [], []
        for i, (idx, demo_id) in enumerate(self._demo_indices):
            int_state = self._all_demos[demo_id]["policy"].get("int_state")
            if int_state is None:
                # No int_state means pure human demo data
                self._human_demo_data.append(i)
            elif int_state[idx] == 0:
                self._human_demo_data.append(i)
            elif int_state[idx] == 1:
                self._pre_int_data.append(i)
            elif int_state[idx] == 2:
                self._int_data.append(i)
            else:
                self._policy_data.append(i)
        # Now, set up the data sampling ratio
        data_ratio = np.array([
            len(self._human_demo_data), len(self._pre_int_data), len(self._int_data), len(self._policy_data)
        ])
        data_ratio = data_ratio / np.sum(data_ratio)
        if self.reweight_strategy is not None and self.reweight_strategy in REWEIGHTING_STRATEGIES:
            adjusted_ratio = REWEIGHTING_STRATEGIES[self.reweight_strategy](data_ratio)
            self._intervention_data_ratio = adjusted_ratio
        else:
            logger.warning(f"Reweighting strategy {self.reweight_strategy} not recognized! Using original data ratio.")
            self._intervention_data_ratio = data_ratio
        self._intervention_data_divider = np.cumsum(self._intervention_data_ratio)
        logger.info(
            f"""\n
            Total data points: {len(self._demo_indices)}, 
            Initial data ratio: {np.round(data_ratio, 3)},
            Reweight strategy: {self.reweight_strategy},
            Data sampling ratio: {np.round(self._intervention_data_ratio, 3)}\n
            """
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # randomly sample a data point
        data_seed = random.random()
        if 0 <= data_seed < self._intervention_data_divider[0] and len(self._human_demo_data) > 0:
            index = random.choice(self._human_demo_data)
        elif self._intervention_data_divider[0] <= data_seed < self._intervention_data_divider[1] and len(self._pre_int_data) > 0:
            index = random.choice(self._pre_int_data)
        elif self._intervention_data_divider[1] <= data_seed < self._intervention_data_divider[2] and len(self._int_data) > 0:
            index = random.choice(self._int_data)
        else:
            index = random.choice(self._policy_data)
        return super().__getitem__(index)


class IIILLeRobotDataset(LeRobotDataset if LeRobotDataset is not None else object):
    def __init__(
        self, 
        sampling_mode: bool = False,
        reweight_strategy: Optional[str] = None,
        *args, 
        **kwargs
    ) -> None:
        if LeRobotDataset is None:
            raise ModuleNotFoundError(
                "LeRobotDataset requires the optional 'lerobot' package, which is not installed in this environment."
            )
        super().__init__(*args, **kwargs)
        # ====== Sampling specific initialization ======
        self.sampling_mode = sampling_mode
        if self.sampling_mode:
            # set take over data ratio
            self.reweight_strategy = reweight_strategy
            # Load sampling indices
            self._human_demo_data, self._pre_int_data, self._int_data, self._policy_data = [], [], [], []
            for idx in range(self.num_frames):
                if self.hf_dataset[idx]["int_state"] == 0:
                    self._human_demo_data.append(idx)
                elif self.hf_dataset[idx]["int_state"] == 1:
                    self._pre_int_data.append(idx)
                elif self.hf_dataset[idx]["int_state"] == 2:
                    self._int_data.append(idx)
                else:
                    self._policy_data.append(idx)
            # Now, set up the data sampling ratio
            data_ratio = np.array([
                len(self._human_demo_data), len(self._pre_int_data), len(self._int_data), len(self._policy_data)
            ])
            data_ratio = data_ratio / np.sum(data_ratio)
            if self.reweight_strategy is not None and self.reweight_strategy in REWEIGHTING_STRATEGIES:
                adjusted_ratio = REWEIGHTING_STRATEGIES[self.reweight_strategy](data_ratio)
                self._intervention_data_ratio = adjusted_ratio
            else:
                logger.warning(f"Reweighting strategy {self.reweight_strategy} not recognized! Using original data ratio.")
                self._intervention_data_ratio = data_ratio
            self._intervention_data_divider = np.cumsum(self._intervention_data_ratio)
            logger.info(
                f"""\n
                Total data points: {len(self._demo_indices)}, 
                Initial data ratio: {np.round(data_ratio, 3)},
                Reweight strategy: {self.reweight_strategy},
                Data sampling ratio: {np.round(self._intervention_data_ratio, 3)}\n
                """
            )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.sampling_mode:
            # randomly sample a data point
            data_seed = random.random()
            if 0 <= data_seed < self._intervention_data_divider[0] and len(self._human_demo_data) > 0:
                index = random.choice(self._human_demo_data)
            elif self._intervention_data_divider[0] <= data_seed < self._intervention_data_divider[1] and len(self._pre_int_data) > 0:
                index = random.choice(self._pre_int_data)
            elif self._intervention_data_divider[1] <= data_seed < self._intervention_data_divider[2] and len(self._int_data) > 0:
                index = random.choice(self._int_data)
            else:
                index = random.choice(self._policy_data)
        return super().__getitem__(index)

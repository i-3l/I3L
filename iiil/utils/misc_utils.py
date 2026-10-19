import os
import numpy as np
import torch
import yaml
from enum import Enum
from omegaconf import DictConfig
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from typing import Callable, Dict, Tuple


# Root directory for data/, outputs/, and logs/. Set IIIL_BASE_PATH to override;
# otherwise defaults to the repository root.
BASE_PATHS = [
    path
    for path in [
        os.environ.get("IIIL_BASE_PATH"),
        os.environ.get("I3L_BASE_PATH"),
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ]
    if path
]


class ActionTypesSelection(str, Enum):
    DEFAULT = "default"
    BASE = "base"
    BODY = "arm"
    ARM = "arm_no_torso"
    ARM_IK = "arm_ik"
    RIGHT_ARM = "right_arm"

def get_config_path() -> str:
    """
    Returns the path to the root folder containing the config files.
    """
    return os.path.join(get_package_root_path(), "configs")

def get_package_root_path() -> str:
    """
    Returns the root path of the targeted exploration package.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config(config_path: str) -> DictConfig:
    """
    Loads a config file from the given path.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return DictConfig(config)


def fk(cmg: CuRoboMotionGenerator,q: torch.Tensor, emb_sel=CuRoboEmbodimentSelection.DEFAULT) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward kinematics using CuRobo's IK solver
    Args:
        cmg (CuRoboMotionGenerator): CuRobo motion generator instance
        q (torch.Tensor): Joint positions (could be batched)
        emb_sel (CuRoboEmbodimentSelection): Embodiment selection for CuRobo
    Returns:
        tuple: batched (ee_position, ee_quaternion)
    """
    q = cmg._tensor_args.to_device(q)
    res = cmg.mg[emb_sel].ik_solver.fk(q)
    return res.ee_position.cpu(), res.ee_quaternion.cpu()


def cotrain(data_ratio: np.ndarray) -> np.ndarray:
    """Keep the data ratio as is for cotraining"""
    assert data_ratio.size == 4 and np.isclose(np.sum(data_ratio), 1.0), "Data ratio should have 4 elements summing to 1.0, got {}".format(data_ratio)
    return data_ratio / data_ratio.sum()

def hg_daggar(data_ratio: np.ndarray) -> np.ndarray:
    """Remove on-policy data for HG-DAGGER"""
    assert data_ratio.size == 4 and np.isclose(np.sum(data_ratio), 1.0), "Data ratio should have 4 elements summing to 1.0, got {}".format(data_ratio)
    adjusted_ratio = data_ratio.copy()
    adjusted_ratio[1] = 0.0  # pre-intervention data
    adjusted_ratio[3] = 0.0  # policy data
    return adjusted_ratio / adjusted_ratio.sum()

def sirius(data_ratio: np.ndarray) -> np.ndarray:
    """For SIRIUS, We keep human demo as is, set pre-intervention to 0, intervention to 0.5, and policy as the res"""
    assert data_ratio.size == 4 and np.isclose(np.sum(data_ratio), 1.0), "Data ratio should have 4 elements summing to 1.0, got {}".format(data_ratio)
    adjusted_ratio = np.zeros_like(data_ratio)
    adjusted_ratio[0] = min(data_ratio[0], 0.5)  # human demo
    adjusted_ratio[2] = 0.0 if data_ratio[2] == 0.0 else 0.5  # intervention
    adjusted_ratio[3] = 0.0 if data_ratio[3] == 0.0 else 1.0 - adjusted_ratio[0] - adjusted_ratio[2]  # policy
    return adjusted_ratio / adjusted_ratio.sum()


REWEIGHTING_STRATEGIES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "cotrain": cotrain,
    "hg_daggar": hg_daggar,
    "sirius": sirius,
}

import yaml
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


ARCH_MAP = {
    "diffusion_rgb_a1": "dp_a1",
    "diffusion_state_a1": "dp_a1_state",
    "wbvima_r1pro": "wbvima_r1pro",
    "wbvima_r1pro_state": "wbvima_r1pro_state",
}


def get_policy_config(policy_type: str) -> DictConfig:
    """
    Loads the policy config for a given policy type.

    :param policy_type: The type of the policy to load (ex: gello).
    :return: The policy config.
    """
    policy_config_yaml = f"{Path(getsourcefile(lambda:0)).parents[1]}/configs/policy/{policy_type}.yaml"
    with open(policy_config_yaml, "r") as file:
        policy_config = yaml.safe_load(file)
    policy_config = OmegaConf.create(policy_config)
    return policy_config


def get_interrupt_detector_config(interrupt_detector_type: str) -> DictConfig:
    """
    Loads the interrupt detector config for a given interrupt detector type.

    :param interrupt_detector_type: The type of the interrupt detector to load (ex: threshold_interrupt_detector, sklearn_interrupt_detector).
    :return: The interrupt detector config.
    """
    # Special handling for voltage_based_detector which uses a unified config
    if interrupt_detector_type == "voltage_based_detector":
        config_file = "voltage_based_detector"
    else:
        config_file = interrupt_detector_type
        
    interrupt_detector_config_yaml = f"{Path(getsourcefile(lambda:0)).parents[1]}/configs/gello/{config_file}.yaml"
    with open(interrupt_detector_config_yaml, "r") as file:
        interrupt_detector_config = yaml.safe_load(file)
    interrupt_detector_config = OmegaConf.create(interrupt_detector_config)
    return interrupt_detector_config


def get_release_detector_config(release_detector_type: str) -> DictConfig:
    """
    Loads the release detector config for a given release detector type.

    :param release_detector_type: The type of the release detector to load (ex: threshold_release_detector).
    :return: The release detector config.
    """
    # Special handling for voltage_based_detector which uses a unified config
    if release_detector_type == "voltage_based_detector":
        config_file = "voltage_based_detector"
    else:
        config_file = release_detector_type
        
    release_detector_config_yaml = f"{Path(getsourcefile(lambda:0)).parents[1]}/configs/gello/{config_file}.yaml"
    with open(release_detector_config_yaml, "r") as file:
        release_detector_config = yaml.safe_load(file)
    release_detector_config = OmegaConf.create(release_detector_config)
    return release_detector_config

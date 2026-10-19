"""Configuration loader for Gello controllers."""

import logging
import os

from omegaconf import OmegaConf

from iiil.utils.misc_utils import get_config_path, load_config

logger = logging.getLogger(__name__)


def load_gello_config(config_name: str, task_overlay: str = None):
    """
    Load Gello configuration from YAML file, optionally merging a per-task overlay.

    Args:
        config_name: Name of config file without .yaml extension (e.g., "gello_r1pro")
        task_overlay: Optional overlay name (e.g., "autotune"). Loaded from
                      {config_name}/{task_overlay}.yaml and deep-merged on top of
                      the base config.

    Returns:
        Loaded configuration object
    """
    config_path = os.path.join(get_config_path(), "gello", f"{config_name}.yaml")
    config = load_config(config_path)

    if task_overlay is not None:
        overlay_path = os.path.join(
            get_config_path(), "gello", config_name, f"{task_overlay}.yaml"
        )
        if os.path.exists(overlay_path):
            overlay = load_config(overlay_path)
            config = OmegaConf.merge(config, overlay)
            logger.info(
                f"Merged task overlay '{task_overlay}' from {overlay_path}"
            )
        else:
            logger.warning(
                f"Task overlay '{task_overlay}' not found at {overlay_path}, "
                f"using base config only"
            )

    return config

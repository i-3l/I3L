import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video
from iiil import IIILTask
from iiil.utils.eval_io import close_video_writer


def _build_run_configuration_banner(cfg) -> str:
    """
    Build a human-readable banner summarizing the active Hydra choices.

    Falls back to config values when Hydra runtime choices are unavailable and
    infers the Gello config from the robot type when not explicitly set.
    """
    try:
        runtime_choices = (
            OmegaConf.to_container(HydraConfig.get().runtime.choices, resolve=True)
            or {}
        )
    except Exception:
        runtime_choices = {}

    def _pick_first(*values):
        for val in values:
            if val is None:
                continue
            if isinstance(val, str) and val.strip() == "":
                continue
            return val
        return "<unknown>"

    task_choice = _pick_first(
        runtime_choices.get("task"),
        getattr(getattr(cfg, "task", None), "type", None),
    )
    env_choice = _pick_first(
        runtime_choices.get("env"),
        getattr(getattr(cfg, "env", None), "env_type", None),
    )
    robot_choice = _pick_first(
        runtime_choices.get("robot"),
        getattr(getattr(cfg, "robot", None), "type", None),
    )
    policy_choice = _pick_first(
        runtime_choices.get("policy"),
        getattr(getattr(cfg, "policy", None), "type", None),
    )

    gello_config = getattr(getattr(cfg, "policy", None), "gello_config", None)
    if gello_config is None:
        inferred = {"A1": "gello_a1", "R1Pro": "gello_r1pro"}.get(robot_choice, None)
        gello_config = inferred if inferred is not None else "auto"

    banner = (
        "\n"
        + "=" * 70
        + "\n"
        + " RUN CONFIGURATION\n"
        + "-" * 70
        + "\n"
        + f" Task   : {task_choice}\n"
        + f" Env    : {env_choice}\n"
        + f" Robot  : {robot_choice}\n"
        + f" Policy : {policy_choice}\n"
        + f" Gello  : {gello_config}\n"
        + "=" * 70
        + "\n"
    )
    return banner


@hydra.main(config_name="base_config", config_path="iiil/configs", version_base="1.1")
def main(cfg):
    OmegaConf.resolve(cfg)

    import numpy as np
    import random
    import torch
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if os.environ.get("IIIL_DETERMINISTIC", "").strip().lower() in {"1", "true", "yes", "on"}:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

    banner = _build_run_configuration_banner(cfg)

    with IIILTask(cfg) as task:
        viewport_video_writer = None
        viewport_video_path = os.environ.get("IIIL_VIEWPORT_VIDEO_PATH")
        if viewport_video_path:
            import omnigibson as og

            viewport_width = int(os.environ.get("IIIL_VIEWPORT_VIDEO_WIDTH", "1920"))
            viewport_height = int(os.environ.get("IIIL_VIEWPORT_VIDEO_HEIGHT", "1080"))
            if (og.sim.viewer_width, og.sim.viewer_height) != (viewport_width, viewport_height):
                raise ValueError(
                    "Viewport recording resolution must match env.cfg.render: "
                    f"requested {viewport_width}x{viewport_height}, simulator is "
                    f"{og.sim.viewer_width}x{og.sim.viewer_height}"
                )
            camera_position = os.environ.get("IIIL_VIEWPORT_CAMERA_POSITION")
            camera_orientation = os.environ.get("IIIL_VIEWPORT_CAMERA_ORIENTATION")
            if camera_position and camera_orientation:
                position = [float(value) for value in camera_position.split(",")]
                orientation = [float(value) for value in camera_orientation.split(",")]
                og.sim.viewer_camera.set_position_orientation(position, orientation)
            viewport_video_writer = create_video_writer(
                fpath=viewport_video_path,
                resolution=(viewport_height, viewport_width),
            )
        if cfg.video_recording_path is not None:
            task.video_writer = create_video_writer(
                fpath=cfg.video_recording_path,
                resolution=(224, 672) if cfg.robot.type == "R1Pro" else (224, 448),
            )
        done = False
        # Show the run configuration immediately before the control loop output
        print(banner, flush=True)
        if cfg.mode == "playback":
            task.env.playback_dataset(record=False)
        else:
            try:
                while not done:
                    terminated, truncated = task.step()
                    if cfg.video_recording_path:
                        task._write_video()
                    if viewport_video_writer is not None:
                        viewport_rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
                        write_video(
                            viewport_rgb[..., :3].cpu().numpy()[None],
                            viewport_video_writer,
                        )
                    if truncated:
                        done = True
                    elif terminated:
                        # Reset the environment and policy to the home pose before the next episode
                        task.reset_episode()
            finally:
                close_video_writer(viewport_video_writer)


if __name__ == "__main__":
    _exit_code = 0
    try:
        main()
    except SystemExit as e:
        _exit_code = e.code if isinstance(e.code, int) else 1
    if os.environ.get("IIIL_HARD_EXIT_AFTER_MAIN", "").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(_exit_code)

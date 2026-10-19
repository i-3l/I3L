#!/usr/bin/env python3
"""Visualize correction-policy actions from an HDF5 file with viser.

This renders two copies of a URDF side-by-side:

- `base_action`
- `oracle_action`

Both action arrays are expected to be joint configurations of shape `(T, nq)`.

Example:
    python IIIL/scripts/vis/visualize_correction_data.py \
        --input data/correction/250_original/pnp.hdf5 \
        --demo 0
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import h5py
import numpy as np

import viser
from viser.extras import ViserUrdf


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HDF5 = REPO_ROOT / "data/correction/250_original/pnp.hdf5"
DEFAULT_URDF = REPO_ROOT / "BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/a1/urdf/a1.urdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize correction HDF5 actions with two URDFs in viser."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_HDF5,
        help="Path to correction HDF5 file.",
    )
    parser.add_argument(
        "--demo",
        type=int,
        default=0,
        help="Demo index to load, e.g. 0 for data/demo_0.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="URDF used to render actions.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the viser server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viser server.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Initial playback FPS.",
    )
    parser.add_argument(
        "--robot-spacing",
        type=float,
        default=0.0,
        help="Y offset between the base and oracle robot renders.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Request a public share URL from viser.",
    )
    return parser.parse_args()


def get_demo_keys(hdf5_path: Path) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        keys = [key for key in data_group.keys() if key.startswith("demo_")]
    return sorted(keys, key=lambda key: int(key.split("_")[-1]))


def get_demo_lengths(hdf5_path: Path, demo_keys: list[str]) -> dict[str, int]:
    lengths: dict[str, int] = {}
    with h5py.File(hdf5_path, "r") as f:
        for demo_key in demo_keys:
            lengths[demo_key] = int(f["data"][demo_key]["policy"]["base_action"].shape[0])
    return lengths


def load_demo(
    hdf5_path: Path,
    demo_key: str,
) -> dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_key]
        payload = {
            "base_action": np.asarray(demo["policy"]["base_action"][:], dtype=np.float32),
            "oracle_action": np.asarray(demo["policy"]["oracle_action"][:], dtype=np.float32),
            "action": np.asarray(demo["action"][:], dtype=np.float32),
        }
        if "obs" in demo and "robot::proprio" in demo["obs"]:
            payload["robot_proprio"] = np.asarray(demo["obs"]["robot::proprio"][:], dtype=np.float32)
        if "obs" in demo:
            for key in demo["obs"].keys():
                if "rgb" in key:
                    payload[key] = np.asarray(demo["obs"][key][:])

        if "is_oracle_active" in demo["policy"]:
            payload["is_oracle_active"] = np.asarray(
                demo["policy"]["is_oracle_active"][:]
            ).reshape(-1)
        if "int_state" in demo["policy"]:
            payload["int_state"] = np.asarray(demo["policy"]["int_state"][:]).reshape(-1)
        if "timestamp_ms" in demo.get("time", {}):
            payload["timestamp_ms"] = np.asarray(demo["time"]["timestamp_ms"][:]).reshape(-1)

    seq_len = payload["base_action"].shape[0]
    if payload["oracle_action"].shape[0] != seq_len:
        raise ValueError("base_action and oracle_action must have the same sequence length.")
    if payload["action"].shape[0] != seq_len:
        raise ValueError("action and policy action arrays must have the same sequence length.")
    return payload


def map_scalar_to_joint_limit_range(
    value: float,
    lower: float | None,
    upper: float | None,
) -> float:
    if lower is None or upper is None:
        return float(value)

    # If the action already looks like a joint position, keep it.
    if lower - 1e-6 <= value <= upper + 1e-6:
        return float(np.clip(value, lower, upper))

    # Common A1 gripper convention: a single scalar in [-1, 1].
    if -1.0001 <= value <= 1.0001:
        alpha = (float(np.clip(value, -1.0, 1.0)) + 1.0) / 2.0
        return float(lower + alpha * (upper - lower))

    return float(np.clip(value, lower, upper))


def build_action_to_urdf_cfg_fn(
    urdf: ViserUrdf,
    action_dim: int,
) -> tuple[callable, str]:
    joint_limits = urdf.get_actuated_joint_limits()
    joint_items = list(joint_limits.items())
    urdf_joint_count = len(joint_items)

    if urdf_joint_count == action_dim:
        return lambda action: np.asarray(action, dtype=np.float32), "identity"

    joint_names = [name.lower() for name, _ in joint_items]
    is_two_finger_gripper_case = (
        urdf_joint_count == action_dim + 1
        and len(joint_names) >= 2
        and "gripper" in joint_names[-1]
        and "gripper" in joint_names[-2]
    )
    if is_two_finger_gripper_case:
        finger_1_limits = joint_items[-2][1]
        finger_2_limits = joint_items[-1][1]

        def _map_action(action: np.ndarray) -> np.ndarray:
            action = np.asarray(action, dtype=np.float32)
            arm_q = action[:-1]
            gripper_cmd = float(action[-1])
            finger_1 = map_scalar_to_joint_limit_range(
                gripper_cmd,
                finger_1_limits[0],
                finger_1_limits[1],
            )
            finger_2 = map_scalar_to_joint_limit_range(
                gripper_cmd,
                finger_2_limits[0],
                finger_2_limits[1],
            )
            return np.concatenate(
                [arm_q, np.array([finger_1, finger_2], dtype=np.float32)],
                axis=0,
            )

        return _map_action, "duplicate_last_gripper_to_two_fingers"

    raise ValueError(
        f"URDF actuated joint count ({urdf_joint_count}) does not match "
        f"action dimension ({action_dim}), and no compatible mapping rule was found."
    )


def build_proprio_to_urdf_cfg_fn(
    urdf_joint_count: int,
) -> tuple[callable, str]:
    def _map_proprio(proprio: np.ndarray) -> np.ndarray:
        proprio = np.asarray(proprio, dtype=np.float32)
        if proprio.shape[0] < urdf_joint_count:
            raise ValueError(
                f"robot::proprio has length {proprio.shape[0]}, "
                f"but URDF needs at least {urdf_joint_count} joint positions."
            )
        return proprio[:urdf_joint_count]

    return _map_proprio, f"first_{urdf_joint_count}_proprio_entries"


def format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:+.3f}" for value in values.tolist()) + "]"


def get_rgb_obs_keys(data: dict[str, np.ndarray]) -> list[str]:
    return sorted([key for key in data.keys() if "rgb" in key])


def to_display_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {image.shape}")
    if image.shape[-1] == 4:
        return image[..., :3]
    return image


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    urdf_path = args.urdf.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {input_path}")
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    demo_keys = get_demo_keys(input_path)
    if not demo_keys:
        raise ValueError(f"No demo_* groups found in {input_path}")
    if args.demo < 0 or args.demo >= len(demo_keys):
        raise IndexError(f"Demo index {args.demo} out of range [0, {len(demo_keys) - 1}]")
    demo_lengths = get_demo_lengths(input_path, demo_keys)
    max_seq_len = max(demo_lengths.values())

    demo_key = demo_keys[args.demo]
    data = load_demo(input_path, demo_key)
    seq_len, action_dim = data["base_action"].shape
    rgb_obs_keys = get_rgb_obs_keys(data)

    server = viser.ViserServer(host=args.host, port=args.port, label="Correction Data Viewer")
    server.scene.set_up_direction("+z")
    server.initial_camera.position = (1.5, 1.5, 1.0)
    server.initial_camera.look_at = (0.0, 0.0, 0.15)

    base_root = server.scene.add_frame("/robots/base", show_axes=False)
    base_root.position = (0.0, -args.robot_spacing / 2.0, 0.0)

    oracle_root = server.scene.add_frame("/robots/oracle", show_axes=False)
    oracle_root.position = (0.0, args.robot_spacing / 2.0, 0.0)

    current_root = server.scene.add_frame("/robots/current", show_axes=False)
    current_root.position = (0.0, 0.0, 0.0)

    base_robot = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/robots/base",
        mesh_color_override=(0.2, 0.45, 1.0, 0.85),
    )
    oracle_robot = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/robots/oracle",
        mesh_color_override=(0.1, 0.8, 0.35, 0.85),
    )
    current_robot = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/robots/current",
        mesh_color_override=(0.75, 0.75, 0.75, 0.95),
    )
    action_to_cfg, mapping_name = build_action_to_urdf_cfg_fn(base_robot, action_dim)
    urdf_joint_count = len(base_robot.get_actuated_joint_limits())
    proprio_to_cfg, proprio_mapping_name = build_proprio_to_urdf_cfg_fn(urdf_joint_count)

    server.scene.add_grid(
        "/grid",
        width=1.8,
        height=1.8,
        position=(0.0, 0.0, 0.0),
    )
    takeover_indicator_root = server.scene.add_frame(
        "/takeover_indicator",
        show_axes=False,
        position=(0.0, 0.0, 0.42),
        visible=False,
    )
    takeover_beacon = server.scene.add_icosphere(
        "/takeover_indicator/beacon",
        radius=0.035,
        color=(255, 40, 40),
        opacity=0.95,
        material="toon5",
        position=(0.0, 0.0, 0.0),
        visible=False,
    )
    takeover_label = server.scene.add_label(
        "/takeover_indicator/label",
        text="ORACLE TAKEOVER",
        position=(0.0, 0.0, 0.08),
        font_size_mode="scene",
        font_scene_height=0.05,
        anchor="bottom-center",
        visible=False,
    )

    with server.gui.add_folder("Dataset"):
        file_text = server.gui.add_text("File", initial_value=str(input_path), disabled=True)
        demo_dropdown = server.gui.add_dropdown(
            "Demo",
            options=demo_keys,
            initial_value=demo_key,
        )
        steps_text = server.gui.add_number("Num steps", initial_value=seq_len, disabled=True)
        mapping_text = server.gui.add_text("URDF mapping", initial_value=mapping_name, disabled=True)
        proprio_mapping_text = server.gui.add_text("Current state mapping", initial_value=proprio_mapping_name, disabled=True)

    image_handles: dict[str, object] = {}
    if rgb_obs_keys:
        with server.gui.add_folder("Frame Image", order=-100):
            for rgb_key in rgb_obs_keys:
                image_handles[rgb_key] = server.gui.add_image(
                    to_display_image(data[rgb_key][0]),
                    label=rgb_key,
                    format="jpeg",
                    jpeg_quality=90,
                )

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max_seq_len - 1,
            step=1,
            initial_value=0,
        )
        play_checkbox = server.gui.add_checkbox("Play", initial_value=False)
        fps_slider = server.gui.add_slider(
            "FPS",
            min=1.0,
            max=60.0,
            step=1.0,
            initial_value=float(np.clip(args.fps, 1.0, 60.0)),
        )
        prev_button = server.gui.add_button("Prev")
        next_button = server.gui.add_button("Next")
        reset_button = server.gui.add_button("Reset")

    with server.gui.add_folder("Display"):
        show_current = server.gui.add_checkbox("Show current position", initial_value=True)
        show_base = server.gui.add_checkbox("Show base_action", initial_value=True)
        show_oracle = server.gui.add_checkbox("Show oracle_action", initial_value=True)
        show_grid = server.gui.add_checkbox("Show grid", initial_value=True)
        show_takeover_indicator = server.gui.add_checkbox("Show takeover indicator", initial_value=True)

    with server.gui.add_folder("Frame Info"):
        oracle_active_text = server.gui.add_text("Oracle active", initial_value="N/A", disabled=True)
        int_state_text = server.gui.add_text("Int state", initial_value="N/A", disabled=True)
        timestamp_text = server.gui.add_text("Timestamp ms", initial_value="N/A", disabled=True)
        base_text = server.gui.add_text("Base q", initial_value="", disabled=True)
        oracle_text = server.gui.add_text("Oracle q", initial_value="", disabled=True)
        current_text = server.gui.add_text("Current q", initial_value="", disabled=True)
        delta_text = server.gui.add_text("Oracle - Base", initial_value="", disabled=True)

    state_lock = threading.RLock()
    current_frame = {"value": 0}
    current_demo_key = {"value": demo_key}
    current_data = {"value": data}
    current_seq_len = {"value": seq_len}
    suppress_frame_slider_callback = {"value": False}

    def apply_frame(frame_idx: int) -> None:
        frame_idx = int(np.clip(frame_idx, 0, current_seq_len["value"] - 1))
        with state_lock:
            current_frame["value"] = frame_idx

            demo_data = current_data["value"]

            base_q = demo_data["base_action"][frame_idx]
            oracle_q = demo_data["oracle_action"][frame_idx]
            base_cfg = action_to_cfg(base_q)
            oracle_cfg = action_to_cfg(oracle_q)
            current_cfg = None
            if "robot_proprio" in demo_data:
                current_cfg = proprio_to_cfg(demo_data["robot_proprio"][frame_idx])

            base_robot.update_cfg(base_cfg)
            oracle_robot.update_cfg(oracle_cfg)
            if current_cfg is not None:
                current_robot.update_cfg(current_cfg)

            base_text.value = format_vector(base_q)
            oracle_text.value = format_vector(oracle_q)
            if current_cfg is not None:
                current_text.value = format_vector(current_cfg)
            else:
                current_text.value = "N/A"
            delta_text.value = format_vector(oracle_q - base_q)

            if "is_oracle_active" in demo_data:
                is_oracle_active = bool(demo_data["is_oracle_active"][frame_idx])
                oracle_active_text.value = str(is_oracle_active)
            else:
                is_oracle_active = False
                oracle_active_text.value = "N/A"
            if "int_state" in demo_data:
                int_state_text.value = str(int(demo_data["int_state"][frame_idx]))
            else:
                int_state_text.value = "N/A"
            if "timestamp_ms" in demo_data:
                timestamp_text.value = f"{float(demo_data['timestamp_ms'][frame_idx]):.1f}"
            else:
                timestamp_text.value = "N/A"
            for rgb_key, image_handle in image_handles.items():
                if rgb_key in demo_data:
                    image_handle.image = to_display_image(demo_data[rgb_key][frame_idx])
                    image_handle.visible = True
                else:
                    image_handle.visible = False

            indicator_visible = show_takeover_indicator.value and is_oracle_active
            takeover_indicator_root.visible = indicator_visible
            takeover_beacon.visible = indicator_visible
            takeover_label.visible = indicator_visible

            if frame_slider.value != frame_idx:
                suppress_frame_slider_callback["value"] = True
                frame_slider.value = frame_idx
                suppress_frame_slider_callback["value"] = False

    def load_selected_demo(new_demo_key: str) -> None:
        new_data = load_demo(input_path, new_demo_key)
        new_seq_len, new_action_dim = new_data["base_action"].shape
        if new_action_dim != action_dim:
            raise ValueError(
                f"Demo {new_demo_key} has action dimension {new_action_dim}, "
                f"expected {action_dim}."
            )

        current_demo_key["value"] = new_demo_key
        current_data["value"] = new_data
        current_seq_len["value"] = new_seq_len
        steps_text.value = new_seq_len

        if image_handles:
            new_rgb_obs_keys = get_rgb_obs_keys(new_data)
            for rgb_key in new_rgb_obs_keys:
                if rgb_key not in image_handles:
                    image_handles[rgb_key] = server.gui.add_image(
                        to_display_image(new_data[rgb_key][0]),
                        label=rgb_key,
                        format="jpeg",
                        jpeg_quality=90,
                    )
        play_checkbox.value = False
        apply_frame(0)

    @frame_slider.on_update
    def _(_event: object) -> None:
        if suppress_frame_slider_callback["value"]:
            return
        apply_frame(int(frame_slider.value))

    @demo_dropdown.on_update
    def _(_event: object) -> None:
        load_selected_demo(demo_dropdown.value)

    @prev_button.on_click
    def _(_event: object) -> None:
        apply_frame(int(frame_slider.value) - 1)

    @next_button.on_click
    def _(_event: object) -> None:
        apply_frame(int(frame_slider.value) + 1)

    @reset_button.on_click
    def _(_event: object) -> None:
        play_checkbox.value = False
        apply_frame(0)

    @show_base.on_update
    def _(_event: object) -> None:
        base_root.visible = show_base.value

    @show_current.on_update
    def _(_event: object) -> None:
        current_root.visible = show_current.value

    @show_oracle.on_update
    def _(_event: object) -> None:
        oracle_root.visible = show_oracle.value

    @show_grid.on_update
    def _(_event: object) -> None:
        grid_handle = server.scene.get_handle_by_name("/grid")
        if grid_handle is not None:
            grid_handle.visible = show_grid.value

    @show_takeover_indicator.on_update
    def _(_event: object) -> None:
        apply_frame(current_frame["value"])

    apply_frame(0)

    if args.share:
        share_url = server.request_share_url(verbose=True)
        if share_url is not None:
            print(f"Share URL: {share_url}")

    print(f"Loaded {demo_key} from {input_path}")
    print(f"Serving viser on http://{server.get_host()}:{server.get_port()}")

    while True:
        if play_checkbox.value:
            apply_frame((current_frame["value"] + 1) % current_seq_len["value"])
            time.sleep(1.0 / float(fps_slider.value))
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())

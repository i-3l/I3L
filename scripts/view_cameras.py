#!/usr/bin/env python3
"""
Live camera viewer for ZED cameras.

Two modes:
  --raw       Show raw V4L2 feed with manual brightness/gain controls (default)
  --config    Show exactly what data collection sees, using settings from
              real_franka_joint.yaml (crop, resize, v4l2 controls)

Usage:
    python scripts/view_cameras.py                    # Raw, both cameras
    python scripts/view_cameras.py --camera wrist     # Raw, wrist only
    python scripts/view_cameras.py --config           # Data collection view

Controls (raw mode only):
    q       - Quit
    +/-     - Adjust brightness (both cameras)
    [/]     - Adjust gain (both cameras)
    s       - Save current frame to /tmp/

Controls (config mode):
    q       - Quit
    s       - Save current frame to /tmp/
"""

import argparse
import cv2
import numpy as np
import subprocess
import time


def get_v4l2_ctrl(device, ctrl):
    """Read a v4l2 control value."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--get-ctrl", ctrl],
            capture_output=True, text=True, timeout=2,
        )
        # e.g. "brightness: 4"
        return int(result.stdout.strip().split(":")[-1].strip())
    except Exception:
        return None


def set_v4l2_ctrl(device, ctrl, value):
    """Set a v4l2 control value."""
    try:
        subprocess.run(
            ["v4l2-ctl", "-d", device, "--set-ctrl", f"{ctrl}={value}"],
            capture_output=True, timeout=2,
        )
    except Exception:
        pass


def run_raw_mode(cameras):
    """Raw V4L2 viewer with manual brightness/gain controls."""
    print("\nControls: q=quit, +/-=brightness, [/]=gain, s=save frame")

    frame_count = 0
    fps_start = time.time()
    fps_display = 0.0

    while True:
        panels = []
        for name, cam in cameras.items():
            ret, frame = cam["cap"].read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            # Take left half for stereo
            if w > h * 1.5:
                frame = frame[:, :w // 2]
            # Resize for display
            frame = cv2.resize(frame, (480, 360))
            # Add label
            cv2.putText(frame, name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            panels.append(frame)

        if not panels:
            continue

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # Add FPS to last panel
        cv2.putText(panels[-1], f"{fps_display:.0f} fps",
                     (panels[-1].shape[1] - 100, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Combine side by side
        if len(panels) == 2:
            display = np.hstack(panels)
        else:
            display = panels[0]

        cv2.imshow("ZED Cameras (raw)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            for cam in cameras.values():
                val = get_v4l2_ctrl(cam["dev"], "brightness")
                if val is not None and val < 8:
                    set_v4l2_ctrl(cam["dev"], "brightness", val + 1)
                    print(f"brightness -> {val + 1}")
        elif key == ord("-"):
            for cam in cameras.values():
                val = get_v4l2_ctrl(cam["dev"], "brightness")
                if val is not None and val > 0:
                    set_v4l2_ctrl(cam["dev"], "brightness", val - 1)
                    print(f"brightness -> {val - 1}")
        elif key == ord("]"):
            for cam in cameras.values():
                val = get_v4l2_ctrl(cam["dev"], "gain")
                if val is not None and val < 8:
                    set_v4l2_ctrl(cam["dev"], "gain", val + 1)
                    print(f"gain -> {val + 1}")
        elif key == ord("["):
            for cam in cameras.values():
                val = get_v4l2_ctrl(cam["dev"], "gain")
                if val is not None and val > 0:
                    set_v4l2_ctrl(cam["dev"], "gain", val - 1)
                    print(f"gain -> {val - 1}")
        elif key == ord("s"):
            for name, cam in cameras.items():
                ret, frame = cam["cap"].read()
                if ret:
                    h, w = frame.shape[:2]
                    if w > h * 1.5:
                        frame = frame[:, :w // 2]
                    path = f"/tmp/cam_{name}_{int(time.time())}.png"
                    cv2.imwrite(path, frame)
                    print(f"Saved: {path}")

    for cam in cameras.values():
        cam["cap"].release()
    cv2.destroyAllWindows()


def run_config_mode():
    """Show exactly what data collection sees, using config settings."""
    from omegaconf import OmegaConf

    # Load the Franka environment config
    config_path = "iiil/configs/env/real_franka_joint.yaml"
    cfg = OmegaConf.load(config_path)
    camera_config = cfg.env.cfg.cameras

    from iiil.utils.camera_capture import ZedCameraCapture

    captures = {}
    for cam_key, cam_cfg in camera_config.items():
        name = cam_cfg["name"]
        device = cam_cfg["device"]
        resolution = tuple(cam_cfg.get("resolution", [240, 240]))
        capture_resolution = cam_cfg.get("capture_resolution", None)
        if capture_resolution is not None:
            capture_resolution = tuple(capture_resolution)
        capture_fps = cam_cfg.get("capture_fps", None)
        crop_size = cam_cfg.get("crop_size", None)
        crop_offset = tuple(cam_cfg.get("crop_offset", [0, 0]))
        v4l2_controls = cam_cfg.get("v4l2", None)
        if v4l2_controls is not None:
            v4l2_controls = dict(v4l2_controls)
        try:
            capture = ZedCameraCapture(
                device_path=device,
                name=name,
                resolution=resolution,
                capture_resolution=capture_resolution,
                capture_fps=capture_fps,
                crop_size=crop_size,
                crop_offset=crop_offset,
                v4l2_controls=v4l2_controls,
            )
            captures[cam_key] = capture
            print(
                f"  {cam_key}: {device} -> {resolution[0]}x{resolution[1]} "
                f"(capture={capture_resolution or 'default'}, "
                f"crop size={crop_size or 'max'}, "
                f"offset x={crop_offset[0]}, y={crop_offset[1]})"
            )
            if v4l2_controls:
                for ctrl, val in v4l2_controls.items():
                    print(f"    {ctrl}={val}")
        except RuntimeError as e:
            print(f"  WARNING: {cam_key}: {e}")

    if not captures:
        print("No cameras available!")
        return

    # Collect device paths for v4l2 control adjustments
    devices = []
    for cam_key, cam_cfg in camera_config.items():
        if cam_key in captures:
            devices.append(cam_cfg["device"])

    # Adjustable parameters: (display_name, v4l2_ctrl_name, min, max, step)
    PARAMS = [
        ("brightness",  "brightness",                0,    8,    1),
        ("contrast",    "contrast",                  0,    8,    1),
        ("saturation",  "saturation",                0,    8,    1),
        ("gain",        "gain",                      0,    8,    1),
        ("gamma",       "gamma",                     1,  500,   10),
        ("sharpness",   "sharpness",                 0,    8,    1),
        ("wb_temp",     "white_balance_temperature", 2800, 6500, 100),
    ]

    # Track current v4l2 values
    v4l2_state = {}
    for dev in devices:
        v4l2_state[dev] = {}
        for display_name, ctrl_name, min_v, max_v, step in PARAMS:
            v4l2_state[dev][display_name] = get_v4l2_ctrl(dev, ctrl_name) or min_v

    # Camera selection and parameter selection
    cam_keys = list(captures.keys())
    selected_cam_idx = 0
    selected_param_idx = 0

    print(f"\nShowing data collection view ({len(captures)} cameras)")
    print("Controls:")
    print("  TAB        = cycle camera")
    print("  1-7        = select parameter")
    print("  +/- or [/] = adjust selected parameter")
    print("  s=save frame, w=write config, q=quit")
    for i, (name, _, min_v, max_v, step) in enumerate(PARAMS):
        print(f"  {i+1}: {name} ({min_v}-{max_v}, step {step})")

    # Display scale factor
    scale = 2

    frame_count = 0
    fps_start = time.time()
    fps_display = 0.0

    while True:
        panels = []
        for i, cam_key in enumerate(cam_keys):
            capture = captures[cam_key]
            frame = capture.get_frame()
            if frame is None:
                continue
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]
            frame_bgr = cv2.resize(frame_bgr, (w * scale, h * scale),
                                   interpolation=cv2.INTER_NEAREST)
            is_selected = (i == selected_cam_idx)
            if is_selected:
                cv2.rectangle(frame_bgr, (0, 0),
                              (frame_bgr.shape[1] - 1, frame_bgr.shape[0] - 1),
                              (0, 255, 255), 3)
            # Show all param values for this camera
            dev = camera_config[cam_key]["device"]
            st = v4l2_state.get(dev, {})
            marker = ">" if is_selected else " "
            line1 = f"{marker}{cam_key}"
            cv2.putText(frame_bgr, line1, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if is_selected else (0, 255, 0), 2)
            # Show params in 2 rows
            p_names = [p[0] for p in PARAMS]
            row1 = "  ".join(f"{p}={st.get(p,'?')}" for p in p_names[:4])
            row2 = "  ".join(f"{p}={st.get(p,'?')}" for p in p_names[4:])
            cv2.putText(frame_bgr, row1, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cv2.putText(frame_bgr, row2, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            # Highlight selected param
            sel_param_name = PARAMS[selected_param_idx][0]
            if is_selected:
                sel_label = f"[{sel_param_name}={st.get(sel_param_name, '?')}]"
                cv2.putText(frame_bgr, sel_label, (10, frame_bgr.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            panels.append(frame_bgr)

        if not panels:
            continue

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        cv2.putText(panels[-1], f"{fps_display:.0f} fps",
                     (panels[-1].shape[1] - 100, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        if len(panels) >= 2:
            display = np.hstack(panels)
        else:
            display = panels[0]

        cv2.imshow("ZED Cameras (config)", display)
        key = cv2.waitKey(1) & 0xFF

        selected_key = cam_keys[selected_cam_idx]
        selected_dev = camera_config[selected_key]["device"]

        if key == ord("q"):
            break
        elif key == 9:  # Tab
            selected_cam_idx = (selected_cam_idx + 1) % len(cam_keys)
            print(f"Selected camera: {cam_keys[selected_cam_idx]}")
        elif ord("1") <= key <= ord("7"):
            idx = key - ord("1")
            if idx < len(PARAMS):
                selected_param_idx = idx
                print(f"Selected param: {PARAMS[selected_param_idx][0]}")
        elif key in (ord("+"), ord("="), ord("]")):
            p_name, ctrl_name, min_v, max_v, step = PARAMS[selected_param_idx]
            val = v4l2_state[selected_dev][p_name]
            if val + step <= max_v:
                val += step
                set_v4l2_ctrl(selected_dev, ctrl_name, val)
                v4l2_state[selected_dev][p_name] = val
            print(f"{selected_key} {p_name} -> {val}")
        elif key in (ord("-"), ord("[")):
            p_name, ctrl_name, min_v, max_v, step = PARAMS[selected_param_idx]
            val = v4l2_state[selected_dev][p_name]
            if val - step >= min_v:
                val -= step
                set_v4l2_ctrl(selected_dev, ctrl_name, val)
                v4l2_state[selected_dev][p_name] = val
            print(f"{selected_key} {p_name} -> {val}")
        elif key == ord("s"):
            for cam_key, capture in captures.items():
                frame = capture.get_frame()
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    path = f"/tmp/cam_{cam_key}_{int(time.time())}.png"
                    cv2.imwrite(path, frame_bgr)
                    print(f"Saved: {path}")
        elif key == ord("w"):
            # Write v4l2 values back to config using ruamel.yaml
            # to preserve comments (especially '# @package _global_')
            import ruamel.yaml
            yaml = ruamel.yaml.YAML()
            yaml.preserve_quotes = True
            with open(config_path, "r") as f:
                raw = yaml.load(f)
            for cam_key in cam_keys:
                dev = camera_config[cam_key]["device"]
                if dev not in v4l2_state:
                    continue
                st = v4l2_state[dev]
                v4l2_node = raw["env"]["cfg"]["cameras"][cam_key]["v4l2"]
                v4l2_node["brightness"] = st["brightness"]
                v4l2_node["contrast"] = st["contrast"]
                v4l2_node["saturation"] = st["saturation"]
                v4l2_node["gain"] = st["gain"]
                v4l2_node["gamma"] = st["gamma"]
                v4l2_node["sharpness"] = st["sharpness"]
                v4l2_node["white_balance_automatic"] = 0
                v4l2_node["hue_automatic"] = 0
                v4l2_node["white_balance_temperature"] = st["wb_temp"]
            with open(config_path, "w") as f:
                yaml.dump(raw, f)
            print(f"Wrote settings to {config_path}")
            for dev, st in v4l2_state.items():
                print(f"  {dev}: {st}")

    for capture in captures.values():
        capture.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live ZED camera viewer")
    parser.add_argument("--camera", "-c", choices=["wrist", "external", "both"],
                        default="both", help="Which camera(s) to show (raw mode)")
    parser.add_argument("--config", action="store_true",
                        help="Show data collection view using config settings (crop, resize, v4l2)")
    args = parser.parse_args()

    if args.config:
        run_config_mode()
        return

    # Raw mode
    cameras = {}
    cam_list = []
    if args.camera in ("wrist", "both"):
        cam_list.append(("wrist", "/dev/zed_wrist"))
    if args.camera in ("external", "both"):
        cam_list.append(("external", "/dev/zed_external"))

    for name, dev in cam_list:
        cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            print(f"WARNING: Could not open {name} ({dev})")
            continue
        # Flush initial frames
        for _ in range(5):
            cap.read()
        cameras[name] = {"cap": cap, "dev": dev}
        brightness = get_v4l2_ctrl(dev, "brightness")
        gain = get_v4l2_ctrl(dev, "gain")
        print(f"{name}: opened {dev} (brightness={brightness}, gain={gain})")

    if not cameras:
        print("No cameras available!")
        return

    run_raw_mode(cameras)


if __name__ == "__main__":
    main()

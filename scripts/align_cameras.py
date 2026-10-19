#!/usr/bin/env python3
"""Overlay saved reference views on the live Franka cameras at full resolution.

The camera devices are opened through the same ``ZedCameraCapture`` class,
Franka YAML configuration, and V4L2 controls as data collection, but in the
camera's 2K mode (4416x1242) so the overlay carries the maximum pixel density
the hardware offers: a 1242x1242 square of native, unresized sensor pixels.

The 2K readout is a centered sensor window with a narrower field of view than
collection's 2560x720 mode (measured 2026-08-20 on both Franka cameras: 720p
is the full sensor binned 2x, and the 2K window starts (176, 99) native px
in).  The square is therefore placed to cover as much of the configured
collection crop as the window allows; thin margins of that crop (about 50
720p px at the top, and the wrist crop's left/right edges) physically cannot
appear in 2K.  Alignment correctness is unaffected — saved references and the
live view go through the identical pipeline, so matching the ghost overlay
reproduces the physical scene exactly.

Run ``./align.sh`` before starting data collection.  On the first run, arrange
the scene correctly and press ``c`` to capture references.  Later runs load
those references automatically and blend them over the live views.

run.sh modes 10 and 11 also record how the scene actually looked when each
session started: they invoke ``./align.sh --snapshot NAME`` to save a headless
reference set under ``alignment_references/sessions/NAME_<timestamp>/``.
Re-create a session's scene later with ``./align.sh TASKNAME`` (that task's
newest snapshot), ``./align.sh last`` (newest snapshot overall, also
``--last``), or ``./align.sh SESSION_NAME`` for an exact one; ``--list`` shows
everything with per-camera health.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from omegaconf import OmegaConf

from iiil.utils.camera_capture import ZedCameraCapture


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "iiil/configs/env/real_franka_joint.yaml"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "alignment_references"
WINDOW_NAME = "Franka rack + mug alignment (2K)"
MAX_WINDOW_SIZE = (1600, 950)  # (width, height) the viewer window is fit into
BASE_PANEL_WIDTH = 600  # reference width for scaling header/footer text

# Alignment always captures in the ZED's 2K mode for maximum pixel density.
# Sensor model, measured 2026-08-20 on both Franka cameras (ORB+RANSAC,
# ~650 inliers): 2560x720 is the full sensor binned 2x, and the 2K readout is
# a centered native window whose top-left sits at (176, 99) native px —
# consistent with (2560-2208)/2 and (1440-1242)/2 on a ~2560x1440 native eye.
ALIGNMENT_CAPTURE_RESOLUTION = (1242, 4416)  # (height, width), stereo pair
ALIGNMENT_CAPTURE_FPS = 15
NATIVE_PIXELS_PER_COLLECTION_PIXEL = 2
SENSOR_2K_WINDOW_ORIGIN = (176, 99)  # (x, y) native px
COLLECTION_CAPTURE_RESOLUTION = (720, 2560)  # the sensor model assumes this
MANIFEST_NAME = "manifest.json"
PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SESSIONS_DIRNAME = "sessions"
SNAPSHOT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
SNAPSHOT_WARMUP_SECONDS = 2.0


def opacity_value(value: str) -> float:
    """Argparse validator for a reference opacity in [0, 1]."""
    try:
        opacity = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("opacity must be a number from 0 to 1") from exc
    if not 0.0 <= opacity <= 1.0:
        raise argparse.ArgumentTypeError("opacity must be between 0 and 1")
    return opacity


def profile_value(value: str) -> str:
    """Keep profile names as a single safe directory component."""
    if not PROFILE_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "profile must use only letters, numbers, '.', '_', and '-'"
        )
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blend persistent full-resolution reference images over the live "
            "Franka camera crops used for rack/mug placement."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        choices=("both", "wrist", "external"),
        default="both",
        help="camera view(s) to open",
    )
    parser.add_argument(
        "--opacity",
        "--alpha",
        type=opacity_value,
        default=0.45,
        help="initial opacity of the reference image",
    )
    parser.add_argument(
        "--profile",
        type=profile_value,
        default="rack_mug",
        help="named set of persistent reference images",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="directory containing named reference profiles",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="CAMERA=IMAGE",
        help=(
            "load a one-off full-resolution image for wrist or external; may "
            "be specified more than once"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Franka environment YAML containing camera settings",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "name",
        nargs="?",
        default=None,
        metavar="NAME",
        help=(
            "what to overlay: a profile name, an exact session snapshot name, "
            "or a task name (opens that task's newest snapshot, e.g. "
            "'./align.sh mugtask'); 'last' opens the newest snapshot overall. "
            "Takes precedence over --profile"
        ),
    )
    modes.add_argument(
        "--check",
        action="store_true",
        help="capture one processed frame from each camera and exit without a GUI",
    )
    modes.add_argument(
        "--snapshot",
        metavar="NAME",
        default=None,
        help=(
            "headlessly save the current scene as a reference set under "
            "<reference-root>/sessions/NAME_<timestamp>/ and exit; run.sh "
            "modes 10 and 11 call this before Franka data collection"
        ),
    )
    modes.add_argument(
        "--last",
        action="store_true",
        help="overlay the most recent session snapshot instead of --profile",
    )
    modes.add_argument(
        "--list",
        dest="list_profiles",
        action="store_true",
        help="list reference profiles and session snapshots, then exit",
    )
    return parser.parse_args(argv)


def selected_camera_keys(selection: str, available: Sequence[str]) -> list[str]:
    if selection == "both":
        keys = [key for key in ("wrist", "external") if key in available]
    else:
        keys = [selection] if selection in available else []
    if not keys:
        raise ValueError(
            f"camera selection '{selection}' is not present in the config; "
            f"available cameras: {', '.join(available) or 'none'}"
        )
    return keys


def load_camera_config(config_path: Path) -> dict[str, dict[str, Any]]:
    """Load plain camera dictionaries from the Franka Hydra config."""
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"camera config does not exist: {config_path}")
    cfg = OmegaConf.load(config_path)
    try:
        cameras = OmegaConf.to_container(cfg.env.cfg.cameras, resolve=True)
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"no env.cfg.cameras mapping in {config_path}") from exc
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError(f"no cameras configured in {config_path}")
    return {str(key): dict(value) for key, value in cameras.items()}


def expected_capture_geometry(cam_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the raw/eye/crop geometry collection is configured to use."""
    capture_resolution = cam_cfg.get("capture_resolution")
    if capture_resolution is None or len(capture_resolution) != 2:
        raise ValueError("alignment requires a configured [height, width] capture_resolution")
    raw_h, raw_w = (int(capture_resolution[0]), int(capture_resolution[1]))
    stereo_split = raw_w > raw_h * 1.5
    source_w = raw_w // 2 if stereo_split else raw_w
    source_h = raw_h
    max_crop_size = min(source_h, source_w)
    requested_crop_size = cam_cfg.get("crop_size")
    crop_size = min(int(requested_crop_size or max_crop_size), max_crop_size)
    offset_x, offset_y = (int(value) for value in cam_cfg.get("crop_offset", [0, 0]))
    max_x = source_w - crop_size
    max_y = source_h - crop_size
    start_x = int(np.clip(max_x // 2 + offset_x, 0, max_x))
    start_y = int(np.clip(max_y // 2 + offset_y, 0, max_y))
    return {
        "raw_resolution": [raw_h, raw_w],
        "stereo_split": stereo_split,
        "source_resolution": [source_h, source_w],
        "effective_crop_size": crop_size,
        "crop_origin": [start_x, start_y],
    }


def alignment_plan(cam_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Plan a 2K capture: the largest native-pixel square the camera offers.

    Alignment maximizes pixel density rather than matching the collection
    framing: the reference and the live view only need to be pixel-identical
    to EACH OTHER for the ghost overlay to reproduce the physical scene.  The
    2K mode is a narrower sensor window than collection's binned 720p mode,
    so the 1242x1242 square (the 2K eye's full height) is positioned via the
    measured sensor model to cover as much of the configured collection crop
    as the window physically allows; the small uncovered margins are accepted.
    """
    base = expected_capture_geometry(cam_cfg)
    if tuple(int(value) for value in cam_cfg["capture_resolution"]) != (
        COLLECTION_CAPTURE_RESOLUTION
    ):
        raise RuntimeError(
            "the 2K alignment sensor model is calibrated for a "
            f"{COLLECTION_CAPTURE_RESOLUTION[1]}x{COLLECTION_CAPTURE_RESOLUTION[0]} "
            "collection capture; update SENSOR_2K_WINDOW_ORIGIN after remeasuring"
        )

    hi_h, hi_w = ALIGNMENT_CAPTURE_RESOLUTION
    stereo_split = hi_w > hi_h * 1.5
    source_h, source_w = (hi_h, hi_w // 2 if stereo_split else hi_w)
    crop_size = min(source_h, source_w)

    # Collection crop region mapped into 2K image coordinates.
    binning = NATIVE_PIXELS_PER_COLLECTION_PIXEL
    window_x, window_y = SENSOR_2K_WINDOW_ORIGIN
    base_x, base_y = base["crop_origin"]
    region_x = base_x * binning - window_x
    region_y = base_y * binning - window_y
    region_size = int(base["effective_crop_size"]) * binning

    # Center the square on that region, clamped into the 2K frame.
    max_x = source_w - crop_size
    max_y = source_h - crop_size
    start_x = int(np.clip(region_x + (region_size - crop_size) // 2, 0, max_x))
    start_y = int(np.clip(region_y + (region_size - crop_size) // 2, 0, max_y))
    # Offsets that make ZedCameraCapture land exactly on this crop origin.
    offset_x = start_x - max_x // 2
    offset_y = start_y - max_y // 2

    v4l2_controls = cam_cfg.get("v4l2")
    return {
        "output_resolution": (crop_size, crop_size),
        "capture_kwargs": {
            "device_path": str(cam_cfg["device"]),
            "name": str(cam_cfg["name"]),
            "resolution": (crop_size, crop_size),
            "capture_resolution": ALIGNMENT_CAPTURE_RESOLUTION,
            "capture_fps": ALIGNMENT_CAPTURE_FPS,
            "crop_size": crop_size,
            "crop_offset": (offset_x, offset_y),
            "v4l2_controls": (
                dict(v4l2_controls) if v4l2_controls is not None else None
            ),
        },
        "expected_geometry": {
            "raw_resolution": [hi_h, hi_w],
            "stereo_split": stereo_split,
            "source_resolution": [source_h, source_w],
            "effective_crop_size": crop_size,
            "crop_origin": [start_x, start_y],
        },
        "collection_geometry": dict(base),
    }


def settings_signature(
    cam_cfg: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Settings that determine whether a stored overlay is pixel-aligned."""
    return {
        "device": str(cam_cfg["device"]),
        "name": str(cam_cfg["name"]),
        "collection_capture_resolution": [
            int(value) for value in cam_cfg["capture_resolution"]
        ],
        "collection_capture_fps": cam_cfg.get("capture_fps"),
        "crop_size": cam_cfg.get("crop_size"),
        "crop_offset": [int(value) for value in cam_cfg.get("crop_offset", [0, 0])],
        "v4l2": dict(cam_cfg["v4l2"]) if cam_cfg.get("v4l2") is not None else None,
        "source_eye": "left",
        "collection_geometry": dict(plan["collection_geometry"]),
        "alignment_capture_resolution": list(
            plan["capture_kwargs"]["capture_resolution"]
        ),
        "alignment_capture_fps": plan["capture_kwargs"]["capture_fps"],
        "alignment_resolution": list(plan["output_resolution"]),
        "alignment_geometry": dict(plan["expected_geometry"]),
    }


def parse_reference_overrides(values: Sequence[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        camera, separator, image_path = value.partition("=")
        if not separator or camera not in {"wrist", "external"} or not image_path:
            raise ValueError(
                f"invalid --reference '{value}'; expected wrist=IMAGE or external=IMAGE"
            )
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"reference image does not exist: {path}")
        overrides[camera] = path
    return overrides


def blend_reference(
    live_bgr: np.ndarray, reference: np.ndarray, opacity: float
) -> np.ndarray:
    """Alpha-composite a BGR/BGRA reference over a same-size live frame."""
    if live_bgr.ndim != 3 or live_bgr.shape[2] != 3:
        raise ValueError("live frame must be an HxWx3 BGR image")
    if reference.shape[:2] != live_bgr.shape[:2]:
        raise ValueError(
            f"reference shape {reference.shape[:2]} does not match live shape "
            f"{live_bgr.shape[:2]}"
        )
    if reference.ndim != 3 or reference.shape[2] not in (3, 4):
        raise ValueError("reference must be an HxWx3 BGR or HxWx4 BGRA image")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0 and 1")

    live = live_bgr.astype(np.float32)
    target = reference[..., :3].astype(np.float32)
    if reference.shape[2] == 4:
        alpha = reference[..., 3:4].astype(np.float32) / 255.0
        alpha *= opacity
    else:
        alpha = opacity
    blended = live * (1.0 - alpha) + target * alpha
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def load_reference_image(path: Path, expected_resolution: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read reference image: {path}")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"reference must be a color PNG/JPEG: {path}")
    if image.shape[:2] != tuple(expected_resolution):
        expected_h, expected_w = expected_resolution
        actual_h, actual_w = image.shape[:2]
        raise ValueError(
            f"reference {path} is {actual_w}x{actual_h}; expected "
            f"{expected_w}x{expected_h}. Capture it with 'c' in this tool."
        )
    return image


def read_manifest(profile_dir: Path) -> dict[str, Any]:
    path = profile_dir / MANIFEST_NAME
    if not path.is_file():
        return {"version": 1, "cameras": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: ignoring unreadable reference manifest {path}: {exc}")
        return {"version": 1, "cameras": {}}
    if not isinstance(data, dict) or not isinstance(data.get("cameras", {}), dict):
        print(f"Warning: ignoring malformed reference manifest {path}")
        return {"version": 1, "cameras": {}}
    return data


def write_manifest(profile_dir: Path, manifest: Mapping[str, Any]) -> None:
    path = profile_dir / MANIFEST_NAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sanitize_session_base(value: str) -> str:
    """Reduce an arbitrary run name (e.g. an HDF5 stem) to a safe profile prefix."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.lstrip("._-")
    return cleaned or "session"


def session_dir_name(base: str, timestamp: datetime) -> str:
    return f"{sanitize_session_base(base)}_{timestamp.strftime(SNAPSHOT_TIMESTAMP_FORMAT)}"


def new_session_dir(reference_root: Path, base: str) -> Path:
    """Pick an unused session directory named after the run and the local time."""
    sessions_root = reference_root / SESSIONS_DIRNAME
    name = session_dir_name(base, datetime.now())
    path = sessions_root / name
    counter = 2
    while path.exists():
        path = sessions_root / f"{name}-{counter}"
        counter += 1
    return path


def session_sort_time(profile_dir: Path) -> datetime:
    """Best-effort capture time: manifest updated_at, else directory mtime."""
    updated_at = read_manifest(profile_dir).get("updated_at")
    if isinstance(updated_at, str):
        try:
            parsed = datetime.fromisoformat(updated_at)
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(profile_dir.stat().st_mtime, tz=timezone.utc)


def find_session_dirs(reference_root: Path) -> list[Path]:
    """Session snapshot directories under the reference root, newest first."""
    sessions_root = reference_root / SESSIONS_DIRNAME
    if not sessions_root.is_dir():
        return []
    dirs = [path for path in sessions_root.iterdir() if path.is_dir()]
    return sorted(dirs, key=session_sort_time, reverse=True)


def display_path(path: Path) -> str:
    """Shorten a path relative to the working directory when possible."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def resolve_named_reference(reference_root: Path, name: str) -> Path | None:
    """Resolve a positional NAME to a reference directory.

    Tried in order: a profile directory, an exact session snapshot directory,
    then the newest session snapshot whose name starts with the (sanitized)
    task name — so './align.sh mugtask' opens mugtask's latest snapshot.
    """
    if PROFILE_PATTERN.fullmatch(name):
        profile = reference_root / name
        if profile.is_dir():
            return profile
        session = reference_root / SESSIONS_DIRNAME / name
        if session.is_dir():
            return session
    prefix = f"{sanitize_session_base(name)}_"
    for path in find_session_dirs(reference_root):
        if path.name.startswith(prefix):
            return path
    return None


def load_references(
    camera_keys: Sequence[str],
    camera_configs: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    profile_dir: Path,
    overrides: Mapping[str, Path],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load references and reject captured images made with stale geometry."""
    manifest = read_manifest(profile_dir)
    manifest_cameras = manifest.get("cameras", {})
    references: dict[str, np.ndarray] = {}
    statuses: dict[str, str] = {}

    for key in camera_keys:
        is_override = key in overrides
        path = overrides.get(key, profile_dir / f"{key}.png")
        if not path.is_file():
            statuses[key] = "no reference - press c to capture"
            continue

        metadata = manifest_cameras.get(key)
        if not is_override and not isinstance(metadata, Mapping):
            statuses[key] = "unverified reference - press C to replace"
            print(
                f"Warning: {key} reference has no valid settings manifest; "
                "use --reference for an intentional one-off image."
            )
            continue
        if not is_override and metadata.get("settings") != settings_signature(
            camera_configs[key], plans[key]
        ):
            statuses[key] = "stale settings - press C to replace"
            print(f"Warning: {key} reference was captured with different camera settings.")
            continue

        try:
            references[key] = load_reference_image(
                path, plans[key]["output_resolution"]
            )
        except ValueError as exc:
            statuses[key] = "invalid reference - see terminal"
            print(f"Warning: {exc}")
            continue
        suffix = "one-off" if is_override else path.name
        if not is_override and metadata is None:
            suffix += ", settings unverified"
        statuses[key] = f"reference: {suffix}"
    return references, statuses


def save_references(
    camera_keys: Sequence[str],
    live_frames: Mapping[str, np.ndarray],
    camera_configs: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    profile_dir: Path,
    *,
    replace: bool,
) -> list[str]:
    """Save clean BGR frames, never the annotated or blended display image."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(profile_dir)
    manifest["version"] = 1
    manifest["profile"] = profile_dir.name
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_cameras = manifest.setdefault("cameras", {})
    saved: list[str] = []

    for key in camera_keys:
        if key not in live_frames:
            continue
        path = profile_dir / f"{key}.png"
        if path.exists() and not replace:
            continue
        temporary = profile_dir / f".{key}.tmp.png"
        if not cv2.imwrite(str(temporary), live_frames[key]):
            raise OSError(f"OpenCV failed to write {temporary}")
        temporary.replace(path)
        captured_at = datetime.now(timezone.utc).isoformat()
        manifest_cameras[key] = {
            "captured_at": captured_at,
            "image": path.name,
            "settings": settings_signature(camera_configs[key], plans[key]),
        }
        saved.append(key)

    if saved:
        write_manifest(profile_dir, manifest)
    return saved


def ensure_display_available() -> None:
    if os.name != "nt" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "no graphical display is available. Run this from the robot desktop "
            "session (or use SSH with X forwarding)."
        )


def parse_v4l2_values(output: str) -> dict[str, int]:
    """Parse ``v4l2-ctl --get-ctrl`` output into integer values."""
    values: dict[str, int] = {}
    for line in output.splitlines():
        name, separator, raw_value = line.partition(":")
        if not separator:
            continue
        try:
            values[name.strip()] = int(raw_value.strip().split()[0])
        except (IndexError, ValueError):
            continue
    return values


def verify_v4l2_controls(device: str, expected: Mapping[str, Any]) -> None:
    """Fail rather than claim alignment used controls that were not applied."""
    if not expected:
        return
    executable = shutil.which("v4l2-ctl")
    if executable is None:
        raise RuntimeError("v4l2-ctl is required to apply and verify camera controls")
    names = list(expected)
    result = subprocess.run(
        [executable, "-d", device, "--get-ctrl", ",".join(names)],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"could not verify V4L2 controls on {device}: {detail}")

    actual = parse_v4l2_values(result.stdout)
    mismatches = []
    for name, expected_value in expected.items():
        if name not in actual:
            mismatches.append(f"{name}=unavailable (wanted {expected_value})")
        elif actual[name] != int(expected_value):
            mismatches.append(f"{name}={actual[name]} (wanted {expected_value})")
    if mismatches:
        raise RuntimeError(
            f"camera controls on {device} do not match the Franka config: "
            + ", ".join(mismatches)
        )


def verify_capture_geometry(
    capture: ZedCameraCapture, plan: Mapping[str, Any]
) -> None:
    """Reject fallback V4L2 modes that would change the alignment field of view."""
    expected = dict(plan["expected_geometry"])
    actual = {
        "raw_resolution": list(capture.raw_resolution),
        "stereo_split": bool(capture.stereo_split),
        "source_resolution": list(capture.source_resolution),
        "effective_crop_size": int(capture.effective_crop_size),
        "crop_origin": list(capture.crop_origin),
    }
    if actual != expected:
        raise RuntimeError(
            "negotiated camera geometry does not match the alignment plan: "
            f"actual={actual}, expected={expected}"
        )
    expected_fps = plan["capture_kwargs"]["capture_fps"]
    if expected_fps is not None and not np.isclose(
        capture.actual_capture_fps, float(expected_fps), atol=0.5
    ):
        raise RuntimeError(
            f"negotiated FPS is {capture.actual_capture_fps:g}; "
            f"expected {float(expected_fps):g}"
        )


def open_captures(
    camera_keys: Sequence[str],
    camera_configs: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, ZedCameraCapture]:
    captures: dict[str, ZedCameraCapture] = {}
    failures: list[str] = []
    for key in camera_keys:
        cfg = camera_configs[key]
        plan = plans[key]
        capture: ZedCameraCapture | None = None
        try:
            capture = ZedCameraCapture(**plan["capture_kwargs"])
            controls = cfg.get("v4l2", {})
            verify_capture_geometry(capture, plan)
            verify_v4l2_controls(str(cfg["device"]), controls)
            captures[key] = capture
            out_h, out_w = plan["output_resolution"]
            print(
                f"Opened {key}: {cfg['device']} -> {out_w}x{out_h} alignment view "
                f"(2K native pixels at origin "
                f"{plan['expected_geometry']['crop_origin']}, "
                f"fps={plan['capture_kwargs']['capture_fps']})"
            )
            if controls:
                values = ", ".join(f"{name}={value}" for name, value in controls.items())
                print(f"  V4L2 verified: {values}")
        except Exception as exc:
            if capture is not None:
                capture.close()
            print(f"Warning: could not open {key} camera ({cfg.get('device')}): {exc}")
            failures.append(key)
    if failures:
        for capture in captures.values():
            capture.close()
        failed = ", ".join(failures)
        raise RuntimeError(
            f"requested camera(s) unavailable: {failed}. Fix the camera(s), or "
            "select one explicitly with --camera wrist or --camera external."
        )
    return captures


def latest_bgr_frames(captures: Mapping[str, ZedCameraCapture]) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for key, capture in captures.items():
        rgb = capture.get_frame()
        if rgb is not None:
            frames[key] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return frames


def validate_alignment_frame(
    key: str, frame: np.ndarray, expected_resolution: tuple[int, int]
) -> None:
    expected_h, expected_w = expected_resolution
    if (
        frame.shape[:2] != tuple(expected_resolution)
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        actual = "x".join(str(size) for size in frame.shape)
        raise RuntimeError(
            f"{key} returned shape {actual}; expected a "
            f"{expected_w}x{expected_h} color frame"
        )


def run_snapshot(
    camera_keys: Sequence[str],
    captures: Mapping[str, ZedCameraCapture],
    camera_configs: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    profile_dir: Path,
) -> int:
    """Headlessly save one clean reference set of the current scene."""
    # Let auto-exposure settle so the reference matches what collection sees.
    deadline = time.monotonic() + SNAPSHOT_WARMUP_SECONDS
    frames = latest_bgr_frames(captures)
    while time.monotonic() < deadline:
        time.sleep(0.05)
        frames.update(latest_bgr_frames(captures))

    missing = [key for key in camera_keys if key not in frames]
    if missing:
        print(
            f"Error: no frame from camera(s): {', '.join(missing)}", file=sys.stderr
        )
        return 1
    for key in camera_keys:
        validate_alignment_frame(key, frames[key], plans[key]["output_resolution"])

    saved = save_references(
        camera_keys, frames, camera_configs, plans, profile_dir, replace=True
    )
    if sorted(saved) != sorted(camera_keys):
        print("Error: could not save all snapshot references", file=sys.stderr)
        return 1

    print(f"Saved scene snapshot: {display_path(profile_dir)} ({', '.join(saved)})")
    print("Re-create this scene later with one of:")
    print("  ./align.sh last")
    if profile_dir.parent == DEFAULT_REFERENCE_ROOT / SESSIONS_DIRNAME:
        print(f"  ./align.sh {profile_dir.name}")
    else:
        print(
            f"  ./align.sh --reference-root {display_path(profile_dir.parent)} "
            f"--profile {profile_dir.name}"
        )
    return 0


def profile_camera_summary(
    profile_dir: Path,
    camera_configs: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
) -> str:
    """Per-camera reference health for --list output."""
    manifest_cameras = read_manifest(profile_dir).get("cameras", {})
    parts: list[str] = []
    for key in sorted(camera_configs):
        if not (profile_dir / f"{key}.png").is_file():
            parts.append(f"{key}: missing")
            continue
        metadata = manifest_cameras.get(key)
        if not isinstance(metadata, Mapping):
            parts.append(f"{key}: unverified")
        elif metadata.get("settings") != settings_signature(
            camera_configs[key], plans[key]
        ):
            parts.append(f"{key}: stale settings")
        else:
            parts.append(f"{key}: ok")
    return ", ".join(parts)


def run_list(
    reference_root: Path,
    camera_configs: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
) -> int:
    """Print named profiles and session snapshots with reference health."""
    named: list[Path] = []
    if reference_root.is_dir():
        named = sorted(
            path
            for path in reference_root.iterdir()
            if path.is_dir() and path.name != SESSIONS_DIRNAME
        )
    sessions = find_session_dirs(reference_root)

    if not named and not sessions:
        print(
            "No reference profiles or session snapshots under "
            f"{display_path(reference_root)}"
        )
        return 0

    if named:
        print("Profiles (open with ./align.sh --profile NAME):")
        for path in named:
            print(
                f"  {path.name:<40} "
                f"{profile_camera_summary(path, camera_configs, plans)}"
            )
    if sessions:
        if named:
            print()
        print("Session snapshots, newest first (open the newest with ./align.sh --last):")
        for path in sessions:
            when = session_sort_time(path).astimezone().strftime("%Y-%m-%d %H:%M")
            print(
                f"  {path.name:<40} {when}  "
                f"{profile_camera_summary(path, camera_configs, plans)}"
            )
        print(
            "Open one with: ./align.sh TASKNAME (newest for that task), "
            "./align.sh SESSION_NAME, or ./align.sh last"
        )
    return 0


def make_panel(
    key: str,
    live_bgr: np.ndarray | None,
    reference: np.ndarray | None,
    status: str,
    opacity: float,
    overlay_enabled: bool,
    output_resolution: tuple[int, int],
) -> np.ndarray:
    frame_h, frame_w = output_resolution
    ui = max(1.0, frame_w / BASE_PANEL_WIDTH)
    header_h = int(round(66 * ui))
    panel = np.full((header_h + frame_h, frame_w, 3), 20, dtype=np.uint8)

    if live_bgr is None:
        image = np.full((frame_h, frame_w, 3), 32, dtype=np.uint8)
        cv2.putText(
            image,
            "Waiting for frame...",
            (int(frame_w * 0.25), int(frame_h * 0.52)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9 * ui,
            (180, 180, 180),
            max(2, int(round(2 * ui))),
            cv2.LINE_AA,
        )
    elif overlay_enabled and reference is not None:
        image = blend_reference(live_bgr, reference, opacity)
    else:
        image = live_bgr
    panel[header_h:, :] = image

    title = f"{key.upper()} - {frame_w}x{frame_h}"
    overlay_state = (
        f"overlay {round(opacity * 100):d}%" if overlay_enabled else "overlay OFF"
    )
    cv2.putText(
        panel,
        title,
        (int(round(12 * ui)), int(round(25 * ui))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62 * ui,
        (245, 245, 245),
        max(2, int(round(2 * ui))),
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"{status} | {overlay_state}",
        (int(round(12 * ui)), int(round(52 * ui))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42 * ui,
        (80, 220, 255) if reference is not None else (100, 170, 255),
        max(1, int(round(ui))),
        cv2.LINE_AA,
    )
    return panel


def make_display(
    camera_keys: Sequence[str],
    live_frames: Mapping[str, np.ndarray],
    references: Mapping[str, np.ndarray],
    statuses: Mapping[str, str],
    opacity: float,
    overlay_enabled: bool,
    output_resolutions: Mapping[str, tuple[int, int]],
) -> np.ndarray:
    panels = [
        make_panel(
            key,
            live_frames.get(key),
            references.get(key),
            statuses.get(key, "NO REFERENCE"),
            opacity,
            overlay_enabled,
            output_resolutions[key],
        )
        for key in camera_keys
    ]
    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.full(
                (max_h - panel.shape[0], panel.shape[1], 3), 20, dtype=np.uint8
            )
            panel = np.vstack((panel, pad))
        padded.append(panel)
    grid = np.hstack(padded)

    ui = max(
        1.0,
        max(output_resolutions[key][1] for key in camera_keys) / BASE_PANEL_WIDTH,
    )
    footer_h = int(round(58 * ui))
    footer = np.full((footer_h, grid.shape[1], 3), 28, dtype=np.uint8)
    cv2.putText(
        footer,
        "q/esc quit   [ ] opacity   o/space overlay",
        (int(round(12 * ui)), int(round(22 * ui))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50 * ui,
        (235, 235, 235),
        max(1, int(round(ui))),
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "c capture missing   C replace all references",
        (int(round(12 * ui)), int(round(46 * ui))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50 * ui,
        (235, 235, 235),
        max(1, int(round(ui))),
        cv2.LINE_AA,
    )
    return np.vstack((grid, footer))


def run_viewer(
    camera_keys: Sequence[str],
    captures: Mapping[str, ZedCameraCapture],
    camera_configs: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    profile_dir: Path,
    overrides: Mapping[str, Path],
    initial_opacity: float,
) -> None:
    output_resolutions = {
        key: tuple(plans[key]["output_resolution"]) for key in camera_keys
    }
    references, statuses = load_references(
        camera_keys, camera_configs, plans, profile_dir, overrides
    )
    opacity = initial_opacity
    overlay_enabled = True
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # Size the window to fit the screen; the canvas itself stays full-res.
    probe_canvas = make_display(
        camera_keys, {}, references, statuses, opacity, overlay_enabled,
        output_resolutions,
    )
    fit = min(
        MAX_WINDOW_SIZE[0] / probe_canvas.shape[1],
        MAX_WINDOW_SIZE[1] / probe_canvas.shape[0],
        1.0,
    )
    cv2.resizeWindow(
        WINDOW_NAME,
        int(probe_canvas.shape[1] * fit),
        int(probe_canvas.shape[0] * fit),
    )

    print(f"Reference profile: {profile_dir}")
    print(
        "Controls: q/esc quit, [/ ] opacity, o/space overlay, "
        "c capture missing, C replace all"
    )
    print("Run this viewer before data collection; camera devices may not be shareable.")

    while True:
        live_frames = latest_bgr_frames(captures)
        display = make_display(
            camera_keys,
            live_frames,
            references,
            statuses,
            opacity,
            overlay_enabled,
            output_resolutions,
        )
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(10) & 0xFF

        if key in (ord("q"), 27):
            return
        if key in (ord("o"), ord(" ")):
            overlay_enabled = not overlay_enabled
        elif key == ord("["):
            opacity = max(0.0, round(opacity - 0.05, 2))
        elif key == ord("]"):
            opacity = min(1.0, round(opacity + 0.05, 2))
        elif key in (ord("c"), ord("C")):
            replace = key == ord("C")
            keys_to_save = (
                list(camera_keys)
                if replace
                else [key for key in camera_keys if key not in references]
            )
            saved = save_references(
                keys_to_save,
                live_frames,
                camera_configs,
                plans,
                profile_dir,
                replace=replace,
            )
            if saved:
                print(f"Saved clean full-resolution reference(s): {', '.join(saved)}")
                references, statuses = load_references(
                    camera_keys, camera_configs, plans, profile_dir, overrides
                )
            elif replace:
                print("No live frames were available to replace.")
            else:
                print("References already exist; press uppercase C to replace them.")

        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return
        except cv2.error:
            return


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        camera_configs = load_camera_config(args.config)
        camera_keys = selected_camera_keys(args.camera, list(camera_configs))
        overrides = parse_reference_overrides(args.reference)
        headless = args.check or args.snapshot is not None or args.list_profiles
        if not headless:
            ensure_display_available()
        # Per-camera plan: the collection capture mode with the crop rendered
        # 1:1 at native pixels. Built for every configured camera so --list
        # can report reference health without touching the devices.
        plans = {key: alignment_plan(cfg) for key, cfg in camera_configs.items()}
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    reference_root = args.reference_root.expanduser().resolve()
    if args.list_profiles:
        return run_list(reference_root, camera_configs, plans)

    if args.snapshot is not None:
        profile_dir = new_session_dir(reference_root, args.snapshot)
    elif args.last or args.name == "last":
        sessions = find_session_dirs(reference_root)
        if not sessions:
            print(
                "Error: no session snapshots under "
                f"{display_path(reference_root / SESSIONS_DIRNAME)}; run.sh modes "
                "10/11 (or ./align.sh --snapshot NAME) create them.",
                file=sys.stderr,
            )
            return 2
        profile_dir = sessions[0]
        print(f"Using latest session snapshot: {display_path(profile_dir)}")
    elif args.name is not None:
        resolved = resolve_named_reference(reference_root, args.name)
        if resolved is None:
            print(
                f"Error: nothing matches '{args.name}' — no profile, session "
                "snapshot, or task by that name. See ./align.sh --list",
                file=sys.stderr,
            )
            return 2
        profile_dir = resolved
        print(f"Using reference: {display_path(profile_dir)}")
    else:
        profile_dir = reference_root / args.profile
    captures: dict[str, ZedCameraCapture] = {}
    try:
        captures = open_captures(camera_keys, camera_configs, plans)
        opened_keys = [key for key in camera_keys if key in captures]
        if args.check:
            frames = latest_bgr_frames(captures)
            for key in opened_keys:
                frame = frames.get(key)
                if frame is None:
                    print(
                        f"Error: {key} opened but returned no processed frame",
                        file=sys.stderr,
                    )
                    return 1
                validate_alignment_frame(key, frame, plans[key]["output_resolution"])
                height, width = frame.shape[:2]
                print(f"OK: {key} processed frame is {width}x{height}")
            return 0
        if args.snapshot is not None:
            return run_snapshot(
                opened_keys, captures, camera_configs, plans, profile_dir
            )
        run_viewer(
            opened_keys,
            captures,
            camera_configs,
            plans,
            profile_dir,
            overrides,
            args.opacity,
        )
        return 0
    except (OSError, RuntimeError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nAlignment viewer stopped.")
        return 130
    finally:
        for capture in captures.values():
            capture.close()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

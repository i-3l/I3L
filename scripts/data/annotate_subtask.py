#!/usr/bin/env python3
"""Annotate subtask start/end frames for a LeRobot dataset.

Example:
    python scripts/data/annotate_subtask.py \
        i3l/popcorn \
        --task popcorn
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

try:
    import av
except ModuleNotFoundError:
    av = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iiil.tasks import SUBTASK_NAME_DICT, TASK_INSTRUCTION_DICT  # noqa: E402


INT_STATE_LABELS = {
    0: "human",
    1: "pre-int",
    2: "intervention",
    3: "policy",
}

INT_STATE_COLORS = {
    0: (90, 210, 90),
    1: (80, 180, 255),
    2: (60, 60, 255),
    3: (170, 170, 170),
}

MIN_INT_STATE_INTERVAL_FRAMES = 15


@dataclass(frozen=True)
class Episode:
    index: int
    name: str
    videos: dict[str, Path]
    parquet: Path | None
    num_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a LeRobot dataset and annotate subtask start/end frames."
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="LeRobot Hugging Face dataset repo id, e.g. i3l/popcorn.",
    )
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=None,
        help=(
            "Local LeRobot dataset root. Defaults to HF_LEROBOT_HOME, "
            "or ~/.cache/huggingface/lerobot if the env var is unset."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset from Hugging Face if it is not already present locally.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task key from iiil.tasks.SUBTASK_NAME_DICT. Defaults to the repo name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to outputs/annotation/<repo_id>/subtask_annotations.json.",
    )
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=None,
        help=(
            "Video feature keys to show, e.g. observation.rgb.left_wrist "
            "observation.rgb.right_wrist observation.rgb.head. Defaults to all video features."
        ),
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to open first.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=640,
        help="Height for each camera view in the concatenated display.",
    )
    parser.add_argument(
        "--max-display-width",
        type=int,
        default=1920,
        help="Scale the concatenated view down if it exceeds this width.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=2560,
        help="Initial GUI window width in pixels.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=1440,
        help="Initial GUI window height in pixels.",
    )
    parser.add_argument(
        "--ui-scale",
        type=float,
        default=1.75,
        help="Scale Tk controls. Keep this near 1.0 if fonts look blurry.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=32,
        help="Tk font size in points.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS. Defaults to dataset metadata FPS or 30.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def default_lerobot_home() -> Path:
    env_home = os.environ.get("HF_LEROBOT_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".cache" / "huggingface" / "lerobot"


def default_output_path(repo_id: str) -> Path:
    return REPO_ROOT.joinpath("outputs", "annotation", *Path(repo_id).parts, "subtask_annotations.json")


def resolve_dataset_path(repo_id: str, lerobot_home: Path | None, download: bool) -> Path:
    if repo_id.startswith("/") or repo_id.startswith("."):
        raise ValueError("Pass a Hugging Face repo id like 'i3l/popcorn', not a dataset path.")
    if len(Path(repo_id).parts) < 2:
        raise ValueError("Expected repo id in '<namespace>/<dataset>' form, e.g. 'i3l/popcorn'.")

    home = (lerobot_home or default_lerobot_home()).expanduser().resolve()
    dataset_path = home.joinpath(*Path(repo_id).parts)
    if dataset_path.exists():
        return dataset_path
    if not download:
        raise FileNotFoundError(
            f"Dataset '{repo_id}' was not found at {dataset_path}. "
            "Use --download to fetch it from Hugging Face, or set --lerobot-home / HF_LEROBOT_HOME."
        )

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Downloading requires huggingface_hub. Install it or download the dataset manually."
        ) from exc

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} to {dataset_path}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dataset_path,
    )
    return dataset_path


def sorted_episode_paths(paths: list[Path]) -> list[Path]:
    def _key(path: Path) -> tuple[int, str]:
        stem = path.stem
        try:
            return int(stem.split("_")[-1]), stem
        except ValueError:
            return 10**9, stem

    return sorted(paths, key=_key)


def parse_episode_index(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse episode index from {path.name}") from exc


def load_info(dataset_path: Path) -> dict[str, Any]:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return {}
    return load_json(info_path)


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    keys = [
        key
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]
    return order_camera_keys(keys)


def order_camera_keys(keys: list[str]) -> list[str]:
    preferred = [
        "left_wrist",
        "right_wrist",
        "head",
        "wrist",
        "external",
    ]

    def _key(camera_key: str) -> tuple[int, str]:
        suffix = camera_key.split(".")[-1]
        try:
            return preferred.index(suffix), camera_key
        except ValueError:
            return len(preferred), camera_key

    return sorted(keys, key=_key)


def find_video_keys(dataset_path: Path, info: dict[str, Any]) -> list[str]:
    keys = video_keys_from_info(info)
    if keys:
        return keys

    videos_dir = dataset_path / "videos"
    if not videos_dir.exists():
        return []
    keys = []
    for path in videos_dir.glob("chunk-*/*"):
        if path.is_dir() and any(path.glob("episode_*.mp4")):
            keys.append(path.name)
    return order_camera_keys(sorted(set(keys)))


def find_parquets(dataset_path: Path) -> dict[int, Path]:
    parquets: dict[int, Path] = {}
    for path in sorted_episode_paths(list((dataset_path / "data").glob("chunk-*/episode_*.parquet"))):
        parquets[parse_episode_index(path)] = path
    return parquets


def frame_count(video_path: Path) -> int:
    if av is not None:
        try:
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                if stream.frames and stream.frames > 0:
                    return int(stream.frames)
                return sum(1 for _frame in container.decode(stream))
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        count = 0
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            count += 1
    cap.release()
    if count <= 0:
        raise ValueError(f"Video has no readable frames: {video_path}")
    return count


def discover_episodes(dataset_path: Path, camera_keys: list[str]) -> list[Episode]:
    if not camera_keys:
        raise ValueError(
            f"No video features found under {dataset_path}. "
            "Pass --camera-keys if metadata is missing."
        )

    per_camera: dict[str, dict[int, Path]] = {}
    for camera_key in camera_keys:
        paths = sorted_episode_paths(
            list((dataset_path / "videos").glob(f"chunk-*/{camera_key}/episode_*.mp4"))
        )
        if not paths:
            raise ValueError(f"No episode videos found for camera key '{camera_key}'.")
        per_camera[camera_key] = {parse_episode_index(path): path for path in paths}

    common_indices = set.intersection(*(set(paths.keys()) for paths in per_camera.values()))
    if not common_indices:
        raise ValueError(f"No common episode indices across camera keys: {camera_keys}")

    parquets = find_parquets(dataset_path)
    episodes = []
    for ep_idx in sorted(common_indices):
        videos = {camera_key: per_camera[camera_key][ep_idx] for camera_key in camera_keys}
        counts = [frame_count(path) for path in videos.values()]
        num_frames = min(counts)
        if len(set(counts)) != 1:
            print(
                f"Warning: episode_{ep_idx:06d} video lengths differ {counts}; using {num_frames}.",
                file=sys.stderr,
            )
        episodes.append(
            Episode(
                index=ep_idx,
                name=f"episode_{ep_idx:06d}",
                videos=videos,
                parquet=parquets.get(ep_idx),
                num_frames=num_frames,
            )
        )
    return episodes


def infer_task_name(dataset_path: Path, explicit_task: str | None) -> str:
    if explicit_task:
        task_name = explicit_task
    else:
        task_name = dataset_path.name
        if task_name not in SUBTASK_NAME_DICT and dataset_path.parent.name == "i3l":
            task_name = dataset_path.name

    if task_name in SUBTASK_NAME_DICT:
        return task_name

    task_text = load_task_text(dataset_path)
    if task_text:
        for candidate, instruction in TASK_INSTRUCTION_DICT.items():
            if task_text == instruction:
                return candidate

    available = ", ".join(sorted(SUBTASK_NAME_DICT))
    raise ValueError(f"Unknown task '{task_name}'. Available task keys: {available}")


def load_task_text(dataset_path: Path) -> str | None:
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return None
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            task = item.get("task")
            if isinstance(task, str):
                return task
    return None


def load_episode_int_state(episode: Episode) -> np.ndarray | None:
    if episode.parquet is None:
        return None
    try:
        df = pd.read_parquet(episode.parquet, columns=["int_state"])
    except Exception as exc:
        print(f"Warning: could not read int_state from {episode.parquet}: {exc}", file=sys.stderr)
        return None

    values = []
    for value in df["int_state"]:
        arr = np.asarray(value).reshape(-1)
        values.append(int(arr[0]) if arr.size else -1)
    int_state = np.asarray(values[: episode.num_frames], dtype=np.int32)
    return filter_short_int_state_intervals(int_state)


def filter_short_int_state_intervals(
    int_state: np.ndarray,
    min_frames: int = MIN_INT_STATE_INTERVAL_FRAMES,
) -> np.ndarray:
    filtered = int_state.copy()
    run_start = 0
    for idx in range(1, filtered.size + 1):
        if idx < filtered.size and int_state[idx] == int_state[run_start]:
            continue
        if idx - run_start < min_frames:
            filtered[run_start:idx] = -1
        run_start = idx
    return filtered


def default_annotation(subtasks: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "subtask_index": idx,
            "subtask": name,
            "start_frame": None,
            "end_frame": None,
            "start_time_sec": None,
            "end_time_sec": None,
            "total_frames": None,
            "intervention_frames": None,
            "on_policy_frames": None,
            "num_interventions": None,
            "intervention_occurred": None,
            "success": None,
        }
        for idx, name in enumerate(subtasks)
    ]


def normalize_existing_annotations(
    annotations: list[dict[str, Any]],
    subtasks: list[str],
    fps: float,
) -> list[dict[str, Any]]:
    by_name = {item.get("subtask"): item for item in annotations}
    normalized = default_annotation(subtasks)
    for item in normalized:
        old = by_name.get(item["subtask"])
        if old is None:
            continue
        item["start_frame"] = old.get("start_frame")
        item["end_frame"] = old.get("end_frame")
        update_times(item, fps)
    return normalized


def make_output_payload(
    repo_id: str,
    dataset_path: Path,
    task_name: str,
    subtasks: list[str],
    camera_keys: list[str],
    fps: float,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        payload = load_json(output_path)
        payload["repo_id"] = repo_id
        payload["dataset"] = str(dataset_path)
        payload["task"] = task_name
        payload["subtasks"] = subtasks
        payload["camera_keys"] = camera_keys
        payload["fps"] = fps
        payload.setdefault("episodes", {})
        return payload

    return {
        "repo_id": repo_id,
        "dataset": str(dataset_path),
        "task": task_name,
        "subtasks": subtasks,
        "camera_keys": camera_keys,
        "fps": fps,
        "created_by": "scripts/data/annotate_subtask.py",
        "frame_interval": "inclusive",
        "episodes": {},
    }


def ensure_episode_payload(
    payload: dict[str, Any],
    episode: Episode,
    subtasks: list[str],
    fps: float,
) -> dict[str, Any]:
    episodes_payload = payload.setdefault("episodes", {})
    ep_payload = episodes_payload.get(episode.name)
    if ep_payload is None:
        ep_payload = {
            "episode_index": episode.index,
            "num_frames": episode.num_frames,
            "annotations": default_annotation(subtasks),
        }
        episodes_payload[episode.name] = ep_payload
    else:
        ep_payload["episode_index"] = episode.index
        ep_payload["num_frames"] = episode.num_frames
        ep_payload["annotations"] = normalize_existing_annotations(
            ep_payload.get("annotations", []),
            subtasks,
            fps,
        )
    return ep_payload


def update_times(annotation: dict[str, Any], fps: float) -> None:
    for frame_key, time_key in (("start_frame", "start_time_sec"), ("end_frame", "end_time_sec")):
        frame = annotation.get(frame_key)
        annotation[time_key] = None if frame is None else round(float(frame) / fps, 6)


def clear_annotation_metrics(annotation: dict[str, Any]) -> None:
    annotation["total_frames"] = None
    annotation["intervention_frames"] = None
    annotation["on_policy_frames"] = None
    annotation["num_interventions"] = None
    annotation["intervention_occurred"] = None
    annotation["success"] = None


def count_true_runs(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    starts = mask & np.concatenate([np.array([True]), ~mask[:-1]])
    return int(np.sum(starts))


def update_annotation_metrics(
    annotation: dict[str, Any],
    int_state: np.ndarray | None,
    episode_num_frames: int,
) -> None:
    start = annotation.get("start_frame")
    end = annotation.get("end_frame")
    if start is None or end is None:
        clear_annotation_metrics(annotation)
        return

    start = clamp(int(start), 0, episode_num_frames - 1)
    end = clamp(int(end), 0, episode_num_frames - 1)
    if start > end:
        start, end = end, start
    annotation["start_frame"] = start
    annotation["end_frame"] = end
    annotation["total_frames"] = end - start + 1

    if int_state is None or int_state.size == 0:
        annotation["intervention_frames"] = None
        annotation["on_policy_frames"] = None
        annotation["num_interventions"] = None
        annotation["intervention_occurred"] = None
        annotation["success"] = None
        return

    state_end = min(end, int_state.size - 1)
    if start > state_end:
        annotation["intervention_frames"] = 0
        annotation["on_policy_frames"] = 0
        annotation["num_interventions"] = 0
        annotation["intervention_occurred"] = False
        annotation["success"] = True
        return

    states = int_state[start : state_end + 1]
    intervention_mask = states == 2
    intervention_frames = int(np.sum(intervention_mask))
    intervention_occurred = intervention_frames > 0
    annotation["intervention_frames"] = intervention_frames
    annotation["on_policy_frames"] = int(np.sum(states == 3))
    annotation["num_interventions"] = count_true_runs(intervention_mask)
    annotation["intervention_occurred"] = intervention_occurred
    annotation["success"] = not intervention_occurred


def update_episode_annotation_metrics(
    annotations: list[dict[str, Any]],
    int_state: np.ndarray | None,
    episode_num_frames: int,
) -> None:
    for annotation in annotations:
        update_annotation_metrics(annotation, int_state, episode_num_frames)


def set_annotation_frame(
    annotation: dict[str, Any],
    key: str,
    frame_idx: int,
    fps: float,
) -> None:
    annotation[key] = int(frame_idx)
    start = annotation.get("start_frame")
    end = annotation.get("end_frame")
    if start is not None and end is not None and start > end:
        annotation["start_frame"], annotation["end_frame"] = end, start
    update_times(annotation, fps)


class PyAvVideoReader:
    def __init__(self, path: Path) -> None:
        if av is None:
            raise ModuleNotFoundError("PyAV is required to decode this video.")
        self.path = path
        self.container: Any | None = None
        self.stream: Any | None = None
        self.decoder: Any | None = None
        self.next_frame_idx = 0
        self.open()

    def open(self) -> None:
        self.close()
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self.decoder = self.container.decode(self.stream)
        self.next_frame_idx = 0

    def read(self, frame_idx: int) -> tuple[bool, np.ndarray | None]:
        if frame_idx < self.next_frame_idx:
            self.open()

        try:
            while self.next_frame_idx <= frame_idx:
                frame = next(self.decoder)
                current_idx = self.next_frame_idx
                self.next_frame_idx += 1
                if current_idx == frame_idx:
                    return True, frame.to_ndarray(format="bgr24")
        except StopIteration:
            return False, None
        return False, None

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
        self.container = None
        self.stream = None
        self.decoder = None


class OpenCvVideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

    def read(self, frame_idx: int) -> tuple[bool, np.ndarray | None]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        return self.cap.read()

    def close(self) -> None:
        self.cap.release()


def make_video_reader(path: Path) -> PyAvVideoReader | OpenCvVideoReader:
    if av is not None:
        return PyAvVideoReader(path)
    return OpenCvVideoReader(path)


def read_video_frame(captures: dict[str, PyAvVideoReader | OpenCvVideoReader], frame_idx: int) -> dict[str, np.ndarray]:
    frames = {}
    for key, cap in captures.items():
        ok, frame = cap.read(frame_idx)
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_idx} from {key}")
        frames[key] = frame
    return frames


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == height:
        return frame
    width = max(1, int(round(w * height / h)))
    interpolation = cv2.INTER_AREA if height < h else cv2.INTER_CUBIC
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def draw_label(frame: np.ndarray, text: str) -> None:
    scale = max(1.0, frame.shape[0] / 360.0)
    bar_h = int(round(34 * scale))
    font_scale = 0.62 * scale
    thickness = max(2, int(round(1.4 * scale)))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (int(round(10 * scale)), int(round(24 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def concatenate_views(frames: dict[str, np.ndarray], camera_keys: list[str], display_height: int) -> np.ndarray:
    views = []
    for key in camera_keys:
        view = resize_to_height(frames[key], display_height)
        draw_label(view, key)
        views.append(view)
    return cv2.hconcat(views)


def draw_timeline(
    width: int,
    num_frames: int,
    frame_idx: int,
    int_state: np.ndarray | None,
    annotations: list[dict[str, Any]],
    active_subtask_idx: int,
) -> np.ndarray:
    ui_scale = max(1.0, width / 1200.0)
    timeline_h = int(round(120 * ui_scale))
    timeline = np.full((timeline_h, width, 3), 32, dtype=np.uint8)
    top = int(round(12 * ui_scale))
    state_h = int(round(22 * ui_scale))
    ann_top = int(round(48 * ui_scale))
    ann_h = int(round(24 * ui_scale))
    font_scale = 0.5 * ui_scale
    thickness = max(1, int(round(ui_scale)))

    if int_state is not None and len(int_state) > 0:
        state_last = len(int_state) - 1
        for x in range(width):
            idx = min(num_frames - 1, int(round(x * (num_frames - 1) / max(1, width - 1))))
            idx = min(idx, state_last)
            color = INT_STATE_COLORS.get(int(int_state[idx]), (100, 100, 100))
            timeline[top : top + state_h, x] = color
    else:
        cv2.rectangle(timeline, (0, top), (width, top + state_h), (80, 80, 80), -1)

    for idx, annotation in enumerate(annotations):
        start = annotation.get("start_frame")
        end = annotation.get("end_frame")
        if start is None or end is None:
            continue
        x1 = int(round(start * (width - 1) / max(1, num_frames - 1)))
        x2 = int(round(end * (width - 1) / max(1, num_frames - 1)))
        color = (240, 220, 80) if idx == active_subtask_idx else (120, 190, 240)
        cv2.rectangle(timeline, (x1, ann_top), (max(x1 + 1, x2), ann_top + ann_h), color, -1)

    cursor_x = int(round(frame_idx * (width - 1) / max(1, num_frames - 1)))
    cv2.line(timeline, (cursor_x, 0), (cursor_x, timeline.shape[0] - 1), (255, 255, 255), 1)
    cv2.putText(
        timeline,
        "int_state",
        (int(round(10 * ui_scale)), top + state_h + int(round(30 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (220, 220, 220),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        timeline,
        "annotations",
        (int(round(10 * ui_scale)), ann_top + ann_h + int(round(30 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (220, 220, 220),
        thickness,
        cv2.LINE_AA,
    )
    return timeline


def compose_display(
    frames: dict[str, np.ndarray],
    camera_keys: list[str],
    episode: Episode,
    frame_idx: int,
    fps: float,
    int_state: np.ndarray | None,
    annotations: list[dict[str, Any]],
    active_subtask_idx: int,
    display_height: int,
    max_display_width: int,
) -> np.ndarray:
    image = concatenate_views(frames, camera_keys, display_height)
    width = image.shape[1]
    timeline = draw_timeline(
        width,
        episode.num_frames,
        frame_idx,
        int_state,
        annotations,
        active_subtask_idx,
    )
    image = cv2.vconcat([image, timeline])

    active = annotations[active_subtask_idx]
    int_state_text = "N/A"
    if int_state is not None and frame_idx < len(int_state):
        state_value = int(int_state[frame_idx])
        int_state_text = f"{state_value} ({INT_STATE_LABELS.get(state_value, 'unknown')})"

    header = (
        f"{episode.name}  frame {frame_idx}/{episode.num_frames - 1}  "
        f"t={frame_idx / fps:.2f}s  int_state={int_state_text}  "
        f"subtask {active_subtask_idx + 1}/{len(annotations)}: {active['subtask']}  "
        f"start={active['start_frame']} end={active['end_frame']}"
    )
    overlay_scale = max(1.0, display_height / 360.0)
    header_h = int(round(40 * overlay_scale))
    header_font = 0.56 * overlay_scale
    header_thickness = max(2, int(round(1.4 * overlay_scale)))
    cv2.rectangle(image, (0, 0), (image.shape[1], header_h), (0, 0, 0), -1)
    cv2.putText(
        image,
        header,
        (int(round(10 * overlay_scale)), int(round(28 * overlay_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        header_font,
        (255, 255, 255),
        header_thickness,
        cv2.LINE_AA,
    )

    help_text = "space play/pause | arrows/,/. step | s start | e end+next | [ ] subtask | n/p episode | r clear | w save | q quit"
    footer_h = int(round(34 * overlay_scale))
    y = image.shape[0] - int(round(11 * overlay_scale))
    cv2.rectangle(image, (0, image.shape[0] - footer_h), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        image,
        help_text,
        (int(round(10 * overlay_scale)), y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46 * overlay_scale,
        (230, 230, 230),
        max(1, int(round(overlay_scale))),
        cv2.LINE_AA,
    )

    if image.shape[1] > max_display_width:
        scale = max_display_width / image.shape[1]
        image = cv2.resize(
            image,
            (max_display_width, int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def open_captures(episode: Episode) -> dict[str, PyAvVideoReader | OpenCvVideoReader]:
    captures = {}
    for key, path in episode.videos.items():
        try:
            cap = make_video_reader(path)
        except Exception:
            for opened in captures.values():
                opened.close()
            raise
        captures[key] = cap
    return captures


class SubtaskAnnotatorApp:
    def __init__(
        self,
        episodes: list[Episode],
        camera_keys: list[str],
        output_payload: dict[str, Any],
        output_path: Path,
        subtasks: list[str],
        fps: float,
        display_height: int,
        max_display_width: int,
        window_width: int,
        window_height: int,
        ui_scale: float,
        font_size: int,
        initial_episode_index: int,
    ) -> None:
        self.episodes = episodes
        self.camera_keys = camera_keys
        self.output_payload = output_payload
        self.output_path = output_path
        self.subtasks = subtasks
        self.fps = fps
        self.display_height = display_height
        self.max_display_width = max_display_width
        self.window_width = window_width
        self.window_height = window_height
        self.ui_scale = ui_scale
        self.font_size = font_size

        episode_pos_by_index = {episode.index: pos for pos, episode in enumerate(episodes)}
        self.episode_pos = episode_pos_by_index.get(initial_episode_index, 0)
        self.subtask_idx = 0
        self.frame_idx = 0
        self.playing = False
        self.captures: dict[str, PyAvVideoReader | OpenCvVideoReader] = {}
        self.int_state: np.ndarray | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.suppress_scale_callback = False

        self.root = tk.Tk()
        self.root.title("I3L Subtask Annotator")
        self._configure_window()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()
        self._bind_keys()
        self.load_episode(self.episode_pos)

    @property
    def episode(self) -> Episode:
        return self.episodes[self.episode_pos]

    @property
    def ep_payload(self) -> dict[str, Any]:
        return ensure_episode_payload(self.output_payload, self.episode, self.subtasks, self.fps)

    @property
    def annotations(self) -> list[dict[str, Any]]:
        return self.ep_payload["annotations"]

    def _configure_window(self) -> None:
        width = max(800, int(self.window_width))
        height = max(600, int(self.window_height))
        scale = max(0.75, float(self.ui_scale))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(width, height)
        self.root.tk.call("tk", "scaling", 1.0)

        base_size = max(8, int(self.font_size))
        for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            font = tkfont.nametofont(font_name)
            font.configure(size=base_size)
        tkfont.nametofont("TkFixedFont").configure(size=base_size)

        style = ttk.Style(self.root)
        pad_x = max(9, int(round(6 * scale)))
        pad_y = max(6, int(round(4 * scale)))
        style.configure("TButton", padding=(pad_x, pad_y))
        style.configure("TCombobox", padding=(max(6, int(round(4 * scale))), pad_y))

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=(8, 8, 8, 4))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(4, weight=1)

        ttk.Label(top, text="Episode").grid(row=0, column=0, sticky="w")
        self.episode_var = tk.StringVar()
        self.episode_combo = ttk.Combobox(
            top,
            textvariable=self.episode_var,
            values=[episode.name for episode in self.episodes],
            state="readonly",
            width=16,
        )
        self.episode_combo.grid(row=0, column=1, sticky="w", padx=(4, 8))
        self.episode_combo.bind("<<ComboboxSelected>>", self.on_episode_selected)

        ttk.Button(top, text="Prev", command=lambda: self.step_episode(-1)).grid(row=0, column=2, padx=2)
        ttk.Button(top, text="Next", command=lambda: self.step_episode(1)).grid(row=0, column=3, padx=2)

        self.status_var = tk.StringVar()
        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=4, sticky="e")

        body = ttk.Frame(self.root, padding=(8, 4, 8, 4))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=0)

        self.image_label = ttk.Label(body, anchor="nw")
        self.image_label.grid(row=0, column=0, sticky="nw")

        side = ttk.Frame(body)
        side.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(10, 0))
        side.rowconfigure(1, weight=1)

        ttk.Label(side, text="Subtasks").grid(row=0, column=0, sticky="w")
        self.subtask_list = tk.Listbox(side, height=12, width=32, exportselection=False)
        self.subtask_list.grid(row=1, column=0, sticky="ns")
        self.subtask_list.bind("<<ListboxSelect>>", self.on_subtask_selected)

        subtask_buttons = ttk.Frame(side)
        subtask_buttons.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(subtask_buttons, text="Set Start", command=self.set_start).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(subtask_buttons, text="Set End", command=self.set_end).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(subtask_buttons, text="Clear", command=self.clear_current).grid(row=0, column=2, sticky="ew", padx=2)

        self.selected_subtask_var = tk.StringVar()
        ttk.Label(side, textvariable=self.selected_subtask_var, wraplength=260).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(10, 0),
        )

        controls = ttk.Frame(body, padding=(0, 4, 0, 0))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)

        self.frame_var = tk.DoubleVar(value=0)
        slider_scale = max(0.75, float(self.ui_scale))
        self.frame_scale = tk.Scale(
            controls,
            from_=0,
            to=1,
            variable=self.frame_var,
            orient=tk.HORIZONTAL,
            command=self.on_scale_changed,
            showvalue=False,
            resolution=1,
            sliderlength=max(36, int(round(28 * slider_scale))),
            width=max(24, int(round(16 * slider_scale))),
            borderwidth=1,
            highlightthickness=0,
        )
        self.frame_scale.grid(row=0, column=0, columnspan=11, sticky="ew", pady=(0, 8))
        self.frame_scale.bind("<Button-1>", self.on_scale_pointer)
        self.frame_scale.bind("<B1-Motion>", self.on_scale_pointer)
        self.frame_scale.bind("<ButtonRelease-1>", self.on_scale_pointer)

        self.time_var = tk.StringVar()
        ttk.Label(controls, textvariable=self.time_var, width=18).grid(row=1, column=0, sticky="w", padx=(0, 8))
        ttk.Button(controls, text="-10", command=lambda: self.step_frame(-10)).grid(row=1, column=1, padx=2)
        ttk.Button(controls, text="-1", command=lambda: self.step_frame(-1)).grid(row=1, column=2, padx=2)
        self.play_button = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_button.grid(row=1, column=3, padx=2)
        ttk.Button(controls, text="+1", command=lambda: self.step_frame(1)).grid(row=1, column=4, padx=2)
        ttk.Button(controls, text="+10", command=lambda: self.step_frame(10)).grid(row=1, column=5, padx=2)
        ttk.Button(controls, text="Prev Task", command=lambda: self.step_subtask(-1)).grid(row=1, column=6, padx=(8, 2))
        ttk.Button(controls, text="Next Task", command=lambda: self.step_subtask(1)).grid(row=1, column=7, padx=2)
        ttk.Button(controls, text="Save", command=self.save).grid(row=1, column=8, padx=(8, 2))
        ttk.Button(controls, text="Quit", command=self.close).grid(row=1, column=9, padx=2)

    def _bind_keys(self) -> None:
        self.root.bind_all("<space>", self.on_space_key)
        self.root.bind_class("TButton", "<space>", self.on_space_key)
        self.root.bind_class("TButton", "<KeyRelease-space>", self.consume_space_key)
        self.root.bind_class("Button", "<space>", self.on_space_key)
        self.root.bind_class("Button", "<KeyRelease-space>", self.consume_space_key)
        self.root.bind("<Left>", lambda _event: self.step_frame(-1))
        self.root.bind("<Right>", lambda _event: self.step_frame(1))
        self.root.bind("<Up>", lambda _event: self.step_frame(10))
        self.root.bind("<Down>", lambda _event: self.step_frame(-10))
        self.root.bind(",", lambda _event: self.step_frame(-1))
        self.root.bind(".", lambda _event: self.step_frame(1))
        self.root.bind("a", lambda _event: self.step_frame(-10))
        self.root.bind("d", lambda _event: self.step_frame(10))
        self.root.bind("[", lambda _event: self.step_subtask(-1))
        self.root.bind("]", lambda _event: self.step_subtask(1))
        self.root.bind("s", lambda _event: self.set_start())
        self.root.bind("e", lambda _event: self.set_end())
        self.root.bind("r", lambda _event: self.clear_current())
        self.root.bind("w", lambda _event: self.save())
        self.root.bind("n", lambda _event: self.step_episode(1))
        self.root.bind("p", lambda _event: self.step_episode(-1))
        self.root.bind("q", lambda _event: self.close())
        self.root.bind("<Escape>", lambda _event: self.close())

    def on_space_key(self, _event: tk.Event) -> str:
        self.toggle_play()
        return "break"

    def consume_space_key(self, _event: tk.Event) -> str:
        return "break"

    def run(self) -> None:
        self.root.mainloop()

    def close_captures(self) -> None:
        for cap in self.captures.values():
            cap.close()
        self.captures = {}

    def load_episode(self, episode_pos: int) -> None:
        self.playing = False
        self.play_button.configure(text="Play")
        self.close_captures()
        self.episode_pos = clamp(episode_pos, 0, len(self.episodes) - 1)
        self.frame_idx = 0
        self.int_state = load_episode_int_state(self.episode)
        self.captures = open_captures(self.episode)
        self.episode_var.set(self.episode.name)
        self.frame_scale.configure(to=max(0, self.episode.num_frames - 1))
        self.update_current_episode_metrics()
        self.refresh_subtask_list()
        self.select_subtask(self.subtask_idx)
        self.render()

    def update_current_episode_metrics(self) -> None:
        update_episode_annotation_metrics(self.annotations, self.int_state, self.episode.num_frames)

    def update_all_episode_metrics(self) -> None:
        current_key = self.episode.name
        for episode in self.episodes:
            ep_payload = ensure_episode_payload(self.output_payload, episode, self.subtasks, self.fps)
            if episode.name == current_key:
                int_state = self.int_state
            else:
                int_state = load_episode_int_state(episode)
            update_episode_annotation_metrics(ep_payload["annotations"], int_state, episode.num_frames)

    def refresh_subtask_list(self) -> None:
        self.subtask_list.delete(0, tk.END)
        for annotation in self.annotations:
            start = annotation.get("start_frame")
            end = annotation.get("end_frame")
            prefix = "*" if start is not None and end is not None else " "
            self.subtask_list.insert(
                tk.END,
                f"{prefix} {annotation['subtask_index'] + 1}. {annotation['subtask']} [{start}, {end}]",
            )

    def select_subtask(self, idx: int) -> None:
        self.subtask_idx = clamp(idx, 0, len(self.annotations) - 1)
        self.subtask_list.selection_clear(0, tk.END)
        self.subtask_list.selection_set(self.subtask_idx)
        self.subtask_list.activate(self.subtask_idx)
        self.update_subtask_label()

    def update_subtask_label(self) -> None:
        annotation = self.annotations[self.subtask_idx]
        self.selected_subtask_var.set(
            f"Selected: {annotation['subtask']}\n"
            f"Start: {annotation['start_frame']}   End: {annotation['end_frame']}"
        )

    def on_subtask_selected(self, _event: object) -> None:
        selection = self.subtask_list.curselection()
        if selection:
            self.subtask_idx = int(selection[0])
            self.update_subtask_label()
            self.render()

    def on_episode_selected(self, _event: object) -> None:
        selected_name = self.episode_var.get()
        for idx, episode in enumerate(self.episodes):
            if episode.name == selected_name:
                self.save()
                self.load_episode(idx)
                return

    def on_scale_changed(self, value: str) -> None:
        if self.suppress_scale_callback:
            return
        self.frame_idx = clamp(int(round(float(value))), 0, self.episode.num_frames - 1)
        self.render()

    def on_scale_pointer(self, event: tk.Event) -> str:
        width = max(1, int(event.widget.winfo_width()))
        usable_left = max(0, int(round(14 * self.ui_scale)))
        usable_right = max(usable_left + 1, width - usable_left)
        x = clamp(int(event.x), usable_left, usable_right)
        ratio = (x - usable_left) / max(1, usable_right - usable_left)
        frame_idx = int(round(ratio * (self.episode.num_frames - 1)))
        self.set_frame(frame_idx)
        return "break"

    def set_frame(self, frame_idx: int) -> None:
        self.frame_idx = clamp(frame_idx, 0, self.episode.num_frames - 1)
        self.suppress_scale_callback = True
        self.frame_var.set(self.frame_idx)
        self.suppress_scale_callback = False
        self.render()

    def step_frame(self, delta: int) -> None:
        self.set_frame(self.frame_idx + delta)

    def step_subtask(self, delta: int) -> None:
        self.select_subtask(self.subtask_idx + delta)
        self.render()

    def step_episode(self, delta: int) -> None:
        new_pos = clamp(self.episode_pos + delta, 0, len(self.episodes) - 1)
        if new_pos != self.episode_pos:
            self.save()
            self.load_episode(new_pos)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self.schedule_next_frame()

    def schedule_next_frame(self) -> None:
        if not self.playing:
            return
        if self.frame_idx >= self.episode.num_frames - 1:
            self.playing = False
            self.play_button.configure(text="Play")
            return
        self.set_frame(self.frame_idx + 1)
        delay_ms = max(1, int(round(1000.0 / self.fps)))
        self.root.after(delay_ms, self.schedule_next_frame)

    def set_start(self) -> None:
        set_annotation_frame(self.annotations[self.subtask_idx], "start_frame", self.frame_idx, self.fps)
        self.update_current_episode_metrics()
        self.refresh_subtask_list()
        self.select_subtask(self.subtask_idx)
        self.render()

    def set_end(self) -> None:
        set_annotation_frame(self.annotations[self.subtask_idx], "end_frame", self.frame_idx, self.fps)
        self.update_current_episode_metrics()
        self.refresh_subtask_list()
        self.select_subtask(min(self.subtask_idx + 1, len(self.annotations) - 1))
        self.render()

    def clear_current(self) -> None:
        annotation = self.annotations[self.subtask_idx]
        annotation["start_frame"] = None
        annotation["end_frame"] = None
        update_times(annotation, self.fps)
        clear_annotation_metrics(annotation)
        self.refresh_subtask_list()
        self.select_subtask(self.subtask_idx)
        self.render()

    def save(self) -> None:
        self.update_all_episode_metrics()
        write_json(self.output_path, self.output_payload)
        self.status_var.set(f"Saved: {self.output_path}")

    def close(self) -> None:
        self.save()
        self.close_captures()
        self.root.destroy()

    def render(self) -> None:
        frames = read_video_frame(self.captures, self.frame_idx)
        display = compose_display(
            frames=frames,
            camera_keys=self.camera_keys,
            episode=self.episode,
            frame_idx=self.frame_idx,
            fps=self.fps,
            int_state=self.int_state,
            annotations=self.annotations,
            active_subtask_idx=self.subtask_idx,
            display_height=self.display_height,
            max_display_width=self.max_display_width,
        )
        image = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=self.photo)
        self.time_var.set(f"{self.frame_idx}/{self.episode.num_frames - 1}  {self.frame_idx / self.fps:.2f}s")
        state_text = "N/A"
        if self.int_state is not None and self.frame_idx < len(self.int_state):
            state = int(self.int_state[self.frame_idx])
            state_text = f"{state} ({INT_STATE_LABELS.get(state, 'unknown')})"
        self.status_var.set(
            f"{self.episode.name} | int_state {state_text} | "
            f"{self.subtask_idx + 1}/{len(self.annotations)} {self.annotations[self.subtask_idx]['subtask']}"
        )


def main() -> int:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.repo_id, args.lerobot_home, args.download)

    info = load_info(dataset_path)
    task_name = infer_task_name(dataset_path, args.task)
    subtasks = SUBTASK_NAME_DICT[task_name]
    camera_keys = args.camera_keys or find_video_keys(dataset_path, info)
    if len(camera_keys) != 3:
        print(
            f"Warning: expected 3 camera views, found {len(camera_keys)}: {camera_keys}",
            file=sys.stderr,
        )

    fps = float(args.fps or info.get("fps") or 30.0)
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")

    output_path = (args.output or default_output_path(args.repo_id)).expanduser().resolve()
    episodes = discover_episodes(dataset_path, camera_keys)
    if not episodes:
        raise ValueError(f"No episodes found in {dataset_path}")

    output_payload = make_output_payload(args.repo_id, dataset_path, task_name, subtasks, camera_keys, fps, output_path)
    for episode in episodes:
        ensure_episode_payload(output_payload, episode, subtasks, fps)

    print("Controls:")
    print("  Use the GUI buttons or drag the frame slider.")
    print("  Keyboard shortcuts still work: space, arrows, s/e, [/], n/p, w, q.")

    app = SubtaskAnnotatorApp(
        episodes=episodes,
        camera_keys=camera_keys,
        output_payload=output_payload,
        output_path=output_path,
        subtasks=subtasks,
        fps=fps,
        display_height=args.display_height,
        max_display_width=args.max_display_width,
        window_width=args.window_width,
        window_height=args.window_height,
        ui_scale=args.ui_scale,
        font_size=args.font_size,
        initial_episode_index=args.episode,
    )
    app.run()
    write_json(output_path, output_payload)
    print(f"Saved annotations to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

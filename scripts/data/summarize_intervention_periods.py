#!/usr/bin/env python3
"""Summarize contiguous intervention periods in LeRobot datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count contiguous int_state==2 intervention periods in LeRobot datasets."
    )
    parser.add_argument(
        "repo_ids",
        nargs="+",
        help="Dataset repo ids. Bare names are interpreted as <default-namespace>/<name>.",
    )
    parser.add_argument(
        "--default-namespace",
        default="wensi-ai",
        help="Namespace used for bare dataset names. Default: wensi-ai.",
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
        help="Download missing datasets from Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "annotation" / "intervention_periods",
        help="Directory for one JSON summary per dataset.",
    )
    parser.add_argument(
        "--min-intervention-length",
        type=int,
        default=15,
        help="Ignore intervention periods shorter than this many frames. Default: 15.",
    )
    return parser.parse_args()


def default_lerobot_home() -> Path:
    env_home = os.environ.get("HF_LEROBOT_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".cache" / "huggingface" / "lerobot"


def normalize_repo_id(repo_id: str, default_namespace: str) -> str:
    if "/" in repo_id:
        return repo_id
    return f"{default_namespace}/{repo_id}"


def resolve_dataset_path(repo_id: str, lerobot_home: Path | None, download: bool) -> Path:
    if repo_id.startswith("/") or repo_id.startswith("."):
        raise ValueError(f"Expected a Hugging Face repo id, got path-like value: {repo_id}")

    parts = Path(repo_id).parts
    if len(parts) < 2:
        raise ValueError(f"Expected repo id in '<namespace>/<dataset>' form, got: {repo_id}")

    home = (lerobot_home or default_lerobot_home()).expanduser().resolve()
    dataset_path = home.joinpath(*parts)
    if dataset_path.exists():
        return dataset_path
    if not download:
        raise FileNotFoundError(
            f"Dataset '{repo_id}' was not found at {dataset_path}. Use --download to fetch it."
        )

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Downloading requires huggingface_hub. Install it or download the dataset manually."
        ) from exc

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} to {dataset_path}")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dataset_path)
    return dataset_path


def episode_index(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return 10**9


def load_fps(dataset_path: Path) -> float | None:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return None
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    fps = info.get("fps")
    return float(fps) if fps else None


def int_state_values(parquet_path: Path) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=["int_state"])
    values = []
    for value in df["int_state"]:
        arr = np.asarray(value).reshape(-1)
        values.append(int(arr[0]) if arr.size else -1)
    return np.asarray(values, dtype=np.int32)


def intervention_periods(states: np.ndarray, min_length: int) -> list[dict[str, int]]:
    mask = states == 2
    periods = []

    def maybe_add_period(start_frame: int, end_frame: int) -> None:
        length = end_frame - start_frame + 1
        if length >= min_length:
            periods.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "length_frames": length,
                }
            )

    start = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            maybe_add_period(start, idx - 1)
            start = None
    if start is not None:
        maybe_add_period(start, len(mask) - 1)
    return periods


def summarize_dataset(repo_id: str, dataset_path: Path, min_intervention_length: int) -> dict[str, Any]:
    fps = load_fps(dataset_path)
    parquets = sorted((dataset_path / "data").glob("chunk-*/episode_*.parquet"), key=episode_index)
    if not parquets:
        raise FileNotFoundError(f"No LeRobot parquet episodes found under {dataset_path / 'data'}")

    episodes = []
    all_lengths = []
    total_frames = 0
    total_intervention_frames = 0
    for parquet_path in parquets:
        states = int_state_values(parquet_path)
        periods = intervention_periods(states, min_intervention_length)
        for period in periods:
            if fps:
                period["length_sec"] = round(period["length_frames"] / fps, 4)
            all_lengths.append(period["length_frames"])

        intervention_frames = int(sum(period["length_frames"] for period in periods))
        raw_intervention_frames = int(np.count_nonzero(states == 2))
        total_frames += int(states.size)
        total_intervention_frames += intervention_frames
        episodes.append(
            {
                "episode": parquet_path.stem,
                "num_frames": int(states.size),
                "num_intervention_periods": len(periods),
                "intervention_frames": intervention_frames,
                "raw_intervention_frames": raw_intervention_frames,
                "periods": periods,
            }
        )

    summary = {
        "repo_id": repo_id,
        "dataset_path": str(dataset_path),
        "fps": fps,
        "min_intervention_length": min_intervention_length,
        "num_episodes": len(episodes),
        "num_frames": total_frames,
        "num_intervention_periods": len(all_lengths),
        "intervention_frames": total_intervention_frames,
        "intervention_lengths_frames": all_lengths,
        "episodes": episodes,
    }
    if fps:
        summary["intervention_lengths_sec"] = [round(length / fps, 4) for length in all_lengths]
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\n{summary['repo_id']}")
    print(
        "  episodes={num_episodes} frames={num_frames} intervention_periods={num_intervention_periods} "
        "intervention_frames={intervention_frames}".format(**summary)
    )
    fps = summary.get("fps")
    for episode in summary["episodes"]:
        lengths = [period["length_frames"] for period in episode["periods"]]
        if fps:
            lengths_text = ", ".join(
                f"{period['length_frames']}f/{period['length_sec']:.2f}s"
                for period in episode["periods"]
            )
        else:
            lengths_text = ", ".join(f"{length}f" for length in lengths)
        if not lengths_text:
            lengths_text = "-"
        print(
            f"  {episode['episode']}: periods={episode['num_intervention_periods']} "
            f"lengths=[{lengths_text}]"
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def output_path_for_repo(output_dir: Path, repo_id: str) -> Path:
    filename = repo_id.replace("/", "__") + ".json"
    return output_dir / filename


def main() -> int:
    args = parse_args()
    errors = []
    output_dir = args.output_dir.expanduser().resolve()
    for raw_repo_id in args.repo_ids:
        repo_id = normalize_repo_id(raw_repo_id, args.default_namespace)
        output_path = output_path_for_repo(output_dir, repo_id)
        try:
            dataset_path = resolve_dataset_path(repo_id, args.lerobot_home, args.download)
            summary = summarize_dataset(repo_id, dataset_path, args.min_intervention_length)
        except Exception as exc:
            print(f"\n{repo_id}")
            print(f"  ERROR: {exc}")
            error_payload = {"repo_id": repo_id, "error": str(exc)}
            write_json(output_path, error_payload)
            print(f"  Saved error JSON to {output_path}")
            errors.append(error_payload)
            continue
        print_summary(summary)
        write_json(output_path, summary)
        print(f"  Saved JSON summary to {output_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Online evaluation script that discovers unevaluated checkpoints and runs eval on them.

Scans output/<data>/<time>/<run>/ckpt folders for .pt files and checks if corresponding
output/<data>/<time>/<run>/eval/<ckpt_name>.json exists. If not, runs evaluation.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import threading
import yaml
from iiil.tasks import TASK_NAME_DICT
from iiil.utils.policy_utils import ARCH_MAP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def find_unevaluated_checkpoints(output_folder: Path) -> list[tuple[Path, Path]]:
    """
    Scan output folder for unevaluated checkpoints.

    Returns list of tuples: (checkpoint_path, results_json_path)
    where results_json_path is the central eval/results.json tracking all evals.
    """
    unevaluated = []

    def _step_key(name: str) -> int:
        """Extract numeric step from ckpt_name like 'step_<n>-...'."""
        m = re.search(r"step_(\d+)", name)
        return int(m.group(1)) if m else -1

    # Iterate through output/<data>/<time>/ directories
    output_folder = Path(output_folder)
    if not output_folder.exists():
        print(f"Error: Output folder not found: {output_folder}")
        return unevaluated

    # Find all .pth checkpoint files in subdirectories
    # Walk directory tree and prune top-level old* folders to avoid expensive traversal
    for root, dirs, files in os.walk(output_folder):
        root_path = Path(root)
        # Only at the top-level, exclude any subfolders starting with 'old'
        if root_path == output_folder:
            dirs[:] = [d for d in dirs if not d.startswith("old")]
        # Also avoid entering any 'eval' folders anywhere
        dirs[:] = [d for d in dirs if d != "eval"]

        # Process checkpoint files in this directory
        for fname in files:
            if not fname.endswith(".pth"):
                continue
            ckpt_file = root_path / fname
            ckpt_name = ckpt_file.stem

            # Skip any checkpoints under paths containing 'EmptyTask' or "residual"
            if "EmptyTask" in str(ckpt_file) or "residual" in str(ckpt_file):
                continue

            # Skip checkpoints with step number <= 50000
            step_num = _step_key(ckpt_name)
            if step_num <= 50000:
                continue

            try:
                # Central results.json should be at same level as the checkpoint parent
                # Pattern: output/<data>/<time>/eval/results.json
                results_json = ckpt_file.parent.parent / "eval" / "results.json"

                # Create results.json if it doesn't exist
                results_json.parent.mkdir(parents=True, exist_ok=True)
                if not results_json.exists():
                    with open(results_json, "w") as f:
                        json.dump({}, f)

                # Load results.json to check if this checkpoint was already evaluated
                already_evaluated = False
                if results_json.exists():
                    try:
                        with open(results_json, "r") as f:
                            results_data = json.load(f) or {}
                        already_evaluated = ckpt_name in results_data
                    except Exception:
                        pass

                needs_eval = not already_evaluated

                if needs_eval:
                    unevaluated.append((ckpt_file, results_json))
                    # print(f"Found unevaluated checkpoint: {ckpt_file}")
            except Exception as e:
                print(f"Warning: Error processing {ckpt_file}: {e}")

    # Sort by step count ascending so we process in order
    return sorted(unevaluated, key=lambda t: _step_key(t[0].stem), reverse=True)


def _stream_output(prefix: str, process: subprocess.Popen) -> None:
    """Stream a subprocess' stdout to the current process with a prefix."""
    try:
        if process.stdout is None:
            return
        for line in process.stdout:
            print(f"[{prefix}] {line.rstrip()}")
    except Exception as e:
        print(f"[{prefix}] Error reading output: {e}")


def log_eval_to_wandb(
    config: dict, ckpt_path: Path, sr: float | None, video_path: Path | None, results: dict | None
) -> None:
    """Log eval metrics to the wandb run that produced this checkpoint."""
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed")
    if not config.get("use_wandb"):
        return

    wandb_project = config.get("wandb_project")
    assert wandb_project, "use_wandb is true but wandb_project missing from config"
    assert sr is not None, "SR is None (eval failed)"

    # run_name IS the wandb run ID: outputs/<run_name>/ckpt/step_50000.pth
    wandb_run_id = ckpt_path.parent.parent.name
    wandb_entity = config.get("wandb_entity")

    # Extract step number from checkpoint filename
    step_match = re.search(r"step_(\d+)", ckpt_path.stem)
    step = int(step_match.group(1)) if step_match else None

    init_kwargs = dict(id=wandb_run_id, project=wandb_project, resume="allow")
    if wandb_entity:
        init_kwargs["entity"] = wandb_entity
    wandb.init(**init_kwargs)

    try:
        log_dict: dict = {"eval/success_rate": sr}

        if video_path and video_path.exists():
            log_dict["eval/video"] = wandb.Video(str(video_path))

        if results:
            for key, val in results.items():
                if key == "SR":
                    continue
                if isinstance(val, (int, float)):
                    log_dict[f"eval/{key}"] = val

        wandb.log(log_dict, step=step)
        print(f"[WANDB] Logged eval metrics to run {wandb_run_id} at step {step}")
    finally:
        wandb.finish()


def save_eval_plot(results_json_path: Path) -> None:
    """Save a success rate vs training step plot next to results.json."""
    with open(results_json_path, "r") as f:
        results_data = json.load(f) or {}

    steps, srs = [], []
    for key, sr in results_data.items():
        m = re.search(r"step_(\d+)", key)
        if m:
            steps.append(int(m.group(1)))
            srs.append(float(sr))

    if not steps:
        return

    # Sort by step
    paired = sorted(zip(steps, srs))
    steps, srs = zip(*paired)

    run_name = results_json_path.parent.parent.name

    plt.figure()
    plt.plot(steps, srs, marker="o")
    plt.xlabel("Training Step")
    plt.ylabel("Success Rate")
    plt.title(run_name)
    plt.ylim(0, 1)

    plot_path = results_json_path.parent / "results.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[RESULTS] Saved plot to {plot_path}")


def run_eval(
    ckpt_path: Path,
    eval_output_path: Path,
    iiil_root: Path,
    il_lib_root: Path,
    no_wandb: bool = False,
) -> bool:
    """
    Run evaluation on a checkpoint by launching two processes:
    1. serve.py (websocket policy server) - runs in background
    2. run_task.py (evaluation task) - waits for completion

    Args:
        ckpt_path: Path to the .pt checkpoint
        eval_output_path: Where to save the eval results JSON

    Returns:
        True if eval succeeded, False otherwise
    """
    serve_process = None
    try:
        # Ensure eval output directory exists
        eval_output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning evaluation on {ckpt_path}...")
        print(f"Output will be saved to {eval_output_path}")

        # Load config.yaml from the checkpoint directory to extract robot and task types
        config_path = ckpt_path.parent.parent / "conf.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        arch_name = config.get("arch_name", {})
        robot_type = config.get("robot", {}).get("type", "a1").lower()
        task_type = config.get("task", {}).get("type", "PickPlaceTask")

        print(f"[CONFIG] Robot: {robot_type}, Task: {task_type}")

        # Step 1: Start serve.py in background (websocket policy server)
        serve_args = [
            sys.executable,
            "serve.py",
            f"robot={robot_type}_iiil",
            f"task={TASK_NAME_DICT[task_type]}/base",
            f"arch={ARCH_MAP[arch_name]}",
            f"ckpt_path={ckpt_path}",
        ]

        print(f"[SERVE] Starting serve.py...")
        serve_env = os.environ.copy()
        serve_env.setdefault("PYTHONUNBUFFERED", "1")
        serve_process = subprocess.Popen(
            serve_args,
            cwd=il_lib_root,
            env=serve_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(
            target=_stream_output, args=("SERVE", serve_process), daemon=True
        ).start()

        # Step 2: Run run_task.py (will complete when evaluation is done)
        eval_results_file = eval_output_path.parent / f"{ckpt_path.stem}.json"
        task_args = [
            sys.executable,
            "run_task.py",
            f"env=sim_{robot_type}_joint",
            f"task={TASK_NAME_DICT[task_type]}/base",
            "policy=il",
            f"video_recording_path={eval_output_path.parent / 'latest.mp4'}",
            f"results_json_path={eval_results_file}",
        ]

        print(f"[TASK] Running run_task.py...")
        task_env = os.environ.copy()
        task_env.setdefault("PYTHONUNBUFFERED", "1")
        task_process = subprocess.Popen(
            task_args,
            cwd=iiil_root,
            env=task_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _stream_output("TASK", task_process)
        task_process.wait()

        # Step 3: Terminate serve.py after evaluation completes
        if serve_process and serve_process.poll() is None:
            print(f"[SERVE] Terminating serve.py...")
            serve_process.terminate()
            try:
                serve_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[SERVE] Force killing serve.py...")
                serve_process.kill()
                serve_process.wait()

        print(f"Evaluation ended for {ckpt_path.name}")
        # Extract SR from results file written by run_task.py
        sr = None
        results = None
        if eval_results_file.exists():
            try:
                with open(eval_results_file, "r") as f:
                    results = json.load(f)
                sr = results.get("SR")
                if sr is not None:
                    sr = float(sr)
                print(f"[RESULTS] Loaded SR={sr} from {eval_results_file}")
            except Exception as e:
                print(f"[RESULTS] Failed to read results file: {e}")

        # Update central results.json with this checkpoint's SR
        ckpt_name = ckpt_path.stem
        results_data = {}
        if eval_output_path.exists():
            try:
                with open(eval_output_path, "r") as f:
                    results_data = json.load(f) or {}
            except Exception:
                pass

        # Add/update entry for this checkpoint (just SR value, no nested dict)
        if sr is not None:
            results_data[ckpt_name] = sr

        # Write back sorted by key
        eval_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_output_path, "w") as f:
            json.dump(
                dict(
                    sorted(
                        results_data.items(),
                        key=lambda kv: int(re.search(r"step_(\d+)", kv[0]).group(1))
                        if re.search(r"step_(\d+)", kv[0])
                        else kv[0],
                    )
                ),
                f,
                indent=2,
            )
        print(f"[RESULTS] Updated {eval_output_path}")
        save_eval_plot(eval_output_path)

        # Log to wandb
        if not no_wandb:
            video_path = eval_output_path.parent / "latest.mp4"
            log_eval_to_wandb(config, ckpt_path, sr, video_path, results)

        # Clean up individual checkpoint results file if it exists
        if eval_results_file.exists():
            try:
                eval_results_file.unlink()
                print(f"[CLEANUP] Removed {eval_results_file}")
            except Exception as e:
                print(f"[CLEANUP] Failed to remove {eval_results_file}: {e}")

        return True

    except Exception as e:
        print(f"✗ Error running evaluation: {e}")
        # Cleanup serve.py if still running
        if serve_process and serve_process.poll() is None:
            serve_process.kill()
            serve_process.wait()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Discover and evaluate unevaluated checkpoints in output folder."
    )
    script_root = Path(__file__).resolve().parents[2]
    workspace_root = script_root.parent
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=workspace_root / "il_lib" / "outputs",
        help="Path to output folder (e.g., outputs or /path/to/outputs)",
    )
    parser.add_argument(
        "--iiil-root",
        type=Path,
        default=script_root,
        help="Path to the IIIL repo root containing run_task.py and scripts/",
    )
    parser.add_argument(
        "--il-lib-root",
        type=Path,
        default=workspace_root / "il_lib",
        help="Path to the il_lib repo root containing serve.py",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Only list checkpoints without running eval",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=600,
        help="Polling interval in seconds to rescan the output folder",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable logging eval results to wandb",
    )

    args = parser.parse_args()

    if args.dry_run:
        # Single scan in dry-run mode
        print(f"Scanning output folder: {args.output}")
        unevaluated = find_unevaluated_checkpoints(args.output)
        if not unevaluated:
            print("No unevaluated checkpoints found.")
        else:
            print(f"\nFound {len(unevaluated)} unevaluated checkpoint(s).")
        return

    print(f"Starting monitor. Output folder: {args.output}; interval: {args.interval}s")
    try:
        while True:
            print("\n--- Scan for unevaluated checkpoints ---")
            unevaluated = find_unevaluated_checkpoints(args.output)
            if not unevaluated:
                print("No unevaluated checkpoints. Sleeping...")
                time.sleep(args.interval)
                continue

            print(
                f"Found {len(unevaluated)} candidate checkpoint(s). Running one eval..."
            )
            ckpt_path, results_json_path = unevaluated[0]
            run_eval(
                ckpt_path,
                results_json_path,
                iiil_root=args.iiil_root,
                il_lib_root=args.il_lib_root,
                no_wandb=args.no_wandb,
            )
            # After each eval, immediately rescan; still only one eval at a time
    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")


if __name__ == "__main__":
    main()

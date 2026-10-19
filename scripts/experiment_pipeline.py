#!/usr/bin/env python3
"""End-to-end experiment orchestration for IIIL + il_lib.

This script is a reusable replacement for one-off study runners. It wires
together the repo's existing entrypoints:

- `run_task.py` for demo collection and evaluation
- `scripts/data/process_data.py` for intervention labels
- `../il_lib/train.py` for policy training
- `../il_lib/serve.py` for websocket serving during correction collection / eval

The default flow is:
1. Collect or load seed demos
2. Train a base policy on demos only
3. Optionally repeat correction collection, retraining, and evaluation for
   multiple rounds
4. Train `cotrain`, `hgdagger`, `sirius`, and/or `residual`
5. Evaluate the resulting checkpoints
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

AVAILABLE_METHODS = ("demos", "cotrain", "hgdagger", "sirius", "residual")
REWEIGHT_METHODS = {
    "cotrain": "cotrain",
    "hgdagger": "hg_daggar",
    "sirius": "sirius",
}
FULL_PHASES = (
    "seed_data",
    "base_train",
    "correction_data",
    "correction_train",
    "eval",
)
DEFAULT_SIM_ENV_BY_ROBOT = {
    "a1_iiil": "sim_a1_joint",
    "a1_7dof_iiil": "sim_a1_joint",
    "r1pro_iiil": "sim_r1pro_joint",
}
DEFAULT_BASE_ARCH_BY_ROBOT = {
    "a1_iiil": "dp_a1",
    "a1_7dof_iiil": "dp_a1_state",
    "r1pro_iiil": "wbvima_r1pro",
}
DEFAULT_RESIDUAL_ARCH_BY_ROBOT = {
    "a1_iiil": "residual_a1",
    "a1_7dof_iiil": "residual_a1_state",
    "r1pro_iiil": "residual_r1pro",
}
DEFAULT_BASE_DEMO_POLICY = "curobo"
SPEC_PATH_KEYS = {
    "root_dir",
    "iiil_dir",
    "il_lib_dir",
    "conda_sh",
    "base_demo_source",
}


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_experiment_name(robot: str, task: str) -> str:
    task_tag = task.replace("/", "_")
    return f"{robot}_{task_tag}"


def parse_csv(value: str | Iterable[object]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def parse_int_csv(value: str | Iterable[object]) -> list[int]:
    items = [int(item) for item in parse_csv(value)]
    if not items:
        raise ValueError("Expected at least one integer value")
    return items


def parse_methods(value: str | Iterable[object]) -> list[str]:
    methods = list(AVAILABLE_METHODS) if value == "all" else parse_csv(value)
    invalid = [method for method in methods if method not in AVAILABLE_METHODS]
    if invalid:
        raise ValueError(
            f"Unsupported method(s): {', '.join(invalid)}. "
            f"Expected one of: {', '.join(AVAILABLE_METHODS)}"
        )
    return methods


def resolve_iteration_method(
    explicit: Optional[str],
    methods: list[str],
    rounds: int,
) -> str:
    if explicit is not None:
        if explicit not in methods and explicit != "demos":
            raise ValueError(
                f"--iteration-method must be one of the selected methods or 'demos', got '{explicit}'"
            )
        if rounds > 1 and explicit == "residual":
            raise ValueError(
                "Iterative collection against 'residual' is not supported yet. "
                "Choose demos, cotrain, hgdagger, or sirius."
            )
        return explicit

    for candidate in ("cotrain", "hgdagger", "sirius", "demos"):
        if candidate in methods:
            return candidate
    return methods[0]


def round_tag(round_idx: int) -> str:
    return f"round_{round_idx:02d}"


def scope_tag(demo_tag: str, round_idx: int) -> str:
    return f"{demo_tag}_r{round_idx:02d}"


def parse_phases(value: str | Iterable[object]) -> set[str]:
    if value == "full":
        return set(FULL_PHASES)
    phases = set(parse_csv(value))
    invalid = phases.difference(FULL_PHASES)
    if invalid:
        raise ValueError(
            f"Unsupported phase(s): {', '.join(sorted(invalid))}. "
            f"Expected any of: {', '.join(FULL_PHASES)} or 'full'"
        )
    return phases


def parse_literalish(raw_value: str) -> object:
    stripped = raw_value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return stripped


@dataclass
class ExperimentSpec:
    path: Path
    values: dict[str, object]
    title: Optional[str]
    tags: list[str]
    notes: str


def _split_spec_text(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "---":
            header = "\n".join(lines[:idx])
            body = "\n".join(lines[idx + 1 :]).strip()
            return header, body
    return text, ""


def parse_experiment_card(text: str) -> tuple[dict[str, object], str]:
    header, notes = _split_spec_text(text)
    values: dict[str, object] = {}
    for lineno, line in enumerate(header.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(
                f"Invalid experiment spec line {lineno}: {line!r}. Expected 'key = value'."
            )
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid experiment spec line {lineno}: missing key.")
        values[key] = parse_literalish(raw_value)
    return values, notes


def normalize_optional_str_list(value: object) -> list[str]:
    if value is None:
        return []
    return parse_csv(value if isinstance(value, str) else list(value))


def load_experiment_spec(path: Path) -> ExperimentSpec:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Experiment spec not found: {resolved}")

    values, notes = parse_experiment_card(resolved.read_text(encoding="utf-8"))
    title = None if "title" not in values else str(values.pop("title"))
    tags = normalize_optional_str_list(values.pop("tags", []))

    for key in SPEC_PATH_KEYS:
        raw_value = values.get(key)
        if raw_value is None or not isinstance(raw_value, str):
            continue
        raw_path = Path(raw_value)
        if raw_path.is_absolute():
            values[key] = str(raw_path)
        else:
            values[key] = str((resolved.parent / raw_path).resolve())

    return ExperimentSpec(
        path=resolved,
        values=values,
        title=title,
        tags=tags,
        notes=notes,
    )


def infer_collection_task(base_task: str) -> str:
    if base_task.endswith("/base"):
        return base_task[:-5] + "/dc"
    raise ValueError(
        f"Cannot infer collection task from '{base_task}'. "
        "Pass --collection-task explicitly."
    )


def infer_collect_env(robot: str, collect_env: Optional[str]) -> str:
    if collect_env is not None:
        return collect_env
    try:
        return DEFAULT_SIM_ENV_BY_ROBOT[robot]
    except KeyError as exc:
        raise ValueError(
            f"No default collect/eval env for robot '{robot}'. "
            "Pass --collect-env / --eval-env explicitly."
        ) from exc


def infer_base_arch(robot: str, arch: Optional[str]) -> str:
    if arch is not None:
        return arch
    try:
        return DEFAULT_BASE_ARCH_BY_ROBOT[robot]
    except KeyError as exc:
        raise ValueError(
            f"No default base arch for robot '{robot}'. Pass --base-arch explicitly."
        ) from exc


def infer_residual_arch(robot: str, arch: Optional[str]) -> str:
    if arch is not None:
        return arch
    try:
        return DEFAULT_RESIDUAL_ARCH_BY_ROBOT[robot]
    except KeyError as exc:
        raise ValueError(
            f"No default residual arch for robot '{robot}'. Pass --residual-arch explicitly."
        ) from exc


def discover_conda_sh() -> Optional[Path]:
    if os.environ.get("CONDA_EXE"):
        conda_base = Path(os.environ["CONDA_EXE"]).resolve().parents[1]
        candidate = conda_base / "etc/profile.d/conda.sh"
        if candidate.exists():
            return candidate
    for candidate in (
        Path.home() / "miniforge3/etc/profile.d/conda.sh",
        Path.home() / "miniconda3/etc/profile.d/conda.sh",
        Path.home() / "anaconda3/etc/profile.d/conda.sh",
    ):
        if candidate.exists():
            return candidate
    return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_top_level_yaml_scalar(path: Path, key: str) -> Optional[str]:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.+?)\s*$")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if line[:1].isspace():
                continue
            match = pattern.match(stripped)
            if match:
                value = match.group(1).strip()
                if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                    value = value[1:-1]
                return value
    return None


def find_latest_run_dir(exp_root: Path, prefix: str) -> Optional[Path]:
    candidates = sorted(
        (path for path in exp_root.glob(f"{prefix}*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _checkpoint_loss(path: Path) -> float:
    match = re.search(r"-loss_([0-9.]+)\.pth$", path.name)
    return float(match.group(1)) if match else float("inf")


def find_best_checkpoint(run_dir: Path) -> Optional[Path]:
    ckpt_dir = run_dir / "ckpt"
    candidates = sorted(ckpt_dir.glob("*.pth"))
    if not candidates:
        return None
    return min(candidates, key=lambda path: (_checkpoint_loss(path), -path.stat().st_mtime))


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    ckpt_dir = run_dir / "ckpt"
    candidates = sorted(ckpt_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def validate_hdf5_has_demos(path: Path) -> bool:
    if h5py is None:
        raise ModuleNotFoundError("h5py is required to inspect HDF5 datasets.")
    if not path.exists():
        return False
    with h5py.File(path, "r") as f:
        return "data" in f and any(name.startswith("demo_") for name in f["data"].keys())


def validate_results_json(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return "SR" in payload


def materialize_subset_hdf5(input_path: Path, output_path: Path, n_demos: int) -> None:
    if h5py is None:
        raise ModuleNotFoundError("h5py is required to create HDF5 demo subsets.")
    source_files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.hdf5"))
    if not source_files:
        raise FileNotFoundError(f"No HDF5 files found under {input_path}")

    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demo_count = 0
    with h5py.File(output_path, "w") as dst:
        dst_data = dst.create_group("data")
        for source_file in source_files:
            with h5py.File(source_file, "r") as src:
                if "data" not in src:
                    continue
                demo_names = sorted(
                    (name for name in src["data"].keys() if name.startswith("demo_")),
                    key=lambda name: int(name.split("_", 1)[1]),
                )
                for demo_name in demo_names:
                    if demo_count >= n_demos:
                        break
                    src.copy(src["data"][demo_name], dst_data, name=f"demo_{demo_count}")
                    demo_count += 1
            if demo_count >= n_demos:
                break

    if demo_count < n_demos:
        raise RuntimeError(
            f"Requested {n_demos} demos but only found {demo_count} under {input_path}"
        )


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target_path)


def wait_for_port(host: str, port: int, timeout_s: float, process: Optional[subprocess.Popen] = None) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Server exited early while waiting for {host}:{port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
            return
        except OSError:
            time.sleep(1.0)
        finally:
            try:
                sock.close()
            except OSError:
                pass
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def read_deployed_action_steps(run_dir: Optional[Path], base_arch: str, iiil_dir: Path) -> int:
    conf_path = run_dir / "conf.yaml" if run_dir is not None else None
    if conf_path is not None and conf_path.exists():
        value = read_top_level_yaml_scalar(conf_path, "deployed_action_steps")
        if value is not None:
            return int(value)
    arch_conf = iiil_dir / "iiil/configs/arch" / f"{base_arch}.yaml"
    if arch_conf.exists():
        value = read_top_level_yaml_scalar(arch_conf, "deployed_action_steps")
        if value is not None:
            return int(value)
    return 1


class CommandRunner:
    def __init__(self, iiil_dir: Path, conda_sh: Optional[Path], dry_run: bool) -> None:
        self.iiil_dir = iiil_dir
        self.conda_sh = conda_sh
        self.dry_run = dry_run

    def _build_script(
        self,
        command: list[str],
        cwd: Path,
        env_name: Optional[str],
        extra_env: Optional[dict[str, str]] = None,
    ) -> str:
        lines = ["set -euo pipefail"]
        if env_name:
            if self.conda_sh is None:
                raise RuntimeError(
                    f"Need conda activation for env '{env_name}', but conda.sh was not found. "
                    "Pass --conda-sh explicitly or omit the env override."
                )
            lines.append(f"source {shlex.quote(str(self.conda_sh))}")
            lines.append(f"conda activate {shlex.quote(env_name)}")
        base_env = {
            "IIIL_BASE_PATH": str(self.iiil_dir),
            "I3L_BASE_PATH": str(self.iiil_dir),
            "IIIL_ENABLE_FLATCACHE": os.environ.get("IIIL_ENABLE_FLATCACHE", "0"),
            "IIIL_USE_GPU_DYNAMICS": os.environ.get("IIIL_USE_GPU_DYNAMICS", "0"),
            "IIIL_ENABLE_TRANSITION_RULES": os.environ.get("IIIL_ENABLE_TRANSITION_RULES", "1"),
            "TORCHDYNAMO_DISABLE": os.environ.get("TORCHDYNAMO_DISABLE", "1"),
            "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED", "1"),
            "OMNIGIBSON_HEADLESS": os.environ.get("OMNIGIBSON_HEADLESS", "1"),
        }
        if extra_env:
            base_env.update(extra_env)
        for key, value in base_env.items():
            lines.append(f"export {key}={shlex.quote(str(value))}")
        lines.append(f"cd {shlex.quote(str(cwd))}")
        lines.append(shlex.join([str(token) for token in command]))
        return "\n".join(lines)

    def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        log_path: Path,
        env_name: Optional[str] = None,
        extra_env: Optional[dict[str, str]] = None,
        success_validator: Optional[Callable[[], bool]] = None,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pretty = shlex.join([str(token) for token in command])
        env_label = env_name or "current"
        print(f"[run:{env_label}] (cd {cwd} && {pretty})")
        if self.dry_run:
            return

        script = self._build_script(command, cwd, env_name, extra_env)
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                ["/bin/bash", "-lc", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log_file.write(line)
            return_code = proc.wait()

        if return_code == 0:
            return
        if success_validator is not None and success_validator():
            print(
                f"[warn] command exited with status {return_code}, but expected artifact is present. Continuing."
            )
            return
        raise subprocess.CalledProcessError(return_code, command)

    def start_background(
        self,
        command: list[str],
        *,
        cwd: Path,
        log_path: Path,
        env_name: Optional[str] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> subprocess.Popen:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pretty = shlex.join([str(token) for token in command])
        env_label = env_name or "current"
        print(f"[serve:{env_label}] (cd {cwd} && {pretty})")
        if self.dry_run:
            class _DummyProc:
                pid = 0

                def poll(self) -> int | None:
                    return None

                def terminate(self) -> None:
                    return None

                def kill(self) -> None:
                    return None

                def wait(self, timeout: Optional[float] = None) -> int:
                    return 0

            return _DummyProc()  # type: ignore[return-value]

        script = self._build_script(command, cwd, env_name, extra_env)
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", script],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        setattr(proc, "_iiil_log_file", log_file)
        return proc

    @staticmethod
    def stop_background(proc: Optional[subprocess.Popen]) -> None:
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        finally:
            log_file = getattr(proc, "_iiil_log_file", None)
            if log_file is not None:
                log_file.close()


@dataclass
class ExperimentConfig:
    name: str
    root_dir: Path
    iiil_dir: Path
    il_lib_dir: Path
    launch_args: list[str]
    spec_path: Optional[Path]
    spec_title: Optional[str]
    spec_tags: list[str]
    spec_notes: str
    conda_sh: Optional[Path]
    collect_env_name: Optional[str]
    process_env_name: Optional[str]
    train_env_name: Optional[str]
    serve_env_name: Optional[str]
    eval_env_name: Optional[str]
    robot: str
    collect_env: str
    eval_env: str
    base_task: str
    collection_task: str
    base_arch: str
    residual_arch: str
    methods: list[str]
    phases: set[str]
    demo_counts: list[int]
    rounds: int
    iteration_method: str
    seed_demo_episodes: int
    correction_episodes: Optional[int]
    eval_episodes: int
    eval_repeats: int
    eval_repeat_start: int
    eval_repeat_end: int
    pre_intervention_length: int
    base_demo_source: Optional[Path]
    base_demo_policy: str
    base_collect_overrides: list[str]
    base_train_overrides: list[str]
    correction_policy: str
    correction_requires_base_server: bool
    correction_oracle_policy: Optional[str]
    correction_collect_overrides: list[str]
    reweight_train_overrides: list[str]
    residual_train_overrides: list[str]
    eval_overrides: list[str]
    train_gpus: int
    train_num_nodes: int
    train_max_steps: Optional[int]
    train_val_check_interval: Optional[int]
    host: str
    port: int
    use_wandb: bool
    train_cuda_visible_devices: Optional[str]
    sim_cuda_visible_devices: Optional[str]
    serve_cuda_visible_devices: Optional[str]
    force_overwrite: bool
    skip_collection: bool
    skip_eval: bool
    save_eval_video: bool
    dry_run: bool

    @property
    def train_root(self) -> Path:
        return self.root_dir / "train_runs"

    @property
    def data_root(self) -> Path:
        return self.root_dir / "datasets"

    @property
    def eval_root(self) -> Path:
        return self.root_dir / "eval"

    @property
    def results_root(self) -> Path:
        return self.root_dir / "results"

    @property
    def log_root(self) -> Path:
        return self.root_dir / "logs"

    @property
    def manifest_path(self) -> Path:
        return self.root_dir / "manifest.tsv"

    @property
    def config_dump_path(self) -> Path:
        return self.root_dir / "experiment_config.json"

    @property
    def run_card_path(self) -> Path:
        return self.root_dir / "RUN.md"

    @property
    def spec_copy_path(self) -> Optional[Path]:
        if self.spec_path is None:
            return None
        return self.root_dir / f"experiment_spec{self.spec_path.suffix or '.txt'}"

    @property
    def registry_path(self) -> Path:
        return self.root_dir.parent / "experiment_registry.tsv"

    @property
    def seed_root(self) -> Path:
        return self.data_root / "seed_master"

    @property
    def seed_demo_h5(self) -> Path:
        return self.seed_root / "seed_demos_master.hdf5"


@dataclass
class MethodArtifact:
    method: str
    arch: str
    data_dir: Path
    run_dir: Path
    checkpoint: Path
    round_idx: int
    base_checkpoint: Optional[Path] = None
    base_run_dir: Optional[Path] = None


class ExperimentPipeline:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.runner = CommandRunner(cfg.iiil_dir, cfg.conda_sh, cfg.dry_run)
        self.manifest_rows: set[tuple[str, str, str, str, str]] = set()
        self.registry_roots: set[str] = set()
        self._active_server: Optional[subprocess.Popen] = None
        self._ensure_layout()
        self._load_manifest()
        self._load_registry()
        self._write_config_dump()
        self._write_spec_copy()
        self._write_run_card()
        self._record_registry()

    def _ensure_layout(self) -> None:
        for path in (
            self.cfg.root_dir,
            self.cfg.train_root,
            self.cfg.data_root,
            self.cfg.eval_root,
            self.cfg.results_root,
            self.cfg.log_root,
            self.cfg.seed_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if not self.cfg.manifest_path.exists():
            self.cfg.manifest_path.write_text(
                "demo_tag\tmethod\tdata_dir\trun_dir\tcheckpoint\n",
                encoding="utf-8",
            )
        if not self.cfg.registry_path.exists():
            self.cfg.registry_path.write_text(
                "root_dir\tname\tspec_path\ttitle\ttags\n",
                encoding="utf-8",
            )

    def _load_manifest(self) -> None:
        if not self.cfg.manifest_path.exists():
            return
        lines = self.cfg.manifest_path.read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            parts = line.split("\t")
            if len(parts) == 5:
                self.manifest_rows.add(tuple(parts))

    def _load_registry(self) -> None:
        if not self.cfg.registry_path.exists():
            return
        lines = self.cfg.registry_path.read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            parts = line.split("\t")
            if parts:
                self.registry_roots.add(parts[0])

    def _write_config_dump(self) -> None:
        payload = {
            "name": self.cfg.name,
            "root_dir": str(self.cfg.root_dir),
            "iiil_dir": str(self.cfg.iiil_dir),
            "il_lib_dir": str(self.cfg.il_lib_dir),
            "launch_args": self.cfg.launch_args,
            "launch_command": shlex.join(["./experiment.sh", *self.cfg.launch_args]),
            "spec_path": str(self.cfg.spec_path) if self.cfg.spec_path else None,
            "spec_copy_path": str(self.cfg.spec_copy_path) if self.cfg.spec_copy_path else None,
            "spec_title": self.cfg.spec_title,
            "spec_tags": self.cfg.spec_tags,
            "spec_notes": self.cfg.spec_notes,
            "robot": self.cfg.robot,
            "collect_env": self.cfg.collect_env,
            "eval_env": self.cfg.eval_env,
            "base_task": self.cfg.base_task,
            "collection_task": self.cfg.collection_task,
            "base_arch": self.cfg.base_arch,
            "residual_arch": self.cfg.residual_arch,
            "methods": self.cfg.methods,
            "phases": sorted(self.cfg.phases),
            "demo_counts": self.cfg.demo_counts,
            "rounds": self.cfg.rounds,
            "iteration_method": self.cfg.iteration_method,
            "seed_demo_episodes": self.cfg.seed_demo_episodes,
            "correction_episodes": self.cfg.correction_episodes,
            "eval_episodes": self.cfg.eval_episodes,
            "eval_repeats": self.cfg.eval_repeats,
            "eval_repeat_start": self.cfg.eval_repeat_start,
            "eval_repeat_end": self.cfg.eval_repeat_end,
            "pre_intervention_length": self.cfg.pre_intervention_length,
            "base_demo_source": str(self.cfg.base_demo_source) if self.cfg.base_demo_source else None,
            "base_demo_policy": self.cfg.base_demo_policy,
            "correction_policy": self.cfg.correction_policy,
            "correction_requires_base_server": self.cfg.correction_requires_base_server,
            "correction_oracle_policy": self.cfg.correction_oracle_policy,
            "use_wandb": self.cfg.use_wandb,
            "host": self.cfg.host,
            "port": self.cfg.port,
            "force_overwrite": self.cfg.force_overwrite,
            "skip_collection": self.cfg.skip_collection,
            "skip_eval": self.cfg.skip_eval,
            "save_eval_video": self.cfg.save_eval_video,
            "dry_run": self.cfg.dry_run,
            "collect_env_name": self.cfg.collect_env_name,
            "process_env_name": self.cfg.process_env_name,
            "train_env_name": self.cfg.train_env_name,
            "serve_env_name": self.cfg.serve_env_name,
            "eval_env_name": self.cfg.eval_env_name,
            "base_collect_overrides": self.cfg.base_collect_overrides,
            "base_train_overrides": self.cfg.base_train_overrides,
            "correction_collect_overrides": self.cfg.correction_collect_overrides,
            "reweight_train_overrides": self.cfg.reweight_train_overrides,
            "residual_train_overrides": self.cfg.residual_train_overrides,
            "eval_overrides": self.cfg.eval_overrides,
        }
        write_json(self.cfg.config_dump_path, payload)

    def _write_spec_copy(self) -> None:
        if self.cfg.spec_path is None or self.cfg.spec_copy_path is None:
            return
        self.cfg.spec_copy_path.write_text(
            self.cfg.spec_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    def _write_run_card(self) -> None:
        lines = [
            f"# {self.cfg.spec_title or self.cfg.name}",
            "",
            "## Run",
            f"- Name: `{self.cfg.name}`",
            f"- Root: `{self.cfg.root_dir}`",
            f"- Launch: `{shlex.join(['./experiment.sh', *self.cfg.launch_args])}`",
            "",
            "## Pipeline",
            f"- Robot: `{self.cfg.robot}`",
            f"- Base task: `{self.cfg.base_task}`",
            f"- Demo counts: `{','.join(str(item) for item in self.cfg.demo_counts)}`",
            f"- Rounds: `{self.cfg.rounds}`",
            f"- Methods: `{','.join(self.cfg.methods)}`",
            f"- Iteration method: `{self.cfg.iteration_method}`",
            "",
            "## Tracking",
            f"- Config dump: `{self.cfg.config_dump_path}`",
            f"- Manifest: `{self.cfg.manifest_path}`",
            f"- Registry: `{self.cfg.registry_path}`",
        ]
        if self.cfg.spec_path is not None:
            lines.extend(
                [
                    f"- Source spec: `{self.cfg.spec_path}`",
                    f"- Copied spec: `{self.cfg.spec_copy_path}`",
                ]
            )
        if self.cfg.spec_tags:
            lines.extend(["", "## Tags", f"`{', '.join(self.cfg.spec_tags)}`"])
        if self.cfg.spec_notes:
            lines.extend(["", "## Notes", self.cfg.spec_notes])
        self.cfg.run_card_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _record_registry(self) -> None:
        root_key = str(self.cfg.root_dir)
        if root_key in self.registry_roots:
            return
        row = (
            root_key,
            self.cfg.name,
            str(self.cfg.spec_path) if self.cfg.spec_path else "",
            self.cfg.spec_title or "",
            ",".join(self.cfg.spec_tags),
        )
        with open(self.cfg.registry_path, "a", encoding="utf-8") as f:
            f.write("\t".join(row) + "\n")
        self.registry_roots.add(root_key)

    def _record_manifest(
        self,
        demo_tag: str,
        method: str,
        data_dir: Path,
        run_dir: Path,
        checkpoint: Path,
    ) -> None:
        row = (demo_tag, method, str(data_dir), str(run_dir), str(checkpoint))
        if row in self.manifest_rows:
            return
        with open(self.cfg.manifest_path, "a", encoding="utf-8") as f:
            f.write("\t".join(row) + "\n")
        self.manifest_rows.add(row)

    def cleanup(self) -> None:
        self._stop_server()

    def _start_server(
        self,
        command: list[str],
        log_path: Path,
        *,
        port: int,
    ) -> None:
        self._stop_server()
        extra_env = {}
        if self.cfg.serve_cuda_visible_devices:
            extra_env["CUDA_VISIBLE_DEVICES"] = self.cfg.serve_cuda_visible_devices
        extra_env["IL_LIB_WEBSOCKET_PORT"] = str(port)
        self._active_server = self.runner.start_background(
            command,
            cwd=self.cfg.il_lib_dir,
            log_path=log_path,
            env_name=self.cfg.serve_env_name,
            extra_env=extra_env,
        )
        if self.cfg.dry_run:
            return
        wait_for_port(self.cfg.host, port, 180.0, self._active_server)

    def _stop_server(self) -> None:
        self.runner.stop_background(self._active_server)
        self._active_server = None

    def _train_method(
        self,
        *,
        prefix: str,
        arch: str,
        data_dir: Path,
        log_path: Path,
        train_overrides: list[str],
    ) -> tuple[Path, Path]:
        if not self.cfg.force_overwrite:
            existing_dir = find_latest_run_dir(self.cfg.train_root, prefix)
            if existing_dir is not None:
                existing_ckpt = find_best_checkpoint(existing_dir) or find_latest_checkpoint(existing_dir)
                if existing_ckpt is not None:
                    print(f"[skip] reusing checkpoint for {prefix}: {existing_ckpt}")
                    return existing_dir, existing_ckpt

        command = [
            "python",
            "train.py",
            f"arch={arch}",
            f"robot={self.cfg.robot}",
            f"task={self.cfg.base_task}",
            f"data_dir={data_dir}",
            f"exp_root_dir={self.cfg.train_root}",
            f"run_name={prefix}",
            f"use_wandb={_bool_str(self.cfg.use_wandb)}",
            f"gpus={self.cfg.train_gpus}",
            f"num_nodes={self.cfg.train_num_nodes}",
        ]
        if self.cfg.train_max_steps is not None:
            command.append(f"max_steps={self.cfg.train_max_steps}")
        if self.cfg.train_val_check_interval is not None:
            command.append(f"val_check_interval={self.cfg.train_val_check_interval}")
        command.extend(train_overrides)

        extra_env = {}
        if self.cfg.train_cuda_visible_devices:
            extra_env["CUDA_VISIBLE_DEVICES"] = self.cfg.train_cuda_visible_devices
        self.runner.run(
            command,
            cwd=self.cfg.il_lib_dir,
            log_path=log_path,
            env_name=self.cfg.train_env_name,
            extra_env=extra_env or None,
        )

        if self.cfg.dry_run:
            run_dir = self.cfg.train_root / f"{prefix}_DRYRUN"
            checkpoint = run_dir / "ckpt" / "best.pth"
            return run_dir, checkpoint

        run_dir = find_latest_run_dir(self.cfg.train_root, prefix)
        if run_dir is None:
            raise RuntimeError(f"Could not find train run directory for prefix {prefix}")
        checkpoint = find_best_checkpoint(run_dir) or find_latest_checkpoint(run_dir)
        if checkpoint is None:
            raise RuntimeError(f"Could not find checkpoint under {run_dir / 'ckpt'}")
        return run_dir, checkpoint

    def _find_existing_method(self, prefix: str) -> Optional[tuple[Path, Path]]:
        run_dir = find_latest_run_dir(self.cfg.train_root, prefix)
        if run_dir is None:
            return None
        checkpoint = find_best_checkpoint(run_dir) or find_latest_checkpoint(run_dir)
        if checkpoint is None:
            return None
        return run_dir, checkpoint

    def _build_dry_run_artifact(
        self,
        *,
        prefix: str,
        method: str,
        arch: str,
        data_dir: Path,
        round_idx: int,
        base_checkpoint: Optional[Path] = None,
        base_run_dir: Optional[Path] = None,
    ) -> MethodArtifact:
        run_dir = self.cfg.train_root / f"{prefix}_DRYRUN"
        checkpoint = run_dir / "ckpt" / "best.pth"
        return MethodArtifact(
            method=method,
            arch=arch,
            data_dir=data_dir,
            run_dir=run_dir,
            checkpoint=checkpoint,
            round_idx=round_idx,
            base_checkpoint=base_checkpoint,
            base_run_dir=base_run_dir,
        )

    def _prepare_merged_dir(
        self,
        *,
        demo_tag: str,
        round_idx: int,
        human_h5: Path,
        correction_files: list[Path],
    ) -> Path:
        merged_dir = self.cfg.data_root / "merged" / demo_tag / round_tag(round_idx)
        merged_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.dry_run:
            print(f"[link] {merged_dir / f'human_{demo_tag}.hdf5'} -> {human_h5}")
            for correction_round_idx, correction_h5 in enumerate(correction_files, start=1):
                link_name = merged_dir / f"corrections_{demo_tag}_{round_tag(correction_round_idx)}.hdf5"
                print(f"[link] {link_name} -> {correction_h5}")
            return merged_dir

        ensure_symlink(merged_dir / f"human_{demo_tag}.hdf5", human_h5)
        for correction_round_idx, correction_h5 in enumerate(correction_files, start=1):
            link_name = merged_dir / f"corrections_{demo_tag}_{round_tag(correction_round_idx)}.hdf5"
            ensure_symlink(link_name, correction_h5)
        return merged_dir

    def _process_hdf5(self, input_path: Path, *, is_human_demo: bool, log_path: Path) -> None:
        command = [
            "python",
            "scripts/data/process_data.py",
            "-i",
            str(input_path),
            "-l",
            str(self.cfg.pre_intervention_length),
        ]
        if is_human_demo:
            command.append("-d")
        self.runner.run(
            command,
            cwd=self.cfg.iiil_dir,
            log_path=log_path,
            env_name=self.cfg.process_env_name,
        )

    def ensure_seed_demo_source(self) -> Path:
        if self.cfg.base_demo_source is not None:
            if not self.cfg.base_demo_source.exists():
                raise FileNotFoundError(f"Base demo source not found: {self.cfg.base_demo_source}")
            print(f"[seed] using external demo source: {self.cfg.base_demo_source}")
            return self.cfg.base_demo_source

        if self.cfg.seed_demo_h5.exists() and not self.cfg.force_overwrite:
            print(f"[skip] seed demo pool already exists: {self.cfg.seed_demo_h5}")
            return self.cfg.seed_demo_h5

        if self.cfg.seed_demo_h5.exists():
            self.cfg.seed_demo_h5.unlink()

        log_dir = self.cfg.log_root / "seed_master"
        log_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "python",
            "run_task.py",
            f"env={self.cfg.collect_env}",
            f"task={self.cfg.collection_task}",
            f"policy={self.cfg.base_demo_policy}",
            f"hdf_path={self.cfg.seed_demo_h5}",
            "overwrite=true",
            f"task.termination_config.total_episode={self.cfg.seed_demo_episodes}",
        ]
        command.extend(self.cfg.base_collect_overrides)
        extra_env = {}
        if self.cfg.sim_cuda_visible_devices:
            extra_env["CUDA_VISIBLE_DEVICES"] = self.cfg.sim_cuda_visible_devices
        self.runner.run(
            command,
            cwd=self.cfg.iiil_dir,
            log_path=log_dir / "collect_seed_demos.log",
            env_name=self.cfg.collect_env_name,
            extra_env=extra_env or None,
            success_validator=lambda: validate_hdf5_has_demos(self.cfg.seed_demo_h5),
        )
        if self.cfg.dry_run:
            return self.cfg.seed_demo_h5
        self._process_hdf5(
            self.cfg.seed_demo_h5,
            is_human_demo=True,
            log_path=log_dir / "process_seed_demos.log",
        )
        return self.cfg.seed_demo_h5

    def _prepare_human_subset(self, n: int, demo_tag: str, source: Path) -> Path:
        human_dir = self.cfg.data_root / "human" / demo_tag
        human_dir.mkdir(parents=True, exist_ok=True)
        human_h5 = human_dir / f"human_{demo_tag}.hdf5"
        if human_h5.exists() and not self.cfg.force_overwrite:
            print(f"[skip] human subset already exists: {human_h5}")
            return human_h5
        if self.cfg.dry_run:
            print(f"[subset] input={source} output={human_h5} n={n}")
            return human_h5
        materialize_subset_hdf5(source, human_h5, n)
        self._process_hdf5(
            human_h5,
            is_human_demo=True,
            log_path=self.cfg.log_root / demo_tag / "human_process.log",
        )
        return human_h5

    def _collect_corrections(
        self,
        *,
        demo_tag: str,
        round_idx: int,
        deployed_artifact: MethodArtifact,
        correction_h5: Path,
    ) -> None:
        if self.cfg.skip_collection:
            print("[skip] correction collection disabled by --skip-collection")
            return
        if correction_h5.exists() and not self.cfg.force_overwrite:
            print(f"[skip] correction dataset already exists: {correction_h5}")
            return
        if correction_h5.exists():
            correction_h5.unlink()

        log_dir = self.cfg.log_root / demo_tag / round_tag(round_idx)
        serve_command = [
            "python",
            "-u",
            "serve.py",
            f"arch={deployed_artifact.arch}",
            f"robot={self.cfg.robot}",
            f"task={self.cfg.base_task}",
            f"ckpt_path={deployed_artifact.checkpoint}",
        ]
        self._start_server(
            serve_command,
            log_dir / f"serve_{deployed_artifact.method}_for_corrections.log",
            port=self.cfg.port,
        )
        try:
            command = [
                "python",
                "run_task.py",
                f"env={self.cfg.collect_env}",
                f"task={self.cfg.collection_task}",
                f"policy={self.cfg.correction_policy}",
                f"hdf_path={correction_h5}",
                "overwrite=true",
            ]
            if self.cfg.correction_episodes is not None:
                command.append(f"task.termination_config.total_episode={self.cfg.correction_episodes}")
            if self.cfg.correction_requires_base_server and self.cfg.correction_policy == "sa":
                command.extend(
                    [
                        "policy.base_policy.type=il",
                        "policy.base_policy.use_websocket=true",
                        f"policy.base_policy.host={self.cfg.host}",
                        f"policy.base_policy.port={self.cfg.port}",
                    ]
                )
                if self.cfg.correction_oracle_policy:
                    command.append(f"policy.oracle_policy.type={self.cfg.correction_oracle_policy}")
            command.extend(self.cfg.correction_collect_overrides)

            extra_env = {}
            if self.cfg.sim_cuda_visible_devices:
                extra_env["CUDA_VISIBLE_DEVICES"] = self.cfg.sim_cuda_visible_devices
            self.runner.run(
                command,
                cwd=self.cfg.iiil_dir,
                log_path=log_dir / "collect_corrections.log",
                env_name=self.cfg.collect_env_name,
                extra_env=extra_env or None,
                success_validator=lambda: validate_hdf5_has_demos(correction_h5),
            )
        finally:
            self._stop_server()

        self._process_hdf5(
            correction_h5,
            is_human_demo=False,
            log_path=log_dir / "process_corrections.log",
        )

    def _aggregate_eval_results(
        self,
        *,
        demo_count: int,
        demo_tag: str,
        round_idx: int,
        scope_name: str,
        method: str,
        eval_dir: Path,
        checkpoint: Path,
        base_checkpoint: Optional[Path],
    ) -> None:
        values = []
        repeat_files = []
        for repeat_file in sorted(eval_dir.glob("repeat_*.json")):
            with open(repeat_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if "SR" not in payload:
                raise RuntimeError(f"Missing SR in {repeat_file}")
            values.append(float(payload["SR"]))
            repeat_files.append(str(repeat_file))
        if not values:
            raise RuntimeError(f"No evaluation files found under {eval_dir}")

        payload = {
            "demo_count": demo_count,
            "demo_tag": demo_tag,
            "round_idx": round_idx,
            "scope_tag": scope_name,
            "method": method,
            "checkpoint": str(checkpoint),
            "base_checkpoint": str(base_checkpoint) if base_checkpoint else None,
            "n_repeats": len(values),
            "success_rates": values,
            "mean_success_rate": statistics.fmean(values),
            "std_success_rate": statistics.stdev(values) if len(values) > 1 else 0.0,
            "repeat_files": repeat_files,
        }
        write_json(eval_dir / "summary.json", payload)

    def _run_eval_suite(
        self,
        *,
        demo_count: int,
        demo_tag: str,
        scope_name: str,
        artifact: MethodArtifact,
    ) -> None:
        if self.cfg.skip_eval or "eval" not in self.cfg.phases:
            return

        eval_dir = self.cfg.eval_root / demo_tag / round_tag(artifact.round_idx) / artifact.method
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_json = eval_dir / "summary.json"
        if summary_json.exists() and not self.cfg.force_overwrite:
            print(f"[skip] eval summary already exists: {summary_json}")
            return

        pending_repeats = []
        for repeat in range(self.cfg.eval_repeat_start, self.cfg.eval_repeat_end + 1):
            repeat_json = eval_dir / f"repeat_{repeat:02d}.json"
            if repeat_json.exists() and not self.cfg.force_overwrite:
                continue
            pending_repeats.append(repeat)

        if not pending_repeats:
            print(f"[skip] no pending eval repeats for {scope_name}/{artifact.method}")
            self._aggregate_eval_results(
                demo_count=demo_count,
                demo_tag=demo_tag,
                round_idx=artifact.round_idx,
                scope_name=scope_name,
                method=artifact.method,
                eval_dir=eval_dir,
                checkpoint=artifact.checkpoint,
                base_checkpoint=artifact.base_checkpoint,
            )
            return

        serve_command = [
            "python",
            "-u",
            "serve.py",
            f"arch={artifact.arch}",
            f"robot={self.cfg.robot}",
            f"task={self.cfg.base_task}",
            f"ckpt_path={artifact.checkpoint}",
        ]
        if artifact.method == "residual":
            if artifact.base_checkpoint is None:
                raise RuntimeError("Residual eval requires base checkpoint")
                serve_command.extend(
                    [
                    f"policy_wrapper.base_policy_ckpt_path={artifact.base_checkpoint}",
                    "policy_wrapper._target_=il_lib.policies.policy_base.ResidualPolicyWrapper",
                    f"policy_wrapper.base_deployed_action_steps={read_deployed_action_steps(artifact.base_run_dir, self.cfg.base_arch, self.cfg.iiil_dir)}",
                    ]
                )

        log_dir = self.cfg.log_root / demo_tag / round_tag(artifact.round_idx) / f"eval_{artifact.method}"
        self._start_server(
            serve_command,
            log_dir / "serve.log",
            port=self.cfg.port,
        )
        try:
            for repeat in pending_repeats:
                repeat_json = eval_dir / f"repeat_{repeat:02d}.json"
                video_path = eval_dir / f"repeat_{repeat:02d}.mp4"
                if repeat_json.exists() and self.cfg.force_overwrite:
                    repeat_json.unlink()
                if video_path.exists() and self.cfg.force_overwrite:
                    video_path.unlink()

                command = [
                    "python",
                    "run_task.py",
                    f"env={self.cfg.eval_env}",
                    f"task={self.cfg.base_task}",
                    "policy=il",
                    f"results_json_path={repeat_json}",
                    f"task.termination_config.total_episode={self.cfg.eval_episodes}",
                    f"policy.host={self.cfg.host}",
                    f"policy.port={self.cfg.port}",
                ]
                if self.cfg.save_eval_video and repeat == self.cfg.eval_repeat_start:
                    command.append(f"video_recording_path={video_path}")
                command.extend(self.cfg.eval_overrides)
                extra_env = {}
                if self.cfg.sim_cuda_visible_devices:
                    extra_env["CUDA_VISIBLE_DEVICES"] = self.cfg.sim_cuda_visible_devices
                self.runner.run(
                    command,
                    cwd=self.cfg.iiil_dir,
                    log_path=log_dir / f"repeat_{repeat:02d}.log",
                    env_name=self.cfg.eval_env_name,
                    extra_env=extra_env or None,
                    success_validator=lambda path=repeat_json: validate_results_json(path),
                )
        finally:
            self._stop_server()

        if self.cfg.dry_run:
            print(f"[dry-run] would aggregate eval summary at {summary_json}")
            return

        self._aggregate_eval_results(
            demo_count=demo_count,
            demo_tag=demo_tag,
            round_idx=artifact.round_idx,
            scope_name=scope_name,
            method=artifact.method,
            eval_dir=eval_dir,
            checkpoint=artifact.checkpoint,
            base_checkpoint=artifact.base_checkpoint,
        )

    def run(self) -> None:
        seed_source: Optional[Path] = None
        if "seed_data" in self.cfg.phases:
            seed_source = self.ensure_seed_demo_source()
        elif self.cfg.base_demo_source is not None:
            seed_source = self.cfg.base_demo_source
        elif self.cfg.seed_demo_h5.exists():
            seed_source = self.cfg.seed_demo_h5

        if seed_source is None:
            raise RuntimeError(
                "No seed demo source is available. Run with phases including seed_data "
                "or pass --base-demo-source."
            )

        for demo_count in self.cfg.demo_counts:
            demo_tag = f"n{demo_count:03d}"
            print("")
            print("=" * 68)
            print(
                f"Running experiment slice {demo_tag} "
                f"(rounds={self.cfg.rounds}, iterate_with={self.cfg.iteration_method})"
            )
            print("=" * 68)

            human_h5 = self._prepare_human_subset(demo_count, demo_tag, seed_source)
            human_dir = human_h5.parent

            base_scope_name = scope_tag(demo_tag, 0)
            base_log_dir = self.cfg.log_root / demo_tag / round_tag(0)
            base_log_dir.mkdir(parents=True, exist_ok=True)
            base_prefix = f"{self.cfg.name}_{base_scope_name}_demos"
            base_run_dir, base_checkpoint = self._train_method(
                prefix=base_prefix,
                arch=self.cfg.base_arch,
                data_dir=human_dir,
                log_path=base_log_dir / "train_demos.log",
                train_overrides=self.cfg.base_train_overrides,
            )
            base_artifact = MethodArtifact(
                method="demos",
                arch=self.cfg.base_arch,
                data_dir=human_dir,
                run_dir=base_run_dir,
                checkpoint=base_checkpoint,
                round_idx=0,
            )
            self._record_manifest(base_scope_name, "demos", human_dir, base_run_dir, base_checkpoint)

            if "demos" in self.cfg.methods:
                self._run_eval_suite(
                    demo_count=demo_count,
                    demo_tag=demo_tag,
                    scope_name=base_scope_name,
                    artifact=base_artifact,
                )

            deploy_artifact = base_artifact
            correction_files: list[Path] = []

            for round_idx in range(1, self.cfg.rounds + 1):
                scope_name = scope_tag(demo_tag, round_idx)
                round_name = round_tag(round_idx)
                round_log_dir = self.cfg.log_root / demo_tag / round_name
                round_log_dir.mkdir(parents=True, exist_ok=True)
                correction_dir = self.cfg.data_root / "corrections" / demo_tag / round_name
                correction_dir.mkdir(parents=True, exist_ok=True)
                correction_h5 = correction_dir / f"corrections_{scope_name}.hdf5"

                if "correction_data" in self.cfg.phases:
                    self._collect_corrections(
                        demo_tag=demo_tag,
                        round_idx=round_idx,
                        deployed_artifact=deploy_artifact,
                        correction_h5=correction_h5,
                    )
                elif not correction_h5.exists() and not self.cfg.dry_run:
                    raise RuntimeError(
                        f"Correction dataset missing for {scope_name}: {correction_h5}. "
                        "Run with phases including correction_data or reuse an existing dataset."
                    )

                correction_data_available = correction_h5.exists() or (
                    self.cfg.dry_run and "correction_data" in self.cfg.phases
                )
                if not correction_data_available:
                    if not self.cfg.dry_run:
                        raise RuntimeError(f"Correction dataset missing for {scope_name}: {correction_h5}")
                    print(f"[dry-run] correction dataset not available for {scope_name}; skipping downstream round work")
                    break

                correction_files.append(correction_h5)
                merged_dir = self._prepare_merged_dir(
                    demo_tag=demo_tag,
                    round_idx=round_idx,
                    human_h5=human_h5,
                    correction_files=correction_files,
                )

                trained: dict[str, MethodArtifact] = {}
                if "correction_train" in self.cfg.phases:
                    for method, reweight in REWEIGHT_METHODS.items():
                        if method not in self.cfg.methods:
                            continue
                        prefix = f"{self.cfg.name}_{scope_name}_{method}"
                        run_dir, checkpoint = self._train_method(
                            prefix=prefix,
                            arch=self.cfg.base_arch,
                            data_dir=merged_dir,
                            log_path=round_log_dir / f"train_{method}.log",
                            train_overrides=[
                                f"data.reweight_strategy={reweight}",
                                *self.cfg.reweight_train_overrides,
                            ],
                        )
                        artifact = MethodArtifact(
                            method=method,
                            arch=self.cfg.base_arch,
                            data_dir=merged_dir,
                            run_dir=run_dir,
                            checkpoint=checkpoint,
                            round_idx=round_idx,
                        )
                        trained[method] = artifact
                        self._record_manifest(scope_name, method, merged_dir, run_dir, checkpoint)

                    if "residual" in self.cfg.methods:
                        prefix = f"{self.cfg.name}_{scope_name}_residual"
                        run_dir, checkpoint = self._train_method(
                            prefix=prefix,
                            arch=self.cfg.residual_arch,
                            data_dir=merged_dir,
                            log_path=round_log_dir / "train_residual.log",
                            train_overrides=self.cfg.residual_train_overrides,
                        )
                        artifact = MethodArtifact(
                            method="residual",
                            arch=self.cfg.residual_arch,
                            data_dir=merged_dir,
                            run_dir=run_dir,
                            checkpoint=checkpoint,
                            round_idx=round_idx,
                            base_checkpoint=base_artifact.checkpoint,
                            base_run_dir=base_artifact.run_dir,
                        )
                        trained["residual"] = artifact
                        self._record_manifest(scope_name, "residual", merged_dir, run_dir, checkpoint)
                else:
                    for method in self.cfg.methods:
                        if method == "demos":
                            continue
                        prefix = f"{self.cfg.name}_{scope_name}_{method}"
                        existing = self._find_existing_method(prefix)
                        if existing is None:
                            continue
                        run_dir, checkpoint = existing
                        trained[method] = MethodArtifact(
                            method=method,
                            arch=self.cfg.residual_arch if method == "residual" else self.cfg.base_arch,
                            data_dir=merged_dir,
                            run_dir=run_dir,
                            checkpoint=checkpoint,
                            round_idx=round_idx,
                            base_checkpoint=base_artifact.checkpoint if method == "residual" else None,
                            base_run_dir=base_artifact.run_dir if method == "residual" else None,
                        )

                for method in self.cfg.methods:
                    if method == "demos":
                        continue
                    artifact = trained.get(method)
                    if artifact is None:
                        continue
                    self._run_eval_suite(
                        demo_count=demo_count,
                        demo_tag=demo_tag,
                        scope_name=scope_name,
                        artifact=artifact,
                    )

                if round_idx >= self.cfg.rounds:
                    continue

                if self.cfg.iteration_method == "demos":
                    deploy_artifact = base_artifact
                    continue

                next_deploy = trained.get(self.cfg.iteration_method)
                if next_deploy is None:
                    prefix = f"{self.cfg.name}_{scope_name}_{self.cfg.iteration_method}"
                    existing = self._find_existing_method(prefix)
                    if existing is not None:
                        run_dir, checkpoint = existing
                        next_deploy = MethodArtifact(
                            method=self.cfg.iteration_method,
                            arch=self.cfg.base_arch,
                            data_dir=merged_dir,
                            run_dir=run_dir,
                            checkpoint=checkpoint,
                            round_idx=round_idx,
                        )
                    elif self.cfg.dry_run:
                        next_deploy = self._build_dry_run_artifact(
                            prefix=prefix,
                            method=self.cfg.iteration_method,
                            arch=self.cfg.base_arch,
                            data_dir=merged_dir,
                            round_idx=round_idx,
                        )
                    else:
                        raise RuntimeError(
                            f"Cannot continue to {round_tag(round_idx + 1)} for {demo_tag} because "
                            f"iteration method '{self.cfg.iteration_method}' has no checkpoint at {scope_name}. "
                            "Run correction_train or reuse an existing trained run."
                        )
                deploy_artifact = next_deploy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end IIIL experiments from demo collection through eval.",
    )
    parser.add_argument(
        "--spec",
        default=None,
        help="Path to an experiment card file with key=value settings plus optional notes.",
    )
    parser.add_argument("--name", default=None, help="Experiment name. Default derives from robot + task.")
    parser.add_argument("--root-dir", default=None, help="Experiment output root. Default: outputs/experiments/<name>.")
    parser.add_argument("--iiil-dir", default=str(repo_root()), help="Path to the IIIL repo root.")
    parser.add_argument("--il-lib-dir", default=None, help="Path to sibling il_lib repo. Default: ../il_lib.")
    parser.add_argument("--conda-sh", default=None, help="Path to conda.sh if using per-phase env activation.")
    parser.add_argument("--collect-env-name", default=None, help="Conda env for collection commands.")
    parser.add_argument("--process-env-name", default=None, help="Conda env for process_data.py.")
    parser.add_argument("--train-env-name", default=None, help="Conda env for il_lib training.")
    parser.add_argument("--serve-env-name", default=None, help="Conda env for il_lib serve.py.")
    parser.add_argument("--eval-env-name", default=None, help="Conda env for evaluation rollouts.")

    parser.add_argument("--robot", default="a1_iiil", help="Hydra robot config, e.g. a1_iiil or r1pro_iiil.")
    parser.add_argument("--collect-env", default=None, help="Hydra env config for base demo and correction collection.")
    parser.add_argument("--eval-env", default=None, help="Hydra env config for evaluation. Defaults to collect env.")
    parser.add_argument("--base-task", default="pnp/base", help="Hydra base task, e.g. pnp/base.")
    parser.add_argument("--collection-task", default=None, help="Hydra collection task, e.g. pnp/dc.")
    parser.add_argument("--base-arch", default=None, help="Hydra arch for base training.")
    parser.add_argument("--residual-arch", default=None, help="Hydra arch for residual training.")

    parser.add_argument(
        "--methods",
        default="demos,cotrain,hgdagger,sirius,residual",
        help="CSV of methods or 'all'. Methods: demos,cotrain,hgdagger,sirius,residual.",
    )
    parser.add_argument(
        "--phases",
        default="full",
        help="CSV of phases or 'full'. Phases: seed_data,base_train,correction_data,correction_train,eval.",
    )
    parser.add_argument("--demo-counts", default="20", help="CSV of demo counts, e.g. 20,40,80.")
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of correction rounds after the demos-only base round. Default: 1.",
    )
    parser.add_argument(
        "--iteration-method",
        default=None,
        help="Method checkpoint to deploy for the next correction round. Default prefers cotrain, then hgdagger, sirius, then demos.",
    )
    parser.add_argument("--seed-demo-episodes", type=int, default=None, help="Episodes for seed demo bootstrapping. Default=max(demo_counts).")
    parser.add_argument("--correction-episodes", type=int, default=None, help="Override collection task episode count for each correction round.")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Episodes per evaluation repeat.")
    parser.add_argument("--eval-repeats", type=int, default=3, help="Number of repeated eval runs.")
    parser.add_argument("--eval-repeat-start", type=int, default=1, help="First eval repeat index.")
    parser.add_argument("--eval-repeat-end", type=int, default=None, help="Last eval repeat index. Defaults to eval-repeats.")
    parser.add_argument("--pre-intervention-length", type=int, default=15, help="Pre-intervention window for process_data.py.")

    parser.add_argument("--base-demo-source", default=None, help="Existing HDF5 file or directory for seed demos.")
    parser.add_argument("--base-demo-policy", default=DEFAULT_BASE_DEMO_POLICY, help="Policy used when bootstrapping seed demos in sim.")
    parser.add_argument("--base-collect-override", action="append", default=[], help="Extra Hydra override for seed demo collection.")
    parser.add_argument("--base-train-override", action="append", default=[], help="Extra Hydra override for base training.")

    parser.add_argument("--correction-policy", default="sa", help="Hydra policy used for correction collection.")
    parser.add_argument("--correction-oracle-policy", default="curobo", help="Oracle policy type for shared autonomy correction collection.")
    parser.add_argument(
        "--correction-requires-base-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether correction collection should serve the base policy first.",
    )
    parser.add_argument("--correction-collect-override", action="append", default=[], help="Extra Hydra override for correction collection.")
    parser.add_argument("--reweight-train-override", action="append", default=[], help="Extra Hydra override for cotrain/hgdagger/sirius training.")
    parser.add_argument("--residual-train-override", action="append", default=[], help="Extra Hydra override for residual training.")
    parser.add_argument("--eval-override", action="append", default=[], help="Extra Hydra override for evaluation run_task.py calls.")

    parser.add_argument("--train-gpus", type=int, default=1, help="il_lib train.py gpus override.")
    parser.add_argument("--train-num-nodes", type=int, default=1, help="il_lib train.py num_nodes override.")
    parser.add_argument("--train-max-steps", type=int, default=None, help="il_lib train.py max_steps override.")
    parser.add_argument("--train-val-check-interval", type=int, default=None, help="il_lib train.py val_check_interval override.")
    parser.add_argument("--host", default="127.0.0.1", help="Websocket host for served policies.")
    parser.add_argument("--port", type=int, default=8000, help="Websocket port for served policies.")
    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to pass use_wandb to il_lib training. Default: true.",
    )
    parser.add_argument("--train-cuda-visible-devices", default=None, help="CUDA_VISIBLE_DEVICES for il_lib train.py.")
    parser.add_argument("--sim-cuda-visible-devices", default=None, help="CUDA_VISIBLE_DEVICES for IIIL collection/eval.")
    parser.add_argument("--serve-cuda-visible-devices", default=None, help="CUDA_VISIBLE_DEVICES for il_lib serve.py.")

    parser.add_argument("--force-overwrite", action="store_true", help="Recreate generated artifacts and rerun completed steps.")
    parser.add_argument("--skip-collection", action="store_true", help="Skip correction collection and reuse an existing correction dataset.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation even if the eval phase is selected.")
    parser.add_argument("--save-eval-video", action="store_true", help="Save one video for the first eval repeat.")
    dry_run_group = parser.add_mutually_exclusive_group()
    dry_run_group.add_argument("--dry-run", dest="dry_run", action="store_true", help="Print commands without executing them.")
    dry_run_group.add_argument("--no-dry-run", dest="dry_run", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(dry_run=False)
    return parser


def apply_spec_defaults(parser: argparse.ArgumentParser, spec: ExperimentSpec) -> None:
    valid_dests = {action.dest for action in parser._actions if action.dest not in {"help"}}
    unknown = sorted(set(spec.values).difference(valid_dests))
    if unknown:
        raise ValueError(
            f"Unsupported key(s) in experiment spec {spec.path}: {', '.join(unknown)}"
        )
    parser.set_defaults(**spec.values)


def parse_config(argv: Optional[list[str]] = None) -> ExperimentConfig:
    launch_args = list(argv) if argv is not None else sys.argv[1:]
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--spec", default=None)
    pre_args, _ = pre_parser.parse_known_args(launch_args)
    spec = None if pre_args.spec is None else load_experiment_spec(Path(pre_args.spec))

    parser = build_arg_parser()
    if spec is not None:
        apply_spec_defaults(parser, spec)
    args = parser.parse_args(launch_args)

    iiil_dir = Path(args.iiil_dir).resolve()
    il_lib_dir = Path(args.il_lib_dir).resolve() if args.il_lib_dir else (iiil_dir.parent / "il_lib")
    if not il_lib_dir.exists():
        raise FileNotFoundError(f"il_lib repo not found: {il_lib_dir}")

    name = args.name or default_experiment_name(args.robot, args.base_task)
    root_dir = Path(args.root_dir).resolve() if args.root_dir else (iiil_dir / "outputs/experiments" / name)

    methods = parse_methods(args.methods)
    phases = parse_phases(args.phases)
    demo_counts = parse_int_csv(args.demo_counts)
    if args.rounds < 0:
        raise ValueError("--rounds must be >= 0")
    if args.rounds > 1 and "residual" in methods:
        raise ValueError(
            "Multi-round runs do not support residual training yet. "
            "Use demos, cotrain, hgdagger, and/or sirius for iterative rounds."
        )
    iteration_method = resolve_iteration_method(args.iteration_method, methods, args.rounds)
    seed_demo_episodes = args.seed_demo_episodes or max(demo_counts)
    eval_repeat_end = args.eval_repeat_end or args.eval_repeats
    if args.eval_repeat_start < 1 or eval_repeat_end < args.eval_repeat_start:
        raise ValueError("Invalid eval repeat range")

    collect_env = infer_collect_env(args.robot, args.collect_env)
    eval_env = args.eval_env or collect_env
    collection_task = args.collection_task or infer_collection_task(args.base_task)
    base_arch = infer_base_arch(args.robot, args.base_arch)
    residual_arch = infer_residual_arch(args.robot, args.residual_arch)

    conda_sh = Path(args.conda_sh).resolve() if args.conda_sh else discover_conda_sh()

    return ExperimentConfig(
        name=name,
        root_dir=root_dir,
        iiil_dir=iiil_dir,
        il_lib_dir=il_lib_dir,
        launch_args=launch_args,
        spec_path=None if spec is None else spec.path,
        spec_title=None if spec is None else spec.title,
        spec_tags=[] if spec is None else spec.tags,
        spec_notes="" if spec is None else spec.notes,
        conda_sh=conda_sh,
        collect_env_name=args.collect_env_name,
        process_env_name=args.process_env_name,
        train_env_name=args.train_env_name,
        serve_env_name=args.serve_env_name,
        eval_env_name=args.eval_env_name,
        robot=args.robot,
        collect_env=collect_env,
        eval_env=eval_env,
        base_task=args.base_task,
        collection_task=collection_task,
        base_arch=base_arch,
        residual_arch=residual_arch,
        methods=methods,
        phases=phases,
        demo_counts=demo_counts,
        rounds=args.rounds,
        iteration_method=iteration_method,
        seed_demo_episodes=seed_demo_episodes,
        correction_episodes=args.correction_episodes,
        eval_episodes=args.eval_episodes,
        eval_repeats=args.eval_repeats,
        eval_repeat_start=args.eval_repeat_start,
        eval_repeat_end=eval_repeat_end,
        pre_intervention_length=args.pre_intervention_length,
        base_demo_source=Path(args.base_demo_source).resolve() if args.base_demo_source else None,
        base_demo_policy=args.base_demo_policy,
        base_collect_overrides=normalize_optional_str_list(args.base_collect_override),
        base_train_overrides=normalize_optional_str_list(args.base_train_override),
        correction_policy=args.correction_policy,
        correction_requires_base_server=args.correction_requires_base_server,
        correction_oracle_policy=args.correction_oracle_policy,
        correction_collect_overrides=normalize_optional_str_list(args.correction_collect_override),
        reweight_train_overrides=normalize_optional_str_list(args.reweight_train_override),
        residual_train_overrides=normalize_optional_str_list(args.residual_train_override),
        eval_overrides=normalize_optional_str_list(args.eval_override),
        train_gpus=args.train_gpus,
        train_num_nodes=args.train_num_nodes,
        train_max_steps=args.train_max_steps,
        train_val_check_interval=args.train_val_check_interval,
        host=args.host,
        port=args.port,
        use_wandb=args.use_wandb,
        train_cuda_visible_devices=args.train_cuda_visible_devices,
        sim_cuda_visible_devices=args.sim_cuda_visible_devices,
        serve_cuda_visible_devices=args.serve_cuda_visible_devices,
        force_overwrite=args.force_overwrite,
        skip_collection=args.skip_collection,
        skip_eval=args.skip_eval,
        save_eval_video=args.save_eval_video,
        dry_run=args.dry_run,
    )


def main(argv: Optional[list[str]] = None) -> int:
    cfg = parse_config(argv)
    pipeline = ExperimentPipeline(cfg)
    try:
        pipeline.run()
    finally:
        pipeline.cleanup()
    print("")
    print("Experiment completed.")
    print(f"Manifest: {cfg.manifest_path}")
    if "eval" in cfg.phases and not cfg.skip_eval:
        print(f"Eval summaries: {cfg.eval_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

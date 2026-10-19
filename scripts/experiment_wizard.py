#!/usr/bin/env python3
"""Interactive helper for creating experiment cards and launching runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from experiment_pipeline import (
    AVAILABLE_METHODS,
    default_experiment_name,
    infer_collection_task,
    load_experiment_spec,
    normalize_optional_str_list,
    parse_csv,
    parse_methods,
    resolve_iteration_method,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARDS_DIR = REPO_ROOT / "experiment_cards"
EXAMPLES_DIR = REPO_ROOT / "examples" / "experiment_specs"
_USE_COLOR = (
    sys.stdout.isatty()
    and sys.stdin.isatty()
    and os.environ.get("TERM", "").lower() != "dumb"
    and "NO_COLOR" not in os.environ
)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_FG_BLUE = "\033[34m"
_FG_CYAN = "\033[36m"
_FG_GREEN = "\033[32m"
_FG_RED = "\033[31m"
_FG_YELLOW = "\033[33m"


def _style(text: str, *codes: str) -> str:
    if not _USE_COLOR or not codes:
        return text
    return f"{''.join(codes)}{text}{_RESET}"


def _bold(text: str) -> str:
    return _style(text, _BOLD)


def _dim(text: str) -> str:
    return _style(text, _DIM)


def _accent(text: str) -> str:
    return _style(text, _BOLD, _FG_CYAN)


def _good(text: str) -> str:
    return _style(text, _BOLD, _FG_GREEN)


def _warn(text: str) -> str:
    return _style(text, _BOLD, _FG_YELLOW)


def _bad(text: str) -> str:
    return _style(text, _BOLD, _FG_RED)


def _menu_number(value: int) -> str:
    return _style(str(value), _BOLD, _FG_BLUE)


def _terminal_width() -> int:
    return max(60, shutil.get_terminal_size((88, 20)).columns)


def _rule(char: str = "-") -> str:
    return char * min(_terminal_width(), 88)


def _print_header(title: str, subtitle: Optional[str] = None) -> None:
    print("")
    print(_accent(title))
    print(_dim(_rule()))
    if subtitle:
        print(_dim(subtitle))


def _print_note(text: str) -> None:
    print(_dim(text))


def _print_error(text: str) -> None:
    print(_bad(f"Error: {text}"))


def _format_status(summary: "ExperimentSummary") -> str:
    if summary.has_run_artifacts:
        status = _good("recorded run")
    else:
        status = _warn("definition only")
    if summary.last_launch_mode:
        launch_mode = _good(summary.last_launch_mode) if summary.last_launch_mode == "run" else _accent(summary.last_launch_mode)
        return f"{status}  last={launch_mode}"
    return status


@dataclass(frozen=True)
class WizardPreset:
    key: str
    label: str
    description: str
    defaults: dict[str, object]


@dataclass(frozen=True)
class ExperimentSummary:
    path: Path
    title: str
    name: str
    robot: str
    base_task: str
    demo_counts: list[str]
    rounds: int
    methods: list[str]
    iteration_method: Optional[str]
    phases: list[str]
    tags: list[str]
    root_dir: Path
    has_run_artifacts: bool
    last_launch_mode: Optional[str]
    launch_command: str


PRESETS = [
    WizardPreset(
        key="a1_demos_only",
        label="A1 demos-only baseline",
        description="Train only the demos policy, no correction rounds.",
        defaults={
            "title": "A1 PnP demos-only baseline",
            "name": "a1_pnp_demos_only",
            "robot": "a1_iiil",
            "base_task": "pnp/base",
            "demo_counts": [20, 40],
            "rounds": 0,
            "methods": ["demos"],
            "train_max_steps": 50000,
            "eval_repeats": 3,
            "use_wandb": True,
            "tags": ["a1", "sim", "baseline", "demos-only"],
            "notes": (
                "Goal:\n"
                "Establish the demos-only baseline before adding correction rounds.\n"
            ),
        },
    ),
    WizardPreset(
        key="a1_iterative_cotrain",
        label="A1 iterative with cotrain deployment",
        description="Collect multiple correction rounds and deploy the latest cotrain checkpoint.",
        defaults={
            "title": "A1 PnP iterative cotrain baseline",
            "name": "a1_pnp_iterative",
            "robot": "a1_iiil",
            "base_task": "pnp/base",
            "demo_counts": [20],
            "rounds": 2,
            "correction_episodes": 20,
            "methods": ["demos", "cotrain", "hgdagger", "sirius"],
            "iteration_method": "cotrain",
            "train_max_steps": 50000,
            "eval_repeats": 3,
            "use_wandb": True,
            "tags": ["a1", "sim", "iterative", "cotrain"],
            "notes": (
                "Goal:\n"
                "Compare whether cumulative correction rounds keep improving over the demos-only baseline.\n"
            ),
        },
    ),
    WizardPreset(
        key="a1_iterative_deploy_demos",
        label="A1 iterative, keep deploying demos",
        description="Collect multiple rounds while still deploying the original demos checkpoint.",
        defaults={
            "title": "A1 PnP iterative collection while still deploying demos",
            "name": "a1_pnp_iterative_deploy_demos",
            "robot": "a1_iiil",
            "base_task": "pnp/base",
            "demo_counts": [20],
            "rounds": 2,
            "correction_episodes": 20,
            "methods": ["demos", "cotrain", "hgdagger"],
            "iteration_method": "demos",
            "train_max_steps": 50000,
            "eval_repeats": 3,
            "use_wandb": True,
            "tags": ["a1", "sim", "iterative", "deploy-demos"],
            "notes": (
                "Goal:\n"
                "Isolate the effect of more correction data from changing the deployed policy.\n"
            ),
        },
    ),
    WizardPreset(
        key="a1_existing_pool",
        label="A1 from existing demo pool",
        description="Reuse an already-collected demo pool instead of recollecting seed demos.",
        defaults={
            "title": "A1 PnP from an existing demo pool",
            "name": "a1_pnp_existing_pool",
            "robot": "a1_iiil",
            "base_task": "pnp/base",
            "base_demo_source": "/path/to/seed_demos_master.hdf5",
            "demo_counts": [20, 40],
            "rounds": 1,
            "correction_episodes": 20,
            "methods": ["demos", "cotrain", "hgdagger", "sirius"],
            "iteration_method": "cotrain",
            "train_max_steps": 50000,
            "eval_repeats": 3,
            "use_wandb": True,
            "tags": ["a1", "sim", "existing-demos"],
            "notes": (
                "Goal:\n"
                "Reuse an existing curated demo pool and skip seed recollection.\n"
            ),
        },
    ),
    WizardPreset(
        key="a1_eval_only",
        label="A1 eval-only rerun",
        description="Rerun evaluation on existing checkpoints without recollecting or retraining.",
        defaults={
            "title": "A1 PnP eval-only rerun",
            "name": "a1_pnp_eval_only",
            "root_dir": "/path/to/existing/outputs/experiments/a1_pnp_iterative",
            "robot": "a1_iiil",
            "base_task": "pnp/base",
            "demo_counts": [20],
            "rounds": 2,
            "methods": ["demos", "cotrain", "hgdagger", "sirius"],
            "phases": ["eval"],
            "eval_repeats": 5,
            "use_wandb": True,
            "tags": ["a1", "sim", "eval-only"],
            "notes": (
                "Goal:\n"
                "Rerun evaluation for an existing experiment root.\n"
            ),
        },
    ),
    WizardPreset(
        key="a1_smoke",
        label="A1 quick smoke test",
        description="Very small run to sanity-check the full orchestration quickly.",
        defaults={
            "title": "A1 PnP smoke test",
            "name": "a1_pnp_smoke",
            "robot": "a1_iiil",
            "base_task": "pnp/base",
            "demo_counts": [5],
            "rounds": 1,
            "correction_episodes": 5,
            "methods": ["demos", "cotrain"],
            "iteration_method": "cotrain",
            "train_max_steps": 2000,
            "eval_repeats": 1,
            "use_wandb": True,
            "tags": ["a1", "sim", "smoke"],
            "notes": (
                "Goal:\n"
                "Quickly validate that collection, training, and eval still connect end to end.\n"
            ),
        },
    ),
    WizardPreset(
        key="r1pro_dryrun",
        label="R1Pro demos-only dry run",
        description="Validate command expansion for an R1Pro setup before running anything expensive.",
        defaults={
            "title": "R1Pro open-task demos-only dry run",
            "name": "r1pro_open_demos_dryrun",
            "robot": "r1pro_iiil",
            "collect_env": "sim_r1pro_joint",
            "eval_env": "sim_r1pro_joint",
            "base_task": "open/base",
            "collection_task": "open/dc",
            "base_arch": "wbvima_r1pro",
            "residual_arch": "residual_r1pro",
            "demo_counts": [20],
            "rounds": 0,
            "methods": ["demos"],
            "dry_run": True,
            "use_wandb": True,
            "tags": ["r1pro", "sim", "dry-run", "demos-only"],
            "notes": (
                "Goal:\n"
                "Dry-run the R1Pro command plan before attempting real collection or training.\n"
            ),
        },
    ),
    WizardPreset(
        key="custom",
        label="Custom from scratch",
        description="Start from minimal defaults and fill everything in yourself.",
        defaults={
            "title": "New experiment",
            "name": "new_experiment",
            "robot": "a1_iiil",
            "base_task": "pnp/base",
            "demo_counts": [20],
            "rounds": 1,
            "methods": ["demos", "cotrain"],
            "train_max_steps": 50000,
            "eval_repeats": 3,
            "use_wandb": True,
            "tags": [],
            "notes": "Goal:\nDescribe what this experiment is trying to test.\n",
        },
    ),
]


def prompt_line(prompt: str, default: Optional[str] = None) -> str:
    suffix = "" if default in {None, ""} else f" [{default}]"
    while True:
        value = input(f"{_bold(prompt)}{_dim(suffix)}: ").strip()
        if value:
            return value
        if default is not None:
            return default


def prompt_bool(prompt: str, default: bool) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        value = input(f"{_bold(prompt)} {_dim(f'[{default_label}]')}: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        _print_error("Enter y or n.")


def prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    print(_bold(prompt))
    default_idx = choices.index(default) + 1 if default in choices else 1
    for idx, choice in enumerate(choices, start=1):
        marker = _dim(" [default]") if idx == default_idx else ""
        print(f"  {_menu_number(idx)}. {choice}{marker}")
    while True:
        value = input(f"{_bold('Select')} {_dim(f'[{default_idx}]')}: ").strip()
        if not value:
            return default
        if value.isdigit():
            idx = int(value)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        if value in choices:
            return value
        _print_error(f"Choose one of: {', '.join(choices)} or a menu number.")


def prompt_csv_list(prompt: str, default: Iterable[object]) -> list[str]:
    default_str = ",".join(str(item) for item in default)
    while True:
        value = input(f"{_bold(prompt)} {_dim(f'[{default_str}]')}: ").strip()
        try:
            items = parse_csv(value or default_str)
        except ValueError as exc:
            _print_error(str(exc))
            continue
        if items:
            return items
        _print_error("Enter at least one item.")


def prompt_csv_ints(prompt: str, default: Iterable[int]) -> list[int]:
    default_str = ",".join(str(item) for item in default)
    while True:
        value = input(f"{_bold(prompt)} {_dim(f'[{default_str}]')}: ").strip()
        raw = value or default_str
        try:
            items = [int(item) for item in parse_csv(raw)]
        except ValueError:
            _print_error("Enter comma-separated integers, for example 20 or 20,40.")
            continue
        if items:
            return items
        _print_error("Enter at least one integer.")


def prompt_optional_int(prompt: str, default: Optional[int]) -> Optional[int]:
    default_str = "" if default is None else str(default)
    display_default = default_str or "blank"
    while True:
        value = input(f"{_bold(prompt)} {_dim(f'[{display_default}]')}: ").strip()
        if not value:
            return default
        if value.lower() in {"none", "null"}:
            return None
        try:
            return int(value)
        except ValueError:
            _print_error("Enter an integer or leave blank.")


def prompt_optional_path(prompt: str, default: Optional[str]) -> Optional[str]:
    display_default = default or "blank"
    value = input(f"{_bold(prompt)} {_dim(f'[{display_default}]')}: ").strip()
    if not value:
        return default
    if value.lower() in {"none", "null"}:
        return None
    return value


def prompt_notes(default: str) -> str:
    _print_header("Notes", "Finish with a single '.' line. Press Enter immediately to keep the default notes.")
    lines: list[str] = []
    first = input(f"{_accent('>')} ")
    if not first.strip():
        return default.strip()
    if first.strip() == ".":
        return ""
    lines.append(first.rstrip())
    while True:
        line = input(f"{_accent('>')} ")
        if line.strip() == ".":
            break
        lines.append(line.rstrip())
    return "\n".join(lines).rstrip()


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        rendered = ", ".join(format_value(item) for item in value)
        return f"[{rendered}]"
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def is_empty_value(value: object) -> bool:
    return value is None or value == "" or value == []


def render_card(values: dict[str, object], notes: str) -> str:
    ordered_keys = [
        "title",
        "name",
        "root_dir",
        "robot",
        "collect_env",
        "eval_env",
        "base_task",
        "collection_task",
        "base_arch",
        "residual_arch",
        "base_demo_source",
        "demo_counts",
        "rounds",
        "correction_episodes",
        "methods",
        "iteration_method",
        "phases",
        "train_max_steps",
        "eval_repeats",
        "use_wandb",
        "dry_run",
        "tags",
    ]
    lines: list[str] = []
    emitted = set()
    for key in ordered_keys:
        if key in values and not is_empty_value(values[key]):
            lines.append(f"{key} = {format_value(values[key])}")
            emitted.add(key)
    for key in sorted(set(values).difference(emitted)):
        if is_empty_value(values[key]):
            continue
        lines.append(f"{key} = {format_value(values[key])}")
    body = "\n".join(lines)
    notes_text = notes.strip()
    if notes_text:
        return f"{body}\n\n---\n{notes_text}\n"
    return body + "\n"


def summarize_card(path: Path) -> ExperimentSummary:
    spec = load_experiment_spec(path)
    values = dict(spec.values)
    robot = str(values.get("robot") or "a1_iiil")
    base_task = str(values.get("base_task") or "")
    name = str(values.get("name") or default_experiment_name(robot, base_task or "experiment/base"))
    title = spec.title or name
    demo_counts = [str(item) for item in values.get("demo_counts", [])]
    rounds = int(values.get("rounds") or 0)
    methods = normalize_optional_str_list(values.get("methods", []))
    phases = normalize_optional_str_list(values.get("phases", []))
    iteration_method = None if values.get("iteration_method") is None else str(values.get("iteration_method"))

    root_dir_value = values.get("root_dir")
    if isinstance(root_dir_value, str) and root_dir_value:
        root_dir = Path(root_dir_value).expanduser().resolve()
    else:
        root_dir = (REPO_ROOT / "outputs" / "experiments" / name).resolve()

    config_path = root_dir / "experiment_config.json"
    has_run_artifacts = config_path.exists()
    last_launch_mode: Optional[str] = None
    launch_command = f"./experiment.sh --spec {shlex.quote(_display_path(path))}"
    if has_run_artifacts:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}
        if "spec_title" in config and config["spec_title"]:
            title = str(config["spec_title"])
        if "spec_tags" in config and config["spec_tags"]:
            tags = normalize_optional_str_list(config["spec_tags"])
        else:
            tags = spec.tags
        if "robot" in config and config["robot"]:
            robot = str(config["robot"])
        if "base_task" in config and config["base_task"]:
            base_task = str(config["base_task"])
        if "name" in config and config["name"]:
            name = str(config["name"])
        if "demo_counts" in config and config["demo_counts"]:
            demo_counts = [str(item) for item in config["demo_counts"]]
        if "rounds" in config and config["rounds"] is not None:
            rounds = int(config["rounds"])
        if "methods" in config and config["methods"]:
            methods = [str(item) for item in config["methods"]]
        if "phases" in config and config["phases"]:
            phases = [str(item) for item in config["phases"]]
        if "iteration_method" in config and config["iteration_method"]:
            iteration_method = str(config["iteration_method"])
        if "root_dir" in config and config["root_dir"]:
            root_dir = Path(str(config["root_dir"])).expanduser().resolve()
        if "dry_run" in config:
            last_launch_mode = "dry-run" if bool(config["dry_run"]) else "run"
        if "launch_command" in config and config["launch_command"]:
            launch_command = str(config["launch_command"])
    else:
        tags = spec.tags

    return ExperimentSummary(
        path=path,
        title=title,
        name=name,
        robot=robot,
        base_task=base_task,
        demo_counts=demo_counts,
        rounds=rounds,
        methods=methods,
        iteration_method=iteration_method,
        phases=phases,
        tags=tags,
        root_dir=root_dir,
        has_run_artifacts=has_run_artifacts,
        last_launch_mode=last_launch_mode,
        launch_command=launch_command,
    )


def _display_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def print_summary_block(summary: ExperimentSummary, idx: Optional[int] = None) -> None:
    prefix = f"{_menu_number(idx)}. " if idx is not None else f"{_accent('-')} "
    print(f"{prefix}{_bold(summary.title)}")
    print(f"   {_dim('status')}: {_format_status(summary)}")
    print(f"   {_dim('robot')}: {summary.robot}    {_dim('task')}: {summary.base_task or '?'}")
    print(
        f"   {_dim('demos')}: {','.join(summary.demo_counts) or '?'}    "
        f"{_dim('rounds')}: {summary.rounds}    "
        f"{_dim('methods')}: {','.join(summary.methods) or '?'}"
    )
    if summary.iteration_method or summary.phases or summary.tags:
        print(
            f"   {_dim('iter')}: {summary.iteration_method or '-'}    "
            f"{_dim('phases')}: {','.join(summary.phases) or 'full'}    "
            f"{_dim('tags')}: {','.join(summary.tags) or '-'}"
        )
    print(f"   {_dim('root')}: {_display_path(summary.root_dir)}")
    print(f"   {_dim('spec')}: {_display_path(summary.path)}")
    print(f"   {_dim('launch')}: {summary.launch_command}")


def list_cards() -> list[Path]:
    cards = sorted(path for path in EXAMPLES_DIR.glob("*.md") if path.name != "README.md")
    if DEFAULT_CARDS_DIR.exists():
        cards.extend(sorted(path for path in DEFAULT_CARDS_DIR.glob("*.md") if path.name != "README.md"))
    return cards


def list_runnable_cards() -> list[Path]:
    return [path for path in list_cards() if path.name != "TEMPLATE.md"]


def list_example_cards() -> list[Path]:
    return sorted(path for path in EXAMPLES_DIR.glob("*.md") if path.name != "README.md")


def list_local_cards() -> list[Path]:
    if not DEFAULT_CARDS_DIR.exists():
        return []
    return sorted(path for path in DEFAULT_CARDS_DIR.glob("*.md") if path.name != "README.md")


def print_card_list() -> None:
    _print_header("Example Cards", "Reference specs you can dry-run or use as a starting point.")
    for path in list_example_cards():
        print_summary_block(summarize_card(path))
        print("")
    user_cards = list_local_cards()
    if user_cards:
        _print_header("Local Cards", "Cards you created locally with the wizard.")
        for path in user_cards:
            print_summary_block(summarize_card(path))
            print("")


def choose_preset(default_key: Optional[str] = None) -> WizardPreset:
    _print_header("Choose A Preset", "Start from the closest experiment shape, then edit the details.")
    for idx, preset in enumerate(PRESETS, start=1):
        print(f"  {_menu_number(idx)}. {_bold(preset.label)}")
        print(f"     {_dim(preset.description)}")
    default_idx = next(
        (idx for idx, preset in enumerate(PRESETS, start=1) if preset.key == default_key),
        2,
    )
    while True:
        raw = input(f"{_bold('Preset')} {_dim(f'[{default_idx}]')}: ").strip()
        if not raw:
            return PRESETS[default_idx - 1]
        try:
            idx = int(raw)
        except ValueError:
            _print_error("Enter a number from the preset list.")
            continue
        if 1 <= idx <= len(PRESETS):
            return PRESETS[idx - 1]
        _print_error("Enter a valid preset number.")


def choose_numbered_action(title: str, items: list[str], default_idx: int = 1) -> int:
    _print_header(title)
    for idx, item in enumerate(items, start=1):
        marker = _dim(" [default]") if idx == default_idx else ""
        print(f"  {_menu_number(idx)}. {_bold(item)}{marker}")
    while True:
        raw = input(f"{_bold('Select')} {_dim(f'[{default_idx}]')}: ").strip()
        if not raw:
            return default_idx
        try:
            idx = int(raw)
        except ValueError:
            _print_error("Enter a menu number.")
            continue
        if 1 <= idx <= len(items):
            return idx
        _print_error("Enter a valid menu number.")


def choose_card_from_list(cards: list[Path], heading: str) -> Optional[Path]:
    if not cards:
        _print_error("No cards found.")
        return None
    _print_header(heading, "Pick the experiment you want to inspect or launch.")
    for idx, path in enumerate(cards, start=1):
        print_summary_block(summarize_card(path), idx=idx)
        print("")
    print(f"  {_menu_number(0)}. {_bold('Back')}")
    while True:
        raw = input(f"{_bold('Select card')} {_dim('[0]')}: ").strip()
        if not raw:
            return None
        try:
            idx = int(raw)
        except ValueError:
            _print_error("Enter a menu number.")
            continue
        if idx == 0:
            return None
        if 1 <= idx <= len(cards):
            return cards[idx - 1]
        _print_error("Enter a valid card number.")


def ensure_valid_methods(methods: list[str], rounds: int) -> list[str]:
    parsed = parse_methods(methods)
    if rounds > 1 and "residual" in parsed:
        raise ValueError(
            "Multi-round runs do not support residual. Use demos, cotrain, hgdagger, and/or sirius."
        )
    return parsed


def prompt_methods(rounds: int, default: Iterable[object]) -> list[str]:
    while True:
        methods = prompt_csv_list(
            f"Methods ({','.join(AVAILABLE_METHODS)})",
            default,
        )
        try:
            return ensure_valid_methods(methods, rounds)
        except ValueError as exc:
            _print_error(str(exc))


def build_interactive_card(preset: WizardPreset, cards_dir: Path) -> tuple[Path, dict[str, object], str]:
    values = dict(preset.defaults)
    notes = str(values.pop("notes", "")).strip()

    _print_header("Create Experiment", f"Using preset: {preset.label}")
    _print_note(preset.description)

    values["title"] = prompt_line("Title", str(values.get("title", "")))
    default_name = str(values.get("name") or values["title"]).replace(" ", "_").lower()
    values["name"] = prompt_line("Experiment name", default_name)
    values["robot"] = prompt_choice("Robot", ["a1_iiil", "r1pro_iiil"], str(values.get("robot", "a1_iiil")))
    values["base_task"] = prompt_line("Base task", str(values.get("base_task", "pnp/base")))

    default_collection_task = str(values.get("collection_task") or infer_collection_task(str(values["base_task"])))
    values["collection_task"] = prompt_line("Collection task", default_collection_task)

    default_collect_env = None if values["robot"] == "a1_iiil" else str(values.get("collect_env", "sim_r1pro_joint"))
    if values["robot"] == "a1_iiil":
        collect_env = prompt_optional_path("Collect env override", str(values.get("collect_env") or "")) or None
        eval_env = prompt_optional_path("Eval env override", str(values.get("eval_env") or "")) or None
        if collect_env is not None:
            values["collect_env"] = collect_env
        else:
            values.pop("collect_env", None)
        if eval_env is not None:
            values["eval_env"] = eval_env
        else:
            values.pop("eval_env", None)
    else:
        values["collect_env"] = prompt_line("Collect env", default_collect_env or "sim_r1pro_joint")
        values["eval_env"] = prompt_line("Eval env", str(values.get("eval_env") or values["collect_env"]))

    values["demo_counts"] = prompt_csv_ints("Demo counts", values.get("demo_counts", [20]))
    values["rounds"] = prompt_optional_int("Correction rounds", int(values.get("rounds", 1))) or 0
    values["methods"] = prompt_methods(int(values["rounds"]), values.get("methods", ["demos", "cotrain"]))

    if int(values["rounds"]) > 0:
        iteration_choices = normalize_optional_str_list(values["methods"])
        if "demos" not in iteration_choices:
            iteration_choices.append("demos")
        default_iteration = str(
            values.get("iteration_method")
            or resolve_iteration_method(None, normalize_optional_str_list(values["methods"]), int(values["rounds"]))
        )
        values["iteration_method"] = prompt_choice(
            "Iteration method",
            iteration_choices,
            default_iteration,
        )
        values["correction_episodes"] = prompt_optional_int(
            "Correction episodes per round",
            values.get("correction_episodes"),  # type: ignore[arg-type]
        )
    else:
        values.pop("iteration_method", None)
        values.pop("correction_episodes", None)

    values["base_demo_source"] = prompt_optional_path(
        "Existing base demo source",
        None if values.get("base_demo_source") is None else str(values["base_demo_source"]),
    )
    values["root_dir"] = prompt_optional_path(
        "Root dir override",
        None if values.get("root_dir") is None else str(values["root_dir"]),
    )
    values["train_max_steps"] = prompt_optional_int(
        "Train max steps",
        values.get("train_max_steps"),  # type: ignore[arg-type]
    )
    values["eval_repeats"] = prompt_optional_int(
        "Eval repeats",
        values.get("eval_repeats"),  # type: ignore[arg-type]
    )
    values["use_wandb"] = prompt_bool("Use W&B", bool(values.get("use_wandb", True)))
    tags_default = normalize_optional_str_list(values.get("tags", []))
    values["tags"] = prompt_csv_list("Tags", tags_default or ["experiment"])

    notes = prompt_notes(notes)

    cards_dir.mkdir(parents=True, exist_ok=True)
    spec_path = cards_dir / f"{values['name']}.md"
    spec_path_input = prompt_optional_path("Card output path", str(spec_path))
    spec_path = Path(spec_path_input or spec_path).expanduser()
    if not spec_path.is_absolute():
        spec_path = (REPO_ROOT / spec_path).resolve()

    cleaned = {key: value for key, value in values.items() if not is_empty_value(value)}
    return spec_path, cleaned, notes


def run_pipeline_for_spec(spec_path: Path, dry_run: bool) -> int:
    command = [sys.executable, str(REPO_ROOT / "scripts" / "experiment_pipeline.py"), "--spec", str(spec_path)]
    if dry_run:
        command.append("--dry-run")
    else:
        spec = load_experiment_spec(spec_path)
        if bool(spec.values.get("dry_run")):
            command.append("--no-dry-run")
    return subprocess.call(command, cwd=REPO_ROOT)


def choose_run_mode(default_idx: int = 1) -> str:
    idx = choose_numbered_action(
        "Launch Mode",
        [
            "Dry-run this experiment",
            "Run this experiment",
        ],
        default_idx=default_idx,
    )
    return ["dry-run", "run"][idx - 1]


def run_existing_card(cards: list[Path], *, dry_run: bool) -> Optional[int]:
    card = choose_card_from_list(
        cards,
        "Choose A Saved Experiment",
    )
    if card is None:
        return None
    display = card.relative_to(REPO_ROOT) if card.is_relative_to(REPO_ROOT) else card
    _print_header("Selected Experiment")
    print(f"{_dim('spec')}: {display}")
    print(f"{_dim('mode')}: {'dry-run' if dry_run else 'run'}")
    return run_pipeline_for_spec(card, dry_run=dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive experiment card wizard.")
    parser.add_argument("--preset", default=None, help="Preset key to start from.")
    parser.add_argument("--cards-dir", default=str(DEFAULT_CARDS_DIR), help="Directory for generated experiment cards.")
    parser.add_argument("--list-cards", action="store_true", help="List available example and local cards.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_cards:
        print_card_list()
        return 0

    if args.preset is not None:
        preset = next((item for item in PRESETS if item.key == args.preset), None)
        if preset is None:
            raise ValueError(f"Unknown preset '{args.preset}'.")
        _print_header("IIIL Experiment Runner", "This wizard saves the experiment definition first, then launches it in dry-run or run mode.")
        spec_path, values, notes = build_interactive_card(preset, Path(args.cards_dir).expanduser())
        card_text = render_card(values, notes)
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(card_text, encoding="utf-8")

        display_path = spec_path.relative_to(REPO_ROOT) if spec_path.is_relative_to(REPO_ROOT) else spec_path
        _print_header("Experiment Saved")
        print(f"{_dim('card')}: {display_path}")
        print(f"{_dim('dry-run')}: ./experiment.sh --spec {shlex.quote(str(display_path))} --dry-run")
        print(f"{_dim('run')}: ./experiment.sh --spec {shlex.quote(str(display_path))}")

        action = choose_run_mode(default_idx=1)
        return run_pipeline_for_spec(spec_path, dry_run=(action == "dry-run"))

    while True:
        choice = choose_numbered_action(
            "IIIL Experiment Runner",
            [
                "Create a new experiment card",
                "Dry-run a saved experiment",
                "Run a saved experiment",
                "List saved experiments",
                "Exit",
            ],
            default_idx=1,
        )
        if choice == 1:
            preset = choose_preset()
            _print_header("IIIL Experiment Runner", "This wizard saves the experiment definition first, then launches it in dry-run or run mode.")
            spec_path, values, notes = build_interactive_card(preset, Path(args.cards_dir).expanduser())
            card_text = render_card(values, notes)
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(card_text, encoding="utf-8")

            display_path = spec_path.relative_to(REPO_ROOT) if spec_path.is_relative_to(REPO_ROOT) else spec_path
            _print_header("Experiment Saved")
            print(f"{_dim('card')}: {display_path}")
            print(f"{_dim('dry-run')}: ./experiment.sh --spec {shlex.quote(str(display_path))} --dry-run")
            print(f"{_dim('run')}: ./experiment.sh --spec {shlex.quote(str(display_path))}")

            action = choose_run_mode(default_idx=1)
            return run_pipeline_for_spec(spec_path, dry_run=(action == "dry-run"))
        if choice == 2:
            result = run_existing_card(list_runnable_cards(), dry_run=True)
            if result is None:
                continue
            return result
        if choice == 3:
            result = run_existing_card(list_runnable_cards(), dry_run=False)
            if result is None:
                continue
            return result
        if choice == 4:
            print_card_list()
            continue
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

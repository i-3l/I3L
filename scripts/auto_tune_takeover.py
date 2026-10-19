#!/usr/bin/env python3
"""
Auto-tune takeover detection thresholds or PID goal-tracking gains.

Starts OmniGibson once, then loops episodes in-process using task.reset_episode().

Threshold modes (current / position / both):
  After each episode, adjusts thresholds:
    - Joints that triggered takeover: bump threshold UP (less sensitive)
    - Joints that didn't trigger:    bump threshold DOWN (more sensitive)
  Converges toward thresholds where only intentional pushes trigger takeover.

PID mode:
  After each episode, collects per-joint max tracking error and adjusts kp/kd/kff:
    - If a joint's error improved vs previous iteration: increase gains
    - If a joint's error got worse: decrease gains
    - If within noise band: hold
  Run until satisfied, Ctrl-C to stop (gains saved each iteration).

Usage:
    python scripts/auto_tune_takeover.py --mode current          # default
    python scripts/auto_tune_takeover.py --mode position
    python scripts/auto_tune_takeover.py --mode both
    python scripts/auto_tune_takeover.py --mode pid              # NEW
    python scripts/auto_tune_takeover.py --mode pid --task open/base --converge 10
"""

import argparse
import re
import sys
import os
import shutil
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from iiil import IIILTask

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent                        # IIIL/IIIL/
GELLO_CONFIG = "gello_r1pro"
GELLO_TASK_OVERLAY = "autotune"
CONFIG_PATH = REPO_DIR / "iiil" / "configs" / "gello" / GELLO_CONFIG / f"{GELLO_TASK_OVERLAY}.yaml"
BASE_CONFIG_PATH = REPO_DIR / "iiil" / "configs" / "gello" / f"{GELLO_CONFIG}.yaml"

# ── Threshold tuning knobs ────────────────────────────────────────
UP_FACTOR = 1.12       # +12% for joints that triggered
DOWN_FACTOR = 0.97     # −3% for joints that didn't trigger

CURRENT_MIN, CURRENT_MAX = 5.0, 500.0     # mA bounds
POSITION_MIN, POSITION_MAX = 0.01, 1.0    # rad bounds

PAUSE_BETWEEN_RUNS = 3  # seconds

# ── PID tuning knobs — per-joint adaptive: increase if improving, back off if worse
PID_KFF_UP = 1.05          # +5% Kff when error improved
PID_KFF_DOWN = 0.95        # -5% Kff when error got worse
PID_KP_UP = 1.02           # +2% Kp when error improved
PID_KP_DOWN = 0.98         # -2% Kp when error got worse
PID_KD_UP = 1.02           # +2% Kd when error improved
PID_KD_DOWN = 0.98         # -2% Kd when error got worse
PID_KFF_MIN, PID_KFF_MAX = 0.01, 2.0
PID_KP_MIN, PID_KP_MAX = 0.05, 5.0
PID_KD_MIN, PID_KD_MAX = 0.005, 1.0
PID_IMPROVE_THRESHOLD = 0.005  # rad — min error change to count as improvement/regression


# ── CLI ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Auto-tune Gello parameters")
    p.add_argument("--mode", choices=["current", "position", "both", "pid"],
                   default="current", help="What to tune (default: current)")
    p.add_argument("--task", default="autotune/base",
                   help="Hydra task override (default: autotune/base)")
    p.add_argument("--converge", type=int, default=5,
                   help="Consecutive clean runs to converge (default: 5)")
    p.add_argument("--episodes", type=int, default=3,
                   help="Episodes to average per gain step in PID mode (default: 3)")
    return p.parse_args()


# ── ANSI helpers ───────────────────────────────────────────────────
_BOLD   = "\033[1m"
_RED    = "\033[1;31m"
_GREEN  = "\033[1;32m"
_YELLOW = "\033[1;33m"
_CYAN   = "\033[1;36m"
_DIM    = "\033[90m"
_RESET  = "\033[0m"


# ── YAML read / write ─────────────────────────────────────────────
def read_thresholds(path: Path):
    """Return (current_list, position_list) parsed from YAML."""
    text = path.read_text(encoding="utf-8")

    # Match only uncommented lines (no leading #)
    m = re.search(r'^[ \t]+current_threshold:\s*\[([^\]]+)\]', text, re.DOTALL | re.MULTILINE)
    if not m:
        raise ValueError("Could not find current_threshold in config")
    current = [float(x) for x in re.findall(r'[\d.]+', m.group(1))]

    m = re.search(r'^[ \t]+position_threshold:\s*\[([^\]]+)\]', text, re.DOTALL | re.MULTILINE)
    if not m:
        raise ValueError("Could not find position_threshold in config")
    position = [float(x) for x in re.findall(r'[\d.]+', m.group(1))]

    return current, position


def write_thresholds(path: Path, current: list, position: list):
    """Write thresholds back to YAML preserving surrounding structure."""
    text = path.read_text(encoding="utf-8")
    half = len(current) // 2

    # ── current_threshold (only match uncommented lines) ──
    r_c = ', '.join(f'{v:.1f}' for v in current[:half])
    l_c = ', '.join(f'{v:.1f}' for v in current[half:])
    new_ct = (f'  current_threshold: [{r_c},   # right arm (mA)\n'
              f'                      {l_c}]   # left arm (mA)')
    text = re.sub(
        r'^([ \t]+)current_threshold:\s*\[.*?\][^\n]*',
        new_ct, text, count=1, flags=re.DOTALL | re.MULTILINE,
    )

    # ── position_threshold (only match uncommented lines) ──
    r_p = ', '.join(f'{v:.2f}' for v in position[:half])
    l_p = ', '.join(f'{v:.2f}' for v in position[half:])
    new_pt = (f'  position_threshold: [{r_p},      # right arm (rad)\n'
              f'                       {l_p}]      # left arm (rad)')
    text = re.sub(
        r'^([ \t]+)position_threshold:\s*\[.*?\][^\n]*',
        new_pt, text, count=1, flags=re.DOTALL | re.MULTILINE,
    )

    path.write_text(text, encoding="utf-8")


def ensure_detection_methods(path: Path, tune_mode: str):
    """Set use_current / use_position based on tune_mode."""
    text = path.read_text(encoding="utf-8")

    if tune_mode == "current":
        text = re.sub(r'(use_current:\s*)false', r'\1true', text)
        text = re.sub(r'(use_position:\s*)true', r'\1false', text)
    elif tune_mode == "position":
        text = re.sub(r'(use_current:\s*)true', r'\1false', text)
        text = re.sub(r'(use_position:\s*)false', r'\1true', text)
    elif tune_mode == "both":
        text = re.sub(r'(use_current:\s*)false', r'\1true', text)
        text = re.sub(r'(use_position:\s*)false', r'\1true', text)

    path.write_text(text, encoding="utf-8")


def get_detector(task):
    """Get the takeover detector from a running task."""
    return task.policy._components.gello_controller._takeover_detector


def build_event_from_detector(detector):
    """Build per-joint takeover event from detector's live rolling buffers.

    Reads the same buffers and thresholds that detect_takeover() uses internally
    (takeover_detector.py:220-230). Produces a dict compatible with adjust().
    """
    event = {'current': {}, 'position': {}}
    if detector.config.use_current:
        avg = np.mean(detector.buffers['current'], axis=0)
        for j in range(detector.dof):
            event['current'][j] = bool(avg[j] > detector.current_threshold[j])
    if detector.config.use_position:
        avg = np.mean(detector.buffers['position'], axis=0)
        for j in range(detector.dof):
            event['position'][j] = bool(avg[j] > detector.position_threshold[j])
    return event


# ── Hydra config ──────────────────────────────────────────────────
def build_config(task: str = "autotune/base"):
    """Build the same config as run.sh mode 7 using Hydra compose API."""
    GlobalHydra.instance().clear()
    config_dir = str(REPO_DIR / "iiil" / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="base_config", overrides=[
            "env=sim_r1pro_joint",
            f"task={task}",
            "policy=gello",
            "policy.base_policy=curobo",
            "policy.require_joycon=false",
            f"policy.gello_task_overlay={GELLO_TASK_OVERLAY}",
            "hdf_path=null",
        ])
    OmegaConf.resolve(cfg)
    return cfg


# ── Episode runner ────────────────────────────────────────────────
def run_episode(task):
    """Run one episode, return takeover event dict or None for clean run."""
    detector = get_detector(task)
    while True:
        terminated, truncated = task.step()

        if detector._last_takeover_state:
            event = build_event_from_detector(detector)
            if event.get('current') or event.get('position'):
                print(f"\n{_YELLOW}>>> Takeover detected (polled) — resetting...{_RESET}")
                task.reset_episode()
                return event

        if terminated or truncated:
            task.reset_episode()
            return None


# ── In-process threshold update ───────────────────────────────────
def apply_thresholds_in_process(task, new_current, new_position):
    """Mutate detector's numpy arrays directly so changes take effect immediately."""
    detector = get_detector(task)
    detector.current_threshold[:] = np.array(new_current, dtype=np.float32)
    detector.position_threshold[:] = np.array(new_position, dtype=np.float32)


# ── Threshold adjustment ───────────────────────────────────────────
def adjust(current: list, position: list, event: dict, dof: int,
           tune_mode: str = "both"):
    """Apply single-event bumps and return new thresholds."""
    c = list(current)
    p = list(position)

    # Current thresholds
    if tune_mode in ("current", "both"):
        for j in range(dof):
            if j in event.get('current', {}):
                if event['current'][j]:     # triggered
                    c[j] = min(c[j] * UP_FACTOR, CURRENT_MAX)
                else:                       # didn't trigger
                    c[j] = max(c[j] * DOWN_FACTOR, CURRENT_MIN)

    # Position thresholds
    if tune_mode in ("position", "both"):
        for j in range(dof):
            if j in event.get('position', {}):
                if event['position'][j]:    # triggered
                    p[j] = min(p[j] * UP_FACTOR, POSITION_MAX)
                else:                       # didn't trigger
                    p[j] = max(p[j] * DOWN_FACTOR, POSITION_MIN)

    return c, p


# ── PID gain YAML read / write ───────────────────────────────────
def read_pid_gains(path: Path):
    """Read kp/kd/kff arrays from YAML (first occurrence of each)."""
    text = path.read_text(encoding="utf-8")
    kp_m = re.search(r'^\s+kp:\s*\[([^\]]+)\]', text, re.MULTILINE)
    kd_m = re.search(r'^\s+kd:\s*\[([^\]]+)\]', text, re.MULTILINE)
    kff_m = re.search(r'^\s+kff:\s*\[([^\]]+)\]', text, re.MULTILINE)
    if not kp_m or not kd_m:
        raise ValueError(f"Could not find kp/kd arrays in {path}")
    kp = [float(x) for x in re.findall(r'[\d.]+', kp_m.group(1))]
    kd = [float(x) for x in re.findall(r'[\d.]+', kd_m.group(1))]
    kff = [float(x) for x in re.findall(r'[\d.]+', kff_m.group(1))] if kff_m else None
    return kp, kd, kff


def write_pid_gains(path: Path, kp: list, kd: list, kff: list = None):
    """Write kp/kd/kff arrays back into YAML via regex substitution (first occurrence)."""
    text = path.read_text(encoding="utf-8")
    fmt_kp = '  kp: [' + ', '.join(f'{v:.3f}' for v in kp) + ']'
    fmt_kd = '  kd: [' + ', '.join(f'{v:.4f}' for v in kd) + ']'
    text = re.sub(r'^(\s+)kp:\s*\[[^\]]+\]', fmt_kp, text, count=1, flags=re.MULTILINE)
    text = re.sub(r'^(\s+)kd:\s*\[[^\]]+\]', fmt_kd, text, count=1, flags=re.MULTILINE)
    if kff is not None and re.search(r'^\s+kff:\s*\[', text, re.MULTILINE):
        fmt_kff = '    kff: [' + ', '.join(f'{v:.3f}' for v in kff) + ']'
        text = re.sub(r'^(\s+)kff:\s*\[[^\]]+\]', fmt_kff, text, count=1, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")


# ── PID episode runner ────────────────────────────────────────────
def run_episode_pid(task):
    """Run a full episode collecting tracking error. Returns metrics dict or None."""
    controller = task.policy._components.gello_controller
    tracker = controller._goal_tracker
    errors = []

    while True:
        terminated, truncated = task.step()
        if tracker.is_tracking and tracker._goal_position is not None:
            pos = controller._cached_positions
            goal = tracker._goal_position
            if pos is not None and goal is not None:
                errors.append(np.abs(goal - pos))
        if terminated or truncated:
            task.reset_episode()
            break

    if not errors:
        return None
    return {
        'max_error': np.max(errors, axis=0),  # (14,)
    }


# ── PID gain adjustment ──────────────────────────────────────────
def adjust_pid(kp: list, kd: list, kff: list, metrics: dict, dof_per_arm: int = 7):
    """Per-joint adaptive adjustment based on error improvement. Returns (new_kp, new_kd, new_kff)."""
    new_kp, new_kd, new_kff = list(kp), list(kd), list(kff)
    max_err = metrics.get('max_error')
    prev_err = metrics.get('prev_max_error')

    for j in range(dof_per_arm):
        if max_err is None:
            continue

        err = max(max_err[j], max_err[j + dof_per_arm])

        if prev_err is None:
            # First iteration: increase all
            new_kff[j] = min(kff[j] * PID_KFF_UP, PID_KFF_MAX)
            new_kp[j] = min(kp[j] * PID_KP_UP, PID_KP_MAX)
            new_kd[j] = min(kd[j] * PID_KD_UP, PID_KD_MAX)
        else:
            prev = prev_err[j]
            delta = prev - err  # positive = improved

            if delta > PID_IMPROVE_THRESHOLD:
                # Error decreased: keep pushing gains up
                new_kff[j] = min(kff[j] * PID_KFF_UP, PID_KFF_MAX)
                new_kp[j] = min(kp[j] * PID_KP_UP, PID_KP_MAX)
                new_kd[j] = min(kd[j] * PID_KD_UP, PID_KD_MAX)
            elif delta < -PID_IMPROVE_THRESHOLD:
                # Error increased: back off
                new_kff[j] = max(kff[j] * PID_KFF_DOWN, PID_KFF_MIN)
                new_kp[j] = max(kp[j] * PID_KP_DOWN, PID_KP_MIN)
                new_kd[j] = max(kd[j] * PID_KD_DOWN, PID_KD_MIN)
            # else: within noise band — hold

    return new_kp, new_kd, new_kff


# ── PID in-process helpers ───────────────────────────────────────
def apply_pid_gains_in_process(task, kp: list, kd: list, kff: list = None):
    """Apply PID gains to the live controller (7 values duplicated to 14)."""
    kp_14 = np.array(kp + kp, dtype=np.float32)
    kd_14 = np.array(kd + kd, dtype=np.float32)
    task.policy._components.gello_controller.set_tracking_gains(kp=kp_14, kd=kd_14)
    if kff is not None:
        tracker = task.policy._components.gello_controller._goal_tracker
        tracker.config.kff[:] = np.array(kff + kff, dtype=np.float32)


def disable_takeover_for_pid(task):
    """Prevent takeover triggering during PID tuning.

    Disables all detection methods so no threshold comparison fires,
    while keeping _runtime_enabled=True so detect_takeover() still runs
    and populates the position buffer for the dashboard display.
    """
    detector = get_detector(task)
    detector.config.use_current = False
    detector.config.use_position = False
    detector.config.use_velocity = False
    detector.config.use_momentum = False


# ── Main loop ──────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.mode == "pid":
        _main_pid(args)
    else:
        _main_threshold(args)


def _main_threshold(args):
    """Threshold tuning modes: current, position, both."""
    if not CONFIG_PATH.exists():
        print(f"{_RED}Config not found: {CONFIG_PATH}{_RESET}")
        sys.exit(1)

    # Set detection methods based on tune mode
    ensure_detection_methods(CONFIG_PATH, args.mode)

    # Build config via Hydra compose API (same as run.sh mode 7)
    cfg = build_config(task=args.task)
    os.chdir(REPO_DIR)

    n_episodes = args.episodes

    print(f"{_CYAN}{'=' * 70}{_RESET}")
    print(f"{_CYAN} AUTO-TUNE TAKEOVER THRESHOLDS (in-process){_RESET}")
    print(f"{_DIM}  Tune mode     : {args.mode}{_RESET}")
    print(f"{_DIM}  Task          : {args.task}{_RESET}")
    print(f"{_DIM}  Episodes      : {n_episodes} per step{_RESET}")
    print(f"{_DIM}  Clean runs    : {args.converge} consecutive clean steps to converge{_RESET}")
    print(f"{_CYAN}{'=' * 70}{_RESET}")

    try:
        with IIILTask(cfg) as task:
            clean_streak = 0
            step = 0

            while True:
                step += 1
                current, position = read_thresholds(CONFIG_PATH)
                dof = len(current)

                print(f"\n{_CYAN}{'=' * 70}{_RESET}")
                print(f"{_CYAN} AUTO-TUNE  step {step}  "
                      f"({datetime.now().strftime('%H:%M:%S')})  "
                      f"— running {n_episodes} episodes{_RESET}")
                print(f"{_DIM}  Current  (mA) : {['%.1f' % v for v in current]}{_RESET}")
                print(f"{_DIM}  Position (rad): {['%.2f' % v for v in position]}{_RESET}")
                print(f"{_CYAN}{'=' * 70}{_RESET}\n")

                # Run N episodes, collect per-joint trigger counts
                trigger_count = {'current': np.zeros(dof, dtype=int),
                                 'position': np.zeros(dof, dtype=int)}
                any_takeover = False

                for ep in range(n_episodes):
                    print(f"  {_DIM}episode {ep + 1}/{n_episodes}...{_RESET}", end="", flush=True)
                    event = run_episode(task)
                    if event is None:
                        print(f"  {_GREEN}clean{_RESET}")
                    else:
                        any_takeover = True
                        triggered_joints = []
                        for kind in ('current', 'position'):
                            for j, fired in event.get(kind, {}).items():
                                if fired:
                                    trigger_count[kind][j] += 1
                                    triggered_joints.append(f"{kind[0].upper()}{j}")
                        print(f"  {_YELLOW}takeover: {', '.join(triggered_joints) or '?'}{_RESET}")
                    if ep < n_episodes - 1:
                        time.sleep(PAUSE_BETWEEN_RUNS)

                # Show trigger summary
                print(f"\n  {_BOLD}Trigger counts across {n_episodes} episodes:{_RESET}")
                if args.mode in ("current", "both"):
                    print(f"    Current : {list(trigger_count['current'])}")
                if args.mode in ("position", "both"):
                    print(f"    Position: {list(trigger_count['position'])}")

                if not any_takeover:
                    clean_streak += 1
                    print(f"\n{_GREEN}Clean step {clean_streak}/{args.converge} "
                          f"— all {n_episodes} episodes clean.{_RESET}")

                    if clean_streak >= args.converge:
                        current, position = read_thresholds(CONFIG_PATH)
                        print(f"\n{_GREEN}{'=' * 70}{_RESET}")
                        print(f"{_GREEN} SUCCESS — {args.converge} consecutive "
                              f"clean steps ({args.converge * n_episodes} episodes), "
                              f"tuning complete!{_RESET}")
                        print(f"{_GREEN}{'=' * 70}{_RESET}")
                        print(f"  {_BOLD}Final current  (mA) : "
                              f"{['%.1f' % v for v in current]}{_RESET}")
                        print(f"  {_BOLD}Final position (rad): "
                              f"{['%.2f' % v for v in position]}{_RESET}")
                        print(f"  Config: {CONFIG_PATH}")
                        break
                else:
                    clean_streak = 0

                    # Build aggregate event: joint triggered if it fired in ANY episode
                    agg_event = {'current': {}, 'position': {}}
                    for j in range(dof):
                        if args.mode in ("current", "both"):
                            agg_event['current'][j] = bool(trigger_count['current'][j] > 0)
                        if args.mode in ("position", "both"):
                            agg_event['position'][j] = bool(trigger_count['position'][j] > 0)

                    new_current, new_position = adjust(current, position, agg_event, dof,
                                                       tune_mode=args.mode)

                    # Show changes
                    print(f"\n{_BOLD}Threshold adjustments:{_RESET}")
                    if args.mode in ("current", "both"):
                        print(f"  {_BOLD}Current (mA):{_RESET}")
                        for j in range(dof):
                            d = new_current[j] - current[j]
                            arrow = f"{_GREEN}+{_RESET}" if d > 0.01 else f"{_RED}-{_RESET}" if d < -0.01 else "="
                            print(f"    J{j:2d}: {current[j]:6.1f} -> {new_current[j]:6.1f}  {arrow}")
                    if args.mode in ("position", "both"):
                        print(f"  {_BOLD}Position (rad):{_RESET}")
                        for j in range(dof):
                            d = new_position[j] - position[j]
                            arrow = f"{_GREEN}+{_RESET}" if d > 0.001 else f"{_RED}-{_RESET}" if d < -0.001 else "="
                            print(f"    J{j:2d}: {position[j]:6.3f} -> {new_position[j]:6.3f}  {arrow}")

                    # Backup then write to YAML
                    backup = CONFIG_PATH.with_suffix(f'.yaml.iter{step}')
                    shutil.copy2(CONFIG_PATH, backup)
                    write_thresholds(CONFIG_PATH, new_current, new_position)
                    print(f"\n{_GREEN}Updated {CONFIG_PATH.name} (backup: {backup.name}){_RESET}")

                    # Live update thresholds in-process (no restart needed)
                    apply_thresholds_in_process(task, new_current, new_position)

                print(f"\n{_DIM}Next step in {PAUSE_BETWEEN_RUNS}s...{_RESET}")
                time.sleep(PAUSE_BETWEEN_RUNS)

    except (KeyboardInterrupt, SystemExit):
        print(f"\n{_YELLOW}Interrupted — keeping current thresholds.{_RESET}")


def _main_pid(args):
    """PID goal-tracking gain tuning mode — per-joint adaptive."""
    for path in (BASE_CONFIG_PATH, CONFIG_PATH):
        if not path.exists():
            print(f"{_RED}Config not found: {path}{_RESET}")
            sys.exit(1)

    # Read initial gains from base config (kff only in base, overlay may not have it)
    kp, kd, kff = read_pid_gains(BASE_CONFIG_PATH)
    if kff is None:
        print(f"{_RED}No kff array found in {BASE_CONFIG_PATH}{_RESET}")
        sys.exit(1)

    # Build config via Hydra compose API
    cfg = build_config(task=args.task)
    os.chdir(REPO_DIR)

    n_episodes = args.episodes

    print(f"{_CYAN}{'=' * 70}{_RESET}")
    print(f"{_CYAN} AUTO-TUNE PID — adaptive per-joint "
          f"(Kff ±{(PID_KFF_UP - 1) * 100:.0f}%, Kp/Kd ±{(PID_KP_UP - 1) * 100:.0f}%){_RESET}")
    print(f"{_DIM}  Task      : {args.task}{_RESET}")
    print(f"{_DIM}  Episodes  : {n_episodes} averaged per gain step{_RESET}")
    print(f"{_DIM}  Threshold : {PID_IMPROVE_THRESHOLD:.3f} rad (noise band){_RESET}")
    print(f"{_DIM}  Ctrl-C when satisfied (gains saved each step){_RESET}")
    print(f"{_CYAN}{'=' * 70}{_RESET}")

    try:
        with IIILTask(cfg) as task:
            # Disable takeover so episodes run to completion
            disable_takeover_for_pid(task)

            # Apply initial gains in-process
            apply_pid_gains_in_process(task, kp, kd, kff)

            step = 0
            dof_per_arm = len(kp)
            prev_avg_err = None  # 7-element array, averaged over N episodes

            while True:
                step += 1

                print(f"\n{_CYAN}{'=' * 70}{_RESET}")
                print(f"{_CYAN} step {step}  "
                      f"({datetime.now().strftime('%H:%M:%S')})  "
                      f"— running {n_episodes} episodes{_RESET}")
                print(f"{_DIM}  Kff: [{', '.join(f'{v:.3f}' for v in kff)}]{_RESET}")
                print(f"{_DIM}  Kp : [{', '.join(f'{v:.3f}' for v in kp)}]{_RESET}")
                print(f"{_DIM}  Kd : [{', '.join(f'{v:.4f}' for v in kd)}]{_RESET}")
                print(f"{_CYAN}{'=' * 70}{_RESET}\n")

                # Run N episodes at current gains, collect per-episode max errors
                episode_errors = []  # list of (14,) arrays
                for ep in range(n_episodes):
                    print(f"  {_DIM}episode {ep + 1}/{n_episodes}...{_RESET}", end="", flush=True)
                    metrics = run_episode_pid(task)
                    if metrics is not None:
                        episode_errors.append(metrics['max_error'])
                        max_err = metrics['max_error']
                        joint_err_ep = [max(max_err[j], max_err[j + dof_per_arm])
                                        for j in range(dof_per_arm)]
                        worst = max(joint_err_ep)
                        print(f"  max_err worst joint: {worst:.3f}")
                    else:
                        print(f"  {_YELLOW}no data{_RESET}")
                    if ep < n_episodes - 1:
                        time.sleep(PAUSE_BETWEEN_RUNS)

                if not episode_errors:
                    print(f"\n{_YELLOW}No tracking data collected in any episode.{_RESET}")
                    time.sleep(PAUSE_BETWEEN_RUNS)
                    continue

                # Average max errors across episodes → (14,)
                avg_max_err = np.mean(episode_errors, axis=0)

                # Compute per-joint worst of L/R from the averaged errors
                joint_err = [max(avg_max_err[j], avg_max_err[j + dof_per_arm])
                             for j in range(dof_per_arm)]

                print(f"\n  {_BOLD}Per-joint avg max error over {len(episode_errors)} episodes (worst of L/R):{_RESET}")
                for j in range(dof_per_arm):
                    err = joint_err[j]
                    if prev_avg_err is not None:
                        delta = prev_avg_err[j] - err
                        if delta > PID_IMPROVE_THRESHOLD:
                            indicator = f"{_GREEN}↓ improved by {delta:.3f}{_RESET}"
                        elif delta < -PID_IMPROVE_THRESHOLD:
                            indicator = f"{_RED}↑ worse by {-delta:.3f}{_RESET}"
                        else:
                            indicator = f"{_DIM}= within noise{_RESET}"
                        print(f"    J{j}: {err:.3f}  (prev {prev_avg_err[j]:.3f})  {indicator}")
                    else:
                        print(f"    J{j}: {err:.3f}  {_DIM}(baseline){_RESET}")

                # Build metrics dict for adjust_pid
                adj_metrics = {
                    'max_error': avg_max_err,
                    'prev_max_error': prev_avg_err,
                }
                prev_avg_err = joint_err

                # Adaptive adjustment
                new_kp, new_kd, new_kff = adjust_pid(kp, kd, kff, adj_metrics, dof_per_arm)

                # Show per-joint gain decisions
                print(f"\n  {_BOLD}Gain adjustments:{_RESET}")
                for j in range(dof_per_arm):
                    kff_d = new_kff[j] - kff[j]
                    kp_d = new_kp[j] - kp[j]
                    kd_d = new_kd[j] - kd[j]
                    if abs(kff_d) < 1e-6 and abs(kp_d) < 1e-6 and abs(kd_d) < 1e-6:
                        tag = f"{_DIM}hold{_RESET}"
                    elif kff_d > 0:
                        tag = f"{_GREEN}↑ increase{_RESET}"
                    else:
                        tag = f"{_RED}↓ decrease{_RESET}"
                    print(f"    J{j}: Kff {kff[j]:.3f}→{new_kff[j]:.3f}  "
                          f"Kp {kp[j]:.3f}→{new_kp[j]:.3f}  "
                          f"Kd {kd[j]:.4f}→{new_kd[j]:.4f}  [{tag}]")

                # Backup then write gains
                backup = BASE_CONFIG_PATH.with_suffix(f'.yaml.iter{step}')
                shutil.copy2(BASE_CONFIG_PATH, backup)
                write_pid_gains(BASE_CONFIG_PATH, new_kp, new_kd, new_kff)

                backup = CONFIG_PATH.with_suffix(f'.yaml.iter{step}')
                shutil.copy2(CONFIG_PATH, backup)
                write_pid_gains(CONFIG_PATH, new_kp, new_kd)

                print(f"{_GREEN}Saved to {BASE_CONFIG_PATH.name}{_RESET}")

                # Apply new gains in-process
                apply_pid_gains_in_process(task, new_kp, new_kd, new_kff)

                kp, kd, kff = new_kp, new_kd, new_kff

                print(f"\n{_DIM}Next step in {PAUSE_BETWEEN_RUNS}s...{_RESET}")
                time.sleep(PAUSE_BETWEEN_RUNS)

    except (KeyboardInterrupt, SystemExit):
        print(f"\n{_YELLOW}Interrupted — gains saved at current values.{_RESET}")
        print(f"  {_BOLD}Kff: [{', '.join(f'{v:.3f}' for v in kff)}]{_RESET}")
        print(f"  {_BOLD}Kp : [{', '.join(f'{v:.3f}' for v in kp)}]{_RESET}")
        print(f"  {_BOLD}Kd : [{', '.join(f'{v:.4f}' for v in kd)}]{_RESET}")
        print(f"  {BASE_CONFIG_PATH}")


if __name__ == "__main__":
    main()

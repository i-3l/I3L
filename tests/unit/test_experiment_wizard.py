#!/usr/bin/env python3
"""Lightweight tests for the experiment wizard helpers."""

import builtins
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/experiment_wizard.py"
_SPEC = importlib.util.spec_from_file_location("experiment_wizard", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.path.insert(0, str(_MODULE_PATH.parent))
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_render_card_handles_list_fields():
    card = _MODULE.render_card(
        {
            "title": "Smoke",
            "name": "smoke",
            "demo_counts": [5, 10],
            "methods": ["demos", "cotrain"],
            "rounds": 1,
        },
        "Goal:\nCheck formatting.",
    )
    assert 'demo_counts = [5, 10]' in card
    assert 'methods = ["demos", "cotrain"]' in card
    assert "Goal:" in card


def test_list_cards_excludes_readme():
    cards = _MODULE.list_cards()
    assert cards
    assert all(path.name != "README.md" for path in cards)


def test_list_runnable_cards_excludes_template():
    cards = _MODULE.list_runnable_cards()
    assert cards
    assert all(path.name != "TEMPLATE.md" for path in cards)


def test_prompt_choice_accepts_number():
    responses = iter(["2"])
    original_input = builtins.input
    try:
        builtins.input = lambda prompt="": next(responses)
        choice = _MODULE.prompt_choice("Robot", ["a1_iiil", "r1pro_iiil"], "a1_iiil")
    finally:
        builtins.input = original_input
    assert choice == "r1pro_iiil"


def test_run_pipeline_for_spec_forces_real_run_over_card_default():
    with tempfile.TemporaryDirectory() as tmp_dir:
        spec_path = Path(tmp_dir) / "dryrun_card.md"
        spec_path.write_text(
            'name = "demo"\n'
            'robot = "a1_iiil"\n'
            'base_task = "pnp/base"\n'
            'demo_counts = [5]\n'
            'methods = ["demos"]\n'
            'dry_run = true\n',
            encoding="utf-8",
        )
        recorded: dict[str, object] = {}
        original_call = _MODULE.subprocess.call
        try:
            _MODULE.subprocess.call = lambda command, cwd=None: recorded.update(command=command, cwd=cwd) or 0
            exit_code = _MODULE.run_pipeline_for_spec(spec_path, dry_run=False)
        finally:
            _MODULE.subprocess.call = original_call
        assert exit_code == 0
        assert "--no-dry-run" in recorded["command"]
        assert "--dry-run" not in recorded["command"]


def test_summarize_card_uses_recorded_run_metadata():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        spec_path = root / "exp.md"
        run_root = root / "runs" / "demo"
        run_root.mkdir(parents=True)
        spec_path.write_text(
            'title = "Spec title"\n'
            'name = "demo"\n'
            'robot = "a1_iiil"\n'
            'base_task = "pnp/base"\n'
            'demo_counts = [5]\n'
            'methods = ["demos"]\n'
            f'root_dir = "{run_root}"\n',
            encoding="utf-8",
        )
        (run_root / "experiment_config.json").write_text(
            json.dumps(
                {
                    "name": "demo",
                    "spec_title": "Recorded title",
                    "spec_tags": ["smoke"],
                    "robot": "a1_iiil",
                    "base_task": "pnp/base",
                    "demo_counts": [5, 10],
                    "rounds": 2,
                    "methods": ["demos", "cotrain"],
                    "iteration_method": "cotrain",
                    "phases": ["eval"],
                    "root_dir": str(run_root),
                    "launch_command": f"./experiment.sh --spec {spec_path} --dry-run",
                    "dry_run": True,
                }
            ),
            encoding="utf-8",
        )
        summary = _MODULE.summarize_card(spec_path)
        assert summary.title == "Recorded title"
        assert summary.demo_counts == ["5", "10"]
        assert summary.methods == ["demos", "cotrain"]
        assert summary.phases == ["eval"]
        assert summary.tags == ["smoke"]
        assert summary.has_run_artifacts is True
        assert summary.last_launch_mode == "dry-run"
        assert "--dry-run" in summary.launch_command


if __name__ == "__main__":
    tests = [
        test_render_card_handles_list_fields,
        test_list_cards_excludes_readme,
        test_list_runnable_cards_excludes_template,
        test_prompt_choice_accepts_number,
        test_run_pipeline_for_spec_forces_real_run_over_card_default,
        test_summarize_card_uses_recorded_run_metadata,
    ]
    for test in tests:
        test()
    print(f"Passed {len(tests)} experiment wizard tests")

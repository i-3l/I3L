#!/usr/bin/env python3
"""Lightweight tests for the experiment pipeline helpers."""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/experiment_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("experiment_pipeline", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_infer_collection_task():
    assert _MODULE.infer_collection_task("pnp/base") == "pnp/dc"
    assert _MODULE.infer_collection_task("real/base") == "real/dc"


def test_parse_methods_all():
    assert _MODULE.parse_methods("all") == list(_MODULE.AVAILABLE_METHODS)


def test_parse_methods_subset():
    assert _MODULE.parse_methods("demos,residual") == ["demos", "residual"]


def test_parse_phases_full():
    assert _MODULE.parse_phases("full") == set(_MODULE.FULL_PHASES)


def test_infer_defaults_for_a1():
    assert _MODULE.infer_collect_env("a1_iiil", None) == "sim_a1_joint"
    assert _MODULE.infer_base_arch("a1_iiil", None) == "dp_a1"
    assert _MODULE.infer_residual_arch("a1_iiil", None) == "residual_a1"


def test_resolve_iteration_method_defaults_to_cotrain():
    assert _MODULE.resolve_iteration_method(None, ["demos", "cotrain", "sirius"], 2) == "cotrain"


def test_resolve_iteration_method_can_fall_back_to_demos():
    assert _MODULE.resolve_iteration_method(None, ["demos"], 3) == "demos"


def test_resolve_iteration_method_rejects_residual_for_multi_round():
    try:
        _MODULE.resolve_iteration_method("residual", ["demos", "residual"], 2)
    except ValueError as exc:
        assert "residual" in str(exc)
    else:
        raise AssertionError("Expected residual iteration method to be rejected")


def test_round_scope_helpers():
    assert _MODULE.round_tag(0) == "round_00"
    assert _MODULE.round_tag(2) == "round_02"
    assert _MODULE.scope_tag("n020", 1) == "n020_r01"


def test_parse_experiment_card():
    values, notes = _MODULE.parse_experiment_card(
        'name = "demo"\nrounds = 2\nuse_wandb = true\ntags = ["a1", "iter"]\n---\nGoal:\nTest'
    )
    assert values["name"] == "demo"
    assert values["rounds"] == 2
    assert values["use_wandb"] is True
    assert values["tags"] == ["a1", "iter"]
    assert "Goal:" in notes


def test_load_experiment_spec_resolves_relative_paths():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        spec_dir = root / "cards"
        spec_dir.mkdir()
        spec_path = spec_dir / "exp.md"
        spec_path.write_text(
            'title = "Card title"\n'
            'base_demo_source = "../data/demos.hdf5"\n'
            'tags = "a1,iter"\n'
            "---\n"
            "Goal:\n"
            "Track notes\n",
            encoding="utf-8",
        )
        spec = _MODULE.load_experiment_spec(spec_path)
        assert spec.title == "Card title"
        assert spec.tags == ["a1", "iter"]
        assert spec.values["base_demo_source"] == str((root / "data" / "demos.hdf5").resolve())
        assert "Track notes" in spec.notes


def test_parse_config_uses_spec_defaults_and_cli_override():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        iiil_dir = root / "iiil"
        il_lib_dir = root / "il_lib"
        iiil_dir.mkdir()
        il_lib_dir.mkdir()
        spec_path = root / "iter_card.md"
        spec_path.write_text(
            'name = "spec_run"\n'
            'robot = "a1_iiil"\n'
            'base_task = "pnp/base"\n'
            'demo_counts = [20]\n'
            'rounds = 2\n'
            'methods = ["demos", "cotrain"]\n'
            'use_wandb = false\n'
            'eval_repeats = 3\n'
            'tags = ["iterative"]\n'
            "---\n"
            "Notes\n",
            encoding="utf-8",
        )
        cfg = _MODULE.parse_config(
            [
                "--spec",
                str(spec_path),
                "--iiil-dir",
                str(iiil_dir),
                "--il-lib-dir",
                str(il_lib_dir),
                "--eval-repeats",
                "1",
            ]
        )
        assert cfg.name == "spec_run"
        assert cfg.rounds == 2
        assert cfg.methods == ["demos", "cotrain"]
        assert cfg.use_wandb is False
        assert cfg.eval_repeats == 1
        assert cfg.spec_path == spec_path.resolve()
        assert cfg.spec_tags == ["iterative"]
        assert "Notes" in cfg.spec_notes


def test_parse_config_no_dry_run_overrides_spec_default():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        iiil_dir = root / "iiil"
        il_lib_dir = root / "il_lib"
        iiil_dir.mkdir()
        il_lib_dir.mkdir()
        spec_path = root / "dryrun_card.md"
        spec_path.write_text(
            'name = "spec_run"\n'
            'robot = "a1_iiil"\n'
            'base_task = "pnp/base"\n'
            'demo_counts = [5]\n'
            'methods = ["demos"]\n'
            'dry_run = true\n',
            encoding="utf-8",
        )
        cfg = _MODULE.parse_config(
            [
                "--spec",
                str(spec_path),
                "--iiil-dir",
                str(iiil_dir),
                "--il-lib-dir",
                str(il_lib_dir),
                "--no-dry-run",
            ]
        )
        assert cfg.dry_run is False


if __name__ == "__main__":
    tests = [
        test_infer_collection_task,
        test_parse_methods_all,
        test_parse_methods_subset,
        test_parse_phases_full,
        test_infer_defaults_for_a1,
        test_resolve_iteration_method_defaults_to_cotrain,
        test_resolve_iteration_method_can_fall_back_to_demos,
        test_resolve_iteration_method_rejects_residual_for_multi_round,
        test_round_scope_helpers,
        test_parse_experiment_card,
        test_load_experiment_spec_resolves_relative_paths,
        test_parse_config_uses_spec_defaults_and_cli_override,
        test_parse_config_no_dry_run_overrides_spec_default,
    ]
    for test in tests:
        test()
    print(f"Passed {len(tests)} experiment pipeline tests")

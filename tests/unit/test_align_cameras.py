from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.align_cameras as align_cameras
from scripts.align_cameras import (
    SESSIONS_DIRNAME,
    alignment_plan,
    blend_reference,
    expected_capture_geometry,
    find_session_dirs,
    load_reference_image,
    load_references,
    make_display,
    new_session_dir,
    opacity_value,
    open_captures,
    parse_args,
    parse_v4l2_values,
    parse_reference_overrides,
    profile_value,
    resolve_named_reference,
    run_list,
    run_snapshot,
    sanitize_session_base,
    save_references,
    session_dir_name,
    settings_signature,
    validate_alignment_frame,
    verify_capture_geometry,
)


EXTERNAL_CFG = {
    "device": "/dev/zed_external",
    "name": "external",
    "capture_resolution": [720, 2560],
    "capture_fps": 30,
    "resolution": [240, 240],
    "crop_size": 600,
    "crop_offset": [-40, -60],
    "v4l2": {"gain": 0, "brightness": 5},
}

WRIST_CFG = {
    "device": "/dev/zed_wrist",
    "name": "wrist",
    "capture_resolution": [720, 2560],
    "capture_fps": 30,
    "resolution": [240, 240],
    "crop_size": 720,
    "crop_offset": [0, 0],
    "v4l2": {"gain": 1},
}


def test_expected_capture_geometry_tracks_collection_crop():
    config = {
        "capture_resolution": [720, 2560],
        "crop_size": 600,
        "crop_offset": [-20, -60],
    }

    geometry = expected_capture_geometry(config)

    assert geometry == {
        "raw_resolution": [720, 2560],
        "stereo_split": True,
        "source_resolution": [720, 1280],
        "effective_crop_size": 600,
        "crop_origin": [320, 0],
    }


def test_alignment_plan_covers_external_collection_region_in_2k():
    plan = alignment_plan(EXTERNAL_CFG)

    assert plan["output_resolution"] == (1242, 1242)
    kwargs = plan["capture_kwargs"]
    assert kwargs["capture_resolution"] == (1242, 4416)
    assert kwargs["capture_fps"] == 15
    assert kwargs["crop_size"] == 1242
    # Collection crop origin [300, 0] -> native (600, 0) -> 2K region origin
    # (424, -99), size 1200; centering the 1242 square gives x 403, y clamped 0.
    assert kwargs["crop_offset"] == (-80, 0)
    assert kwargs["resolution"] == (1242, 1242)
    assert kwargs["v4l2_controls"] == {"gain": 0, "brightness": 5}
    assert plan["expected_geometry"] == {
        "raw_resolution": [1242, 4416],
        "stereo_split": True,
        "source_resolution": [1242, 2208],
        "effective_crop_size": 1242,
        "crop_origin": [403, 0],
    }
    assert plan["collection_geometry"]["crop_origin"] == [300, 0]


def test_alignment_plan_centers_wrist_square_on_its_wider_region():
    plan = alignment_plan(WRIST_CFG)

    assert plan["output_resolution"] == (1242, 1242)
    assert plan["capture_kwargs"]["crop_offset"] == (0, 0)
    assert plan["expected_geometry"]["crop_origin"] == [483, 0]


def test_alignment_plan_rejects_uncalibrated_collection_mode():
    config = dict(EXTERNAL_CFG)
    config["capture_resolution"] = [1080, 3840]

    with pytest.raises(RuntimeError, match="calibrated"):
        alignment_plan(config)


@pytest.mark.parametrize("config", [EXTERNAL_CFG, WRIST_CFG])
def test_alignment_plan_offsets_reproduce_origin_through_capture_formula(config):
    plan = alignment_plan(config)
    source_h, source_w = plan["expected_geometry"]["source_resolution"]
    crop = plan["expected_geometry"]["effective_crop_size"]
    offset_x, offset_y = plan["capture_kwargs"]["crop_offset"]

    # Replicate ZedCameraCapture._crop_geometry to prove the planned offsets
    # land the capture exactly on the intended crop origin.
    max_x, max_y = source_w - crop, source_h - crop
    start_x = int(np.clip(max_x // 2 + offset_x, 0, max_x))
    start_y = int(np.clip(max_y // 2 + offset_y, 0, max_y))

    assert [start_x, start_y] == plan["expected_geometry"]["crop_origin"]


def test_alignment_plan_requires_collection_capture_resolution():
    with pytest.raises(ValueError, match="capture_resolution"):
        alignment_plan({"device": "/dev/x", "name": "x", "crop_size": 600})


def test_signature_tracks_collection_and_alignment_settings():
    plan = alignment_plan(EXTERNAL_CFG)

    signature = settings_signature(EXTERNAL_CFG, plan)

    assert signature["collection_capture_resolution"] == [720, 2560]
    assert signature["crop_offset"] == [-40, -60]
    assert signature["alignment_capture_resolution"] == [1242, 4416]
    assert signature["alignment_capture_fps"] == 15
    assert signature["alignment_resolution"] == [1242, 1242]
    assert signature["alignment_geometry"]["crop_origin"] == [403, 0]
    assert signature["collection_geometry"]["crop_origin"] == [300, 0]
    assert "resolution" not in signature


def test_blend_reference_uses_uniform_opacity_for_rgb_image():
    live = np.full((2, 2, 3), 100, dtype=np.uint8)
    reference = np.full((2, 2, 3), 200, dtype=np.uint8)

    result = blend_reference(live, reference, 0.25)

    assert np.all(result == 125)


def test_blend_reference_combines_png_alpha_with_global_opacity():
    live = np.full((1, 2, 3), 100, dtype=np.uint8)
    reference = np.full((1, 2, 4), 200, dtype=np.uint8)
    reference[..., 3] = [0, 255]

    result = blend_reference(live, reference, 0.5)

    assert np.all(result[0, 0] == 100)
    assert np.all(result[0, 1] == 150)


def test_load_reference_requires_exact_native_geometry(tmp_path: Path):
    path = tmp_path / "wrong-size.png"
    assert cv2.imwrite(str(path), np.zeros((240, 240, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="expected 600x600"):
        load_reference_image(path, (600, 600))


def test_captured_reference_loads_and_is_rejected_after_crop_change(tmp_path: Path):
    config = {"external": dict(EXTERNAL_CFG)}
    plans = {"external": alignment_plan(config["external"])}
    frame = np.full((1242, 1242, 3), 123, dtype=np.uint8)

    saved = save_references(
        ["external"],
        {"external": frame},
        config,
        plans,
        tmp_path,
        replace=False,
    )
    references, statuses = load_references(
        ["external"], config, plans, tmp_path, overrides={}
    )

    assert saved == ["external"]
    assert np.array_equal(references["external"], frame)
    assert statuses["external"] == "reference: external.png"

    config["external"]["crop_offset"] = [0, 0]
    plans = {"external": alignment_plan(config["external"])}
    stale_references, stale_statuses = load_references(
        ["external"], config, plans, tmp_path, overrides={}
    )
    assert stale_references == {}
    assert stale_statuses["external"].startswith("stale settings")


def test_persistent_reference_without_manifest_is_rejected(tmp_path: Path):
    path = tmp_path / "external.png"
    assert cv2.imwrite(str(path), np.zeros((1242, 1242, 3), dtype=np.uint8))
    config = {"external": dict(EXTERNAL_CFG)}
    plans = {"external": alignment_plan(config["external"])}

    references, statuses = load_references(
        ["external"], config, plans, tmp_path, overrides={}
    )

    assert references == {}
    assert statuses["external"].startswith("unverified reference")


def test_open_captures_fails_if_any_requested_camera_is_missing(monkeypatch):
    opened = []

    class FakeCapture:
        def __init__(self, **kwargs):
            if kwargs["device_path"].endswith("external"):
                raise RuntimeError("missing")
            self.closed = False
            opened.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(align_cameras, "ZedCameraCapture", FakeCapture)
    monkeypatch.setattr(align_cameras, "verify_capture_geometry", lambda *_: None)
    monkeypatch.setattr(align_cameras, "verify_v4l2_controls", lambda *_: None)
    configs = {
        "wrist": {"device": "/dev/zed_wrist", "name": "wrist"},
        "external": {"device": "/dev/zed_external", "name": "external"},
    }
    plans = {
        key: {
            "output_resolution": (1242, 1242),
            "capture_kwargs": {
                "device_path": cfg["device"],
                "name": key,
                "capture_fps": 15,
            },
            "expected_geometry": {"crop_origin": [0, 0]},
        }
        for key, cfg in configs.items()
    }

    with pytest.raises(RuntimeError, match="requested camera.*external"):
        open_captures(["wrist", "external"], configs, plans)

    assert opened[0].closed


def test_verify_capture_geometry_rejects_fallback_raw_mode():
    plan = alignment_plan(EXTERNAL_CFG)

    class FallbackCapture:
        raw_resolution = (376, 1344)
        stereo_split = True
        source_resolution = (376, 672)
        effective_crop_size = 376
        crop_origin = (148, 0)
        actual_capture_fps = 30.0

    with pytest.raises(RuntimeError, match="negotiated camera geometry"):
        verify_capture_geometry(FallbackCapture(), plan)


def test_validate_alignment_frame_rejects_wrong_resolution():
    validate_alignment_frame(
        "wrist", np.zeros((720, 720, 3), dtype=np.uint8), (720, 720)
    )

    with pytest.raises(RuntimeError, match="expected a 720x720 color frame"):
        validate_alignment_frame(
            "wrist", np.zeros((600, 600, 3), dtype=np.uint8), (720, 720)
        )


@pytest.mark.parametrize("value", ["-0.1", "1.1", "words"])
def test_opacity_validator_rejects_invalid_values(value: str):
    with pytest.raises(Exception):
        opacity_value(value)


@pytest.mark.parametrize("value", ["../rack", "rack/mug", ".", ""])
def test_profile_validator_rejects_unsafe_names(value: str):
    with pytest.raises(Exception):
        profile_value(value)


def test_reference_overrides_require_camera_mapping(tmp_path: Path):
    image = tmp_path / "reference.png"
    image.touch()

    parsed = parse_reference_overrides([f"external={image}"])

    assert parsed == {"external": image.resolve()}
    with pytest.raises(ValueError, match="expected wrist=IMAGE or external=IMAGE"):
        parse_reference_overrides([str(image)])


def test_parse_v4l2_values_handles_control_output():
    output = "brightness: 6\ngain: 0\nwhite_balance_temperature: 3800\n"

    assert parse_v4l2_values(output) == {
        "brightness": 6,
        "gain": 0,
        "white_balance_temperature": 3800,
    }


def test_single_600px_camera_display_matches_original_layout():
    frame = np.zeros((600, 600, 3), dtype=np.uint8)

    display = make_display(
        ["external"],
        {"external": frame},
        references={},
        statuses={"external": "no reference"},
        opacity=0.45,
        overlay_enabled=True,
        output_resolutions={"external": (600, 600)},
    )

    # header 66 + frame 600 + footer 58
    assert display.shape == (724, 600, 3)


def test_display_pads_mixed_panel_sizes_to_common_height():
    frames = {
        "wrist": np.zeros((720, 720, 3), dtype=np.uint8),
        "external": np.zeros((600, 600, 3), dtype=np.uint8),
    }

    display = make_display(
        ["wrist", "external"],
        frames,
        references={},
        statuses={},
        opacity=0.45,
        overlay_enabled=True,
        output_resolutions={"wrist": (720, 720), "external": (600, 600)},
    )

    assert display.shape[1] == 720 + 600
    # wrist panel: header 79 (66 * 1.2) + 720; footer 70 (58 * 1.2)
    assert display.shape[0] == 79 + 720 + 70


def test_sanitize_session_base_yields_profile_safe_names():
    assert sanitize_session_base("franka demos (v2)") == "franka_demos_v2_"
    assert sanitize_session_base("--..weird") == "weird"
    assert sanitize_session_base("") == "session"
    assert profile_value(sanitize_session_base("franka demos (v2)"))


def test_session_dir_name_appends_local_timestamp():
    stamp = datetime(2026, 8, 20, 14, 32, 1)

    assert session_dir_name("demos.extra", stamp) == "demos.extra_20260820_143201"


def test_new_session_dir_never_reuses_an_existing_directory(
    tmp_path: Path, monkeypatch
):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 20, 14, 32, 1)

    monkeypatch.setattr(align_cameras, "datetime", FixedDatetime)

    first = new_session_dir(tmp_path, "demo run")
    assert first == tmp_path / SESSIONS_DIRNAME / "demo_run_20260820_143201"

    first.mkdir(parents=True)
    second = new_session_dir(tmp_path, "demo run")
    assert second == tmp_path / SESSIONS_DIRNAME / "demo_run_20260820_143201-2"


def test_find_session_dirs_orders_newest_first(tmp_path: Path):
    sessions = tmp_path / SESSIONS_DIRNAME
    older = sessions / "older"
    newer = sessions / "newer"
    stamps = {older: "2026-08-19T10:00:00+00:00", newer: "2026-08-20T10:00:00+00:00"}
    for path, updated_at in stamps.items():
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(
            json.dumps({"version": 1, "cameras": {}, "updated_at": updated_at}),
            encoding="utf-8",
        )

    assert find_session_dirs(tmp_path) == [newer, older]
    assert find_session_dirs(tmp_path / "does-not-exist") == []


def test_parse_args_modes_are_mutually_exclusive():
    assert parse_args(["--snapshot", "run1"]).snapshot == "run1"
    assert parse_args(["--last"]).last is True
    assert parse_args(["--list"]).list_profiles is True
    assert parse_args([]).snapshot is None
    assert parse_args([]).name is None
    assert parse_args(["mugtask"]).name == "mugtask"

    with pytest.raises(SystemExit):
        parse_args(["--snapshot", "run1", "--check"])
    with pytest.raises(SystemExit):
        parse_args(["--last", "--list"])
    with pytest.raises(SystemExit):
        parse_args(["mugtask", "--last"])
    with pytest.raises(SystemExit):
        parse_args(["mugtask", "--snapshot", "run1"])


def test_resolve_named_reference_prefers_profile_then_session_then_task(
    tmp_path: Path,
):
    (tmp_path / "rack_mug").mkdir()
    sessions = tmp_path / SESSIONS_DIRNAME
    older = sessions / "hi_20260819_090000"
    newer = sessions / "hi_20260820_010600"
    other = sessions / "hi2_20260820_020000"
    stamps = {
        older: "2026-08-19T09:00:00+00:00",
        newer: "2026-08-20T01:06:00+00:00",
        other: "2026-08-20T02:00:00+00:00",
    }
    for path, updated_at in stamps.items():
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(
            json.dumps({"version": 1, "cameras": {}, "updated_at": updated_at}),
            encoding="utf-8",
        )

    assert resolve_named_reference(tmp_path, "rack_mug") == tmp_path / "rack_mug"
    assert resolve_named_reference(tmp_path, "hi_20260819_090000") == older
    # Task name resolves to that task's newest snapshot; 'hi2' is newer overall
    # but its name does not start with 'hi_'.
    assert resolve_named_reference(tmp_path, "hi") == newer
    assert resolve_named_reference(tmp_path, "nosuch") is None
    assert resolve_named_reference(tmp_path, "../rack_mug") is None


def test_run_snapshot_saves_references_that_load_verified(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(align_cameras, "SNAPSHOT_WARMUP_SECONDS", 0.0)

    class FakeCapture:
        def get_frame(self):
            return np.full((1242, 1242, 3), 90, dtype=np.uint8)

    config = {"external": dict(EXTERNAL_CFG)}
    plans = {"external": alignment_plan(config["external"])}
    profile_dir = tmp_path / SESSIONS_DIRNAME / "run_20260820_143201"

    result = run_snapshot(
        ["external"], {"external": FakeCapture()}, config, plans, profile_dir
    )

    assert result == 0
    references, statuses = load_references(
        ["external"], config, plans, profile_dir, overrides={}
    )
    assert statuses["external"] == "reference: external.png"
    assert np.all(references["external"] == 90)


def test_run_snapshot_fails_without_a_frame(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(align_cameras, "SNAPSHOT_WARMUP_SECONDS", 0.0)

    class DeadCapture:
        def get_frame(self):
            return None

    result = run_snapshot(
        ["external"], {"external": DeadCapture()}, {}, {}, tmp_path / "unused"
    )

    assert result == 1
    assert not (tmp_path / "unused").exists()


def test_run_list_reports_profile_and_session_health(tmp_path: Path, capsys):
    config = {"external": dict(EXTERNAL_CFG)}
    plans = {"external": alignment_plan(config["external"])}
    (tmp_path / "rack_mug").mkdir()
    session = tmp_path / SESSIONS_DIRNAME / "runA_20260820_143201"
    frame = np.zeros((1242, 1242, 3), dtype=np.uint8)
    save_references(
        ["external"], {"external": frame}, config, plans, session, replace=True
    )

    assert run_list(tmp_path, config, plans) == 0

    output = capsys.readouterr().out
    assert "rack_mug" in output
    assert "external: missing" in output
    assert "runA_20260820_143201" in output
    assert "external: ok" in output

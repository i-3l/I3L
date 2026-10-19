"""Live preview for the processed camera observations used during collection."""

from __future__ import annotations

import math
import os
from typing import Mapping

import cv2
import numpy as np
import torch


class CameraStreamViewer:
    """Render camera observations and collection counts in one OpenCV window.

    The viewer consumes the observation dictionary after the environment camera
    capture has cropped / resized it.  It therefore shows the same image arrays
    that are handed to the data-collection wrapper, without opening each camera
    device a second time.
    """

    _TILE_WIDTH = 426
    _TILE_HEIGHT = 240
    _HEADER_HEIGHT = 64

    def __init__(
        self,
        enabled: bool = False,
        show_takeover_count: bool = False,
        window_name: str = "Franka collection cameras (processed / saved)",
    ) -> None:
        self.enabled = bool(enabled)
        self.show_takeover_count = bool(show_takeover_count)
        self.window_name = window_name
        self._window_created = False
        self._warned = False

        # OpenCV's Qt backend can abort the entire process when no display is
        # available, so do not attempt to create a window in a headless shell.
        if self.enabled and os.name != "nt":
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                self._disable("no DISPLAY or WAYLAND_DISPLAY is available")

    def update(
        self,
        obs: Mapping[str, object],
        *,
        total_saved: int = 0,
        takeover_saved: int = 0,
        recording: bool = False,
    ) -> None:
        """Draw the newest processed frames. Press ``q`` to close the preview."""
        if not self.enabled:
            return

        camera_items = [
            (key, value)
            for key, value in obs.items()
            if self._is_camera_key(key)
        ]
        camera_items.sort(key=lambda item: self._sort_key(item[0]))
        if not camera_items:
            return

        try:
            panels = [self._make_panel(key, value) for key, value in camera_items]
            panels = [panel for panel in panels if panel is not None]
            if not panels:
                return

            columns = min(3, len(panels))
            rows = math.ceil(len(panels) / columns)
            blank = np.full(
                (self._TILE_HEIGHT, self._TILE_WIDTH, 3), 24, dtype=np.uint8
            )
            while len(panels) < rows * columns:
                panels.append(blank.copy())

            grid_rows = []
            for row in range(rows):
                start = row * columns
                grid_rows.append(np.hstack(panels[start : start + columns]))
            grid = np.vstack(grid_rows)

            header = np.full(
                (self._HEADER_HEIGHT, grid.shape[1], 3), (28, 28, 28), dtype=np.uint8
            )
            saved_text = f"Saved demos: {int(total_saved)}"
            if self.show_takeover_count:
                saved_text = (
                    f"Saved demos: {int(takeover_saved)} with takeover"
                    f"  |  {int(total_saved)} total"
                )
            status_text = "RECORDING" if recording else "READY"
            status_color = (80, 220, 80) if recording else (185, 185, 185)
            cv2.putText(
                header,
                saved_text,
                (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                header,
                f"{status_text}   |   q: close preview",
                (18, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                status_color,
                1,
                cv2.LINE_AA,
            )
            display = np.vstack((header, grid))

            if not self._window_created:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, display.shape[1], display.shape[0])
                self._window_created = True
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(
                self.window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                self.close()
        except cv2.error as exc:
            self._disable(f"OpenCV could not create or update the window: {exc}")

    def close(self) -> None:
        """Close only the preview; data collection continues."""
        if self._window_created:
            try:
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass
        self._window_created = False
        self.enabled = False

    def _disable(self, reason: str) -> None:
        self.enabled = False
        if not self._warned:
            print(f"Camera preview disabled: {reason}", flush=True)
            self._warned = True

    @staticmethod
    def _is_camera_key(key: str) -> bool:
        lower = key.lower()
        return "camera" in lower and (
            lower.endswith("::rgb")
            or lower.endswith("::rgb_right")
            or lower.endswith("::depth")
        )

    @classmethod
    def _sort_key(cls, key: str) -> tuple[str, int]:
        lower = key.lower()
        camera = "wrist" if "wrist" in lower else "external" if "external" in lower else lower
        if lower.endswith("::rgb"):
            modality = 0
        elif lower.endswith("::rgb_right"):
            modality = 1
        else:
            modality = 2
        return camera, modality

    @classmethod
    def _make_panel(cls, key: str, value: object) -> np.ndarray | None:
        frame = cls._to_numpy(value)
        if frame is None:
            return None

        lower = key.lower()
        if lower.endswith("::depth"):
            bgr = cls._colorize_depth(frame)
        else:
            bgr = cls._rgb_to_bgr(frame)
        if bgr is None:
            return None

        source_h, source_w = bgr.shape[:2]
        scale = min(cls._TILE_WIDTH / source_w, cls._TILE_HEIGHT / source_h)
        width = max(1, int(round(source_w * scale)))
        height = max(1, int(round(source_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(bgr, (width, height), interpolation=interpolation)
        panel = np.full((cls._TILE_HEIGHT, cls._TILE_WIDTH, 3), 16, dtype=np.uint8)
        x = (cls._TILE_WIDTH - width) // 2
        y = (cls._TILE_HEIGHT - height) // 2
        panel[y : y + height, x : x + width] = resized

        label = f"{cls._label(key)}  {source_w}x{source_h}"
        cv2.rectangle(panel, (0, 0), (cls._TILE_WIDTH, 29), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (9, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        return panel

    @staticmethod
    def _to_numpy(value: object) -> np.ndarray | None:
        if torch.is_tensor(value):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim not in (2, 3) or array.size == 0:
            return None
        return array

    @staticmethod
    def _rgb_to_bgr(frame: np.ndarray) -> np.ndarray | None:
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            return None
        rgb = frame[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
            if rgb.size and float(np.max(rgb)) <= 1.0:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _colorize_depth(frame: np.ndarray) -> np.ndarray:
        depth = np.squeeze(frame).astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            low, high = np.percentile(depth[valid], (2, 98))
            if high <= low:
                high = low + 1.0
            scaled = (np.clip(depth, low, high) - low) / (high - low)
            normalized[valid] = (scaled[valid] * 255.0).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

    @staticmethod
    def _label(key: str) -> str:
        lower = key.lower()
        camera = "Wrist" if "wrist" in lower else "External" if "external" in lower else "Camera"
        if lower.endswith("::rgb_right"):
            modality = "right RGB"
        elif lower.endswith("::depth"):
            modality = "depth"
        else:
            modality = "left RGB"
        return f"{camera} - {modality}"


__all__ = ["CameraStreamViewer"]

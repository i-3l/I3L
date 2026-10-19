"""OpenCV-based camera capture with background thread for ZED cameras.

ZED cameras without the ZED SDK present frames as side-by-side stereo via V4L2.
This module captures the left image, crops to square, and resizes to the target
resolution in a background thread so get_frame() is near-instant.
"""

import cv2
import logging
import numpy as np
import threading
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ZedCameraCapture:
    """Background-threaded OpenCV capture for a ZED camera.

    Opens the device, continuously grabs frames in a daemon thread,
    and provides the latest processed frame via get_frame().

    Args:
        device_path: V4L2 device path (e.g. "/dev/zed_wrist" or "/dev/video2").
        name: Logical camera name for logging (e.g. "robot:zed_wrist:Camera:0").
        resolution: Target output resolution as (H, W). Frames are center-cropped
            to square then resized to this size.
        capture_resolution: Optional raw stereo capture resolution as (H, W).
        capture_fps: Optional raw camera frame rate.
        crop_size: Optional square crop size in source pixels. By default the
            largest square that fits in one eye is used.
        crop_offset: Crop displacement as (x, y) pixels from center before
            resize. Positive x moves the crop right; positive y moves it down.
            Values are clamped so the crop always remains inside the image.
        v4l2_controls: Optional dict of V4L2 control names to values
            (e.g. {"brightness": 4, "gain": 0}). Applied on startup for
            consistent settings across runs.
        open_timeout: Seconds to keep retrying open/first-frame before giving
            up. Cameras need a few seconds after USB re-enumeration.
    """

    def __init__(
        self,
        device_path: str,
        name: str,
        resolution: Tuple[int, int] = (240, 240),
        capture_resolution: Optional[Tuple[int, int]] = None,
        capture_fps: Optional[int] = None,
        crop_size: Optional[int] = None,
        crop_offset: Tuple[int, int] = (0, 0),
        v4l2_controls: Optional[Dict[str, int]] = None,
        open_timeout: float = 10.0,
    ):
        self.name = name
        self.device_path = device_path
        self.resolution = resolution
        self.capture_resolution = capture_resolution
        self.capture_fps = capture_fps
        if crop_size is not None and int(crop_size) <= 0:
            raise ValueError("crop_size must be positive")
        self.crop_size = int(crop_size) if crop_size is not None else None
        if len(crop_offset) != 2:
            raise ValueError("crop_offset must contain exactly [x, y]")
        self.crop_offset = (int(crop_offset[0]), int(crop_offset[1]))

        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False

        if capture_resolution is not None and len(capture_resolution) != 2:
            raise ValueError("capture_resolution must contain exactly [height, width]")

        # Open video device with V4L2 backend. A camera that just re-enumerated
        # on the USB bus (controller reset, replug) can refuse to open or
        # deliver frames for several seconds, so retry until open_timeout
        # instead of failing the whole session on the first attempt.
        deadline = time.monotonic() + float(open_timeout)
        while True:
            error = None
            test_frame = None
            self._cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            if not self._cap.isOpened():
                error = (
                    f"Failed to open camera '{name}' at {device_path}. "
                    f"Check that the device exists and you have permission (video group)."
                )
            else:
                # OpenCV otherwise negotiates the ZED's VGA stereo mode (1344x376).
                if capture_resolution is not None:
                    capture_h, capture_w = map(int, capture_resolution)
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_w)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_h)
                if capture_fps is not None:
                    self._cap.set(cv2.CAP_PROP_FPS, int(capture_fps))

                # Apply V4L2 controls for consistent settings across runs
                if v4l2_controls:
                    self._apply_v4l2_controls(device_path, v4l2_controls)

                # Read one frame to determine the raw resolution
                ret, test_frame = self._cap.read()
                if not ret or test_frame is None:
                    error = (
                        f"Camera '{name}' opened but failed to read initial "
                        f"frame from {device_path}."
                    )

            if error is None:
                break
            self._cap.release()
            if time.monotonic() >= deadline:
                raise RuntimeError(error)
            logger.warning(f"{error} Retrying...")
            time.sleep(1.0)

        raw_h, raw_w = test_frame.shape[:2]
        self.raw_resolution = (raw_h, raw_w)
        self.actual_capture_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if capture_resolution is not None:
            requested_h, requested_w = map(int, capture_resolution)
            if (raw_h, raw_w) != (requested_h, requested_w):
                logger.warning(
                    f"Camera '{name}' requested {requested_w}x{requested_h}, "
                    f"but V4L2 negotiated {raw_w}x{raw_h}"
                )
        # ZED side-by-side: width is 2x the single-image width
        self._stereo_split = raw_w > raw_h * 1.5
        self.stereo_split = self._stereo_split
        if self._stereo_split:
            single_w = raw_w // 2
            logger.info(
                f"Camera '{name}': detected side-by-side stereo "
                f"({raw_w}x{raw_h}), using left image ({single_w}x{raw_h})"
            )
        else:
            single_w = raw_w
            logger.info(f"Camera '{name}': single image ({raw_w}x{raw_h})")

        self.source_resolution = (raw_h, single_w)
        crop_size, start_x, start_y = self._crop_geometry(raw_h, single_w)
        self.effective_crop_size = crop_size
        self.crop_origin = (start_x, start_y)

        # Process and store the initial frame
        self._latest_frame = self._process_frame(test_frame)

        # Start background capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Camera '{name}' capture started on {device_path} "
            f"(output {resolution[0]}x{resolution[1]}, "
            f"crop_size={self.crop_size or 'max'}, crop_offset={self.crop_offset})"
        )

    @staticmethod
    def _apply_v4l2_controls(device_path: str, controls: Dict[str, int]) -> None:
        """Apply V4L2 controls via v4l2-ctl for consistent camera settings."""
        import subprocess
        for ctrl, value in controls.items():
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", device_path, "--set-ctrl", f"{ctrl}={value}"],
                    capture_output=True, timeout=2,
                )
                logger.info(f"Camera {device_path}: set {ctrl}={value}")
            except Exception as e:
                logger.warning(f"Camera {device_path}: failed to set {ctrl}={value}: {e}")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract left image from stereo, center-crop to square, convert BGR->RGB, resize."""
        if self._stereo_split:
            h, w = frame.shape[:2]
            frame = frame[:, : w // 2]

        # Crop to square at the configured displacement from center.
        h, w = frame.shape[:2]
        crop_size, start_x, start_y = self._crop_geometry(h, w)
        frame = frame[
            start_y : start_y + crop_size,
            start_x : start_x + crop_size,
        ]

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to target resolution
        target_h, target_w = self.resolution
        frame = cv2.resize(
            frame, (target_w, target_h), interpolation=cv2.INTER_AREA
        )

        return frame

    def _crop_geometry(self, height: int, width: int) -> Tuple[int, int, int]:
        """Return the effective square size and top-left crop coordinate."""
        max_crop_size = min(height, width)
        crop_size = min(self.crop_size or max_crop_size, max_crop_size)
        max_x = width - crop_size
        max_y = height - crop_size
        start_x = int(np.clip(max_x // 2 + self.crop_offset[0], 0, max_x))
        start_y = int(np.clip(max_y // 2 + self.crop_offset[1], 0, max_y))
        return crop_size, start_x, start_y

    def _capture_loop(self) -> None:
        """Continuously grab and process frames in background."""
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                processed = self._process_frame(frame)
                with self._lock:
                    self._latest_frame = processed
            else:
                # Brief pause on read failure to avoid busy-looping
                import time

                time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest captured frame (H, W, 3) uint8 RGB, or None."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def close(self) -> None:
        """Stop capture thread and release the video device."""
        self._running = False
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()
        logger.info(f"Camera '{self.name}' closed.")

import atexit
import logging
import select
import sys
import termios
import tty
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

from iiil.detectors.interrupt_detector import InterruptDetector
from iiil.detectors.release_detector import ReleaseDetector


logger = logging.getLogger(__name__)


class KeyboardInterruptDetector(InterruptDetector, ReleaseDetector):
    """
    Manual detector that switches shared autonomy from keyboard input.

    Press interrupt_key, '[' by default, to request oracle control. Press
    release_key, ']' by default, to request base-policy control. Press
    discard_key, '\\' by default, to mark the current trajectory as a failure.
    """

    def __init__(self, config: DictConfig):
        self._interrupt_key = config.get("interrupt_key", "[")
        self._release_key = config.get("release_key", "]")
        self._discard_key = config.get("discard_key", "\\")
        self._enabled = sys.stdin.isatty()
        self._fd = None
        self._old_settings = None
        self._pending_key = None

        if not self._enabled:
            logger.warning("KeyboardInterruptDetector disabled because stdin is not a TTY.")
            return

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        atexit.register(self.close)

    def is_interrupted(self, detector_obs: dict[str, Any]) -> bool:
        return self._consume_key(self._interrupt_key)

    def is_released(self, detector_obs: dict[str, Any]) -> bool:
        return self._consume_key(self._release_key)

    def consume_discard_requested(self) -> bool:
        """Return True once when the discard key has been pressed."""
        return self._consume_key(self._discard_key)

    def _consume_key(self, expected_key: str) -> bool:
        self._poll()
        if self._pending_key == expected_key:
            self._pending_key = None
            return True
        return False

    def _poll(self) -> None:
        if not self._enabled:
            return

        while select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key in (self._interrupt_key, self._release_key, self._discard_key):
                self._pending_key = key

    def reset(self):
        self._pending_key = None

    def close(self) -> None:
        if self._enabled and self._old_settings is not None and self._fd is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def get_current_state(self) -> Optional[Dict[str, torch.Tensor]]:
        return None

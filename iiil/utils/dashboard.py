"""
Compact CLI dashboard for IIIL runtime status.

Uses a terminal **scroll region** to reserve the bottom 4 rows for the
dashboard.  Normal output (log events, Isaac Sim messages) scrolls in the
region above it.  The dashboard is redrawn at absolute row positions via
``/dev/tty`` so it is unaffected by any ``sys.stdout`` wrapper that
Isaac Sim may install.

Thread-safe: the takeover detector calls log_event from a 100 Hz thread.
Auto-disables when /dev/tty cannot be opened or the terminal is too small.
"""

import os
import sys
import time
import threading
from dataclasses import dataclass


@dataclass
class DashboardState:
    """Snapshot of all display values for one render."""
    trial_num: int = 0
    trial_total: int = 0
    trial_successes: int = 0
    task_name: str = ""
    demo_num: int = 0
    recording: bool = False
    gate_state: str = "POLICY"
    peak_joint_name: str = ""
    peak_value: float = 0.0
    peak_threshold: float = 1.0
    takeover_detected: bool = False
    # Release timer state
    release_timer_active: bool = False
    release_elapsed: float = 0.0
    release_timeout: float = 3.0
    release_vel_max: float = 0.0
    release_vel_threshold: float = 0.05
    release_curr_std: float = 0.0
    release_curr_threshold: float = 10.0
    # Tracking error state
    control_mode: str = ""              # "gello_to_env", "env_to_gello", "admittance"
    tracking_error: list = None         # per-joint position error (list of floats)
    position_threshold: list = None     # per-joint threshold (list of floats)
    robot_type: str = ""                # "R1Pro", "A1", etc.
    robot_dof: int = 0                  # 6 or 14


# ANSI colour helpers
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_BRIGHT_GREEN = "\033[92m"
_YELLOW = "\033[33m"
_BOLD_YELLOW = "\033[1;33m"
_RED = "\033[31m"

_DASHBOARD_LINES = 5


class Dashboard:
    """Fixed-bottom dashboard using terminal scroll regions.

    On construction the terminal's scroll region is narrowed so that the
    bottom ``_DASHBOARD_LINES`` rows never scroll.  ``update()`` draws the
    dashboard into those rows via ``/dev/tty``.  ``log_event()`` prints
    into the scrolling region via ``sys.stdout``.

    ``clear()`` **must** be called on shutdown to restore the full scroll
    region, otherwise the terminal stays in the reduced region.
    """

    _MIN_RENDER_INTERVAL = 0.05  # ~20 Hz

    def __init__(self, enabled: bool = True, extra_rows: int = 0):
        self._enabled = enabled
        self._dashboard_lines = _DASHBOARD_LINES + extra_rows
        self._tty = None
        self._lock = threading.Lock()
        self._last_text = ""
        self._last_render_time = 0.0
        self._scroll_bottom = 0
        self._height = 0
        self._width = 58

        if not self._enabled:
            return

        # /dev/tty goes straight to the terminal, bypassing any stdout
        # wrapper that Isaac Sim may have installed.
        try:
            self._tty = open("/dev/tty", "w", encoding="utf-8")
        except OSError:
            self._enabled = False
            return

        try:
            size = os.get_terminal_size()
            self._height = size.lines
            self._width = min(size.columns, 80)
        except OSError:
            self._enabled = False
            self._tty.close()
            self._tty = None
            return

        if self._height < self._dashboard_lines + 4:
            self._enabled = False
            self._tty.close()
            self._tty = None
            return

        self._scroll_bottom = self._height - self._dashboard_lines

        # Shrink the scroll region so normal output cannot overwrite the
        # dashboard rows, then place the cursor at the bottom of that
        # region so new output starts there.
        self._tty.write(
            f"\033[1;{self._scroll_bottom}r"
            f"\033[{self._scroll_bottom};1H"
        )
        self._tty.flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, state: DashboardState) -> None:
        """Draw the dashboard into the reserved bottom rows."""
        if not self._enabled:
            return

        now = time.monotonic()
        if now - self._last_render_time < self._MIN_RENDER_INTERVAL:
            return

        with self._lock:
            text = self._build_text(state)
            if text == self._last_text:
                return

            row_start = self._scroll_bottom + 1
            lines = text.split("\n")

            # Build one atomic buffer: save cursor → draw each row at its
            # absolute position → restore cursor.  Because the entire
            # payload is <4 KiB the kernel delivers it in a single write.
            buf = "\0337"  # DECSC — save cursor + attrs
            for i, line in enumerate(lines):
                buf += f"\033[{row_start + i};1H\033[2K{line}"
            buf += "\0338"  # DECRC — restore cursor + attrs

            # Flush stdout first so any pending output reaches the
            # terminal before we jump to the dashboard rows.
            sys.stdout.flush()
            self._tty.write(buf)
            self._tty.flush()

            self._last_text = text
            self._last_render_time = now

    def log_event(self, msg: str) -> None:
        """Print a persistent message in the scrolling region."""
        with self._lock:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()

    def clear(self) -> None:
        """Restore the full scroll region and erase the dashboard rows."""
        if not self._enabled:
            return

        with self._lock:
            buf = "\033[r"  # reset scroll region to full terminal
            for i in range(self._dashboard_lines):
                buf += f"\033[{self._scroll_bottom + 1 + i};1H\033[2K"
            buf += f"\033[{self._height};1H"
            self._tty.write(buf)
            self._tty.flush()
            self._enabled = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _format_joint_error(self, idx: int, val: float, thresh: float) -> str:
        """Format a single joint's tracking error with color coding."""
        ratio = val / thresh if thresh > 0 else 0
        pct = int(ratio * 100)
        if ratio > 1.0:
            color = _RED
        elif ratio > 0.8:
            color = _YELLOW
        else:
            color = ""
        end = _RESET if color else ""
        return f"{color}J{idx}={val:.2f}/{thresh:.2f}({pct}%){end}"

    def _build_tracking_error_lines(self, state: DashboardState) -> list:
        """Build 0, 1, or 2 tracking error lines depending on robot type."""
        extra = self._dashboard_lines - _DASHBOARD_LINES
        if extra <= 0:
            return []

        show_data = (
            state.tracking_error is not None
            and state.position_threshold is not None
            and state.control_mode not in ("env_to_gello", "gello_to_env", "")
        )

        is_r1pro = "R1Pro" in state.robot_type or "r1pro" in state.robot_type

        if is_r1pro and extra >= 2:
            if show_data and len(state.tracking_error) >= 14:
                parts_l = [self._format_joint_error(i, state.tracking_error[i], state.position_threshold[i]) for i in range(7)]
                parts_r = [self._format_joint_error(i, state.tracking_error[i], state.position_threshold[i]) for i in range(7, 14)]
                return [
                    " Err L: " + " ".join(parts_l),
                    " Err R: " + " ".join(parts_r),
                ]
            else:
                return [
                    f" {_DIM}Err L: ---{_RESET}",
                    f" {_DIM}Err R: ---{_RESET}",
                ]
        else:
            # A1 or other single-arm: 1 row
            if show_data and state.tracking_error:
                dof = len(state.tracking_error)
                parts = [self._format_joint_error(i, state.tracking_error[i], state.position_threshold[i]) for i in range(dof)]
                return [" Err: " + " ".join(parts)]
            else:
                return [f" {_DIM}Err: ---{_RESET}"]

    def _build_text(self, state: DashboardState) -> str:
        w = self._width
        border = "\u2500" * w

        # Line 2 — trial / task / demo / recording
        trial_str = f"Trial {state.trial_num}/{state.trial_total}"
        if state.trial_successes > 0:
            trial_str += f" ({state.trial_successes} ok)"

        task_str = state.task_name or ""
        demo_str = f"Demo #{state.demo_num}" if state.demo_num > 0 else ""

        if state.recording:
            rec_str = f"{_BRIGHT_GREEN}{_BOLD}\u25cf REC{_RESET}"
        else:
            rec_str = f"{_DIM}\u25cf REC{_RESET}"

        parts = [p for p in (trial_str, task_str, demo_str, rec_str) if p]
        line2 = " " + " | ".join(parts)

        # Line 3 — gate / peak / detected
        if state.gate_state == "TAKEOVER":
            gate_str = f"Gate: {_BOLD_YELLOW}TAKEOVER{_RESET}"
        else:
            gate_str = f"Gate: {_GREEN}POLICY{_RESET}"

        ratio = (state.peak_value / state.peak_threshold
                 if state.peak_threshold > 0 else 0)
        pct = int(ratio * 100)
        if ratio > 1.0:
            color = _RED
        elif ratio > 0.8:
            color = _YELLOW
        else:
            color = ""
        peak_str = (
            f"Peak: {color}{state.peak_joint_name}={state.peak_value:.1f}"
            f"/{state.peak_threshold:.0f} mA ({pct}%)"
            f"{_RESET if color else ''}"
        )

        if state.takeover_detected:
            det_str = f"Detected: {_BOLD_YELLOW}yes{_RESET}"
        else:
            det_str = "Detected: no"

        line3 = " " + " | ".join((gate_str, peak_str, det_str))

        # Line 4 — release timer (only meaningful during takeover)
        if state.takeover_detected:
            if state.release_timer_active:
                elapsed_str = f"{_BOLD_YELLOW}{state.release_elapsed:.1f}s{_RESET}/{state.release_timeout:.0f}s"
            else:
                elapsed_str = f"waiting"
            vel_ok = state.release_vel_max < state.release_vel_threshold
            curr_ok = state.release_curr_std < state.release_curr_threshold
            vel_mark = f"{_GREEN}ok{_RESET}" if vel_ok else f"{_RED}x{_RESET}"
            curr_mark = f"{_GREEN}ok{_RESET}" if curr_ok else f"{_RED}x{_RESET}"
            line4 = (
                f" Release: {elapsed_str}"
                f" | vel={state.release_vel_max:.3f}/{state.release_vel_threshold} [{vel_mark}]"
                f" | curr_std={state.release_curr_std:.1f}/{state.release_curr_threshold:.0f} [{curr_mark}]"
            )
        else:
            line4 = f" {_DIM}Release: ---{_RESET}"

        # Tracking error rows (only when extra_rows > 0)
        err_lines = self._build_tracking_error_lines(state)

        parts = [border, line2, line3, line4] + err_lines + [border]
        return "\n".join(parts)

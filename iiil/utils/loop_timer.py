"""Lightweight per-section timer for profiling loops.

IIIL_PROFILE_CSV=1 or profile_csv=true in config or env var to enable loop timer profiling.

Usage:
    timer = LoopTimer("control_loop")
    while running:
        timer.mark("start")
        do_usb_read()
        timer.mark("usb_read")
        do_compensation()
        timer.mark("compensation")
        timer.end_iteration()

    timer.save_csv("profile_control_loop.csv")
"""
import csv
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class LoopTimer:
    """Accumulates per-section timing for a loop, writes to CSV."""

    def __init__(self, name: str, print_interval_s: float = 5.0):
        self.name = name
        self._print_interval_s = print_interval_s
        self._marks: list[tuple[str, float]] = []
        self._records: list[dict] = []
        self._last_print_time = 0.0
        self._since_last_print: list[dict] = []

    def mark(self, label: str) -> None:
        """Record a timestamp for a section boundary."""
        self._marks.append((label, time.perf_counter()))

    def end_iteration(self) -> None:
        """Compute section deltas, store record, optionally print summary."""
        marks = self._marks
        if len(marks) < 2:
            self._marks = []
            return

        record = {"timestamp": marks[0][1]}
        for i in range(len(marks) - 1):
            key = f"{marks[i][0]}_to_{marks[i + 1][0]}_ms"
            record[key] = (marks[i + 1][1] - marks[i][1]) * 1000
        record["total_ms"] = (marks[-1][1] - marks[0][1]) * 1000

        self._records.append(record)
        self._since_last_print.append(record)
        self._marks = []

        now = time.perf_counter()
        if now - self._last_print_time >= self._print_interval_s:
            self._print_summary()
            self._last_print_time = now

    def _print_summary(self) -> None:
        """Print mean/max of recent interval."""
        batch = self._since_last_print
        if not batch:
            return

        n = len(batch)
        keys = [k for k in batch[0] if k.endswith("_ms")]
        parts = []
        for k in keys:
            vals = [r[k] for r in batch]
            mean = sum(vals) / n
            mx = max(vals)
            short = k.replace("_ms", "")
            parts.append(f"{short}={mean:.1f}/{mx:.1f}")

        logger.info(f"[{self.name}] n={n} (mean/max ms): {' | '.join(parts)}")
        self._since_last_print = []

    def save_csv(self, path: str) -> None:
        """Write all accumulated records to CSV."""
        if not self._records:
            logger.warning(f"[{self.name}] No records to save")
            return

        fieldnames = list(self._records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)

        logger.info(f"[{self.name}] Saved {len(self._records)} records to {path}")

    @property
    def record_count(self) -> int:
        return len(self._records)

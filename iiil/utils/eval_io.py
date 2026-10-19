"""
I/O helpers for evaluation: video writer cleanup and atomic results writing.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def close_video_writer(video_writer):
    """Flush and close a PyAV (container, stream) video writer tuple.

    Returns None (caller should assign back to clear the reference).
    """
    if video_writer is None:
        return

    container = None
    stream = None
    if isinstance(video_writer, tuple):
        if len(video_writer) >= 1:
            container = video_writer[0]
        if len(video_writer) >= 2:
            stream = video_writer[1]
    else:
        container = video_writer

    try:
        if container is not None and stream is not None:
            for packet in stream.encode():
                container.mux(packet)
    except Exception:
        logger.exception("Failed to flush pending video packets during shutdown")
    finally:
        try:
            if container is not None and hasattr(container, "close"):
                container.close()
        except Exception:
            logger.exception("Failed to close video writer container during shutdown")


def write_results_json(path, n_trials: int, n_success: int) -> None:
    """Atomically write a results JSON file with success rate.

    Uses tmp-file + fsync + rename to avoid partial writes.
    """
    results_json_path = Path(path)
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = results_json_path.with_suffix(results_json_path.suffix + ".tmp")
    payload = {
        "SR": n_success / n_trials if n_trials > 0 else 0.0
    }
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, results_json_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

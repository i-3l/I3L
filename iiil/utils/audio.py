"""Non-blocking text-to-speech announcements via espeak-ng."""
import subprocess
import logging

logger = logging.getLogger(__name__)


def speak(text: str) -> None:
    """Speak text asynchronously using espeak-ng. Non-blocking, fire-and-forget."""
    try:
        subprocess.Popen(
            ["espeak-ng", "-s", "160", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.warning("espeak-ng not found — install with: sudo apt install espeak-ng")
    except Exception as e:
        logger.debug(f"Audio announce failed: {e}")

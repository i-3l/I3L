"""
Capacitive touch sensor reader for Gello takeover detection.

Reads touch state from an ESP32-S3 + MPR121 capacitive sensor
over USB serial (CDC). Provides a thread-safe cached binary
touch state for the 100Hz control loop.

Protocol (from ESP32-S3 firmware):
    "T<hex-bitmask>\\n"  - touch data (e.g., "T800", "T000")
    "E<2-hex-digit-code>\\n"   - error codes (e.g., "E01")
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TouchSensorConfig:
    """Configuration for the capacitive touch sensor."""
    enable: bool = False
    port: str = "/dev/tty_touch_sensor"
    baudrate: int = 115200
    num_channels: int = 1
    reconnect_interval: float = 2.0  # Seconds between reconnection attempts
    stale_timeout: float = 0.5       # Seconds after which cached data is considered stale


class TouchSensorReader:
    """
    Thread-safe serial reader for capacitive touch sensor.

    Runs a background thread that continuously reads the sensor
    and caches the latest touch state. The 100Hz control loop
    reads the cached state without blocking on serial I/O.

    Thread safety:
    - _lock protects _touched_mask, _last_update_time, _last_error, _connected
    - The reader thread writes; the control loop reads.
    - Lock hold time is minimal (reading/writing a few primitives).
    """

    def __init__(self, config: TouchSensorConfig):
        self._config = config
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Cached state (protected by _lock)
        self._touched_mask: int = 0
        self._last_update_time: float = 0.0
        self._last_error: Optional[str] = None
        self._connected: bool = False

        # Serial connection (only accessed from reader thread)
        self._serial = None

    def start(self) -> None:
        """Start the background reader thread."""
        if not self._config.enable:
            logger.info("[TouchSensor] Disabled by config")
            return
        if self._thread is not None and self._thread.is_alive():
            logger.warning("[TouchSensor] Already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="touch-sensor-reader",
        )
        self._thread.start()
        logger.info(f"[TouchSensor] Reader started on {self._config.port}")

    def stop(self) -> None:
        """Stop the background reader thread and close serial port."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._close_serial()
        with self._lock:
            self._connected = False
        logger.info("[TouchSensor] Reader stopped")

    @property
    def is_touched(self) -> bool:
        """Check if ANY channel is currently touched.

        Returns False if sensor is disconnected or data is stale.
        This is the primary API for the TakeoverDetector.
        """
        if not self._config.enable:
            return False
        with self._lock:
            if not self._connected:
                return False
            if time.time() - self._last_update_time > self._config.stale_timeout:
                return False
            return self._touched_mask != 0

    @property
    def touched_channels(self) -> int:
        """Get the raw bitmask of touched channels.

        For multi-channel expansion. Bit N set = channel N touched.
        Returns 0 if disconnected or stale.
        """
        if not self._config.enable:
            return 0
        with self._lock:
            if not self._connected:
                return 0
            if time.time() - self._last_update_time > self._config.stale_timeout:
                return 0
            return self._touched_mask

    @property
    def is_connected(self) -> bool:
        """Check if sensor is connected and responding."""
        with self._lock:
            return self._connected

    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message, if any."""
        with self._lock:
            return self._last_error

    def _reader_loop(self) -> None:
        """Background thread: read serial data and update cached state."""
        while not self._stop_event.is_set():
            # Connect if needed
            if self._serial is None or not self._serial.is_open:
                if not self._try_connect():
                    self._stop_event.wait(self._config.reconnect_interval)
                    continue

            # Read one line
            try:
                line = self._serial.readline()
                if not line:
                    continue  # Timeout, no data
                self._parse_line(line.decode("ascii", errors="ignore").strip())
            except Exception as e:
                # Catch serial errors and any OS-level I/O errors
                logger.warning(f"[TouchSensor] Serial error: {e}")
                with self._lock:
                    self._connected = False
                    self._last_error = str(e)
                self._close_serial()
                self._stop_event.wait(self._config.reconnect_interval)

    def _try_connect(self) -> bool:
        """Attempt to open the serial port."""
        try:
            import serial
            self._serial = serial.Serial(
                port=self._config.port,
                baudrate=self._config.baudrate,
                timeout=0.1,  # 100ms read timeout
            )
            with self._lock:
                self._connected = True
                self._last_error = None
            logger.info(f"[TouchSensor] Connected to {self._config.port}")
            return True
        except Exception as e:
            with self._lock:
                self._connected = False
                self._last_error = f"Connection failed: {e}"
            return False

    def _parse_line(self, line: str) -> None:
        """Parse a single line from the sensor.

        Expected formats:
            "T<hex>" - touch data (e.g., "T01", "T00", "TFF")
            "E<hex>" - error code (e.g., "E01")
        """
        if len(line) < 2:
            return

        prefix = line[0]
        hex_data = line[1:]

        if prefix == "T":
            try:
                mask = int(hex_data, 16)
                with self._lock:
                    self._touched_mask = mask
                    self._last_update_time = time.time()
                    self._last_error = None
            except ValueError:
                pass  # Malformed hex data, skip
        elif prefix == "E":
            with self._lock:
                self._last_error = f"Sensor error: {hex_data}"
            logger.warning(f"[TouchSensor] Sensor error code: {hex_data}")

    def _close_serial(self) -> None:
        """Close serial connection if open."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

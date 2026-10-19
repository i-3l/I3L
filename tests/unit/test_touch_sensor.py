"""Unit tests for capacitive touch sensor reader."""

import time
import pytest

from iiil.gello.control.touch_sensor import TouchSensorReader, TouchSensorConfig


class TestTouchSensorConfig:
    """Test TouchSensorConfig defaults and construction."""

    def test_defaults(self):
        config = TouchSensorConfig()
        assert config.enable is False
        assert config.port == "/dev/tty_touch_sensor"
        assert config.baudrate == 115200
        assert config.num_channels == 1
        assert config.reconnect_interval == 2.0
        assert config.stale_timeout == 0.5

    def test_custom_config(self):
        config = TouchSensorConfig(
            enable=True, port="/dev/ttyACM0", num_channels=4, stale_timeout=1.0
        )
        assert config.enable is True
        assert config.port == "/dev/ttyACM0"
        assert config.num_channels == 4
        assert config.stale_timeout == 1.0


class TestTouchSensorReader:
    """Test TouchSensorReader parsing and state management."""

    def _make_reader(self, enable=True, stale_timeout=10.0):
        """Helper to create a reader with sensible test defaults."""
        config = TouchSensorConfig(enable=enable, stale_timeout=stale_timeout)
        reader = TouchSensorReader(config)
        # Simulate connected state (normally set by _try_connect)
        reader._connected = True
        return reader

    def test_disabled_sensor_returns_false(self):
        reader = TouchSensorReader(TouchSensorConfig(enable=False))
        assert reader.is_touched is False
        assert reader.touched_channels == 0

    def test_parse_touch_active(self):
        reader = self._make_reader()
        reader._parse_line("T01")
        assert reader.is_touched is True
        assert reader.touched_channels == 1

    def test_parse_touch_released(self):
        reader = self._make_reader()
        reader._parse_line("T00")
        assert reader.is_touched is False
        assert reader.touched_channels == 0

    def test_parse_multi_channel(self):
        reader = self._make_reader()
        reader._parse_line("T05")  # Channels 0 and 2
        assert reader.is_touched is True
        assert reader.touched_channels == 0x05

    def test_parse_full_byte(self):
        reader = self._make_reader()
        reader._parse_line("TFF")  # All 8 channels
        assert reader.is_touched is True
        assert reader.touched_channels == 0xFF

    def test_parse_error_line(self):
        reader = self._make_reader()
        reader._parse_line("E01")
        assert reader.last_error == "Sensor error: 01"
        # Touch state should remain unchanged (was 0)
        assert reader.is_touched is False

    def test_error_does_not_clear_touch(self):
        reader = self._make_reader()
        reader._parse_line("T01")
        assert reader.is_touched is True
        reader._parse_line("E02")
        # Error doesn't update touch mask; previous touch state remains
        assert reader.touched_channels == 1

    def test_stale_data_returns_false(self):
        reader = self._make_reader(stale_timeout=0.1)
        reader._parse_line("T01")
        # Manually age the data
        reader._last_update_time = time.time() - 0.2
        assert reader.is_touched is False
        assert reader.touched_channels == 0

    def test_fresh_data_returns_true(self):
        reader = self._make_reader(stale_timeout=0.5)
        reader._parse_line("T01")
        # Data is fresh (just set by _parse_line)
        assert reader.is_touched is True

    def test_disconnected_returns_false(self):
        reader = self._make_reader()
        reader._connected = False
        reader._parse_line("T01")
        reader._last_update_time = time.time()
        assert reader.is_touched is False
        assert reader.touched_channels == 0

    def test_malformed_data_ignored(self):
        reader = self._make_reader()
        reader._parse_line("X")      # Too short
        reader._parse_line("Tgg")    # Invalid hex
        reader._parse_line("")        # Empty
        reader._parse_line("T")       # Prefix only, no hex
        assert reader.touched_channels == 0

    def test_valid_after_malformed(self):
        reader = self._make_reader()
        reader._parse_line("Tgg")     # Invalid
        reader._parse_line("T03")     # Valid
        assert reader.touched_channels == 0x03

    def test_sequential_updates(self):
        reader = self._make_reader()
        reader._parse_line("T01")
        assert reader.touched_channels == 0x01
        reader._parse_line("T03")
        assert reader.touched_channels == 0x03
        reader._parse_line("T00")
        assert reader.touched_channels == 0x00

    def test_is_connected_default(self):
        config = TouchSensorConfig(enable=True)
        reader = TouchSensorReader(config)
        assert reader.is_connected is False

    def test_last_error_default(self):
        config = TouchSensorConfig(enable=True)
        reader = TouchSensorReader(config)
        assert reader.last_error is None

    def test_start_when_disabled_is_noop(self):
        reader = TouchSensorReader(TouchSensorConfig(enable=False))
        reader.start()  # Should not raise or start a thread
        assert reader._thread is None or not reader._thread.is_alive()

    def test_stop_when_not_started(self):
        reader = TouchSensorReader(TouchSensorConfig(enable=True))
        reader.stop()  # Should not raise

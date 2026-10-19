"""Unit tests for touch-based takeover detection."""

import time
import numpy as np
import pytest
from unittest.mock import MagicMock

from iiil.gello.control.takeover_detector import TakeoverDetector, TakeoverConfig
from iiil.gello.configs.motor_type import MotorType


def _make_detector(
    use_touch=True,
    use_current=False,
    use_position=False,
    use_velocity=False,
    touch_release_cooldown=0.5,
    auto_return_timeout=0.3,
    dof=6,
):
    """Helper to create a TakeoverDetector with touch config."""
    config = TakeoverConfig(
        enable=True,
        use_touch=use_touch,
        use_current=use_current,
        use_position=use_position,
        use_velocity=use_velocity,
        use_momentum=False,
        touch_release_cooldown=touch_release_cooldown,
        enable_auto_return=True,
        auto_return_timeout=auto_return_timeout,
        takeover_grace_period=0.0,  # No grace period for tests
    )
    motor_types = [MotorType.XM430_W350_T] * min(3, dof) + [MotorType.XC330_M288_T] * max(0, dof - 3)
    detector = TakeoverDetector(config, dof, motor_types)
    detector.enable()
    # Suppress grace period for testing
    detector._suppress_until = 0.0
    return detector


def _zeros(dof=6):
    """Helper to create zero arrays for detect_takeover args."""
    return np.zeros(dof, dtype=np.float32)


class TestTouchOnlyDetection:
    """Test touch-only takeover detection mode."""

    def test_touch_triggers_takeover(self):
        detector = _make_detector()
        mock_sensor = MagicMock()
        mock_sensor.is_touched = True
        detector.set_touch_sensor(mock_sensor)

        result = detector.detect_takeover(
            _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros()
        )
        assert result is True

    def test_no_touch_no_takeover(self):
        detector = _make_detector()
        mock_sensor = MagicMock()
        mock_sensor.is_touched = False
        detector.set_touch_sensor(mock_sensor)

        result = detector.detect_takeover(
            _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros()
        )
        assert result is False

    def test_touch_state_tracks_sensor(self):
        detector = _make_detector()
        mock_sensor = MagicMock()
        detector.set_touch_sensor(mock_sensor)

        # Touch on
        mock_sensor.is_touched = True
        assert detector.detect_takeover(
            _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros()
        ) is True

        # Touch off
        mock_sensor.is_touched = False
        assert detector.detect_takeover(
            _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros()
        ) is False

        # Touch on again
        mock_sensor.is_touched = True
        assert detector.detect_takeover(
            _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros()
        ) is True

    def test_no_sensor_no_takeover(self):
        """If use_touch=True but no sensor injected, should not trigger."""
        detector = _make_detector()
        # Don't set a touch sensor
        result = detector.detect_takeover(
            _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros()
        )
        assert result is False

    def test_last_takeover_state_updated(self):
        detector = _make_detector()
        mock_sensor = MagicMock()
        mock_sensor.is_touched = True
        detector.set_touch_sensor(mock_sensor)

        detector.detect_takeover(
            _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros()
        )
        assert detector._last_takeover_state is True


class TestTouchReleaseDetection:
    """Test touch-based release detection."""

    def test_touch_release_cooldown(self):
        detector = _make_detector(touch_release_cooldown=0.1)
        mock_sensor = MagicMock()
        mock_sensor.is_touched = False
        detector.set_touch_sensor(mock_sensor)

        # First call: cooldown starts, not released yet
        assert detector.detect_release(_zeros(), _zeros()) is False

        # Wait for cooldown
        time.sleep(0.15)
        assert detector.detect_release(_zeros(), _zeros()) is True

    def test_touch_active_prevents_release(self):
        detector = _make_detector(touch_release_cooldown=0.05)
        mock_sensor = MagicMock()
        detector.set_touch_sensor(mock_sensor)

        # Touch still active
        mock_sensor.is_touched = True
        assert detector.detect_release(_zeros(), _zeros()) is False

        time.sleep(0.1)
        # Still touching, should not release
        assert detector.detect_release(_zeros(), _zeros()) is False

    def test_touch_reactivation_resets_cooldown(self):
        detector = _make_detector(touch_release_cooldown=0.1)
        mock_sensor = MagicMock()
        detector.set_touch_sensor(mock_sensor)

        # Start release timer
        mock_sensor.is_touched = False
        detector.detect_release(_zeros(), _zeros())

        time.sleep(0.05)

        # Touch again -- should reset timer
        mock_sensor.is_touched = True
        detector.detect_release(_zeros(), _zeros())

        # Release again -- timer restarted
        mock_sensor.is_touched = False
        detector.detect_release(_zeros(), _zeros())

        # Not enough time since last touch reactivation
        time.sleep(0.05)
        assert detector.detect_release(_zeros(), _zeros()) is False

        # Now enough time
        time.sleep(0.1)
        assert detector.detect_release(_zeros(), _zeros()) is True

    def test_reset_clears_touch_state(self):
        detector = _make_detector(touch_release_cooldown=0.05)
        mock_sensor = MagicMock()
        mock_sensor.is_touched = False
        detector.set_touch_sensor(mock_sensor)

        # Start release timer
        detector.detect_release(_zeros(), _zeros())

        # Reset clears the timer
        detector.reset()
        assert detector._touch_released_time is None


class TestMixedModeDetection:
    """Test touch + analog mixed mode detection."""

    def test_touch_triggers_in_mixed_mode(self):
        detector = _make_detector(use_touch=True, use_current=True)
        mock_sensor = MagicMock()
        mock_sensor.is_touched = True
        detector.set_touch_sensor(mock_sensor)

        # Current is below threshold, but touch is active
        result = detector.detect_takeover(
            _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros(), _zeros()
        )
        assert result is True

    def test_mixed_mode_touch_prevents_release(self):
        """In mixed mode, if touch is still active, release should return False."""
        detector = _make_detector(
            use_touch=True, use_current=True, touch_release_cooldown=0.05
        )
        mock_sensor = MagicMock()
        mock_sensor.is_touched = True
        detector.set_touch_sensor(mock_sensor)

        # Touch active prevents release regardless of current/velocity
        assert detector.detect_release(_zeros(), _zeros()) is False

        time.sleep(0.1)
        assert detector.detect_release(_zeros(), _zeros()) is False

    def test_mixed_mode_falls_through_to_analog_release(self):
        """In mixed mode with touch released, should use analog release detection."""
        detector = _make_detector(
            use_touch=True,
            use_current=True,
            auto_return_timeout=0.05,
        )
        mock_sensor = MagicMock()
        mock_sensor.is_touched = False  # Touch released
        detector.set_touch_sensor(mock_sensor)

        # Touch released, falls through to velocity/current checks
        # With zero velocity and zero current, the analog checks should pass
        # after the auto_return_timeout
        detector.detect_release(_zeros(), _zeros())
        time.sleep(0.1)
        result = detector.detect_release(_zeros(), _zeros())
        assert result is True


class TestIsTouchOnlyMode:
    """Test the _is_touch_only_mode helper."""

    def test_touch_only(self):
        detector = _make_detector(
            use_touch=True, use_current=False, use_position=False, use_velocity=False
        )
        assert detector._is_touch_only_mode() is True

    def test_mixed_mode(self):
        detector = _make_detector(
            use_touch=True, use_current=True, use_position=False, use_velocity=False
        )
        assert detector._is_touch_only_mode() is False

    def test_no_touch(self):
        detector = _make_detector(
            use_touch=False, use_current=True, use_position=True, use_velocity=False
        )
        assert detector._is_touch_only_mode() is False

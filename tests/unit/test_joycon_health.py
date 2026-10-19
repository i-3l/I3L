"""Unit tests for Joy-Con health tracking and reconnection logic.

Run with: python tests/unit/test_joycon_health.py
"""

import sys
import os
import time
import importlib.util

# Load joycon_utils directly to avoid omnigibson dependency
_spec = importlib.util.spec_from_file_location(
    'joycon_utils',
    os.path.join(os.path.dirname(__file__), '../../iiil/utils/joycon_utils.py')
)
_joycon_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_joycon_utils)

JoyConHealthStatus = _joycon_utils.JoyConHealthStatus
BaseJoyConInterface = _joycon_utils.BaseJoyConInterface


class MockJoyCon:
    """Mock JoyCon for testing."""

    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.call_count = 0

    def get_button_x(self):
        self.call_count += 1
        if self.should_fail:
            raise Exception("Connection lost")
        return True

    def get_status(self):
        if self.should_fail:
            raise Exception("Connection lost")
        return {"battery": 3}

    def get_battery_level(self):
        return 3


class TestableJoyConInterface(BaseJoyConInterface):
    """Testable subclass that doesn't connect to real hardware."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mock_joycon = MockJoyCon()
        self.jc_right = self.mock_joycon
        self.connected = True

    def _get_required_joycons(self):
        return ["right"]

    def get_gripper_command(self):
        return {"gripper": 0.0}


def run_tests():
    """Run all tests and return (passed, failed) counts."""
    tests_passed = 0
    tests_failed = 0

    print("=== Testing Health Tracking ===")

    # Test 1
    print("Test 1: initial_state_is_healthy...", end=" ")
    try:
        interface = TestableJoyConInterface(required=False)
        assert interface.is_healthy is True
        assert interface.is_reconnecting is False
        print("PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"FAILED: {e}")
        tests_failed += 1

    # Test 2
    print("Test 2: successful_read_stays_healthy...", end=" ")
    try:
        interface = TestableJoyConInterface(required=False)
        result = interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        assert result is True
        assert interface.is_healthy is True
        assert interface._health_status["right"]["failures"] == 0
        print("PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"FAILED: {e}")
        tests_failed += 1

    # Test 3
    print("Test 3: failure_tracking_to_degraded...", end=" ")
    try:
        interface = TestableJoyConInterface(required=False, failure_threshold=5)
        interface.mock_joycon.should_fail = True
        interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        assert interface._health_status["right"]["failures"] == 2
        assert interface._health_status["right"]["status"] == JoyConHealthStatus.DEGRADED
        assert interface.is_healthy is False
        print("PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"FAILED: {e}")
        tests_failed += 1

    # Test 4
    print("Test 4: threshold_triggers_disconnect...", end=" ")
    try:
        interface = TestableJoyConInterface(required=False, failure_threshold=3)
        interface.mock_joycon.should_fail = True
        for i in range(3):
            interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        assert interface._health_status["right"]["failures"] == 3
        assert interface._health_status["right"]["status"] == JoyConHealthStatus.DISCONNECTED
        interface.close()
        print("PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"FAILED: {e}")
        tests_failed += 1

    # Test 5
    print("Test 5: success_resets_degraded_to_healthy...", end=" ")
    try:
        interface = TestableJoyConInterface(required=False, failure_threshold=5)
        interface.mock_joycon.should_fail = True
        for _ in range(3):
            interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        assert interface._health_status["right"]["failures"] == 3
        assert interface._health_status["right"]["status"] == JoyConHealthStatus.DEGRADED
        interface.mock_joycon.should_fail = False
        interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        assert interface._health_status["right"]["failures"] == 0
        assert interface._health_status["right"]["status"] == JoyConHealthStatus.HEALTHY
        print("PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"FAILED: {e}")
        tests_failed += 1

    # Test 6
    print("Test 6: disconnected_skips_reads...", end=" ")
    try:
        interface = TestableJoyConInterface(required=False, failure_threshold=2)
        interface.mock_joycon.should_fail = True
        interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        assert interface._health_status["right"]["status"] == JoyConHealthStatus.DISCONNECTED
        interface.mock_joycon.call_count = 0
        interface.mock_joycon.should_fail = False
        result = interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        assert result is False
        assert interface.mock_joycon.call_count == 0
        interface.close()
        print("PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"FAILED: {e}")
        tests_failed += 1

    # Test 7
    print("Test 7: reconnecting_skips_reads...", end=" ")
    try:
        interface = TestableJoyConInterface(required=False)
        interface._health_status["right"] = {
            "status": JoyConHealthStatus.RECONNECTING,
            "failures": 2,
            "reconnect_attempts": 1,
        }
        interface.mock_joycon.call_count = 0
        result = interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        assert result is False
        assert interface.mock_joycon.call_count == 0
        print("PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"FAILED: {e}")
        tests_failed += 1

    # Test 8
    print("Test 8: dual_joycon_health...", end=" ")
    try:
        interface = TestableJoyConInterface(required=False)
        interface._get_required_joycons = lambda: ["left", "right"]
        interface.jc_left = MockJoyCon()
        assert interface.is_healthy is True
        interface._health_status["left"] = {
            "status": JoyConHealthStatus.DISCONNECTED,
            "failures": 5,
            "reconnect_attempts": 0,
        }
        assert interface.is_healthy is False
        print("PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"FAILED: {e}")
        tests_failed += 1

    print()
    print("=== Testing Stale Data Detection ===")

    # Test 9: stale data triggers disconnect after threshold iterations
    print("Test 9: stale_data_triggers_disconnect...", end=" ")
    try:
        interface = TestableJoyConInterface(required=False, failure_threshold=10)
        # Simulate Joy-Con with frozen _input_report
        interface.mock_joycon._input_report = bytes([0x30] + [0] * 48)
        interface._stale_threshold = 3  # 3 iterations for faster test

        # First read - initializes tracking
        interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        assert interface.is_healthy is True

        # Subsequent reads with unchanged data should increment stale counter
        # and eventually trigger failure
        for i in range(4):
            interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")

        # Should have recorded a failure due to stale data
        assert interface._health_status["right"]["failures"] >= 1
        print("PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"FAILED: {e}")
        tests_failed += 1

    # Test 10: changing data stays healthy
    print("Test 10: changing_data_stays_healthy...", end=" ")
    try:
        interface = TestableJoyConInterface(required=False, failure_threshold=10)
        interface.mock_joycon._input_report = bytes([0x30] + [0] * 48)
        interface._stale_threshold = 3

        # First read
        interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")

        # Change the report data each time (simulating live updates)
        for i in range(5):
            interface.mock_joycon._input_report = bytes([0x30] + [i + 1] * 48)
            interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")

        assert interface._health_status["right"]["failures"] == 0
        assert interface.is_healthy is True
        print("PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"FAILED: {e}")
        tests_failed += 1

    print()
    print("=== Testing Reconnection Logic ===")

    # Test 11
    print("Test 11: disconnect_triggers_reconnect...", end=" ")
    try:
        interface = TestableJoyConInterface(
            required=False, failure_threshold=2, reconnect_interval=0.1
        )
        interface.mock_joycon.should_fail = True
        interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        time.sleep(0.05)
        assert interface._reconnect_thread is not None
        assert interface._reconnect_thread.is_alive()
        interface.close()
        print("PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"FAILED: {e}")
        tests_failed += 1

    # Test 12
    print("Test 12: close_stops_thread...", end=" ")
    try:
        interface = TestableJoyConInterface(
            required=False, failure_threshold=2, reconnect_interval=1.0
        )
        interface.mock_joycon.should_fail = True
        interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        interface._safe_get(interface.jc_right, "get_button_x", joycon_id="right")
        time.sleep(0.1)
        assert interface._reconnect_thread.is_alive()
        interface.close()
        time.sleep(0.1)
        assert interface._reconnect_thread is None or not interface._reconnect_thread.is_alive()
        print("PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"FAILED: {e}")
        tests_failed += 1

    print()
    return tests_passed, tests_failed


if __name__ == "__main__":
    passed, failed = run_tests()
    print(f"=== Results: {passed}/{passed + failed} tests passed ===")
    sys.exit(0 if failed == 0 else 1)

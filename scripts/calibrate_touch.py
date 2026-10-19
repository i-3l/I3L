#!/usr/bin/env python3
"""
Auto-tune script for the MPR121 capacitive touch sensor.

Guides the user through touch/no-touch measurements, computes optimal
thresholds, and updates the firmware source with the new values.

Usage:
    python scripts/calibrate_touch.py                  # auto-detect port
    python scripts/calibrate_touch.py /dev/ttyACM0     # explicit port
"""

import sys
import time
import glob
import re
from pathlib import Path

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

# ---------- Config ----------

BAUDRATE = 115200
SAMPLES_PER_PHASE = 1000     # Number of samples to collect per phase
SAMPLE_SETTLE_SKIP = 50      # Skip first N samples after prompt (settling)
FIRMWARE_PATH = (
    Path(__file__).parent.parent
    / "iiil" / "gello" / "capacitive_sensor" / "capacitive_sensor.ino"
)


def find_port():
    """Auto-detect the touch sensor serial port."""
    for pattern in ["/dev/tty_touch_sensor", "/dev/ttyACM*", "/dev/ttyUSB*"]:
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    return None


def parse_calibration_line(line):
    """Parse a calibration line: C<filtered>,<baseline>,<hex>.

    Returns (filtered, baseline) or None if not a calibration line.
    """
    line = line.strip()
    if not line.startswith("C"):
        return None
    try:
        parts = line[1:].split(",")
        if len(parts) != 3:
            return None
        filtered = int(parts[0])
        baseline = int(parts[1])
        return filtered, baseline
    except (ValueError, IndexError):
        return None


def collect_samples(ser, n_samples, skip=0):
    """Collect n_samples calibration readings, skipping the first `skip`."""
    samples = []
    skipped = 0
    # Drain any stale data
    ser.reset_input_buffer()

    report_interval = max(1, n_samples // 20)  # Report ~20 times

    while len(samples) < n_samples:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("ascii", errors="ignore").strip()
        parsed = parse_calibration_line(line)
        if parsed is None:
            continue
        if skipped < skip:
            skipped += 1
            continue
        samples.append(parsed)
        n = len(samples)
        if n == 1 or n % report_interval == 0 or n == n_samples:
            filtered, baseline = parsed
            delta = baseline - filtered
            deltas = [b - f for f, b in samples]
            mean_d = sum(deltas) / len(deltas)
            print(f"  [{n:4d}/{n_samples}] filtered={filtered:4d}  "
                  f"baseline={baseline:4d}  delta={delta:4d}  "
                  f"(mean delta: {mean_d:.1f})")

    return samples


def percentile(data, p):
    """Compute the p-th percentile of a list."""
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    d = k - f
    return s[f] + d * (s[c] - s[f])


def std_dev(data):
    """Compute standard deviation."""
    m = sum(data) / len(data)
    return (sum((x - m) ** 2 for x in data) / len(data)) ** 0.5


def compute_thresholds(no_touch_samples, touch_samples):
    """Compute optimal touch and release thresholds from calibration data.

    The MPR121 triggers touch when: baseline - filtered > touch_threshold
    and triggers release when: baseline - filtered < release_threshold.

    Returns (touch_threshold, release_threshold) as integers.
    """
    # Delta = baseline - filtered
    no_touch_deltas = [b - f for f, b in no_touch_samples]
    touch_deltas = [b - f for f, b in touch_samples]

    no_touch_mean = sum(no_touch_deltas) / len(no_touch_deltas)
    no_touch_std = std_dev(no_touch_deltas)
    touch_mean = sum(touch_deltas) / len(touch_deltas)
    touch_std = std_dev(touch_deltas)

    no_touch_p50 = percentile(no_touch_deltas, 50)
    no_touch_p95 = percentile(no_touch_deltas, 95)
    no_touch_p99 = percentile(no_touch_deltas, 99)
    no_touch_max = max(no_touch_deltas)

    touch_min = min(touch_deltas)
    touch_p01 = percentile(touch_deltas, 1)
    touch_p05 = percentile(touch_deltas, 5)
    touch_p50 = percentile(touch_deltas, 50)

    print(f"\n{'=' * 50}")
    print("Calibration Results")
    print(f"{'=' * 50}")
    print(f"\n  No-touch deltas ({len(no_touch_deltas)} samples):")
    print(f"    mean={no_touch_mean:.1f}  std={no_touch_std:.1f}")
    print(f"    p50={no_touch_p50:.0f}  p95={no_touch_p95:.0f}  "
          f"p99={no_touch_p99:.0f}  max={no_touch_max}")
    print(f"\n  Touch deltas ({len(touch_deltas)} samples):")
    print(f"    mean={touch_mean:.1f}  std={touch_std:.1f}")
    print(f"    min={touch_min}  p1={touch_p01:.0f}  "
          f"p5={touch_p05:.0f}  p50={touch_p50:.0f}")
    print(f"\n  Separation (touch p5 - no-touch p95): "
          f"{touch_p05 - no_touch_p95:.0f}")

    if touch_p05 <= no_touch_p95:
        print("\n  WARNING: Touch and no-touch ranges overlap!")
        print("  The sensor may not reliably distinguish touch.")
        print("  Check electrode connection and try again.")

    # Touch threshold: set between the no-touch p99 and touch p5,
    # biased 30% from no-touch side for margin against noise
    gap = touch_p05 - no_touch_p99
    touch_thresh = max(1, int(no_touch_p99 + gap * 0.3))

    # Release threshold: ~60% of touch threshold (hysteresis)
    release_thresh = max(1, int(touch_thresh * 0.6))

    # Ensure touch > release
    if release_thresh >= touch_thresh:
        release_thresh = max(1, touch_thresh - 1)

    return touch_thresh, release_thresh


def update_firmware(touch_thresh, release_thresh):
    """Update TOUCH_THRESHOLD and RELEASE_THRESHOLD in the .ino file."""
    if not FIRMWARE_PATH.exists():
        print(f"\n  Firmware file not found: {FIRMWARE_PATH}")
        print(f"  Manually set: TOUCH_THRESHOLD={touch_thresh}, "
              f"RELEASE_THRESHOLD={release_thresh}")
        return False

    content = FIRMWARE_PATH.read_text()

    new_content = re.sub(
        r"#define TOUCH_THRESHOLD\s+\d+",
        f"#define TOUCH_THRESHOLD  {touch_thresh}",
        content,
    )
    new_content = re.sub(
        r"#define RELEASE_THRESHOLD\s+\d+",
        f"#define RELEASE_THRESHOLD {release_thresh}",
        new_content,
    )

    if new_content == content:
        print("\n  No changes needed (thresholds already match).")
        return False

    FIRMWARE_PATH.write_text(new_content)
    print(f"\n  Updated {FIRMWARE_PATH}")
    return True


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_port()
    if port is None:
        print("ERROR: No serial port found. Plug in the sensor or specify a port.")
        sys.exit(1)

    print(f"{'=' * 50}")
    print("Touch Sensor Calibration")
    print(f"{'=' * 50}")
    print(f"  Port: {port}")
    print(f"  Samples per phase: {SAMPLES_PER_PHASE}")
    print()

    ser = serial.Serial(port, BAUDRATE, timeout=1.0)
    time.sleep(0.5)  # Let connection settle

    # Enter calibration mode
    ser.write(b"C")
    ser.flush()
    time.sleep(0.2)

    # Verify we're getting calibration data
    ser.reset_input_buffer()
    got_cal = False
    for _ in range(20):
        raw = ser.readline()
        if raw and raw.decode("ascii", errors="ignore").strip().startswith("C"):
            got_cal = True
            break
    if not got_cal:
        print("ERROR: Not receiving calibration data.")
        print("Make sure the firmware with calibration support is flashed.")
        ser.write(b"N")
        ser.close()
        sys.exit(1)

    # Phase 1: No touch
    print(f"{'=' * 50}")
    print("Phase 1: NO TOUCH")
    print(f"{'=' * 50}")
    input("  Remove your hands from the sensor, then press Enter...")
    print()
    no_touch_samples = collect_samples(
        ser, SAMPLES_PER_PHASE, skip=SAMPLE_SETTLE_SKIP
    )

    # Phase 2: Touch
    print(f"\n{'=' * 50}")
    print("Phase 2: TOUCH")
    print(f"{'=' * 50}")
    input("  Touch the sensor pad firmly, then press Enter...")
    print()
    touch_samples = collect_samples(
        ser, SAMPLES_PER_PHASE, skip=SAMPLE_SETTLE_SKIP
    )

    # Return to normal mode
    ser.write(b"N")
    ser.flush()
    ser.close()

    # Compute thresholds
    touch_thresh, release_thresh = compute_thresholds(
        no_touch_samples, touch_samples
    )

    print(f"\n  >> Touch threshold:   {touch_thresh}")
    print(f"  >> Release threshold: {release_thresh}")

    # Update firmware
    print(f"\n{'=' * 50}")
    print("Update Firmware")
    print(f"{'=' * 50}")

    resp = input(f"\n  Write thresholds to firmware source? (y/n): ").strip().lower()
    if resp == "y":
        changed = update_firmware(touch_thresh, release_thresh)
        if changed:
            print("\n  Reflash to apply:")
            print("    ./flash.sh /dev/ttyACM0")
    else:
        print(f"\n  To apply manually, edit capacitive_sensor.ino:")
        print(f"    #define TOUCH_THRESHOLD  {touch_thresh}")
        print(f"    #define RELEASE_THRESHOLD {release_thresh}")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCalibration cancelled.")
        sys.exit(1)
    except serial.SerialException as e:
        print(f"\nSerial error: {e}")
        sys.exit(1)

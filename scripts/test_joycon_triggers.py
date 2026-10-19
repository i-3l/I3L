#!/usr/bin/env python3
"""Quick test to verify JoyCon trigger buttons are readable.

Usage:
    python scripts/test_joycon_triggers.py

Press R and ZR triggers on the right JoyCon. You should see
trigger_r and trigger_zr toggle True/False.
Press Ctrl+C to exit.
"""

import time


def main():
    print("Connecting to JoyCon...")
    from iiil.utils.joycon_utils import get_joycon_interface

    jc = get_joycon_interface("Franka", required=True)
    print(f"Connected: {jc.connected}")
    print(f"Healthy: {jc.is_healthy}")
    print()
    print("Press R or ZR triggers (Ctrl+C to quit):")
    print()

    try:
        while True:
            r = jc.trigger_r
            zr = jc.trigger_zr
            gripper = jc.get_gripper_command()
            # Also check buttons to confirm JoyCon is working
            x = jc.button_x
            a = jc.button_a

            print(
                f"\rtrigger_r={r!s:5s}  trigger_zr={zr!s:5s}  "
                f"gripper_cmd={gripper}  "
                f"button_x={x!s:5s}  button_a={a!s:5s}",
                end="", flush=True,
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()

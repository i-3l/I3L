#!/usr/bin/env python3
"""
Read and print GELLO joint positions from hardware without moving the arm.

Usage:
    python scripts/read_gello_joints.py --config gello_a1 --repeat 1 --interval 1.0
    python scripts/read_gello_joints.py --config gello_r1pro --repeat 0 --interval 0.5
"""

import argparse
import sys
import time
import numpy as np

from iiil.gello.factory import (
    load_gello_config,
    build_motor_groups,
    build_compensation_config,
    build_controller,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read GELLO joint positions from connected servos.")
    parser.add_argument(
        "--config",
        default="gello_a1",
        help="GELLO config name (YAML in iiil/configs/gello, without extension, e.g., gello_a1 or gello_r1pro).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to read (use 0 for infinite loop).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between reads when repeating.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config and build a controller; do NOT start control (no motion).
    cfg = load_gello_config(args.config)
    motor_groups = build_motor_groups(cfg, port_overrides=None)
    comp_cfg = build_compensation_config(cfg)
    controller = build_controller(cfg.bilateral_control_type, motor_groups, comp_cfg, cfg)

    try:
        dof = controller.dof
        count = 0
        while True:
            joint_angles, valid_mask = controller.q
            joint_angles_deg = np.rad2deg(joint_angles)
            print(f"[{time.strftime('%H:%M:%S')}]")
            print(f"  DOF         : {dof}")
            print("  Joints (rad):", np.array2string(joint_angles, precision=4, max_line_width=200))
            print("  Joints (deg):", np.array2string(joint_angles_deg, precision=2, max_line_width=200))
            print("  Valid mask  :", valid_mask.astype(int))
            print("-" * 50)

            count += 1
            if args.repeat != 0 and count >= args.repeat:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        controller.close()


if __name__ == "__main__":
    # Ensure repository root is on sys.path when running from scripts/
    sys.path.append(".")
    main()

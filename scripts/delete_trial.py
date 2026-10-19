#!/usr/bin/env python3
"""
Delete a trial from an HDF5 dataset file.

Usage:
    python scripts/delete_trial.py path/to/data.hdf5
    python scripts/delete_trial.py path/to/data.hdf5 --demo 3
    python scripts/delete_trial.py path/to/data.hdf5 --demo 3 --no-confirm
"""

import argparse
import h5py
import os
import sys
from pathlib import Path


def list_demos(hdf5_path: str) -> list:
    """List all demos in the HDF5 file with their step counts."""
    demos = []
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            return demos
        for key in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]) if x.startswith("demo_") else float('inf')):
            if key.startswith("demo_"):
                demo_id = int(key.split("_")[1])
                num_steps = f["data"][key].attrs.get("num_samples", "?")
                demos.append((demo_id, key, num_steps))
    return demos


def backup_trial(src_path: str, demo_key: str, demo_id: int) -> str:
    """
    Backup a single trial to a separate HDF5 file.

    Returns the backup file path.
    """
    src_path = Path(src_path)
    src_dir = src_path.parent
    src_name = src_path.stem

    backup_path = src_dir / f"deleted_{src_name}_trajectory_{demo_id}.hdf5"

    # If backup already exists, add a suffix
    counter = 1
    while backup_path.exists():
        backup_path = src_dir / f"deleted_{src_name}_trajectory_{demo_id}_{counter}.hdf5"
        counter += 1

    with h5py.File(src_path, "r") as src, h5py.File(backup_path, "w") as dst:
        # Create data group
        dst_data = dst.create_group("data")

        # Copy the demo as demo_0 in backup
        src.copy(f"data/{demo_key}", dst_data, name="demo_0")

        # Copy metadata
        dst_data.attrs["n_episodes"] = 1
        if "env_args" in src["data"].attrs:
            dst_data.attrs["env_args"] = src["data"].attrs["env_args"]
        if "target_frequency_hz" in src["data"].attrs:
            dst_data.attrs["target_frequency_hz"] = src["data"].attrs["target_frequency_hz"]

    return str(backup_path)


def delete_trial(hdf5_path: str, demo_id: int) -> bool:
    """
    Delete a trial and renumber remaining demos to keep sequential numbering.

    Returns True if successful.
    """
    demo_key = f"demo_{demo_id}"

    with h5py.File(hdf5_path, "a") as f:
        if "data" not in f or demo_key not in f["data"]:
            print(f"Error: {demo_key} not found in {hdf5_path}")
            return False

        # Get all demo keys sorted by ID
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1])
        )

        # Delete the target demo
        del f["data"][demo_key]

        # Renumber remaining demos to keep sequential
        # First, collect demos that need renumbering (those after the deleted one)
        demos_to_rename = []
        for key in demo_keys:
            current_id = int(key.split("_")[1])
            if current_id > demo_id:
                demos_to_rename.append((key, current_id))

        # Rename in order (lower IDs first to avoid conflicts)
        for old_key, old_id in demos_to_rename:
            new_id = old_id - 1
            new_key = f"demo_{new_id}"
            f["data"].move(old_key, new_key)

        # Update n_episodes attribute if it exists
        if "n_episodes" in f["data"].attrs:
            f["data"].attrs["n_episodes"] = f["data"].attrs["n_episodes"] - 1

    return True


def confirm_deletion(demo_id: int, num_steps: int) -> bool:
    """Show confirmation prompt and return True if user confirms."""
    print(f"\n\033[1m\033[93m{'='*60}\033[0m")
    print(f"\033[1m\033[93m  WARNING: About to delete demo_{demo_id} ({num_steps} steps)\033[0m")
    print(f"\033[1m\033[93m{'='*60}\033[0m\n")

    while True:
        response = input("Are you sure you want to delete this trial? [y/N]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no", ""):
            return False
        else:
            print("Please enter 'y' or 'n'")


def main():
    parser = argparse.ArgumentParser(description="Delete a trial from an HDF5 dataset")
    parser.add_argument("hdf5_path", help="Path to the HDF5 file")
    parser.add_argument("--demo", "-d", type=int, help="Demo ID to delete (if not specified, will show interactive list)")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--list", "-l", action="store_true", help="Just list demos without deleting")
    args = parser.parse_args()

    hdf5_path = os.path.expanduser(args.hdf5_path)

    if not os.path.exists(hdf5_path):
        print(f"Error: File not found: {hdf5_path}")
        sys.exit(1)

    # List all demos
    demos = list_demos(hdf5_path)

    if not demos:
        print(f"No demos found in {hdf5_path}")
        sys.exit(1)

    # Print demo list
    print(f"\n\033[1mDemos in {hdf5_path}:\033[0m\n")
    print(f"  {'ID':<6} {'Name':<12} {'Steps':<10}")
    print(f"  {'-'*6} {'-'*12} {'-'*10}")
    for demo_id, demo_key, num_steps in demos:
        print(f"  {demo_id:<6} {demo_key:<12} {num_steps:<10}")
    print()

    if args.list:
        sys.exit(0)

    # Get demo ID to delete
    if args.demo is not None:
        demo_id = args.demo
    else:
        try:
            demo_id = int(input("Enter demo ID to delete: ").strip())
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(0)

    # Check if demo exists
    demo_info = next((d for d in demos if d[0] == demo_id), None)
    if demo_info is None:
        print(f"Error: demo_{demo_id} not found")
        sys.exit(1)

    demo_id, demo_key, num_steps = demo_info

    # Confirm deletion
    if not args.no_confirm:
        if not confirm_deletion(demo_id, num_steps):
            print("Cancelled.")
            sys.exit(0)

    # Backup the trial
    print(f"\nBacking up {demo_key}...", end=" ", flush=True)
    backup_path = backup_trial(hdf5_path, demo_key, demo_id)
    print(f"\033[92mDone\033[0m")
    print(f"  Backup saved to: {backup_path}")

    # Delete the trial
    print(f"Deleting {demo_key}...", end=" ", flush=True)
    if delete_trial(hdf5_path, demo_id):
        print(f"\033[92mDone\033[0m")

        # Show updated list
        demos = list_demos(hdf5_path)
        print(f"\n\033[1mRemaining demos:\033[0m")
        if demos:
            for did, dkey, nsteps in demos:
                print(f"  {dkey} ({nsteps} steps)")
        else:
            print("  (none)")
        print()
    else:
        print(f"\033[91mFailed\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()

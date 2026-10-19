#!/usr/bin/env python3
"""
Interactive script to detect and configure USB symlinks for ZED cameras.

This script walks you through:
1. Detecting ZED cameras by plug/unplug events
2. Reading the USB physical port path for stable identification
3. Verifying the camera with an OpenCV test capture
4. Creating udev rules so symlinks persist across reboots

Prerequisites:
  - User must be in the 'video' group:  sudo usermod -aG video $USER
  - Re-login after adding to video group
"""

import glob
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Symlink names for each camera role
CAMERA_ROLES = {
    "wrist": "/dev/zed_wrist",
    "external": "/dev/zed_external",
}

UDEV_RULES_PATH = "/etc/udev/rules.d/99-zed-cameras.rules"


def get_video_devices() -> List[str]:
    """Get list of /dev/video* devices."""
    return sorted(glob.glob("/dev/video*"))


def get_usb_port_path(device: str) -> Optional[str]:
    """Get the USB physical port path (KERNELS) for a video device.

    This identifies the physical USB port, not the device itself, so
    the symlink stays stable regardless of plug order.
    """
    try:
        output = subprocess.check_output(
            ["udevadm", "info", "--attribute-walk", "--name", device],
            text=True, stderr=subprocess.DEVNULL,
        )
        # Walk up the device tree to find the USB port KERNELS entry
        # We want the first KERNELS that looks like a USB port (e.g. "8-1")
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("ATTRS{idVendor}") and "2b03" in line:
                # Found a STEREOLABS device in the ancestry — good
                pass
            if line.startswith('KERNELS=="') and line.endswith('"'):
                kernel = line.split('"')[1]
                # USB port paths look like "8-1", "1-2.3", etc.
                if kernel and "-" in kernel and not kernel.startswith("usb"):
                    return kernel
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


def get_device_info(device: str) -> Dict[str, str]:
    """Get udevadm info properties for a device."""
    info = {}
    try:
        output = subprocess.check_output(
            ["udevadm", "info", "--query=all", "--name", device],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines():
            if line.startswith("E: "):
                key, _, value = line[3:].partition("=")
                info[key] = value
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def is_zed_device(device: str) -> bool:
    """Check if a video device is a ZED camera."""
    info = get_device_info(device)
    model = info.get("ID_MODEL", "").lower()
    vendor = info.get("ID_VENDOR_ID", "")
    return "zed" in model or vendor == "2b03"


def is_primary_video_node(device: str) -> bool:
    """Check if this is the primary video node (index 0), not a metadata node."""
    info = get_device_info(device)
    # ZED cameras create multiple /dev/video* nodes; the primary one has index 0
    devpath = info.get("DEVPATH", "")
    # Also check via v4l2 — primary nodes support VIDEO_CAPTURE
    try:
        output = subprocess.check_output(
            ["udevadm", "info", "--attribute-walk", "--name", device],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines():
            if 'ATTR{index}' in line and '"0"' in line:
                return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # Fallback: assume the lower-numbered device is primary
    return True


def verify_camera_opencv(device: str) -> bool:
    """Try to open the camera with OpenCV and read a frame."""
    try:
        import cv2
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"  Frame captured: {w}x{h}")
            if w > h * 1.5:
                print(f"  Side-by-side stereo detected (left image: {w//2}x{h})")
            return True
        return False
    except ImportError:
        print("  WARNING: OpenCV not available, skipping frame verification")
        return True
    except Exception as e:
        print(f"  Error during verification: {e}")
        return False


def wait_for_new_video_device(initial_devices: List[str], timeout: float = 30.0) -> Optional[str]:
    """Wait for a new /dev/video* device to appear. Returns the primary ZED node."""
    print(f"\nWaiting for you to plug in the USB cable...")
    print(f"(Timeout: {timeout} seconds)")

    start_time = time.time()
    while time.time() - start_time < timeout:
        current = get_video_devices()
        new_devices = [d for d in current if d not in initial_devices]

        if new_devices:
            time.sleep(1.0)  # Let all device nodes settle
            current = get_video_devices()
            new_devices = sorted([d for d in current if d not in initial_devices])

            # Find the primary ZED video node among new devices
            for dev in new_devices:
                if is_zed_device(dev) and is_primary_video_node(dev):
                    return dev
            # Fallback: return the first new device
            if new_devices:
                return new_devices[0]

        time.sleep(0.5)
        print(".", end="", flush=True)

    print()
    return None


def detect_camera_interactive(role: str, existing_ports: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """Interactive detection of one camera. Returns (device_path, usb_port_path) or None."""
    symlink = CAMERA_ROLES[role]

    print(f"\nPlease UNPLUG the {role} camera USB if it's connected.")
    input("Press Enter when ready...")

    initial_devices = get_video_devices()
    print(f"Current video devices: {initial_devices if initial_devices else '(none)'}")

    print(f"\nNow PLUG IN the {role} camera.")
    device = wait_for_new_video_device(initial_devices)

    if device is None:
        print(f"ERROR: No new video device detected for {role} camera.")
        return None

    print(f"\nDetected new device: {device}")

    # Get device info
    info = get_device_info(device)
    model = info.get("ID_MODEL", "unknown")
    serial = info.get("ID_SERIAL", "unknown")
    vendor_id = info.get("ID_VENDOR_ID", "")
    model_id = info.get("ID_MODEL_ID", "")
    print(f"  Model: {model}")
    print(f"  Serial: {serial}")
    print(f"  USB ID: {vendor_id}:{model_id}")

    # Get USB port path
    usb_port = get_usb_port_path(device)
    if usb_port:
        print(f"  USB port path: {usb_port}")
    else:
        print("  WARNING: Could not determine USB port path")

    # Check it's not already assigned
    for assigned_role, (assigned_dev, assigned_port, *_ids) in existing_ports.items():
        if usb_port and usb_port == assigned_port:
            print(f"ERROR: USB port {usb_port} already assigned to {assigned_role} camera.")
            return None

    # Verify with OpenCV
    print(f"\nVerifying camera with OpenCV...")
    if verify_camera_opencv(device):
        print(f"Camera verified successfully!")
    else:
        print(f"WARNING: Could not verify camera with OpenCV.")
        resp = input("  Use this device anyway? (y/n): ").strip().lower()
        if resp != "y":
            return None

    return (device, usb_port, vendor_id, model_id)


def generate_udev_rules(port_mapping: Dict[str, Tuple[str, str, str, str]]) -> str:
    """Generate udev rules content.

    Prefers matching by USB product ID (port-independent, survives replugs
    and controller re-enumeration). Falls back to the physical port path
    only when two cameras share the same model, or to a vendor-only match
    when neither is available.

    Args:
        port_mapping: {role: (device_path, usb_port_path, vendor_id, model_id)}
    """
    lines = [
        "# ZED camera symlinks (generated by setup_zed_cameras.py)",
        "# Matched by USB product ID where camera models differ (stable across",
        "# ports); by physical port path only when models are identical.",
        "",
    ]

    model_ids = [ids[3] for ids in port_mapping.values() if ids[3]]
    unique_models = len(set(model_ids)) == len(model_ids)

    for role, (device, usb_port, vendor_id, model_id) in port_mapping.items():
        symlink = CAMERA_ROLES[role].replace("/dev/", "")
        if vendor_id and model_id and unique_models:
            lines.append(f'# ZED {role.capitalize()} Camera ({vendor_id}:{model_id})')
            lines.append(
                f'SUBSYSTEM=="video4linux", ATTR{{index}}=="0", '
                f'ATTRS{{idVendor}}=="{vendor_id}", ATTRS{{idProduct}}=="{model_id}", '
                f'SYMLINK+="{symlink}", MODE="0666"'
            )
        elif usb_port:
            lines.append(f'# ZED {role.capitalize()} Camera (USB port: {usb_port})')
            lines.append(
                f'SUBSYSTEM=="video4linux", ATTR{{index}}=="0", '
                f'KERNELS=="{usb_port}", SYMLINK+="{symlink}", MODE="0666"'
            )
        else:
            lines.append(f"# ZED {role.capitalize()} Camera (WARNING: no USB port path, using vendor match)")
            lines.append(
                f'SUBSYSTEM=="video4linux", ATTR{{index}}=="0", '
                f'ATTRS{{idVendor}}=="2b03", SYMLINK+="{symlink}", MODE="0666"'
            )
        lines.append("")

    return "\n".join(lines)


def write_udev_rules(rules_content: str) -> Tuple[bool, List[str]]:
    """Write udev rules and reload. Returns (success, manual_commands)."""
    manual_commands = [
        f"sudo tee {UDEV_RULES_PATH} << 'RULES'\n{rules_content}\nRULES",
        "sudo udevadm control --reload-rules",
        "sudo udevadm trigger",
    ]

    # Try writing directly
    try:
        with open(UDEV_RULES_PATH, "w") as f:
            f.write(rules_content)
        print(f"Wrote udev rules to {UDEV_RULES_PATH}")

        # Reload udev
        subprocess.run(["udevadm", "control", "--reload-rules"], check=True)
        subprocess.run(["udevadm", "trigger"], check=True)
        print("Udev rules reloaded successfully.")
        return True, manual_commands
    except PermissionError:
        print(f"\nPermission denied writing to {UDEV_RULES_PATH}")
        print("Run these commands manually:")
        print()
        for cmd in manual_commands:
            print(f"  {cmd}")
        return False, manual_commands
    except Exception as e:
        print(f"\nError: {e}")
        return False, manual_commands


def check_prerequisites():
    """Check user is in video group and has necessary tools."""
    import grp
    try:
        video_gid = grp.getgrnam("video").gr_gid
        user_groups = os.getgroups()
        if video_gid not in user_groups:
            username = os.environ.get("USER", "user")
            print(f"WARNING: You are not in the 'video' group.")
            print(f"  Run: sudo usermod -aG video {username}")
            print(f"  Then log out and back in.")
            print()
            resp = input("Continue anyway? (y/n): ").strip().lower()
            if resp != "y":
                sys.exit(1)
    except KeyError:
        pass  # No video group on this system

    # Check udevadm
    if not any(
        os.path.isfile(os.path.join(d, "udevadm"))
        for d in os.environ.get("PATH", "").split(":")
    ):
        print("WARNING: udevadm not found. Udev rule creation may fail.")


def main():
    print("=" * 70)
    print("ZED Camera USB Setup Wizard")
    print("=" * 70)
    print()
    print("This script will help you set up stable /dev symlinks for ZED cameras.")
    print(f"  Wrist camera  -> {CAMERA_ROLES['wrist']}")
    print(f"  External camera -> {CAMERA_ROLES['external']}")
    print()

    check_prerequisites()

    # Check for existing video devices
    existing_devices = get_video_devices()
    print(f"Current video devices: {existing_devices if existing_devices else '(none)'}")
    print()

    port_mapping: Dict[str, Tuple[str, str, str, str]] = {}

    # Detect cameras
    for role in ["wrist", "external"]:
        print("=" * 70)
        print(f"STEP: {role.upper()} CAMERA ({CAMERA_ROLES[role]})")
        print("=" * 70)

        result = detect_camera_interactive(role, port_mapping)
        if result:
            port_mapping[role] = result
            print(f"\n{role.capitalize()} camera configured!")
        else:
            print(f"\nSkipping {role} camera.")
            resp = input("Continue with remaining cameras? (y/n): ").strip().lower()
            if resp != "y":
                break

    if not port_mapping:
        print("\nNo cameras detected. Exiting.")
        sys.exit(1)

    # Generate and write udev rules
    print("\n" + "=" * 70)
    print("CREATING UDEV RULES")
    print("=" * 70)

    rules = generate_udev_rules(port_mapping)
    print(f"\nGenerated rules:\n")
    print(rules)

    success, manual_commands = write_udev_rules(rules)

    # Summary
    print("\n" + "=" * 70)
    if success:
        print("Setup Complete!")
    else:
        print("Setup Complete (manual steps needed)")
    print("=" * 70)

    print(f"\nConfigured cameras:")
    for role, (device, usb_port, vendor_id, model_id) in port_mapping.items():
        print(
            f"  {role}: {device} (USB port: {usb_port}, "
            f"id: {vendor_id}:{model_id}) -> {CAMERA_ROLES[role]}"
        )

    if not success:
        print("\nRun these commands to finish setup:")
        for cmd in manual_commands:
            print(f"  {cmd}")

    # Verify symlinks exist
    print("\nVerifying symlinks:")
    for role in port_mapping:
        symlink = CAMERA_ROLES[role]
        if os.path.exists(symlink):
            target = os.path.realpath(symlink)
            print(f"  {symlink} -> {target}")
        else:
            print(f"  {symlink}: NOT FOUND (unplug and replug camera, or run udevadm trigger)")

    print(f"\nYour IIIL config uses these devices:")
    print(f"  iiil/configs/env/real_franka_joint.yaml")
    print(f"    cameras.wrist.device: {CAMERA_ROLES['wrist']}")
    print(f"    cameras.external.device: {CAMERA_ROLES['external']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

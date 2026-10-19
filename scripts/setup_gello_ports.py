#!/usr/bin/env python3
"""
Interactive script to detect and configure USB ports for Gello hardware.

This script walks you through:
1. Selecting your Gello configuration (A1 or R1Pro)
2. Detecting USB ports when you plug in motor groups
3. Counting motors to verify correct connection
4. Optionally setting up a capacitive touch sensor (ESP32-S3 + MPR121)
5. Creating /dev symlinks for all detected devices
"""

import sys
import time
import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dynamixel_sdk.port_handler import PortHandler
    from dynamixel_sdk.packet_handler import PacketHandler
except ImportError:
    print("ERROR: dynamixel_sdk not found. Please install it:")
    print("  pip install dynamixel-sdk")
    sys.exit(1)


# Expected motor counts for each configuration
EXPECTED_MOTORS = {
    "a1": {
        "large": 3,   # High power servos (joints 0-2)
        "small": 3,   # Low power servos (joints 3-5)
    },
    "r1pro": {
        "large": 8,   # High power servos: 4 per arm × 2 arms (joints 0-3 right, 7-10 left)
        "small": 6,   # Low power servos: 3 per arm × 2 arms (joints 4-6 right, 11-13 left)
    },
    "franka": {
        "large": 3,   # XL430 (12V) → Franka joints 0, 1, 3
        "small": 4,   # XL330 (5V) → Franka joints 2, 4, 5, 6
    },
}

# Symlink names per configuration
SYMLINK_NAMES = {
    "a1": {"large": "/dev/tty_large_motor", "small": "/dev/tty_small_motor", "touch": "/dev/tty_touch_sensor"},
    "r1pro": {"large": "/dev/tty_large_motor", "small": "/dev/tty_small_motor", "touch": "/dev/tty_touch_sensor"},
    "franka": {"large": "/dev/franka_large", "small": "/dev/franka_small", "touch": "/dev/tty_touch_sensor"},
}

# Motor ID ranges to scan (Dynamixel IDs typically 0-253)
SCAN_ID_RANGE = range(0, 254)


def get_available_ports() -> List[str]:
    """Get list of available USB serial ports."""
    ports = []
    
    # Linux ports
    ports.extend(glob.glob("/dev/ttyUSB*"))
    ports.extend(glob.glob("/dev/ttyACM*"))
    
    # macOS ports
    ports.extend(glob.glob("/dev/tty.usbserial-*"))
    ports.extend(glob.glob("/dev/cu.usbserial-*"))
    ports.extend(glob.glob("/dev/tty.usbmodem-*"))
    ports.extend(glob.glob("/dev/cu.usbmodem-*"))
    
    return sorted(ports)


def scan_motors_on_port(port: str, baudrate: int = 1000000) -> List[int]:
    """
    Scan for Dynamixel motors on a given port.
    
    Returns list of motor IDs found.
    """
    port_handler = PortHandler(port)
    packet_handler = PacketHandler(2.0)
    
    found_ids = []
    
    try:
        if not port_handler.openPort():
            # Retry once after a short wait (udev may not have applied permissions yet)
            print(f"  Port not ready, retrying in 2s...")
            time.sleep(2.0)
            if not port_handler.openPort():
                print(f"  Failed to open port {port}")
                print(f"  Try: sudo chmod 666 {port}")
                return []
        
        # Try common baudrates if default fails
        baudrates = [1000000, 57600, 115200, 2000000, 3000000]
        if baudrate not in baudrates:
            baudrates.insert(0, baudrate)
        
        success = False
        for baud in baudrates:
            if port_handler.setBaudRate(baud):
                print(f"  Trying baud rate: {baud}")
                # Scan for motors using ping
                # Limit scan to reasonable range to speed up
                for motor_id in range(0, 20):  # Scan IDs 0-19 (most common range)
                    model_number, comm_result, error = packet_handler.ping(port_handler, motor_id)
                    if comm_result == 0:  # SUCCESS
                        found_ids.append(motor_id)
                        print(f"    Found motor ID {motor_id} (model: {model_number})")
                if found_ids:
                    success = True
                    break
        
        if not success and not found_ids:
            print(f"  No motors found. Make sure motors are powered and baud rate is correct.")
        
        port_handler.closePort()
    except Exception as e:
        print(f"  Warning: Error scanning {port}: {e}")
        try:
            port_handler.closePort()
        except:
            pass
    
    return sorted(found_ids)


def wait_for_new_port(initial_ports: List[str], timeout: float = 30.0) -> Optional[str]:
    """
    Wait for a new USB port to appear.
    
    Returns the new port path, or None if timeout.
    """
    print(f"\nWaiting for you to plug in the USB cable...")
    print(f"(Timeout: {timeout} seconds)")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        current_ports = get_available_ports()
        new_ports = [p for p in current_ports if p not in initial_ports]
        
        if new_ports:
            # Wait for udev rules to apply permissions (MODE="0666")
            time.sleep(2.0)
            if new_ports[0] in get_available_ports():
                return new_ports[0]
        
        time.sleep(0.5)
        print(".", end="", flush=True)
    
    print("\n")
    return None


def identify_motor_group(motor_ids: List[int], config_type: str) -> Optional[str]:
    """
    Identify which motor group this is based on motor count.
    
    Returns: "large", "small", or None if unclear
    """
    expected = EXPECTED_MOTORS[config_type]
    count = len(motor_ids)
    
    if count == expected["large"]:
        return "large"
    elif count == expected["small"]:
        return "small"
    else:
        return None


def create_symlinks(port_mapping: Dict[str, str], config_type: str = "a1") -> Tuple[bool, List[str]]:
    """
    Create symlinks for detected ports.

    Symlink names depend on the configuration:
    - A1/R1Pro: /dev/tty_large_motor, /dev/tty_small_motor
    - Franka:   /dev/franka_large, /dev/franka_small

    Args:
        port_mapping: Dict mapping "large", "small", or "touch" to port path
        config_type: "a1", "r1pro", or "franka"

    Returns:
        Tuple of (success: bool, manual_commands: List[str])
    """
    import os

    symlinks = SYMLINK_NAMES.get(config_type, SYMLINK_NAMES["a1"])
    
    manual_commands = []
    created = []
    needs_sudo = os.geteuid() != 0
    
    if needs_sudo:
        print("\n⚠ NOTE: Creating symlinks in /dev requires root privileges.")
        print("Attempting to create symlinks... (will show manual commands if needed)")
        print()
    
    for group_type, symlink_path in symlinks.items():
        if group_type not in port_mapping:
            continue
        
        target = port_mapping[group_type]
        cmd = f"sudo ln -sf {target} {symlink_path}"
        manual_commands.append(cmd)
        
        # Remove existing symlink if it exists
        if os.path.islink(symlink_path):
            existing = os.path.realpath(symlink_path)
            if existing == target:
                print(f"  ✓ {symlink_path} already points to {target}")
                created.append(symlink_path)
                continue
            else:
                print(f"  Removing existing symlink: {symlink_path} -> {existing}")
                try:
                    os.unlink(symlink_path)
                except PermissionError:
                    print(f"  ✗ Permission denied. Run manually: sudo rm {symlink_path}")
                    manual_commands.insert(-1, f"sudo rm {symlink_path}")
                    continue
        elif os.path.exists(symlink_path):
            print(f"  ⚠ WARNING: {symlink_path} exists and is not a symlink.")
            response = input(f"  Remove it and create symlink? (y/n): ").strip().lower()
            if response == 'y':
                try:
                    os.remove(symlink_path)
                except PermissionError:
                    print(f"  ✗ Permission denied. Run manually: sudo rm {symlink_path}")
                    manual_commands.insert(-1, f"sudo rm {symlink_path}")
                    continue
            else:
                print(f"  Skipping {symlink_path}")
                continue
        
        # Create symlink
        try:
            os.symlink(target, symlink_path)
            print(f"  ✓ Created: {symlink_path} -> {target}")
            created.append(symlink_path)
        except PermissionError:
            print(f"  ✗ Permission denied creating {symlink_path}")
        except Exception as e:
            print(f"  ✗ Error creating {symlink_path}: {e}")
    
    success = len(created) == len([k for k in port_mapping.keys() if k in symlinks])
    return success, manual_commands


def verify_touch_sensor(port: str, baudrate: int = 115200, timeout: float = 3.0) -> bool:
    """
    Verify a port is sending touch sensor data.

    Reads serial lines for up to `timeout` seconds, looking for the
    expected "T<hex>" protocol from the ESP32-S3 capacitive sensor firmware.

    Returns True if valid touch data was received.
    """
    import serial

    try:
        ser = serial.Serial(port, baudrate, timeout=1.0)
    except Exception as e:
        print(f"  Could not open {port} at {baudrate} baud: {e}")
        return False

    start = time.time()
    valid_count = 0
    try:
        while time.time() - start < timeout:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            if len(line) >= 3 and line[0] == "T":
                try:
                    int(line[1:], 16)
                    valid_count += 1
                    if valid_count >= 3:
                        print(f"  Received valid touch data: {line}")
                        return True
                except ValueError:
                    pass
            elif len(line) >= 3 and line[0] == "E":
                print(f"  Sensor reporting error: {line}")
    except Exception as e:
        print(f"  Error reading from {port}: {e}")
    finally:
        ser.close()

    if valid_count > 0:
        print(f"  Got {valid_count} valid line(s) but fewer than expected.")
        return True
    return False


def setup_touch_sensor(port_mapping: Dict[str, str]) -> None:
    """
    Interactive touch sensor setup step.

    Detects the ESP32-S3 touch sensor port, verifies it sends valid data,
    and adds the 'touch' key to port_mapping for symlink creation.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Capacitive Touch Sensor (ESP32-S3 + MPR121)")
    print("=" * 70)
    print()

    response = input("Do you want to set up a capacitive touch sensor? (y/n): ").strip().lower()
    if response != "y":
        print("Skipping touch sensor setup.")
        return

    print()
    print("Please UNPLUG the touch sensor USB if it's connected.")
    print("Then plug it in when prompted.")
    input("Press Enter when ready...")

    initial_ports = get_available_ports()
    new_port = wait_for_new_port(initial_ports, timeout=30.0)

    if new_port is None:
        print("ERROR: No new port detected. Skipping touch sensor setup.")
        return

    # Make sure it's not one of the motor ports
    for group, motor_port in port_mapping.items():
        if new_port == motor_port:
            print(f"ERROR: Detected port {new_port} is already assigned to {group} motors.")
            print("Please use a different USB port for the touch sensor.")
            return

    print(f"\n✓ Detected new port: {new_port}")
    print("Verifying touch sensor data...")

    try:
        if verify_touch_sensor(new_port):
            print("✓ Touch sensor verified!")
        else:
            print("⚠ WARNING: Could not read valid touch data from this port.")
            print("  Make sure the ESP32-S3 firmware is flashed and the MPR121 is connected.")
            resp = input("  Use this port anyway? (y/n): ").strip().lower()
            if resp != "y":
                print("Skipping touch sensor.")
                return
    except ImportError:
        print("⚠ pyserial not installed -- skipping verification.")
        print(f"  Assuming {new_port} is the touch sensor.")

    port_mapping["touch"] = new_port
    print(f"✓ Touch sensor port: {new_port}")


def main():
    print("=" * 70)
    print("Gello USB Port Setup Wizard")
    print("=" * 70)
    print()

    # Step 1: Select configuration
    print("Select your Gello configuration:")
    print("  1. 6 DOF A1 (single arm)")
    print("  2. 7 DOF R1Pro (bilateral - two arms)")
    print("  3. 7 DOF Franka Joylo (single arm)")
    print()

    while True:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        if choice == "1":
            config_type = "a1"
            config_name = "gello_a1"
            break
        elif choice == "2":
            config_type = "r1pro"
            config_name = "gello_r1pro"
            break
        elif choice == "3":
            config_type = "franka"
            config_name = "gello_franka"
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    print(f"\nSelected: {config_type}")
    expected = EXPECTED_MOTORS[config_type]
    print(f"Expected motors:")
    print(f"  Large motor group (high power): {expected['large']} motors")
    print(f"  Small motor group (low power):  {expected['small']} motors")
    print()

    # Get initial port list
    initial_ports = get_available_ports()
    print(f"Currently connected ports: {initial_ports if initial_ports else '(none)'}")
    print()

    port_mapping = {}

    # Step 2: Detect large motor group
    print("=" * 70)
    print("STEP 1: Large Motor Group (High Power Servos)")
    print("=" * 70)
    print(f"Please UNPLUG the large motor group USB if it's connected.")
    input("Press Enter when ready...")

    # Wait for user to plug in
    initial_ports_step1 = get_available_ports()
    new_port = wait_for_new_port(initial_ports_step1, timeout=30.0)

    if new_port is None:
        print("ERROR: No new port detected. Please check your USB connection.")
        return

    print(f"\n✓ Detected new port: {new_port}")
    print("Scanning for motors...")

    motor_ids = scan_motors_on_port(new_port)

    if not motor_ids:
        print(f"ERROR: No motors found on {new_port}")
        print("Please check:")
        print("  - Motors are powered on")
        print("  - USB cable is properly connected")
        print("  - Baud rate matches (trying 1Mbps)")
        return

    print(f"✓ Found {len(motor_ids)} motors with IDs: {motor_ids}")

    group_type = identify_motor_group(motor_ids, config_type)
    if group_type == "large":
        print("✓ This matches the LARGE motor group!")
        port_mapping["large"] = new_port
    elif group_type == "small":
        print("⚠ WARNING: Motor count matches SMALL group, but you said this is LARGE.")
        print("  Continuing anyway...")
        port_mapping["large"] = new_port
    else:
        print(f"⚠ WARNING: Expected {expected['large']} motors, found {len(motor_ids)}")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return
        port_mapping["large"] = new_port

    # Step 3: Detect small motor group
    print("\n" + "=" * 70)
    print("STEP 2: Small Motor Group (Low Power Servos)")
    print("=" * 70)
    print(f"Please UNPLUG the small motor group USB if it's connected.")
    input("Press Enter when ready...")

    # Wait for user to plug in
    initial_ports_step2 = get_available_ports()
    new_port = wait_for_new_port(initial_ports_step2, timeout=30.0)

    if new_port is None:
        print("ERROR: No new port detected.")
        return

    if new_port == port_mapping.get("large"):
        print("ERROR: Same port as large motor group detected.")
        print("Please use a different USB port for the small motor group.")
        return

    print(f"\n✓ Detected new port: {new_port}")
    print("Scanning for motors...")

    motor_ids = scan_motors_on_port(new_port)

    if not motor_ids:
        print(f"ERROR: No motors found on {new_port}")
        return

    print(f"✓ Found {len(motor_ids)} motors with IDs: {motor_ids}")

    group_type = identify_motor_group(motor_ids, config_type)
    if group_type == "small":
        print("✓ This matches the SMALL motor group!")
        port_mapping["small"] = new_port
    elif group_type == "large":
        print("⚠ WARNING: Motor count matches LARGE group, but you said this is SMALL.")
        print("  Continuing anyway...")
        port_mapping["small"] = new_port
    else:
        print(f"⚠ WARNING: Expected {expected['small']} motors, found {len(motor_ids)}")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return
        port_mapping["small"] = new_port

    # Step 4: Touch sensor (optional)
    setup_touch_sensor(port_mapping)

    # Step 5: Create symlinks
    print("\n" + "=" * 70)
    step_num = "4" if "touch" in port_mapping else "3"
    print(f"STEP {step_num}: Create USB Port Symlinks")
    print("=" * 70)
    print(f"\nDetected ports:")
    print(f"  Large motor group: {port_mapping.get('large', 'NOT FOUND')}")
    print(f"  Small motor group: {port_mapping.get('small', 'NOT FOUND')}")
    if "touch" in port_mapping:
        print(f"  Touch sensor:      {port_mapping['touch']}")
    print()
    symlink_names = SYMLINK_NAMES.get(config_type, SYMLINK_NAMES["a1"])
    print("Creating symlinks:")
    print(f"  {symlink_names['large']} -> (large motor port)")
    print(f"  {symlink_names['small']} -> (small motor port)")
    if "touch" in port_mapping:
        print(f"  {symlink_names['touch']} -> (touch sensor port)")
    print()

    success, manual_commands = create_symlinks(port_mapping, config_type)

    print("\n" + "=" * 70)
    if success:
        print("✓ Setup Complete!")
        print("=" * 70)
        print(f"\nSymlinks created successfully!")
    else:
        print("⚠ Setup Complete (with manual steps needed)")
        print("=" * 70)
        print(f"\nSome symlinks could not be created automatically.")
        print("Run these commands manually (requires sudo):")
        print()
        for cmd in manual_commands:
            print(f"  {cmd}")
        print()

    print("Your Gello config files use these symlinks:")
    print(f"  {symlink_names['large']} (high power servos)")
    print(f"  {symlink_names['small']} (low power servos)")
    if "touch" in port_mapping:
        print(f"  {symlink_names['touch']} (capacitive touch sensor)")
    print("\nYou can now test your setup with:")
    print(f"  python scripts/read_gello_joints.py --config {config_name}")

    if not success and manual_commands:
        print("\n" + "=" * 70)
        print("Quick copy-paste commands:")
        print("=" * 70)
        for cmd in manual_commands:
            print(cmd)


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


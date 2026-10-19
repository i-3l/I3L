#!/bin/bash
#
# Recover FrankaJoylo Dynamixel USB adapters that dropped off the USB bus.
#
# The FTDI adapters (/dev/franka_large FT9BTJY0, /dev/franka_small FT8J0W3N)
# can disappear when their xHCI controller runtime-suspends (2026-08-21:
# FT9BTJY0 and a ZED-M vanished together from controller 0000:15:00.3 hours
# after boot). Rebinding xHCI controllers that currently have no attached
# devices forces re-enumeration and restores the udev symlinks, without
# touching controllers that host live devices.
#
# Usage: scripts/recover_franka_usb.sh   (self-elevates with sudo)
# Exits 0 if all ports are present (or recovered), 1 if still missing.

set -uo pipefail

PORTS=(/dev/franka_large /dev/franka_small)

missing_ports() {
    local p
    for p in "${PORTS[@]}"; do
        [[ -e "$p" ]] || echo "$p"
    done
}

list_ports() {
    local p
    for p in "${PORTS[@]}"; do
        [[ -e "$p" ]] && echo "  $p -> $(readlink -f "$p")"
    done
}

if [[ -z "$(missing_ports)" ]]; then
    echo "All FrankaJoylo ports present:"
    list_ports
    exit 0
fi

if [[ $EUID -ne 0 ]]; then
    exec sudo "$0" "$@"
fi

echo "Missing:"
missing_ports | sed 's/^/  /'
echo "Rebinding idle xHCI controllers to force USB re-enumeration..."

DRIVER=/sys/bus/pci/drivers/xhci_hcd
rebound=0
for ctrl in "$DRIVER"/0000:*; do
    [[ -e "$ctrl" ]] || continue
    pci=$(basename "$ctrl")
    # Count devices attached to this controller's USB buses (root hubs and
    # interface nodes like "7-0:1.0" excluded).
    attached=0
    for bus in "$ctrl"/usb*; do
        [[ -d "$bus" ]] || continue
        busnum=$(basename "$bus")
        busnum=${busnum#usb}
        count=$(find /sys/bus/usb/devices/ -maxdepth 1 -name "${busnum}-[0-9]*" ! -name "*:*" | wc -l)
        attached=$((attached + count))
    done
    if [[ $attached -eq 0 ]]; then
        echo "  Rebinding $pci (no devices attached)"
        echo "$pci" > "$DRIVER/unbind"
        sleep 1
        echo "$pci" > "$DRIVER/bind"
        rebound=$((rebound + 1))
    else
        echo "  Skipping $pci ($attached device(s) attached)"
    fi
done

if [[ $rebound -eq 0 ]]; then
    echo "No idle controllers to rebind."
fi

sleep 3
udevadm settle --timeout=5 2>/dev/null || true

still_missing=$(missing_ports)
if [[ -z "$still_missing" ]]; then
    echo "Recovered:"
    list_ports
    exit 0
fi

echo "Still missing after re-enumeration:"
echo "$still_missing" | sed 's/^/  /'
echo "Check the adapter's USB cable and power, then re-run this script."
exit 1

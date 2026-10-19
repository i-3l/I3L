#!/usr/bin/env bash
# Guided Bluetooth pairing for the single Right Joy-Con used by Franka.
#
# This intentionally follows the known-working calder launcher flow:
# remove stale Joy-Con records, hold a full classic-Bluetooth inquiry open,
# then pair, trust, and connect with bluetoothctl.

set -uo pipefail

if [ -t 1 ]; then
    C_RED=$'\033[31m' C_GRN=$'\033[32m' C_YLW=$'\033[33m' C_CYN=$'\033[36m'
    C_BLD=$'\033[1m' C_OFF=$'\033[0m'
else
    C_RED='' C_GRN='' C_YLW='' C_CYN='' C_BLD='' C_OFF=''
fi

say()  { printf '%s\n' "$*"; }
ok()   { printf '%s[ok]%s %s\n' "$C_GRN" "$C_OFF" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$C_YLW" "$C_OFF" "$*"; }
err()  { printf '%s[err]%s %s\n' "$C_RED" "$C_OFF" "$*" >&2; }
hdr()  { printf '\n%s%s%s\n' "$C_BLD$C_CYN" "== $* ==" "$C_OFF"; }

right_joycon_hidraw() {
    local mac=${1,,} device_dir uevent uniq hidraw node

    for uevent in /sys/bus/hid/devices/*/uevent; do
        [ -r "$uevent" ] || continue
        grep -qx 'HID_ID=0005:0000057E:00002007' "$uevent" || continue

        uniq=$(awk -F= '$1 == "HID_UNIQ" {print tolower($2); exit}' "$uevent")
        [ "$uniq" = "$mac" ] || continue

        device_dir=${uevent%/uevent}
        for hidraw in "$device_dir"/hidraw/hidraw*; do
            [ -e "$hidraw" ] || continue
            node="/dev/${hidraw##*/}"
            if [ -r "$node" ] && [ -w "$node" ]; then
                printf '%s\n' "$node"
                return 0
            fi
        done
    done
    return 1
}

usage() {
    say "Usage: ./connect.sh"
    say ""
    say "Remove saved Joy-Con pairings, then pair Franka's blue Right Joy-Con."
}

case "${1:-}" in
    ""|--pair|--select) ;;
    -h|--help) usage; exit 0 ;;
    *) err "unknown option: $1"; usage >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { err "too many arguments"; usage >&2; exit 2; }

command -v bluetoothctl >/dev/null 2>&1 || {
    err "bluetoothctl not found — install bluez"
    exit 1
}
if ! bluetoothctl show >/dev/null 2>&1; then
    err "no Bluetooth adapter available (rfkill blocked? adapter missing?)"
    exit 1
fi
bluetoothctl power on >/dev/null 2>&1 || {
    err "could not power on the Bluetooth adapter"
    exit 1
}

if [ ! -t 0 ]; then
    err "pairing requires an interactive terminal"
    exit 1
fi

adapter=$(bluetoothctl show 2>/dev/null | awk '/^Controller / {print $2; exit}')
hdr "Pair Franka's blue Right Joy-Con"
[ -z "$adapter" ] || say "Bluetooth adapter: $adapter"

# Clear every cached Joy-Con first. A controller re-synced to another host can
# leave a paired/trusted record here whose old link key no longer works.
if ! device_output=$(bluetoothctl devices 2>/dev/null); then
    err "could not list saved Bluetooth devices"
    exit 1
fi
old_joycon_output=$(printf '%s\n' "$device_output" \
    | awk '$1 == "Device" && index($0, "Joy-Con (") {print $2}' \
    | awk '!seen[$0]++')
old_joycons=()
if [ -n "$old_joycon_output" ]; then
    mapfile -t old_joycons <<<"$old_joycon_output"
fi

if [ "${#old_joycons[@]}" -gt 0 ]; then
    warn "this will remove ${#old_joycons[@]} saved Joy-Con Bluetooth entries"
    printf 'Continue? [Y/n] '
    IFS= read -r answer
    case "$answer" in
        n|N|no|NO|No) say "No pairings were changed."; exit 1 ;;
    esac

    for mac in "${old_joycons[@]}"; do
        bluetoothctl disconnect "$mac" >/dev/null 2>&1 || true
        if bluetoothctl remove "$mac" >/dev/null 2>&1; then
            ok "removed Joy-Con $mac"
        else
            err "failed to remove Joy-Con $mac"
            exit 1
        fi
    done
else
    ok "no saved Joy-Con entries to remove"
fi

name='Joy-Con (R)'
say ""
say "${C_BLD}Hold the small SYNC button on the blue $name${C_OFF}"
say "${C_BLD}(on its rail, between SL and SR) until all four LEDs sweep.${C_OFF}"
say "${C_BLD}Then release SYNC and press Enter here.${C_OFF}"
IFS= read -r _

mac=""
tries=0
while [ -z "$mac" ] && [ "$tries" -lt 5 ]; do
    # Joy-Cons are CLASSIC Bluetooth. A complete inquiry takes roughly 10–13
    # seconds, so keep each discovery round open for the full 15 seconds.
    bluetoothctl --timeout 15 scan on >/dev/null 2>&1 || true

    if ! device_output=$(bluetoothctl devices 2>/dev/null); then
        err "Bluetooth discovery completed, but its device list could not be read"
        exit 1
    fi
    candidates=$(printf '%s\n' "$device_output" \
        | awk -v n="$name" '$1 == "Device" && index($0, n) {print $2}' \
        | awk '!seen[$0]++')
    count=$(printf '%s\n' "$candidates" | awk 'NF {count++} END {print count+0}')

    if [ "$count" -eq 1 ]; then
        mac=$candidates
    elif [ "$count" -gt 1 ]; then
        err "$count devices answer to \"$name\""
        say "Keep every other Joy-Con asleep, then rerun ./connect.sh."
        say "Candidates:"
        printf '  %s\n' $candidates
        exit 1
    fi

    tries=$((tries + 1))
    if [ -z "$mac" ] && [ "$tries" -lt 5 ]; then
        say "  scanning... (round $tries/5; press SYNC again if the LEDs stopped sweeping)"
    fi
done

if [ -z "$mac" ]; then
    err "$name never appeared"
    say "Charge it if needed, hold SYNC until all four LEDs sweep, then rerun ./connect.sh."
    exit 1
fi

say "Found $name at $mac — pairing..."
if ! bluetoothctl pair "$mac" >/dev/null 2>&1; then
    err "pairing $name ($mac) failed"
    say "Keep its LEDs sweeping for the whole attempt, then rerun ./connect.sh."
    exit 1
fi
if ! bluetoothctl trust "$mac" >/dev/null 2>&1; then
    err "$name paired, but trusting $mac failed"
    exit 1
fi

# Do not use Device1.Connected as a reason to skip this call. Immediately
# after pairing, BlueZ reports the temporary pairing ACL as connected even
# though the HID input profile has not been opened yet. The explicit connect
# below is the step that requests that profile.
tries=0
hidraw=""
while [ "$tries" -lt 10 ] && [ -z "$hidraw" ]; do
    bluetoothctl connect "$mac" >/dev/null 2>&1 || true
    tries=$((tries + 1))
    hidraw=$(right_joycon_hidraw "$mac" || true)
    if [ -z "$hidraw" ] && [ "$tries" -lt 10 ]; then
        sleep 1
    fi
done

# The final Connect reply can arrive just before HID and udev finish creating
# the device node, so give that last attempt the same settling time.
if [ -z "$hidraw" ]; then
    for _ in {1..10}; do
        sleep 0.5
        hidraw=$(right_joycon_hidraw "$mac" || true)
        [ -z "$hidraw" ] || break
    done
fi

info=$(bluetoothctl info "$mac" 2>/dev/null || true)
if ! grep -q 'Paired: yes' <<<"$info" \
   || ! grep -q 'Trusted: yes' <<<"$info" \
   || ! grep -q 'Connected: yes' <<<"$info"; then
    err "$name did not finish pairing/trusting/connecting"
    say "Press a button while rerunning ./connect.sh."
    exit 1
fi

if [ -z "$hidraw" ]; then
    err "Bluetooth connected, but the blue Right Joy-Con is not available to Franka"
    say "No readable HID device appeared for $mac after 10 explicit connect attempts."
    say "Press SYNC and rerun ./connect.sh."
    exit 1
fi

ok "$name paired, trusted, connected, and readable at $hidraw ($mac)"
say "Franka only uses this controller: R/ZR drive the gripper; A/B/X/Y control trials."

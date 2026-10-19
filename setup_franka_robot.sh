#!/bin/bash
#
# Franka Robot Setup — Pre-Flight Check & Guided Startup
#
# Checks everything needed before ./run.sh 10 or 11, and offers
# to start services that aren't running (auto_arm on NUC, etc.).
#
# Usage: ./setup_franka_robot.sh
#
set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

IIIL_DIR="$(cd "$(dirname "$0")" && pwd)"
NUC_IP="${NUC_IP:-172.16.0.3}"
NUC_USER="${NUC_USER:-franka3}"
NUC_TMUX_SESSION="franka"
FRANKA_IP="${FRANKA_IP:-172.16.0.2}"
DEOXYS_CONFIG="${DEOXYS_CONFIG:-config/charmander.yml}"  # deoxys robot config on the NUC
FRANKA_DESK_URL="https://${FRANKA_IP}"
FRANKA_DESK_LEGACY_URL="http://${FRANKA_IP}/desk"

PASS=0
WARN=0
FAIL=0

pass() { echo -e "  ${GREEN}✓${NC} $1"; PASS=$((PASS + 1)); }
warn() { echo -e "  ${YELLOW}!${NC} $1"; WARN=$((WARN + 1)); }
fail() { echo -e "  ${RED}✗${NC} $1"; FAIL=$((FAIL + 1)); }

ask_yn() {
    # ask_yn "prompt" → returns 0 for yes, 1 for no
    local prompt="$1" answer=""
    while true; do
        if ! read -rp "  $prompt [y/n]: " answer; then
            echo ""
            return 1
        fi
        case "$answer" in
            [Yy]*) return 0 ;;
            [Nn]*) return 1 ;;
            *) echo "  Please answer y or n." ;;
        esac
    done
}

# Helper: check if we can SSH to NUC
NUC_SSH_OK=false
check_nuc_ssh() {
    if $NUC_SSH_OK; then return 0; fi
    if ssh -o BatchMode=yes -o ConnectTimeout=3 "${NUC_USER}@${NUC_IP}" "true" 2>/dev/null; then
        NUC_SSH_OK=true
        return 0
    fi
    return 1
}

# Franka Control may ignore ICMP even while its web interface is available, so
# check the service operators actually need and use ping only as a fallback.
FRANKA_DESK_HTTP_OK=false
check_franka_control_box() {
    local url

    if command -v curl >/dev/null 2>&1; then
        for url in "$FRANKA_DESK_URL" "$FRANKA_DESK_LEGACY_URL"; do
            if curl --insecure --silent --show-error --output /dev/null \
                --noproxy '*' --connect-timeout 2 --max-time 4 "$url" 2>/dev/null; then
                FRANKA_DESK_HTTP_OK=true
                return 0
            fi
        done
    fi

    ping -c 1 -W 2 "$FRANKA_IP" >/dev/null 2>&1
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Franka Robot Pre-Flight Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ── 1. Conda environment ─────────────────────────────────────────
echo -e "${BLUE}[1/7] Conda Environment${NC}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
    fail "No conda env active. Run: conda activate behavior"
else
    pass "Conda env: $CONDA_DEFAULT_ENV"
fi

# ── 2. Python dependencies ────────────────────────────────────────
echo ""
echo -e "${BLUE}[2/7] Python Dependencies${NC}"

missing_deps=()
for mod in deoxys franka_joylo dynamixel_sdk cv2 torch h5py; do
    if python -c "import $mod" 2>/dev/null; then
        pass "$mod"
    else
        echo -e "  ${YELLOW}!${NC} $mod not importable"
        missing_deps+=("$mod")
    fi
done

if [[ ${#missing_deps[@]} -gt 0 ]]; then
    if ask_yn "Run ./setup_franka.sh to install missing dependencies?"; then
        echo ""
        "$IIIL_DIR/setup_franka.sh"
        echo ""
    fi

    # Count only the final state. A dependency repaired above must not leave a
    # stale failure that makes this pre-flight exit unsuccessfully.
    for mod in "${missing_deps[@]}"; do
        if python -c "import $mod" 2>/dev/null; then
            pass "$mod now importable"
        else
            fail "$mod is still not importable"
        fi
    done
fi

# ── 3. Joylo USB ports ───────────────────────────────────────────
echo ""
echo -e "${BLUE}[3/7] Joylo Motor Ports${NC}"

missing_ports=()
for port in /dev/franka_small /dev/franka_large; do
    if [[ -e "$port" ]]; then
        target=$(readlink -f "$port")
        pass "$port -> $target"
    else
        echo -e "  ${YELLOW}!${NC} $port not found"
        missing_ports+=("$port")
    fi
done

if [[ ${#missing_ports[@]} -gt 0 ]]; then
    # Adapters can drop off a runtime-suspended xHCI controller (2026-08-21:
    # FT9BTJY0 + a ZED-M vanished together). Rebinding idle controllers
    # re-enumerates them without a replug; the script self-elevates via sudo.
    echo -e "  ${YELLOW}Attempting USB re-enumeration (scripts/recover_franka_usb.sh)...${NC}"
    "$IIIL_DIR/scripts/recover_franka_usb.sh" || true

    still_missing=()
    for port in "${missing_ports[@]}"; do
        if [[ -e "$port" ]]; then
            target=$(readlink -f "$port")
            pass "$port recovered -> $target"
        else
            still_missing+=("$port")
        fi
    done
    missing_ports=("${still_missing[@]}")
fi

if [[ ${#missing_ports[@]} -gt 0 ]]; then
    FRANKA_JOYLO_DIR="${HOME}/FrankaJoylo"
    if [[ -d "$FRANKA_JOYLO_DIR" ]]; then
        if ask_yn "Run FrankaJoylo setup to configure motor ports?"; then
            echo ""
            (cd "$FRANKA_JOYLO_DIR" && ./setup.sh)
            echo ""
        fi
    else
        echo -e "       ${YELLOW}FrankaJoylo not found at $FRANKA_JOYLO_DIR${NC}"
    fi

    for port in "${missing_ports[@]}"; do
        if [[ -e "$port" ]]; then
            target=$(readlink -f "$port")
            pass "$port -> $target"
        else
            fail "$port is still not available"
        fi
    done
fi

# ── 4. ZED cameras ───────────────────────────────────────────────
echo ""
echo -e "${BLUE}[4/7] ZED Cameras${NC}"

missing_cameras=()
for cam in /dev/zed_wrist /dev/zed_external; do
    if [[ -e "$cam" ]]; then
        target=$(readlink -f "$cam")
        pass "$cam -> $target"
    else
        echo -e "  ${YELLOW}!${NC} $cam not found"
        missing_cameras+=("$cam")
    fi
done

if [[ ${#missing_cameras[@]} -gt 0 ]]; then
    # Distinguish "camera unplugged / udev rule missing" from "USB3 link dead".
    # The wrist ZED-M exposes its IMU over USB2 (2b03:f681) and its camera
    # over USB3 (2b03:f682). IMU present + camera absent means the USB3 pairs
    # in the cable/connector are not training — no rescan or wizard fixes
    # that; the cable must be reseated or replaced (seen 2026-08-21).
    if lsusb -d 2b03:f681 >/dev/null 2>&1 && ! lsusb -d 2b03:f682 >/dev/null 2>&1; then
        fail "wrist ZED-M USB3 link is down (its USB2 IMU is visible, but the camera is not enumerating)"
        echo -e "       ${YELLOW}Reseat or replace the wrist camera's USB3 cable (both ends).${NC}"
        echo -e "       ${YELLOW}The plug is seated (USB2 pins work); the USB3 pairs are not connecting.${NC}"
    fi

    if ask_yn "Run ZED camera setup wizard?"; then
        echo ""
        python "$IIIL_DIR/scripts/setup_zed_cameras.py"
        echo ""
    fi

    for cam in "${missing_cameras[@]}"; do
        if [[ -e "$cam" ]]; then
            target=$(readlink -f "$cam")
            pass "$cam -> $target"
        else
            fail "$cam is still not available"
        fi
    done
fi

# Quick OpenCV capture test for available cameras
for cam in /dev/zed_wrist /dev/zed_external; do
    if [[ -e "$cam" ]]; then
        result=$(python -c "
import cv2, sys
cap = cv2.VideoCapture('$cam', cv2.CAP_V4L2)
if not cap.isOpened():
    print('FAIL:cannot open')
    sys.exit(1)
ret, frame = cap.read()
cap.release()
if ret and frame is not None:
    h, w = frame.shape[:2]
    print(f'OK:{w}x{h}')
else:
    print('FAIL:no frame')
    sys.exit(1)
" 2>/dev/null) || result="FAIL:exception"
        cam_name=$(basename "$cam")
        if [[ "$result" == OK:* ]]; then
            res="${result#OK:}"
            pass "$cam_name capture OK ($res)"
        else
            fail "$cam_name capture failed (${result#FAIL:})"
        fi
    fi
done

# ── 5. Network: NUC ──────────────────────────────────────────────
echo ""
echo -e "${BLUE}[5/7] NUC (${NUC_IP})${NC}"

if ping -c 1 -W 2 "$NUC_IP" &>/dev/null; then
    pass "NUC reachable at $NUC_IP"

    if check_nuc_ssh; then
        pass "SSH to NUC OK"

        # Match the script process name exactly. `pgrep -f` also matches the
        # remote shell running the check and can falsely report success.
        arm_count=$(ssh -o BatchMode=yes -o ConnectTimeout=3 "${NUC_USER}@${NUC_IP}" \
            "pgrep -xc auto_arm.sh || true" 2>/dev/null) || arm_count=""

        if [[ "$arm_count" == "0" ]]; then
            echo -e "  ${YELLOW}!${NC} auto_arm.sh is not running on NUC"
            if ask_yn "Start auto_arm.sh on NUC?"; then
                echo -e "  ${YELLOW}Starting auto_arm.sh in tmux session '${NUC_TMUX_SESSION}' on NUC...${NC}"
                # Create or reuse tmux session on NUC and launch auto_arm.sh
                ssh -o BatchMode=yes -o ConnectTimeout=5 "${NUC_USER}@${NUC_IP}" \
                    "tmux has-session -t ${NUC_TMUX_SESSION} 2>/dev/null || tmux new-session -d -s ${NUC_TMUX_SESSION}" 2>/dev/null
                ssh -o BatchMode=yes -o ConnectTimeout=5 "${NUC_USER}@${NUC_IP}" \
                    "tmux send-keys -t ${NUC_TMUX_SESSION} 'cd ~/deoxys_control/deoxys && ./auto_scripts/auto_arm.sh ${DEOXYS_CONFIG}' Enter" 2>/dev/null

                sleep 3
            fi

            arm_count=$(ssh -o BatchMode=yes -o ConnectTimeout=3 "${NUC_USER}@${NUC_IP}" \
                "pgrep -xc auto_arm.sh || true" 2>/dev/null) || arm_count=""
        fi

        if [[ "$arm_count" == "1" ]]; then
            pass "one auto_arm.sh service is running on NUC"
        elif [[ "$arm_count" =~ ^[0-9]+$ ]] && ((arm_count > 1)); then
            fail "$arm_count auto_arm.sh services are running on NUC; exactly one is required"
            echo -e "       ${YELLOW}Inspect: ssh ${NUC_USER}@${NUC_IP} 'ps -C auto_arm.sh -o pid,ppid,args'${NC}"
        else
            fail "auto_arm.sh is not running on NUC"
            echo -e "       ${YELLOW}Attach with: ssh ${NUC_USER}@${NUC_IP} -t 'tmux attach -t ${NUC_TMUX_SESSION}'${NC}"
        fi

        # I3L drives the physical gripper from the Right Joy-Con, so this
        # service and its state stream are required for real collection.
        gripper_count=$(ssh -o BatchMode=yes -o ConnectTimeout=3 "${NUC_USER}@${NUC_IP}" \
            "pgrep -xc auto_gripper.sh || true" 2>/dev/null) || gripper_count=""

        if [[ "$gripper_count" == "0" ]]; then
            echo -e "  ${YELLOW}!${NC} auto_gripper.sh is not running on NUC"
            if ask_yn "Start auto_gripper.sh on NUC?"; then
                echo -e "  ${YELLOW}Starting auto_gripper.sh on NUC...${NC}"
                ssh -o BatchMode=yes -o ConnectTimeout=5 "${NUC_USER}@${NUC_IP}" \
                    "tmux has-session -t ${NUC_TMUX_SESSION} 2>/dev/null || tmux new-session -d -s ${NUC_TMUX_SESSION}" 2>/dev/null
                # Create a new window for the gripper
                ssh -o BatchMode=yes -o ConnectTimeout=5 "${NUC_USER}@${NUC_IP}" \
                    "tmux new-window -t ${NUC_TMUX_SESSION} -n gripper 2>/dev/null || true" 2>/dev/null
                ssh -o BatchMode=yes -o ConnectTimeout=5 "${NUC_USER}@${NUC_IP}" \
                    "tmux send-keys -t ${NUC_TMUX_SESSION}:gripper 'cd ~/deoxys_control/deoxys && ./auto_scripts/auto_gripper.sh ${DEOXYS_CONFIG}' Enter" 2>/dev/null
                sleep 3
            fi

            gripper_count=$(ssh -o BatchMode=yes -o ConnectTimeout=3 "${NUC_USER}@${NUC_IP}" \
                "pgrep -xc auto_gripper.sh || true" 2>/dev/null) || gripper_count=""
        fi

        if [[ "$gripper_count" == "1" ]]; then
            pass "one auto_gripper.sh service is running on NUC"
        elif [[ "$gripper_count" =~ ^[0-9]+$ ]] && ((gripper_count > 1)); then
            fail "$gripper_count auto_gripper.sh services are running on NUC; exactly one is required"
            echo -e "       ${YELLOW}Inspect: ssh ${NUC_USER}@${NUC_IP} 'ps -C auto_gripper.sh -o pid,ppid,args'${NC}"
        else
            fail "auto_gripper.sh is not running on NUC"
        fi
    else
        warn "Cannot SSH to NUC (key auth not set up)"
        echo -e "       ${YELLOW}Set up SSH key: ssh-copy-id ${NUC_USER}@${NUC_IP}${NC}"
        if ask_yn "Set up SSH key auth to NUC now?"; then
            ssh-copy-id -o StrictHostKeyChecking=accept-new "${NUC_USER}@${NUC_IP}"
            if check_nuc_ssh; then
                pass "SSH key auth to NUC configured"
                echo -e "  ${YELLOW}Re-run this script to check/start NUC services.${NC}"
            else
                fail "SSH key auth setup failed"
            fi
        fi
    fi
else
    fail "NUC not reachable at $NUC_IP"
    echo -e "       ${YELLOW}Check ethernet connection and NUC power.${NC}"
    echo -e "       ${YELLOW}Workstation should be 172.16.0.1, NUC should be 172.16.0.3${NC}"
fi

# ── 6. Franka Desk ───────────────────────────────────────────────
echo ""
echo -e "${BLUE}[6/7] Franka Control Box & Desk (${FRANKA_IP})${NC}"
echo -e "  Dashboard: ${BLUE}${FRANKA_DESK_URL}${NC}"
echo -e "  Legacy Desk fallback: ${BLUE}${FRANKA_DESK_LEGACY_URL}${NC}"
echo -e "  ${YELLOW}A browser certificate warning for the direct HTTPS address is expected.${NC}"

FRANKA_DESK_CONFIRMED=false
if check_franka_control_box; then
    if $FRANKA_DESK_HTTP_OK; then
        pass "Franka Desk web service reachable at $FRANKA_IP"
    else
        pass "Franka control box responds at $FRANKA_IP"
        warn "The Desk web probe did not answer yet; it may still be starting"
    fi

    if command -v xdg-open >/dev/null 2>&1 \
       && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] \
       && ask_yn "Open Franka Desk in this workstation's browser?"; then
        nohup xdg-open "$FRANKA_DESK_URL" >/dev/null 2>&1 &
    fi

    echo ""
    echo -e "  ${YELLOW}In Franka Desk, with the workcell clear and E-stop available:${NC}"
    echo -e "  ${YELLOW}  1. Unlock the joints (LED should turn blue).${NC}"
    echo -e "  ${YELLOW}  2. Open the sidebar menu and activate FCI (LED should turn green).${NC}"
    echo ""
    if ask_yn "Have you unlocked joints and activated FCI?"; then
        pass "Franka Desk confirmed ready"
        FRANKA_DESK_CONFIRMED=true
    else
        fail "Franka Desk is not ready; unlock the joints and activate FCI"
    fi
else
    fail "Franka control box not reachable at $FRANKA_IP"
    echo -e "       ${YELLOW}The workstation route is 172.16.0.1 -> 172.16.0.2.${NC}"
    echo -e "       ${YELLOW}Check control-box power, boot state, and its Ethernet cable/switch port.${NC}"
fi

# ── 7. End-to-end control path ────────────────────────────────────
echo ""
echo -e "${BLUE}[7/7] Franka State Stream (workstation -> NUC -> robot)${NC}"

if $FRANKA_DESK_CONFIRMED; then
    if timeout 15s python -c "
from franka_joylo.deoxys_franka import DeoxysFrankaInterface
import time

iface = None
try:
    iface = DeoxysFrankaInterface(control_freq=100)
    deadline = time.time() + 10.0
    while (iface.robot_interface.state_buffer_size == 0 or
           iface.robot_interface.gripper_state_buffer_size == 0):
        if time.time() >= deadline:
            raise TimeoutError('arm or gripper state missing from the NUC')
        time.sleep(0.1)
    q = iface.robot_interface.last_q
    print('  Joint positions: [' + ', '.join(f'{value:.3f}' for value in q) + ']')
    print(f'  Gripper width: {iface.robot_interface.last_gripper_q}')
finally:
    if iface is not None:
        iface.robot_interface.close()
" 2>/dev/null; then
        pass "Franka arm and gripper state streams are live"
    else
        fail "Franka arm/gripper state was not received through Deoxys"
        echo -e "       ${YELLOW}A wrapper may be running while its interface child is failing.${NC}"
        echo -e "       ${YELLOW}Check: ssh ${NUC_USER}@${NUC_IP} -t 'tmux attach -t ${NUC_TMUX_SESSION}'${NC}"
    fi
else
    echo -e "  ${YELLOW}!${NC} Skipped until Franka Desk is reachable and FCI is enabled"
fi

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  ${GREEN}✓ $PASS passed${NC}    ${YELLOW}! $WARN warnings${NC}    ${RED}✗ $FAIL failed${NC}"
echo ""

if [[ $FAIL -gt 0 ]]; then
    echo -e "${RED}Fix the failures above before running ./run.sh 10 or 11.${NC}"
    exit 1
elif [[ $WARN -gt 0 ]]; then
    echo -e "${YELLOW}Warnings present — data collection may work with limited features.${NC}"
    echo -e "${GREEN}Ready for: ./run.sh 10  (data collection) or ./run.sh 11  (correction)${NC}"
    exit 0
else
    echo -e "${GREEN}All checks passed! Ready for:${NC}"
    echo -e "${GREEN}  ./run.sh 10  (Franka data collection)${NC}"
    echo -e "${GREEN}  ./run.sh 11  (Franka policy correction)${NC}"
    exit 0
fi

#!/usr/bin/env bash
#
# Flash capacitive touch sensor firmware to LilyGo T-Display ESP32-S3.
# Uses arduino-cli (installed automatically if missing).
#
# Usage:
#   ./flash.sh              # auto-detect port
#   ./flash.sh /dev/ttyACM0 # explicit port
#   ./flash.sh --install    # just install deps, don't flash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKETCH_DIR="$SCRIPT_DIR/iiil/gello/capacitive_sensor"
SKETCH="$SKETCH_DIR/capacitive_sensor.ino"

BOARD_FQBN="esp32:esp32:esp32s3"
BOARD_FLAGS="CDCOnBoot=cdc,FlashSize=16M,PSRAM=opi"
ESP32_CORE_URL="https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json"
MPR121_LIB="Adafruit MPR121"
GFX_LIB="GFX Library for Arduino"

# ---------- helpers ----------

info()  { echo -e "\033[1;34m==>\033[0m $*"; }
ok()    { echo -e "\033[1;32m==>\033[0m $*"; }
warn()  { echo -e "\033[1;33m==>\033[0m $*"; }
err()   { echo -e "\033[1;31m==>\033[0m $*" >&2; }

# ---------- install arduino-cli if missing ----------

ensure_arduino_cli() {
    if command -v arduino-cli &>/dev/null; then
        return
    fi

    info "arduino-cli not found -- installing..."

    if command -v brew &>/dev/null; then
        brew install arduino-cli
    elif command -v curl &>/dev/null; then
        curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh -s -- --dest "$HOME/.local/bin"
        export PATH="$HOME/.local/bin:$PATH"
    else
        err "Cannot install arduino-cli. Install it manually:"
        err "  https://arduino.github.io/arduino-cli/installation/"
        exit 1
    fi

    if ! command -v arduino-cli &>/dev/null; then
        err "arduino-cli still not on PATH after install."
        err "Add ~/.local/bin to your PATH and re-run."
        exit 1
    fi

    ok "arduino-cli installed: $(arduino-cli version)"
}

# ---------- install board core + library ----------

install_deps() {
    info "Updating board index..."
    arduino-cli config init --overwrite 2>/dev/null || true
    arduino-cli config add board_manager.additional_urls "$ESP32_CORE_URL" 2>/dev/null || true
    arduino-cli core update-index

    if ! arduino-cli core list | grep -q "esp32:esp32"; then
        info "Installing ESP32 core (this may take a few minutes on first run)..."
        arduino-cli core install esp32:esp32
    else
        ok "ESP32 core already installed."
    fi

    if ! arduino-cli lib list 2>/dev/null | grep -qi "Adafruit MPR121"; then
        info "Installing Adafruit MPR121 library..."
        arduino-cli lib install "$MPR121_LIB"
    else
        ok "Adafruit MPR121 library already installed."
    fi

    if ! arduino-cli lib list 2>/dev/null | grep -qi "GFX Library for Arduino"; then
        info "Installing GFX Library for Arduino..."
        arduino-cli lib install "$GFX_LIB"
    else
        ok "GFX Library for Arduino already installed."
    fi
}

# ---------- detect port ----------

detect_port() {
    local ports
    ports=$(arduino-cli board list --format text 2>/dev/null \
        | grep -i "esp32\|ttyACM\|ttyUSB\|usbmodem" \
        | head -1 \
        | awk '{print $1}')

    if [[ -z "$ports" ]]; then
        # Fallback: grab first ttyACM device
        for p in /dev/ttyACM*; do
            if [[ -e "$p" ]]; then
                ports="$p"
                break
            fi
        done
    fi
    echo "$ports"
}

# ---------- main ----------

main() {
    local port="${1:-}"

    ensure_arduino_cli
    install_deps

    if [[ "$port" == "--install" ]]; then
        ok "Dependencies installed. Run again without --install to flash."
        exit 0
    fi

    if [[ ! -f "$SKETCH" ]]; then
        err "Sketch not found: $SKETCH"
        exit 1
    fi

    # Detect or validate port
    if [[ -z "$port" ]]; then
        info "Auto-detecting board..."
        port=$(detect_port)
        if [[ -z "$port" ]]; then
            err "No board detected. Make sure the T-Display S3 is plugged in."
            err "If it won't enumerate, hold BOOT, press RESET, release BOOT, then re-run."
            err "Or specify the port manually:  ./flash.sh /dev/ttyACM0"
            exit 1
        fi
        info "Detected port: $port"
    else
        if [[ ! -e "$port" ]]; then
            err "Port $port does not exist."
            exit 1
        fi
        info "Using port: $port"
    fi

    # Compile
    info "Compiling sketch..."
    arduino-cli compile \
        --fqbn "${BOARD_FQBN}:${BOARD_FLAGS}" \
        "$SKETCH_DIR"

    # Upload
    info "Uploading to $port..."
    arduino-cli upload \
        --fqbn "${BOARD_FQBN}:${BOARD_FLAGS}" \
        --port "$port" \
        "$SKETCH_DIR"

    ok "Flash complete!"
    echo ""
    echo "Verify with:  cat $port    (Ctrl-C to stop)"
    echo "You should see T00 / T01 lines at 115200 baud."
}

main "$@"

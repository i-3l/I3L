#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ALIGN_ENV="${IIIL_ALIGN_ENV:-behavior}"

if [[ -n "${IIIL_ALIGN_PYTHON:-}" ]]; then
    PYTHON_BIN="$IIIL_ALIGN_PYTHON"
else
    if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniforge3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    else
        echo "Error: conda was not found. Set IIIL_ALIGN_PYTHON to a Python with cv2 and omegaconf." >&2
        exit 1
    fi

    if ! conda activate "$ALIGN_ENV"; then
        echo "Error: could not activate conda environment '$ALIGN_ENV'." >&2
        echo "Set IIIL_ALIGN_ENV or IIIL_ALIGN_PYTHON to use another environment." >&2
        exit 1
    fi
    # Resolve Python from the activated environment to avoid PATH shims.
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi

if ! "$PYTHON_BIN" -c "import cv2, numpy, omegaconf" 2>/dev/null; then
    echo "Error: $PYTHON_BIN cannot import cv2, numpy, and omegaconf." >&2
    echo "Set IIIL_ALIGN_ENV or IIIL_ALIGN_PYTHON to a working project environment." >&2
    exit 1
fi

cd "$SCRIPT_DIR"

# With no arguments, ask which task to align with instead of silently opening
# the default profile. Task names are the session snapshot prefixes
# (NAME_<timestamp>) under alignment_references/sessions/.
if [[ $# -eq 0 && -t 0 ]]; then
    SESSIONS_DIR="$SCRIPT_DIR/alignment_references/sessions"
    mapfile -t TASKS < <(
        find "$SESSIONS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
            | sed -E 's/_[0-9]{8}_[0-9]{6}$//' | sort -u
    )
    if [[ ${#TASKS[@]} -gt 0 ]]; then
        echo "Choose a task to align with (newest snapshot for that task is used):"
        PS3="Task number: "
        select TASK in "${TASKS[@]}" "last (newest snapshot overall)"; do
            if [[ -z "${TASK:-}" ]]; then
                echo "Invalid choice; pick a number from the list." >&2
                continue
            fi
            [[ "$TASK" == "last (newest snapshot overall)" ]] && TASK="last"
            set -- "$TASK"
            break
        done
        if [[ $# -eq 0 ]]; then
            echo "No task selected; aborting." >&2
            exit 1
        fi
    fi
fi

# Viewer modes need an X display, but SSH/tmux shells usually have none.
# Borrow the workstation's physical display the same way the
# on-physical-display bashrc helper does. Headless modes must stay headless:
# run.sh modes 10/11 call --snapshot mid-startup and must never hit a sudo
# prompt or open a window.
NEEDS_DISPLAY=true
for arg in "$@"; do
    case "$arg" in
        --check|--list|--snapshot|--snapshot=*|--help|-h)
            NEEDS_DISPLAY=false
            ;;
    esac
done

TEMP_XAUTHORITY=""
cleanup() {
    if [[ -n "$TEMP_XAUTHORITY" ]]; then
        rm -f -- "$TEMP_XAUTHORITY"
    fi
}
trap cleanup EXIT

if $NEEDS_DISPLAY && [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    TARGET_DISPLAY="${PHYSICAL_DISPLAY:-:1}"
    SOURCE_XAUTHORITY="${PHYSICAL_XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

    if [[ "$TARGET_DISPLAY" == :* ]]; then
        display_number="${TARGET_DISPLAY#:}"
        display_number="${display_number%%.*}"
        if [[ ! -S "/tmp/.X11-unix/X${display_number}" ]]; then
            echo "Error: no DISPLAY is set and the physical display ${TARGET_DISPLAY} is not running." >&2
            echo "Run this from the robot desktop, or set PHYSICAL_DISPLAY to the right X display." >&2
            exit 1
        fi
    fi

    echo "No DISPLAY in this shell; opening the viewer on ${TARGET_DISPLAY} (sudo may prompt)."
    TEMP_XAUTHORITY="$(mktemp /tmp/align-xauthority.XXXXXX)"
    if sudo install -m 600 -o "$(id -un)" -g "$(id -gn)" \
        "$SOURCE_XAUTHORITY" "$TEMP_XAUTHORITY"; then
        export DISPLAY="$TARGET_DISPLAY"
        export XAUTHORITY="$TEMP_XAUTHORITY"
    else
        echo "Error: could not copy ${SOURCE_XAUTHORITY} to access display ${TARGET_DISPLAY}." >&2
        echo "Run this from the robot desktop, or: on-physical-display ./align.sh" >&2
        exit 1
    fi
fi

STATUS=0
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/align_cameras.py" "$@" || STATUS=$?
exit "$STATUS"

#!/bin/bash
set -e

# Franka Workstation Setup Script
# Installs all dependencies for running I3L on a Franka workstation
# into the active conda environment.
#
# Prerequisites:
#   - conda environment activated (e.g. `conda activate behavior`)
#   - Repos cloned:
#       ~/deoxys_control/deoxys
#       ~/FrankaJoylo
#       ~/il_lib  (optional, for training)
#
# Usage: ./setup_franka.sh

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DEOXYS_DIR="${HOME}/deoxys_control/deoxys"
FRANKA_JOYLO_DIR="${HOME}/FrankaJoylo"
IL_LIB_DIR="${HOME}/il_lib"
IIIL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check conda env is active
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
    echo -e "${RED}Error: Activate a conda environment first (e.g. conda activate behavior)${NC}"
    exit 1
fi
echo -e "${GREEN}Using conda env: ${CONDA_DEFAULT_ENV}${NC}"

# Check repos exist
for dir in "$DEOXYS_DIR" "$FRANKA_JOYLO_DIR"; do
    if [[ ! -d "$dir" ]]; then
        echo -e "${RED}Error: $dir not found${NC}"
        exit 1
    fi
done

# 1. Install deoxys runtime dependencies (its setup.py declares none)
echo -e "${YELLOW}[1/5] Installing deoxys dependencies...${NC}"
pip install \
    pyzmq \
    PyYAML \
    easydict \
    google-api-python-client \
    pybullet \
    numpy \
    hidapi \
    glfw \
    numba \
    termcolor \
    simplejson \
    "mujoco>=3.0"

# 2. Install deoxys editable
echo -e "${YELLOW}[2/5] Installing deoxys (editable)...${NC}"
pip install -e "$DEOXYS_DIR"

# 3. Regenerate deoxys protobuf files (checked-in _pb2.py files are incompatible with protobuf 4+)
echo -e "${YELLOW}[3/5] Regenerating deoxys protobuf files...${NC}"
if command -v protoc &>/dev/null; then
    (
        cd "$DEOXYS_DIR"
        protoc --proto_path=proto --python_out=deoxys/proto \
            franka-interface/franka_controller.proto \
            franka-interface/franka_robot_state.proto
        protoc --proto_path=proto --python_out=deoxys/proto \
            sample.proto
    )
    echo -e "${GREEN}  Protobuf files regenerated.${NC}"
else
    echo -e "${RED}  Warning: protoc not found. If you get protobuf errors at runtime, install protobuf compiler:${NC}"
    echo -e "${RED}    pip install grpcio-tools  # then re-run this script${NC}"
fi

# 4. Install FrankaJoylo editable
echo -e "${YELLOW}[4/5] Installing FrankaJoylo (editable)...${NC}"
pip install -e "$FRANKA_JOYLO_DIR"

# 5. Install il_lib if present
if [[ -d "$IL_LIB_DIR" ]]; then
    echo -e "${YELLOW}[5/5] Installing il_lib (editable)...${NC}"
    pip install -e "$IL_LIB_DIR"
else
    echo -e "${YELLOW}[5/5] Skipping il_lib (not found at $IL_LIB_DIR)${NC}"
fi

# Verify imports
echo ""
echo -e "${YELLOW}Verifying imports...${NC}"
FAILED=0
for mod in deoxys franka_joylo easydict hidapi pybullet glfw mujoco simplejson; do
    if python -c "import $mod" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $mod"
    else
        echo -e "  ${RED}✗${NC} $mod"
        FAILED=1
    fi
done

if [[ $FAILED -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}All Franka dependencies installed successfully.${NC}"
else
    echo ""
    echo -e "${RED}Some imports failed. Check the output above.${NC}"
    exit 1
fi

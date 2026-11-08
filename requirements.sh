#!/usr/bin/env bash
# ==============================================================================
# Drosophila Fly Brain — System & Environment Setup
#
# This script configures all dependencies, datasets, and runtime components
# required to run the Drosophila Connectome Robot Simulation & Live Laboratory.
#
# Usage:
#   cd fly_brain_ros
#   ./requirements.sh
# ==============================================================================

set -euo pipefail

# Avoid RMW DDS incompatibility issues during inspection
unset RMW_IMPLEMENTATION

# Visual formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}${BOLD}"
echo "=============================================================================="
echo " Drosophila Connectome Robot Simulation — Environment Setup"
echo "=============================================================================="
echo -e "${NC}"

# Script and repository root resolution
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
REPO_ROOT="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
cd "$REPO_ROOT"

# ------------------------------------------------------------------------------
# 1. System & Architecture Checks
# ------------------------------------------------------------------------------
echo -e "${BOLD}[1/7] Checking Operating System & Architecture...${NC}"
OS_NAME="$(uname -s)"
ARCH="$(uname -m)"
echo "  - OS: $OS_NAME ($ARCH)"

if [ "$OS_NAME" != "Linux" ]; then
    echo -e "${YELLOW}  Warning: Non-Linux OS detected ($OS_NAME). Gazebo Fortress and ROS 2 Humble require Linux (Ubuntu 22.04 LTS recommended).${NC}"
fi

# Select target Python interpreter:
# Priority:
#   1. Explicit PYTHON_BIN env variable
#   2. Active Python virtual environment ($VIRTUAL_ENV/bin/python3)
#   3. System Python matching ROS 2 Humble (/usr/bin/python3 or /usr/bin/python3.10)
#   4. Default system python3 in PATH
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python3" ]; then
        PYTHON_BIN="${VIRTUAL_ENV}/bin/python3"
    elif [ -f "/opt/ros/humble/setup.bash" ] && [ -x "/usr/bin/python3" ]; then
        PYTHON_BIN="/usr/bin/python3"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo -e "${RED}  Error: No python3 interpreter found. Please install Python 3.10 or higher.${NC}" >&2
        exit 1
    fi
fi

PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "  - Target Python: $PYTHON_BIN (v$PY_VERSION)"

# ------------------------------------------------------------------------------
# 2. ROS 2 Humble & Gazebo Fortress Check
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[2/7] Checking ROS 2 Humble & Gazebo Fortress...${NC}"
if [ -f "/opt/ros/humble/setup.bash" ]; then
    echo -e "${GREEN}  ✔ ROS 2 Humble found at /opt/ros/humble/setup.bash.${NC}"
else
    echo -e "${YELLOW}  Notice: ROS 2 Humble was not found at /opt/ros/humble/setup.bash.${NC}"
    echo "  The pure neural engine, 3D connectome viewer, and unit tests can run without ROS 2,"
    echo "  but physical Gazebo robot simulation requires ROS 2 Humble on Ubuntu 22.04 LTS."
    echo "  To install ROS 2 Humble, follow:"
    echo "    https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html"
fi

if command -v ign >/dev/null 2>&1; then
    IGN_VER=$(ign gazebo --versions 2>/dev/null | head -n 1 || echo "installed")
    echo -e "${GREEN}  ✔ Gazebo Fortress (ign gazebo) detected: $IGN_VER.${NC}"
else
    echo -e "${YELLOW}  Notice: Ignition/Gazebo Fortress ('ign') not found in PATH.${NC}"
    echo "  For 3D physics robot simulation, install Gazebo Fortress via:"
    echo "    sudo apt-get install -y ros-humble-ros-gz"
fi

# ------------------------------------------------------------------------------
# 3. System Apt Dependencies
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[3/7] Checking System Utilities & Libraries...${NC}"
APT_PKGS=(python3-pip python3-dev git curl jq lsof psmisc build-essential)
MISSING_PKGS=()

for pkg in "${APT_PKGS[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo "  Missing system packages: ${MISSING_PKGS[*]}"
    if [ "$EUID" -eq 0 ]; then
        apt-get update && apt-get install -y "${MISSING_PKGS[@]}"
    elif command -v sudo >/dev/null 2>&1; then
        echo "  Attempting to install missing packages via sudo..."
        sudo apt-get update && sudo apt-get install -y "${MISSING_PKGS[@]}" || {
            echo -e "${YELLOW}  Warning: Could not automatically install system packages via sudo.${NC}"
            echo "  Please install them manually: sudo apt-get install -y ${MISSING_PKGS[*]}"
        }
    else
        echo -e "${YELLOW}  Warning: Sudo not available. Please install: ${MISSING_PKGS[*]}${NC}"
    fi
else
    echo -e "${GREEN}  ✔ All core system utilities are installed.${NC}"
fi

# ------------------------------------------------------------------------------
# 4. Python Package Dependencies
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[4/7] Installing Python Requirements...${NC}"

PIP_CMD="$PYTHON_BIN -m pip"
PIP_INSTALL_FLAGS=""

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if "$PYTHON_BIN" -c "import sys; sys.exit(0 if sys.version_info < (3, 12) else 1)" 2>/dev/null; then
        PIP_INSTALL_FLAGS="--user"
    else
        PIP_INSTALL_FLAGS="--user --break-system-packages"
    fi
fi

echo "  - Pip command: $PIP_CMD install $PIP_INSTALL_FLAGS"

# Enforce NumPy < 2.0.0 first to maintain PyTorch C-API compatibility
echo "  - Verifying NumPy compatibility (< 2.0.0)..."
$PIP_CMD install $PIP_INSTALL_FLAGS "numpy>=1.21.5,<2.0.0"

# Install remaining requirements
echo "  - Installing requirements from requirements.txt..."
$PIP_CMD install $PIP_INSTALL_FLAGS -r "$REPO_ROOT/requirements.txt"
echo -e "${GREEN}  ✔ Python dependencies verified successfully.${NC}"

# ------------------------------------------------------------------------------
# 5. Connectome Biological Dataset Setup
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[5/7] Verifying Drosophila Connectome Dataset (138,639 Neurons)...${NC}"
mkdir -p "$REPO_ROOT/data"

COMPLETENESS_FILE="$REPO_ROOT/data/2025_Completeness_783.csv"
CONNECTIVITY_FILE="$REPO_ROOT/data/2025_Connectivity_783.parquet"

DATA_FOUND=0

if [ -f "$COMPLETENESS_FILE" ] && [ -f "$CONNECTIVITY_FILE" ]; then
    DATA_FOUND=1
    echo -e "${GREEN}  ✔ Connectome files already exist in local data/ directory.${NC}"
fi

# Check sibling directory (e.g. ../fly-brain/data)
if [ "$DATA_FOUND" -eq 0 ]; then
    SIBLING_DATA="$(cd "$REPO_ROOT/.." && pwd)/fly-brain/data"
    if [ -f "$SIBLING_DATA/2025_Completeness_783.csv" ] && [ -f "$SIBLING_DATA/2025_Connectivity_783.parquet" ]; then
        echo "  - Found connectome dataset in sibling repository: $SIBLING_DATA"
        rm -rf "$REPO_ROOT/data"
        ln -sfn "$SIBLING_DATA" "$REPO_ROOT/data"
        DATA_FOUND=1
        echo -e "${GREEN}  ✔ Linked connectome dataset from $SIBLING_DATA to data/.${NC}"
    fi
fi

# If still not found, clone upstream repository from Eon Systems PBC
if [ "$DATA_FOUND" -eq 0 ]; then
    echo "  - Connectome dataset not found locally. Cloning from upstream repository..."
    UPSTREAM_DIR="$REPO_ROOT/fly-brain-upstream"
    if [ ! -d "$UPSTREAM_DIR" ]; then
        git clone --depth 1 https://github.com/eonsystemspbc/fly-brain.git "$UPSTREAM_DIR"
    fi
    if [ -d "$UPSTREAM_DIR/data" ]; then
        rm -rf "$REPO_ROOT/data"
        ln -sfn "$UPSTREAM_DIR/data" "$REPO_ROOT/data"
        DATA_FOUND=1
        echo -e "${GREEN}  ✔ Connectome dataset initialized from upstream repository.${NC}"
    else
        echo -e "${RED}  Error: Failed to obtain connectome data from upstream repository.${NC}" >&2
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# 6. Spatial Atlas & Checkpoint Directory Initialization
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[6/7] Initializing 3D Spatial Atlas & Checkpoint State...${NC}"
mkdir -p "$REPO_ROOT/checkpoints"

# Generate 3D spatial coordinates bin for WebGL 3D Connectome Viewer if missing
SPATIAL_BIN="$REPO_ROOT/data/connectome_spatial_coords.bin"
if [ ! -f "$SPATIAL_BIN" ]; then
    echo "  - Generating 3D spatial layout coordinates (138k neurons)..."
    PYTHONPATH="$REPO_ROOT:$REPO_ROOT/virtual_world:$REPO_ROOT/live_lab${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" -m fly_brain_ros.connectome_spatial_layout \
        --comp-csv "$REPO_ROOT/data/2025_Completeness_783.csv" \
        --output "$SPATIAL_BIN"
    echo -e "${GREEN}  ✔ 3D spatial coordinate atlas generated successfully.${NC}"
else
    echo -e "${GREEN}  ✔ 3D spatial coordinate atlas already present.${NC}"
fi

# Initial baseline checkpoint
LOCAL_CP="$REPO_ROOT/checkpoints/fly_brain_checkpoint.json"
PARENT_CP="$(cd "$REPO_ROOT/.." && pwd)/checkpoints/fly_brain_checkpoint.json"
if [ ! -f "$LOCAL_CP" ] && [ -f "$PARENT_CP" ]; then
    echo "  - Initializing persistent baseline checkpoint from $PARENT_CP..."
    cp "$PARENT_CP" "$LOCAL_CP"
    echo -e "${GREEN}  ✔ Baseline checkpoint initialized in checkpoints/fly_brain_checkpoint.json.${NC}"
fi

# ------------------------------------------------------------------------------
# 7. File Permissions & Alias Links
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[7/7] Setting Permissions and Helper Links...${NC}"
chmod +x "$REPO_ROOT/run.sh" "$REPO_ROOT/requirements.sh"

# Create requirments.sh symlink for spelling compatibility
if [ ! -e "$REPO_ROOT/requirments.sh" ]; then
    ln -sf "requirements.sh" "$REPO_ROOT/requirments.sh"
    echo -e "${GREEN}  ✔ Created spelling alias: requirments.sh -> requirements.sh${NC}"
fi

# ------------------------------------------------------------------------------
# Summary & Next Steps
# ------------------------------------------------------------------------------
echo -e "\n${GREEN}${BOLD}=============================================================================="
echo " Setup Completed Successfully!"
echo "=============================================================================="
echo -e "${NC}"
echo "To build the ROS 2 workspace (if not already built):"
echo -e "  ${BLUE}cd ~/robo_fly_ros_ws && colcon build --symlink-install${NC}"
echo ""
echo "To start the Drosophila Robot Simulation and Live Laboratory Dashboard:"
echo -e "  ${BLUE}./run.sh${NC}                  # Full simulation with Gazebo 3D Arena"
echo -e "  ${BLUE}./run.sh headless:=true${NC}   # Headless simulation (low CPU/GPU overhead)"
echo ""
echo "Interactive Web Dashboard:"
echo -e "  ${BLUE}http://127.0.0.1:5000${NC}"
echo ""
echo "To run the automated verification test suite (142 unit & integration tests):"
echo -e "  ${BLUE}unset RMW_IMPLEMENTATION && python3 -m pytest -p no:anyio test/${NC}"
echo "=============================================================================="

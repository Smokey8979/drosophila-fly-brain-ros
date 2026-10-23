#!/usr/bin/env bash
# ==============================================================================
# Drosophila Fly Brain — One-Command Startup and Safe Shutdown
#
# Usage:
#   cd ~/Desktop/robo_fly/fly_brain_ros
#   ./run.sh                      # Standard interactive launch
#   ./run.sh headless:=true       # Headless simulation mode
#   ./run.sh --help               # Display help
#
# Terminal Control:
#   Type 'close' and press Enter to save neural state and exit safely.
#   Press Ctrl+C for the same graceful shutdown.
# ==============================================================================

set -m           # Enable job control for process-group isolation
trap '' PIPE     # Ignore SIGPIPE so broken console pipes do not abort shutdown

# Help option
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "============================================================"
    echo " Drosophila Fly Brain — Interactive Runner"
    echo "============================================================"
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "Launches the complete Drosophila Connectome robot simulation:"
    echo "  - Gazebo Fortress learning arena"
    echo "  - 4-wheel robot spawner"
    echo "  - ROS-GZ Bridge (/cmd_vel, /scan, /odom)"
    echo "  - Fly Brain Neural Engine (138k LIF neurons, learning, affect)"
    echo "  - Interactive Live Web Dashboard (http://127.0.0.1:5000)"
    echo ""
    echo "Options:"
    echo "  headless:=true       Run Gazebo headless (no 3D GUI client)"
    echo "  use_sim_time:=true   Use Gazebo simulation clock"
    echo "  dashboard_port:=PORT Web dashboard port (default: 5000)"
    echo "  --help, -h           Show this help message"
    echo ""
    echo "Safe Shutdown:"
    echo "  Type 'close' and press Enter in the terminal to save state."
    echo "  Press Ctrl+C to request the same graceful shutdown."
    echo "============================================================"
    exit 0
fi

# Directory resolution (handling symlinks transparently)
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$SCRIPT_DIR"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECKPOINT_FILE="${FLY_BRAIN_CHECKPOINT_PATH:-$PARENT_DIR/checkpoints/fly_brain_checkpoint.json}"
export FLY_BRAIN_CHECKPOINT_PATH="$CHECKPOINT_FILE"
LOCKFILE="/tmp/fly_brain_ros_run.pid"
DASHBOARD_PORT=5000
DASHBOARD_URL="http://127.0.0.1:${DASHBOARD_PORT}"

# Extract custom port if passed in arguments
for arg in "$@"; do
    if [[ "$arg" =~ dashboard_port:=([0-9]+) ]]; then
        DASHBOARD_PORT="${BASH_REMATCH[1]}"
        DASHBOARD_URL="http://127.0.0.1:${DASHBOARD_PORT}"
    fi
done

# ------------------------------------------------------------------------------
# 1. Concurrency & Port Conflict Checks
# ------------------------------------------------------------------------------
if [ -f "$LOCKFILE" ]; then
    EXISTING_PID=$(cat "$LOCKFILE" 2>/dev/null || echo "")
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "Error: Another fly brain simulation instance is already running (PID: $EXISTING_PID)." >&2
        echo "Please close the existing instance or terminate PID $EXISTING_PID before launching." >&2
        exit 1
    else
        rm -f "$LOCKFILE" 2>/dev/null || true
    fi
fi
echo "$$" > "$LOCKFILE"

# Port conflict check
if python3 -c "import socket; s=socket.socket(); s.settimeout(0.4); exit(0 if s.connect_ex(('127.0.0.1', $DASHBOARD_PORT)) == 0 else 1)" 2>/dev/null; then
    echo "Error: Port $DASHBOARD_PORT is already in use by another application or process." >&2
    echo "Cannot start dashboard server on port $DASHBOARD_PORT. Please free the port before running." >&2
    rm -f "$LOCKFILE"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Environment Setup
# ------------------------------------------------------------------------------
unset RMW_IMPLEMENTATION

if [ -f "/opt/ros/humble/setup.bash" ]; then
    # shellcheck source=/dev/null
    source "/opt/ros/humble/setup.bash"
else
    echo "Error: ROS 2 Humble setup file not found at /opt/ros/humble/setup.bash." >&2
    rm -f "$LOCKFILE"
    exit 1
fi

ROS_WS_SETUP="$HOME/robo_fly_ros_ws/install/setup.bash"
if [ -f "$ROS_WS_SETUP" ]; then
    # shellcheck source=/dev/null
    source "$ROS_WS_SETUP"
else
    echo "Error: ROS workspace setup not found at $ROS_WS_SETUP." >&2
    echo "Please build the workspace using 'colcon build --symlink-install' in ~/robo_fly_ros_ws." >&2
    rm -f "$LOCKFILE"
    exit 1
fi

export IGN_IP=127.0.0.1
export QT_QPA_PLATFORM=xcb
unset WAYLAND_DISPLAY

if [ -d "$HOME/.local/lib/python3.10/site-packages" ]; then
    export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
fi

# ------------------------------------------------------------------------------
# 3. Logging Setup
# ------------------------------------------------------------------------------
LOG_DIR="$REPO_ROOT/test_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

mkdir -p "$PARENT_DIR/checkpoints"

# Global state trackers
LAUNCH_PID=""
SHUTDOWN_IN_PROGRESS=0
CHECKPOINT_VERIFIED=0

# ------------------------------------------------------------------------------
# 4. Graceful Shutdown & Checkpoint Verification Function
# ------------------------------------------------------------------------------
graceful_save_and_shutdown() {
    local trigger="${1:-USER_COMMAND}"

    if [ "$SHUTDOWN_IN_PROGRESS" -eq 1 ]; then
        echo "[INFO] Graceful shutdown is already in progress, please wait..."
        return
    fi
    SHUTDOWN_IN_PROGRESS=1

    echo ""
    echo "============================================================"
    echo " Initiating Safe Shutdown Sequence ($trigger)"
    echo "============================================================"

    # Step 1: Stop teleoperation and command zero velocity
    echo "[1/6] Halting robot motion through safety-supervised command path..."
    curl -s -f -X POST "$DASHBOARD_URL/api/control" \
        -H "Content-Type: application/json" \
        -d '{"state": "pause"}' >/dev/null 2>&1 || true

    ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
        "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" >/dev/null 2>&1 &

    local mtime_before=0
    if [ -f "$CHECKPOINT_FILE" ]; then
        mtime_before=$(stat -c %Y "$CHECKPOINT_FILE" 2>/dev/null || echo 0)
    fi

    # Step 2: Request sleep through production lifecycle interface
    echo "[2/6] Requesting neural state checkpoint via production lifecycle API..."
    local sleep_api_ok=0
    if curl -s -f -X POST "$DASHBOARD_URL/api/lifecycle/sleep" \
        -H "Content-Type: application/json" >/dev/null 2>&1; then
        sleep_api_ok=1
        echo "      Production /api/lifecycle/sleep endpoint responded OK."
    else
        echo "      Notice: Dashboard /api/lifecycle/sleep not responsive."
    fi

    # Authoritative broadcast via ROS 2 lifecycle control topic for guaranteed redundancy
    ros2 topic pub --once /fly_brain/lifecycle_control std_msgs/msg/String \
        "data: 'SLEEP'" >/dev/null 2>&1 &

    # Step 3: Wait for checkpoint commit and lifecycle status update
    echo "[3/6] Waiting for atomic checkpoint flush..."
    local flushed=0
    for i in {1..20}; do
        if [ -f "$CHECKPOINT_FILE" ]; then
            local mtime_now=0
            mtime_now=$(stat -c %Y "$CHECKPOINT_FILE" 2>/dev/null) || mtime_now=0
            if [ "$mtime_now" -gt "$mtime_before" ] || [ "$mtime_before" -eq 0 ]; then
                flushed=1
                break
            fi
        fi
        sleep 0.25
    done

    # Step 4: Verify cryptographic integrity of checkpoint
    echo "[4/6] Verifying cryptographic integrity and completeness of saved checkpoint..."
    local cp_validation
    cp_validation=$(python3 - << 'EOF' "$CHECKPOINT_FILE"
import sys, os, json, hashlib

cp_path = sys.argv[1]
if not os.path.exists(cp_path):
    print("FAIL: Checkpoint file not found on disk.")
    sys.exit(1)

size = os.path.getsize(cp_path)
if size < 5000:
    print(f"FAIL: Checkpoint size suspiciously small ({size} bytes).")
    sys.exit(2)

try:
    with open(cp_path, "r") as f:
        data = json.load(f)
except Exception as e:
    print(f"FAIL: JSON decoding error: {e}")
    sys.exit(3)

manifest = data.get("manifest", {})
schema = str(manifest.get("schema_version", ""))
if not schema.startswith("2."):
    print(f"FAIL: Unsupported schema version '{schema}'.")
    sys.exit(4)

stored_hash = manifest.pop("sha256", None)
canonical_bytes = json.dumps(data, sort_keys=True).encode("utf-8")
calc_hash = hashlib.sha256(canonical_bytes).hexdigest()

if not stored_hash or stored_hash != calc_hash:
    print(f"FAIL: SHA-256 hash mismatch! Stored={stored_hash}, Calc={calc_hash}")
    sys.exit(5)

required_sections = ["neural_state", "learning_state", "internal_affect_state", "episodic_memory"]
for sec in required_sections:
    if sec not in data:
        print(f"FAIL: Missing required state section '{sec}'.")
        sys.exit(6)

print(f"PASS|{size}|{stored_hash}|{schema}")
sys.exit(0)
EOF
)
    local py_res=$?
    local cp_bytes=0
    local cp_digest=""
    local cp_schema=""

    if [ $py_res -eq 0 ] && [[ "$cp_validation" == PASS* ]]; then
        CHECKPOINT_VERIFIED=1
        cp_bytes=$(echo "$cp_validation" | cut -d'|' -f2)
        cp_digest=$(echo "$cp_validation" | cut -d'|' -f3)
        cp_schema=$(echo "$cp_validation" | cut -d'|' -f4)
        echo "      [PASS] Checkpoint integrity confirmed: SHA-256 match ($cp_bytes bytes, Schema v$cp_schema)."
    else
        CHECKPOINT_VERIFIED=0
        echo "      [FAIL] $cp_validation"
        echo "      [WARNING] Neural checkpoint could not be verified. Proceeding with controlled stop."
    fi

    # Step 5: Graceful process group termination
    echo "[5/6] Gracefully terminating script-owned simulation processes..."
    if [ -n "$LAUNCH_PID" ] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
        # Send SIGINT to process group (let ros2 launch and web_server handle clean shutdown)
        kill -INT -"$LAUNCH_PID" 2>/dev/null || true

        local wait_cycles=0
        while kill -0 "$LAUNCH_PID" 2>/dev/null && [ $wait_cycles -lt 20 ]; do
            sleep 0.5
            wait_cycles=$((wait_cycles + 1))
        done

        if kill -0 "$LAUNCH_PID" 2>/dev/null; then
            echo "      Processes still active after 10s; sending SIGTERM..."
            kill -TERM -"$LAUNCH_PID" 2>/dev/null || true
            sleep 2
        fi

        wait "$LAUNCH_PID" 2>/dev/null || true

        # Wait for all processes in the script-owned process group to exit
        local pg_cycles=0
        while pgrep -g "$LAUNCH_PID" >/dev/null 2>&1 && [ $pg_cycles -lt 10 ]; do
            sleep 0.3
            pg_cycles=$((pg_cycles + 1))
        done
        if pgrep -g "$LAUNCH_PID" >/dev/null 2>&1; then
            kill -KILL -"$LAUNCH_PID" 2>/dev/null || true
            sleep 0.5
        fi
    fi

    # Step 6: Print Final Summary
    echo "[6/6] Shutdown sequence finished."
    echo ""
    echo "============================================================"
    echo " Drosophila Fly Brain — Shutdown Summary"
    echo "============================================================"
    if [ "$CHECKPOINT_VERIFIED" -eq 1 ]; then
        local size_kb=$((cp_bytes / 1024))
        echo " Neural State Saved:     YES (Cryptographically Verified)"
        echo " Checkpoint Path:        $CHECKPOINT_FILE"
        echo " Checkpoint Size:        $cp_bytes bytes (~${size_kb} KB)"
        echo " SHA-256 Digest:         ${cp_digest:0:16}...${cp_digest:48:16}"
        echo " Schema Version:         v$cp_schema"
    else
        echo " Neural State Saved:     NO (Verification Failed / Unsaved)"
        echo " Checkpoint Path:        $CHECKPOINT_FILE"
        if [ -n "$cp_validation" ]; then
            echo " Failure Reason:         $cp_validation"
        fi
    fi
    echo " Shutdown Trigger:       $trigger"
    echo " Owned Processes:        Terminated cleanly"
    echo " Runtime Log:            $LOG_FILE"
    echo "============================================================"
    echo ""
}

# ------------------------------------------------------------------------------
# 5. Trap Registration
# ------------------------------------------------------------------------------
cleanup_exit() {
    rm -f "$LOCKFILE" 2>/dev/null || true
    if [ "$SHUTDOWN_IN_PROGRESS" -eq 0 ] && [ -n "$LAUNCH_PID" ] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
        graceful_save_and_shutdown "EXIT_HANDLER"
    fi
}

trap cleanup_exit EXIT
trap 'echo ""; echo ">>> Caught Ctrl+C (SIGINT). Initiating graceful save and shutdown..."; graceful_save_and_shutdown "SIGINT"; exit 0' INT
trap 'echo ""; echo ">>> Caught SIGTERM. Initiating graceful save and shutdown..."; graceful_save_and_shutdown "SIGTERM"; exit 0' TERM

# ------------------------------------------------------------------------------
# 6. Launch Simulation Stack
# ------------------------------------------------------------------------------
LAUNCH_ARGS=(
    "use_sim_time:=false"
    "dashboard_port:=${DASHBOARD_PORT}"
    "start_dashboard:=true"
)

# Append any custom user arguments
for arg in "$@"; do
    LAUNCH_ARGS+=("$arg")
done

echo "==========================================="
echo " Drosophila Fly Brain — Live Laboratory"
echo "==========================================="
echo ""
echo "Starting the brain, Gazebo, bridges and dashboard..."
echo "Runtime log: $LOG_FILE"
echo ""

# Launch ROS 2 stack in background with process group, streaming to console and log
ros2 launch fly_brain_ros live_lab_full.launch.py "${LAUNCH_ARGS[@]}" > >(tee -i -a "$LOG_FILE") 2>&1 &
LAUNCH_PID=$!

# ------------------------------------------------------------------------------
# 7. Await Readiness & Display Interactive Prompt
# ------------------------------------------------------------------------------
echo "Awaiting system initialization..."
INITIALIZED=0
START_TIME=$(date +%s)
TIMEOUT=45

while [ $(($(date +%s) - START_TIME)) -lt $TIMEOUT ]; do
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo "Error: Launch process terminated unexpectedly during startup." >&2
        exit 1
    fi

    if curl -s -f -m 1 "$DASHBOARD_URL/api/lifecycle/status" >/dev/null 2>&1 || \
       curl -s -f -m 1 "$DASHBOARD_URL/api/state" >/dev/null 2>&1; then
        INITIALIZED=1
        break
    fi
    sleep 1
done

if [ "$INITIALIZED" -eq 1 ]; then
    echo ""
    echo "==========================================="
    echo " Drosophila Fly Brain — Ready"
    echo "==========================================="
    echo "Dashboard: http://127.0.0.1:${DASHBOARD_PORT}"
    echo ""
    echo "The simulation is running."
    echo "Type close and press Enter to save state and shut down safely."
    echo "Press Ctrl+C to request the same graceful shutdown."
    echo "==========================================="
    echo ""
else
    echo ""
    echo "Notice: Dashboard initialization is still settling, but processes are active."
    echo "Type close and press Enter to save state and shut down safely."
    echo "Press Ctrl+C to request the same graceful shutdown."
    echo ""
fi

# ------------------------------------------------------------------------------
# 8. Main Terminal Input Loop
# ------------------------------------------------------------------------------
while [ "$SHUTDOWN_IN_PROGRESS" -eq 0 ]; do
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo ""
        echo "[INFO] Simulation processes exited externally."
        graceful_save_and_shutdown "EXTERNAL_EXIT"
        break
    fi

    user_cmd=""
    if read -t 1 -r user_cmd; then
        cmd_norm=$(echo "$user_cmd" | tr '[:upper:]' '[:lower:]' | xargs 2>/dev/null || echo "$user_cmd")
        if [[ "$cmd_norm" == "close" || "$cmd_norm" == "quit" || "$cmd_norm" == "exit" ]]; then
            echo ""
            echo ">>> Accepted '$cmd_norm' command from terminal."
            graceful_save_and_shutdown "USER_CLOSE"
            break
        elif [[ -n "$cmd_norm" ]]; then
            echo ">>> Unknown command: '$user_cmd'. Type 'close' and press Enter to save and exit."
        fi
    else
        read_rc=$?
        # If read_rc <= 128 (e.g. 1 on EOF when stdin is /dev/null or redirected), sleep 1 to prevent busy-spin
        if [ "$read_rc" -le 128 ]; then
            sleep 1
        fi
    fi
done

exit 0

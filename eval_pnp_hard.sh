#!/bin/bash
# ==============================================================================
# eval_pnp_hard.sh — Evaluate one or more policies on a1 pnp_hard
#
# Runs each policy with CuRobo oracle in the background (via policy=residual_eval)
# to record per-step base_action vs oracle_action and detector scores.
# Then generates per-rollout joint-level comparison graphs.
#
# Usage:
#   ./eval_pnp_hard.sh <ckpt1> [ckpt2] [ckpt3] ...
#   ./eval_pnp_hard.sh --tmux --episodes 100 <ckpt1> <ckpt2>
#   ./eval_pnp_hard.sh --tmux --episodes 50 --videos 10 data/path/to/ckpt.pth
#
# Options:
#   --tmux           Launch inside a tmux session (attach with: tmux attach -t eval_pnp_hard)
#   --episodes N     Episodes per policy (default: 100)
#   --videos N       Record N individual episode videos per policy (default: 0)
#   --arch ARCH      Serve architecture (default: dp_a1)
#   --port PORT      Websocket port (default: 8000)
#   --seed SEED      Evaluation seed (default: 43)
#   --skip-graphs    Skip graph generation
#   --output-dir DIR Custom output directory
# ==============================================================================

set -euo pipefail

# ====== Tmux launch: if --tmux flag is passed, relaunch inside a tmux session ======
if [[ "${1:-}" == "--tmux" ]]; then
  shift
  SESSION_NAME="eval_pnp_hard"
  # Kill existing session if any
  tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
  # Re-invoke this script without --tmux inside a new tmux session
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  tmux new-session -d -s "$SESSION_NAME" "$SCRIPT_PATH $*"
  echo "Eval launched in tmux session '$SESSION_NAME'"
  echo "  Attach:  tmux attach -t $SESSION_NAME"
  echo "  Detach:  Ctrl-b d"
  echo "  Kill:    tmux kill-session -t $SESSION_NAME"
  exit 0
fi

# ====== Activate conda env ======
set +u
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate behavior
set -u
export OMNI_KIT_ACCEPT_EULA=YES
export DISPLAY="${DISPLAY:-:1}"
export OMNIGIBSON_HEADLESS=1

# ====== Defaults ======
EPISODES=100
ARCH="dp_a1"
PORT=8000
VIDEOS=0
SKIP_GRAPHS=false
OUTPUT_DIR=""
SEED=43
CKPTS=()

# ====== Parse args ======
while [[ $# -gt 0 ]]; do
  case $1 in
    --episodes) EPISODES="$2"; shift 2;;
    --arch) ARCH="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --videos) VIDEOS="$2"; shift 2;;
    --skip-graphs) SKIP_GRAPHS=true; shift;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    -*) echo "Unknown option: $1"; exit 1;;
    *) CKPTS+=("$1"); shift;;
  esac
done

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "Usage: $0 [options] <ckpt1> [ckpt2] ..."
  echo "  --episodes N     Episodes per policy (default: 100)"
  echo "  --videos N       Record N episode videos per policy (default: 0)"
  echo "  --arch ARCH      Serve architecture (default: dp_a1)"
  echo "  --port PORT      Websocket port (default: 8000)"
  echo "  --seed SEED      Evaluation seed (default: 43)"
  echo "  --skip-graphs    Skip graph generation"
  echo "  --output-dir DIR Custom output directory"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IL_LIB_DIR="$(dirname "$SCRIPT_DIR")/il_lib"
export IIIL_BASE_PATH="$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$SCRIPT_DIR/data/eval_pnp_hard/$TIMESTAMP"
fi
[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo " EVAL: a1 pnp_hard"
echo "----------------------------------------------------------------------"
echo " Policies:   ${#CKPTS[@]}"
echo " Episodes:   $EPISODES per policy"
echo " Arch:       $ARCH"
echo " Output:     $OUTPUT_DIR"
echo "======================================================================"

# ====== Helper functions ======

wait_for_port() {
  local port=$1
  local elapsed=0
  while ! python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',$port)); s.close()" 2>/dev/null; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [[ $elapsed -ge 120 ]]; then
      echo "ERROR: Timeout waiting for port $port"
      return 1
    fi
  done
  echo "  Server ready (${elapsed}s)"
}

kill_port() {
  fuser -k "$1/tcp" 2>/dev/null || true
  sleep 2
}

# ====== Evaluate each policy ======

POLICY_IDX=0
FAILED_POLICIES=0
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
> "$SUMMARY_FILE"

for CKPT in "${CKPTS[@]}"; do
  POLICY_IDX=$((POLICY_IDX + 1))

  # Resolve to absolute path
  [[ "$CKPT" != /* ]] && CKPT="$SCRIPT_DIR/$CKPT"

  if [[ ! -f "$CKPT" ]]; then
    echo "WARNING: checkpoint not found, skipping: $CKPT"
    echo "[$POLICY_IDX] $CKPT: CHECKPOINT NOT FOUND" >> "$SUMMARY_FILE"
    FAILED_POLICIES=$((FAILED_POLICIES + 1))
    continue
  fi

  # Extract a short name from the checkpoint path
  POLICY_NAME=$(basename "$(dirname "$(dirname "$CKPT")")")/$(basename "$CKPT" .pth)
  POLICY_DIR="$(realpath -m "$OUTPUT_DIR/policy_$(printf '%02d' $POLICY_IDX)")"
  mkdir -p "$POLICY_DIR/graphs"

  echo ""
  echo "======================================================================"
  echo " [$POLICY_IDX/${#CKPTS[@]}] $POLICY_NAME"
  echo "----------------------------------------------------------------------"
  echo " Checkpoint: $CKPT"
  echo " Output:     $POLICY_DIR"
  echo "======================================================================"

  # Save metadata
  echo "$CKPT" > "$POLICY_DIR/checkpoint_path.txt"
  echo "$POLICY_NAME" > "$POLICY_DIR/policy_name.txt"

  # Kill anything on port
  kill_port $PORT

  # Start policy server
  echo "  Starting policy server ($ARCH)..."
  cd "$IL_LIB_DIR"
  python serve.py arch=$ARCH robot=a1_iiil task=pnp_hard/base \
    ckpt_path="$CKPT" > "$POLICY_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  cd "$SCRIPT_DIR"

  if ! wait_for_port "$PORT"; then
    kill "$SERVER_PID" 2>/dev/null || true
    echo "[$POLICY_IDX] $POLICY_NAME: SERVER FAILED" >> "$SUMMARY_FILE"
    FAILED_POLICIES=$((FAILED_POLICIES + 1))
    continue
  fi

  # Run eval
  RESULTS_JSON="$POLICY_DIR/results.json"
  rm -f "$RESULTS_JSON" "$RESULTS_JSON.tmp"

  echo "  Running $EPISODES episodes (seed=$SEED)..."
  EVAL_STATUS=0
  python run_task.py \
    env=sim_a1_joint \
    task=pnp_hard/base \
    policy=il \
    policy.host=127.0.0.1 \
    policy.port=$PORT \
    task.termination_config.total_episode=$EPISODES \
    results_json_path="$RESULTS_JSON" \
    seed=$SEED \
    > "$POLICY_DIR/eval.log" 2>&1 || EVAL_STATUS=$?

  TOTAL_TRIALS=$(awk '/Total trials:/ {trials=$NF} END {print trials}' "$POLICY_DIR/eval.log")
  SR=""
  if [[ -f "$RESULTS_JSON" ]]; then
    SR=$(python -c 'import json, sys; sr = float(json.load(open(sys.argv[1]))["SR"]); assert 0.0 <= sr <= 1.0; print(f"{sr:.1%}")' "$RESULTS_JSON" 2>/dev/null || true)
  fi

  EVAL_COMPLETE=false
  if [[ "$TOTAL_TRIALS" == "$EPISODES" && -n "$SR" ]]; then
    EVAL_COMPLETE=true
    if [[ $EVAL_STATUS -ne 0 ]]; then
      echo "  WARNING: eval exited with status $EVAL_STATUS after writing complete results"
    fi
  fi

  # Record individual episode videos (same seed so same object placements)
  if [[ "$EVAL_COMPLETE" == "true" && $VIDEOS -gt 0 ]]; then
    echo "  Recording $VIDEOS episode videos (seed=$SEED)..."
    mkdir -p "$POLICY_DIR/videos"
    for i in $(seq 0 $((VIDEOS - 1))); do
      python run_task.py \
        env=sim_a1_joint \
        task=pnp_hard/base \
        policy=il \
        policy.host=127.0.0.1 \
        policy.port=$PORT \
        task.termination_config.total_episode=1 \
        video_recording_path="$POLICY_DIR/videos/ep${i}.mp4" \
        seed=$((42 + i)) \
        > "$POLICY_DIR/videos/ep${i}.log" 2>&1 || true
    done
  fi

  # Cleanup server
  kill $SERVER_PID 2>/dev/null || true
  kill_port $PORT

  # Print result
  if [[ "$EVAL_COMPLETE" == "true" ]]; then
    echo "  Result: SR=$SR"
    echo "[$POLICY_IDX] $POLICY_NAME: SR=$SR" >> "$SUMMARY_FILE"
  else
    echo "  Eval failed or incomplete (status=$EVAL_STATUS, trials=${TOTAL_TRIALS:-unknown}; check $POLICY_DIR/eval.log)"
    echo "[$POLICY_IDX] $POLICY_NAME: FAILED" >> "$SUMMARY_FILE"
    FAILED_POLICIES=$((FAILED_POLICIES + 1))
  fi
done

# ====== Generate per-rollout graphs ======

if [[ "$SKIP_GRAPHS" == "false" ]]; then
  echo ""
  echo "======================================================================"
  echo " Generating per-rollout graphs..."
  echo "======================================================================"

  python "$SCRIPT_DIR/scripts/plot_eval_rollouts.py" --output-dir "$OUTPUT_DIR"
fi

# ====== Final summary ======

echo ""
echo "======================================================================"
echo " RESULTS SUMMARY"
echo "----------------------------------------------------------------------"
cat "$SUMMARY_FILE"
echo ""
echo " Output: $OUTPUT_DIR"
echo " Graphs: $OUTPUT_DIR/policy_*/graphs/"
echo "======================================================================"

if [[ $FAILED_POLICIES -gt 0 ]]; then
  echo "ERROR: $FAILED_POLICIES policy evaluation(s) failed or were incomplete."
  exit 1
fi

#!/bin/bash
# ==============================================================================
# eval.sh — Automated evaluation for base and residual policies
#
# Usage:
#   ./eval.sh --base-ckpt <path>                          # Eval base policy only
#   ./eval.sh --base-ckpt <path> --residual-ckpt <path>   # Eval both, compare
#   ./eval.sh --residual-ckpt <path> --base-ckpt <path> --episodes 50 --repeats 3
#
# Options:
#   --base-ckpt PATH       Base policy checkpoint (required)
#   --residual-ckpt PATH   Residual policy checkpoint (optional)
#   --episodes N            Episodes per eval repeat (default: 50)
#   --repeats N             Number of eval repeats (default: 1)
#   --videos N              Record N individual episode videos per policy (default: 5)
#   --output-dir DIR        Output directory (default: data/eval_auto/<timestamp>)
#   --env ENV               Hydra env config (default: sim_a1_joint)
#   --task TASK             Hydra task config (default: pnp/base)
#   --robot ROBOT           Hydra robot config (default: a1_iiil)
#   --arch ARCH             Residual arch config (default: simple_residual_a1_v13)
#   --port PORT             Websocket port (default: 8000)
#   --skip-videos           Skip video recording
#   --skip-graphs           Skip graph generation
# ==============================================================================

set -euo pipefail

# ====== Defaults ======
BASE_CKPT=""
RESIDUAL_CKPT=""
EPISODES=50
REPEATS=1
VIDEOS=5
OUTPUT_DIR=""
ENV="sim_a1_joint"
TASK="pnp/base"
ROBOT="a1_iiil"
ARCH="simple_residual_a1_v13"
PORT=8000
SKIP_VIDEOS=false
SKIP_GRAPHS=false

# ====== Parse args ======
while [[ $# -gt 0 ]]; do
  case $1 in
    --base-ckpt) BASE_CKPT="$2"; shift 2;;
    --residual-ckpt) RESIDUAL_CKPT="$2"; shift 2;;
    --episodes) EPISODES="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --videos) VIDEOS="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --env) ENV="$2"; shift 2;;
    --task) TASK="$2"; shift 2;;
    --robot) ROBOT="$2"; shift 2;;
    --arch) ARCH="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --skip-videos) SKIP_VIDEOS=true; shift;;
    --skip-graphs) SKIP_GRAPHS=true; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$BASE_CKPT" ]]; then
  echo "Error: --base-ckpt is required"
  exit 1
fi

# Absolutize checkpoint paths
[[ "$BASE_CKPT" != /* ]] && BASE_CKPT="$(pwd)/$BASE_CKPT"
[[ -n "$RESIDUAL_CKPT" && "$RESIDUAL_CKPT" != /* ]] && RESIDUAL_CKPT="$(pwd)/$RESIDUAL_CKPT"

# ====== Paths ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IL_LIB_DIR="$(dirname "$SCRIPT_DIR")/il_lib"
export IIIL_BASE_PATH="$SCRIPT_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$SCRIPT_DIR/data/eval_auto/$TIMESTAMP"
fi
mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo " EVAL CONFIG"
echo "----------------------------------------------------------------------"
echo " Base ckpt:     $BASE_CKPT"
echo " Residual ckpt: ${RESIDUAL_CKPT:-none}"
echo " Arch config:   $ARCH"
echo " Episodes:      $EPISODES x $REPEATS repeats"
echo " Videos:        $VIDEOS per policy"
echo " Output:        $OUTPUT_DIR"
echo " Env/Task:      $ENV / $TASK"
echo "======================================================================"

# ====== Helper functions ======

wait_for_port() {
  local port=$1
  local timeout=${2:-120}
  local elapsed=0
  while ! python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',$port)); s.close()" 2>/dev/null; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [[ $elapsed -ge $timeout ]]; then
      echo "ERROR: Timeout waiting for port $port"
      return 1
    fi
  done
  echo "  Port $port is ready (${elapsed}s)"
}

kill_port() {
  fuser -k "$1/tcp" 2>/dev/null || true
  sleep 2
}

run_eval() {
  local name=$1
  local episodes=$2
  local results_json=$3
  local video_path=${4:-}

  local hdf5_path=${5:-}

  local cmd="python run_task.py env=$ENV task=$TASK policy=il"
  cmd+=" task.termination_config.total_episode=$episodes"
  cmd+=" results_json_path=$results_json"
  cmd+=" policy.host=127.0.0.1 policy.port=$PORT"
  if [[ -n "$video_path" ]]; then
    cmd+=" video_recording_path=$video_path"
  fi
  if [[ -n "$hdf5_path" ]]; then
    cmd+=" mode=collection hdf_path=$hdf5_path"
  fi

  echo "  Running: $name ($episodes episodes)"
  (cd "$SCRIPT_DIR" && eval "$cmd") > "$OUTPUT_DIR/${name}.log" 2>&1 || true
  # Extract summary from log
  grep -E "Total success|Success rate" "$OUTPUT_DIR/${name}.log" 2>/dev/null || true
}

run_residual_eval() {
  # Like run_eval but uses policy=residual_eval (SA + CuRobo oracle for ground truth)
  # Always collects HDF5 so we get per-step base_action + oracle_action
  local name=$1
  local episodes=$2
  local results_json=$3
  local video_path=${4:-}
  local hdf5_path=${5:-}

  local cmd="python run_task.py env=$ENV task=$TASK policy=residual_eval"
  cmd+=" task.termination_config.total_episode=$episodes"
  cmd+=" results_json_path=$results_json"
  cmd+=" +policy.base_policy.host=127.0.0.1 +policy.base_policy.port=$PORT"
  if [[ -n "$video_path" ]]; then
    cmd+=" video_recording_path=$video_path"
  fi
  if [[ -n "$hdf5_path" ]]; then
    cmd+=" mode=collection hdf_path=$hdf5_path"
  fi

  echo "  Running: $name ($episodes episodes, with CuRobo oracle)"
  (cd "$SCRIPT_DIR" && eval "$cmd") > "$OUTPUT_DIR/${name}.log" 2>&1 || true
  grep -E "Total success|Success rate" "$OUTPUT_DIR/${name}.log" 2>/dev/null || true
}

# ====== Eval base policy (skip if cached result exists) ======

BASE_CACHE="$SCRIPT_DIR/data/eval_results/base_n060.json"

if [[ -f "$BASE_CACHE" ]]; then
  echo ""
  echo "====== BASE POLICY: Using cached result ======"
  echo "  Source: $BASE_CACHE ($(cat $BASE_CACHE))"
  mkdir -p "$OUTPUT_DIR/base"
  cp "$BASE_CACHE" "$OUTPUT_DIR/base/repeat_01.json"
else
  echo ""
  echo "====== Evaluating BASE POLICY ======"
  kill_port $PORT

  echo "  Starting base policy server..."
  cd "$IL_LIB_DIR"
  python serve.py arch=dp_a1 robot=$ROBOT task=$TASK \
    ckpt_path="$BASE_CKPT" &
  BASE_SERVER_PID=$!
  cd "$SCRIPT_DIR"

  wait_for_port $PORT
  mkdir -p "$OUTPUT_DIR/base"

  run_eval "base/repeat_01" $EPISODES "$OUTPUT_DIR/base/repeat_01.json"

  # Save to cache for future runs
  cp "$OUTPUT_DIR/base/repeat_01.json" "$BASE_CACHE" 2>/dev/null || true

  kill $BASE_SERVER_PID 2>/dev/null || true
  kill_port $PORT
  echo "  Base policy evaluation complete."
fi

# ====== Eval residual policy (if provided) ======

if [[ -n "$RESIDUAL_CKPT" ]]; then
  echo ""
  echo "====== Evaluating RESIDUAL POLICY ======"

  echo "  Starting residual policy server..."
  cd "$SCRIPT_DIR"
  python scripts/eval_simple_residual.py \
    arch=$ARCH robot=$ROBOT task=$TASK data_dir=/tmp/fake \
    module.base_policy_ckpt_path="$BASE_CKPT" \
    ckpt_path="$RESIDUAL_CKPT" \
    --port=$PORT \
    --stats-path="$OUTPUT_DIR/residual/inference_stats.json" \
    --traces-path="$OUTPUT_DIR/residual/action_traces.json" &
  RESIDUAL_SERVER_PID=$!

  wait_for_port $PORT

  mkdir -p "$OUTPUT_DIR/residual"

  # Run repeats with CuRobo oracle (collects HDF5 with oracle ground truth)
  for r in $(seq 1 $REPEATS); do
    hdf5_arg="$OUTPUT_DIR/residual/rollouts_repeat$(printf '%02d' $r).hdf5"
    run_residual_eval "residual/repeat_$(printf '%02d' $r)" $EPISODES \
      "$OUTPUT_DIR/residual/repeat_$(printf '%02d' $r).json" "" "$hdf5_arg"
  done

  # Record individual episode videos
  if [[ "$SKIP_VIDEOS" == "false" && $VIDEOS -gt 0 ]]; then
    echo "  Recording $VIDEOS episode videos..."
    mkdir -p "$OUTPUT_DIR/residual/videos"
    for i in $(seq 0 $((VIDEOS - 1))); do
      run_residual_eval "residual/videos/ep$i" 1 \
        "$OUTPUT_DIR/residual/videos/ep${i}_results.json" \
        "$OUTPUT_DIR/residual/videos/ep${i}.mp4"
    done
  fi

  # Send SIGINT so server writes stats in its finally block
  kill -INT $RESIDUAL_SERVER_PID 2>/dev/null || true
  sleep 3
  kill $RESIDUAL_SERVER_PID 2>/dev/null || true
  kill_port $PORT

  # Display inference stats if available
  if [[ -f "$OUTPUT_DIR/residual/inference_stats.json" ]]; then
    echo "  Inference stats:"
    python -c "
import json
with open('$OUTPUT_DIR/residual/inference_stats.json') as f:
    s = json.load(f)
print(f'    Total steps: {s[\"total_steps\"]}')
print(f'    Intervention rate: {s[\"intervention_rate\"]:.1%} ({s[\"intervention_steps\"]}/{s[\"total_steps\"]})')
print(f'    Residual magnitude: mean={s[\"residual_mag_mean\"]:.4f}, p95={s[\"residual_mag_p95\"]:.4f}')
print(f'    Residual as % of base: {s[\"residual_pct_of_base\"]:.1f}%')
if 'residual_mag_when_intervened' in s:
    print(f'    Residual when intervened: {s[\"residual_mag_when_intervened\"]:.4f}')
if 'residual_mag_when_passthrough' in s:
    print(f'    Residual when passthrough: {s[\"residual_mag_when_passthrough\"]:.4f}')
" 2>/dev/null || true
  fi
  echo "  Residual policy evaluation complete."
fi

# ====== Generate graphs and comparison ======

if [[ "$SKIP_GRAPHS" == "false" ]]; then
  echo ""
  echo "====== Generating graphs ======"
  cd "$SCRIPT_DIR"
  python scripts/plot_eval_auto.py --output-dir "$OUTPUT_DIR"
  # Offline analysis: run residual on correction data (has ground truth intervention labels)
  if [[ -n "$RESIDUAL_CKPT" && -f "$SCRIPT_DIR/data/corrections_n060.hdf5" ]]; then
    echo "  Running offline analysis on correction data..."
    python scripts/analyze_residual_on_corrections.py \
      --correction-hdf5 data/corrections_n060.hdf5 \
      --residual-ckpt "$RESIDUAL_CKPT" \
      --base-ckpt "$BASE_CKPT" \
      --arch "$ARCH" \
      --output-json "$OUTPUT_DIR/offline_analysis.json" 2>/dev/null || echo "  (offline analysis failed)"
  fi

  # Analyze eval rollouts (with CuRobo oracle ground truth)
  if [[ -n "$RESIDUAL_CKPT" ]]; then
    for hdf5 in "$OUTPUT_DIR"/residual/rollouts_repeat*.hdf5; do
      if [[ -f "$hdf5" ]]; then
        echo "  Analyzing eval rollout: $(basename $hdf5)"
        python scripts/analyze_eval_rollouts.py \
          --hdf5 "$hdf5" \
          --output-dir "$OUTPUT_DIR" \
          --output-json "$OUTPUT_DIR/eval_oracle_analysis.json" 2>/dev/null || echo "  (eval rollout analysis failed)"
        break  # Only analyze first repeat
      fi
    done
  fi

  # Principled intervention accuracy analysis (gate vs oracle ground truth)
  if [[ -n "$RESIDUAL_CKPT" ]]; then
    for hdf5 in "$OUTPUT_DIR"/residual/rollouts_repeat*.hdf5; do
      if [[ -f "$hdf5" ]]; then
        echo "  Running intervention accuracy analysis..."
        python scripts/analyze_intervention_accuracy.py \
          --eval-hdf5 "$hdf5" \
          --residual-ckpt "$RESIDUAL_CKPT" \
          --base-ckpt "$BASE_CKPT" \
          --arch "$ARCH" \
          --output-dir "$OUTPUT_DIR" \
          --output-json "$OUTPUT_DIR/intervention_accuracy.json" 2>/dev/null || echo "  (intervention accuracy analysis failed)"
        break  # Only analyze first repeat
      fi
    done
  fi

  # Generate trace graphs if correction data + eval HDF5 exist
  python scripts/plot_residual_traces.py \
    --correction-hdf5 data/corrections_n060.hdf5 \
    --base-hdf5 "$OUTPUT_DIR/base/rollouts.hdf5" \
    --residual-hdf5 "$OUTPUT_DIR/residual/rollouts.hdf5" \
    --output-dir "$OUTPUT_DIR" 2>/dev/null || echo "  (some trace graphs skipped — missing HDF5 data)"

  # Residual effectiveness analysis (requires action traces + HDF5 with oracle)
  if [[ -n "$RESIDUAL_CKPT" && -f "$OUTPUT_DIR/residual/action_traces.json" ]]; then
    for hdf5 in "$OUTPUT_DIR"/residual/rollouts_repeat*.hdf5; do
      if [[ -f "$hdf5" ]]; then
        echo "  Generating residual effectiveness plots..."
        python scripts/plot_residual_effectiveness.py \
          --traces "$OUTPUT_DIR/residual/action_traces.json" \
          --hdf5 "$hdf5" \
          --output-dir "$OUTPUT_DIR" \
          --max-rollouts 10 || echo "  (effectiveness plots failed)"
        break
      fi
    done
  fi
fi

# ====== Stitch comparison videos ======

if [[ "$SKIP_VIDEOS" == "false" && -n "$RESIDUAL_CKPT" && $VIDEOS -gt 0 ]]; then
  echo ""
  echo "====== Stitching comparison videos ======"
  mkdir -p "$OUTPUT_DIR/comparison"
  python scripts/make_comparison_videos_auto.py \
    --base-dir "$OUTPUT_DIR/base/videos" \
    --residual-dir "$OUTPUT_DIR/residual/videos" \
    --output-dir "$OUTPUT_DIR/comparison" \
    --episodes $VIDEOS
fi

# ====== Summary ======

echo ""
echo "======================================================================"
echo " RESULTS SUMMARY"
echo "----------------------------------------------------------------------"
echo " Output directory: $OUTPUT_DIR"
echo ""
echo " Base policy:"
for f in "$OUTPUT_DIR"/base/repeat_*.json; do
  [[ -f "$f" ]] && echo "   $(basename $f): $(cat $f)"
done
if [[ -n "$RESIDUAL_CKPT" ]]; then
  echo ""
  echo " Residual policy:"
  for f in "$OUTPUT_DIR"/residual/repeat_*.json; do
    [[ -f "$f" ]] && echo "   $(basename $f): $(cat $f)"
  done
fi
echo ""
if [[ -f "$OUTPUT_DIR/residual/inference_stats.json" ]]; then
  echo ""
  echo " Inference stats: $OUTPUT_DIR/residual/inference_stats.json"
fi
if [[ -f "$OUTPUT_DIR/intervention_accuracy.json" ]]; then
  echo ""
  echo " Intervention accuracy:"
  python -c "
import json
with open('$OUTPUT_DIR/intervention_accuracy.json') as f:
    s = json.load(f)
r = s.get('roc', {})
b = s.get('best_threshold_match', {})
m = s.get('mse_by_gate', {})
print(f'    ROC AUC: {r.get(\"auc\", 0):.3f}')
print(f'    Best F1: {b.get(\"f1\", 0):.3f} (acc={b.get(\"accuracy\", 0):.3f} prec={b.get(\"precision\", 0):.3f} rec={b.get(\"recall\", 0):.3f})')
print(f'    Gate fire rate: {s.get(\"gate_summary\", {}).get(\"gate_fire_rate\", 0):.1%}')
if m.get('n_gate_on', 0) > 0:
    imp = (1 - m['gate_on_mse_exec_oracle'] / max(m['gate_on_mse_base_oracle'], 1e-8)) * 100
    print(f'    Gate ON MSE improvement: {imp:+.1f}%')
" 2>/dev/null || true
fi
echo ""
echo " Graphs: $OUTPUT_DIR/*.png"
echo " Videos: $OUTPUT_DIR/base/videos/, $OUTPUT_DIR/residual/videos/"
if [[ -n "$RESIDUAL_CKPT" ]]; then
  echo " Comparisons: $OUTPUT_DIR/comparison/"
fi
echo "======================================================================"

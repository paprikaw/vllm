#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/gpfs/projects/punim2715/vllm_workbench/vllm}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/python}"
TARGET_HOST="${TARGET_HOST:-spartan-gpgpu069}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/vllm_exp/configs/qwen_pp4_autoscaling_expand_100.yaml}"
RUN_LOG_DIR="${RUN_LOG_DIR:-$ROOT_DIR/new-logs/qwen32b-pp4-autoscaling-expand-100-a100-$(date +%Y%m%d-%H%M%S)}"
TMP_ROOT="${TMP_ROOT:-/home/bxb1/data/tmp/vllm_exp/qwen_autoscaling_expand_100}"
PORT="${PORT:-8000}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
CLEANUP_ON_EXIT="${CLEANUP_ON_EXIT:-1}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-300}"
GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-0}"
REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-4}"
FREE_GPU_MIN_MB="${FREE_GPU_MIN_MB:-70000}"
VLLM_AUTOSCALING_WAIT_CLEANUP_BEFORE_SCHEDULE="${VLLM_AUTOSCALING_WAIT_CLEANUP_BEFORE_SCHEDULE:-0}"
LOCAL_MODE=0

usage() {
  cat <<EOF
Usage:
  bash scripts/run_qwen_autoscaling_expand_100.sh [--local]

Environment overrides:
  TARGET_HOST=$TARGET_HOST
  CONFIG_PATH=$CONFIG_PATH
  RUN_LOG_DIR=$RUN_LOG_DIR
  RUN_TIMEOUT_SECONDS=$RUN_TIMEOUT_SECONDS
  CLEANUP_ON_EXIT=$CLEANUP_ON_EXIT
  WAIT_FOR_GPUS=$WAIT_FOR_GPUS
  GPU_WAIT_INTERVAL_SECONDS=$GPU_WAIT_INTERVAL_SECONDS
  GPU_WAIT_TIMEOUT_SECONDS=$GPU_WAIT_TIMEOUT_SECONDS
  REQUIRED_GPU_COUNT=$REQUIRED_GPU_COUNT
  FREE_GPU_MIN_MB=$FREE_GPU_MIN_MB
  VLLM_AUTOSCALING_WAIT_CLEANUP_BEFORE_SCHEDULE=$VLLM_AUTOSCALING_WAIT_CLEANUP_BEFORE_SCHEDULE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      LOCAL_MODE=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

short_host() {
  hostname -s
}

if [[ "$LOCAL_MODE" -eq 0 && "$(short_host)" != "$TARGET_HOST" ]]; then
  exec ssh "$TARGET_HOST" \
    "cd '$ROOT_DIR' && ROOT_DIR='$ROOT_DIR' VENV_ACTIVATE='$VENV_ACTIVATE' PYTHON_BIN='$PYTHON_BIN' TARGET_HOST='$TARGET_HOST' CONFIG_PATH='$CONFIG_PATH' RUN_LOG_DIR='$RUN_LOG_DIR' TMP_ROOT='$TMP_ROOT' PORT='$PORT' RUN_TIMEOUT_SECONDS='$RUN_TIMEOUT_SECONDS' CLEANUP_ON_EXIT='$CLEANUP_ON_EXIT' WAIT_FOR_GPUS='$WAIT_FOR_GPUS' GPU_WAIT_INTERVAL_SECONDS='$GPU_WAIT_INTERVAL_SECONDS' GPU_WAIT_TIMEOUT_SECONDS='$GPU_WAIT_TIMEOUT_SECONDS' REQUIRED_GPU_COUNT='$REQUIRED_GPU_COUNT' FREE_GPU_MIN_MB='$FREE_GPU_MIN_MB' VLLM_AUTOSCALING_WAIT_CLEANUP_BEFORE_SCHEDULE='$VLLM_AUTOSCALING_WAIT_CLEANUP_BEFORE_SCHEDULE' bash '$ROOT_DIR/scripts/run_qwen_autoscaling_expand_100.sh' --local"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

count_free_a100s() {
  nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits \
    | awk -F, -v min="$FREE_GPU_MIN_MB" '
      {
        name=$2
        free=$3
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        gsub(/^[ \t]+|[ \t]+$/, "", free)
        if (name ~ /A100/ && free + 0 >= min) {
          count++
        }
      }
      END { print count + 0 }
    '
}

has_active_vllm_job() {
  pgrep -af "vllm_exp.run sweep-test|vllm serve|benchmark_serving.py" \
    | grep -v "pgrep -af" >/dev/null 2>&1
}

port_in_use() {
  ss -ltn "( sport = :$PORT )" | tail -n +2 | grep -q .
}

wait_for_resources() {
  if [[ "$WAIT_FOR_GPUS" != "1" ]]; then
    return
  fi

  local start_ts now elapsed free_count
  start_ts="$(date +%s)"
  while true; do
    free_count="$(count_free_a100s)"
    if [[ "$free_count" -ge "$REQUIRED_GPU_COUNT" ]] \
      && ! has_active_vllm_job \
      && ! port_in_use; then
      return
    fi

    echo "Waiting for resources on $(hostname -s): free_a100s=${free_count}/${REQUIRED_GPU_COUNT}, min_free_mb=${FREE_GPU_MIN_MB}, port=${PORT}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader
    if has_active_vllm_job; then
      pgrep -af "vllm_exp.run sweep-test|vllm serve|benchmark_serving.py" || true
    fi
    if port_in_use; then
      ss -ltnp "( sport = :$PORT )" || true
    fi

    if [[ "$GPU_WAIT_TIMEOUT_SECONDS" -gt 0 ]]; then
      now="$(date +%s)"
      elapsed=$((now - start_ts))
      if [[ "$elapsed" -ge "$GPU_WAIT_TIMEOUT_SECONDS" ]]; then
        echo "Timed out waiting for resources after ${elapsed}s" >&2
        exit 124
      fi
    fi
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}

mkdir -p "$RUN_LOG_DIR" "$TMP_ROOT"
RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
BENCHMARK_CONFIG_PATH="$TMP_ROOT/$RUN_ID/benchmark_config.json"
DEPLOYMENT_CONFIG_PATH="$TMP_ROOT/$RUN_ID/vllm_config.json"
mkdir -p "$(dirname "$BENCHMARK_CONFIG_PATH")"

cleanup() {
  local status=$?
  if [[ "$CLEANUP_ON_EXIT" = "1" ]]; then
    pkill -TERM -f "vllm serve .*/Qwen3-32B-FP8" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "vllm serve .*/Qwen3-32B-FP8" 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_for_resources

if pgrep -af "raylet|gcs_server|ray::" >/dev/null 2>&1; then
  echo "Stopping stale Ray processes before starting this run."
  ray stop --force >/dev/null 2>&1 || true
fi

echo "host=$(hostname -f)"
echo "config=$CONFIG_PATH"
echo "log_dir=$RUN_LOG_DIR"
echo "benchmark_config=$BENCHMARK_CONFIG_PATH"
echo "deployment_config=$DEPLOYMENT_CONFIG_PATH"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

source "$VENV_ACTIVATE"
cd "$ROOT_DIR"

export BENCHMARK_CONFIG_PATH
export DEPLOYMENT_CONFIG_PATH
export VLLM_AUTOSCALING_WAIT_CLEANUP_BEFORE_SCHEDULE

timeout "$RUN_TIMEOUT_SECONDS" \
  "$PYTHON_BIN" -m vllm_exp.run sweep-test \
    --config "$CONFIG_PATH" \
    --log-dir "$RUN_LOG_DIR" \
    --single-server

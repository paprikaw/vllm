#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/bxb1/vllm_workbench/vllm}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/python}"
TARGET_HOST="${TARGET_HOST:-spartan-gpgpu169}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/vllm_exp/configs/llama70b_pp4_autoscaling_shrink_rdt_race_120_a100.yaml}"
RUN_LOG_DIR="${RUN_LOG_DIR:-$ROOT_DIR/new-logs/llama70b-pp4-shrink-rdt-race-a100-$(date +%Y%m%d-%H%M%S)}"
TMP_ROOT="${TMP_ROOT:-/home/bxb1/data/tmp/vllm_exp/pp4_shrink_rdt_race}"
PORT="${PORT:-8000}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-3600}"
CLEANUP_ON_EXIT="${CLEANUP_ON_EXIT:-1}"
LOCAL_MODE=0

usage() {
  cat <<EOF
Usage:
  bash scripts/run_pp4_shrink_rdt_race.sh [--local]

Environment overrides:
  TARGET_HOST=$TARGET_HOST
  CONFIG_PATH=$CONFIG_PATH
  RUN_LOG_DIR=$RUN_LOG_DIR
  RUN_TIMEOUT_SECONDS=$RUN_TIMEOUT_SECONDS
  CLEANUP_ON_EXIT=$CLEANUP_ON_EXIT
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
    "cd '$ROOT_DIR' && ROOT_DIR='$ROOT_DIR' VENV_ACTIVATE='$VENV_ACTIVATE' PYTHON_BIN='$PYTHON_BIN' TARGET_HOST='$TARGET_HOST' CONFIG_PATH='$CONFIG_PATH' RUN_LOG_DIR='$RUN_LOG_DIR' TMP_ROOT='$TMP_ROOT' PORT='$PORT' RUN_TIMEOUT_SECONDS='$RUN_TIMEOUT_SECONDS' CLEANUP_ON_EXIT='$CLEANUP_ON_EXIT' bash '$ROOT_DIR/scripts/run_pp4_shrink_rdt_race.sh' --local"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

if ss -ltn "( sport = :$PORT )" | tail -n +2 | grep -q .; then
  echo "Port $PORT is already in use on $(hostname -s); refusing to start." >&2
  exit 98
fi

mkdir -p "$RUN_LOG_DIR" "$TMP_ROOT"
RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
BENCHMARK_CONFIG_PATH="$TMP_ROOT/$RUN_ID/benchmark_config.json"
DEPLOYMENT_CONFIG_PATH="$TMP_ROOT/$RUN_ID/vllm_config.json"
mkdir -p "$(dirname "$BENCHMARK_CONFIG_PATH")"

cleanup() {
  local status=$?
  if [[ "$CLEANUP_ON_EXIT" = "1" ]]; then
    pkill -TERM -f "vllm serve .*Llama-3.3-70B-Instruct-FP8" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "vllm serve .*Llama-3.3-70B-Instruct-FP8" 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

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

timeout "$RUN_TIMEOUT_SECONDS" \
  "$PYTHON_BIN" -m vllm_exp.run sweep-test \
    --config "$CONFIG_PATH" \
    --log-dir "$RUN_LOG_DIR" \
    --single-server

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BASE_LOG_DIR="${BASE_LOG_DIR:-$ROOT_DIR/new-logs/dynamic-nixl-compare-$(date +%Y%m%d-%H%M%S)}"
EXPAND_SCRIPT="${EXPAND_SCRIPT:-$ROOT_DIR/scripts/run_qwen_expand_1to2_smoke.sh}"
SHRINK_SCRIPT="${SHRINK_SCRIPT:-$ROOT_DIR/scripts/run_qwen_expand_1to2_smoke.sh}"
DEFAULT_EXPAND_CONFIG="$ROOT_DIR/vllm_exp/configs/qwen_pp4_autoscaling_expand_1to2_smoke.yaml"
DEFAULT_SHRINK_CONFIG="$ROOT_DIR/vllm_exp/configs/qwen_pp4_autoscaling_shrink_smoke.yaml"
EXPAND_NCCL_CONFIG_PATH="${EXPAND_NCCL_CONFIG_PATH:-${EXPAND_CONFIG_PATH:-$DEFAULT_EXPAND_CONFIG}}"
EXPAND_NIXL_CONFIG_PATH="${EXPAND_NIXL_CONFIG_PATH:-${EXPAND_CONFIG_PATH:-$DEFAULT_EXPAND_CONFIG}}"
SHRINK_NCCL_CONFIG_PATH="${SHRINK_NCCL_CONFIG_PATH:-${SHRINK_CONFIG_PATH:-$DEFAULT_SHRINK_CONFIG}}"
SHRINK_NIXL_CONFIG_PATH="${SHRINK_NIXL_CONFIG_PATH:-${SHRINK_CONFIG_PATH:-$DEFAULT_SHRINK_CONFIG}}"
EXPAND_TARGET_HOST="${EXPAND_TARGET_HOST:-spartan-gpgpu069}"
SHRINK_TARGET_HOST="${SHRINK_TARGET_HOST:-spartan-gpgpu169}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-3600}"
CLEANUP_ON_EXIT="${CLEANUP_ON_EXIT:-1}"
RUN_CASES="${RUN_CASES:-expand:nccl,expand:nixl,shrink:nccl,shrink:nixl}"
NCCL_PYTHON_BIN="${NCCL_PYTHON_BIN:-/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/python}"
NCCL_VENV_ACTIVATE="${NCCL_VENV_ACTIVATE:-/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/activate}"
NIXL_PYTHON_BIN="${NIXL_PYTHON_BIN:-$NCCL_PYTHON_BIN}"
NIXL_VENV_ACTIVATE="${NIXL_VENV_ACTIVATE:-$NCCL_VENV_ACTIVATE}"
LOCAL_ARGS=()

usage() {
  cat <<EOF
Usage:
  bash scripts/run_dynamic_nixl_expand_shrink_compare.sh [--local]

Runs expand and shrink once with the existing NCCL data path and once with
direct NIXL enabled for both dynamic PP handoff and KV synchronizer data.

Environment overrides:
  ROOT_DIR=$ROOT_DIR
  BASE_LOG_DIR=$BASE_LOG_DIR
  EXPAND_TARGET_HOST=$EXPAND_TARGET_HOST
  SHRINK_TARGET_HOST=$SHRINK_TARGET_HOST
  RUN_TIMEOUT_SECONDS=$RUN_TIMEOUT_SECONDS
  CLEANUP_ON_EXIT=$CLEANUP_ON_EXIT
  NCCL_PYTHON_BIN=$NCCL_PYTHON_BIN
  NCCL_VENV_ACTIVATE=$NCCL_VENV_ACTIVATE
  NIXL_PYTHON_BIN=$NIXL_PYTHON_BIN
  NIXL_VENV_ACTIVATE=$NIXL_VENV_ACTIVATE
  EXPAND_NCCL_CONFIG_PATH=$EXPAND_NCCL_CONFIG_PATH
  EXPAND_NIXL_CONFIG_PATH=$EXPAND_NIXL_CONFIG_PATH
  SHRINK_NCCL_CONFIG_PATH=$SHRINK_NCCL_CONFIG_PATH
  SHRINK_NIXL_CONFIG_PATH=$SHRINK_NIXL_CONFIG_PATH
  RUN_CASES=$RUN_CASES
EOF
}

case_requested() {
  local wanted="$1:$2"
  [[ ",$RUN_CASES," == *",$wanted,"* ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      LOCAL_ARGS+=(--local)
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

run_case() {
  local name="$1"
  local transport="$2"
  local script="$3"
  local target_host="$4"
  local config_path="$5"
  local log_dir="$BASE_LOG_DIR/$name-$transport"

  if ! case_requested "$name" "$transport"; then
    echo "=== skipping $name with $transport (RUN_CASES=$RUN_CASES) ==="
    return
  fi

  mkdir -p "$log_dir"
  echo "=== running $name with $transport ==="
  echo "log_dir=$log_dir"

  if [[ "$transport" == "nixl" ]]; then
    VLLM_DYNAMIC_KV_TRANSPORT=nixl \
    VLLM_DYNAMIC_PP_TRANSPORT=nixl \
    VLLM_DYNAMIC_PP_NCCL_TRANSPORT=0 \
    VLLM_DYNAMIC_KV_NIXL_PORT="${VLLM_DYNAMIC_KV_NIXL_PORT:-35557}" \
    VLLM_DYNAMIC_PP_NIXL_PORT="${VLLM_DYNAMIC_PP_NIXL_PORT:-45557}" \
    TARGET_HOST="$target_host" \
    ROOT_DIR="$ROOT_DIR" \
    PYTHON_BIN="$NIXL_PYTHON_BIN" \
    VENV_ACTIVATE="$NIXL_VENV_ACTIVATE" \
    CONFIG_PATH="$config_path" \
    RUN_LOG_DIR="$log_dir" \
    RUN_TIMEOUT_SECONDS="$RUN_TIMEOUT_SECONDS" \
    CLEANUP_ON_EXIT="$CLEANUP_ON_EXIT" \
      bash "$script" "${LOCAL_ARGS[@]}"
  else
    VLLM_DYNAMIC_KV_TRANSPORT=nccl \
    VLLM_DYNAMIC_PP_TRANSPORT= \
    VLLM_DYNAMIC_PP_NCCL_TRANSPORT=1 \
    TARGET_HOST="$target_host" \
    ROOT_DIR="$ROOT_DIR" \
    PYTHON_BIN="$NCCL_PYTHON_BIN" \
    VENV_ACTIVATE="$NCCL_VENV_ACTIVATE" \
    CONFIG_PATH="$config_path" \
    RUN_LOG_DIR="$log_dir" \
    RUN_TIMEOUT_SECONDS="$RUN_TIMEOUT_SECONDS" \
    CLEANUP_ON_EXIT="$CLEANUP_ON_EXIT" \
      bash "$script" "${LOCAL_ARGS[@]}"
  fi
}

mkdir -p "$BASE_LOG_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
echo "base_log_dir=$BASE_LOG_DIR"

run_case expand nccl "$EXPAND_SCRIPT" "$EXPAND_TARGET_HOST" "$EXPAND_NCCL_CONFIG_PATH"
run_case expand nixl "$EXPAND_SCRIPT" "$EXPAND_TARGET_HOST" "$EXPAND_NIXL_CONFIG_PATH"
run_case shrink nccl "$SHRINK_SCRIPT" "$SHRINK_TARGET_HOST" "$SHRINK_NCCL_CONFIG_PATH"
run_case shrink nixl "$SHRINK_SCRIPT" "$SHRINK_TARGET_HOST" "$SHRINK_NIXL_CONFIG_PATH"

echo "All dynamic NIXL comparison runs completed."

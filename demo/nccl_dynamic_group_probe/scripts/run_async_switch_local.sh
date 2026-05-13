#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
NPROC="${NPROC:-4}"
MATRIX_SIZE="${MATRIX_SIZE:-4096}"
COMPUTE_REPEATS="${COMPUTE_REPEATS:-2}"
TOTAL_SEC="${TOTAL_SEC:-12}"
ASYNC_CREATE_SEC="${ASYNC_CREATE_SEC:-2}"
SWITCH_NOT_BEFORE_SEC="${SWITCH_NOT_BEFORE_SEC:-6}"
COLLECTIVE_MB="${COLLECTIVE_MB:-8}"
STATE_MB="${STATE_MB:-32}"
DTYPE="${DTYPE:-float16}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/async_switch_local_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  "${PROJECT_DIR}/async_group_switch_demo.py" \
  --out-dir "${OUT_DIR}" \
  --matrix-size "${MATRIX_SIZE}" \
  --compute-repeats "${COMPUTE_REPEATS}" \
  --total-sec "${TOTAL_SEC}" \
  --async-create-sec "${ASYNC_CREATE_SEC}" \
  --switch-not-before-sec "${SWITCH_NOT_BEFORE_SEC}" \
  --collective-mb "${COLLECTIVE_MB}" \
  --state-mb "${STATE_MB}" \
  --dtype "${DTYPE}"

echo "Wrote results to ${OUT_DIR}"

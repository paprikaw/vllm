#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
NPROC="${NPROC:-4}"
MATRIX_SIZE="${MATRIX_SIZE:-8192}"
COMPUTE_REPEATS="${COMPUTE_REPEATS:-3}"
TOTAL_SEC="${TOTAL_SEC:-20}"
TRIGGER_SEC="${TRIGGER_SEC:-8}"
COLLECTIVE_EVERY="${COLLECTIVE_EVERY:-1}"
COLLECTIVE_MB="${COLLECTIVE_MB:-16}"
EXPANDED_COLLECTIVE_MB="${EXPANDED_COLLECTIVE_MB:-64}"
DTYPE="${DTYPE:-float16}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/local_a100_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  "${PROJECT_DIR}/dynamic_group_probe.py" \
  --out-dir "${OUT_DIR}" \
  --matrix-size "${MATRIX_SIZE}" \
  --compute-repeats "${COMPUTE_REPEATS}" \
  --total-sec "${TOTAL_SEC}" \
  --trigger-sec "${TRIGGER_SEC}" \
  --collective-every "${COLLECTIVE_EVERY}" \
  --collective-mb "${COLLECTIVE_MB}" \
  --expanded-collective-mb "${EXPANDED_COLLECTIVE_MB}" \
  --dtype "${DTYPE}"

echo "Wrote results to ${OUT_DIR}"

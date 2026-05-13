#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
INITIAL_NPROC="${INITIAL_NPROC:-3}"
MATRIX_SIZE="${MATRIX_SIZE:-2048}"
COMPUTE_REPEATS="${COMPUTE_REPEATS:-1}"
TOTAL_SEC="${TOTAL_SEC:-12}"
SPAWN_SEC="${SPAWN_SEC:-2}"
SWITCH_SEC="${SWITCH_SEC:-6}"
COLLECTIVE_MB="${COLLECTIVE_MB:-4}"
STATE_MB="${STATE_MB:-16}"
DTYPE="${DTYPE:-float16}"
NEW_MASTER_ADDR="${NEW_MASTER_ADDR:-127.0.0.1}"
NEW_MASTER_PORT="${NEW_MASTER_PORT:-29631}"
NEW_RDZV_BACKEND="${NEW_RDZV_BACKEND:-file}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/dynamic_world_reinit_local_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${INITIAL_NPROC}" \
  "${PROJECT_DIR}/dynamic_world_reinit_demo.py" \
  --mode old-rank \
  --out-dir "${OUT_DIR}" \
  --matrix-size "${MATRIX_SIZE}" \
  --compute-repeats "${COMPUTE_REPEATS}" \
  --total-sec "${TOTAL_SEC}" \
  --spawn-sec "${SPAWN_SEC}" \
  --switch-sec "${SWITCH_SEC}" \
  --collective-mb "${COLLECTIVE_MB}" \
  --state-mb "${STATE_MB}" \
  --dtype "${DTYPE}" \
  --new-master-addr "${NEW_MASTER_ADDR}" \
  --new-master-port "${NEW_MASTER_PORT}" \
  --new-rdzv-backend "${NEW_RDZV_BACKEND}"

echo "Wrote results to ${OUT_DIR}"

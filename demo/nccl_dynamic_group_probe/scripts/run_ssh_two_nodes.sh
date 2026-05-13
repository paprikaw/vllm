#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"

NODE0="${NODE0:-spartan-gpgpu066}"
NODE1="${NODE1:-spartan-gpgpu007}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MASTER_ADDR="${MASTER_ADDR:-${NODE0}}"
MASTER_PORT="${MASTER_PORT:-29577}"
MATRIX_SIZE="${MATRIX_SIZE:-6144}"
COMPUTE_REPEATS="${COMPUTE_REPEATS:-3}"
TOTAL_SEC="${TOTAL_SEC:-30}"
TRIGGER_SEC="${TRIGGER_SEC:-10}"
COLLECTIVE_EVERY="${COLLECTIVE_EVERY:-1}"
COLLECTIVE_MB="${COLLECTIVE_MB:-16}"
EXPANDED_COLLECTIVE_MB="${EXPANDED_COLLECTIVE_MB:-64}"
DTYPE="${DTYPE:-float16}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/ssh_two_nodes_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}"

run_node() {
  local node="$1"
  local node_rank="$2"
  local log_file="${OUT_DIR}/node${node_rank}_${node}.log"

  ssh -o BatchMode=yes "${node}" \
    "cd '${REPO_DIR}' && \
     export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0,1,2,3}' && \
     export NCCL_DEBUG='${NCCL_DEBUG:-WARN}' && \
     export NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME:-bond0.3027}' && \
     export TORCH_NCCL_ASYNC_ERROR_HANDLING='${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}' && \
     export TORCH_NCCL_BLOCKING_WAIT='${TORCH_NCCL_BLOCKING_WAIT:-1}' && \
     '${PYTHON_BIN}' -m torch.distributed.run \
       --nnodes=2 \
       --node_rank='${node_rank}' \
       --nproc_per_node='${GPUS_PER_NODE}' \
       --master_addr='${MASTER_ADDR}' \
       --master_port='${MASTER_PORT}' \
       '${PROJECT_DIR}/dynamic_group_probe.py' \
       --out-dir '${OUT_DIR}' \
       --matrix-size '${MATRIX_SIZE}' \
       --compute-repeats '${COMPUTE_REPEATS}' \
       --total-sec '${TOTAL_SEC}' \
       --trigger-sec '${TRIGGER_SEC}' \
       --collective-every '${COLLECTIVE_EVERY}' \
       --collective-mb '${COLLECTIVE_MB}' \
       --expanded-collective-mb '${EXPANDED_COLLECTIVE_MB}' \
       --dtype '${DTYPE}'" \
    >"${log_file}" 2>&1 &
}

run_node "${NODE0}" 0
run_node "${NODE1}" 1

wait

echo "Wrote results to ${OUT_DIR}"
echo "Node logs:"
printf '  %s\n' "${OUT_DIR}"/node*.log

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NCCL_ROOT="${SCRIPT_DIR}/vendor/nvidia/nccl"

if [[ ! -x "${SCRIPT_DIR}/nccl_grow_shrink_demo" ]]; then
  "${SCRIPT_DIR}/build.sh"
fi

export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

exec "${SCRIPT_DIR}/nccl_grow_shrink_demo" "${@}"

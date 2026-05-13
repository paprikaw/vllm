#!/usr/bin/env bash
# Source this file to prefer the vendored NCCL 2.30.4 for local experiments.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NCCL_GROW_SHRINK_ROOT="${SCRIPT_DIR}"
export NCCL_ROOT="${SCRIPT_DIR}/vendor/nvidia/nccl"
export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export CPATH="${NCCL_ROOT}/include:${CPATH:-}"
export LIBRARY_PATH="${NCCL_ROOT}/lib:${LIBRARY_PATH:-}"

echo "NCCL_ROOT=${NCCL_ROOT}"
echo "LD_LIBRARY_PATH begins with ${NCCL_ROOT}/lib"

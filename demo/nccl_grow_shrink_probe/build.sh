#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NCCL_ROOT="${SCRIPT_DIR}/vendor/nvidia/nccl"

if [[ ! -f "${NCCL_ROOT}/include/nccl.h" || ! -f "${NCCL_ROOT}/lib/libnccl.so.2" ]]; then
  echo "Missing vendored NCCL under ${NCCL_ROOT}."
  echo "Install with: python -m pip install --target ${SCRIPT_DIR}/vendor nvidia-nccl-cu12==2.30.4"
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  module load CUDA/12.4.1 >/dev/null 2>&1 || true
fi

nvcc -std=c++17 -O2 \
  -I"${NCCL_ROOT}/include" \
  "${SCRIPT_DIR}/src/nccl_grow_shrink_demo.cu" \
  -L"${NCCL_ROOT}/lib" -l:libnccl.so.2 \
  -Xlinker -rpath -Xlinker "${NCCL_ROOT}/lib" \
  -o "${SCRIPT_DIR}/nccl_grow_shrink_demo"

nvcc -std=c++17 -O2 \
  -I"${NCCL_ROOT}/include" \
  "${SCRIPT_DIR}/src/nccl_grow_shrink_multinode.cu" \
  -L"${NCCL_ROOT}/lib" -l:libnccl.so.2 \
  -Xlinker -rpath -Xlinker "${NCCL_ROOT}/lib" \
  -o "${SCRIPT_DIR}/nccl_grow_shrink_multinode"

echo "Built ${SCRIPT_DIR}/nccl_grow_shrink_demo"
echo "Built ${SCRIPT_DIR}/nccl_grow_shrink_multinode"

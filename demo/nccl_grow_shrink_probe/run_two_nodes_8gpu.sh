#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
NCCL_ROOT="${SCRIPT_DIR}/vendor/nvidia/nccl"

A100_HOST="${A100_HOST:-spartan-gpgpu066}"
L40_HOST="${L40_HOST:-spartan-gpgpu007}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/results/two_node_8gpu_$(date +%Y%m%d_%H%M%S)}"

"${SCRIPT_DIR}/build.sh" >/dev/null
mkdir -p "${RUN_DIR}/logs"
rm -f "${RUN_DIR}"/*.uid "${RUN_DIR}"/*.{0,1,2,3,4,5,6,7} 2>/dev/null || true

COMMON_ENV=(
  "LD_LIBRARY_PATH=${NCCL_ROOT}/lib:\${LD_LIBRARY_PATH:-}"
  "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
  "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}"
  "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0.3027}"
  "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"
  "NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}"
)
MODE="${MODE:-grow-shrink}"

launch_local() {
  local rank="$1"
  (
    cd "${REPO_ROOT}"
    env "${COMMON_ENV[@]}" \
      "${SCRIPT_DIR}/nccl_grow_shrink_multinode" \
      --rank "${rank}" --local-device "${rank}" --world-size 8 \
      --parent-size 4 --run-dir "${RUN_DIR}" --mode "${MODE}"
  ) >"${RUN_DIR}/logs/rank${rank}.${A100_HOST}.log" 2>&1 &
  pids+=("$!")
}

launch_remote() {
  local rank="$1"
  local dev="$2"
  ssh -o BatchMode=yes "${L40_HOST}" \
    "cd '${REPO_ROOT}' && env ${COMMON_ENV[*]} '${SCRIPT_DIR}/nccl_grow_shrink_multinode' --rank '${rank}' --local-device '${dev}' --world-size 8 --parent-size 4 --run-dir '${RUN_DIR}' --mode '${MODE}'" \
    >"${RUN_DIR}/logs/rank${rank}.${L40_HOST}.log" 2>&1 &
  pids+=("$!")
}

echo "A100 host: ${A100_HOST}"
echo "L40 host:  ${L40_HOST}"
echo "Run dir:   ${RUN_DIR}"
echo "Mode:      ${MODE}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0.3027}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"

pids=()
for r in 0 1 2 3; do
  launch_local "${r}"
done
for r in 4 5 6 7; do
  launch_remote "${r}" "$((r - 4))"
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done

cat "${RUN_DIR}"/logs/rank*.log | sort -k2,2n || true

if [[ "${rc}" == 0 ]]; then
  echo "PASS: two-node 8-GPU grow/shrink completed"
else
  echo "FAIL: see ${RUN_DIR}/logs"
fi
exit "${rc}"

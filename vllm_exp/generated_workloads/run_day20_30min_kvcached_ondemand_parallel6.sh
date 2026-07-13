#!/usr/bin/env bash
set -u

CHECKOUT=/home/bxb1/.codex/worktrees/c60f/vllm
CONFIG_DIR="$CHECKOUT/vllm_exp/configs/day20_30min_kvcached_ondemand_parallel"
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be set}

mkdir -p "$RUN_ROOT"
cd "$CHECKOUT"
export RAY_ADDRESS=auto
export PYTHONUNBUFFERED=1

declare -A pids=()

cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

launch() {
  local label=$1
  local driver_host=$2
  local config="$CONFIG_DIR/$label.yaml"
  local command
  command="cd $CHECKOUT && RAY_ADDRESS=auto PYTHONUNBUFFERED=1 "
  command+="$CHECKOUT/.venv/bin/python -m vllm_exp.run sweep-test "
  command+="--config $config --log-dir $RUN_ROOT --single-server"
  if [[ "$(hostname -s)" == "$driver_host" ]]; then
    bash -lc "$command" >"$RUN_ROOT/launcher_$label.log" 2>&1 &
  else
    ssh -o BatchMode=yes "$driver_host" "$command" \
      >"$RUN_ROOT/launcher_$label.log" 2>&1 &
  fi
  pids["$label"]=$!
  printf '%s,%s,%s,%s\n' "$label" "${pids[$label]}" "$driver_host" "$config" \
    >>"$RUN_ROOT/launcher_pids.csv"
}

printf 'batch1_start,%s\n' "$(date --iso-8601=seconds)" >"$RUN_ROOT/batch_timestamps.csv"
launch dynamic300 spartan-gpgpu138
launch static44-36 spartan-gpgpu141
sleep 30
launch static36-44 spartan-gpgpu138
launch static48-32 spartan-gpgpu141

overall=0
if wait "${pids[dynamic300]}"; then
  dynamic_status=0
else
  dynamic_status=$?
  overall=1
fi
printf 'dynamic300,%s\n' "$dynamic_status" \
  >"$RUN_ROOT/dynamic_exit_status.csv"

for label in static36-44 static44-36 static48-32; do
  if wait "${pids[$label]}"; then
    status=0
  else
    status=$?
    overall=1
  fi
  printf '%s,%s\n' "$label" "$status" \
    >>"$RUN_ROOT/static_exit_status.csv"
done

printf 'batch2_start,%s\n' "$(date --iso-8601=seconds)" \
  >>"$RUN_ROOT/batch_timestamps.csv"
launch static40-40 spartan-gpgpu138
launch static52-28 spartan-gpgpu141

for label in static40-40 static52-28; do
  if wait "${pids[$label]}"; then
    status=0
  else
    status=$?
    overall=1
  fi
  printf '%s,%s\n' "$label" "$status" \
    >>"$RUN_ROOT/static_exit_status.csv"
done

exit "$overall"

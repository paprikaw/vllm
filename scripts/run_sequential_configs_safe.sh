#!/usr/bin/env bash

set -u
set -o pipefail

ROOT_DIR="/home/bxb1/vllm_workbench/vllm"
VENV_ACTIVATE="/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/activate"
PYTHON_BIN="/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/python"
TMP_ROOT="/home/bxb1/data/tmp/vllm_exp/sequential_runs"
SLEEP_BETWEEN_RUNS=5
CONTINUE_ON_ERROR=0
RESUME_MODE=0

CONFIGS=()
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_LOG_DIR=""
META_DIR=""
REPORT_PATH=""
SUMMARY_TSV=""
SQUEUE_SNAPSHOT=""

usage() {
  cat <<EOF
用法:
  bash scripts/run_sequential_configs_safe.sh [选项]

选项:
  --config <path>            添加一个配置文件，可重复使用多次
  --run-log-dir <path>       指定传给 vllm_exp 的 --log-dir（可选）
  --meta-dir <path>          指定本脚本的 summary/report 输出目录
  --resume                   强制 overwrite=false，跳过已完成子试验
  --continue-on-error        某个实验失败后继续执行后续实验
  --help                     显示帮助

说明:
  必须至少提供一个 --config。
  如果未提供 --run-log-dir，默认使用: new-logs/sequential-run-$TIMESTAMP
  如果未提供 --meta-dir，默认使用: memory/sequential-run-$TIMESTAMP
EOF
}

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    echo "$path"
  else
    echo "$ROOT_DIR/$path"
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] 文件不存在: $path" >&2
    exit 1
  fi
}

port_in_use() {
  local host="$1"
  local port="$2"

  if [[ "$host" = "$(hostname -s)" ]]; then
    ss -ltn "( sport = :$port )" | tail -n +2 | grep -q .
  else
    ssh "$host" "ss -ltn '( sport = :$port )' | tail -n +2 | grep -q ."
  fi
}

yaml_query() {
  local config_path="$1"
  local query_name="$2"
  "$PYTHON_BIN" - "$config_path" "$query_name" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
query_name = sys.argv[2]
with open(config_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

if query_name == 'rank0':
    print(data.get('static_config', {}).get('network', {}).get('rank_to_ip', {}).get(0)
          or data.get('static_config', {}).get('network', {}).get('rank_to_ip', {}).get('0', ''))
elif query_name == 'project':
    print(data.get('project', 'unknown-project'))
elif query_name == 'port':
    print(data.get('static_config', {}).get('vllm', {}).get('port', 8000))
elif query_name == 'overwrite':
    print(data.get('overwrite', True))
else:
    raise SystemExit(f'unknown query: {query_name}')
PY
}

patch_config_for_resume() {
  local src_config="$1"
  local dst_config="$2"

  "$PYTHON_BIN" - "$src_config" "$dst_config" <<'PY'
import sys
import yaml

src = sys.argv[1]
dst = sys.argv[2]

with open(src, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

data['overwrite'] = False

with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
PY
}

append_report_header() {
  mkdir -p "$META_DIR" "$TMP_ROOT"
  {
    echo "# 顺序实验运行报告"
    echo
    echo "- 时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- vllm_exp 日志根目录: $RUN_LOG_DIR"
    echo "- 脚本元数据目录: $META_DIR"
    echo "- Summary TSV: $SUMMARY_TSV"
    echo "- Resume 模式: $([[ "$RESUME_MODE" -eq 1 ]] && echo yes || echo no)"
    echo
    echo "## 配置列表"
    echo
    for config in "${CONFIGS[@]}"; do
      echo "- $config"
    done
    echo
    echo "## 运行结果"
    echo
    echo "| 序号 | 配置 | project | 执行节点 | 端口 | 状态 | 退出码 | 日志目录 | 备注 |"
    echo "|---|---|---|---|---:|---|---:|---|---|"
  } > "$REPORT_PATH"

  echo -e "index\tconfig\tproject\thost\tport\tstatus\texit_code\tlog_dir\tnote" > "$SUMMARY_TSV"
}

log_result() {
  local idx="$1"
  local config="$2"
  local project="$3"
  local host="$4"
  local port="$5"
  local status_label="$6"
  local exit_code="$7"
  local log_dir="$8"
  local note="$9"

  echo -e "${idx}\t${config}\t${project}\t${host}\t${port}\t${status_label}\t${exit_code}\t${log_dir}\t${note}" >> "$SUMMARY_TSV"
  echo "| ${idx} | ${config} | ${project} | ${host} | ${port} | ${status_label} | ${exit_code} | ${log_dir} | ${note} |" >> "$REPORT_PATH"
}

run_single_config() {
  local idx="$1"
  local config_path="$2"

  local project
  local rank0_host
  local port
  local local_host
  local run_name
  local run_tmp_dir
  local benchmark_tmp
  local deploy_tmp
  local effective_config_path
  local status=0
  local note=""
  local target_host

  project="$(yaml_query "$config_path" project)"
  rank0_host="$(yaml_query "$config_path" rank0)"
  port="$(yaml_query "$config_path" port)"
  local_host="$(hostname -s)"
  target_host="$local_host"

  if [[ -n "$rank0_host" ]]; then
    target_host="$rank0_host"
  fi

  run_name="$(printf '%02d-%s' "$idx" "$(basename "${config_path%.yaml}")")"
  run_tmp_dir="$TMP_ROOT/$TIMESTAMP/$run_name"
  benchmark_tmp="$run_tmp_dir/benchmark_config.json"
  deploy_tmp="$run_tmp_dir/vllm_config.json"
  effective_config_path="$config_path"

  mkdir -p "$run_tmp_dir"

  if [[ "$RESUME_MODE" -eq 1 ]]; then
    effective_config_path="$run_tmp_dir/$(basename "$config_path")"
    patch_config_for_resume "$config_path" "$effective_config_path"
  fi

  echo
  echo "============================================================"
  echo "[RUN $idx] 开始配置: $config_path"
  if [[ "$RESUME_MODE" -eq 1 ]]; then
    echo "effective_config: $effective_config_path (overwrite=false)"
  fi
  echo "project: $project"
  echo "target_host: $target_host"
  echo "port: $port"
  echo "run_log_dir(base): $RUN_LOG_DIR"
  echo "project_log_dir: $RUN_LOG_DIR/project-$project"
  echo "tmp_dir: $run_tmp_dir"
  echo "============================================================"

  if port_in_use "$target_host" "$port"; then
    note="启动前端口 $port 已被占用，环境不干净，停止以避免污染后续实验"
    echo "[ERROR] $note"
    log_result "$idx" "$config_path" "$project" "$target_host" "$port" "blocked" 98 "$RUN_LOG_DIR/project-$project" "$note"
    return 98
  fi

  if [[ "$target_host" = "$local_host" ]]; then
    (
      source "$VENV_ACTIVATE"
      cd "$ROOT_DIR"
      export BENCHMARK_CONFIG_PATH="$benchmark_tmp"
      export DEPLOYMENT_CONFIG_PATH="$deploy_tmp"
      "$PYTHON_BIN" -m vllm_exp.run sweep-test --config "$effective_config_path" --log-dir "$RUN_LOG_DIR" --single-server
    )
    status=$?
  else
    ssh "$target_host" \
      "source '$VENV_ACTIVATE' && cd '$ROOT_DIR' && export BENCHMARK_CONFIG_PATH='$benchmark_tmp' DEPLOYMENT_CONFIG_PATH='$deploy_tmp' && '$PYTHON_BIN' -m vllm_exp.run sweep-test --config '$effective_config_path' --log-dir '$RUN_LOG_DIR' --single-server"
    status=$?
  fi

  if port_in_use "$target_host" "$port"; then
    note="运行结束后端口 $port 仍被占用，可能存在残留服务；为避免污染，建议停止后续实验并人工检查"
    echo "[WARN] $note"
    log_result "$idx" "$config_path" "$project" "$target_host" "$port" "dirty-exit" "$status" "$RUN_LOG_DIR/project-$project" "$note"
    return 97
  fi

  if [[ "$status" -eq 0 ]]; then
    log_result "$idx" "$config_path" "$project" "$target_host" "$port" "success" "$status" "$RUN_LOG_DIR/project-$project" ""
    echo "[OK] 配置执行成功: $config_path"
    return 0
  fi

  note="实验返回非零退出码，但未检测到端口残留"
  log_result "$idx" "$config_path" "$project" "$target_host" "$port" "failed" "$status" "$RUN_LOG_DIR/project-$project" "$note"
  echo "[FAIL] 配置执行失败: $config_path (exit=$status)"
  return "$status"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      shift
      [[ $# -gt 0 ]] || { echo "[ERROR] --config 需要参数" >&2; exit 1; }
      CONFIGS+=("$(resolve_path "$1")")
      ;;
    --run-log-dir)
      shift
      [[ $# -gt 0 ]] || { echo "[ERROR] --run-log-dir 需要参数" >&2; exit 1; }
      RUN_LOG_DIR="$(resolve_path "$1")"
      ;;
    --meta-dir)
      shift
      [[ $# -gt 0 ]] || { echo "[ERROR] --meta-dir 需要参数" >&2; exit 1; }
      META_DIR="$(resolve_path "$1")"
      ;;
    --resume)
      RESUME_MODE=1
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "[ERROR] 必须显式提供至少一个 --config。" >&2
  usage
  exit 1
fi

if [[ -z "$RUN_LOG_DIR" ]]; then
  RUN_LOG_DIR="$ROOT_DIR/new-logs/sequential-run-$TIMESTAMP"
fi

if [[ -z "$META_DIR" ]]; then
  META_DIR="$ROOT_DIR/memory/sequential-run-$TIMESTAMP"
fi

REPORT_PATH="$META_DIR/run-report.md"
SUMMARY_TSV="$META_DIR/summary.tsv"
SQUEUE_SNAPSHOT="$META_DIR/squeue.txt"

for config in "${CONFIGS[@]}"; do
  require_file "$config"
done

mkdir -p "$RUN_LOG_DIR" "$META_DIR"
if command -v squeue >/dev/null 2>&1; then
  squeue --me > "$SQUEUE_SNAPSHOT" 2>&1 || true
fi
append_report_header

exit_code=0
for i in "${!CONFIGS[@]}"; do
  idx=$((i + 1))
  config="${CONFIGS[$i]}"

  run_single_config "$idx" "$config"
  status=$?

  if [[ "$status" -ne 0 ]]; then
    exit_code="$status"
    if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
      echo
      echo "[STOP] 第 $idx 个配置未安全完成，停止后续执行。"
      break
    fi
  fi

  if [[ "$idx" -lt "${#CONFIGS[@]}" ]]; then
    echo "[INFO] 等待 ${SLEEP_BETWEEN_RUNS}s 后继续下一个配置..."
    sleep "$SLEEP_BETWEEN_RUNS"
  fi
done

echo
if [[ "$exit_code" -eq 0 ]]; then
  echo "[DONE] 全部配置安全执行完成。"
else
  echo "[DONE] 批量执行结束，最后非零状态码: $exit_code"
fi

echo "[INFO] vllm_exp 日志根目录: $RUN_LOG_DIR"
echo "[INFO] 脚本元数据目录: $META_DIR"
echo "[INFO] Summary TSV: $SUMMARY_TSV"
echo "[INFO] 报告: $REPORT_PATH"

exit "$exit_code"

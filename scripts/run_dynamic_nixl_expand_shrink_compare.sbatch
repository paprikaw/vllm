#!/usr/bin/env bash
#SBATCH --job-name=vllm-nixl-expand-shrink
#SBATCH --account=punim2715
#SBATCH --qos=publicgpu
#SBATCH --partition=gpu-a100-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2715/vllm_workbench/vllm/new-logs/dynamic-nixl-compare-%j.out
#SBATCH --error=/data/gpfs/projects/punim2715/vllm_workbench/vllm/new-logs/dynamic-nixl-compare-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-/home/bxb1/.codex/worktrees/6e47/vllm}}"
BASE_LOG_DIR="${BASE_LOG_DIR:-$ROOT_DIR/new-logs/dynamic-nixl-compare-${SLURM_JOB_ID:-manual}}"
NCCL_PYTHON_BIN="${NCCL_PYTHON_BIN:-/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/python}"
NCCL_VENV_ACTIVATE="${NCCL_VENV_ACTIVATE:-/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/activate}"
NIXL_PYTHON_BIN="${NIXL_PYTHON_BIN:-/home/bxb1/projects/A100/.venv/bin/python}"
NIXL_VENV_ACTIVATE="${NIXL_VENV_ACTIVATE:-/home/bxb1/projects/A100/.venv/bin/activate}"

mkdir -p "$BASE_LOG_DIR" /data/gpfs/projects/punim2715/vllm_workbench/vllm/new-logs
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "node=$(hostname -f)"
echo "root_dir=$ROOT_DIR"
echo "base_log_dir=$BASE_LOG_DIR"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
date
nvidia-smi -L

module load GCCcore/11.3.0 || true
module load Miniconda3/23.10.0-1 || true
module load NCCL/2.21.5-CUDA-12.4.1 || true

CONFIG_TMP_DIR="$BASE_LOG_DIR/generated-configs"
mkdir -p "$CONFIG_TMP_DIR"
export ROOT_DIR CONFIG_TMP_DIR
"$NCCL_PYTHON_BIN" - <<'PY'
import copy
import os
from pathlib import Path

import yaml

root = Path(os.environ["ROOT_DIR"])
tmp = Path(os.environ["CONFIG_TMP_DIR"])
host = os.environ.get("SLURMD_NODENAME") or os.uname().nodename.split(".")[0]


def load_config(path: Path):
    with path.open() as f:
        return yaml.safe_load(f)


def normalize(cfg: dict, transport: str) -> dict:
    cfg = copy.deepcopy(cfg)
    envs = cfg.setdefault("envs", {})
    envs["VLLM_DYNAMIC_KV_TRANSPORT"] = transport
    if transport == "nixl":
        envs["VLLM_DYNAMIC_PP_TRANSPORT"] = "nixl"
        envs["VLLM_DYNAMIC_PP_NIXL_TRANSPORT"] = "1"
        envs["VLLM_DYNAMIC_PP_NCCL_TRANSPORT"] = "0"
        envs["VLLM_DYNAMIC_PP_RDT_TRANSPORT"] = ""
    else:
        envs["VLLM_DYNAMIC_PP_TRANSPORT"] = ""
        envs["VLLM_DYNAMIC_PP_NIXL_TRANSPORT"] = "0"
        envs["VLLM_DYNAMIC_PP_NCCL_TRANSPORT"] = "1"
        envs["VLLM_DYNAMIC_PP_RDT_TRANSPORT"] = ""

    static = cfg.setdefault("static_config", {})
    network = static.setdefault("network", {})
    placements = network.get("placements") or []
    for placement in placements:
        placement["node"] = host
        placement["ip"] = host
    benchmark = static.setdefault("benchmark", {})
    benchmark["benchmark_script_path"] = str(root / "benchmarks" /
                                            "benchmark_serving.py")
    return cfg


def write_config(name: str, src: Path, transport: str):
    cfg = normalize(load_config(src), transport)
    dst = tmp / f"{name}-{transport}.yaml"
    with dst.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"{name}_{transport}={dst}")
    return dst


expand_src = root / "vllm_exp/configs/qwen_pp4_autoscaling_expand_1to2_smoke.yaml"
shrink_src = root / "vllm_exp/configs/qwen_pp4_autoscaling_shrink_smoke.yaml"
paths = {
    "EXPAND_NCCL_CONFIG_PATH": write_config("expand", expand_src, "nccl"),
    "EXPAND_NIXL_CONFIG_PATH": write_config("expand", expand_src, "nixl"),
    "SHRINK_NCCL_CONFIG_PATH": write_config("shrink", shrink_src, "nccl"),
    "SHRINK_NIXL_CONFIG_PATH": write_config("shrink", shrink_src, "nixl"),
}
with (tmp / "config_paths.env").open("w") as f:
    for key, value in paths.items():
        f.write(f"{key}={value}\n")
PY
source "$CONFIG_TMP_DIR/config_paths.env"

ROOT_DIR="$ROOT_DIR" \
BASE_LOG_DIR="$BASE_LOG_DIR" \
NCCL_PYTHON_BIN="$NCCL_PYTHON_BIN" \
NCCL_VENV_ACTIVATE="$NCCL_VENV_ACTIVATE" \
NIXL_PYTHON_BIN="$NIXL_PYTHON_BIN" \
NIXL_VENV_ACTIVATE="$NIXL_VENV_ACTIVATE" \
EXPAND_NCCL_CONFIG_PATH="$EXPAND_NCCL_CONFIG_PATH" \
EXPAND_NIXL_CONFIG_PATH="$EXPAND_NIXL_CONFIG_PATH" \
SHRINK_NCCL_CONFIG_PATH="$SHRINK_NCCL_CONFIG_PATH" \
SHRINK_NIXL_CONFIG_PATH="$SHRINK_NIXL_CONFIG_PATH" \
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-5400}" \
CLEANUP_ON_EXIT="${CLEANUP_ON_EXIT:-1}" \
  bash scripts/run_dynamic_nixl_expand_shrink_compare.sh --local

if [[ "${CHECK_LOGS:-1}" == "1" ]]; then
  "$NCCL_PYTHON_BIN" scripts/check_dynamic_nixl_compare_logs.py "$BASE_LOG_DIR"
else
  echo "Skipping full comparison log check because CHECK_LOGS=$CHECK_LOGS"
fi

date

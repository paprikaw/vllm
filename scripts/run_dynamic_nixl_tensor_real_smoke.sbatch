#!/usr/bin/env bash
#SBATCH --job-name=vllm-nixl-direct-smoke
#SBATCH --account=punim2715
#SBATCH --qos=publicgpu
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/data/gpfs/projects/punim2715/vllm_workbench/vllm/new-logs/direct-nixl-smoke-%j.out
#SBATCH --error=/data/gpfs/projects/punim2715/vllm_workbench/vllm/new-logs/direct-nixl-smoke-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
A100_VENV_BIN="${A100_VENV_BIN:-/home/bxb1/projects/A100/.venv/bin}"
BASE_PORT="${BASE_PORT:-$((32000 + SLURM_JOB_ID % 20000))}"

mkdir -p "$ROOT_DIR/new-logs"
cd "$ROOT_DIR"

echo "job_id=${SLURM_JOB_ID:-}"
echo "node=$(hostname -f)"
echo "root_dir=$ROOT_DIR"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
date
nvidia-smi -L

module load GCCcore/11.3.0 || true
module load Miniconda3/23.10.0-1 || true
module load NCCL/2.21.5-CUDA-12.4.1 || true
export PATH="$A100_VENV_BIN:$PATH"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
hash -r

which python
python - <<'PY'
import importlib.util
import sys

print("python", sys.executable)
for name in ["torch", "nixl", "nixl_cu12", "vllm"]:
    spec = importlib.util.find_spec(name)
    print(name, "FOUND" if spec else "missing", spec.origin if spec else "")

import torch
import nixl

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("nixl", nixl.__file__)
PY

python verification_tests/dynamic_nixl_tensor_real.py \
  --root-dir "$ROOT_DIR" \
  --use-cuda \
  --base-port "$BASE_PORT" \
  --sender-cuda-visible-devices 0 \
  --receiver-cuda-visible-devices 1 \
  --quick-exit

date

#!/usr/bin/env bash
set -euo pipefail

NODE0="${NODE0:-spartan-gpgpu066}"
NODE1="${NODE1:-spartan-gpgpu007}"

for node in "${NODE0}" "${NODE1}"; do
  echo "== ${node} =="
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${node}" \
    'hostname -f; nvidia-smi -L; python - <<'"'"'PY'"'"'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
PY'
done

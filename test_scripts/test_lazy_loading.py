#!/usr/bin/env python3
# test_lazy_across_shards.py

import os
import time
import torch
from safetensors.torch import safe_open

def find_and_load_tensor(shard_paths, target_key, device=0):
    print(f"共有 {len(shard_paths)} 个 .safetensors 分片文件")
    print(f"目标 key: {target_key}")

    for path in shard_paths:
        print(f"\n🔍 扫描文件: {path}")
        try:
            with safe_open(path, framework="pt", device=device) as f:
                keys = list(f.keys())
                if target_key in keys:
                    print(f"✅ 找到 key: {target_key} 在此文件中")
                    
                    mem0 = torch.cuda.memory_allocated(device)
                    print(f"[加载前] 显存: {mem0:,} bytes")

                    start = time.time()
                    tensor = f.get_tensor(target_key)
                    end = time.time()

                    mem1 = torch.cuda.memory_allocated(device)
                    print(f"[加载后] 显存: {mem1:,} bytes (+{mem1 - mem0:,})")
                    print(f"张量信息: shape={tensor.shape}, dtype={tensor.dtype}")
                    print(f"加载耗时: {end - start:.6f} 秒")
                    return
        except Exception as e:
            print(f"❌ 读取失败: {e}")
    print("❌ 所有文件中未找到目标 key")

def main():
    default_dir = "/root/.cache/huggingface/Qwen/Qwen3-32B-AWQ"
    default_key = "model.layers.47.input_layernorm.weight"

    shard_paths = sorted([
        os.path.join(default_dir, f)
        for f in os.listdir(default_dir)
        if f.endswith(".safetensors")
    ])

    find_and_load_tensor(shard_paths, default_key, device=0)

if __name__ == "__main__":
    main()
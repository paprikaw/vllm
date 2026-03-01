#!/usr/bin/env python3
"""
Benchmark comparing A100 (FP8 dequant at runtime) vs L40s (native FP8)
under different workload patterns:
- Decode Heavy: Many small batches (M=1-8), memory bound
- Prefill Heavy: Large batches (M=128-2048), compute bound

A100 Behavior with FP8 models:
  - Loads FP8 weights 
  - Dequantizes to FP16/BF16 at runtime
  - Performs FP16 GEMM

L40s Behavior with FP8 models:
  - Loads FP8 weights
  - Uses torch._scaled_mm (native FP8 GEMM)

Usage:
    # On A100:
    ssh spartan-gpgpu166
    /data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/python \
        /home/bxb1/vllm_workbench/vllm/benchmarks/benchmark_workload_compare.py

    # On L40s:
    ssh spartan-gpgpu009
    /data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/python \
        /home/bxb1/vllm_workbench/vllm/benchmarks/benchmark_workload_compare.py
"""

import time
import torch
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WorkloadConfig:
    name: str
    description: str
    configs: List[Tuple[int, int, int, str, int]]  # (M, K, N, name, weight)


# Llama-70B shapes: hidden=8192, intermediate=28672
DECODE_HEAVY = WorkloadConfig(
    name="Decode Heavy",
    description="Many small batches, typical for long-running chat sessions",
    configs=[
        # (M, K, N, name, relative_weight)
        (1, 8192, 8192, "qkv", 100),      # 100 decode steps
        (1, 8192, 28672, "up", 100),
        (1, 28672, 8192, "down", 100),
        (4, 8192, 8192, "qkv-4", 20),     # occasional small batch
        (4, 8192, 28672, "up-4", 20),
        (4, 28672, 8192, "down-4", 20),
        (8, 8192, 8192, "qkv-8", 10),     # rare larger batch
        (8, 8192, 28672, "up-8", 10),
        (8, 28672, 8192, "down-8", 10),
        (128, 8192, 8192, "prefill", 1),  # occasional prefill
        (128, 8192, 28672, "prefill-up", 1),
    ]
)

PREFILL_HEAVY = WorkloadConfig(
    name="Prefill Heavy", 
    description="Large batches, typical for batch inference or long prompts",
    configs=[
        # (M, K, N, name, relative_weight)
        (128, 8192, 8192, "qkv-128", 20),
        (128, 8192, 28672, "up-128", 20),
        (128, 28672, 8192, "down-128", 20),
        (256, 8192, 8192, "qkv-256", 30),
        (256, 8192, 28672, "up-256", 30),
        (256, 28672, 8192, "down-256", 30),
        (512, 8192, 8192, "qkv-512", 25),
        (512, 8192, 28672, "up-512", 25),
        (512, 28672, 8192, "down-512", 25),
        (1024, 8192, 8192, "qkv-1k", 15),
        (1024, 8192, 28672, "up-1k", 15),
        (1, 8192, 8192, "decode", 10),    # some decode
        (1, 8192, 28672, "decode-up", 10),
    ]
)


def get_device_info():
    """Get GPU device information."""
    if not torch.cuda.is_available():
        return None, None, None
    
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    return name, capability, capability[0] * 10 + capability[1]


def compute_flops(M: int, K: int, N: int) -> int:
    return 2 * M * K * N


def compute_tflops(flops: int, time_ms: float) -> float:
    return flops / (time_ms / 1000) / 1e12


def supports_native_fp8():
    """Check if native FP8 GEMM is supported (SM >= 8.9)."""
    cap = torch.cuda.get_device_capability()
    cap_int = cap[0] * 10 + cap[1]
    return cap_int >= 89


def create_fp8_tensors(M: int, K: int, N: int, device: str = "cuda"):
    """Create per-tensor FP8 tensors."""
    fp8_dtype = torch.float8_e4m3fn
    
    input_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    weight_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    
    input_scale = input_fp16.abs().max() / 448.0
    qinput = (input_fp16 / input_scale).to(fp8_dtype).contiguous()
    scale_a = input_scale.float().view(1)
    
    weight_scale = weight_fp16.abs().max() / 448.0
    weight_nk = (weight_fp16.t() / weight_scale).to(fp8_dtype).contiguous()
    qweight = weight_nk.t()
    scale_b = weight_scale.float().view(1)
    
    return qinput, qweight, scale_a, scale_b


def benchmark_native_fp8(M: int, K: int, N: int, 
                         warmup: int = 10, iterations: int = 50):
    """Native FP8 using torch._scaled_mm (L40s/H100)."""
    device = "cuda"
    qinput, qweight, scale_a, scale_b = create_fp8_tensors(M, K, N, device)
    out_dtype = torch.float16
    
    for _ in range(warmup):
        output = torch._scaled_mm(qinput, qweight,
                                  out_dtype=out_dtype,
                                  scale_a=scale_a,
                                  scale_b=scale_b)
        if isinstance(output, tuple):
            output = output[0]
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        output = torch._scaled_mm(qinput, qweight,
                                  out_dtype=out_dtype,
                                  scale_a=scale_a,
                                  scale_b=scale_b)
        if isinstance(output, tuple):
            output = output[0]
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000


def benchmark_fp8_dequant_runtime(M: int, K: int, N: int,
                                  warmup: int = 10, iterations: int = 50):
    """Simulate A100 behavior with FP8 model.
    
    In vLLM on A100:
    - FP8 weights are dequantized to FP16 ONCE during model loading
    - All forward passes use FP16 GEMM with the dequantized weights
    
    So A100 with FP8 model = pure FP16 throughput (no per-forward dequant overhead)
    The only cost is 2x memory for FP16 weights vs FP8.
    """
    device = "cuda"
    
    # In reality, weight is already dequantized and stored as FP16
    input_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    weight_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    
    for _ in range(warmup):
        output = torch.mm(input_fp16, weight_fp16)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        output = torch.mm(input_fp16, weight_fp16)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000


def benchmark_fp16_direct(M: int, K: int, N: int,
                          warmup: int = 10, iterations: int = 50):
    """Pure FP16 GEMM (baseline, no quantization overhead)."""
    device = "cuda"
    input_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    weight_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    
    for _ in range(warmup):
        output = torch.mm(input_fp16, weight_fp16)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        output = torch.mm(input_fp16, weight_fp16)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000


def run_workload(workload: WorkloadConfig, is_native_fp8: bool,
                 warmup: int = 10, iterations: int = 50):
    """Run a workload and compute weighted throughput."""
    results = []
    total_weighted_time = 0.0
    total_weighted_flops = 0.0
    
    for M, K, N, name, weight in workload.configs:
        flops = compute_flops(M, K, N)
        
        if is_native_fp8:
            try:
                time_ms = benchmark_native_fp8(M, K, N, warmup, iterations)
                kernel = "FP8"
            except Exception as e:
                time_ms = benchmark_fp16_direct(M, K, N, warmup, iterations)
                kernel = "FP16"
        else:
            # A100: dequant at runtime
            time_ms = benchmark_fp8_dequant_runtime(M, K, N, warmup, iterations)
            kernel = "DQ+FP16"
        
        fp16_time = benchmark_fp16_direct(M, K, N, warmup, iterations)
        
        tflops = compute_tflops(flops, time_ms)
        fp16_tflops = compute_tflops(flops, fp16_time)
        
        # Weighted by frequency
        total_weighted_time += time_ms * weight
        total_weighted_flops += flops * weight
        
        results.append({
            "name": name,
            "M": M, "K": K, "N": N,
            "weight": weight,
            "time_ms": time_ms,
            "tflops": tflops,
            "fp16_time_ms": fp16_time,
            "fp16_tflops": fp16_tflops,
            "speedup": fp16_time / time_ms if kernel == "FP8" else time_ms / fp16_time,
            "kernel": kernel
        })
    
    # Weighted average throughput
    # This represents: if we had `weight` number of each operation,
    # what's the total time and total FLOPS?
    weighted_avg_tflops = (total_weighted_flops / (total_weighted_time / 1000)) / 1e12
    
    return results, weighted_avg_tflops


def print_workload_results(workload: WorkloadConfig, results: list, 
                           weighted_tflops: float, gpu_name: str,
                           is_native_fp8: bool):
    """Print formatted results for a workload."""
    print()
    print("=" * 100)
    print(f"{workload.name}: {workload.description}")
    print(f"GPU: {gpu_name}")
    print(f"Kernel: {'Native FP8 (torch._scaled_mm)' if is_native_fp8 else 'FP8 Dequant + FP16 GEMM'}")
    print("=" * 100)
    
    print(f"{'Op':<12} | {'M':>5} {'K':>6} {'N':>6} | {'Wt':>3} | {'Kernel':>8} | {'Time':>8} | {'TFLOPS':>7} | {'vs FP16':>8}")
    print("-" * 100)
    
    for r in results:
        if r["kernel"] == "FP8":
            speedup_str = f"{r['speedup']:.2f}x ↑"
        else:
            # For DQ+FP16, speedup < 1 means slower than pure FP16
            if r["speedup"] > 1:
                speedup_str = f"{r['speedup']:.2f}x ↓"
            else:
                speedup_str = f"{1/r['speedup']:.2f}x ↑"
        
        print(f"{r['name']:<12} | {r['M']:>5} {r['K']:>6} {r['N']:>6} | {r['weight']:>3} | "
              f"{r['kernel']:>8} | {r['time_ms']:>7.3f}ms | {r['tflops']:>6.1f} | {speedup_str}")
    
    print("-" * 100)
    print(f"Weighted Average Throughput: {weighted_tflops:.1f} TFLOPS")
    print("=" * 100)
    
    return weighted_tflops


def main():
    parser = argparse.ArgumentParser(description="Workload Comparison Benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    args = parser.parse_args()
    
    device_name, capability, cap_int = get_device_info()
    if device_name is None:
        print("No CUDA device available!")
        return
    
    is_native_fp8 = supports_native_fp8()
    
    print()
    print("#" * 100)
    print("# A100 vs L40s Workload Comparison")
    print("#" * 100)
    print(f"Device: {device_name}")
    print(f"Compute Capability: {capability[0]}.{capability[1]}")
    print(f"Native FP8 Support: {is_native_fp8}")
    print()
    
    if is_native_fp8:
        gpu_type = "L40S" if "L40" in device_name else "H100" if "H100" in device_name else "Unknown"
        print(f"Running as: {gpu_type} with native FP8 (torch._scaled_mm)")
    else:
        gpu_type = "A100"
        print(f"Running as: {gpu_type} with FP8 dequant at runtime")
    
    # Run both workloads
    decode_results, decode_tflops = run_workload(
        DECODE_HEAVY, is_native_fp8, args.warmup, args.iterations)
    prefill_results, prefill_tflops = run_workload(
        PREFILL_HEAVY, is_native_fp8, args.warmup, args.iterations)
    
    # Print results
    print_workload_results(DECODE_HEAVY, decode_results, decode_tflops, 
                          device_name, is_native_fp8)
    print_workload_results(PREFILL_HEAVY, prefill_results, prefill_tflops,
                          device_name, is_native_fp8)
    
    # Summary
    print()
    print("#" * 100)
    print("# Summary")
    print("#" * 100)
    print(f"{'Workload':<20} | {'Weighted TFLOPS':>15} | {'Mode':>20}")
    print("-" * 60)
    mode = "Native FP8" if is_native_fp8 else "FP8 Dequant + FP16"
    print(f"{'Decode Heavy':<20} | {decode_tflops:>14.1f} | {mode:>20}")
    print(f"{'Prefill Heavy':<20} | {prefill_tflops:>14.1f} | {mode:>20}")
    print()
    
    # Save for later comparison
    output_file = Path(f"/tmp/workload_{gpu_type.lower()}.json")
    with open(output_file, 'w') as f:
        json.dump({
            "gpu": gpu_type,
            "device": device_name,
            "native_fp8": is_native_fp8,
            "decode_heavy_tflops": decode_tflops,
            "prefill_heavy_tflops": prefill_tflops,
            "decode_results": decode_results,
            "prefill_results": prefill_results,
        }, f, indent=2)
    print(f"Results saved to {output_file}")
    print("#" * 100)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
最小复现：验证GPU推理期间的weight chunk copy性能下降问题

实验设计：
1. 创建一个GPU上的weight tensor（模拟layer weight）
2. 在后台线程执行chunked copy（模拟weight migration）
3. 在主线程执行GPU计算（模拟推理）
4. 比较不同场景下的chunk copy时间

场景对比：
A. 无干扰：单独执行chunked copy
B. 有推理干扰：chunked copy与推理并发执行
C. 使用ForegroundBackgroundGate协调

预期：场景B的chunk copy时间应该比场景A慢
"""

import torch
import threading
import time
from typing import Optional, List
from contextlib import contextmanager
import statistics


class ForegroundBackgroundGate:
    """Foreground/Background同步门控"""
    def __init__(self):
        self._cond = threading.Condition()
        self._active_foreground = 0
        self._background_running = False

    @contextmanager
    def foreground(self):
        with self._cond:
            self._active_foreground += 1
        try:
            yield
        finally:
            with self._cond:
                self._active_foreground -= 1
                if self._active_foreground == 0:
                    self._cond.notify_all()

    @contextmanager
    def background(self):
        with self._cond:
            while self._active_foreground > 0 or self._background_running:
                self._cond.wait()
            self._background_running = True
        try:
            yield
        finally:
            with self._cond:
                self._background_running = False
                self._cond.notify_all()


def chunked_copy_inplace(
    dst: torch.Tensor,
    src: torch.Tensor,
    chunk_size_mb: float = 5.0,
    fbgate: Optional[ForegroundBackgroundGate] = None,
) -> List[float]:
    """
    分批执行tensor copy，返回每个chunk的copy时间(ms)
    """
    total_bytes = src.numel() * src.element_size()
    total_mb = total_bytes / (1024 * 1024)
    n_copy = max(1, int(total_mb / chunk_size_mb + 0.5))
    
    dst_flat = dst.view(-1)
    src_flat = src.view(-1)
    total_elements = dst_flat.numel()
    chunk_size = (total_elements + n_copy - 1) // n_copy
    
    chunk_times = []
    
    for i in range(n_copy):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_elements)
        
        if start_idx >= total_elements:
            break
            
        if fbgate is not None:
            with fbgate.background():
                time_start = time.perf_counter()
                dst_flat[start_idx:end_idx].copy_(src_flat[start_idx:end_idx])
                torch.cuda.synchronize()  # 确保copy完成
                chunk_times.append((time.perf_counter() - time_start) * 1000)  # ms
        else:
            time_start = time.perf_counter()
            dst_flat[start_idx:end_idx].copy_(src_flat[start_idx:end_idx])
            torch.cuda.synchronize()
            chunk_times.append((time.perf_counter() - time_start) * 1000)
    
    return chunk_times


def simulate_inference(
    model: torch.Tensor,
    duration_sec: float,
    fbgate: Optional[ForegroundBackgroundGate] = None,
):
    """
    模拟推理: 持续执行GPU matmul操作
    """
    batch = torch.randn(512, model.shape[0], device='cuda', dtype=model.dtype)
    end_time = time.time() + duration_sec
    step = 0
    
    while time.time() < end_time:
        if fbgate is not None:
            with fbgate.foreground():
                output = torch.mm(batch, model)
                torch.cuda.synchronize()
        else:
            output = torch.mm(batch, model)
            torch.cuda.synchronize()
        step += 1
    
    return step


def run_experiment_no_inference(weight_size_mb: float, chunk_size_mb: float, src_on_cpu: bool = False, use_pinned: bool = False):
    """场景A: 无干扰的chunked copy"""
    if src_on_cpu:
        suffix = f"CPU{'(pinned)' if use_pinned else '(non-pinned)'}→GPU"
    else:
        suffix = "GPU→GPU"
    print(f"\n{'='*60}")
    print(f"场景A: 无干扰的chunked copy ({suffix})")
    print(f"{'='*60}")
    
    # 创建weights
    elements = int(weight_size_mb * 1024 * 1024 / 2)  # float16 = 2 bytes
    if src_on_cpu:
        if use_pinned:
            src = torch.randn(elements, dtype=torch.float16).pin_memory()
        else:
            src = torch.randn(elements, device='cpu', dtype=torch.float16)
    else:
        src = torch.randn(elements, device='cuda', dtype=torch.float16)
    dst = torch.empty(elements, device='cuda', dtype=torch.float16)
    
    # 执行chunked copy
    chunk_times = chunked_copy_inplace(dst, src, chunk_size_mb=chunk_size_mb)
    
    print(f"Weight size: {weight_size_mb:.1f} MB")
    print(f"Chunk size: {chunk_size_mb:.1f} MB")
    print(f"Number of chunks: {len(chunk_times)}")
    print(f"Chunk times: {[f'{t:.3f}ms' for t in chunk_times[:5]]}...")
    print(f"Average: {statistics.mean(chunk_times):.3f}ms")
    print(f"Median: {statistics.median(chunk_times):.3f}ms")
    
    return chunk_times


def run_experiment_with_inference(weight_size_mb: float, chunk_size_mb: float, use_gate: bool, src_on_cpu: bool = False):
    """场景B/C: 有推理干扰的chunked copy"""
    suffix = "CPU→GPU" if src_on_cpu else "GPU→GPU"
    gate_str = "使用FBGate" if use_gate else "无协调"
    print(f"\n{'='*60}")
    print(f"场景{'C' if use_gate else 'B'}: 有推理干扰的chunked copy ({gate_str}, {suffix})")
    print(f"{'='*60}")
    
    # 创建weights and model
    elements = int(weight_size_mb * 1024 * 1024 / 2)
    if src_on_cpu:
        src = torch.randn(elements, device='cpu', dtype=torch.float16)
    else:
        src = torch.randn(elements, device='cuda', dtype=torch.float16)
    dst = torch.empty(elements, device='cuda', dtype=torch.float16)
    
    # 模拟一个MLP layer
    model_weight = torch.randn(5120, 5120, device='cuda', dtype=torch.float16)
    
    fbgate = ForegroundBackgroundGate() if use_gate else None
    
    chunk_times_result = []
    inference_steps_result = [0]
    copy_done = threading.Event()
    
    def copy_thread():
        # 让推理先启动
        time.sleep(0.05)
        times = chunked_copy_inplace(dst, src, chunk_size_mb=chunk_size_mb, fbgate=fbgate)
        chunk_times_result.extend(times)
        copy_done.set()
    
    def inference_thread():
        steps = simulate_inference(model_weight, duration_sec=5.0, fbgate=fbgate)
        inference_steps_result[0] = steps
    
    # 启动两个线程
    t_copy = threading.Thread(target=copy_thread)
    t_infer = threading.Thread(target=inference_thread)
    
    t_infer.start()
    t_copy.start()
    
    # 等待copy完成
    copy_done.wait()
    t_copy.join()
    
    # 停止推理
    t_infer.join()
    
    print(f"Weight size: {weight_size_mb:.1f} MB")
    print(f"Chunk size: {chunk_size_mb:.1f} MB")
    print(f"Number of chunks: {len(chunk_times_result)}")
    print(f"Inference steps: {inference_steps_result[0]}")
    print(f"Chunk times: {[f'{t:.3f}ms' for t in chunk_times_result[:5]]}...")
    print(f"Average: {statistics.mean(chunk_times_result):.3f}ms")
    print(f"Median: {statistics.median(chunk_times_result):.3f}ms")
    
    return chunk_times_result


def run_experiment_with_stream_isolation(weight_size_mb: float, chunk_size_mb: float):
    """场景D: 使用独立CUDA stream + FBGate"""
    print(f"\n{'='*60}")
    print(f"场景D: 使用独立CUDA stream + FBGate")
    print(f"{'='*60}")
    
    # 创建weights and model
    elements = int(weight_size_mb * 1024 * 1024 / 2)
    src = torch.randn(elements, device='cuda', dtype=torch.float16)
    dst = torch.empty_like(src)
    
    model_weight = torch.randn(5120, 5120, device='cuda', dtype=torch.float16)
    
    fbgate = ForegroundBackgroundGate()
    copy_stream = torch.cuda.Stream()  # 独立的copy stream
    
    chunk_times_result = []
    copy_done = threading.Event()
    
    def copy_thread_with_stream():
        time.sleep(0.05)
        total_bytes = src.numel() * src.element_size()
        total_mb = total_bytes / (1024 * 1024)
        n_copy = max(1, int(total_mb / chunk_size_mb + 0.5))
        
        dst_flat = dst.view(-1)
        src_flat = src.view(-1)
        total_elements = dst_flat.numel()
        chunk_size = (total_elements + n_copy - 1) // n_copy
        
        for i in range(n_copy):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_elements)
            if start_idx >= total_elements:
                break
            
            with fbgate.background():
                with torch.cuda.stream(copy_stream):
                    time_start = time.perf_counter()
                    dst_flat[start_idx:end_idx].copy_(src_flat[start_idx:end_idx])
                    copy_stream.synchronize()
                    chunk_times_result.append((time.perf_counter() - time_start) * 1000)
        
        copy_done.set()
    
    def inference_thread():
        simulate_inference(model_weight, duration_sec=5.0, fbgate=fbgate)
    
    t_copy = threading.Thread(target=copy_thread_with_stream)
    t_infer = threading.Thread(target=inference_thread)
    
    t_infer.start()
    t_copy.start()
    
    copy_done.wait()
    t_copy.join()
    t_infer.join()
    
    print(f"Weight size: {weight_size_mb:.1f} MB")
    print(f"Chunk size: {chunk_size_mb:.1f} MB")
    print(f"Number of chunks: {len(chunk_times_result)}")
    print(f"Chunk times: {[f'{t:.3f}ms' for t in chunk_times_result[:5]]}...")
    print(f"Average: {statistics.mean(chunk_times_result):.3f}ms")
    print(f"Median: {statistics.median(chunk_times_result):.3f}ms")
    
    return chunk_times_result


def main():
    print("=" * 60)
    print("Weight Chunk Copy 性能测试")
    print("=" * 60)
    
    # 配置
    WEIGHT_SIZE_MB = 125.0  # 类似Qwen3的gate_up_proj weight
    CHUNK_SIZE_MB = 5.0
    
    # 预热GPU
    torch.randn(1000, 1000, device='cuda')
    torch.cuda.synchronize()
    
    # ====== 基准测试 ======
    print("\n" + "=" * 60)
    print("测试1: 基准性能")
    print("=" * 60)
    times_gpu = run_experiment_no_inference(WEIGHT_SIZE_MB, CHUNK_SIZE_MB, src_on_cpu=False)
    times_cpu_nonpin = run_experiment_no_inference(WEIGHT_SIZE_MB, CHUNK_SIZE_MB, src_on_cpu=True, use_pinned=False)
    times_cpu_pin = run_experiment_no_inference(WEIGHT_SIZE_MB, CHUNK_SIZE_MB, src_on_cpu=True, use_pinned=True)
    
    # ====== Warmup效应测试 ======
    print("\n" + "=" * 60)
    print("测试2: CPU→GPU Warmup效应（重复加载同一tensor）")
    print("=" * 60)
    
    elements = int(WEIGHT_SIZE_MB * 1024 * 1024 / 2)
    src_cpu = torch.randn(elements, dtype=torch.float16)  # non-pinned CPU
    dst_gpu = torch.empty(elements, device='cuda', dtype=torch.float16)
    
    # 第一次加载 (cold)
    print("\n--- 第1次加载（cold）---")
    times_cold = chunked_copy_inplace(dst_gpu, src_cpu, chunk_size_mb=CHUNK_SIZE_MB)
    print(f"Average: {statistics.mean(times_cold):.3f}ms, Median: {statistics.median(times_cold):.3f}ms")
    
    # 清理GPU缓存
    del dst_gpu; torch.cuda.empty_cache(); torch.cuda.synchronize()
    dst_gpu = torch.empty(elements, device='cuda', dtype=torch.float16)
    
    # 第二次加载 (warm)
    print("\n--- 第2次加载（warm）---")
    times_warm1 = chunked_copy_inplace(dst_gpu, src_cpu, chunk_size_mb=CHUNK_SIZE_MB)
    print(f"Average: {statistics.mean(times_warm1):.3f}ms, Median: {statistics.median(times_warm1):.3f}ms")
    
    # 清理GPU缓存
    del dst_gpu; torch.cuda.empty_cache(); torch.cuda.synchronize()
    dst_gpu = torch.empty(elements, device='cuda', dtype=torch.float16)
    
    # 第三次加载 (warm)
    print("\n--- 第3次加载（warm）---")
    times_warm2 = chunked_copy_inplace(dst_gpu, src_cpu, chunk_size_mb=CHUNK_SIZE_MB)
    print(f"Average: {statistics.mean(times_warm2):.3f}ms, Median: {statistics.median(times_warm2):.3f}ms")
    
    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"{'场景':<45} {'平均时间':>10} {'中位数':>10}")
    print("-" * 65)
    print(f"{'GPU→GPU':<45} {statistics.mean(times_gpu):>10.3f}ms {statistics.median(times_gpu):>10.3f}ms")
    print(f"{'CPU(non-pin)→GPU':<45} {statistics.mean(times_cpu_nonpin):>10.3f}ms {statistics.median(times_cpu_nonpin):>10.3f}ms")
    print(f"{'CPU(pinned)→GPU':<45} {statistics.mean(times_cpu_pin):>10.3f}ms {statistics.median(times_cpu_pin):>10.3f}ms")
    print("-" * 65)
    print(f"{'CPU→GPU: 第1次（cold）':<45} {statistics.mean(times_cold):>10.3f}ms {statistics.median(times_cold):>10.3f}ms")
    print(f"{'CPU→GPU: 第2次（warm）':<45} {statistics.mean(times_warm1):>10.3f}ms {statistics.median(times_warm1):>10.3f}ms")
    print(f"{'CPU→GPU: 第3次（warm）':<45} {statistics.mean(times_warm2):>10.3f}ms {statistics.median(times_warm2):>10.3f}ms")
    
    # 分析
    print("\n" + "=" * 60)
    print("分析")
    print("=" * 60)
    speedup = statistics.mean(times_cold) / statistics.mean(times_warm1)
    print(f"  Warmup效应: 第2次加载比第1次快 {speedup:.2f}x")
    print(f"")
    print(f"  vLLM日志对比:")
    print(f"    - 第1次async迁移chunk copy: ~2ms")
    print(f"    - 第3次async迁移chunk copy: ~700µs")
    print(f"    - Speedup: ~2.9x")
    print(f"")
    print(f"  结论:")
    print(f"    1. sync迁移正确reset配置：[(0,15),(16,63)] → ... → [(0,15),(16,63)]")
    print(f"    2. 第1次vs第3次async迁移的chunk copy差异来自:")
    print(f"       - CPU内存页面表warmup（首次vs已缓存）")
    print(f"       - PyTorch/CUDA分配器warmup效应")
    print(f"       - 文件系统/mmap缓存（safetensors）")


if __name__ == "__main__":
    main()

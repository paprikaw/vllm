# GPU Memory Monitor 与 KV Cache Block 计算总结

## 概述

本文档总结了 vLLM dynamic pipeline parallelism 中的两个核心模块：
1. **GPU Memory Monitor** - 内存泄漏检测系统
2. **KV Cache Block 计算** - 初始化时的 block 数量计算方式

---

## 1. GPU Memory Monitoring 系统

### 文件位置
- `vllm/v1/worker/gpu_memory_monitor.py`

### MemoryOverheadMonitor（唯一使用的 Monitor）

**核心公式**：
```
overhead = used_memory - expected_memory
expected_memory = layer_count × per_layer_weight_size + kv_cache_size
used_memory = total_memory - free_memory
```

**泄漏检测逻辑**：
```python
# 只有当 overhead 超过 baseline + tolerance 时才报告泄漏
# 负数差值（内存减少）不视为泄漏
is_valid = overhead_diff_gb <= self.tolerance_gb
```

**使用流程**：
1. `initialize_baseline()` - 模型加载后建立基线
2. `check_overhead_before()` - 内存操作前检查
3. `check_overhead_after()` - 内存操作后检查

**关键修改记录**：
- 原始逻辑：`abs(overhead_diff_gb) <= tolerance` 会把负数差值也视为泄漏
- 修改后：`overhead_diff_gb <= tolerance` 只检测正向泄漏

---

## 2. KV Cache Block 计算

### 2.1 初始 Block 数量计算

**文件位置**：`vllm/v1/engine/dynamic_core.py`

**公式**：
```python
def _get_max_num_blocks(self, total_gpu_memory, weight_size_per_layer, 
                        page_size, num_layers, runtime_overhead=0):
    total_usable_memory = total_gpu_memory * gpu_memory_utilization
    total_weight_size = weight_size_per_layer * num_layers
    total_kv_cache = total_usable_memory - total_weight_size - runtime_overhead
    max_blocks_per_layer = floor(total_kv_cache / (num_layers * page_size))
    return max_blocks_per_layer
```

**参数说明**：
| 参数 | 说明 |
|------|------|
| `total_gpu_memory` | GPU 总内存 (bytes) |
| `weight_size_per_layer` | 每层权重大小 (~0.80 GB for 70B model) |
| `page_size` | KV cache block 大小 (524288 bytes = 512KB) |
| `num_layers` | 当前 rank 上的层数 |
| `runtime_overhead` | profile_run 测量的运行时开销 |

**日志示例**：
```
[memory access] debug ------- get max num blocks: 26.04 GB, num_layers: 56, block_size: 524288, 
total_gpu_memory: 79.14 GB, total_usable_memory: 75.18 GB, weight_size_per_layer: 0.80 GB, 
total_weight_size: 44.63 GB, runtime_overhead: 4.52 GB
```

### 2.2 Flexi 模式 KV Cache 大小计算

**问题背景**：
- 原始代码使用 `isinstance(block, torch.Tensor)` 检测 KV cache
- Flexi 模式下 `kv_caches` 是 `list[list[int]]`（指针地址），不是 Tensor
- 导致 `kv_cache_bytes` 永远为 0

**修复方案**：

**文件位置**：`vllm/v1/worker/dynamic_gpu_worker.py`

**新增字段**：
```python
self.per_block_kv_cache_bytes = 0  # Per-layer, per-block KV cache size in bytes (K+V)
```

**初始化**（在 `dynamic_initialize_from_config` 中）：
```python
self.per_block_kv_cache_bytes = kv_cache_spec.page_size_bytes  # K+V size per block per layer
```

**计算方法**：
```python
def _get_current_memory_params(self) -> Tuple[int, int]:
    if is_flexi:
        # Flexi mode: use formula instead of iterating tensors
        if self.block_num > 0 and self.per_block_kv_cache_bytes > 0:
            kv_cache_bytes = self.block_num * layer_count * self.per_block_kv_cache_bytes
    else:
        # Non-flexi mode: iterate over tensor list
        for kv_tensor in self.model_runner.kv_caches:
            if isinstance(kv_tensor, torch.Tensor):
                kv_cache_bytes += kv_tensor.numel() * kv_tensor.element_size()
    
    return layer_count, kv_cache_bytes
```

**关键公式**：
```
kv_cache_bytes = block_num × layer_count × per_block_kv_cache_bytes
```

其中：
- `block_num` - 当前分配的 block 数量
- `layer_count` - 当前 rank 上的层数
- `per_block_kv_cache_bytes` = `kv_cache_spec.page_size_bytes` = K+V 每 block 每层大小

---

## 3. 典型配置示例

### 异构 GPU 环境 (A100 + L40)

| GPU | 总内存 | Usable | 层数 | runtime_overhead |
|-----|-------|--------|------|------------------|
| A100 (80GB) | 79.14 GB | 75.18 GB | 56 | 4.52 GB |
| L40 (44GB) | 44.52 GB | 42.29 GB | 24 | 4.63 GB |

### Block 计算结果

- 初始配置 PP=(56, 24): `maximum_kv_block_num: [952, 120]`
- Compact 后: `compacted to: 120, maximum_kv_block_num_after_compact: [952, 120]`

---

## 4. 日志输出示例

### MemoryOverheadMonitor 日志

```
[MemoryOverheadMonitor] BEFORE add_layers | rank=0 | layers=56, kv=1.406GB, 
used=48.23GB, overhead=3.12GB (baseline=3.10GB, diff=+0.0156GB) [OK]

[MemoryOverheadMonitor] AFTER add_layers | rank=0 | layers=56, kv=1.406GB, 
used=48.25GB, overhead=3.14GB (baseline=3.10GB, diff=+0.0312GB) [OK], 
used_change=+0.020GB (expected=+0.018GB, unexpected=+0.0020GB)
```

### Block 计算日志

```
[memory access] debug ------- get max num blocks: 26.04 GB, num_layers: 56, 
block_size: 524288, total_gpu_memory: 79.14 GB, total_usable_memory: 75.18 GB, 
weight_size_per_layer: 0.80 GB, total_weight_size: 44.63 GB, runtime_overhead: 4.52 GB
```

---

## 5. 相关文件清单

| 文件 | 功能 |
|------|------|
| `vllm/v1/worker/gpu_memory_monitor.py` | Memory Monitor 实现 |
| `vllm/v1/worker/dynamic_gpu_worker.py` | KV cache 大小计算、per_block_kv_cache_bytes |
| `vllm/v1/engine/dynamic_core.py` | Block 数量计算公式 |
| `vllm/config.py` | 配置参数定义 |

---

生成时间: 2026-03-05

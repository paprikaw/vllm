# Combined Layers 实现报告

## 概述

本次实现了 **combined_layers** 功能，允许多个 transformer 层共享同一个 VMM 物理块。这解决了 VMM 2MB 最小分配粒度导致的内部碎片问题。

### 问题背景
- VMM 最小分配粒度为 2MB
- 原本每层单独分配 2MB 块 → 每块可存 512 tokens
- 当 `layer_group_granularity=4` 时 → 4层共享一个2MB块 → 每块可存 128 tokens
- **自动计算粒度**：`layer_group_granularity=0`（默认）会根据 block_size 自动计算最优值

### 公式
```
granularity = (2MB / bytes_per_token_kv) / block_size
            = 512 / block_size  (对于典型配置: 8 heads × 128 dim × bf16)

例：block_size=128 → granularity=4 → 4层共享一个2MB块
```

## 修改的文件

### 1. vllm/config.py
在 `DynamicConfig` 中添加：

```python
layer_group_granularity: int = 0
"""Number of layers to pack into a single VMM physical block.
- 0 (default): Auto-calculate based on block_size to fill 2MB VMM blocks
- 1: Disabled (each layer gets separate VMM allocation)
- >1: Manually specify grouping
"""

use_vmm: bool = True
"""Whether to use CUDA Virtual Memory Management (VMM) for KV cache allocation.
- True (default): Use VMM for efficient memory management with 2MB blocks
- False: Use cudaMallocAsync instead (has memory leak, for testing only)
"""
```

### 2. vllm_exp/data.py (实验框架)
添加配置支持：

| 类 | 字段 | 说明 |
|---|---|---|
| `SweepVllmParams` | `use_vmm: Optional[List[bool]]` | 支持 sweep |
| `StaticVllmCfg` | `use_vmm: bool = True` | 静态配置 |
| `ExpVllmConfig` | `use_vmm: bool = True` | 实验配置 |
| `VllmServerSpec` | `use_vmm: bool = True` | 服务器规格 |

同时更新了 `get_sweep_axes()`, `_create_from_combo()`, `to_dynamic_cfg()`。

### 3. csrc/kv_cache_allocator_optimized.cpp
新增两个 C++ 函数：

**allocate_with_cuda_vmm_combined_layers()**
- 为多层分配共享的 VMM 块
- 返回每层的 K/V 指针（指向同一物理块的不同偏移）
- 内存布局: `[K0][V0][K1][V1][K2][V2][K3][V3]`

**free_vmm_blocks_combined_layers()**
- 释放 combined_layers 模式分配的内存
- 遍历所有 groups 释放

### 4. vllm/kv_allocator.py
添加 Python wrapper 函数：
- `allocate_with_cuda_vmm_combined_layers()`
- `free_vmm_blocks_combined_layers()`

### 5. vllm/v1/worker/dynamic_gpu_model_runner.py

**dynamic_initialize_kv_cache_flexi()**
- **Auto-calculate granularity**: 当 `layer_group_granularity=0` 时自动计算最优值
- **use_vmm check**: 如果 `use_vmm=False`，强制 granularity=1 使用 cudaMallocAsync
- 验证层数整除性
- 存储 `grouped_handles`, `layer_group_granularity`, `vmm_bytes_per_kv`

**其他函数更新**:
- `flexi_atomic_switch_kv_cache_config_for_layers()` - 更新 grouped_handles
- `_migrate_block_by_swapping_ptrs()` - 交换时同步 grouped_handles

### 6. vllm/v1/worker/dynamic_gpu_worker.py

**代码重构** - `_flexi_resize_kv_cache` 从 580+ 行重构为清晰的结构：

| 函数 | 作用 |
|------|------|
| `_flexi_resize_kv_cache()` | 主入口，dispatcher（~40行）|
| `_get_flexi_resize_context()` | 获取共享上下文 |
| `_log_memory_metrics()` | 统一的内存监控日志 |
| `_do_flexi_shrink()` | shrink 主逻辑 |
| `_free_shrunk_caches()` | 释放缓存（修复了所有 groups 释放） |
| `_do_flexi_grow()` | grow 主逻辑 |
| `_grow_combined_layers_mode()` | combined layers 分配 |
| `_grow_per_layer_mode()` | per-layer 分配 |

**其他更新**:
- `atomic_shelve_kv_cache()` - 收集 `grouped_handles_to_free`
- `release_kv_cache_for_layers()` - 使用 `free_vmm_blocks_combined_layers`

## 关键数据结构

```python
# grouped_handles[group_idx][block_idx] - 按组存储的 VMM handles
grouped_handles: list[list[int]]

# 每组包含的层数
layer_group_granularity: int

# 每层单个 K 或 V 的字节数
vmm_bytes_per_kv: int

# VMM 对齐字节数（2MB）
vmm_aligned_bytes: int

# 是否使用 combined_layers 模式
use_combined_layers: bool  # = (layer_group_granularity > 1 and grouped_handles exists)
```

## 使用方式

### 方式1：自动计算（推荐）
```python
DynamicConfig(
    use_flexi_kv=True,
    layer_group_granularity=0,  # 自动计算最优值
    block_size=128,
    ...
)
```

### 方式2：手动指定
```python
DynamicConfig(
    use_flexi_kv=True,
    layer_group_granularity=4,  # 4层共享一个2MB块
    ...
)
```

### 方式3：禁用 VMM（测试内存泄漏）
```python
DynamicConfig(
    use_flexi_kv=True,
    use_vmm=False,  # 使用 cudaMallocAsync（有内存泄漏）
    ...
)
```

### 实验框架配置（vllm_exp）
```yaml
vllm:
  use_vmm: true
  layer_group_granularity: 0  # auto
  block_size: 128
```

## 约束条件

1. `num_layers % layer_group_granularity == 0` - 总层数必须是 granularity 的整数倍
2. 所有增删操作必须以 granularity 为单位
3. 仅在 `use_flexi_kv=True` 时生效
4. 当 `use_vmm=False` 时，强制 granularity=1

## 已修复的 Bug

### [2026-03-04] combined_layers 模式内存泄漏
**症状**: shrink 后只释放了约 1/4 内存
```
expected_freed=9130.00 MB, actual_freed=830.00 MB, leak=8300.00 MB
```

**原因**: `_free_shrunk_caches()` 只释放了 `old_grouped_handles[0]`，漏掉其他 groups

**修复**: 遍历所有 groups 释放：
```python
for group_idx, group_handles in enumerate(old_grouped_handles):
    # 释放每个 group 的 handles
    kv_allocator.free_vmm_blocks_combined_layers(...)
```

## 内存监控

添加了 Memory Monitor 日志用于追踪内存使用：
```
[Memory Monitor SHRINK] rank=0, blocks_shrinked=415, layers=44, 
    expected_freed_MB=9130.00, actual_freed_MB=9100.00, leak_MB=30.00

[Memory Monitor GROW] rank=0, blocks_allocated=100, layers=44,
    expected_allocated_MB=2200.00, actual_allocated_MB=2280.00, overhead_MB=80.00
```

# 2026-03-20 FP8 Marlin allocator 崩溃修复记录

## 现象
- `project-kernal_cmp_7` / `flash rr=3 in=512 out=16 vmm=1 cpu_cache=1` 会在 rank 1 崩溃。
- 典型栈：
  - `c10::cuda::CUDACachingAllocator::Native::DeviceCachingAllocator::malloc()`
  - `_custom_ops.py:gptq_marlin_gemm`
  - `marlin_utils_fp8.py:apply_fp8_marlin_linear`
- 后续会级联成：
  - `Fatal Python error: Segmentation fault`
  - `ActorDiedError`
  - `EngineDeadError`
  - peer rank recv 0 bytes

## 根因判断
- 不是普通 CUDA OOM。
- 更像是 **FP8 Marlin linear 输出张量反复临时分配** 导致 allocator churn / 内存损坏暴露。
- 触发路径在高并发 distributed forward 中更明显。

## 错误修复尝试
### 尝试 1：每层持久 output buffer
- 在 layer 上挂 `_fp8_marlin_output_buffer`，并把 `c=output` 传给 `gptq_marlin_gemm`。
- 结果：避免了内部临时分配，但在 startup `profile_run()` 时造成显存暴涨。
- 新错误：`torch.OutOfMemoryError`。
- 原因：每个 FP8 Marlin layer 都常驻一份大 output buffer。

### 尝试 2：runner/forward 级共享 workspace（最终方案）
- 复用 `ForwardContext.workspace_buffers`，参考 `SiluAndMul` 的实现模式。
- 在 `apply_fp8_marlin_linear()` 中：
  - 以 `fp8_marlin_linear_{size_n}` 为 key 从共享 workspace 取 buffer
  - 不足时扩容
  - 将切片后的 output 通过 `c=output` 传给 `gptq_marlin_gemm`
- 这样只保留 **forward 级共享临时 buffer**，而不是每层持久 buffer。

## 修改文件
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`
- `vllm/model_executor/layers/quantization/fbgemm_fp8.py`
- `vllm/model_executor/layers/quantization/fp8.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py`

## 验证
### 最小复现（166 本地双 rank）
- 配置：`/tmp/tmp_flash_rr3_segfault_20260320.yaml`
- 结果：`100/100 successful requests`
- 无：`SIGSEGV` / `Fatal Python error` / `ActorDiedError` / `OutOfMemoryError`

### 全量 kernel cmp 验证
- 基于 `vllm_exp/configs/a-kernal_cmp.yaml`
- 结果：`6 experiments, All succeeded`
- 包含之前高风险 case：`direct rr=3 in=512 out=16 vmm=1`

### 140 集群复验
- 单机 `140->140` 因资源不足无法建 2-GPU placement group，不属于代码回归。
- 改用 `140 + 001` 两节点配置：
  - `memory/tmp_flash_rr3_segfault_20260320_on140_cluster.yaml`
- 结果：`1 experiments, All succeeded`
- `100 successful requests`

## 结论
- 原始崩溃的直接修复点是：
  **让 FP8 Marlin linear 输出走 forward 级共享 workspace，而不是每次临时分配，也不是每层常驻分配。**
- 该修复已在 166 与 140 集群上完成验证。

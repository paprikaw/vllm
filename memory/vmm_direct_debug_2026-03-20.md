# VMM + direct 性能回退调试记录（2026-03-20）

## 结论摘要
- 目前证据表明：**问题不在 `flexi_direct_varlen_fwd` 这个纯 attention kernel 本身**。
- 用独立微基准分别测试了：
  - async vs VMM 分配
  - direct vs flexi
  - decode / prefill
  - MHA / GQA
  - 单层 / 40层顺序调用
- 这些纯 kernel 场景下，**direct+VMM 没有复现显著回退**，有些场景甚至更快。
- 真正的回退出现在 **服务端 `execute_model()` 的集成路径**，不是裸 kernel。

## 关键证据
### 1. 端到端日志
在 `project-kernal_cmp_debug_1` 中：
- direct + vmm=1:
  - request TTFT ≈ 885.5 ms
  - TPOT ≈ 167.85 ms
  - E2EL ≈ 3403.23 ms
- direct + vmm=0:
  - request TTFT ≈ 624.2 ms
  - TPOT ≈ 121.38 ms
  - E2EL ≈ 2444.96 ms

### 2. 纯 kernel 微基准
对 `flexi_flash_attn_varlen_func` / `flexi_direct_flash_attn_varlen_func` 直接调用，
使用 `KVAllocator` 分别分配 async/VMM 页，结果：
- 单层 decode / prefill：无 direct+VMM 回退
- GQA (`num_heads=32, num_heads_k=4`)：无 direct+VMM 回退
- 40 层顺序调用：无 direct+VMM 回退

这说明：
- VMM 地址布局本身 **不是** direct attention kernel 变慢的根因。
- `block_table -> ptr_table` 的 direct 访存形式 **不是**根因。

### 3. execute_model 集成阶段变慢
在 `server.log` 中聚合 `[forward]: execute time`：
- direct_vmm1: avg ≈ 0.0745 s
- direct_vmm0: avg ≈ 0.0620 s
- flash_vmm1: avg ≈ 0.0631 s
- flash_vmm0: avg ≈ 0.0621 s

=> **只有 direct + VMM 在完整执行路径里明显变慢**。

### 4. upstream / NCCL 读阶段也被连带放大
在分析报告中：
- `Communication Time from Upstream`
  - direct_vmm1 ≈ 2.88 ms
  - direct_vmm0 ≈ 2.16 ms
  - flash_vmm1 ≈ 1.85 ms
  - flash_vmm0 ≈ 1.87 ms
- `[RAY.NCCLGROUP.Read] Received Data From Comm Time`
  - direct_vmm1 明显高于 direct_vmm0

说明 direct+VMM 会让 pipeline 下游等待更久，但这更像是 **上游 execute 路径变慢后的连锁反应**，不是通信栈本身独立变慢。

## 当前最可信判断
### 根因定位
- **不是 allocator 初始化后的裸 kernel 性能问题**。
- **是 direct 模式完整推理集成路径中的某个 GPU 异步工作，在 VMM 打开后变慢，并被记账到 `execute_model()` / pipeline 等待中。**

### 特别重要的点
`_dynamic_prepare_inputs()` 中的：
- `block_table.commit(num_reqs)`
- `update_ptr_table_from_block_table(...)`

虽然日志里 `kv ptr_tables update took 60~100µs`，但这只是 **CPU launch 时间**，没有同步，因此**不代表真实 GPU 执行耗时**。

## 下一步建议
1. 在 `DynamicGPUModelRunner._execute_model()` 内部加同步计时，把以下阶段拆开：
   - `_dynamic_prepare_inputs()`
   - `sync_and_slice_intermediate_tensors()`
   - `with set_forward_context(...): self.model(...)`
   - `maybe_wait_for_kv_save()`
2. 对 `update_ptr_table_from_block_table()` 单独做同步计时，确认真实 GPU 时间。
3. 如果 `_dynamic_prepare_inputs()` 仍然很小，则继续细分 `self.model(...)` 内 attention 前后的时间。

## 暂时排除项
- `flexi_direct_varlen_fwd` 裸 kernel 本身
- direct ptr_table 的单层地址解析
- VMM 地址对单层 direct kernel 的基础访存效率影响
- 40层顺序 direct attention 的纯 kernel 累积回退

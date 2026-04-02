# 2026-03-22 KV patch 链路优化：预生成缓冲

## 背景
在 `rr=3` 的最小复现实验里，receiver ready 之后仍有较长的 patch phase。代码检查发现 sender 端当前流程是：
1. 发完整 KV tensor
2. 等 receiver `notify_for_kv_patch()`
3. 才开始 `get_kv_patch()` 做 slot mapping 收集和 KV gather
4. 再发送 patch

这意味着 **patch 生成完全落在 receiver ready 之后的暴露阶段**。

## 本次修改
- 在 `dynamic_kv_synchronizer.py` 中加入 `buffer_kv_patches()` / `get_buffered_kv_patch()`。
- sender 在发完 full KV tensor 后，立即启动一个后台线程预生成 patch，并放入已有 `KVPatchBuffer`。
- sender 仍然会等待 receiver ready，但 ready 之后改为直接消费 buffer 中已生成的 patch 发送。
- 为避免 buffered 路径提前把 `kv_cache_transfer_in_process` 置回 `False`，给 `_get_patch()` / `_flexi_get_patch()` 增加了 `finalize_transfer` 开关：
  - 直接发送路径：`True`
  - 预生成缓冲路径：`False`
  - 最终由 `_buffer_get_kv_patch()` 在消费到 `kv_patch_finished` 时统一 reset 状态。

## 预期收益
把 patch gather / patch materialization 与 receiver 的 weight loading / bind 阶段重叠，缩短 receiver ready 之后的 exposed patch 阶段。

## 已验证
- `py_compile` 通过：
  - `vllm/distributed/kv_transfer/kv_connector/dynamic_kv_synchronizer.py`
  - `vllm/v1/worker/dynamic_gpu_worker.py`
- 静态类型错误基本都是文件内原有问题，新增修改未引入新的语法错误。

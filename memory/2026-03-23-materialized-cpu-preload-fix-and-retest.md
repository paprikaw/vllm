# 2026-03-23 materialized CPU preload fix and retest

## 修改
- 文件: `vllm/model_executor/model_loader/dynamic_qwen3_loader.py`
- 变更: `_preload_all_weights()` 不再把 `safe_open(...).get_tensor(name)` 返回的 mmap/file-backed tensor 直接放进 `_preloaded_weights`。
- 新逻辑: 对每个 tensor 执行真实 CPU 拷贝：
  - `materialized_param = torch.empty_like(param, device="cpu")`
  - `materialized_param.copy_(param)`
  - 缓存 `materialized_param`
- 保留 `madvise` 作为源 mmap 的预取提示，但缓存对象本身已经变成真实 CPU resident tensor。

## 复测配置
- 新配置: `vllm_exp/configs/debug-async-weight-loading-outlier-min-materialized-preload.yaml`
- 节点: `spartan-gpgpu140` + `spartan-gpgpu001`
- 运行命令:
  - `python -m vllm_exp.run sweep-test --config vllm_exp/configs/debug-async-weight-loading-outlier-min-materialized-preload.yaml --log-dir /home/bxb1/vllm_workbench/vllm/logs --single-server`

## preload 开销
- rank0/140: 67.67 GB, 约 61.3s
- rank1/001: 67.67 GB, 约 67.5s
- 第二组 sweep 也类似，约 67-85s
- 结论: 修复显著增加 server 启动阶段的 preload 时间与 CPU RAM 占用，这是预期 tradeoff。

## 关键复测结果
### 48 -> 32
- 第一轮 target migration:
  - `load_weights_took=1.4s`
  - `process_after_loading_summary=93ms`
- 第二轮另一侧:
  - `load_weights_took=756ms`
  - `process_after_loading_summary=126ms`
- 第二次 repetition:
  - `load_weights_took=1.8s`
- 不再出现旧结果里的 `25.0s` 异常尖峰。

### 56 -> 24
- 第一轮 target migration:
  - `load_weights_took=2.8s`
  - `process_after_loading_summary=177ms`
- 另一侧:
  - `load_weights_took=1.6s`
  - `process_after_loading_summary=250ms`
- 第二次 repetition:
  - `load_weights_took=4.7s`
  - `process_after_loading_summary=158ms`
- 不再出现旧结果里的 `41.4s` 异常尖峰；16 层总体显著高于 8 层，趋势恢复接近单调。

## 与修复前对比
- 修复前:
  - 48->32 第一轮 `load_weights_took=25.0s`
  - 56->24 第一轮 `load_weights_took=41.4s`
- 修复后:
  - 48->32 约 `0.76s ~ 1.8s`
  - 56->24 约 `1.6s ~ 4.7s`
- 结论: 主要异常已消失，说明根因确实是 mmap-backed 假 CPU cache 导致的首次 page fault/H2D 抖动。

## 结论
- 真正 materialize CPU preload 后，async migration 的 weight loading 不再受冷页 page fault 严重污染。
- migration time 与 layer 数的关系恢复正常得多。
- 代价是启动时要付出一次性大约 68 GB/worker 的 CPU 内存和约 1 分钟级别 preload 时间。

# 2026-03-19 bind_kv_cache 慢路径分析

## 结论
在 async migration 过程中，`bind_kv_cache` 的 10.1s 不是单纯分配内存导致：

1. 预分配只用了约 1.5s。
2. 真正的慢点是逐层 bind 时，与在线 inference 并发执行，且 bind 路径没有显式使用独立 migration stream，`stream=None` 会落到 default stream。
3. 同时每层 bind 后都会执行 `gc.collect()` + `torch.cuda.empty_cache()`，这会进一步放大同步/串行化成本。

## 关键证据
- 日志：pre-allocation done in 1.5s
- 日志：bind kv cache time taken: 10.1s
- 日志中 bind 期间大量出现：`[forward]: inference stream synchronize time: 0.23 seconds`
- 接收侧 bind 调用：`dynamic_flexi_bind_single_kv_tensor(..., stream=None, ...)`
- `apply_kv_tensor_to_allocated_cache(..., stream=None)` 内部 `with torch.cuda.stream(stream):`，因此实际落在 default stream。
- worker 明确有高优先级 `inference_stream`，并在 forward 末尾 `self.worker.inference_stream.synchronize()`。

## 涉及代码
- `vllm/v1/worker/dynamic_gpu_worker.py`
- `vllm/v1/utils.py`
- `vllm/v1/worker/dynamic_gpu_model_runner.py`
- `vllm/v1/executor/dynamic_utils.py`

## 推测的主要瓶颈
- bind kernel (`flexi_reshape_and_cache_flash`) 与 pointer tensor 的 H2D 拷贝跑在 default stream。
- 在线推理跑在高优先级 `inference_stream`。
- default stream / empty_cache 使迁移与推理形成频繁同步，导致每几层就出现一次 ~0.23s 的等待。

## 后续建议
1. 接收侧 bind 显式传入 `self.migration_stream`。
2. 避免每层都 `gc.collect()` / `torch.cuda.empty_cache()`；改为批量或末尾做一次。
3. 给 `apply_kv_tensor_to_allocated_cache`、`create_ptr_tensor_from_list` 增加分段计时，确认真实占比。

# 2026-03-22 async vs sync migration throughput debug

## 问题
用户观察到在 `36-44 -> 52-28`、`mig=1`、`n_req=200`、`chunk=40`、`kv_resize=1`、`cpu_cache=1` 配置下，`sync` 的总 token throughput 比 `async` 高约 4~5%，与“async 可以边迁移边推理，理论上应更快”的直觉不符。

## 核心结论
1. **用户看到的“5 秒左右”只对应 stop-the-world / engine lock 时间，不等于 async migration 的完整生命周期。**
   - sync 的 stop time 确实约 5.2~6.1s。
   - async 的 engine lock 只有 0.34~0.43ms，但完整 migration process 是 **12.9s / 14.7s / 14.4s / 22.5s / 14.5s**。
2. **async 不是“免费 overlap”**。它虽然几乎消掉 stop time，但把代价转移成了：
   - 后台 weight loading
   - receiver 等待 layer ready + bind KV
   - patch catch-up / lag threshold 等待
   - 这些过程与在线 decode 共享 GPU copy / HBM / NCCL / stream / forward gate 资源。
3. **这组 workload 下，真正拖慢吞吐的不是 async 的 engine lock，而是 migration overlap 期间 decode 被拖慢。**
   - 从 `timestamp_metrics.csv` 统计高并发窗口（`running_reqs >= 70`）的 generation throughput：
     - async ≈ **515.93 tok/s**
     - sync ≈ **543.52 tok/s**
     - async 低 **5.08%**，与用户看到的总吞吐差几乎一致。
4. **async 的关键瓶颈是 sender/receiver 的 patch-ready 链路太长，导致新配置真正生效很晚。**
   - sender 在 `_sender_loop()` 里 `wait_for_kv_patch()`。
   - receiver 在 `_listen_loop()` 里必须等 `has_layer()`（weight loading 完成）后才能 bind KV，再 `notify_for_kv_patch()`。
   - 因此日志里的 `wait for kv patch preparation` 本质上不是“纯 patch 构造”，而是 sender 在等 receiver 完成 weight load + bind + ready。

## 关键证据
### benchmark 结果
- sync total token throughput: **826.17 tok/s**
- async total token throughput: **790.25 tok/s**
- 差距约 **4.35%**。

### async analysis
- migration process total: **12.9s / 14.7s / 14.4s / 22.5s / 14.5s**
- wait for kv patch preparation: **3.9s / 4.7s / 5.3s / 17.1s / 3.8s**
- after weight loading 平均约 **4.48s**，且有一次达到 **16.7s**
- stop_time async engine_lock_total: **0.34~0.43ms**

### sync analysis
- stop_time sync engine_lock_total: **5.23s ~ 6.12s**
- drain: **约 1.4s**
- after weight loading 平均约 **2.02s**

## 代码路径定位
- `change_model_configuration_by_kv_transfer_async()`：`vllm/vllm/v1/engine/dynamic_core.py`
  - 先异步 `async_add_layers()`、`start_kv_cache_migration_async()`
  - 再轮询 `sync_by_checking_leftover_tokens()` 决定何时真正 `async_change_configuration()`
- `_add_layers()`：`vllm/vllm/v1/worker/dynamic_gpu_worker.py`
  - weight loading 在 migration stream 上执行
  - 会消耗 GPU 资源，即便 forward_lock 持有时间很短
- `_sender_loop()`：`vllm/vllm/v1/worker/dynamic_gpu_worker.py`
  - 发送完 full KV tensor 后等待 `wait_for_kv_patch()`
- `_listen_loop()`：`vllm/vllm/v1/worker/dynamic_gpu_worker.py`
  - 等 layer loaded -> bind KV -> notify_for_kv_patch
  - 所以 receiver ready 是 async 的真正关键路径之一

## 解释框架
- **sync**：一次较大的停顿，迁移后 decode 路径更干净。
- **async**：几乎无停顿，但迁移后台工作与 decode 并发，导致 decode token 速率下降；同时新配置切换还要等待 patch lag 条件满足，因此 migration 全生命周期反而更长。

## 最终判断
这次不是“async 实现失效”，而是 **async overlap 的收益 < overlap 带来的 decode 干扰成本**，并且 async 的“完整迁移完成时间”远大于用户直觉里的 5 秒 stop time。

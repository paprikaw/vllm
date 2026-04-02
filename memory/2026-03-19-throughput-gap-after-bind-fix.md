# 2026-03-19 吞吐差距继续分析

## 结论
- 当前 `n_req=200, chunk=160` 的 migration 实验已经不再是 `bind_kv_cache` 主导。
- 与静态目标配置 `pp=40-24` 的同 workload 基线相比：
  - migration: `Duration=88.75s`, `Request throughput=2.25 req/s`
  - static target: `Duration=82.23s`, `Request throughput=2.43 req/s`
- 差距约 `6.5s`，与 migration timeline 的 `5.1s` 基本同量级，说明当前剩余吞吐损失主要就是 migration 自身开销，而不是 steady-state token 速度明显退化。

## 关键观察
- 当前 mixed workload 的 Stage 2 本身就是 `128 -> 512`，所以总生成 token 很大；吞吐下降不能简单归因于 bind。
- 当前 migration timeline：
  - `After Weight Loading = 2.90s`
  - `Receive KV Tensor = 973ms`
  - `Bind KV Cache = 447ms`
  - `Wait For KV Patch Preparation = 2.1s`
  - `After Receiving KV Cache Patches = 240ms`
- 其中日志里的 `wait for kv patch preparation` 实际上是 sender 在等 receiver `notify_for_kv_patch()`，本质上反映 receiver 侧还没完成“层可用 + bind 完成”的准备。

## 推断
- 现在最大头部瓶颈更像是：
  1. weight loading 完成前 receiver 不能 ready
  2. bind 后仍有一次全局 `torch.cuda.synchronize()`，会把不相关 stream 也一起等掉
- 由于 bind 已经切到 `migration_stream`，bind 完成后只需要保证 migration stream 上的 bind 写入完成，不需要再做 device-wide synchronize。

## 已做的下一步优化
- 将 receiver 在 bind 完成后的 `torch.cuda.synchronize()` 改为：
  - 优先 `self.migration_stream.synchronize()`
  - 仅在没有 migration stream 时 fallback 到 `torch.cuda.synchronize()`
- 同时增加了 post-bind synchronize 的独立 timeline 日志，便于下一轮验证这部分是否仍然有额外等待。

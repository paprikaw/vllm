# 2026-03-19 migration 期间推理性能分析

## 结论
migration 现在不是“完全同步停机”，而是“旧请求仍在 decode，但 decode 吞吐明显下降，同时新配置 worker 的 ready 路径仍然偏串行”。因此用户看到：
- migration timeline 约 5.1s
- 总实验时长也差不多多了 5~6s

本质上是 migration overlap 很弱，且 overlap 期间推理性能显著退化。

## 证据
### 1. 高并发 generation 吞吐明显低于静态目标配置
- migration 实验高并发窗口 generation throughput 大约在 971~1086 tok/s。
- 静态目标配置相同 workload 的对应窗口是 1256~1401 tok/s。
- 说明 migration 期间 decode 吞吐下降约 20%~30%。

### 2. Stage 2 长输出请求的 token 速度变慢
- migration 的 Stage 2 长输出请求样本里，TPOT 大约 82~89ms。
- 静态目标配置对应样本 TPOT 大约 68~72ms。
- 这是约 20%~25% 的 per-token 退化。

### 3. Stage 2 的首 token 与端到端延迟也被拉长
- migration: P99 TTFT = 786ms
- static target: P99 TTFT = 343ms
- migration 中首批 Stage 2 请求出现 600ms~1.0s 的 TTFT spike。

### 4. receiver ready 路径仍然比较串
当前 timeline:
- After Weight Loading = 2.90s
- Receive KV Tensor = 973ms
- Bind KV Cache = 447ms
- Wait For KV Patch Preparation = 2.1s

这里的 `wait for kv patch preparation` 更准确地说，是 sender 在等 receiver 完成 layer ready + bind 完成 + notify ready，不是纯 patch 构造本身慢。

## 解释
1. bind 本身已经不是大头了。
2. 真正的问题是 migration 期间，weight loading / KV recv / bind / patch 与在线 decode 共享 GPU copy / HBM / NCCL 资源。
3. 所以系统没有停，但 decode token rate 明显下来了。
4. 同时 receiver 侧 ready 仍要等一长条关键路径走完，导致 overlap 不充分。

## 下一步方向
1. 优先继续压缩 receiver ready 路径（尤其是 weight loading -> ready）。
2. 研究能否让 patch phase 更早开始，而不是等 bind 全部完成后才 notify。
3. 限制 migration 期间 patch/gather 的批量，减少对 decode 的带宽竞争。
4. 如果需要，再单独打点 migration window 内的 decode step latency / prefill latency。

# 2026-03-20 migration 吞吐低于静态 36-28 的原因

## 结论
在 `project-config-cmp-with-mig-qwen-debug` 中，`mig=1, 16-48 -> 36-28` 的吞吐低于静态 `36-28`，主要原因不是 steady-state decode 能力不足，而是 **一次性 migration wall-clock 开销约 6.2s** 直接拉长了 benchmark 总时长。

- migration 配置：Duration 83.58s，Request throughput 2.39 req/s，Output token throughput 601.03 tok/s，Mean TTFT 217.95ms，Mean ITL 90.95ms
- 静态 36-28：Duration 77.48s，Request throughput 2.58 req/s，Output token throughput 646.75 tok/s，Mean TTFT 708.90ms，Mean ITL 91.40ms
- 两者时长差：`83.58 - 77.48 = 6.10s`，与 analysis 中 `MIGRATION PROCESS TOTAL TIME = 6.2s` 几乎一致。
- 若把 migration 的 6.2s 一次性税去掉：
  - `200 / (83.58 - 6.2) = 2.585 req/s`，几乎等于静态 36-28 的 2.581 req/s
  - `50237 / (83.58 - 6.2) = 649.22 tok/s`，略高于静态 36-28 的 646.79 tok/s

## 解释
migration 让系统先享受 `16-48` 的 prompt/TTFT 优势，再切到 `36-28` 的 decode/ITL/吞吐优势，因此平均 TTFT/ITL 看起来更优；但 benchmark throughput 是按 **总完成时间** 算的，所以一次性迁移成本会完整计入分母。

## 主要开销拆分
来自 analysis：
- Migration process total: 6.2s
- After weight loading: 3.40s
- Wait for KV patch preparation: 2.6s
- Receive KV tensor total: 1.3s
- Bind KV cache total: 592ms
- Resize KV cache: 平均 1.16s（4 次）

## 优化方向
1. 提前加载目标层权重，消掉 3.4s
2. 提前准备 KV patches，缩短 2.6s wait
3. 压缩/并行化 KV tensor receive + bind
4. 更早触发 migration，让一次性税更多落在低并发阶段
5. 对长跑 benchmark，把迁移 warmup 从统计窗口中剔除

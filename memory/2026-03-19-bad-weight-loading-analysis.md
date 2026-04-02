# 2026-03-19 慢 weight loading 分析

## 现象
- 本次实验 `chunk=160`，analysis 显示：
  - After Weight Loading = 10.4s
  - Receive KV Tensor = 3.9s
  - Wait For KV Patch Preparation = 7.5s
- receiver 在 22:45:43 就收完了 KV tensor，但直到 22:45:50 才完成 weight loading，22:45:51 才 bind 完并 notify ready。

## 关键原因
1. `weight_chunk_size_mb=160` 太大，导致 125MB 大权重不再 chunked copy，而是直接整块 copy。
   - `linear.py` 中阈值是 `min_size_for_chunking = chunk_size_mb * 2`。
   - 160MB 配置下阈值变成 320MB，所以 125MB 权重都走 direct copy。
2. weight loading 与在线 inference / compiled DAG / NCCL 写操作强烈重叠。
   - 日志里在 weight loading 中间持续出现 compiled DAG write / NCCL send queued。
3. 因此很多 `Loaded weight for ... took XXXms` 不是 copy 本身慢，而是包含了等待 GPU/stream/通信资源的时间。
   - 典型例子：
     - `layers.4.input_layernorm.weight` 本身 loader 只要 82µs，但总耗时 127ms。
     - `layers.4.self_attn.q_proj.weight_scale_inv` 实际 direct copy 只有 62µs，但总耗时 307ms。
     - `layers.28.self_attn.v_proj.weight` 实际 direct copy 729µs，但总耗时 221ms，并且紧接着出现 NCCL/CompiledDAG 写日志。
4. `wait for kv patch preparation = 7.5s` 实际上基本是在等 receiver 的 weight loading + bind 完成，不是 patch 生成本身慢。

## 判断
- 这次慢的根因不是 safetensors 读取本身坏掉，而是：
  - 大 chunk 让大权重失去 chunked/background copy 的温和行为；
  - 同时 migration 与在线推理/通信重叠，导致 weight loading 大量时间消耗在等待共享 GPU/通信资源。
- 次要可能性：CPU cache 虽然开启，但当前实现是 preloaded tensor cache，不是真正固定驻留的 pinned staging buffer，因此小 tensor 仍可能出现页/缓存抖动；但主因仍是 overlap contention。

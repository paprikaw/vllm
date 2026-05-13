# Qwen e2e rr=5: 为什么 Prefill-Optimal throughput 低，但 TTFT/TPOT 也低

日期：2026-04-01

## 关键结论
- `12-52`（Prefill-Optimal）在 **prefill-heavy** 请求上非常强，但会把 **decode-heavy** 请求拖成很长的 decode tail，因此总体 makespan 最长，导致 throughput 最低。
- 图里的 `TTFT` / `TPOT` 是 **request-level mean**，而 throughput 是 `(total_input + total_output) / total_duration`，受 **整个 run 的完成时间** 支配。
- 在 mixed workload (`[[512,16],[128,512]]`, `running_num_requests=[100,100]`) 下，`12-52` 的低平均 TTFT / TPOT 并不意味着高 throughput，因为真正拖慢 run 的是长输出请求的累计 decode 时间。
- `PipeLive (12-52 -> 36-28)` 基本保留了 `12-52` 的 prefill-heavy 性能，同时恢复了 `36-28` 的 decode-heavy 吞吐能力，因此 overall score 最优。

## 数据（Qwen, rr=5）
### Overall
- Static `12-52`
  - TTFT = 200.84 ms
  - TPOT = 143.87 ms
  - Throughput = 1146.69 tok/s
  - total_time = 99.58 s
- Static `36-28`
  - TTFT = 1115.05 ms
  - TPOT = 406.36 ms
  - Throughput = 1479.21 tok/s
  - total_time = 77.27 s
- PipeLive `12-52 -> 36-28`
  - TTFT = 214.31 ms
  - TPOT = 124.24 ms
  - Throughput = 1441.64 tok/s
  - total_time = 79.26 s

### Per-workload split
#### Static `12-52`
- prefill-heavy:
  - TTFT = 243.05 ms
  - TPOT = 170.44 ms
  - E2E = 2799.59 ms
- decode-heavy:
  - TTFT = 158.64 ms
  - TPOT = 117.30 ms
  - E2E = 57242.93 ms

#### Static `36-28`
- prefill-heavy:
  - TTFT = 1795.55 ms
  - TPOT = 734.95 ms
  - E2E = 12819.85 ms
- decode-heavy:
  - TTFT = 434.56 ms
  - TPOT = 77.76 ms
  - E2E = 38360.64 ms

#### PipeLive `12-52 -> 36-28`
- prefill-heavy:
  - TTFT = 244.50 ms
  - TPOT = 166.43 ms
  - E2E = 2741.00 ms
- decode-heavy:
  - TTFT = 184.12 ms
  - TPOT = 82.04 ms
  - E2E = 39886.81 ms

## 可写进 paper 的解释
`12-52` minimizes prefill cost and therefore achieves the best request-level TTFT. However, under the mixed workload, it leaves insufficient decode capacity for long-generation requests, which creates a long decode tail and stretches the end-to-end completion time of the whole run. Since total token throughput is determined by the aggregate token volume divided by the full experiment makespan, `12-52` ends up having the lowest throughput despite its low average TTFT and TPOT. In contrast, PipeLive starts from the prefill-optimal split to preserve low first-token latency, then reconfigures to the decode-optimal split to shorten the long decode tail. This allows PipeLive to retain near-prefill-optimal TTFT while recovering almost all of the decode-optimal throughput, leading to the best overall performance.

## 补充
- benchmark 中 `ttft` / `tpot` / `e2els` 的定义在 `benchmarks/benchmark_serving.py` 与 `benchmarks/backend_request_func.py`。
- Throughput 使用全 run 时长 `dur_s` 计算，而不是由平均 TTFT / TPOT 推导。

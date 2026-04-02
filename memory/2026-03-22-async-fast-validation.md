# 2026-03-22 async-fast 实验核查

## 目标
确认 `new-logs/project-async-fast_5` 中 `async` / `async_fast` / `sync` 三种模式是否真实执行，而不是因为实验失败导致结果相似。

## 核查结论
- 三种模式都跑到了各自的 mode-specific stop-time 路径：
  - `async` -> `[STOP_TIME] ASYNC MODE - FULL BREAKDOWN`
  - `async_fast` -> `[STOP_TIME] ASYNC-FAST MODE - FULL BREAKDOWN`
  - `sync` -> `[STOP_TIME] SYNC MODE - FULL BREAKDOWN`
- 每个测试目录都有 `request_metrics.csv`，且均为 1000 行，对应 `rep=5` 且每次 200 request，说明主 benchmark 已完整产出。
- rr=2.5 的关键差异明显：
  - async: migration total `32.8s`，但 stop time 约 `10.09ms`
  - async_fast: migration total `5.7s`，stop time 约 `2248.78ms`
  - sync: weight loading 平均 `1.56s`，stop time 约 `4060.40ms`
- 因此“模式没有生效”这个判断不成立；更可能是 notebook 画的是 TTFT / TPOT / throughput 这类**全程平均指标**，把单次迁移的 stop-time 差异摊薄了。

## 额外发现
- `server_raw.log` 中存在 sync 模式 teardown 阶段的 Ray traceback（`compiled_dag_node.py` cancel/close 路径），但发生在任务取消/收尾阶段，且 metrics 文件已完整生成，更像收尾噪声而非主实验失败。
- notebook 当前编辑器上下文与磁盘文件可能不一致：编辑器里看到 `project-async-fast_5`，但磁盘上的 `draws/fast_async.ipynb` 仍指向 `project-async-fast_2`。

# 顺序实验运行报告

- 时间: 2026-03-22 00:12:28
- vllm_exp 日志根目录: /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260322-001228
- 脚本元数据目录: /home/bxb1/vllm_workbench/vllm/memory/sequential-run-20260322-001228
- Summary TSV: /home/bxb1/vllm_workbench/vllm/memory/sequential-run-20260322-001228/summary.tsv
- Resume 模式: no

## 配置列表

- /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-async-fast-stop-time.yaml
- /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-async-fast.yaml

## 运行结果

| 序号 | 配置 | project | 执行节点 | 端口 | 状态 | 退出码 | 日志目录 | 备注 |
|---|---|---|---|---:|---|---:|---|---|
| 1 | /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-async-fast-stop-time.yaml | async-fast_stop_time_layer-2 | spartan-gpgpu140 | 8000 | failed | 130 | /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260322-001228/project-async-fast_stop_time_layer-2 | 实验返回非零退出码，但未检测到端口残留 |

# 顺序实验运行报告

- 时间: 2026-03-23 00:09:53
- vllm_exp 日志根目录: /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260323-000953
- 脚本元数据目录: /home/bxb1/vllm_workbench/vllm/memory/sequential-run-20260323-000953
- Summary TSV: /home/bxb1/vllm_workbench/vllm/memory/sequential-run-20260323-000953/summary.tsv
- Resume 模式: no

## 配置列表

- /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-e2e-cmp-llama-patch.yaml
- /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-e2e-cmp-qwen-patch.yaml
- /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-async-fast-stop-time.yaml

## 运行结果

| 序号 | 配置 | project | 执行节点 | 端口 | 状态 | 退出码 | 日志目录 | 备注 |
|---|---|---|---|---:|---|---:|---|---|
| 1 | /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-e2e-cmp-llama-patch.yaml | e2e-cmp-llama-7 | spartan-gpgpu140 | 8000 | success | 0 | /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260323-000953/project-e2e-cmp-llama-7 |  |
| 2 | /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-e2e-cmp-qwen-patch.yaml | e2e-cmp-qwen-patch | spartan-gpgpu140 | 8000 | success | 0 | /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260323-000953/project-e2e-cmp-qwen-patch |  |
| 3 | /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-async-fast-stop-time.yaml | async-fast_stop_time_layer-2 | spartan-gpgpu140 | 8000 | success | 0 | /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260323-000953/project-async-fast_stop_time_layer-2 |  |

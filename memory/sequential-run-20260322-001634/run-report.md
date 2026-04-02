# 顺序实验运行报告

- 时间: 2026-03-22 00:16:34
- vllm_exp 日志根目录: /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260322-001634
- 脚本元数据目录: /home/bxb1/vllm_workbench/vllm/memory/sequential-run-20260322-001634
- Summary TSV: /home/bxb1/vllm_workbench/vllm/memory/sequential-run-20260322-001634/summary.tsv
- Resume 模式: no

## 配置列表

- /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-e2e-cmp-llama.yaml
- /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-e2e-cmp-qwen.yaml
- /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-async-fast-duration.yaml

## 运行结果

| 序号 | 配置 | project | 执行节点 | 端口 | 状态 | 退出码 | 日志目录 | 备注 |
|---|---|---|---|---:|---|---:|---|---|
| 1 | /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-e2e-cmp-llama.yaml | e2e-cmp-llama-5 | spartan-gpgpu166 | 8000 | success | 0 | /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260322-001634/project-e2e-cmp-llama-5 |  |
| 2 | /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-e2e-cmp-qwen.yaml | e2e-cmp-qwen-2 | spartan-gpgpu166 | 8000 | success | 0 | /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260322-001634/project-e2e-cmp-qwen-2 |  |
| 3 | /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-async-fast-duration.yaml | async-fast_duration_5 | spartan-gpgpu166 | 8000 | success | 0 | /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260322-001634/project-async-fast_duration_5 |  |

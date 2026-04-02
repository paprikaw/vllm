# 顺序实验运行报告

- 时间: 2026-03-20 00:05:05
- vllm_exp 日志根目录: /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260320-000505
- 脚本元数据目录: /home/bxb1/vllm_workbench/vllm/memory/sequential-run-20260320-000505
- Summary TSV: /home/bxb1/vllm_workbench/vllm/memory/sequential-run-20260320-000505/summary.tsv
- Resume 模式: no

## 配置列表

- /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-kernal_cmp.yaml
- /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-migration_cmp.yaml
- /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/qwen_config_finder.yaml
- /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-async-fast-duration.yaml

## 运行结果

| 序号 | 配置 | project | 执行节点 | 端口 | 状态 | 退出码 | 日志目录 | 备注 |
|---|---|---|---|---:|---|---:|---|---|
| 1 | /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-kernal_cmp.yaml | kernal_cmp_6 | spartan-gpgpu166 | 8000 | success | 0 | /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260320-000505/project-kernal_cmp_6 |  |
| 2 | /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-migration_cmp.yaml | config-cmp-with-mig-2 | spartan-gpgpu166 | 8000 | success | 0 | /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260320-000505/project-config-cmp-with-mig-2 |  |
| 3 | /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/qwen_config_finder.yaml | config_finder_qwen-1 | spartan-gpgpu166 | 8000 | success | 0 | /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260320-000505/project-config_finder_qwen-1 |  |
| 4 | /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/a-async-fast-duration.yaml | async-fast_duration_3 | spartan-gpgpu166 | 8000 | success | 0 | /home/bxb1/vllm_workbench/vllm/new-logs/sequential-run-20260320-000505/project-async-fast_duration_3 |  |

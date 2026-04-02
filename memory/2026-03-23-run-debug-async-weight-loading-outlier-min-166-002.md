# 2026-03-23 在 166/002 上运行最小 async weight loading 复现实验

## 配置
- 配置文件：`vllm_exp/configs/debug-async-weight-loading-outlier-min-166-002.yaml`
- rank 0: `spartan-gpgpu166`
- rank 1: `spartan-gpgpu002`

## 运行命令
```bash
python -m vllm_exp.run sweep-test \
  --config vllm_exp/configs/debug-async-weight-loading-outlier-min-166-002.yaml \
  --log-dir /home/bxb1/vllm_workbench/vllm/logs \
  --single-server
```

## 结果
- 运行成功
- `Single-server sweep test completed: 2 experiments, All succeeded`

## 日志目录
- `/home/bxb1/vllm_workbench/vllm/logs/project-debug-async-weight-loading-outlier-min-166-002`

包含两组实验：
1. `tgt_pp=48-32`
2. `tgt_pp=56-24`

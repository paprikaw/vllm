---
name: run-exp
description: 运行 vllm_exp 实验。检查 squeue 节点状态，验证配置文件中的节点配置是否匹配，然后执行实验。
argument-hint: config-file-path
allowed-tools: Bash, Read, Glob
---

# Run Experiment Skill

执行以下步骤来运行实验：

## 步骤 1: 检查节点状态
运行 `squeue --me` 查看当前预约的节点信息，获取节点名称和 GPU 类型。

## 步骤 2: 读取配置文件
读取用户指定的配置文件: `$ARGUMENTS`

如果用户没有指定配置文件，询问用户配置文件路径。

## 步骤 3: 验证节点配置
从配置文件中提取 `rank_to_ip` 或节点相关配置，与 `squeue --me` 的结果进行对比：
- 检查配置中的节点是否与当前预约的节点匹配
- 检查 GPU 类型是否符合预期

## 步骤 4: 执行或报告
**如果匹配**：
- 激活虚拟环境：`source /home/bxb1/vllm_workbench/.venv/bin/activate`
- 运行实验命令：
```bash
python -m vllm_exp.run sweep-test --config <config_path> --log-dir new-logs/ --single-server
```

**如果不匹配**：
- 向用户报告不匹配的具体内容
- 询问用户是否需要修改配置文件或等待正确的节点

## 注意事项
- 使用 `timeout: 0` 确保完整追踪命令执行
- 不要在后台运行实验命令
- 如果配置中的 rank 0 在远程节点，需要 ssh 到远程节点执行

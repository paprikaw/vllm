````skill
---
name: run-exp
description: 运行 vllm_exp 实验。区分开发/实验环境，检查 GPU 空闲状态，验证配置，然后执行实验。
argument-hint: config-file-path
allowed-tools: Bash, Read, Glob, Agent
---

# Run Experiment Skill

执行以下步骤来运行实验：

## 步骤 1: 确认当前机器
运行 `hostname` 确认当前所在的节点。

## 步骤 2: 获取并分类所有节点
运行 `squeue --me --format="%.18i %.9P %.20j %.8u %.2t %.10M %.6D %.20R %b"` 获取所有预约的节点信息。

根据 GPU 数量将节点分为两组：
- **开发环境（Development）**：GPU 数量 = 1 的节点（单卡预约）
- **实验环境（Experiment）**：GPU 数量 > 1 的节点（整机预约）

向用户报告分类结果，格式如下：
```
开发环境节点：
  - <hostname> (<GPU类型>, <GPU数量> GPU)
实验环境节点：
  - <hostname> (<GPU类型>, <GPU数量> GPU)
当前所在机器：<hostname>
```

## 步骤 3: 检查 GPU 空闲状态
**优先检查开发环境**，依次在每个节点上运行：
```bash
ssh -o ConnectTimeout=5 <hostname> "ray status 2>/dev/null; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"
```
> 注意：ssh 可能卡住，使用 `ssh -o ConnectTimeout=5` 避免长时间阻塞。本机节点直接运行，不需要 ssh。

判断 GPU 是否空闲的标准：
- `nvidia-smi` 中 GPU 利用率接近 0%，显存使用很少
- `ray status` 中没有正在运行的 actor/task，或 ray 未启动

**如果开发环境有空闲 GPU**：优先使用开发环境节点。

**如果开发环境无空闲 GPU**：检查实验环境节点的状态。如果实验环境节点 GPU 也处于空闲状态（没有实验在运行），可以使用实验环境。

向用户报告各环境的状态，并推荐使用哪个环境。

## 步骤 4: 读取配置文件
读取用户指定的配置文件: `$ARGUMENTS`

如果用户没有指定配置文件，询问用户配置文件路径。

不要要求用户额外指定 log 目录。实验输出目录应由配置本身所对应的 project/实验配置决定，skill 只需要接收配置文件路径。

## 步骤 5: 验证节点配置
从配置文件中提取 `rank_to_ip` 或节点相关配置，与步骤 3 中确认的可用节点进行对比：
- 检查配置中的节点是否为当前可用（空闲）的节点
- 检查 GPU 类型是否符合预期
- 如果配置中的节点不在空闲列表中，提示用户是否需要修改配置

## 步骤 6: 执行或报告
**如果匹配**：
- 激活虚拟环境：`source /home/bxb1/vllm_workbench/.venv/bin/activate`
- 运行实验命令：
```bash
python -m vllm_exp.run sweep-test --config <config_path> --single-server
```

**如果不匹配**：
- 向用户报告不匹配的具体内容（哪些节点忙碌、哪些空闲）
- 建议用户修改配置文件使用空闲的节点，或等待节点释放

## 注意事项
- 使用 `timeout: 0` 确保完整追踪命令执行
- 不要在后台运行实验命令
- 如果配置中的 rank 0 在远程节点，需要 ssh 到远程节点执行
- 不要向用户询问或要求提供 `--log-dir`
- SSH 命令始终使用 `-o ConnectTimeout=5` 防止挂起
- 同一时间只 ssh 一个节点，不要并行多个 ssh
````

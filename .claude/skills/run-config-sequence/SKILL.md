````skill
---
name: run-config-sequence
description: 安全地按顺序运行多个 vllm_exp 配置。为每个实验隔离临时配置文件、日志目录，并在发现残留服务时停止，避免失败实验污染后续实验。配置列表必须由用户或调用方显式提供。
argument-hint: [config1] [config2] [config3]
````skill
---
name: run-config-sequence
description: 安全地按顺序运行多个 vllm_exp 配置。为每个实验隔离临时配置文件，并在发现残留服务时停止，避免失败实验污染后续实验。配置列表与运行日志目录必须显式提供；支持基于已有日志目录跳过已完成子试验，实现断点续跑。
argument-hint: --run-log-dir <dir> --config <cfg1> --config <cfg2> ...
allowed-tools: Bash, Read, Glob, Write
---

# 顺序运行多个配置 Skill

当用户希望连续运行多个 `vllm_exp` 配置，并且要求：

- 某个实验失败后不污染后面的实验
- 可以复用已有日志目录
- 已完成子试验自动跳过

使用这个 skill。

## 核心思路

`vllm_exp` 本身已经支持：

- `overwrite=false` 时，检查子试验对应的 `benchmark.log` / `request_metrics.csv`
- 如果子试验已完成，则自动跳过

因此，只要：

1. 复用同一个 `--run-log-dir`
2. 将配置中的 `overwrite` 设为 `false`

就可以实现**断点续跑**。

这个 skill 做的事情，是在此基础上提供：

- 顺序执行
- 端口污染检查
- rank0 自动选节点
- 临时 JSON 隔离
- summary/report

## 设计目标

1. **顺序执行，不并行**
2. **运行日志目录由用户显式指定**，不强制 sequential 命名
3. **每个实验使用独立临时配置路径**
   - 覆盖：
     - `BENCHMARK_CONFIG_PATH`
     - `DEPLOYMENT_CONFIG_PATH`
4. **运行前后检查端口占用**
   - 默认读配置里的 `static_config.vllm.port`，缺省为 `8000`
   - 若发现残留服务，直接停止，避免污染后续实验
5. **自动判断 rank0 是否在远程节点**
   - 若 `static_config.network.rank_to_ip.0` 不是本机，则 `ssh` 到 rank0 节点执行
6. **支持断点续跑**
   - `--resume` 时临时写入 `overwrite=false`
   - 复用用户指定的 `--run-log-dir`
7. **生成 summary**
   - 包含每个配置的开始时间、退出码、日志目录、执行节点、备注

## 推荐脚本

- `scripts/run_sequential_configs_safe.sh`

## 推荐运行方式

```bash
bash scripts/run_sequential_configs_safe.sh \
  --run-log-dir new-logs/config-finder-runs \
  --config vllm_exp/configs/qwen_config_finder.yaml \
  --config vllm_exp/configs/a-llama-config-finder.yaml \
  --config vllm_exp/configs/a-async-fast-duration.yaml
```

如果要断点续跑：

```bash
bash scripts/run_sequential_configs_safe.sh \
  --run-log-dir new-logs/config-finder-runs \
  --resume \
  --config vllm_exp/configs/qwen_config_finder.yaml \
  --config vllm_exp/configs/a-llama-config-finder.yaml \
  --config vllm_exp/configs/a-async-fast-duration.yaml
```

如果希望某个配置失败后仍继续后续配置：

```bash
bash scripts/run_sequential_configs_safe.sh \
  --run-log-dir new-logs/config-finder-runs \
  --resume \
  --continue-on-error \
  --config vllm_exp/configs/qwen_config_finder.yaml \
  --config vllm_exp/configs/a-llama-config-finder.yaml \
  --config vllm_exp/configs/a-async-fast-duration.yaml
```

## 为什么不要直接用 `&&`

`cmd1 && cmd2 && cmd3` 的问题是：

- `cmd1` 一旦失败，后面的实验不会继续执行；
- `&&` 不能处理**残留服务**、**共享临时文件**、**端口污染**；
- `&&` 也不能利用 `vllm_exp` 的 completed-check 实现断点续跑。

## 必做检查

### 1. 运行前记录节点状态

执行：

```bash
squeue --me
```

### 2. 检查配置中的 rank0

读取每个配置的：

- `static_config.network.rank_to_ip.0`

如果 rank0 不在本机，需要在 rank0 节点执行实验。

### 3. 不要后台运行

实验命令必须前台完整追踪，不要用 `&`。

### 4. 不要重启 ray cluster

如果怀疑 ray cluster 有问题，只报告给用户，不要自动重启。

## 输出结果

脚本会输出：

1. 用户指定的 vllm_exp 日志根目录
2. 脚本元数据目录（summary/report）
3. summary TSV
4. summary Markdown 报告

## 适用场景

- 比较多个模型配置
- 扫描多个论文图配置
- 连续跑用户显式指定的一组实验
- 基于已有日志目录继续未完成实验

````

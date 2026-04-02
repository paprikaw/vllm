## vllm_exp 使用说明（配置参数指南）

本文帮助你在不同测试类型下，清晰填写需要的配置字段，并给出可直接运行的示例。

### 运行方式

- 入口：`vllm_exp/run.py`（Typer CLI）
- 在容器内运行simple_test：

```bash
python3 -m vllm_exp.run sweep-test --config /root/vllm_workbench/vllm/vllm_exp/configs/migration_test.yaml
```

注意：
- 需要在配置顶层或每个 project 的 `envs` 中提供：
  - `BENCHMARK_CONFIG_PATH`: 例如 `/tmp/vllm_exp/benchmark_config.json`
  - `DEPLOYMENT_CONFIG_PATH`: 例如 `/tmp/vllm_exp/vllm_config.json`
- 默认基准将访问 `http://head:<port>`，确保容器内或主机上 `head` 解析到本机（如添加 `/etc/hosts` 条目 `127.0.0.1 head`）。

### 配置总体结构

```yaml
envs:
  NCCL_DEBUG: INFO
  BENCHMARK_CONFIG_PATH: "/tmp/vllm_exp/benchmark_config.json"
  DEPLOYMENT_CONFIG_PATH: "/tmp/vllm_exp/vllm_config.json"
projects:
  - project: "你的项目名"
    type: "测试类型"
    path_policy:        # 可选：用于决定日志目录的“变量维度”
      variables: []
    vllm:
      pipeline_parallel_size: 2
      gpu_memory_utilization: 0.9
      start_pp_layer_partitions: ["28,36"]
    model:
      path: "/root/.cache/huggingface/Qwen3-32B-AWQ"
      name: "Qwen3-32B-AWQ"
    migration:           # 迁移/压缩KV相关
      is_migration: false
    benchmark:           # 基准负载配置（见下文）
      ...
is_log_cover: true
```

### Benchmark 字段语义（核心）

- 必填组合（数据组成）：
  - `data_num_requests: [N1, N2, ...]`：第 i 组样本的请求数
  - `input_output_lens: [[in1, out1], [in2, out2], ...]`：第 i 组样本的输入/输出长度
  - 约束：`len(data_num_requests) == len(input_output_lens)`，否则报错

- 请求速率（两种模式，二选一）：
  - 扫参模式：`sweep_request_rates: [r1, r2, ...]`（每个 r > 0）
    - 框架会对每个速率独立跑一轮完整基准
  - 分段曲线模式：`running_request_rates: [r1, r2, ...]` + `running_num_requests: [k1, k2, ...]`
    - 在同一轮内分段施压，第 i 段以速率 `ri` 发送 `ki` 个请求
    - 要求：长度一致；一般建议 `sum(running_num_requests) == sum(data_num_requests)`

- 其他常用项：
  - `pattern_batch_size`: pattern 数据集每批请求条数（用于 pattern 负载模式）
  - `profile`: 是否启用 benchmark 侧性能采样

重要：避免 `request_rate == 0`，否则 `benchmark_serving.py` 会出现除零错误。

### 多机双向通道（可选）

若需要让 KV 传输在控制面支持双向监听（每个 rank 同时作为本地 server 接收、作为 client 连接对端），请在 `projects[].network.rank_to_ip` 中为各 PP rank 指定可达 IP：

```yaml
network:
  delays: [0]
  rank_to_ip:
    0: "10.0.0.1"
    1: "10.0.0.2"
```

框架会将其注入到运行环境变量，供双向通道初始化使用。

### 不同测试类型的必填项

下表给出三种测试类型在 `projects[].type` 不同取值下需要的字段。

#### 1) type: `test_migration_with_different_pp`

用途：在固定请求速率下，遍历不同的 `start_pp_layer_partitions`，用来测试动态迁移/不同流水线划分。

- `path_policy.variables` 必须包含：
  - `start_pp_layer_partition`
  - `is_migration`
- `migration.is_migration`: 是否开启迁移（true/false）
- `benchmark.sweep_request_rates`: 只能给一个数（代码中有 `assert len(...) == 1`）
- `vllm.start_pp_layer_partitions`: 如 `["8,56", "16,48", "28,36", ...]`

示例：
```yaml
envs:
  BENCHMARK_CONFIG_PATH: "/tmp/vllm_exp/benchmark_config.json"
  DEPLOYMENT_CONFIG_PATH: "/tmp/vllm_exp/vllm_config.json"
projects:
  - project: "migration_test"
    type: "test_migration_with_different_pp"
    path_policy:
      variables: [start_pp_layer_partition, is_migration]
    vllm:
      pipeline_parallel_size: 2
      gpu_memory_utilization: 0.9
      start_pp_layer_partitions: ["28,36"]  # Initial config (used at startup)
    migration:
      is_migration: true
      migration_steps: [100]
      # alternative_configs: Migration target configs ONLY (not initial config)
      # "0" = first migration target, "1" = second migration target, etc.
      # The initial config comes from start_pp_layer_partitions above
      alternative_configs:
        "0": [16, 48]  # First migration: switch to [16, 48]
        "1": [28, 36]  # Second migration: switch back to [28, 36]
    benchmark:
      sweep_request_rates: [2.5]
      pattern_batch_size: 250
      data_num_requests: [200]
      input_output_lens: [[512, 64]]
is_log_cover: true
```

#### 2) type: `partition_and_request_rate`

用途：对于每个 `start_pp_layer_partition`，遍历一组请求速率，形成二维 sweep（分区 × 速率）。

- `path_policy.variables` 必须包含：
  - `start_pp_layer_partition`
  - `request_rate`
- `benchmark.sweep_request_rates`: 至少一个且都 > 0

示例：
```yaml
envs:
  BENCHMARK_CONFIG_PATH: "/tmp/vllm_exp/benchmark_config.json"
  DEPLOYMENT_CONFIG_PATH: "/tmp/vllm_exp/vllm_config.json"
projects:
  - project: "sweep_pp_and_rr"
    type: "partition_and_request_rate"
    path_policy:
      variables: [start_pp_layer_partition, request_rate]
    vllm:
      pipeline_parallel_size: 2
      gpu_memory_utilization: 0.9
      start_pp_layer_partitions: ["28,36", "24,40"]
    migration:
      is_migration: false
    benchmark:
      sweep_request_rates: [1.0, 2.0, 3.0]
      data_num_requests: [1000]
      input_output_lens: [[512, 64]]
is_log_cover: true
```

#### 3) type: `test_compact_kv`

用途：单次运行中，基准负载的请求速率分段变化（跑阶梯/变速），用于验证 KV 压缩/紧凑迁移。

- `benchmark.running_request_rates` 与 `benchmark.running_num_requests` 必须成对出现、长度一致、值 > 0
- `vllm.start_pp_layer_partitions` 目前只支持单元素（代码中有 `assert len(...) == 1`）

示例：
```yaml
envs:
  BENCHMARK_CONFIG_PATH: "/tmp/vllm_exp/benchmark_config.json"
  DEPLOYMENT_CONFIG_PATH: "/tmp/vllm_exp/vllm_config.json"
projects:
  - project: "compact_kv_test"
    type: "test_compact_kv"
    vllm:
      pipeline_parallel_size: 2
      gpu_memory_utilization: 0.9
      start_pp_layer_partitions: ["28,36"]   # 仅单元素
    migration:
      is_migration: false
      is_compact_kv: true
      compact_steps: []
    benchmark:
      running_request_rates: [1.0, 3.0]
      running_num_requests:  [500, 500]
      data_num_requests: [1000]
      input_output_lens: [[512, 64]]
is_log_cover: true
```

### 常见问题（FAQ）

- ZeroDivisionError: request_rate 为 0
  - 请确保 `sweep_request_rates` 或 `running_request_rates` 中所有值 > 0。

- 报错 “Variable X is not declared in path policy.”
  - 需要在 `path_policy.variables` 中声明将用于目录命名的“变量键”。例如：
    - `test_migration_with_different_pp` 需要 `[start_pp_layer_partition, is_migration]`
    - `partition_and_request_rate` 需要 `[start_pp_layer_partition, request_rate]`

- `input_output_lens` 校验失败
  - 同时提供 `data_num_requests`，且两者长度一致。

- 访问 `http://head:8000` 失败
  - 确保 `head` 解析到本机（容器加 `--add-host head:127.0.0.1`，或本机 `/etc/hosts` 配置）。

### 最小可用示例（快速起跑）

```yaml
envs:
  BENCHMARK_CONFIG_PATH: "/tmp/vllm_exp/benchmark_config.json"
  DEPLOYMENT_CONFIG_PATH: "/tmp/vllm_exp/vllm_config.json"
projects:
  - project: "simple_test"
    type: "test_migration_with_different_pp"
    path_policy:
      variables: [start_pp_layer_partition, is_migration]
    vllm:
      pipeline_parallel_size: 2
      gpu_memory_utilization: 0.9
      start_pp_layer_partitions: ["28,36"]
    migration:
      is_migration: false
    benchmark:
      sweep_request_rates: [2.5]
      data_num_requests: [10]
      input_output_lens: [[512, 64]]
is_log_cover: true
```

### 日志与输出

- 所有日志与指标写入 `--log-dir` 下按 `path_policy.variables` 组成的层级目录。
- 框架会在 `constants.json` 中写入“常量参数指纹”，便于区分不同实验。



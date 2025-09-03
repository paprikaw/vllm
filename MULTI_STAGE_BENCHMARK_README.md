# 多阶段基准测试功能使用说明

## 概述

vLLM 基准测试工具现在支持多阶段基准测试功能，允许您在不同的请求速率下测试不同数量的请求，从而更全面地评估模型的性能特征。

## 新功能特性

### 1. 多阶段请求速率控制
- 可以为每个阶段设置不同的请求速率 (requests/second)
- 每个阶段可以处理不同数量的请求
- 支持动态负载测试和压力测试

### 2. 阶段化性能分析
- 每个阶段独立计算性能指标
- 提供阶段间的性能对比
- 生成综合性能报告

### 3. 向后兼容性
- 原有的单阶段基准测试功能完全保留
- 新的多阶段功能作为可选功能提供

## 使用方法

### 配置文件格式

多阶段基准测试使用JSON配置文件来定义各阶段的参数。配置文件应包含以下字段：

```json
{
  "benchmark_type": "multi_stage",
  "description": "配置描述",
  "request_rates": [5.0, 15.0, 25.0],
  "num_requests": [200, 300, 500],
  "stage_descriptions": [
    "阶段1: 预热阶段，低负载 (5 req/s)",
    "阶段2: 正常负载阶段 (15 req/s)",
    "阶段3: 高负载压力测试 (25 req/s)"
  ]
}
```

#### 必需字段

- `request_rates`: 浮点数数组，每个阶段的请求速率 (requests/second)
- `num_requests`: 整数数组，每个阶段处理的请求数量

#### 可选字段

- `benchmark_type`: 基准测试类型标识
- `description`: 配置描述
- `stage_descriptions`: 各阶段的描述信息
- `total_requests`: 总请求数
- `expected_duration`: 预期持续时间

### 基本语法

```bash
python benchmarks/benchmark_serving.py \
    --backend <backend> \
    --model <your_model> \
    --dataset-name <dataset> \
    --dataset-path <path_to_dataset> \
    --benchmark-config <config_file.json> \
    --num-prompts <total_prompts>
```

### 参数说明

#### 新增参数

- `--benchmark-config`: 基准测试配置文件路径
  - 类型: 字符串 (JSON文件路径)
  - 示例: `--benchmark-config config.json`
  - 说明: 指定包含基准测试配置的JSON文件路径，支持多阶段配置和其他扩展配置

#### 重要约束

1. **配置文件格式**: JSON文件必须包含 `request_rates` 和 `num_requests` 两个数组
2. **长度一致**: 两个数组的长度必须相同
3. **请求总数**: 所有阶段的请求数量总和不应超过 `--num-prompts` 指定的总数
4. **文件路径**: 配置文件路径必须有效且文件可读

### 使用示例

#### 示例1: 三阶段渐进式负载测试

```bash
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset-name sharegpt \
    --dataset-path /path/to/sharegpt \
    --benchmark-config multi_stage_config_example.json \
    --num-prompts 1000 \
    --max-concurrency 10
```

这个命令将执行：
- 阶段1: 200个请求，速率5 req/s
- 阶段2: 300个请求，速率15 req/s  
- 阶段3: 500个请求，速率25 req/s

#### 示例2: 压力测试模式

```bash
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset-name random \
    --dataset-path /path/to/sharegpt \
    --benchmark-config stress_test_config.json \
    --num-prompts 1000 \
    --max-concurrency 20
```

这个命令将执行：
- 阶段1: 50个请求，速率1 req/s (预热)
- 阶段2: 100个请求，速率10 req/s (正常负载)
- 阶段3: 200个请求，速率50 req/s (高负载)
- 阶段4: 300个请求，速率100 req/s (压力测试)

## 输出结果

### 控制台输出

多阶段基准测试会在控制台显示：

1. **阶段信息**: 每个阶段的配置和进度
2. **阶段结果**: 每个阶段的详细性能指标
3. **综合摘要**: 所有阶段的汇总表格

示例输出：
```
==================== Stage 1 ====================
Request rate: 10.0 req/s
Number of requests: 200

Stage 1 Results:
---------------- Stage Benchmark Result ----------------
Successful requests:          200
Stage duration (s):          20.15
Total input tokens:          40000
Total generated tokens:       20000
Request throughput (req/s):   9.93
Output token throughput (tok/s): 992.56
Total Token throughput (tok/s): 2977.67

============================================================
MULTI-STAGE BENCHMARK SUMMARY
============================================================
Stage  Rate      Requests   Duration     Throughput      Success
------------------------------------------------------------
1      10.0      200        20.15        9.93           200
2      15.0      300        20.12        14.91          300
3      25.0      500        20.18        24.78          500
------------------------------------------------------------
Total  -          1000       60.45        16.54          1000
```

### 结果文件

结果将保存为JSON文件，文件名格式：
```
{backend}-multi_stage-{rate1}qps_{rate2}qps_{rate3}qps-concurrency{max_concurrency}-{model}-{timestamp}.json
```

文件内容包含：
- 每个阶段的详细指标
- 总体性能统计
- 原始请求和响应数据
- 配置参数

## 性能指标

### 每个阶段计算的指标

- **吞吐量指标**:
  - 请求吞吐量 (req/s)
  - 输出令牌吞吐量 (tok/s)
  - 总令牌吞吐量 (tok/s)

- **延迟指标**:
  - 首令牌延迟 (TTFT)
  - 每输出令牌时间 (TPOT)
  - 令牌间延迟 (ITL)
  - 端到端延迟 (E2EL)

- **统计指标**:
  - 平均值、中位数、标准差
  - 百分位数 (P50, P90, P99等)

### 总体指标

- 所有阶段的综合性能
- 加权平均吞吐量
- 总体资源利用率

## 最佳实践

### 1. 阶段设计建议

- **预热阶段**: 使用较低的请求速率，让系统稳定
- **正常负载**: 使用预期的生产负载速率
- **压力测试**: 逐步增加负载，观察性能拐点
- **恢复测试**: 降低负载，观察系统恢复能力

### 2. 参数配置建议

- **请求数量**: 每个阶段至少50个请求以获得统计意义
- **速率递增**: 建议相邻阶段速率差异不超过2-3倍
- **总测试时间**: 考虑系统稳定性和测试效率的平衡

### 3. 监控建议

- 监控系统资源使用情况 (CPU, GPU, 内存)
- 观察延迟和吞吐量的变化趋势
- 记录性能拐点和异常情况

## 故障排除

### 常见问题

1. **参数不匹配错误**
   ```
   ValueError: Both --request-rate-list and --num-requests must be provided together
   ```
   解决: 确保两个参数同时提供

2. **长度不一致错误**
   ```
   ValueError: Length of --request-rate-list (3) must match length of --num-requests (4)
   ```
   解决: 确保两个列表长度相同

3. **请求数量超出限制**
   ```
   Warning: Total requests in stages (1200) exceeds total prompts (1000)
   ```
   解决: 减少各阶段的请求数量或增加总请求数

### 调试技巧

- 使用 `--disable-tqdm` 禁用进度条以获得更清晰的输出
- 从小规模测试开始，逐步增加复杂度
- 检查日志文件中的详细错误信息

## 技术实现

### 架构设计

多阶段基准测试采用以下架构：

1. **阶段管理器**: 负责阶段切换和配置管理
2. **请求分发器**: 根据阶段配置分发请求
3. **指标收集器**: 收集和计算各阶段指标
4. **结果聚合器**: 汇总所有阶段的结果

### 性能优化

- 异步并发处理
- 内存高效的数据结构
- 最小化阶段间开销
- 智能的资源管理

## 未来扩展

计划中的功能增强：

1. **动态负载调整**: 根据系统响应自动调整负载
2. **A/B测试支持**: 比较不同配置的性能差异
3. **实时监控**: 实时显示性能指标和趋势
4. **报告生成**: 自动生成详细的性能分析报告

## 贡献指南

欢迎贡献代码和改进建议：

1. Fork 项目仓库
2. 创建功能分支
3. 提交代码更改
4. 创建 Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [项目仓库](https://github.com/vllm-project/vllm)
- 邮件列表: [vllm-dev](mailto:vllm-dev@googlegroups.com)
- 文档: [vLLM文档](https://docs.vllm.ai/)

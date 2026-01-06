# Memory Stress Tester for vLLM

## 目的

在vLLM推理过程中持续分配list of tensors（类似KV cache blocks），测试这种内存分配压力对Flexi Flash Attention vs Regular实现的性能影响。

## 原理

创建一个后台线程，不断执行以下操作：
1. 分配一个list，包含N个independent tensors（每个tensor形状如`(16, 8, 128)`）
2. 保持M个这样的allocation cycles在内存中
3. 当达到上限时，释放最旧的cycle，分配新的cycle
4. 记录每次分配的时间

这模拟了KV cache动态分配的场景，可以测试：
- Flexi版本在内存压力下的分配性能
- Regular版本在内存压力下的分配性能
- 是否存在内存碎片化问题

## 配置说明

在`vllm_exp`配置文件的`migration`部分添加：

```yaml
migration:
  is_migration: false
  memory_stress_tester:
    enabled: true                       # 启用stress测试
    num_tensors_per_allocation: 1000    # 每次分配的tensor数量
    tensor_shape: [16, 8, 128]          # 每个tensor的形状 (num_heads, block_size, head_dim)
    allocation_interval_ms: 100         # 分配间隔（毫秒）
    num_allocation_cycles: 10           # 同时保持的allocation cycles数量
```

### 参数说明

- **num_tensors_per_allocation**: 每个allocation cycle中tensor的数量，对应KV cache的block数量
- **tensor_shape**: tensor形状，默认`[16, 8, 128]`对应typical KV block shape
- **allocation_interval_ms**: 两次分配之间的间隔时间（毫秒）
- **num_allocation_cycles**: 同时保持多少个allocation cycles，影响内存占用量

### 内存计算

每个allocation cycle的内存占用：
```
memory_per_cycle = num_tensors * (tensor_shape元素数) * 2 bytes (float16)
例如: 1000 * (16*8*128) * 2 = 32.77 MB
```

总内存占用：
```
total_memory = memory_per_cycle * num_allocation_cycles
例如: 32.77 MB * 10 = 327.7 MB
```

## 测试配置示例

完整的测试配置已创建在：
```
/home/bxb1/vllm_workbench/vllm/vllm_exp/configs/migration_test_agent.yaml
```

包含4个测试场景：
1. `stress_test_regular` - Regular版本 + memory stress
2. `stress_test_flexi` - Flexi版本 + memory stress
3. `baseline_regular` - Regular版本 baseline（无stress）
4. `baseline_flexi` - Flexi版本 baseline（无stress）

## 运行测试

```bash
python3 -m vllm_exp.run \
  --config /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/migration_test_agent.yaml \
  --log-dir /home/bxb1/vllm_workbench/vllm/logs/agent/
```

## 查看结果

### 1. Stress Tester日志

在server log中搜索：
```bash
grep "MemoryStressTester" server.log
```

输出示例：
```
MemoryStressTester initialized: device=cuda:0, tensors_per_alloc=1000, ...
MemoryStressTester started (KV cache allocation style)
MemoryStressTester: cycle #10, alloc_time=72.15ms, avg=71.23ms, current_cycles=10
MemoryStressTester stopped. Stats: allocations=100, avg_alloc_time=72.45ms
```

### 2. 性能对比

对比以下指标：
- **TTFT (Time to First Token)**: Prefill阶段延迟
- **TPOT (Time Per Output Token)**: Decode阶段延迟
- **Throughput**: 吞吐量

比较4个场景的结果：
```
Scenario                    TTFT    TPOT    Throughput  Stress Alloc Time
baseline_regular            X ms    Y ms    Z tok/s     N/A
baseline_flexi              X ms    Y ms    Z tok/s     N/A
stress_test_regular         X ms    Y ms    Z tok/s     A ms
stress_test_flexi           X ms    Y ms    Z tok/s     B ms
```

### 3. 分析

关键问题：
1. Flexi vs Regular在有memory stress时性能差异是否扩大？
2. Memory stress是否导致Flexi分配时间增加更多？
3. 是否存在内存碎片化导致的性能退化？

## 实现细节

### 文件结构

```
vllm/v1/engine/memory_stress_tester.py  - Stress tester实现
vllm/v1/engine/dynamic_core.py          - 集成到engine
vllm/dynamic_config.py                  - 配置定义
vllm_exp/data.py                        - 配置模型
vllm_exp/experiments.py                 - 配置传递
```

### 线程安全

- Stress tester在独立线程运行
- 使用`threading.Lock`保护共享状态
- 使用`threading.Event`进行优雅停止

### 性能监控

Stress tester会记录：
- `total_allocations`: 总分配次数
- `total_deallocations`: 总释放次数  
- `allocation_errors`: 分配错误次数
- `avg_allocation_time_ms`: 平均分配时间（毫秒）

## 注意事项

1. **GPU内存**: 确保有足够GPU内存运行stress tester + vLLM推理
2. **OOM风险**: 如果出现OOM，减少`num_allocation_cycles`或`num_tensors_per_allocation`
3. **性能影响**: Stress tester本身会占用GPU资源，这是预期行为
4. **Baseline对比**: 务必同时运行有/无stress的测试进行对比

## 预期发现

通过这个测试，可以回答：
1. Flexi的慢是否因为内存分配本身慢，还是其他原因？
2. 在内存压力下，PyTorch caching allocator对两种实现的影响差异
3. 是否需要优化KV cache allocation策略

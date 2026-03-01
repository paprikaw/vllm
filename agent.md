# vLLM Extended Agent Guide

## 命令执行规则

1. **永远追踪命令**: 执行命令时必须使用 `timeout: 0`，确保完整等待命令执行完成
2. **禁止擅自后台运行**: 如果认为必须使用 `isBackground: true` 或不追踪命令，**必须先询问用户**
3. **原因**: 提前跳出等待状态会导致需要反复检查命令是否执行成功，影响工作效率

---

## vllm_exp Experimental Framework Usage

### Overview
`vllm_exp` is a comprehensive testing framework for vLLM experiments. It manages vLLM server startup, benchmark execution, and result collection with various testing modes.

### Running Experiments

**Basic Command:**
```bash
python3 -m vllm_exp.run --config <config_file.yaml> --log-dir <log_directory>
```

**Example:**
```bash
python3 -m vllm_exp.run \
  --config /home/bxb1/vllm_workbench/vllm/vllm_exp/configs/migration_test_A100.yaml \
  --log-dir /home/bxb1/vllm_workbench/vllm/logs/
```


### Multi-Machine Network Configuration

For distributed setups with KV transfer:
```yaml
network:
  delays: [0]
  rank_to_ip:
    0: "192.168.1.10"
    1: "192.168.1.11"
```

### Common Pitfalls

1. **ZeroDivisionError: request_rate is 0**
   - Ensure all values in `sweep_request_rates` or `running_request_rates` are > 0

2. **"Variable X not declared in path policy"**
   - Add required variables to `path_policy.variables`:
     - `test_migration_with_different_pp`: `[start_pp_layer_partition, is_migration]`
     - `partition_and_request_rate`: `[start_pp_layer_partition, request_rate]`

3. **Process hangs indefinitely**
   - **Most common**: Missing or incorrect migration configuration
     - Even when `is_migration: false`, you MUST include `alternative_configs["0"]` with initial config
     - Example: `alternative_configs: {"0": [32, 32]}`
     - Without this, engine core will fail with `KeyError: '0'`
   - When using `one_off_test`, ensure only one `start_pp_layer_partitions` value
   - Check that `data_num_requests` matches `input_output_lens` length

4. **Cannot access http://head:8000**
   - Ensure `head` resolves to localhost in `/etc/hosts` or use `head_addr: "localhost"`

### Log Output

Logs are written to `--log-dir` with subdirectories based on:
- Project name: `project-{project_name}/`
- Variables from `path_policy`: Used to create nested directory structure
- Server logs: `server-{pp=X,Y}-{flexi=0/1}.log`
- Benchmark logs: `benchmark-*.log`
- Metrics: `request_metrics-*.csv`

### Minimal Working Example

```yaml
envs:
  BENCHMARK_CONFIG_PATH: "/tmp/vllm_exp/benchmark_config.json"
  DEPLOYMENT_CONFIG_PATH: "/tmp/vllm_exp/vllm_config.json"
projects:
  - project: "simple_test"
    type: "one_off_test"
    model:
      path: "/path/to/model"
    vllm:
      pipeline_parallel_size: 2
      start_pp_layer_part
      alternative_configs:
        "0": [32, 32]  # Required even when migration is disableditions: ["32,32"]
    migration:
      is_migration: false
    benchmark:
      data_num_requests: [100]
      input_output_lens: [[512, 128]]
      running_request_rates: [2]
      running_num_requests: [100]
is_log_cover: true
```

### Performance Analysis Workflow

For comparing attention kernels:

1. **Create a sweep configuration** with `attention_kernel` options
   ```yaml
   # Sweep across all three kernels
   sweep_config:
     vllm:
       attention_kernel: [flash, flexi, direct]
   ```
   Options: `flash` (standard FlashAttention), `flexi` (flexi kernel), `direct` (flexi direct kernel)

2. **Run both experiments**
   ```bash
   python3 -m vllm_exp.run --config regular.yaml --log-dir logs/
   python3 -m vllm_exp.run --config flexi.yaml --log-dir logs/
   ```

3. **Analyze server logs** for timing information
   - Server logs contain `[perf_analysis]`, `[timeline]`, `[forward]` tags
   - Request metrics in CSV files for aggregate statistics

4. **Compare results**
   - TTFT (Time To First Token)
   - TPOT (Time Per Output Token)
   - E2E latency
   - Forward pass counts and timings

---

## Stream Priority Usage

### CUDA Stream Priority Levels (A100)

On A100 GPUs, stream priority ranges from:
- **Lowest priority**: `0`
- **Highest priority**: `-2`

**Query available range:**
```python
import torch
low, high = torch.cuda.Stream().priority_range()
# A100: low=0, high=-2
```

### Creating High Priority Stream

```python
# Create highest priority stream
high_priority_stream = torch.cuda.Stream(priority=-2)

# Use with proper synchronization
with torch.cuda.stream(high_priority_stream):
    # Compute operations
    output = model(input)

# CRITICAL: Always synchronize before using results
torch.cuda.synchronize(high_priority_stream)
```

### Stream Priority Best Practices

1. **Always synchronize** after stream operations before accessing results
2. Use high priority for critical compute kernels
3. Avoid creating too many streams (overhead)
4. Profile to verify priority is effective

---

## Automated Monitoring

### Smart Auto-Monitor (Recommended)

The smart monitor automatically tracks experiments and provides status updates without manual intervention.

**Start the monitor:**
```bash
python /tmp/smart_monitor.py > /tmp/smart_monitor_console.log 2>&1 &
```

**View live status updates:**
```bash
tail -f /tmp/experiment_status.txt
```

**Features:**
- ✅ Automatic progress tracking every 30 seconds
- ✅ Detects experiment completion automatically
- ✅ Runs performance analysis when both experiments finish
- ✅ Generates analysis report automatically
- ✅ No manual intervention needed

**Status file shows:**
- Current phase (Regular/Flexi/Analyzing/Done)
- Running time for each experiment
- Log file sizes (progress indicator)
- Completion status
- Analysis results summary

### Agent Auto-Check Loop

For continuous monitoring within agent conversation, use 30-second check intervals:

```python
# Agent will automatically check status every 30 seconds
# and report progress updates until experiments complete
while not completed:
    status = check_experiment_status()
    report_to_user(status)
    wait(30)
```

**Advantages:**
- Proactive updates without user prompts
- Catches issues immediately
- Seamless workflow
- User can interrupt anytime

## Manual Debugging Tips

### Check Experiment Status
```bash
# Check running processes
ps aux | grep vllm_exp.run

# Monitor log file growth
watch -n 5 'ls -lh logs/agent/*.log'

# Check for errors in logs
tail -f logs/agent/iter1_regular_full.log | grep -i error
```

### Kill Hung Experiments
```bash
pkill -f vllm_exp.run
```

### Verify Configuration
```bash
# Dry run to check config parsing (if supported)
python -m vllm_exp.run --config myconfig.yaml --help
```

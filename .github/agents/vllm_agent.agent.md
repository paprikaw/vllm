---
description: 'AI assistant specialized in vLLM extended version with dynamic pipeline parallelism support, KV cache migration, and experimental environment management. Expert in dynamic GPU workers, flexi attention kernels, and distributed inference optimization.'
tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'pylance-mcp-server/*', 'todo', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment']
---

# vLLM Extended Agent

## Overview

This agent is specialized for working with an extended version of vLLM that enables dynamic pipeline parallelism configuration switching and KV cache migration capabilities. The agent has deep knowledge of the codebase architecture, experimental setup, and best practices for development and testing.

## Basic setup

Python Environment: /data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/python

## Basic Rules
1. You should always follows the user defines basic setup above. 

2. If you fell that there is a need to update the agent.md, please do so in a concise manner, the agent.md file is ".github/agents/vllm_agent.agent.md"

3. If you have come to a conclusion for a problem, please generate a reports under ./reports directory

4. In the process of solving a problem, you will find youself generating all sorts of tools to analyse log, trying to use the exisiting tools provided in the **./vllm_exp/tools**, if you need a new tools, please think of maybe adding feature to exising one. Try to make it easy to maintain, you should think these set of tools as a framework of log analysing. If the things you want to do can only be done by temporary scripts and not be easily integrated to existing tools, you should place it in /tmp directory.

5. When you are performing a long task, sometimes you need to run tools and waiting for the results. You shouldn't hand the control back to user, instead you should keep monitoring the progress of the task. You can "sleep 30" to wait for 30 seconds before checking again.

6. If you are going to test a vllm inferencing process without migration, simply set the migration_step higher than the total number of requests, so that no migration will happen.

## Background

### Repository Context

This repository is an **extended version of vLLM** that addresses limitations in the original implementation:

- **Problem**: Traditional vLLM cannot perform dynamic pipeline parallelism configuration switching during runtime
- **Solution**: This repository extends vLLM to support dynamic reconfiguration of pipeline parallelism stages, enabling flexible model serving and efficient resource utilization
- **Additional Features**: Includes experimental environment code for testing and validation

### Code Architecture

#### Dynamic Extensions (`dynamic_*` prefix)

The codebase uses **inheritance-based extension pattern** to augment vLLM's original functionality without modifying core code:

- **`dynamic_gpu_worker.py`**: Extended GPU worker with dynamic pipeline stage management
- **`dynamic_gpu_model_runner.py`**: Model runner supporting runtime reconfiguration
- **`dynamic_kv_synchronizer.py`**: KV cache synchronization across pipeline stages
- **Other dynamic modules**: Various components enabling dynamic behavior

**Key Design Principle**: All dynamic code inherits from original vLLM classes, allowing seamless integration and easy maintenance.

#### Flexible KV Cache (`flexi_*` prefix)

Files with `flexi` prefix implement **flexible KV cache expansion** with custom Flash Attention kernels:

- **`flexi_attention_kernels.cu`**: Custom CUDA kernels for variable-size KV cache
- **`flexi_bind_kv_cache.py`**: Dynamic binding utilities for flexible cache management
- **Purpose**: Enable efficient memory usage and support for dynamic sequence length handling

### Key Features

1. **Dynamic Pipeline Parallelism**
   - Runtime stage reconfiguration
   - KV cache migration between pipeline stages
   - Flexible model partitioning

2. **Flexible KV Cache Management**
   - Variable-size KV cache allocation
   - Efficient memory utilization
   - Custom Flash Attention integration

3. **Experimental Environment**
   - Comprehensive testing framework
   - Configuration-based experiments
   - Performance benchmarking tools

## Experiment Guidelines

When completing tasks that require code verification, follow these guidelines:

### 1. Short Python Scripts

**Use Case**: Quick validation, utility scripts, or standalone tests

**Best Practice**: Place scripts in `/tmp` directory to avoid repository clutter

```bash
# Example: Testing a specific function
python /tmp/test_stream_priority.py
```

**Rationale**: Keeps the repository clean and prevents accidental commits of temporary code

### 2. vLLM Integration Testing

**Use Case**: Testing vLLM functionality, pipeline behavior, or end-to-end workflows

**Best Practice**: Use the experimental environment in `vllm_exp/`

#### Experimental Environment Structure

```
vllm_exp/
├── configs/
│   ├── migration_test_A100.yaml    # A100 GPU configuration
│   └── ...                          # Other configuration files
├── run.py                           # Main experiment runner
└── README.md                        # Environment documentation
```

#### Running Experiments
If you need to run the experiments, you should reference the existing using configuration  migration_test_A100.yaml create and use your own configuration in : 
vllm_exp/
├── configs/
│   ├── migration_test_agent.yaml  

You should also output the log to a individual log directory such as:
/home/bxb1/data/vllm_workbench/vllm/logs/agent/

**Standard Command**:
running a experiment:
```bash
python3 -m vllm_exp.run \
  --config /home/bxb1/data/vllm_workbench/vllm/vllm_exp/configs/migration_test_A100.yaml \
  --log-dir /home/bxb1/data/vllm_workbench/vllm/logs/agent/
```

using log analysis tools:
```bash
python /home/bxb1/vllm_workbench/vllm/logs/analyze_log_metrics.py "/home/bxb1/vllm_workbench/vllm/logs/project-migration_test_async_with_migration/server-{pp=32,32}-{flexi=0}.log" --top 50 > /home/bxb1/vllm_workbench/vllm/logs/agents/analyse_regular.txt
```

**Configuration File**: `migration_test_A100.yaml`
- Optimized for A100 GPUs
- Configured for pipeline parallelism testing
- Includes KV migration settings

**Log Directory**: `/home/bxb1/data/vllm_workbench/vllm/logs`
- Centralized logging for all experiments
- Structured output for analysis
- Timestamped experiment runs

### Test Types
Currently, please only use one_off_test as the test type
#### 1. `one_off_test` - Single Configuration Test
**Purpose**: Run a single configuration without sweeping parameters. Ideal for performance profiling and detailed analysis.

**Key Characteristics:**
- Single `start_pp_layer_partitions` value (must have length 1)
- Uses `running_request_rates` + `running_num_requests` (staged rate changes)
- No `path_policy` needed (defaults to empty)

**Configuration Example:**
```yaml
projects:
  - project: "my_test"
    type: "one_off_test"
    vllm:
      pipeline_parallel_size: 2
      enable_flexi_flash_attn: false
      start_pp_layer_partitions: ["32,32"]  # Must be single element
    migration:
      is_migration: false  # No migration
      # DO NOT set migration_steps or alternative_configs when is_migration: false
    benchmark:
      data_num_requests: [100]
      input_output_lens: [[512, 128]]
      running_request_rates: [2]      # Rate for each stage
      running_num_requests: [100]     # Requests per stage
```

**⚠️ Important Notes:**
- **When `is_migration: false`**: Do NOT include `migration_steps` or `alternative_configs`
- **When `is_migration: true`**: Must include `migration_steps` (list of request numbers when migration occurs) and `alternative_configs` (pipeline configurations to switch to)


## Development Best Practices

### Code Modification Guidelines

1. **Inheritance Over Modification**: Extend existing classes rather than modifying original vLLM code
2. **Naming Conventions**: 
   - Prefix dynamic extensions with `dynamic_`
   - Prefix flexible KV cache code with `flexi_`
3. **Documentation**: Document all extensions with clear purpose and usage
4. **Testing**: Always test changes using the experimental environment

## Important Paths

- **Configuration**: `/home/bxb1/data/vllm_workbench/vllm/vllm_exp/configs/migration_test_A100.yaml`
- **Logs**: `/home/bxb1/data/vllm_workbench/vllm/logs`
- **Experimental Code**: `vllm_exp/`
- **Dynamic Extensions**: `vllm/v1/worker/dynamic_*.py`
- **Flexible Kernels**: `../flash-attention/`

## Notes

- Always check the experimental environment README for the latest usage instructions
- Configuration files may be updated frequently; verify paths before running experiments
- When in doubt, place temporary code in `/tmp` to maintain repository cleanliness
- The agent is aware of both the original vLLM API and the extended dynamic features

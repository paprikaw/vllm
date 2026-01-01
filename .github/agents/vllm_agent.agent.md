---
description: 'AI assistant specialized in vLLM extended version with dynamic pipeline parallelism support, KV cache migration, and experimental environment management. Expert in dynamic GPU workers, flexi attention kernels, and distributed inference optimization.'
tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'pylance-mcp-server/*', 'todo', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment']
---

# vLLM Extended Agent

## Overview

This agent is specialized for working with an extended version of vLLM that enables dynamic pipeline parallelism configuration switching and KV cache migration capabilities. The agent has deep knowledge of the codebase architecture, experimental setup, and best practices for development and testing.

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

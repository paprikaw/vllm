---
description: 'AI assistant specialized in vLLM extended version with dynamic pipeline parallelism support, KV cache migration, and experimental environment management. Expert in dynamic GPU workers, flexi attention kernels, and distributed inference optimization.'
tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'pylance-mcp-server/*', 'todo', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment', 'ms-toolsai.jupyter/configureNotebook', 'ms-toolsai.jupyter/listNotebookPackages', 'ms-toolsai.jupyter/installNotebookPackages']
---

# vLLM Extended Agent
## Overview
This agent is specialized for working with an extended version of vLLM that enables dynamic pipeline parallelism configuration switching and KV cache migration capabilities. The agent has deep knowledge of the codebase architecture, experimental setup, and best practices for development and testing.

This repository is an **extended version of vLLM** that addresses limitations in the original implementation:
- **Problem**: Traditional vLLM cannot perform dynamic pipeline parallelism configuration switching during runtime
- **Solution**: This repository extends vLLM to support dynamic reconfiguration of pipeline parallelism stages, enabling flexible model serving and efficient resource utilization
- **Additional Features**: Includes experimental environment code for testing and validation

## Basic Rules
1. You should always follows the user defines basic setup above. 
2. If you have come to a conclusion for a problem, please generate a reports under ./reports directory
3. In the process of solving a problem, you will find youself generating all sorts of tools to analyse log, trying to use the exisiting tools provided in the **./vllm_exp/tools**, if you need a new tools, please think of maybe adding feature to exising one. Try to make it easy to maintain, you should think these set of tools as a framework of log analysing. If the things you want to do can only be done by temporary scripts and not be easily integrated to existing tools, you should place it in /tmp directory.
4. If you are going to test a vllm inferencing process without migration, simply set the migration_step higher than the total number of requests, so that no migration will happen.
5. Keeps the repository clean and prevents accidental commits of temporary code.
6. When investigating whether a log file has error, rather than print the tail of the log, you should search whether there is errors happening in the log.
7: When investigating a benchmark file, you should **only** consider it as normal when all the output has <think> as begining and produce full sentence with requred num of tokens.
8: When running any commands, you should always try to add | head at the end to prevent copilot from interupting your commands.

## Development
### Code Modification Guidelines
1. Inheritance Over Modification: Extend existing classes rather than modifying original vLLM code
2. Naming Conventions: 
   - Prefix dynamic extensions with `dynamic_`
   - Prefix flexible KV cache code with `flexi_`
3. Documentation: Document all extensions with clear purpose and usage
4. Testing: Always test changes using the experimental environment
5. When adding new variables to vllm, making sure to also add them to the experiment framework exp_vllm to make sure user can directly configure them via the experiment configuration files. Making sure not using environment variables, add them to the data.py and pass them through vllm's args.
6. You should always write most succint, clear and elegent code. Don't over engineer stuff.
### Environment Setup
Python Environment: /data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/python
### Important Paths
- **Configuration**: 'vllm_exp/configs/debug_cross_node.yaml vllm_exp/configs/debug_single_node_l40.yaml vllm_exp/configs/debug_single_node.yaml/'
- **Logs**: `/home/bxb1/data/vllm_workbench/vllm/logs`
- **Experimental Code**: `vllm_exp/`
- **Dynamic Extensions**: `vllm/v1/worker/dynamic_*.py`
- **Flexible Kernels**: `../flash-attention/`
### Basic Commands
1. Checking out current gpu cluster status: ```ray status```
2. running a experiment:
```bash
python3 -m vllm_exp.run \
  --config /home/bxb1/data/vllm_workbench/vllm/vllm_exp/configs/config.yaml \
  --log-dir /home/bxb1/data/vllm_workbench/vllm/logs/agent/
```
3. using log analysis tools:
```bash
python /home/bxb1/vllm_workbench/vllm/logs/analyze_log_metrics.py "/home/bxb1/vllm_workbench/vllm/logs/project-migration_test_async_with_migration/server-{pp=32,32}-{flexi=0}.log" --top 50 > /home/bxb1/vllm_workbench/vllm/logs/agents/analyse_regular.txt
```
### Experiment Guidelines
1. You should also output the log to a individual log directory such as: /home/bxb1/data/vllm_workbench/vllm/logs/agent/
2. If you need to run the experiments, you should reference the existing using configuration  migration_test_A100.yaml create and use your own configurations like "vllm_exp/configs/debug_cross_node.yaml vllm_exp/configs/debug_single_node_l40.yaml vllm_exp/configs/debug_single_node.yaml/'"
3. There are different test types for the experiment, currently, please only use one_off_test as the test type
4. When `is_migration: false`: Do NOT include `migration_steps` or `alternative_configs`
5. When `is_migration: true`: Must include `migration_steps` (list of request numbers when migration occurs) and `alternative_configs` (pipeline configurations to switch to). Note: `start_pp_layer_partitions` defines the initial config, while `alternative_configs` defines ONLY the migration target configs (starting from "0" as the first migration target).
6. When adding new args to config, please use dynamic_config to pass it to vllm and benchmark.




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
The real implementation of flexi kernal is under the **../flash-attention/** directory.


```skill
---
name: compile-vllm
description: 编译 vLLM。使用 ccache 缓存加速编译，加载必要的模块，设置正确的环境变量。
argument-hint: none
allowed-tools: Bash, Read
---

# Compile vLLM Skill

使用 ccache 编译 vLLM，确保环境配置正确。

## 编译脚本路径
```
/home/bxb1/vllm_workbench/scripts/running_compile_vllm.sh
```

## 执行步骤

### 步骤 1: 激活虚拟环境
```bash
source /home/bxb1/vllm_workbench/.venv/bin/activate
```

### 步骤 2: 执行编译脚本
```bash
bash /home/bxb1/vllm_workbench/scripts/running_compile_vllm.sh
```

## 脚本功能说明

编译脚本会自动完成以下操作：

1. **加载必要模块**：
   - NCCL/2.21.5-CUDA-12.4.1
   - Miniconda3/23.10.0-1
   - ccache/4.6.3

2. **配置 ccache**：
   - 缓存目录：`/home/bxb1/data/.cache/ccache`
   - 日志目录：`/home/bxb1/data/ccache_logs/`
   - 启用调试模式和路径哈希忽略

3. **设置编译环境变量**：
   - `USE_CUDA=1`
   - `USE_SYSTEM_NCCL=1`
   - `TORCH_CUDA_ARCH_LIST="8.0"`
   - `VLLM_FLASH_ATTN_SRC_DIR=/home/bxb1/vllm_workbench/flash-attention`

4. **安装依赖并编译**：
   - 安装 build 和 dev 依赖
   - 使用 `uv pip install -e .` 进行 editable 安装
   - 编译日志保存到：`/home/bxb1/vllm_workbench/logs/vllm_build/`

5. **输出 ccache 统计**：
   - 显示缓存命中率
   - 显示 cache miss 记录

## 注意事项

1. **只有修改 C++ 代码时才需要重新编译**
   - 修改 Python 代码不需要重新编译
   
2. **文件系统共享**
   - 在 Spartan 环境下，文件系统是共享的
   - 只需要在一个节点上编译，不需要在多个节点重复编译

3. **编译超时**
   - 使用 `timeout: 0` 确保完整等待编译完成
   - 首次编译可能需要较长时间（约 30-60 分钟）
   - 有 ccache 缓存时，增量编译通常只需几分钟

4. **查看编译日志**
   - 编译日志位于：`/home/bxb1/vllm_workbench/logs/vllm_build/`
   - 文件名格式：`build_YYYYMMDD_HHMMSS.log`

```

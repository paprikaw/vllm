---
name: add-exp-param
description: 在 vllm_exp 实验框架中添加或修改配置参数。涵盖 data.py 数据定义、experiments.py 服务器分组逻辑、vllm 代码集成等完整流程。
argument-hint: parameter-name
allowed-tools: Bash, Read, Glob, Write, Edit
---

# Add/Modify vLLM Experiment Parameter Skill

在 vllm_exp 框架中添加新配置参数或修改现有参数时，必须按照以下检查清单逐步完成。**遗漏任何一步都可能导致参数不生效或实验行为异常。**

## 核心文件位置

- **数据定义**: `vllm_exp/data.py`
- **实验执行**: `vllm_exp/experiments.py`
- **vLLM 配置**: `vllm/config.py`
- **具体实现**: 取决于参数用途（如 model_loader, worker 等）

---

## 检查清单

### 阶段 1: 数据定义 (data.py)

#### 1.1 添加 Sweep 轴定义（如果需要 sweep）
在 `SweepConfig` 类中添加：
```python
class SweepConfig:
    # 其他字段...
    new_param: Optional[List[Type]] = None
    """List of new_param values to sweep. Description of what it does."""
```

#### 1.2 添加到 sweep 轴生成
在 `SweepConfig.get_sweep_axes()` 方法中添加：
```python
if self.vllm.new_param is not None:
    axes['new_param'] = self.vllm.new_param
```

#### 1.3 添加静态默认值
在以下类中添加默认值：
- `VllmStaticServerConfigV2` (静态服务器配置)
- `VllmServerSpec` (服务器规格)
- `FinalExperimentConfig` (最终配置)

```python
class VllmStaticServerConfigV2:
    new_param: Type = default_value
```

#### 1.4 添加短名称映射
在 `ExperimentConfig.SHORT_NAMES` 字典中添加：
```python
SHORT_NAMES = {
    # 其他映射...
    "new_param": "np",  # 用于文件名和日志
}
```

#### 1.5 添加到 vars_dict
在 `ExperimentConfig.get_vars_dict()` 方法中添加：
```python
vars_dict["np"] = self.vllm.new_param
```

#### 1.6 添加到配置合并函数（如果需要 sweep）
在 `build_vllm_spec()` 或类似函数中处理参数：
```python
def build_vllm_spec(..., new_param: Optional[Type] = None, ...):
    np = new_param if new_param is not None else static_cfg.vllm.new_param
    # 使用 np 构建配置
```

#### 1.7 添加到 VllmServerSpec 构造
确保参数传递到 `VllmServerSpec(...)` 构造函数：
```python
return VllmServerSpec(
    # 其他参数...
    new_param=np,
)
```

#### 1.8 添加到命令行参数生成
在 `VllmServerSpec.to_cmd_dict()` 或 `generate_server_command()` 中添加：
```python
def to_cmd_dict(self) -> dict:
    return {
        # 其他参数...
        "new_param": self.new_param,
    }
```

---

### 阶段 2: 服务器生命周期管理 (experiments.py)

#### 2.1 ⚠️ 关键：检查是否需要服务器重启

**问题**：不同参数值的实验可能共享同一服务器实例，导致参数不生效。

**判断标准**：
- 如果参数影响模型加载、初始化或服务器状态 → **需要重启**
- 如果参数只影响运行时行为且可以动态更新 → **不需要重启**

#### 2.2 添加到服务器分组键（如果需要重启）

在 `get_server_group_key()` 函数中添加参数：

```python
def get_server_group_key(exp: 'Experiment') -> tuple:
    """获取服务器分组键，相同键的实验共享服务器"""
    return (
        exp.sweep_config_idx,
        exp.vllm_spec.attention_kernel,
        exp.vllm_spec.block_size,
        exp.vllm_spec.enable_vmm_allocator,
        exp.vllm_spec.new_param,  # ← 添加新参数
    )
```

#### 2.3 更新日志输出

更新服务器分组信息的日志格式：
```python
# 在启动服务器时的日志中包含新参数
print(f"Starting server for sweep[{idx}] ... new_param={new_param_value}")
```

#### 2.4 更新日志文件命名

如果需要区分不同参数的日志：
```python
log_filename = f"server_global_sweep{idx}_kernel_{kernel}_np{new_param}.log"
```

---

### 阶段 3: vLLM 代码集成

#### 3.1 添加到 DynamicConfig (config.py)

如果参数需要动态更新（运行时可变）：
```python
@dataclass
class DynamicConfig:
    # 其他字段...
    new_param: Type = default_value
```

#### 3.2 在实际代码中使用参数

在相关模块中读取和使用参数：
```python
# 从 vllm_config 获取
new_param = vllm_config.dynamic_config.new_param
```
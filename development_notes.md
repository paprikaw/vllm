# Development notebook
在这里记录所有开发过程中值得记录的内容

# 添加配置

## 在 DynamicConfig 中添加新配置参数

### 操作步骤

1. **在 `vllm/config.py` 的 `DynamicConfig` 类中添加字段**
   ```python
   @dataclass
   class DynamicConfig:
       enable_flexi_flash_attn: bool = False
       your_new_param: bool = False  # 添加新参数
       """参数说明文档"""
   ```
   - 必须提供默认值
   - 如果参数影响计算图，在 `compute_hash()` 方法的 `factors` 列表中添加该参数

2. **在 `vllm_exp/data.py` 中同步更新**
   - 在 `VllmCfg` 类中添加相同字段
   - 在 `collect_variables()` 方法中添加收集逻辑

3. **在代码中使用**
   ```python
   if vllm_config.dynamic_config.your_new_param:
       # 你的逻辑
   ```

### 使用方式

- **CLI**: `vllm serve <model> -D '{"your_new_param": true}'`
- **YAML**: `vllm: {your_new_param: true}`
- **Python**: `DynamicConfig(your_new_param=True)`

### 注意事项

- `-D` 参数解析由 `get_kwargs()` 自动处理，无需手动编写解析代码
- 参数会通过 `VllmConfig.dynamic_config` 传递到整个系统
- vllm_exp 通过 YAML 配置自动生成 `-D` 参数

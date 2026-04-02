# 2026-03-20 VMM 修复正式化与清理

## 目标
将用于定位 VMM 回退的临时计时日志全部清理掉，并把激活输出复用从 `LlamaMLP` 中的实验性实现，下沉成正式实现。

## 代码调整
1. `vllm/model_executor/layers/activation.py`
   - 在 `SiluAndMul` 中加入 `_output_buffer`。
   - `forward_cuda()` / `forward_xpu()` 改为复用缓冲区，而不是每次重新 `torch.empty(...)`。
   - 这样修复被封装在激活层本身，不再依赖某个具体模型的 MLP 特判。
2. `vllm/model_executor/models/llama.py`
   - 删除 `LlamaMLP.forward()` 中的临时 profiling / 同步 / `_act_buffer` 实验代码。
   - 删除 `LlamaDecoderLayer.forward()` 中的子阶段计时日志。
3. `vllm/v1/worker/dynamic_gpu_model_runner.py`
   - 删除 `INTEGRATION_TIMING` 相关同步计时与日志，恢复异步执行路径。
4. `vllm/v1/attention/backends/flexi_flash_attn.py`
   - 删除 `DIRECT_LAYER_TIMING` 计时与同步。
5. `vllm/model_executor/models/dynamic_llama.py`
   - 将慢层日志阈值从调试期的 `20ms` 恢复到 `100ms`。

## 验证
### 语法检查
- `python -m py_compile` 覆盖上述 5 个文件，成功。

### 定向实验
配置：`vllm_exp/configs/tmp_direct_vmm_stage_debug_20260320.yaml`

结果：
- `vmm=1`: `ttft_mean=563.66ms`, `tpot_mean=106.40ms`, `e2el_mean=2159.62ms`
- `vmm=0`: `ttft_mean=562.98ms`, `tpot_mean=108.26ms`, `e2el_mean=2186.84ms`

结论：
- 正式实现后，`VMM on/off` 仍然基本重合。
- 新日志中不再出现 `INTEGRATION_TIMING` / `DIRECT_LAYER_TIMING` / `DECODER_SUBTIMING` / `MLP_SUBTIMING`。

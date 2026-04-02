# 2026-03-20 Qwen 显存问题修复：runner 级共享 activation workspace

## 问题
`SiluAndMul` 之前被改成了模块级持久 `_output_buffer`。
在 Qwen 这类每层都有独立 `SiluAndMul` 的模型上，会变成“每层持有一份最大激活输出 buffer”。

对 `max_num_batched_tokens=8128`、`intermediate_size=25600` 的 Qwen：
- 单层 buffer 约 `0.388 GiB`
- 48 层 rank 约 `18.6 GiB`

这会让 `profile_run()` 测得的 `runtime_overhead` 被严重放大，继而在 KV cache 初始化前就报：
`No available memory for the cache blocks`。

## 修复
改为 **runner 级共享 workspace**：
1. 在 `ForwardContext` 中新增 `workspace_buffers`
2. 在 v0/v1 GPU runner 上持有 `self.workspace_buffers`
3. `set_forward_context(...)` 进入 forward 时把 runner 的共享 workspace 传下去
4. `SiluAndMul` 不再持有模块级 `_output_buffer`
5. `SiluAndMul` 改为从 `get_forward_context().workspace_buffers["silu_and_mul"]` 里取共享 buffer；不足时再扩容

## 结果
- 仍保留 activation 输出复用
- 但显存占用从“每层一份”降为“每个 runner 一份”
- 理论上 Qwen rank1 会从额外 ~18.6 GiB 降到 ~0.388 GiB 量级

## 影响文件
- `vllm/forward_context.py`
- `vllm/model_executor/layers/activation.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/dynamic_gpu_model_runner.py`
- `vllm/worker/model_runner.py`

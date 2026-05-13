# project-kernal_cmp_6 direct/rr=3 崩溃排查

## 结论
- 复现目录：`new-logs/project-kernal_cmp_6/test-{kernel=direct}-{pp=40-40}-{rr=3}-{mig=0}-{in=512}-{out=16}-{n_req=100}-{rep=5}-{chunk=5}-{mig_mode=async}-{blk=128}-{vmm=1}-{kv_resize=1}-{cpu_cache=1}`
- 直接根因：rank 1 worker (`pid=81752`) 在 `activation.py:81` 对应的 `torch.ops._C.silu_and_mul` 路径执行期间触发 **SIGSEGV**。
- 崩溃点栈顶在 `c10::cuda::CUDACachingAllocator::Native::DeviceCachingAllocator::malloc()`，但 Python 栈显示正在 `dynamic_llama -> llama -> activation.forward_cuda` 中执行。
- 后续的 `Cannot connect to host localhost:8000`、`EngineDeadError`、`All requests failed` 都是该 worker 崩溃后的级联错误。

## 关键日志
- `server.log:12033` 左右：
  - `*** SIGSEGV received ...`
  - `PC: ... CUDACachingAllocator::...::malloc()`
  - `Fatal Python error: Segmentation fault`
- Python 栈：
  - `vllm/model_executor/layers/activation.py:81 in forward_cuda`
  - `vllm/model_executor/models/llama.py:98`
  - `vllm/model_executor/models/dynamic_llama.py:355`
  - `vllm/v1/worker/dynamic_gpu_model_runner.py:588`
- 同时 upstream rank 0 看到：
  - `dynamic_gpu_worker.py:2828 [FATAL] _listen_loop(from_rank=1) crashed: failed to recv, got 0 bytes`
  - 说明 rank 1 先崩，rank 0 的 recv 失败是次生现象。

## 现象特征
- 只有 `kernel=direct` 且 `rr=3` 的 `in=512/out=16` case 崩；
  - `flash rr=3` 无同类错误
  - `direct rr=1/2` 无同类错误
- 崩溃前日志显示较大 batch：
  - `Scheduled tokens: 3968`
  - 但 rank 1 收到的 `hidden_states` 形状是 `torch.Size([6673, 8192])`，`residual` 同 shape，总通信量约 `208.53 MB`
- 这说明高并发 direct 路径下，**实际前向张量规模异常放大/与调度数字不一致**，很可能导致 earlier kernel 写坏内存，最终在后续 `silu_and_mul` 申请输出张量时触发 allocator segfault。

## 初步判断
- 更像是 **direct kernel / ptr_table / flexi_direct 路径的内存破坏**，而不是普通 OOM：
  - 没有标准 CUDA OOM 异常
  - 是 `SIGSEGV`，并且只出现在 direct 路径
- `silu_and_mul` 大概率不是原始 bug 点，只是最先暴露损坏的位置。

## 建议
1. 优先在 direct 路径增加 fail-fast 校验：发送/接收 `IntermediateTensors` 时检查 `tensor.shape[0]` 与 `scheduler_output.total_num_scheduled_tokens` 是否一致。
2. 在 `flexi_direct` attention 调用前后增加 shape / ptr_table / num_blocks 一致性断言。
3. 若要先保证实验可跑，临时规避：
   - 使用 `flash`
   - 或降低 `rr` 到 `<=2`
   - 或降低 `max_num_batched_tokens`

# 2026-03-23 async migration layer-size 异常值排查

## 现象
在 `new-logs/project-async-fast_stop_time_layer-patch` 中，`migration_mode=async` 的 `migration process time` 没有随迁移 layer 数单调增大。

## 现有日志结论
对四组目标配置的 `server.log` 汇总后发现：

- `44-36`：`migration_total` 平均约 `5.64s`，但首个 target 迁移达 `18.0s`
- `48-32`：target 迁移稳定出现大值，约 `27.9s~29.3s`
- `52-28`：出现一次更极端离群值，约 `37.3s`
- `56-24`：反而更稳定，target 迁移约 `5.4s`

真正更像“异常源”的不是 KV 传输阶段，而是 `after weight loading`：

- `44-36`: `[15700, 650, 1100, 668, 1100] ms`
- `48-32`: `[26000, 1100, 25200, 1100, 24900] ms`
- `52-28`: `[4300, 1500, 33100, 1500, 2600] ms`
- `56-24`: `[3300, 1900, 3300, 2000, 3400] ms`

相比之下：

- `bind_kv_cache` 大多在 `270~615ms`
- `wait_for_kv_patch` 大多在 `320~616ms`
- `after_remove_layers` 大多在 `545~667ms`

因此当前更像是：**headline migration time 被 weight loading 阶段主导，而不是 layer 越多 -> KV migration 越慢。**

## 已添加的日志
### 1. worker 侧 `_add_layers`
文件：`vllm/v1/worker/dynamic_gpu_worker.py`

新增：
- `pre_add_cleanup` 耗时
- add_layers enqueue 耗时
- `migration_stream.synchronize()` 单独耗时
- add 前后 GPU free memory

### 2. loader 侧 `load_dynamic_layers`
文件：`vllm/model_executor/model_loader/dynamic_qwen3_loader.py`

新增：
- dynamic load 开始时的 layer 数、model range、cpu cache 状态
- `model.add_layers()` 结构扩展耗时
- `weights_to_load` 数量与样本
- `model.load_weights()` 产出的 tensor 数、总字节数、首个 tensor 延迟、每层字节分布
- `process_layer_weights_after_loading()` 单独耗时
- 各阶段前后 GPU free memory

### 3. engine 侧 weight loading dispatch
文件：`vllm/v1/engine/dynamic_core.py`

新增：
- `adding_per_rank` 派发时输出 `total_layers`

## 最小复现实验配置
新增：`vllm_exp/configs/debug-async-weight-loading-outlier-min.yaml`

特点：
- 只保留两组 target：`48-32`（异常）与 `56-24`（相对正常）
- `num_total_requests=120`
- `repetition=2`
- 其余关键参数保持与原实验一致：
  - `migration_approach=async`
  - `weight_loading_mode=sync`
  - `attention_kernel=direct`
  - `use_vmm=true`
  - `enable_kv_resize=true`
  - `enable_cpu_weight_cache=true`
  - `block_size=128`

## 当前判断
最需要确认的两个问题：
1. 大头时间究竟卡在 `model.load_weights()`，还是卡在 `process_layer_weights_after_loading()`
2. `migration_stream.synchronize()` 是否在等额外的历史 CUDA 工作，而不仅仅是本次 weight loading

下一步建议直接跑最小配置，先比较 `48-32` 和 `56-24` 两组新增日志。
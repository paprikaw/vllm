# 2026-03-23 async migration 异常值根因定位

## 最终结论
`migration process time` 的异常大值，根因在 **async 模式下 target 侧的 weight loading**，更具体地说，是 `cpu weight cache` 实现并没有把权重真正复制到常驻 CPU 内存，而是把 `safe_open(...).get_tensor()` 返回的 **mmap/file-backed tensor** 存进了 `_preloaded_weights`。

因此：
- 代码认为自己在走 `cpu_cache`
- 但第一次（或页缓存被回收后的再次）`model.load_weights()` 时，实际仍会在 H2D copy 过程中触发大量 page fault / 文件页回填
- 最终表现为：
  - `first_tensor_latency` 很小（说明 iterator 很快）
  - 但 `load_weights_took` 极大（说明慢在 tensor copy / page fault）

## 关键证据

### 1. 166/002 上日志正常，说明代码路径本身可线性扩展
在 `logs/project-debug-async-weight-loading-outlier-min-166-002` 中：
- 8 layers: `load_weights_took ≈ 1.4s`
- 16 layers: `load_weights_took ≈ 2.8s`

近似线性，`process_after_loading_summary` 仅 `~0.1-0.25s`，`migration_stream synchronize` 仅 `µs~ms`。

### 2. 140/001 上复现了异常，而且异常精确落在 `load_weights()`
在 `logs/project-debug-async-weight-loading-outlier-min` 中：

#### `tgt_pp=48-32`
- 第一轮 target migration:
  - `add_layers_structure`: `284ms`
  - `load_weights_took`: **25.0s**
  - `process_after_loading_summary`: `117ms`
  - `migration_stream synchronize`: `28µs`
  - `after weight loading`: `25.7s`
- 第二轮 target migration:
  - `load_weights_took`: `1.4s`

#### `tgt_pp=56-24`
- 第一轮 target migration:
  - `add_layers_structure`: `166ms`
  - `load_weights_took`: **41.4s**
  - `process_after_loading_summary`: `222ms`
  - `migration_stream synchronize`: `41µs`
  - `after weight loading`: `42.0s`
- 第二轮 target migration:
  - `load_weights_took`: `2.8s`

这说明：
- 慢不在 `process_layer_weights_after_loading`
- 慢不在 worker stream 同步
- 慢不在 KV transfer
- **慢就在 `model.load_weights()` 本身**

### 3. `first_tensor_latency` 始终只有 `~1ms`
这非常关键：
- 如果慢在 Python iterator / 查字典 / 正则过滤，首 tensor 延迟会很大
- 实际 `first_tensor_latency=1ms`，说明 generator 很快
- 但总 `load_weights_took` 高达 `25s / 41s`

因此瓶颈是：
**tensor 被取出来之后，真正被 `model.load_weights()` 使用时，底层内存页并不热，H2D copy 被 page fault 拖慢。**

## 根因代码位置
文件：`vllm/model_executor/model_loader/dynamic_qwen3_loader.py`

### 伪 CPU cache 的来源
`_preload_all_weights()` 中：
- `param = f.get_tensor(name)`
- 直接 `self._preloaded_weights[name] = param`

这里并没有 `clone()` / `copy()` 到真正独立的 CPU 常驻 buffer。
所以 `_preloaded_weights` 只是持有了 safetensors/mmap 返回的 tensor 引用。

`madvise(MADV_POPULATE_READ)` / `MADV_WILLNEED` 只是 hint，不保证页一直常驻，也不保证在 GPFS/NFS 场景下完全避免后续 page fault。

## 为什么会出现“中间 layer 数更差”
不是 layer 数本身有非单调 bug，而是：
- 某次 target migration 恰好命中了冷页 / 被回收页
- 那一轮 `load_weights()` 就会非常慢
- 于是 `migration process time` 被这段 hidden weight loading 放大

所以你看到的是 **页缓存冷热/回收行为** 叠加在不同实验上，而不是 migration layer 数与时间的简单函数关系。

## 最可能的修复方向
1. **把 CPU weight cache 变成真正的 CPU resident cache**
   - preload 时对 tensor 做实体复制，而不是仅保存 `get_tensor()` 返回值
   - 例如保存为独立 CPU tensor（必要时可考虑 pinned memory）
2. 如果内存成本太高，则至少：
   - 对即将迁移的层做显式 warm-up / materialization
   - 不要依赖 `madvise` 作为“已预加载”的充分条件

## 一句话总结
问题不是 async migration 的 KV patch 逻辑，而是 **所谓 CPU weight cache 实际上是 mmap-backed lazy cache，导致 `model.load_weights()` 在部分迁移轮次中被 page fault 拖慢。**

# Block Table 查找 KV Cache 地址的实现

本文档说明在 Flash Attention kernel 中，如何使用 `block_table` 来查找对应的 token 的 KV cache 地址。

## 核心函数

### `resolve_thread_kv_page_slice_offset` 函数

**文件**: `.deps/vllm-flash-attn-src/csrc/flash_attn/src/utils.h:300`

这是根据 `block_table` 计算 KV cache 地址的核心函数。

```cpp
template <typename Kernel_traits>
__forceinline__ __device__
int64_t resolve_thread_kv_page_slice_offset(
    const int tidx,                    // 线程索引
    const int n_block,                 // 当前处理的 N block 索引
    const int page_block_size,        // 每个 page block 的大小（通常是 16 的倍数）
    const int* block_table,           // block_table 指针
    const int page_stride,            // page 之间的步长
    const int row_stride,             // 行之间的步长
    std::optional<int> partial_block_size = std::nullopt  // 部分 block 的大小
)
```

#### 计算步骤：

1. **计算列偏移** (col_offset):
   ```cpp
   const int64_t col_offset = tidx % kGmemThreadsPerRow * kGmemElemsPerLoad;
   ```
   - 根据线程索引计算该线程负责的列偏移

2. **计算行偏移** (block_row_offset):
   ```cpp
   int64_t block_row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread;
   ```
   - 计算线程在 block 内的行偏移

3. **计算全局行偏移** (global_row_offset):
   ```cpp
   const int64_t global_row_offset = block_row_offset + n_block * kBlockN;
   ```
   - `n_block * kBlockN`: 当前 N block 的起始行
   - 加上线程在 block 内的行偏移

4. **计算虚拟页索引和页内偏移**:
   ```cpp
   const int64_t page_offset = global_row_offset % page_block_size;
   const int64_t virtual_page_idx = global_row_offset / page_block_size;
   ```
   - `virtual_page_idx`: 虚拟页索引（逻辑页号）
   - `page_offset`: 在页内的行偏移

5. **查找物理页地址并计算最终偏移**:
   ```cpp
   return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
       + page_offset * ((int64_t) row_stride)
       + col_offset;
   ```
   - `block_table[virtual_page_idx]`: 通过 block_table 将虚拟页索引转换为物理页索引
   - `* page_stride`: 物理页的起始地址偏移
   - `+ page_offset * row_stride`: 页内行偏移
   - `+ col_offset`: 列偏移

## 在 Kernel 中的使用

### 1. 初始化 block_table 指针

**文件**: `.deps/vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_kernel.h:584`

```cpp
const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
const int *block_table = params.block_table == nullptr ? nullptr 
    : params.block_table + bidb * params.block_table_batch_stride;
```

- 根据当前 batch 索引 (`bidb`) 获取对应的 `block_table` 行
- `block_table_batch_stride`: batch 之间的步长

### 2. 初始化 KV 地址（首次加载）

**文件**: `.deps/vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_kernel.h:632-638`

```cpp
if (block_table != nullptr) {
    auto final_block_size = binfo.actual_seqlen_k - (n_block_max - 1) * kBlockN;
    tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(
        tidx, n_block_max - 1, params.page_block_size,
        block_table, params.k_batch_stride, params.k_row_stride, final_block_size);
    tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(
        tidx, n_block_max - 1, params.page_block_size,
        block_table, params.v_batch_stride, params.v_row_stride, final_block_size);
}
```

- 在循环开始前，为最后一个 N block 计算初始地址
- 使用 `n_block_max - 1` 因为循环是从后往前遍历

### 3. 主循环中更新 K 地址

**文件**: `.deps/vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_kernel.h:921-928`

```cpp
if (n_block > n_block_min) {
    // Advance gK
    if (block_table == nullptr) {
        // 非 paged KV: 直接使用连续内存，通过步长计算
        tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
    } else {
        // Paged KV: 使用 block_table 重新计算地址
        tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(
            tidx, n_block - 1, params.page_block_size, 
            block_table, params.k_batch_stride, params.k_row_stride);
    }
    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKsK, tKVcKV, tKVpKV);
}
```

- 在每次循环迭代中，当需要加载下一个 K block 时
- 如果使用 paged KV，调用 `resolve_thread_kv_page_slice_offset` 重新计算地址
- 如果非 paged KV，直接通过步长偏移

### 4. 主循环中更新 V 地址

**文件**: `.deps/vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_kernel.h:886-892`

```cpp
// Advance gV
if (masking_step > 0) {
    if (block_table == nullptr) {
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
    } else {
        tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(
            tidx, n_block, params.page_block_size,
            block_table, params.v_batch_stride, params.v_row_stride);
    }
    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tVgV, tVsV, tKVcKV, tKVpKV);
}
```

- 类似地，在加载 V block 时也使用相同的机制

### 5. 其他使用位置

在代码中还有其他几个地方使用 `resolve_thread_kv_page_slice_offset`:

- **第 794-797 行**: 在 Append_KV 模式中更新 K 和 V 地址
- **第 890-891 行**: 在另一个循环路径中更新 V 地址
- **第 926-927 行**: 在另一个循环路径中更新 K 地址
- **第 966-967 行**: 在非 causal 路径中更新 V 地址
- **第 988-989 行**: 在非 causal 路径中更新 K 地址

## 地址计算原理

### Paged KV Cache 布局

```
KV Cache 物理内存布局:
[Block 0] [Block 1] [Block 2] ... [Block N]
每个 Block 包含 page_block_size 个 tokens

逻辑序列布局:
Seq 0: [Token 0-15] [Token 16-31] ...
Seq 1: [Token 0-15] [Token 16-31] ...

block_table 映射:
block_table[seq_idx][virtual_page_idx] = physical_block_number
```

### 地址计算示例

假设：
- `page_block_size = 16`
- `kBlockN = 128` (每个 N block 处理 128 个 tokens)
- `n_block = 2` (处理第 2 个 N block)
- `virtual_page_idx = 5` (逻辑页 5)

计算过程：
1. `global_row_offset = 2 * 128 = 256` (第 2 个 N block 的起始行)
2. `virtual_page_idx = 256 / 16 = 16` (逻辑页索引)
3. `page_offset = 256 % 16 = 0` (页内偏移)
4. `physical_block_number = block_table[16]` (查找物理块号)
5. `final_offset = physical_block_number * page_stride + 0 * row_stride + col_offset`

## 关键数据结构

### Flash_fwd_params 中的相关字段

```cpp
int *block_table;                    // block_table 指针
index_t block_table_batch_stride;     // batch 之间的步长
index_t k_batch_stride;               // K 的 batch 步长
index_t v_batch_stride;               // V 的 batch 步长
index_t k_row_stride;                 // K 的行步长
index_t v_row_stride;                 // V 的行步长
int page_block_size;                  // 每个 page block 的大小
```

## 总结

1. **核心函数**: `resolve_thread_kv_page_slice_offset` 负责根据 `block_table` 计算地址偏移
2. **使用位置**: 在 kernel 的多个位置调用，用于在循环中更新 K 和 V 的地址
3. **计算流程**: 
   - 计算逻辑页索引 (`virtual_page_idx`)
   - 通过 `block_table[virtual_page_idx]` 查找物理块号
   - 计算物理地址偏移
4. **优势**: 支持非连续的 KV cache 存储（PagedAttention），提高内存利用率


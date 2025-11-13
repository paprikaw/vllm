# Flash Attention 调用链 (Call Chain)

从 vLLM Python 层到 CUDA Kernel 的完整调用路径。

## 1. Python 层 (vLLM)

### 1.1 Attention Layer
**文件**: `vllm/attention/layer.py`

```python
# Attention.forward() 方法
self.impl.forward(self, query, key, value, self_kv_cache, attn_metadata, output=output)
```

### 1.2 FlashAttentionImpl
**文件**: `vllm/attention/backends/flash_attn.py`

```python
# FlashAttentionImpl.forward() 方法 (约第 808 行)
flash_attn_varlen_func(
    q=query,
    k=key,
    v=value,
    cu_seqlens_q=q_seq_start_loc,
    cu_seqlens_k=k_seq_start_loc,
    max_seqlen_q=q_seq_len,
    max_seqlen_k=k_seq_len,
    softmax_scale=softmax_scale,
    causal=_get_causal_option(attn_type),
    window_size=window_size,
    alibi_slopes=alibi_slopes,
    softcap=logits_soft_cap,
    out=prefill_output,
    fa_version=self.vllm_flash_attn_version,
    ...
)
```

## 2. Python 接口层

**文件**: `vllm/vllm_flash_attn/flash_attn_interface.py`

```python
# flash_attn_varlen_func() 函数 (约第 230 行)
out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(
    q, k, v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    None,
    block_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    False,
    causal,
    real_window_size[0],
    real_window_size[1],
    softcap,
    False,  # return_softmax
    None,   # gen
)
```

## 3. PyTorch C++ 绑定层

**文件**: `.deps/vllm-flash-attn-src/csrc/flash_attn/flash_api_torch_lib.cpp`

```cpp
// PyTorch 操作注册
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("varlen_fwd(...) -> Tensor[]");
    ops.impl("varlen_fwd", torch::kCUDA, make_pytorch_shim(&mha_varlen_fwd));
}
```

## 4. C++ API 层

**文件**: `.deps/vllm-flash-attn-src/csrc/flash_attn/flash_api.cpp`

### 4.1 入口函数
```cpp
// mha_varlen_fwd() 函数 (约第 516 行)
std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    std::optional<at::Tensor> &out_,
    const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k,
    ...
)
```

### 4.2 参数设置
```cpp
// 设置 Flash_fwd_params 结构体
Flash_fwd_params params;
set_params_fprop(params, ...);
```

### 4.3 Kernel 启动
```cpp
// run_mha_fwd() 函数 (约第 242 行)
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}
```

## 5. Kernel 分发层

**文件**: `.deps/vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_launch_template.h`

### 5.1 模板特化
```cpp
// 针对不同 head_dim 的特化实现
// 例如: flash_fwd_hdim128_bf16_sm80.cu
template<>
void run_mha_fwd_<cutlass::bfloat16_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128<cutlass::bfloat16_t, false>(params, stream);
}
```

### 5.2 Head Dimension 分发
```cpp
// run_mha_fwd_hdim128() 函数 (约第 226 行)
template<typename T, bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, 
                         Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, 
                         Is_dropout, Is_causal>(params, stream);
        }
    });
}
```

## 6. Kernel 启动层

**文件**: `.deps/vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_launch_template.h`

```cpp
// run_flash_fwd() 函数 (约第 54 行)
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    
    // 根据各种条件选择 kernel 模板参数
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH(...);
            BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                ALIBI_SWITCH(...);
                SOFTCAP_SWITCH(...);
                
                // 获取 kernel 函数指针
                auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout, Is_causal, ...>;
                
                // 设置共享内存大小
                if (smem_size >= 48 * 1024) {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                
                // 启动 CUDA kernel
                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}
```

## 7. CUDA Kernel 层

**文件**: `.deps/vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_launch_template.h`

```cpp
// flash_fwd_kernel 全局 kernel 函数
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, 
    bool Is_dropout, bool Is_causal, bool Is_local, 
    bool Has_alibi, bool Is_even_MN, bool Is_even_K, 
    bool Is_softcap, bool Return_softmax) {
    
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        FLASH_NAMESPACE::compute_attn<Kernel_traits, Is_dropout, Is_causal, 
            Is_local, Has_alibi, Is_even_MN, Is_even_K, 
            Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
```

## 8. 核心计算层

**文件**: `.deps/vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_kernel.h`

### 8.1 compute_attn 入口
```cpp
// compute_attn() 函数 (约第 63 行)
inline __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x;
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;
    
    FLASH_NAMESPACE::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, 
        Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
        params, bidb, bidh, m_block);
}
```

### 8.2 compute_attn_1rowblock 核心实现
```cpp
// compute_attn_1rowblock() 函数 (约第 52 行)
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, ...>
inline __device__ void compute_attn_1rowblock(
    const Params &params, const int bidb, const int bidh, const int m_block) {
    
    // 1. 加载 Q, K, V 到共享内存
    // 2. 计算 QK^T (GEMM)
    // 3. 应用 mask 和 softmax
    // 4. 计算 softmax(QK^T) * V
    // 5. 写回输出到全局内存
}
```

## 调用链总结

```
Python 层:
  Attention.forward()
    └─> FlashAttentionImpl.forward()
        └─> flash_attn_varlen_func()

Python 接口层:
  flash_attn_varlen_func()
    └─> torch.ops._vllm_fa2_C.varlen_fwd()

PyTorch C++ 绑定:
  torch.ops._vllm_fa2_C.varlen_fwd
    └─> mha_varlen_fwd()

C++ API 层:
  mha_varlen_fwd()
    ├─> set_params_fprop()          // 设置参数
    └─> run_mha_fwd()               // 启动 kernel

Kernel 分发层:
  run_mha_fwd()
    └─> run_mha_fwd_<T, Headdim, Is_causal>()
        └─> run_mha_fwd_hdim128()  // 或其他 head_dim
            └─> run_flash_fwd<Kernel_traits, Is_dropout, Is_causal>()

Kernel 启动层:
  run_flash_fwd()
    └─> flash_fwd_kernel<<<grid, threads, smem, stream>>>(params)

CUDA Kernel 层:
  flash_fwd_kernel()
    └─> compute_attn()
        └─> compute_attn_1rowblock()  // 核心计算逻辑
```

## 关键文件位置

1. **Python 层**:
   - `vllm/attention/layer.py` - Attention 类
   - `vllm/attention/backends/flash_attn.py` - FlashAttentionImpl 类

2. **Python 接口**:
   - `vllm/vllm_flash_attn/flash_attn_interface.py` - flash_attn_varlen_func()

3. **C++ API**:
   - `.deps/vllm-flash-attn-src/csrc/flash_attn/flash_api.cpp` - mha_varlen_fwd()
   - `.deps/vllm-flash-attn-src/csrc/flash_attn/flash_api_torch_lib.cpp` - PyTorch 绑定

4. **Kernel 启动**:
   - `.deps/vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_launch_template.h` - run_flash_fwd()

5. **Kernel 实现**:
   - `.deps/vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_kernel.h` - compute_attn_1rowblock()
   - `.deps/vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_hdim*.cu` - 各种 head_dim 的特化

## 关键数据结构

- **Flash_fwd_params**: 包含所有 kernel 需要的参数（指针、步长、形状等）
- **Kernel_traits**: 模板参数，定义 kernel 的特性（block 大小、线程数、数据类型等）
- **BlockInfo**: 处理变长序列的块信息


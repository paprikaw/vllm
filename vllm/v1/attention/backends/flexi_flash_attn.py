# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate
import os
import time

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch

from vllm import _custom_ops as ops
# yapf conflicts with isort for this block
# yapf: disable
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType,
                                              is_quantized_kv_cache)
# yapf: enable
from vllm.attention.backends.utils import (
    PAD_SLOT_ID, CommonAttentionState, compute_slot_mapping,
    compute_slot_mapping_start_idx, get_num_prefill_decode_query_kv_tokens,
    get_seq_len_block_table_args, is_all_cross_attn_metadata_set,
    is_all_encoder_attn_metadata_set, is_block_tables_empty)
from vllm.attention.utils.fa_utils import (flash_attn_supports_fp8,
                                           get_flash_attn_version)
from vllm.logger import init_logger
from vllm.multimodal import MultiModalPlaceholderMap
from vllm.utils import async_tensor_h2d, make_tensor_with_pad
from vllm.v1.utils import human_readable_duration
from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                  flash_attn_with_kvcache)
from vllm.v1.attention.backends.flash_attn import (FlashAttentionBackend, FlashAttentionImpl,FlashAttentionMetadata)
from vllm.vllm_flash_attn.flash_attn_interface import flexi_flash_attn_varlen_func, flexi_direct_flash_attn_varlen_func, prepare_flexi_kv_ptrs
if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)

logger = init_logger(__name__)


class FlexiFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls() -> Type["FlashAttentionImpl"]:
        from vllm.config import get_current_vllm_config
        vllm_config = get_current_vllm_config()
        # Both 'flexi' and 'direct' kernels use FlexiFlashAttentionImpl;
        # 'flash' uses the standard FlashAttentionImpl.
        if vllm_config.dynamic_config.use_flexi_kv:
            return FlexiFlashAttentionImpl
        return FlashAttentionImpl


class FlexiFlashAttentionImpl(FlashAttentionImpl):
    """
    Copied from FlashAttentionImpl with modifications to support
    flexible attention
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """
    def flexi_forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache_dev_ptr: int,
        v_cache_dev_ptr: int,
        page_meta: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        num_blocks: int,
        k_ptr_tables: Optional[torch.Tensor] = None,
        v_ptr_tables: Optional[torch.Tensor] = None,
        start_layer: int = 0,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
            k_ptr_tables: (num_layers, batch_size, max_num_blocks_per_seq), uint64, optional
            v_ptr_tables: (num_layers, batch_size, max_num_blocks_per_seq), uint64, optional
            start_layer: Starting layer index for computing local layer index
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run.
            return output

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens
        # Reshape the input keys and values and store them in the cache.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens] and
        # value[:num_actual_tokens] because the reshape_and_cache_flash op uses
        # the slot_mapping's shape to determine the number of actual tokens.
        
        
        # DEBUG: Verify that key_cache_list and key_cache share the same memory
        # assert key_cache_list[0].data_ptr() == key_cache[0].data_ptr(), \
        #     f"Memory mismatch! key_cache_list[0].data_ptr()={key_cache_list[0].data_ptr()}, key_cache[0].data_ptr()={key_cache[0].data_ptr()}" 

        torch.ops._C_cache_ops.flexi_reshape_and_cache_flash(
            key,
            value,
            k_cache_dev_ptr,
            v_cache_dev_ptr,
            page_meta,
            page_meta,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
        # torch.ops._C_cache_ops.reshape_and_cache_flash(
        #     key,
        #     value,
        #     key_cache,
        #     value_cache,
        #     attn_metadata.slot_mapping,
        #     self.kv_cache_dtype,
        #     layer._k_scale,
        #     layer._v_scale,
        # )
        
        # DEBUG: Verify KV cache write
        # torch.cuda.synchronize()
        # # Check first token's slot
        # if attn_metadata.slot_mapping.numel() > 0:
        #     slot_0 = attn_metadata.slot_mapping[0].item()
        #     block_idx = slot_0 // key_cache_meta.shape[0]  # block_size
        #     block_offset = slot_0 % key_cache_meta.shape[0]
        #     if block_idx < len(key_cache):
        #         cached_key = key_cache[block_idx][block_offset]
        #         input_key = key[0]
        #         diff = (cached_key - input_key).abs().max().item()
        #         logger.info(f"[FLEXI DEBUG] After write: slot_0={slot_0}, block_idx={block_idx}, block_offset={block_offset}")
        #         logger.info(f"[FLEXI DEBUG] Key diff (should be ~0): {diff}")
        #         if diff > 1e-3:
        #             logger.warning(f"[FLEXI DEBUG] Key mismatch! input_key[:5]={input_key.flatten()[:5].tolist()}, cached_key[:5]={cached_key.flatten()[:5].tolist()}")

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "Flexi FlashAttention does not support FP8 query on this "
                "device.")
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
            num_tokens, num_heads, head_size = query.shape
            query, _ = ops.scaled_fp8_quant(
                query.reshape(
                    (num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale)
            query = query.reshape((num_tokens, num_heads, head_size))

        # Compute attention and update output up to `num_actual_tokens`.
        use_local_attn = \
            (self.use_irope and attn_metadata.local_attn_metadata is not None)

        if not attn_metadata.use_cascade or use_local_attn:
            if use_local_attn:
                assert attn_metadata.local_attn_metadata is not None
                local_metadata = attn_metadata.local_attn_metadata
                cu_seqlens_q = local_metadata.local_query_start_loc
                seqused_k = local_metadata.local_seqused_k
                max_seqlen_q = local_metadata.local_max_query_len
                max_seqlen_k = local_metadata.local_max_seq_len
                block_table = local_metadata.local_block_table
                scheduler_metadata = local_metadata.local_scheduler_metadata
            else:
                cu_seqlens_q = attn_metadata.query_start_loc
                seqused_k = attn_metadata.seq_lens
                max_seqlen_q = attn_metadata.max_query_len
                max_seqlen_k = attn_metadata.max_seq_len
                block_table = attn_metadata.block_table
                scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])
            
            # Use flexi_direct if k_ptr_tables/v_ptr_tables are available
            use_flexi_direct = (k_ptr_tables is not None and 
                               v_ptr_tables is not None and 
                               k_ptr_tables.numel() > 0)

            try:
                if use_flexi_direct:
                    # Extract layer index from layer name to get the correct ptr_table
                    from vllm.v1.utils import extract_layer_index
                    # layer has layer_name attribute from Attention base class
                    layer_name = getattr(layer, 'layer_name', None)
                    if layer_name is None:
                        # Fallback to non-direct if layer_name not available
                        use_flexi_direct = False
                    else:
                        layer_idx = extract_layer_index(layer_name)
                        # k_ptr_tables is (num_layers, batch_size, max_num_blocks_per_seq)
                        # Convert global layer_idx to local index using start_layer
                        local_layer_idx = layer_idx - start_layer
                        
                        # Validate local_layer_idx is within bounds
                        if local_layer_idx < 0 or local_layer_idx >= k_ptr_tables.shape[0]:
                            # [LAYER_MISMATCH] Critical: layer index out of bounds for ptr_table
                            from vllm.logger import init_logger
                            _mismatch_logger = init_logger(__name__)
                            _mismatch_logger.error(f"[LAYER_MISMATCH] layer_idx={layer_idx}, start_layer={start_layer}, "
                                                  f"local_layer_idx={local_layer_idx}, k_ptr_tables.shape={k_ptr_tables.shape}, "
                                                  f"layer_name={layer_name}. Falling back to flexi mode!")
                            use_flexi_direct = False
                        else:
                            # Get per-batch ptr_table for this layer
                            k_ptr_table = k_ptr_tables[local_layer_idx]  # (batch_size, max_num_blocks_per_seq)
                            v_ptr_table = v_ptr_tables[local_layer_idx]  # (batch_size, max_num_blocks_per_seq)
                            flexi_direct_flash_attn_varlen_func(
                            q=query[:num_actual_tokens],
                            k_meta=page_meta,
                            v_meta=page_meta,
                            num_blocks=num_blocks,
                            k_ptr_table=k_ptr_table,
                            v_ptr_table=v_ptr_table,
                            max_seqlen_q=max_seqlen_q,
                            cu_seqlens_q=cu_seqlens_q,
                            max_seqlen_k=max_seqlen_k,
                            seqused_k=seqused_k,
                            softmax_scale=self.scale,
                            causal=True,
                            window_size=list(self.sliding_window) if self.sliding_window else None,
                            softcap=self.logits_soft_cap,
                            alibi_slopes=self.alibi_slopes,
                            out=output[:num_actual_tokens],
                        )

                if not use_flexi_direct:
                    # Flexi kernel: uses flexi_flash_attn_varlen_func with cached pointers
                    flexi_flash_attn_varlen_func(
                        q=query[:num_actual_tokens],
                        k_meta=page_meta,
                        v_meta=page_meta,
                        num_blocks=num_blocks,
                        out=output[:num_actual_tokens],
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=max_seqlen_q,
                        seqused_k=seqused_k,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=self.scale,
                        causal=True,
                        alibi_slopes=self.alibi_slopes,
                        window_size=self.sliding_window,
                        block_table=block_table,
                        softcap=self.logits_soft_cap,
                        scheduler_metadata=scheduler_metadata,
                        fa_version=self.vllm_flash_attn_version,
                        q_descale=layer._q_scale.expand(descale_shape),
                        k_descale=layer._k_scale.expand(descale_shape),
                        v_descale=layer._v_scale.expand(descale_shape),
                        cached_k_ptrs=k_cache_dev_ptr,
                        cached_v_ptrs=v_cache_dev_ptr,
                    )

            except Exception as e:
                logger.info(f"Error is happening, k_meta device: {page_meta.device}, v_meta device: {page_meta.device}, query device: {query.device}, output device: {output.device}, cached_k_ptrs: {k_cache_dev_ptr}, cached_v_ptrs: {v_cache_dev_ptr}, use_flexi_direct: {use_flexi_direct}")
                raise e
            # torch.testing.assert_close(output, copied_output, atol=2e-2, rtol=1e-2), \
            #     f"{torch.max(torch.abs(output - copied_output))}"
            return output
        raise NotImplementedError(
            "Flexi FlashAttention does not support cascade attention on this "
            "device.")
        assert not use_local_attn, (
            "Cascade attention does not support local attention.")
        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
        )
        return output

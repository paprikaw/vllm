# SPDX-License-Identifier: Apache-2.0
"""Attention layer."""
from typing import Any, Dict, List, Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm.envs as envs
from vllm.attention import AttentionType
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.platforms import _Backend, current_platform
from vllm.utils import direct_register_custom_op
from vllm.utils import init_logger
from vllm.attention.layer import Attention, unified_attention_with_output_fake
import time

from vllm.v1.attention.backends.flexi_flash_attn import FlexiFlashAttentionBackend, FlexiFlashAttentionImpl

logger = init_logger(__name__)

class FlexiAttention(Attention):
    """Copied from Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int | None = None, alibi_slopes: List[float] | None = None, cache_config: CacheConfig | None = None, quant_config: QuantizationConfig | None = None, blocksparse_params: Dict[str, Any] | None = None, logits_soft_cap: float | None = None, per_layer_sliding_window: int | None = None, use_mla: bool = False, prefix: str = "", attn_type: str = AttentionType.DECODER, **extra_impl_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads, alibi_slopes, cache_config, quant_config, blocksparse_params, logits_soft_cap, per_layer_sliding_window, use_mla, prefix, attn_type, **extra_impl_args)

        if per_layer_sliding_window is not None:
            # per-layer sliding window
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            # model-level sliding window
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            is_attention_free = cache_config.is_attention_free
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            is_attention_free = False
            calculate_kv_scales = False
        if num_kv_heads is None:
            num_kv_heads = num_heads
        attn_backend = FlexiFlashAttentionBackend
        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, sliding_window, kv_cache_dtype,
                             blocksparse_params, logits_soft_cap, attn_type,
                             **extra_impl_args)
        self.backend = backend_name_to_enum(attn_backend.get_name())
        self.key_dev_ptr: int
        self.value_dev_ptr: int
        self.num_blocks: int
        self.page_meta: torch.Tensor

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        if self.calculate_kv_scales:
            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata.enable_kv_scales_calculation:
                self.calc_kv_scales(query, key, value)
        if self.use_output:
            output_shape = (output_shape
                            if output_shape is not None else query.shape)
            output = torch.empty(output_shape,
                                 dtype=query.dtype,
                                 device=query.device)
            hidden_size = output_shape[-1]
            # We skip reshaping query, key and value tensors for the MLA
            # backend since these tensors have different semantics and are
            # processed differently.
            if not self.use_mla:
                # Reshape the query, key, and value tensors.
                # NOTE(woosuk): We do this outside the custom op to minimize the
                # CPU overheads from the non-CUDA-graph regions.
                query = query.view(-1, self.num_heads, self.head_size)
                output = output.view(-1, self.num_heads, self.head_size)
                if key is not None:
                    key = key.view(-1, self.num_kv_heads, self.head_size)
                if value is not None:
                    value = value.view(-1, self.num_kv_heads, self.head_size)
            if self.use_direct_call:
                forward_context: ForwardContext = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = torch.empty(0)
                # Debug assert: KV must be bound and non-empty
                # if os.environ.get("VLLM_DEBUG_ASSERT_KV", "1").lower() not in ("0", "", "false", "no"):
                #     assert isinstance(self_kv_cache, torch.Tensor) and self_kv_cache.numel() > 0, (
                #         f"Attention {self.layer_name} has empty KV cache bound")
                assert isinstance(self.impl, FlexiFlashAttentionImpl)
                assert False
                self.impl.flexi_forward(self,
                              query,
                              key,
                              value,
                              self_kv_cache,
                              self.key_dev_ptr,
                              self.value_dev_ptr,
                              attn_metadata,
                              output=output)
            else:
                import time
                attn_start = time.time()
                torch.ops.vllm.flexi_unified_attention_with_output(
                    query, key, value, output, self.layer_name)
                attn_time = time.time() - attn_start
                from vllm.logger import init_logger
                logger = init_logger(__name__)
                logger.debug(f"[perf_analysis] FlexiAttention {self.layer_name}: flexi_unified_attention took {attn_time:.4f}s")
            return output.view(-1, hidden_size)
        else:
            if self.use_direct_call:
                forward_context = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                assert isinstance(self.impl, FlexiFlashAttentionImpl)
                return self.impl.flexi_forward(self,
                                      query,
                                      key,
                                      value,
                                      self.key_dev_ptr,
                                      self.value_dev_ptr,
                                      self.page_meta,
                                        attn_metadata,
                                      self.num_blocks,
                                      )
            else:
                res = torch.ops.vllm.flexi_unified_attention(
                    query, key, value, self.layer_name)
                return res


def flexi_unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    assert False
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]

    from vllm.v1.attention.backends.flexi_flash_attn import FlexiFlashAttentionImpl
    assert isinstance(self.impl, FlexiFlashAttentionImpl)
    # self_kv_cache = self.kv_cache[forward_context.virtual_engine]
    output = self.impl.flexi_forward(self,
                  query,
                  key,
                  value,
                  self.key_cache,
                  self.value_cache,
                  self.key_dev_ptr,
                  self.value_dev_ptr,
                  attn_metadata)
    return output

def flexi_unified_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()

direct_register_custom_op(
    op_name="flexi_unified_attention",
    op_func=flexi_unified_attention,
    mutates_args=[],
    fake_impl=flexi_unified_attention_fake,
    dispatch_key=current_platform.dispatch_key,
)

def flexi_unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    # wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    from vllm.v1.attention.backends.flexi_flash_attn import FlexiFlashAttentionImpl
    assert isinstance(self.impl, FlexiFlashAttentionImpl)
    self.impl.flexi_forward(self,
                  query,
                  key,
                  value,
                  self.key_dev_ptr,
                  self.value_dev_ptr,
                  self.page_meta,
                  attn_metadata,
                  self.num_blocks,
                  output=output)
    # maybe_save_kv_layer_to_connector(layer_name, kv_cache)

def flexi_unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return

direct_register_custom_op(
    op_name="flexi_unified_attention_with_output",
    op_func=flexi_unified_attention_with_output,
    mutates_args=["output"],
    fake_impl=flexi_unified_attention_with_output_fake,
    dispatch_key=current_platform.dispatch_key,
)

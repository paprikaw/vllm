# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple

import torch
import vllm.envs as envs
from dataclasses import dataclass

def sanity_check_mm_encoder_outputs(
    mm_embeddings: object,
    expected_num_items: int,
) -> None:
    """
    Perform sanity checks for the result of
    [`vllm.model_executor.models.SupportsMultiModal.get_multimodal_embeddings`][].
    """
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")


def scatter_mm_placeholders(
    embeds: torch.Tensor,
    is_embed: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Scatter the multimodal embeddings into a contiguous tensor that represents
    the placeholder tokens.

    [`vllm.multimodal.processing.PromptUpdateDetails.is_embed`][].

    Args:
        embeds: The multimodal embeddings.
          Shape: `(num_embeds, embed_dim)`
        is_embed: A boolean mask indicating which positions in the placeholder
          tokens need to be filled with multimodal embeddings.
          Shape: `(num_placeholders, num_embeds)`
    """
    if is_embed is None:
        return embeds

    placeholders = embeds.new_full(
        (is_embed.shape[0], embeds.shape[-1]),
        fill_value=torch.nan,
    )
    placeholders[is_embed] = embeds
    return placeholders


def gather_mm_placeholders(
    placeholders: torch.Tensor,
    is_embed: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Reconstructs the embeddings from the placeholder tokens.

    This is the operation of [scatter_mm_placeholders][].
    """
    if is_embed is None:
        return placeholders

    return placeholders[is_embed]

def get_total_gpu_memory(rank: int) -> int:
    if envs.VLLM_PIPELINE_MEMORY_LIMIT:
        total_gpu_memory = envs.VLLM_PIPELINE_MEMORY_LIMIT[rank]
    else:   
        _, total_gpu_memory = torch.cuda.mem_get_info()
    return total_gpu_memory

def get_flexi_kv_cache(size: int, block_shape: Tuple[int, int, int], kv_cache_dtype: torch.dtype, device: torch.device) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    return [torch.empty(block_shape, dtype=kv_cache_dtype, device=device) for _ in range(size)], \
        [torch.empty(block_shape, dtype=kv_cache_dtype, device=device) for _ in range(size)]

def get_flexi_kv_cache_multi_stream(size: int, block_shape: Tuple[int, int, int], kv_cache_dtype: torch.dtype, device: torch.device, stream: Optional[torch.cuda.Stream] = None) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Create KV cache tensors on the specified device.
    
    Args:
        size: Number of cache blocks to create
        block_shape: Shape of each cache block
        kv_cache_dtype: Data type for cache tensors
        device: Target device for allocation
        stream: Optional CUDA stream for async allocation. If provided, allocations
                will be performed on this stream, enabling parallel creation of
                multiple KV caches.
    
    Returns:
        Tuple of (k_cache_list, v_cache_list)
    """
    k_cache = []
    v_cache = []
    if stream is not None and device.type == 'cuda':
        with torch.cuda.stream(stream):
            with torch.no_grad():
                for _ in range(size):
                    k_tensor = torch.empty(block_shape, dtype=kv_cache_dtype, device=device)
                    v_tensor = torch.empty(block_shape, dtype=kv_cache_dtype, device=device)
                    k_cache.append(k_tensor)
                    v_cache.append(v_tensor)
                # k_cache = [torch.empty(block_shape, dtype=kv_cache_dtype, device=device) for _ in range(size)]
                # v_cache = [torch.empty(block_shape, dtype=kv_cache_dtype, device=device) for _ in range(size)]
    else:
        with torch.no_grad():
            for _ in range(size):
                k_tensor = torch.empty(block_shape, dtype=kv_cache_dtype, device=device)
                v_tensor = torch.empty(block_shape, dtype=kv_cache_dtype, device=device)
                k_cache.append(k_tensor)
                v_cache.append(v_tensor)
    return k_cache, v_cache

@dataclass
class KVBufferStatus:
    used_tokens: dict[int, int]
    free_tokens: dict[int, int]
    capacity_tokens: dict[int, int]


def validate_layers_granularity(
    layers_list: list[Tuple[int, int]], 
    start_layer: int, 
    granularity: int,
    operation: str = "operation"
) -> None:
    """
    验证 layers_list 是否符合 combined_layers 的 granularity 约束。
    
    要求：
    1. 每个 layer range 的本地起始索引必须对齐到 group 边界 (local_start % granularity == 0)
    2. 每个 layer range 的本地结束索引+1 必须对齐到 group 边界 ((local_end + 1) % granularity == 0)
    
    Args:
        layers_list: 要操作的全局 layer 范围列表，如 [(36, 39)]
        start_layer: 当前 rank 的起始 layer 索引
        granularity: combined_layers 的 granularity 值
        operation: 操作名称（用于错误信息）
    
    Raises:
        ValueError: 如果 layers_list 不符合 granularity 约束
    """
    if granularity <= 1:
        return  # 非 combined_layers 模式，不需要验证
    
    for layers in layers_list:
        local_start = layers[0] - start_layer
        local_end = layers[1] - start_layer
        
        # 检查本地起始索引是否对齐到 group 边界
        if local_start % granularity != 0:
            raise ValueError(
                f"[{operation}] combined_layers mode requires layers aligned to group boundary. "
                f"layers={layers}, local_start={local_start}, granularity={granularity}, "
                f"local_start % granularity = {local_start % granularity} != 0"
            )
        
        # 检查本地结束索引+1是否对齐到 group 边界（确保删除完整的 groups）
        if (local_end + 1) % granularity != 0:
            raise ValueError(
                f"[{operation}] combined_layers mode requires layers aligned to group boundary. "
                f"layers={layers}, local_end={local_end}, granularity={granularity}, "
                f"(local_end + 1) % granularity = {(local_end + 1) % granularity} != 0"
            )


def validate_layers_count_granularity(
    layers_list: list[Tuple[int, int]],
    granularity: int,
    operation: str = "operation"
) -> None:
    """
    验证 layers_list 的总层数是否符合 granularity 约束。
    
    用于添加层操作：确保添加的层数是完整的 groups。
    
    Args:
        layers_list: 要操作的 layer 范围列表，如 [(36, 39)]
        granularity: combined_layers 的 granularity 值
        operation: 操作名称（用于错误信息）
    
    Raises:
        ValueError: 如果总层数不是 granularity 的整数倍
    """
    if granularity <= 1:
        return  # 非 combined_layers 模式，不需要验证
    
    total_layers = sum(layers[1] - layers[0] + 1 for layers in layers_list)
    if total_layers % granularity != 0:
        raise ValueError(
            f"[{operation}] combined_layers mode requires complete groups. "
            f"layers_list={layers_list}, total_layers={total_layers}, granularity={granularity}, "
            f"total_layers % granularity = {total_layers % granularity} != 0"
        )
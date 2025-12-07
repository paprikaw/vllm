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

@dataclass
class KVBufferStatus:
    used_tokens: dict[int, int]
    free_tokens: dict[int, int]
    capacity_tokens: dict[int, int]
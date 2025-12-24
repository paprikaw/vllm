# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch

from tests.kernels.utils import DEFAULT_OPCHECK_TEST_UTILS, opcheck
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    is_fa_version_supported,
    prepare_flexi_kv_ptrs,
    free_flexi_kv_ptrs
)

COPYING_DIRECTION = [('cuda', 'cpu'), ('cuda', 'cuda'), ('cpu', 'cuda')]
DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 120, 256]
BLOCK_SIZES = [8, 16, 32]
CACHE_LAYOUTS = ["NHD", "HND"]

# Parameters for MLA tests.
KV_LORA_RANKS = [512]
QK_ROPE_HEAD_DIMS = [64]
NUM_TOKENS_MLA = [42]
BLOCK_SIZES_MLA = [16]
NUM_BLOCKS_MLA = [8]

# Arbitrary values for testing
# don't make it too large. e.g. [1024, 36000] will OOM
NUM_BLOCKS = [1024, 10000]

NUM_MAPPINGS = [256]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

# We assume fp8 is always enabled for testing.
KV_CACHE_DTYPE = ["auto", "fp8"]


@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_copy_blocks(
    kv_cache_factory,
    num_mappings: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    kv_cache_dtype: str,
    device: str,
) -> None:
    if kv_cache_dtype == "fp8" and head_size % 16:
        pytest.skip()
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    # Generate random block mappings where each source block is mapped to two
    # destination blocks.
    assert 2 * num_mappings <= num_blocks
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remainig_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remainig_blocks, 2 * num_mappings)
    block_mapping: list[tuple[int, int]] = []
    for i in range(num_mappings):
        src = src_blocks[i]
        dst1 = dst_blocks[2 * i]
        dst2 = dst_blocks[2 * i + 1]
        block_mapping.append((src, dst1))
        block_mapping.append((src, dst2))

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size,
                                                num_layers, num_heads,
                                                head_size, kv_cache_dtype,
                                                dtype, seed, device)

    # Clone the KV caches.
    cloned_key_caches = [key_cache.clone() for key_cache in key_caches]
    cloned_value_caches = [value_cache.clone() for value_cache in value_caches]

    # Call the copy blocks kernel.
    block_mapping_tensor = torch.tensor(block_mapping,
                                        dtype=torch.int64,
                                        device=device).view(-1, 2)

    opcheck(torch.ops._C_cache_ops.copy_blocks,
            (key_caches, value_caches, block_mapping_tensor),
            test_utils=DEFAULT_OPCHECK_TEST_UTILS,
            cond=(head_size == HEAD_SIZES[0]))
    ops.copy_blocks(key_caches, value_caches, block_mapping_tensor)

    # Run the reference implementation.
    for src, dst in block_mapping:
        for cloned_key_cache in cloned_key_caches:
            cloned_key_cache[dst].copy_(cloned_key_cache[src])
        for cloned_value_cache in cloned_value_caches:
            cloned_value_cache[dst].copy_(cloned_value_cache[src])

    # Compare the results.
    for key_cache, cloned_key_cache in zip(key_caches, cloned_key_caches):
        torch.testing.assert_close(key_cache, cloned_key_cache)
    for value_cache, cloned_value_cache in zip(value_caches,
                                               cloned_value_caches):
        torch.testing.assert_close(value_cache, cloned_value_cache)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_reshape_and_cache(
    kv_cache_factory,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    if kv_cache_dtype == "fp8" and head_size % 16:
        pytest.skip()
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long)

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    # Clone the KV caches.
    if kv_cache_dtype == "fp8":
        cloned_key_cache = torch.empty_like(key_cache, dtype=torch.float16)
        ops.convert_fp8(cloned_key_cache, key_cache, k_scale.item())
        cloned_value_cache = torch.empty_like(value_cache, dtype=torch.float16)
        ops.convert_fp8(cloned_value_cache, value_cache, v_scale.item())
    else:
        cloned_key_cache = key_cache.clone()
        cloned_value_cache = value_cache.clone()

    # Call the reshape_and_cache kernel.
    opcheck(torch.ops._C_cache_ops.reshape_and_cache,
            (key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype,
             k_scale, v_scale),
            cond=(head_size == HEAD_SIZES[0]))
    ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping,
                          kv_cache_dtype, k_scale, v_scale)

    if kv_cache_dtype == "fp8":
        result_key_cache = torch.empty_like(key_cache, dtype=torch.float16)
        ops.convert_fp8(result_key_cache, key_cache, k_scale.item())
        result_value_cache = torch.empty_like(value_cache, dtype=torch.float16)
        ops.convert_fp8(result_value_cache, value_cache, v_scale.item())

    # Run the reference implementation.
    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies_lst = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies_lst[i]
        block_offset = block_offsets_lst[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    if kv_cache_dtype == "fp8":
        torch.testing.assert_close(result_key_cache,
                                   cloned_key_cache,
                                   atol=0.001,
                                   rtol=0.1)
        torch.testing.assert_close(result_value_cache,
                                   cloned_value_cache,
                                   atol=0.001,
                                   rtol=0.1)
    else:
        torch.testing.assert_close(key_cache, cloned_key_cache)
        torch.testing.assert_close(value_cache, cloned_value_cache)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("kv_cache_layout", CACHE_LAYOUTS)
@torch.inference_mode()
def test_reshape_and_cache_flash(
    kv_cache_factory_flashinfer,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
    kv_cache_layout: str,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    # fp8 conversion requires continugous memory buffer. Reduce the number of
    # blocks and tokens to consume less memory.
    num_tokens = num_tokens // 2
    num_blocks = num_blocks // 2
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst,
                                dtype=torch.long,
                                device=device)
    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device=device)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory_flashinfer(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        device=device,
        cache_layout=kv_cache_layout,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    del key_caches
    del value_caches


    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    def permute_and_compact(x):
        y = x if kv_cache_layout == "NHD" else x.permute(0, 2, 1, 3)
        return y.contiguous()

    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    # Clone the KV caches.
    if kv_cache_dtype == "fp8":
        cloned_key_cache = torch.empty_like(key_cache_compact,
                                            dtype=torch.float16)
        ops.convert_fp8(cloned_key_cache, key_cache_compact, k_scale.item(),
                        kv_cache_dtype)
        cloned_value_cache = torch.empty_like(value_cache_compact,
                                              dtype=torch.float16)
        ops.convert_fp8(cloned_value_cache, value_cache_compact,
                        v_scale.item(), kv_cache_dtype)
    else:
        cloned_key_cache = key_cache_compact.clone()
        cloned_value_cache = value_cache_compact.clone()
    # Call the reshape_and_cache kernel.
    opcheck(torch.ops._C_cache_ops.reshape_and_cache_flash,
            (key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype,
             k_scale, v_scale),
            cond=(head_size == HEAD_SIZES[0]))
    ops.reshape_and_cache_flash(key, value, key_cache, value_cache,
                                slot_mapping, kv_cache_dtype, k_scale, v_scale)
    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    if kv_cache_dtype == "fp8":
        result_key_cache = torch.empty_like(key_cache_compact,
                                            dtype=torch.float16)
        ops.convert_fp8(result_key_cache,
                        key_cache_compact,
                        k_scale.item(),
                        kv_dtype=kv_cache_dtype)
        result_value_cache = torch.empty_like(value_cache_compact,
                                              dtype=torch.float16)
        ops.convert_fp8(result_value_cache,
                        value_cache_compact,
                        v_scale.item(),
                        kv_dtype=kv_cache_dtype)

    # Run the reference implementation.
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies_lst = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies_lst[i]
        block_offset = block_offsets_lst[i]
        if kv_cache_layout == "NHD":
            cloned_key_cache[block_idx, block_offset, :, :] = key[i]
            cloned_value_cache[block_idx, block_offset, :, :] = value[i]
        else:
            cloned_key_cache[block_idx, :, block_offset, :] = key[i]
            cloned_value_cache[block_idx, :, block_offset, :] = value[i]

    if kv_cache_dtype == "fp8":
        torch.testing.assert_close(result_key_cache,
                                   cloned_key_cache,
                                   atol=0.001,
                                   rtol=0.1)
        torch.testing.assert_close(result_value_cache,
                                   cloned_value_cache,
                                   atol=0.001,
                                   rtol=0.1)
    else:
        torch.testing.assert_close(key_cache_compact, cloned_key_cache)
        torch.testing.assert_close(value_cache_compact, cloned_value_cache)

@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("kv_cache_layout", CACHE_LAYOUTS)
@torch.inference_mode()
def test_flexi_reshape_and_cache_flash(
    kv_cache_factory_flashinfer,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
    kv_cache_layout: str,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    # fp8 conversion requires continugous memory buffer. Reduce the number of
    # blocks and tokens to consume less memory.
    num_tokens = num_tokens // 2
    num_blocks = num_blocks // 2
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst,
                                dtype=torch.long,
                                device=device)
    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device=device)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory_flashinfer(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        device=device,
        cache_layout=kv_cache_layout,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    del key_caches
    del value_caches

    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    def permute_and_compact(x):
        y = x if kv_cache_layout == "NHD" else x.permute(0, 2, 1, 3)
        return y.contiguous()

    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)
    k_pages = []
    v_pages = []

    for k_tensor in key_cache:
        k_pages.append(k_tensor)
    for v_tensor in value_cache:
        v_pages.append(v_tensor)
    k_ptrs, v_ptrs = prepare_flexi_kv_ptrs(k_pages, v_pages);

    # Clone the KV caches.
    if kv_cache_dtype == "fp8":
        cloned_key_cache = torch.empty_like(key_cache_compact,
                                            dtype=torch.float16)
        ops.convert_fp8(cloned_key_cache, key_cache_compact, k_scale.item(),
                        kv_cache_dtype)
        cloned_value_cache = torch.empty_like(value_cache_compact,
                                              dtype=torch.float16)
        ops.convert_fp8(cloned_value_cache, value_cache_compact,
                        v_scale.item(), kv_cache_dtype)
    else:
        cloned_key_cache = key_cache_compact.clone()
        cloned_value_cache = value_cache_compact.clone()
    # Call the reshape_and_cache kernel.
    opcheck(torch.ops._C_cache_ops.flexi_reshape_and_cache_flash,
            (key, value, k_ptrs, v_ptrs,
                                k_pages[0], v_pages[0], slot_mapping, kv_cache_dtype, k_scale, v_scale),
            cond=(head_size == HEAD_SIZES[0]))
    ops.flexi_reshape_and_cache_flash(key, value, k_ptrs, v_ptrs,
                                k_pages[0], v_pages[0], slot_mapping, kv_cache_dtype, k_scale, v_scale)
    
    # After kernel execution, read back the results
    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    if kv_cache_dtype == "fp8":
        result_key_cache = torch.empty_like(key_cache_compact,
                                            dtype=torch.float16)
        ops.convert_fp8(result_key_cache,
                        key_cache_compact,
                        k_scale.item(),
                        kv_dtype=kv_cache_dtype)
        result_value_cache = torch.empty_like(value_cache_compact,
                                              dtype=torch.float16)
        ops.convert_fp8(result_value_cache,
                        value_cache_compact,
                        v_scale.item(),
                        kv_dtype=kv_cache_dtype)

    # Run the reference implementation.
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies_lst = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies_lst[i]
        block_offset = block_offsets_lst[i]
        if kv_cache_layout == "NHD":
            cloned_key_cache[block_idx, block_offset, :, :] = key[i]
            cloned_value_cache[block_idx, block_offset, :, :] = value[i]
        else:
            cloned_key_cache[block_idx, :, block_offset, :] = key[i]
            cloned_value_cache[block_idx, :, block_offset, :] = value[i]

    if kv_cache_dtype == "fp8":
        torch.testing.assert_close(result_key_cache,
                                   cloned_key_cache,
                                   atol=0.001,
                                   rtol=0.1)
        torch.testing.assert_close(result_value_cache,
                                   cloned_value_cache,
                                   atol=0.001,
                                   rtol=0.1)
    else:
        torch.testing.assert_close(key_cache_compact, cloned_key_cache)
        torch.testing.assert_close(value_cache_compact, cloned_value_cache)



@pytest.mark.parametrize("direction", COPYING_DIRECTION)
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_swap_blocks(
    kv_cache_factory,
    direction: tuple[str, str],
    num_mappings: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    if kv_cache_dtype == "fp8" and "cpu" in direction:
        pytest.skip()
    if kv_cache_dtype == "fp8" and head_size % 16:
        pytest.skip()

    current_platform.seed_everything(seed)

    src_device = device if direction[0] == "cuda" else 'cpu'
    dst_device = device if direction[1] == "cuda" else 'cpu'

    src_blocks = random.sample(range(num_blocks), num_mappings)
    # For the same device, mapping must not overlap
    if src_device == dst_device:
        remaining_blocks = list(set(range(num_blocks)) - set(src_blocks))
        dst_blocks = random.sample(remaining_blocks, num_mappings)
    else:
        dst_blocks = random.sample(range(num_blocks), num_mappings)

    block_mapping = list(zip(src_blocks, dst_blocks))
    block_mapping_tensor = torch.tensor(block_mapping,
                                        dtype=torch.int64,
                                        device="cpu").view(-1, 2)

    # Create the KV caches on the first device.
    src_key_caches, src_value_caches = kv_cache_factory(
        num_blocks, block_size, 1, num_heads, head_size, kv_cache_dtype, dtype,
        seed, src_device)

    # Create the KV caches on the second device.
    dist_key_caches, dist_value_caches = kv_cache_factory(
        num_blocks, block_size, 1, num_heads, head_size, kv_cache_dtype, dtype,
        seed, dst_device)

    src_key_caches_clone = src_key_caches[0].clone()
    src_value_caches_clone = src_value_caches[0].clone()

    # Call the swap_blocks kernel.
    do_opcheck = (head_size == HEAD_SIZES[0])
    opcheck(torch.ops._C_cache_ops.swap_blocks,
            (src_key_caches[0], dist_key_caches[0], block_mapping_tensor),
            cond=do_opcheck)
    opcheck(torch.ops._C_cache_ops.swap_blocks,
            (src_value_caches[0], dist_value_caches[0], block_mapping_tensor),
            cond=do_opcheck)

    ops.swap_blocks(src_key_caches[0], dist_key_caches[0],
                    block_mapping_tensor)
    ops.swap_blocks(src_value_caches[0], dist_value_caches[0],
                    block_mapping_tensor)

    for src, dst in block_mapping:
        torch.testing.assert_close(src_key_caches_clone[src].cpu(),
                                   dist_key_caches[0][dst].cpu())
        torch.testing.assert_close(src_value_caches_clone[src].cpu(),
                                   dist_value_caches[0][dst].cpu())


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_fp8_e4m3_conversion(
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    current_platform.seed_everything(seed)

    low = -224.0
    high = 224.0
    shape = (num_blocks, num_heads, head_size, block_size)
    cache = torch.empty(shape, dtype=dtype, device=device)
    cache.uniform_(low, high)

    cache_fp8 = torch.empty_like(cache, dtype=torch.uint8)
    ops.convert_fp8(cache_fp8, cache)

    converted_cache = torch.empty_like(cache)
    ops.convert_fp8(converted_cache, cache_fp8)

    torch.testing.assert_close(cache, converted_cache, atol=0.001, rtol=0.1)


def _create_mla_cache(
    num_blocks: int,
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> torch.Tensor:
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    return torch.zeros(num_blocks,
                       block_size,
                       entry_size,
                       dtype=cache_dtype,
                       device=device)


def _fill_mla_cache(cache: torch.Tensor, kv_cache_dtype: str):
    rand_dtype = torch.float16 if kv_cache_dtype == "fp8" else cache.dtype

    vals = torch.randn(*cache.shape, device=cache.device, dtype=rand_dtype)
    if kv_cache_dtype == "fp8":
        temp = torch.zeros_like(cache)
        ops.convert_fp8(temp, vals, 1.0, kv_dtype=kv_cache_dtype)
        vals = temp
    cache.copy_(vals)


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_concat_and_cache_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst,
                                dtype=torch.long,
                                device=device)

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens,
                       qk_rope_head_dim,
                       dtype=dtype,
                       device=device)
    entry_size = kv_lora_rank + qk_rope_head_dim

    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    kv_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                 kv_cache_dtype, device)
    ref_temp = torch.zeros(*kv_cache.shape, dtype=dtype, device=device)

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
        ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

    if kv_cache_dtype == "fp8":
        ref_kv_cache = torch.empty_like(ref_temp, dtype=kv_cache.dtype)
        ops.convert_fp8(ref_kv_cache,
                        ref_temp,
                        scale.item(),
                        kv_dtype=kv_cache_dtype)
    else:
        ref_kv_cache = ref_temp

    opcheck(
        torch.ops._C_cache_ops.concat_and_cache_mla,
        (kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
                             kv_cache_dtype, scale)

    if kv_cache_dtype == "fp8":
        result_temp = torch.empty_like(kv_cache, dtype=torch.float16)
        ops.convert_fp8(result_temp,
                        kv_cache.contiguous(),
                        scale.item(),
                        kv_dtype=kv_cache_dtype)
        expected_temp = torch.empty_like(ref_kv_cache, dtype=torch.float16)
        ops.convert_fp8(expected_temp,
                        ref_kv_cache,
                        scale.item(),
                        kv_dtype=kv_cache_dtype)
        torch.testing.assert_close(result_temp,
                                   expected_temp,
                                   atol=0.001,
                                   rtol=0.1)
    else:
        torch.testing.assert_close(kv_cache, ref_kv_cache)


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_copy_blocks_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    num_blocks: int,
    num_layers: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    entry_size = kv_lora_rank + qk_rope_head_dim

    kv_caches = []
    for _ in range(num_layers):
        kv_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                     kv_cache_dtype, device)
        _fill_mla_cache(kv_cache, kv_cache_dtype=kv_cache_dtype)
        kv_caches.append(kv_cache)

    ref_caches = [kv_cache.clone() for kv_cache in kv_caches]

    num_mappings = min(2, num_blocks // 2)
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remaining = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remaining, 2 * num_mappings)
    block_mapping = []
    for i in range(num_mappings):
        src = src_blocks[i]
        dst1 = dst_blocks[2 * i]
        dst2 = dst_blocks[2 * i + 1]
        block_mapping.append((src, dst1))
        block_mapping.append((src, dst2))
    block_mapping_tensor = torch.tensor(block_mapping,
                                        dtype=torch.int64,
                                        device=device).view(-1, 2)

    for src, dst in block_mapping:
        for ref_cache in ref_caches:
            ref_cache[dst].copy_(ref_cache[src])

    opcheck(
        torch.ops._C_cache_ops.copy_blocks_mla,
        (kv_caches, block_mapping_tensor),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )
    ops.copy_blocks_mla(kv_caches, block_mapping_tensor)

    for kv_cache, ref_cache in zip(kv_caches, ref_caches):
        torch.testing.assert_close(kv_cache, ref_cache)


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_swap_blocks_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    entry_size = kv_lora_rank + qk_rope_head_dim

    src_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                  kv_cache_dtype, device)
    dst_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                  kv_cache_dtype, device)

    _fill_mla_cache(src_cache, kv_cache_dtype)
    _fill_mla_cache(dst_cache, kv_cache_dtype)

    src_cache_clone = src_cache.clone()

    num_mappings = min(2, num_blocks // 2)
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remaining_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remaining_blocks, num_mappings)
    block_mapping = list(zip(src_blocks, dst_blocks))
    block_mapping_tensor = torch.tensor(block_mapping,
                                        dtype=torch.int64,
                                        device="cpu").view(-1, 2)

    opcheck(
        torch.ops._C_cache_ops.swap_blocks,
        (src_cache, dst_cache, block_mapping_tensor),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.swap_blocks(src_cache, dst_cache, block_mapping_tensor)

    for src, dst in block_mapping:
        torch.testing.assert_close(
            src_cache_clone[src].cpu(),
            dst_cache[dst].cpu(),
            msg=f"Block {src} from src should have been swapped to block "
            f"{dst} in dst_cache.")


@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [1024])
@pytest.mark.parametrize("max_seq_len", [512])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype",
                         ["auto"])  # You can also test "fp8" if needed.
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_gather_cache_mla(kv_lora_rank, qk_rope_head_dim, block_size,
                          num_blocks, max_seq_len, batch_size, dtype,
                          kv_cache_dtype, device):
    entry_size = kv_lora_rank + qk_rope_head_dim
    src_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                  kv_cache_dtype, device)
    _fill_mla_cache(src_cache, kv_cache_dtype=kv_cache_dtype)

    seq_len_tensor = torch.randint(0,
                                   max_seq_len + 1, (batch_size, ),
                                   device=device)

    total_tokens = seq_len_tensor.sum()
    cu_seq_lens = torch.empty((batch_size + 1),
                              dtype=torch.int32,
                              device=device)
    cu_seq_lens[0] = 0
    cu_seq_lens[1:] = seq_len_tensor.cumsum(dim=0).to(dtype=torch.int32)
    print("seq_len_tensor", seq_len_tensor)

    tot_blocks_tensor = (seq_len_tensor + block_size - 1) // block_size
    block_table = torch.empty((batch_size, num_blocks),
                              dtype=torch.int32,
                              device=device)

    for b in range(batch_size):
        perm = torch.randperm(num_blocks, device=device)
        block_table[b, :] = perm

    dst = torch.zeros((total_tokens, entry_size),
                      dtype=src_cache.dtype,
                      device=device)

    expected_batches = []
    for b in range(batch_size):
        s = seq_len_tensor[b]
        if s == 0:
            continue
        tot = tot_blocks_tensor[b]
        blocks = block_table[b, :tot].tolist()

        gathered_rows = []
        for i in range(tot - 1):
            gathered_rows.append(src_cache[blocks[i]])
        remaining = s - (tot - 1) * block_size
        gathered_rows.append(src_cache[blocks[-1], :remaining, :])

        batch_expected = torch.cat(gathered_rows, dim=0)
        expected_batches.append(batch_expected)
    expected = torch.cat(expected_batches, dim=0)

    opcheck(
        torch.ops._C_cache_ops.gather_cache,
        (src_cache, dst, block_table, cu_seq_lens, batch_size, None),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.gather_cache(src_cache, dst, block_table, cu_seq_lens, batch_size)
    torch.testing.assert_close(dst, expected)


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.cpu_model
@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
@torch.inference_mode()
def test_concat_and_cache_mla_cpu(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    device = "cpu"
    kv_cache_dtype = "auto"
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst,
                                dtype=torch.long,
                                device=device)

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens,
                       qk_rope_head_dim,
                       dtype=dtype,
                       device=device)
    entry_size = kv_lora_rank + qk_rope_head_dim

    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    kv_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                 kv_cache_dtype, device)
    ref_temp = torch.zeros(*kv_cache.shape, dtype=dtype, device=device)

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
        ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

    if kv_cache_dtype == "fp8":
        ref_kv_cache = torch.empty_like(ref_temp, dtype=kv_cache.dtype)
        ops.convert_fp8(ref_kv_cache,
                        ref_temp,
                        scale.item(),
                        kv_dtype=kv_cache_dtype)
    else:
        ref_kv_cache = ref_temp

    opcheck(
        torch.ops._C_cache_ops.concat_and_cache_mla,
        (kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
                             kv_cache_dtype, scale)
    torch.testing.assert_close(kv_cache, ref_kv_cache)

@pytest.mark.parametrize("num_tokens", [1, 83, 100])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [100, 1000])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_flexi_gather_and_reshape_roundtrip(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    """
    Test that simulates sender-receiver KV cache migration:
    1. Sender: Use flexi_gather_pages to gather KV data from cache into a contiguous tensor
    2. Receiver: Use flexi_reshape_and_cache_flash to write the gathered data into a new cache
    3. Verify that the original and recovered data match
    """
    torch.set_default_device(device)
    
    # === SENDER SIDE: Create source cache with random data ===
    # Each page: [block_size, num_heads, head_size]
    src_key_pages = [torch.randn(block_size, num_heads, head_size, dtype=dtype, device=device) for _ in range(num_blocks)]
    src_value_pages = [torch.randn(block_size, num_heads, head_size, dtype=dtype, device=device) for _ in range(num_blocks)]
    
    # Prepare source pointers
    src_key_ptrs, src_value_ptrs = prepare_flexi_kv_ptrs(src_key_pages, src_value_pages)
    
    # Create slot mapping - simulate which tokens to migrate
    # Use random slots that don't exceed the available slots
    total_slots = num_blocks * block_size
    slot_mapping = torch.tensor(
        random.sample(range(total_slots), num_tokens),
        dtype=torch.int64, 
        device=device
    )
    slot_mapping = torch.tensor(
        random.sample(range(total_slots), num_tokens),
        dtype=torch.int64, 
        device=device
    )
    
    # === SENDER: Gather KV data from source cache ===
    # Output shape: [num_tokens, num_heads, head_size]
    gathered_key = torch.empty(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    gathered_value = torch.empty(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    
    ops.flexi_gather_pages(src_key_ptrs, src_value_ptrs, slot_mapping, gathered_key, gathered_value, block_size)
    
    # === RECEIVER SIDE: Create destination cache ===
    # Receiver allocates new cache - same size as sender for this test
    dst_key_pages = [torch.zeros(block_size, num_heads, head_size, dtype=dtype, device=device) for _ in range(num_blocks)]
    dst_value_pages = [torch.zeros(block_size, num_heads, head_size, dtype=dtype, device=device) for _ in range(num_blocks)]
    
    # Prepare destination pointers
    dst_key_ptrs, dst_value_ptrs = prepare_flexi_kv_ptrs(dst_key_pages, dst_value_pages)
    
    # Dummy scales for auto dtype
    k_scale = torch.tensor(2.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    # === RECEIVER: Write gathered data into destination cache ===
    # Use the SAME slot_mapping - this simulates sender and receiver using compatible slot mappings
    ops.flexi_reshape_and_cache_flash(
        gathered_key, gathered_value,
        dst_key_ptrs, dst_value_ptrs,
        dst_key_pages[0], dst_value_pages[0],  # meta tensors for stride info
        slot_mapping,
        "auto",
        k_scale, v_scale
    )
    
    # === VERIFICATION: Compare source and destination at the same slots ===
    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        page_idx = slot // block_size
        page_offset = slot % block_size
        
        # Check key
        src_key_data = src_key_pages[page_idx][page_offset]
        dst_key_data = dst_key_pages[page_idx][page_offset]
        assert torch.allclose(src_key_data, dst_key_data, atol=1e-5, rtol=1e-5), \
            f"Key mismatch at token {i}, slot {slot}: src={src_key_data}, dst={dst_key_data}"
        
        # Check value
        src_value_data = src_value_pages[page_idx][page_offset]
        dst_value_data = dst_value_pages[page_idx][page_offset]
        assert torch.allclose(src_value_data, dst_value_data, atol=1e-5, rtol=1e-5), \
            f"Value mismatch at token {i}, slot {slot}: src={src_value_data}, dst={dst_value_data}"
    
    # Cleanup
    free_flexi_kv_ptrs(src_key_ptrs, src_value_ptrs)
    free_flexi_kv_ptrs(dst_key_ptrs, dst_value_ptrs)


@pytest.mark.parametrize("num_tokens", [1, 83, 100])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [1000])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_flexi_gather_pages(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    torch.set_default_device(device)
    
    # Create random pages
    # Each page: [block_size, num_heads, head_size]
    key_pages = [torch.randn(block_size, num_heads, head_size, dtype=dtype, device=device) for _ in range(num_blocks)]
    value_pages = [torch.randn(block_size, num_heads, head_size, dtype=dtype, device=device) for _ in range(num_blocks)]
    
    # Prepare pointers
    key_page_ptrs, value_page_ptrs = prepare_flexi_kv_ptrs(key_pages, value_pages)
    
    # Create slot mapping
    # Map each token to a random slot in the available blocks
    total_token = num_blocks * block_size
    slot_mapping = torch.randint(0, total_token, (num_tokens,), dtype=torch.int64, device=device)
    
    # Output tensors
    key_out = torch.empty(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    value_out = torch.empty(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    
    # Run kernel
    ops.flexi_gather_pages(key_page_ptrs, value_page_ptrs, slot_mapping, key_out, value_out, block_size)
    
    # Verification
    expected_key_out = torch.empty_like(key_out)
    expected_value_out = torch.empty_like(value_out)
    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        page_idx = slot // block_size
        page_offset = slot % block_size
        
        expected_key_out[i] = key_pages[page_idx][page_offset]
        expected_value_out[i] = value_pages[page_idx][page_offset]
        
    assert torch.allclose(key_out, expected_key_out, atol=1e-3, rtol=1e-3)
    assert torch.allclose(value_out, expected_value_out, atol=1e-3, rtol=1e-3)
    
    # Cleanup
    free_flexi_kv_ptrs(key_page_ptrs, value_page_ptrs)


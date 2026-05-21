# SPDX-License-Identifier: Apache-2.0

"""
KV Cache Allocator - Optimized allocation strategies for Flexi Attention

This module provides optimized KV cache allocation with multiple strategies:
- torch_empty: Standard PyTorch allocation (uses caching allocator)
- cuda_malloc_async: Stream-ordered allocation (CUDA 11.2+)

The cuda_malloc_async strategy is particularly useful for:
- Reducing allocation overhead during inference
- Better memory management in multi-stream scenarios
- Integration with FlexiCache for per-layer cache management
"""

import threading
import time
from typing import List, Tuple, Optional, Union
import torch

kv_allocator_available = False
_cpp_module = None

from vllm.dynamic_utils import ForegroundBackgroundGate
from vllm.logger import init_logger
logger = init_logger(__name__)

try:
    import vllm.kv_cache_allocator as _cpp_module
    kv_allocator_available = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        f"Failed to import kv_cache_allocator C++ extension: {e}. "
        f"cuda_malloc_async allocation strategy will not be available."
    )
class KVAllocator():
    def __init__(self):
        self.fb_gate: Optional[ForegroundBackgroundGate] = None

    def set_lock(self, lock: ForegroundBackgroundGate) -> None:
        self.fb_gate = lock

    def allocate_python_style(
        self,
        size: int,
        block_shape: List[int],
        dtype: torch.dtype,
        device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
        """
        Baseline C++ implementation mimicking Python approach.

        This is useful for performance comparison with pure Python implementations.

        Args:
            size: Total size in bytes to allocate
            block_shape: Shape of each block [num_blocks, block_size, ...]
            dtype: PyTorch data type for the tensors
            device: Target CUDA device

        Returns:
            Tuple of (key_tensors, value_tensors, allocation_time_ms)

        Raises:
            RuntimeError: If kv_cache_allocator extension is not available
            TypeError: If arguments have incorrect types
            ValueError: If size or block_shape are invalid
        """
        if not kv_allocator_available:
            raise RuntimeError("kv_cache_allocator extension not available")

        return _cpp_module.allocate_python_style(size, block_shape, dtype, device)


    def allocate_with_cuda_async(
        self,
        size: int,
        block_shape: List[int],
        dtype: torch.dtype,
        device: torch.device,
        stream_ptr: int = 0
    ) -> Tuple[List[int], List[int], int, int, float]:
        """
        cudaMallocAsync (CUDA 11.2+, stream-ordered) - minimal PyTorch overhead.

        Stream-ordered memory allocation that provides better performance in
        multi-stream scenarios. Returns raw pointers instead of tensors.

        Args:
            size: Total size in bytes to allocate
            block_shape: Shape of each block [num_blocks, block_size, ...]
            dtype: PyTorch data type for the tensors
            device: Target CUDA device
            stream_ptr: CUDA stream pointer (cudaStream_t as int), 0 for default stream

        Returns:
            Tuple of (key_ptrs, value_ptrs, total_bytes, num_blocks, allocation_time_ms)
            where key_ptrs and value_ptrs are lists of raw memory addresses (int)

        Raises:
            RuntimeError: If kv_cache_allocator extension is not available or CUDA version < 11.2
            TypeError: If arguments have incorrect types
            ValueError: If size or block_shape are invalid
        """
        if not kv_allocator_available:
            raise RuntimeError("kv_cache_allocator extension not available")
        start_time = time.perf_counter()
        if stream_ptr == 0:
            current_stream = torch.cuda.current_stream(device)
            stream_ptr = current_stream.cuda_stream
        if self.fb_gate is not None:
            with self.fb_gate.background():
                alloc_time_ms = (time.perf_counter() - start_time) * 1000
                logger.info(f"async kv allocation with lock: alloc_time={alloc_time_ms:.2f}ms")
                results =  _cpp_module.allocate_with_cuda_async(size, block_shape, dtype, device, stream_ptr)
        else:
            results = _cpp_module.allocate_with_cuda_async(size, block_shape, dtype, device, stream_ptr)
        torch.cuda.synchronize(device)
        alloc_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"async kv allocation: alloc_time={alloc_time_ms:.2f}ms")
        return results

    def free_cache(
        self,
        ptrs: List[int],
        device: torch.device,
        stream_ptr: int = 0
    ) -> None:
        """
        Free memory allocated by cudaMallocAsync.

        Frees memory previously allocated with allocate_with_cuda_async.
        Must be called with the same stream that was used for allocation.

        Args:
            ptrs: List of raw memory addresses (int) to free
            device_id: CUDA device ID (e.g., 0 for cuda:0)
            stream_ptr: CUDA stream pointer (cudaStream_t as int), 0 for default stream

        Raises:
            RuntimeError: If kv_cache_allocator extension is not available
            TypeError: If arguments have incorrect types
            ValueError: If device_id or stream_ptr are invalid
        """
        if not kv_allocator_available:
            raise RuntimeError("kv_cache_allocator extension not available")

        device_id = device.index 
        if stream_ptr == 0:
            current_stream = torch.cuda.current_stream(device_id)
            stream_ptr = current_stream.cuda_stream
        _cpp_module.free_cache(ptrs, device_id, stream_ptr)


    def allocate_with_cuda_vmm(
        self,
        size: int,
        block_shape: List[int],
        dtype: torch.dtype,
        device: torch.device
    ) -> Tuple[List[int], List[int], int, int, int, List[int], List[int], float]:
        """
        VMM API allocation (supports fine-grained 2MB release).

        Uses CUDA Virtual Memory Management API for allocation, which allows
        non-contiguous memory release at 2MB granularity. This is useful for
        dynamic KV cache management where blocks need to be freed independently.

        Args:
            size: Number of blocks to allocate
            block_shape: Shape of each block [num_blocks, block_size, ...]
            dtype: PyTorch data type for the tensors
            device: Target CUDA device

        Returns:
            Tuple of (k_ptrs, v_ptrs, k_ptrs_dev, v_ptrs_dev, aligned_bytes, 
                      k_handles, v_handles, allocation_time_ms)
            - k_ptrs, v_ptrs: Lists of virtual addresses for K/V blocks
            - k_ptrs_dev, v_ptrs_dev: Device pointers to pointer arrays
            - aligned_bytes: Size of each block (2MB aligned)
            - k_handles, v_handles: Physical memory handles for each block
            - allocation_time_ms: Allocation time in milliseconds

        Raises:
            RuntimeError: If kv_cache_allocator extension is not available
        """
        if not kv_allocator_available:
            raise RuntimeError("kv_cache_allocator extension not available")
        
        start_time = time.perf_counter()
        
        if self.fb_gate is not None:
            with self.fb_gate.background():
                results = _cpp_module.allocate_with_cuda_vmm(size, block_shape, dtype, device)
        else:
            results = _cpp_module.allocate_with_cuda_vmm(size, block_shape, dtype, device)
        
        torch.cuda.synchronize(device)
        alloc_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"VMM kv allocation: alloc_time={alloc_time_ms:.2f}ms, aligned_bytes={results[4]}")
        return results

    def free_vmm_blocks(
        self,
        ptrs: List[int],
        handles: List[int],
        aligned_bytes: int,
        device: torch.device
    ) -> None:
        """
        Free VMM allocated blocks (supports non-contiguous release).

        Frees individual KV cache blocks allocated with allocate_with_cuda_vmm.
        Can release blocks independently at 2MB granularity.

        Args:
            ptrs: List of virtual addresses to free
            handles: Corresponding physical memory handles
            aligned_bytes: Size of each block (from allocation)
            device: Target CUDA device
        """
        if not kv_allocator_available:
            raise RuntimeError("kv_cache_allocator extension not available")
        
        device_id = device.index
        _cpp_module.free_vmm_blocks(ptrs, handles, aligned_bytes, device_id)

    def free_vmm_va_range(
        self,
        va_base: int,
        total_size: int,
        device: torch.device
    ) -> None:
        """
        Free VMM virtual address range.

        Should be called after all blocks in the range have been freed.

        Args:
            va_base: Base virtual address of the range
            total_size: Total size of the virtual address range
            device: Target CUDA device
        """
        if not kv_allocator_available:
            raise RuntimeError("kv_cache_allocator extension not available")
        
        device_id = device.index
        _cpp_module.free_vmm_va_range(va_base, total_size, device_id)

    def allocate_with_cuda_vmm_combined(
        self,
        size: int,
        block_shape: List[int],
        dtype: torch.dtype,
        device: torch.device
    ) -> Tuple[List[int], List[int], int, int, int, int, List[int], float]:
        """
        VMM API allocation with K and V combined in same physical page.

        This method saves memory by placing K and V in the same 2MB VMM page.
        K occupies the first half, V occupies the second half.
        This avoids memory waste when K/V blocks are smaller than 2MB.

        Args:
            size: Number of blocks to allocate
            block_shape: Shape of each block [num_blocks, block_size, ...]
            dtype: PyTorch data type for the tensors
            device: Target CUDA device

        Returns:
            Tuple of (k_ptrs, v_ptrs, k_ptrs_dev, v_ptrs_dev, aligned_combined_bytes,
                      bytes_per_tensor, kv_handles, allocation_time_ms)
            - k_ptrs: List of virtual addresses for K blocks
            - v_ptrs: List of virtual addresses for V blocks (offset by bytes_per_tensor)
            - k_ptrs_dev, v_ptrs_dev: Device pointers to pointer arrays
            - aligned_combined_bytes: Size of each combined KV block (2MB aligned)
            - bytes_per_tensor: Size of one K or V tensor (for offset calculation)
            - kv_handles: Physical memory handles for combined KV blocks
            - allocation_time_ms: Allocation time in milliseconds

        Raises:
            RuntimeError: If kv_cache_allocator extension is not available
        """
        if not kv_allocator_available:
            raise RuntimeError("kv_cache_allocator extension not available")
        
        start_time = time.perf_counter()
        
        if self.fb_gate is not None:
            with self.fb_gate.background():
                results = _cpp_module.allocate_with_cuda_vmm_combined(size, block_shape, dtype, device)
        else:
            results = _cpp_module.allocate_with_cuda_vmm_combined(size, block_shape, dtype, device)
        
        torch.cuda.synchronize(device)
        alloc_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"VMM combined kv allocation: alloc_time={alloc_time_ms:.2f}ms, "
                    f"aligned_combined_bytes={results[4]}, bytes_per_tensor={results[5]}")
        return results

    def free_vmm_blocks_combined(
        self,
        k_ptrs: List[int],
        handles: List[int],
        aligned_combined_bytes: int,
        device: torch.device
    ) -> None:
        """
        Free VMM combined blocks (K and V share same physical page).

        Frees combined KV cache blocks allocated with allocate_with_cuda_vmm_combined.
        Each handle represents a combined KV block.

        Args:
            k_ptrs: List of K virtual addresses (start of combined block)
            handles: Corresponding physical memory handles
            aligned_combined_bytes: Size of each combined block (from allocation)
            device: Target CUDA device
        """
        if not kv_allocator_available:
            raise RuntimeError("kv_cache_allocator extension not available")
        
        device_id = device.index
        _cpp_module.free_vmm_blocks_combined(k_ptrs, handles, aligned_combined_bytes, device_id)

    def allocate_with_cuda_vmm_combined_layers(
        self,
        num_blocks: int,
        block_shape: List[int],
        dtype: torch.dtype,
        device: torch.device,
        granularity: int,
        gate_mode: str = "background",
    ) -> Tuple[List[List[int]], List[List[int]], List[int], List[int], int, int, List[int], float]:
        """
        VMM API allocation with multiple layers combined in same physical pages.

        This method saves memory by placing multiple layers' K and V caches in the
        same VMM pages. Each 2MB VMM block contains `granularity` layers' KV cache.

        Memory layout for granularity=4:
            Each block: [K0][V0][K1][V1][K2][V2][K3][V3]

        Args:
            num_blocks: Number of blocks to allocate per layer
            block_shape: Shape of each block [block_size, num_heads, head_dim]
            dtype: PyTorch data type for the tensors
            device: Target CUDA device
            granularity: Number of layers to pack per VMM block

        Returns:
            Tuple of (k_ptrs_per_layer, v_ptrs_per_layer, k_ptrs_dev_per_layer, 
                      v_ptrs_dev_per_layer, aligned_combined_bytes, bytes_per_kv,
                      kv_handles, allocation_time_ms)
            - k_ptrs_per_layer: [granularity][num_blocks] K virtual addresses
            - v_ptrs_per_layer: [granularity][num_blocks] V virtual addresses
            - k_ptrs_dev_per_layer: [granularity] GPU pointer arrays for K
            - v_ptrs_dev_per_layer: [granularity] GPU pointer arrays for V
            - aligned_combined_bytes: Size of each combined block (2MB aligned)
            - bytes_per_kv: Size of one K or V tensor per layer
            - kv_handles: [num_blocks] Physical memory handles (shared across layers)
            - allocation_time_ms: Allocation time in milliseconds

        Raises:
            RuntimeError: If kv_cache_allocator extension is not available
        """
        if not kv_allocator_available:
            raise RuntimeError("kv_cache_allocator extension not available")
        
        if granularity < 1:
            raise ValueError("granularity must be >= 1")
        
        start_time = time.perf_counter()
        
        if self.fb_gate is not None and gate_mode == "background":
            with self.fb_gate.background():
                results = _cpp_module.allocate_with_cuda_vmm_combined_layers(
                    num_blocks, block_shape, dtype, device, granularity)
        elif self.fb_gate is not None and gate_mode == "foreground":
            with self.fb_gate.foreground():
                results = _cpp_module.allocate_with_cuda_vmm_combined_layers(
                    num_blocks, block_shape, dtype, device, granularity)
        else:
            results = _cpp_module.allocate_with_cuda_vmm_combined_layers(
                num_blocks, block_shape, dtype, device, granularity)
        
        torch.cuda.synchronize(device)
        alloc_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"VMM combined layers allocation: alloc_time={alloc_time_ms:.2f}ms, "
                    f"granularity={granularity}, gate_mode={gate_mode}, "
                    f"aligned_combined_bytes={results[4]}, "
                    f"bytes_per_kv={results[5]}")
        return results

    def free_vmm_blocks_combined_layers(
        self,
        base_k_ptrs: List[int],
        handles: List[int],
        aligned_combined_bytes: int,
        device: torch.device,
        granularity: int,
        bytes_per_kv: int
    ) -> None:
        """
        Free VMM combined layers blocks.

        Frees combined KV cache blocks allocated with allocate_with_cuda_vmm_combined_layers.
        The handles are shared across layers in a group, so this frees the entire group.

        Args:
            base_k_ptrs: List of K virtual addresses from the first layer (base of combined block)
            handles: Corresponding physical memory handles (one per block, shared across layers)
            aligned_combined_bytes: Size of each combined block (from allocation)
            device: Target CUDA device
            granularity: Number of layers per VMM block
            bytes_per_kv: Size of one K or V tensor per layer

        Note:
            This must be called with handles from a complete layer group.
            Partial group freeing is not supported.
        """
        if not kv_allocator_available:
            raise RuntimeError("kv_cache_allocator extension not available")
        
        device_id = device.index
        _cpp_module.free_vmm_blocks_combined_layers(
            base_k_ptrs, handles, aligned_combined_bytes, device_id, granularity, bytes_per_kv)

    def prepare_flexi_kv_ptrs(
        self,
        k_list: List[int],
        v_list: List[int]
    ) -> Tuple[int, int]:
        """
        Prepare and cache KV pointers on GPU for flexi attention.

        Allocates device memory for pointer arrays and copies the K/V cache block
        pointers to GPU. These cached pointers can be used directly in flexi attention
        kernels without repeated host-to-device transfers.

        Args:
            k_list: List of key cache block pointers (raw memory addresses as int)
            v_list: List of value cache block pointers (raw memory addresses as int)

        Returns:
            Tuple of (k_ptrs_dev, v_ptrs_dev) where:
            - k_ptrs_dev: Device pointer to the array of key cache pointers (int)
            - v_ptrs_dev: Device pointer to the array of value cache pointers (int)

        Raises:
            RuntimeError: If kv_cache_allocator extension is not available
            TypeError: If arguments have incorrect types
            ValueError: If k_list or v_list are invalid

        Note:
            The returned device pointers must be freed explicitly using cudaFree
            or similar mechanisms to avoid memory leaks.
        """
        if not kv_allocator_available:
            raise RuntimeError("kv_cache_allocator extension not available")

        if not isinstance(k_list, list) or not all(isinstance(x, int) for x in k_list):
            raise TypeError(f"k_list must be a list of integers, got {type(k_list)}")
        if not isinstance(v_list, list) or not all(isinstance(x, int) for x in v_list):
            raise TypeError(f"v_list must be a list of integers, got {type(v_list)}")

        return _cpp_module.prepare_flexi_kv_ptrs(k_list, v_list)

    def free_page_list(
        self,
        ptrs: int,
        device: torch.device,
        stream_ptr: int = 0
    ) -> None:
        """
        Free page list memory allocated by cudaMallocAsync.

        Frees memory previously allocated with prepare_flexi_kv_ptrs.
        Must be called with the same stream that was used for allocation.

        Args:
            ptrs: Device pointer to the array of pointers (int)
            device_id: CUDA device ID (e.g., 0 for cuda:0)
            stream_ptr: CUDA stream pointer (cudaStream_t as int), 0 for default stream

        Raises:
            RuntimeError: If kv_cache_allocator extension is not available
            TypeError: If arguments have incorrect types
            ValueError: If device_id or stream_ptr are invalid
        """
        if not kv_allocator_available:
            raise RuntimeError("kv_cache_allocator extension not available")

        device_id = device.index 
        if stream_ptr == 0:
            current_stream = torch.cuda.current_stream(device_id)
            stream_ptr = current_stream.cuda_stream
        _cpp_module.free_page_list(ptrs, device_id, stream_ptr)

kv_allocator = KVAllocator()

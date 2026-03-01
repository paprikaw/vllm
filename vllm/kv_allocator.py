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

    def trim_memory_pool(
        self,
        device: torch.device,
        min_bytes_to_keep: int = 0
    ) -> None:
        """
        Trim CUDA memory pool to release retained memory back to OS.

        cudaMallocAsync uses a memory pool that retains freed memory for
        future allocations. This function forces the pool to release
        retained memory back to the operating system.

        Args:
            device: CUDA device
            min_bytes_to_keep: Minimum bytes the pool should keep (default 0 = release all)

        Raises:
            RuntimeError: If kv_cache_allocator extension is not available
        """
        if not kv_allocator_available:
            logger.warning("kv_cache_allocator extension not available, cannot trim memory pool")
            return

        device_id = device.index
        _cpp_module.trim_memory_pool(device_id, min_bytes_to_keep)

kv_allocator = KVAllocator()
"""
FlexiCache - Single Layer KV Cache Management for Flexi Attention

This module provides a cache manager for a single layer's K or V cache in Flexi Attention mode.
Flexi Attention uses list-of-tensors representation where each block is an independent tensor.

Usage Pattern:
    # Create separate lists for all layers
    key_caches = [FlexiCache(...) for _ in range(num_layers)]
    value_caches = [FlexiCache(...) for _ in range(num_layers)]
    
    # Operate on individual layer cache
    key_caches[0].allocate_blocks(num_blocks=100)
    key_caches[0].resize(new_num_blocks=150)
    key_caches[0].free_all()

Key Features:
- Allocation strategies: torch_empty (default) or cuda_malloc_async (stream-ordered)
- Block operations: allocate, free, resize, swap in/out
- GPU pointer management for kernel usage
- Integration with kv_cache_allocator_optimized C++ extension
"""

import torch
from typing import List, Optional, Literal, Union
from vllm.logger import init_logger

logger = init_logger(__name__)

AllocationStrategy = Literal["torch_empty", "cuda_malloc_async"]


class FlexiCache:
    """
    Manages KV cache allocation, deallocation, and operations for Flexi Attention.
    
    Responsibilities:
    1. Allocate KV cache as list of tensors (per-layer, per-block)
    2. Free allocated memory (with proper cleanup for cuda_malloc_async)
    3. Resize KV cache (grow or shrink block count)
    4. Swap operations (move blocks between GPU/CPU)
    5. Prepare GPU pointer arrays for kernel usage
    
    Usage:
        allocator = FlexiKVAllocator(
            num_layers=64,
            num_blocks=1000,
            block_shape=(16, 8, 128),  # (num_heads, block_size, head_dim)
            dtype=torch.float16,
            device=torch.device("cuda:0"),
            allocation_strategy="cuda_malloc_async",
            kv_allocator_cpp=kv_cache_allocator_optimized
        )
        
        # Allocate
        key_caches, value_caches = allocator.allocate_kv_cache()
        
        # Prepare for kernel usage
        k_ptrs_dev, v_ptrs_dev = allocator.prepare_gpu_pointers(key_caches, value_caches)
        
        # Resize
        allocator.resize_kv_cache(new_num_blocks=1500)
        
        # Free
        allocator.free_kv_cache(key_caches, value_caches)
    """
    
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_shape: Tuple[int, int, int],
        dtype: torch.dtype,
        device: torch.device,
        allocation_strategy: AllocationStrategy = "torch_empty",
        kv_allocator_cpp = None,
    ):
        """
        Initialize FlexiKVAllocator.
        
        Args:
            num_layers: Number of attention layers
            num_blocks: Initial number of KV cache blocks per layer
            block_shape: Shape of each block (num_heads, block_size, head_dim)
            dtype: Data type for KV cache (typically torch.float16)
            device: CUDA device for allocation
            allocation_strategy: "torch_empty" or "cuda_malloc_async"
            kv_allocator_cpp: C++ extension module (required for cuda_malloc_async)
        """
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_shape = block_shape
        self.dtype = dtype
        self.device = device
        self.allocation_strategy = allocation_strategy
        self.kv_allocator_cpp = kv_allocator_cpp
        
        # Validate cuda_malloc_async requirements
        if self.allocation_strategy == "cuda_malloc_async":
            if self.kv_allocator_cpp is None:
                logger.warning(
                    "cuda_malloc_async strategy requires kv_allocator_cpp extension, "
                    "falling back to torch_empty"
                )
                self.allocation_strategy = "torch_empty"
            else:
                logger.info("Using cuda_malloc_async allocation strategy")
        
        # Current CUDA stream (cached for cuda_malloc_async)
        self._current_stream = None
        self._stream_ptr = None
        
        # GPU pointer arrays (cached for kernel usage)
        # Format: {layer_idx: (k_ptrs_dev, v_ptrs_dev)}
        self._gpu_pointers_cache: dict[int, Tuple[int, int]] = {}
        
        # Track allocated memory for statistics
        self._total_allocated_blocks = 0
        self._total_freed_blocks = 0
        
        logger.info(
            f"FlexiKVAllocator initialized: "
            f"layers={num_layers}, blocks={num_blocks}, "
            f"block_shape={block_shape}, dtype={dtype}, "
            f"device={device}, strategy={self.allocation_strategy}"
        )
    
    def _get_stream_ptr(self) -> int:
        """Get current CUDA stream pointer (cached)"""
        if self._stream_ptr is None:
            self._current_stream = torch.cuda.current_stream(self.device)
            self._stream_ptr = self._current_stream.cuda_stream
        return self._stream_ptr
    
    def _allocate_single_layer_torch(
        self
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Allocate KV cache for a single layer using torch.empty.
        
        Returns:
            (key_cache_list, value_cache_list) where each is a list of tensors
        """
        key_cache = []
        value_cache = []
        
        with torch.no_grad():
            for _ in range(self.num_blocks):
                k_block = torch.empty(
                    self.block_shape,
                    dtype=self.dtype,
                    device=self.device
                )
                v_block = torch.empty(
                    self.block_shape,
                    dtype=self.dtype,
                    device=self.device
                )
                key_cache.append(k_block)
                value_cache.append(v_block)
        
        return key_cache, value_cache
    
    def _allocate_single_layer_cuda_async(
        self
    ) -> Tuple[List[int], List[int], int, int]:
        """
        Allocate KV cache for a single layer using cudaMallocAsync.
        
        Returns:
            (k_ptrs, v_ptrs, k_ptrs_dev, v_ptrs_dev)
            - k_ptrs/v_ptrs: List of host-side pointers (int64)
            - k_ptrs_dev/v_ptrs_dev: Device pointer arrays (int64)
        """
        stream_ptr = self._get_stream_ptr()
        
        k_ptrs, v_ptrs, k_ptrs_dev, v_ptrs_dev, alloc_time = \
            self.kv_allocator_cpp.allocate_with_cuda_async(
                size=self.num_blocks,
                block_shape=list(self.block_shape),
                dtype=self.dtype,
                device=self.device,
                stream_ptr=stream_ptr
            )
        
        logger.debug(
            f"cudaMallocAsync allocation: {self.num_blocks} blocks, "
            f"time={alloc_time:.2f}ms"
        )
        
        return k_ptrs, v_ptrs, k_ptrs_dev, v_ptrs_dev
    
    def allocate_kv_cache(
        self
    ) -> Tuple[List[List], List[List]]:
        """
        Allocate KV cache for all layers.
        
        Returns:
            (key_caches, value_caches) where:
            - key_caches[layer_idx] = list of key blocks (or list of pointers)
            - value_caches[layer_idx] = list of value blocks (or list of pointers)
        """
        key_caches = []
        value_caches = []
        
        if self.allocation_strategy == "torch_empty":
            for layer_idx in range(self.num_layers):
                k_cache, v_cache = self._allocate_single_layer_torch()
                key_caches.append(k_cache)
                value_caches.append(v_cache)
                self._total_allocated_blocks += self.num_blocks * 2  # K + V
        
        elif self.allocation_strategy == "cuda_malloc_async":
            for layer_idx in range(self.num_layers):
                k_ptrs, v_ptrs, k_ptrs_dev, v_ptrs_dev = \
                    self._allocate_single_layer_cuda_async()
                key_caches.append(k_ptrs)
                value_caches.append(v_ptrs)
                # Cache GPU pointer arrays
                self._gpu_pointers_cache[layer_idx] = (k_ptrs_dev, v_ptrs_dev)
                self._total_allocated_blocks += self.num_blocks * 2
        
        logger.info(
            f"Allocated KV cache: {self.num_layers} layers × {self.num_blocks} blocks"
        )
        return key_caches, value_caches
    
    def free_kv_cache(
        self,
        key_caches: List[List],
        value_caches: List[List]
    ) -> None:
        """
        Free allocated KV cache.
        
        Args:
            key_caches: Key cache to free
            value_caches: Value cache to free
        """
        if self.allocation_strategy == "torch_empty":
            # PyTorch garbage collector will handle this
            freed_blocks = len(key_caches) * len(key_caches[0]) * 2
            key_caches.clear()
            value_caches.clear()
            self._total_freed_blocks += freed_blocks
            logger.info(f"Freed KV cache: {freed_blocks} blocks (torch_empty)")
        
        elif self.allocation_strategy == "cuda_malloc_async":
            stream_ptr = self._get_stream_ptr()
            device_id = self.device.index if self.device.type == 'cuda' else 0
            
            # Free each layer's allocations
            for layer_idx in range(len(key_caches)):
                k_ptrs = key_caches[layer_idx]
                v_ptrs = value_caches[layer_idx]
                
                # Free device memory
                self.kv_allocator_cpp.free_cuda_async(k_ptrs, device_id, stream_ptr)
                self.kv_allocator_cpp.free_cuda_async(v_ptrs, device_id, stream_ptr)
                
                # Free GPU pointer arrays
                if layer_idx in self._gpu_pointers_cache:
                    k_ptrs_dev, v_ptrs_dev = self._gpu_pointers_cache[layer_idx]
                    # TODO: Add free_gpu_pointers in C++ extension
                    del self._gpu_pointers_cache[layer_idx]
                
                self._total_freed_blocks += len(k_ptrs) + len(v_ptrs)
            
            key_caches.clear()
            value_caches.clear()
            logger.info(f"Freed KV cache using cudaFreeAsync")
    
    def resize_kv_cache(
        self,
        key_caches: List[List],
        value_caches: List[List],
        new_num_blocks: int
    ) -> Tuple[List[List], List[List]]:
        """
        Resize KV cache to new block count.
        
        Strategy:
        - If growing: allocate new blocks and append
        - If shrinking: free excess blocks
        
        Args:
            key_caches: Current key cache
            value_caches: Current value cache
            new_num_blocks: Target number of blocks
            
        Returns:
            (resized_key_caches, resized_value_caches)
        """
        current_blocks = len(key_caches[0]) if key_caches else 0
        delta = new_num_blocks - current_blocks
        
        if delta == 0:
            logger.info("KV cache size unchanged")
            return key_caches, value_caches
        
        if delta > 0:
            # Growing: allocate additional blocks
            logger.info(f"Growing KV cache: {current_blocks} -> {new_num_blocks} (+{delta})")
            return self._grow_kv_cache(key_caches, value_caches, delta)
        else:
            # Shrinking: free excess blocks
            logger.info(f"Shrinking KV cache: {current_blocks} -> {new_num_blocks} ({delta})")
            return self._shrink_kv_cache(key_caches, value_caches, new_num_blocks)
    
    def _grow_kv_cache(
        self,
        key_caches: List[List],
        value_caches: List[List],
        num_new_blocks: int
    ) -> Tuple[List[List], List[List]]:
        """Grow KV cache by allocating additional blocks"""
        old_num_blocks = self.num_blocks
        self.num_blocks = num_new_blocks  # Temporarily set for allocation
        
        if self.allocation_strategy == "torch_empty":
            for layer_idx in range(len(key_caches)):
                for _ in range(num_new_blocks):
                    k_block = torch.empty(
                        self.block_shape,
                        dtype=self.dtype,
                        device=self.device
                    )
                    v_block = torch.empty(
                        self.block_shape,
                        dtype=self.dtype,
                        device=self.device
                    )
                    key_caches[layer_idx].append(k_block)
                    value_caches[layer_idx].append(v_block)
        
        elif self.allocation_strategy == "cuda_malloc_async":
            # Allocate new blocks
            for layer_idx in range(len(key_caches)):
                k_ptrs, v_ptrs, k_ptrs_dev, v_ptrs_dev = \
                    self._allocate_single_layer_cuda_async()
                
                # Append new pointers to existing lists
                key_caches[layer_idx].extend(k_ptrs)
                value_caches[layer_idx].extend(v_ptrs)
                
                # Update GPU pointer arrays (need to reallocate with new size)
                # TODO: Implement efficient pointer array reallocation
        
        self.num_blocks = old_num_blocks + num_new_blocks
        return key_caches, value_caches
    
    def _shrink_kv_cache(
        self,
        key_caches: List[List],
        value_caches: List[List],
        new_num_blocks: int
    ) -> Tuple[List[List], List[List]]:
        """Shrink KV cache by freeing excess blocks"""
        current_blocks = len(key_caches[0])
        blocks_to_remove = current_blocks - new_num_blocks
        
        if self.allocation_strategy == "torch_empty":
            # Simply truncate lists (GC will free)
            for layer_idx in range(len(key_caches)):
                key_caches[layer_idx] = key_caches[layer_idx][:new_num_blocks]
                value_caches[layer_idx] = value_caches[layer_idx][:new_num_blocks]
        
        elif self.allocation_strategy == "cuda_malloc_async":
            stream_ptr = self._get_stream_ptr()
            device_id = self.device.index if self.device.type == 'cuda' else 0
            
            # Free excess blocks
            for layer_idx in range(len(key_caches)):
                excess_k_ptrs = key_caches[layer_idx][new_num_blocks:]
                excess_v_ptrs = value_caches[layer_idx][new_num_blocks:]
                
                self.kv_allocator_cpp.free_cuda_async(excess_k_ptrs, device_id, stream_ptr)
                self.kv_allocator_cpp.free_cuda_async(excess_v_ptrs, device_id, stream_ptr)
                
                # Truncate lists
                key_caches[layer_idx] = key_caches[layer_idx][:new_num_blocks]
                value_caches[layer_idx] = value_caches[layer_idx][:new_num_blocks]
        
        self.num_blocks = new_num_blocks
        return key_caches, value_caches
    
    def prepare_gpu_pointers(
        self,
        key_caches: List[List],
        value_caches: List[List],
        layer_idx: Optional[int] = None
    ) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Prepare GPU pointer arrays for kernel usage.
        
        For cuda_malloc_async: return cached device pointer arrays
        For torch_empty: create pointer arrays on-the-fly
        
        Args:
            key_caches: Key cache
            value_caches: Value cache
            layer_idx: If specified, return pointers for single layer
            
        Returns:
            If layer_idx is None: list of (k_ptrs_dev, v_ptrs_dev) for all layers
            If layer_idx specified: (k_ptrs_dev, v_ptrs_dev) for that layer
        """
        if self.allocation_strategy == "cuda_malloc_async":
            # Use cached GPU pointer arrays
            if layer_idx is not None:
                if layer_idx in self._gpu_pointers_cache:
                    return self._gpu_pointers_cache[layer_idx]
                else:
                    raise ValueError(f"Layer {layer_idx} not in GPU pointer cache")
            else:
                return [self._gpu_pointers_cache[i] for i in range(len(key_caches))]
        
        elif self.allocation_strategy == "torch_empty":
            # Use prepare_flexi_kv_ptrs from flash_attn_interface
            from vllm.vllm_flash_attn.flash_attn_interface import prepare_flexi_kv_ptrs
            
            if layer_idx is not None:
                k_ptrs_dev, v_ptrs_dev = prepare_flexi_kv_ptrs(
                    key_caches[layer_idx],
                    value_caches[layer_idx]
                )
                return (k_ptrs_dev, v_ptrs_dev)
            else:
                result = []
                for idx in range(len(key_caches)):
                    k_ptrs_dev, v_ptrs_dev = prepare_flexi_kv_ptrs(
                        key_caches[idx],
                        value_caches[idx]
                    )
                    result.append((k_ptrs_dev, v_ptrs_dev))
                return result
    
    def swap_out_blocks(
        self,
        key_caches: List[List],
        value_caches: List[List],
        layer_idx: int,
        block_indices: List[int]
    ) -> Tuple[List, List]:
        """
        Swap blocks from GPU to CPU.
        
        Args:
            key_caches: Key cache
            value_caches: Value cache
            layer_idx: Layer index
            block_indices: Indices of blocks to swap out
            
        Returns:
            (cpu_key_blocks, cpu_value_blocks)
        """
        cpu_key_blocks = []
        cpu_value_blocks = []
        
        if self.allocation_strategy == "torch_empty":
            for block_idx in block_indices:
                k_block = key_caches[layer_idx][block_idx]
                v_block = value_caches[layer_idx][block_idx]
                
                # Copy to CPU
                cpu_key_blocks.append(k_block.cpu())
                cpu_value_blocks.append(v_block.cpu())
        
        elif self.allocation_strategy == "cuda_malloc_async":
            # TODO: Implement for cuda_malloc_async
            # Need to create temporary tensors from pointers, then copy
            raise NotImplementedError("swap_out_blocks not yet implemented for cuda_malloc_async")
        
        return cpu_key_blocks, cpu_value_blocks
    
    def swap_in_blocks(
        self,
        key_caches: List[List],
        value_caches: List[List],
        layer_idx: int,
        cpu_key_blocks: List[torch.Tensor],
        cpu_value_blocks: List[torch.Tensor],
        dst_block_indices: List[int]
    ) -> None:
        """
        Swap blocks from CPU to GPU.
        
        Args:
            key_caches: Key cache
            value_caches: Value cache
            layer_idx: Layer index
            cpu_key_blocks: CPU key blocks to swap in
            cpu_value_blocks: CPU value blocks to swap in
            dst_block_indices: Destination block indices on GPU
        """
        if self.allocation_strategy == "torch_empty":
            for i, block_idx in enumerate(dst_block_indices):
                key_caches[layer_idx][block_idx].copy_(cpu_key_blocks[i])
                value_caches[layer_idx][block_idx].copy_(cpu_value_blocks[i])
        
        elif self.allocation_strategy == "cuda_malloc_async":
            # TODO: Implement for cuda_malloc_async
            raise NotImplementedError("swap_in_blocks not yet implemented for cuda_malloc_async")
    
    def get_stats(self) -> dict:
        """Get allocation statistics"""
        return {
            "num_layers": self.num_layers,
            "num_blocks": self.num_blocks,
            "block_shape": self.block_shape,
            "allocation_strategy": self.allocation_strategy,
            "total_allocated_blocks": self._total_allocated_blocks,
            "total_freed_blocks": self._total_freed_blocks,
            "active_blocks": self._total_allocated_blocks - self._total_freed_blocks,
            "cached_gpu_pointers": len(self._gpu_pointers_cache),
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"FlexiKVAllocator("
            f"layers={stats['num_layers']}, "
            f"blocks={stats['num_blocks']}, "
            f"strategy={stats['allocation_strategy']}, "
            f"active={stats['active_blocks']})"
        )

"""
FlexiCache - Single Layer Cache Management for Flexi Attention

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
- Integration with kv_cache_allocator C++ extension
"""

import torch
from typing import List, Optional, Literal, Union
from vllm.logger import init_logger
from vllm.kv_allocator import kv_allocator_available

logger = init_logger(__name__)

AllocationStrategy = Literal["torch_empty", "cuda_malloc_async"]


class FlexiCache:
    """
    Manages a single layer's K or V cache as a list of blocks.
    
    Each FlexiCache instance represents one cache (either K or V) for one layer.
    The cache is a list of blocks, where each block is either:
    - torch.Tensor (for torch_empty strategy)
    - int64 pointer (for cuda_malloc_async strategy)
    
    Responsibilities:
    1. Allocate blocks (grow the list)
    2. Free blocks (remove from list and cleanup)
    3. Resize cache (allocate more or free excess)
    4. Prepare GPU pointer array for kernel usage
    5. Swap blocks between GPU/CPU
    
    Usage:
        cache = FlexiCache(
            layer_idx=0,
            cache_type="key",  # or "value"
            block_shape=(16, 8, 128),
            dtype=torch.float16,
            device=torch.device("cuda:0"),
            allocation_strategy="torch_empty",
            kv_allocator_cpp=None
        )
        
        # Allocate blocks
        cache.allocate_blocks(num_blocks=100)
        
        # Access blocks
        blocks = cache.get_blocks()  # List[torch.Tensor] or List[int]
        
        # Resize
        cache.resize(new_num_blocks=150)
        
        # Prepare for kernel
        ptr_dev = cache.prepare_gpu_pointer_array()
        
        # Free all
        cache.free_all()
    """
    
    def __init__(
        self,
        layer_idx: int,
        cache_type: Literal["key", "value"],
        block_shape: tuple[int, int, int],
        dtype: torch.dtype,
        device: torch.device,
        allocation_strategy: AllocationStrategy = "torch_empty",
        kv_allocator_cpp = None,
    ):
        """
        Initialize FlexiCache for a single layer's K or V cache.
        
        Args:
            layer_idx: Layer index (for logging/debugging)
            cache_type: "key" or "value"
            block_shape: Shape of each block (num_heads, block_size, head_dim)
            dtype: Data type (typically torch.float16)
            device: CUDA device
            allocation_strategy: "torch_empty" or "cuda_malloc_async"
            kv_allocator_cpp: C++ extension (required for cuda_malloc_async)
        """
        self.layer_idx = layer_idx
        self.cache_type = cache_type
        self.block_shape = block_shape
        self.dtype = dtype
        self.device = device
        self.allocation_strategy = allocation_strategy
        self.kv_allocator_cpp = kv_allocator_cpp
        
        # Validate cuda_malloc_async requirements
        if self.allocation_strategy == "cuda_malloc_async":
            if not kv_allocator_available:
                logger.warning(
                    f"Layer {layer_idx} {cache_type}: cuda_malloc_async requires kv_allocator extension, "
                    f"but it's not available. Falling back to torch_empty"
                )
                self.allocation_strategy = "torch_empty"
            elif self.kv_allocator_cpp is None:
                logger.warning(
                    f"Layer {layer_idx} {cache_type}: cuda_malloc_async requires kv_allocator_cpp, "
                    f"falling back to torch_empty"
                )
                self.allocation_strategy = "torch_empty"
        
        # Block storage: List[torch.Tensor] or List[int]
        self._blocks: List[Union[torch.Tensor, int]] = []
        
        # GPU pointer array (int64 device pointer, cached for kernel usage)
        # Only used for cuda_malloc_async (returned from C++ allocate_with_cuda_async)
        self._gpu_ptr_array: Optional[int] = None
        
        # Current CUDA stream (cached)
        self._stream_ptr: Optional[int] = None
        
        # Statistics
        self._total_allocated = 0
        self._total_freed = 0
    
    def _get_stream_ptr(self) -> int:
        """Get current CUDA stream pointer (cached)"""
        if self._stream_ptr is None:
            stream = torch.cuda.current_stream(self.device)
            self._stream_ptr = stream.cuda_stream
        return self._stream_ptr
    
    def allocate_blocks(self, num_blocks: int) -> None:
        """
        Allocate new blocks and append to cache.
        
        Args:
            num_blocks: Number of blocks to allocate
        """
        if num_blocks <= 0:
            return
        
        if self.allocation_strategy == "torch_empty":
            self._allocate_blocks_torch(num_blocks)
        elif self.allocation_strategy == "cuda_malloc_async":
            self._allocate_blocks_cuda_async(num_blocks)
        
        self._total_allocated += num_blocks
        
        logger.debug(
            f"Layer {self.layer_idx} {self.cache_type}: allocated {num_blocks} blocks "
            f"(total: {len(self._blocks)})"
        )
    
    def _allocate_blocks_torch(self, num_blocks: int) -> None:
        """Allocate blocks using torch.empty"""
        with torch.no_grad():
            for _ in range(num_blocks):
                block = torch.empty(
                    self.block_shape,
                    dtype=self.dtype,
                    device=self.device
                )
                self._blocks.append(block)
    
    def _allocate_blocks_cuda_async(self, num_blocks: int) -> None:
        """Allocate blocks using cudaMallocAsync"""
        stream_ptr = self._get_stream_ptr()
        
        # Use C++ extension to allocate
        # Note: allocate_with_cuda_async returns (k_ptrs, v_ptrs, k_ptrs_dev, v_ptrs_dev, time)
        # We use it for single cache (k or v), so we only take first output
        ptrs, _, ptrs_dev, _, alloc_time = self.kv_allocator_cpp.allocate_with_cuda_async(
            size=num_blocks,
            block_shape=list(self.block_shape),
            dtype=self.dtype,
            device=self.device,
            stream_ptr=stream_ptr
        )
        
        # Append pointers to block list
        self._blocks.extend(ptrs)
        
        # Update GPU pointer array (need to reallocate if we already had one)
        # For now, we store the new one and invalidate the old
        # TODO: Optimize by extending the existing array
        if self._gpu_ptr_array is not None:
            logger.warning(
                f"Layer {self.layer_idx} {self.cache_type}: reallocating GPU pointer array "
                f"(not optimal, consider pre-allocating)"
            )
        self._gpu_ptr_array = ptrs_dev
        
        logger.debug(
            f"Layer {self.layer_idx} {self.cache_type}: cudaMallocAsync "
            f"{num_blocks} blocks, time={alloc_time:.2f}ms"
        )
    
    def free_blocks(self, block_indices: List[int]) -> None:
        """
        Free specific blocks.
        
        Args:
            block_indices: Indices of blocks to free (must be sorted descending)
        """
        if not block_indices:
            return
        
        if self.allocation_strategy == "torch_empty":
            # Remove from list (GC will handle cleanup)
            for idx in sorted(block_indices, reverse=True):
                del self._blocks[idx]
        
        elif self.allocation_strategy == "cuda_malloc_async":
            stream_ptr = self._get_stream_ptr()
            device_id = self.device.index if self.device.type == 'cuda' else 0
            
            # Free specific pointers
            ptrs_to_free = [self._blocks[idx] for idx in block_indices]
            self.kv_allocator_cpp.free_cuda_async(ptrs_to_free, device_id, stream_ptr)
            
            # Remove from list
            for idx in sorted(block_indices, reverse=True):
                del self._blocks[idx]
            
            # Invalidate GPU pointer array (needs rebuild)
            self._gpu_ptr_array = None
        
        self._total_freed += len(block_indices)
        logger.debug(
            f"Layer {self.layer_idx} {self.cache_type}: freed {len(block_indices)} blocks "
            f"(remaining: {len(self._blocks)})"
        )
    
    def free_all(self) -> None:
        """Free all blocks in this cache"""
        if not self._blocks:
            return
        
        num_blocks = len(self._blocks)
        
        if self.allocation_strategy == "cuda_malloc_async":
            stream_ptr = self._get_stream_ptr()
            device_id = self.device.index if self.device.type == 'cuda' else 0
            
            # Free all pointers
            self.kv_allocator_cpp.free_cuda_async(self._blocks, device_id, stream_ptr)
            
            # Free GPU pointer array if exists
            # TODO: Add free_gpu_pointer_array in C++ extension
            self._gpu_ptr_array = None
        
        # Clear list
        self._blocks.clear()
        self._total_freed += num_blocks
        
        logger.debug(
            f"Layer {self.layer_idx} {self.cache_type}: freed all {num_blocks} blocks"
        )
    
    def resize(self, new_num_blocks: int) -> None:
        """
        Resize cache to target block count.
        
        Args:
            new_num_blocks: Target number of blocks
        """
        current_blocks = len(self._blocks)
        
        if new_num_blocks == current_blocks:
            return
        
        if new_num_blocks > current_blocks:
            # Grow: allocate more blocks
            delta = new_num_blocks - current_blocks
            logger.info(
                f"Layer {self.layer_idx} {self.cache_type}: growing {current_blocks} -> "
                f"{new_num_blocks} (+{delta} blocks)"
            )
            self.allocate_blocks(delta)
        
        else:
            # Shrink: free excess blocks
            delta = current_blocks - new_num_blocks
            logger.info(
                f"Layer {self.layer_idx} {self.cache_type}: shrinking {current_blocks} -> "
                f"{new_num_blocks} (-{delta} blocks)"
            )
            # Free from the end
            indices_to_free = list(range(new_num_blocks, current_blocks))
            self.free_blocks(indices_to_free)
    
    def prepare_gpu_pointer_array(self) -> int:
        """
        Prepare GPU pointer array for kernel usage.
        
        Returns:
            Device pointer (int64) to the pointer array
            
        For cuda_malloc_async: return cached device pointer
        For torch_empty: create pointer array on-the-fly using prepare_flexi_kv_ptrs
        """
        if self.allocation_strategy == "cuda_malloc_async":
            if self._gpu_ptr_array is None:
                raise RuntimeError(
                    f"Layer {self.layer_idx} {self.cache_type}: GPU pointer array not available. "
                    f"Did you allocate blocks?"
                )
            return self._gpu_ptr_array
        
        elif self.allocation_strategy == "torch_empty":
            # Create pointer array from tensor list
            from vllm.vllm_flash_attn.flash_attn_interface import prepare_flexi_kv_ptrs
            
            # prepare_flexi_kv_ptrs expects (k_list, v_list), but we only have one
            # So we call it with empty list for the other
            if self.cache_type == "key":
                ptr_dev, _ = prepare_flexi_kv_ptrs(self._blocks, [])
            else:
                _, ptr_dev = prepare_flexi_kv_ptrs([], self._blocks)
            
            return ptr_dev
    
    def swap_out_blocks(self, block_indices: List[int]) -> List[torch.Tensor]:
        """
        Swap blocks from GPU to CPU.
        
        Args:
            block_indices: Indices of blocks to swap out
            
        Returns:
            List of CPU tensors
        """
        cpu_blocks = []
        
        if self.allocation_strategy == "torch_empty":
            for idx in block_indices:
                gpu_block = self._blocks[idx]
                cpu_blocks.append(gpu_block.cpu())
        
        elif self.allocation_strategy == "cuda_malloc_async":
            # TODO: Create temporary tensor from pointer, then copy to CPU
            raise NotImplementedError(
                "swap_out_blocks not yet implemented for cuda_malloc_async"
            )
        
        return cpu_blocks
    
    def swap_in_blocks(
        self,
        cpu_blocks: List[torch.Tensor],
        dst_block_indices: List[int]
    ) -> None:
        """
        Swap blocks from CPU to GPU.
        
        Args:
            cpu_blocks: CPU tensors to swap in
            dst_block_indices: Destination block indices on GPU
        """
        if len(cpu_blocks) != len(dst_block_indices):
            raise ValueError("cpu_blocks and dst_block_indices must have same length")
        
        if self.allocation_strategy == "torch_empty":
            for cpu_block, dst_idx in zip(cpu_blocks, dst_block_indices):
                self._blocks[dst_idx].copy_(cpu_block)
        
        elif self.allocation_strategy == "cuda_malloc_async":
            # TODO: Copy from CPU to GPU pointer location
            raise NotImplementedError(
                "swap_in_blocks not yet implemented for cuda_malloc_async"
            )
    
    def get_blocks(self) -> List[Union[torch.Tensor, int]]:
        """
        Get the list of blocks.
        
        Returns:
            List[torch.Tensor] for torch_empty
            List[int] for cuda_malloc_async (pointers)
        """
        return self._blocks
    
    def num_blocks(self) -> int:
        """Get current number of blocks"""
        return len(self._blocks)
    
    def is_empty(self) -> bool:
        """Check if cache is empty"""
        return len(self._blocks) == 0
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "layer_idx": self.layer_idx,
            "cache_type": self.cache_type,
            "num_blocks": len(self._blocks),
            "block_shape": self.block_shape,
            "allocation_strategy": self.allocation_strategy,
            "total_allocated": self._total_allocated,
            "total_freed": self._total_freed,
            "has_gpu_ptr_array": self._gpu_ptr_array is not None,
        }
    
    def __len__(self) -> int:
        """Return number of blocks"""
        return len(self._blocks)
    
    def __repr__(self) -> str:
        return (
            f"FlexiCache(layer={self.layer_idx}, type={self.cache_type}, "
            f"blocks={len(self._blocks)}, strategy={self.allocation_strategy})"
        )

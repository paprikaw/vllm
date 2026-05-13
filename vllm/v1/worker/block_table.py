# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils import cdiv

logger = init_logger(__name__)


class BlockTable:

    def __init__(
        self,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        self.block_table = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            device=self.device,
            dtype=torch.int32,
        )
        self.block_table_cpu = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.block_table_np = self.block_table_cpu.numpy()
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping_cpu = torch.zeros(self.max_num_batched_tokens,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.slot_mapping = torch.zeros(self.max_num_batched_tokens,
                                        dtype=torch.int64,
                                        device=self.device)

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table_np[row_idx, start:start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table_np[tgt, :num_blocks] = self.block_table_np[
            src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        num_blocks_src = self.num_blocks_per_row[src]
        num_blocks_tgt = self.num_blocks_per_row[tgt]
        self.num_blocks_per_row[src] = num_blocks_tgt
        self.num_blocks_per_row[tgt] = num_blocks_src

        self.block_table_np[[src, tgt]] = self.block_table_np[[tgt, src]]

    def commit(self, num_reqs: int) -> None:
        self.block_table[:num_reqs].copy_(self.block_table_cpu[:num_reqs],
                                          non_blocking=True)

    def clear(self) -> None:
        self.block_table.fill_(0)
        self.block_table_cpu.fill_(0)

    def get_device_tensor(self) -> torch.Tensor:
        """Ruturns the device tensor of the block table."""
        return self.block_table

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table_cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table_np


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(self, max_num_reqs: int, max_model_len: int,
                 max_num_batched_tokens: int, pin_memory: bool,
                 device: torch.device, block_size: int) -> None:
        self.block_tables = [
            BlockTable(max_num_reqs, cdiv(max_model_len, block_size),
                       max_num_batched_tokens, pin_memory, device)
        ]

    def append_row(self, block_ids: list[list[int]], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: list[list[int]], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def commit(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit(num_reqs)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]


class PtrTable:
    """
    A table for storing KV cache pointers for flexi_direct implementation.
    Similar to BlockTable but stores uint64 pointers instead of int32 block indices.
    
    For each layer, stores a mapping from (batch_idx, block_idx) to actual memory pointer.
    Stores either K or V pointers (not both) - create two instances for K and V.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        num_layers: int,
        pin_memory: bool,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.num_layers = num_layers
        self.pin_memory = pin_memory
        self.device = device

        # GPU tensor: (num_layers, max_num_reqs, max_num_blocks_per_req)
        # Use int64 internally for computation (PyTorch CUDA supports it better)
        # then view as uint64 when returning
        self.ptr_table = torch.zeros(
            (num_layers, max_num_reqs, max_num_blocks_per_req),
            device=self.device,
            dtype=torch.int64,
        )
        # CPU tensor with pinned memory for async copy (for potential future use)
        self.ptr_table_cpu = torch.zeros(
            (num_layers, max_num_reqs, max_num_blocks_per_req),
            device="cpu",
            dtype=torch.int64,
            pin_memory=pin_memory,
        )
        # Cached stacked ptr_tensors to avoid repeated stacking
        # Shape: (num_layers, num_blocks_per_layer) int64
        self._cached_stacked_ptr_tensors: Optional[torch.Tensor] = None
        self._cached_ptr_tensor_id: Optional[int] = None  # Use id() to detect changes

    def prepare_stacked_tensors(self, ptr_tensors: list[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Prepare new stacked ptr_tensors without switching.
        Use this to pre-compute the stacked tensor, then call commit_stacked_tensors()
        to atomically switch to it.
        
        Args:
            ptr_tensors: List of per-layer pointer tensors, each (num_blocks,) uint64
        
        Returns:
            The prepared stacked tensor, or None if no valid layers
        """
        num_layers = len(ptr_tensors)
        if num_layers == 0:
            return None
        
        # Find valid layers and get max num_blocks
        valid_layers = [(i, ptr_tensors[i]) for i in range(num_layers) if ptr_tensors[i].numel() > 0]
        if not valid_layers:
            return None
        
        max_blocks = max(t.numel() for _, t in valid_layers)
        
        # Pre-allocate stacked tensor: (num_layers, max_blocks)
        stacked = torch.zeros((num_layers, max_blocks), dtype=torch.int64, device=self.device)
        
        # Fill in valid layers
        for layer_idx, t in valid_layers:
            t_int64 = t.view(torch.int64)
            stacked[layer_idx, :t_int64.numel()] = t_int64
        
        return stacked

    def commit_stacked_tensors(self, stacked: Optional[torch.Tensor], ptr_tensors_id: int) -> None:
        """
        Atomically switch to the prepared stacked tensor.
        This is a fast pointer assignment, safe to call during inference.
        
        Args:
            stacked: The prepared stacked tensor from prepare_stacked_tensors()
            ptr_tensors_id: id(ptr_tensors) for cache invalidation check
        """
        self._cached_stacked_ptr_tensors = stacked
        self._cached_ptr_tensor_id = ptr_tensors_id

    def set_ptr_tensors(self, ptr_tensors: list[torch.Tensor]) -> None:
        """
        Pre-stack ptr_tensors for efficient reuse.
        Call this when ptr_tensors are initialized or updated (rare).
        
        Args:
            ptr_tensors: List of per-layer pointer tensors, each (num_blocks,) uint64
        """
        assert self.num_layers == len(ptr_tensors)
        
        # Find valid layers and get max num_blocks
        not_valid_layers = [(i, ptr_tensors[i]) for i in range(self.num_layers) if ptr_tensors[i].numel() == 0]
        if not_valid_layers:
            for i, t in not_valid_layers:
                logger.info(f"PtrTable: Layer {i} has zero blocks in ptr_tensors, tensor: {t}")
            raise ValueError("All layers must have non-zero blocks in ptr_tensors.")
        
        max_blocks = max(t.numel() for t in ptr_tensors)
        
        # Pre-allocate stacked tensor: (num_layers, max_blocks)
        stacked = torch.zeros((self.num_layers, max_blocks), dtype=torch.int64, device=self.device)
        
        # Fill in valid layers
        for layer_idx, t in enumerate(ptr_tensors):
            t_int64 = t.view(torch.int64)
            stacked[layer_idx, :t_int64.numel()] = t_int64
        
        self._cached_stacked_ptr_tensors = stacked
        self._cached_ptr_tensor_id = id(ptr_tensors)

    def update_from_block_table_and_ptr_tensors(
        self,
        block_table: torch.Tensor,
        ptr_tensors: list[torch.Tensor],
        num_reqs: int,
        use_fused_kernel: bool = True,
    ) -> torch.Tensor:
        """
        Update ptr_table from block_table using efficient GPU indexing.
        
        Args:
            block_table: (num_reqs, max_num_blocks_per_req), int32, block indices on GPU
            ptr_tensors: List of per-layer pointer tensors, each (num_blocks,) uint64
            num_reqs: Number of active requests
            use_fused_kernel: If True, use the fused CUDA kernel (faster).
                              If False, use PyTorch operations (for fallback/debugging).
        
        Returns:
            ptr_table: (num_layers, num_reqs, max_num_blocks_per_req), uint64
        """
        import time
        from vllm.logger import init_logger
        _ptr_logger = init_logger(__name__)
        
        t0 = time.time()
        batch_size = min(num_reqs, block_table.shape[0])
        max_num_blocks = block_table.shape[1]
        num_layers = min(self.num_layers, len(ptr_tensors))
        t_prep = time.time() - t0
        
        if num_layers == 0 or batch_size == 0:
            raise ValueError(f"No valid layers or requests to update PtrTable., num_reqs: {num_reqs}, num_layers: {num_layers}, block_table shape: {block_table.shape}, ptr_tensors length: {len(ptr_tensors)}")
            # return self.ptr_table[:num_layers, :batch_size].view(torch.uint64)
        
        # Use cached stacked tensors if available
        # Note: commit_stacked_tensors() should be called explicitly after migration
        # to update the cache. This check is just a fallback for initial setup.
        t1 = time.time()
        stacked_ptr_tensors = self._cached_stacked_ptr_tensors
        assert stacked_ptr_tensors is not None
        t_cache = time.time() - t1
        
        t2 = time.time()
        if use_fused_kernel:
            # Use fused CUDA kernel for maximum performance
            from vllm._custom_ops import update_ptr_table_from_block_table
            update_ptr_table_from_block_table(
                self.ptr_table,
                block_table,
                stacked_ptr_tensors,
                num_layers,
                batch_size,
            )
        else:
            # Fallback: use PyTorch operations
            num_blocks = stacked_ptr_tensors.shape[1]
            
            # Flatten and clamp block_table: (batch_size * max_num_blocks,)
            flat_block_table = block_table[:batch_size].flatten().long()
            clamped_indices = flat_block_table.clamp(0, num_blocks - 1)
            
            # Create valid mask: (1, batch_size * max_num_blocks)
            valid_mask = (flat_block_table >= 0).to(torch.int64).unsqueeze(0)
            
            # Expand indices for all layers: (num_layers, batch_size * max_num_blocks)
            indices_expanded = clamped_indices.unsqueeze(0).expand(num_layers, -1)
            
            # Gather from stacked tensors: (num_layers, batch_size * max_num_blocks)
            gathered = torch.gather(stacked_ptr_tensors[:num_layers], 1, indices_expanded)
            
            # Apply mask (broadcast)
            gathered = gathered * valid_mask
            
            # Reshape and store: (num_layers, batch_size, max_num_blocks)
            self.ptr_table[:num_layers, :batch_size] = gathered.view(num_layers, batch_size, max_num_blocks)
        
        t_kernel = time.time() - t2
        
        # Log detailed timing (only for slow updates or periodically)
        total_time = t_prep + t_cache + t_kernel
        if total_time > 0.001:  # Log if > 1ms
            _ptr_logger.info(
                "[PTR_TABLE_TIMING] update: prep=%.3fms, cache=%.3fms, kernel=%.3fms, total=%.3fms "
                "(layers=%d, batch=%d, blocks=%d)",
                t_prep * 1000, t_cache * 1000, t_kernel * 1000, total_time * 1000,
                num_layers, batch_size, max_num_blocks)
        
        return self.ptr_table[:num_layers, :batch_size].view(torch.uint64)

    def get_device_tensor(self) -> torch.Tensor:
        """Returns the device tensor of the ptr table as uint64."""
        return self.ptr_table.view(torch.uint64)

    def clear(self) -> None:
        self.ptr_table.fill_(0)
        self.ptr_table_cpu.fill_(0)

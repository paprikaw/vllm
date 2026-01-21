"""
Test for the fused update_ptr_table_from_block_table CUDA kernel.

This kernel fuses the following operations:
1. Flatten and convert block_table to long
2. Clamp indices to valid range  
3. Create valid mask (block_id >= 0)
4. Gather from stacked_ptr_tensors for all layers
5. Apply mask and reshape

The test compares the fused kernel output against a reference PyTorch implementation.
"""

import pytest
import torch

from vllm._custom_ops import update_ptr_table_from_block_table


def reference_update_ptr_table(
    block_table: torch.Tensor,
    ptr_tensors: torch.Tensor,
    num_layers: int,
    batch_size: int,
    max_blocks_per_req: int,
) -> torch.Tensor:
    """
    Reference implementation using standard PyTorch operations.
    
    Args:
        block_table: (batch_size, max_blocks_per_req), int32
        ptr_tensors: (num_layers, num_blocks), int64 (representing uint64 pointers)
        num_layers: Number of layers
        batch_size: Number of requests
        max_blocks_per_req: Maximum blocks per request
        
    Returns:
        ptr_table: (num_layers, batch_size, max_blocks_per_req), int64
    """
    num_blocks = ptr_tensors.shape[1]
    
    # Flatten block_table: (batch_size * max_blocks_per_req,)
    flat_block_table = block_table[:batch_size].flatten().long()
    
    # Clamp indices to valid range
    clamped_indices = flat_block_table.clamp(0, num_blocks - 1)
    
    # Create valid mask: (1, batch_size * max_blocks_per_req)
    valid_mask = (flat_block_table >= 0).to(torch.int64).unsqueeze(0)
    
    # Expand indices for all layers: (num_layers, batch_size * max_blocks_per_req)
    indices_expanded = clamped_indices.unsqueeze(0).expand(num_layers, -1)
    
    # Gather from stacked tensors: (num_layers, batch_size * max_blocks_per_req)
    gathered = torch.gather(ptr_tensors[:num_layers], 1, indices_expanded)
    
    # Apply mask (broadcast)
    gathered = gathered * valid_mask
    
    # Reshape: (num_layers, batch_size, max_blocks_per_req)
    ptr_table = gathered.view(num_layers, batch_size, max_blocks_per_req)
    
    return ptr_table


class TestUpdatePtrTableFromBlockTable:
    """Test cases for the fused update_ptr_table_from_block_table kernel."""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")
    
    @pytest.mark.parametrize("num_layers", [1, 8, 32, 64])
    @pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
    @pytest.mark.parametrize("max_blocks_per_req", [8, 32, 128])
    @pytest.mark.parametrize("num_blocks", [256, 1024, 4096])
    def test_correctness(
        self,
        device,
        num_layers: int,
        batch_size: int, 
        max_blocks_per_req: int,
        num_blocks: int,
    ):
        """Test that fused kernel produces same results as reference."""
        # Create random block_table with some invalid entries (-1)
        block_table = torch.randint(
            -1, num_blocks, 
            (batch_size, max_blocks_per_req),
            dtype=torch.int32,
            device=device
        )
        
        # Create random pointer tensors (simulating memory addresses)
        ptr_tensors = torch.randint(
            0, 2**62,  # Use int64 range (will be viewed as uint64)
            (num_layers, num_blocks),
            dtype=torch.int64,
            device=device
        )
        
        # Allocate output tensor
        ptr_table = torch.zeros(
            (num_layers, batch_size, max_blocks_per_req),
            dtype=torch.int64,
            device=device
        )
        
        # Run fused kernel
        update_ptr_table_from_block_table(
            ptr_table, block_table, ptr_tensors, num_layers, batch_size
        )
        
        # Run reference implementation
        ref_ptr_table = reference_update_ptr_table(
            block_table, ptr_tensors, num_layers, batch_size, max_blocks_per_req
        )
        
        # Compare results
        torch.testing.assert_close(ptr_table, ref_ptr_table)
    
    def test_all_invalid_blocks(self, device):
        """Test when all block indices are invalid (-1)."""
        num_layers = 8
        batch_size = 4
        max_blocks_per_req = 16
        num_blocks = 256
        
        # All blocks are invalid
        block_table = torch.full(
            (batch_size, max_blocks_per_req),
            -1,
            dtype=torch.int32,
            device=device
        )
        
        ptr_tensors = torch.randint(
            0, 2**62,
            (num_layers, num_blocks),
            dtype=torch.int64,
            device=device
        )
        
        ptr_table = torch.zeros(
            (num_layers, batch_size, max_blocks_per_req),
            dtype=torch.int64,
            device=device
        )
        
        update_ptr_table_from_block_table(
            ptr_table, block_table, ptr_tensors, num_layers, batch_size
        )
        
        # All outputs should be 0
        assert torch.all(ptr_table == 0)
    
    def test_all_valid_blocks(self, device):
        """Test when all block indices are valid."""
        num_layers = 8
        batch_size = 4
        max_blocks_per_req = 16
        num_blocks = 256
        
        # All blocks are valid
        block_table = torch.randint(
            0, num_blocks,
            (batch_size, max_blocks_per_req),
            dtype=torch.int32,
            device=device
        )
        
        ptr_tensors = torch.randint(
            0, 2**62,
            (num_layers, num_blocks),
            dtype=torch.int64,
            device=device
        )
        
        ptr_table = torch.zeros(
            (num_layers, batch_size, max_blocks_per_req),
            dtype=torch.int64,
            device=device
        )
        
        update_ptr_table_from_block_table(
            ptr_table, block_table, ptr_tensors, num_layers, batch_size
        )
        
        ref_ptr_table = reference_update_ptr_table(
            block_table, ptr_tensors, num_layers, batch_size, max_blocks_per_req
        )
        
        torch.testing.assert_close(ptr_table, ref_ptr_table)
    
    def test_partial_batch(self, device):
        """Test when only part of block_table rows are used."""
        num_layers = 8
        total_batch_size = 16
        actual_batch_size = 4
        max_blocks_per_req = 16
        num_blocks = 256
        
        block_table = torch.randint(
            -1, num_blocks,
            (total_batch_size, max_blocks_per_req),
            dtype=torch.int32,
            device=device
        )
        
        ptr_tensors = torch.randint(
            0, 2**62,
            (num_layers, num_blocks),
            dtype=torch.int64,
            device=device
        )
        
        # Larger output tensor than needed
        ptr_table = torch.zeros(
            (num_layers, total_batch_size, max_blocks_per_req),
            dtype=torch.int64,
            device=device
        )
        
        update_ptr_table_from_block_table(
            ptr_table, block_table, ptr_tensors, num_layers, actual_batch_size
        )
        
        ref_ptr_table = reference_update_ptr_table(
            block_table, ptr_tensors, num_layers, actual_batch_size, max_blocks_per_req
        )
        
        # Only check the actual_batch_size portion
        torch.testing.assert_close(
            ptr_table[:num_layers, :actual_batch_size],
            ref_ptr_table
        )
    
    def test_single_element(self, device):
        """Test edge case with single element."""
        num_layers = 1
        batch_size = 1
        max_blocks_per_req = 1
        num_blocks = 1
        
        block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
        ptr_tensors = torch.tensor([[12345678]], dtype=torch.int64, device=device)
        
        ptr_table = torch.zeros(
            (num_layers, batch_size, max_blocks_per_req),
            dtype=torch.int64,
            device=device
        )
        
        update_ptr_table_from_block_table(
            ptr_table, block_table, ptr_tensors, num_layers, batch_size
        )
        
        assert ptr_table[0, 0, 0].item() == 12345678
    
    def test_large_scale(self, device):
        """Test with large tensors to ensure performance."""
        num_layers = 80  # Large model like Llama 70B
        batch_size = 256
        max_blocks_per_req = 512
        num_blocks = 8192
        
        block_table = torch.randint(
            -1, num_blocks,
            (batch_size, max_blocks_per_req),
            dtype=torch.int32,
            device=device
        )
        
        ptr_tensors = torch.randint(
            0, 2**62,
            (num_layers, num_blocks),
            dtype=torch.int64,
            device=device
        )
        
        ptr_table = torch.zeros(
            (num_layers, batch_size, max_blocks_per_req),
            dtype=torch.int64,
            device=device
        )
        
        # Warm up
        for _ in range(3):
            update_ptr_table_from_block_table(
                ptr_table, block_table, ptr_tensors, num_layers, batch_size
            )
        
        torch.cuda.synchronize()
        
        # Benchmark
        import time
        start = time.time()
        num_iters = 100
        for _ in range(num_iters):
            update_ptr_table_from_block_table(
                ptr_table, block_table, ptr_tensors, num_layers, batch_size
            )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / num_iters) * 1000
        print(f"\nLarge scale benchmark: {avg_time_ms:.3f} ms per call")
        
        # Verify correctness
        ref_ptr_table = reference_update_ptr_table(
            block_table, ptr_tensors, num_layers, batch_size, max_blocks_per_req
        )
        torch.testing.assert_close(ptr_table, ref_ptr_table)


@pytest.mark.benchmark
class TestUpdatePtrTableBenchmark:
    """Benchmark tests comparing fused kernel vs reference implementation."""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")
    
    def test_benchmark_comparison(self, device):
        """Compare performance of fused kernel vs reference."""
        num_layers = 32
        batch_size = 64
        max_blocks_per_req = 128
        num_blocks = 2048
        
        block_table = torch.randint(
            -1, num_blocks,
            (batch_size, max_blocks_per_req),
            dtype=torch.int32,
            device=device
        )
        
        ptr_tensors = torch.randint(
            0, 2**62,
            (num_layers, num_blocks),
            dtype=torch.int64,
            device=device
        )
        
        ptr_table = torch.zeros(
            (num_layers, batch_size, max_blocks_per_req),
            dtype=torch.int64,
            device=device
        )
        
        import time
        num_iters = 100
        
        # Warm up
        for _ in range(10):
            update_ptr_table_from_block_table(
                ptr_table, block_table, ptr_tensors, num_layers, batch_size
            )
            _ = reference_update_ptr_table(
                block_table, ptr_tensors, num_layers, batch_size, max_blocks_per_req
            )
        
        torch.cuda.synchronize()
        
        # Benchmark fused kernel
        start = time.time()
        for _ in range(num_iters):
            update_ptr_table_from_block_table(
                ptr_table, block_table, ptr_tensors, num_layers, batch_size
            )
        torch.cuda.synchronize()
        fused_time = (time.time() - start) / num_iters * 1000
        
        # Benchmark reference
        start = time.time()
        for _ in range(num_iters):
            _ = reference_update_ptr_table(
                block_table, ptr_tensors, num_layers, batch_size, max_blocks_per_req
            )
        torch.cuda.synchronize()
        ref_time = (time.time() - start) / num_iters * 1000
        
        print(f"\n{'='*50}")
        print(f"Benchmark Results (num_layers={num_layers}, batch_size={batch_size})")
        print(f"{'='*50}")
        print(f"Fused kernel:      {fused_time:.3f} ms")
        print(f"Reference PyTorch: {ref_time:.3f} ms")
        print(f"Speedup:           {ref_time/fused_time:.2f}x")
        print(f"{'='*50}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

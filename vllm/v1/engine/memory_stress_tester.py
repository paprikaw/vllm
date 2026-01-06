"""
Memory Stress Tester for vLLM
Continuously allocates list of tensors (KV cache style) during inference 
to test impact on Flexi vs Regular KV cache allocation performance
"""

import threading
import time
import torch
from typing import Optional, List, Literal, Union
from vllm.logger import init_logger
from vllm.v1.utils import human_readable_duration
from vllm.kv_allocator import kv_allocator

logger = init_logger(__name__)

# Allocation strategy types
AllocationStrategy = Literal["torch_empty", "cuda_malloc_async"]


class MemoryStressTester:
    """
    Background thread that continuously allocates list of tensors (similar to KV cache blocks)
    to test the impact on Flexi vs Regular allocation performance during inference.
    
    Simulates KV cache allocation pattern: list of independent tensors with specific shape.
    """
    
    def __init__(
        self,
        forwarding_lock: threading.Lock,
        device: str = "cuda:0",
        num_tensors_per_allocation: int = 1000,
        tensor_shape: tuple = (16, 8, 128),  # Default: (num_heads, block_size, head_dim)
        allocation_interval_ms: int = 100,
        num_allocation_cycles: int = 10,
        max_total_allocations: Optional[int] = None,
        allocation_strategy: AllocationStrategy = "torch_empty",
        enabled: bool = False,
    ):
        """
        Args:
            device: CUDA device to allocate on
            num_tensors_per_allocation: Number of tensors in each list allocation (like num_blocks)
            tensor_shape: Shape of each tensor (default matches KV cache block)
            allocation_interval_ms: Interval between allocation cycles in milliseconds
            num_allocation_cycles: Number of allocation cycles to maintain
            max_total_allocations: Maximum number of total allocations (None = unlimited)
            allocation_strategy: Strategy for allocating tensors ("torch_empty" or "cuda_malloc_async")
            enabled: Whether the stress tester is enabled
        """
        self.device = torch.device(device)
        self.num_tensors_per_allocation = num_tensors_per_allocation
        self.tensor_shape = tensor_shape
        self.allocation_interval_ms = allocation_interval_ms
        self.num_allocation_cycles = num_allocation_cycles
        self.max_total_allocations = max_total_allocations
        self.allocation_strategy = allocation_strategy
        self.enabled = enabled
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._allocation_cycles = []  # List of allocation cycles
        self._lock = threading.Lock()
        
        self._total_allocations = 0
        self._total_deallocations = 0
        self._allocation_errors = 0
        self._total_allocation_time_ms = 0.0
        
        if self.allocation_strategy == "cuda_malloc_async" and self.enabled and kv_allocator is None:
            logger.warning("cuda_malloc_async strategy requires kv_allocator, but it was not provided. Falling back to torch_empty.")
            self.allocation_strategy = "torch_empty"
        self.forwarding_lock = forwarding_lock 
        # Calculate memory footprint
        numel = 1
        for dim in tensor_shape:
            numel *= dim
        bytes_per_tensor = numel * 2  # float16
        total_mb = (bytes_per_tensor * num_tensors_per_allocation) / (1024 * 1024)
        
        if self.enabled:
            logger.info(
                f"MemoryStressTester initialized: "
                f"device={device}, "
                f"tensors_per_alloc={num_tensors_per_allocation}, "
                f"tensor_shape={tensor_shape}, "
                f"interval={allocation_interval_ms}ms, "
                f"cycles={num_allocation_cycles}, "
                f"max_allocations={max_total_allocations if max_total_allocations else 'unlimited'}, "
                f"strategy={allocation_strategy}, "
                f"memory_per_cycle={total_mb:.1f}MB"
            )
    
    def _allocate_tensor_list_torch_empty(self) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Allocate tensor list using torch.empty (default PyTorch allocation)"""
        key_tensor_list = []
        value_tensor_list = []
        with torch.no_grad():
            for _ in range(self.num_tensors_per_allocation):
                tensor = torch.empty(
                    self.tensor_shape,
                    dtype=torch.float16,
                    device=self.device
                )
                key_tensor_list.append(tensor)
        with torch.no_grad():
            for _ in range(self.num_tensors_per_allocation):
                tensor = torch.empty(
                    self.tensor_shape,
                    dtype=torch.float16,
                    device=self.device
                )
                value_tensor_list.append(tensor)
        
        return key_tensor_list, value_tensor_list
    
    def _allocate_tensor_list_cuda_malloc_async(self) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Allocate tensor list using cudaMallocAsync (stream-ordered allocation)"""
        # Use pre-loaded C++ extension
        # Get the current CUDA stream to ensure we use the same stream as other PyTorch operations
        current_stream = torch.cuda.current_stream(self.device)
        stream_ptr = current_stream.cuda_stream
        # Use C++ implementation for cudaMallocAsync
        # allocate_with_cuda_async returns (k_ptrs: List[int], v_ptrs: List[int], time_ms)
        k_ptrs, v_ptrs, _ = kv_allocator.allocate_with_cuda_async(
            size=self.num_tensors_per_allocation,
            block_shape=list(self.tensor_shape),
            dtype=torch.float16,
            device=self.device,
            stream_ptr=stream_ptr
        )
        
        # Store pointers directly without wrapping in tensors
        # Return as lists to maintain compatibility with the interface
        return k_ptrs, v_ptrs

    
    def _allocate_tensor_list(self) -> Union[tuple[List[int], List[int]], tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Allocate tensor list using the configured strategy"""
        if self.allocation_strategy == "torch_empty":
            return self._allocate_tensor_list_torch_empty()
        elif self.allocation_strategy == "cuda_malloc_async":
            return self._allocate_tensor_list_cuda_malloc_async()
        else:
            raise ValueError(f"Unknown allocation strategy: {self.allocation_strategy}")
    
    def start(self):
        """Start the memory stress testing thread"""
        if not self.enabled:
            logger.info("MemoryStressTester is disabled, not starting")
            return
        
        if self._thread is not None and self._thread.is_alive():
            logger.warning("MemoryStressTester is already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._allocation_loop,
            name="MemoryStressTester",
            daemon=True
        )
        self._thread.start()
        logger.info("MemoryStressTester started")
    
    def stop(self):
        """Stop the memory stress testing thread"""
        if not self.enabled:
            return
        
        # Free any remaining allocations if using cuda_malloc_async
        with self._lock:
            if self.allocation_strategy == "cuda_malloc_async":
                try:
                    current_stream = torch.cuda.current_stream(self.device)
                    stream_ptr = current_stream.cuda_stream
                    device_id = self.device.index if self.device.type == 'cuda' else 0
                    # Free all remaining allocation cycles (pointer lists)
                    for k_ptrs, v_ptrs in self._allocation_cycles:
                        kv_allocator.free_cache(k_ptrs, device_id, stream_ptr)
                        kv_allocator.free_cache(v_ptrs, device_id, stream_ptr)
                    logger.info(f"Freed {len(self._allocation_cycles)} allocation cycles using cudaFreeAsync")
                except Exception as e:
                    logger.warning(f"Failed to free allocations: {e}")
            
            # Clear allocation cycles
            self._allocation_cycles.clear()
        
        avg_time = (self._total_allocation_time_ms / self._total_allocations 
                   if self._total_allocations > 0 else 0)
        
        logger.info(
            f"MemoryStressTester stopped. "
            f"Stats: allocations={self._total_allocations}, "
            f"deallocations={self._total_deallocations}, "
            f"errors={self._allocation_errors}, "
            f"avg_alloc_time={avg_time:.2f}ms"
        )
        
        logger.info(
            f"MemoryStressTester stopped. "
            f"Stats: allocations={self._total_allocations}, "
            f"deallocations={self._total_deallocations}, "
            f"errors={self._allocation_errors}"
        )
    
    def get_stats(self) -> dict:
        """Get current statistics"""
        with self._lock:
            avg_time = (self._total_allocation_time_ms / self._total_allocations 
                       if self._total_allocations > 0 else 0)
            return {
                "enabled": self.enabled,
                "total_allocations": self._total_allocations,
                "total_deallocations": self._total_deallocations,
                "allocation_errors": self._allocation_errors,
                "current_cycles": len(self._allocation_cycles),
                "avg_allocation_time_ms": avg_time,
            }
    
    def _allocation_loop(self):
        """Main loop that performs continuous list-of-tensors allocations (KV cache style)"""
        logger.info("MemoryStressTester allocation loop started")
        time_start = time.time()
        try:
            while not self._stop_event.is_set():
                # Check if we've reached the maximum number of allocations
                if self.max_total_allocations is not None and self._total_allocations >= self.max_total_allocations:
                    logger.info(
                        f"MemoryStressTester reached max allocations: {self.max_total_allocations}, stopping"
                    )
                    break
                with self.forwarding_lock:
                    try:
                        start_time = time.perf_counter()

                        # Allocate list of tensors using the configured strategy
                        key_tensor_list, value_tensor_list = self._allocate_tensor_list()

                        alloc_time_ms = (time.perf_counter() - start_time) * 1000

                        # If we have too many cycles, remove the oldest one
                        if len(self._allocation_cycles) >= self.num_allocation_cycles:
                            old_cycle = self._allocation_cycles.pop(0)
                            self._total_deallocations += 1

                            # Explicitly free memory if using cuda_malloc_async
                            if self.allocation_strategy == "cuda_malloc_async":
                                try:
                                    current_stream = torch.cuda.current_stream(self.device)
                                    stream_ptr = current_stream.cuda_stream
                                    device_id = self.device.index if self.device.type == 'cuda' else 0
                                    # old_cycle contains pointer lists directly
                                    k_ptrs = old_cycle[0]
                                    v_ptrs = old_cycle[1]
                                    kv_allocator.free_cache(k_ptrs, device_id, stream_ptr)
                                    kv_allocator.free_cache(v_ptrs, device_id, stream_ptr)
                                except Exception as e:
                                    logger.warning(f"Failed to free old cycle: {e}")
                        # Store new allocation cycle
                        self._allocation_cycles.append((key_tensor_list, value_tensor_list))
                        self._total_allocations += 1
                        self._total_allocation_time_ms += alloc_time_ms
                        avg_time = self._total_allocation_time_ms / self._total_allocations
                        logger.info(
                            f"MemoryStressTester: cycle #{self._total_allocations}, "
                            f"alloc_time={alloc_time_ms:.2f}ms, "
                            f"avg={avg_time:.2f}ms, "
                            f"current_cycles={len(self._allocation_cycles)}, "
                            f"total_deallocations={self._total_deallocations}, "
                        )
                    except Exception as e:
                        self._allocation_errors += 1
                        if self._allocation_errors % 10 == 1:
                            logger.warning(
                                f"MemoryStressTester allocation error (#{self._allocation_errors}): {e}"
                            )
                time.sleep(0.003)
                # Wait for next allocation cycle
                self._stop_event.wait(timeout=self.allocation_interval_ms / 1000.0)
            logger.info(f"MemoryStressTester allocation loop time: {human_readable_duration(time.time() - time_start)}") 
        except Exception as e:
            logger.error(f"MemoryStressTester loop crashed: {e}", exc_info=True)
        finally:
            logger.info("MemoryStressTester allocation loop exiting")


# Global instance for easy access
_global_stress_tester: Optional[MemoryStressTester] = None


def get_global_stress_tester() -> Optional[MemoryStressTester]:
    """Get the global memory stress tester instance"""
    return _global_stress_tester


def initialize_stress_tester(
    forwarding_lock: threading.Lock,
    device: str = "cuda:0",
    num_tensors_per_allocation: int = 1000,
    tensor_shape: tuple = (16, 8, 128),
    allocation_interval_ms: int = 100,
    num_allocation_cycles: int = 10,
    max_total_allocations: Optional[int] = None,
    allocation_strategy: AllocationStrategy = "torch_empty",
    enabled: bool = False,
) -> MemoryStressTester:
    """Initialize and return the global memory stress tester"""
    global _global_stress_tester
    
    if _global_stress_tester is not None:
        logger.warning("MemoryStressTester already initialized, returning existing instance")
        return _global_stress_tester
    
    _global_stress_tester = MemoryStressTester(
        forwarding_lock=forwarding_lock,
        device=device,
        num_tensors_per_allocation=num_tensors_per_allocation,
        tensor_shape=tensor_shape,
        allocation_interval_ms=allocation_interval_ms,
        num_allocation_cycles=num_allocation_cycles,
        max_total_allocations=max_total_allocations,
        allocation_strategy=allocation_strategy,
        enabled=enabled,
    )
    
    return _global_stress_tester


def cleanup_stress_tester():
    """Cleanup the global memory stress tester"""
    global _global_stress_tester
    
    if _global_stress_tester is not None:
        _global_stress_tester.stop()
        _global_stress_tester = None

"""GPU Memory Monitor for Dynamic Pipeline Parallelism Migration.

This module tracks GPU memory usage across different pipeline configurations
and detects memory leaks by comparing post-migration memory with recorded baselines.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global step counter for memory snapshots
_snapshot_counter = 0
_snapshot_lock = threading.Lock()


def memory_snapshot(tag: str, device: Optional[torch.device] = None) -> Dict[str, float]:
    """Take a detailed GPU memory snapshot and log it.
    
    Args:
        tag: A descriptive tag for this snapshot point (e.g. "before_resize", "after_delete_layers")
        device: CUDA device to query. If None, uses current device.
        
    Returns:
        Dict with memory values in GB.
    """
    global _snapshot_counter
    with _snapshot_lock:
        _snapshot_counter += 1
        step = _snapshot_counter
    
    if device is not None:
        torch.cuda.set_device(device)
    
    torch.cuda.synchronize()
    
    free_mem, total_mem = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    
    # Memory in the PyTorch cache but not currently allocated
    cached_not_used = reserved - allocated
    
    snapshot = {
        'free_gb': free_mem / 1024**3,
        'total_gb': total_mem / 1024**3,
        'allocated_gb': allocated / 1024**3,
        'reserved_gb': reserved / 1024**3,
        'cached_unused_gb': cached_not_used / 1024**3,
        'max_allocated_gb': max_allocated / 1024**3,
    }
    
    logger.info(
        f"[MEM_SNAPSHOT #{step}] [{tag}] "
        f"free={snapshot['free_gb']:.3f}GB, "
        f"allocated={snapshot['allocated_gb']:.3f}GB, "
        f"reserved={snapshot['reserved_gb']:.3f}GB, "
        f"cached_unused={snapshot['cached_unused_gb']:.3f}GB, "
        f"max_alloc={snapshot['max_allocated_gb']:.3f}GB, "
        f"total={snapshot['total_gb']:.3f}GB"
    )
    
    return snapshot


def gc_and_empty_cache():
    """Force garbage collection and empty CUDA cache."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# Unified Memory Overhead Monitor
# ============================================================================

# Global singleton instance for MemoryOverheadMonitor
_memory_overhead_monitor: Optional["MemoryOverheadMonitor"] = None
_overhead_monitor_lock = threading.Lock()


class MemoryOverheadMonitor:
    """Unified memory overhead monitor for detecting memory leaks.
    
    Overhead is defined as the memory used beyond weights and KV cache:
        overhead = used_memory - expected_memory
        expected_memory = layer_count * per_layer_weight_size + kv_cache_size
        used_memory = total_memory - free_memory
    
    The baseline overhead is captured after initial model loading and should
    remain stable throughout migration operations. Any increase in overhead
    indicates a potential memory leak.
    
    Usage:
        1. After model loading, call `initialize_baseline()` to establish baseline
        2. Before memory operations, call `check_overhead_before()`
        3. After memory operations, call `check_overhead_after()`
    """
    
    # Default tolerance in GB
    DEFAULT_TOLERANCE_GB = 0.1  # 100 MB
    
    def __init__(
        self,
        rank: int,
        device: Optional[torch.device] = None,
        per_layer_weight_bytes: int = 0,
        per_block_kv_cache_bytes: int = 0,
        tolerance_gb: float = DEFAULT_TOLERANCE_GB
    ):
        """Initialize the memory overhead monitor.
        
        Args:
            rank: Worker rank
            device: CUDA device
            per_layer_weight_bytes: Size of a single transformer layer's weights in bytes
            per_block_kv_cache_bytes: Size of KV cache per block (for resize tracking)
            tolerance_gb: Tolerance for overhead deviation in GB
        """
        self.rank = rank
        self.device = device
        self.per_layer_weight_bytes = per_layer_weight_bytes
        self.per_block_kv_cache_bytes = per_block_kv_cache_bytes
        self.tolerance_gb = tolerance_gb
        
        # Baseline overhead (captured after initialization)
        self.baseline_overhead_bytes: Optional[int] = None
        self.baseline_layer_count: int = 0
        self.baseline_kv_cache_bytes: int = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Operation tracking
        self._current_operation: Optional[str] = None
        self._before_state: Optional[Dict] = None
        
        logger.info(
            f"[MemoryOverheadMonitor] Initialized for rank {rank}, "
            f"per_layer_weight={per_layer_weight_bytes / 1024**3:.3f}GB, "
            f"per_block_kv={per_block_kv_cache_bytes / 1024**3:.6f}GB, "
            f"tolerance={tolerance_gb:.3f}GB"
        )
    
    def _get_memory_state(self) -> Dict:
        """Get current memory state."""
        if self.device is not None:
            torch.cuda.set_device(self.device)
        
        torch.cuda.synchronize()
        gc_and_empty_cache()
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        return {
            'free_bytes': free_mem,
            'total_bytes': total_mem,
            'used_bytes': total_mem - free_mem,
            'allocated_bytes': allocated,
            'reserved_bytes': reserved,
        }
    
    def _calculate_expected_memory(self, layer_count: int, kv_cache_bytes: int) -> int:
        """Calculate expected memory usage (weights + KV cache)."""
        return layer_count * self.per_layer_weight_bytes + kv_cache_bytes
    
    def _calculate_overhead(self, used_bytes: int, layer_count: int, kv_cache_bytes: int) -> int:
        """Calculate overhead = used - expected."""
        expected = self._calculate_expected_memory(layer_count, kv_cache_bytes)
        return used_bytes - expected
    
    def initialize_baseline(
        self,
        layer_count: int,
        kv_cache_bytes: int = 0,
        per_layer_weight_bytes: Optional[int] = None
    ) -> None:
        """Initialize baseline overhead after model loading.
        
        This should be called once after weights are loaded and KV cache is initialized.
        The overhead captured here includes:
        - CUDA context
        - NCCL buffers
        - Ray runtime
        - Intermediate tensors
        - Any other fixed allocations
        
        Args:
            layer_count: Current number of layers loaded
            kv_cache_bytes: Total KV cache size in bytes (0 if not yet allocated)
            per_layer_weight_bytes: Override per-layer weight size if needed
        """
        if per_layer_weight_bytes is not None:
            self.per_layer_weight_bytes = per_layer_weight_bytes
        
        state = self._get_memory_state()
        
        with self._lock:
            self.baseline_overhead_bytes = self._calculate_overhead(
                state['used_bytes'], layer_count, kv_cache_bytes
            )
            self.baseline_layer_count = layer_count
            self.baseline_kv_cache_bytes = kv_cache_bytes
        
        expected_gb = self._calculate_expected_memory(layer_count, kv_cache_bytes) / 1024**3
        overhead_gb = self.baseline_overhead_bytes / 1024**3
        
        logger.info(
            f"[MemoryOverheadMonitor] Baseline initialized | rank={self.rank} | "
            f"layers={layer_count}, kv_cache={kv_cache_bytes / 1024**3:.3f}GB, "
            f"used={state['used_bytes'] / 1024**3:.3f}GB, "
            f"expected={expected_gb:.3f}GB, "
            f"baseline_overhead={overhead_gb:.3f}GB"
        )
    
    def initialize_baseline_with_profile(
        self,
        layer_count: int,
        profile_peak_memory_bytes: int,
        non_torch_allocations_bytes: int = 0,
        per_layer_weight_bytes: Optional[int] = None
    ) -> None:
        """Initialize baseline overhead using profile_run results.
        
        This should be called after weights are loaded but BEFORE KV cache is allocated.
        The profile_run executes a dummy forward pass to measure peak memory including:
        - Model weights
        - Intermediate activations during forward pass
        - CUDA context and runtime allocations
        - NCCL buffers
        
        The baseline overhead is calculated as:
            runtime_overhead = peak_memory + non_torch_allocations - expected_weight_memory
            expected_weight_memory = layer_count * per_layer_weight_bytes
        
        Args:
            layer_count: Current number of layers loaded
            profile_peak_memory_bytes: Peak memory from torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            non_torch_allocations_bytes: Memory allocated outside of torch (e.g., NCCL)
            per_layer_weight_bytes: Override per-layer weight size if needed
        """
        if per_layer_weight_bytes is not None:
            self.per_layer_weight_bytes = per_layer_weight_bytes
        
        # Calculate expected weight memory
        expected_weight_memory = layer_count * self.per_layer_weight_bytes
        
        # Runtime overhead = peak + non_torch - expected_weights
        # This captures intermediate activations, CUDA context, NCCL buffers, etc.
        runtime_overhead = profile_peak_memory_bytes + non_torch_allocations_bytes - expected_weight_memory
        
        with self._lock:
            self.baseline_overhead_bytes = runtime_overhead
            self.baseline_layer_count = layer_count
            self.baseline_kv_cache_bytes = 0  # KV cache not yet allocated
        
        expected_weight_gb = expected_weight_memory / 1024**3
        peak_gb = profile_peak_memory_bytes / 1024**3
        non_torch_gb = non_torch_allocations_bytes / 1024**3
        overhead_gb = runtime_overhead / 1024**3
        
        logger.info(
            f"[MemoryOverheadMonitor] Baseline initialized via profile_run | rank={self.rank} | "
            f"layers={layer_count}, expected_weight={expected_weight_gb:.3f}GB, "
            f"peak_memory={peak_gb:.3f}GB, non_torch={non_torch_gb:.3f}GB, "
            f"baseline_overhead={overhead_gb:.3f}GB"
        )
    
    def check_overhead_before(
        self,
        operation: str,
        current_layer_count: int,
        current_kv_cache_bytes: int = 0,
        expected_kv_change_bytes: Optional[int] = None
    ) -> Tuple[float, bool]:
        """Check overhead before a memory operation.
        
        Args:
            operation: Description of the operation (e.g., "add_layers", "resize_kv_cache")
            current_layer_count: Current number of layers
            current_kv_cache_bytes: Current KV cache size in bytes
            expected_kv_change_bytes: Expected KV cache memory change (for resize tracking)
            
        Returns:
            Tuple of (current_overhead_gb, is_within_tolerance)
        """
        if self.baseline_overhead_bytes is None:
            logger.warning(
                f"[MemoryOverheadMonitor] Baseline not initialized, skipping check"
            )
            return 0.0, True
        
        state = self._get_memory_state()
        current_overhead = self._calculate_overhead(
            state['used_bytes'], current_layer_count, current_kv_cache_bytes
        )
        
        overhead_diff = current_overhead - self.baseline_overhead_bytes
        overhead_diff_gb = overhead_diff / 1024**3
        current_overhead_gb = current_overhead / 1024**3
        baseline_gb = self.baseline_overhead_bytes / 1024**3
        
        # Only flag as leak if overhead is HIGHER than baseline + tolerance
        # Negative diff means less memory used, which is fine
        is_valid = overhead_diff_gb <= self.tolerance_gb
        
        # Store state for after check
        with self._lock:
            self._current_operation = operation
            self._before_state = {
                'layer_count': current_layer_count,
                'kv_cache_bytes': current_kv_cache_bytes,
                'overhead_bytes': current_overhead,
                'used_bytes': state['used_bytes'],
                'expected_kv_change_bytes': expected_kv_change_bytes,
            }
        
        status = "OK" if is_valid else "LEAK_DETECTED"
        expected_info = ""
        if expected_kv_change_bytes is not None:
            expected_info = f", expected_kv_change={expected_kv_change_bytes / 1024**3:+.3f}GB"
        logger.info(
            f"[MemoryOverheadMonitor] BEFORE {operation} | rank={self.rank} | "
            f"layers={current_layer_count}, kv={current_kv_cache_bytes / 1024**3:.3f}GB, "
            f"used={state['used_bytes'] / 1024**3:.3f}GB, "
            f"overhead={current_overhead_gb:.3f}GB (baseline={baseline_gb:.3f}GB, "
            f"diff={overhead_diff_gb:+.4f}GB) [{status}]{expected_info}"
        )
        
        if not is_valid:
            logger.error(
                f"[MemoryOverheadMonitor] MEMORY LEAK DETECTED before {operation}! "
                f"Expected overhead ~{baseline_gb:.3f}GB, actual {current_overhead_gb:.3f}GB, "
                f"leak={overhead_diff_gb:.4f}GB ({overhead_diff / 1024**2:.2f}MB)"
            )
        
        return current_overhead_gb, is_valid
    
    def check_overhead_after(
        self,
        operation: str,
        new_layer_count: int,
        new_kv_cache_bytes: int = 0
    ) -> Tuple[float, bool]:
        """Check overhead after a memory operation.
        
        Args:
            operation: Description of the operation
            new_layer_count: New number of layers after operation
            new_kv_cache_bytes: New KV cache size in bytes after operation
            
        Returns:
            Tuple of (current_overhead_gb, is_within_tolerance)
        """
        if self.baseline_overhead_bytes is None:
            logger.warning(
                f"[MemoryOverheadMonitor] Baseline not initialized, skipping check"
            )
            return 0.0, True
        
        state = self._get_memory_state()
        current_overhead = self._calculate_overhead(
            state['used_bytes'], new_layer_count, new_kv_cache_bytes
        )
        
        overhead_diff = current_overhead - self.baseline_overhead_bytes
        overhead_diff_gb = overhead_diff / 1024**3
        current_overhead_gb = current_overhead / 1024**3
        baseline_gb = self.baseline_overhead_bytes / 1024**3
        
        # Only flag as leak if overhead is HIGHER than baseline + tolerance
        # Negative diff means less memory used, which is fine
        is_valid = overhead_diff_gb <= self.tolerance_gb
        
        # Calculate memory change from before state if available
        change_info = ""
        kv_resize_info = ""
        with self._lock:
            if self._before_state is not None:
                before_used = self._before_state['used_bytes']
                used_change = state['used_bytes'] - before_used
                expected_change = (
                    (new_layer_count - self._before_state['layer_count']) * self.per_layer_weight_bytes +
                    (new_kv_cache_bytes - self._before_state['kv_cache_bytes'])
                )
                unexpected_change = used_change - expected_change
                change_info = (
                    f", used_change={used_change / 1024**3:+.3f}GB "
                    f"(expected={expected_change / 1024**3:+.3f}GB, "
                    f"unexpected={unexpected_change / 1024**3:+.4f}GB)"
                )
                
                # For resize operations, add detailed KV cache change analysis
                # Use actual GPU used_bytes change to estimate real KV change
                expected_kv_change = self._before_state.get('expected_kv_change_bytes')
                if expected_kv_change is not None:
                    # Actual KV change = total used change - layer weight change
                    layer_weight_change = (
                        (new_layer_count - self._before_state['layer_count']) * 
                        self.per_layer_weight_bytes
                    )
                    actual_kv_change = used_change - layer_weight_change
                    kv_diff = actual_kv_change - expected_kv_change
                    kv_resize_info = (
                        f" | KV_RESIZE: expected={expected_kv_change / 1024**3:+.3f}GB, "
                        f"actual={actual_kv_change / 1024**3:+.3f}GB, "
                        f"diff={kv_diff / 1024**3:+.4f}GB"
                    )
                    if abs(kv_diff) > self.tolerance_gb * 1024**3:
                        kv_resize_info += " [KV_MISMATCH]"
                
                self._before_state = None
                self._current_operation = None
        
        status = "OK" if is_valid else "LEAK_DETECTED"
        logger.info(
            f"[MemoryOverheadMonitor] AFTER {operation} | rank={self.rank} | "
            f"layers={new_layer_count}, kv={new_kv_cache_bytes / 1024**3:.3f}GB, "
            f"used={state['used_bytes'] / 1024**3:.3f}GB, "
            f"overhead={current_overhead_gb:.3f}GB (baseline={baseline_gb:.3f}GB, "
            f"diff={overhead_diff_gb:+.4f}GB) [{status}]{change_info}{kv_resize_info}"
        )
        
        if not is_valid:
            logger.error(
                f"[MemoryOverheadMonitor] MEMORY LEAK DETECTED after {operation}! "
                f"Expected overhead ~{baseline_gb:.3f}GB, actual {current_overhead_gb:.3f}GB, "
                f"leak={overhead_diff_gb:.4f}GB ({overhead_diff / 1024**2:.2f}MB)"
            )
        
        return current_overhead_gb, is_valid
    
    def get_baseline_overhead_gb(self) -> Optional[float]:
        """Get baseline overhead in GB."""
        if self.baseline_overhead_bytes is None:
            return None
        return self.baseline_overhead_bytes / 1024**3


def get_memory_overhead_monitor(
    rank: int = 0,
    device: Optional[torch.device] = None,
    per_layer_weight_bytes: int = 0,
    per_block_kv_cache_bytes: int = 0,
    tolerance_gb: float = MemoryOverheadMonitor.DEFAULT_TOLERANCE_GB
) -> MemoryOverheadMonitor:
    """Get or create the global memory overhead monitor."""
    global _memory_overhead_monitor
    with _overhead_monitor_lock:
        if _memory_overhead_monitor is None:
            _memory_overhead_monitor = MemoryOverheadMonitor(
                rank=rank,
                device=device,
                per_layer_weight_bytes=per_layer_weight_bytes,
                per_block_kv_cache_bytes=per_block_kv_cache_bytes,
                tolerance_gb=tolerance_gb
            )
        return _memory_overhead_monitor


def reset_memory_overhead_monitor(
    rank: int = 0,
    device: Optional[torch.device] = None,
    per_layer_weight_bytes: int = 0,
    per_block_kv_cache_bytes: int = 0,
    tolerance_gb: float = MemoryOverheadMonitor.DEFAULT_TOLERANCE_GB
) -> MemoryOverheadMonitor:
    """Reset and create a new memory overhead monitor."""
    global _memory_overhead_monitor
    with _overhead_monitor_lock:
        _memory_overhead_monitor = MemoryOverheadMonitor(
            rank=rank,
            device=device,
            per_layer_weight_bytes=per_layer_weight_bytes,
            per_block_kv_cache_bytes=per_block_kv_cache_bytes,
            tolerance_gb=tolerance_gb
        )
        return _memory_overhead_monitor

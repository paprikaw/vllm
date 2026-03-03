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

# Global singleton instance
_gpu_memory_monitor: Optional["GPUMemoryMonitor"] = None
_monitor_lock = threading.Lock()

# Global step counter for memory snapshots
_snapshot_counter = 0
_snapshot_lock = threading.Lock()

# Global checkpoint tracker
_checkpoint_tracker: Optional["MemoryCheckpointTracker"] = None
_checkpoint_lock = threading.Lock()


def get_gpu_memory_monitor() -> "GPUMemoryMonitor":
    """Get the global GPU memory monitor instance."""
    global _gpu_memory_monitor
    with _monitor_lock:
        if _gpu_memory_monitor is None:
            _gpu_memory_monitor = GPUMemoryMonitor()
        return _gpu_memory_monitor


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


@dataclass
class MemoryCheckpoint:
    """A single memory checkpoint with expected change."""
    tag: str
    operation: str  # "add_layers", "remove_layers", "resize_kv_cache", etc.
    timestamp: float
    free_gb: float
    allocated_gb: float
    reserved_gb: float
    expected_delta_gb: Optional[float] = None  # Expected free memory change
    actual_delta_gb: Optional[float] = None    # Actual free memory change from previous checkpoint
    is_before: bool = True  # True for "before" checkpoint, False for "after"
    rank: int = 0
    details: Dict = field(default_factory=dict)  # Additional context info


class MemoryCheckpointTracker:
    """Track memory checkpoints throughout migration for detailed analysis.
    
    This tracker allows us to:
    1. Record memory state before/after each operation
    2. Calculate expected memory changes based on operation parameters
    3. Compare actual vs expected changes to identify leak sources
    """
    
    def __init__(self, rank: int, device: Optional[torch.device] = None):
        self.rank = rank
        self.device = device
        self.checkpoints: List[MemoryCheckpoint] = []
        self._lock = threading.Lock()
        self._last_snapshot: Optional[Dict[str, float]] = None
        logger.info(f"[MemoryCheckpointTracker] Initialized for rank {rank}")
    
    def _take_snapshot(self) -> Dict[str, float]:
        """Take a memory snapshot and return values."""
        if self.device is not None:
            torch.cuda.set_device(self.device)
        
        torch.cuda.synchronize()
        gc_and_empty_cache()
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        return {
            'free_gb': free_mem / 1024**3,
            'allocated_gb': allocated / 1024**3,
            'reserved_gb': reserved / 1024**3,
            'total_gb': total_mem / 1024**3,
        }
    
    def checkpoint_before(
        self, 
        tag: str, 
        operation: str, 
        expected_delta_gb: Optional[float] = None,
        details: Optional[Dict] = None
    ) -> int:
        """Record a checkpoint BEFORE an operation.
        
        Args:
            tag: Descriptive tag for this checkpoint
            operation: Type of operation (add_layers, remove_layers, etc.)
            expected_delta_gb: Expected change in free memory (positive = more free memory)
            details: Additional context (e.g., layer info, cache sizes)
            
        Returns:
            Checkpoint index for matching with after checkpoint
        """
        snapshot = self._take_snapshot()
        
        actual_delta = None
        if self._last_snapshot is not None:
            actual_delta = snapshot['free_gb'] - self._last_snapshot['free_gb']
        
        cp = MemoryCheckpoint(
            tag=tag,
            operation=operation,
            timestamp=time.time(),
            free_gb=snapshot['free_gb'],
            allocated_gb=snapshot['allocated_gb'],
            reserved_gb=snapshot['reserved_gb'],
            expected_delta_gb=expected_delta_gb,
            actual_delta_gb=actual_delta,
            is_before=True,
            rank=self.rank,
            details=details or {}
        )
        
        with self._lock:
            idx = len(self.checkpoints)
            self.checkpoints.append(cp)
            self._last_snapshot = snapshot
        
        logger.info(
            f"[MEM_CHECKPOINT] BEFORE {operation} | {tag} | "
            f"free={snapshot['free_gb']:.3f}GB, alloc={snapshot['allocated_gb']:.3f}GB, "
            f"reserved={snapshot['reserved_gb']:.3f}GB"
        )
        
        return idx
    
    def checkpoint_after(
        self, 
        tag: str, 
        operation: str, 
        before_idx: int,
        expected_delta_gb: Optional[float] = None
    ) -> Tuple[float, float, bool]:
        """Record a checkpoint AFTER an operation and validate.
        
        Args:
            tag: Descriptive tag for this checkpoint
            operation: Type of operation
            before_idx: Index of corresponding before checkpoint
            expected_delta_gb: Expected change in free memory from before checkpoint
            
        Returns:
            Tuple of (actual_delta_gb, expected_delta_gb, is_within_tolerance)
        """
        snapshot = self._take_snapshot()
        
        with self._lock:
            if before_idx >= len(self.checkpoints):
                logger.error(f"Invalid before_idx: {before_idx}")
                return 0, 0, False
            
            before_cp = self.checkpoints[before_idx]
            actual_delta = snapshot['free_gb'] - before_cp.free_gb
            
            if expected_delta_gb is None:
                expected_delta_gb = before_cp.expected_delta_gb or 0
            
            # Tolerance: 100MB
            tolerance_gb = 0.1
            is_valid = abs(actual_delta - expected_delta_gb) <= tolerance_gb
            
            cp = MemoryCheckpoint(
                tag=tag,
                operation=operation,
                timestamp=time.time(),
                free_gb=snapshot['free_gb'],
                allocated_gb=snapshot['allocated_gb'],
                reserved_gb=snapshot['reserved_gb'],
                expected_delta_gb=expected_delta_gb,
                actual_delta_gb=actual_delta,
                is_before=False,
                rank=self.rank,
                details=before_cp.details
            )
            
            self.checkpoints.append(cp)
            self._last_snapshot = snapshot
        
        status = "OK" if is_valid else "MISMATCH"
        diff = actual_delta - expected_delta_gb
        logger.info(
            f"[MEM_CHECKPOINT] AFTER {operation} | {tag} | "
            f"free={snapshot['free_gb']:.3f}GB, alloc={snapshot['allocated_gb']:.3f}GB | "
            f"delta={actual_delta:+.3f}GB (expected={expected_delta_gb:+.3f}GB, diff={diff:+.4f}GB) [{status}]"
        )
        
        if not is_valid:
            logger.warning(
                f"[MEM_CHECKPOINT] MEMORY MISMATCH in {operation}: "
                f"expected {expected_delta_gb:+.3f}GB change, got {actual_delta:+.3f}GB change, "
                f"discrepancy={diff:+.4f}GB ({diff*1024:.2f}MB)"
            )
        
        return actual_delta, expected_delta_gb, is_valid
    
    def checkpoint_verify(
        self, 
        tag: str, 
        operation: str, 
        expected_free_gb: float,
        tolerance_gb: float = 0.1
    ) -> Tuple[float, bool]:
        """Verify free memory matches expected value.
        
        Args:
            tag: Descriptive tag
            operation: Type of operation
            expected_free_gb: Expected free memory value
            tolerance_gb: Tolerance in GB (default 100MB)
            
        Returns:
            Tuple of (actual_free_gb, is_within_tolerance)
        """
        snapshot = self._take_snapshot()
        actual_free = snapshot['free_gb']
        diff = actual_free - expected_free_gb
        is_valid = abs(diff) <= tolerance_gb
        
        status = "OK" if is_valid else "MISMATCH"
        logger.info(
            f"[MEM_VERIFY] {operation} | {tag} | "
            f"free={actual_free:.3f}GB (expected={expected_free_gb:.3f}GB, diff={diff:+.4f}GB) [{status}]"
        )
        
        if not is_valid:
            logger.warning(
                f"[MEM_VERIFY] MISMATCH in {operation}: "
                f"expected {expected_free_gb:.3f}GB free, got {actual_free:.3f}GB, "
                f"discrepancy={diff:+.4f}GB ({abs(diff)*1024:.2f}MB)"
            )
        
        return actual_free, is_valid
    
    def get_summary(self) -> str:
        """Generate a summary of all checkpoints and mismatches."""
        lines = [f"=== Memory Checkpoint Summary (rank {self.rank}) ==="]
        mismatches = []
        
        with self._lock:
            for i, cp in enumerate(self.checkpoints):
                if not cp.is_before and cp.actual_delta_gb is not None and cp.expected_delta_gb is not None:
                    diff = cp.actual_delta_gb - cp.expected_delta_gb
                    if abs(diff) > 0.1:  # 100MB tolerance
                        mismatches.append((cp, diff))
        
        lines.append(f"Total checkpoints: {len(self.checkpoints)}")
        lines.append(f"Mismatches detected: {len(mismatches)}")
        
        for cp, diff in mismatches:
            lines.append(
                f"  - [{cp.operation}] {cp.tag}: "
                f"expected={cp.expected_delta_gb:+.3f}GB, actual={cp.actual_delta_gb:+.3f}GB, "
                f"diff={diff:+.4f}GB ({abs(diff)*1024:.2f}MB)"
            )
        
        return "\n".join(lines)


def gc_and_empty_cache():
    """Force garbage collection and empty CUDA cache."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def get_checkpoint_tracker(rank: int = 0, device: Optional[torch.device] = None) -> MemoryCheckpointTracker:
    """Get or create the global checkpoint tracker."""
    global _checkpoint_tracker
    with _checkpoint_lock:
        if _checkpoint_tracker is None:
            _checkpoint_tracker = MemoryCheckpointTracker(rank, device)
        return _checkpoint_tracker


def reset_checkpoint_tracker(rank: int = 0, device: Optional[torch.device] = None) -> MemoryCheckpointTracker:
    """Reset and create a new checkpoint tracker."""
    global _checkpoint_tracker
    with _checkpoint_lock:
        _checkpoint_tracker = MemoryCheckpointTracker(rank, device)
        return _checkpoint_tracker


def config_to_key(pp_layer_config: list[Tuple[int, int]], rank: int) -> str:
    """Convert pp_layer_config and rank to a unique string key.
    
    Args:
        pp_layer_config: List of (start_layer, end_layer) tuples per rank
        rank: The current worker's rank
        
    Returns:
        A string key like "rank0:[(0,31),(32,63)]" that uniquely identifies
        this configuration for this rank.
    """
    # Include both global config and rank for uniqueness
    config_str = ",".join(f"({s},{e})" for s, e in pp_layer_config)
    return f"rank{rank}:[{config_str}]"


def local_config_to_key(start_layer: int, end_layer: int, rank: int) -> str:
    """Convert local layer config and rank to a unique string key.
    
    This is a simpler key format that uses just the local rank's layer range.
    
    Args:
        start_layer: Start layer index for this rank
        end_layer: End layer index for this rank
        rank: The current worker's rank
        
    Returns:
        A string key like "rank0:(0,31)" that identifies this configuration.
    """
    return f"rank{rank}:({start_layer},{end_layer})"


class GPUMemoryMonitor:
    """Monitor GPU memory usage across pipeline configurations.
    
    Records baseline GPU free memory for each configuration at first encounter,
    then validates that subsequent visits to the same configuration have
    similar memory usage (within tolerance).
    """
    
    # Default memory difference threshold in bytes (50 MB)
    DEFAULT_THRESHOLD_BYTES = 50 * 1024 * 1024
    
    def __init__(self, threshold_mb: float = 50.0):
        """Initialize the GPU memory monitor.
        
        Args:
            threshold_mb: Memory difference threshold in MB. If post-migration
                         memory differs from recorded baseline by more than this,
                         an error is raised.
        """
        self.threshold_bytes = int(threshold_mb * 1024 * 1024)
        # Dict mapping config_key -> recorded free GPU memory in bytes
        self._config_memory: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._enabled = True
        logger.info(f"[GPUMemoryMonitor] Initialized with threshold={threshold_mb}MB")
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the memory monitor."""
        self._enabled = enabled
        logger.info(f"[GPUMemoryMonitor] {'Enabled' if enabled else 'Disabled'}")
    
    def _get_free_gpu_memory(self, device: Optional[torch.device] = None) -> int:
        """Get current free GPU memory after clearing cache.
        
        Args:
            device: CUDA device to query. If None, uses current device.
            
        Returns:
            Free GPU memory in bytes.
        """
        if device is not None:
            torch.cuda.set_device(device)
        # Clear cache to get accurate free memory reading
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_memory, _ = torch.cuda.mem_get_info()
        return free_memory
    
    def record_initial_memory(
        self,
        pp_layer_config: list[Tuple[int, int]],
        rank: int,
        device: Optional[torch.device] = None
    ) -> None:
        """Record GPU memory for initial configuration.
        
        Called once during initialization after KV cache is set up.
        This establishes the baseline memory for the starting configuration.
        
        Args:
            pp_layer_config: Current pipeline layer configuration
            rank: Current worker's rank
            device: CUDA device to query
        """
        if not self._enabled:
            return
            
        config_key = config_to_key(pp_layer_config, rank)
        free_memory = self._get_free_gpu_memory(device)
        
        with self._lock:
            if config_key not in self._config_memory:
                self._config_memory[config_key] = free_memory
                logger.info(
                    f"[GPUMemoryMonitor] Recorded initial memory for {config_key}: "
                    f"{free_memory / 1024**3:.3f} GB free"
                )
            else:
                logger.warning(
                    f"[GPUMemoryMonitor] Config {config_key} already recorded, skipping"
                )
    
    def check_memory_after_migration(
        self,
        pp_layer_config: list[Tuple[int, int]],
        rank: int,
        device: Optional[torch.device] = None,
        migration_type: str = "unknown"
    ) -> None:
        """Check GPU memory after migration and compare with recorded baseline.
        
        If this configuration was seen before, verify memory usage is within
        the threshold. If not seen before, record it as baseline.
        
        Args:
            pp_layer_config: Target pipeline layer configuration after migration
            rank: Current worker's rank
            device: CUDA device to query
            migration_type: "sync" or "async" for logging purposes
            
        Raises:
            RuntimeError: If memory differs from baseline by more than threshold
        """
        if not self._enabled:
            return
            
        config_key = config_to_key(pp_layer_config, rank)
        free_memory = self._get_free_gpu_memory(device)
        
        with self._lock:
            if config_key in self._config_memory:
                recorded_memory = self._config_memory[config_key]
                diff_bytes = abs(free_memory - recorded_memory)
                diff_mb = diff_bytes / (1024 * 1024)
                
                logger.info(
                    f"[GPUMemoryMonitor] [{migration_type}] Config {config_key}: "
                    f"current={free_memory / 1024**3:.3f} GB, "
                    f"recorded={recorded_memory / 1024**3:.3f} GB, "
                    f"diff={diff_mb:.2f} MB"
                )
                
                if diff_bytes > self.threshold_bytes:
                    error_msg = (
                        f"[GPUMemoryMonitor] MEMORY LEAK DETECTED after {migration_type} migration! "
                        f"Config: {config_key}, "
                        f"Expected free: {recorded_memory / 1024**3:.3f} GB, "
                        f"Actual free: {free_memory / 1024**3:.3f} GB, "
                        f"Difference: {diff_mb:.2f} MB (threshold: {self.threshold_bytes / (1024*1024):.2f} MB)"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                # First time seeing this config, record it
                self._config_memory[config_key] = free_memory
                logger.info(
                    f"[GPUMemoryMonitor] [{migration_type}] Recorded new config {config_key}: "
                    f"{free_memory / 1024**3:.3f} GB free"
                )

    def record_initial_memory_local(
        self,
        start_layer: int,
        end_layer: int,
        rank: int,
        device: Optional[torch.device] = None
    ) -> None:
        """Record GPU memory for initial configuration using local layer range.
        
        Args:
            start_layer: Start layer index for this rank
            end_layer: End layer index for this rank  
            rank: Current worker's rank
            device: CUDA device to query
        """
        if not self._enabled:
            return
            
        config_key = local_config_to_key(start_layer, end_layer, rank)
        free_memory = self._get_free_gpu_memory(device)
        
        with self._lock:
            if config_key not in self._config_memory:
                self._config_memory[config_key] = free_memory
                logger.info(
                    f"[GPUMemoryMonitor] Recorded initial memory for {config_key}: "
                    f"{free_memory / 1024**3:.3f} GB free"
                )
            else:
                logger.warning(
                    f"[GPUMemoryMonitor] Config {config_key} already recorded, skipping"
                )

    def check_memory_after_migration_local(
        self,
        start_layer: int,
        end_layer: int,
        rank: int,
        device: Optional[torch.device] = None,
        migration_type: str = "unknown"
    ) -> None:
        """Check GPU memory after migration using local layer range.
        
        Args:
            start_layer: Start layer index for this rank after migration
            end_layer: End layer index for this rank after migration
            rank: Current worker's rank
            device: CUDA device to query
            migration_type: "sync" or "async" for logging purposes
        """
        if not self._enabled:
            return
            
        config_key = local_config_to_key(start_layer, end_layer, rank)
        free_memory = self._get_free_gpu_memory(device)
        
        # Also log detailed memory breakdown for debugging
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        
        with self._lock:
            if config_key in self._config_memory:
                recorded_memory = self._config_memory[config_key]
                diff_bytes = recorded_memory - free_memory  # positive = leak
                diff_mb = diff_bytes / (1024 * 1024)
                abs_diff_bytes = abs(diff_bytes)
                abs_diff_mb = abs_diff_bytes / (1024 * 1024)
                
                logger.info(
                    f"[GPUMemoryMonitor] [{migration_type}] Config {config_key}: "
                    f"current_free={free_memory / 1024**3:.3f} GB, "
                    f"recorded_free={recorded_memory / 1024**3:.3f} GB, "
                    f"diff={diff_mb:+.2f} MB, "
                    f"allocated={allocated / 1024**3:.3f} GB, "
                    f"reserved={reserved / 1024**3:.3f} GB, "
                    f"cached_unused={(reserved - allocated) / 1024**3:.3f} GB"
                )
                
                if abs_diff_bytes > self.threshold_bytes:
                    leak_or_gain = "LEAK" if diff_bytes > 0 else "EXTRA_FREE"
                    warn_msg = (
                        f"[GPUMemoryMonitor] MEMORY {leak_or_gain} DETECTED after {migration_type} migration! "
                        f"Config: {config_key}, "
                        f"Expected free: {recorded_memory / 1024**3:.3f} GB, "
                        f"Actual free: {free_memory / 1024**3:.3f} GB, "
                        f"Difference: {abs_diff_mb:.2f} MB (threshold: {self.threshold_bytes / (1024*1024):.2f} MB), "
                        f"allocated={allocated / 1024**3:.3f} GB, "
                        f"reserved={reserved / 1024**3:.3f} GB"
                    )
                    logger.error(warn_msg)
                    # WARNING ONLY - do not crash, allow experiment to continue for debugging
                    # raise RuntimeError(warn_msg)
            else:
                # First time seeing this config, record it
                self._config_memory[config_key] = free_memory
                logger.info(
                    f"[GPUMemoryMonitor] [{migration_type}] Recorded new config {config_key}: "
                    f"{free_memory / 1024**3:.3f} GB free, "
                    f"allocated={allocated / 1024**3:.3f} GB, "
                    f"reserved={reserved / 1024**3:.3f} GB"
                )
    
    def get_recorded_configs(self) -> Dict[str, int]:
        """Get a copy of all recorded configurations and their memory baselines."""
        with self._lock:
            return dict(self._config_memory)
    
    def clear(self) -> None:
        """Clear all recorded configurations."""
        with self._lock:
            self._config_memory.clear()
            logger.info("[GPUMemoryMonitor] Cleared all recorded configurations")

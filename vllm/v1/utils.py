# SPDX-License-Identifier: Apache-2.0

import os
import time
import weakref
from collections import defaultdict
from collections.abc import Sequence
from multiprocessing import Process, connection
from typing import (TYPE_CHECKING, Callable, Generic, Optional, TypeVar, Union,
                    overload)
from dataclasses import dataclass
from datetime import timedelta

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.utils import get_mp_context, kill_process_tree
from vllm.v1.executor.abstract import Executor
from vllm.dynamic_config import DynamicConfig

if TYPE_CHECKING:
    from vllm.attention.layer import Attention

logger = init_logger(__name__)

T = TypeVar("T")


# =========================
# Time utilities
# =========================

def human_readable_duration(seconds: float) -> str:
    """Return a concise human-readable duration string.

    Examples:
        0.532 -> "532ms"
        12.3  -> "12.3s"
        75.0  -> "1m 15.0s"
        3720  -> "1h 2m 0.0s"
    """
    try:
        if seconds < 0:
            # Guard against negative inputs; show absolute value with prefix
            return f"-{human_readable_duration(-seconds)}"
        if seconds < 1e-3:
            # microseconds
            return f"{seconds * 1e6:.0f}µs"
        if seconds < 1:
            # milliseconds
            return f"{seconds * 1e3:.0f}ms"

        # For >= 1 second, format as h m s with a decimal on seconds
        total_seconds = float(seconds)
        td = timedelta(seconds=total_seconds)
        # Extract hours, minutes, seconds
        total_sec_int = int(td.total_seconds())
        hours, rem = divmod(total_sec_int, 3600)
        minutes, secs_int = divmod(rem, 60)
        secs_rem = total_seconds - (hours * 3600 + minutes * 60)

        parts: list[str] = []
        if hours:
            parts.append(f"{hours}h")
        if minutes or hours:
            parts.append(f"{minutes}m")
        parts.append(f"{secs_rem:.1f}s")
        return " ".join(parts)
    except Exception:
        # Fallback to raw seconds if any unexpected error happens
        return f"{seconds:.3f}s"


def now_s() -> float:
    """High-resolution monotonic time in seconds (for elapsed measurements)."""
    return time.perf_counter()


def elapsed_s(start_s: float, end_s: Optional[float] = None) -> float:
    """Compute elapsed seconds from start to now or provided end."""
    return (end_s if end_s is not None else now_s()) - start_s


def elapsed_str(start_s: float, end_s: Optional[float] = None) -> str:
    """Human-readable elapsed time string."""
    return human_readable_duration(elapsed_s(start_s, end_s))


class DurationTimer:
    """Context manager + utility for timing code blocks.

    Usage:
        with DurationTimer() as t:
            ...
        logger.info("took %s", t.elapsed_str)
    """

    def __init__(self) -> None:
        self._start_s: Optional[float] = None
        self._end_s: Optional[float] = None

    def __enter__(self):
        self._start_s = now_s()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_s = now_s()

    @property
    def elapsed(self) -> float:
        return elapsed_s(self._start_s or now_s(), self._end_s)

    @property
    def elapsed_str(self) -> str:
        return human_readable_duration(self.elapsed)

# Shared dataclasses for memory assessment
@dataclass
class WorkerMemInfo:
    """Per-worker memory snapshot.

    layer_size: bytes of a single layer's weights
    free_mem: current free GPU memory reported by driver (bytes)
    kv_tensor_size: bytes of a single layer's KV cache tensor
    """
    layer_size: int
    free_mem: int
    kv_tensor_size: int


@dataclass
class AssessResult:
    """Assessment result for adding layers on a worker.

    enough_without_compact: free_mem >= required_with_margin
    can_fit_after_compact: free_mem + freed_estimate > required_with_margin
    required_mem: bytes needed by new layers' weights + their KV cache
                          (estimated) with safety margin
    free_mem: current free GPU memory
    freed_estimate: estimated bytes that could be freed by compacting KV (on existing layers)
    """
    enough_without_compact: bool
    can_fit_after_compact: bool
    required_mem: int
    free_mem: int
    freed_estimate: int


class ConstantList(Generic[T], Sequence):

    def __init__(self, x: list[T]) -> None:
        self._x = x

    def append(self, item):
        raise Exception("Cannot append to a constant list")

    def extend(self, item):
        raise Exception("Cannot extend a constant list")

    def insert(self, item):
        raise Exception("Cannot insert into a constant list")

    def pop(self, item):
        raise Exception("Cannot pop from a constant list")

    def remove(self, item):
        raise Exception("Cannot remove from a constant list")

    def clear(self):
        raise Exception("Cannot clear a constant list")

    def index(self,
              item: T,
              start: int = 0,
              stop: Optional[int] = None) -> int:
        return self._x.index(item, start,
                             stop if stop is not None else len(self._x))

    @overload
    def __getitem__(self, item: int) -> T:
        ...

    @overload
    def __getitem__(self, s: slice, /) -> list[T]:
        ...

    def __getitem__(self, item: Union[int, slice]) -> Union[T, list[T]]:
        return self._x[item]

    @overload
    def __setitem__(self, item: int, value: T):
        ...

    @overload
    def __setitem__(self, s: slice, value: T, /):
        ...

    def __setitem__(self, item: Union[int, slice], value: Union[T, list[T]]):
        raise Exception("Cannot set item in a constant list")

    def __delitem__(self, item):
        raise Exception("Cannot delete item from a constant list")

    def __iter__(self):
        return iter(self._x)

    def __contains__(self, item):
        return item in self._x

    def __len__(self):
        return len(self._x)

    def __repr__(self):
        return f"ConstantList({self._x})"


class CoreEngineProcManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of background processes used by the AsyncLLM and LLMEngine.
    """

    def __init__(
        self,
        target_fn: Callable,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        dynamic_config: DynamicConfig,
        on_head_node: bool,
        input_address: str,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        context = get_mp_context()
        common_kwargs = {
            "vllm_config": vllm_config,
            "on_head_node": on_head_node,
            "input_address": input_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "dynamic_config": dynamic_config,
        }

        self.processes: list[Process] = []
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index
            # Start EngineCore in background process.
            self.processes.append(
                context.Process(target=target_fn,
                                name=f"EngineCore_{global_index}",
                                kwargs=common_kwargs | {
                                    "dp_rank": global_index,
                                    "local_dp_rank": local_index,
                                }))

        self._finalizer = weakref.finalize(self, shutdown, self.processes,
                                           input_address)
        try:
            for proc in self.processes:
                proc.start()
        finally:
            # Kill other procs if not all are running.
            if self.finished_procs():
                self.close()

    def close(self):
        """Shutdown all procs."""
        self._finalizer()

    def join_first(self):
        """Wait for any process to exit."""
        connection.wait(proc.sentinel for proc in self.processes)

    def sentinels(self) -> list:
        return [proc.sentinel for proc in self.processes]

    def finished_procs(self) -> dict[str, int]:
        """Returns dict of proc name -> exit code for any finished procs."""
        return {
            proc.name: proc.exitcode
            for proc in self.processes if proc.exitcode is not None
        }


# Note(rob): shutdown function cannot be a bound method,
# else the gc cannot collect the objedecoupct.
def shutdown(procs: list[Process], input_address: str):
    # Shutdown the process.
    for proc in procs:
        if proc.is_alive():
            proc.terminate()

    # Allow 5 seconds for remaining procs to terminate.
    deadline = time.monotonic() + 5
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    for proc in procs:
        if proc.is_alive() and (pid := proc.pid) is not None:
            kill_process_tree(pid)

    # Remove zmq ipc socket files.
    if input_address.startswith("ipc://"):
        socket_file = input_address[len("ipc://"):]
        if os and os.path.exists(socket_file):
            os.remove(socket_file)


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, "Attention"],
    runner_kv_caches: list[torch.Tensor],
) -> None:
    """
    Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its 
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention 
        layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
    """
    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.
            raise NotImplementedError
        layer_name = layer_names[0]
        runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].kv_cache = [kv_cache]


def copy_slice(from_tensor: torch.Tensor, to_tensor: torch.Tensor,
               length: int) -> torch.Tensor:
    """
    Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.

    Returns the sliced target tensor.
    """
    return to_tensor[:length].copy_(from_tensor[:length], non_blocking=True)


def report_usage_stats(
        vllm_config,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT) -> None:
    """Report usage statistics if enabled."""

    if not is_usage_stats_enabled():
        return

    from vllm.model_executor.model_loader import get_architecture_class_name

    usage_message.report_usage(
        get_architecture_class_name(vllm_config.model_config),
        usage_context,
        extra_kvs={
            # Common configuration
            "dtype":
            str(vllm_config.model_config.dtype),
            "tensor_parallel_size":
            vllm_config.parallel_config.tensor_parallel_size,
            "block_size":
            vllm_config.cache_config.block_size,
            "gpu_memory_utilization":
            vllm_config.cache_config.gpu_memory_utilization,

            # Quantization
            "quantization":
            vllm_config.model_config.quantization,
            "kv_cache_dtype":
            str(vllm_config.cache_config.cache_dtype),

            # Feature flags
            "enable_lora":
            bool(vllm_config.lora_config),
            "enable_prompt_adapter":
            bool(vllm_config.prompt_adapter_config),
            "enable_prefix_caching":
            vllm_config.cache_config.enable_prefix_caching,
            "enforce_eager":
            vllm_config.model_config.enforce_eager,
            "disable_custom_all_reduce":
            vllm_config.parallel_config.disable_custom_all_reduce,
        })

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
from vllm.distributed.kv_transfer.kv_connector.dynamic_kv_synchronizer import DynamicKVSynchronizer
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.utils import get_mp_context, kill_process_tree
from vllm.v1.executor.abstract import Executor
from vllm.dynamic_config import DynamicConfig
# from vllm.v1.worker.dynamic_gpu_model_runner import DynamicGPUModelRunner

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
def human_readable_size(size: int) -> str:
    """Return a concise human-readable size string.
    """
    if size < 1024:
        return f"{size}B"
    if size < 1024 ** 2:
        return f"{size / 1024:.2f}KB"
    if size < 1024 ** 3:
        return f"{size / 1024 ** 2:.2f}MB"

    return f"{size / 1024 ** 3:.2f}GB"


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
    block_size: bytes of a single block of a KV cache
    free_mem: current free GPU memory reported by driver (bytes)
    kv_tensor_size: bytes of a single layer's KV cache tensor
    """
    layer_size: int
    kv_tensor_size: int
    free_mem: int
    total_gpu_memory: int


@dataclass
class LayerAddingAssessResult:
    """Assessment result for adding layers on a worker.

    enough_without_compact: free_mem >= required_with_margin
    can_fit_after_compact: free_mem + freed_estimate > required_with_margin
    required_mem: bytes needed by new layers' weights + their KV cache
                          (estimated) with safety margin
    max_blocks_per_layer: number of blocks after compacting KV, this will only be used when can_fit_after_compact is True and enough_without_compact is False
    """
    enough_without_compact: bool
    can_fit_after_compact: bool
    max_blocks_per_layer: int

@dataclass
class LayerDeletionAssessResult:
    """Assessment result for deleting layers on a worker.
    This result shows the maximum number of blocks one layer can have after adding certain number of weights
    """
    max_blocks_per_layer: int


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

def dynamic_bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, "Attention"],
    runner_kv_caches: list[torch.Tensor],
    kv_synchronizer: "DynamicKVSynchronizer",
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
        kv_synchronizer.kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].kv_cache = [kv_cache]

def dynamic_flexi_bind_kv_cache(
    key_cache: dict[str, list[torch.Tensor]],
    value_cache: dict[str, list[torch.Tensor]],
    key_dev_ptr: dict[str, int],
    value_dev_ptr: dict[str, int],
    forward_context: dict[str, "Attention"],
    kv_synchronizer: "DynamicKVSynchronizer",
    runner_key_caches: list[list[torch.Tensor]],
    runner_value_caches: list[list[torch.Tensor]],
    runner_key_dev_ptrs: list[int],
    runner_value_dev_ptrs: list[int],
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
        key_cache: The allocated key caches with layer names as keys.
        value_cache: The allocated value caches with layer names as keys.
        forward_context: The global forward context containing all Attention 
        layers with layer names as keys.
        runner_key_caches: The key_cache declared by ModelRunner.
        runner_value_caches: The value_cache declared by ModelRunner.
    """
    # Bind kv_caches to ModelRunner
    assert len(runner_key_caches) == 0
    assert len(runner_value_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in key_cache:
        index2name[extract_layer_index(layer_name)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.
            raise NotImplementedError
        layer_name = layer_names[0]
        runner_key_caches.append(key_cache[layer_name])
        runner_value_caches.append(value_cache[layer_name])
        runner_key_dev_ptrs.append(key_dev_ptr[layer_name])
        runner_value_dev_ptrs.append(value_dev_ptr[layer_name])
        kv_synchronizer.key_cache_list.append(key_cache[layer_name])
        kv_synchronizer.value_cache_list.append(value_cache[layer_name])
        kv_synchronizer.key_cache_ptrs.append(key_dev_ptr[layer_name])
        kv_synchronizer.value_cache_ptrs.append(value_dev_ptr[layer_name])
    
    # Bind kv_caches to forward context
    for layer_name, attn in forward_context.items():
        # NOTE: Use list because of v0 PP virtual engine.
        # assert isinstance(attn, FlexiAttention)
        attn.key_cache = key_cache[layer_name]
        attn.value_cache = value_cache[layer_name]
        attn.key_dev_ptr = key_dev_ptr[layer_name]
        attn.value_dev_ptr = value_dev_ptr[layer_name]


def dynamic_bind_single_kv_tensor(
        layer_index: int, 
        start_layer: int,
        end_layer: int,
        forward_context: dict[str, "Attention"],
        kv_synchronizer: "DynamicKVSynchronizer",
        runner,
        kv_tensor: torch.Tensor,
        ) -> None:
    """Bind a single layer's KV tensor to runner caches and forward context.

    - 更新本 runner 的 `self.kv_caches`
    - 将 forward context 中对应 Attention 的 `kv_cache[ve]` 指向该张量
    - 如有必要，补齐 kv_cache_config 的 layer_names，确保后续 attn_metadata 构建覆盖到该层

    线程安全：内部获取 forward_lock。
    """
    
    # Determine local index in runner kv cache list
    assert layer_index >= start_layer and layer_index < end_layer, f"Layer {layer_index} outside of current model range [{start_layer}, {end_layer}]"
    local_index = layer_index - start_layer
    assert local_index < len(runner.kv_caches), f"Local index {local_index} is out of range, kv_tensor length: {len(runner.kv_caches)}"
    logger.info(f"bind kv tensor for {layer_index}, local_index={local_index}, kv_tensor length: {len(runner.kv_caches)}")
    # Basic sanity: non-empty tensor
    assert isinstance(kv_tensor, torch.Tensor) and kv_tensor.numel() > 0, (
        f"Binding empty KV tensor for layer {layer_index}")
    runner.kv_caches[local_index] = kv_tensor
    kv_synchronizer.kv_caches[local_index] = kv_tensor
    # Bind to forward context
    layer_name: str = get_layer_name_for_index(layer_index, forward_context)
    if layer_name not in forward_context:
        raise KeyError(
            f"No attention layer named {layer_name} in forward_context.")

    attn_module = forward_context[layer_name]
    attn_module.kv_cache = [kv_tensor]
    group = runner.kv_cache_config.kv_cache_groups[0]
    # 3) 补齐 kv_cache_config 的 layer_names，保证后续 attn_metadata 覆盖
    if layer_name not in group.layer_names:
        # 按 layer_index 位置插入，保持有序
        insert_idx = len(group.layer_names)
        target_idx = extract_layer_index(layer_name)
        for i, name in enumerate(group.layer_names):
            if extract_layer_index(name) > target_idx:
                insert_idx = i
                break
        group.layer_names.insert(insert_idx, layer_name)

def dynamic_flexi_bind_single_kv_tensor(
    start_layer: int,
    end_layer: int,
    layer_index: int,
    slot_mapping: torch.Tensor,
    block_num: int,
    kv_tensor: torch.Tensor,
    forward_context: dict[str, "Attention"],
    kv_synchronizer: "DynamicKVSynchronizer",
    runner: "DynamicGPUModelRunner",
    device: torch.device,
    stream: Optional[torch.cuda.streams.Stream] = None) -> None:
    """Bind a single layer's KV tensor to runner caches and forward context.

    - 更新本 runner 的 `self.kv_caches`
    - 将 forward context 中对应 Attention 的 `kv_cache[ve]` 指向该张量
    - 如有必要，补齐 kv_cache_config 的 layer_names，确保后续 attn_metadata 构建覆盖到该层

    线程安全：内部获取 forward_lock。
    """
    with device:
        torch.cuda.set_device(device)
        # Determine local index in runner kv cache list
        assert layer_index >= start_layer and layer_index < end_layer, f"Layer {layer_index} outside of current model range [{start_layer}, {end_layer}]"
        local_index = layer_index - start_layer
        assert local_index < len(runner.key_caches), f"Local index {local_index} is out of range, key_cache length: {len(runner.key_caches)}"
        logger.info(f"bind single kv tensor for {layer_index}, slot_mapping:{slot_mapping}")
        key_cache_list, value_cache_list, key_cache_ptr, value_cache_ptr = runner.get_flexi_kv_cache_from_gathered_kv_tensor(slot_mapping,layer_index, kv_tensor, block_num, stream)
        runner.key_caches[local_index] = key_cache_list
        runner.value_caches[local_index] = value_cache_list
        runner.key_cache_ptrs[local_index] = key_cache_ptr
        runner.value_cache_ptrs[local_index] = value_cache_ptr
        kv_synchronizer.key_cache_list[local_index] = key_cache_list
        kv_synchronizer.value_cache_list[local_index] = value_cache_list
        kv_synchronizer.key_cache_ptrs[local_index] = key_cache_ptr
        kv_synchronizer.value_cache_ptrs[local_index] = value_cache_ptr

        # Bind to forward context
        layer_name: str = get_layer_name_for_index(layer_index, forward_context)
        if layer_name not in forward_context:
            raise KeyError(
                f"No attention layer named {layer_name} in forward_context.")
        attn_module = forward_context[layer_name]
        # assert isinstance(attn_module, FlexiAttention), f"Attention module for layer {layer_name} is not FlexiAttention"
        attn_module.key_cache = key_cache_list
        attn_module.value_cache = value_cache_list
        attn_module.key_dev_ptr = key_cache_ptr
        attn_module.value_dev_ptr = value_cache_ptr

        group = runner.kv_cache_config.kv_cache_groups[0]
        # 3) 补齐 kv_cache_config 的 layer_names，保证后续 attn_metadata 覆盖
        if layer_name not in group.layer_names:
            # 按 layer_index 位置插入，保持有序
            insert_idx = len(group.layer_names)
            target_idx = extract_layer_index(layer_name)
            for i, name in enumerate(group.layer_names):
                if extract_layer_index(name) > target_idx:
                    insert_idx = i
                    break
            group.layer_names.insert(insert_idx, layer_name)

def dynamic_flexi_bind_single_kv_cache(
    start_layer: int,
    end_layer: int,
    layer_index: int,
    key_cache_list: list[torch.Tensor],
    value_cache_list: list[torch.Tensor],
    key_cache_ptr: int,
    value_cache_ptr: int,
    forward_context: dict[str, "Attention"],
    kv_synchronizer: "DynamicKVSynchronizer",
    runner: "DynamicGPUModelRunner"):
    """Bind a single layer's KV tensor to runner caches and forward context.

    - 更新本 runner 的 `self.kv_caches`
    - 将 forward context 中对应 Attention 的 `kv_cache[ve]` 指向该张量
    - 如有必要，补齐 kv_cache_config 的 layer_names，确保后续 attn_metadata 构建覆盖到该层

    线程安全：内部获取 forward_lock。
    """
    # Determine local index in runner kv cache list
    assert layer_index >= start_layer and layer_index < end_layer, f"Layer {layer_index} outside of current model range [{start_layer}, {end_layer}]"
    local_index = layer_index - start_layer
    assert local_index < len(runner.key_caches), f"Local index {local_index} is out of range, key_cache length: {len(runner.key_caches)}"

    runner.key_caches[local_index] = key_cache_list
    runner.value_caches[local_index] = value_cache_list
    runner.key_cache_ptrs[local_index] = key_cache_ptr
    runner.value_cache_ptrs[local_index] = value_cache_ptr
    kv_synchronizer.key_cache_list[local_index] = key_cache_list
    kv_synchronizer.value_cache_list[local_index] = value_cache_list
    kv_synchronizer.key_cache_ptrs[local_index] = key_cache_ptr
    kv_synchronizer.value_cache_ptrs[local_index] = value_cache_ptr

    # Bind to forward context
    layer_name: str = get_layer_name_for_index(layer_index, forward_context)
    if layer_name not in forward_context:
        raise KeyError(
            f"No attention layer named {layer_name} in forward_context.")
    attn_module = forward_context[layer_name]
    # assert isinstance(attn_module, FlexiAttention), f"Attention module for layer {layer_name} is not FlexiAttention"
    old_cache = attn_module.key_cache
    attn_module.key_cache = key_cache_list
    old_value_cache = attn_module.value_cache
    attn_module.value_cache = value_cache_list
    attn_module.key_dev_ptr = key_cache_ptr
    attn_module.value_dev_ptr = value_cache_ptr

    group = runner.kv_cache_config.kv_cache_groups[0]
    # 3) 补齐 kv_cache_config 的 layer_names，保证后续 attn_metadata 覆盖
    if layer_name not in group.layer_names:
        # 按 layer_index 位置插入，保持有序
        insert_idx = len(group.layer_names)
        target_idx = extract_layer_index(layer_name)
        for i, name in enumerate(group.layer_names):
            if extract_layer_index(name) > target_idx:
                insert_idx = i
                break
        group.layer_names.insert(insert_idx, layer_name)
    return old_cache, old_value_cache
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


def get_layer_name_for_index(layer_index: int, forward_context) -> str:
    """Return the layer name in forward_context for a given global layer index.

    Raises KeyError if not found or ambiguous.
    """
    candidates = [name for name in forward_context.keys()
                  if extract_layer_index(name) == layer_index]
    if not candidates:
        raise KeyError(f"No layer name found for index {layer_index} in forward_context, forward_context keys: {list(forward_context.keys())}")
    if len(candidates) > 1:
        raise KeyError(f"Multiple layer names found for index {layer_index}: {candidates}")
    return candidates[0]
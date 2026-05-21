# SPDX-License-Identifier: Apache-2.0
from pprint import pp
import re
from cv2 import resize
from numpy import isin
from torch import jagged

from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
from .core import EngineCore
import traceback
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
import math

import msgspec
import zmq
from bitarray import bitarray

from vllm.config import ParallelConfig, VllmConfig
from vllm import envs
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.executor.multiproc_worker_utils import _add_prefix
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.utils import make_zmq_socket, resolve_obj_by_qualname, zmq_socket_ctx
from vllm.v1.core.kv_cache_utils import (get_kv_cache_config,
                                         unify_kv_cache_configs)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.dynamic_kv_cache_manager import DynamicKVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as V1Scheduler
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.mm_input_cache import MirroredProcessingCache
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.dynamic_ray_distributed_executor import DynamicRayDistributedExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.core.sched.dynamic_scheduler import ChangeConfigurationType
from vllm.v1.core.sched.dynamic_scheduler import DynamicScheduler
from vllm.v1.core.sched.dynamic_scheduler import MigrationStatus
from vllm.dynamic_config import PPLayerConfigs
from vllm.version import __version__ as VLLM_VERSION
from .utils import get_new_layer_config_with_migration_action
from vllm.v1.utils import WorkerMemInfo, LayerAddingAssessResult, human_readable_duration, StopTimeMetrics
from dataclasses import dataclass
from vllm.dynamic_config import MigrationConfig
from copy import deepcopy


logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar('_R')  # Return type for collective_rpc


def _safe_queue_size(q: Any) -> int:
    if q is None:
        return 0
    try:
        return q.qsize()
    except (AttributeError, NotImplementedError):
        return 0

class DynamicEngineCore(EngineCore):

    def __init__(self,
                 vllm_config: VllmConfig,
                 executor_class: type[Executor],
                 log_stats: bool,
                 migration_config: MigrationConfig,
                 executor_fail_callback: Optional[Callable] = None
                 ):

        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins
        load_general_plugins()

        self.vllm_config = vllm_config
        logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(
                executor_fail_callback)

        # TODO: Load initial configurations properly
        self.migration_config = migration_config
        # Initialize cur_pp_layer_config from dynamic_config.pp_layer_partition
        # This is the actual initial configuration used by vLLM at startup
        partition_list_str = self.vllm_config.dynamic_config.pp_layer_partition
        if partition_list_str is None:
            raise ValueError("dynamic_config.pp_layer_partition must be set")
        partitions = [int(layer) for layer in partition_list_str.split(",")]
        pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        if len(partitions) > pp_size:
            raise ValueError(
                f"pp_layer_partition has {len(partitions)} entries, larger "
                f"than pipeline_parallel_size={pp_size}")
        partitions.extend([0] * (pp_size - len(partitions)))
        initial_layer_configs = []
        start_layer = 0
        for num_layers in partitions:
            if num_layers <= 0:
                initial_layer_configs.append((start_layer, start_layer - 1))
                continue
            end_layer = start_layer + num_layers - 1
            initial_layer_configs.append((start_layer, end_layer))
            start_layer = end_layer + 1
        self.cur_pp_layer_config = initial_layer_configs
        # Store initial config for reference
        self.initial_pp_layer_config = list(initial_layer_configs)
        self._placement_generation = 0
        self.cur_active_pp_ranks = self._active_ranks_for_config(
            self.cur_pp_layer_config)
        self.migration_in_process = False
        
        # Event to signal migration_thread to reset its request counter
        # This is triggered by set_pp_config between benchmark repetitions
        self._migration_reset_event = threading.Event()
        
        # Mutable migration configuration - can be updated by set_pp_config
        # These are used by migration_thread to determine when to trigger migrations
        self._migration_alternative_configs: Dict[int, Any] = {}
        self._migration_steps: set = set()
        self._migration_config_lock = threading.Lock()
        
        # Initialize from dynamic_config if available
        if (vllm_config.dynamic_config and
                (vllm_config.dynamic_config.is_migration
                 or vllm_config.dynamic_config.autoscaling_sequence)):
            if vllm_config.dynamic_config.autoscaling_sequence:
                sequence_configs: dict[int, list[Tuple[int, int]]] = {}
                sequence_steps: list[int] = []
                for idx, step_cfg in enumerate(
                        vllm_config.dynamic_config.autoscaling_sequence):
                    if "step" not in step_cfg:
                        raise ValueError(
                            "autoscaling_sequence entries must include step")
                    pp_config = step_cfg.get("pp_layer_config")
                    if pp_config is None:
                        pp_partition = step_cfg.get("pp_layer_partition")
                        if pp_partition is None:
                            raise ValueError(
                                "autoscaling_sequence entries must include "
                                "pp_layer_config or pp_layer_partition")
                        pp_config = self._parse_pp_layer_partition(pp_partition)
                    sequence_steps.append(int(step_cfg["step"]))
                    sequence_configs[idx] = self._normalize_pp_layer_config(
                        pp_config)
                self._migration_alternative_configs = sequence_configs
                self._migration_steps = set(sequence_steps)
            if vllm_config.dynamic_config.alternative_configs:
                # alternative_configs is a dict like {"pp_layer_configs": {"0": [[0,31],[32,63]], ...}}
                pp_layer_configs = vllm_config.dynamic_config.alternative_configs.get("pp_layer_configs", {})
                self._migration_alternative_configs = {
                    int(k): self._normalize_pp_layer_config(v)
                    for k, v in pp_layer_configs.items()
                }
            if vllm_config.dynamic_config.migration_steps:
                self._migration_steps = set(vllm_config.dynamic_config.migration_steps)
            if vllm_config.dynamic_config.autoscaling_sequence:
                sequence_configs = {}
                sequence_steps = []
                for idx, step_cfg in enumerate(
                        vllm_config.dynamic_config.autoscaling_sequence):
                    pp_config = step_cfg.get("pp_layer_config")
                    if pp_config is None:
                        pp_config = self._parse_pp_layer_partition(
                            step_cfg["pp_layer_partition"])
                    sequence_steps.append(int(step_cfg["step"]))
                    sequence_configs[idx] = self._normalize_pp_layer_config(
                        pp_config)
                self._migration_alternative_configs = sequence_configs
                self._migration_steps = set(sequence_steps)

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
            self._initialize_kv_caches(vllm_config)

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        if isinstance(vllm_config.scheduler_config.scheduler_cls, str):
            Scheduler = resolve_obj_by_qualname(
                vllm_config.scheduler_config.scheduler_cls)
        else:
            Scheduler = vllm_config.scheduler_config.scheduler_cls

        # This warning can be removed once the V1 Scheduler interface is
        # finalized and we can maintain support for scheduler classes that
        # implement it
        if Scheduler is not V1Scheduler:
            logger.warning(
                "Using configured V1 scheduler class %s. "
                "This scheduler interface is not public and "
                "compatibility may not be maintained.",
                vllm_config.scheduler_config.scheduler_cls)

        assert Scheduler is DynamicScheduler
        self.scheduler = DynamicScheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size
            > 1,
            log_stats=self.log_stats,
        )
        self._apply_active_pp_ranks_for_config(self.cur_pp_layer_config)

        # Setup MM Input Mapper.
        self.mm_input_cache_server = MirroredProcessingCache(
            vllm_config.model_config)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: Optional[queue.Queue[tuple[Future[ModelRunnerOutput],
                                                     SchedulerOutput]]] = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d",
                        self.batch_queue_size)
            self.batch_queue = queue.Queue(self.batch_queue_size)
        self.vllm_config = vllm_config
        assert isinstance(self.scheduler, DynamicScheduler)

        # Make sure dynamic_config is not in the kwargs, but pulling out from it.
        self.migration_status = MigrationStatus.NOT_MIGRATING
        self.engine_lock = threading.Lock()
        # Event to signal that an async migration has fully completed
        # (including do_resize on workers). set_pp_config waits on this.
        self._migration_done_event = threading.Event()
        self._migration_done_event.set()  # initially no migration in progress

        self.scheduler_kv_cache_config: Optional[KVCacheConfig]

    def _normalize_pp_layer_config(
        self,
        pp_layer_config: list[Tuple[int, int]],
    ) -> list[Tuple[int, int]]:
        pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        normalized = [(int(lo), int(hi)) for lo, hi in pp_layer_config]
        if len(normalized) > pp_size:
            raise ValueError(
                f"pp_layer_config has {len(normalized)} stages, larger than "
                f"pipeline_parallel_size={pp_size}")
        next_layer = normalized[-1][1] + 1 if normalized else 0
        normalized.extend([(next_layer, next_layer - 1)] *
                          (pp_size - len(normalized)))
        return normalized

    def _parse_pp_layer_partition(self, pp_layer_partition: str) -> list[Tuple[int, int]]:
        parts = [int(x.strip()) for x in pp_layer_partition.split(",")]
        ranges: list[Tuple[int, int]] = []
        start = 0
        for num_layers in parts:
            if num_layers <= 0:
                ranges.append((start, start - 1))
                continue
            end = start + num_layers - 1
            ranges.append((start, end))
            start = end + 1
        return ranges

    def _active_ranks_for_config(
        self,
        pp_layer_config: list[Tuple[int, int]],
    ) -> list[int]:
        return [rank for rank, (lo, hi) in enumerate(pp_layer_config)
                if hi >= lo]

    def _apply_active_pp_ranks_for_config(
        self,
        pp_layer_config: list[Tuple[int, int]],
    ) -> None:
        active_ranks = self._active_ranks_for_config(pp_layer_config)
        self.cur_active_pp_ranks = active_ranks
        self._placement_generation += 1
        if isinstance(self.model_executor, DynamicRayDistributedExecutor):
            self.model_executor.set_active_pp_ranks(active_ranks)
        logger.info(
            "Applied pipeline autoscaling placement generation=%s active_ranks=%s",
            self._placement_generation, active_ranks)
        self._refresh_batch_queue_for_active_pp_ranks()

    def _refresh_batch_queue_for_active_pp_ranks(self) -> None:
        if not hasattr(self, "batch_queue_size"):
            return
        new_size = self.model_executor.max_concurrent_batches
        pending_resize = getattr(self, "_pending_batch_queue_size", None)
        if new_size == self.batch_queue_size and pending_resize is None:
            return
        current_queue = self.batch_queue
        if current_queue is not None and not current_queue.empty():
            old_size = self.batch_queue_size
            if new_size < old_size:
                self.batch_queue_size = new_size
                logger.info(
                    "Applied smaller batch queue scheduling limit from %s to "
                    "%s; queue object resize deferred until %s queued "
                    "batches finish",
                    old_size, new_size, _safe_queue_size(current_queue))
            self._pending_batch_queue_size = new_size
            logger.info(
                "Deferring batch queue resize from %s to %s until %s "
                "queued batches finish",
                old_size, new_size, _safe_queue_size(current_queue))
            return

        old_size = self.batch_queue_size
        self.batch_queue_size = new_size
        self.batch_queue = (queue.Queue(new_size) if new_size > 1 else None)
        if hasattr(self, "step_fn"):
            self.step_fn = (self.step if self.batch_queue is None else
                            self.step_with_batch_queue)
        self._pending_batch_queue_size = None
        logger.info("Updated batch queue size from %s to %s for active PP ranks",
                    old_size, new_size)

    def _estimate_max_blocks_per_layer(self, gpu_total_memory: int, memory_after_adding_weight: int, num_layers_on_rank: int, block_size: int) -> int:
        safe_margin = (1 - self.vllm_config.cache_config.gpu_memory_utilization) * gpu_total_memory
        memory_after_adding_weight -=  math.ceil(safe_margin)
        logger.info(f"[memory access] debug ------- estimate max blocks per layer: {memory_after_adding_weight / 1024 ** 3:.2f} GB, num_layers_on_rank: {num_layers_on_rank}, safe_margin: {safe_margin / 1024 ** 3:.2f} GB, gpu_total_memory: {gpu_total_memory / 1024 ** 3:.2f} GB")
        return math.floor(memory_after_adding_weight / (num_layers_on_rank * block_size))


    def _assess_memory_for_layer_reconfiguration(
        self,
        rank: int,
        num_changed_layers: int,
        mem_info: WorkerMemInfo,
        current_pp_layer_config: list[Tuple[int, int]],
    ) -> LayerAddingAssessResult:
        """Assess whether the target rank has enough memory to add layers.

        Returns AssessResult with:
          - enough_without_compact: free_mem >= required_mem
          - can_fit_after_compact: free_mem + freed_estimate > required_mem
          - required_mem: bytes needed by new layers' weights + their KV
            cache (estimated) with safety margin
        注意，这里的所有memory都是对于整体的memory而言，而不是对于单个kv cache tensor而言的。
        """
        assert isinstance(self.scheduler, DynamicScheduler)
        assert self.scheduler_kv_cache_config is not None

        num_layers_on_rank = current_pp_layer_config[rank][1] - current_pp_layer_config[rank][0] + 1
        if num_layers_on_rank <= 0 and num_changed_layers <= 0:
            return LayerAddingAssessResult(True, True, sys.maxsize)
        # 计算在加入当前layer之后，kv cache的最大block数量
        total_layer_num = num_changed_layers + num_layers_on_rank
        if total_layer_num <= 0:
            return LayerAddingAssessResult(True, True, sys.maxsize)
        block_size = self.scheduler_kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
        max_blocks_per_layer = self._get_max_num_blocks(mem_info.total_gpu_memory, mem_info.layer_size, block_size, total_layer_num, mem_info.runtime_overhead_bytes)


        total_gpu_memory = int(mem_info.total_gpu_memory)
        total_usable_memory = total_gpu_memory * self.vllm_config.cache_config.gpu_memory_utilization
        weight_size_per_layer = mem_info.layer_size
        block_num =  self.scheduler.kv_cache_manager.block_pool.num_gpu_blocks
        # 计算当前GPU的可用内存
        current_used_memory = (weight_size_per_layer + block_size * block_num) * num_layers_on_rank
        free_gpu_memory = total_usable_memory - current_used_memory
        
        if num_changed_layers <= 0:
            return LayerAddingAssessResult(True, True, max_blocks_per_layer)

        # 如果当前GPU的可用内存大于需要添加的layer的内存，则直接可用（包含余量）
        if (weight_size_per_layer + block_size * block_num)  * num_changed_layers <= free_gpu_memory:
            logger.info(f"[memory access] assessed memory for adding layers: {num_changed_layers}, memory can directly fit, max_blocks_per_layer: {max_blocks_per_layer}")
            return LayerAddingAssessResult(True, True, max_blocks_per_layer)

        free_blocks = self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        used_blocks =  block_num - free_blocks

        # 或许需要compact, 此时我们计算在加入当前layer之后，kv cache的最大block数量
        if max_blocks_per_layer > used_blocks:
            logger.info(f"[memory access] assessed memory for adding layers: {num_changed_layers}, memory can fit after compact, max_blocks_per_layer: {max_blocks_per_layer}")
            return LayerAddingAssessResult(False, True, max_blocks_per_layer)

        logger.info(f"[memory access] assessed memory for adding layers: {num_changed_layers}, memory can not directly fit, max_blocks_per_layer: {max_blocks_per_layer}")
        return LayerAddingAssessResult(False, False, max_blocks_per_layer)

    def _get_max_num_blocks(self, total_gpu_memory: int, weight_size_per_layer: int, page_size: int, num_layers: int, runtime_overhead: int = 0) -> int:
        """Calculate max number of KV blocks per layer.
        
        Args:
            total_gpu_memory: Total GPU memory in bytes
            weight_size_per_layer: Weight size per layer in bytes
            page_size: KV cache page/block size in bytes
            num_layers: Number of layers
            runtime_overhead: Runtime overhead measured by profile_run (activations,
                CUDA context, NCCL buffers, etc.) in bytes
        
        Returns:
            Maximum number of blocks per layer
        
        Formula:
            total_kv_cache = total_gpu_memory * gpu_utilization - total_weight_size - runtime_overhead
            max_blocks_per_layer = floor(total_kv_cache / (num_layers * page_size))
        """
        total_usable_memory = total_gpu_memory * self.vllm_config.cache_config.gpu_memory_utilization
        total_weight_size = weight_size_per_layer * num_layers
        total_kv_cache = total_usable_memory - total_weight_size - runtime_overhead
        logger.info(f"[memory access] debug ------- get max num blocks: {total_kv_cache / 1024 ** 3:.2f} GB, num_layers: {num_layers}, block_size: {page_size}, total_gpu_memory: {total_gpu_memory / 1024 ** 3:.2f} GB, total_usable_memory: {total_usable_memory / 1024 ** 3:.2f} GB, weight_size_per_layer: {weight_size_per_layer / 1024 ** 3:.2f} GB, total_weight_size: {total_weight_size / 1024 ** 3:.2f} GB, runtime_overhead: {runtime_overhead / 1024 ** 3:.2f} GB")
        max_blocks_per_layer = math.floor(total_kv_cache / (num_layers * page_size))
        return max_blocks_per_layer

    def _assess_memory_for_delete(
        self,
        rank: int,
        deleting_layer_list: list[Tuple[int, int]],
        mem_info: WorkerMemInfo,
        current_pp_layer_config: list[Tuple[int, int]],
    ) -> int:
        """Assess whether the target rank has enough memory to add layers.
        Returns the maximum number of blocks one layer can have after deleting certain number of layers
        """
        logger.info(f"[memory access] rank {rank}: assessing memory for deleting layers: {deleting_layer_list}")
        assert isinstance(self.scheduler, DynamicScheduler)
        assert len(deleting_layer_list) != 0, f"deleting_layer_list is empty for rank {rank}"

        for layers in deleting_layer_list:
            assert layers[0] in current_pp_layer_config[rank] and layers[1] in current_pp_layer_config[rank], f"deleting layers {layers} is not in the current pp layer config {current_pp_layer_config[rank]}"
        total_gpu_memory = int(mem_info.total_gpu_memory)
        total_usable_memory = total_gpu_memory * (1 - self.vllm_config.cache_config.gpu_memory_utilization)
        weight_size_per_layer = mem_info.layer_size
        block_size = self.vllm_config.cache_config.block_size
        num_deleted_layers = sum((hi - lo + 1) for lo, hi in deleting_layer_list)
        num_layers_on_rank = current_pp_layer_config[rank][1] - current_pp_layer_config[rank][0] + 1

        # 计算在加入当前layer之后，kv cache的最大block数量
        total_layer_num = num_layers_on_rank - num_deleted_layers
        total_weight_size = weight_size_per_layer * total_layer_num
        total_kv_cache = total_usable_memory - total_weight_size
        max_blocks_per_layer = math.floor(total_kv_cache / (total_layer_num * block_size))
        return max_blocks_per_layer

    # def _get_max_blocks_per_layer_after_deletion(self, rank: int, deleting_layer_list: list[Tuple[int, int]], mem_info: WorkerMemInfo) -> int:
    #     """Assess after deleting certain number of layers, how many blocks one layer can have
    #     """

    #     assert isinstance(self.scheduler, DynamicScheduler)
    #     assert len(deleting_layer_list) != 0

    #     num_layers_on_rank = self.cur_pp_layer_config[rank][1] - self.cur_pp_layer_config[rank][0] + 1
    #     assert num_layers_on_rank > 0
    #     num_deleted_layers = sum((hi - lo + 1) for lo, hi in deleting_layer_list)
    #     layer_size = int(mem_info.layer_size)
    #     free_mem = int(mem_info.free_mem)
    #     kv_tensor_size = int(mem_info.kv_tensor_size)
    #     total_gpu_memory = int(mem_info.total_gpu_memory)

    #     # Calculate the new space for kv cache after the deletion of layers
    #     freed_memory_from_delete_weight = layer_size * num_deleted_layers 
    #     freed_memory_from_free_kv_cache = kv_tensor_size * num_layers_on_rank
    #     maximum_mem_for_kv_cache = free_mem + freed_memory_from_delete_weight + freed_memory_from_free_kv_cache

    #     logger.info(f"rank {rank}: layer_size{layer_size}, num_of_deleted_layers: {num_deleted_layers}, freed memory from delete weight: {freed_memory_from_delete_weight / 1024 ** 3:.2f} GB, freed memory from free kv cache: {freed_memory_from_free_kv_cache / 1024 ** 3:.2f} GB, maximum memory for kv cache: {maximum_mem_for_kv_cache / 1024 ** 3:.2f} GB, total gpu memory: {total_gpu_memory / 1024 ** 3:.2f} GB, num layers on rank: {num_layers_on_rank}, block size: {mem_info.block_size}")

    #     max_blocks_per_layer = self._estimate_max_blocks_per_layer(
    #         total_gpu_memory,
    #         maximum_mem_for_kv_cache, 
    #         num_layers_on_rank - num_deleted_layers, 
    #         mem_info.block_size)
    #     logger.info(f"rank {rank}: after deleting layers, maximum blocks per layer: {max_blocks_per_layer}")
    #     return int(max_blocks_per_layer)
    def _initialize_kv_caches(
            self, vllm_config: VllmConfig) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        # Here we are not using kv_cache_config to initialize the kv cache
        # We keep these logic just for compatibility with the parent class
        # Also we want to invoke the profile run within determine_available_memory()
        available_gpu_memory = self.model_executor.determine_available_memory()

        assert len(kv_cache_specs) == len(available_gpu_memory)
        # Get the kv cache tensor size
        kv_cache_configs = [
            get_kv_cache_config(vllm_config, kv_cache_spec_one_worker,
                                available_gpu_memory_one_worker)
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in
            zip(kv_cache_specs, available_gpu_memory)
        ]

        # Since we use a shared centralized controller, we need the
        # `kv_cache_config` to be consistent across all workers to make sure
        # all the memory operators can be applied to all workers.
        unify_kv_cache_configs(kv_cache_configs)

        # All workers have the same kv_cache_config except layer names, so use
        # an arbitrary one to initialize the scheduler.
        assert all([
            cfg.num_blocks == kv_cache_configs[0].num_blocks
            for cfg in kv_cache_configs
        ])
        num_cpu_blocks = 0
        self.scheduler_kv_cache_config = kv_cache_configs[0]

        # rather than initialize from kv config, we invoke our own logic 
        assert len(kv_cache_configs) >= 1 # Support PP>=1
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
        fixed_blocks = self.vllm_config.dynamic_config.fixed_num_gpu_blocks
        if fixed_blocks > 0:
            max_blocks_per_layer = fixed_blocks
            logger.info(f"[operation]: using fixed_num_gpu_blocks={fixed_blocks} for kv cache initialization")
        else:
            mem_infos = self.model_executor.get_workers_mem_info()
            max_blocks_per_layer = 6666666666
            for rank, mem_info in enumerate(mem_infos):
                num_layers_on_rank = self.cur_pp_layer_config[rank][1] - self.cur_pp_layer_config[rank][0] + 1
                if num_layers_on_rank <= 0:
                    continue
                max_blocks_per_layer = min(max_blocks_per_layer, self._get_max_num_blocks(mem_info.total_gpu_memory, mem_info.layer_size, self.scheduler_kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes, num_layers_on_rank, mem_info.runtime_overhead_bytes))
            if max_blocks_per_layer == 6666666666:
                raise RuntimeError("No active layers found for KV cache initialization")
            logger.info(f"[operation]: initialize kv cache with max blocks per layer: {max_blocks_per_layer}")
        
        # Update kv_cache_configs with the calculated max_blocks_per_layer
        # This is necessary because unify_kv_cache_configs doesn't consider layer count differences
        for cfg in kv_cache_configs:
            cfg.num_blocks = max_blocks_per_layer
        
        self.model_executor.dynamic_initialize_from_config(kv_cache_configs, max_blocks_per_layer)


        # Override num_gpus in the kv cache
        self.scheduler_kv_cache_config.num_blocks = max_blocks_per_layer
        self.vllm_config.cache_config.num_gpu_blocks = max_blocks_per_layer
        # self.model_executor.initialize_from_config(kv_cache_configs)


        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)
        return max_blocks_per_layer, num_cpu_blocks, self.scheduler_kv_cache_config
    def _reinitialize_kv_caches(
            self, vllm_config: VllmConfig) -> tuple[int, int, KVCacheConfig]:
        start = time.time()
        assert(isinstance(self.model_executor, DynamicRayDistributedExecutor))
        # Get all kv cache specs needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()
        # print(f"kv_cache_specs: {kv_cache_specs}")
        # Profiles the peak memory usage of the model to determine how much
        # memory can be allocated for kv cache.
        available_gpu_memory = self.model_executor.determine_available_memory()
        # print(f"available_gpu_memory: {available_gpu_memory}")
        assert len(kv_cache_specs) == len(available_gpu_memory)
        # Get the kv cache tensor size
        kv_cache_configs = [
            get_kv_cache_config(vllm_config, kv_cache_spec_one_worker,
                                available_gpu_memory_one_worker)
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in
            zip(kv_cache_specs, available_gpu_memory)
        ]
        # Since we use a shared centralized controller, we need the
        # `kv_cache_config` to be consistent across all workers to make sure
        # all the memory operators can be applied to all workers.
        unify_kv_cache_configs(kv_cache_configs)

        # All workers have the same kv_cache_config except layer names, so use
        # an arbitrary one to initialize the scheduler.
        assert all([
            cfg.num_blocks == kv_cache_configs[0].num_blocks
            for cfg in kv_cache_configs
        ])

        # All layers have the same kv cache size
        # We assume this so we manage all kv cache tensor all together 
        for cfg in kv_cache_configs:
            sizes = [tensor.size for tensor in cfg.tensors.values()]
            assert all(s == sizes[0] for s in sizes), "Inconsistent KV sizes within config"

        num_gpu_blocks = kv_cache_configs[0].num_blocks
        num_cpu_blocks = 0
        scheduler_kv_cache_config = kv_cache_configs[0]

        # Here we initialize the unified kv cache size and num blocks
        # We maintain these variables to dynamically manage the kv cache
        self.kv_cache_size = next(iter(kv_cache_configs[0].tensors.values())).size
        self.kv_cache_num_blocks = num_gpu_blocks

        # Reinitialize kv cache and warmup the execution
        self.model_executor.reinitialize_kv_cache(kv_cache_configs)

        # This is different from the scheduler's migration status
        # - Scheduler migration status is used to trace whether the old
        #   requests before migration are finished
        # - Engine migration status is used to trace whether the migration is done.
        self.migration_status = MigrationStatus.NOT_MIGRATING

        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def _drain_out_running_queue(
        self,
        finish_waiting: bool = False,
    ) -> list[EngineCoreOutputs]:
        assert isinstance(self.scheduler, DynamicScheduler)
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)

        engine_core_outputs = []
        if self.batch_queue is None:
            # PP=1: no batch queue, nothing to drain
            return engine_core_outputs

        is_ray_use_cpu = os.getenv("VLLM_USE_CPU_MODEL", "0") == "1"
        execute_func = (self.model_executor.execute_cpu_model
                        if is_ray_use_cpu
                        else self.model_executor.execute_model)

        if finish_waiting:
            self.scheduler.begin_sync_drain()

        try:
            while True:
                if not self.batch_queue.empty():
                    future, scheduler_output = self.batch_queue.get_nowait()
                    # Blocking until the first result is available.
                    model_output = future.result()
                    self.batch_queue.task_done()
                    engine_core_outputs.append(self.scheduler.update_from_output(
                        scheduler_output, model_output))
                    continue

                if not finish_waiting:
                    break

                _, cur_running = self.scheduler.running_controller.get_cur()
                cur_waiting = self.scheduler.waiting_controller.get_cur()
                if not cur_running and not cur_waiting:
                    break

                scheduler_output = self.scheduler.dynamic_schedule()
                if scheduler_output.total_num_scheduled_tokens == 0:
                    logger.warning(
                        "sync drain could not schedule remaining work: "
                        "running=%d waiting=%d",
                        len(cur_running), len(cur_waiting))
                    break

                future = execute_func(scheduler_output)
                model_output = future.result()
                engine_core_outputs.append(self.scheduler.update_from_output(
                    scheduler_output, model_output))
        finally:
            if finish_waiting:
                self.scheduler.finish_sync_drain()
        return engine_core_outputs

    def step_with_batch_queue(self) -> Optional[EngineCoreOutputs]:
        """
        Copied from vllm/vllm/v1/engine/core.py
        We will try to change the communcation channel of pipeline parallelism when doing kv cache migration. 
        Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
        time_start = time.time()
        with self.engine_lock:
            assert self.batch_queue is not None
            engine_core_outputs = None
            scheduler_output = None
            is_ray_use_cpu = os.getenv("VLLM_USE_CPU_MODEL", "0") == "1"
            execute_func = self.model_executor.execute_cpu_model if is_ray_use_cpu else self.model_executor.execute_model
                # if not self.migration_in_process \
                # else self.model_executor.execute_cpu_model
            # Try to schedule a new batch if the batch queue is not full, but
            # the scheduler may return an empty batch if all requests are scheduled.
            # Note that this is not blocking.
            isinstance(self.scheduler, DynamicScheduler)
            if _safe_queue_size(self.batch_queue) < self.batch_queue_size:
                schedule_start = time.time()
                scheduler_output = self.scheduler.dynamic_schedule()
                schedule_time = time.time() - schedule_start
                assert isinstance(scheduler_output, DynamicSchedulerOutput)
                # if scheduler_output.total_migration_tokens > 0:
                    # logger.info(f"[forward]: scheduled a total {scheduler_output.total_migration_tokens} tokens, total_num_scheduled_tokens: {scheduler_output.total_num_scheduled_tokens}, is_sync_after_migration: {scheduler_output.is_sync_after_migration}")
                if scheduler_output.total_num_scheduled_tokens > 0:
                    exec_start = time.time()
                    future = execute_func(scheduler_output)
                    exec_submit_time = time.time() - exec_start
                    logger.info(f"Scheduled tokens: {scheduler_output.total_num_scheduled_tokens}")
                    self.batch_queue.put_nowait(
                        (future, scheduler_output))  # type: ignore

            scheduled_batch = (scheduler_output is not None
                               and scheduler_output.total_num_scheduled_tokens > 0)
            # If no more requests can be scheduled and the job queue is not empty,
            # block until the first batch in the job queue is finished.
            # TODO(comaniac): Ideally we should peek the first batch in the
            # job queue to check if it's finished before scheduling a new batch,
            # but peeking the first element in a queue is not thread-safe,
            # so we need more work.
            if not scheduled_batch and not self.batch_queue.empty():
                future, scheduler_output = self.batch_queue.get_nowait()
                # Blocking until the first result is available.
                result_start = time.time()
                model_output = future.result()
                result_time = time.time() - result_start
                self.batch_queue.task_done()
                update_start = time.time()
                engine_core_outputs = self.scheduler.update_from_output(
                    scheduler_output, model_output)
                update_time = time.time() - update_start
                if (getattr(self, "_pending_batch_queue_size", None)
                        is not None and self.batch_queue.empty()):
                    self._refresh_batch_queue_for_active_pp_ranks()
            logger.info(f"[forward]: step with batch queue in {time.time() - time_start:.2f} seconds")

            return engine_core_outputs

    def step(self) -> EngineCoreOutputs:
        """Single-batch step for PP=1 (no batch queue).
        
        This is the non-pipelined path used when batch_queue is None
        (i.e., max_concurrent_batches == 1, which happens with PP=1).
        """
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
        with self.engine_lock:
            if not self.scheduler.has_requests():
                return EngineCoreOutputs(
                    outputs=[],
                    scheduler_stats=self.scheduler.make_stats(),
                )
            assert isinstance(self.scheduler, DynamicScheduler)
            scheduler_output = self.scheduler.dynamic_schedule()
            assert isinstance(scheduler_output, DynamicSchedulerOutput)
            
            is_ray_use_cpu = os.getenv("VLLM_USE_CPU_MODEL", "0") == "1"
            execute_func = (self.model_executor.execute_cpu_model
                            if is_ray_use_cpu
                            else self.model_executor.execute_model)
            # With PP=1, execute_model returns ModelRunnerOutput directly
            # (not a future), since max_concurrent_batches==1.
            model_output = execute_func(scheduler_output)
            if hasattr(model_output, 'result'):
                model_output = model_output.result()
            engine_core_outputs = self.scheduler.update_from_output(
                scheduler_output, model_output)
            return engine_core_outputs

    def change_model_configuration_by_reinitialize_kv_cache(self, pp_layer_config: list[Tuple[int, int]]) -> list[EngineCoreOutputs]:
        """
        This is a very naive implementation, we use it as a baseline
        Change the model configuration by reinitializing the kv cache
        In this implementation, we drain out the running queue, preempt all requests, release the kv cache, add the layers, remove the layers, and reinitialize the kv cache
        """
        pp_layer_config = self._normalize_pp_layer_config(pp_layer_config)
        with self.engine_lock:
            logger.info(f"Change configuration from {self.cur_pp_layer_config} to {pp_layer_config}")
            start_time = time.time()
            # First, we need to wait for all batch to be finished
            assert self.batch_queue is not None
            assert isinstance(self.scheduler, DynamicScheduler)
            assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
            engine_core_outputs = self._drain_out_running_queue()
            drain_out_time = time.time()
            self.scheduler.preempt_all_requests()
            preempt_time = time.time()
            # Release the kv cache
            self.model_executor.release_kv_cache()
            release_kv_cache_time = time.time()

            # Adding the layers 
            for rank, layers in enumerate(pp_layer_config):
                start_layer, end_layer = self.cur_pp_layer_config[rank][0], self.cur_pp_layer_config[rank][1]
                adding_layer_list = []
                if layers[0] < start_layer:
                    adding_layer_list.append((layers[0], start_layer - 1))
                elif layers[1] > end_layer:
                    adding_layer_list.append((end_layer + 1, layers[1]))
                if len(adding_layer_list) == 0:
                    continue
                self.model_executor.add_layers(rank, adding_layer_list)

            # remove the layers 
            for rank, layers in enumerate(pp_layer_config):
                start_layer, end_layer = self.cur_pp_layer_config[rank][0], self.cur_pp_layer_config[rank][1]
                deleting_layer_list = []
                if layers[0] > start_layer:
                    deleting_layer_list.append((start_layer, layers[0] - 1))
                if layers[1] < end_layer:
                    deleting_layer_list.append((layers[1] + 1, end_layer))
                if len(deleting_layer_list) == 0:
                    continue
                self.model_executor.remove_layers(rank, deleting_layer_list)
            
            self.scheduler.update_layer_config(pp_layer_config)
            weight_migration_time = time.time()

            # Reinitialize the kv cache
            _, _, kv_cache_config = self._reinitialize_kv_caches(self.vllm_config)
            reinitialize_kv_cache_time = time.time()
            self.scheduler.re_initialize_kv_cache_manager(kv_cache_config)
            end_time = time.time()

            self.cur_pp_layer_config = pp_layer_config
            self._apply_active_pp_ranks_for_config(pp_layer_config)
            from datetime import timedelta

            def format_duration(seconds):
                return str(timedelta(seconds=seconds))

            logger.info(f"drain_out_time: {format_duration(drain_out_time - start_time)}")
            logger.info(f"preempt_time: {format_duration(preempt_time - drain_out_time)}")
            logger.info(f"release_kv_cache_time: {format_duration(release_kv_cache_time - preempt_time)}")
            logger.info(f"weight_migration_time: {format_duration(weight_migration_time - release_kv_cache_time)}")
            logger.info(f"reinitialize_kv_cache_time: {format_duration(reinitialize_kv_cache_time - weight_migration_time)}")
            logger.info(f"end_time: {format_duration(end_time - reinitialize_kv_cache_time)}")

            return engine_core_outputs

    def _get_weight_loading_mode(self) -> str:
        mode = self.vllm_config.dynamic_config.weight_loading_mode
        if mode not in ("async", "sync"):
            raise ValueError(
                f"Invalid weight_loading_mode: {mode}. Must be 'async' or 'sync'.")
        return mode

    def _dispatch_weight_loading(
        self,
        adding_per_rank: dict[int, list[Tuple[int, int]]],
        migration_kind: str,
    ) -> str:
        weight_loading_mode = self._get_weight_loading_mode()
        add_layers_fn = (self.model_executor.async_add_layers
                         if weight_loading_mode == "async"
                         else self.model_executor.sync_add_layers)
        for r, add_list in adding_per_rank.items():
            total_layers = sum(hi - lo + 1 for lo, hi in add_list)
            logger.info(
                f"[{migration_kind}] rank {r}: {weight_loading_mode} adding layers {add_list} "
                f"(total_layers={total_layers})")
            add_layers_fn(r, add_list)
        return weight_loading_mode

    def change_model_configuration_by_kv_transfer_async(self, pp_layer_config: list[Tuple[int, int]]):
        """
        Our fancy implementation of model configuration change
        """
        pp_layer_config = self._normalize_pp_layer_config(pp_layer_config)
        logger.info(f"Start migrating to new configuration {pp_layer_config}")
        self.migration_status = MigrationStatus.MIGRATING
        self._migration_done_event.clear()  # signal that migration is in progress
        assert isinstance(self.scheduler, DynamicScheduler)
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        fixed_blocks = self.vllm_config.dynamic_config.fixed_num_gpu_blocks
        enable_kv_resize = self.vllm_config.dynamic_config.enable_kv_resize
        # allow_resize is True only when both conditions are met:
        # 1. fixed_num_gpu_blocks <= 0 (not fixed)
        # 2. enable_kv_resize is True (resize allowed during migration)
        allow_resize = (fixed_blocks <= 0) and enable_kv_resize
        if not allow_resize:
            if not enable_kv_resize:
                logger.info(f"[memory access] enable_kv_resize=False, resize disabled for async migration")
            else:
                logger.info(f"[memory access] fixed_num_gpu_blocks={fixed_blocks}, resize disabled for async migration")
        # 若任一 rank 需要 compact，则在持有引擎锁时再次校验一次可用显存，
        # 仍不足时再统一压缩 KV cache，最后再进行 add_layers
        # 先获取一次内存快照
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
        mem_infos = self.model_executor.get_workers_mem_info()

        # Adding/removing
        need_compact = False
        adding_per_rank: dict[int, list[Tuple[int, int]]] = {}
        maximum_kv_block_num_after_compact: list[int] = []

        # 记录当前gpu上的layer configuration
        # 除了migration前的config和migration后的config之外，在migration的过程中
        # 还有可能出现一些中间状态的migration config
        tmp_pp_layer_config = deepcopy(self.cur_pp_layer_config)
        bitmap = bitarray()

        # When shrinking, do not deactivate ranks before existing batches have
        # naturally completed. Each scheduled batch carries its own pp config,
        # so new batches can use the target placement while old batches keep
        # routing through the old ranks.
        target_active_ranks = self._active_ranks_for_config(pp_layer_config)
        deactivating_ranks = sorted(set(self.cur_active_pp_ranks) - set(target_active_ranks))
        if deactivating_ranks:
            logger.info(
                "[autoscaling] shrink detected: ranks %s will be deactivated "
                "for newly scheduled batches after migration sync",
                deactivating_ranks)

        with self.engine_lock:
            time_start = time.time()
            for rank, layers in enumerate(pp_layer_config):
                # Skip ranks that are being deactivated (target start > end indicates inactive).
                # These ranks will have their layers removed later; no layers are added here.
                if layers[0] > layers[1]:
                    assess = self._assess_memory_for_layer_reconfiguration(rank, 0, mem_infos[rank], self.cur_pp_layer_config)
                    maximum_kv_block_num_after_compact.append(assess.max_blocks_per_layer)
                    continue
                start_layer, end_layer = tmp_pp_layer_config[rank][0], tmp_pp_layer_config[rank][1]
                adding_layer_list = []
                if start_layer > end_layer:
                    adding_layer_list.append((layers[0], layers[1]))
                    tmp_pp_layer_config[rank] = (layers[0], layers[1])
                elif layers[0] < start_layer:
                    adding_layer_list.append((layers[0], start_layer - 1))
                    tmp_pp_layer_config[rank] = (layers[0], tmp_pp_layer_config[rank][1])
                elif layers[1] > end_layer:
                    adding_layer_list.append((end_layer + 1, layers[1]))
                    tmp_pp_layer_config[rank] = (tmp_pp_layer_config[rank][0], layers[1])
                logger.info(f"rank {rank}:\n adding_layer_list: {adding_layer_list}")
                # 在 adding 之前，校验对应 GPU 是否有足够可用显存容纳"将要新增的层"的权重；
                # 如果不足，尝试评估：压缩（compact）KV cache 后释放的显存，是否足以腾挪出空间。
                # 同时，如果当前的GPU available memory已经足够，则需要评估是否需要resize kv cache
                adding_layer_num = sum(layers[1] - layers[0] + 1 for layers in adding_layer_list)
                assess = self._assess_memory_for_layer_reconfiguration(rank, adding_layer_num, mem_infos[rank], self.cur_pp_layer_config)
                maximum_kv_block_num_after_compact.append(assess.max_blocks_per_layer)
                if len(adding_layer_list) > 0:
                    adding_per_rank[rank] = adding_layer_list
                    if assess.enough_without_compact:
                        pass
                    elif assess.can_fit_after_compact:
                        need_compact = True
                    else:
                        raise RuntimeError(f"Rank {rank} lacks memory for adding layers even after KV compact estimate: max_blocks_per_layer: {assess.max_blocks_per_layer}, current_used_blocks: {self.scheduler.kv_cache_manager.block_pool.num_gpu_blocks - self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()}")
                        # logger.info(
                        #     "Rank %s lacks memory even after KV compact estimate: max_blocks_per_layer: %s, current_used_blocks: %s",
                        #     rank, assess.max_blocks_per_layer, self.scheduler.kv_cache_manager.block_pool.num_gpu_blocks - self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks())
                        # return []
            assert isinstance(self.scheduler, DynamicScheduler)
            assert isinstance(self.scheduler.kv_cache_manager, DynamicKVCacheManager)
            compacted_length = min(maximum_kv_block_num_after_compact)
            original_length = self.scheduler.kv_cache_manager.block_pool.num_gpu_blocks

            if need_compact and not allow_resize:
                raise RuntimeError(
                    "Migration requires KV cache compact/resize but resize is disabled "
                    f"(enable_kv_resize={enable_kv_resize}, fixed_num_gpu_blocks={fixed_blocks}). "
                    f"compacted_length={compacted_length}, original_length={original_length}")

            if allow_resize and need_compact:
                time_start_compact_kv = time.time()
                bitmap = self.scheduler.compact_kv_cache(compacted_length)
                logger.info(f"[memory access] compacted_length for scheduler in {human_readable_duration(time.time() - time_start_compact_kv)}, compacted to: {compacted_length}, maximum_kv_block_num_after_compact: {maximum_kv_block_num_after_compact}")

                if compacted_length < original_length:
                    self.scheduler.shrink_block_pool(compacted_length)
            engine_lock_time_ms = (time.time() - time_start) * 1000
            logger.info(f"[timeline]: engine locking time: {human_readable_duration(time.time() - time_start)}")
            logger.info(f"[STOP_TIME][async]: engine_lock_total={engine_lock_time_ms:.2f}ms")
        # with self.engine_lock:
        if allow_resize and need_compact:
            time_start_compact_kv = time.time()
            self._compact_kv_cache(compacted_length, bitmap)
            logger.info(f"[memory access] KV cache compacted to {compacted_length} blocks before adding layers")
            logger.info(f"[timeline]: compact kv cache take: {human_readable_duration(time.time() - time_start_compact_kv)}")

            logger.info(f"important: when compacting kv cache, the kv cache size needs to be resized before migration")

            # 需要resize kv cache来进行migration，这里的resize一定是缩小
            # Only shrink when compacted_length < original_length (see sync path comment).
            if compacted_length < original_length:
                self.model_executor.resize_kv_cache(compacted_length)
                logger.info(f"[timeline]: after resize kv cache, time taken: {human_readable_duration(time.time() - time_start)}")
                logger.info(f"[timeline]: after shrink block pool, time taken: {human_readable_duration(time.time() - time_start)}")
            
        # self._compact_kv_cache(1700)
        # self.model_executor.resize_kv_cache(1700)
        # self.scheduler.shrink_block_pool(1700)
        # 对所有需要新增层的 rank 执行 add_layers（异步 fire-and-forget）
        logger.info(f"adding_per_rank: {adding_per_rank}")
        if not adding_per_rank:
            # No layers need to be transferred — target config matches current config.
            # This can happen when set_pp_config didn't migrate back (e.g. already at target).
            logger.info("No layers to add — skipping async migration (no-op)")
            self.migration_status = MigrationStatus.NOT_MIGRATING
            self._migration_done_event.set()
            return

        weight_loading_mode = self._dispatch_weight_loading(
            adding_per_rank, migration_kind="async")
        time_kv_compact_end = time.time()

        logger.info(f"[timeline]: before start actual kv cache migration, time taken to dispatch {weight_loading_mode} weight loading, compact, resize kv cache: {human_readable_duration(time_kv_compact_end - time_start)}")

        # 在异步传输/扩容前，抓取 KV cache 的快照，记录各请求的已计算 token 与 block 映射
        # 用于后续传输完成后对齐新增 token，实现两 GPU 之间 KV 同步
        # Ensure no in-flight work before snapshotting KV state

        # 计算 src->dst 传输对
        # 辅助函数：根据当前分片配置找到包含区间 [lo, hi] 的所有源 rank
        # 在 shrink 场景下，一个 target range 可能跨越多个 source rank
        def find_src_ranks_for_range(lo: int, hi: int) -> list[tuple[int, list[int]]]:
            result: list[tuple[int, list[int]]] = []
            remaining_lo = lo
            for src_rank, (cur_lo, cur_hi) in enumerate(self.cur_pp_layer_config):
                if cur_lo > cur_hi:
                    continue  # skip inactive ranks
                if remaining_lo > hi:
                    break
                if remaining_lo <= cur_hi and hi >= cur_lo:
                    overlap_lo = max(remaining_lo, cur_lo)
                    overlap_hi = min(hi, cur_hi)
                    if overlap_lo <= overlap_hi:
                        result.append((src_rank, list(range(overlap_lo, overlap_hi + 1))))
                        remaining_lo = overlap_hi + 1
            if remaining_lo <= hi:
                raise AssertionError(
                    f"Could not cover full range [{lo}, {hi}] from current config {self.cur_pp_layer_config}. "
                    f"Remaining: [{remaining_lo}, {hi}]")
            return result

        # 聚合为以 src_rank 为键的 rank_to_layers_ids 映射
        # src_to_plan[src_rank] = { dst_rank: [layer_ids...] }
        src_to_plan: dict[int, dict[int, list[int]]] = {}
        for dst_rank, add_list in adding_per_rank.items():
            for lo, hi in add_list:
                for src_rank, layer_ids in find_src_ranks_for_range(lo, hi):
                    plan = src_to_plan.setdefault(src_rank, {})
                    dst_layer_ids = plan.setdefault(dst_rank, [])
                    dst_layer_ids.extend(layer_ids)

        sender_list = list(src_to_plan.keys())
        receiver_list = list(adding_per_rank.keys())
        # Only increment scheduler_output_version when compact_kv_cache actually
        # ran on workers (which unconditionally bumps their version counter).
        # compact_kv_cache runs iff `allow_resize and need_compact`.
        # Without need_compact the workers never compact and stay at version 0,
        # so the scheduler must not advance either.
        should_increase_version = allow_resize and need_compact
        slot_mapping = self.scheduler.start_migration(sender_list, receiver_list, should_increase_version)
        assert slot_mapping is not None if is_flexi else True
        self.model_executor.start_kv_cache_migration_async(pp_layer_config, src_to_plan, slot_mapping)

        time_kv_migration_end = time.time()
        logger.info(f"[timeline]: after start kv cache migration, time taken: {human_readable_duration(time_kv_migration_end - time_kv_compact_end)}")

        def sync_by_checking_buffer_status() -> bool:
            token_to_send_threshold = int(os.environ.get("VLLM_PATCH_ID_DIFF_THRESHOLD", 1024))
            should_sync = True
            assert isinstance(self.model_executor, DynamicRayDistributedExecutor) 
            buffer_status_list = self.model_executor.get_kv_buffer_status()
            logger.info(f"buffer_status_list: {buffer_status_list}")
            for src_rank, rank_to_layers_ids in src_to_plan.items():
                src_rank_buffer_status = buffer_status_list[src_rank]
                for dst_rank, _ in rank_to_layers_ids.items():
                    sender_token_number = src_rank_buffer_status.used_tokens[dst_rank]
                    if sender_token_number > token_to_send_threshold:
                        should_sync = False
                        break
                if not should_sync:
                    break
            return should_sync

        def sync_by_checking_leftover_tokens() -> bool:
            token_to_send_threshold = int(os.environ.get("VLLM_PATCH_ID_DIFF_THRESHOLD", 500))
            assert isinstance(self.model_executor, DynamicRayDistributedExecutor) 
            assert isinstance(self.scheduler, DynamicScheduler)
            receiver_list = list(adding_per_rank.keys())
            
            # If no receivers (no layers being added), we can sync immediately
            if not receiver_list:
                logger.info("No receivers in migration, can sync immediately")
                return True
            
            applied_token_list = self.model_executor.get_applied_token_num(receiver_list)
            logger.info(f"applied_token_list: {applied_token_list}")
            logger.info(f"num of tokens for migration: {self.scheduler.num_tokens_for_migration}")
            
            relevant_applied_tokens: list[int] = []
            world_size = self.vllm_config.parallel_config.pipeline_parallel_size
            for src_rank, rank_to_layers_ids in src_to_plan.items():
                for dst_rank in rank_to_layers_ids:
                    applied_tokens = applied_token_list[dst_rank]
                    if applied_tokens is None:
                        logger.info(
                            "Receiver rank %s did not report applied tokens yet",
                            dst_rank)
                        return False
                    peer_ranks = [
                        rank for rank in range(world_size)
                        if rank != dst_rank
                    ]
                    src_index = peer_ranks.index(src_rank)
                    relevant_applied_tokens.append(applied_tokens[src_index])

            if not relevant_applied_tokens:
                logger.info("No relevant applied tokens yet, waiting...")
                return False

            min_applied_token = min(relevant_applied_tokens)

            lag = [
                self.scheduler.num_tokens_for_migration - applied
                for applied in relevant_applied_tokens
            ]
            logger.info(f"lag between sent and applied tokens: {lag}")
            return min_applied_token > 0 and max(lag) < token_to_send_threshold

        '''
        Following checks the patch sending process and synchronize the kv cache when all tokens in the sender to be sent are less than the threshold
        '''
        assert isinstance(self.scheduler, DynamicScheduler)
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
        resized_block_num = 0
        while True:
            time.sleep(0.3)

            should_sync = sync_by_checking_leftover_tokens()

            final_pp_layer_config = deepcopy(tmp_pp_layer_config)
            if should_sync:
                deleting_layer_assesses: list[int] = []
                for rank, layers in enumerate(pp_layer_config):
                    # 计算对于每一个rank而言，需要删除哪一些layers
                    start_layer, end_layer = final_pp_layer_config[rank][0], final_pp_layer_config[rank][1]
                    deleting_layer_list = []
                    if layers[0] > layers[1]:
                        if start_layer <= end_layer:
                            deleting_layer_list.append((start_layer, end_layer))
                        final_pp_layer_config[rank] = (layers[0], layers[1])
                    else:
                        if layers[0] > start_layer:
                            deleting_layer_list.append((start_layer, layers[0] - 1))
                            final_pp_layer_config[rank] = (layers[0], final_pp_layer_config[rank][1])
                        if layers[1] < end_layer:
                            deleting_layer_list.append((layers[1] + 1, end_layer))
                            final_pp_layer_config[rank] = (final_pp_layer_config[rank][0], layers[1])
                    deleting_layer_num = -sum(layers[1] - layers[0] + 1 for layers in deleting_layer_list)
                    assess = self._assess_memory_for_layer_reconfiguration(rank, deleting_layer_num, mem_infos[rank], tmp_pp_layer_config)
                    deleting_layer_assesses.append(assess.max_blocks_per_layer)

                for rank, layers in enumerate(final_pp_layer_config):
                    assert layers[0] == pp_layer_config[rank][0] and layers[1] == pp_layer_config[rank][1]

                resized_block_num = min(deleting_layer_assesses)
                if not allow_resize:
                    # When fixed_num_gpu_blocks is set, keep the current block count
                    resized_block_num = self.scheduler.kv_cache_manager.num_gpu_blocks
                    logger.info(f"[memory access] fixed_num_gpu_blocks={fixed_blocks}, keeping current block count: {resized_block_num}")
                else:
                    logger.info(f"[memory access] all tokens to be sent is less than the threshold, start to synchronize the kv cache, resized_block_num: {resized_block_num}")
                    if resized_block_num != self.scheduler.kv_cache_manager.num_gpu_blocks:
                        assert resized_block_num > self.scheduler.kv_cache_manager.num_gpu_blocks, f"resized_block_num: {resized_block_num} is less than the current kv cache size: {self.scheduler.kv_cache_manager.num_gpu_blocks}"
                        logger.info(f"[memory access] start to synchronize the kv cache after resizing from {self.scheduler.kv_cache_manager.num_gpu_blocks} to {resized_block_num} blocks")

                with self.engine_lock:
                    self.scheduler.async_change_configuration(
                        pp_layer_config, resized_block_num)
                    self.cur_pp_layer_config = pp_layer_config
                    if not self.vllm_config.dynamic_config.pipeline_autoscaling_enabled:
                        self._apply_active_pp_ranks_for_config(pp_layer_config)
                    else:
                        logger.info(
                            "[autoscaling async] queued PP config switch via "
                            "scheduler output; active ranks remain batch-local")
                break

        assert resized_block_num != 0
        if resized_block_num == self.scheduler.kv_cache_manager.num_gpu_blocks:
            logger.info(f"no need to resize kv cache during migration, directly synchronize the kv cache, sleep for 4 seconds")
            time.sleep(4)
            logger.info(f"[timeline]: migration process time taken: {human_readable_duration(time.time() - time_start)}")
            self.migration_status = MigrationStatus.NOT_MIGRATING
            self._migration_done_event.set()
            return
        time_start_checking_resizing_done = time.time()
        # Checking if the resizing is done and needed to extend the block pool
        while True:
            time.sleep(0.3)
            resizing_done = self.model_executor.get_is_kv_resizing_done()
            logger.info(f"resizing_done status: {resizing_done}")
            if all(resizing_done):
                with self.scheduler.lock:
                    self.scheduler.extend_block_pool(resized_block_num)
                break

        self.migration_status = MigrationStatus.NOT_MIGRATING
        self._migration_done_event.set()
        logger.info(f"[timeline]: after check resizing done process, time taken: {human_readable_duration(time.time() - time_start_checking_resizing_done)}")
        logger.info(f"[timeline]: migration process time taken: {human_readable_duration(time.time() - time_start)}")

    def change_model_configuration_by_kv_transfer_async_fast(
        self,
        pp_layer_config: list[Tuple[int, int]],
    ) -> list[EngineCoreOutputs]:
        """
        Async-fast migration: combines async weight loading with fast KV transfer.
        
        Key differences from async:
        - Weight loading is still async (same as async)
        - When sending KV tensor: drain_running_queue first
        - Send all KV tensors at once (no patch phase)
        - After sending, directly restart inference
        
        This is faster than async because it avoids the continuous kv_patch sending phase,
        but requires draining the running queue similar to sync migration.
        """
        pp_layer_config = self._normalize_pp_layer_config(pp_layer_config)
        logger.info(f"[async_fast] Start migrating to new configuration {pp_layer_config}")
        self.migration_status = MigrationStatus.MIGRATING
        self._migration_done_event.clear()
        assert isinstance(self.scheduler, DynamicScheduler)
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        fixed_blocks = self.vllm_config.dynamic_config.fixed_num_gpu_blocks
        enable_kv_resize = self.vllm_config.dynamic_config.enable_kv_resize
        allow_resize = (fixed_blocks <= 0) and enable_kv_resize
        if not allow_resize:
            if not enable_kv_resize:
                logger.info(f"[memory access] enable_kv_resize=False, resize disabled for async_fast migration")
            else:
                logger.info(f"[memory access] fixed_num_gpu_blocks={fixed_blocks}, resize disabled for async_fast migration")
        
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
        mem_infos = self.model_executor.get_workers_mem_info()

        need_compact = False
        adding_per_rank: dict[int, list[Tuple[int, int]]] = {}
        maximum_kv_block_num_after_compact: list[int] = []
        tmp_pp_layer_config = deepcopy(self.cur_pp_layer_config)
        bitmap = bitarray()
        engine_core_outputs = []
        
        
        with self.engine_lock:
            time_start = time.time()
            # Phase 1: Assess memory and prepare layer lists
            for rank, layers in enumerate(pp_layer_config):
                # Skip ranks that are being deactivated (target start > end indicates inactive).
                if layers[0] > layers[1]:
                    assess = self._assess_memory_for_layer_reconfiguration(rank, 0, mem_infos[rank], self.cur_pp_layer_config)
                    maximum_kv_block_num_after_compact.append(assess.max_blocks_per_layer)
                    continue
                start_layer, end_layer = tmp_pp_layer_config[rank][0], tmp_pp_layer_config[rank][1]
                adding_layer_list = []
                if start_layer > end_layer:
                    adding_layer_list.append((layers[0], layers[1]))
                    tmp_pp_layer_config[rank] = (layers[0], layers[1])
                elif layers[0] < start_layer:
                    adding_layer_list.append((layers[0], start_layer - 1))
                    tmp_pp_layer_config[rank] = (layers[0], tmp_pp_layer_config[rank][1])
                elif layers[1] > end_layer:
                    adding_layer_list.append((end_layer + 1, layers[1]))
                    tmp_pp_layer_config[rank] = (tmp_pp_layer_config[rank][0], layers[1])
                logger.info(f"[async_fast] rank {rank}: adding_layer_list: {adding_layer_list}")

                adding_layer_num = sum(layers[1] - layers[0] + 1 for layers in adding_layer_list)
                assess = self._assess_memory_for_layer_reconfiguration(rank, adding_layer_num, mem_infos[rank], self.cur_pp_layer_config)
                maximum_kv_block_num_after_compact.append(assess.max_blocks_per_layer)
                if len(adding_layer_list) > 0:
                    adding_per_rank[rank] = adding_layer_list
                    if assess.enough_without_compact:
                        pass
                    elif assess.can_fit_after_compact:
                        need_compact = True
                    else:
                        raise RuntimeError(f"[async_fast] Rank {rank} lacks memory for adding layers even after KV compact estimate")

            assert isinstance(self.scheduler, DynamicScheduler)
            assert isinstance(self.scheduler.kv_cache_manager, DynamicKVCacheManager)
            compacted_length = min(maximum_kv_block_num_after_compact)
            original_length = self.scheduler.kv_cache_manager.block_pool.num_gpu_blocks

            if need_compact and not allow_resize:
                raise RuntimeError(
                    "[async_fast] Migration requires KV cache compact/resize but resize is disabled "
                    f"(enable_kv_resize={enable_kv_resize}, fixed_num_gpu_blocks={fixed_blocks}). "
                    f"compacted_length={compacted_length}, original_length={original_length}")

            if allow_resize and need_compact:
                time_start_compact_kv = time.time()
                bitmap = self.scheduler.compact_kv_cache(compacted_length)
                logger.info(f"[async_fast] compacted to: {compacted_length} in {human_readable_duration(time.time() - time_start_compact_kv)}")

            if allow_resize and compacted_length < original_length:
                self.scheduler.shrink_block_pool(compacted_length)
            engine_lock_1_time_ms = (time.time() - time_start) * 1000
            logger.info(f"[async_fast timeline]: engine locking time: {human_readable_duration(time.time() - time_start)}")
        
        # Compact KV cache outside of engine lock
        if allow_resize and need_compact:
            time_start_compact_kv = time.time()
            self._compact_kv_cache(compacted_length, bitmap)
            logger.info(f"[async_fast] KV cache compacted to {compacted_length} blocks in {human_readable_duration(time.time() - time_start_compact_kv)}")

        if allow_resize and compacted_length < original_length:
            self.model_executor.resize_kv_cache(compacted_length)
            logger.info(f"[async_fast timeline]: after resize kv cache: {human_readable_duration(time.time() - time_start)}")

        logger.info(f"[async_fast] adding_per_rank: {adding_per_rank}")
        if not adding_per_rank:
            logger.info("[async_fast] No layers to add - skipping migration (no-op)")
            self.migration_status = MigrationStatus.NOT_MIGRATING
            self._migration_done_event.set()
            return engine_core_outputs

        # Phase 2: Start weight loading with configured mode
        weight_loading_mode = self._dispatch_weight_loading(
            adding_per_rank, migration_kind="async_fast")

        logger.info(f"[async_fast timeline]: {weight_loading_mode} weight loading dispatched: {human_readable_duration(time.time() - time_start)}")

        # Phase 3: Compute src->dst transfer plan
        original_pp_layer_config = deepcopy(self.cur_pp_layer_config)

        def find_src_ranks_for_range(lo: int, hi: int) -> list[tuple[int, list[int]]]:
            result: list[tuple[int, list[int]]] = []
            remaining_lo = lo
            for src_rank, (cur_lo, cur_hi) in enumerate(original_pp_layer_config):
                if cur_lo > cur_hi:
                    continue
                if remaining_lo > hi:
                    break
                if remaining_lo <= cur_hi and hi >= cur_lo:
                    overlap_lo = max(remaining_lo, cur_lo)
                    overlap_hi = min(hi, cur_hi)
                    if overlap_lo <= overlap_hi:
                        result.append((src_rank, list(range(overlap_lo, overlap_hi + 1))))
                        remaining_lo = overlap_hi + 1
            if remaining_lo <= hi:
                raise AssertionError(
                    f"[async_fast] Could not cover full range [{lo}, {hi}] from original config {original_pp_layer_config}. "
                    f"Remaining: [{remaining_lo}, {hi}]")
            return result

        src_to_plan: dict[int, dict[int, list[int]]] = {}
        sending_plan_for_ranks: dict[int, dict[int, list[Tuple[int, int]]]] = {}
        for dst_rank, add_list in adding_per_rank.items():
            for lo, hi in add_list:
                for src_rank, layer_ids in find_src_ranks_for_range(lo, hi):
                    # For src_to_plan (layer_ids as list)
                    plan = src_to_plan.setdefault(src_rank, {})
                    dst_layer_ids = plan.setdefault(dst_rank, [])
                    dst_layer_ids.extend(layer_ids)
                    # For sending_plan_for_ranks (layer ranges as tuples)
                    plan_ranges = sending_plan_for_ranks.setdefault(src_rank, {})
                    layer_ranges = plan_ranges.setdefault(dst_rank, [])
                    layer_ranges.append((layer_ids[0], layer_ids[-1]))

        sender_list = list(src_to_plan.keys())
        receiver_list = list(adding_per_rank.keys())
        
        # Phase 4: Drain running queue first, then execute one-shot KV migration.
        # Do NOT wait for async add-layers; weight loading and migration can still overlap.
        with self.engine_lock:
            drain_start = time.time()
            outputs = self._drain_out_running_queue(finish_waiting=True)
            engine_core_outputs.extend(outputs)
            logger.info(f"[async_fast timeline]: drain_running_queue took {human_readable_duration(time.time() - drain_start)}")
            logger.info(f"[async_fast] drained {len(outputs)} outputs during migration")

            should_increase_version = allow_resize and need_compact
            slot_mapping = self.scheduler.start_migration(sender_list, receiver_list, should_increase_version)
            assert slot_mapping is not None if is_flexi else True

            self.cur_pp_layer_config = deepcopy(tmp_pp_layer_config)
            logger.info(f"[async_fast] updated cur_pp_layer_config to intermediate state {self.cur_pp_layer_config}")

            self.model_executor.start_kv_cache_migration_async_fast(
                pp_layer_config,
                sending_plan_for_ranks,
                adding_per_rank,
                slot_mapping,
            )
            engine_lock_2_time_ms = (time.time() - drain_start) * 1000
            drain_time_ms = (time.time() - drain_start) * 1000  # Approximately includes drain + kv_transfer
            logger.info(f"[async_fast timeline]: one-shot KV transfer completed in {human_readable_duration(time.time() - drain_start)}")
            logger.info(f"[STOP_TIME][async_fast]: engine_lock_total={engine_lock_1_time_ms + engine_lock_2_time_ms:.2f}ms (lock1={engine_lock_1_time_ms:.2f}ms, lock2={engine_lock_2_time_ms:.2f}ms) drain+kv_transfer={drain_time_ms:.2f}ms")

        time_kv_migration_end = time.time()
        logger.info(f"[async_fast timeline]: migration initiated, time taken: {human_readable_duration(time_kv_migration_end - time_start)}")

        # Phase 5: finalize scheduler/config after one-shot transfer is completed.
        assert isinstance(self.scheduler, DynamicScheduler)
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
        final_pp_layer_config = deepcopy(tmp_pp_layer_config)
        deleting_layer_assesses: list[int] = []
        for rank, layers in enumerate(pp_layer_config):
            start_layer, end_layer = final_pp_layer_config[rank][0], final_pp_layer_config[rank][1]
            deleting_layer_list = []
            if layers[0] > layers[1]:
                if start_layer <= end_layer:
                    deleting_layer_list.append((start_layer, end_layer))
                final_pp_layer_config[rank] = (layers[0], layers[1])
            else:
                if layers[0] > start_layer:
                    deleting_layer_list.append((start_layer, layers[0] - 1))
                    final_pp_layer_config[rank] = (layers[0], final_pp_layer_config[rank][1])
                if layers[1] < end_layer:
                    deleting_layer_list.append((layers[1] + 1, end_layer))
                    final_pp_layer_config[rank] = (final_pp_layer_config[rank][0], layers[1])
            deleting_layer_num = -sum(layers[1] - layers[0] + 1 for layers in deleting_layer_list)
            assess = self._assess_memory_for_layer_reconfiguration(rank, deleting_layer_num, mem_infos[rank], tmp_pp_layer_config)
            deleting_layer_assesses.append(assess.max_blocks_per_layer)

        for rank, layers in enumerate(final_pp_layer_config):
            assert layers[0] == pp_layer_config[rank][0] and layers[1] == pp_layer_config[rank][1]

        resized_block_num = min(deleting_layer_assesses)
        if not allow_resize:
            resized_block_num = self.scheduler.kv_cache_manager.num_gpu_blocks
            logger.info(f"[async_fast] fixed_num_gpu_blocks={fixed_blocks}, keeping current block count: {resized_block_num}")

        with self.scheduler.lock:
            self.scheduler.async_change_configuration(pp_layer_config,
                                                      resized_block_num)

        self.cur_pp_layer_config = pp_layer_config
        if not self.vllm_config.dynamic_config.pipeline_autoscaling_enabled:
            self._apply_active_pp_ranks_for_config(pp_layer_config)
        else:
            logger.info(
                "[autoscaling async_fast] queued PP config switch via "
                "scheduler output; active ranks remain batch-local")
        logger.info(f"[async_fast] config synced to {pp_layer_config}")

        assert resized_block_num != 0
        if resized_block_num == self.scheduler.kv_cache_manager.num_gpu_blocks:
            logger.info(f"[async_fast] no need to resize kv cache, migration setup complete in {human_readable_duration(time.time() - time_start)}")
            self.migration_status = MigrationStatus.NOT_MIGRATING
            self._migration_done_event.set()
            return engine_core_outputs
        
        time_start_checking_resizing_done = time.time()
        # Check if KV resizing is done
        while True:
            time.sleep(0.3)
            resizing_done = self.model_executor.get_is_kv_resizing_done()
            logger.info(f"resizing_done status: {resizing_done}")
            if all(resizing_done):
                with self.scheduler.lock:
                    self.scheduler.extend_block_pool(resized_block_num)
                break

        self.migration_status = MigrationStatus.NOT_MIGRATING
        self._migration_done_event.set()
        logger.info(f"[timeline]: check resizing done in {human_readable_duration(time.time() - time_start_checking_resizing_done)}")
        logger.info(f"[timeline]: migration process time taken: {human_readable_duration(time.time() - time_start)}")
        return engine_core_outputs

    def change_model_configuration_by_kv_transfer_sync(
        self,
        pp_layer_config: list[Tuple[int, int]],
        target_kv_blocks: Optional[int] = None,
        log_stop_time: bool = True,
    ) -> list[EngineCoreOutputs]:
        """
        Our fancy implementation of model synchronized configuration change.

        Args:
            pp_layer_config: Target layer configuration per rank.
            target_kv_blocks: If provided and > 0, the final KV block count
                after migration.  None means use the calculated maximum.
            log_stop_time: If True, emit [STOP_TIME] logs. Set to False when
                called from set_pp_config to avoid polluting migration metrics.
        """
        pp_layer_config = self._normalize_pp_layer_config(pp_layer_config)
        logger.info(f"Start migrating to new configuration {pp_layer_config}")
        assert isinstance(self.scheduler, DynamicScheduler)
        time_start = None 
        engine_core_outputs = []
        fixed_blocks = self.vllm_config.dynamic_config.fixed_num_gpu_blocks
        enable_kv_resize = self.vllm_config.dynamic_config.enable_kv_resize
        # allow_resize is True only when both conditions are met:
        # 1. fixed_num_gpu_blocks <= 0 (not fixed)
        # 2. enable_kv_resize is True (resize allowed during migration)
        allow_resize = (fixed_blocks <= 0) and enable_kv_resize
        if not allow_resize:
            if not enable_kv_resize:
                logger.info(f"[memory access] enable_kv_resize=False, resize disabled for sync migration")
            else:
                logger.info(f"[memory access] fixed_num_gpu_blocks={fixed_blocks}, resize disabled for sync migration")
        # 若任一 rank 需要 compact，则在持有引擎锁时再次校验一次可用显存，
        # 仍不足时再统一压缩 KV cache，最后再进行 add_layers
        with self.engine_lock:
            time_start =  time.time()
            # 先获取一次内存快照
            assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
            mem_infos = self.model_executor.get_workers_mem_info()

            # Adding/removing
            need_compact = False
            adding_per_rank: dict[int, list[Tuple[int, int]]] = {}
            maximum_kv_block_num_after_compact: list[int] = []

            # 记录当前gpu上的layer configuration
            # 除了migration前的config和migration后的config之外，在migration的过程中
            # 还有可能出现一些中间状态的migration config
            tmp_pp_layer_config = deepcopy(self.cur_pp_layer_config)
            for rank, layers in enumerate(pp_layer_config):
                # Skip ranks that are being deactivated (target start > end indicates inactive).
                if layers[0] > layers[1]:
                    assess = self._assess_memory_for_layer_reconfiguration(rank, 0, mem_infos[rank], self.cur_pp_layer_config)
                    maximum_kv_block_num_after_compact.append(assess.max_blocks_per_layer)
                    continue
                start_layer, end_layer = tmp_pp_layer_config[rank][0], tmp_pp_layer_config[rank][1]
                adding_layer_list = []
                if start_layer > end_layer:
                    adding_layer_list.append((layers[0], layers[1]))
                    tmp_pp_layer_config[rank] = (layers[0], layers[1])
                elif layers[0] < start_layer:
                    adding_layer_list.append((layers[0], start_layer - 1))
                    tmp_pp_layer_config[rank] = (layers[0], tmp_pp_layer_config[rank][1])
                if layers[1] > end_layer:
                    adding_layer_list.append((end_layer + 1, layers[1]))
                    tmp_pp_layer_config[rank] = (tmp_pp_layer_config[rank][0], layers[1])
                logger.info(f"rank {rank}:\n adding_layer_list: {adding_layer_list}")
                # 在 adding 之前，校验对应 GPU 是否有足够可用显存容纳"将要新增的层"的权重；
                # 如果不足，尝试评估：压缩（compact）KV cache 后释放的显存，是否足以腾挪出空间。
                # 同时，如果当前的GPU available memory已经足够，则需要评估是否需要resize kv cache
                adding_layer_num = sum(layers[1] - layers[0] + 1 for layers in adding_layer_list)
                assess = self._assess_memory_for_layer_reconfiguration(rank, adding_layer_num, mem_infos[rank], self.cur_pp_layer_config)
                maximum_kv_block_num_after_compact.append(assess.max_blocks_per_layer)
                if len(adding_layer_list) > 0:
                    adding_per_rank[rank] = adding_layer_list
                    if assess.enough_without_compact:
                        pass
                    elif assess.can_fit_after_compact:
                        need_compact = True
                    else:
                        logger.info(
                            "Rank %s lacks memory even after KV compact estimate: max_blocks_per_layer: %s, current_used_blocks: %s",
                            rank, assess.max_blocks_per_layer, self.scheduler.kv_cache_manager.block_pool.num_gpu_blocks - self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks())
                        return []
            compacted_length = min(maximum_kv_block_num_after_compact)
            original_length = self.scheduler.kv_cache_manager.block_pool.num_gpu_blocks

            # For sync migration, running queue should be empty after drain
            # No slot_mapping is needed - receiver will create fresh empty KV caches

            outputs = self._drain_out_running_queue(finish_waiting=True)
            engine_core_outputs.extend(outputs)
            drain_time_ms = (time.time() - time_start) * 1000
            logger.info(f"[timeline]: after drain out running queue, time taken: {human_readable_duration(time.time() - time_start)}")
            logger.info(f"[debug]: tmp pp layer config: {tmp_pp_layer_config}")

            assert isinstance(self.scheduler, DynamicScheduler)
            assert isinstance(self.scheduler.kv_cache_manager, DynamicKVCacheManager)
            compacted_length = min(maximum_kv_block_num_after_compact)

            if need_compact and not allow_resize:
                raise RuntimeError(
                    "Sync migration requires KV cache compact/resize but resize is disabled "
                    f"(enable_kv_resize={enable_kv_resize}, fixed_num_gpu_blocks={fixed_blocks}). "
                    f"compacted_length={compacted_length}")

            if need_compact:
                logger.info(f"[memory access] compacted_length: {compacted_length}, maximum_kv_block_num_after_compact: {maximum_kv_block_num_after_compact}")
                # Get bitmap from scheduler before compacting
                bitmap = self.scheduler.compact_kv_cache(compacted_length)
                self._compact_kv_cache(compacted_length, bitmap)
                logger.info(f"[memory access] KV cache compacted to {compacted_length} blocks before adding layers")
                logger.info(f"[timeline]: after compact kv cache, time taken: {human_readable_duration(time.time() - time_start)}")
            else:
                logger.info(f"[memory access] no need to compact kv cache")
            logger.info(f"[memory access] allow_resize: {allow_resize}, compacted_length: {compacted_length}, original_length: {original_length}") 

            # 需要resize kv cache来进行migration，这里的resize一定是缩小
            # Only shrink when compacted_length < original_length.
            # When fixed_num_gpu_blocks constrains original to be smaller than
            # the theoretical max, compacted > original — skip pre-migration
            # resize; post-migration phase handles expansion.
            if allow_resize and compacted_length < original_length:
                # 我理解这里只要shrink block pool在resize kv cache之前调用就行了
                # TODO: 在resize之前，理论上需要保证没有shrink_kv_cache之前的in-flight request
                # 这可能需要我实现一个scheduleroutput中的同步机制。
                self.model_executor.resize_kv_cache(compacted_length)
                self.scheduler.shrink_block_pool(compacted_length)
                logger.info(f"[timeline]: after resize kv cache, time taken: {human_readable_duration(time.time() - time_start)}")
                logger.info(f"[timeline]: after shrink block pool, time taken: {human_readable_duration(time.time() - time_start)}")
            # 对所有需要新增层的 rank 执行 add_layers
            # 注意：在 sync migration 中必须使用同步的 add_layers，
            # 否则 _listen_loop 在绑定 KV cache 时会等待 _layer_loaded_cv 锁，
            # 但该锁被 async_add_layers 线程持有，导致死锁。
            
            # Save original config before updating - needed for find_src_rank_for_range
            original_pp_layer_config = deepcopy(self.cur_pp_layer_config)
            
            weight_loading_mode = self._dispatch_weight_loading(
                adding_per_rank, migration_kind="sync")
            if weight_loading_mode == "async":
                logger.info("[sync_migration] waiting for async weight loading to complete before KV migration")
                self.model_executor.wait_for_all_async_add_layers()
            # Update cur_pp_layer_config immediately after add_layers succeeds
            # This ensures Engine state matches Worker state even if later operations fail
            self.cur_pp_layer_config = deepcopy(tmp_pp_layer_config)
            logger.info(f"[sync_migration]: updated cur_pp_layer_config to intermediate state {self.cur_pp_layer_config} after add_layers")
            time_kv_compact_end = time.time()
            logger.info(f"[timeline]: before start actual kv cache migration, time taken to complete {weight_loading_mode} weight loading, compact, resize kv cache: {human_readable_duration(time_kv_compact_end - time_start)}")


            # 在异步传输/扩容前，抓取 KV cache 的快照，记录各请求的已计算 token 与 block 映射
            # 用于后续传输完成后对齐新增 token，实现两 GPU 之间 KV 同步
            # Ensure no in-flight work before snapshotting KV state

            # 计算 src->dst 传输对
            # 辅助函数：根据原始分片配置找到包含区间 [lo, hi] 的所有源 rank
            # Note: Use original_pp_layer_config (saved before add_layers), not cur_pp_layer_config
            # 在 shrink 场景下，一个 target range 可能跨越多个 source rank
            def find_src_ranks_for_range(lo: int, hi: int) -> list[tuple[int, list[int]]]:
                result: list[tuple[int, list[int]]] = []
                remaining_lo = lo
                for src_rank, (cur_lo, cur_hi) in enumerate(original_pp_layer_config):
                    if cur_lo > cur_hi:
                        continue
                    if remaining_lo > hi:
                        break
                    if remaining_lo <= cur_hi and hi >= cur_lo:
                        overlap_lo = max(remaining_lo, cur_lo)
                        overlap_hi = min(hi, cur_hi)
                        if overlap_lo <= overlap_hi:
                            result.append((src_rank, list(range(overlap_lo, overlap_hi + 1))))
                            remaining_lo = overlap_hi + 1
                if remaining_lo <= hi:
                    raise AssertionError(
                        f"Could not cover full range [{lo}, {hi}] from original config {original_pp_layer_config}. "
                        f"Remaining: [{remaining_lo}, {hi}]")
                return result

            # 聚合为以 src_rank 为键的 rank_to_layers_ids 映射
            sending_plan_for_ranks: dict[int, dict[int, list[Tuple[int, int]]]] = {}
            for dst_rank, add_list in adding_per_rank.items():
                for lo, hi in add_list:
                    for src_rank, layer_ids in find_src_ranks_for_range(lo, hi):
                        plan = sending_plan_for_ranks.setdefault(src_rank, {})
                        layer_ranges = plan.setdefault(dst_rank, [])
                        layer_ranges.append((layer_ids[0], layer_ids[-1]))

            # For sync migration, slot_mapping is None - receiver creates fresh empty KV caches
            self.model_executor.start_kv_cache_migration_sync(pp_layer_config, sending_plan_for_ranks, adding_per_rank, None)
            time_kv_migration_end = time.time()
            logger.info(f"[timeline]: after start kv cache migration, time taken: {human_readable_duration(time_kv_migration_end - time_kv_compact_end)}")
            time_kv_migration_end = time.time()
            logger.info(f"time taken to start kv cache migration: {human_readable_duration(time_kv_migration_end - time_start)}")
            final_pp_layer_config = deepcopy(tmp_pp_layer_config)
            deleting_layer_assesses: list[int] = []
            for rank, layers in enumerate(pp_layer_config):
                # 计算对于每一个rank而言，需要删除哪一些layers
                start_layer, end_layer = final_pp_layer_config[rank][0], final_pp_layer_config[rank][1]
                deleting_layer_list = []
                if layers[0] > layers[1]:
                    if start_layer <= end_layer:
                        deleting_layer_list.append((start_layer, end_layer))
                    final_pp_layer_config[rank] = (layers[0], layers[1])
                else:
                    if layers[0] > start_layer:
                        deleting_layer_list.append((start_layer, layers[0] - 1))
                        final_pp_layer_config[rank] = (layers[0], final_pp_layer_config[rank][1])
                    if layers[1] < end_layer:
                        deleting_layer_list.append((layers[1] + 1, end_layer))
                        final_pp_layer_config[rank] = (final_pp_layer_config[rank][0], layers[1])
                deleting_layer_num = -sum(layers[1] - layers[0] + 1 for layers in deleting_layer_list)
                assess = self._assess_memory_for_layer_reconfiguration(rank, deleting_layer_num, mem_infos[rank], tmp_pp_layer_config)
                deleting_layer_assesses.append(assess.max_blocks_per_layer)

            for rank, layers in enumerate(final_pp_layer_config):
                assert layers[0] == pp_layer_config[rank][0] and layers[1] == pp_layer_config[rank][1]
            calculated_max_blocks = min(deleting_layer_assesses)
            # Determine final KV block count
            if target_kv_blocks is not None and target_kv_blocks > 0:
                resized_block_num = target_kv_blocks
                logger.info(f"Using explicit target_kv_blocks={target_kv_blocks} "
                            f"(calculated max would be {calculated_max_blocks})")
            else:
                resized_block_num = calculated_max_blocks

            # Always update scheduler's pp_layer_config, regardless of whether block num changes
            self.scheduler.sync_change_configuration(pp_layer_config)
            current_blocks = self.scheduler.kv_cache_manager.num_gpu_blocks
            if allow_resize and resized_block_num != current_blocks:
                logger.info(f"[memory access] resizing KV cache from {current_blocks} to {resized_block_num} blocks")
                self.model_executor.resize_kv_cache(resized_block_num)
                if resized_block_num > current_blocks:
                    self.scheduler.extend_block_pool(resized_block_num)
                else:
                    self.scheduler.shrink_block_pool(resized_block_num)
            elif not allow_resize:
                logger.info(f"[memory access] fixed_num_gpu_blocks={fixed_blocks}, skipping end-of-migration resize (would be {resized_block_num} blocks)")
            self.cur_pp_layer_config = pp_layer_config
            self._apply_active_pp_ranks_for_config(pp_layer_config)
            engine_lock_total_ms = (time.time() - time_start) * 1000
            logger.info(f"[sync_migration]: updated cur_pp_layer_config to {self.cur_pp_layer_config}, time taken: {human_readable_duration(time.time() - time_start)}")
            logger.info(f"[timeline]: migration process time taken: {human_readable_duration(time.time() - time_start)}")
            if log_stop_time:
                logger.info(f"[STOP_TIME][sync]: engine_lock_total={engine_lock_total_ms:.2f}ms drain={drain_time_ms:.2f}ms")

            return engine_core_outputs

    def set_pp_config(
        self,
        pp_layer_config: list[Tuple[int, int]],
        alternative_configs: Optional[Dict[int, Any]] = None,
        migration_steps: Optional[list[int]] = None,
        migration_mode: Optional[str] = None,
        weight_loading_mode: Optional[str] = None,
        weight_chunk_size_mb: Optional[float] = None,
        fixed_num_gpu_blocks: Optional[int] = None,
    ) -> list[EngineCoreOutputs]:
        """Set pipeline configuration to a specific target config.
        
        This method allows setting any valid pp_layer_config dynamically.
        If the target config is the same as the current config, the migration
        is skipped (only migration thread state is reset).
        
        This is used in sweep_test mode where a single server instance
        serves multiple experiments with different PP configurations.
        
        Args:
            pp_layer_config: Target configuration as list of (start, end) tuples per rank.
                             Example: [(0, 39), (40, 63)] for 2 ranks.
            alternative_configs: Optional new migration targets. If provided, updates
                                 the migration_thread's configuration.
                                 Format: {0: [(0,17), (18,63)], 1: [(0,19), (20,63)]}
            migration_steps: Optional new migration steps (request indices at which
                            to trigger migration). If provided, updates the migration_thread.
            migration_mode: Optional migration mode ('sync', 'async', or 'async_fast'). If provided,
                           updates self.dynamic_config.migration_mode for subsequent migrations.
            weight_loading_mode: Optional weight loading mode ('sync' or 'async'). If provided,
                           updates self.dynamic_config.weight_loading_mode for subsequent migrations.
        
        Returns:
            List of EngineCoreOutputs from processing any pending requests.
        """
        outputs = []
        pp_layer_config = self._normalize_pp_layer_config(pp_layer_config)
        
        # Wait for any ongoing async migration to fully complete
        # (including worker-side do_resize / finish_migration) before resetting state
        if not self._migration_done_event.wait(timeout=120):
            raise RuntimeError("Timeout waiting for ongoing migration to complete before setting new PP config. Current migration may be stuck.")
        
        # Update migration_mode if provided
        if migration_mode is not None:
            logger.info(f"set_pp_config: Updating migration_mode to '{migration_mode}'")
            self.migration_config.migration_mode = migration_mode

        if weight_loading_mode is not None:
            if weight_loading_mode not in ("async", "sync"):
                raise ValueError(
                    f"Invalid weight_loading_mode: {weight_loading_mode}. Must be 'async' or 'sync'.")
            old_mode = self.vllm_config.dynamic_config.weight_loading_mode
            if old_mode != weight_loading_mode:
                logger.info("set_pp_config: Updating weight_loading_mode from '%s' to '%s'",
                            old_mode, weight_loading_mode)
                self.vllm_config.dynamic_config.weight_loading_mode = weight_loading_mode
            else:
                logger.info(f"set_pp_config: weight_loading_mode already '{weight_loading_mode}', no change")
        
        # Update weight_chunk_size_mb if provided
        if weight_chunk_size_mb is not None:
            logger.info(f"set_pp_config: Updating weight_chunk_size_mb to {weight_chunk_size_mb}")
            os.environ["VLLM_WEIGHT_CHUNK_SIZE_MB"] = str(weight_chunk_size_mb)
            self.model_executor.set_env_var("VLLM_WEIGHT_CHUNK_SIZE_MB", str(weight_chunk_size_mb))
        
        # Update fixed_num_gpu_blocks if provided
        if fixed_num_gpu_blocks is not None:
            old_val = self.vllm_config.dynamic_config.fixed_num_gpu_blocks
            if old_val != fixed_num_gpu_blocks:
                logger.info(f"set_pp_config: Updating fixed_num_gpu_blocks from {old_val} to {fixed_num_gpu_blocks}")
                self.vllm_config.dynamic_config.fixed_num_gpu_blocks = fixed_num_gpu_blocks
            else:
                logger.info(f"set_pp_config: fixed_num_gpu_blocks already {fixed_num_gpu_blocks}, no change")
        
        # Update migration configuration if provided
        with self._migration_config_lock:
            if alternative_configs is not None:
                # alternative_configs may be in format {"pp_layer_configs": {"0": [...], "1": [...]}}
                # Extract and convert keys to int
                if "pp_layer_configs" in alternative_configs:
                    pp_layer_configs = alternative_configs.get("pp_layer_configs", {})
                    self._migration_alternative_configs = {
                        int(k): self._normalize_pp_layer_config(v)
                        for k, v in pp_layer_configs.items()
                    }
                else:
                    # Already in {int: config} format
                    self._migration_alternative_configs = {
                        int(k) if isinstance(k, str) else k:
                        self._normalize_pp_layer_config(v)
                        for k, v in alternative_configs.items()
                    }
                logger.info(f"set_pp_config: Updated alternative_configs to {self._migration_alternative_configs}")
            if migration_steps is not None:
                self._migration_steps = set(migration_steps)
                logger.info(f"set_pp_config: Updated migration_steps to {migration_steps}")
        
        # Determine the target KV block count based on fixed_num_gpu_blocks
        fixed_blocks = self.vllm_config.dynamic_config.fixed_num_gpu_blocks
        target_kv = fixed_blocks if fixed_blocks > 0 else None

        # Check if already at target configuration
        normalized_cur = [tuple(x) for x in self.cur_pp_layer_config]
        normalized_target = [tuple(x) for x in pp_layer_config]
        if normalized_cur == normalized_target:
            logger.info(f"set_pp_config: Already at target configuration {pp_layer_config}")
            # PP config unchanged, but may need to resize KV cache
            # (e.g. previous experiment was resizable, this one is fixed)
            current_blocks = self.scheduler.kv_cache_manager.num_gpu_blocks
            if target_kv is not None and target_kv != current_blocks:
                logger.info(f"set_pp_config: adjusting KV cache from {current_blocks} to {target_kv} blocks (same PP config)")
                self.model_executor.resize_kv_cache(target_kv)
                if target_kv > current_blocks:
                    self.scheduler.extend_block_pool(target_kv)
                else:
                    self.scheduler.shrink_block_pool(target_kv)
            else:
                logger.info(f"set_pp_config: KV cache already at {current_blocks} blocks, no resize needed")
        else:
            logger.info(f"set_pp_config: changing from {self.cur_pp_layer_config} to {pp_layer_config}, target_kv_blocks={target_kv}")
            # Temporarily set fixed_num_gpu_blocks=-1 and enable_kv_resize=True so sync migration
            # can freely compact/shrink/expand.  target_kv_blocks tells it
            # the desired final KV block count.
            # Note: enable_kv_resize only affects migration triggered by migration_steps,
            # but set_pp_config should always be able to resize.
            saved_fixed = self.vllm_config.dynamic_config.fixed_num_gpu_blocks
            saved_enable_kv_resize = self.vllm_config.dynamic_config.enable_kv_resize
            self.vllm_config.dynamic_config.fixed_num_gpu_blocks = -1
            self.vllm_config.dynamic_config.enable_kv_resize = True
            # Disable STOP_TIME logging on workers to avoid polluting migration metrics
            self.model_executor.set_log_stop_time(False)
            try:
                outputs.extend(self.change_model_configuration_by_kv_transfer_sync(
                    pp_layer_config,
                    target_kv_blocks=target_kv,
                    log_stop_time=False,  # Don't pollute migration metrics
                ))
            finally:
                # Restore to the NEW target value (already set by the
                # "Update fixed_num_gpu_blocks" block above)
                self.vllm_config.dynamic_config.fixed_num_gpu_blocks = fixed_blocks
                self.vllm_config.dynamic_config.enable_kv_resize = saved_enable_kv_resize
                # Re-enable STOP_TIME logging on workers
                self.model_executor.set_log_stop_time(True)
        
        # Signal migration_thread to reset request counter for next benchmark run
        logger.info("set_pp_config: Signaling migration_thread to reset request counter")
        # Also clear the request_num_queue to discard any pending request counts
        # from the previous repetition
        self._migration_reset_event.set()
        logger.info(f"set_pp_config completed, now at {self.cur_pp_layer_config}")
        return outputs

    def get_engine_state(self) -> Dict[str, Any]:
        assert isinstance(self.scheduler, DynamicScheduler)

        batch_queue_size = _safe_queue_size(self.batch_queue)
        input_queue_size = _safe_queue_size(self.input_queue)
        unfinished_requests = self.scheduler.get_num_unfinished_requests()
        running_requests = self.scheduler.running_controller.get_total_length()
        waiting_requests = self.scheduler.waiting_controller.get_total_length()
        migration_in_process = (self.migration_in_process
                                or not self._migration_done_event.is_set())

        is_idle = (unfinished_requests == 0 and batch_queue_size == 0
                   and input_queue_size == 0 and not self.engines_running
                   and not migration_in_process)

        return {
            "is_idle": is_idle,
            "unfinished_requests": unfinished_requests,
            "running_requests": running_requests,
            "waiting_requests": waiting_requests,
            "tracked_requests": len(self.scheduler.requests),
            "batch_queue_size": batch_queue_size,
            "input_queue_size": input_queue_size,
            "engines_running": bool(self.engines_running),
            "migration_in_process": migration_in_process,
            "current_pp_layer_config": self.cur_pp_layer_config,
        }

    def migrate_layer_v1(self, rank_from: int, rank_to: int, num_layers: int) -> list[EngineCoreOutputs]:
        with self.engine_lock:
            logger.info(f"migrating layer from {rank_from} to {rank_to} with {num_layers} layers")
            start_time = time.time()
            # First, we need to wait for all batch to be finished
            assert self.batch_queue is not None
            assert isinstance(self.scheduler, DynamicScheduler)
            assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
            engine_core_outputs = []
            while not self.batch_queue.empty():
                future, scheduler_output = self.batch_queue.get_nowait()
                # Blocking until the first result is available.
                model_output = future.result()
                self.batch_queue.task_done()
                engine_core_outputs.append(self.scheduler.update_from_output(
                    scheduler_output, model_output))
            drain_out_time = time.time()
            self.scheduler.preempt_all_requests()
            preempt_time = time.time()
            # Release the kv cache
            self.model_executor.release_kv_cache()
            release_kv_cache_time = time.time()
            # Doing the model layer weight migration 
            layers, next_layer_config = \
                get_new_layer_config_with_migration_action(
                    rank_from,
                    rank_to, 
                    num_layers, 
                    self.scheduler.pp_layer_config_status.get_cur_pp_layer_config()
                    )
            
            self.model_executor.async_add_layers(rank_to, [layers])
            self.model_executor.remove_layers(rank_from, [layers])
            self.scheduler.update_layer_config(next_layer_config)
            weight_migration_time = time.time()

            # Reinitialize the kv cache
            _, _, kv_cache_config = self._reinitialize_kv_caches(self.vllm_config)
            reinitialize_kv_cache_time = time.time()
            self.scheduler.re_initialize_kv_cache_manager(kv_cache_config)
            end_time = time.time()

            from datetime import timedelta

            def format_duration(seconds):
                return str(timedelta(seconds=seconds))

            logger.info(f"drain_out_time: {format_duration(drain_out_time - start_time)}")
            logger.info(f"preempt_time: {format_duration(preempt_time - drain_out_time)}")
            logger.info(f"release_kv_cache_time: {format_duration(release_kv_cache_time - preempt_time)}")
            logger.info(f"weight_migration_time: {format_duration(weight_migration_time - release_kv_cache_time)}")
            logger.info(f"reinitialize_kv_cache_time: {format_duration(reinitialize_kv_cache_time - weight_migration_time)}")
            logger.info(f"end_time: {format_duration(end_time - reinitialize_kv_cache_time)}")

            return engine_core_outputs

    def migrate_layers_v0(self, rank_from: int, rank_to: int, num_layers: int) -> Future:
        assert False
        logger.info(f"migrating layers from {rank_from} to {rank_to} with {num_layers} layers")
        start = time.time()
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
        assert isinstance(self.scheduler, DynamicScheduler)
        assert self.scheduler.migration_status == MigrationStatus.NOT_MIGRATING
        assert self.migration_status == MigrationStatus.NOT_MIGRATING
        # Check if the memory is enough for the new layers

        self.migration_status = MigrationStatus.MIGRATING
        available_gpu_memory = self.model_executor.get_current_available_memory()[rank_to]
        logger.info(f"available_gpu_memory: {available_gpu_memory}")
        logger.info(f"self.kv_cache_size: {self.kv_cache_size}")
        logger.info(f"self.kv_cache_num_blocks: {self.kv_cache_num_blocks}")
        logger.info(f"num_layers: {num_layers}")
        assert available_gpu_memory > self.kv_cache_size * num_layers

        # Get the new layer configuration 
        layers, next_layer_config = get_new_layer_config_with_migration_action(rank_from, 
                                                                       rank_to, 
                                                                       num_layers, 
                                                                       self.scheduler.pp_layer_config_status.get_cur_pp_layer_config())
        self.migrating_layers = layers
        self.rank_from = rank_from
        self.model_executor.add_layers(rank_to, layers)
        # Get the kv cache spec for the new layers
        kv_cache_spec = self.model_executor.get_kv_cache_spec_for_layers(rank_to, layers)
        self.model_executor.initialize_kv_cache_for_layers(rank_to, 
            kv_cache_spec, 
            self.kv_cache_size, 
            self.kv_cache_num_blocks, 
            layers)

        # Start migration process in the scheduler
        future = self.scheduler.v1_start_migration(next_layer_config)
        end = time.time()
        logger.info(f"Added layers to {rank_to} took {end - start} seconds")
        return future

    def execute_model(self, scheduler_output: SchedulerOutput):
        with self.engine_lock:
            return super().execute_model(scheduler_output)

    def done_migration(self):
        assert False
        assert self.migration_status == MigrationStatus.MIGRATING
        assert self.migrating_layers is not None
        assert self.rank_from is not None
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
        self.model_executor.remove_layers(self.rank_from, self.migrating_layers)
        self.model_executor.release_kv_cache_for_layers(self.rank_from, self.migrating_layers)
        self.migration_status = MigrationStatus.NOT_MIGRATING
        self.layers_to_be_removed = None
        self.rank_from = None

    def _compact_kv_cache(self, compacted_length: int, bitmap: bitarray):
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
        assert isinstance(self.scheduler, DynamicScheduler)
        self.model_executor.compact_kv_cache(compacted_length, bitmap)

class DynamicEngineCoreProc(DynamicEngineCore):
    """ZMQ-wrapper for running EngineCore in background process.
    This whole class is replicated from the vllm EngineCoreProc.
    We use this class to adapt DynamicEngineCore
    """

    ENGINE_CORE_DEAD = b'ENGINE_CORE_DEAD'

    def __init__(
        self,
        vllm_config: VllmConfig,
        dynamic_config: MigrationConfig,
        on_head_node: bool,
        input_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        engine_index: int = 0,
    ):
        input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()

        executor_fail_callback = lambda: input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        # Create input socket.
        input_ctx = zmq.Context()
        identity = engine_index.to_bytes(length=2, byteorder="little")
        input_socket = make_zmq_socket(input_ctx,
                                       input_address,
                                       zmq.DEALER,
                                       identity=identity,
                                       bind=False)
        try:
            # Register engine with front-end.
            output_address = self.startup_handshake(
                input_socket, on_head_node, vllm_config.parallel_config)

            # Update config which may have changed from the handshake.
            vllm_config.__post_init__()

            # Set up data parallel environment.
            self._init_data_parallel(vllm_config)

            # Initialize engine core and model.
            super().__init__(vllm_config, 
                            executor_class, 
                            log_stats, 
                            dynamic_config,
                            executor_fail_callback,
                            )

            self.step_fn = (self.step if self.batch_queue is None else
                            self.step_with_batch_queue)
            self.engines_running = False

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            input_socket.send(
                msgspec.msgpack.encode({
                    "status": "READY",
                    "local": on_head_node,
                    "num_gpu_blocks": num_gpu_blocks,
                }))

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            self.input_queue = input_queue
            self.output_queue = queue.Queue[Union[EngineCoreOutputs, bytes]]()
            self.metrics_queue = queue.Queue[DynamicMetricsOutput]()
            self.request_num_queue = queue.Queue[int]()
            threading.Thread(target=self.process_input_socket,
                             args=(input_socket, ),
                             daemon=True).start()
            input_socket = None
            self.output_thread = threading.Thread(
                target=self.process_output_socket,
                args=(output_address, engine_index),
                daemon=True)
            self.output_thread.start()
            threading.Thread(target=self.migration_thread, daemon=True).start()
            threading.Thread(target=self.stress_tester_thread, daemon=True).start()
            threading.Thread(target=self.metrics_thread, daemon=True).start()
            threading.Thread(target=self.test_kv_cache_compact_thread, daemon=True).start()
            logger.info(f"kv cache config: {self.vllm_config.cache_config}")
        finally:
            if input_socket is not None:
                input_socket.close(linger=0)

    @staticmethod
    def startup_handshake(input_socket: zmq.Socket, on_head_node: bool,
                          parallel_config: ParallelConfig) -> str:

        # Send registration message.
        input_socket.send(
            msgspec.msgpack.encode({
                "status": "HELLO",
                "local": on_head_node,
            }))

        # Receive initialization message.
        logger.info("Waiting for init message from front-end.")
        if not input_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60 * 1000):
            raise RuntimeError("Did not receive response from front-end "
                               f"process within {HANDSHAKE_TIMEOUT_MINS} "
                               f"minutes")
        init_bytes = input_socket.recv()
        init_message = msgspec.msgpack.decode(init_bytes)
        logger.debug("Received init message: %s", init_message)

        output_socket_address = init_message["output_socket_address"]
        #TBD(nick) maybe replace IP with configured head node address

        received_parallel_config = init_message["parallel_config"]
        for key, value in received_parallel_config.items():
            setattr(parallel_config, key, value)

        return output_socket_address

    @staticmethod
    def run_engine_core(*args,
                        dp_rank: int = 0,
                        local_dp_rank: int = 0,
                        **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: Optional[DynamicEngineCoreProc] = None
        try:
            parallel_config: ParallelConfig = kwargs[
                "vllm_config"].parallel_config
            assert parallel_config.data_parallel_size == 1, "DynamicEngineCoreProc only supports data parallel size 1"
            
            engine_core = DynamicEngineCoreProc(*args,**kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            time_start = time.time()
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()
            logger.info(f"debug: ---------------------process the engine step, time: {time.time() - time_start:.2f} seconds")

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""
        waited = False
        while (not self.engines_running
               and not self.scheduler.has_requests()
               and (self.batch_queue is None or self.batch_queue.empty())):
            if logger.isEnabledFor(DEBUG) and self.input_queue.empty():
                logger.debug("EngineCore waiting for work.")
                waited = True
            req = self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self):
        """Called only when there are unfinished local requests."""
        # Step the engine core.
        assert isinstance(self.scheduler, DynamicScheduler)
        outputs = self.step_fn()
        kv_cache_utilization = self.scheduler.get_kv_cache_utilization()
        # Put EngineCoreOutputs into the output queue.
        if outputs is not None:
            self.output_queue.put_nowait(outputs)
        self.metrics_queue.put_nowait(DynamicMetricsOutput(kv_cache_utilization=kv_cache_utilization))

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            self.request_num_queue.put_nowait(1)
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                output.result = method(
                    *self._convert_msgspec_args(method, args))
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (f"Call to {method_name} method"
                                          f" failed: {str(e)}")
            self.output_queue.put_nowait(
                EngineCoreOutputs(utility_output=output))
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error("Unrecognized input request type encountered: %s",
                         request_type)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
         arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation) if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation) else v
            for v, p in zip(args, arg_types))

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(DynamicEngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal("vLLM shutdown signal from EngineCore failed "
                         "to send. Please report this issue.")

    def process_input_socket(self, input_socket: zmq.Socket):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        while True:
            # (RequestType, RequestData)
            type_frame, *data_frames = input_socket.recv_multipart(copy=False)
            request_type = EngineCoreRequestType(bytes(type_frame.buffer))

            # Deserialize the request data.
            decoder = add_request_decoder if (
                request_type == EngineCoreRequestType.ADD) else generic_decoder
            request = decoder.decode(data_frames)

            # Push to input queue for core busy loop.
            self.input_queue.put_nowait((request_type, request))

    def process_output_socket(self, output_path: str, engine_index: int):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with zmq_socket_ctx(output_path, zmq.constants.PUSH,
                            linger=4000) as socket:
            while True:
                outputs = self.output_queue.get()
                if outputs == DynamicEngineCoreProc.ENGINE_CORE_DEAD:
                    socket.send(outputs, copy=False)
                    break
                assert not isinstance(outputs, bytes)
                outputs.engine_index = engine_index

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = socket.send_multipart(buffers,
                                                copy=False,
                                                track=True)
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < 2:
                    # Keep at most 2 buffers to reuse.
                    reuse_buffers.append(buffer)

    def migration_thread_rr(self):
        assert False
        if self.vllm_config.dynamic_config.is_migration:
            migration_interval = envs.MIGRATION_INTERVAL
            assert migration_interval > 0
            step = 0
            while True:
                logger.info(f"sleep for {migration_interval} seconds")
                time.sleep(migration_interval)
                if step % 2 == 0:
                    engine_core_outputs = self.migrate_layer_v0(0, 1, 10)
                else:
                    engine_core_outputs = self.migrate_layer_v0(1, 0, 10)
                if engine_core_outputs is not None:
                    for output in engine_core_outputs:
                        self.output_queue.put_nowait(output)
                step += 1
        else:
            logger.info(f"is_migration is not set, skipping migration")
            return

    def stress_tester_thread(self):
        """Thread to start memory stress tester at specified request count."""
        import os

        if self.vllm_config.dynamic_config.tester_start_step is None:
            logger.info("tester_start_step is not set, skipping stress tester thread")
            return
        
        if not hasattr(self.vllm_config.dynamic_config, 'memory_stress_tester') or \
           self.vllm_config.dynamic_config.memory_stress_tester is None or \
           not self.vllm_config.dynamic_config.memory_stress_tester.get('enabled', False):
            logger.info("Memory stress tester not enabled, skipping")
            return

        tester_start_step = self.vllm_config.dynamic_config.tester_start_step
        logger.info(f"Stress tester will start at request #{tester_start_step}")
        
        num_of_requests = 0
        while True:
            num_of_requests += self.request_num_queue.get()
            logger.info(f"stress_tester_thread: num_of_requests={num_of_requests}")
            
            if num_of_requests >= tester_start_step:
                logger.info(f"Starting memory stress tester at request #{num_of_requests}")
                # Send RPC to all workers to start stress tester
                self.model_executor.collective_rpc(
                    "start_stress_tester",
                )
                logger.info("Memory stress tester started on all workers")
                break

    def migration_thread(self):
        import os
        from collections import deque

        if not self.vllm_config.dynamic_config.is_migration:
            logger.info("is_migration is not set, skipping migration")
            return

        # --- sliding window settings ---
        WINDOW = 50 

        num_of_requests = 0 
        # Start from 0: alternative_configs now represents migration targets only
        # (initial config is from VLLM_PP_LAYER_PARTITION, not from alternative_configs)
        cur_config = 0
        while True:
            # Check if reset was requested (between benchmark repetitions or experiments)
            if self._migration_reset_event.is_set():
                logger.info(f"migration_thread: resetting request counter from {num_of_requests} to 0, cur_config from {cur_config} to 0")
                num_of_requests = 0
                cur_config = 0
                self._migration_reset_event.clear()
            
            num_of_requests += self.request_num_queue.get()
            logger.info("num_of_requests: " + str(num_of_requests))
            
            # Read current migration config under lock
            with self._migration_config_lock:
                alternative_configs = dict(self._migration_alternative_configs)
                migration_steps = set(self._migration_steps)
            
            # Log current config for debugging (only occasionally)
            if num_of_requests % 100 == 0:
                logger.info(f"migration_thread: current alternative_configs={alternative_configs}, migration_steps={migration_steps}")
            
            if num_of_requests in migration_steps:
                # First increment cur_config to get the target configuration
                # (cur_config=0 is initial config, cur_config=1 is first migration target, etc.)
                cur_config += 1
                if cur_config in alternative_configs:
                    logger.info(f"change model configuration to config index {cur_config}")
                    # Use migration_mode to determine which method to call
                    migration_mode = self.migration_config.migration_mode
                    if migration_mode == "sync":
                        logger.info(f"Using sync migration mode")
                        # Do NOT temporarily disable fixed_num_gpu_blocks here.
                        # The engine core and workers have separate vllm_config
                        # copies. Disabling here only affects the engine core,
                        # causing it to expand the scheduler block pool while
                        # workers keep their KV tensors at the fixed size —
                        # leading to CUDA illegal memory access when the
                        # scheduler allocates blocks beyond that size.
                        # The sync/async migration code already handles
                        # fixed_num_gpu_blocks correctly (skipping resize and
                        # keeping the current block count).
                        engine_core_outputs = self.change_model_configuration_by_kv_transfer_sync(alternative_configs[cur_config])
                        # Drain outputs must be sent to output_queue so the
                        # client receives responses for requests that finished
                        # during the drain; otherwise the client hangs forever
                        # waiting for the missing responses.
                        for output in engine_core_outputs:
                            if output is not None:
                                self.output_queue.put_nowait(output)
                        logger.info(f"migration_thread: flushed {len(engine_core_outputs)} drain outputs to output_queue")
                    elif migration_mode == "async":
                        logger.info(f"Using async migration mode")
                        self.change_model_configuration_by_kv_transfer_async(alternative_configs[cur_config])
                    elif migration_mode == "async_fast":
                        logger.info(f"Using async_fast migration mode")
                        engine_core_outputs = self.change_model_configuration_by_kv_transfer_async_fast(alternative_configs[cur_config])
                        for output in engine_core_outputs:
                            if output is not None:
                                self.output_queue.put_nowait(output)
                        logger.info(f"migration_thread: flushed {len(engine_core_outputs)} async_fast drain outputs to output_queue")
                    else:
                        raise ValueError(f"Invalid migration_mode: {migration_mode}. Must be 'sync', 'async', or 'async_fast'.")

                else:
                    logger.warning(f"migration_thread: cur_config {cur_config} not found in alternative_configs, skipping migration")
                # outputs = self.migrate_layer_v1(1, 0, 24)
                # logger.info("debug-------- engineoutput when migration: " + str(outputs))
                # for output in outputs:
                #     self.output_queue.put_nowait(output)

    def test_kv_cache_compact_thread(self):
        return
        import os
        assert isinstance(self.scheduler, DynamicScheduler)
        if not self.vllm_config.dynamic_config.is_compact_kv:
            logger.info("is_compact_kv is not set, skipping migration")
            return

        compact_steps = set(self.migration_config.compact_steps)
        logger.info(f"compact kv when steps: {compact_steps}")
        assert len(compact_steps) > 0, "compact_steps must be set"

        num_of_requests = 0
        while True:
            num_of_requests += self.request_num_queue.get()
            logger.info("num_of_requests: " + str(num_of_requests))
            if num_of_requests in compact_steps:
                logger.info("compact kv cache")
                num_kv_blocks = self.scheduler.kv_cache_manager.num_gpu_blocks
                kv_cache_ratio = self.scheduler.get_kv_cache_utilization()
                if kv_cache_ratio >= 0.9:
                    continue
                compact_ratio = kv_cache_ratio + 0.1
                compacted_length = int(compact_ratio * num_kv_blocks)
                bitmap = self.scheduler.shrink_block_pool(compacted_length)
                self._compact_kv_cache(compacted_length, bitmap)
    def metrics_thread(self):
        while True:
            metrics = self.metrics_queue.get()
            util = getattr(metrics, "kv_cache_utilization", None)
            # logger.info(f"kv_cache_utilization: {util}")

@dataclass
class DynamicMetricsOutput:
    kv_cache_utilization: float
    num_of_total_serving_request: int = 0


# def derive_migration_actions_from_deployments(
#         from_deployment: list[Tuple[int, int]],
#         to_deployment: list[Tuple[int, int]],
#         ) -> 'list[Tuple[int, int, int]]':
#     """Derive migration actions from two deployments.
#     Args:
#         from_deployment: The current deployment.
#         to_deployment: The target deployment.
#     Returns:
#         A list of migration actions. Each action is a tuple of
#         (rank_from, rank_to, num_layers).
#     """
#     # [1, 4], [5, 6], [, 7]
#     # [1, 1], [2, 3], [4, 7]
#     output = []
#     assert len(from_deployment) == len(to_deployment)
#     for i in range(len(from_deployment)-1):
#         # when from_deployments have layers to migrate to right GPU
#         from_deployment_layers = from_deployment[i]
#         to_deployment_layers = to_deployment[i]

#         num_migrate_layer = abs(from_deployment_layers[1] - to_deployment_layers[1])
#         if from_deployment_layers[1] > to_deployment_layers[1]:
#             output.append((i, i+1, num_migrate_layer))

#         if from_deployment_layers[1] < to_deployment_layers[1]:
#             output.append((i+1, i, num_migrate_layer))

#         # Update the state
#         from_deployment[i], \
#             from_deployment[i+1] = \
#         (from_deployment[i][0], to_deployment[i][1]), \
#             (to_deployment[i+1][0], from_deployment[i+1][1])

#     return output

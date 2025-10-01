# SPDX-License-Identifier: Apache-2.0
from torch import jagged
from .core import EngineCore
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
from typing import Any, Callable, Optional, Tuple, TypeVar, Union
import math

import msgspec
import zmq
from bitarray import bitarray

from vllm.config import ParallelConfig, VllmConfig
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
from vllm.v1.core.sched.dynamic_scheduler import DynamicScheduler
from vllm.v1.core.sched.dynamic_scheduler import MigrationStatus
from vllm.dynamic_config import PPLayerConfigs
from vllm.version import __version__ as VLLM_VERSION
from .utils import get_new_layer_config_with_migration_action
from vllm.v1.utils import WorkerMemInfo, AssessResult, human_readable_duration
from dataclasses import dataclass
from vllm.dynamic_config import DynamicConfig


logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar('_R')  # Return type for collective_rpc

class DynamicEngineCore(EngineCore):

    def __init__(self, *args, 
                        dynamic_config: DynamicConfig, 
                        **kwargs):

        super().__init__(*args, **kwargs)
        assert isinstance(self.scheduler, DynamicScheduler)

        # Make sure dynamic_config is not in the kwargs, but pulling out from it.
        assert "dynamic_config" not in kwargs
        self.migration_status = MigrationStatus.NOT_MIGRATING
        self.engine_lock = threading.Lock()
        self.dynamic_config = dynamic_config
        # TODO: Load initial configurations properly
        self.cur_pp_layer_config = self.scheduler.pp_layer_config_status.get_cur_pp_layer_config()
        self.migration_in_process = False

    def _assess_memory_for_add(
        self,
        rank: int,
        adding_layer_list: list[Tuple[int, int]],
        mem_info: WorkerMemInfo,
    ) -> AssessResult:
        """Assess whether the target rank has enough memory to add layers.

        Returns AssessResult with:
          - enough_without_compact: free_mem >= required_mem
          - can_fit_after_compact: free_mem + freed_estimate > required_mem
          - required_mem: bytes needed by new layers' weights + their KV
            cache (estimated) with safety margin
          - free_mem: current free GPU memory (driver)
          - freed_estimate: estimated bytes that could be freed by compacting KV
        注意，这里的所有memory都是对于整体的memory而言，而不是对于单个kv cache tensor而言的。
        """
        if len(adding_layer_list) == 0:
            return AssessResult(True, True, 0, 0, 0)
        assert isinstance(self.scheduler, DynamicScheduler)
        # 汇总 rank 上要新增的层数
        num_added_layers = sum((hi - lo + 1) for lo, hi in adding_layer_list)
        layer_size = int(mem_info.layer_size)
        free_mem = int(mem_info.free_mem)
        kv_tensor_size = int(mem_info.kv_tensor_size)

        weight_required = num_added_layers * max(layer_size, 0)
        # 估算新增层的需求（按压缩后的长度 u）
        # 在这里，我们按照当前kv cache有可能的最小值来估算新增层对于kv cache的需求
        # 安全余量从环境变量 VLLM_DYNAMIC_MIGRATION_SAFETY_MARGIN 读取，默认 10%
        u = float(self.scheduler.get_kv_cache_utilization())
        assert u >= 0.0
        minimal_kv_needed_for_added = int(num_added_layers * kv_tensor_size * u)
        minimal_total_needed = weight_required + minimal_kv_needed_for_added
        margin_pct = float(os.getenv("VLLM_DYNAMIC_MIGRATION_SAFETY_MARGIN", "0.10"))
        assert margin_pct >= 0.0
        required = int(math.ceil(minimal_total_needed * (1.0 + margin_pct)))

        assert required > 0

        # 直接可用（包含余量）
        if free_mem > required:
            return AssessResult(True, True, required, free_mem, 0)

        # 估算通过压缩 KV cache 可释放的显存（仅针对现有层）
        # 当前 rank 上的层数（用于估算该 rank 的 KV 总占用）
        cur_lo, cur_hi = self.cur_pp_layer_config[rank]
        num_layers_on_rank = cur_hi - cur_lo + 1
        assert num_layers_on_rank > 0 and kv_tensor_size > 0
        freed_estimate = math.floor(int((1.0 - u) * kv_tensor_size * max(num_layers_on_rank, 0)))
        assert freed_estimate >= 0
        can_fit_after = (free_mem + freed_estimate > required)
        return AssessResult(False, can_fit_after, required, free_mem, freed_estimate)

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

    def _drain_out_running_queue(self) -> list[EngineCoreOutputs]:
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
        with self.engine_lock:
            assert self.batch_queue is not None
            engine_core_outputs = None
            scheduler_output = None
            execute_func = self.model_executor.execute_model
                # if not self.migration_in_process \
                # else self.model_executor.execute_cpu_model
            # Try to schedule a new batch if the batch queue is not full, but
            # the scheduler may return an empty batch if all requests are scheduled.
            # Note that this is not blocking.
            if not self.batch_queue.full():
                scheduler_output = self.scheduler.schedule()
                if scheduler_output.total_num_scheduled_tokens > 0:
                    logger.info(f"schedule a new batch, execute_func: {execute_func}")
                    future = execute_func(scheduler_output)
                    logger.info(f"debug: put the new batch into the batch queue")
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
                logger.info(f"debug: get the first batch from the batch queue")
                model_output = future.result()
                logger.info(f"debug: get the result from the batch queue")
                self.batch_queue.task_done()
                engine_core_outputs = self.scheduler.update_from_output(
                    scheduler_output, model_output)

            return engine_core_outputs

    def step(self) -> EngineCoreOutputs:
        assert False, "step is not supported for dynamic engine core"

    def change_model_configuration(self, pp_layer_config: list[Tuple[int, int]]) -> list[EngineCoreOutputs]:
        with self.engine_lock:
            logger.info(f"Change configuration  from {self.cur_pp_layer_config} to {pp_layer_config}")
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
                if layers[1] > end_layer:
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

    def start_migrate_to_new_config(self, pp_layer_config: list[Tuple[int, int]]) -> list[EngineCoreOutputs]:
        logger.info(f"Start migrating to new configuration {pp_layer_config}")
        assert isinstance(self.scheduler, DynamicScheduler)

        engine_core_outputs = []
        # 若任一 rank 需要 compact，则在持有引擎锁时再次校验一次可用显存，
        # 仍不足时再统一压缩 KV cache，最后再进行 add_layers
        with self.engine_lock:
            time_start = time.time()
            # 先获取一次内存快照
            assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
            mem_infos = self.model_executor.get_workers_mem_info()

            # Adding/removing
            need_compact = False
            adding_per_rank: dict[int, list[Tuple[int, int]]] = {}
            assesses: list[AssessResult] = []
            for rank, layers in enumerate(pp_layer_config):
                start_layer, end_layer = self.cur_pp_layer_config[rank][0], self.cur_pp_layer_config[rank][1]
                adding_layer_list = []
                deleting_layer_list = []
                if layers[0] < start_layer:
                    adding_layer_list.append((layers[0], start_layer - 1))
                if layers[1] > end_layer:
                    adding_layer_list.append((end_layer + 1, layers[1]))

                if layers[0] > start_layer:
                    deleting_layer_list.append((start_layer, layers[0] - 1))
                if layers[1] < end_layer:
                    deleting_layer_list.append((layers[1] + 1, end_layer))

                if len(adding_layer_list) == 0 and len(deleting_layer_list) == 0:
                    continue
                logger.info(f"rank {rank}:\n adding_layer_list: {adding_layer_list}\n deleting_layer_list: {deleting_layer_list}")
                # 在 adding 之前，校验对应 GPU 是否有足够可用显存容纳“将要新增的层”的权重；
                # 如果不足，尝试评估：压缩（compact）KV cache 后释放的显存，是否足以腾挪出空间。
                if len(adding_layer_list) > 0:
                    adding_per_rank[rank] = adding_layer_list
                    assess = self._assess_memory_for_add(rank, adding_layer_list, mem_infos[rank])
                    assesses.append(assess)
                    if assess.required_mem > 0:
                        if assess.enough_without_compact:
                            pass
                        elif assess.can_fit_after_compact:
                            need_compact = True
                        else:
                            logger.warning(
                                "Rank %s lacks memory even after KV compact estimate: free=%s, freed_est≈%s, required=%s",
                                rank, assess.free_mem, assess.freed_estimate, assess.required_mem)
                            return []
            # TODO: 在这里实际上我们也需要加上expanding kv cache的逻辑，可以在后面再实现
            if need_compact:
                # 计算每个需要新增层的 rank 所需的压缩目标，取最严格（最小）的目标统一压缩一次
                bitmap = self.scheduler.get_bitmap()
                used_blocks = int(bitmap.count())
                total_blocks = int(self.scheduler.kv_cache_manager.num_gpu_blocks)
                target_candidates: list[int] = []
                for (r, add_list), assess in zip(adding_per_rank.items(), assesses):
                    deficit = max(0, assess.required_mem - assess.free_mem)
                    target_len_r = used_blocks
                    if deficit > 0:
                        cur_lo, cur_hi = self.cur_pp_layer_config[r]
                        num_layers_on_rank = cur_hi - cur_lo + 1
                        assert num_layers_on_rank > 0 and mem_infos[r].kv_tensor_size > 0

                        total_kv_cache_size = num_layers_on_rank * mem_infos[r].kv_tensor_size
                        reduction_blocks = int(math.ceil(deficit / total_kv_cache_size * total_blocks))
                        target_len_r = total_blocks - reduction_blocks
                        assert target_len_r >= used_blocks
                        target_candidates.append(target_len_r)
                    logger.info(
                        "Check rank %s for kv cache compression: free=%s, required=%s, deficit=%s, used=%d, target_len=%d",
                        r, assess.free_mem, assess.required_mem, deficit, used_blocks, target_len_r)

                if target_candidates:
                    # 进行一次统一压缩
                    compacted_length = min(target_candidates)
                    assert used_blocks <= compacted_length <= total_blocks
                    engine_core_outputs.extend(self._compact_kv_cache(compacted_length))
                    logger.info(
                        "KV cache compacted to %d blocks before adding layers (ranks %s)",
                        compacted_length, list(adding_per_rank.keys()))
                else:
                    logger.info("Skip KV compact: all ranks sufficient after recheck")
        # 对所有需要新增层的 rank 执行 add_layers（异步 fire-and-forget）
        for r, add_list in adding_per_rank.items():
            self.model_executor.async_add_layers(r, add_list)
        time_kv_compact_end = time.time()

        logger.info(f"time taken to compact kv cache: {human_readable_duration(time_kv_compact_end - time_start)}")
        # 在异步传输/扩容前，抓取 KV cache 的快照，记录各请求的已计算 token 与 block 映射
        # 用于后续传输完成后对齐新增 token，实现两 GPU 之间 KV 同步
        with self.engine_lock:
            # Ensure no in-flight work before snapshotting KV state
            self._drain_out_running_queue()

            # 计算 src->dst 传输对
            # 辅助函数：根据当前分片配置找到包含区间 [lo, hi] 的源 rank
            def find_src_rank_for_range(lo: int, hi: int) -> int:
                for src_rank, (cur_lo, cur_hi) in enumerate(self.cur_pp_layer_config):
                    if lo >= cur_lo and hi <= cur_hi:
                        return src_rank
                raise AssertionError(f"No source rank found for range [{lo}, {hi}] in {self.cur_pp_layer_config}")

            # 聚合为以 src_rank 为键的 rank_to_layers_ids 映射
            # src_to_plan[src_rank] = { dst_rank: [layer_ids...] }
            src_to_plan: dict[int, dict[int, list[int]]] = {}
            for dst_rank, add_list in adding_per_rank.items():
                for lo, hi in add_list:
                    src_rank = find_src_rank_for_range(lo, hi)
                    plan = src_to_plan.setdefault(src_rank, {})
                    layer_ids = plan.setdefault(dst_rank, [])
                    layer_ids.extend(range(lo, hi + 1))

            # 去重并排序每个 dst_rank 的 layer 列表，然后按 src 发起迁移
            for src_rank, rank_to_layers_ids in src_to_plan.items():
                for dst_rank, layers in rank_to_layers_ids.items():
                    # 去重且保持有序
                    unique_sorted = sorted(set(layers))
                    rank_to_layers_ids[dst_rank] = unique_sorted
                logger.info(f"rank {src_rank} start kv cache migration to rank {rank_to_layers_ids}")
                self.model_executor.start_kv_cache_migration(src_rank, rank_to_layers_ids)
            self.migration_in_process = True
        
        def check_patch_id_diff_thread():
            assert isinstance(self.scheduler, DynamicScheduler)
            patch_id_diff_threshold = int(os.environ.get("VLLM_PATCH_ID_DIFF_THRESHOLD", 5))
            assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
            while True:
                time.sleep(1)
                should_sync = True
                buffer_status_list = self.model_executor.get_kv_buffer_status()
                logger.info(f"buffer_status_list: {buffer_status_list}")
                for src_rank, rank_to_layers_ids in src_to_plan.items():
                    for dst_rank, _ in rank_to_layers_ids.items():
                        src_rank_buffer_status = buffer_status_list[src_rank]
                        dst_rank_buffer_status = buffer_status_list[dst_rank]
                        # Use .get with default 0 to tolerate missing keys
                        if dst_rank not in src_rank_buffer_status.send_patch_ids or src_rank not in dst_rank_buffer_status.recv_patch_ids:
                            logger.info(f"rank {src_rank} or rank {dst_rank} not in the buffer status")
                            should_sync = False
                            break
                        sender_patch_id = src_rank_buffer_status.send_patch_ids[dst_rank]
                        recv_patch_id = dst_rank_buffer_status.recv_patch_ids[src_rank]
                        assert sender_patch_id >= recv_patch_id, "The patch id of the rank should be larger than the last patch"
                        patch_id_diff = sender_patch_id - recv_patch_id
                        if patch_id_diff > patch_id_diff_threshold:
                            should_sync = False
                            break
                    if not should_sync:
                        break
                
                if should_sync:
                    logger.info(f"all patch id diff is less than the threshold, start to synchronize the kv cache")
                    self.scheduler.synchronize_kv_cache_and_change_configuration(pp_layer_config)
                    break

        threading.Thread(target=check_patch_id_diff_thread, daemon=True).start()
        return engine_core_outputs

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
            
            self.model_executor.add_layers(rank_to, [layers])
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

    def _compact_kv_cache(self, compacted_length: int) -> list[EngineCoreOutputs]:
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
        assert isinstance(self.scheduler, DynamicScheduler)
        engine_outputs = self._drain_out_running_queue()

        bitmap = self.scheduler.get_bitmap()
        self.model_executor.compact_kv_cache(compacted_length, bitmap)
        self.scheduler.compact_kv_cache(compacted_length)
        return engine_outputs

class DynamicEngineCoreProc(DynamicEngineCore):
    """ZMQ-wrapper for running EngineCore in background process.
    This whole class is replicated from the vllm EngineCoreProc.
    We use this class to adapt DynamicEngineCore
    """

    ENGINE_CORE_DEAD = b'ENGINE_CORE_DEAD'

    def __init__(
        self,
        vllm_config: VllmConfig,
        dynamic_config: DynamicConfig,
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
                            executor_fail_callback,
                            dynamic_config=dynamic_config
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

        engine_core: DynamicEngineCoreProc
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
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""
        waited = False
        while not self.engines_running and not (self.scheduler.has_requests()):
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
        if os.environ.get("TEST_MIGRATION") == "1":
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
            logger.info(f"TEST_MIGRATION is not set, skipping migration")
            return

    def migration_thread(self):
        import os
        from collections import deque

        if os.environ.get("TEST_MIGRATION") != "1":
            logger.info("TEST_MIGRATION is not set, skipping migration")
            return

        # --- config choices & indices ---
        # Each configuration name must be an integer
        alternative_configs = {
            int(k): v for k, v in self.dynamic_config.alternative_configs.pp_layer_configs.items()
        }
        migration_steps = set(self.dynamic_config.migration_steps)
        max_index = max(alternative_configs.keys())
        cur_config_index = min(alternative_configs.keys())  # 若你希望从0开始，也可直接置0

        logger.info(f"alternative_configs: {alternative_configs}")
        logger.info(f"migration_steps: {migration_steps}")
        # --- sliding window settings ---
        WINDOW = 50 
        UP_THRESHOLD = 0.8   # 只有窗口满且均值>0.6才上调
        DOWN_THRESHOLD = 0.5 # 滞回：窗口满且均值<0.5才下调
        kv_utilizations = deque(maxlen=WINDOW)

        num_of_requests = 0 
        while True:
            num_of_requests += self.request_num_queue.get()
            logger.info("num_of_requests: " + str(num_of_requests))
            if num_of_requests in migration_steps:
                logger.info("change model configuration")
                outputs = self.start_migrate_to_new_config(alternative_configs[1])
                for output in outputs:
                    self.output_queue.put_nowait(output)

    def test_kv_cache_compact_thread(self):
        return
        import os
        assert isinstance(self.scheduler, DynamicScheduler)
        if os.environ.get("TEST_KV_COMPACT") != "1":
            logger.info("TEST_KV_COMPACT is not set, skipping migration")
            return

        compact_steps = set(self.dynamic_config.compact_steps)
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
                self._compact_kv_cache(int(compact_ratio * num_kv_blocks))
    def metrics_thread(self):
        while True:
            metrics = self.metrics_queue.get()
            util = getattr(metrics, "kv_cache_utilization", None)
            logger.info(f"kv_cache_utilization: {util}")

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

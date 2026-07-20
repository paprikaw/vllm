# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.dynamic_kv_cache_manager import DynamicKVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from typing import Any, List, Tuple, Optional, Union
from threading import Lock
import vllm.envs as envs
from enum import Enum
import os
import time
import torch
import gc
from collections import defaultdict, deque
from collections.abc import Iterable
from bitarray import bitarray

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.kvcached_integration import (
    use_flexi_kv_for_runtime,
    use_kvcached_backend,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.distributed.kv_events import EventPublisherFactory
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole

from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager, compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData, SchedulerOutput)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


def _autoscaling_kvcached_debug_enabled() -> bool:
    return os.environ.get("VLLM_AUTOSCALING_KVCACHED_DEBUG",
                          "").lower() in {"1", "true", "yes", "on"}


def _short_req_id(req_id: str) -> str:
    return req_id[-8:] if isinstance(req_id, str) else str(req_id)


class DynamicScheduler(Scheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        """
        Completely copied from parent class, we need to initialize new dynamic kv cache manager in __init_, including the pp layer config status
        """
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.include_finished_set = include_finished_set

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            self.connector = KVConnectorFactory.create_connector_v1(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config)

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # P/D: requests in process of recving KV transfers
        self.finished_recving_kv_req_ids: set[str] = set()

        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        # Request id -> deque of CachedRequestData
        self._cached_reqs_data: dict[
            str, deque[CachedRequestData]] = defaultdict(deque)

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config

        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = DynamicKVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
        )


        # Initialize pp layer config status
        # Use dynamic_config.pp_layer_partition, falls back to VLLM_PP_LAYER_PARTITION env var
        partition_list_str = self.vllm_config.dynamic_config.pp_layer_partition
        if partition_list_str is None:
            partition_list_str = envs.VLLM_PP_LAYER_PARTITION
        assert partition_list_str is not None, "Either dynamic_config.pp_layer_partition or VLLM_PP_LAYER_PARTITION must be set"
        partitions = [
            int(layer) for layer in partition_list_str.split(",")
        ]
        layer_configs = []
        for pp_rank in range(len(partitions)):
            start_layer = sum(partitions[:pp_rank])
            end_layer = start_layer + partitions[pp_rank] - 1
            layer_configs.append((start_layer, end_layer))
        self.pp_layer_config = layer_configs
        self._sync_drain_pending_waiting: Optional[deque[Request]] = None

        # When scheduler executes schedule or migration operation, it needs
        # to acquire the lock.
        self.lock = Lock()
        self._pending_pp_layer_config: Optional[List[Tuple[int, int]]] = None
        self._pending_new_kv_cache_block_num = 0

        self.migration_in_process = False # 控制在每一次scheduler schedule的时候是否需要同时发送slot_mapping
        self.num_tokens_for_migration = 0 # 记录已经发送的slot数量，用来和worker端已处理的slot数量进行对比

        self.cur_scheduler_output_version = 0
        self._debug_scheduler_step_id = 0
        self._debug_request_free_epoch = 0
        self._debug_block_free_epoch = 0
        self._debug_request_free_epochs: dict[str, int] = {}

        # 用以记录在迁移过程中，哪些rank是sender，哪些rank是receiver
        self.sender_list_during_migration: Optional[set[int]] = None 
        self.receiver_list_during_migration: Optional[set[int]] = None

    def async_change_configuration(self, pp_layer_config: List[Tuple[int,int]], new_kv_cache_block_num: int):
        assert self._pending_new_kv_cache_block_num == 0
        assert self._pending_pp_layer_config is None
        self._pending_pp_layer_config = pp_layer_config
        self._pending_new_kv_cache_block_num = new_kv_cache_block_num

    def start_migration(self, sender_list: list[int], receiver_list: list[int], should_increase_scheduler_output_version: bool) -> Union[list[int], None]:
        with self.lock:
            self.migration_in_process = True
            self.sender_list_during_migration = set(sender_list) 
            self.receiver_list_during_migration = set(receiver_list)
            is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
            if should_increase_scheduler_output_version:
                self.cur_scheduler_output_version += 1
            if is_flexi or use_kvcached_backend():
                slot_mapping = self.get_slot_mapping_from_reqs(self.running)
                logger.info(f"[num tokens]: slot_mapping length:{len(slot_mapping)} ")
                logger.info(f"[num tokens]: scheduler side num_tokens:{self.kv_cache_manager} ")
                self.num_tokens_for_migration = len(slot_mapping) 
                return slot_mapping
            else:
                self.num_tokens_for_migration = 0
                return None


    def get_slot_mapping_from_reqs(self, reqs: Union[list[Request], list[NewRequestData]]):
        slot_mapping = []
        block_size = self.block_size
        for req in reqs:
            req_id = req.request_id
            if req_id in self.kv_cache_manager.single_type_manager.req_to_blocks:
                # get_block_ids returns list[list[int]], we want the first list
                block_ids = self.kv_cache_manager.get_block_ids(req_id)[0]
            else:
                block_ids = []
            
            num_computed_tokens = int(req.num_computed_tokens)
            
            for i, block_id in enumerate(block_ids):
                start_token_idx = i * block_size
                assert start_token_idx <= num_computed_tokens
                
                valid_tokens = min(block_size, num_computed_tokens - start_token_idx)
                base_slot = block_id * block_size
                slot_mapping.extend(range(base_slot, base_slot + valid_tokens))
        return slot_mapping

    def sync_change_configuration(self, pp_layer_config: List[Tuple[int,int]]):
        """
        This function is used to synchronize the kv cache and change the configuration
        Note that this function is only used when there is no running batch remain in the pipeline and the scheduling process is suspended.
        Therefore it is common to firstly call _drain_out_running_queue to drain out the running queue, then call this function to change the scheduler configuration.
        The actual configuration change will be done by invoking collective operation "start_kv_cache_migration_sync" in the model executor.
        """
        self.update_layer_config(pp_layer_config)

    def update_layer_config(self, layer_config: List[Tuple[int,int]]):
        self.pp_layer_config = layer_config

    def _schedule(self) -> DynamicSchedulerOutput:
        """
        In this function, we control the behavior of scheduler during the migration process
        There are three types of migration process at the moment:
            1. sync migration with kv cache transfer 
            2. async migration with kv cache transfer
            3. drain-out style of migration (deprecated)

        sync migration: 
            Drain out the running batches and interupt the inference process.
            Change the configuration of gpus and scheduler synchronously.
            In the next scheduling step, the scheduler will schedule the requests using the new configuration. Also scheduled with the next kv cache block num to tell the gpu worker to resize the kv cache *before* the execution of inferenc of inferencee.
        async migration:
            Don't interupt the inference process, the kv tensor is transmitting asynchronously.
            When the timing is right, we inject a sync msg to scheduleroutput to tell the gpu worker sending out all the kv patches. After this async msg, we change the configuration of the scheduler.
            In the next scheduling step, the scheduler is scheduling with the new configuration.
        """
        pp_layer_config = self.pp_layer_config
        scheduler_output = super().schedule()
        self._debug_scheduler_step_id += 1
        scheduler_step_id = self._debug_scheduler_step_id
        is_sync_after_migration = self._pending_pp_layer_config is not None

        # When inject_sync_msg is True, it means we need to sync the kv cache between layers belongs to old and new configuration
        # The logic here will result in a two step scheduling process
        # 1. First step: use the old configuration to schedule the requests, but the is_sync_after_migration is True
        # This will result in different rank to send sync msg to each other
        # 2. Second step: use the new configuration to schedule the requests, but the is_sync_after_migration is False
        #  Note when rank receive this sync msg, it is possible that the kv cache patch is not fully applied to the sync point,
        #  Therefore we will let gpu worker to wait for all kv cache patch to be applied before using the new configuration
        output = DynamicSchedulerOutput(
                scheduled_new_reqs=scheduler_output.scheduled_new_reqs,
                scheduled_cached_reqs=scheduler_output.scheduled_cached_reqs,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
                scheduled_spec_decode_tokens=scheduler_output.scheduled_spec_decode_tokens,
                scheduled_encoder_inputs=scheduler_output.scheduled_encoder_inputs,
                num_common_prefix_blocks=scheduler_output.num_common_prefix_blocks,
                finished_req_ids=scheduler_output.finished_req_ids,
                free_encoder_input_ids=scheduler_output.free_encoder_input_ids,
                structured_output_request_ids=scheduler_output.structured_output_request_ids,
                grammar_bitmask=scheduler_output.grammar_bitmask,
                kv_connector_metadata=scheduler_output.kv_connector_metadata,
                pp_layer_config=pp_layer_config,
                current_scheduler_output_version=self.cur_scheduler_output_version,
                scheduler_step_id=scheduler_step_id,
                scheduler_request_free_epoch=self._debug_request_free_epoch,
                scheduler_block_free_epoch=self._debug_block_free_epoch,
                is_sync_after_migration=is_sync_after_migration,
                total_migration_tokens=self.num_tokens_for_migration + scheduler_output.total_num_scheduled_tokens,
                # total_migration_tokens=self.num_tokens_for_migration,
                new_kv_cache_block_num=self._pending_new_kv_cache_block_num,
                migration_in_process=self.migration_in_process,
                sender_list= self.sender_list_during_migration,
                receiver_list= self.receiver_list_during_migration,
                # slot_mapping = self.get_slot_mapping_from_reqs(scheduler_output.scheduled_new_reqs) if self.sending_slot_mapping else None,
            )
        if _autoscaling_kvcached_debug_enabled():
            logger.warning(
                "[KVCACHED_SCHED_OUTPUT_TRACE] phase=create step_id=%s "
                "sched_ver=%s request_free_epoch=%s block_free_epoch=%s "
                "total_tokens=%s scheduled_req_ids=%s finished_req_ids=%s "
                "migration_in_process=%s is_sync_after_migration=%s",
                scheduler_step_id, self.cur_scheduler_output_version,
                self._debug_request_free_epoch, self._debug_block_free_epoch,
                output.total_num_scheduled_tokens,
                [_short_req_id(req_id)
                 for req_id in output.num_scheduled_tokens.keys()],
                [_short_req_id(req_id) for req_id in output.finished_req_ids],
                output.migration_in_process, output.is_sync_after_migration)
        # scheduled token为0的请求不应该发送给worker，因此在这里我们跳过后续的asynchronise处理
        if output.total_num_scheduled_tokens == 0:
            return output

        if self.migration_in_process:
            self.num_tokens_for_migration += output.total_num_scheduled_tokens
            logger.info(f"[num tokens]: scheduler output token {output.total_num_scheduled_tokens} added, total tokens for migration: {self.num_tokens_for_migration} ")


        if self._pending_pp_layer_config is not None:
            # 在最后的阶段，只有可能是expand，不可能shrink
            assert self._pending_new_kv_cache_block_num >= self.kv_cache_manager.num_gpu_blocks, f"pending_new_kv_cache_block_num: {self._pending_new_kv_cache_block_num} is less than the current kv cache size: {self.kv_cache_manager.num_gpu_blocks}"
            # In here, we update the layer configuration to the next configuration
            # In the next scheduling step, we will use the next configuration
            self.update_layer_config(self._pending_pp_layer_config)

            self._pending_pp_layer_config = None
            self._pending_new_kv_cache_block_num = 0
            self.migration_in_process = False
            self.sender_list_during_migration = None
            self.receiver_list_during_migration = None
            # self.total_migration_tokens = self.num_tokens_for_migration 
            self.num_tokens_for_migration = 0

        return output

    def re_initialize_kv_cache_manager(self,  kv_cache_config: KVCacheConfig):
        # When doing naive stop and go layer migration, we free the old kv cache
        # and allocate a new one after the layer is migrated. 

        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
        )
        # Free the old kv cache manager
        gc.collect()
        torch.cuda.empty_cache()

    def preempt_all_requests(self):
        while self.running:
            preempt_request = self.running.pop()
            # Mabe not free preempt request, because we will reconstruct it anyway
            self.kv_cache_manager.free(preempt_request)
            preempt_request.status = RequestStatus.PREEMPTED
            preempt_request.num_computed_tokens = 0
            self.waiting.appendleft(preempt_request)

    def dynamic_schedule(self) -> DynamicSchedulerOutput:
        with self.lock:
            return self._schedule()

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()
        self._debug_request_free_epoch += 1
        request_free_epoch = self._debug_request_free_epoch
        self._debug_request_free_epochs[
            request.request_id] = request_free_epoch
        if _autoscaling_kvcached_debug_enabled():
            logger.warning(
                "[KVCACHED_REQUEST_FREE_TRACE] phase=request_free_enter "
                "trace_ns=%s request_free_epoch=%s block_free_epoch=%s "
                "req_id=%s status=%s",
                time.time_ns(), request_free_epoch,
                self._debug_block_free_epoch,
                _short_req_id(request.request_id), request.status)

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        self._cached_reqs_data.pop(request.request_id, None)
        self.finished_req_ids.add(request.request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        if not _autoscaling_kvcached_debug_enabled():
            return super()._free_blocks(request)

        self._debug_block_free_epoch += 1
        block_free_epoch = self._debug_block_free_epoch
        request_free_epoch = self._debug_request_free_epochs.get(
            request.request_id, -1)
        self._log_kvcached_free_invariant(request, "before_free")
        block_ids = getattr(request, "block_ids", None)
        block_ids_trace = [
            [int(block_id) for block_id in block_group]
            for block_group in (block_ids or [])
        ]
        logger.warning(
            "[KVCACHED_REQUEST_FREE_TRACE] phase=before_free trace_ns=%s "
            "request_free_epoch=%s block_free_epoch=%s req_id=%s "
            "block_ids=%s",
            time.time_ns(), request_free_epoch, block_free_epoch,
            request.request_id, block_ids_trace)
        try:
            return super()._free_blocks(request)
        finally:
            logger.warning(
                "[KVCACHED_REQUEST_FREE_TRACE] phase=after_free trace_ns=%s "
                "request_free_epoch=%s block_free_epoch=%s req_id=%s "
                "block_ids=%s",
                time.time_ns(), request_free_epoch, block_free_epoch,
                request.request_id, block_ids_trace)
            self._log_kvcached_free_invariant(request, "after_free")

    def _live_kvcached_block_refs(
        self,
        exclude_req_id: Optional[str] = None,
    ) -> dict[int, list[str]]:
        manager = getattr(self.kv_cache_manager, "single_type_manager", None)
        req_to_blocks = getattr(manager, "req_to_blocks", {})
        live_refs: dict[int, list[str]] = defaultdict(list)
        for req_id, blocks in req_to_blocks.items():
            if req_id == exclude_req_id:
                continue
            req = self.requests.get(req_id)
            if req is None or req.is_finished():
                continue
            for block in blocks:
                live_refs[int(block.block_id)].append(_short_req_id(req_id))
        return live_refs

    def _log_kvcached_free_invariant(
        self,
        request: Request,
        phase: str,
    ) -> None:
        manager = getattr(self.kv_cache_manager, "single_type_manager", None)
        req_to_blocks = getattr(manager, "req_to_blocks", {})
        req_blocks = list(req_to_blocks.get(request.request_id, []))
        req_block_ids = [int(block.block_id) for block in req_blocks]
        physical_candidates = [
            int(block.block_id)
            for block in req_blocks
            if int(getattr(block, "ref_cnt", 0)) == 1
        ]
        live_refs = self._live_kvcached_block_refs(
            exclude_req_id=request.request_id)
        overlap = {
            block_id: live_refs[block_id]
            for block_id in physical_candidates
            if block_id in live_refs
        }
        if overlap:
            logger.error(
                "[KVCACHED_SCHED_FREE_OVERLAP] phase=%s trace_ns=%s "
                "req_id=%s status=%s candidate_blocks=%s overlap=%s",
                phase, time.time_ns(), _short_req_id(request.request_id),
                request.status, physical_candidates[:128], overlap)
        logger.warning(
            "[KVCACHED_SCHED_FREE_INVARIANT] phase=%s trace_ns=%s req_id=%s "
            "status=%s req_blocks=%d physical_candidates=%d "
            "scheduler_live_blocks=%d overlap_count=%d "
            "sample_req_blocks=%s sample_candidates=%s sample_overlap=%s",
            phase, time.time_ns(), _short_req_id(request.request_id),
            request.status, len(req_block_ids), len(physical_candidates),
            len(live_refs), len(overlap), req_block_ids[:64],
            physical_candidates[:64], {
                block_id: overlap[block_id]
                for block_id in list(overlap)[:16]
            })

    def log_kvcached_deferred_release_invariant(self, reason: str) -> None:
        if not _autoscaling_kvcached_debug_enabled():
            return
        block_pool = getattr(self.kv_cache_manager, "block_pool", None)
        physical_manager = getattr(block_pool, "kv_cache_manager", None)
        if physical_manager is None:
            return
        pages = list(getattr(physical_manager, "deferred_free_page_ids", []))
        if not pages:
            logger.warning(
                "[KVCACHED_DEFERRED_RELEASE_INVARIANT] reason=%s "
                "trace_ns=%s deferred_pages=0 scheduler_live_blocks=%d",
                reason, time.time_ns(),
                len(self._live_kvcached_block_refs()))
            return
        blocks_per_page = int(
            getattr(physical_manager, "blocks_per_physical_page", 1) or 1)
        deferred_blocks: set[int] = set()
        for page_id in pages:
            start_block = int(page_id) * blocks_per_page
            deferred_blocks.update(
                range(start_block, start_block + blocks_per_page))
        live_refs = self._live_kvcached_block_refs()
        overlap = {
            block_id: live_refs[block_id]
            for block_id in sorted(deferred_blocks & set(live_refs))
        }
        if overlap:
            logger.error(
                "[KVCACHED_DEFERRED_RELEASE_OVERLAP] reason=%s trace_ns=%s "
                "blocks_per_page=%s deferred_pages=%s overlap=%s",
                reason, time.time_ns(), blocks_per_page, pages[:128], {
                    block_id: overlap[block_id]
                    for block_id in list(overlap)[:64]
                })
        logger.warning(
            "[KVCACHED_DEFERRED_RELEASE_INVARIANT] reason=%s trace_ns=%s "
            "deferred_pages=%d deferred_blocks=%d blocks_per_page=%d "
            "scheduler_live_blocks=%d overlap_count=%d sample_pages=%s "
            "sample_overlap=%s",
            reason, time.time_ns(), len(pages), len(deferred_blocks),
            blocks_per_page, len(live_refs), len(overlap), pages[:64], {
                block_id: overlap[block_id]
                for block_id in list(overlap)[:16]
            })

    def add_request(self, request: Request) -> None:
        if self._sync_drain_pending_waiting is not None:
            self._sync_drain_pending_waiting.append(request)
            self.requests[request.request_id] = request
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)
            return

        super().add_request(request)

    def begin_sync_drain(self) -> None:
        assert self._sync_drain_pending_waiting is None, (
            "sync drain is already active")
        self._sync_drain_pending_waiting = deque()

    def finish_sync_drain(self) -> None:
        if self._sync_drain_pending_waiting is None:
            return

        assert len(self.running) == 0, (
            f"sync drain finished with running requests: {self.running}")
        assert len(self.waiting) == 0, (
            f"sync drain finished with waiting requests: {self.waiting}")

        self.waiting = self._sync_drain_pending_waiting
        self._sync_drain_pending_waiting = None

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        logger.warning("Finishing requests: %s", request_ids)
        with self.lock:
            assert RequestStatus.is_finished(finished_status)
            if isinstance(request_ids, str):
                request_ids = (request_ids, )
            else:
                request_ids = set(request_ids)

            for req_id in request_ids:
                request = self.requests.get(req_id)
                if request is None:
                    # Invalid request ID.
                    continue

                if request.status == RequestStatus.RUNNING:
                    self.running.remove(request)
                elif request in self.waiting:
                    self.waiting.remove(request)
                elif (self._sync_drain_pending_waiting is not None
                      and request in self._sync_drain_pending_waiting):
                    self._sync_drain_pending_waiting.remove(request)
                else:
                    logger.warning("Request %s is not in scheduler queues",
                                   req_id)
                    continue
                request.status = finished_status
                self._free_request(request)

    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests."""
        return len(self.waiting) + len(self.running)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:        
        # Convert scheduler_output to DynamicSchedulerOutput
        assert isinstance(scheduler_output, DynamicSchedulerOutput), \
            "Expected DynamicSchedulerOutput"
        with self.lock:
            outputs = super().update_from_output(
                create_from_dynamic_scheduler_output(scheduler_output),
                model_runner_output)
            return outputs

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        
        # Get KV memory stats only if enabled (may have performance impact)
        actual_kv_mem = 0
        allocated_kv_mem = 0
        if self.vllm_config.dynamic_config.log_kv_memory_stats:
            page_size_bytes = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
            num_layers = self.vllm_config.model_config.hf_text_config.num_hidden_layers
            # Use self.requests dict to ensure consistency with req_to_blocks
            active_requests = {
                req_id: req
                for req_id, req in self.requests.items()
                if not req.is_finished()
            }
            actual_kv_mem, allocated_kv_mem = self.kv_cache_manager.get_kv_memory_stats(
                active_requests, page_size_bytes, num_layers)
        
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            gpu_cache_usage=self.kv_cache_manager.usage,
            actual_kv_memory_bytes=actual_kv_mem,
            allocated_kv_memory_bytes=allocated_kv_mem,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
        )
    def get_kv_cache_utilization(self) -> float:
        return self.kv_cache_manager.usage
    
    def shrink_block_pool(self, compacted_length: int):
        assert isinstance(self.kv_cache_manager, DynamicKVCacheManager)
        self.kv_cache_manager.shrink_kv_cache(compacted_length)

    def compact_kv_cache(self, compacted_length: int) -> bitarray:
        assert isinstance(self.kv_cache_manager, DynamicKVCacheManager)
        assert compacted_length < self.kv_cache_manager.num_gpu_blocks, f"compacted_length {compacted_length} should be smaller than current kv cache block num {self.kv_cache_manager.num_gpu_blocks}"
        bitmap = self.get_bitmap()
        self.kv_cache_manager.compact_kv_cache(compacted_length, bitmap)
        return bitmap

    def extend_block_pool(self, extended_length: int) -> None:
        assert isinstance(self.kv_cache_manager, DynamicKVCacheManager)
        assert extended_length > self.kv_cache_manager.num_gpu_blocks, f"extended_length should be larger than current kv cache block num, {extended_length}: {extended_length}, current: {self.kv_cache_manager.num_gpu_blocks}"
        self.kv_cache_manager.extend_kv_cache(extended_length)

    def get_bitmap(self) -> bitarray:
        assert isinstance(self.kv_cache_manager, DynamicKVCacheManager)
        return self.kv_cache_manager.get_bitmap()

class MigrationStatus(Enum):
    NOT_MIGRATING = 0 
    MIGRATING = 1


def create_from_dynamic_scheduler_output(dynamic_scheduler_output: DynamicSchedulerOutput) -> SchedulerOutput:
    return SchedulerOutput(
                scheduled_new_reqs=dynamic_scheduler_output.scheduled_new_reqs,
                scheduled_cached_reqs=dynamic_scheduler_output.scheduled_cached_reqs,
                num_scheduled_tokens=dynamic_scheduler_output.num_scheduled_tokens,
                total_num_scheduled_tokens=dynamic_scheduler_output.total_num_scheduled_tokens,
                scheduled_spec_decode_tokens=dynamic_scheduler_output.scheduled_spec_decode_tokens,
                scheduled_encoder_inputs=dynamic_scheduler_output.scheduled_encoder_inputs,
                num_common_prefix_blocks=dynamic_scheduler_output.num_common_prefix_blocks,
                finished_req_ids=dynamic_scheduler_output.finished_req_ids,
                free_encoder_input_ids=dynamic_scheduler_output.free_encoder_input_ids,
                structured_output_request_ids=dynamic_scheduler_output.structured_output_request_ids,
                grammar_bitmask=dynamic_scheduler_output.grammar_bitmask,
                kv_connector_metadata=dynamic_scheduler_output.kv_connector_metadata,
                scheduler_step_id=dynamic_scheduler_output.scheduler_step_id,
                current_scheduler_output_version=(
                    dynamic_scheduler_output.current_scheduler_output_version),
                scheduler_request_free_epoch=(
                    dynamic_scheduler_output.scheduler_request_free_epoch),
                scheduler_block_free_epoch=(
                    dynamic_scheduler_output.scheduler_block_free_epoch),
                autoscaling_request_state_sync=(
                    dynamic_scheduler_output.autoscaling_request_state_sync),
            )

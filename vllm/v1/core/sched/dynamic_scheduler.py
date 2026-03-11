# SPDX-License-Identifier: Apache-2.0

from regex import P
from vllm.v1.core.dynamic_kv_cache_manager import DynamicKVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from typing import List, Tuple, Optional, TypeVar, Union, Any, Dict
from vllm.v1.core.sched.snapshot import KVCacheSnapshot, KVCacheSnapshotEntry
from threading import Lock
import vllm.envs as envs
from enum import Enum
from concurrent.futures import Future
import torch
import gc
from collections import defaultdict, deque
from collections.abc import Iterable
from bitarray import bitarray

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.dynamic_config import PPLayerConfigs
from vllm.distributed.kv_events import EventPublisherFactory
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole

from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager, compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData, SchedulerOutput)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)
T = TypeVar("T")

class ChangeConfigurationType(Enum):
    NOT_CHANGING = 0
    ASYNC_CHANGING = 1
    SYNC_CHANGING = 2

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
        self.pp_layer_config_status = SchedulerPPLayerConfigStatus(layer_configs)
        
        # # Used to track the config of each request
        # self.config_request_ids: Dict[str, List[str]] = {}

        self.migration_status = MigrationStatus.NOT_MIGRATING

        # Used to track the round robin index
        # The round robin is implmented to schedule the requests from the
        # current configuration and next configuration.
        self.round_robin_index = 0
        self.running_controller = RunningQueueMigrationController()
        self.waiting_controller = WaitingQueueMigrationController()
        # self.cur_running = []
        # self.cur_waiting = deque()
        # self.next_running = []
        # self.next_waiting = deque()
        self.next_pp_layer_config = None # This is only used when switching fron old configuration to new configuration in the 
        # When scheduler executes schedule or migration operation, it needs
        # to acquire the lock.
        self.lock = Lock()
        self._migration_future: Optional[Future] = None

        # These flags are used to indicate the type of configuration change
        # Once the flag is set, the scheduler will schedule the according scheduleroutput
        # on the next scheduling step
        self.change_configuration_status = ChangeConfigurationType.NOT_CHANGING
        self.next_new_kv_cache_block_num = 0

        self.migration_in_process = False # 控制在每一次scheduler schedule的时候是否需要同时发送slot_mapping
        self.num_tokens_for_migration = 0 # 记录已经发送的slot数量，用来和worker端已处理的slot数量进行对比

        self.cur_scheduler_output_version = 0

        # 用以记录在迁移过程中，哪些rank是sender，哪些rank是receiver
        self.sender_list_during_migration: Optional[set[int]] = None 
        self.receiver_list_during_migration: Optional[set[int]] = None

    def async_change_configuration(self, pp_layer_config: List[Tuple[int,int]], new_kv_cache_block_num: int):
        self.change_configuration_status = ChangeConfigurationType.ASYNC_CHANGING
        assert self.next_new_kv_cache_block_num == 0 
        assert self.next_pp_layer_config is None
        self.next_pp_layer_config = pp_layer_config
        self.next_new_kv_cache_block_num = new_kv_cache_block_num

    def start_migration(self, sender_list: list[int], receiver_list: list[int], should_increase_scheduler_output_version: bool) -> Union[list[int], None]:
        with self.lock:
            self.migration_in_process = True
            self.sender_list_during_migration = set(sender_list) 
            self.receiver_list_during_migration = set(receiver_list)
            is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
            if should_increase_scheduler_output_version:
                self.cur_scheduler_output_version += 1
            if is_flexi:
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
        self.change_configuration_status = ChangeConfigurationType.SYNC_CHANGING
        self.update_layer_config(pp_layer_config)
        # if new_kv_cache_block_num > self.kv_cache_manager.num_gpu_blocks:
        #     self.extend_kv_cache(new_kv_cache_block_num)

    def v1_start_migration(self, layer_config: List[Tuple[int,int]])->Future:
        # Start the migration process
        # This function will only be called when doing concurrent serving old and new requests in migration 
        # In the latest style of migration, we don't use this function
        # 1. When doing migration, we maintain two sets of running 
        #    and waiting requests. 
        # 2. cur_waiting and cur_running are the requests that are 
        #    running when the migration starts. 
        # 3. next_waiting and next_running are the new requests that 
        #    will be run when the migraion is in process. 
        assert self.migration_status == MigrationStatus.NOT_MIGRATING, \
            "Migration is already in process, cannot start a new one"
        with self.lock:
            self.migration_status = MigrationStatus.MIGRATING
            # self.next_running = []
            # self.next_waiting = self.cur_waiting
            # self.cur_waiting = deque()
            self.running_controller.start_migration()
            self.waiting_controller.start_migration()
            self._add_layer_config(layer_config)
            self._migration_future = Future()
            return self._migration_future

    def update_layer_config(self, layer_config: List[Tuple[int,int]]):
        # Update the configuration of layers stored in the scheduler
        self.pp_layer_config_status.update_pp_layer_config(layer_config)

    def _complete_migration(self):
        assert self.migration_status == MigrationStatus.MIGRATING
        assert len(self.running_controller.get_cur()[1]) == 0
        assert len(self.waiting_controller.get_cur()) == 0
        self.running_controller.finish_migration()
        self.waiting_controller.finish_migration()

        # Update the running and waiting status
        self.migration_status = MigrationStatus.NOT_MIGRATING

        # Update the pp layer config status
        self.pp_layer_config_status.finish_migration_and_update_config()

        self.round_robin_index = 0

         # resolve the future
        assert self._migration_future is not None and not self._migration_future.done()
        self._migration_future.set_result(True)

    def _schedule(self,is_old_request: bool) -> DynamicSchedulerOutput:
        """
        In this function, we control the behavior of scheduler during the migration process
        There are three types of migration process at the moment:
            1. sync migration with kv cache transfer 
            2. async migration with kv cache transfer
            3. drain-out style of migration (depreciated)

        sync migration: 
            Drain out the running batches and interupt the inference process.
            Change the configuration of gpus and scheduler synchronously.
            In the next scheduling step, the scheduler will schedule the requests using the new configuration. Also scheduled with the next kv cache block num to tell the gpu worker to resize the kv cache *before* the execution of inferenc of inferencee.
        async migration:
            Don't interupt the inference process, the kv tensor is transmitting asynchronously.
            When the timing is right, we inject a sync msg to scheduleroutput to tell the gpu worker sending out all the kv patches. After this async msg, we change the configuration of the scheduler.
            In the next scheduling step, the scheduler is scheduling with the new configuration.
        """
        if is_old_request:
            id, cur_running = self.running_controller.get_cur()
            cur_waiting = self.waiting_controller.get_cur()
            self.running = cur_running
            self.waiting = cur_waiting 
            pp_layer_config = self.pp_layer_config_status.get_cur_pp_layer_config()
            scheduler_output = super().schedule()

            self.waiting_controller.cur = self.waiting
            self.running_controller.set_queue_by_id(id, self.running)
        else:
            id, next_running = self.running_controller.get_next()
            next_waiting = self.waiting_controller.get_next()

            self.running = next_running
            self.waiting = next_waiting
            pp_layer_config = self.pp_layer_config_status.get_next_pp_layer_config()
            scheduler_output = super().schedule()

            self.waiting_controller.next = self.waiting
            self.running_controller.set_queue_by_id(id, self.running)

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
                request_queue_id=id,
                current_scheduler_output_version=self.cur_scheduler_output_version,
                is_sync_after_migration=True if self.change_configuration_status == ChangeConfigurationType.ASYNC_CHANGING else False,
                total_migration_tokens=self.num_tokens_for_migration + scheduler_output.total_num_scheduled_tokens,
                # total_migration_tokens=self.num_tokens_for_migration,
                new_kv_cache_block_num=self.next_new_kv_cache_block_num,
                migration_in_process=self.migration_in_process,
                sender_list= self.sender_list_during_migration,
                receiver_list= self.receiver_list_during_migration,
                # slot_mapping = self.get_slot_mapping_from_reqs(scheduler_output.scheduled_new_reqs) if self.sending_slot_mapping else None,
            )
        # scheduled token为0的请求不应该发送给worker，因此在这里我们跳过后续的asynchronise处理
        if output.total_num_scheduled_tokens == 0:
            return output

        if self.migration_in_process:
            self.num_tokens_for_migration += output.total_num_scheduled_tokens
            logger.info(f"[num tokens]: scheduler output token {output.total_num_scheduled_tokens} added, total tokens for migration: {self.num_tokens_for_migration} ")


        if self.change_configuration_status == ChangeConfigurationType.ASYNC_CHANGING:
            self.change_configuration_status = ChangeConfigurationType.NOT_CHANGING
            assert self.next_pp_layer_config is not None
            # 在最后的阶段，只有可能是expand，不可能shrink
            assert self.next_new_kv_cache_block_num >= self.kv_cache_manager.num_gpu_blocks, f"next_new_kv_cache_block_num: {self.next_new_kv_cache_block_num} is less than the current kv cache size: {self.kv_cache_manager.num_gpu_blocks}"
            # In here, we update the layer configuration to the next configuration
            # In the next scheduling step, we will use the next configuration
            self.update_layer_config(self.next_pp_layer_config)

            self.next_pp_layer_config = None
            self.next_new_kv_cache_block_num = 0
            self.migration_in_process = False
            self.sender_list_during_migration = None
            self.receiver_list_during_migration = None
            # self.total_migration_tokens = self.num_tokens_for_migration 
            self.num_tokens_for_migration = 0

        if self.change_configuration_status == ChangeConfigurationType.SYNC_CHANGING:
            assert self.next_new_kv_cache_block_num == 0
            self.change_configuration_status = ChangeConfigurationType.NOT_CHANGING
        return output

    def get_kv_cache_snapshot(self) -> KVCacheSnapshot:
        """Capture a snapshot of current KV cache state per request.

        Returns a mapping: request_id -> {"num_computed_tokens": int,
                                           "block_ids": list[list[int]]}
        It includes requests from both current/next running & waiting queues
        when migration is ongoing; otherwise only the current queues.
        """
        entries: dict[str, KVCacheSnapshotEntry] = {}
        with self.lock:
            # Collect from both CUR and NEXT queues to be robust during migration
            cur_run_id, cur_running = self.running_controller.get_cur()
            cur_waiting = self.waiting_controller.get_cur()
            next_run_id, next_running = self.running_controller.get_next()
            next_waiting = self.waiting_controller.get_next()
            # 此时我们并不涉及到next的迁移，所以next_running和next_waiting都为空
            assert next_run_id is None and next_waiting is None

            def _collect(reqs: list[Request]) -> None:
                for req in reqs:
                    req_id = req.request_id
                    # Some waiting requests may have no allocated blocks yet
                    if req_id in self.kv_cache_manager.single_type_manager.req_to_blocks:
                        block_ids = self.kv_cache_manager.get_block_ids(req_id)
                    else:
                        block_ids = []
                    entries[req_id] = KVCacheSnapshotEntry(
                        request_id=req_id,
                        num_computed_tokens=int(req.num_computed_tokens),
                        block_ids=block_ids,
                    )

            _collect(cur_running)
            _collect(list(cur_waiting))

        return KVCacheSnapshot(entries_by_id=entries)

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
        import time
        schedule_start_time = time.time()
        with self.lock:
            if self.migration_status == MigrationStatus.NOT_MIGRATING:
                _, next_running  = self.running_controller.get_next()
                assert len(next_running) == 0, f"next_running should be empty:{next_running}"
                assert len(self.waiting_controller.get_next()) == 0, "next_waiting should be empty"
                assert self.round_robin_index == 0, "round_robin_index should be 0"
                # Log queue status
                _, cur_running = self.running_controller.get_cur()
                cur_waiting = self.waiting_controller.get_cur()
                for req in list(cur_waiting)[:5]:  # Log first 5 waiting requests
                    wait_time = (time.time() - req.arrival_time) * 1000
                    # logger.info(f"[ttft_trace] Waiting request {req.request_id}: wait_time={wait_time:.2f}ms, num_tokens={req.num_tokens}, num_computed_tokens={req.num_computed_tokens}")
                scheduler_output = self._schedule(is_old_request=True)
                schedule_total_time = (time.time() - schedule_start_time) * 1000
                # logger.info(f"[ttft_trace] Scheduler: dynamic_schedule() took {schedule_total_time:.2f}ms")
                return scheduler_output

            _, cur_running = self.running_controller.get_cur()
            if len(cur_running) == 0:
                self._complete_migration()
                return self._schedule(is_old_request=True)
                                     
            scheduler_output = self._schedule(is_old_request=self.round_robin_index == 0)
            self.round_robin_index = (self.round_robin_index + 1) % 2
            return scheduler_output

    def add_request(self, request: Request) -> None:
        if self.migration_status == MigrationStatus.MIGRATING:
            self.waiting = self.waiting_controller.get_next()
            super().add_request(request)
            self.waiting_controller.next = self.waiting
        else:
            self.waiting = self.waiting_controller.get_cur()
            super().add_request(request)
            self.waiting_controller.cur = self.waiting

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
                    _, cur_running = self.running_controller.get_cur()
                    _, next_running = self.running_controller.get_next()
                    cur_running.remove(request)
                    next_running.remove(request)
                else:
                    self.waiting_controller.cur.remove(request)
                    self.waiting_controller.next.remove(request)
                request.status = finished_status
                self._free_request(request)

    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests."""
        return self.running_controller.get_total_length() + \
            self.waiting_controller.get_total_length()

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:        
        # Convert scheduler_output to DynamicSchedulerOutput
        assert isinstance(scheduler_output, DynamicSchedulerOutput), \
            "Expected DynamicSchedulerOutput"
        with self.lock:
            self.running = self.running_controller.get_by_id(scheduler_output.request_queue_id)
            if self.migration_status == MigrationStatus.MIGRATING:
                self.waiting = self.waiting_controller.get_next()
            else:
                self.waiting = self.waiting_controller.get_cur()
            output =  super().update_from_output(
            create_from_dynamic_scheduler_output(scheduler_output), 
            model_runner_output)
            self.running_controller.set_queue_by_id(
                scheduler_output.request_queue_id, self.running)

            if self.migration_status == MigrationStatus.MIGRATING:
                self.waiting_controller.next = self.waiting
            else:
                self.waiting_controller.cur = self.waiting
            return output

            # self.running = self.cur_running
            # self.waiting = self.cur_waiting
            # output =  super().update_from_output(
            # create_from_dynamic_scheduler_output(scheduler_output), 
            # model_runner_output)
            # self.cur_running = self.running
            # self.cur_waiting = self.waiting

            # return output

    def _add_layer_config(self, layer_config: List[Tuple[int,int]]):
            self.pp_layer_config_status.update_with_next_pp_layer_config(layer_config)

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
            actual_kv_mem, allocated_kv_mem = self.kv_cache_manager.get_kv_memory_stats(
                self.requests, page_size_bytes, num_layers)
        
        return SchedulerStats(
            num_running_reqs=self.running_controller.get_total_length(),
            num_waiting_reqs=self.waiting_controller.get_total_length(),
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

class SchedulerPPLayerConfigStatus:
    #   Keep two list, first one represent current pp layer configuration, 
    #   second one represent next pp layer configuration
    #   For each configuration, we keep a list of layer range ids
    #   .e.g: 
    #       Assume we have a 10 layers model and a size 2 pipeline deployment 
    #       we want pipeline stage 1 has first 5 layers, stage 2 has last 5 layers, 
    #       then the pp_layer_configs will be:
    #           cur_pp_layer_configs: [(0, 4), (5, 9)]
    #           next_pp_layer_configs: None # Haven't update yet
    #       After a while, we decide to update the layer configuration with 3 layers 
    #       in stage 1 and 2 layers in stage 2. When live migration is processing, 
    #       the pp_layer_configs will be:
    #           cur_pp_layer_configs: [(0, 4), (5, 9)]
    #           next_pp_layer_configs: [(0, 3), (4, 9)]
    #       After live migration is finished, the pp_layer_configs will be updated to:
    #           cur_pp_layer_configs: [(0, 3), (4, 9)]
    #           next_pp_layer_configs: None
    def __init__(self, pp_layer_configs: Optional[List[Tuple[int,int]]] = None):
        self.pp_layer_configs = PPLayerConfigs()
        if pp_layer_configs is not None:
            self.pp_layer_configs.set_pp_layer_config("cur", pp_layer_configs)

    def update_with_next_pp_layer_config(self, pp_layer_configs: List[Tuple[int,int]]):
        self.pp_layer_configs.set_pp_layer_config("next", pp_layer_configs)

    def finish_migration_and_update_config(self):
        self.pp_layer_configs.set_pp_layer_config("cur", self.pp_layer_configs.get_pp_layer_config("next"))
        self.pp_layer_configs.delete_pp_layer_config("next")

    def get_cur_pp_layer_config(self) -> List[Tuple[int,int]]:
        return self.pp_layer_configs.get_pp_layer_config("cur")
        
    def get_next_pp_layer_config(self) -> List[Tuple[int,int]]:
        return self.pp_layer_configs.get_pp_layer_config("next")

    def update_pp_layer_config(self, pp_layer_configs: List[Tuple[int,int]]):
        # At the moment, we only support update current pp layer config
        # without next pp layer config
        # This is because we only update current pp layer configuration 
        # when using v0 style migration 
        assert not self.pp_layer_configs.is_key_exist("next"), "Next pp layer config is not None"
        self.pp_layer_configs.set_pp_layer_config("cur", pp_layer_configs)

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
                kv_connector_metadata=dynamic_scheduler_output.kv_connector_metadata
            )



# 我希望构造一个MigrationList的数据结构
# 这个数据结构是用于对当前数据结构内的collection进行迁移的
# 我们假设collection内的元素都是可被消费的
# 所以这个迁移的本质是，在迁移的过程中，保持旧的collection中的元素可以被消费
# 新的元素被加入的时候会被添加到新的collection中
# 当所有旧的collection中的元素都被消费完毕后，新的collection就成为了当前的collection，迁移操作完毕

# 这个数据结构有一个catch，即当迁移操作开始的时候，新的元素会被添加到新的collection中，而此时迁移操作可能会结束
# 在collection中的新元素会变成旧元素。当消费者开始进行消费的时候，消费者需要能够知道这个元素在哪一个collection中
# 所以我们需要一个迁移列表，记录每个元素的collection id
# 当消费者消费的时候，需要知道这个元素的collection id，使用collection id进行消费

# 关于数据结构本身的要求：
# 支持使用数据结构本身可以支持list
class RunningQueueMigrationController():
    def __init__(self):

        self.migration_in_progress = False
        self.head_id = 0

        self.id_map: dict[int, list] = {self.head_id: []}

    def start_migration(self):
        assert not self.migration_in_progress, "Migration already in progress"
        self.migration_in_progress = True
        self.head_id += 1
        self.id_map[self.head_id] = []

    def get_cur(self) -> Tuple[int, list]:
        if self.migration_in_progress:
            return self.head_id - 1, self.id_map[self.head_id - 1] 
        else:
            return self.head_id, self.id_map[self.head_id]

    def get_next(self) -> Tuple[int, list]:
        if self.migration_in_progress:
            return self.head_id, self.id_map[self.head_id]
        else:
            return self.head_id+1, []

    def finish_migration(self):
        assert self.migration_in_progress, "No migration in progress"

        del self.id_map[self.head_id-1] # remove the old list
        self.migration_in_progress = False

    def get_by_id(self, id: int) -> list:
        assert id in self.id_map, f"ID {id} not found in id_map"
        return self.id_map[id]

    def set_queue_by_id(self, id: int, queue: list):
        assert id in self.id_map, f"ID {id} not found in id_map"
        self.id_map[id] = queue

    def get_total_length(self) -> int:
        """Get the total length of the current queue."""
        sum = 0
        for id, queue in self.id_map.items():
            sum += len(queue)
        return sum

class WaitingQueueMigrationController():
    def __init__(self):

        self.migration_in_progress = False
        self.head_id = 0

        self.cur = deque()
        self.next = deque()

    def start_migration(self):
        assert not self.migration_in_progress, "Migration already in progress"
        self.next = self.cur
        self.cur = deque()
        self.migration_in_progress = True

    def get_cur(self) -> deque:
        return self.cur

    def get_next(self) -> deque:
        return self.next

    def finish_migration(self):
        assert self.migration_in_progress, "No migration in progress"
        self.cur = self.next
        self.next = deque()
        self.migration_in_progress = False
    def get_total_length(self) -> int:
        """Get the total length of the current queue."""
        return len(self.cur) + len(self.next)
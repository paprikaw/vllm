from regex import P
from vllm.v1.core.sched.scheduler import Scheduler
from typing import List, Tuple, Optional, TypeVar, Union
from threading import Lock
import vllm.envs as envs
from enum import Enum
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
from vllm.v1.core.sched.output import SchedulerOutput
from concurrent.futures import Future
from collections import deque
from vllm.v1.request import Request
from vllm.logger import init_logger
from collections import deque
from collections.abc import Iterable
from typing import Optional, Union

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.dynamic_config import PPLayerConfigs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
import torch
import gc
logger = init_logger(__name__)
T = TypeVar("T")


class DynamicScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize pp layer config status
        assert envs.VLLM_PP_LAYER_PARTITION is not None, "VLLM_PP_LAYER_PARTITION is not set"
        partition_list_str: str = envs.VLLM_PP_LAYER_PARTITION 
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

        # When scheduler executes schedule or migration operation, it needs
        # to acquire the lock.
        self.lock = Lock()
        self._migration_future: Optional[Future] = None


    def start_migration(self, layer_config: List[Tuple[int,int]])->Future:
        # Start the migration process
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

        return DynamicSchedulerOutput(
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
                    request_queue_id=id
                )

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
            # self.kv_cache_manager.free(preempt_request)
            preempt_request.status = RequestStatus.PREEMPTED
            preempt_request.num_computed_tokens = 0
            self.waiting.appendleft(preempt_request)

    def schedule(self) -> DynamicSchedulerOutput:
        with self.lock:
            if self.migration_status == MigrationStatus.NOT_MIGRATING:
                _, next_running  = self.running_controller.get_next()
                assert len(next_running) == 0, f"next_running should be empty:{next_running}"
                assert len(self.waiting_controller.get_next()) == 0, "next_waiting should be empty"
                assert self.round_robin_index == 0, "round_robin_index should be 0"
                scheduler_output = self._schedule(is_old_request=True)
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
        return SchedulerStats(
            num_running_reqs=self.running_controller.get_total_length(),
            num_waiting_reqs=self.waiting_controller.get_total_length(),
            gpu_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
        )
    def get_kv_cache_utilization(self) -> float:
        return self.kv_cache_manager.usage

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
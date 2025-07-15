from vllm.v1.core.sched.scheduler import Scheduler
from typing import List, Tuple, Dict, Optional
from threading import Lock
import vllm.envs as envs
from enum import Enum
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
from vllm.v1.core.sched.output import SchedulerOutput
from dataclasses import asdict
from concurrent.futures import Future
from collections import deque
from vllm.v1.request import Request
from vllm.logger import init_logger

logger = init_logger(__name__)


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
        self.pp_layer_config_status = PPLayerConfigStatus(layer_configs)

        # # Used to track the config of each request
        # self.config_request_ids: Dict[str, List[str]] = {}

        self.migration_status = MigrationStatus.NOT_MIGRATING

        # Used to track the round robin index
        # The round robin is implmented to schedule the requests from the
        # current configuration and next configuration.
        self.round_robin_index = 0
        self.cur_running: Optional[List[Request]] = None
        self.cur_waiting: Optional[deque] = None
        self.next_running: Optional[List[Request]] = None
        self.next_waiting: Optional[deque] = None

        # When scheduler executes schedule or migration operation, it needs
        # to acquire the lock.
        self.lock = Lock()
        self._migration_future: Optional[Future] = None


    def start_migration(self, layer_config: List[Tuple[int,int]])->Future:
        logger.info(f"scheduler start migration before lock")
        with self.lock:
            logger.info(f"scheduler start migration, layer_config: {layer_config}")
            self.migration_status = MigrationStatus.MIGRATING
            self.cur_running = self.running
            self.cur_waiting = deque()
            self.next_running = []
            self.next_waiting = self.waiting
            self._add_layer_config(layer_config)
            self._migration_future = Future()
            return self._migration_future

    def _complete_migration(self):
        assert self.migration_status == MigrationStatus.MIGRATING
        assert self.cur_running is not None
        assert self.cur_waiting is not None and len(self.cur_waiting) == 0
        assert self.next_running is not None
        assert self.next_waiting is not None

        if len(self.cur_running) != 0:
            return False
        # Update the running and waiting status
        self.migration_status = MigrationStatus.NOT_MIGRATING
        # Update the running and waiting queue 
        self.running = self.next_running
        self.waiting = self.next_waiting

        # Update the pp layer config status
        self.pp_layer_config_status.finish_migration_and_update_config()

        self.cur_running = None
        self.cur_waiting = None
        self.next_running = None
        self.next_waiting = None
        self.round_robin_index = 0

         # resolve the future
        assert self._migration_future is not None and not self._migration_future.done()
        self._migration_future.set_result(True)

    def schedule(self) -> DynamicSchedulerOutput:
        with self.lock:
            if self.migration_status == MigrationStatus.MIGRATING:
                assert self.cur_running is not None, "cur_running is not set"
                assert self.cur_waiting is not None, "cur_waiting is not set"
                assert self.next_running is not None, "next_running is not set"
                assert self.next_waiting is not None, "next_waiting is not set"

                if len(self.cur_running) == 0:
                    self._complete_migration()
                    return create_dynamic_scheduler_output(super().schedule(), self.pp_layer_config_status.get_cur_pp_layer_config())

                if self.round_robin_index == 0:
                    self.running = self.cur_running 
                    self.waiting = self.cur_waiting
                    pp_layer_config = self.pp_layer_config_status.get_cur_pp_layer_config()
                else:
                    self.running = self.next_running
                    self.waiting = self.next_waiting
                    pp_layer_config = self.pp_layer_config_status.get_next_pp_layer_config()
                self.round_robin_index = (self.round_robin_index + 1) % 2
            else:
                assert self.cur_running is None, "cur_running should be None"
                assert self.cur_waiting is None, "cur_waiting should be None"
                assert self.next_running is None, "next_running should be None"
                assert self.next_waiting is None, "next_waiting should be None"
                assert self.round_robin_index == 0, "round_robin_index should be 0"
                pp_layer_config = self.pp_layer_config_status.get_cur_pp_layer_config()


        return create_dynamic_scheduler_output(super().schedule(), pp_layer_config)

    def _add_layer_config(self, layer_config: List[Tuple[int,int]]):
            self.pp_layer_config_status.update_with_next_pp_layer_config(layer_config)


class PPLayerConfigStatus:
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

    pp_layer_configs: Dict[str, Optional[List[Tuple[int,int]]]]

    def __init__(self, pp_layer_config:List[Tuple[int,int]]):
        self.pp_layer_configs = {
            "cur": pp_layer_config,
            "next": None
        }
        self.request_id_config_map = {}

    def update_with_next_pp_layer_config(self, pp_layer_configs: List[Tuple[int,int]]):
        self.pp_layer_configs["next"] = pp_layer_configs

    def finish_migration_and_update_config(self):
        self.pp_layer_configs["cur"] = self.pp_layer_configs["next"]
        self.pp_layer_configs["next"] = None

    def get_cur_pp_layer_config(self) -> List[Tuple[int,int]]:
        assert self.pp_layer_configs["cur"] is not None, "Current pp layer config is not set"
        return self.pp_layer_configs["cur"]
        
    def get_next_pp_layer_config(self) -> List[Tuple[int,int]]:
        assert self.pp_layer_configs["next"] is not None, "Next pp layer config is not set"
        return self.pp_layer_configs["next"]

class MigrationStatus(Enum):
    NOT_MIGRATING = 0 
    MIGRATING = 1

def create_dynamic_scheduler_output(scheduler_output: SchedulerOutput, pp_layer_config: List[Tuple[int,int]]) -> DynamicSchedulerOutput:
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
            )

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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple
from .output import NewRequestData, CachedRequestData
import numpy as np
import numpy.typing as npt

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorMetadata)

@dataclass
class DynamicSchedulerOutput():
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: list[CachedRequestData]

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of (req_id, encoder_input_index) tuples.
    # Used to free the encoder cache.
    free_encoder_input_ids: list[tuple[str, int]]

    # Dict of request ids to their index within the batch
    # for filling the next token bitmask
    structured_output_request_ids: dict[str, int]

    # Execution layer configs for each pipeline workers.
    # Here each pipeline worker is able to execute a subset of the layers.
    # This allows us to perform pipeline parallelism with different configurations.
    #   .e.g With a model with 12 layers, we can have two workers with below layers:
    #       worker 0: [0, 1, 2, 3, 4, 5, 6]
    #       worker 1: [6, 7, 8, 9, 10, 11]
    #   With this setup, we can have two different configurations:
    #     Configuration 1: [0, 1, 2, 3, 4, 5] -> [6, 7, 8, 9, 10, 11]
    #     Configuration 2: [0, 1, 2, 3, 4, 5, 6] -> [7, 8, 9, 10, 11]
    #   This is useful for the case where we want to perform pipeline parallelism
    #   with different configurations.
    pp_layer_config: list[Tuple[int, int]]
    
    # Whether the scheduling output is from before the migration was started.  
    request_queue_id: int

    current_scheduler_output_version: int

    
#     slot_mapping: Optional[list[int]] = None
    # the bitmask for the whole batch
    grammar_bitmask: Optional[npt.NDArray[np.int32]]

    total_migration_tokens: int = 0

    # KV Cache Connector metadata.
    kv_connector_metadata: Optional[KVConnectorMetadata] = None

    # Whether the scheduling output is from before the migration was started.  
    is_sync_after_migration: bool = False

    migration_in_process: bool = False

    new_kv_cache_block_num: int = 0

    sender_list: Optional[set[int]] = None
    receiver_list: Optional[set[int]] = None

    # Debug-only PP NCCL tracing metadata. These fields let the dynamic
    # executor correlate each in-flight micro-batch across PP NCCL send/recv
    # logs without changing the scheduler contract.
    pp_nccl_seq: int = -1
    pp_nccl_active_ranks: Optional[tuple[int, ...]] = None
    pp_nccl_request_ids: tuple[str, ...] = ()
    pp_nccl_config_fingerprint: str = ""

    # Autoscaling active-rank update carried by the sync batch. The executor
    # starts a lightweight Ray actor chain so workers can update PP routing
    # before the scheduler pause is released.
    autoscaling_activate_pp_ranks: Optional[list[int]] = None
    autoscaling_activate_pp_ranks_generation: int = -1

    # Autoscaling request-state handoff carried by the first target-topology
    # batch. This replaces the separate export/import collective RPC path:
    # the first worker that already has the request states attaches them to
    # this scheduler output, and newly activated ranks import them before
    # running the batch.
    autoscaling_request_state_sync: bool = False
    autoscaling_request_states: Optional[dict[str, Any]] = None
    autoscaling_request_state_source_rank: int = -1

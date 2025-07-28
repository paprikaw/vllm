# SPDX-License-Identifier: Apache-2.0
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
from typing import Any, Callable, Optional, TypeVar, Union

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
from vllm.version import __version__ as VLLM_VERSION
from .utils import get_new_layer_config_with_migration_action

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar('_R')  # Return type for collective_rpc

class DynamicEngineCore(EngineCore):

    def _initialize_kv_caches(
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

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        # This is different from the scheduler's migration status
        # - Scheduler migration status is used to trace whether the old
        #   requests before migration are finished
        # - Engine migration status is used to trace whether the migration is done.
        self.migration_status = MigrationStatus.NOT_MIGRATING

        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def migrate_layers(self, rank_from: int, rank_to: int, num_layers: int) -> Future:
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
        future = self.scheduler.start_migration(next_layer_config)
        end = time.time()
        logger.info(f"Added layers to {rank_to} took {end - start} seconds")
        return future
    
    def done_migration(self):
        assert self.migration_status == MigrationStatus.MIGRATING
        assert self.migrating_layers is not None
        assert self.rank_from is not None
        assert isinstance(self.model_executor, DynamicRayDistributedExecutor)
        self.model_executor.remove_layers(self.rank_from, self.migrating_layers)
        self.model_executor.release_kv_cache_for_layers(self.rank_from, self.migrating_layers)
        self.migration_status = MigrationStatus.NOT_MIGRATING
        self.layers_to_be_removed = None
        self.rank_from = None
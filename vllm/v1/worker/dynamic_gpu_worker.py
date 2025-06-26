import gc
import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils import GiB_bytes
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.worker_base import WorkerBase
from .utils import get_total_gpu_memory
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.dynamic_gpu_model_runner import DynamicGPUModelRunner
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment, _check_if_gpu_supports_dtype
logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput

class DynamicGPUWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_device(self):
        # This function is copied from Worker.init_device
        # We override this becuase this function is used to initialize the GPUModelRunner and we want to use our own DynamicGPUModelRunner

        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            assert isinstance(self.model_config.dtype, torch.dtype)
            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)

        assert self.model_config.seed is not None
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner: DynamicGPUModelRunner = DynamicGPUModelRunner(
            self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def add_layers(self, layers: Tuple[int, int]) -> None:
        logger.info(f"Add Model Layers: {layers}")
        self.model_runner.add_model_layers(layers)
        # self.model_runner.load_model_layers(layer_names)

    def get_kv_cache_spec_for_layers(self, rank: int, layer_range: Tuple[int, int]) -> dict[str, KVCacheSpec]:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip getting kv cache spec for layers")
            return {}
        logger.debug(f"Get kv cache spec for layers: {layer_range}")
        return self.model_runner.get_kv_cache_spec_for_layers(layer_range)

    def initialize_kv_cache_for_layers(self, kv_cache_specs: dict[str, KVCacheSpec], 
                                       kv_cache_size: int, 
                                       kv_cache_num_blocks: int) -> None:
        self.model_runner.initialize_kv_cache_for_layers(kv_cache_specs, kv_cache_size, kv_cache_num_blocks)
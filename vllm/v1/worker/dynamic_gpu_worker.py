import gc
import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.distributed
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.dynamic_gpu_model_runner import DynamicGPUModelRunner
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment, _check_if_gpu_supports_dtype
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
from dataclasses import asdict
from vllm.sequence import IntermediateTensors
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
import time
logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput

class DynamicGPUWorker(Worker):
    def __init__(self, *args, **kwargs):
        logger.info("start to initialize")
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
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "DynamicSchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        logger.info(f"start to execute_model in gpu worker")
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(
            SchedulerOutput(**asdict(scheduler_output)), 
            scheduler_output.pp_layer_config[self.rank],
            intermediate_tensors)

        parallel_config = self.vllm_config.parallel_config
        if parallel_config.distributed_executor_backend != "external_launcher" \
            and not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return None
        assert isinstance(output, ModelRunnerOutput)
        return output if self.is_driver_worker else None


    @torch.inference_mode()
    def get_current_available_memory(self) -> int:
        """Get the current available memory in bytes.
        """
        free_gpu_memory, _ = torch.cuda.mem_get_info()
        return int(free_gpu_memory)

    def add_layers(self, rank: int, layers: Tuple[int, int]) -> None:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip adding model layers")
            return None
        logger.info(f"Add Model Layers: {layers}")
        self.model_runner.add_layers(layers)

    def remove_layers(self, rank: int, layers: Tuple[int, int]) -> None:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip removing model layers")
            return None
        logger.info(f"Remove Model Layers: {layers}")
        self.model_runner.remove_layers(layers)

    def release_kv_cache_for_layers(self, rank: int, layers: Tuple[int, int]) -> None:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip releasing kv cache for layers")
            return None
        logger.info(f"Release kv cache for layers: {layers}")
        self.model_runner.release_kv_cache_for_layers(layers)

    def get_kv_cache_spec_for_layers(self, rank: int, layer_range: Tuple[int, int]) -> dict[str, KVCacheSpec]:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip getting kv cache spec for layers")
            return {}
        logger.debug(f"Get kv cache spec for layers: {layer_range}")
        return self.model_runner.get_kv_cache_spec_for_layers(layer_range)

    def initialize_kv_cache_for_layers(self, rank: int, kv_cache_specs: dict[str, KVCacheSpec], 
                                       kv_cache_size: int, 
                                       kv_cache_num_blocks: int,
                                       layers: Tuple[int, int]) -> None:
        if self.rank != rank:
            return
        self.model_runner.initialize_kv_cache_for_layers(kv_cache_specs, kv_cache_size, kv_cache_num_blocks, layers)
        return
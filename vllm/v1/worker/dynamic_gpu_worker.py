from collections import defaultdict
import gc
import os
from pdb import run
from typing import TYPE_CHECKING, Optional, Tuple, Union
import threading
import math
from regex import F
import torch
import torch.distributed
from torch.cuda import Stream
from vllm._custom_ops import flexi_reshape_and_cache_flash
from vllm.distributed.kv_transfer.kv_connector.dynamic_utils import FlexiKVTensorMeta
from vllm.logger import init_logger
from vllm.lora import layers
from vllm.model_executor import set_random_seed
from vllm.v1.worker.utils import get_total_gpu_memory
from vllm.model_executor.models.dynamic_qwen3 import DynamicQwen3ForCausalLM
from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import dynamic_bind_single_kv_tensor, dynamic_flexi_bind_single_kv_tensor, get_layer_name_for_index, report_usage_stats, WorkerMemInfo
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.dynamic_gpu_model_runner import DynamicGPUModelRunner
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment, _check_if_gpu_supports_dtype
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
from vllm.utils import DeadlockTimeoutContext
from vllm.v1.utils import human_readable_duration
from vllm.v1.worker.utils import get_flexi_kv_cache
from vllm.device_allocator.cumem import CuMemAllocator
import gc


from vllm.vllm_flash_attn.flash_attn_interface import prepare_flexi_kv_ptrs



from .utils import KVBufferStatus
from vllm.sequence import IntermediateTensors
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.distributed.kv_transfer.kv_connector.dynamic_kv_synchronizer import KVPatchMeta, KVTensorMeta
from bitarray import bitarray
import time
from vllm.distributed.kv_transfer.kv_connector.dynamic_kv_synchronizer import DynamicKVSynchronizer
import signal
import sys
import traceback
import faulthandler
logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput



class DynamicGPUWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 缓存等待绑定到 forward_context 的 KV tensor（按全局 layer_id）
        self._pending_kv_by_layer: dict[int, torch.Tensor] = {}
        # Debug flag: when enabled, re-raise exceptions for easier debugging
        self._debug_raise: bool = str(os.getenv("VLLM_DEBUG_RAISE", "0")).lower() not in ("0", "", "false", "no")
        # Debug assertions for KV binding/shape
        self._debug_assert_kv: bool = str(os.getenv("VLLM_DEBUG_ASSERT_KV", "1")).lower() not in ("0", "", "false", "no")
        self._listen_kv_cache_threads: list[threading.Thread] = []
        self.rank_to_layers_ids: dict[int, list[int]] = {}

        self.migration_in_process: bool = False 
        self.sending_kv_cache_in_process: bool = False
        self.receive_in_process: bool = False

        # 独立的条件变量与锁：等待新层加载（避免在等待时占用 forward_lock）
        self._layer_loaded_lock = threading.Lock()
        self._layer_loaded_cv = threading.Condition(self._layer_loaded_lock)
        # 独立的条件变量：等待 KV cache 绑定到位（与“层已加载”解耦）
        self._kv_bound_lock = threading.Lock()
        self._kv_bound_cv = threading.Condition(self._kv_bound_lock)

        self._sync_migration_lock = threading.Lock()
        self._sync_migration_cv = threading.Condition(self._sync_migration_lock)

        pp_size = int(self.vllm_config.parallel_config.pipeline_parallel_size)
        # Used for waiting all patch applied
        self._all_patch_applied_cv = threading.Condition(threading.Lock()) 
        self.is_all_patch_applied: dict[int, bool] = {}
        self.is_all_patch_applied = {}
        for rank in range(pp_size):
            self.is_all_patch_applied[rank] = True

        self.new_kv_cache_block_num = 0 # 用于记录新配置的kv cache block数量，用于后续的kv cache resize
        logger.info(torch.__config__.show())

        # 用于记录在migration过程中，receiver已经applied的token数量
        self.receiver_num_applied_token_dict = defaultdict(int)

        self.block_size = 0
        
    def _assert_layers_kv_bound(self, layer_ids: set[int]) -> None:
        if not self._debug_assert_kv:
            return
        assert isinstance(self.model_runner, DynamicGPUModelRunner)
        start_layer = self.model_runner.model.model.start_layer
        fctx = self.vllm_config.compilation_config.static_forward_context
        is_flexi = self.vllm_config.dynamic_config.enable_flexi_flash_attn
        for layer_id in sorted(layer_ids):
            # 1) 层已加载
            assert self.model_runner.has_layer(layer_id), (
                f"Layer {layer_id} not loaded yet (start={start_layer})")
            local_index = layer_id - start_layer

            if is_flexi:
                # runner kv cache 非空且形状合理
                assert 0 <= local_index < len(self.model_runner.key_caches), (
                    f"Local index {local_index} out of range for key_cache_list len={len(self.model_runner.value_caches)}")
                key_t = self.model_runner.key_caches[local_index]
                assert isinstance(key_t, list) and len(key_t) > 0, (
                    f"Key cache for layer {layer_id} is empty/uninitialized")
                assert 0 <= local_index < len(self.model_runner.value_caches), (
                    f"Local index {local_index} out of range for value_cache_list len={len(self.model_runner.value_caches)}")
                value_t = self.model_runner.value_caches[local_index]
                assert isinstance(value_t, list) and len(value_t) > 0, (
                    f"Value cache for layer {layer_id} is empty/uninitialized")
                
                # 2) dynamic_kv_synchronizer
                assert 0 <= local_index < len(self.dynamic_kv_synchronizer.key_cache_list), (
                    f"Local index {local_index} out of range for sync key_cache_list")
                sync_key_t = self.dynamic_kv_synchronizer.key_cache_list[local_index]
                assert sync_key_t is key_t, f"Sync key cache mismatch for layer {layer_id}"
                
                assert 0 <= local_index < len(self.dynamic_kv_synchronizer.value_cache_list), (
                    f"Local index {local_index} out of range for sync value_cache_list")
                sync_value_t = self.dynamic_kv_synchronizer.value_cache_list[local_index]
                assert sync_value_t is value_t, f"Sync value cache mismatch for layer {layer_id}"

                # 3) forward context
                layer_name = get_layer_name_for_index(layer_id, fctx)
                assert layer_name in fctx, f"Layer {layer_name} missing in forward context"
                attn = fctx[layer_name]
                # attn.key_cache should be the same list object as key_t
                assert hasattr(attn, 'key_cache'), f"Attention layer {layer_name} missing key_cache"
                assert attn.key_cache is key_t, f"Forward context key cache mismatch for layer {layer_id}"
                assert hasattr(attn, 'value_cache'), f"Attention layer {layer_name} missing value_cache"
                assert attn.value_cache is value_t, f"Forward context value cache mismatch for layer {layer_id}"
                
            else:
                # runner kv cache 非空且形状合理
                assert 0 <= local_index < len(self.model_runner.kv_caches), (
                    f"Local index {local_index} out of range for kv_caches len={len(self.model_runner.kv_caches)}")
                kv_t = self.model_runner.kv_caches[local_index]
                assert isinstance(kv_t, torch.Tensor) and kv_t.numel() > 0, (
                    f"KV cache for layer {layer_id} is empty/uninitialized")

                # 2) dynamic_kv_synchronizer
                assert 0 <= local_index < len(self.dynamic_kv_synchronizer.kv_caches), (
                    f"Local index {local_index} out of range for sync kv_caches len={len(self.dynamic_kv_synchronizer.kv_caches)}")
                sync_kv_t = self.dynamic_kv_synchronizer.kv_caches[local_index]
                assert sync_kv_t is kv_t, f"Sync KV cache mismatch for layer {layer_id}"

                # 3) forward context kv_cache 指向同一张量，且形状一致
                layer_name = get_layer_name_for_index(layer_id, fctx)
                assert layer_name in fctx, f"Layer {layer_name} missing in forward context"
                attn = fctx[layer_name]
                assert isinstance(attn.kv_cache, list) and len(attn.kv_cache) >= 1, (
                    f"Attention.kv_cache invalid for {layer_name}")
                bound_kv = attn.kv_cache[0]
                assert isinstance(bound_kv, torch.Tensor) and bound_kv.numel() > 0, (
                    f"Forward context KV for {layer_name} is empty")
                assert kv_t.shape == bound_kv.shape, (
                    f"KV shape mismatch for layer {layer_id}: runner={kv_t.shape}, fctx={bound_kv.shape}")
                # 张量对象应当一致（引用同一存储）
                assert kv_t.data_ptr() == bound_kv.data_ptr(), (
                    f"KV tensor for layer {layer_id} is not the same object between runner and fctx")

    def _add_layers(self, layer_list: list[Tuple[int, int]]) -> None:
        """添加新的模型层
        
        死锁调试提示：如果此处发生死锁，可以：
        1. 运行 kill -SIGUSR1 <pid> 查看所有线程栈
        2. 或将 with 语句改为：
           with DeadlockTimeoutContext(self._layer_loaded_cv, "_layer_loaded_cv", timeout=30):
           来自动检测和报告死锁
        """
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        logger.info(f"[operation]: Add Model Layers: {layer_list}")
        time_start = time.time()
        is_flexi = self.vllm_config.dynamic_config.enable_flexi_flash_attn
        with DeadlockTimeoutContext(self._layer_loaded_cv, "_layer_loaded_cv", timeout=2):
            logger.info(f"start to load layers, inside forward lock")
            # 记录扩容前的起始 layer，便于在 start_layer 左移时重排本地 kv 索引基准
            old_start_layer = self.model_runner.model.model.start_layer
            old_end_layer = self.model_runner.model.model.end_layer
            self.model_runner.add_layers(layer_list)

            # 接收完weights之后，在这里进行kv cache数据结构的扩展，保证后续kv cache tensor绑定的正确性
            new_start_layer = self.model_runner.model.model.start_layer
            new_end_layer = self.model_runner.model.model.end_layer
            logger.info(f"debug: ---------------------add layers, old_start_layer: {old_start_layer}, new_start_layer: {new_start_layer}, old_end_layer: {old_end_layer}, new_end_layer: {new_end_layer}")

            # 在这里更新kv_cache_group
            if is_flexi:
                # 如果 start_layer 向更小的下标移动，需要对现有 self.kv_caches 做前置填充，
                # 使其索引基准与新的 start_layer 对齐
                left_added_layer_num = int(old_start_layer - new_start_layer)
                right_added_layer_num = int(new_end_layer - old_end_layer)
                if left_added_layer_num > 0:
                    self.model_runner.key_caches = [[] for _ in range(left_added_layer_num)] + \
                                                    self.model_runner.key_caches
                    self.model_runner.value_caches = [[] for _ in range(left_added_layer_num)] + \
                                                    self.model_runner.value_caches
                    self.model_runner.key_cache_ptrs = [0] * left_added_layer_num + \
                                                    self.model_runner.key_cache_ptrs
                    self.model_runner.value_cache_ptrs = [0] * left_added_layer_num + \
                                                    self.model_runner.value_cache_ptrs
                    
                    self.dynamic_kv_synchronizer.key_cache_list = [[] for _ in range(left_added_layer_num)] + \
                                                    self.dynamic_kv_synchronizer.key_cache_list
                    self.dynamic_kv_synchronizer.value_cache_list = [[] for _ in range(left_added_layer_num)] + \
                                                    self.dynamic_kv_synchronizer.value_cache_list
                    self.dynamic_kv_synchronizer.key_cache_ptrs = [0] * left_added_layer_num + \
                                                    self.dynamic_kv_synchronizer.key_cache_ptrs
                    self.dynamic_kv_synchronizer.value_cache_ptrs = [0] * left_added_layer_num + \
                                                    self.dynamic_kv_synchronizer.value_cache_ptrs

                if right_added_layer_num > 0:
                    logger.info(f"pad the kv cache for {right_added_layer_num}")
                    pad_right = right_added_layer_num
                    self.model_runner.key_caches.extend([[] for _ in range(pad_right)])
                    self.model_runner.value_caches.extend([[] for _ in range(pad_right)])
                    self.model_runner.key_cache_ptrs.extend([0] * pad_right)
                    self.model_runner.value_cache_ptrs.extend([0] * pad_right)

                    self.dynamic_kv_synchronizer.key_cache_list.extend([[] for _ in range(pad_right)])
                    self.dynamic_kv_synchronizer.value_cache_list.extend([[] for _ in range(pad_right)])
                    self.dynamic_kv_synchronizer.key_cache_ptrs.extend([0] * pad_right)
                    self.dynamic_kv_synchronizer.value_cache_ptrs.extend([0] * pad_right)
                logger.info(f"[timeline]: after add layers, time taken: {human_readable_duration(time.time() - time_start)}")
                for i in range(len(self.model_runner.key_caches)):
                    logger.info(f"key cache len: {len(self.model_runner.key_caches)}")
            else:
                # 如果 start_layer 向更小的下标移动，需要对现有 self.kv_caches 做前置填充，
                # 使其索引基准与新的 start_layer 对齐
                left_added_layer_num = int(old_start_layer - new_start_layer)
                right_added_layer_num = int(new_end_layer - old_end_layer)
                if left_added_layer_num > 0:
                    self.model_runner.kv_caches = [torch.tensor([])] * left_added_layer_num + \
                                                  self.model_runner.kv_caches
                    self.dynamic_kv_synchronizer.kv_caches = [torch.tensor([])] * left_added_layer_num + \
                                                  self.dynamic_kv_synchronizer.kv_caches
                if right_added_layer_num > 0:
                    logger.info(f"pad the kv cache for {right_added_layer_num}")
                    pad_right = right_added_layer_num
                    self.model_runner.kv_caches.extend([torch.tensor([])] * pad_right)
                    self.dynamic_kv_synchronizer.kv_caches.extend([torch.tensor([])] * pad_right)
                logger.info(f"[timeline]: after add layers, time taken: {human_readable_duration(time.time() - time_start)}")
                for i in range(len(self.model_runner.kv_caches)):
                    logger.info(f"kv cache {i}: {self.model_runner.kv_caches[i].shape}")
        
            # 唤醒等待层加载的线程（kv tensor的绑定线程和kv patch 应用线程）。
            self._layer_loaded_cv.notify_all()
            logger.info(f"[debug]: notified all layer loaded cv waiters")


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

        # Construct the model runner first
        self.model_runner: DynamicGPUModelRunner = DynamicGPUModelRunner(
            self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def load_model(self) -> None:
        super().load_model()

        # Wait for all ranks to finish loading model before initializing KV synchronizer
        # This prevents deadlock where faster ranks (fewer layers) enter barrier
        # while slower ranks (more layers) are still loading
        if torch.distributed.is_initialized():
            logger.info(f"Rank {self.rank}: waiting for all ranks to finish model loading before KV synchronizer init")
            torch.distributed.barrier()
            logger.info(f"Rank {self.rank}: all ranks ready, initializing KV synchronizer")

        # Initialize KV synchronizer AFTER model is ready (needs model for args)
        self.dynamic_kv_synchronizer = DynamicKVSynchronizer(
            rank=self.rank,
            local_rank=self.local_rank,
            config=self.vllm_config,
            model_executable=self.model_runner.model
        )
        # 启动所有监听其它rank的发送过来的kv cache的线程
        for rank in range(self.vllm_config.parallel_config.pipeline_parallel_size):
            if rank != self.rank:
                # threading.Thread(target=self.listen_to_kv_cache_tensor_and_patches, args=(rank,), daemon=True).start()
                self.listen_to_kv_cache_tensor_and_patches(rank)

    def dynamic_initialize_from_config(self, kv_cache_configs: list[KVCacheConfig], num_blocks: int) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.block_size = kv_cache_configs[self.rank].kv_cache_groups[0].kv_cache_spec.block_size
        self.block_num = num_blocks
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            if self.vllm_config.dynamic_config.enable_flexi_flash_attn:
                logger.info("Using flexi flash attention dynamic initialize kv cache")
                self.model_runner.dynamic_initialize_kv_cache_flexi(kv_cache_configs[self.rank], self.dynamic_kv_synchronizer, num_blocks)
            else:
                logger.info("Using standard flash attention dynamic initialize kv cache")
                self.model_runner.dynamic_initialize_kv_cache(kv_cache_configs[self.rank], self.dynamic_kv_synchronizer, num_blocks)
            self.dynamic_kv_synchronizer.create_slot_mappings(num_blocks * self.block_size)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        layer_config: Tuple[int, int],
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        assert False, "This function is not used"
        assert(isinstance(self.model_runner.model, DynamicQwen3ForCausalLM))
        logger.info(f"start to execute_model in gpu worker")
        result = self.model_runner.execute_model(scheduler_output, layer_config, intermediate_tensors=intermediate_tensors)
        assert(len(self.model_runner.input_batch.block_table.block_tables) == 0) 
        slot_mapping = self.model_runner.input_batch.block_table[0].slot_mapping
        assert slot_mapping.device == torch.device("cuda")
        if self.model_runner.migration_in_process:
            self.dynamic_kv_synchronizer.send_kv_cache_patch(self.rank_to_layers_ids, slot_mapping, self.model_runner.kv_caches, self.model_runner.model.model.start_layer)
        return result


    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """
        Copied from gpu_worker.py
        Profiles the peak memory usage of the model to determine how much 
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        # assert self.model_runner.model.get_sched_layers() == (self.model_runner.model.model.start_layer, self.model_runner.model.model.end_layer), "model should be in the initial state"
        self.model_runner.initialize_intermediate_states()
        torch.cuda.empty_cache()

        # torch.cuda.reset_peak_memory_stats()
        available_memory, _ = torch.cuda.mem_get_info()
        total_gpu_memory = get_total_gpu_memory(self.rank)
        safe_margin = (1 - self.cache_config.gpu_memory_utilization) * total_gpu_memory
        available_memory -= safe_margin

        logger.info(f"debug ------- determine available memory: {available_memory / 1024 ** 3:.2f} GB, safe_margin: {safe_margin / 1024 ** 3:.2f} GB, total_gpu_memory: {total_gpu_memory / 1024 ** 3:.2f} GB")
        return int(available_memory)

    @torch.inference_mode()
    def get_current_available_memory(self) -> int:
        """Get the current available memory in bytes.
        """
        free_gpu_memory, _ = torch.cuda.mem_get_info()
        return int(free_gpu_memory)

    def reinitialize_kv_cache(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        self.model_runner.reinitialize_kv_cache(kv_cache_configs[self.rank], self.dynamic_kv_synchronizer)
        return

    def async_add_layers(self, rank: int, layer_list: list[Tuple[int, int]]) -> None:
        if self.rank != rank:
            logger.info(f"Worker {self.rank} is not the target rank {rank}, skip adding model layers")
            return None
        logger.info(f"[operation]: Async Add Model Layers: {layer_list}")
        time_start = time.time()
        def _do_add():
            self._add_layers(layer_list)
        threading.Thread(target=_do_add, daemon=True).start()
        logger.info(f"[timeline]: after add layers, time taken: {human_readable_duration(time.time() - time_start)}")

    def remove_layers(self, rank: int, layer_list: list[Tuple[int, int]]) -> None:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip removing model layers")
            return None
        logger.info(f"Remove Model Layers: {layer_list}")
        self.model_runner.remove_layers(layer_list)

    def release_kv_cache_for_layers(self, rank: int, layers_list: list[Tuple[int, int]]) -> None:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip releasing kv cache for layers")
            return None
        # self.model_runner.release_kv_cache_for_layers(layers_list)
        from vllm.config import get_current_vllm_config
        vllm_config = get_current_vllm_config()
        is_flexi = vllm_config.dynamic_config.enable_flexi_flash_attn
        
        start_layer = self.model_runner.model.model.start_layer

        if is_flexi:
            self.model_runner.flexi_release_kv_cache_for_layers(layers_list)
            
            self.dynamic_kv_synchronizer.key_cache_list = [
                key_cache for idx, key_cache in enumerate(self.dynamic_kv_synchronizer.key_cache_list) 
                if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
            ]
            self.dynamic_kv_synchronizer.value_cache_list = [
                value_cache for idx, value_cache in enumerate(self.dynamic_kv_synchronizer.value_cache_list)
                if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
            ]
            self.dynamic_kv_synchronizer.key_cache_ptrs = [
                ptr for idx, ptr in enumerate(self.dynamic_kv_synchronizer.key_cache_ptrs)
                if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
            ]
            self.dynamic_kv_synchronizer.value_cache_ptrs = [
                ptr for idx, ptr in enumerate(self.dynamic_kv_synchronizer.value_cache_ptrs)
                if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
            ]
        else:
            self.model_runner.release_kv_cache_for_layers(layers_list)
            
            self.dynamic_kv_synchronizer.kv_caches = [
                kv_cache for idx, kv_cache in enumerate(self.dynamic_kv_synchronizer.kv_caches) 
                if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
            ]

    def release_kv_cache(self) -> None:
        self.model_runner.release_kv_cache()

    def get_mem_info(self) -> WorkerMemInfo:
        """Return per-layer weight size, current free GPU memory, and one-layer
        KV cache tensor size (bytes).

        - layer_size: bytes of a single Transformer layer's weights, as
          recorded by DynamicQwen3 during weight loading.
        - free_memory: current free GPU memory in bytes (driver reported).
        - single_kv_cache_tensor_size: bytes of one layer's KV cache tensor
          (includes both K and V within the tensor shape).
        """
        logger.info(f"before get_mem_info, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        time_start = time.time()
        torch.cuda.empty_cache()
        logger.info(f" after empty cache, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        # Layer weight size (may raise if not recorded yet)
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        layer_size = int(self.model_runner.model.get_layer_weight_size())

        is_kv_cache_initialized = len(self.model_runner.kv_caches) != 0 and self.model_runner.kv_caches[0].numel() != 0



        # Free memory from driver
        free_memory, _ = torch.cuda.mem_get_info()

        # Size of a single KV cache tensor (for one layer) from model runner
        kv_tensor_size = 0
        assert isinstance(self.model_runner, DynamicGPUModelRunner)
        if is_kv_cache_initialized:
            kv_tensor_size = int(self.model_runner.get_single_kv_tensor_size())

        # get the size of total gpu memory
        total_gpu_memory = get_total_gpu_memory(self.rank)

        return WorkerMemInfo(layer_size, kv_tensor_size, int(free_memory), int(total_gpu_memory))

    def get_kv_buffer_status(self) -> KVBufferStatus:
        """Return KV patch buffer status per rank.

        used_tokens = capacity - size
        free_tokens = size
        capacity_tokens = capacity
        """
        assert False
        capacity_list = {} 
        free_tokens_list = {} 
        used_tokens_list = {}
        for peer_rank in self.dynamic_kv_synchronizer.buffers:
            buf = self.dynamic_kv_synchronizer.buffers[peer_rank]
            capacity = buf.capacity
            free_tokens = buf.size
            used_tokens = max(0, capacity - free_tokens)
            if self.dynamic_kv_synchronizer.is_kv_patch_sending():
                capacity_list[peer_rank] = capacity
                free_tokens_list[peer_rank] = free_tokens
                used_tokens_list[peer_rank] = used_tokens
            else:
                capacity_list[peer_rank] = capacity
                free_tokens_list[peer_rank] = 0
                used_tokens_list[peer_rank] = capacity

        return KVBufferStatus(used_tokens_list, free_tokens_list, capacity_list)

    def get_applied_token_num(self, receiver_list: list[int]) -> Union[list[int], None]:
        """Return KV patch buffer status per rank.

        """
        if self.rank not in receiver_list:
            logger.info(f"Worker {self.rank} is not in the receiver list {receiver_list}, skip getting applied token num")
            return None
        result = [self.receiver_num_applied_token_dict[rank] for rank in range(self.vllm_config.parallel_config.pipeline_parallel_size) if rank != self.rank]
        logger.info(f"Worker {self.rank} get applied token num: {result}, receiver_list: {receiver_list}")
        return result

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
        self.model_runner.initialize_kv_cache_for_layers(kv_cache_specs, kv_cache_size, kv_cache_num_blocks, self.dynamic_kv_synchronizer, layers)
        return

    def compact_kv_cache(self, compacted_length: int, bitmap: bitarray) -> None:
        self.block_num = compacted_length
        self.model_runner.compact_kv_cache(compacted_length, bitmap)
        self.dynamic_kv_synchronizer.create_slot_mappings(compacted_length * self.block_size)

    def resize_kv_cache(self, new_length: int) -> None:
        self.block_num = new_length
        self.model_runner.resize_kv_cache(new_length)
        self.dynamic_kv_synchronizer.create_slot_mappings(new_length * self.block_size)
    # ####################################### #
    # Sender side of KV migration functions   #
    # ####################################### #
    def start_kv_cache_migration_async(self, src_to_plan: dict[int, dict[int, list[int]]], slot_mapping: Optional[list[int]] = None) -> None:
        """Collective-RPC entry used by executor.

        Only the worker whose `self.rank == source_rank` performs the actual
        send; other ranks are no-ops. The receiver side should already be
        listening via `listen_to_kv_cache_tensor(source_rank)`.
        """
        assert self.migration_in_process == False, "The migration should not be in process"
        assert self.sending_kv_cache_in_process == False, "The sending kv cache should not be in process"
        logger.info(f"[operation]: Start KV Cache Migration: {src_to_plan}")
        with self.model_runner.forward_lock:
            self.migration_in_process = True
            if self.rank not in src_to_plan:
                return None

            self.sending_kv_cache_in_process = True
        logger.info(f"[operation]: rank {self.rank} start sending KV Cache Migration")
        rank_to_layers_ids = src_to_plan[self.rank]
        assert len(self.rank_to_layers_ids) == 0
        self.rank_to_layers_ids = rank_to_layers_ids

        def migration_thread(rank: int, layer_ids: list[int]):
            assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
            time_start = time.time()
            # self.dynamic_kv_synchronizer.start_kv_tensor_transfer_async(rank_to_layers_ids, self.model_runner.kv_caches, self.model_runner.model.model.start_layer)
            self._kv_migration_sender_loop(rank, layer_ids,slot_mapping)
            logger.info(f"[timeline]: after start kv cache tensor, time taken: {human_readable_duration(time.time() - time_start)}")

        for rank, layer_ids in rank_to_layers_ids.items():
            threading.Thread(target=migration_thread, args=(rank, layer_ids), daemon=True).start()
        return None

    def _kv_migration_sender_loop(self, 
                                rank: int,
                                layer_ids: list[int],
                                slot_mapping: Optional[list[int]] = None
                                ) -> None:
        """Send layers' KV cache to a peer rank.
            In each of the migration process, this function should only be called once.
            Different from the buffered version, this function directly gathers the kv cache from the memory pointers 
            and sends them without using intermediate buffers.
            Slot mappings is updated by the worker threads, which essentially replace the role of buffer.
            The good thing about slot mapping is that is is implemented with dirty page, which avoids duplicated token ids.
        Args:
            rank: peer global rank to send to.
            kv_cache: tensor to send (GPU or CPU tensor; will be moved to local GPU).
            start_layer_id: which layer this KV cache belongs to (for receiver to demux).
            layer_ids: which layers this KV cache belongs to (for receiver to demux).
            slot_mapping: the slot mapping to gather the kv cache.
        """
        assert all(transfer_in_process == False for transfer_in_process in self.dynamic_kv_synchronizer.kv_cache_transfer_in_process.values()), "In each of the migration process, this function should only be called once."
        assert all(patch_id == 0 for patch_id in self.dynamic_kv_synchronizer.last_patch_ids.values()), "The patch id of the rank should be 0."
        slot_mapping_dev = torch.tensor(slot_mapping, device=self.device) if slot_mapping is not None else None
        start_layer_id = self.model_runner.model.model.start_layer        # Firstly send the kv tensor to remote rank
        for layer_id in layer_ids:
            time_start = time.time()
            kv_tensor_meta, kv_tensor_data = self.dynamic_kv_synchronizer.get_kv_tensor_from_cache(layer_ids, layer_id, start_layer_id, slot_mapping_dev)
            logger.info(f"start to send kv tensor for layer {layer_id} to rank {rank}, kv_tensor_meta: {kv_tensor_meta}, kv_tensor_data shape: {kv_tensor_data.shape}, time taken to get kv tensor: {human_readable_duration(time.time() - time_start)} seconds")

            self.dynamic_kv_synchronizer.send_kv_tensor_to_rank(rank, kv_tensor_meta, kv_tensor_data, slot_mapping_dev)
            logger.info(f"[debug]: sent kv tensor for layer {layer_id} to rank {rank}")

        logger.info(f"[debug]: finished sending kv tensor, start to send kv patch")

        # 在发送完kv tensor之后开始发送kv patch
        for kv_patch in self.dynamic_kv_synchronizer.get_kv_patch(rank, start_layer_id, layer_ids):
            time_start = time.time()
            patch_payload_size = kv_patch.kv_payload.numel() * kv_patch.kv_payload.element_size()
            patch_slot_mapping_size = kv_patch.slot_mapping.numel() * kv_patch.slot_mapping.element_size()
            logger.info(f"[timeline]: send kv patch to rank {rank}, time taken: {time.time() - time_start}, data_size: {patch_slot_mapping_size / 1024 ** 2:.2f}MB + {patch_payload_size / 1024 ** 2:.2f}MB, kv patch id: {kv_patch.meta.id}")
            self.dynamic_kv_synchronizer.send_kv_patch_to_rank(rank, kv_patch)


        # 最后发送一个 finished 的控制消息，表示本次kv cache的发送完成


    def start_kv_cache_migration_sync(self, src_to_sending_layers: dict[int, dict[int, list[Tuple[int, int]]]], rank_to_layer_ids: dict[int, list[Tuple[int, int]]]) -> None:
        """Collective-RPC entry used by executor.

        Only the worker whose `self.rank == source_rank` performs the actual
        send; other ranks are no-ops. The receiver side should already be
        listening via `listen_to_kv_cache_tensor(source_rank)`.
        """

        with self.model_runner.forward_lock:
            time_start = time.time()
            assert self.receive_in_process == False, "The sending kv cache should not be in process"
            if self.rank in rank_to_layer_ids:
                logger.info(f"debug: rank {self.rank} is receiving kv cache")
                self.receive_in_process = True
                # layer_list = rank_to_layer_ids[self.rank]
                # self._add_layers(layer_list)
            if self.rank not in src_to_sending_layers:
                logger.info(f"[timeline]: after start kv cache migration sync, time taken: {human_readable_duration(time.time() - time_start)}")
                return None
            self.sending_kv_cache_in_process = True
            assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)

            sending_layers_plans = src_to_sending_layers[self.rank]
            sending_layers_list: dict[int, list[int]] = {}
            for rank, layer_ranges in sending_layers_plans.items():
                for layer_range in layer_ranges:
                    layer_ids = sending_layers_list.setdefault(rank, [])
                    layer_ids.extend(list(range(layer_range[0], layer_range[1] + 1)))

            # with self.model_runner.forward_lock:
            self.dynamic_kv_synchronizer.start_kv_tensor_transfer_sync(sending_layers_list, self.model_runner.kv_caches, self.model_runner.model.model.start_layer)
            # Asynchronize with cuda stream
            torch.cuda.synchronize()

            for _, layer_ranges in sending_layers_plans.items():
                self.release_kv_cache_for_layers(self.rank, layer_ranges)
                self.remove_layers(self.rank, layer_ranges)
            self.sending_kv_cache_in_process = False

        logger.info(f"-------------启动 KV cache 迁移所用时间: {time.time() - time_start:.2f} 秒")
        return None

    # ####################################### #
    # Receiver side of KV migration functions #
    # ####################################### # 
    def listen_to_kv_cache_tensor_and_patches(self, from_rank: int) -> None:
        """
        异步监听指定 from_rank 的 KV：先接收完整 KV 张量，再持续接收小 patch，并按需同步。
        """
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        assert self.rank != from_rank, "The rank should not listen to its own kv cache"

        logger.info(f"Worker {self.rank} listening KV stream from rank {from_rank}")

        threading.Thread(target=self._listen_loop, args=(from_rank,), daemon=True).start()
        return None

    def _listen_loop(self, from_rank: int):
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        is_flexi = self.vllm_config.dynamic_config.enable_flexi_flash_attn
        time_start = time.time()
        while True:
            received_layer = set()
            layers_to_be_received  = set()
            self.receiver_num_applied_token_dict[from_rank] = 0
            logger.info(f"[operation]: start to listen to kv cache tensor and patches")
            for i in range(len(self.model_runner.kv_caches)):
                logger.info(f"[debug]: kv_caches[{i}] shape: {self.model_runner.kv_caches[i].shape}")
            first_time = True
            tmp_kv_tensors_dict: dict[int, torch.Tensor] = {}
            tmp_slot_mapping_dict: dict[int, torch.Tensor] = {}
            tmp_slot_token_num: int = 0

            while True:
                # 1) Firstly wait for all tensor the be received
                meta = self.dynamic_kv_synchronizer.recv_controller(from_rank)
                logger.info("received kv tensor meta")
                if first_time:
                    assert isinstance(meta, KVTensorMeta) or isinstance(meta, FlexiKVTensorMeta)
                    layers_to_be_received = set(meta.layer_to_be_received)
                    time_start = time.time()
                    first_time = False
                    self.is_all_patch_applied[from_rank] = False # 第一次接受到kv tensor时，需要将is_all_patch_applied设置为False，表示需要等待所有patch都应用完毕后才能进行配置的转换
                    tmp_slot_token_num = meta.num_tokens
                if meta.type != "kv_tensor":
                    assert layers_to_be_received == received_layer, "The layers to be received should be the same as the received layers"
                    break

                assert meta.layer_id not in received_layer, "The layer should not be received twice"
                assert meta.num_tokens == tmp_slot_token_num, "The num tokens should be the same for all kv tensor metas"
                received_layer.add(meta.layer_id)
                logger.info(f"[debug]: rank {self.rank} receive kv tensor meta {meta}")

                if is_flexi:
                    assert isinstance(meta, FlexiKVTensorMeta)
                    slot_mapping, kv_tensor = self.dynamic_kv_synchronizer.recv_kv_tensor(from_rank, meta)
                    tmp_slot_mapping_dict[meta.layer_id] = slot_mapping 
                else:
                    assert isinstance(meta, KVTensorMeta)
                    kv_tensor = self.dynamic_kv_synchronizer.recv_kv_tensor(from_rank, meta)
                    assert isinstance(kv_tensor, torch.Tensor)
                tmp_kv_tensors_dict[meta.layer_id] = kv_tensor

                # logger.info(f"available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")

            logger.info(f"[debug]: receive kv tensor finished, start to bind kv cache")
            # 在kv cache绑定前，weight必须loading结束
            for layer_id, kv_tensor in tmp_kv_tensors_dict.items():
                with self._layer_loaded_cv:
                    while not self.model_runner.has_layer(layer_id):
                        logger.info(f"[debug]: rank {self.rank} waiting for layer {layer_id} to be loaded")
                        self._layer_loaded_cv.wait()
                with self.model_runner.forward_lock:
                    logger.info(f"[debug]: rank {self.rank} bind layer {layer_id}'s kv cache and weight")
                    if is_flexi:
                        slot_mapping = tmp_slot_mapping_dict[layer_id]
                        dynamic_flexi_bind_single_kv_tensor(
                            self.model_runner.model.model.start_layer,
                            self.model_runner.model.model.end_layer,
                            layer_id,
                            slot_mapping,
                            self.block_num,
                            kv_tensor,
                            self.vllm_config.compilation_config.static_forward_context,
                            self.dynamic_kv_synchronizer,
                            self.model_runner
                        )
                    else:
                        dynamic_bind_single_kv_tensor(
                            layer_id,
                            self.model_runner.model.model.start_layer,
                            self.model_runner.model.model.end_layer,
                            self.compilation_config.static_forward_context,
                            self.dynamic_kv_synchronizer,
                            runner=self.model_runner,
                            kv_tensor=kv_tensor
                        )
                logger.info(f"[operation]: rank {self.rank} layer {layer_id}'s kv cache and weight is loaded")
          
            self.receiver_num_applied_token_dict[from_rank] = tmp_slot_token_num
            # 在同步迁移中，断言：所有接收层的 KV 已完成绑定且形状一致
            try:
                self._assert_layers_kv_bound(layers_to_be_received)
            except Exception as e:
                logger.exception(f"KV binding assertion failed: {e}")
                raise

            logger.info(f"[debug]: after receive kv tensor, kv cache:")
            if is_flexi:
                for i in range(len(self.model_runner.key_caches)):
                    logger.info(f"[debug]: key_caches[{i}] length: {len(self.model_runner.key_caches[i])}")
            else:
                for i in range(len(self.model_runner.kv_caches)):
                    logger.info(f"[debug]: kv_caches[{i}] shape: {self.model_runner.kv_caches[i].shape}")
            cur_patch_id = 0
            logger.info(f"[timeline]: after receive kv tensor, time taken: {human_readable_duration(time.time() - time_start)}")
            logger.info(f"[operation]: start to listen to kv cache patches")
            # 2) Then wait for all patch to be received
            while True:
                assert meta.type == "kv_patch_meta" or meta.type == "kv_patch_finished", "The type of the meta should be kv_patch_meta or kv_patch_finished"

                # Ensure the order of the patch
                assert meta.id == cur_patch_id, f"The patch id should be the next id of the last patch, meta id: {meta.id} vs cur patch id{cur_patch_id}"
                cur_patch_id += 1

                slot_mapping, kv_payload = self.dynamic_kv_synchronizer.recv_kv_patch(from_rank, meta)
                self.dynamic_kv_synchronizer.apply_one_patch_to_kv_cache(self.model_runner.model.model.start_layer, meta, kv_payload, slot_mapping)
                if meta.type == "kv_patch_meta":
                    logger.info(f"debug: ------------ Worker {self.rank} received kv patch meta from rank {from_rank}, kv payload shape: {kv_payload.shape}, slot mapping shape: {slot_mapping.shape}, num tokens: {meta.num_tokens}, patch id: {meta.id}")
                    self.receiver_num_applied_token_dict[from_rank] += meta.num_tokens
                elif meta.type == "kv_patch_finished":
                    logger.info(f"Worker {self.rank} received kv patch finished message from rank {from_rank}")
                    # 通知所有等待所有KV cache patch都applied完成的线程
                    with self._all_patch_applied_cv:
                        self.is_all_patch_applied[from_rank] = True
                        self._all_patch_applied_cv.notify_all()
                    break # Exit point from current migration listening loop
                else:
                    assert False, f"Unexpected message type: {meta.type}"
                meta = self.dynamic_kv_synchronizer.recv_controller(from_rank)
            logger.info(f"[timeline]: after listen to kv cache patches, time taken: {human_readable_duration(time.time() - time_start)}")
        
    def all_patch_applied(self) -> bool:
        return all(self.is_all_patch_applied.values())

    # #######################################     #
    # Callback funtions executed inside the forwarding loop #
    # ####################################### # 

    def async_migration_before_execute_callback(self,
                                               is_sync_after_migration: bool
                                               ) -> None:
        # 处于迁移中时，或这是同步批（用于发送 finished 信号），直接放行。
        if self.migration_in_process:
            logger.info(f"debug: rank {self.rank} is in migration or is sync after migration, skip kv synchronize before execute callback")
            return

        # 等待所有KV cache补丁都应用完毕后再使用新的pp配置
        with self._all_patch_applied_cv:
            # logger.info(f"debug: ---------------------rank {self.rank} waiting for patch to be all applied")
            is_not_applied = False
            while not self.all_patch_applied():
                if not is_not_applied:
                    time_start = time.time()
                    is_not_applied = True
                self._all_patch_applied_cv.wait()
            if is_not_applied:
                logger.info(f"[timeline]: after waiting for all kv cache patch to be applied, time taken: {human_readable_duration(time.time() - time_start)}")

        if self.new_kv_cache_block_num != 0:
            time_start = time.time()
            self.resize_kv_cache(self.new_kv_cache_block_num)
            logger.info(f"[timeline]: after resize kv cache, time taken: {human_readable_duration(time.time() - time_start)}")
            self.new_kv_cache_block_num = 0

    def sync_migration_before_execute_callback(self, new_kv_cache_block_num: int) -> None:
        assert self.sending_kv_cache_in_process == False, "The sending kv cache should not be in process"
        with self._sync_migration_cv:
            while self.receive_in_process:
                logger.info(f"debug: ---------------------rank {self.rank} waiting for migration to be finished")
                self._sync_migration_cv.wait()
            logger.info(f"debug: ---------------------rank {self.rank} is good for forwarding")
        # if new_kv_cache_block_num != 0:
        #     self.resize_kv_cache(new_kv_cache_block_num)

    def async_migration_after_execute_callback(self, is_sync: bool, new_kv_cache_block_num: int):
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)

        # target_device = self.device  # set in init_device to cuda:self.local_rank
        # if slot_mapping.device != target_device:
        #     slot_mapping = slot_mapping.to(target_device, non_blocking=True)

        # First, enqueue this step's KV patch (if sending) so the latest tokens are transferred.
        if self.sending_kv_cache_in_process:
            time_start = time.time()
            # Send the kv cache patch to the other ranks if the migration is in process
            slot_mapping = self.model_runner.input_batch.block_table[0].slot_mapping
            # Ensure slot_mapping is on this worker's CUDA device
            assert slot_mapping.device.type == "cuda", f"slot_mapping must be on CUDA, got {slot_mapping.device}"
            for rank in self.rank_to_layers_ids:
                self.dynamic_kv_synchronizer.add_new_tokens_to_kv_synchronizer(
                    rank,
                    self.model_runner.kv_caches,
                    self.rank_to_layers_ids[rank],
                    self.model_runner.model.model.start_layer,
                    slot_mapping,
                    is_sync
                )

            if is_sync:
                time_start = time.time()
                logger.info(f"[operation]: before finish kv cache transfer start to finish kv cache tansfer, delete layers, release kv cache, reinitialize kv cache")
                # Sender finishes transfer and cleans up; receiver no-ops.
                logger.info(f"debug: finish kv cache transfer for rank {self.rank}")
                logger.info(f"[timeline]: after finish all kv cache transfer, time taken: {human_readable_duration(time.time() - time_start)}")
                # 在这个节点，我们可以free旧的layer weights和kv cache
                layer_ranges = []
                for layers in self.rank_to_layers_ids.values():
                    # Assert the layers are sorted
                    assert layers == sorted(layers), "The layers should be sorted"
                    layer_ranges.append((layers[0], layers[-1]))
                self.release_kv_cache_for_layers(self.rank, layer_ranges)
                logger.info(f"[timeline]: after release kv cache for layers, time taken: {human_readable_duration(time.time() - time_start)}")
                self.remove_layers(self.rank, layer_ranges)
                logger.info(f"[timeline]: after remove layers, time taken: {human_readable_duration(time.time() - time_start)}")
                self.new_kv_cache_block_num = new_kv_cache_block_num
                self.finish_migration()
        else:
            assert not is_sync, "If not sending kv cache, it should not be sync migration"

    def finish_migration(self):
        self.migration_in_process = False
        self.sending_kv_cache_in_process = False
        self.rank_to_layers_ids = {}

        # assert self.dynamic_layer_kv_connector.is_all_patch_applied(), "All patch should be applied"

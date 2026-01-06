from collections import defaultdict
from concurrent.futures import thread
from copy import deepcopy
import gc
from hmac import new
from operator import is_
import os
from pdb import run
from sched import scheduler
from typing import TYPE_CHECKING, Optional, Tuple, Union
import threading
import math
from regex import F
from responses import start
import torch
import torch.distributed
from torch.cuda import Stream
from vllm._custom_ops import flexi_reshape_and_cache_flash
from vllm.attention import layer
from vllm.attention.dynamic_layer import FlexiAttention
from vllm.distributed.kv_transfer.kv_connector.dynamic_utils import FlexiKVTensorMeta
# from vllm.kv_allocator import allocate_with_cuda_async, free_cache, free_page_list, prepare_flexi_kv_ptrs
from vllm.kv_allocator import kv_allocator
from vllm.logger import init_logger
from vllm.lora import layers
from vllm.model_executor import set_random_seed
from vllm.v1.core.dynamic_kv_cache_utils import compact_cache_with_record
from vllm.v1.worker.utils import get_total_gpu_memory
from vllm.model_executor.models.dynamic_qwen3 import DynamicQwen3ForCausalLM
from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import dynamic_bind_single_kv_tensor, dynamic_flexi_bind_single_kv_cache, dynamic_flexi_bind_single_kv_tensor, get_layer_name_for_index, report_usage_stats, WorkerMemInfo
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.dynamic_gpu_model_runner import DynamicGPUModelRunner
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment, _check_if_gpu_supports_dtype
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
from vllm.utils import DeadlockTimeoutContext, hash_tensor_list
from vllm.v1.utils import human_readable_duration
from vllm.v1.worker.utils import get_flexi_kv_cache
from vllm.device_allocator.cumem import CuMemAllocator
import gc


from vllm.v1.engine.memory_stress_tester import initialize_stress_tester, get_global_stress_tester


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

        self.sending_kv_cache_patch_in_process: bool = False
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
        self.is_all_patch_applied = {}
        for rank in range(pp_size):
            self.is_all_patch_applied[rank] = True

        self.new_kv_cache_block_num = 0 # 用于记录新配置的kv cache block数量，用于后续的kv cache resize
        logger.info(torch.__config__.show())

        # 用于记录在migration过程中，receiver已经applied的token数量
        self.receiver_num_applied_token_dict = defaultdict(int)
        # 当所有的patch都applied之后，记录总的applied token数量
        self.after_migration_applied_token_num = 0

        self.block_size = 0

        self.migration_records_for_specific_scheduler_output_version: dict[int, dict[int, int]] = {}
         # 记录在migration过程中，已经迁移的block id映射关系

        self.cur_scheduler_output_version = 0

        # 记录scheduler_output中的最后的total token number，用来和receiver端已经applied的token num进行对比 
        self.after_migration_total_token = 0

        # 记录当前rank是否已经结束 kv resizing
        self.kv_resizing_done = False

        # stream
        self.high_priority_stream: Optional[torch.cuda.Stream] = None


        # page meta
    def _load_kv_cache_allocator(self):
        """Load kv_cache_allocator_optimized C++ extension"""
        try:
            import kv_cache_allocator_optimized
            logger.info("Loaded kv_cache_allocator_optimized C++ extension")
            return kv_cache_allocator_optimized
        except ImportError:
            # Try to load it dynamically using torch.utils.cpp_extension
            try:
                from torch.utils.cpp_extension import load
                
                # Get the path to the source file
                csrc_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                    'csrc',
                    'kv_cache_allocator_optimized.cpp'
                )
                
                logger.info(f"Loading kv_cache_allocator_optimized from {csrc_path}")
                kv_allocator = load(
                    name='kv_cache_allocator_optimized',
                    sources=[csrc_path],
                    extra_cuda_cflags=['-O3', '--use_fast_math'],
                    verbose=False
                )
                logger.info("Successfully loaded kv_cache_allocator_optimized")
                return kv_allocator
            except Exception as e:
                logger.warning(f"Failed to load kv_cache_allocator_optimized: {e}")
                return None
    
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
                assert isinstance(attn, FlexiAttention), f"Layer {layer_name} is not FlexiAttention"
                # attn.key_cache should be the same list object as key_t
                assert hasattr(attn, 'key_dev_ptr'), f"Attention layer {layer_name} missing key_dev_ptr"
                assert hasattr(attn, 'value_dev_ptr'), f"Attention layer {layer_name} missing value_dev_ptr"
                assert hasattr(attn, 'num_blocks'), f"Attention layer {layer_name} missing num_blocks"
                assert hasattr(attn, 'page_meta'), f"Attention layer {layer_name} missing page_meta"
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
            logger.info(f"start to load layer lock")
            # 记录扩容前的起始 layer，便于在 start_layer 左移时重排本地 kv 索引基准
            old_start_layer = self.model_runner.model.model.start_layer
            old_end_layer = self.model_runner.model.model.end_layer
            assert self.device is not None
            self.model_runner.add_layers(layer_list, self.device)

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
        
            logger.info(f"kv scynchronizer kv cache list length after adding layers: {len(self.dynamic_kv_synchronizer.key_cache_ptrs)}")
            # 唤醒等待层加载的线程（kv tensor的绑定线程和kv patch 应用线程）。
            self._layer_loaded_cv.notify_all()
            logger.info(f"[debug]: notified all layer loaded cv waiters")
            torch.cuda.synchronize()


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
        kv_allocator.set_lock(self.model_runner.forward_lock)
        self.high_priority_stream = torch.cuda.Stream(device=self.device, priority=-5)
        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)
        
        
        # Initialize memory stress tester (for testing KV cache allocation impact)
        # Will be started by tester_thread based on tester_start_step
        stress_tester_config = self.vllm_config.dynamic_config.memory_stress_tester
        ranks_to_perform_test = stress_tester_config.get('test_rank', []) if stress_tester_config is not None else []
        if self.rank in ranks_to_perform_test and stress_tester_config is not None:
            self.memory_stress_tester = initialize_stress_tester(
                self.model_runner.forward_lock,
                device=f"cuda:{self.vllm_config.parallel_config.rank}",
                num_tensors_per_allocation=stress_tester_config.get('num_tensors_per_allocation', 1000),
                tensor_shape=tuple(stress_tester_config.get('tensor_shape', [16, 8, 128])),
                allocation_interval_ms=stress_tester_config.get('allocation_interval_ms', 100),
                num_allocation_cycles=stress_tester_config.get('num_allocation_cycles', 10),
                max_total_allocations=stress_tester_config.get('max_total_allocations', None),
                allocation_strategy=stress_tester_config.get('allocation_strategy', 'torch_empty'),
                enabled=stress_tester_config.get('enabled', False),
            )
            logger.info(f"Memory stress tester initialized (will start at configured step)")
        else:
            self.memory_stress_tester = None
        

    def load_model(self) -> None:
        super().load_model()

        # Wait for all ranks to finish loading model before initializing KV synchronizer
        # This prevents deadlock where faster ranks (fewer layers) enter barrier
        # while slower ranks (more layers) are still loading
        if torch.distributed.is_initialized():
            logger.info(f"Rank {self.rank}: waiting for all ranks to finish model loading before KV synchronizer init")
            torch.distributed.barrier()
            logger.info(f"Rank {self.rank}: all ranks ready, initializing KV synchronizer")
        assert self.device is not None
        # Initialize KV synchronizer AFTER model is ready (needs model for args)
        self.dynamic_kv_synchronizer = DynamicKVSynchronizer(
            rank=self.rank,
            local_rank=self.local_rank,
            config=self.vllm_config,
            model_executable=self.model_runner.model,
            device=self.device
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
        # 异步调用：与工作的 commit 9ed5ec00f2c94 保持一致
        # 注意：daemon=False 确保线程在进程退出前完成
        threading.Thread(target=_do_add, daemon=False).start()
        logger.info(f"[timeline]: after add layers, time taken: {human_readable_duration(time.time() - time_start)}")

    def remove_layers(self, rank: int, layer_list: list[Tuple[int, int]]) -> None:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip removing model layers")
            return None
        logger.info(f"Remove Model Layers: {layer_list}")
        assert self.device is not None
        self.model_runner.remove_layers(layer_list, self.device)

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
        # torch.cuda.empty_cache()
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

    def get_is_kv_resizing_done(self) -> bool:
        return self.kv_resizing_done

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
        # self.block_num = compacted_length
        self._compact_kv_cache(compacted_length, bitmap)


    def _compact_kv_cache(self, compacted_length: int, bitmap: bitarray) -> None:
        runner = self.model_runner
        start_layer, end_layer = runner.model.model.start_layer, runner.model.model.end_layer
        is_flexi = self.vllm_config.dynamic_config.enable_flexi_flash_attn
        assert isinstance(runner.model, DynamicQwen3ForCausalLM)
        logger.info(f"start to compact kv cache for layers {runner.model.model.start_layer} to {runner.model.model.end_layer}")
        num_blocks = len(bitmap)
        if is_flexi:
            assert num_blocks == len(runner.key_caches[0]), f"bitmap length mismatch: num_blocks: {num_blocks} != kv_cache_tensor_length: {len(runner.key_caches)}"
        else:
            assert num_blocks == runner.kv_caches[0].size(1), f"bitmap length mismatch: num_blocks: {num_blocks} != kv_cache_tensor_length: {len(runner.kv_caches[0])}"

        def is_used(idx):
            return bitmap[idx]
        migrate_record: dict[int, int] = {}

        tmp_key_cache = []
        tmp_value_cache = []
        for key_cache, value_cache in zip(runner.key_caches, runner.value_caches):
            tmp_key_cache.append([page for page in key_cache])
            tmp_value_cache.append([page for page in value_cache])
         # Create temporary copies of the caches and pointers
        tmp_old_key_ptrs = deepcopy(self.model_runner.key_cache_ptrs)
        tmp_old_value_ptrs = deepcopy(self.model_runner.value_cache_ptrs)
        tmp_new_key_ptrs: list[int] = []
        tmp_new_value_ptrs: list[int] = []
        
        def _migrate_block_by_swapping_ptrs(old_block_id: int, new_block_id: int, migrate_record: dict[int, int]):
            assert len(tmp_key_cache) != 0
            for key_cache in tmp_key_cache:
                key_cache[new_block_id], key_cache[old_block_id] = key_cache[ old_block_id], key_cache[ new_block_id]
            for value_cache in tmp_value_cache:
                value_cache[ new_block_id], value_cache[ old_block_id] = value_cache[ old_block_id], value_cache[ new_block_id]
            migrate_record[old_block_id] = new_block_id

        if runner.vllm_config.dynamic_config.enable_flexi_flash_attn:
            compact_cache_with_record(_migrate_block_by_swapping_ptrs, is_used, compacted_length, num_blocks, migrate_record)

            # Prepare new kv ptrs
            forward_context = runner.vllm_config.compilation_config.static_forward_context
            for layer_name, attn_module in forward_context.items():
                idx = extract_layer_index(layer_name)
                local_idx = idx - start_layer
                # Free old GPU pointer arrays to avoid memory leak
                new_k_ptrs, new_v_ptrs = kv_allocator.prepare_flexi_kv_ptrs(
                    tmp_key_cache[local_idx], tmp_value_cache[local_idx])
                logger.info(f"prepare kv ptr for layer {layer_name}, new_k_ptrs: {new_k_ptrs}, new_v_ptrs: {new_v_ptrs}")
                tmp_new_key_ptrs.append(new_k_ptrs)
                tmp_new_value_ptrs.append(new_v_ptrs)
            # Bind the new kv cache
            with self.model_runner.forward_lock:
                time_start_within_lock = time.time()
                for layer_name, attn_module in forward_context.items():
                    idx = extract_layer_index(layer_name)
                    local_idx = idx - start_layer
                    new_k_ptrs, new_v_ptrs = tmp_new_key_ptrs[local_idx], tmp_new_value_ptrs[local_idx]
                    new_k_cache, new_v_cache = tmp_key_cache[local_idx], tmp_value_cache[local_idx]
                    # Free old GPU pointer arrays to avoid memory leak
                    dynamic_flexi_bind_single_kv_cache(start_layer, end_layer, idx, new_k_cache, new_v_cache, new_k_ptrs, new_v_ptrs, forward_context, self.dynamic_kv_synchronizer, runner)
                logger.debug(f"after binding, synchronizer key cache ptr list: {self.dynamic_kv_synchronizer.key_cache_ptrs}, value cache ptr list: {self.dynamic_kv_synchronizer.value_cache_ptrs}") 
                # 遍历CachedRequestState更新block_ids
                for req_state in runner.requests.values():
                    for i, block_ids in enumerate(req_state.block_ids):
                        for j, block_id in enumerate(block_ids):
                            if block_id in migrate_record:
                                req_state.block_ids[i][j] = migrate_record[block_id]
                # 遍历InputBatch中的block_table更新block_ids
                for block_table in runner.input_batch.block_table:
                    for row in block_table.block_table_np:
                        for local_idx, block_id in enumerate(row):
                            if block_id in migrate_record:
                                row[local_idx] = migrate_record[block_id]
                logger.info(f"kv cache compaction flexi take {human_readable_duration(time.time() - time_start_within_lock)} seconds within lock")
            assert self.device is not None
            for old_k_ptrs, old_v_ptrs in zip(tmp_old_key_ptrs, tmp_old_value_ptrs):
                kv_allocator.free_page_list(old_k_ptrs, self.device)
                kv_allocator.free_page_list(old_v_ptrs, self.device)
        else:
            with self.model_runner.forward_lock:
                time_start_within_lock = time.time()
                compact_cache_with_record(runner._migrate_block_by_copy_data, is_used, compacted_length, num_blocks, migrate_record)
                # 遍历CachedRequestState更新block_ids
                for req_state in runner.requests.values():
                    for i, block_ids in enumerate(req_state.block_ids):
                        for j, block_id in enumerate(block_ids):
                            if block_id in migrate_record:
                                req_state.block_ids[i][j] = migrate_record[block_id]
                # 遍历InputBatch中的block_table更新block_ids
                for block_table in runner.input_batch.block_table:
                    for row in block_table.block_table_np:
                        for local_idx, block_id in enumerate(row):
                            if block_id in migrate_record:
                                row[local_idx] = migrate_record[block_id]
                torch.cuda.synchronize()
                logger.info(f"kv cache compaction regular take {human_readable_duration(time.time() - time_start_within_lock)} seconds within lock")
        self.migration_records_for_specific_scheduler_output_version[self.cur_scheduler_output_version] = migrate_record
        self.cur_scheduler_output_version += 1
        logger.info(f"[debug]: migrate_record: {migrate_record}")
        
        # # ===== 关键修复：将更新后的 block table 同步到 GPU =====
        # # block_table_np 的修改已自动同步到 block_table_cpu (numpy view)
        # # 但必须显式调用 commit() 将 CPU tensor 拷贝到 GPU tensor
        # num_reqs = len(self.requests)
        # logger.info(f"[debug]: committing block table updates to GPU for {num_reqs} requests")
        # self.input_batch.block_table.commit(num_reqs)
        # logger.info(f"[debug]: block table committed to GPU")
        torch.cuda.synchronize()
    
    def _maybe_switch_block(self, scheduler_output: "DynamicSchedulerOutput") -> None:
        if scheduler_output.current_scheduler_output_version != self.cur_scheduler_output_version:
            time_start = time.time()
            assert scheduler_output.current_scheduler_output_version < self.cur_scheduler_output_version, f"Scheduler output version {scheduler_output.current_scheduler_output_version} is greater than current version {self.cur_scheduler_output_version}"
            assert scheduler_output.current_scheduler_output_version in self.migration_records_for_specific_scheduler_output_version, f"Migration record for version {scheduler_output.current_scheduler_output_version} not found"
            migrate_record = self.migration_records_for_specific_scheduler_output_version[scheduler_output.current_scheduler_output_version]
            for req in scheduler_output.scheduled_new_reqs:
                block_ids = req.block_ids
                for i, block_ids in enumerate(req.block_ids):
                    for j, block_id in enumerate(block_ids):
                        if block_id in migrate_record:
                            req.block_ids[i][j] = migrate_record[block_id]
            
            for req in scheduler_output.scheduled_cached_reqs:
                block_ids = req.new_block_ids
                for i, block_ids in enumerate(req.new_block_ids):
                    for j, block_id in enumerate(block_ids):
                        if block_id in migrate_record:
                            req.new_block_ids[i][j] = migrate_record[block_id]
            logger.info(f"Worker {self.rank} switched block ids for scheduler output version {scheduler_output.current_scheduler_output_version} with time taken: {human_readable_duration(time.time() - time_start)}")

    def resize_kv_cache(self, new_length: int) -> None:
        start_time = time.time()
        is_flexi = self.vllm_config.dynamic_config.enable_flexi_flash_attn
        if new_length == self.block_num:
            logger.info(f"kv cache length is already {new_length}, no need to resize")
            return
        self.block_num = new_length
        # logger.info(f"before resize kv cache, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        if is_flexi:
            self._flexi_resize_kv_cache(new_length)
        else:
            self._resize_kv_cache(new_length)

        logger.info(f"inside resize_kv_cache, after resizing kv cache, taking {human_readable_duration(time.time() - start_time)}")
        self.dynamic_kv_synchronizer.create_slot_mappings(new_length * self.block_size)
        logger.info(f"inside resize_kv_cache, after create slot mappings, taking {human_readable_duration(time.time() - start_time)}")

    def _resize_kv_cache(self, new_length: int) -> None:
        runner = self.model_runner
        assert new_length > 0
        with runner.forward_lock:
            time_start = time.time()
            assert isinstance(runner.model, DynamicQwen3ForCausalLM)
            logger.info(f"resizing kv cache from {len(runner.kv_caches[0][0])} to {new_length}")
            forward_context = self.vllm_config.compilation_config.static_forward_context
            kv, kv_length, T, H, Dh = runner.kv_caches[0].shape
            logger.info(f"num of kv tensors{len(runner.kv_caches)}")

            for layer_name, attn_module in forward_context.items():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.info(f"resizing kv cache for layer {layer_name}")
                layer_idx = extract_layer_index(layer_name)
                idx = layer_idx - runner.model.model.start_layer
                cache = runner.kv_caches[idx]
                tmp_cache = torch.zeros((kv, new_length, T, H, Dh), device=self.device, dtype=cache.dtype)
                logger.info(f"current thread: {threading.current_thread().name} tmp_cache shape: {tmp_cache.shape}, cache shape: {cache.shape}")
                if new_length > kv_length:
                    # 只复制旧的有效部分，新增的部分已经是0了
                    tmp_cache[:, :kv_length, ...].copy_(cache[:, :kv_length, ...])
                else:
                    tmp_cache[:, :new_length, ...].copy_(cache[:, :new_length, ...])
                dynamic_bind_single_kv_tensor(
                    layer_idx,
                    runner.model.model.start_layer,
                    runner.model.model.end_layer,
                    forward_context,
                    self.dynamic_kv_synchronizer,
                    runner,
                    tmp_cache
                )
            logger.info(f"[timeline]: resize kv cache within: {human_readable_duration(time.time() - time_start)}")

    def _flexi_resize_kv_cache(self, new_length: int) -> None:
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        logger.info(f"resizing kv cache from {len(self.model_runner.key_caches)} to {new_length}")
        logger.info(f"before resize kv cache, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        time_start = time.time()
        forward_context = self.vllm_config.compilation_config.static_forward_context
        T, H, Dh = self.model_runner.page_meta.shape
        cache_length = len(self.model_runner.key_caches[0])
        logger.info(f"num of kv tensors{cache_length}")

        start_layer, end_layer = self.model_runner.model.model.start_layer, self.model_runner.model.model.end_layer
        tmp_key_cache_list = []
        tmp_value_cache_list = []
        tmp_key_cache_ptr_list = []
        tmp_value_cache_ptr_list = []

        tmp_old_key_cache_list = []
        tmp_old_value_cache_list = []
        if new_length < cache_length:
            for layer_name, _ in forward_context.items():
                logger.info(f"resizing kv cache for layer {layer_name}")
                idx = extract_layer_index(layer_name)
                local_idx = idx - start_layer

                new_key_cache = self.model_runner.key_caches[local_idx][:new_length]
                new_value_cache = self.model_runner.value_caches[local_idx][:new_length]
                old_key_cache = self.model_runner.key_caches[local_idx][new_length:]
                old_value_cache = self.model_runner.value_caches[local_idx][new_length:]

                tmp_old_key_cache_list.append(old_key_cache)
                tmp_old_value_cache_list.append(old_value_cache)

                new_key_ptrs, new_value_ptrs = kv_allocator.prepare_flexi_kv_ptrs(new_key_cache, new_value_cache)
                tmp_key_cache_list.append(new_key_cache)
                tmp_value_cache_list.append(new_value_cache)
                tmp_key_cache_ptr_list.append(new_key_ptrs)
                tmp_value_cache_ptr_list.append(new_value_ptrs)

            torch.cuda.synchronize()
            tmp_old_key_cache_ptr_list = []
            tmp_old_value_cache_ptr_list = []
            with self.model_runner.forward_lock:
                start_time = time.time()
                for layer_name, _ in forward_context.items():
                    logger.info(f"resizing kv cache for layer {layer_name}")
                    idx = extract_layer_index(layer_name)
                    local_idx = idx - start_layer

                    # Free old GPU pointer arrays before allocating new ones
                    old_k_ptrs, old_v_ptrs = self.model_runner.key_cache_ptrs[local_idx], self.model_runner.value_cache_ptrs[local_idx]
                    tmp_old_key_cache_ptr_list.append(old_k_ptrs)
                    tmp_old_value_cache_ptr_list.append(old_v_ptrs)

                    key_cache = tmp_key_cache_list[local_idx]
                    value_cache = tmp_value_cache_list[local_idx]
                    new_k_ptrs, new_v_ptrs = tmp_key_cache_ptr_list[local_idx], tmp_value_cache_ptr_list[local_idx]

                    before_bind_time = time.time()
                    dynamic_flexi_bind_single_kv_cache(start_layer, end_layer,idx, key_cache, value_cache, new_k_ptrs, new_v_ptrs, forward_context, self.dynamic_kv_synchronizer, self.model_runner)

                    logger.info(f"[timeline]: bind single kv cache for layer {layer_name} take {human_readable_duration(time.time() - before_bind_time)}")
                logger.info(f"[timeline]: bound kv cache with smaller size, time taken: {human_readable_duration(time.time() - start_time)}")

            assert self.device is not None
            # Free old key and value ptrs
            for old_key_ptr, old_value_ptr in zip(tmp_old_key_cache_ptr_list, tmp_old_value_cache_ptr_list):
                kv_allocator.free_page_list(old_key_ptr, self.device)
                kv_allocator.free_page_list(old_value_ptr, self.device)

            # Free old key and value cache tensors
            for old_key_cache, old_value_cache in zip( tmp_old_key_cache_list, tmp_old_value_cache_list):
                kv_allocator.free_cache(old_key_cache, self.device)
                kv_allocator.free_cache(old_value_cache, self.device)

        elif new_length > cache_length:
            extended_kv_cache_shape = (T, H, Dh)
            tmp_new_allocated_key_cache = []
            tmp_new_allocated_value_cache = []
            torch.cuda.synchronize()
            tmp_old_key_cache_ptr_list = []
            tmp_old_value_cache_ptr_list = []
            # torch.cuda.empty_cache()
            for layer_name, attn_module in forward_context.items():
                local_idx = extract_layer_index(layer_name) - self.model_runner.model.model.start_layer

                new_allocated_block_num = new_length - cache_length
                new_allocated_key_cache, new_allocated_value_cache, new_k_ptrs, new_v_ptrs, _ = kv_allocator.allocate_with_cuda_async(new_allocated_block_num, list(extended_kv_cache_shape), self.model_runner.kv_cache_dtype, self.model_runner.device)
                get_flexi_kv_cache( new_allocated_block_num, extended_kv_cache_shape, self.model_runner.kv_cache_dtype, self.model_runner.device)


                key_cache = self.model_runner.key_caches[local_idx]
                value_cache = self.model_runner.value_caches[local_idx]

                new_key_cache = key_cache + new_allocated_key_cache
                new_value_cache = value_cache + new_allocated_value_cache

                tmp_new_allocated_key_cache.append(new_key_cache)
                tmp_new_allocated_value_cache.append(new_value_cache)
                
                # Add old ptrs for freeing later
                old_k_ptrs, old_v_ptrs = self.model_runner.key_cache_ptrs[local_idx], self.model_runner.value_cache_ptrs[local_idx]
                tmp_old_key_cache_ptr_list.append(old_k_ptrs)
                tmp_old_value_cache_ptr_list.append(old_v_ptrs)

                tmp_key_cache_ptr_list.append(new_k_ptrs)
                tmp_value_cache_ptr_list.append(new_v_ptrs)
            torch.cuda.synchronize()
            with self.model_runner.forward_lock:
                start_time = time.time()
                for layer_name, attn_module in forward_context.items():
                    idx = extract_layer_index(layer_name)
                    local_idx = idx - self.model_runner.model.model.start_layer
                    new_key_cache = tmp_new_allocated_key_cache[local_idx]
                    new_value_cache = tmp_new_allocated_value_cache[local_idx]
                    new_k_ptrs =  tmp_key_cache_ptr_list[local_idx]
                    new_v_ptrs =  tmp_value_cache_ptr_list[local_idx]
                    dynamic_flexi_bind_single_kv_cache(start_layer, end_layer, idx, new_key_cache, new_value_cache, new_k_ptrs, new_v_ptrs, forward_context, self.dynamic_kv_synchronizer, self.model_runner)
                logger.info(f"[timeline]: bound kv cache with larger size, time taken: {human_readable_duration(time.time() - start_time)} seconds")

            # Free old key and value ptrs
            for old_key_ptr, old_value_ptr in zip(tmp_old_key_cache_ptr_list, tmp_old_value_cache_ptr_list):
                kv_allocator.free_page_list(old_key_ptr, self.model_runner.device)
                kv_allocator.free_page_list(old_value_ptr, self.model_runner.device)

        torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        logger.info(f"[forward]: resized kv cache ,available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")

    # ####################################### #
    # Sender side of KV migration functions   #
    # ####################################### #
    def start_stress_tester(self) -> None:
        """Start the memory stress tester if enabled."""
        if self.memory_stress_tester is None:
            logger.info(f"Memory stress tester is not initialized in rank {self.rank}")
            return
            
        if not self.memory_stress_tester.enabled:
            logger.info("Memory stress tester is disabled")
            return
        
        # Check if already started
        if hasattr(self.memory_stress_tester, '_thread') and \
           self.memory_stress_tester._thread is not None and \
           self.memory_stress_tester._thread.is_alive():
            logger.warning("Memory stress tester already started and running")
            return
        
        # Start the tester
        self.memory_stress_tester.start()
        logger.info("Memory stress tester started in GPU worker (KV cache allocation style)")

    def start_kv_cache_migration_async(self, src_to_plan: dict[int, dict[int, list[int]]], slot_mapping: Optional[list[int]] = None) -> None:
        """Collective-RPC entry used by executor.

        Only the worker whose `self.rank == source_rank` performs the actual
        send; other ranks are no-ops. The receiver side should already be
        listening via `listen_to_kv_cache_tensor(source_rank)`.
        """
        # assert self.migration_in_process == False, "The migration should not be in process"
        logger.info(f"[operation]: Start KV Cache Migration: {src_to_plan}")
        logger.info(f"[operation]: rank {self.rank} start sending KV Cache Migration")
        assert len(self.rank_to_layers_ids) == 0
        if self.rank not in src_to_plan:
            logger.info(f"debug: rank {self.rank} is not sending kv cache")
            return None
        self.rank_to_layers_ids = src_to_plan[self.rank]
        self.kv_resizing_done = False

        def migration_thread(rank: int, layer_ids: list[int]):
            assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
            time_start = time.time()
            with self.model_runner.forward_lock:
                if self.rank not in src_to_plan:
                    return None
            # self.dynamic_kv_synchronizer.start_kv_tensor_transfer_async(rank_to_layers_ids, self.model_runner.kv_caches, self.model_runner.model.model.start_layer)
            self._sender_loop(rank, layer_ids,slot_mapping)
            logger.info(f"[timeline]: after start kv cache tensor, time taken: {human_readable_duration(time.time() - time_start)}")
        for rank, layer_ids in self.rank_to_layers_ids.items():
            threading.Thread(target=migration_thread, args=(rank, layer_ids), daemon=True).start()
        logger.info("finished starting kv cache migration async")
        return None

    def _sender_loop(self, 
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
            kv_tensor_meta, kv_tensor_data = self.dynamic_kv_synchronizer.get_kv_tensor_from_cache(layer_ids, layer_id, start_layer_id, self.model_runner.page_meta,  slot_mapping_dev)
            logger.info(f"start to send kv tensor for layer {layer_id} to rank {rank}, kv_tensor_meta: {kv_tensor_meta}, kv_tensor_data shape: {kv_tensor_data.shape}, time taken to get kv tensor: {human_readable_duration(time.time() - time_start)} seconds")

            self.dynamic_kv_synchronizer.send_kv_tensor_to_rank(rank, kv_tensor_meta, kv_tensor_data, slot_mapping_dev)
            logger.info(f"[debug]: sent kv tensor for layer {layer_id} to rank {rank}")

            # 将layer_ids中的第一个layer id对应的原生key cache list保存为一个文件
            # if is_flexi:
            #     first_layer_id = layer_ids[0]
            #     local_layer_idx = first_layer_id - start_layer_id
            #     key_cache_list = self.model_runner.key_caches[local_layer_idx]
            #     value_cache_list = self.model_runner.value_caches[local_layer_idx]
            #     save_path = f"/root/vllm_workbench/vllm/logs/tensors/sender_kv_cache_layer_{first_layer_id}.pt"
            #     torch.save({
            #         'key_cache': [k.cpu().clone() for k in key_cache_list],
            #         'value_cache': [v.cpu().clone() for v in value_cache_list],
            #         'layer_id': first_layer_id,
            #         'slot_mapping': slot_mapping_dev.cpu().clone() if slot_mapping_dev is not None else None,
            #         'block_num': len(key_cache_list),
            #     }, save_path)
            #     logger.info(f"[debug]: saved sender key_cache_list for layer {first_layer_id} to {save_path}")
        logger.info(f"[debug]: finished sending kv tensor, start to send kv patch")
        # 在发送完kv tensor之后开始发送kv patch
        for kv_patch in self.dynamic_kv_synchronizer.get_kv_patch(rank, start_layer_id, layer_ids, self.model_runner.page_meta):
            time_start = time.time()
            patch_payload_size = kv_patch.kv_payload.numel() * kv_patch.kv_payload.element_size()
            patch_slot_mapping_size = kv_patch.slot_mapping.numel() * kv_patch.slot_mapping.element_size()
            self.dynamic_kv_synchronizer.send_kv_patch_to_rank(rank, kv_patch)
            logger.info(f"[timeline]: send kv patch to rank {rank}, time taken: {time.time() - time_start}, data_size: {patch_slot_mapping_size / 1024 ** 2:.2f}MB + {patch_payload_size / 1024 ** 2:.2f}MB, kv patch id: {kv_patch.meta.id}, kv patch type: {kv_patch.meta.type}")


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
            time_start_bind_kv_cache = time.time()
            # 在kv cache绑定前，weight必须loading结束
            # streams = [torch.cuda.Stream() for _ in range(len(tmp_kv_tensors_dict))]
            for layer_id, kv_tensor in tmp_kv_tensors_dict.items():
                with self._layer_loaded_cv:
                    while not self.model_runner.has_layer(layer_id):
                        logger.info(f"[debug]: rank {self.rank} waiting for layer {layer_id} to be loaded")
                        self._layer_loaded_cv.wait()
                logger.info(f"[debug]: rank {self.rank} bind layer {layer_id}'s kv cache and weight")
                if is_flexi:
                    slot_mapping = tmp_slot_mapping_dict[layer_id]
                    assert self.device is not None
                    dynamic_flexi_bind_single_kv_tensor(
                        self.model_runner.model.model.start_layer,
                        self.model_runner.model.model.end_layer,
                        layer_id,
                        slot_mapping,
                        self.block_num,
                        kv_tensor,
                        self.vllm_config.compilation_config.static_forward_context,
                        self.dynamic_kv_synchronizer,
                        self.model_runner,
                        self.device,
                    )
                else:
                    # dynamic_bind_single_kv_tensor内部不使用所，因此我们需要在外面加锁
                    with self.model_runner.forward_lock:
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
            torch.cuda.synchronize()
            logger.info(f"[timeline]: bind kv cache time taken: {human_readable_duration(time.time() - time_start_bind_kv_cache)}")
            self.receiver_num_applied_token_dict[from_rank] = tmp_slot_token_num
            # 在同步迁移中，断言：所有接收层的 KV 已完成绑定且形状一致
            self._assert_layers_kv_bound(layers_to_be_received)


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
                self.dynamic_kv_synchronizer.apply_one_patch_to_kv_cache(self.model_runner.model.model.start_layer, meta, kv_payload, slot_mapping, self.model_runner.page_meta)
                self.receiver_num_applied_token_dict[from_rank] += meta.num_tokens
                logger.info(f"[num tokens]: receiver side: rank {from_rank} applied token: {meta.num_tokens}, total applied token: {self.receiver_num_applied_token_dict[from_rank]}")
                if meta.type == "kv_patch_meta":
                    logger.info(f"debug: ------------ Worker {self.rank} received kv patch meta from rank {from_rank}, kv payload shape: {kv_payload.shape}, slot mapping shape: {slot_mapping.shape}, num tokens: {meta.num_tokens}, patch id: {meta.id}")
                elif meta.type == "kv_patch_finished":
                    logger.info(f"Worker {self.rank} received kv patch finished message from rank {from_rank}")
                    # 通知所有等待所有KV cache patch都applied完成的线程
                    with self._all_patch_applied_cv:
                        logger.info(f"before notify, all applied status: {self.is_all_patch_applied}, applied_token_num: {self.after_migration_applied_token_num}")
                        self.is_all_patch_applied[from_rank] = True
                        self.after_migration_applied_token_num = self.receiver_num_applied_token_dict[from_rank]
                        self._all_patch_applied_cv.notify_all()
                        logger.info(f"after notify, all applied status: {self.is_all_patch_applied}, applied_token_num: {self.after_migration_applied_token_num}")
                    logger.info(f"[num tokens]: num of applied tokens from rank {from_rank}: {self.receiver_num_applied_token_dict[from_rank]}")
                    break # Exit point from current migration listening loop
                else:
                    assert False, f"Unexpected message type: {meta.type}"
                meta = self.dynamic_kv_synchronizer.recv_controller(from_rank)
            logger.info(f"[timeline]: after listen to kv cache patches, time taken: {human_readable_duration(time.time() - time_start)}")
        
    def all_patch_applied(self) -> bool:
        return all(self.is_all_patch_applied.values())

    # ########################################
    # Callback funtions executed inside the forwarding loop #
    # ######################################## 

    def async_migration_before_execute_callback(self,
                                                scheduler_output: "DynamicSchedulerOutput" 
                                               ) -> None:
        self._maybe_switch_block(scheduler_output)
        # 处于迁移中时，或这是同步批（用于发送 finished 信号），直接放行。
        if scheduler_output.migration_in_process:
            assert self.after_migration_total_token == 0, "If in migration process, the total after migration token num should be zero"
            logger.info(f"debug: rank {self.rank} is in migration or is sync after migration, skip kv synchronize before execute callback")
            return

        if self.after_migration_total_token != 0:
            # 等待所有KV cache补丁都应用完毕后再使用新的pp配置
            time_start = time.time()
            with self._all_patch_applied_cv:
                while not self.all_patch_applied():
                    self._all_patch_applied_cv.wait()
                logger.info(f"[timeline]: after waiting for all kv cache patch to be applied, time taken: {human_readable_duration(time.time() - time_start)}")

                assert self.after_migration_total_token == self.after_migration_applied_token_num, f"The after migration token num should be equal to the total migration token num, self.after_migration_total_token: {self.after_migration_total_token}, total_migration_token_num: {self.after_migration_applied_token_num}"
                # for rank in range(self.vllm_config.parallel_config.pipeline_parallel_size):
                #     if rank != self.rank:
                #         continue
                #     assert total_migration_token_num == self.receiver_num_applied_token_dict[rank], f"The total migration token num should be equal to the applied token num, total_migration_token_num: {total_migration_token_num}, applied token num: {self.receiver_num_applied_token_dict[rank]}"
            self.after_migration_total_token = 0
            self.after_migration_applied_token_num = 0

    def sync_migration_before_execute_callback(self, new_kv_cache_block_num: int) -> None:
        with self._sync_migration_cv:
            while self.receive_in_process:
                logger.info(f"debug: ---------------------rank {self.rank} waiting for migration to be finished")
                self._sync_migration_cv.wait()
            logger.info(f"debug: ---------------------rank {self.rank} is good for forwarding")
        # if new_kv_cache_block_num != 0:
        #     self.resize_kv_cache(new_kv_cache_block_num)

    def async_migration_after_execute_callback(self, scheduler_output: "DynamicSchedulerOutput"):
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)

        # target_device = self.device  # set in init_device to cuda:self.local_rank
        # if slot_mapping.device != target_device:
        #     slot_mapping = slot_mapping.to(target_device, non_blocking=True)

        # First, enqueue this step's KV patch (if sending) so the latest tokens are transferred.
        is_sync = scheduler_output.is_sync_after_migration
        num_total_new_tokens = scheduler_output.total_num_scheduled_tokens
        num_total_migration_tokens = scheduler_output.total_migration_tokens

        if scheduler_output.migration_in_process:
            assert scheduler_output.sender_list is not None and scheduler_output.receiver_list is not None, "If in migration process, the sender list should not be None"
            is_sender = self.rank in scheduler_output.sender_list
            is_receiver = self.rank in scheduler_output.receiver_list

            if is_sender:
                # Send the kv cache patch to the other ranks if the migration is in process
                slot_mapping = self.model_runner.input_batch.block_table[0].slot_mapping
                # Ensure slot_mapping is on this worker's CUDA device
                assert slot_mapping.device.type == "cuda", f"slot_mapping must be on CUDA, got {slot_mapping.device}"
                for rank in self.rank_to_layers_ids:
                    self.dynamic_kv_synchronizer.add_new_tokens_to_kv_synchronizer(
                        rank,
                        slot_mapping,
                        is_sync,
                        num_total_new_tokens
                    )

            if is_sync:
                def do_resize():
                    assert scheduler_output.sender_list is not None, "If is sync migration, the sender list should not be None"
                    time_start = time.time()
                    logger.info(f"[operation]: before finish kv cache transfer start to finish kv cache tansfer, delete layers, release kv cache, reinitialize kv cache")
                    if is_sender:
                        # Sender finishes transfer and cleans up; receiver no-ops.
                        layer_ranges = []
                        for layers in self.rank_to_layers_ids.values():
                            # Assert the layers are sorted
                            assert layers == sorted(layers), "The layers should be sorted"
                            layer_ranges.append((layers[0], layers[-1]))
                        self.release_kv_cache_for_layers(self.rank, layer_ranges)
                        logger.info(f"[timeline]: after release kv cache for layers, time taken: {human_readable_duration(time.time() - time_start)}")
                        self.remove_layers(self.rank, layer_ranges)
                        logger.info(f"[timeline]: after remove layers, time taken: {human_readable_duration(time.time() - time_start)}")
                    self.resize_kv_cache(scheduler_output.new_kv_cache_block_num)
                    self.kv_resizing_done = True
                    self.finish_migration()
                    logger.info(f"[timeline]: after resize kv cache for layers, time taken: {human_readable_duration(time.time() - time_start)}")

                if is_receiver:
                    self.after_migration_total_token = num_total_migration_tokens
                threading.Thread(target=do_resize, daemon=True).start()

        # else:
        #     assert not is_sync, "If not sending kv cache, it should not be sync migration"

    def finish_migration(self):
        self.rank_to_layers_ids = {}

        # assert self.dynamic_layer_kv_connector.is_all_patch_applied(), "All patch should be applied"

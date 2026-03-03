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
from vllm.model_executor.models.dynamic_model_base import DynamicModelBase
from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import create_ptr_tensor_from_list, dynamic_bind_single_kv_tensor, dynamic_flexi_bind_single_kv_cache, dynamic_flexi_bind_single_kv_tensor, get_layer_name_for_index, report_usage_stats, WorkerMemInfo
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
from vllm.v1.worker.gpu_memory_monitor import (
    get_gpu_memory_monitor, memory_snapshot, 
    get_checkpoint_tracker, reset_checkpoint_tracker
)
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
        self._receive_finished_lock = threading.Lock()
        self._receive_finished_cv = threading.Condition(self._receive_finished_lock)

        # 独立的条件变量与锁：等待新层加载（避免在等待时占用 forward_lock）
        self._layer_loaded_lock = threading.Lock()
        self._layer_loaded_cv = threading.Condition(self._layer_loaded_lock)
        # 独立的条件变量：等待 KV cache 绑定到位（与“层已加载”解耦）
        self._kv_bound_lock = threading.Lock()
        self._kv_bound_cv = threading.Condition(self._kv_bound_lock)


        pp_size = int(self.vllm_config.parallel_config.pipeline_parallel_size)
        # Used for waiting all patch applied
        self._all_patch_applied_cv = threading.Condition(threading.Lock()) 
        self.is_all_patch_applied = {}
        for rank in range(pp_size):
            self.is_all_patch_applied[rank] = True
        self.resizing_done_cv = threading.Condition(threading.Lock())
        self.resizing_done = True
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

        # streams
        self.inference_stream = None
        self.migration_stream = None
        self._nccl_lock = threading.Lock()

        # 记录target pp config
        self.target_pp_layer_config: Optional[list[Tuple[int, int]]] = None


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
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
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
        assert isinstance(self.model_runner.model, DynamicModelBase)
        logger.info(f"[operation]: Add Model Layers: {layer_list}")
        time_start = time.time()
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        torch.cuda.synchronize() 
        gc.collect()
        torch.cuda.empty_cache()
        
        # Calculate expected memory change for adding layers
        num_added_layers = sum([layer[1] - layer[0] + 1 for layer in layer_list])
        weight_size_per_layer_gb = self.model_runner.model.get_layer_weight_size() / (1024**3)
        expected_weight_delta_gb = -weight_size_per_layer_gb * num_added_layers  # Negative = less free mem
        
        # Memory checkpoint: track expected weight loading impact
        tracker = get_checkpoint_tracker(self.rank, self.device)
        before_idx = tracker.checkpoint_before(
            tag=f"add_layers_{layer_list}",
            operation="add_layers_weights",
            expected_delta_gb=expected_weight_delta_gb,
            details={
                'num_layers': num_added_layers, 
                'weight_per_layer_gb': weight_size_per_layer_gb,
                'layer_list': layer_list
            }
        )
        memory_snapshot(f"rank{self.rank}_before_add_layers_{layer_list}", self.device)

        with DeadlockTimeoutContext(self._layer_loaded_cv, "_layer_loaded_cv", timeout=2):
            assert self.migration_stream is not None
            with self.migration_stream:
                # with torch.cuda.stream(self.migration_stream):
                logger.info(f"start to load layer, current stream:{torch.cuda.current_stream()}")
                # 记录扩容前的起始 layer，便于在 start_layer 左移时重排本地 kv 索引基准
                old_start_layer = self.model_runner.model.model.start_layer
                old_end_layer = self.model_runner.model.model.end_layer
                assert self.device is not None
                self.model_runner.add_layers(layer_list, self.device)

            self.migration_stream.synchronize()
            logger.info(f"[timeline]: after weight loading, time taken: {human_readable_duration(time.time() - time_start)}")
            memory_snapshot(f"rank{self.rank}_after_weight_loading_{layer_list}", self.device)
            
            # Verify memory change after weight loading
            tracker.checkpoint_after(
                tag=f"add_layers_{layer_list}_weights_loaded",
                operation="add_layers_weights",
                before_idx=before_idx,
                expected_delta_gb=expected_weight_delta_gb
            )

            # 接收完weights之后，在这里进行kv cache数据结构的扩展，保证后续kv cache tensor绑定的正确性
            new_start_layer = self.model_runner.model.model.start_layer
            new_end_layer = self.model_runner.model.model.end_layer
            logger.info(f"debug: ---------------------add layers, old_start_layer: {old_start_layer}, new_start_layer: {new_start_layer}, old_end_layer: {old_end_layer}, new_end_layer: {new_end_layer}")

            with self.model_runner.forward_lock:
                time_within_lock_start = time.time()
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
                        # Pad key_handles and value_handles for VMM mode
                        self.model_runner.key_handles = [[] for _ in range(left_added_layer_num)] + \
                                                        self.model_runner.key_handles
                        self.model_runner.value_handles = [[] for _ in range(left_added_layer_num)] + \
                                                        self.model_runner.value_handles
                        # Pad k_ptr_tensors and v_ptr_tensors with empty tensors (direct mode only)
                        if self.vllm_config.dynamic_config.use_direct_ptr:
                            empty_tensor = torch.tensor([], dtype=torch.uint64, device=self.device)
                            self.model_runner.k_ptr_tensors = [empty_tensor.clone() for _ in range(left_added_layer_num)] + \
                                                            self.model_runner.k_ptr_tensors
                            self.model_runner.v_ptr_tensors = [empty_tensor.clone() for _ in range(left_added_layer_num)] + \
                                                            self.model_runner.v_ptr_tensors

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
                        # Pad key_handles and value_handles for VMM mode
                        self.model_runner.key_handles.extend([[] for _ in range(pad_right)])
                        self.model_runner.value_handles.extend([[] for _ in range(pad_right)])
                        # Pad k_ptr_tensors and v_ptr_tensors with empty tensors (direct mode only)
                        if self.vllm_config.dynamic_config.use_direct_ptr:
                            empty_tensor = torch.tensor([], dtype=torch.uint64, device=self.device)
                            self.model_runner.k_ptr_tensors.extend([empty_tensor.clone() for _ in range(pad_right)])
                            self.model_runner.v_ptr_tensors.extend([empty_tensor.clone() for _ in range(pad_right)])

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
                    for i in range(len(self.model_runner.kv_caches)):
                        logger.info(f"kv cache {i}: {self.model_runner.kv_caches[i].shape}")
                logger.info(f"[timeline]: time within lock when add layers time taken: {human_readable_duration(time.time() - time_within_lock_start)}")
        
            logger.info(f"kv scynchronizer kv cache list length after adding layers: {len(self.dynamic_kv_synchronizer.key_cache_ptrs)}")
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
            logger.info(f"Rank {self.rank}: initializing on device {self.device}")
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
        kv_allocator.set_lock(self.model_runner.fbgate)
        self.inference_stream = torch.cuda.Stream(device=self.device, priority=-5)
        logger.info(f"high priority inference stream created: {self.inference_stream}")
        # Set inference stream in model_runner for stream-specific synchronization
        self.model_runner.set_inference_stream(self.inference_stream)
        self.migration_stream = torch.cuda.Stream(device=self.device)
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
        
    def _is_dynamic_model(self) -> bool:
        """Check if the loaded model is a DynamicModelBase instance."""
        return isinstance(self.model_runner.model, DynamicModelBase)

    def load_model(self) -> None:
        super().load_model()
        
        # Check if model supports dynamic features
        if not self._is_dynamic_model():
            logger.warning(
                "Loaded model is not a DynamicModelBase instance. "
                "Dynamic PP reconfiguration features will be disabled. "
                f"Model type: {type(self.model_runner.model).__name__}"
            )
            self.dynamic_kv_synchronizer = None
            return
            
        self.model_runner.model.model.add_fbgate(self.model_runner.fbgate)
        # Wait for all ranks to finish loading model before initializing KV synchronizer
        # This prevents deadlock where faster ranks (fewer layers) enter barrier
        # while slower ranks (more layers) are still loading
        if torch.distributed.is_initialized():
            logger.info(f"Rank {self.rank}: waiting for all ranks to finish model loading before KV synchronizer init")
            torch.distributed.barrier()
            logger.info(f"Rank {self.rank}: all ranks ready, initializing KV synchronizer")
        assert self.device is not None
        # Initialize KV synchronizer AFTER model is ready (needs model for args)
        # Pass the NCCL lock to prevent deadlock with Ray's compiled_dag
        self.dynamic_kv_synchronizer = DynamicKVSynchronizer(
            rank=self.rank,
            local_rank=self.local_rank,
            config=self.vllm_config,
            model_executable=self.model_runner.model,
            device=self.device,
            nccl_lock=self._nccl_lock
        )
        # 启动所有监听其它rank的发送过来的kv cache的线程
        for rank in range(self.vllm_config.parallel_config.pipeline_parallel_size):
            if rank != self.rank:
                # threading.Thread(target=self.listen_to_kv_cache_tensor_and_patches, args=(rank,), daemon=True).start()
                self.listen_to_kv_cache_tensor_and_patches(rank)

    def dynamic_initialize_from_config(self, kv_cache_configs: list[KVCacheConfig], num_blocks: int) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        # For non-dynamic models, use the parent class's standard initialization
        if not self._is_dynamic_model():
            logger.info("Using standard KV cache initialization for non-dynamic model")
            # Recalculate num_blocks based on tensor_config to avoid assertion failure
            # The dynamic core sets num_blocks globally, but we need to use what's
            # actually available based on tensor_config.size
            kv_cache_config = kv_cache_configs[self.rank]
            min_num_blocks = float('inf')
            for kv_cache_group in kv_cache_config.kv_cache_groups:
                kv_cache_spec = kv_cache_group.kv_cache_spec
                for layer_name in kv_cache_group.layer_names:
                    tensor_config = kv_cache_config.tensors[layer_name]
                    local_num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                    min_num_blocks = min(min_num_blocks, local_num_blocks)
            if min_num_blocks < float('inf'):
                kv_cache_config.num_blocks = int(min_num_blocks)
                logger.info(f"Recalculated num_blocks for non-dynamic model: {kv_cache_config.num_blocks}")
            super().initialize_from_config(kv_cache_config)
            return
            
        self.block_size = kv_cache_configs[self.rank].kv_cache_groups[0].kv_cache_spec.block_size
        self.block_num = num_blocks
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            if self.vllm_config.dynamic_config.use_flexi_kv:
                logger.info("Using flexi flash attention dynamic initialize kv cache")
                self.model_runner.dynamic_initialize_kv_cache_flexi(kv_cache_configs[self.rank], self.dynamic_kv_synchronizer, num_blocks)
            else:
                logger.info("Using standard flash attention dynamic initialize kv cache")
                self.model_runner.dynamic_initialize_kv_cache(kv_cache_configs[self.rank], self.dynamic_kv_synchronizer, num_blocks)
            self.dynamic_kv_synchronizer.create_slot_mappings(num_blocks * self.block_size)
            
            # Record initial GPU memory for this configuration
            # This establishes a baseline to detect memory leaks after migration
            if isinstance(self.model_runner.model, DynamicModelBase):
                start_layer = self.model_runner.model.model.start_layer
                end_layer = self.model_runner.model.model.end_layer
                memory_monitor = get_gpu_memory_monitor()
                memory_monitor.record_initial_memory_local(
                    start_layer, end_layer, self.rank, self.device
                )

    def set_env_var(self, key: str, value: str) -> None:
        """Update an environment variable in this worker process."""
        os.environ[key] = value
        logger.info(f"Worker {self.rank}: Updated env var {key}={value}")

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        layer_config: Tuple[int, int],
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        assert False, "This function is not used"
        assert(isinstance(self.model_runner.model, DynamicModelBase))
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
        assert isinstance(self.model_runner.model, DynamicModelBase) or isinstance(self.model_runner.model, torch.nn.Module), "model should be an instance of DynamicModelBase or torch.nn.Module"
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
            # Set CUDA device for this thread - threads don't inherit CUDA context
            torch.cuda.set_device(self.device)
            self._add_layers(layer_list)
        # 异步调用：与工作的 commit 9ed5ec00f2c94 保持一致
        # 注意：daemon=False 确保线程在进程退出前完成
        threading.Thread(target=_do_add, daemon=False).start()
        logger.info(f"[timeline]: after add layers, time taken: {human_readable_duration(time.time() - time_start)}")

    def add_layers(self, rank: int, layer_list: list[Tuple[int, int]]) -> None:
        if self.rank != rank:
            logger.info(f"Worker {self.rank} is not the target rank {rank}, skip adding model layers")
            return None
        logger.info(f"[operation]: Add Model Layers: {layer_list}")
        time_start = time.time()
        self._add_layers(layer_list)
        logger.info(f"[timeline]: after add layers, time taken: {human_readable_duration(time.time() - time_start)}")

    def remove_layers(self, rank: int, layer_list: list[Tuple[int, int]]) -> None:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip removing model layers")
            return None
        logger.info(f"Remove Model Layers: {layer_list}")
        assert self.device is not None
        
        # # Calculate expected memory change for removing layers
        # num_removed_layers = sum([layer[1] - layer[0] + 1 for layer in layer_list])
        # assert isinstance(self.model_runner.model, DynamicModelBase)
        # weight_size_per_layer_gb = self.model_runner.model.get_layer_weight_size() / (1024**3)
        # expected_weight_delta_gb = weight_size_per_layer_gb * num_removed_layers  # Positive = more free mem
        
        # # Memory checkpoint: track expected weight removal impact
        # tracker = get_checkpoint_tracker(self.rank, self.device)
        # before_idx = tracker.checkpoint_before(
        #     tag=f"remove_layers_{layer_list}",
        #     operation="remove_layers",
        #     expected_delta_gb=expected_weight_delta_gb,
        #     details={
        #         'num_layers': num_removed_layers, 
        #         'weight_per_layer_gb': weight_size_per_layer_gb,
        #         'layer_list': layer_list
        #     }
        # )
        
        self.model_runner.remove_layers(layer_list, self.device)
        
        # # Verify memory change after removing layers (weights should be freed)
        # tracker.checkpoint_after(
        #     tag=f"remove_layers_{layer_list}_done",
        #     operation="remove_layers",
        #     before_idx=before_idx,
        #     expected_delta_gb=expected_weight_delta_gb
        # )

    def atomic_shelve_kv_cache(self, rank: int, layers_list: list[Tuple[int, int]]) ->Tuple[list[list[int]], list[list[int]], list[int], list[int], list[list[int]], list[list[int]]]:
        """
        Atomically shelve KV cache for specified layers.
        
        Returns:
            Tuple of (caches_to_free_key, caches_to_free_value, ptrs_to_free_key, ptrs_to_free_value,
                      handles_to_free_key, handles_to_free_value)
        """
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip releasing kv cache for layers")
            return [], [], [], [], [], []
        # self.model_runner.release_kv_cache_for_layers(layers_list)
        from vllm.config import get_current_vllm_config
        vllm_config = get_current_vllm_config()
        is_flexi = vllm_config.dynamic_config.use_flexi_kv
        start_layer = self.model_runner.model.model.start_layer
        free_before, total = torch.cuda.mem_get_info()
        logger.info(f"before release_kv_cache_for_layers: free={free_before / 1024 ** 3:.2f} GB, total={total / 1024 ** 3:.2f} GB")
        caches_to_free_key: list[list[int]] = []
        caches_to_free_value: list[list[int]] = []
        ptrs_to_free_key: list[int] = []
        ptrs_to_free_value: list[int] = []
        handles_to_free_key: list[list[int]] = []
        handles_to_free_value: list[list[int]] = []
        if is_flexi:
            # Collect the key/value caches, ptrs, and handles to be freed BEFORE removing from lists
            for idx, (key_cache, value_cache) in enumerate(zip(self.model_runner.key_caches, self.model_runner.value_caches)):
                if any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list):
                    caches_to_free_key.append(key_cache)
                    caches_to_free_value.append(value_cache)
                    # Also collect VMM handles if available
                    if self.model_runner.key_handles and idx < len(self.model_runner.key_handles):
                        handles_to_free_key.append(self.model_runner.key_handles[idx])
                    else:
                        handles_to_free_key.append([])
                    if self.model_runner.value_handles and idx < len(self.model_runner.value_handles):
                        handles_to_free_value.append(self.model_runner.value_handles[idx])
                    else:
                        handles_to_free_value.append([])
            for idx, (k_ptr, v_ptr) in enumerate(zip(self.model_runner.key_cache_ptrs, self.model_runner.value_cache_ptrs)):
                if any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list):
                    ptrs_to_free_key.append(k_ptr)
                    ptrs_to_free_value.append(v_ptr)

            self.model_runner.flexi_atomic_switch_kv_cache_config_for_layers(layers_list)
            
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
            self.model_runner.atomic_switch_kv_cache_config_for_layers(layers_list)
            self.dynamic_kv_synchronizer.kv_caches = [
                kv_cache for idx, kv_cache in enumerate(self.dynamic_kv_synchronizer.kv_caches) 
                if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
            ]
        logger.info(f"after atomic shelve kv cache for layers, kv ptr tensor:{self.model_runner.k_ptr_tensors}, {self.model_runner.v_ptr_tensors}")
        return caches_to_free_key, caches_to_free_value, ptrs_to_free_key, ptrs_to_free_value, handles_to_free_key, handles_to_free_value


    def release_kv_cache_for_layers(
        self, 
        rank: int,  
        caches_to_free_key: list[list[int]], 
        caches_to_free_value: list[list[int]], 
        ptrs_to_free_key: list[int], 
        ptrs_to_free_value: list[int],
        handles_to_free_key: Optional[list[list[int]]] = None,
        handles_to_free_value: Optional[list[list[int]]] = None
    ) -> None:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip releasing kv cache for layers")
            return None
        # self.model_runner.release_kv_cache_for_layers(layers_list)
        from vllm.config import get_current_vllm_config
        vllm_config = get_current_vllm_config()
        is_flexi = vllm_config.dynamic_config.use_flexi_kv
        free_before, total = torch.cuda.mem_get_info()
        logger.info(f"before release_kv_cache_for_layers: free={free_before / 1024 ** 3:.2f} GB, total={total / 1024 ** 3:.2f} GB")
        
        # Calculate expected memory freed from KV cache using page_meta
        kv_cache_bytes_to_free = 0
        ptr_bytes_to_free = 0
        num_layers = len(caches_to_free_key) if caches_to_free_key else 0
        num_blocks = len(caches_to_free_key[0]) if caches_to_free_key and len(caches_to_free_key) > 0 else 0
        
        # if is_flexi and caches_to_free_key and self.model_runner.page_meta is not None:
        #     # Use page_meta to calculate block size: (block_size, num_heads, head_size)
        #     page_meta = self.model_runner.page_meta
        #     T, H, Dh = page_meta.shape
        #     dtype_size = page_meta.element_size()
        #     bytes_per_block = T * H * Dh * dtype_size  # Size of one K or V block
            
        #     # Total KV cache bytes = num_layers * num_blocks * bytes_per_block * 2 (K+V)
        #     kv_cache_bytes_to_free = num_layers * num_blocks * bytes_per_block * 2
            
        #     # Pointer arrays: num_blocks * sizeof(void*) = num_blocks * 8 bytes per layer
        #     ptr_bytes_to_free = num_layers * num_blocks * 8 * 2  # 2 for K+V pointer arrays
        
        # expected_delta_gb = (kv_cache_bytes_to_free + ptr_bytes_to_free) / (1024**3)
        
        # # Memory checkpoint: track expected KV cache release impact
        # tracker = get_checkpoint_tracker(self.rank, self.device)
        # before_idx = tracker.checkpoint_before(
        #     tag=f"release_kv_cache_{num_layers}layers_{num_blocks}blocks",
        #     operation="release_kv_cache_for_layers",
        #     expected_delta_gb=expected_delta_gb,
        #     details={
        #         'num_layers': num_layers, 
        #         'num_blocks': num_blocks,
        #         'kv_cache_bytes': kv_cache_bytes_to_free,
        #         'ptr_bytes': ptr_bytes_to_free
        #     }
        # )
        
        if is_flexi:
            assert len(caches_to_free_key) != 0
            assert len(caches_to_free_value) != 0
            assert len(ptrs_to_free_key) != 0
            assert len(ptrs_to_free_value) != 0
            # Free GPU memory for the released KV caches
            assert self.device is not None
            logger.info(f"[release_kv_cache_for_layers] Freeing {len(caches_to_free_key)} layers of KV cache GPU memory")
            logger.info(f"[release_kv_cache_for_layers] Each layer has {len(caches_to_free_key[0]) if caches_to_free_key else 0} blocks to free")
            
            # Check allocation mode to use correct free function
            use_vmm = self.model_runner.vmm_aligned_bytes > 0
            combined_mode = getattr(self.model_runner, 'vmm_combined_mode', False)
            aligned_bytes = self.model_runner.vmm_aligned_bytes
            
            # Ensure handles lists have correct length
            if handles_to_free_key is None:
                handles_to_free_key = [[] for _ in caches_to_free_key]
            if handles_to_free_value is None:
                handles_to_free_value = [[] for _ in caches_to_free_value]
            
            for i, (key_cache, value_cache) in enumerate(zip(caches_to_free_key, caches_to_free_value)):
                key_handles = handles_to_free_key[i] if i < len(handles_to_free_key) else []
                value_handles = handles_to_free_value[i] if i < len(handles_to_free_value) else []
                logger.info(f"[release_kv_cache_for_layers] Freeing layer {i}: key_cache len={len(key_cache)}, "
                           f"value_cache len={len(value_cache)}, use_vmm={use_vmm}, combined_mode={combined_mode}, "
                           f"key_handles len={len(key_handles)}, value_handles len={len(value_handles)}")
                with self.model_runner.fbgate.background():
                    if use_vmm and key_handles:
                        # VMM mode with handles available
                        if combined_mode:
                            # Combined mode: K and V share same physical page, only free once using K ptr and handle
                            kv_allocator.free_vmm_blocks_combined(key_cache, key_handles, aligned_bytes, self.device)
                        else:
                            # Separate mode: free K and V independently
                            kv_allocator.free_vmm_blocks(key_cache, key_handles, aligned_bytes, self.device)
                            if value_handles:
                                kv_allocator.free_vmm_blocks(value_cache, value_handles, aligned_bytes, self.device)
                    elif use_vmm:
                        # VMM mode but no handles - warn and skip (memory leak)
                        logger.warning(f"[release_kv_cache_for_layers] VMM mode but no handles for layer {i}. "
                                       f"Memory will be leaked.")
                    else:
                        # cudaMallocAsync mode
                        kv_allocator.free_cache(key_cache, self.device)
                        kv_allocator.free_cache(value_cache, self.device)
            
            for k_ptr, v_ptr in zip(ptrs_to_free_key, ptrs_to_free_value):
                with self.model_runner.fbgate.background():
                    kv_allocator.free_page_list(k_ptr, self.device)
                    kv_allocator.free_page_list(v_ptr, self.device)
        
        # # Synchronize to ensure all frees complete before measuring
        # torch.cuda.synchronize()
        
        # # Verify memory change after release
        # tracker.checkpoint_after(
        #     tag=f"release_kv_cache_{num_layers}layers_{num_blocks}blocks_done",
        #     operation="release_kv_cache_for_layers",
        #     before_idx=before_idx,
        #     expected_delta_gb=expected_delta_gb
        # )
        
        final_free, _ = torch.cuda.mem_get_info()
        logger.info(f"after release_kv_cache_for_layers: free={final_free / 1024 ** 3:.2f} GB, model runner's kv cache length: {len(self.model_runner.kv_caches)}")

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
        if hasattr(self.model_runner.model, 'get_layer_weight_size'):
            layer_size = int(self.model_runner.model.get_layer_weight_size())
        else:
            # Fallback for models without get_layer_weight_size
            layer_size = 0
            logger.warning("get_mem_info: Model has no get_layer_weight_size method, layer_size set to 0")

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
        runner = self.model_runner
        start_layer, end_layer = runner.model.model.start_layer, runner.model.model.end_layer
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        assert isinstance(runner.model, DynamicModelBase)
        logger.info(f"start to compact kv cache for layers {runner.model.model.start_layer} to {runner.model.model.end_layer}")
        time_start = time.time()
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
        # Always track old GPU pointer arrays for freeing after rebind
        tmp_old_key_ptrs = list(self.model_runner.key_cache_ptrs)
        tmp_old_value_ptrs = list(self.model_runner.value_cache_ptrs)
        use_direct = self.vllm_config.dynamic_config.use_direct_ptr
        
        def _migrate_block_by_swapping_ptrs(old_block_id: int, new_block_id: int, migrate_record: dict[int, int]):
            assert len(tmp_key_cache) != 0
            for key_cache in tmp_key_cache:
                key_cache[new_block_id], key_cache[old_block_id] = key_cache[ old_block_id], key_cache[ new_block_id]
            for value_cache in tmp_value_cache:
                value_cache[ new_block_id], value_cache[ old_block_id] = value_cache[ old_block_id], value_cache[ new_block_id]
            migrate_record[old_block_id] = new_block_id

        if runner.vllm_config.dynamic_config.use_flexi_kv:
            compact_cache_with_record(_migrate_block_by_swapping_ptrs, is_used, compacted_length, num_blocks, migrate_record)

            forward_context = runner.vllm_config.compilation_config.static_forward_context
            # 🔴 FIX: Sort forward_context by layer index to ensure consistent ordering
            sorted_forward_context = sorted(forward_context.items(), key=lambda x: extract_layer_index(x[0]))

            # Rebuild GPU pointer arrays AFTER compaction (needed by both flexi and direct kernels)
            tmp_new_key_ptrs: list[int] = []
            tmp_new_value_ptrs: list[int] = []
            for layer_name, attn_module in sorted_forward_context:
                idx = extract_layer_index(layer_name)
                local_idx = idx - start_layer
                with self.model_runner.fbgate.background():
                    new_k_ptrs, new_v_ptrs = kv_allocator.prepare_flexi_kv_ptrs(
                        tmp_key_cache[local_idx], tmp_value_cache[local_idx])
                logger.info(f"prepare kv ptr for layer {layer_name}, new_k_ptrs: {hex(new_k_ptrs)}, new_v_ptrs: {hex(new_v_ptrs)}")
                tmp_new_key_ptrs.append(new_k_ptrs)
                tmp_new_value_ptrs.append(new_v_ptrs)

            # PtrTensors only needed by direct kernel — build AFTER compaction
            if use_direct:
                tmp_new_key_ptr_tensors: list[torch.Tensor] = [
                    create_ptr_tensor_from_list(kc, self.model_runner.device) for kc in tmp_key_cache]
                tmp_new_value_ptr_tensors: list[torch.Tensor] = [
                    create_ptr_tensor_from_list(vc, self.model_runner.device) for vc in tmp_value_cache]

            # Bind the new kv cache
            with self.model_runner.forward_lock:
                time_start_within_lock = time.time()
                for layer_name, attn_module in sorted_forward_context:
                    idx = extract_layer_index(layer_name)
                    local_idx = idx - start_layer
                    new_k_cache, new_v_cache = tmp_key_cache[local_idx], tmp_value_cache[local_idx]
                    new_k_ptrs, new_v_ptrs = tmp_new_key_ptrs[local_idx], tmp_new_value_ptrs[local_idx]
                    if use_direct:
                        new_k_ptr_tensor = tmp_new_key_ptr_tensors[local_idx]
                        new_v_ptr_tensor = tmp_new_value_ptr_tensors[local_idx]
                    else:
                        new_k_ptr_tensor = None
                        new_v_ptr_tensor = None
                    dynamic_flexi_bind_single_kv_cache(start_layer, end_layer, idx, new_k_cache, new_v_cache, new_k_ptrs, new_v_ptrs, new_k_ptr_tensor, new_v_ptr_tensor, forward_context, self.dynamic_kv_synchronizer, runner)
                
                # Update stacked tensors cache (direct mode only, must be before freeing old pointers)
                if use_direct:
                    self.model_runner.update_kv_ptr_tensor(tmp_new_key_ptr_tensors, tmp_new_value_ptr_tensors)

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
                logger.info(f"[timeline]: kv cache compaction within lock take {human_readable_duration(time.time() - time_start_within_lock)}")
                self.migration_records_for_specific_scheduler_output_version[self.cur_scheduler_output_version] = migrate_record
                self.cur_scheduler_output_version += 1
            assert self.device is not None
            # Free old GPU pointer arrays (needed by both flexi and direct)
            for old_k_ptrs, old_v_ptrs in zip(tmp_old_key_ptrs, tmp_old_value_ptrs):
                with self.model_runner.fbgate.background():
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
                logger.info(f"[timeline]: kv cache compaction within lock take {human_readable_duration(time.time() - time_start_within_lock)}")
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
        logger.info(f"[timeline]: compact kv cache total time taken: {human_readable_duration(time.time() - time_start)}")
    
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
        # Ensure correct CUDA device is set for this worker (important for Ray RPC calls)
        if self.device is not None:
            torch.cuda.set_device(self.device)
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        old_length = self.block_num
        if new_length == old_length:
            logger.info(f"kv cache length is already {new_length}, no need to resize, sleep 2 seconds")
            return
        
        # # Calculate expected memory change for KV cache resize
        # # Each block contains: 2 (K+V) * tokens_per_block * num_heads * head_size * dtype_size
        # # For flexi mode: num_layers * num_blocks_diff * page_size
        # assert isinstance(self.model_runner.model, DynamicModelBase)
        # num_layers = len(self.model_runner.key_caches) if is_flexi else len(self.model_runner.kv_caches)
        
        # if is_flexi:
        #     # Flexi mode: each block is T * H * Dh * dtype_size bytes
        #     T, H, Dh = self.model_runner.page_meta.shape
        #     dtype_size = self.model_runner.page_meta.dtype.itemsize
        #     bytes_per_block = T * H * Dh * dtype_size * 2  # 2 for K+V
        # else:
        #     # Non-flexi mode: full tensor shape
        #     if len(self.model_runner.kv_caches) > 0:
        #         kv_tensor = self.model_runner.kv_caches[0]
        #         numel_per_block = kv_tensor[0][0].numel()  # Single block size
        #         dtype_size = kv_tensor.element_size()
        #         bytes_per_block = numel_per_block * dtype_size * 2  # 2 for K+V in [2, blocks, ...]
        #     else:
        #         bytes_per_block = 0
        
        # block_diff = old_length - new_length  # Positive if shrinking
        # kv_cache_delta_gb = (block_diff * num_layers * bytes_per_block) / (1024**3)
        
        # # Memory checkpoint: track expected KV cache resize impact
        # tracker = get_checkpoint_tracker(self.rank, self.device)
        # before_idx = tracker.checkpoint_before(
        #     tag=f"resize_kv_{old_length}_to_{new_length}",
        #     operation="resize_kv_cache",
        #     expected_delta_gb=kv_cache_delta_gb,
        #     details={
        #         'old_length': old_length, 
        #         'new_length': new_length,
        #         'num_layers': num_layers,
        #         'bytes_per_block': bytes_per_block,
        #         'block_diff': block_diff
        #     }
        # )
        
        # memory_snapshot(f"rank{self.rank}_before_resize_{old_length}_to_{new_length}", self.device)
        self.block_num = new_length
        # logger.info(f"before resize kv cache, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        if is_flexi:
            self._flexi_resize_kv_cache(new_length)
        else:
            self._resize_kv_cache(new_length)

        # Verify memory change after resize
        # tracker.checkpoint_after(
        #     tag=f"resize_kv_{old_length}_to_{new_length}_done",
        #     operation="resize_kv_cache",
        #     before_idx=before_idx,
        #     expected_delta_gb=kv_cache_delta_gb
        # )
        
        self.dynamic_kv_synchronizer.create_slot_mappings(new_length * self.block_size)
        logger.info(f"[timeline]: total resize kv cache time taken: {human_readable_duration(time.time() - start_time)}")

    def _resize_kv_cache(self, new_length: int) -> None:
        runner = self.model_runner
        assert new_length > 0
        with runner.forward_lock:
            time_start = time.time()
            assert isinstance(runner.model, DynamicModelBase)
            logger.info(f"resizing kv cache from {len(runner.kv_caches[0][0])} to {new_length}")
            forward_context = self.vllm_config.compilation_config.static_forward_context
            kv, kv_length, T, H, Dh = runner.kv_caches[0].shape
            logger.info(f"num of kv tensors{len(runner.kv_caches)}")

            for layer_name, attn_module in forward_context.items():
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"rresizing kv cache for laye {layer_name}")
                layer_idx = extract_layer_index(layer_name)
                idx = layer_idx - runner.model.model.start_layer
                cache = runner.kv_caches[idx]
                tmp_cache = torch.zeros((kv, new_length, T, H, Dh), device=self.device, dtype=cache.dtype)
                logger.info(f"current thread: {threading.current_thread().name} tmp_cache shape: {tmp_cache.shape}, cache shape: {cache.shape}, tmp cache device: {tmp_cache.device}")
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
        
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[timeline]: resize kv cache within: {human_readable_duration(time.time() - time_start)}")

    def _flexi_resize_kv_cache(self, new_length: int) -> None:
        assert isinstance(self.model_runner.model, DynamicModelBase)
        use_direct_ptr = self.vllm_config.dynamic_config.use_direct_ptr
        time_start = time.time()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info(f"resizing kv cache from {len(self.model_runner.key_caches[0])} to {new_length}")
        logger.info(f"before resize kv cache, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        forward_context = self.vllm_config.compilation_config.static_forward_context
        T, H, Dh = self.model_runner.page_meta.shape
        cache_length = len(self.model_runner.key_caches[0])

        start_layer, end_layer = self.model_runner.model.model.start_layer, self.model_runner.model.model.end_layer
        
        # 🔴 FIX: Sort forward_context by layer index to ensure consistent ordering
        # After async migration, forward_context may have layers out of numerical order
        # because deleted layers are removed and new layers are appended to the end.
        # This causes mismatch between temp list index (iteration order) and local_idx
        # (numerical layer_index - start_layer), leading to wrong KV cache bindings.
        sorted_forward_context = sorted(forward_context.items(), key=lambda x: extract_layer_index(x[0]))
        
        tmp_key_cache_list = []
        tmp_value_cache_list = []
        tmp_key_cache_ptr_list = []
        tmp_value_cache_ptr_list = []
        tmp_new_key_ptr_tensors: list[Optional[torch.Tensor]] = []
        tmp_new_value_ptr_tensors: list[Optional[torch.Tensor]] = []

        tmp_old_key_cache_list = []
        tmp_old_value_cache_list = []
        tmp_old_key_cache_ptr_list = []
        tmp_old_value_cache_ptr_list = []
        # VMM handle tracking for shrinking
        tmp_old_key_handles_list: list[list[int]] = []
        tmp_old_value_handles_list: list[list[int]] = []
        tmp_new_key_handles_list: list[list[int]] = []
        tmp_new_value_handles_list: list[list[int]] = []
        if new_length < cache_length:
            # Check if we should use VMM (block_size >= 2MB)
            # vmm_aligned_bytes > 0 means initialization used VMM
            use_vmm = self.model_runner.vmm_aligned_bytes > 0
            
            for layer_name, _ in sorted_forward_context:
                logger.info(f"resizing kv cache for layer {layer_name}")
                idx = extract_layer_index(layer_name)
                local_idx = idx - start_layer

                # Track old GPU pointer arrays for freeing later
                old_k_ptrs, old_v_ptrs = self.model_runner.key_cache_ptrs[local_idx], self.model_runner.value_cache_ptrs[local_idx]
                old_key_cache = self.model_runner.key_caches[local_idx][new_length:]
                old_value_cache = self.model_runner.value_caches[local_idx][new_length:]
                
                # Track VMM handles only when using VMM allocation
                # Check both existence AND length to avoid IndexError after layer reconfiguration
                handles_len = len(self.model_runner.key_handles[local_idx]) if (self.model_runner.key_handles and local_idx < len(self.model_runner.key_handles)) else 0
                cache_len = len(self.model_runner.key_caches[local_idx])
                
                if use_vmm and handles_len > 0:
                    # VMM mode: only track handles that actually exist
                    # handles may be shorter than cache if some blocks were received via migration (non-VMM)
                    if handles_len < cache_len:
                        logger.warning(f"[VMM shrink] layer {layer_name}: handles_len={handles_len} < cache_len={cache_len}, some blocks not VMM allocated")
                    
                    # Calculate how many handles we can actually free
                    # Only blocks allocated with VMM have handles
                    if handles_len > new_length:
                        old_key_handles = self.model_runner.key_handles[local_idx][new_length:handles_len]
                        old_value_handles = self.model_runner.value_handles[local_idx][new_length:handles_len] if self.model_runner.value_handles else []
                        new_key_handles = self.model_runner.key_handles[local_idx][:min(new_length, handles_len)]
                        new_value_handles = self.model_runner.value_handles[local_idx][:min(new_length, handles_len)] if self.model_runner.value_handles else []
                        
                        # Adjust old_key_cache to match old_key_handles length for VMM freeing
                        # The remaining blocks without handles will be freed by GC
                        vmm_free_count = len(old_key_handles)
                        old_key_cache_for_vmm = self.model_runner.key_caches[local_idx][new_length:new_length + vmm_free_count]
                        old_value_cache_for_vmm = self.model_runner.value_caches[local_idx][new_length:new_length + vmm_free_count]
                    else:
                        # All handles are kept, nothing to free with VMM
                        old_key_handles = []
                        old_value_handles = []
                        new_key_handles = self.model_runner.key_handles[local_idx][:handles_len]
                        new_value_handles = self.model_runner.value_handles[local_idx][:handles_len] if self.model_runner.value_handles else []
                        old_key_cache_for_vmm = []
                        old_value_cache_for_vmm = []
                else:
                    # cudaMallocAsync mode or no handles: no VMM freeing needed
                    old_key_handles = []
                    old_value_handles = []
                    new_key_handles = []
                    new_value_handles = []
                    old_key_cache_for_vmm = []
                    old_value_cache_for_vmm = []
                
                tmp_old_key_cache_ptr_list.append(old_k_ptrs)
                tmp_old_value_cache_ptr_list.append(old_v_ptrs)
                tmp_old_key_cache_list.append(old_key_cache)
                tmp_old_value_cache_list.append(old_value_cache)
                # Store VMM-specific cache/handles pairs for proper freeing
                tmp_old_key_handles_list.append((old_key_cache_for_vmm, old_key_handles))
                tmp_old_value_handles_list.append((old_value_cache_for_vmm, old_value_handles))
                tmp_new_key_handles_list.append(new_key_handles)
                tmp_new_value_handles_list.append(new_value_handles)

                new_key_cache = self.model_runner.key_caches[local_idx][:new_length]
                new_value_cache = self.model_runner.value_caches[local_idx][:new_length]
                with self.model_runner.fbgate.background():
                    # Both flexi and direct need GPU pointer arrays for the kernel
                    new_key_ptrs, new_value_ptrs = kv_allocator.prepare_flexi_kv_ptrs(new_key_cache, new_value_cache)
                    logger.info(f"resizing for kv cache, new_key_ptrs: {hex(new_key_ptrs)}, new_value_ptrs: {hex(new_value_ptrs)}")
                    # PtrTensors only needed by direct kernel
                    if use_direct_ptr:
                        new_key_ptr_tensor = create_ptr_tensor_from_list(new_key_cache, self.model_runner.device)
                        new_value_ptr_tensor = create_ptr_tensor_from_list(new_value_cache, self.model_runner.device)
                    else:
                        new_key_ptr_tensor = None
                        new_value_ptr_tensor = None
                tmp_key_cache_list.append(new_key_cache)
                tmp_value_cache_list.append(new_value_cache)
                tmp_key_cache_ptr_list.append(new_key_ptrs)
                tmp_value_cache_ptr_list.append(new_value_ptrs)
                tmp_new_key_ptr_tensors.append(new_key_ptr_tensor)
                tmp_new_value_ptr_tensors.append(new_value_ptr_tensor)
            torch.cuda.synchronize()
            time_before_in_lock = time.time()
            with self.model_runner.forward_lock:
                start_time = time.time()
                for layer_name, _ in sorted_forward_context:
                    logger.info(f"resizing kv cache for layer {layer_name}")
                    idx = extract_layer_index(layer_name)
                    local_idx = idx - start_layer

                    key_cache = tmp_key_cache_list[local_idx]
                    value_cache = tmp_value_cache_list[local_idx]
                    new_k_ptrs, new_v_ptrs = tmp_key_cache_ptr_list[local_idx], tmp_value_cache_ptr_list[local_idx]
                    new_key_ptrs_tensor = tmp_new_key_ptr_tensors[local_idx]
                    new_value_ptrs_tensor = tmp_new_value_ptr_tensors[local_idx]

                    before_bind_time = time.time()
                    dynamic_flexi_bind_single_kv_cache(start_layer, end_layer,idx, key_cache, value_cache, new_k_ptrs, new_v_ptrs,  new_key_ptrs_tensor, new_value_ptrs_tensor, forward_context, self.dynamic_kv_synchronizer, self.model_runner)

                    logger.info(f"[timeline]: bind single kv cache for layer {layer_name} take {human_readable_duration(time.time() - before_bind_time)}")
                    # Update VMM handles in model_runner
                    if self.model_runner.key_handles:
                        self.model_runner.key_handles[local_idx] = tmp_new_key_handles_list[local_idx]
                        self.model_runner.value_handles[local_idx] = tmp_new_value_handles_list[local_idx]
                if use_direct_ptr:
                    self.model_runner.update_kv_ptr_tensor(tmp_new_key_ptr_tensors, tmp_new_value_ptr_tensors)
            time_after_in_lock = time.time()
            with self.resizing_done_cv:
                self.resizing_done = True
                self.resizing_done_cv.notify_all()

            assert self.device is not None
            # Free old GPU pointer arrays (needed by both flexi and direct)
            # memory_snapshot(f"rank{self.rank}_before_free_old_ptrs_and_caches_shrink", self.device)
            for old_key_ptr, old_value_ptr in zip(tmp_old_key_cache_ptr_list, tmp_old_value_cache_ptr_list):
                with self.model_runner.fbgate.background():
                    kv_allocator.free_page_list(old_key_ptr, self.device)
                    kv_allocator.free_page_list(old_value_ptr, self.device)

            # Free old key and value cache based on allocation mode
            if use_vmm:
                # Use VMM freeing for fine-grained 2MB release
                aligned_bytes = self.model_runner.vmm_aligned_bytes
                combined_mode = getattr(self.model_runner, 'vmm_combined_mode', False)
                vmm_freed_layers = 0
                for (old_key_cache_vmm, old_key_handles), (old_value_cache_vmm, old_value_handles) in zip(
                    tmp_old_key_handles_list, tmp_old_value_handles_list
                ):
                    with self.model_runner.fbgate.background():
                        if combined_mode:
                            # Combined mode: key_handles contains kv_handles
                            # K and V share same physical page, only need to free once
                            if old_key_handles and old_key_cache_vmm:
                                kv_allocator.free_vmm_blocks_combined(old_key_cache_vmm, old_key_handles, aligned_bytes, self.device)
                                vmm_freed_layers += 1
                        else:
                            # Separate mode: free K and V independently
                            if old_key_handles and old_key_cache_vmm:
                                kv_allocator.free_vmm_blocks(old_key_cache_vmm, old_key_handles, aligned_bytes, self.device)
                                vmm_freed_layers += 1
                            if old_value_handles and old_value_cache_vmm:
                                kv_allocator.free_vmm_blocks(old_value_cache_vmm, old_value_handles, aligned_bytes, self.device)
                logger.info(f"[VMM] Freed {vmm_freed_layers} layers using VMM API with aligned_bytes={aligned_bytes}, combined_mode={combined_mode}")
            else:
                # Use cudaFreeAsync for smaller blocks
                for old_key_cache, old_value_cache in zip(tmp_old_key_cache_list, tmp_old_value_cache_list):
                    with self.model_runner.fbgate.background():
                        kv_allocator.free_cache(old_key_cache, self.device)
                        kv_allocator.free_cache(old_value_cache, self.device)
                logger.info(f"[cudaFreeAsync] Freed {len(tmp_old_key_cache_list)} layers using cudaFreeAsync")
            # memory_snapshot(f"rank{self.rank}_after_free_old_ptrs_and_caches_shrink", self.device)
            time_after_free = time.time()

        elif new_length > cache_length:
            extended_kv_cache_shape = (T, H, Dh)
            tmp_new_allocated_key_cache = []
            tmp_new_allocated_value_cache = []
            tmp_new_key_ptr_tensors: list[Optional[torch.Tensor]] = []
            tmp_new_value_ptr_tensors: list[Optional[torch.Tensor]] = []
            # VMM handle tracking for new allocations
            tmp_new_key_handles: list[list[int]] = []
            tmp_new_value_handles: list[list[int]] = []

            tmp_old_key_cache_ptr_list = []
            tmp_old_value_cache_ptr_list = []
            
            # Check if we should use VMM (block_size >= 2MB)
            # vmm_aligned_bytes > 0 means initialization used VMM
            use_vmm = self.model_runner.vmm_aligned_bytes > 0

            # torch.cuda.empty_cache()
            for layer_name, attn_module in sorted_forward_context:
                local_idx = extract_layer_index(layer_name) - self.model_runner.model.model.start_layer
                logger.info(f"[resize_kv_cache debug] layer_name={layer_name}, local_idx={local_idx}, start_layer={self.model_runner.model.model.start_layer}, key_caches_len={len(self.model_runner.key_caches)}")

                new_allocated_block_num = new_length - cache_length


                key_cache = self.model_runner.key_caches[local_idx]
                value_cache = self.model_runner.value_caches[local_idx]
                # Get existing handles (may be empty for non-VMM allocations or after migration)
                # Check both existence AND length to avoid IndexError after layer reconfiguration
                old_key_handles = self.model_runner.key_handles[local_idx] if (self.model_runner.key_handles and local_idx < len(self.model_runner.key_handles)) else []
                old_value_handles = self.model_runner.value_handles[local_idx] if (self.model_runner.value_handles and local_idx < len(self.model_runner.value_handles)) else []

                if use_vmm:
                    # Use VMM API for fine-grained 2MB release support
                    combined_mode = getattr(self.model_runner, 'vmm_combined_mode', False)
                    if combined_mode:
                        # Combined mode: K and V share same 2MB physical page
                        vmm_result = kv_allocator.allocate_with_cuda_vmm_combined(
                            new_allocated_block_num, 
                            list(extended_kv_cache_shape), 
                            self.model_runner.kv_cache_dtype, 
                            self.model_runner.device
                        )
                        new_allocated_key_cache = vmm_result[0]  # k_ptrs
                        new_allocated_value_cache = vmm_result[1]  # v_ptrs
                        new_allocated_key_handles = vmm_result[6]  # kv_handles (combined)
                        new_allocated_value_handles = []  # Empty for combined mode
                        aligned_bytes = vmm_result[4]  # aligned_combined_bytes
                        self.model_runner.vmm_aligned_bytes = aligned_bytes
                    else:
                        # Separate mode: K and V each have their own 2MB pages
                        vmm_result = kv_allocator.allocate_with_cuda_vmm(
                            new_allocated_block_num, 
                            list(extended_kv_cache_shape), 
                            self.model_runner.kv_cache_dtype, 
                            self.model_runner.device
                        )
                        new_allocated_key_cache = vmm_result[0]  # k_ptrs
                        new_allocated_value_cache = vmm_result[1]  # v_ptrs
                        new_allocated_key_handles = vmm_result[5]  # k_handles
                        new_allocated_value_handles = vmm_result[6]  # v_handles
                        aligned_bytes = vmm_result[4]  # aligned_bytes
                        self.model_runner.vmm_aligned_bytes = aligned_bytes
                else:
                    # Use cudaMallocAsync for smaller blocks
                    new_allocated_key_cache, new_allocated_value_cache, _, _, _ = kv_allocator.allocate_with_cuda_async(
                        new_allocated_block_num, 
                        list(extended_kv_cache_shape), 
                        self.model_runner.kv_cache_dtype, 
                        self.model_runner.device
                    )
                    new_allocated_key_handles = []
                    new_allocated_value_handles = []
                
                new_key_cache = key_cache + new_allocated_key_cache
                new_value_cache = value_cache + new_allocated_value_cache
                # Combine old and new handles
                new_key_handles = old_key_handles + list(new_allocated_key_handles)
                new_value_handles = old_value_handles + list(new_allocated_value_handles)
                
                logger.info(f"[resize_kv_cache] use_vmm={use_vmm}, new_key_cache len={len(new_key_cache)}, handles len={len(new_key_handles)}")
                with self.model_runner.fbgate.background():
                    # Both flexi and direct need GPU pointer arrays for the kernel
                    new_k_ptrs, new_v_ptrs = kv_allocator.prepare_flexi_kv_ptrs(new_key_cache, new_value_cache)
                    # PtrTensors only needed by direct kernel
                    if use_direct_ptr:
                        new_key_ptr_tensor = create_ptr_tensor_from_list(new_key_cache, self.model_runner.device)
                        new_value_ptr_tensor = create_ptr_tensor_from_list(new_value_cache, self.model_runner.device)
                    else:
                        new_key_ptr_tensor = None
                        new_value_ptr_tensor = None

                tmp_new_allocated_key_cache.append(new_key_cache)
                tmp_new_allocated_value_cache.append(new_value_cache)
                tmp_new_key_ptr_tensors.append(new_key_ptr_tensor)
                tmp_new_value_ptr_tensors.append(new_value_ptr_tensor)
                tmp_key_cache_ptr_list.append(new_k_ptrs)
                tmp_value_cache_ptr_list.append(new_v_ptrs)
                tmp_new_key_handles.append(new_key_handles)
                tmp_new_value_handles.append(new_value_handles)
                
                # Track old ptrs for freeing later
                old_k_ptrs, old_v_ptrs = self.model_runner.key_cache_ptrs[local_idx], self.model_runner.value_cache_ptrs[local_idx]
                tmp_old_key_cache_ptr_list.append(old_k_ptrs)
                tmp_old_value_cache_ptr_list.append(old_v_ptrs)
            time_before_in_lock = time.time()
            torch.cuda.synchronize()
            with self.model_runner.forward_lock:
                start_time = time.time()
                new_key_ptr_tensor_list = []
                new_value_ptr_tensor_list = []
                for layer_name, attn_module in sorted_forward_context:
                    idx = extract_layer_index(layer_name)
                    local_idx = idx - self.model_runner.model.model.start_layer
                    new_key_cache = tmp_new_allocated_key_cache[local_idx]
                    new_value_cache = tmp_new_allocated_value_cache[local_idx]
                    new_k_ptrs =  tmp_key_cache_ptr_list[local_idx]
                    new_v_ptrs =  tmp_value_cache_ptr_list[local_idx]
                    new_k_ptr_tensor = tmp_new_key_ptr_tensors[local_idx]
                    new_v_ptr_tensor = tmp_new_value_ptr_tensors[local_idx]
                    dynamic_flexi_bind_single_kv_cache(start_layer, end_layer, idx, new_key_cache, new_value_cache, new_k_ptrs, new_v_ptrs, new_k_ptr_tensor, new_v_ptr_tensor, forward_context, self.dynamic_kv_synchronizer, self.model_runner)
                    new_key_ptr_tensor_list.append(new_k_ptr_tensor)
                    new_value_ptr_tensor_list.append(new_v_ptr_tensor)
                    # Update VMM handles in model_runner
                    if len(self.model_runner.key_handles) <= local_idx:
                        self.model_runner.key_handles.extend([[] for _ in range(local_idx + 1 - len(self.model_runner.key_handles))])
                        self.model_runner.value_handles.extend([[] for _ in range(local_idx + 1 - len(self.model_runner.value_handles))])
                    self.model_runner.key_handles[local_idx] = tmp_new_key_handles[local_idx]
                    self.model_runner.value_handles[local_idx] = tmp_new_value_handles[local_idx]
                # Update stacked tensors cache to reflect new ptr_tensors (direct mode only)
                if use_direct_ptr:
                    self.model_runner.update_kv_ptr_tensor(new_key_ptr_tensor_list, new_value_ptr_tensor_list)
            time_after_in_lock = time.time() 
            with self.resizing_done_cv:
                self.resizing_done = True
                self.resizing_done_cv.notify_all()

            # Free old GPU pointer arrays (needed by both flexi and direct)
            for old_key_ptr, old_value_ptr in zip(tmp_old_key_cache_ptr_list, tmp_old_value_cache_ptr_list):
                with self.model_runner.fbgate.background():
                    kv_allocator.free_page_list(old_key_ptr, self.model_runner.device)
                    kv_allocator.free_page_list(old_value_ptr, self.model_runner.device)
            time_after_free = time.time()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # memory_snapshot(f"rank{self.rank}_after_resize_to_{new_length}", self.device)
        logger.info(f"[forward]: resized kv cache ,available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        logger.info(f"[timeline]: time before in lock: {time_after_in_lock - time_before_in_lock:.4f} seconds")
        logger.info(f"[timeline]: time after free: {time_after_free - time_after_in_lock:.4f} seconds")
        logger.info(f"[timeline]: total resize kv cache time taken: {human_readable_duration(time.time() - time_start)}")

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

    def start_kv_cache_migration_async(self, pp_layer_config: list[Tuple[int, int]], src_to_plan: dict[int, dict[int, list[int]]], slot_mapping: Optional[list[int]] = None) -> None:
        """Collective-RPC entry used by executor.

        Only the worker whose `self.rank == source_rank` performs the actual
        send; other ranks are no-ops. The receiver side should already be
        listening via `listen_to_kv_cache_tensor(source_rank)`.
        """
        # assert self.migration_in_process == False, "The migration should not be in process"
        logger.info(f"[operation]: Start KV Cache Migration: {src_to_plan}")
        logger.info(f"[operation]: rank {self.rank} start sending KV Cache Migration")
        num_layers = pp_layer_config[self.rank][1] - pp_layer_config[self.rank][0] + 1
        self.target_pp_layer_config = pp_layer_config
        # Only prepare ptr_tables in direct mode (flash/flexi don't use ptr_tables)
        if self.vllm_config.dynamic_config.use_direct_ptr:
            self.model_runner.prepare_ptr_tables(num_layers)
        assert len(self.rank_to_layers_ids) == 0
        if self.rank not in src_to_plan:
            logger.info(f"debug: rank {self.rank} is not sending kv cache")
            return None
        self.rank_to_layers_ids = src_to_plan[self.rank]
        self.kv_resizing_done = False
        def migration_thread(rank: int, layer_ids: list[int]):
            # Set CUDA device for this thread - threads don't inherit CUDA context
            torch.cuda.set_device(self.device)
            assert isinstance(self.model_runner.model, DynamicModelBase)
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
        assert self.migration_stream is not None
        with self.migration_stream:
            for layer_id in layer_ids:
                time_start = time.time()
                kv_tensor_meta, kv_tensor_data = self.dynamic_kv_synchronizer.get_kv_tensor_from_cache(layer_ids, layer_id, start_layer_id, self.model_runner.page_meta,  slot_mapping_dev)
                logger.info(f"start to send kv tensor for layer {layer_id} to rank {rank}, kv_tensor_meta: {kv_tensor_meta}, kv_tensor_data shape: {kv_tensor_data.shape}, time taken to get kv tensor: {human_readable_duration(time.time() - time_start)} seconds")
                with self.model_runner.fbgate.background():
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
            time_start_to_wait = time.time()
            logger.info(f"[timeline]: wait for kv patch")
            self.dynamic_kv_synchronizer.wait_for_kv_patch(rank)
            logger.info(f"[timeline]: wait for kv patch preparation take {human_readable_duration(time.time() - time_start_to_wait)}")
            for kv_patch in self.dynamic_kv_synchronizer.get_kv_patch(rank, start_layer_id, layer_ids, self.model_runner.page_meta):
                time_start = time.time()
                patch_payload_size = kv_patch.kv_payload.numel() * kv_patch.kv_payload.element_size()
                patch_slot_mapping_size = kv_patch.slot_mapping.numel() * kv_patch.slot_mapping.element_size()
                with self.model_runner.fbgate.foreground():
                    self.dynamic_kv_synchronizer.send_kv_patch_to_rank(rank, kv_patch)
                logger.info(f"[timeline]: send kv patch to rank {rank}, time taken: {time.time() - time_start}, data_size: {patch_slot_mapping_size / 1024 ** 2:.2f}MB + {patch_payload_size / 1024 ** 2:.2f}MB, kv patch id: {kv_patch.meta.id}, kv patch type: {kv_patch.meta.type}")
            logger.info(f"[debug]: finished sending kv patch for layer_ids {layer_ids} to rank {rank}")


    def start_kv_cache_migration_sync(self, pp_layer_config: list[Tuple[int, int]], src_to_sending_layers: dict[int, dict[int, list[Tuple[int, int]]]], 
                                       rank_to_layer_ids: dict[int, list[Tuple[int, int]]],
                                       slot_mapping: Optional[list[int]] = None) -> None:
        """Collective-RPC entry used by executor.

        Only the worker whose `self.rank == source_rank` performs the actual
        send; other ranks are no-ops. The receiver side should already be
        listening via `listen_to_kv_cache_tensor(source_rank)`.
        
        Args:
            src_to_sending_layers: Mapping of source rank to {dest_rank: layer_ranges}
            rank_to_layer_ids: Mapping of destination rank to layer ranges to receive
            slot_mapping: Optional list of slot indices for flexi mode. If None, no data is sent.
        """

        time_start = time.time()
        
        # Reset checkpoint tracker for this migration to track memory changes
        old_config = (self.model_runner.model.model.start_layer, self.model_runner.model.model.end_layer)
        new_config = pp_layer_config[self.rank] if self.rank < len(pp_layer_config) else old_config
        tracker = reset_checkpoint_tracker(self.rank, self.device)
        logger.info(f"[MEM_CHECKPOINT] Starting sync migration: rank{self.rank} config {old_config} -> {new_config}")
        
        with self._receive_finished_cv:
            assert self.receive_in_process == False, "The receiving kv cache should not be in process"

        if self.rank in rank_to_layer_ids:
            logger.info(f"debug: rank {self.rank} is receiving kv cache")
            with self._receive_finished_cv:
                self.receive_in_process = True

        assert isinstance(self.model_runner.model, DynamicModelBase)
        memory_snapshot(f"rank{self.rank}_sync_migration_entry", self.device)

        # Sender Side, Send KV Cache
        logger.info(f"Enter sync migration process")
        if self.rank in src_to_sending_layers:
            sending_layers_plans = src_to_sending_layers[self.rank]
            logger.info(f"Enter Sender Side in sync migration process, sending_layer_plans:{sending_layers_plans}")
            sending_layers_list: dict[int, list[int]] = {}
            for rank, layer_ranges in sending_layers_plans.items():
                for layer_range in layer_ranges:
                    layer_ids = sending_layers_list.setdefault(rank, [])
                    layer_ids.extend(list(range(layer_range[0], layer_range[1] + 1)))

            # Convert slot_mapping to tensor if provided
            slot_mapping_tensor: Optional[torch.Tensor] = None
            if slot_mapping is not None:
                slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int64, device=self.device)

            with self.model_runner.forward_lock:
                self.dynamic_kv_synchronizer.send_kv_tensor_sync(
                    sending_layers_list, 
                    self.model_runner.kv_caches, 
                    self.model_runner.model.model.start_layer,
                    slot_mapping_tensor)
            all_layer_ranges = []
            for _, ranges in sending_layers_plans.items():
                all_layer_ranges.extend(ranges)
            caches_to_free_key, caches_to_free_value, ptrs_to_free_key, ptrs_to_free_value, handles_to_free_key, handles_to_free_value = self.atomic_shelve_kv_cache(self.rank, all_layer_ranges)
            # memory_snapshot(f"rank{self.rank}_after_atomic_shelve", self.device)
            self.release_kv_cache_for_layers(self.rank, caches_to_free_key, caches_to_free_value, ptrs_to_free_key, ptrs_to_free_value, handles_to_free_key, handles_to_free_value)
            # memory_snapshot(f"rank{self.rank}_after_release_kv_cache", self.device)
            self.remove_layers(self.rank, all_layer_ranges)
            # memory_snapshot(f"rank{self.rank}_after_remove_layers", self.device)
            
            # Check GPU memory after removing layers (sender side of sync migration)
            if isinstance(self.model_runner.model, DynamicModelBase):
                new_start_layer = self.model_runner.model.model.start_layer
                new_end_layer = self.model_runner.model.model.end_layer
                memory_monitor = get_gpu_memory_monitor()
                memory_monitor.check_memory_after_migration_local(
                    new_start_layer, new_end_layer, self.rank, self.device,
                    migration_type="sync_sender"
                )
                
        start_layer = pp_layer_config[self.rank][0]
        # Receiver Side, Wait for receive to finish
        if self.rank not in src_to_sending_layers:
            # Wait for receive to finish if this rank is a receiver
            if self.rank in rank_to_layer_ids:
                with self._receive_finished_cv:
                    while self.receive_in_process:
                        logger.info(f"[sync migration]: rank {self.rank} waiting for receive to finish")
                        self._receive_finished_cv.wait()
            logger.info(f"[timeline]: after start kv cache migration sync, time taken: {human_readable_duration(time.time() - time_start)}")
        is_direct = self.vllm_config.dynamic_config.use_direct_ptr  
        if is_direct:
            memory_snapshot(f"rank{self.rank}_before_commit_ptr_tables", self.device)
            self.model_runner.commit_ptr_tables(self.model_runner.k_ptr_tensors, self.model_runner.v_ptr_tensors, target_start_layer=start_layer)
            memory_snapshot(f"rank{self.rank}_after_commit_ptr_tables", self.device)
        
        # Print checkpoint summary for this migration
        tracker = get_checkpoint_tracker(self.rank, self.device)
        logger.info(f"\n{tracker.get_summary()}")
        
        logger.info(f"finish sync migration in {time.time() - time_start:.2f}")
        return None

    # ####################################### #
    # Receiver side of KV migration functions #
    # ####################################### # 
    def listen_to_kv_cache_tensor_and_patches(self, from_rank: int) -> None:
        """
        异步监听指定 from_rank 的 KV：先接收完整 KV 张量，再持续接收小 patch，并按需同步。
        """
        assert isinstance(self.model_runner.model, DynamicModelBase)
        assert self.rank != from_rank, "The rank should not listen to its own kv cache"

        logger.info(f"Worker {self.rank} listening KV stream from rank {from_rank}")

        threading.Thread(target=self._listen_loop, args=(from_rank,), daemon=True).start()
        return None

    def _listen_loop(self, from_rank: int):
        # IMPORTANT: Set CUDA device for this thread - threads don't inherit CUDA context
        torch.cuda.set_device(self.device)
        
        assert isinstance(self.model_runner.model, DynamicModelBase)
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        time_start = None
        try:
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

                # NOTE: Use foreground() instead of background() here.
                # background() waits for ALL foreground tasks to finish, which
                # deadlocks with compiled DAG (which uses foreground() for its
                # NCCL recv and execute_model). KV migration NCCL uses a separate
                # communicator and writes to newly allocated buffers, so it is
                # safe to run concurrently with compiled DAG foreground tasks.
                with self.model_runner.fbgate.foreground():
                    if is_flexi:
                        assert isinstance(meta, FlexiKVTensorMeta)
                        slot_mapping, kv_tensor = self.dynamic_kv_synchronizer.recv_kv_tensor(from_rank, meta)
                        tmp_slot_mapping_dict[meta.layer_id] = slot_mapping 
                    else:
                        assert isinstance(meta, KVTensorMeta)
                        kv_tensor = self.dynamic_kv_synchronizer.recv_kv_tensor(from_rank, meta)
                        assert isinstance(kv_tensor, torch.Tensor)
                tmp_kv_tensors_dict[meta.layer_id] = kv_tensor

                # Check if all layers have been received - break early to avoid deadlock
                # (Sender is waiting for our notify_for_kv_patch before sending kv_patch_meta)
                if layers_to_be_received == received_layer:
                    logger.info(f"[debug]: all layers received, breaking out of loop early to avoid deadlock")
                    meta = None  # Set meta to None to indicate we broke early
                    break

                # logger.info(f"available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
            assert time_start is not None
            logger.info(f"[timeline]: receive kv tensor finished, time taken {human_readable_duration(time.time() - time_start)} start to bind kv cache")
            # 在kv cache绑定前，weight必须loading结束
            layer_ids = list(tmp_kv_tensors_dict.keys())
            for layer_id in layer_ids:
                with self._layer_loaded_cv:
                    logger.info(f"inside the layer_loaded_cv")
                    while not self.model_runner.has_layer(layer_id):
                        logger.info(f"[debug]: rank {self.rank} waiting for layer {layer_id} to be loaded")
                        self._layer_loaded_cv.wait()
            time_start_bind_kv_cache = time.time()
            
            # Pre-allocate KV cache memory for all layers before binding
            preallocated_caches: dict[int, tuple] = {}
            if is_flexi:
                logger.info(f"[timeline]: pre-allocating KV cache for {len(layer_ids)} layers")
                preallocated_caches = self.model_runner.preallocate_flexi_kv_caches_for_migration(
                    layer_ids, self.block_num
                )
                logger.info(f"[timeline]: pre-allocation done in {human_readable_duration(time.time() - time_start_bind_kv_cache)}")
            
            for layer_id in layer_ids:
                logger.info(f"current memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB, start to bind kv cache for layer {layer_id}")
                kv_tensor = tmp_kv_tensors_dict[layer_id]
                if is_flexi:
                    slot_mapping = tmp_slot_mapping_dict[layer_id]
                    assert self.device is not None
                    # Use pre-allocated memory
                    preallocated = preallocated_caches.get(layer_id)
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
                        preallocated=preallocated,
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
                tmp_kv_tensors_dict.pop(layer_id)
                gc.collect()
                torch.cuda.empty_cache()  # Free GPU memory occupied by received kv tensor before binding, to make room for new cache if needed
            torch.cuda.synchronize()
            # memory_snapshot(f"rank{self.rank}_after_bind_all_kv_caches_from_rank{from_rank}", self.device)
            logger.info(f"[timeline]: bind kv cache time taken: {human_readable_duration(time.time() - time_start_bind_kv_cache)}")
            
            logger.info(f"[operation]: start to listen to kv cache patches")
            self.dynamic_kv_synchronizer.notify_for_kv_patch(from_rank)
            logger.info(f"[timeline]: notify for kv patches took {human_readable_duration(time.time() - time_start_bind_kv_cache)}")
            meta = self.dynamic_kv_synchronizer.recv_controller(from_rank)
            # Check if this is sync migration - if so, skip kv_patch receiving
            if meta.type == "sync_finished":
                logger.info(f"[sync migration]: Worker {self.rank} received sync_finished from rank {from_rank}, skipping kv_patch phase")
                # Reset async migration state after sync migration completes
                # This prevents the next async migration from waiting on stale patch-applied states
                with self._all_patch_applied_cv:
                    # Reset is_all_patch_applied to False for next migration
                    # (not True, because next migration needs to wait for patches)
                    self.is_all_patch_applied[from_rank] = True
                    self._all_patch_applied_cv.notify_all()
                # Notify waiting threads that sync migration is complete
                # Notify that receive is finished for sync migration
                with self._receive_finished_cv:
                    self.receive_in_process = False
                    self._receive_finished_cv.notify_all()
                
                # Check GPU memory after sync migration to detect memory leaks
                if isinstance(self.model_runner.model, DynamicModelBase):
                    start_layer = self.model_runner.model.model.start_layer
                    end_layer = self.model_runner.model.model.end_layer
                    memory_monitor = get_gpu_memory_monitor()
                    memory_monitor.check_memory_after_migration_local(
                        start_layer, end_layer, self.rank, self.device,
                        migration_type="sync"
                    )
                
                logger.info(f"[timeline]: sync migration complete, time taken: {human_readable_duration(time.time() - time_start)}")
                continue  # Go back to outer loop, ready for next migration
            self.receiver_num_applied_token_dict[from_rank] = tmp_slot_token_num
            # 在同步迁移中，断言：所有接收层的 KV 已完成绑定且形状一致
            self._assert_layers_kv_bound(layers_to_be_received)


            # logger.info(f"[debug]: after receive kv tensor, kv cache:")
            # if is_flexi:
            #     for i in range(len(self.model_runner.key_caches)):
            #         logger.info(f"[debug]: key_caches[{i}] length: {len(self.model_runner.key_caches[i])}")
            # else:
            #     for i in range(len(self.model_runner.kv_caches)):
            #         logger.info(f"[debug]: kv_caches[{i}] shape: {self.model_runner.kv_caches[i].shape}")
            cur_patch_id = 0
            # logger.info(f"[timeline]: after receive kv tensor, time taken: {human_readable_duration(time.time() - time_start)}")
            time_before_start_receive_patches = time.time()
            # 2) Then wait for all patch to be received
            while True:
                assert meta.type == "kv_patch_meta" or meta.type == "kv_patch_finished", "The type of the meta should be kv_patch_meta or kv_patch_finished"

                # Ensure the order of the patch
                assert meta.id == cur_patch_id, f"The patch id should be the next id of the last patch, meta id: {meta.id} vs cur patch id{cur_patch_id}"
                cur_patch_id += 1
                slot_mapping, kv_payload = self.dynamic_kv_synchronizer.recv_kv_patch(from_rank, meta)
                logger.info(f"[listen loop]: Received Meta: {meta}")
                logger.info(f"[listen loop]: slot mapping device: {slot_mapping.device}, kv payload device: {kv_payload.device}, kv payload shape: {kv_payload.shape}, slot mapping shape: {slot_mapping.shape}")
                
                # Only apply patch if there's actual data (skip for empty kv_patch_finished)
                if kv_payload.numel() > 0:
                    self.dynamic_kv_synchronizer.apply_one_patch_to_kv_cache(self.model_runner.model.model.start_layer, meta, kv_payload, slot_mapping, self.model_runner.page_meta)
                    self.receiver_num_applied_token_dict[from_rank] += meta.num_tokens
                    logger.info(f"[num tokens]: receiver side: rank {from_rank} applied token: {meta.num_tokens}, total applied token: {self.receiver_num_applied_token_dict[from_rank]}")
                else:
                    logger.info(f"[listen loop]: skipping apply for empty patch (type={meta.type})")
                    
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
                    
                    # Check GPU memory after async migration to detect memory leaks (receiver side)
                    if isinstance(self.model_runner.model, DynamicModelBase):
                        start_layer = self.model_runner.model.model.start_layer
                        end_layer = self.model_runner.model.model.end_layer
                        memory_monitor = get_gpu_memory_monitor()
                        memory_monitor.check_memory_after_migration_local(
                            start_layer, end_layer, self.rank, self.device,
                            migration_type="async"
                        )
                    
                    break # Exit point from current migration listening loop
                else:
                    assert False, f"Unexpected message type: {meta.type}"
                meta = self.dynamic_kv_synchronizer.recv_controller(from_rank)
            logger.info(f"[timeline]: after listen to kv cache patches, time taken: {human_readable_duration(time.time() - time_before_start_receive_patches)}")
        except Exception as e:
            logger.exception(f"[FATAL] _listen_loop(from_rank={from_rank}) crashed: {e}")
            raise
        
    def all_patch_applied(self) -> bool:
        return all(self.is_all_patch_applied.values())

    # ########################################
    # Callback funtions executed inside the forwarding loop #
    # ######################################## 

    def wait_for_resize_done(self) -> None:
        """Wait for KV cache resize to complete. 
        MUST be called BEFORE acquiring forward_lock to avoid deadlock!
        """
        with self.resizing_done_cv:
            while not self.resizing_done:
                self.resizing_done_cv.wait()

    def sync_migration_before_execute_callback(self, new_kv_cache_block_num: int) -> None:
        with self._receive_finished_cv:
            while self.receive_in_process:
                logger.info(f"debug: ---------------------rank {self.rank} waiting for migration to be finished")
                self._receive_finished_cv.wait()
            logger.info(f"debug: ---------------------rank {self.rank} is good for forwarding")
        # if new_kv_cache_block_num != 0:
        #     self.resize_kv_cache(new_kv_cache_block_num)

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
            # with self._all_patch_applied_cv:
            #     while not self.all_patch_applied():
            #         self._all_patch_applied_cv.wait()
            # NOTE: resizing_done wait is now done BEFORE forward_lock in dynamic_utils.py
            # to avoid deadlock (do_resize thread needs forward_lock to complete)
            logger.info(f"[timeline]: after waiting for all kv cache patch to be applied, time taken: {human_readable_duration(time.time() - time_start)}")

            # assert self.after_migration_total_token == self.after_migration_applied_token_num, f"The after migration token num should be equal to the total migration token num, self.after_migration_total_token: {self.after_migration_total_token}, total_migration_token_num: {self.after_migration_applied_token_num}"
            # for rank in range(self.vllm_config.parallel_config.pipeline_parallel_size):
            #     if rank != self.rank:
            #         continue
            #     assert total_migration_token_num == self.receiver_num_applied_token_dict[rank], f"The total migration token num should be equal to the applied token num, total_migration_token_num: {total_migration_token_num}, applied token num: {self.receiver_num_applied_token_dict[rank]}"
            # Reset all async migration state after successful completion
            self.after_migration_total_token = 0
            self.after_migration_applied_token_num = 0


    def async_migration_after_execute_callback(self, scheduler_output: "DynamicSchedulerOutput"):
        if not isinstance(self.model_runner.model, DynamicModelBase):
            return

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
                logger.info(f"Sender added kv patch to synchronizer, is finished: {is_sync}, num total new tokens: {num_total_new_tokens}")

            if is_sync:
                time_start_to_sync = time.time()
                # Set CUDA device for this thread - threads don't inherit CUDA context
                torch.cuda.set_device(self.device)
                assert scheduler_output.sender_list is not None, "If is sync migration, the sender list should not be None"
                time_start = time.time()
                logger.info(f"[operation]: before finish kv cache transfer start to finish kv cache tansfer, delete layers, release kv cache, reinitialize kv cache")
                # Sender finishes transfer and cleans up; receiver no-ops.
                layer_ranges = []
                for layers in self.rank_to_layers_ids.values():
                    # Assert the layers are sorted
                    assert layers == sorted(layers), "The layers should be sorted"
                    layer_ranges.append((layers[0], layers[-1]))

                assert self.target_pp_layer_config is not None, "target_pp_layer_config must be set"
                handles_to_free_key: list[list[int]] = []
                handles_to_free_value: list[list[int]] = []
                if layer_ranges:
                    caches_to_free_key, caches_to_free_value, ptrs_to_free_key, ptrs_to_free_value, handles_to_free_key, handles_to_free_value = self.atomic_shelve_kv_cache(self.rank, layer_ranges)

                start_layer = self.target_pp_layer_config[self.rank][0]
                is_direct = self.vllm_config.dynamic_config.use_direct_ptr
                if is_direct:
                    self.model_runner.commit_ptr_tables(self.model_runner.k_ptr_tensors, self.model_runner.v_ptr_tensors, target_start_layer=start_layer)
                self.target_pp_layer_config = None

                # Set resizing_done = False BEFORE starting the thread, unconditionally
                # This ensures wait_for_resize_done() will block until do_resize completes
                with self.resizing_done_cv:
                    self.resizing_done = False

                def do_resize():
                    try:
                        # Set CUDA device for this thread - threads don't inherit CUDA context
                        torch.cuda.set_device(self.device)
                        if is_sender:
                            self.remove_layers(self.rank, layer_ranges)
                            self.release_kv_cache_for_layers(self.rank, caches_to_free_key, caches_to_free_value, ptrs_to_free_key, ptrs_to_free_value, handles_to_free_key, handles_to_free_value)
                            logger.info(f"[timeline]: after remove layers, time taken: {human_readable_duration(time.time() - time_start)}")
                        self.after_migration_total_token = num_total_migration_tokens
                        # For sender: directly set applied token count since sender doesn't receive patches
                        # For receiver: this will be overwritten by _listen_loop when all patches are applied
                        if is_sender:
                            self.after_migration_applied_token_num = num_total_migration_tokens
                        fixed_blocks = self.vllm_config.dynamic_config.fixed_num_gpu_blocks
                        if fixed_blocks <= 0:
                            self.resize_kv_cache(scheduler_output.new_kv_cache_block_num)
                        else:
                            logger.info(f"fixed_num_gpu_blocks={fixed_blocks}, skipping worker-side resize (would be {scheduler_output.new_kv_cache_block_num} blocks), sleep for 4 seconds")
                        self.finish_migration()
                        self.kv_resizing_done = True
                    finally:
                        # Always signal completion, even on error
                        with self.resizing_done_cv:
                            self.resizing_done = True
                            self.resizing_done_cv.notify_all()

                threading.Thread(target=do_resize, daemon=True).start()
                logger.info(f"[timeline]: finish sync migration kv cache transfer, time taken: {human_readable_duration(time.time() - time_start_to_sync)}")

        # else:
        #     assert not is_sync, "If not sending kv cache, it should not be sync migration"

    def finish_migration(self):
        self.rank_to_layers_ids = {}
        
        # Atomically rebuild and commit PtrTable stacked tensors after migration
        # This ensures all ptr_tensors changes are reflected in a single atomic switch
        from vllm.config import get_current_vllm_config
        vllm_config = get_current_vllm_config()
        if vllm_config.dynamic_config.use_direct_ptr:
            time_start = time.time()
            logger.info(f"[timeline]: commit ptr tables after migration take {human_readable_duration(time.time() - time_start)}")
            logger.info(f"finish_migration: committed ptr_tables for flexi_direct")

        # Check GPU memory after migration to detect memory leaks
        # Compare with recorded baseline for this configuration
        if isinstance(self.model_runner.model, DynamicModelBase):
            start_layer = self.model_runner.model.model.start_layer
            end_layer = self.model_runner.model.model.end_layer
            memory_monitor = get_gpu_memory_monitor()
            memory_monitor.check_memory_after_migration_local(
                start_layer, end_layer, self.rank, self.device, 
                migration_type="async"
            )

        # assert self.dynamic_layer_kv_connector.is_all_patch_applied(), "All patch should be applied"

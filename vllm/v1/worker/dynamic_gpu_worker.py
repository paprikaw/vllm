from collections import defaultdict
from contextlib import nullcontext
from concurrent.futures import thread
from copy import deepcopy
import gc
from hmac import new
from operator import is_
import os
from pdb import run
from sched import scheduler
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union
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
from vllm.distributed.kv_transfer.kv_connector.dynamic_utils import (
    FlexiKVTensorMeta,
    KVCachedSparseKVTensorMeta,
)
# from vllm.kv_allocator import allocate_with_cuda_async, free_cache, free_page_list, prepare_flexi_kv_ptrs
from vllm.kv_allocator import kv_allocator
from vllm.kvcached_integration import (
    materialize_kvcached_received_kv_tensor,
    materialize_kvcached_sparse_received_kv_tensor,
    maybe_apply_kvcached_vllm_patches,
    use_direct_ptr_for_runtime,
    use_flexi_kv_for_runtime,
    use_kvcached_backend,
)
from vllm.logger import init_logger
from vllm.lora import layers
from vllm.model_executor import set_random_seed
from vllm.v1.core.dynamic_kv_cache_utils import compact_cache_with_record
from vllm.v1.worker.utils import get_total_gpu_memory
from vllm.model_executor.models.dynamic_model_base import DynamicModelBase
from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheConfig, KVCacheGroupSpec
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
from vllm.distributed.parallel_state import (get_pp_group, get_tp_group,
                                             set_pp_group_active_ranks)
from vllm.distributed.kv_transfer.kv_connector.dynamic_kv_synchronizer import KVPatch, KVPatchMeta, KVTensorMeta
from bitarray import bitarray
import time
from vllm.distributed.kv_transfer.kv_connector.dynamic_kv_synchronizer import DynamicKVSynchronizer
from vllm.v1.worker.gpu_memory_monitor import (
    memory_snapshot, 
    get_memory_overhead_monitor, reset_memory_overhead_monitor
)
import signal
import sys
import traceback
import faulthandler
logger = init_logger(__name__)


def _sync_current_cuda_stream(device: Optional[torch.device]) -> None:
    if device is not None:
        torch.cuda.current_stream(device).synchronize()
    else:
        torch.cuda.current_stream().synchronize()


def _memory_snapshot_no_device_sync(
    tag: str,
    device: Optional[torch.device],
) -> dict[str, float]:
    if device is not None:
        torch.cuda.set_device(device)

    free_mem, total_mem = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    cached_not_used = reserved - allocated
    snapshot = {
        "free_gb": free_mem / 1024**3,
        "total_gb": total_mem / 1024**3,
        "allocated_gb": allocated / 1024**3,
        "reserved_gb": reserved / 1024**3,
        "cached_unused_gb": cached_not_used / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    logger.info(
        "[MEM_SNAPSHOT_STREAM_LOCAL] [%s] free=%.3fGB allocated=%.3fGB "
        "reserved=%.3fGB cached_unused=%.3fGB max_alloc=%.3fGB total=%.3fGB",
        tag,
        snapshot["free_gb"],
        snapshot["allocated_gb"],
        snapshot["reserved_gb"],
        snapshot["cached_unused_gb"],
        snapshot["max_allocated_gb"],
        snapshot["total_gb"],
    )
    return snapshot

# Import validate_layers_granularity from utils
from vllm.v1.worker.utils import validate_layers_granularity, validate_layers_count_granularity


if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput



class DynamicGPUWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 缓存等待绑定到 forward_context 的 KV tensor（按全局 layer_id）
        self._pending_kv_by_layer: dict[int, torch.Tensor] = {}
        self._autoscale_sender_key_cache_ptrs: Optional[list[int]] = None
        self._autoscale_sender_value_cache_ptrs: Optional[list[int]] = None
        self._autoscale_sender_start_layer: Optional[int] = None
        self._migration_logical_num_blocks: Optional[int] = None
        # Debug flag: when enabled, re-raise exceptions for easier debugging
        self._debug_raise: bool = str(os.getenv("VLLM_DEBUG_RAISE", "0")).lower() not in ("0", "", "false", "no")
        # Debug assertions for KV binding/shape
        self._debug_assert_kv: bool = str(os.getenv("VLLM_DEBUG_ASSERT_KV", "1")).lower() not in ("0", "", "false", "no")
        self._listen_kv_cache_threads: list[threading.Thread] = []
        self._kv_patch_recv_apply_lock = threading.Lock()
        self.rank_to_layers_ids: dict[int, list[int]] = {}
        self._pending_deleted_model_layers: list[list[object]] = []
        self._pending_deleted_model_layers_events: list[threading.Event] = []
        self._pending_deleted_model_layers_lock = threading.Lock()

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
        self._async_resize_lock = threading.Lock()
        self._async_resize_thread: Optional[threading.Thread] = None
        self._async_resize_error: Optional[BaseException] = None
        self._async_resize_target: Optional[int] = None
        self.new_kv_cache_block_num = 0 # 用于记录新配置的kv cache block数量，用于后续的kv cache resize
        logger.info(torch.__config__.show())

        # 用于记录在migration过程中，receiver已经applied的token数量
        self.receiver_num_applied_token_dict = defaultdict(int)
        # 当所有的patch都applied之后，记录总的applied token数量
        self.after_migration_applied_token_num = 0

        self.block_size = 0
        self.per_block_kv_cache_bytes = 0  # Per-layer, per-block KV cache size in bytes (K+V)

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
        
        # Async layer loading tracking
        self._async_add_layers_threads: list[threading.Thread] = []
        self._async_add_layers_lock = threading.Lock()
        
        # Sender thread completion tracking (避免 do_resize 释放 migration_thread 仍在使用的内存)
        self._sender_threads_cv = threading.Condition(threading.Lock())
        self._num_active_sender_threads = 0
        
        # Control STOP_TIME logging (disabled during set_pp_config to avoid polluting metrics)
        self.log_stop_time = True

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
    
    def _get_current_memory_params(self) -> Tuple[int, int]:
        """Get current layer count and KV cache bytes for overhead monitoring.
        
        Returns:
            Tuple of (layer_count, kv_cache_bytes)
        """
        if not isinstance(self.model_runner.model, DynamicModelBase):
            return 0, 0
        
        start_layer = self._kv_cache_start_layer()
        end_layer = self.model_runner.model.model.end_layer
        layer_count = end_layer - start_layer
        
        # Calculate current KV cache size
        kv_cache_bytes = 0
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
        if is_flexi:
            # Flexi mode: key_caches/value_caches are lists of pointer addresses, not Tensors.
            # Use block_num * layer_count * per_block_kv_cache_bytes to calculate KV cache size.
            # per_block_kv_cache_bytes = page_size_bytes = K+V size per block per layer
            if self.block_num > 0 and self.per_block_kv_cache_bytes > 0:
                kv_cache_bytes = self.block_num * layer_count * self.per_block_kv_cache_bytes
        else:
            # Non-flexi mode: single KV tensor per layer
            if hasattr(self.model_runner, 'kv_caches') and len(self.model_runner.kv_caches) > 0:
                for kv_tensor in self.model_runner.kv_caches:
                    if isinstance(kv_tensor, torch.Tensor) and kv_tensor.numel() > 0:
                        kv_cache_bytes += kv_tensor.numel() * kv_tensor.element_size()
        
        return layer_count, kv_cache_bytes
    
    def _assert_layers_kv_bound(self, layer_ids: set[int]) -> None:
        if not self._debug_assert_kv:
            return
        assert isinstance(self.model_runner, DynamicGPUModelRunner)
        start_layer = self._kv_cache_start_layer()
        fctx = self.vllm_config.compilation_config.static_forward_context
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
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

    def _kv_layer_structures_ready(self, layer_ids: set[int]) -> bool:
        if not layer_ids:
            return True
        assert isinstance(self.model_runner, DynamicGPUModelRunner)
        start_layer = self._kv_cache_start_layer()
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
        use_direct_ptr = use_direct_ptr_for_runtime(self.vllm_config)

        for layer_id in layer_ids:
            if not self.model_runner.has_layer(layer_id):
                return False
            local_index = layer_id - start_layer
            if local_index < 0:
                return False
            if is_flexi:
                if local_index >= len(self.model_runner.key_caches):
                    return False
                if local_index >= len(self.model_runner.value_caches):
                    return False
                if local_index >= len(self.dynamic_kv_synchronizer.key_cache_list):
                    return False
                if local_index >= len(self.dynamic_kv_synchronizer.value_cache_list):
                    return False
                if use_direct_ptr:
                    if local_index >= len(self.model_runner.k_ptr_tensors):
                        return False
                    if local_index >= len(self.model_runner.v_ptr_tensors):
                        return False
            else:
                if local_index >= len(self.model_runner.kv_caches):
                    return False
                if local_index >= len(self.dynamic_kv_synchronizer.kv_caches):
                    return False
        return True

    def _wait_for_kv_layer_structures_ready(
        self,
        layer_ids: set[int],
        from_rank: int,
    ) -> None:
        if not layer_ids:
            return
        wait_start = time.time()
        with self._layer_loaded_cv:
            while not self._kv_layer_structures_ready(layer_ids):
                logger.info(
                    "[autoscaling async] rank %s waiting for local layer/KV "
                    "structures before receiving KV from rank %s: layers=%s",
                    self.rank, from_rank, sorted(layer_ids))
                self._layer_loaded_cv.wait(timeout=1.0)
        waited = time.time() - wait_start
        if waited > 0:
            logger.info(
                "[autoscaling async] rank %s local layer/KV structures ready "
                "for KV from rank %s: layers=%s wait=%s",
                self.rank, from_rank, sorted(layer_ids),
                human_readable_duration(waited))

    def _clear_autoscale_sender_pointer_view(self, reason: str) -> None:
        if (self._autoscale_sender_key_cache_ptrs is not None
                or self._autoscale_sender_value_cache_ptrs is not None
                or self._autoscale_sender_start_layer is not None):
            logger.info(
                "[autoscaling async] rank %s cleared captured sender KV "
                "pointer view after %s",
                self.rank, reason)
        self._autoscale_sender_key_cache_ptrs = None
        self._autoscale_sender_value_cache_ptrs = None
        self._autoscale_sender_start_layer = None

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
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
        free_before_cleanup, total_gpu_memory = torch.cuda.mem_get_info()
        
        # Validate granularity for combined_layers mode
        if hasattr(self.model_runner, 'layer_group_granularity') and self.model_runner.layer_group_granularity > 1:
            validate_layers_count_granularity(
                layer_list,
                self.model_runner.layer_group_granularity,
                operation="add_layers"
        )
        
        with self._pending_deleted_model_layers_lock:
            has_pending_deleted_layers = bool(self._pending_deleted_model_layers)

        cleanup_start = time.time()
        if has_pending_deleted_layers:
            if self.migration_stream is not None:
                self.migration_stream.synchronize()
            else:
                _sync_current_cuda_stream(self.device)
            gc.collect()
        free_after_cleanup, _ = torch.cuda.mem_get_info()
        logger.info(
            "[dynamic-load-worker]: pre_add_cleanup layers=%s ran=%s took=%s free_gpu_before=%.2fGB free_gpu_after=%.2fGB total_gpu=%.2fGB",
            layer_list,
            has_pending_deleted_layers,
            human_readable_duration(time.time() - cleanup_start),
            free_before_cleanup / 1024 ** 3,
            free_after_cleanup / 1024 ** 3,
            total_gpu_memory / 1024 ** 3,
        )
        
        # Check overhead BEFORE operation (if monitoring enabled)
        if not self.vllm_config.dynamic_config.disable_memory_overhead_monitor:
            layer_count_before, kv_cache_bytes = self._get_current_memory_params()
            overhead_monitor = get_memory_overhead_monitor()
            overhead_monitor.check_overhead_before(
                operation=f"add_layers_{layer_list}",
                current_layer_count=layer_count_before,
                current_kv_cache_bytes=kv_cache_bytes
            )
        
        with DeadlockTimeoutContext(self._layer_loaded_cv, "_layer_loaded_cv", timeout=2):
            assert self.migration_stream is not None
            with torch.cuda.stream(self.migration_stream):
                logger.info(f"start to load layer, current stream:{torch.cuda.current_stream()}")
                # 记录扩容前的起始 layer，便于在 start_layer 左移时重排本地 kv 索引基准
                old_start_layer = self.model_runner.model.model.start_layer
                old_end_layer = self.model_runner.model.model.end_layer
                old_kv_cache_start_layer = self._kv_cache_start_layer()
                assert self.device is not None
                enqueue_start = time.time()
                self.model_runner.add_layers(layer_list, self.device)
                logger.info(
                    "[dynamic-load-worker]: add_layers enqueued layers=%s took=%s on stream=%s",
                    layer_list,
                    human_readable_duration(time.time() - enqueue_start),
                    self.migration_stream,
                )

            stream_sync_start = time.time()
            self.migration_stream.synchronize()
            free_after_weight_loading, _ = torch.cuda.mem_get_info()
            logger.info(
                "[dynamic-load-worker]: migration_stream synchronize after add_layers layers=%s took=%s free_gpu_after=%.2fGB",
                layer_list,
                human_readable_duration(time.time() - stream_sync_start),
                free_after_weight_loading / 1024 ** 3,
            )
            logger.info(f"[timeline]: after weight loading, time taken: {human_readable_duration(time.time() - time_start)}")

            # 接收完weights之后，在这里进行kv cache数据结构的扩展，保证后续kv cache tensor绑定的正确性
            new_start_layer = self.model_runner.model.model.start_layer
            new_end_layer = self.model_runner.model.model.end_layer
            old_layers_empty = old_end_layer <= old_start_layer
            logger.info(f"debug: ---------------------add layers, old_start_layer: {old_start_layer}, new_start_layer: {new_start_layer}, old_end_layer: {old_end_layer}, new_end_layer: {new_end_layer}")

            forward_lock_wait_start = time.time()
            with self.model_runner.forward_lock:
                time_within_lock_start = time.time()
                logger.info(
                    "[dynamic-load-worker]: add_layers acquired forward_lock "
                    "for KV structure update after %s for layers=%s",
                    human_readable_duration(time_within_lock_start -
                                            forward_lock_wait_start),
                    layer_list)
                # 在这里更新kv_cache_group
                if is_flexi:
                    if self.vllm_config.dynamic_config.pipeline_autoscaling_enabled:
                        self._autoscale_sender_key_cache_ptrs = list(
                            self.dynamic_kv_synchronizer.key_cache_ptrs)
                        self._autoscale_sender_value_cache_ptrs = list(
                            self.dynamic_kv_synchronizer.value_cache_ptrs)
                        self._autoscale_sender_start_layer = (
                            old_kv_cache_start_layer)
                        logger.info(
                            "[autoscaling async] captured sender KV pointer "
                            "view before add_layers: start_layer=%s, "
                            "num_layers=%s",
                            old_kv_cache_start_layer,
                            len(self._autoscale_sender_key_cache_ptrs))

                    # 如果 start_layer 向更小的下标移动，需要对现有 self.kv_caches 做前置填充，
                    # 使其索引基准与新的 start_layer 对齐
                    if old_layers_empty:
                        left_added_layer_num = 0
                        right_added_layer_num = int(new_end_layer -
                                                    new_start_layer)
                        logger.info(
                            "[autoscaling async] old PP range is empty; "
                            "initializing KV slots directly for new range "
                            "[%s, %s) with %s layers",
                            new_start_layer, new_end_layer,
                            right_added_layer_num)
                    else:
                        left_added_layer_num = int(old_start_layer -
                                                   new_start_layer)
                        right_added_layer_num = int(new_end_layer -
                                                    old_end_layer)
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
                        if use_direct_ptr_for_runtime(self.vllm_config):
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

                        # Pad grouped_handles for combined_layers mode (VMM)
                        # This ensures grouped_handles indices align with key_caches after left-padding
                        if hasattr(self.model_runner, 'grouped_handles') and hasattr(self.model_runner, 'layer_group_granularity'):
                            granularity = self.model_runner.layer_group_granularity
                            n_groups_to_pad = left_added_layer_num // granularity
                            if n_groups_to_pad > 0:
                                self.model_runner.grouped_handles = [[] for _ in range(n_groups_to_pad)] + \
                                                                    self.model_runner.grouped_handles
                                logger.info(f"[_add_layers] Padded grouped_handles left: added {n_groups_to_pad} empty groups, "
                                           f"total groups now: {len(self.model_runner.grouped_handles)}")

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
                        if use_direct_ptr_for_runtime(self.vllm_config):
                            empty_tensor = torch.tensor([], dtype=torch.uint64, device=self.device)
                            self.model_runner.k_ptr_tensors.extend([empty_tensor.clone() for _ in range(pad_right)])
                            self.model_runner.v_ptr_tensors.extend([empty_tensor.clone() for _ in range(pad_right)])

                        self.dynamic_kv_synchronizer.key_cache_list.extend([[] for _ in range(pad_right)])
                        self.dynamic_kv_synchronizer.value_cache_list.extend([[] for _ in range(pad_right)])
                        self.dynamic_kv_synchronizer.key_cache_ptrs.extend([0] * pad_right)
                        self.dynamic_kv_synchronizer.value_cache_ptrs.extend([0] * pad_right)

                        # Pad grouped_handles for combined_layers mode (VMM)
                        # This ensures grouped_handles indices align with key_caches after right-padding
                        if hasattr(self.model_runner, 'grouped_handles') and hasattr(self.model_runner, 'layer_group_granularity'):
                            granularity = self.model_runner.layer_group_granularity
                            n_groups_to_pad = pad_right // granularity
                            if n_groups_to_pad > 0:
                                self.model_runner.grouped_handles.extend([[] for _ in range(n_groups_to_pad)])
                                logger.info(f"[_add_layers] Padded grouped_handles right: added {n_groups_to_pad} empty groups, "
                                           f"total groups now: {len(self.model_runner.grouped_handles)}")

                    if old_layers_empty or left_added_layer_num > 0:
                        self._set_kv_cache_start_layer(
                            new_start_layer,
                            "add_layers left/empty KV cache padding")

                    logger.info(f"[timeline]: after add layers, time taken: {human_readable_duration(time.time() - time_start)}")
                    for i in range(len(self.model_runner.key_caches)):
                        logger.info(f"key cache len: {len(self.model_runner.key_caches)}")
                else:
                    # 如果 start_layer 向更小的下标移动，需要对现有 self.kv_caches 做前置填充，
                    # 使其索引基准与新的 start_layer 对齐
                    if old_layers_empty:
                        left_added_layer_num = 0
                        right_added_layer_num = int(new_end_layer -
                                                    new_start_layer)
                    else:
                        left_added_layer_num = int(old_start_layer -
                                                   new_start_layer)
                        right_added_layer_num = int(new_end_layer -
                                                    old_end_layer)
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
                    if old_layers_empty or left_added_layer_num > 0:
                        self._set_kv_cache_start_layer(
                            new_start_layer,
                            "add_layers left/empty KV cache padding")
                    for i in range(len(self.model_runner.kv_caches)):
                        logger.info(f"kv cache {i}: {self.model_runner.kv_caches[i].shape}")
                forward_lock_hold_ms = (time.time() - time_within_lock_start) * 1000
                if self.log_stop_time:
                    logger.info(f"[STOP_TIME][worker][add_layers][forward_lock]: hold={forward_lock_hold_ms:.2f}ms")
                logger.info(f"[timeline]: time within lock when add layers time taken: {human_readable_duration(time.time() - time_within_lock_start)}")
        
            logger.info(f"kv scynchronizer kv cache list length after adding layers: {len(self.dynamic_kv_synchronizer.key_cache_ptrs)}")
            # 唤醒等待层加载的线程（kv tensor的绑定线程和kv patch 应用线程）。
            self._layer_loaded_cv.notify_all()
            logger.info(f"[debug]: notified all layer loaded cv waiters")
        
        # Check overhead AFTER operation (if monitoring enabled)
        if not self.vllm_config.dynamic_config.disable_memory_overhead_monitor:
            layer_count_after, kv_cache_bytes_after = self._get_current_memory_params()
            overhead_monitor = get_memory_overhead_monitor()
            overhead_monitor.check_overhead_after(
                operation=f"add_layers_{layer_list}",
                new_layer_count=layer_count_after,
                new_kv_cache_bytes=kv_cache_bytes_after
            )

    def init_device(self):
        # This function is copied from Worker.init_device
        # We override this becuase this function is used to initialize the GPUModelRunner and we want to use our own DynamicGPUModelRunner
        maybe_apply_kvcached_vllm_patches("dynamic GPU worker init_device")

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

    def _kv_cache_start_layer(self) -> int:
        runner_start = getattr(
            self.model_runner,
            "kv_cache_start_layer",
            getattr(self.model_runner.model.model, "start_layer", 0))
        sync = getattr(self, "dynamic_kv_synchronizer", None)
        if sync is not None:
            sync_start = getattr(sync, "kv_cache_start_layer", runner_start)
            if sync_start != runner_start:
                raise RuntimeError(
                    "KV cache start layer mismatch between runner and "
                    f"synchronizer: runner={runner_start}, "
                    f"synchronizer={sync_start}")
        return int(runner_start)

    def _set_kv_cache_start_layer(self, start_layer: int, reason: str) -> None:
        assert isinstance(self.model_runner, DynamicGPUModelRunner)
        self.model_runner.set_kv_cache_start_layer(start_layer, reason)
        sync = getattr(self, "dynamic_kv_synchronizer", None)
        if sync is not None:
            old_sync_start = getattr(sync, "kv_cache_start_layer", None)
            sync.kv_cache_start_layer = int(start_layer)
            if old_sync_start != int(start_layer):
                logger.info(
                    "dynamic_kv_synchronizer.kv_cache_start_layer: %s -> %s "
                    "(%s)", old_sync_start, start_layer, reason)

    @staticmethod
    def _kv_ptr_view_covers_layers(start_layer: Optional[int],
                                   key_cache_ptrs: Optional[list[int]],
                                   value_cache_ptrs: Optional[list[int]],
                                   layer_ids: list[int]) -> bool:
        if (start_layer is None or key_cache_ptrs is None
                or value_cache_ptrs is None):
            return False
        if len(key_cache_ptrs) != len(value_cache_ptrs):
            return False
        for layer_id in layer_ids:
            local_layer_id = layer_id - start_layer
            if local_layer_id < 0 or local_layer_id >= len(key_cache_ptrs):
                return False
            if (key_cache_ptrs[local_layer_id] == 0
                    or value_cache_ptrs[local_layer_id] == 0):
                return False
        return True

    @staticmethod
    def _kv_cache_start_after_deleting_layers(
            current_start: int,
            layers_list: list[Tuple[int, int]]) -> int:
        new_start = int(current_start)
        for start, end in sorted(layers_list):
            if start <= new_start <= end:
                new_start = end + 1
            elif start > new_start:
                break
        return new_start

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
        from vllm.v1.executor.dynamic_utils import (
            set_vllm_ray_rdt_kv_nccl_lock,
        )
        set_vllm_ray_rdt_kv_nccl_lock(
            self.dynamic_kv_synchronizer.get_nccl_lock())
        logger.info("Registered KV NCCL lock for Ray RDT transport")
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
            
        kv_cache_config = kv_cache_configs[self.rank]
        if not kv_cache_config.kv_cache_groups:
            template_group = next(
                (cfg.kv_cache_groups[0] for cfg in kv_cache_configs
                 if cfg.kv_cache_groups), None)
            if template_group is None:
                raise RuntimeError("No KV cache spec found on any worker")
            kv_cache_config = KVCacheConfig(
                num_blocks=num_blocks,
                tensors={},
                kv_cache_groups=[
                    KVCacheGroupSpec([], template_group.kv_cache_spec)
                ],
            )
            kv_cache_configs[self.rank] = kv_cache_config

        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = kv_cache_spec.block_size
        self.per_block_kv_cache_bytes = kv_cache_spec.page_size_bytes  # K+V size per block per layer
        self.block_num = num_blocks
        
        # NOTE: Profile run and MemoryOverheadMonitor baseline initialization 
        # is now done in determine_available_memory() which is called before this.
        # This ensures overhead is measured and used for KV cache block calculation.
        
        # Initialize KV cache
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            if use_flexi_kv_for_runtime(self.vllm_config):
                logger.info("Using flexi flash attention dynamic initialize kv cache")
                self.model_runner.dynamic_initialize_kv_cache_flexi(kv_cache_config, self.dynamic_kv_synchronizer, num_blocks)
            else:
                logger.info("Using standard flash attention dynamic initialize kv cache")
                self.model_runner.dynamic_initialize_kv_cache(kv_cache_config, self.dynamic_kv_synchronizer, num_blocks)
            self.dynamic_kv_synchronizer.create_slot_mappings(num_blocks * self.block_size)

    def set_env_var(self, key: str, value: str) -> None:
        """Update an environment variable in this worker process."""
        os.environ[key] = value
        logger.info(f"Worker {self.rank}: Updated env var {key}={value}")

    def set_log_stop_time(self, enabled: bool) -> None:
        """Enable or disable STOP_TIME logging.
        
        Used to disable logging during set_pp_config to avoid polluting
        migration metrics with initialization overhead.
        """
        self.log_stop_time = enabled

    def set_active_pp_ranks(self, active_ranks: Optional[list[int]]) -> None:
        """Update the active PP routing subset for pipeline autoscaling.

        The underlying distributed PP group remains the candidate-rank
        superset. This only changes first/last/next/prev PP routing semantics
        used by model forward and pipeline output handling.
        """
        set_pp_group_active_ranks(active_ranks)
        logger.info("Worker %s: updated active PP ranks to %s",
                    self.rank, active_ranks)

    def prepare_autoscaling_request_states_from_sync_batch(
        self,
        scheduler_output: DynamicSchedulerOutput,
    ) -> None:
        if not getattr(scheduler_output, "autoscaling_request_state_sync",
                       False):
            return

        request_states = getattr(scheduler_output,
                                 "autoscaling_request_states", None)
        if request_states:
            request_states = {
                req_id: self._normalize_autoscaling_request_state(req_id,
                                                                  req_state)
                for req_id, req_state in request_states.items()
            }
            scheduler_output.autoscaling_request_states = request_states

        refresh_ranks = set(
            getattr(scheduler_output,
                    "autoscaling_request_state_refresh_ranks", ()) or ())
        should_refresh_local_states = self.rank in refresh_ranks
        if should_refresh_local_states:
            if not request_states:
                raise RuntimeError(
                    "Autoscaling request-state sync asked rank "
                    f"{self.rank} to refresh, but the target batch carried "
                    "no request states")
            start = time.time()
            for req_id, req_state in request_states.items():
                self.model_runner.requests[req_id] = deepcopy(req_state)
            logger.info(
                "[autoscaling request states] rank %s refreshed %d local "
                "request states from sync batch source_rank=%s in %s",
                self.rank, len(request_states),
                scheduler_output.autoscaling_request_state_source_rank,
                human_readable_duration(time.time() - start))
        else:
            exported_states = self._export_autoscaling_request_states_for_batch(
                scheduler_output)
            if exported_states:
                if request_states is None:
                    request_states = exported_states
                    scheduler_output.autoscaling_request_state_source_rank = (
                        self.rank)
                    added_states = len(exported_states)
                    skipped_states = 0
                else:
                    added_states = 0
                    skipped_states = 0
                    for req_id, req_state in exported_states.items():
                        if req_id in request_states:
                            skipped_states += 1
                            continue
                        request_states[req_id] = req_state
                        added_states += 1
                scheduler_output.autoscaling_request_states = request_states
                logger.info(
                    "[autoscaling request states] rank %s attached %d worker "
                    "request states to target PP batch; added=%d skipped=%d "
                    "total_states=%d source_rank=%s refresh_ranks=%s",
                    self.rank, len(exported_states), added_states,
                    skipped_states, len(request_states),
                    scheduler_output.autoscaling_request_state_source_rank,
                    sorted(refresh_ranks))

        request_states = getattr(scheduler_output,
                                 "autoscaling_request_states", None)
        missing = [
            req.req_id for req in scheduler_output.scheduled_cached_reqs
            if req.req_id not in self.model_runner.requests
        ]
        if missing and not request_states:
            raise RuntimeError(
                "Autoscaling target batch was marked for request-state sync "
                f"but did not carry states for missing requests on rank "
                f"{self.rank}: {missing}")

        start = time.time()
        imported = 0
        request_states = request_states or {}
        for req_id, req_state in request_states.items():
            if req_id not in self.model_runner.requests:
                self.model_runner.requests[req_id] = deepcopy(
                    self._normalize_autoscaling_request_state(req_id,
                                                             req_state))
                imported += 1

        unresolved = [
            req_id for req_id in missing
            if req_id not in self.model_runner.requests
        ]
        if unresolved:
            raise RuntimeError(
                "Autoscaling target batch did not receive request states for "
                f"rank {self.rank}: missing={unresolved} "
                f"available_states={list(request_states.keys())[:16]}")

        if imported:
            logger.info(
                "[autoscaling request states] rank %s imported %d "
                "missing request states from sync batch source_rank=%s in %s",
                self.rank, imported,
                scheduler_output.autoscaling_request_state_source_rank,
                human_readable_duration(time.time() - start))

        self._reset_input_batch_for_autoscaling_sync_batch(scheduler_output)

    def _normalize_autoscaling_request_state(
        self,
        req_id: str,
        req_state: Any,
    ) -> Any:
        normalized = deepcopy(req_state)
        kv_cache_config = getattr(self.model_runner, "kv_cache_config", None)
        kv_cache_groups = getattr(kv_cache_config, "kv_cache_groups", None)
        expected_groups = len(kv_cache_groups or ()) or 1
        block_ids = getattr(normalized, "block_ids", None)
        if not block_ids:
            normalized.block_ids = [[] for _ in range(expected_groups)]
            return normalized
        if len(block_ids) != expected_groups:
            raise RuntimeError(
                "Autoscaling request-state sync received malformed "
                f"block_ids for request {req_id} on rank {self.rank}: "
                f"groups={len(block_ids)} expected={expected_groups}")
        normalized.block_ids = [list(group) for group in block_ids]
        return normalized

    def _reset_input_batch_for_autoscaling_sync_batch(
        self,
        scheduler_output: DynamicSchedulerOutput,
    ) -> None:
        input_batch = self.model_runner.input_batch
        old_num_reqs = input_batch.num_reqs
        planned_num_reqs = (len(scheduler_output.scheduled_new_reqs) +
                            len(scheduler_output.scheduled_cached_reqs))
        removed_req_indices: list[int] = []
        for req_id in list(input_batch.req_id_to_index.keys()):
            req_index = input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)
        removed_req_indices.sort(reverse=True)
        input_batch.condense(removed_req_indices)
        input_batch.refresh_sampling_metadata()
        logger.info(
            "[autoscaling request states] rank %s reset input_batch before "
            "target sync batch; old_num_reqs=%d planned_num_reqs=%d "
            "removed=%d",
            self.rank, old_num_reqs, planned_num_reqs,
            len(removed_req_indices))

    def _export_autoscaling_request_states_for_batch(
        self,
        scheduler_output: DynamicSchedulerOutput,
    ) -> dict[str, Any]:
        request_states: dict[str, Any] = {}
        req_ids = getattr(scheduler_output,
                          "autoscaling_request_state_req_ids", None)
        if req_ids is not None:
            for req_id in req_ids:
                req_state = self.model_runner.requests.get(req_id)
                if req_state is not None:
                    request_states[req_id] = (
                        self._normalize_autoscaling_request_state(
                            req_id, req_state))
            return request_states

        for req in scheduler_output.scheduled_cached_reqs:
            req_state = self.model_runner.requests.get(req.req_id)
            if req_state is not None:
                request_states[req.req_id] = (
                    self._normalize_autoscaling_request_state(
                        req.req_id, req_state))
        return request_states

    def _autoscaling_live_request_block_ids(self) -> set[int]:
        """Return blocks for requests that still need KV after migration."""
        live_block_ids: set[int] = set()
        requests = getattr(self.model_runner, "requests", {})
        for req_state in list(requests.values()):
            block_groups = getattr(req_state, "block_ids", None)
            if not block_groups:
                continue
            for block_id in block_groups[0]:
                block_id = int(block_id)
                if block_id >= 0:
                    live_block_ids.add(block_id)
        return live_block_ids

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
            self.dynamic_kv_synchronizer.send_kv_cache_patch(
                self.rank_to_layers_ids, slot_mapping,
                self.model_runner.kv_caches, self._kv_cache_start_layer())
        return result


    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """
        Profiles the peak memory usage of the model using a dummy forward pass
        to determine the runtime overhead (activations, CUDA context, NCCL buffers).
        
        This follows the original vLLM implementation but also initializes
        the MemoryOverheadMonitor baseline for leak detection during migration.

        Returns:
            Available memory for KV cache in bytes (after subtracting safe margin)
        """
        assert isinstance(self.model_runner.model, DynamicModelBase) or isinstance(self.model_runner.model, torch.nn.Module), "model should be an instance of DynamicModelBase or torch.nn.Module"
        
        # Initialize intermediate states before profiling
        self.model_runner.initialize_intermediate_states()
        
        # Get layer info for overhead calculation
        if isinstance(self.model_runner.model, DynamicModelBase):
            start_layer = self.model_runner.model.model.start_layer
            end_layer = self.model_runner.model.model.end_layer
            layer_count = end_layer - start_layer
            per_layer_weight_bytes = self.model_runner.model.get_layer_weight_size()
        else:
            layer_count = 0
            per_layer_weight_bytes = 0
        
        total_gpu_memory = get_total_gpu_memory(self.rank)
        
        # Profile run to measure peak memory (following original vLLM implementation)
        logger.info(f"[MemoryOverheadMonitor] Running profile_run to measure runtime overhead...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Execute dummy forward pass to measure peak memory
        self.model_runner.profile_run()
        
        # Get peak memory from profile run
        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        
        # Check for non-torch allocations (NCCL, etc.) - following original vLLM
        torch.cuda.empty_cache()
        torch_allocated = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        total_allocated = total_gpu_memory - torch.cuda.mem_get_info()[0]
        non_torch_allocations = max(0, total_allocated - torch_allocated)
        
        # Add non-torch allocations to peak memory (like original vLLM)
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations
        
        # Calculate runtime overhead = peak_memory - expected_weight_memory
        expected_weight_memory = layer_count * per_layer_weight_bytes
        runtime_overhead = peak_memory - expected_weight_memory
        
        logger.info(
            f"[MemoryOverheadMonitor] Profile run complete | rank={self.rank} | "
            f"layers={layer_count}, weight={expected_weight_memory / 1024**3:.3f}GB, "
            f"peak_memory={peak_memory / 1024**3:.3f}GB, non_torch={non_torch_allocations / 1024**3:.3f}GB, "
            f"runtime_overhead={runtime_overhead / 1024**3:.3f}GB"
        )
        
        # Initialize memory overhead monitor with profile results
        if isinstance(self.model_runner.model, DynamicModelBase):
            overhead_monitor = reset_memory_overhead_monitor(
                rank=self.rank,
                device=self.device,
                per_layer_weight_bytes=per_layer_weight_bytes,
                tolerance_gb=0.15  # 150MB tolerance
            )
            overhead_monitor.initialize_baseline_with_profile(
                layer_count=layer_count,
                profile_peak_memory_bytes=peak_memory - non_torch_allocations,  # torch peak only
                non_torch_allocations_bytes=non_torch_allocations
            )
        
        # Calculate available memory using original vLLM formula
        available_kv_cache_memory = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_memory
        )
        
        logger.info(
            f"debug ------- determine available memory: {available_kv_cache_memory / 1024 ** 3:.2f} GB, "
            f"peak_memory: {peak_memory / 1024 ** 3:.2f} GB, "
            f"total_gpu_memory: {total_gpu_memory / 1024 ** 3:.2f} GB"
        )
        return int(available_kv_cache_memory)

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
            self._wait_for_async_resize("async add layers")
            self._add_layers(layer_list)
        # 异步调用：与工作的 commit 9ed5ec00f2c94 保持一致
        # 注意：daemon=False 确保线程在进程退出前完成
        thread = threading.Thread(target=_do_add, daemon=False)
        with self._async_add_layers_lock:
            self._async_add_layers_threads.append(thread)
        thread.start()
        logger.info(f"[timeline]: after add layers, time taken: {human_readable_duration(time.time() - time_start)}")

    def wait_for_async_add_layers(self) -> None:
        """Wait for all async_add_layers threads to complete."""
        threads_to_wait = []
        with self._async_add_layers_lock:
            threads_to_wait = list(self._async_add_layers_threads)
        
        logger.info(f"[async_fast] Waiting for {len(threads_to_wait)} async_add_layers threads to complete")
        for thread in threads_to_wait:
            thread.join()
        
        with self._async_add_layers_lock:
            self._async_add_layers_threads.clear()
        logger.info(f"[async_fast] All async_add_layers threads completed")

    def add_layers(self, rank: int, layer_list: list[Tuple[int, int]]) -> None:
        if self.rank != rank:
            logger.info(f"Worker {self.rank} is not the target rank {rank}, skip adding model layers")
            return None
        logger.info(f"[operation]: sync Add Model Layers: {layer_list}")
        time_start = time.time()
        def _do_add():
            # Set CUDA device for this thread - threads don't inherit CUDA context
            torch.cuda.set_device(self.device)
            self._add_layers(layer_list)
        _do_add()
        logger.info(f"[timeline]: after add layers, time taken: {human_readable_duration(time.time() - time_start)}")

    def remove_layers(self,
                      rank: int,
                      layer_list: list[Tuple[int, int]],
                      release_immediately: bool = True) -> list[list[object]]:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip removing model layers")
            return []
        layer_list = self._merge_contiguous_layer_ranges(layer_list)
        logger.info(f"Remove Model Layers: {layer_list}")
        assert self.device is not None
        
        # Check overhead BEFORE operation (if monitoring enabled)
        if not self.vllm_config.dynamic_config.disable_memory_overhead_monitor:
            layer_count_before, kv_cache_bytes = self._get_current_memory_params()
            overhead_monitor = get_memory_overhead_monitor()
            overhead_monitor.check_overhead_before(
                operation=f"remove_layers_{layer_list}",
                current_layer_count=layer_count_before,
                current_kv_cache_bytes=kv_cache_bytes
            )
        
        detached_layers = self.model_runner.remove_layers(
            layer_list, self.device, release_immediately=release_immediately)
        
        # Check overhead AFTER operation (if monitoring enabled)
        if not self.vllm_config.dynamic_config.disable_memory_overhead_monitor:
            layer_count_after, kv_cache_bytes_after = self._get_current_memory_params()
            overhead_monitor = get_memory_overhead_monitor()
            overhead_monitor.check_overhead_after(
                operation=f"remove_layers_{layer_list}",
                new_layer_count=layer_count_after,
                new_kv_cache_bytes=kv_cache_bytes_after
            )
        return detached_layers

    def _queue_deleted_model_layers_after_inference(
            self,
            detached_layers: list[list[object]]) -> Optional[threading.Event]:
        if not detached_layers:
            return None
        release_event = threading.Event()
        with self._pending_deleted_model_layers_lock:
            self._pending_deleted_model_layers.extend(detached_layers)
            self._pending_deleted_model_layers_events.append(release_event)
            pending_count = sum(len(group)
                                for group in self._pending_deleted_model_layers)
        logger.info("[delete_layers] queued %d detached layer objects for "
                    "after-inference release", pending_count)
        return release_event

    def _release_deleted_model_layers_after_inference(self) -> None:
        with self._pending_deleted_model_layers_lock:
            if not self._pending_deleted_model_layers:
                return
            detached_layers = self._pending_deleted_model_layers
            release_events = self._pending_deleted_model_layers_events
            self._pending_deleted_model_layers = []
            self._pending_deleted_model_layers_events = []
        assert self.device is not None
        release_start = time.time()
        layer_count = sum(len(group) for group in detached_layers)
        logger.info("[delete_layers] after inference releasing %d detached "
                    "layer objects", layer_count)
        try:
            self.model_runner.release_removed_layers(detached_layers,
                                                     self.device)
        finally:
            for event in release_events:
                event.set()
            logger.info("[delete_layers] after inference release took %s",
                        human_readable_duration(time.time() - release_start))

    def atomic_shelve_kv_cache(self, rank: int, layers_list: list[Tuple[int, int]]) -> Tuple[list[list[int]], list[list[int]], list[int], list[int], list[list[int]], list[list[int]], list[tuple[list[int], list[int]]]]:
        """
        Atomically shelve KV cache for specified layers.
        
        Returns:
            Tuple of (caches_to_free_key, caches_to_free_value, ptrs_to_free_key, ptrs_to_free_value,
                      handles_to_free_key, handles_to_free_value, grouped_handles_to_free)
            
            Note: grouped_handles_to_free is a list of (handles, base_k_ptrs) tuples to ensure
            correct pairing between VMM handles and virtual addresses during memory release.
        """
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip releasing kv cache for layers")
            return [], [], [], [], [], [], []
        layers_list = self._merge_contiguous_layer_ranges(layers_list)
        # self.model_runner.release_kv_cache_for_layers(layers_list)
        from vllm.config import get_current_vllm_config
        vllm_config = get_current_vllm_config()
        is_flexi = use_flexi_kv_for_runtime(vllm_config)
        start_layer = self._kv_cache_start_layer()
        free_before, total = torch.cuda.mem_get_info()
        logger.info(
            "before release_kv_cache_for_layers: free=%.2f GB, total=%.2f "
            "GB, kv_cache_start_layer=%s, model_start_layer=%s",
            free_before / 1024 ** 3, total / 1024 ** 3, start_layer,
            self.model_runner.model.model.start_layer)
        caches_to_free_key: list[list[int]] = []
        caches_to_free_value: list[list[int]] = []
        ptrs_to_free_key: list[int] = []
        ptrs_to_free_value: list[int] = []
        handles_to_free_key: list[list[int]] = []
        handles_to_free_value: list[list[int]] = []
        grouped_handles_to_free: list[tuple[list[int], list[int]]] = []  # For combined_layers mode: (handles, base_k_ptrs)
        
        # Check for combined_layers mode
        use_combined_layers = (hasattr(self.model_runner, 'layer_group_granularity') and 
                               self.model_runner.layer_group_granularity > 1 and
                               hasattr(self.model_runner, 'grouped_handles'))
        
        if use_combined_layers:
            granularity = self.model_runner.layer_group_granularity
            # Validate: deleted layers must be complete groups AND align to group boundary
            validate_layers_granularity(
                layers_list, start_layer, granularity, 
                operation="atomic_shelve_kv_cache"
            )
            
            # Calculate which groups will be deleted based on layers_list
            # layers_list contains global layer indices, convert to local indices first
            # then map to group indices
            groups_to_delete = set()
            for layers in layers_list:
                # Convert global layers to local layer indices
                local_start = layers[0] - start_layer
                local_end = layers[1] - start_layer
                # Map local layer indices to group indices
                start_group = local_start // granularity
                end_group = local_end // granularity
                for g in range(start_group, end_group + 1):
                    groups_to_delete.add(g)
            groups_to_delete = sorted(groups_to_delete)
            num_groups_to_delete = len(groups_to_delete)
            
            logger.info(f"combined_layers shelve: deleting {num_groups_to_delete} groups, "
                        f"groups_to_delete={groups_to_delete}")
            
            # Collect grouped_handles to free WITH their corresponding base_k_ptrs
            # This ensures handles and VAs are correctly paired even after migrations
            for group_idx in groups_to_delete:
                if group_idx < len(self.model_runner.grouped_handles):
                    handles = self.model_runner.grouped_handles[group_idx]
                    # Get the base K pointers for this group (first layer in the group)
                    first_layer_in_group = group_idx * granularity
                    if first_layer_in_group < len(self.model_runner.key_caches):
                        base_k_ptrs = self.model_runner.key_caches[first_layer_in_group]
                        # Store as tuple (handles, base_k_ptrs) to keep them bound together
                        grouped_handles_to_free.append((handles, base_k_ptrs))
                        # Debug: print handles and VAs being collected
                        if len(handles) > 0:
                            logger.info(f"[DEBUG COLLECT] group_idx={group_idx}, handles[:3]={handles[:3]}, "
                                       f"base_k_ptrs[:3]={[hex(p) for p in base_k_ptrs[:3]] if base_k_ptrs else []}")
                    else:
                        logger.warning(f"[COLLECT WARNING] group_idx={group_idx}: first_layer_in_group={first_layer_in_group} "
                                      f">= key_caches len={len(self.model_runner.key_caches)}")
        
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

            self.model_runner.flexi_atomic_switch_kv_cache_config_for_layers(
                layers_list, kv_cache_start_layer=start_layer)
            
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
            self.model_runner.atomic_switch_kv_cache_config_for_layers(
                layers_list, kv_cache_start_layer=start_layer)
            self.dynamic_kv_synchronizer.kv_caches = [
                kv_cache for idx, kv_cache in enumerate(self.dynamic_kv_synchronizer.kv_caches) 
                if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
            ]
            if use_kvcached_backend():
                deleted_layers = {
                    layer for layers in layers_list
                    for layer in range(layers[0], layers[1] + 1)
                }
                layer_names = getattr(self.model_runner,
                                      "_kvcached_layer_names", None)
                if layer_names is not None:
                    self.model_runner._kvcached_layer_names = [
                        name for name in layer_names
                        if extract_layer_index(name) not in deleted_layers
                    ]
        new_start_layer = self._kv_cache_start_after_deleting_layers(
            start_layer, layers_list)
        self._set_kv_cache_start_layer(new_start_layer,
                                       "atomic_shelve_kv_cache")
        return caches_to_free_key, caches_to_free_value, ptrs_to_free_key, ptrs_to_free_value, handles_to_free_key, handles_to_free_value, grouped_handles_to_free

    @staticmethod
    def _merge_contiguous_layer_ranges(
        layer_ranges: list[Tuple[int, int]],
    ) -> list[Tuple[int, int]]:
        if not layer_ranges:
            return []

        normalized = sorted(
            (start, end) for start, end in layer_ranges if start <= end)
        if not normalized:
            return []

        merged: list[Tuple[int, int]] = []
        for start, end in normalized:
            if not merged or start > merged[-1][1] + 1:
                merged.append((start, end))
            else:
                prev_start, prev_end = merged[-1]
                merged[-1] = (prev_start, max(prev_end, end))
        return merged

    def release_kv_cache_for_layers(
        self, 
        rank: int,  
        caches_to_free_key: list[list[int]], 
        caches_to_free_value: list[list[int]], 
        ptrs_to_free_key: list[int], 
        ptrs_to_free_value: list[int],
        handles_to_free_key: Optional[list[list[int]]] = None,
        handles_to_free_value: Optional[list[list[int]]] = None,
        grouped_handles_to_free: Optional[list[tuple[list[int], list[int]]]] = None  # Changed: list of (handles, base_k_ptrs) tuples
    ) -> None:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip releasing kv cache for layers")
            return None
        # self.model_runner.release_kv_cache_for_layers(layers_list)
        from vllm.config import get_current_vllm_config
        from vllm.v1.worker.gpu_memory_monitor import memory_snapshot
        vllm_config = get_current_vllm_config()
        is_flexi = use_flexi_kv_for_runtime(vllm_config)
        
        # Take memory snapshot before release without a device-wide sync.
        before_snapshot = _memory_snapshot_no_device_sync(
            f"rank{self.rank}_before_release_kv", self.device)
        free_before = before_snapshot['free_gb'] * 1024 ** 3
        total = before_snapshot['total_gb'] * 1024 ** 3
        logger.info(f"before release_kv_cache_for_layers: free={free_before / 1024 ** 3:.2f} GB, total={total / 1024 ** 3:.2f} GB")
        
        # Calculate expected memory to free
        num_layers = len(caches_to_free_key) if caches_to_free_key else 0
        num_blocks = len(caches_to_free_key[0]) if caches_to_free_key and len(caches_to_free_key) > 0 else 0
        
        # Check for combined_layers mode
        use_combined_layers = (hasattr(self.model_runner, 'layer_group_granularity') and 
                               self.model_runner.layer_group_granularity > 1 and
                               grouped_handles_to_free is not None and
                               len(grouped_handles_to_free) > 0)
        
        expected_kv_bytes = 0
        expected_ptr_bytes = num_layers * 8 * 2  # K_ptr and V_ptr arrays per layer
        
        if use_combined_layers:
            # combined_layers mode: all layers in a group share the same VMM blocks
            # Expected = num_blocks * aligned_combined_bytes (per group)
            aligned_bytes = self.model_runner.vmm_aligned_bytes
            num_groups = len(grouped_handles_to_free)
            expected_kv_bytes = num_blocks * aligned_bytes * num_groups
            logger.info(f"[Memory Monitor RELEASE] combined_layers mode: "
                        f"num_groups={num_groups}, num_blocks={num_blocks}, "
                        f"aligned_bytes={aligned_bytes}, "
                        f"expected_kv_bytes={expected_kv_bytes / 1024**2:.2f} MB")
        else:
            # Per-layer mode
            use_vmm = getattr(self.model_runner, 'vmm_aligned_bytes', 0) > 0
            if use_vmm:
                aligned_bytes = self.model_runner.vmm_aligned_bytes
                combined_mode = getattr(self.model_runner, 'vmm_combined_mode', False)
                if combined_mode:
                    # VMM combined (K+V share): num_layers * num_blocks * aligned_bytes
                    expected_kv_bytes = num_layers * num_blocks * aligned_bytes
                else:
                    # VMM separate: num_layers * num_blocks * aligned_bytes * 2
                    expected_kv_bytes = num_layers * num_blocks * aligned_bytes * 2
                logger.info(f"[Memory Monitor RELEASE] vmm mode (combined={combined_mode}): "
                            f"num_layers={num_layers}, num_blocks={num_blocks}, "
                            f"aligned_bytes={aligned_bytes}, "
                            f"expected_kv_bytes={expected_kv_bytes / 1024**2:.2f} MB")
            else:
                # cudaMallocAsync: use actual tensor size
                if hasattr(self.model_runner, 'page_meta') and self.model_runner.page_meta is not None:
                    T, H, Dh = self.model_runner.page_meta.shape
                    dtype_size = self.model_runner.page_meta.element_size()
                    bytes_per_block = T * H * Dh * dtype_size
                    expected_kv_bytes = num_layers * num_blocks * bytes_per_block * 2
                    logger.info(f"[Memory Monitor RELEASE] cudaMallocAsync mode: "
                                f"num_layers={num_layers}, num_blocks={num_blocks}, "
                                f"bytes_per_block={bytes_per_block}, "
                                f"expected_kv_bytes={expected_kv_bytes / 1024**2:.2f} MB")
        
        total_expected_bytes = expected_kv_bytes + expected_ptr_bytes
        logger.info(f"[Memory Monitor RELEASE] EXPECTED to free: "
                    f"kv_bytes={expected_kv_bytes / 1024**2:.2f} MB, "
                    f"ptr_bytes={expected_ptr_bytes / 1024**2:.2f} MB, "
                    f"total={total_expected_bytes / 1024**2:.2f} MB")
        
        # Calculate expected memory freed from KV cache using page_meta
        kv_cache_bytes_to_free = 0
        ptr_bytes_to_free = 0
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
            
            # Check for combined_layers mode (multiple layers share VMM blocks)
            use_combined_layers = (hasattr(self.model_runner, 'layer_group_granularity') and 
                                   self.model_runner.layer_group_granularity > 1 and
                                   grouped_handles_to_free is not None and
                                   len(grouped_handles_to_free) > 0)
            
            if use_combined_layers:
                # combined_layers mode: free using grouped_handles
                # FIXED: Now using pre-bound (handles, base_k_ptrs) tuples to ensure correct pairing
                granularity = self.model_runner.layer_group_granularity
                vmm_bytes_per_kv = self.model_runner.vmm_bytes_per_kv
                aligned_bytes = self.model_runner.vmm_aligned_bytes
                
                logger.info(f"[release_kv_cache_for_layers] Combined layers mode: freeing {len(grouped_handles_to_free)} groups "
                            f"(granularity={granularity})")
                
                # Use pre-bound (handles, base_k_ptrs) tuples - handles and VAs are already correctly paired
                for group_idx, (handles, base_k_ptrs) in enumerate(grouped_handles_to_free):
                    if handles and base_k_ptrs:
                        logger.info(f"[release_kv_cache_for_layers] Freeing group {group_idx}: "
                                    f"{len(base_k_ptrs)} blocks, {len(handles)} handles, "
                                    f"first_handle={handles[0] if handles else 'N/A'}, "
                                    f"first_va={hex(base_k_ptrs[0]) if base_k_ptrs else 'N/A'}")
                        with self.model_runner.fbgate.background():
                            kv_allocator.free_vmm_blocks_combined_layers(
                                base_k_ptrs, handles, aligned_bytes, self.device, granularity, vmm_bytes_per_kv
                            )
                    else:
                        logger.warning(f"[release_kv_cache_for_layers] Skipping group {group_idx}: "
                                      f"handles empty={not handles}, base_k_ptrs empty={not base_k_ptrs}")
                
                # Free pointer arrays for all layers
                for k_ptr, v_ptr in zip(ptrs_to_free_key, ptrs_to_free_value):
                    with self.model_runner.fbgate.background():
                        kv_allocator.free_page_list(k_ptr, self.device)
                        kv_allocator.free_page_list(v_ptr, self.device)
            else:
                # Non-combined_layers mode: original per-layer freeing logic
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
                            # VMM mode but no handles - this layer was allocated with cudaMallocAsync
                            # (e.g. during migration when block_size is too small for VMM)
                            # Fall back to free_cache instead of leaking memory
                            # Note: cudaMallocAsync always allocates K and V separately,
                            # regardless of the global combined_mode flag
                            logger.info(f"[release_kv_cache_for_layers] VMM mode but no handles for layer {i}. "
                                           f"Falling back to cudaMallocAsync free.")
                            kv_allocator.free_cache(key_cache, self.device)
                            kv_allocator.free_cache(value_cache, self.device)
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
        
        # Take memory snapshot after release. Only synchronize the stream that
        # enqueued the release work.
        _sync_current_cuda_stream(self.device)
        after_snapshot = _memory_snapshot_no_device_sync(
            f"rank{self.rank}_after_release_kv", self.device)
        
        # Calculate actual freed memory
        actual_freed_bytes = (after_snapshot['free_gb'] - before_snapshot['free_gb']) * 1024 ** 3
        diff_bytes = actual_freed_bytes - total_expected_bytes
        
        # Log the comparison
        logger.info(f"[Memory Monitor RELEASE] ACTUAL freed: {actual_freed_bytes / 1024**2:.2f} MB")
        logger.info(f"[Memory Monitor RELEASE] DIFF (actual - expected): {diff_bytes / 1024**2:.2f} MB")
        if abs(diff_bytes) > 10 * 1024 * 1024:  # More than 10 MB difference
            logger.warning(f"[Memory Monitor RELEASE] ⚠️  MEMORY LEAK DETECTED! "
                          f"Expected to free {total_expected_bytes / 1024**2:.2f} MB, "
                          f"but actually freed {actual_freed_bytes / 1024**2:.2f} MB "
                          f"(diff={diff_bytes / 1024**2:.2f} MB)")
        
        final_free = after_snapshot['free_gb'] * 1024 ** 3
        logger.info(f"after release_kv_cache_for_layers: free={final_free / 1024 ** 3:.2f} GB, model runner's kv cache length: {len(self.model_runner.kv_caches)}")

    def release_kv_cache(self) -> None:
        self.model_runner.release_kv_cache()

    def get_mem_info(self) -> WorkerMemInfo:
        """Return per-layer weight size, current free GPU memory, one-layer
        KV cache tensor size, and runtime overhead (bytes).

        - layer_size: bytes of a single Transformer layer's weights, as
          recorded by DynamicQwen3 during weight loading.
        - free_memory: current free GPU memory in bytes (driver reported).
        - single_kv_cache_tensor_size: bytes of one layer's KV cache tensor
          (includes both K and V within the tensor shape).
        - runtime_overhead_bytes: overhead measured by profile_run (activations,
          CUDA context, NCCL buffers, etc.)
        """
        def collect_mem_info() -> WorkerMemInfo:
            _sync_current_cuda_stream(self.device)
            gc.collect()  # Trigger Python GC to free any unreferenced memory
            if hasattr(self.model_runner.model, 'get_layer_weight_size'):
                layer_size = int(
                    self.model_runner.model.get_layer_weight_size())
            else:
                # Fallback for models without get_layer_weight_size
                layer_size = 0
                logger.warning(
                    "get_mem_info: Model has no get_layer_weight_size "
                    "method, layer_size set to 0")

            is_kv_cache_initialized = (
                len(self.model_runner.kv_caches) != 0
                and self.model_runner.kv_caches[0].numel() != 0)

            # Free memory from driver
            free_memory, _ = torch.cuda.mem_get_info()

            # Size of a single KV cache tensor (for one layer) from model runner
            kv_tensor_size = 0
            assert isinstance(self.model_runner, DynamicGPUModelRunner)
            if is_kv_cache_initialized:
                kv_tensor_size = int(
                    self.model_runner.get_single_kv_tensor_size())

            # get the size of total gpu memory
            total_gpu_memory = get_total_gpu_memory(self.rank)

            # Get runtime overhead from MemoryOverheadMonitor baseline
            # This was measured during determine_available_memory() via profile_run
            overhead_monitor = get_memory_overhead_monitor()
            runtime_overhead = 0
            if (overhead_monitor is not None
                    and overhead_monitor.baseline_overhead_bytes is not None):
                runtime_overhead = overhead_monitor.baseline_overhead_bytes

            return WorkerMemInfo(layer_size, kv_tensor_size,
                                 int(free_memory), int(total_gpu_memory),
                                 int(runtime_overhead))

        autoscaling_enabled = (
            self.vllm_config.dynamic_config.pipeline_autoscaling_enabled)
        fbgate = getattr(getattr(self, "model_runner", None), "fbgate", None)
        if autoscaling_enabled and fbgate is not None:
            gate = getattr(fbgate, "exclusive_background", fbgate.background)
            with gate():
                return collect_mem_info()
        return collect_mem_info()

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

    def get_async_migration_state(self) -> dict[str, Any]:
        with self._all_patch_applied_cv:
            patch_status = dict(self.is_all_patch_applied)
            all_patch_applied = all(patch_status.values())
        with self._sender_threads_cv:
            active_sender_threads = self._num_active_sender_threads
        with self.resizing_done_cv:
            resizing_done = self.resizing_done
        return {
            "rank": self.rank,
            "all_patch_applied": all_patch_applied,
            "patch_status": patch_status,
            "active_sender_threads": active_sender_threads,
            "receive_in_process": self.receive_in_process,
            "resizing_done": resizing_done,
            "kv_resizing_done": self.kv_resizing_done,
            "after_migration_total_token": self.after_migration_total_token,
            "after_migration_applied_token_num": (
                self.after_migration_applied_token_num),
        }

    def finish_idle_async_kv_cache_transfer(
        self,
        sender_list: list[int],
    ) -> None:
        """Finish async KV transfer when no sync batch can be scheduled.

        The normal autoscaling path sends the final ``kv_patch_finished`` marker
        from ``async_migration_after_execute_callback`` on a scheduler output
        with ``is_sync_after_migration=True``. If the workload has already
        drained, there is no next scheduler output. In that idle case, the full
        KV tensor snapshot is the last data transfer, so enqueue an empty
        finished patch for each active sender destination.
        """
        if self.rank not in set(sender_list):
            return

        target_ranks = list(self.rank_to_layers_ids.keys())
        if not target_ranks:
            logger.info(
                "[autoscaling sync] rank %s has no active async KV senders "
                "to finish in idle path", self.rank)
            return

        for dst_rank in target_ranks:
            self.dynamic_kv_synchronizer.add_new_tokens_to_kv_synchronizer(
                dst_rank,
                [],
                is_finished=True,
                num_total_new_tokens=0,
            )
        logger.info(
            "[autoscaling sync] rank %s enqueued idle async KV transfer "
            "finish markers for destinations %s", self.rank, target_ranks)

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
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
        start_layer = self._kv_cache_start_layer()
        local_layer_count = (len(runner.key_caches) if is_flexi
                             else len(runner.kv_caches))
        end_layer = start_layer + local_layer_count
        assert isinstance(runner.model, DynamicModelBase)
        logger.info(
            "start to compact kv cache for KV layers %s to %s "
            "(model layers %s to %s)", start_layer, end_layer,
            runner.model.model.start_layer, runner.model.model.end_layer)
        time_start = time.time()
        num_blocks = len(bitmap)
        if is_flexi and not runner.key_caches:
            logger.info("Skipping KV compaction on rank %s with no local KV caches",
                        self.rank)
            return
        if not is_flexi and not runner.kv_caches:
            logger.info("Skipping KV compaction on rank %s with no local KV caches",
                        self.rank)
            return
        if is_flexi:
            local_num_blocks = len(runner.key_caches[0])
            assert num_blocks == local_num_blocks, (
                "bitmap length mismatch: "
                f"num_blocks: {num_blocks} != "
                f"kv_cache_tensor_length: {local_num_blocks}")
        else:
            local_num_blocks = runner.kv_caches[0].size(1)
            assert num_blocks == local_num_blocks, (
                "bitmap length mismatch: "
                f"num_blocks: {num_blocks} != "
                f"kv_cache_tensor_length: {local_num_blocks}")

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
        use_direct = use_direct_ptr_for_runtime(self.vllm_config)
        
        # 🔴 FIX: Also track grouped_handles for combined_layers mode
        # When swapping block pointers, we must also swap their handles
        # Otherwise shrink will release wrong physical memory!
        use_combined_layers = hasattr(runner, 'grouped_handles') and len(runner.grouped_handles) > 0
        
        def _migrate_block_by_swapping_ptrs(old_block_id: int, new_block_id: int, migrate_record: dict[int, int]):
            assert len(tmp_key_cache) != 0
            for key_cache in tmp_key_cache:
                key_cache[new_block_id], key_cache[old_block_id] = key_cache[ old_block_id], key_cache[ new_block_id]
            for value_cache in tmp_value_cache:
                value_cache[ new_block_id], value_cache[ old_block_id] = value_cache[ old_block_id], value_cache[ new_block_id]
            # 🔴 FIX: Swap handles to keep them in sync with pointers
            # This is critical for combined_layers mode where handles are shared across layers
            if use_combined_layers:
                for group_handles in runner.grouped_handles:
                    if old_block_id < len(group_handles) and new_block_id < len(group_handles):
                        group_handles[new_block_id], group_handles[old_block_id] = \
                            group_handles[old_block_id], group_handles[new_block_id]
            migrate_record[old_block_id] = new_block_id

        if use_flexi_kv_for_runtime(runner.vllm_config):
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
                runner.input_batch.block_table.commit(runner.input_batch.num_reqs)
                logger.info(f"[timeline]: kv cache compaction within lock take {human_readable_duration(time.time() - time_start_within_lock)}")
                forward_lock_hold_ms = (time.time() - time_start_within_lock) * 1000
                if self.log_stop_time:
                    logger.info(f"[STOP_TIME][worker][compact_flexi][forward_lock]: hold={forward_lock_hold_ms:.2f}ms")
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
                runner.input_batch.block_table.commit(runner.input_batch.num_reqs)
                _sync_current_cuda_stream(self.device)
                logger.info(f"[timeline]: kv cache compaction within lock take {human_readable_duration(time.time() - time_start_within_lock)}")
                forward_lock_hold_ms = (time.time() - time_start_within_lock) * 1000
                if self.log_stop_time:
                    logger.info(f"[STOP_TIME][worker][compact_nonflexi][forward_lock]: hold={forward_lock_hold_ms:.2f}ms")
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
        _sync_current_cuda_stream(self.device)
        logger.info(f"[timeline]: compact kv cache total time taken: {human_readable_duration(time.time() - time_start)}")
    
    def _maybe_switch_block(self, scheduler_output: "DynamicSchedulerOutput") -> None:
        if (scheduler_output.current_scheduler_output_version >
                self.cur_scheduler_output_version and
                not self.migration_records_for_specific_scheduler_output_version):
            logger.info(
                "Worker %s has no local block remap records; advancing "
                "scheduler output version from %s to %s",
                self.rank, self.cur_scheduler_output_version,
                scheduler_output.current_scheduler_output_version)
            self.cur_scheduler_output_version = (
                scheduler_output.current_scheduler_output_version)
            return
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

    def start_resize_kv_cache_async(self, new_length: int) -> None:
        with self._async_resize_lock:
            if (self._async_resize_thread is not None
                    and self._async_resize_thread.is_alive()):
                if self._async_resize_target == new_length:
                    logger.info(
                        "[autoscaling async resize] rank %s resize to %s "
                        "already in progress", self.rank, new_length)
                    return
                raise RuntimeError(
                    f"rank {self.rank} already resizing KV cache to "
                    f"{self._async_resize_target}, cannot start {new_length}")

            self._async_resize_error = None
            self._async_resize_target = new_length
            with self.resizing_done_cv:
                self.resizing_done = False

            def _run_resize() -> None:
                try:
                    self._resize_kv_cache_impl(new_length)
                except BaseException as exc:
                    with self._async_resize_lock:
                        self._async_resize_error = exc
                    logger.exception(
                        "[autoscaling async resize] rank %s failed resizing "
                        "KV cache to %s", self.rank, new_length)
                finally:
                    with self.resizing_done_cv:
                        self.resizing_done = True
                        self.resizing_done_cv.notify_all()

            thread = threading.Thread(
                target=_run_resize,
                name=f"kv-resize-rank{self.rank}-to-{new_length}",
                daemon=True)
            self._async_resize_thread = thread
            thread.start()
            logger.info(
                "[autoscaling async resize] rank %s started background KV "
                "resize to %s", self.rank, new_length)

    def _wait_for_async_resize(self, reason: str) -> None:
        with self._async_resize_lock:
            thread = self._async_resize_thread
        if thread is not None and thread.is_alive():
            wait_start = time.time()
            logger.info(
                "[autoscaling async resize] rank %s waiting for background "
                "KV resize before %s", self.rank, reason)
            thread.join()
            logger.info(
                "[autoscaling async resize] rank %s waited %s before %s",
                self.rank,
                human_readable_duration(time.time() - wait_start),
                reason)
        with self._async_resize_lock:
            error = self._async_resize_error
            if error is not None:
                self._async_resize_error = None
                raise error

    def resize_kv_cache(self, new_length: int) -> None:
        self._wait_for_async_resize("synchronous KV resize")
        self._resize_kv_cache_impl(new_length)

    def _resize_kv_cache_impl(self, new_length: int) -> None:
        start_time = time.time()
        # Ensure correct CUDA device is set for this worker (important for Ray RPC calls)
        if self.device is not None:
            torch.cuda.set_device(self.device)
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
        old_length = self.block_num
        if new_length == old_length:
            logger.info(f"kv cache length is already {new_length}, no need to resize, sleep 2 seconds")
            return
        if is_flexi and not self.model_runner.key_caches:
            logger.info("Skipping KV resize on rank %s with no local KV caches; "
                        "recording block_num=%s", self.rank, new_length)
            self.block_num = new_length
            self.dynamic_kv_synchronizer.create_slot_mappings(
                new_length * self.block_size)
            return
        if not is_flexi and not self.model_runner.kv_caches:
            logger.info("Skipping KV resize on rank %s with no local KV caches; "
                        "recording block_num=%s", self.rank, new_length)
            self.block_num = new_length
            self.dynamic_kv_synchronizer.create_slot_mappings(
                new_length * self.block_size)
            return
        
        # Check overhead BEFORE operation (if monitoring enabled)
        if not self.vllm_config.dynamic_config.disable_memory_overhead_monitor:
            layer_count, kv_cache_bytes_before = self._get_current_memory_params()
            overhead_monitor = get_memory_overhead_monitor()
            # For resize, also track expected memory change using self.per_block_kv_cache_bytes
            expected_kv_change_bytes = (new_length - old_length) * self.per_block_kv_cache_bytes * layer_count
            overhead_monitor.check_overhead_before(
                operation=f"resize_kv_{old_length}_to_{new_length}",
                current_layer_count=layer_count,
                current_kv_cache_bytes=kv_cache_bytes_before,
                expected_kv_change_bytes=expected_kv_change_bytes
            )
        
        self.block_num = new_length
        if is_flexi:
            self._flexi_resize_kv_cache(new_length)
        else:
            self._resize_kv_cache(new_length)

        # Check overhead AFTER operation (if monitoring enabled)
        if not self.vllm_config.dynamic_config.disable_memory_overhead_monitor:
            layer_count_after, kv_cache_bytes_after = self._get_current_memory_params()
            overhead_monitor = get_memory_overhead_monitor()
            overhead_monitor.check_overhead_after(
                operation=f"resize_kv_{old_length}_to_{new_length}",
                new_layer_count=layer_count_after,
                new_kv_cache_bytes=kv_cache_bytes_after
            )
        
        self.dynamic_kv_synchronizer.create_slot_mappings(new_length * self.block_size)
        logger.info(f"[timeline]: total resize kv cache time taken: {human_readable_duration(time.time() - start_time)}")

    def _resize_kv_cache(self, new_length: int) -> None:
        runner = self.model_runner
        assert new_length > 0
        forward_lock_start = time.time()
        with runner.forward_lock:
            time_start = time.time()
            assert isinstance(runner.model, DynamicModelBase)
            logger.info(f"resizing kv cache from {len(runner.kv_caches[0][0])} to {new_length}")
            forward_context = self.vllm_config.compilation_config.static_forward_context
            kv, kv_length, T, H, Dh = runner.kv_caches[0].shape
            logger.info(f"num of kv tensors{len(runner.kv_caches)}")
            start_layer = self._kv_cache_start_layer()

            for layer_name, attn_module in forward_context.items():
                _sync_current_cuda_stream(self.device)
                gc.collect()
                logger.info(f"rresizing kv cache for laye {layer_name}")
                layer_idx = extract_layer_index(layer_name)
                idx = layer_idx - start_layer
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
                    start_layer,
                    start_layer + len(runner.kv_caches),
                    forward_context,
                    self.dynamic_kv_synchronizer,
                    runner,
                    tmp_cache
                )
        
        forward_lock_hold_ms = (time.time() - forward_lock_start) * 1000
        if self.log_stop_time:
            logger.info(f"[STOP_TIME][worker][resize_nonflexi][forward_lock]: hold={forward_lock_hold_ms:.2f}ms")
        _sync_current_cuda_stream(self.device)
        gc.collect()
        logger.info(f"[timeline]: resize kv cache within: {human_readable_duration(time.time() - time_start)}")

    # =========================================================================
    # Flexi KV Cache Resize - Refactored Helper Methods
    # =========================================================================
    
    def _get_flexi_resize_context(self) -> tuple:
        """Get common context needed for flexi resize operations.
        
        Returns:
            Tuple of (forward_context, sorted_forward_context, cache_shape, 
                     start_layer, end_layer, use_direct_ptr, layer_group_granularity,
                     num_local_layers, use_combined_layers)
        """
        forward_context = self.vllm_config.compilation_config.static_forward_context
        sorted_forward_context = sorted(forward_context.items(), key=lambda x: extract_layer_index(x[0]))
        
        T, H, Dh = self.model_runner.page_meta.shape
        cache_shape = (T, H, Dh)
        
        start_layer = self._kv_cache_start_layer()
        end_layer = start_layer + len(self.model_runner.key_caches)
        use_direct_ptr = use_direct_ptr_for_runtime(self.vllm_config)
        
        layer_group_granularity = getattr(self.model_runner, 'layer_group_granularity', 1)
        num_local_layers = len(self.model_runner.key_caches)
        
        # Validate combined_layers mode
        use_combined_layers = False
        if layer_group_granularity > 1 and hasattr(self.model_runner, 'grouped_handles') and len(self.model_runner.grouped_handles) > 0:
            expected_layers = len(self.model_runner.grouped_handles) * layer_group_granularity
            if expected_layers == num_local_layers:
                use_combined_layers = True
            else:
                logger.warning(
                    f"[resize_kv_cache] Skipping combined_layers mode: grouped_handles implies "
                    f"{expected_layers} layers, but key_caches has {num_local_layers} layers. "
                    f"Falling back to per-layer allocation."
                )
        
        return (forward_context, sorted_forward_context, cache_shape, 
                start_layer, end_layer, use_direct_ptr, layer_group_granularity,
                num_local_layers, use_combined_layers)
    
    def _log_memory_metrics(self, operation: str, blocks_count: int, num_layers: int, 
                            cache_shape: tuple, expected_bytes: int, actual_bytes: int,
                            use_vmm: bool, use_combined_layers: bool) -> None:
        """Log memory monitoring metrics for resize operations."""
        T, H, Dh = cache_shape
        from vllm.utils import get_kv_cache_torch_dtype
        kv_dtype = get_kv_cache_torch_dtype(self.model_runner.kv_cache_dtype, self.model_runner.model_config.dtype)
        bytes_per_block = T * H * Dh * kv_dtype.itemsize
        
        if operation == "SHRINK":
            diff_label = "leak"
            diff_value = expected_bytes - actual_bytes
        else:  # GROW
            diff_label = "overhead"
            diff_value = actual_bytes - expected_bytes
        
        logger.info(
            f"[Memory Monitor {operation}] rank={self.rank}, "
            f"blocks_{operation.lower()}ed={blocks_count}, layers={num_layers}, "
            f"block_shape=({T},{H},{Dh}), bytes_per_block={bytes_per_block}, "
            f"expected_{'freed' if operation == 'SHRINK' else 'allocated'}_MB={expected_bytes / 1024**2:.2f}, "
            f"actual_{'freed' if operation == 'SHRINK' else 'allocated'}_MB={actual_bytes / 1024**2:.2f}, "
            f"{diff_label}_MB={diff_value / 1024**2:.2f}, "
            f"use_vmm={use_vmm}, use_combined_layers={use_combined_layers}"
        )
    
    def _do_flexi_shrink(self, new_length: int, cache_length: int, ctx: tuple) -> tuple:
        """Execute the shrink phase of flexi KV cache resize.
        
        Returns:
            Tuple of (time_before_in_lock, time_after_in_lock, time_after_free)
        """
        (forward_context, sorted_forward_context, cache_shape, 
         start_layer, end_layer, use_direct_ptr, layer_group_granularity,
         num_local_layers, use_combined_layers) = ctx
        T, H, Dh = cache_shape
        
        # Memory monitoring: capture before state. Keep the dependency local to
        # the current stream instead of draining the whole device.
        _sync_current_cuda_stream(self.device)
        gc.collect()
        mem_before = torch.cuda.mem_get_info()
        gpu_free_before, gpu_total = mem_before
        gpu_used_before = gpu_total - gpu_free_before
        
        use_vmm = self.model_runner.vmm_aligned_bytes > 0
        
        # Track old grouped_handles for combined_layers mode
        # FIXED: Collect handles WITH their corresponding base_k_ptrs to ensure correct pairing
        old_grouped_handles: list[tuple[list[int], list[int]]] = []  # (handles, base_k_ptrs) tuples
        if use_combined_layers and hasattr(self.model_runner, 'grouped_handles'):
            total_handles_to_free = 0
            granularity = layer_group_granularity
            logger.info(f"[Shrink DEBUG] Collecting old_grouped_handles: "
                       f"model_runner.grouped_handles has {len(self.model_runner.grouped_handles)} groups, "
                       f"new_length={new_length}, cache_length={cache_length}")
            for group_idx, group_handles in enumerate(self.model_runner.grouped_handles):
                handles_len = len(group_handles)
                if handles_len > new_length:
                    to_free_handles = group_handles[new_length:]
                    # Get the corresponding base_k_ptrs for this group from the current key_caches
                    # CRITICAL: Get VAs BEFORE they are modified
                    first_layer_in_group = group_idx * granularity
                    if first_layer_in_group < len(self.model_runner.key_caches):
                        # Get the VAs that will be freed (indices new_length:handles_len)
                        to_free_k_ptrs = self.model_runner.key_caches[first_layer_in_group][new_length:new_length + len(to_free_handles)]
                        old_grouped_handles.append((to_free_handles, to_free_k_ptrs))
                        total_handles_to_free += len(to_free_handles)
                        logger.info(f"[Shrink DEBUG] group {group_idx}: handles_len={handles_len}, "
                                   f"freeing {len(to_free_handles)} handles (indices {new_length}:{handles_len}), "
                                   f"VAs from layer {first_layer_in_group}")
                    else:
                        logger.warning(f"[Shrink DEBUG] group {group_idx}: first_layer_in_group={first_layer_in_group} "
                                      f">= key_caches len={len(self.model_runner.key_caches)}")
                        old_grouped_handles.append(([], []))
                else:
                    old_grouped_handles.append(([], []))
                    logger.info(f"[Shrink DEBUG] group {group_idx}: handles_len={handles_len} <= new_length={new_length}, nothing to free")
            logger.info(f"[Shrink DEBUG] Total handles to free across all groups: {total_handles_to_free}")
        
        # Temporary storage for new and old caches
        tmp_key_cache_list = []
        tmp_value_cache_list = []
        tmp_key_cache_ptr_list = []
        tmp_value_cache_ptr_list = []
        tmp_new_key_ptr_tensors: list[Optional[torch.Tensor]] = []
        tmp_new_value_ptr_tensors: list[Optional[torch.Tensor]] = []
        
        tmp_old_key_cache_ptr_list = []
        tmp_old_value_cache_ptr_list = []
        tmp_old_key_cache_list = []
        tmp_old_value_cache_list = []
        tmp_old_key_handles_list: list[tuple] = []
        tmp_old_value_handles_list: list[tuple] = []
        tmp_new_key_handles_list: list[list[int]] = []
        tmp_new_value_handles_list: list[list[int]] = []
        
        # Phase 1: Prepare new cache references for each layer
        for layer_name, _ in sorted_forward_context:
            logger.info(f"resizing kv cache for layer {layer_name}")
            idx = extract_layer_index(layer_name)
            local_idx = idx - start_layer
            
            # Track old pointers and caches
            old_k_ptrs = self.model_runner.key_cache_ptrs[local_idx]
            old_v_ptrs = self.model_runner.value_cache_ptrs[local_idx]
            old_key_cache = self.model_runner.key_caches[local_idx][new_length:]
            old_value_cache = self.model_runner.value_caches[local_idx][new_length:]
            
            # Handle VMM tracking
            handles_len = len(self.model_runner.key_handles[local_idx]) if (
                self.model_runner.key_handles and local_idx < len(self.model_runner.key_handles)
            ) else 0
            cache_len = len(self.model_runner.key_caches[local_idx])
            
            if use_vmm and handles_len > 0:
                if handles_len < cache_len:
                    logger.warning(f"[VMM shrink] layer {layer_name}: handles_len={handles_len} < cache_len={cache_len}")
                
                if handles_len > new_length:
                    old_key_handles = self.model_runner.key_handles[local_idx][new_length:handles_len]
                    old_value_handles = self.model_runner.value_handles[local_idx][new_length:handles_len] if self.model_runner.value_handles else []
                    new_key_handles = self.model_runner.key_handles[local_idx][:min(new_length, handles_len)]
                    new_value_handles = self.model_runner.value_handles[local_idx][:min(new_length, handles_len)] if self.model_runner.value_handles else []
                    vmm_free_count = len(old_key_handles)
                    old_key_cache_for_vmm = self.model_runner.key_caches[local_idx][new_length:new_length + vmm_free_count]
                    old_value_cache_for_vmm = self.model_runner.value_caches[local_idx][new_length:new_length + vmm_free_count]
                else:
                    old_key_handles, old_value_handles = [], []
                    new_key_handles = self.model_runner.key_handles[local_idx][:handles_len]
                    new_value_handles = self.model_runner.value_handles[local_idx][:handles_len] if self.model_runner.value_handles else []
                    old_key_cache_for_vmm, old_value_cache_for_vmm = [], []
            else:
                old_key_handles, old_value_handles = [], []
                new_key_handles, new_value_handles = [], []
                old_key_cache_for_vmm, old_value_cache_for_vmm = [], []
            
            tmp_old_key_cache_ptr_list.append(old_k_ptrs)
            tmp_old_value_cache_ptr_list.append(old_v_ptrs)
            tmp_old_key_cache_list.append(old_key_cache)
            tmp_old_value_cache_list.append(old_value_cache)
            tmp_old_key_handles_list.append((old_key_cache_for_vmm, old_key_handles))
            tmp_old_value_handles_list.append((old_value_cache_for_vmm, old_value_handles))
            tmp_new_key_handles_list.append(new_key_handles)
            tmp_new_value_handles_list.append(new_value_handles)
            
            # Prepare new cache references
            new_key_cache = self.model_runner.key_caches[local_idx][:new_length]
            new_value_cache = self.model_runner.value_caches[local_idx][:new_length]
            
            with self.model_runner.fbgate.background():
                new_key_ptrs, new_value_ptrs = kv_allocator.prepare_flexi_kv_ptrs(new_key_cache, new_value_cache)
                logger.info(f"resizing for kv cache, new_key_ptrs: {hex(new_key_ptrs)}, new_value_ptrs: {hex(new_value_ptrs)}")
                if use_direct_ptr:
                    new_key_ptr_tensor = create_ptr_tensor_from_list(new_key_cache, self.model_runner.device)
                    new_value_ptr_tensor = create_ptr_tensor_from_list(new_value_cache, self.model_runner.device)
                else:
                    new_key_ptr_tensor, new_value_ptr_tensor = None, None
            
            tmp_key_cache_list.append(new_key_cache)
            tmp_value_cache_list.append(new_value_cache)
            tmp_key_cache_ptr_list.append(new_key_ptrs)
            tmp_value_cache_ptr_list.append(new_value_ptrs)
            tmp_new_key_ptr_tensors.append(new_key_ptr_tensor)
            tmp_new_value_ptr_tensors.append(new_value_ptr_tensor)
        
        _sync_current_cuda_stream(self.device)
        time_before_in_lock = time.time()
        
        # Phase 2: Bind new caches under lock
        with self.model_runner.forward_lock:
            for layer_name, _ in sorted_forward_context:
                logger.info(f"resizing kv cache for layer {layer_name}")
                idx = extract_layer_index(layer_name)
                local_idx = idx - start_layer
                
                before_bind_time = time.time()
                dynamic_flexi_bind_single_kv_cache(
                    start_layer, end_layer, idx,
                    tmp_key_cache_list[local_idx], tmp_value_cache_list[local_idx],
                    tmp_key_cache_ptr_list[local_idx], tmp_value_cache_ptr_list[local_idx],
                    tmp_new_key_ptr_tensors[local_idx], tmp_new_value_ptr_tensors[local_idx],
                    forward_context, self.dynamic_kv_synchronizer, self.model_runner
                )
                logger.info(f"[timeline]: bind single kv cache for layer {layer_name} take {human_readable_duration(time.time() - before_bind_time)}")
                
                if not use_combined_layers and self.model_runner.key_handles:
                    self.model_runner.key_handles[local_idx] = tmp_new_key_handles_list[local_idx]
                    self.model_runner.value_handles[local_idx] = tmp_new_value_handles_list[local_idx]
            
            if use_combined_layers:
                for group_idx in range(len(self.model_runner.grouped_handles)):
                    old_handles = self.model_runner.grouped_handles[group_idx]
                    if len(old_handles) > new_length:
                        self.model_runner.grouped_handles[group_idx] = old_handles[:new_length]
            
            if use_direct_ptr:
                self.model_runner.update_kv_ptr_tensor(tmp_new_key_ptr_tensors, tmp_new_value_ptr_tensors)
        
        time_after_in_lock = time.time()
        forward_lock_hold_ms = (time_after_in_lock - time_before_in_lock) * 1000
        if self.log_stop_time:
            logger.info(f"[STOP_TIME][worker][resize_shrink][forward_lock]: hold={forward_lock_hold_ms:.2f}ms")
        
        with self.resizing_done_cv:
            self.resizing_done = True
            self.resizing_done_cv.notify_all()
        
        # Phase 3: Free old resources with detailed memory tracking
        assert self.device is not None
        
        # Track memory at each step using stream-local synchronization.
        _sync_current_cuda_stream(self.device)
        mem_step0 = torch.cuda.mem_get_info()
        used_step0 = (gpu_total - mem_step0[0]) / 1024**2  # MB
        
        # Step 1: Free old pointer arrays
        for old_key_ptr, old_value_ptr in zip(tmp_old_key_cache_ptr_list, tmp_old_value_cache_ptr_list):
            with self.model_runner.fbgate.background():
                kv_allocator.free_page_list(old_key_ptr, self.device)
                kv_allocator.free_page_list(old_value_ptr, self.device)
        
        _sync_current_cuda_stream(self.device)
        mem_step1 = torch.cuda.mem_get_info()
        used_step1 = (gpu_total - mem_step1[0]) / 1024**2
        freed_step1 = used_step0 - used_step1

        # Step 2: Free VMM blocks
        self._free_shrunk_caches(
            use_vmm, use_combined_layers, old_grouped_handles,  # old_grouped_handles now contains (handles, base_k_ptrs) tuples
            tmp_old_key_cache_list, tmp_old_value_cache_list,
            tmp_old_key_handles_list, tmp_old_value_handles_list,
            layer_group_granularity
        )
        
        _sync_current_cuda_stream(self.device)
        mem_step2 = torch.cuda.mem_get_info()
        used_step2 = (gpu_total - mem_step2[0]) / 1024**2
        freed_step2 = used_step1 - used_step2
        
        logger.info(f"[SHRINK Memory Steps] rank={self.rank}, "
                   f"step0_used={used_step0:.2f}MB, "
                   f"step1_freed_ptrs={freed_step1:.2f}MB, "
                   f"step2_freed_vmm={freed_step2:.2f}MB, "
                   f"total_freed={used_step0 - used_step2:.2f}MB")
        
        time_after_free = time.time()
        
        # Memory monitoring: calculate metrics
        gc.collect()
        mem_after = mem_step2  # Use already captured value
        gpu_used_after = gpu_total - mem_after[0]
        actual_freed = gpu_used_before - gpu_used_after
        
        from vllm.utils import get_kv_cache_torch_dtype
        kv_dtype = get_kv_cache_torch_dtype(self.model_runner.kv_cache_dtype, self.model_runner.model_config.dtype)
        bytes_per_block = T * H * Dh * kv_dtype.itemsize
        blocks_freed = cache_length - new_length
        expected_freed = blocks_freed * bytes_per_block * num_local_layers * 2
        
        self._log_memory_metrics("SHRINK", blocks_freed, num_local_layers, cache_shape,
                                  expected_freed, actual_freed, use_vmm, use_combined_layers)
        
        return time_before_in_lock, time_after_in_lock, time_after_free
    
    def _free_shrunk_caches(self, use_vmm: bool, use_combined_layers: bool, 
                            old_grouped_handles: list[tuple[list[int], list[int]]], tmp_old_key_cache_list: list,
                            tmp_old_value_cache_list: list, tmp_old_key_handles_list: list,
                            tmp_old_value_handles_list: list, layer_group_granularity: int) -> None:
        """Free old KV caches after shrinking.
        
        Args:
            old_grouped_handles: For combined_layers mode, list of (handles, base_k_ptrs) tuples
                                 to ensure correct pairing during memory release.
        """
        if use_vmm:
            aligned_bytes = self.model_runner.vmm_aligned_bytes
            combined_mode = getattr(self.model_runner, 'vmm_combined_mode', False)
            vmm_freed_layers = 0
            
            if use_combined_layers:
                bytes_per_kv = getattr(self.model_runner, 'vmm_bytes_per_kv', 0)
                total_blocks_freed = 0
                groups_freed = 0
                groups_skipped_empty = 0
                
                logger.info(f"[VMM combined layers DEBUG] Starting free: "
                           f"old_grouped_handles size={len(old_grouped_handles)}, "
                           f"tmp_old_key_cache_list size={len(tmp_old_key_cache_list)}, "
                           f"layer_group_granularity={layer_group_granularity}")
                
                # FIXED: Use pre-bound (handles, base_k_ptrs) tuples
                for group_idx, (group_handles, base_k_ptrs) in enumerate(old_grouped_handles):
                    if not group_handles or not base_k_ptrs:
                        groups_skipped_empty += 1
                        continue
                    
                    logger.info(f"[VMM combined layers] Freeing group {group_idx}: "
                               f"handles={len(group_handles)}, base_k_ptrs={len(base_k_ptrs)}, "
                               f"first_handle={group_handles[0] if group_handles else 'N/A'}, "
                               f"first_va={hex(base_k_ptrs[0]) if base_k_ptrs else 'N/A'}")
                    with self.model_runner.fbgate.background():
                        kv_allocator.free_vmm_blocks_combined_layers(
                            base_k_ptrs, group_handles, aligned_bytes, self.device,
                            layer_group_granularity, bytes_per_kv
                        )
                    total_blocks_freed += len(group_handles)
                    groups_freed += 1
                
                logger.info(f"[VMM combined layers] Freed {total_blocks_freed} blocks from {groups_freed}/{len(old_grouped_handles)} groups "
                           f"(skipped: empty={groups_skipped_empty})")
            else:
                for (old_key_cache_vmm, old_key_handles), (old_value_cache_vmm, old_value_handles) in zip(
                    tmp_old_key_handles_list, tmp_old_value_handles_list
                ):
                    with self.model_runner.fbgate.background():
                        if combined_mode:
                            if old_key_handles and old_key_cache_vmm:
                                kv_allocator.free_vmm_blocks_combined(old_key_cache_vmm, old_key_handles, aligned_bytes, self.device)
                                vmm_freed_layers += 1
                        else:
                            if old_key_handles and old_key_cache_vmm:
                                kv_allocator.free_vmm_blocks(old_key_cache_vmm, old_key_handles, aligned_bytes, self.device)
                                vmm_freed_layers += 1
                            if old_value_handles and old_value_cache_vmm:
                                kv_allocator.free_vmm_blocks(old_value_cache_vmm, old_value_handles, aligned_bytes, self.device)
                logger.info(f"[VMM] Freed {vmm_freed_layers} layers, aligned_bytes={aligned_bytes}, combined_mode={combined_mode}")
        else:
            for old_key_cache, old_value_cache in zip(tmp_old_key_cache_list, tmp_old_value_cache_list):
                with self.model_runner.fbgate.background():
                    kv_allocator.free_cache(old_key_cache, self.device)
                    kv_allocator.free_cache(old_value_cache, self.device)
            logger.info(f"[cudaFreeAsync] Freed {len(tmp_old_key_cache_list)} layers")
    
    def _do_flexi_grow(self, new_length: int, cache_length: int, ctx: tuple) -> tuple:
        """Execute the grow phase of flexi KV cache resize.
        
        Returns:
            Tuple of (time_before_in_lock, time_after_in_lock, time_after_free)
        """
        (forward_context, sorted_forward_context, cache_shape, 
         start_layer, end_layer, use_direct_ptr, layer_group_granularity,
         num_local_layers, use_combined_layers) = ctx
        T, H, Dh = cache_shape
        
        # Memory monitoring: capture before state. This is stream-local so
        # inference work on other streams is not drained.
        _sync_current_cuda_stream(self.device)
        mem_before = torch.cuda.mem_get_info()
        gpu_free_before, gpu_total = mem_before
        gpu_used_before = gpu_total - gpu_free_before
        
        use_vmm = self.model_runner.vmm_aligned_bytes > 0
        new_allocated_block_num = new_length - cache_length
        
        # Temporary storage
        tmp_new_allocated_key_cache = []
        tmp_new_allocated_value_cache = []
        tmp_new_key_ptr_tensors: list[Optional[torch.Tensor]] = []
        tmp_new_value_ptr_tensors: list[Optional[torch.Tensor]] = []
        tmp_key_cache_ptr_list = []
        tmp_value_cache_ptr_list = []
        tmp_new_key_handles: list[list[int]] = []
        tmp_new_value_handles: list[list[int]] = []
        tmp_old_key_cache_ptr_list = []
        tmp_old_value_cache_ptr_list = []
        new_grouped_handles: list[list[int]] = []
        
        if use_combined_layers:
            self._grow_combined_layers_mode(
                new_allocated_block_num, cache_shape, use_direct_ptr, layer_group_granularity,
                tmp_new_allocated_key_cache, tmp_new_allocated_value_cache,
                tmp_new_key_ptr_tensors, tmp_new_value_ptr_tensors,
                tmp_key_cache_ptr_list, tmp_value_cache_ptr_list,
                tmp_new_key_handles, tmp_new_value_handles,
                tmp_old_key_cache_ptr_list, tmp_old_value_cache_ptr_list,
                new_grouped_handles
            )
        else:
            self._grow_per_layer_mode(
                new_allocated_block_num, cache_length, cache_shape, sorted_forward_context,
                use_vmm, use_direct_ptr,
                tmp_new_allocated_key_cache, tmp_new_allocated_value_cache,
                tmp_new_key_ptr_tensors, tmp_new_value_ptr_tensors,
                tmp_key_cache_ptr_list, tmp_value_cache_ptr_list,
                tmp_new_key_handles, tmp_new_value_handles,
                tmp_old_key_cache_ptr_list, tmp_old_value_cache_ptr_list
            )
        
        time_before_in_lock = time.time()
        _sync_current_cuda_stream(self.device)
        
        # Bind new caches under lock
        with self.model_runner.forward_lock:
            new_key_ptr_tensor_list = []
            new_value_ptr_tensor_list = []
            
            for layer_name, attn_module in sorted_forward_context:
                idx = extract_layer_index(layer_name)
                local_idx = idx - start_layer
                
                if local_idx < 0 or local_idx >= len(tmp_new_allocated_key_cache):
                    raise IndexError(
                        f"[flexi_resize_kv_cache] Invalid local_idx={local_idx} for layer {layer_name}. "
                        f"use_combined_layers={use_combined_layers}, grouped_handles={len(self.model_runner.grouped_handles) if hasattr(self.model_runner, 'grouped_handles') else 'N/A'}"
                    )
                
                dynamic_flexi_bind_single_kv_cache(
                    start_layer, end_layer, idx,
                    tmp_new_allocated_key_cache[local_idx], tmp_new_allocated_value_cache[local_idx],
                    tmp_key_cache_ptr_list[local_idx], tmp_value_cache_ptr_list[local_idx],
                    tmp_new_key_ptr_tensors[local_idx], tmp_new_value_ptr_tensors[local_idx],
                    forward_context, self.dynamic_kv_synchronizer, self.model_runner
                )
                new_key_ptr_tensor_list.append(tmp_new_key_ptr_tensors[local_idx])
                new_value_ptr_tensor_list.append(tmp_new_value_ptr_tensors[local_idx])
                
                if not use_combined_layers:
                    if len(self.model_runner.key_handles) <= local_idx:
                        self.model_runner.key_handles.extend([[] for _ in range(local_idx + 1 - len(self.model_runner.key_handles))])
                        self.model_runner.value_handles.extend([[] for _ in range(local_idx + 1 - len(self.model_runner.value_handles))])
                    self.model_runner.key_handles[local_idx] = tmp_new_key_handles[local_idx]
                    self.model_runner.value_handles[local_idx] = tmp_new_value_handles[local_idx]
            
            if use_combined_layers and new_grouped_handles:
                for group_idx, new_handles in enumerate(new_grouped_handles):
                    if group_idx < len(self.model_runner.grouped_handles):
                        self.model_runner.grouped_handles[group_idx].extend(new_handles)
                    else:
                        self.model_runner.grouped_handles.append(new_handles)
            
            if use_direct_ptr:
                self.model_runner.update_kv_ptr_tensor(new_key_ptr_tensor_list, new_value_ptr_tensor_list)
        
        time_after_in_lock = time.time()
        forward_lock_hold_ms = (time_after_in_lock - time_before_in_lock) * 1000
        if self.log_stop_time:
            logger.info(f"[STOP_TIME][worker][resize_grow][forward_lock]: hold={forward_lock_hold_ms:.2f}ms")
        
        with self.resizing_done_cv:
            self.resizing_done = True
            self.resizing_done_cv.notify_all()
        
        # Free old pointer arrays
        for old_key_ptr, old_value_ptr in zip(tmp_old_key_cache_ptr_list, tmp_old_value_cache_ptr_list):
            with self.model_runner.fbgate.background():
                kv_allocator.free_page_list(old_key_ptr, self.model_runner.device)
                kv_allocator.free_page_list(old_value_ptr, self.model_runner.device)
        
        time_after_free = time.time()
        
        # Memory monitoring: calculate metrics.
        _sync_current_cuda_stream(self.device)
        mem_after = torch.cuda.mem_get_info()
        gpu_used_after = gpu_total - mem_after[0]
        actual_allocated = gpu_used_after - gpu_used_before
        
        from vllm.utils import get_kv_cache_torch_dtype
        kv_dtype = get_kv_cache_torch_dtype(self.model_runner.kv_cache_dtype, self.model_runner.model_config.dtype)
        bytes_per_block = T * H * Dh * kv_dtype.itemsize
        expected_allocated = new_allocated_block_num * bytes_per_block * num_local_layers * 2
        
        self._log_memory_metrics("GROW", new_allocated_block_num, num_local_layers, cache_shape,
                                  expected_allocated, actual_allocated, use_vmm, use_combined_layers)
        
        return time_before_in_lock, time_after_in_lock, time_after_free
    
    def _grow_combined_layers_mode(
        self, new_allocated_block_num: int, cache_shape: tuple, use_direct_ptr: bool,
        layer_group_granularity: int,
        tmp_new_allocated_key_cache: list, tmp_new_allocated_value_cache: list,
        tmp_new_key_ptr_tensors: list, tmp_new_value_ptr_tensors: list,
        tmp_key_cache_ptr_list: list, tmp_value_cache_ptr_list: list,
        tmp_new_key_handles: list, tmp_new_value_handles: list,
        tmp_old_key_cache_ptr_list: list, tmp_old_value_cache_ptr_list: list,
        new_grouped_handles: list
    ) -> None:
        """Grow KV cache using combined layers mode."""
        num_groups = len(self.model_runner.grouped_handles)
        
        for group_idx in range(num_groups):
            vmm_result = kv_allocator.allocate_with_cuda_vmm_combined_layers(
                new_allocated_block_num, list(cache_shape),
                self.model_runner.kv_cache_dtype, self.model_runner.device,
                layer_group_granularity
            )
            
            k_ptrs_per_layer, v_ptrs_per_layer = vmm_result[0], vmm_result[1]
            k_ptrs_dev_per_layer, v_ptrs_dev_per_layer = vmm_result[2], vmm_result[3]
            aligned_bytes, new_bytes_per_kv = vmm_result[4], vmm_result[5]
            new_handles = list(vmm_result[6])
            
            self.model_runner.vmm_aligned_bytes = aligned_bytes
            self.model_runner.vmm_bytes_per_kv = new_bytes_per_kv
            new_grouped_handles.append(new_handles)
            
            for layer_in_group in range(layer_group_granularity):
                global_layer_idx = group_idx * layer_group_granularity + layer_in_group
                if global_layer_idx < len(self.model_runner.key_caches):
                    old_key_cache = self.model_runner.key_caches[global_layer_idx]
                    old_value_cache = self.model_runner.value_caches[global_layer_idx]
                    new_key_cache = old_key_cache + list(k_ptrs_per_layer[layer_in_group])
                    new_value_cache = old_value_cache + list(v_ptrs_per_layer[layer_in_group])
                    
                    tmp_old_key_cache_ptr_list.append(self.model_runner.key_cache_ptrs[global_layer_idx])
                    tmp_old_value_cache_ptr_list.append(self.model_runner.value_cache_ptrs[global_layer_idx])
                    
                    with self.model_runner.fbgate.background():
                        new_k_ptrs, new_v_ptrs = kv_allocator.prepare_flexi_kv_ptrs(new_key_cache, new_value_cache)
                        if use_direct_ptr:
                            new_key_ptr_tensor = create_ptr_tensor_from_list(new_key_cache, self.model_runner.device)
                            new_value_ptr_tensor = create_ptr_tensor_from_list(new_value_cache, self.model_runner.device)
                        else:
                            new_key_ptr_tensor, new_value_ptr_tensor = None, None
                    
                    tmp_new_allocated_key_cache.append(new_key_cache)
                    tmp_new_allocated_value_cache.append(new_value_cache)
                    tmp_new_key_ptr_tensors.append(new_key_ptr_tensor)
                    tmp_new_value_ptr_tensors.append(new_value_ptr_tensor)
                    tmp_key_cache_ptr_list.append(new_k_ptrs)
                    tmp_value_cache_ptr_list.append(new_v_ptrs)
                    tmp_new_key_handles.append([])
                    tmp_new_value_handles.append([])
        
        logger.info(f"[resize_kv_cache] combined_layers mode: allocated {new_allocated_block_num} blocks for {num_groups} groups")
    
    def _grow_per_layer_mode(
        self, new_allocated_block_num: int, cache_length: int, cache_shape: tuple,
        sorted_forward_context: list, use_vmm: bool, use_direct_ptr: bool,
        tmp_new_allocated_key_cache: list, tmp_new_allocated_value_cache: list,
        tmp_new_key_ptr_tensors: list, tmp_new_value_ptr_tensors: list,
        tmp_key_cache_ptr_list: list, tmp_value_cache_ptr_list: list,
        tmp_new_key_handles: list, tmp_new_value_handles: list,
        tmp_old_key_cache_ptr_list: list, tmp_old_value_cache_ptr_list: list
    ) -> None:
        """Grow KV cache using per-layer allocation mode."""
        for layer_name, attn_module in sorted_forward_context:
            local_idx = extract_layer_index(layer_name) - self._kv_cache_start_layer()
            logger.info(f"[resize_kv_cache debug] layer_name={layer_name}, local_idx={local_idx}")
            
            key_cache = self.model_runner.key_caches[local_idx]
            value_cache = self.model_runner.value_caches[local_idx]
            
            old_key_handles = self.model_runner.key_handles[local_idx] if (
                self.model_runner.key_handles and local_idx < len(self.model_runner.key_handles)
            ) else []
            old_value_handles = self.model_runner.value_handles[local_idx] if (
                self.model_runner.value_handles and local_idx < len(self.model_runner.value_handles)
            ) else []
            
            if use_vmm:
                combined_mode = getattr(self.model_runner, 'vmm_combined_mode', False)
                if combined_mode:
                    vmm_result = kv_allocator.allocate_with_cuda_vmm_combined(
                        new_allocated_block_num, list(cache_shape),
                        self.model_runner.kv_cache_dtype, self.model_runner.device
                    )
                    new_allocated_key_cache = vmm_result[0]
                    new_allocated_value_cache = vmm_result[1]
                    new_allocated_key_handles = vmm_result[6]
                    new_allocated_value_handles = []
                    self.model_runner.vmm_aligned_bytes = vmm_result[4]
                else:
                    vmm_result = kv_allocator.allocate_with_cuda_vmm(
                        new_allocated_block_num, list(cache_shape),
                        self.model_runner.kv_cache_dtype, self.model_runner.device
                    )
                    new_allocated_key_cache = vmm_result[0]
                    new_allocated_value_cache = vmm_result[1]
                    new_allocated_key_handles = vmm_result[5]
                    new_allocated_value_handles = vmm_result[6]
                    self.model_runner.vmm_aligned_bytes = vmm_result[4]
            else:
                result = kv_allocator.allocate_with_cuda_async(
                    new_allocated_block_num, list(cache_shape),
                    self.model_runner.kv_cache_dtype, self.model_runner.device
                )
                new_allocated_key_cache, new_allocated_value_cache = result[0], result[1]
                new_allocated_key_handles, new_allocated_value_handles = [], []
            
            new_key_cache = key_cache + new_allocated_key_cache
            new_value_cache = value_cache + new_allocated_value_cache
            new_key_handles = old_key_handles + list(new_allocated_key_handles)
            new_value_handles = old_value_handles + list(new_allocated_value_handles)
            
            logger.info(f"[resize_kv_cache] use_vmm={use_vmm}, new_key_cache len={len(new_key_cache)}, handles len={len(new_key_handles)}")
            
            with self.model_runner.fbgate.background():
                new_k_ptrs, new_v_ptrs = kv_allocator.prepare_flexi_kv_ptrs(new_key_cache, new_value_cache)
                if use_direct_ptr:
                    new_key_ptr_tensor = create_ptr_tensor_from_list(new_key_cache, self.model_runner.device)
                    new_value_ptr_tensor = create_ptr_tensor_from_list(new_value_cache, self.model_runner.device)
                else:
                    new_key_ptr_tensor, new_value_ptr_tensor = None, None
            
            tmp_new_allocated_key_cache.append(new_key_cache)
            tmp_new_allocated_value_cache.append(new_value_cache)
            tmp_new_key_ptr_tensors.append(new_key_ptr_tensor)
            tmp_new_value_ptr_tensors.append(new_value_ptr_tensor)
            tmp_key_cache_ptr_list.append(new_k_ptrs)
            tmp_value_cache_ptr_list.append(new_v_ptrs)
            tmp_new_key_handles.append(new_key_handles)
            tmp_new_value_handles.append(new_value_handles)
            
            # Track old ptrs
            old_k_ptrs = self.model_runner.key_cache_ptrs[local_idx]
            old_v_ptrs = self.model_runner.value_cache_ptrs[local_idx]
            tmp_old_key_cache_ptr_list.append(old_k_ptrs)
            tmp_old_value_cache_ptr_list.append(old_v_ptrs)

    def _flexi_resize_kv_cache(self, new_length: int) -> None:
        """Resize flexi KV cache to new_length blocks.
        
        This is the main entry point that dispatches to _do_flexi_shrink or _do_flexi_grow.
        
        Args:
            new_length: Target number of KV cache blocks per layer.
        """
        assert isinstance(self.model_runner.model, DynamicModelBase)
        time_start = time.time()
        
        _sync_current_cuda_stream(self.device)
        
        cache_length = len(self.model_runner.key_caches[0])
        logger.info(f"resizing kv cache from {cache_length} to {new_length}")
        logger.info(f"before resize kv cache, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        
        # Get common context
        ctx = self._get_flexi_resize_context()
        
        # Dispatch to appropriate handler
        if new_length < cache_length:
            time_before_in_lock, time_after_in_lock, time_after_free = \
                self._do_flexi_shrink(new_length, cache_length, ctx)
        elif new_length > cache_length:
            time_before_in_lock, time_after_in_lock, time_after_free = \
                self._do_flexi_grow(new_length, cache_length, ctx)
        else:
            # No change needed
            logger.info(f"[resize_kv_cache] new_length == cache_length ({new_length}), no resize needed")
            return
        
        _sync_current_cuda_stream(self.device)
        
        logger.debug(
            "[forward]: resized kv cache, available gpu memory: %.2f GB",
            torch.cuda.mem_get_info()[0] / 1024**3)
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

    def start_kv_cache_migration_async(
            self,
            pp_layer_config: list[Tuple[int, int]],
            src_to_plan: dict[int, dict[int, list[int]]],
            slot_mapping: Optional[list[int]] = None,
            logical_num_blocks: Optional[int] = None) -> None:
        """Collective-RPC entry used by executor.

        Only the worker whose `self.rank == source_rank` performs the actual
        send; other ranks are no-ops. The receiver side should already be
        listening via `listen_to_kv_cache_tensor(source_rank)`.
        """
        # assert self.migration_in_process == False, "The migration should not be in process"
        logger.info(f"[operation]: Start KV Cache Migration: {src_to_plan}")
        logger.info(f"[operation]: rank {self.rank} start sending KV Cache Migration")
        
        # CRITICAL FIX: Reset kv_resizing_done for ALL workers (sender and receiver)
        # before migration starts. Previously this was only set for senders, causing
        # scheduler to read stale True value from receivers and prematurely extend
        # block pool during VMM allocation.
        self.kv_resizing_done = False
        self._migration_logical_num_blocks = logical_num_blocks
        if logical_num_blocks is not None:
            logger.info(
                "[autoscaling async] rank %s using migration logical block "
                "count %s for receiver KV binding",
                self.rank, logical_num_blocks)

        receiver_sources: list[int] = []
        for src_rank, rank_to_layers_ids in src_to_plan.items():
            if self.rank in rank_to_layers_ids:
                receiver_sources.append(src_rank)
        if receiver_sources:
            with self._all_patch_applied_cv:
                for src_rank in receiver_sources:
                    self.is_all_patch_applied[src_rank] = False
                self._all_patch_applied_cv.notify_all()
            logger.info(
                "[autoscaling async] rank %s waiting for KV patches from "
                "sources %s", self.rank, receiver_sources)
        
        num_layers = pp_layer_config[self.rank][1] - pp_layer_config[self.rank][0] + 1
        self.target_pp_layer_config = pp_layer_config
        assert len(self.rank_to_layers_ids) == 0
        if self.rank not in src_to_plan:
            logger.info(f"debug: rank {self.rank} is not sending kv cache")
            return None
        self.rank_to_layers_ids = src_to_plan[self.rank]

        # In pipeline-autoscaling no-drain mode, sender ranks must keep the
        # old PtrTable/key_cache_ptrs until old-config batches and KV sending
        # are done. Preparing target PtrTables here is especially unsafe for
        # shrink senders whose target layer count is zero.
        if (use_direct_ptr_for_runtime(self.vllm_config)
                and not self.vllm_config.dynamic_config.
                pipeline_autoscaling_enabled):
            self.model_runner.prepare_ptr_tables(num_layers)
        elif use_direct_ptr_for_runtime(self.vllm_config):
            logger.info(
                "[autoscaling async] rank %s keeps existing PtrTable during "
                "KV migration start; target_num_layers=%s",
                self.rank, num_layers)
        
        # 初始化 sender 线程计数
        with self._sender_threads_cv:
            self._num_active_sender_threads = len(self.rank_to_layers_ids)
        
        with self._layer_loaded_cv:
            sender_key_cache_ptrs = None
            sender_value_cache_ptrs = None
            sender_kv_caches = None
            if (use_direct_ptr_for_runtime(self.vllm_config)
                    and self.vllm_config.dynamic_config.
                    pipeline_autoscaling_enabled):
                sender_layer_ids = sorted({
                    layer_id
                    for layer_ids in self.rank_to_layers_ids.values()
                    for layer_id in layer_ids
                })
                if self._kv_ptr_view_covers_layers(
                        self._autoscale_sender_start_layer,
                        self._autoscale_sender_key_cache_ptrs,
                        self._autoscale_sender_value_cache_ptrs,
                        sender_layer_ids):
                    sender_start_layer_id = self._autoscale_sender_start_layer
                    sender_key_cache_ptrs = self._autoscale_sender_key_cache_ptrs
                    sender_value_cache_ptrs = (
                        self._autoscale_sender_value_cache_ptrs)
                    logger.info(
                        "[autoscaling async] rank %s using captured sender KV "
                        "pointer view: start_layer=%s, num_layers=%s",
                        self.rank, sender_start_layer_id,
                        0 if sender_key_cache_ptrs is None else
                        len(sender_key_cache_ptrs))
                else:
                    sender_start_layer_id = self._kv_cache_start_layer()
                    sender_key_cache_ptrs = list(
                        self.dynamic_kv_synchronizer.key_cache_ptrs)
                    sender_value_cache_ptrs = list(
                        self.dynamic_kv_synchronizer.value_cache_ptrs)
                    logger.info(
                        "[autoscaling async] rank %s captured sender KV "
                        "pointer view at migration start: start_layer=%s, "
                        "num_layers=%s",
                        self.rank, sender_start_layer_id,
                        len(sender_key_cache_ptrs))

                if not self._kv_ptr_view_covers_layers(
                        sender_start_layer_id, sender_key_cache_ptrs,
                        sender_value_cache_ptrs, sender_layer_ids):
                    raise RuntimeError(
                        "KV migration sender could not capture a pointer view "
                        "covering all sending layers: "
                        f"rank={self.rank}, start_layer={sender_start_layer_id}, "
                        f"num_key_cache_ptrs={0 if sender_key_cache_ptrs is None else len(sender_key_cache_ptrs)}, "
                        f"layers={sender_layer_ids}")
            else:
                sender_start_layer_id = self._kv_cache_start_layer()
                sender_kv_caches = list(self.dynamic_kv_synchronizer.kv_caches)
                logger.info(
                    "[autoscaling async] rank %s captured sender KV tensor "
                    "view at migration start: start_layer=%s, num_layers=%s",
                    self.rank, sender_start_layer_id, len(sender_kv_caches))

        def migration_thread(rank: int, layer_ids: list[int],
                             start_layer_id: int,
                             key_cache_ptrs: Optional[list[int]],
                             value_cache_ptrs: Optional[list[int]],
                             kv_caches: Optional[list[torch.Tensor]]):
            try:
                # Set CUDA device for this thread - threads don't inherit CUDA context
                torch.cuda.set_device(self.device)
                assert isinstance(self.model_runner.model, DynamicModelBase)
                time_start = time.time()
                self._wait_for_async_resize("KV sender start")
                if self.rank not in src_to_plan:
                    return None
                # self.dynamic_kv_synchronizer.start_kv_tensor_transfer_async(rank_to_layers_ids, self.model_runner.kv_caches, self.model_runner.model.model.start_layer)
                self._sender_loop(rank, layer_ids, slot_mapping,
                                  start_layer_id, key_cache_ptrs,
                                  value_cache_ptrs, logical_num_blocks,
                                  kv_caches)
                logger.info(f"[timeline]: after start kv cache tensor, time taken: {human_readable_duration(time.time() - time_start)}")
            finally:
                # 通知 do_resize 此 sender 线程已完成
                with self._sender_threads_cv:
                    self._num_active_sender_threads -= 1
                    if self._num_active_sender_threads == 0:
                        self._sender_threads_cv.notify_all()
        for rank, layer_ids in self.rank_to_layers_ids.items():
            threading.Thread(target=migration_thread,
                             args=(rank, layer_ids,
                                   sender_start_layer_id,
                                   sender_key_cache_ptrs,
                                   sender_value_cache_ptrs,
                                   sender_kv_caches),
                             daemon=True).start()
        logger.info("finished starting kv cache migration async")
        return None

    def _sender_loop(self, 
                                rank: int,
                                layer_ids: list[int],
                                slot_mapping: Optional[list[int]] = None,
                                sender_start_layer_id: Optional[int] = None,
                                sender_key_cache_ptrs: Optional[list[int]] = None,
                                sender_value_cache_ptrs: Optional[list[int]] = None,
                                logical_num_blocks: Optional[int] = None,
                                sender_kv_caches: Optional[list[torch.Tensor]] = None,
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
        assert not self.dynamic_kv_synchronizer.kv_cache_transfer_in_process[rank], (
            "KV cache transfer to the target rank is already in process.")
        assert self.dynamic_kv_synchronizer.last_patch_ids[rank] == 0, (
            "The patch id of the target rank should be 0.")
        if sender_start_layer_id is None:
            sender_start_layer_id = self._kv_cache_start_layer()
        assert self.migration_stream is not None
        with torch.cuda.stream(self.migration_stream):
            slot_mapping_dev = None
            slot_mapping_host = None
            slot_mapping_already_valid = False
            if slot_mapping is not None:
                slot_mapping_host = (
                    slot_mapping.detach().cpu().tolist()
                    if isinstance(slot_mapping, torch.Tensor)
                    else slot_mapping)
                if use_kvcached_backend():
                    slot_mapping_host = [
                        int(slot) for slot in slot_mapping_host
                        if int(slot) >= 0
                    ]
                    slot_mapping_already_valid = True
                    if os.environ.get(
                            "VLLM_AUTOSCALING_KVCACHED_DEBUG",
                            "").lower() in {"1", "true", "yes", "on"}:
                        trace_blocks = sorted({
                            int(slot) // int(self.block_size)
                            for slot in slot_mapping_host
                            if int(slot) >= 0
                        })
                        if os.environ.get(
                                "VLLM_AUTOSCALING_KVCACHED_DEBUG_VERBOSE",
                                "").lower() in {"1", "true", "yes", "on"}:
                            logger.warning(
                                "[KVCACHED_MIGRATION_SLOT_TRACE] "
                                "sender_rank=%s target_rank=%s "
                                "phase=initial_snapshot slot_count=%s "
                                "block_size=%s blocks=%s slots_min=%s "
                                "slots_max=%s",
                                self.rank, rank, len(slot_mapping_host),
                                self.block_size, trace_blocks,
                                min(slot_mapping_host)
                                if slot_mapping_host else None,
                                max(slot_mapping_host)
                                if slot_mapping_host else None)
                        else:
                            logger.warning(
                                "[KVCACHED_MIGRATION_SLOT_TRACE] "
                                "sender_rank=%s target_rank=%s "
                                "phase=initial_snapshot slot_count=%s "
                                "block_size=%s blocks_count=%s "
                                "sample_blocks=%s slots_min=%s slots_max=%s",
                                self.rank, rank, len(slot_mapping_host),
                                self.block_size, len(trace_blocks),
                                trace_blocks[:64],
                                min(slot_mapping_host)
                                if slot_mapping_host else None,
                                max(slot_mapping_host)
                                if slot_mapping_host else None)
                if not use_kvcached_backend():
                    slot_mapping_dev = torch.tensor(slot_mapping_host,
                                                    dtype=torch.int64,
                                                    device=self.device)
                    _sync_current_cuda_stream(self.device)
            autoscaling_enabled = (
                self.vllm_config.dynamic_config.
                pipeline_autoscaling_enabled)
            send_gate = self.model_runner.fbgate.background
            for layer_id in layer_ids:
                time_start = time.time()
                # Sender KV indexing must stay tied to the old layout captured
                # when migration starts. Autoscaling updates the target layout
                # asynchronously, while old-config batches are still running.
                nccl_lock = getattr(self, "_nccl_lock", None)
                nccl_lock_context = (nccl_lock if nccl_lock is not None
                                     else nullcontext())
                autoscale_snapshot_context = (
                    nullcontext() if autoscaling_enabled else send_gate())
                with autoscale_snapshot_context:
                    # The live KV cache is still writable by the foreground
                    # model execution stream. For KVCacheD, take the page
                    # lifetime lock before the forward lock so unmap cannot
                    # invalidate a page between slot filtering and gather.
                    page_lifetime_context = (
                        self.dynamic_kv_synchronizer
                        .kvcached_page_lifetime_context()
                        if use_kvcached_backend() else nullcontext())
                    with page_lifetime_context:
                        forward_lock_context = self.model_runner.forward_lock
                        with forward_lock_context:
                            inference_sync_start = time.time()
                            if self.inference_stream is not None:
                                self.inference_stream.synchronize()
                            inference_sync_duration = (
                                time.time() - inference_sync_start)
                            layer_slot_mapping_dev = slot_mapping_dev
                            layer_slot_mapping_already_valid = (
                                slot_mapping_already_valid)
                            if (use_kvcached_backend()
                                    and slot_mapping_host is not None):
                                layer_slot_mapping_host = (
                                    self.dynamic_kv_synchronizer
                                    .filter_slots_for_kvcached_live_blocks(
                                        slot_mapping_host,
                                        int(self.block_size),
                                        reason=(
                                            "initial_snapshot:"
                                            f"layer={layer_id}:target={rank}")))
                                layer_slot_mapping_dev = torch.tensor(
                                    layer_slot_mapping_host,
                                    dtype=torch.int64,
                                    device=self.device)
                                layer_slot_mapping_already_valid = True
                                _sync_current_cuda_stream(self.device)
                            with nccl_lock_context:
                                kv_tensor_meta, kv_tensor_data = self.dynamic_kv_synchronizer.get_kv_tensor_from_cache(
                                    layer_ids,
                                    layer_id,
                                    sender_start_layer_id,
                                    self.model_runner.page_meta,
                                    layer_slot_mapping_dev,
                                    key_cache_ptrs=sender_key_cache_ptrs,
                                    value_cache_ptrs=sender_value_cache_ptrs,
                                    logical_num_blocks=logical_num_blocks,
                                    kv_caches=sender_kv_caches,
                                    slot_mapping_already_valid=layer_slot_mapping_already_valid)
                            # Keep the KV snapshot read ordered before the next
                            # foreground forward can overwrite the same cache pages.
                            migration_sync_start = time.time()
                            self.migration_stream.synchronize()
                            logger.debug(
                                "[autoscaling async] rank %s synchronized streams "
                                "while snapshotting KV layer %s for rank %s: "
                                "inference_wait=%s migration_wait=%s",
                                self.rank, layer_id, rank,
                                human_readable_duration(
                                    inference_sync_duration),
                                human_readable_duration(
                                    time.time() - migration_sync_start))
                logger.info(f"start to send kv tensor for layer {layer_id} to rank {rank}, kv_tensor_meta: {kv_tensor_meta}, kv_tensor_data shape: {kv_tensor_data.shape}, time taken to get kv tensor: {human_readable_duration(time.time() - time_start)} seconds")
                if autoscaling_enabled:
                    self.dynamic_kv_synchronizer.send_kv_tensor_to_rank(
                        rank,
                        kv_tensor_meta,
                        kv_tensor_data,
                        layer_slot_mapping_dev)
                if not autoscaling_enabled:
                    with send_gate():
                        self.dynamic_kv_synchronizer.send_kv_tensor_to_rank(
                            rank,
                            kv_tensor_meta,
                            kv_tensor_data,
                            layer_slot_mapping_dev)
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
            for kv_patch in self.dynamic_kv_synchronizer.get_kv_patch(
                    rank, sender_start_layer_id, layer_ids,
                    self.model_runner.page_meta,
                    cuda_op_lock=self.model_runner.forward_lock,
                    key_cache_ptrs=sender_key_cache_ptrs,
                    value_cache_ptrs=sender_value_cache_ptrs,
                    kv_caches=sender_kv_caches,
                    live_block_ids_getter=(
                        self._autoscaling_live_request_block_ids
                        if self.vllm_config.dynamic_config.
                        pipeline_autoscaling_enabled else None),
                    block_size=self.block_size):
                time_start = time.time()
                patch_payload_size = kv_patch.kv_payload.numel() * kv_patch.kv_payload.element_size()
                patch_slot_mapping_size = kv_patch.slot_mapping.numel() * kv_patch.slot_mapping.element_size()
                patch_send_context = (
                    nullcontext() if self.vllm_config.dynamic_config.
                    pipeline_autoscaling_enabled else
                    self.model_runner.fbgate.background())
                with patch_send_context:
                    with (self.dynamic_kv_synchronizer
                          .kvcached_page_lifetime_context()):
                        try:
                            self.dynamic_kv_synchronizer.send_kv_patch_to_rank(
                                rank,
                                kv_patch)
                        finally:
                            (self.dynamic_kv_synchronizer
                             .release_kvcached_outbound_patch_snapshot(
                                 kv_patch))
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

            forward_lock_wait_start = time.time()
            with self.model_runner.forward_lock:
                forward_lock_acquired = time.time()
                if self.log_stop_time:
                    logger.info(f"[STOP_TIME][worker][sync][forward_lock]: wait={forward_lock_acquired - forward_lock_wait_start:.4f}s")
                self.dynamic_kv_synchronizer.send_kv_tensor_sync(
                    sending_layers_list,
                    self.model_runner.kv_caches,
                    self._kv_cache_start_layer(),
                    slot_mapping_tensor)
            forward_lock_total = time.time() - forward_lock_wait_start
            if self.log_stop_time:
                logger.info(f"[STOP_TIME][worker][sync][forward_lock]: total_hold={forward_lock_total:.4f}s")
            all_layer_ranges = []
            for _, ranges in sending_layers_plans.items():
                all_layer_ranges.extend(ranges)
            with self.model_runner.forward_lock:
                caches_to_free_key, caches_to_free_value, ptrs_to_free_key, ptrs_to_free_value, handles_to_free_key, handles_to_free_value, grouped_handles_to_free = self.atomic_shelve_kv_cache(self.rank, all_layer_ranges)
            # memory_snapshot(f"rank{self.rank}_after_atomic_shelve", self.device)
            self.release_kv_cache_for_layers(self.rank, caches_to_free_key, caches_to_free_value, ptrs_to_free_key, ptrs_to_free_value, handles_to_free_key, handles_to_free_value, grouped_handles_to_free)
            # memory_snapshot(f"rank{self.rank}_after_release_kv_cache", self.device)
            self.remove_layers(self.rank, all_layer_ranges)
            # memory_snapshot(f"rank{self.rank}_after_remove_layers", self.device)
                
        start_layer = pp_layer_config[self.rank][0]
        # Receiver Side, Wait for receive to finish.
        #
        # For PP>2 a middle rank can be both a sender and a receiver in the
        # same reconfiguration, e.g. 16,16,16,16 -> 8,8,24,24 makes rank 1
        # send layers 16-31 while receiving layers 8-15. The old PP=2-oriented
        # branch skipped the receive wait for sender ranks, so direct pointer
        # tables could be committed before all newly received layer KV pointers
        # were bound.
        if self.rank in rank_to_layer_ids:
            with self._receive_finished_cv:
                while self.receive_in_process:
                    logger.info(f"[sync migration]: rank {self.rank} waiting for receive to finish")
                    self._receive_finished_cv.wait()
        logger.info(f"[timeline]: after start kv cache migration sync, time taken: {human_readable_duration(time.time() - time_start)}")
        is_direct = use_direct_ptr_for_runtime(self.vllm_config)
        if is_direct:
            memory_snapshot(f"rank{self.rank}_before_commit_ptr_tables", self.device)
            self.model_runner.commit_ptr_tables(self.model_runner.k_ptr_tensors, self.model_runner.v_ptr_tensors, target_start_layer=start_layer)
            memory_snapshot(f"rank{self.rank}_after_commit_ptr_tables", self.device)
        
        logger.info(f"finish sync migration in {time.time() - time_start:.2f}")
        return None

    def start_kv_cache_migration_async_fast(self, pp_layer_config: list[Tuple[int, int]], 
                                             src_to_sending_layers: dict[int, dict[int, list[Tuple[int, int]]]], 
                                             rank_to_layer_ids: dict[int, list[Tuple[int, int]]],
                                             slot_mapping: Optional[list[int]] = None) -> None:
        """Async-fast KV cache migration - similar to sync but used after async weight loading.
        
        This performs a one-shot KV transfer without the continuous patch phase.
        Weight loading should have already completed before this is called.
        
        Args:
            pp_layer_config: Target layer configuration per rank
            src_to_sending_layers: Mapping of source rank to {dest_rank: layer_ranges}
            rank_to_layer_ids: Mapping of destination rank to layer ranges to receive
            slot_mapping: Optional list of slot indices for flexi mode
        """
        time_start = time.time()
        logger.info(f"[operation]: Start KV Cache Migration AsyncFast: {src_to_sending_layers}")

        # Keep async_fast lifecycle consistent with async migration:
        # cleanup/remove/resize will happen in async_migration_after_execute_callback.
        self.kv_resizing_done = False
        self.target_pp_layer_config = pp_layer_config
        num_layers = pp_layer_config[self.rank][1] - pp_layer_config[self.rank][0] + 1
        if (use_direct_ptr_for_runtime(self.vllm_config)
                and not self.vllm_config.dynamic_config.
                pipeline_autoscaling_enabled):
            self.model_runner.prepare_ptr_tables(num_layers)
        elif use_direct_ptr_for_runtime(self.vllm_config):
            logger.info(
                "[autoscaling async_fast] rank %s keeps existing PtrTable "
                "during KV migration start; target_num_layers=%s",
                self.rank, num_layers)

        assert len(self.rank_to_layers_ids) == 0
        
        with self._receive_finished_cv:
            assert self.receive_in_process == False, "The receiving kv cache should not be in process"

        if self.rank in rank_to_layer_ids:
            logger.info(f"[async_fast] rank {self.rank} is receiving kv cache")
            with self._receive_finished_cv:
                self.receive_in_process = True

        assert isinstance(self.model_runner.model, DynamicModelBase)
        memory_snapshot(f"rank{self.rank}_async_fast_migration_entry", self.device)

        # Sender Side, Send KV Cache (same as sync)
        logger.info(f"[async_fast] Enter async_fast migration process")
        if self.rank in src_to_sending_layers:
            sending_layers_plans = src_to_sending_layers[self.rank]
            logger.info(f"[async_fast] Enter Sender Side, sending_layer_plans:{sending_layers_plans}")
            sending_layers_list: dict[int, list[int]] = {}
            for rank, layer_ranges in sending_layers_plans.items():
                for layer_range in layer_ranges:
                    layer_ids = sending_layers_list.setdefault(rank, [])
                    layer_ids.extend(list(range(layer_range[0], layer_range[1] + 1)))

            # Required by async_migration_after_execute_callback for final cleanup.
            self.rank_to_layers_ids = sending_layers_list

            # Convert slot_mapping to tensor if provided
            slot_mapping_tensor: Optional[torch.Tensor] = None
            if slot_mapping is not None:
                slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int64, device=self.device)

            forward_lock_wait_start = time.time()
            with self.model_runner.forward_lock:
                forward_lock_acquired = time.time()
                if self.log_stop_time:
                    logger.info(f"[STOP_TIME][worker][async_fast][forward_lock]: wait={forward_lock_acquired - forward_lock_wait_start:.4f}s")
                self.dynamic_kv_synchronizer.send_kv_tensor_sync(
                    sending_layers_list,
                    self.model_runner.kv_caches,
                    self._kv_cache_start_layer(),
                    slot_mapping_tensor)
            forward_lock_total = time.time() - forward_lock_wait_start
            if self.log_stop_time:
                logger.info(f"[STOP_TIME][worker][async_fast][forward_lock]: total_hold={forward_lock_total:.4f}s")

        # Receiver Side, Wait for receive to finish
        if self.rank not in src_to_sending_layers:
            # Wait for receive to finish if this rank is a receiver
            if self.rank in rank_to_layer_ids:
                with self._receive_finished_cv:
                    while self.receive_in_process:
                        logger.info(f"[async_fast]: rank {self.rank} waiting for receive to finish")
                        self._receive_finished_cv.wait()
            logger.info(f"[async_fast timeline]: after kv cache migration, time taken: {human_readable_duration(time.time() - time_start)}")

        logger.info(f"[async_fast] finish migration in {time.time() - time_start:.2f}")
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
        self._wait_for_async_resize("KV receiver listen loop")
        
        assert isinstance(self.model_runner.model, DynamicModelBase)
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
        time_start = None
        try:
          while True:
            received_layer = set()
            layers_to_be_received  = set()
            logger.info(f"[operation]: start to listen to kv cache tensor and patches")
            kv_cache_snapshot = list(self.model_runner.kv_caches)
            for i, kv_cache in enumerate(kv_cache_snapshot):
                logger.info(f"[debug]: kv_caches[{i}] shape: {kv_cache.shape}")
            first_time = True
            tmp_kv_tensors_dict: dict[int, torch.Tensor] = {}
            tmp_slot_mapping_dict: dict[int, torch.Tensor] = {}
            tmp_slot_token_num: int = 0

            while True:
                # 1) Firstly wait for all tensor the be received
                meta = self.dynamic_kv_synchronizer.recv_controller(from_rank)
                logger.info("received kv tensor meta")
                if first_time:
                    assert isinstance(meta, (KVTensorMeta, FlexiKVTensorMeta,
                                             KVCachedSparseKVTensorMeta))
                    layers_to_be_received = set(meta.layer_to_be_received)
                    time_start = time.time()
                    first_time = False
                    self.receiver_num_applied_token_dict[from_rank] = 0
                    self.is_all_patch_applied[from_rank] = False # 第一次接受到kv tensor时，需要将is_all_patch_applied设置为False，表示需要等待所有patch都应用完毕后才能进行配置的转换
                    tmp_slot_token_num = meta.num_tokens
                if meta.type not in ("kv_tensor", "kv_tensor_sparse"):
                    assert layers_to_be_received == received_layer, "The layers to be received should be the same as the received layers"
                    break

                assert meta.layer_id not in received_layer, "The layer should not be received twice"
                if meta.num_tokens != tmp_slot_token_num:
                    logger.warning(
                        "[KVCACHED_MIGRATION_SLOT_TRACE] receiver_rank=%s "
                        "from_rank=%s phase=initial_snapshot_recv "
                        "layer=%s first_num_tokens=%s layer_num_tokens=%s "
                        "using_max_for_progress=%s",
                        self.rank, from_rank, meta.layer_id,
                        tmp_slot_token_num, meta.num_tokens,
                        max(tmp_slot_token_num, meta.num_tokens))
                    tmp_slot_token_num = max(tmp_slot_token_num,
                                             meta.num_tokens)
                received_layer.add(meta.layer_id)
                logger.info(f"[debug]: rank {self.rank} receive kv tensor meta {meta}")

                # Bulk KV receives share the local NCCL lock with PP transfer.
                # Autoscaling receivers must not wait for the migration
                # foreground gate here: a middle PP rank may already have a
                # foreground task queued behind the sender's forward_lock while
                # this listener needs to ACCEPT another rank's transfer.
                recv_gate = nullcontext
                autoscaling_enabled = (
                    self.vllm_config.dynamic_config.
                    pipeline_autoscaling_enabled)
                # Bulk receives land in temporary tensors. Holding the local
                # forward lock here can deadlock a middle autoscaling rank that
                # is concurrently sending downstream patches while receiving
                # upstream snapshots. Binding/applying into the local KV cache
                # remains serialized below.
                recv_cuda_op_lock = (None if autoscaling_enabled
                                     else self.model_runner.forward_lock)
                with recv_gate():
                    if isinstance(meta, (FlexiKVTensorMeta,
                                         KVCachedSparseKVTensorMeta)):
                        slot_mapping, kv_tensor = (
                            self.dynamic_kv_synchronizer.recv_kv_tensor(
                                from_rank,
                                meta,
                                cuda_op_lock=recv_cuda_op_lock))
                        tmp_slot_mapping_dict[meta.layer_id] = slot_mapping
                    else:
                        assert isinstance(meta, KVTensorMeta)
                        kv_tensor = (
                            self.dynamic_kv_synchronizer.recv_kv_tensor(
                                from_rank,
                                meta,
                                cuda_op_lock=recv_cuda_op_lock))
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
            # Store incoming KV tensors in the temporary dictionaries above
            # first. Only wait for local layer/KV structures once we are ready
            # to bind them, so the sender does not hold the NCCL lock while this
            # rank is still loading weights.
            layer_ids = list(tmp_kv_tensors_dict.keys())
            self._wait_for_kv_layer_structures_ready(set(layer_ids),
                                                     from_rank)
            time_start_bind_kv_cache = time.time()
            
            # Pre-allocate KV cache memory for all layers before binding.
            # IMPORTANT: Must acquire forward_lock to prevent concurrent allocation
            # with main forward thread. Otherwise both threads may try to allocate
            # memory simultaneously, causing OOM even when total memory would suffice.
            preallocated_caches: dict[int, tuple] = {}
            receiver_block_num = (
                self._migration_logical_num_blocks
                if self._migration_logical_num_blocks is not None
                else self.block_num)
            if receiver_block_num != self.block_num:
                logger.info(
                    "[autoscaling async] rank %s receiver block count "
                    "switching from %s to migration logical blocks %s "
                    "before binding layers %s",
                    self.rank, self.block_num, receiver_block_num,
                    layer_ids)
                self.block_num = receiver_block_num
            if is_flexi:
                _sync_current_cuda_stream(self.device)
                gc.collect()

                # Check overhead BEFORE preallocate (if monitoring enabled)
                if not self.vllm_config.dynamic_config.disable_memory_overhead_monitor:
                    layer_count, kv_cache_bytes = self._get_current_memory_params()
                    overhead_monitor = get_memory_overhead_monitor()
                    overhead_monitor.check_overhead_before(
                        operation=f"preallocate_kv_for_{len(layer_ids)}_layers",
                        current_layer_count=layer_count,
                        current_kv_cache_bytes=kv_cache_bytes
                    )

                free_before, _ = torch.cuda.mem_get_info()
                logger.info(f"[timeline]: free memory before preallocate: {free_before / 1024**3:.2f} GB")
                preallocated_caches = self.model_runner.preallocate_flexi_kv_caches_for_migration(
                    layer_ids, receiver_block_num
                )
                free_after, _ = torch.cuda.mem_get_info()
                logger.info(f"[timeline]: pre-allocation done, free memory after: {free_after / 1024**3:.2f} GB, allocated: {(free_before - free_after) / 1024**3:.2f} GB")
                logger.info(f"[timeline]: pre-allocation done in {human_readable_duration(time.time() - time_start_bind_kv_cache)}")

            if not is_flexi and use_kvcached_backend():
                # KVCacheD materialization can trigger lazy page mapping and an
                # async copy into the backing FTensor. Keep the whole bind phase
                # serialized with forward until the stream is synchronized.
                receiver_start_layer = self._kv_cache_start_layer()
                receiver_end_layer = (
                    receiver_start_layer + len(self.model_runner.kv_caches))
                with self.model_runner.forward_lock:
                    for layer_id in layer_ids:
                        logger.info(f"current memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB, start to bind kv cache for layer {layer_id}")
                        kv_tensor = tmp_kv_tensors_dict[layer_id]
                        if layer_id in tmp_slot_mapping_dict:
                            bind_kv_tensor = (
                                materialize_kvcached_sparse_received_kv_tensor(
                                    self.model_runner,
                                    layer_id,
                                    receiver_start_layer,
                                    receiver_block_num,
                                    kv_tensor,
                                    tmp_slot_mapping_dict[layer_id]))
                            tmp_slot_mapping_dict.pop(layer_id)
                        else:
                            bind_kv_tensor = (
                                materialize_kvcached_received_kv_tensor(
                                    self.model_runner,
                                    layer_id,
                                    receiver_start_layer,
                                    receiver_block_num,
                                    kv_tensor))
                        dynamic_bind_single_kv_tensor(
                            layer_id,
                            receiver_start_layer,
                            receiver_end_layer,
                            self.compilation_config.static_forward_context,
                            self.dynamic_kv_synchronizer,
                            runner=self.model_runner,
                            kv_tensor=bind_kv_tensor)
                        tmp_kv_tensors_dict.pop(layer_id)
                    sync_start = time.time()
                    _sync_current_cuda_stream(self.device)
                    logger.info(
                        "[timeline]: current stream synchronize after "
                        "KVCacheD bind took %s",
                        human_readable_duration(time.time() - sync_start))
                    sync_start = time.time()
                    if self.migration_stream is not None:
                        self.migration_stream.synchronize()
                        logger.info(
                            f"[timeline]: migration stream synchronize after bind took {human_readable_duration(time.time() - sync_start)}"
                        )
                    else:
                        _sync_current_cuda_stream(self.device)
                        logger.info(
                            f"[timeline]: current stream synchronize after bind took {human_readable_duration(time.time() - sync_start)}"
                        )
            else:
                for layer_id in layer_ids:
                    logger.info(f"current memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB, start to bind kv cache for layer {layer_id}")
                    kv_tensor = tmp_kv_tensors_dict[layer_id]
                    if is_flexi:
                        slot_mapping = tmp_slot_mapping_dict[layer_id]
                        assert self.device is not None
                        logger.info(f"[timeline]: binding layer {layer_id} on migration stream {self.migration_stream}")
                        # Use pre-allocated memory
                        preallocated = preallocated_caches.get(layer_id)
                        dynamic_flexi_bind_single_kv_tensor(
                            self._kv_cache_start_layer(),
                            self._kv_cache_start_layer() + len(self.model_runner.key_caches),
                            layer_id,
                            slot_mapping,
                            receiver_block_num,
                            kv_tensor,
                            self.vllm_config.compilation_config.static_forward_context,
                            self.dynamic_kv_synchronizer,
                            self.model_runner,
                            self.device,
                            stream=self.migration_stream,
                            preallocated=preallocated,
                        )
                    else:
                        # dynamic_bind_single_kv_tensor内部不使用所，因此我们需要在外面加锁
                        with self.model_runner.forward_lock:
                            dynamic_bind_single_kv_tensor(
                                layer_id,
                                self._kv_cache_start_layer(),
                                self._kv_cache_start_layer() + len(self.model_runner.kv_caches),
                                self.compilation_config.static_forward_context,
                                self.dynamic_kv_synchronizer,
                                runner=self.model_runner,
                                kv_tensor=kv_tensor
                            )
                    tmp_kv_tensors_dict.pop(layer_id)
                sync_start = time.time()
                if self.migration_stream is not None:
                    self.migration_stream.synchronize()
                    logger.info(
                        f"[timeline]: migration stream synchronize after bind took {human_readable_duration(time.time() - sync_start)}"
                    )
                else:
                    _sync_current_cuda_stream(self.device)
                    logger.info(
                        f"[timeline]: current stream synchronize after bind took {human_readable_duration(time.time() - sync_start)}"
                    )
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
                with self._kv_patch_recv_apply_lock:
                    with (self.dynamic_kv_synchronizer
                          .kvcached_page_lifetime_context()):
                        recv_patch_gate = nullcontext
                        recv_patch_cuda_op_lock = self.model_runner.forward_lock
                        with recv_patch_gate():
                            slot_mapping, kv_payload = (
                                self.dynamic_kv_synchronizer.recv_kv_patch(
                                    from_rank, meta,
                                    cuda_op_lock=recv_patch_cuda_op_lock))
                        logger.info(f"[listen loop]: Received Meta: {meta}")
                        logger.info(f"[listen loop]: slot mapping device: {slot_mapping.device}, kv payload device: {kv_payload.device}, kv payload shape: {kv_payload.shape}, slot mapping shape: {slot_mapping.shape}")

                        # Only apply patch if there's actual data (skip for empty kv_patch_finished)
                        if kv_payload.numel() > 0:
                            with self._layer_loaded_cv:
                                receiver_start_layer_id = self._kv_patch_apply_start_layer()
                                receiver_key_cache_ptrs = None
                                receiver_value_cache_ptrs = None
                                if use_flexi_kv_for_runtime(self.vllm_config):
                                    receiver_key_cache_ptrs = list(
                                        self.dynamic_kv_synchronizer.key_cache_ptrs)
                                    receiver_value_cache_ptrs = list(
                                        self.dynamic_kv_synchronizer.value_cache_ptrs)
                                    if not self._kv_ptr_view_covers_layers(
                                            receiver_start_layer_id,
                                            receiver_key_cache_ptrs,
                                            receiver_value_cache_ptrs,
                                            list(meta.layer_ids)):
                                        raise RuntimeError(
                                            "KV migration receiver could not capture "
                                            "a pointer view covering patch layers: "
                                            f"rank={self.rank}, from_rank={from_rank}, "
                                            f"start_layer={receiver_start_layer_id}, "
                                            f"num_key_cache_ptrs={len(receiver_key_cache_ptrs)}, "
                                            f"layers={meta.layer_ids}")
                            with self.model_runner.forward_lock:
                                self.dynamic_kv_synchronizer.apply_one_patch_to_kv_cache(
                                    receiver_start_layer_id, meta,
                                    kv_payload, slot_mapping,
                                    self.model_runner.page_meta,
                                    key_cache_ptrs=receiver_key_cache_ptrs,
                                    value_cache_ptrs=receiver_value_cache_ptrs)
                        else:
                            logger.info(f"[listen loop]: skipping apply for empty patch (type={meta.type})")
                        self.receiver_num_applied_token_dict[from_rank] += meta.num_tokens
                        logger.info(f"[num tokens]: receiver side: rank {from_rank} applied token: {meta.num_tokens}, total applied token: {self.receiver_num_applied_token_dict[from_rank]}")
                        self.dynamic_kv_synchronizer.notify_kv_patch_applied(
                            from_rank, meta.id)
                    
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
                        # memory_monitor = get_gpu_memory_monitor()
                        # memory_monitor.check_memory_after_migration_local(
                        #     start_layer, end_layer, self.rank, self.device,
                        #     migration_type="async"
                        # )
                    
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

    def wait_for_all_patch_applied(self, reason: str) -> None:
        time_start = time.time()
        with self._all_patch_applied_cv:
            while not self.all_patch_applied():
                logger.info(
                    "[autoscaling sync] rank %s waiting for all KV patches "
                    "before %s: status=%s",
                    self.rank, reason, dict(self.is_all_patch_applied))
                self._all_patch_applied_cv.wait(timeout=5.0)
        logger.info(
            "[timeline]: rank %s waited for all KV patches before %s, "
            "time taken: %s",
            self.rank, reason,
            human_readable_duration(time.time() - time_start))

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

    def _kv_patch_apply_start_layer(self) -> int:
        # Patch application indexes dynamic_kv_synchronizer.key_cache_ptrs,
        # whose layout follows kv_cache_start_layer, not model.start_layer.
        # During shrink, KV pointers can be switched to the final layout
        # before model.delete_layers() updates model.start_layer.
        return self._kv_cache_start_layer()

    def async_migration_before_execute_callback(self,
                                                scheduler_output: "DynamicSchedulerOutput" 
                                               ) -> None:
        self._maybe_switch_block(scheduler_output)
        # 处于迁移中时，或这是同步批（用于发送 finished 信号），直接放行。
        if scheduler_output.migration_in_process:
            assert self.after_migration_total_token == 0, "If in migration process, the total after migration token num should be zero"
            logger.debug(
                "rank %s is in migration or sync after migration; skip KV "
                "synchronize before execute callback",
                self.rank)
            return

        if self.after_migration_total_token != 0:
            # 等待所有KV cache补丁都应用完毕后再使用新的pp配置
            time_start = time.time()
            with self._all_patch_applied_cv:
                while not self.all_patch_applied():
                    self._all_patch_applied_cv.wait()
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
        trace_info = {
            "scheduler_step_id":
            getattr(scheduler_output, "scheduler_step_id", -1),
            "scheduler_output_version":
            getattr(scheduler_output, "current_scheduler_output_version", -1),
            "scheduler_request_free_epoch":
            getattr(scheduler_output, "scheduler_request_free_epoch", -1),
            "scheduler_block_free_epoch":
            getattr(scheduler_output, "scheduler_block_free_epoch", -1),
        }

        if scheduler_output.migration_in_process:
            assert scheduler_output.sender_list is not None and scheduler_output.receiver_list is not None, "If in migration process, the sender list should not be None"
            is_sender = self.rank in scheduler_output.sender_list
            is_receiver = self.rank in scheduler_output.receiver_list

            if is_sender:
                # Send the kv cache patch to the other ranks if the migration is in process
                block_table = self.model_runner.input_batch.block_table[0]
                slot_mapping = (
                    block_table.slot_mapping_np[:num_total_new_tokens]
                    .copy()
                    .tolist())
                for rank in self.rank_to_layers_ids:
                    self.dynamic_kv_synchronizer.add_new_tokens_to_kv_synchronizer(
                        rank,
                        slot_mapping,
                        is_sync,
                        num_total_new_tokens,
                        trace_info=trace_info,
                    )
                logger.info(
                    "Sender added kv patch to synchronizer, is finished: %s, "
                    "num total new tokens: %s, trace_info=%s",
                    is_sync, num_total_new_tokens, trace_info)

            if is_sync:
                logger.info(
                    "[autoscaling sync] rank %s deferred sync KV cleanup "
                    "until after PP hidden-state transfer",
                    self.rank)
        else:
            self._release_deleted_model_layers_after_inference()

    def async_migration_after_pp_transfer_callback(
            self, scheduler_output: "DynamicSchedulerOutput"):
        if not isinstance(self.model_runner.model, DynamicModelBase):
            return
        if (not scheduler_output.migration_in_process
                or not scheduler_output.is_sync_after_migration):
            return
        assert scheduler_output.sender_list is not None
        assert scheduler_output.receiver_list is not None
        if self.vllm_config.dynamic_config.pipeline_autoscaling_enabled:
            logger.info(
                "[autoscaling sync] rank %s defers migration finalization "
                "to all-worker RPC", self.rank)
            return
        self._finalize_async_migration_after_sync(
            list(scheduler_output.sender_list),
            list(scheduler_output.receiver_list),
            scheduler_output.total_migration_tokens,
            scheduler_output.new_kv_cache_block_num)

    def finalize_async_migration_after_sync(
        self,
        sender_list: list[int],
        receiver_list: list[int],
        total_migration_tokens: int,
        new_kv_cache_block_num: int,
    ) -> None:
        if not isinstance(self.model_runner.model, DynamicModelBase):
            return
        self._finalize_async_migration_after_sync(
            sender_list, receiver_list, total_migration_tokens,
            new_kv_cache_block_num)

    def _finalize_async_migration_after_sync(
        self,
        sender_list: list[int],
        receiver_list: list[int],
        num_total_migration_tokens: int,
        new_kv_cache_block_num: int,
    ) -> None:
        sender_set = set(sender_list)
        receiver_set = set(receiver_list)
        is_sender = self.rank in sender_set
        is_receiver = self.rank in receiver_set

        if not is_sender and not is_receiver:
            fixed_blocks = self.vllm_config.dynamic_config.fixed_num_gpu_blocks
            if fixed_blocks <= 0:
                if self.block_num != new_kv_cache_block_num:
                    logger.info(
                        "[autoscaling sync] rank %s has no sender/receiver "
                        "work; resizing local KV cache from %s to %s to match "
                        "the global block pool",
                        self.rank, self.block_num, new_kv_cache_block_num)
                    self.resize_kv_cache(new_kv_cache_block_num)
                else:
                    logger.info(
                        "[autoscaling sync] rank %s has no sender/receiver "
                        "work and KV cache already matches %s blocks",
                        self.rank, new_kv_cache_block_num)
            else:
                logger.info(
                    "fixed_num_gpu_blocks=%s, skipping no-op rank "
                    "worker-side resize (would be %s blocks)",
                    fixed_blocks, new_kv_cache_block_num)
            self.after_migration_total_token = 0
            self.after_migration_applied_token_num = 0
            self.kv_resizing_done = True
            self._clear_autoscale_sender_pointer_view(
                "no-op migration finalize")
            with self.resizing_done_cv:
                self.resizing_done = True
                self.resizing_done_cv.notify_all()
            logger.info(
                "[autoscaling sync] rank %s has no sender/receiver work; "
                "marked migration finalize done", self.rank)
            return

        if self.target_pp_layer_config is None:
            logger.info(
                "[autoscaling sync] rank %s migration already finalized; "
                "sender=%s receiver=%s", self.rank, is_sender, is_receiver)
            self.kv_resizing_done = True
            self._clear_autoscale_sender_pointer_view(
                "already-finalized migration")
            with self.resizing_done_cv:
                self.resizing_done = True
                self.resizing_done_cv.notify_all()
            return

        time_start_to_sync = time.time()
        time_start = time.time()
        # Set CUDA device for this thread - threads don't inherit CUDA context
        # torch.cuda.set_device(self.device)
        # logger.info(f"[operation]: before finish kv cache transfer start to finish kv cache tansfer, delete layers, release kv cache, reinitialize kv cache")
        # Sender finishes transfer and cleans up; receiver no-ops.
        layer_ranges = []
        for layers in self.rank_to_layers_ids.values():
            # Assert the layers are sorted
            assert layers == sorted(layers), "The layers should be sorted"
            layer_ranges.append((layers[0], layers[-1]))

        if is_sender and layer_ranges:
            # All migration KV patches have already been applied before this
            # finalize path runs. Keep using the copied layer_ranges for
            # cleanup, but stop marking target-topology forwards as sender
            # work while background KV resize is still in progress.
            self.rank_to_layers_ids = {}
            logger.info(
                "[autoscaling sync] rank %s cleared sender routing state "
                "before background cleanup/resize", self.rank)

        assert self.target_pp_layer_config is not None, "target_pp_layer_config must be set"
        caches_to_free_key = []
        caches_to_free_value = []
        ptrs_to_free_key = []
        ptrs_to_free_value = []
        handles_to_free_key: list[list[int]] = []
        handles_to_free_value: list[list[int]] = []
        grouped_handles_to_free: list[tuple[list[int], list[int]]] = []  # (handles, base_k_ptrs) tuples
        if is_receiver:
            # A rank can be both sender and receiver during autoscaling
            # shrink. Drain incoming patches before mutating kv_cache_start_layer
            # or the local KV cache list, otherwise the listener may index the
            # old patch layer ids against the new target-local view.
            self.wait_for_all_patch_applied(
                "pre-shelve autoscaling KV cache cleanup")
        if layer_ranges:
            with self.model_runner.forward_lock:
                caches_to_free_key, caches_to_free_value, ptrs_to_free_key, ptrs_to_free_value, handles_to_free_key, handles_to_free_value, grouped_handles_to_free = self.atomic_shelve_kv_cache(self.rank, layer_ranges)

        start_layer, end_layer = self.target_pp_layer_config[self.rank]
        target_num_layers = end_layer - start_layer + 1
        is_direct = use_direct_ptr_for_runtime(self.vllm_config)
        if is_direct and target_num_layers > 0:
            self.model_runner.commit_ptr_tables(self.model_runner.k_ptr_tensors, self.model_runner.v_ptr_tensors, target_start_layer=start_layer)
        elif is_direct:
            logger.info(
                "[autoscaling async] rank %s target PP range is empty "
                "(%s, %s); skip PtrTable commit",
                self.rank, start_layer, end_layer)
        self.target_pp_layer_config = None

        # Set resizing_done = False BEFORE starting the thread, unconditionally
        # This ensures wait_for_resize_done() will block until do_resize completes
        with self.resizing_done_cv:
            self.resizing_done = False

        # Also reset kv_resizing_done as a safety net (already reset in
        # start_kv_cache_migration_async, but reinforce here at sync point)
        self.kv_resizing_done = False

        def do_resize():
            try:
                # Set CUDA device for this thread - threads don't inherit CUDA context
                torch.cuda.set_device(self.device)
                assert self.migration_stream is not None
                if is_receiver:
                    self.wait_for_all_patch_applied("sync KV cache cleanup")
                if is_sender:
                    # 等待所有 sender 线程完成后再释放 KV 缓存内存
                    with self._sender_threads_cv:
                        while self._num_active_sender_threads > 0:
                            logger.info(f"[do_resize] Waiting for {self._num_active_sender_threads} sender threads...")
                            self._sender_threads_cv.wait(timeout=5.0)

                    release_weights_immediately = (
                        target_num_layers <= 0
                        or self.vllm_config.dynamic_config.
                        pipeline_autoscaling_enabled)
                    detached_layers = self.remove_layers(
                        self.rank,
                        layer_ranges,
                        release_immediately=release_weights_immediately)
                    deleted_layers_released = None
                    if release_weights_immediately:
                        logger.info(
                            "[delete_layers] rank %s released sender weights "
                            "immediately after sync; target_num_layers=%s "
                            "autoscaling=%s",
                            self.rank, target_num_layers,
                            self.vllm_config.dynamic_config.
                            pipeline_autoscaling_enabled)
                    else:
                        deleted_layers_released = (
                            self._queue_deleted_model_layers_after_inference(
                                detached_layers))
                    with torch.cuda.stream(self.migration_stream):
                        self.release_kv_cache_for_layers(self.rank, caches_to_free_key, caches_to_free_value, ptrs_to_free_key, ptrs_to_free_value, handles_to_free_key, handles_to_free_value, grouped_handles_to_free)
                    if deleted_layers_released is not None:
                        wait_deleted_layers_start = time.time()
                        deleted_layers_released.wait()
                        logger.info(
                            "[delete_layers] background resize waited %s for "
                            "after-inference model weight release",
                            human_readable_duration(
                                time.time() - wait_deleted_layers_start))
                        empty_cache_start = time.time()
                        gc.collect()
                        torch.cuda.empty_cache()
                        logger.info(
                            "[delete_layers] background empty_cache before KV "
                            "resize took %s",
                            human_readable_duration(time.time() -
                                                    empty_cache_start))
                    logger.info(f"[timeline]: after remove layers, time taken: {human_readable_duration(time.time() - time_start)}")
                self.after_migration_total_token = num_total_migration_tokens
                # For sender: directly set applied token count since sender doesn't receive patches
                # For receiver: this will be overwritten by _listen_loop when all patches are applied
                if is_sender:
                    self.after_migration_applied_token_num = num_total_migration_tokens
                fixed_blocks = self.vllm_config.dynamic_config.fixed_num_gpu_blocks
                if fixed_blocks <= 0:
                    with torch.cuda.stream(self.migration_stream):
                        self.resize_kv_cache(new_kv_cache_block_num)
                    stream_sync_start = time.time()
                    self.migration_stream.synchronize()
                    logger.info(
                        "[autoscaling async] migration stream synchronized "
                        "after worker cleanup/resize, took %s",
                        human_readable_duration(time.time() - stream_sync_start))
                else:
                    logger.info(f"fixed_num_gpu_blocks={fixed_blocks}, skipping worker-side resize (would be {new_kv_cache_block_num} blocks), sleep for 4 seconds")
                self.finish_migration()
                self.kv_resizing_done = True
            finally:
                # Always signal completion, even on error
                with self.resizing_done_cv:
                    self.resizing_done = True
                    self.resizing_done_cv.notify_all()
        threading.Thread(target=do_resize, daemon=True).start()
        logger.info(f"[timeline]: finish sync migration kv cache transfer, time taken: {human_readable_duration(time.time() - time_start_to_sync)}")

    def finish_migration(self):
        self.rank_to_layers_ids = {}
        self._clear_autoscale_sender_pointer_view("finish_migration")
        
        # Atomically rebuild and commit PtrTable stacked tensors after migration
        # This ensures all ptr_tensors changes are reflected in a single atomic switch
        from vllm.config import get_current_vllm_config
        vllm_config = get_current_vllm_config()
        if use_direct_ptr_for_runtime(vllm_config):
            time_start = time.time()
            logger.info(f"[timeline]: commit ptr tables after migration take {human_readable_duration(time.time() - time_start)}")
            logger.info(f"finish_migration: committed ptr_tables for flexi_direct")

        # assert self.dynamic_layer_kv_connector.is_all_patch_applied(), "All patch should be applied"

import gc
import os
from typing import TYPE_CHECKING, Optional, Tuple, Union
import threading

import torch
import torch.distributed
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.model_executor.models.dynamic_qwen3 import DynamicQwen3ForCausalLM
from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import report_usage_stats, WorkerMemInfo
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.dynamic_gpu_model_runner import DynamicGPUModelRunner
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment, _check_if_gpu_supports_dtype
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput

from .utils import KVBufferStatus
from vllm.sequence import IntermediateTensors
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.distributed.kv_transfer.kv_connector.dynamic_layer_kv_connector import KVPatchMeta, KVTensorMeta
from bitarray import bitarray
import time
from vllm.distributed.kv_transfer.kv_connector.dynamic_layer_kv_connector import DynamicKVSynchronizer
logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput

class DynamicGPUWorker(Worker):
    def __init__(self, *args, **kwargs):
        logger.info("start to initialize")
        super().__init__(*args, **kwargs)
        # 缓存等待绑定到 forward_context 的 KV tensor（按全局 layer_id）
        self._pending_kv_by_layer: dict[int, torch.Tensor] = {}
        # Debug flag: when enabled, re-raise exceptions for easier debugging
        self._debug_raise: bool = str(os.getenv("VLLM_DEBUG_RAISE", "0")).lower() not in ("0", "", "false", "no")
        self._listen_kv_cache_threads: list[threading.Thread] = []
        self.rank_to_layers_ids: dict[int, list[int]] = {}

        self.migration_in_process: bool = False

        # 独立的条件变量与锁：等待新层加载（避免在等待时占用 forward_lock）
        self._layer_loaded_lock = threading.Lock()
        self._layer_loaded_cv = threading.Condition(self._layer_loaded_lock)

        # Used for waiting all patch applied
        self._all_patch_applied_cv = threading.Condition(threading.Lock()) 

    def _add_layers(self, layer_list: list[Tuple[int, int]]) -> None:
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        logger.info(f"Add Model Layers: {layer_list}")
        self.model_runner.add_layers(layer_list)

        # 刚刚加载的层若之前已经接收到对应 KV，则在此处补绑定
        with self.model_runner.forward_lock:
            start_layer = self.model_runner.model.model.start_layer
            new_start = start_layer
            for lo, hi in layer_list:
                for layer_id in range(lo, hi + 1):
                    if layer_id not in self._pending_kv_by_layer:
                        continue
                    kv_tensor = self._pending_kv_by_layer[layer_id]
                    self.model_runner.bind_layer_kv_tensor(layer_id, kv_tensor)
                    self._pending_kv_by_layer.pop(layer_id, None)
                    new_start = min(new_start, layer_id)

            if new_start != start_layer:
                self.model_runner.model.model.start_layer = new_start

        # 唤醒等待层加载的线程（例如 kv patch 应用线程）。
        # 注意：在 forward_lock 外进行 notify，避免双锁顺序问题导致的潜在死锁。
        with self._layer_loaded_cv:
            self._layer_loaded_cv.notify_all()


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

        # Initialize KV synchronizer AFTER model is ready (needs model for args)
        self.dynamic_layer_kv_connector = DynamicKVSynchronizer(
            rank=self.rank,
            local_rank=self.local_rank,
            config=self.vllm_config,
            model_executable=self.model_runner.model,
        )

        # 启动所有监听其它rank的发送过来的kv cache的线程
        for rank in range(self.vllm_config.parallel_config.pipeline_parallel_size):
            if rank != self.rank:
                threading.Thread(target=self.listen_to_kv_cache_tensor_and_patches, args=(rank,), daemon=True).start()

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
            self.dynamic_layer_kv_connector.send_kv_cache_patch(self.rank_to_layers_ids, slot_mapping, self.model_runner.kv_caches, self.model_runner.model.model.start_layer)
        return result


    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Get the current available memory in bytes.
        """
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        assert self.model_runner.model.get_sched_layers() == (self.model_runner.model.model.start_layer, self.model_runner.model.model.end_layer), "model should be in the initial state"
        return super().determine_available_memory()

    @torch.inference_mode()
    def get_current_available_memory(self) -> int:
        """Get the current available memory in bytes.
        """
        free_gpu_memory, _ = torch.cuda.mem_get_info()
        return int(free_gpu_memory)

    def reinitialize_kv_cache(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        self.model_runner.reinitialize_kv_cache(kv_cache_configs[self.rank])
        return

    def async_add_layers(self, rank: int, layer_list: list[Tuple[int, int]]) -> None:
        if self.rank != rank:
            logger.debug(f"Worker {self.rank} is not the target rank {rank}, skip adding model layers")
            return None
        logger.info(f"Async Add Model Layers: {layer_list}")
        def _do_add():
            self._add_layers(layer_list)
            logger.info(f"Async Add Model Layers done: {layer_list}")
        threading.Thread(target=_do_add, daemon=True).start()


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
        self.model_runner.release_kv_cache_for_layers(layers_list)

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
        # Layer weight size (may raise if not recorded yet)
        layer_size = 0
        if isinstance(self.model_runner.model, DynamicQwen3ForCausalLM):
            try:
                layer_size = int(self.model_runner.model.get_layer_weight_size())
            except Exception:
                if self._debug_raise:
                    raise
                layer_size = 0

        # Free memory from driver
        free_memory, _ = torch.cuda.mem_get_info()

        # Size of a single KV cache tensor (for one layer) from model runner
        kv_tensor_size = 0
        if isinstance(self.model_runner, DynamicGPUModelRunner):
            kv_tensor_size = int(self.model_runner.get_single_kv_tensor_size())

        return WorkerMemInfo(layer_size, int(free_memory), kv_tensor_size)

    def get_kv_buffer_status(self) -> KVBufferStatus:
        """Return KV patch buffer status per rank.

        used_tokens = capacity - size
        free_tokens = size
        capacity_tokens = capacity
        """
        capacity_list = []
        free_tokens_list = []
        used_tokens_list = []
        for peer_rank in self.dynamic_layer_kv_connector.buffers:
            buf = self.dynamic_layer_kv_connector.buffers[peer_rank]
            capacity = int(getattr(buf, "capacity", 0))
            free_tokens = int(getattr(buf, "size", 0))
            used_tokens = max(0, capacity - free_tokens)
            capacity_list.append(capacity)
            free_tokens_list.append(free_tokens)
            used_tokens_list.append(used_tokens)

        return KVBufferStatus(used_tokens_list, free_tokens_list, capacity_list, self.dynamic_layer_kv_connector.patch_ids, self.dynamic_layer_kv_connector.last_applied_patch_ids)

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

    def compact_kv_cache(self, compacted_length: int, bitmap: bitarray) -> None:
        self.model_runner.compact_kv_cache(compacted_length, bitmap)
        self.model_runner.resize_kv_cache(compacted_length)

    def start_kv_cache_migration(self, rank, rank_to_layers_ids: dict[int, list[int]]) -> None:
        """Collective-RPC entry used by executor.

        Only the worker whose `self.rank == source_rank` performs the actual
        send; other ranks are no-ops. The receiver side should already be
        listening via `listen_to_kv_cache_tensor(source_rank)`.
        """
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        if self.rank != rank:
            return None
        logger.info(f"debug: invoking start_kv_cache_migration")
        with self.model_runner.forward_lock:
            self.dynamic_layer_kv_connector.start_kv_cache_transfer(rank_to_layers_ids, self.model_runner.kv_caches, self.model_runner.model.model.start_layer)
            self.migration_in_process = True
            assert len(self.rank_to_layers_ids) == 0
            self.rank_to_layers_ids = rank_to_layers_ids
        return None

    def listen_to_kv_cache_tensor_and_patches(self, from_rank: int) -> None:
        """
        异步监听指定 from_rank 的 KV：先接收完整 KV 张量，再持续接收小 patch，并按需同步。
        """
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        assert self.rank != from_rank, "The rank should not listen to its own kv cache"

        logger.info(f"Worker {self.rank} listening KV stream from rank {from_rank}")

        def _listen_loop():
            def _apply_patch_loop() -> None:
                # Apply the kv patches in the queue
                assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
                # 确保针对 from_rank 的通道与缓冲区已创建，避免首个补丁到达前访问 buffers[from_rank] 抛 KeyError
                # 确保接收方向的通道已就绪
                while True:
                    meta, kv_payload, slot_mapping = self.dynamic_layer_kv_connector.get_one_patch(from_rank)
                    if meta.type == "kv_patch_finished":
                        logger.info(f"Worker {self.rank} received kv patch finished message from rank {from_rank}, stop draining out thread")
                        assert all(self.dynamic_layer_kv_connector.buffers[from_rank].size == 0 for from_rank in self.dynamic_layer_kv_connector.buffers), "All buffers should be empty"
                        # 通知所有等待所有KV cache patch都applied完成的线程
                        with self._all_patch_applied_cv:
                            self._all_patch_applied_cv.notify_all()
                        break

                    # Notice that this thread might be interleaved with the thread that receive kv tensors,
                    # Therefore we need to make sure all the  layers of kv caches loaded before apply the patch
                    for layer_id in meta.layer_ids:
                        while True:
                            with self.model_runner.forward_lock:
                                if self.model_runner.has_layer(layer_id):
                                    break
                            logger.info(f"Worker {self.rank} received kv patch for layer {layer_id} but layer {layer_id} is not loaded, wait until loaded")
                            with self._layer_loaded_cv:
                                self._layer_loaded_cv.wait()

                    with self.model_runner.forward_lock:
                        patch_id = self.dynamic_layer_kv_connector.apply_one_patch(self.model_runner.kv_caches, self.model_runner.model.model.start_layer, meta, kv_payload, slot_mapping)
                        self.dynamic_layer_kv_connector.update_last_applied_patch_id(from_rank, patch_id)

            # 启动后台线程用于持续应用 buffer 中的 patch
            drain_thread = threading.Thread(target=_apply_patch_loop, daemon=False)
            drain_thread.start()

            # 1) Wait and apply the first full batch (kv_stage_batch)
            while True:
                meta = self.dynamic_layer_kv_connector.recv_controller(from_rank)
                logger.info(f"Worker {self.rank} received msg from rank {from_rank}: {meta}")
                if meta.type == "kv_tensor":
                    kv_tensor = self.dynamic_layer_kv_connector._recv_data_from_rank(from_rank, meta.dtype, meta.shape)
                    # 如果此时已经收到layer weights，则直接将kv cache绑定
                    with self.model_runner.forward_lock:
                        if self.model_runner.has_layer(meta.layer_id):
                            self.model_runner.bind_layer_kv_tensor(meta.layer_id, kv_tensor)
                        else:
                            self._pending_kv_by_layer[meta.layer_id] = kv_tensor
                elif meta.type == "kv_patch_meta":
                    self.dynamic_layer_kv_connector.kv_patch_handler(from_rank, meta)
                elif meta.type == "kv_patch_finished":
                    logger.info(f"debug: received kv patch finished from rank {from_rank}")
                    self.dynamic_layer_kv_connector.kv_patch_handler(from_rank, meta)
                    break
                else:
                    assert False, f"Unexpected message type: {meta.type}"


            # 等待排水线程结束，确保所有 patch 已应用
            try:
                drain_thread.join()
            except Exception as e:
                logger.exception(f"Drain thread join failed: {e}")

            logger.info(f"Worker {self.rank} listening KV stream from rank {from_rank} finished, waiting for drain thread to finish")

        threading.Thread(target=_listen_loop, daemon=True).start()
        return None

    def kv_synchronize_before_execute_callback(self):

        # 如果migration没有在执行，则等待所有KV cache patch都applied完成，我们才能使用新的model pp configuration
        if not self.migration_in_process:
            # 在这里等待所有KV cache patch都applied完成
            while True:
                if self.dynamic_layer_kv_connector.is_all_patch_applied():
                    logger.info(f"debug: all kv cache patch are applied for rank {self.rank}")
                    break
                with self._all_patch_applied_cv:
                    logger.info(f"debug: all kv cache patch are not applied for rank {self.rank}, wait for all kv cache patch to be applied")
                    self._all_patch_applied_cv.wait()


    def kv_synchronize_after_execute_callback(self, is_sync: bool):
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)

        # Send the kv cache patch to the other ranks if the migration is in process
        slot_mapping = self.model_runner.input_batch.block_table[0].slot_mapping
        # Ensure slot_mapping is on this worker's CUDA device
        assert slot_mapping.device.type == "cuda", f"slot_mapping must be on CUDA, got {slot_mapping.device}"
        # target_device = self.device  # set in init_device to cuda:self.local_rank
        # if slot_mapping.device != target_device:
        #     slot_mapping = slot_mapping.to(target_device, non_blocking=True)
        if is_sync:
            # Sender finishes transfer and cleans up; receiver no-ops.
            if self.migration_in_process:
                self.dynamic_layer_kv_connector.finish_kv_cache_transfer()

                # 在这个节点，我们可以free旧的layer weights和kv cache
                layer_ranges = []
                for layers in self.rank_to_layers_ids.values():
                    # Assert the layers are sorted
                    assert layers == sorted(layers), "The layers should be sorted"
                    layer_ranges.append((layers[0], layers[-1]))
                self.remove_layers(self.rank, layer_ranges)
                self.release_kv_cache_for_layers(self.rank, layer_ranges)

                self.finish_migration()

        if self.migration_in_process:
            self.dynamic_layer_kv_connector.send_kv_cache_patch(self.rank_to_layers_ids, slot_mapping, self.model_runner.kv_caches, self.model_runner.model.model.start_layer)
            logger.info(f"debug: sent kv cache patch to the other ranks")

    def finish_migration(self):
        self.migration_in_process = False
        self.rank_to_layers_ids = {}

        assert self.dynamic_layer_kv_connector.is_all_patch_applied(), "All patch should be applied"

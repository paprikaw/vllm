import gc
import os
from typing import TYPE_CHECKING, Optional, Tuple, Union
import threading
import math
import torch
import torch.distributed
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.v1.worker.utils import get_total_gpu_memory
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
from vllm.distributed.kv_transfer.kv_connector.dynamic_kv_synchronizer import KVPatchMeta, KVTensorMeta
from bitarray import bitarray
import time
from vllm.distributed.kv_transfer.kv_connector.dynamic_kv_synchronizer import DynamicKVSynchronizer
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

    def _add_layers(self, layer_list: list[Tuple[int, int]]) -> None:
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        logger.info(f"Add Model Layers: {layer_list}")
        with self._layer_loaded_cv:
            # 记录扩容前的起始 layer，便于在 start_layer 左移时重排本地 kv 索引基准
            old_start_layer = self.model_runner.model.model.start_layer
            self.model_runner.add_layers(layer_list)

            # 接收完weights之后，在这里进行有可能的左扩
            with self.model_runner.forward_lock:
                start_layer = self.model_runner.model.model.start_layer
                new_start = start_layer

                # 如果 start_layer 向更小的下标移动，需要对现有 self.kv_caches 做前置填充，
                # 使其索引基准与新的 start_layer 对齐
                shift = int(old_start_layer - start_layer)
                if shift > 0:
                    self.model_runner.kv_caches = [torch.tensor([])] * shift + \
                                                  self.model_runner.kv_caches
                if new_start != start_layer:
                    self.model_runner.model.model.start_layer = new_start

            # 唤醒等待层加载的线程（例如 kv patch 应用线程）。
            # 注意：在 forward_lock 外进行 notify，避免双锁顺序问题导致的潜在死锁。
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
                # threading.Thread(target=self.listen_to_kv_cache_tensor_and_patches, args=(rank,), daemon=True).start()
                threading.Thread(target=self.listen_to_kv_cache_tensor, args=(rank,), daemon=True).start()

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
        assert self.model_runner.model.get_sched_layers() == (self.model_runner.model.model.start_layer, self.model_runner.model.model.end_layer), "model should be in the initial state"
        self.model_runner.profile_run()
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
        logger.info(f"before get_mem_info, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        time_start = time.time()
        torch.cuda.empty_cache()
        logger.info(f" after empty cache, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        # Layer weight size (may raise if not recorded yet)
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        layer_size = int(self.model_runner.model.get_layer_weight_size())
        k_block = self.model_runner.kv_caches[0][0][0]
        block_size = int(k_block.numel() * k_block.element_size() * 2)  # 乘以2包含V


        # Free memory from driver
        free_memory, _ = torch.cuda.mem_get_info()

        # Size of a single KV cache tensor (for one layer) from model runner
        kv_tensor_size = 0
        if isinstance(self.model_runner, DynamicGPUModelRunner):
            kv_tensor_size = int(self.model_runner.get_single_kv_tensor_size())

        # get the size of total gpu memory
        total_gpu_memory = get_total_gpu_memory(self.rank)

        return WorkerMemInfo(layer_size, block_size, kv_tensor_size, int(free_memory), int(total_gpu_memory))

    def get_kv_buffer_status(self) -> KVBufferStatus:
        """Return KV patch buffer status per rank.

        used_tokens = capacity - size
        free_tokens = size
        capacity_tokens = capacity
        """
        capacity_list = {} 
        free_tokens_list = {} 
        used_tokens_list = {}
        for peer_rank in self.dynamic_layer_kv_connector.buffers:
            buf = self.dynamic_layer_kv_connector.buffers[peer_rank]
            capacity = buf.capacity
            free_tokens = buf.size
            used_tokens = max(0, capacity - free_tokens)
            if self.dynamic_layer_kv_connector.is_kv_patch_sending():
                capacity_list[peer_rank] = capacity
                free_tokens_list[peer_rank] = free_tokens
                used_tokens_list[peer_rank] = used_tokens
            else:
                capacity_list[peer_rank] = capacity
                free_tokens_list[peer_rank] = 0
                used_tokens_list[peer_rank] = capacity



        return KVBufferStatus(used_tokens_list, free_tokens_list, capacity_list)

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

    def resize_kv_cache(self, new_length: int) -> None:
        self.model_runner.resize_kv_cache(new_length)

    def start_kv_cache_migration_async(self, src_to_plan: dict[int, dict[int, list[int]]]) -> None:
        """Collective-RPC entry used by executor.

        Only the worker whose `self.rank == source_rank` performs the actual
        send; other ranks are no-ops. The receiver side should already be
        listening via `listen_to_kv_cache_tensor(source_rank)`.
        """
        assert self.migration_in_process == False, "The migration should not be in process"
        assert self.sending_kv_cache_in_process == False, "The sending kv cache should not be in process"

        with self.model_runner.forward_lock:
            self.migration_in_process = True
            if self.rank not in src_to_plan:
                return None

            self.sending_kv_cache_in_process = True

        def migration_thread():
            assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
            rank_to_layers_ids = src_to_plan[self.rank]
            time_start = time.time()
            # with self.model_runner.forward_lock:
            self.dynamic_layer_kv_connector.start_kv_tensor_transfer_async(rank_to_layers_ids, self.model_runner.kv_caches, self.model_runner.model.model.start_layer)
            assert len(self.rank_to_layers_ids) == 0
            self.rank_to_layers_ids = rank_to_layers_ids
            logger.info(f"-------------启动 KV cache 迁移所用时间: {time.time() - time_start:.2f} 秒")

        threading.Thread(target=migration_thread, daemon=True).start()
        return None

    def start_kv_cache_migration_sync(self, src_to_sending_layers: dict[int, dict[int, list[Tuple[int, int]]]], rank_to_layer_ids: dict[int, list[Tuple[int, int]]]) -> None:
        """Collective-RPC entry used by executor.

        Only the worker whose `self.rank == source_rank` performs the actual
        send; other ranks are no-ops. The receiver side should already be
        listening via `listen_to_kv_cache_tensor(source_rank)`.
        """
        time_start = time.time()
        assert self.receive_in_process == False, "The sending kv cache should not be in process"

        with self.model_runner.forward_lock:
            if self.rank in rank_to_layer_ids:
                self.receive_in_process = True
            if self.rank not in src_to_sending_layers:
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
            self.dynamic_layer_kv_connector.start_kv_tensor_transfer_sync(sending_layers_list, self.model_runner.kv_caches, self.model_runner.model.model.start_layer)
            # Asynchronize with cuda stream
            torch.cuda.synchronize()

        for _, layer_ranges in sending_layers_plans.items():
            self.release_kv_cache_for_layers(self.rank, layer_ranges)
            self.remove_layers(self.rank, layer_ranges)
        self.sending_kv_cache_in_process = False

        logger.info(f"-------------启动 KV cache 迁移所用时间: {time.time() - time_start:.2f} 秒")
        return None

    def listen_to_kv_cache_tensor_and_patches(self, from_rank: int) -> None:
        """
        异步监听指定 from_rank 的 KV：先接收完整 KV 张量，再持续接收小 patch，并按需同步。
        """
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        assert self.rank != from_rank, "The rank should not listen to its own kv cache"

        logger.info(f"Worker {self.rank} listening KV stream from rank {from_rank}")

        def _listen_loop():

            assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
            while True:
                received_layer = set()
                layers_to_be_received: set[int]
                while True:
                    # 1) Firstly wait for all tensor the be received
                    meta = self.dynamic_layer_kv_connector.recv_controller(from_rank)
                    if meta.type != "kv_tensor":
                        assert layers_to_be_received == received_layer, "The layers to be received should be the same as the received layers"
                        break
                    layers_to_be_received = meta.layer_to_be_received
                    assert meta.layer_id not in received_layer, "The layer should not be received twice"
                    received_layer.add(meta.layer_id)
                    kv_tensor = self.dynamic_layer_kv_connector._recv_data_from_rank(from_rank, meta.dtype, meta.shape)
                    # 在kv cache绑定前，weight必须loading结束
                    with self._layer_loaded_cv:
                        while not self.model_runner.has_layer(meta.layer_id):
                            self._layer_loaded_cv.wait()
                        # 确认 layer 已经存在，再绑定
                        with self.model_runner.forward_lock:
                            self.model_runner.bind_layer_kv_tensor(meta.layer_id, kv_tensor)
                cur_patch_id = 0
                logger.info(f"--------------shape of the self kv_caches: {len(self.model_runner.kv_caches)}, {self.model_runner.kv_caches[0].shape}")
                # 2) Then wait for all patch to be received
                while True:
                    assert meta.type == "kv_patch_meta" or meta.type == "kv_patch_finished", "The type of the meta should be kv_patch_meta or kv_patch_finished"

                    # Ensure the order of the patch
                    assert meta.id == cur_patch_id, "The patch id should be the next id of the last patch"
                    cur_patch_id += 1

                    if meta.type == "kv_patch_meta":
                        logger.info(f"debug: ------------ Worker {self.rank} received kv patch meta from rank {from_rank}")
                        self.is_all_patch_applied[from_rank] = False
                        pipe = self.dynamic_layer_kv_connector._ensure_pipe_and_buffer(from_rank, 'recv')
                        pipe = self.dynamic_layer_kv_connector._pair_pipes_recv[from_rank]
                        slot_mapping = pipe.recv_data(meta.slot_mapping_dtype, meta.slot_mapping_shape)
                        kv_payload = pipe.recv_data(meta.kv_payload_dtype, meta.kv_payload_shape)
                        assert kv_payload.dim() == 5 and kv_payload.size(0) == 2
                        # 该线程与权重/整层 KV 加载可能交错，因此分两步等待：先等“层已加载”，再等“KV 已绑定”
                        for layer_id in meta.layer_ids:
                            # 先等待层加载完成
                            while True:
                                logger.info(f"Worker {self.rank} received kv patch for layer {layer_id} but layer is not loaded, waiting...")
                                with self._layer_loaded_cv:
                                    with self.model_runner.forward_lock:
                                        if self.model_runner.has_layer(layer_id):
                                            break
                                    self._layer_loaded_cv.wait()

                        self.dynamic_layer_kv_connector.apply_one_patch(self.model_runner.kv_caches, self.model_runner.model.model.start_layer, meta, kv_payload, slot_mapping)
                    elif meta.type == "kv_patch_finished":
                        logger.info(f"Worker {self.rank} received kv patch finished message from rank {from_rank}")
                        self.is_all_patch_applied[from_rank] = True
                        # 通知所有等待所有KV cache patch都applied完成的线程
                        with self._all_patch_applied_cv:
                            self._all_patch_applied_cv.notify_all()
                        break
                    else:
                        assert False, f"Unexpected message type: {meta.type}"
                    meta = self.dynamic_layer_kv_connector.recv_controller(from_rank)

        threading.Thread(target=_listen_loop, daemon=True).start()
        return None
    def listen_to_kv_cache_tensor(self, from_rank: int) -> None:
        """
        异步监听指定 from_rank 的 KV：只接收完整 KV 张量，不考虑patch的情况
        """
        assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
        assert self.rank != from_rank, "The rank should not listen to its own kv cache"

        logger.info(f"Worker {self.rank} listening KV stream from rank {from_rank}")

        def _listen_loop():

            assert isinstance(self.model_runner.model, DynamicQwen3ForCausalLM)
            while True:
                received_layer = set()
                layers_to_be_received: set[int]
                while True:
                    # 1) Firstly wait for all tensor the be received
                    meta = self.dynamic_layer_kv_connector.recv_controller(from_rank)
                    if meta.type != "kv_tensor":
                        assert layers_to_be_received == received_layer, "The layers to be received should be the same as the received layers"
                        break
                    layers_to_be_received = meta.layer_to_be_received
                    assert meta.layer_id not in received_layer, "The layer should not be received twice"
                    received_layer.add(meta.layer_id)
                    kv_tensor = self.dynamic_layer_kv_connector._recv_data_from_rank(from_rank, meta.dtype, meta.shape)
                    # 在kv cache绑定前，weight必须loading结束
                    with self._layer_loaded_cv:
                        while not self.model_runner.has_layer(meta.layer_id):
                            self._layer_loaded_cv.wait()
                        # 确认 layer 已经存在，再绑定
                        with self.model_runner.forward_lock:
                            self.model_runner.bind_layer_kv_tensor(meta.layer_id, kv_tensor)
                logger.info(f"debug: ---------- listen loop: receive all kv tensor, migration is over")
                with self._sync_migration_cv:
                    assert self.receive_in_process
                    self.receive_in_process = False
                    self._sync_migration_cv.notify_all()


        threading.Thread(target=_listen_loop, daemon=True).start()
        return None
    def all_patch_applied(self) -> bool:
        return all(self.is_all_patch_applied.values())

    def kv_synchronize_before_execute_callback(self,
                                               is_sync_after_migration: bool
                                               ) -> None:
        # 处于迁移中时，或这是同步批（用于发送 finished 信号），直接放行。
        if self.migration_in_process:
            logger.info(f"debug: rank {self.rank} is in migration or is sync after migration, skip kv synchronize before execute callback")
            return

        # 等待所有KV cache补丁都应用完毕后再使用新的pp配置
        with self._all_patch_applied_cv:
            # logger.info(f"debug: ---------------------rank {self.rank} waiting for patch to be all applied")
            while not self.all_patch_applied():
                logger.info(f"debug: ---------------------all kv cache patch are not applied for rank {self.rank}, wait for all kv cache patch to be applied, is_all_patch_applie: {self.all_patch_applied()}")
                self._all_patch_applied_cv.wait()

    def sync_migration_before_execute_callback(self, new_kv_cache_block_num: int) -> None:
        logger.info("debug --- new_kv_cache_block_num: {new_kv_cache_block_num}")
        with self._sync_migration_cv:
            while self.receive_in_process:
                logger.info(f"debug: ---------------------rank {self.rank} waiting for migration to be finished")
                self._sync_migration_cv.wait()
        if new_kv_cache_block_num != 0:
            self.resize_kv_cache(new_kv_cache_block_num)

    def kv_synchronize_after_execute_callback(self, is_sync: bool, new_kv_cache_block_num: int):
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
                self.dynamic_layer_kv_connector.put_kv_patch_to_buffer(
                    rank,
                    self.model_runner.kv_caches,
                    self.rank_to_layers_ids[rank],
                    self.model_runner.model.model.start_layer,
                    slot_mapping,
                )
            logger.info(
                f"debug: ---------------------sent kv cache patch to all ranks, takes time: {time.time() - time_start:.2f} seconds")

        if is_sync:
            # Sender finishes transfer and cleans up; receiver no-ops.
            assert new_kv_cache_block_num != 0, "The new kv cache block num should not be 0"
            if self.sending_kv_cache_in_process:
                logger.info(f"debug: finish kv cache transfer for rank {self.rank}")
                self.dynamic_layer_kv_connector.finish_kv_cache_transfer()

                # 在这个节点，我们可以free旧的layer weights和kv cache
                layer_ranges = []
                for layers in self.rank_to_layers_ids.values():
                    # Assert the layers are sorted
                    assert layers == sorted(layers), "The layers should be sorted"
                    layer_ranges.append((layers[0], layers[-1]))
                self.release_kv_cache_for_layers(self.rank, layer_ranges)
                self.remove_layers(self.rank, layer_ranges)
            self.resize_kv_cache(new_kv_cache_block_num)
            self.finish_migration()

    def finish_migration(self):
        self.migration_in_process = False
        self.sending_kv_cache_in_process = False
        self.rank_to_layers_ids = {}

        # assert self.dynamic_layer_kv_connector.is_all_patch_applied(), "All patch should be applied"

# SPDX-License-Identifier: Apache-2.0
"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The SimpleConnector transfers KV caches between prefill vLLM worker (KV cache
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
from typing import TYPE_CHECKING, Optional, Union, Dict, Tuple, Literal
import threading
import os

import torch
from pydantic import TypeAdapter

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.dynamic_utils import (
    kv_synchronizer_helper as kv_helper,
    KVPatch,
    KVPatchMeta,
    KVTensorMeta,
)
from vllm.logger import init_logger
from vllm.distributed.utils import StatelessProcessGroup
import time
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)

# ===================== Pydantic models for control metadata =====================
class KVPatchBuffer:
    def __init__(self, size: int):
        self.buffer: list[KVPatch] = []
        # 总容量（token 数）与剩余可用容量（token 数）
        self.capacity = int(size)
        self.size = int(size)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def add_patch(self, patch: KVPatch) -> bool:
        token_num = 0
        assert patch.meta.type == "kv_patch_meta" or patch.meta.type == "kv_patch_finished", "The type of the meta should be kv_patch_meta or kv_patch_finished"
        meta, kv_payload, slot_mapping = patch.meta, patch.kv_payload, patch.slot_mapping
        if meta.type == "kv_patch_meta":
            # 校验 kv_payload 的维度: 期望 [2, L, T, H, D]
            assert patch.kv_payload.dim() == 5, (
                f"kv_payload must be 5D [2, L, T, H, D], got {kv_payload.dim()}D with shape {tuple(kv_payload.shape)}")
            assert patch.kv_payload.size(0) == 2, (
                f"kv_payload first dim must be 2 (K/V), got {kv_payload.size(0)} with shape {tuple(kv_payload.shape)}")

            assert kv_payload.size(2) == slot_mapping.size(0), "The number of tokens in the kv_payload and slot_mapping must be the same"

            token_num = int(slot_mapping.size(0))

        with self._cv:
            # 当剩余容量不足时阻塞等待
            while self.size < token_num:
                logger.info(f"the size of the buffer is {self.size}, the token num is {token_num}, wait for the buffer to be free")
                self._cv.wait()

            # 检查 meta id 顺序
            if len(self.buffer) > 0:
                last_id = self.buffer[-1].meta.id
                assert patch.meta.id == last_id + 1, "The id of the patch is not the next id of the last patch"

            self.buffer.append(patch)
            self.size -= token_num
            self._cv.notify_all()
            logger.info(f"add patch to buffer, the size of the buffer is {self.size}, the token num is {token_num}")
            return True

    def pop_patch(self) -> KVPatch:
        with self._cv:
            # 当缓冲区为空时阻塞等待
            while not self.buffer:
                self._cv.wait()
            patch = self.buffer.pop(0)
            token_num = int(patch.slot_mapping.size(0))
            self.size += token_num
            # 通知可能等待容量的生产者
            self._cv.notify_all()
            return patch


class PairPipe:
    """A minimal 1:1 NCCL pipe between two actors (world_size=2).

    - Uses a StatelessProcessGroup(TCPStore) for rendezvous + metadata.
    - Uses PyNcclCommunicator for GPU tensor p2p send/recv.
    - Sender/receiver roles are symmetric; peer is 1 - pair_rank.
    """

    def __init__(self, local_rank: int, host: str, port: int, pair_rank: int, store_timeout_s: int = 30000, device: Optional[torch.device] = None):
        assert pair_rank in (0, 1)
        self.pair_rank = pair_rank
        self.peer_rank = 1 - pair_rank
        # 发送/接收张量所使用的设备
        self.device = device if device is not None else torch.device("cuda", local_rank)
        logger.info(f"initilized {pair_rank} rank to with port {port} and ip {host}")
        # Rendezvous store (metadata/control plane)
        self.meta_group = StatelessProcessGroup.create(host=host,
                                                  port=port,
                                                  rank=pair_rank,
                                                  world_size=2,
                                                  store_timeout=store_timeout_s)
        self.data_group = StatelessProcessGroup.create(host=host,
                                                  port=port+100,
                                                  rank=pair_rank,
                                                  world_size=2,
                                                  store_timeout=store_timeout_s)
        # Ensure both sides are ready
        self.meta_group.barrier()
        self.data_group.barrier()


        self.is_use_nccl = not os.getenv("KV_SYNC_USE_CPU", "0") == "1"
        if self.is_use_nccl:
        # NCCL data-plane communicator
            self._nccl = PyNcclCommunicator(group=self.meta_group, device=local_rank)
        else:
            self._nccl = None

    def _prepare_recv_buffer(self, dtype: torch.dtype, shape: torch.Size) -> torch.Tensor:
        assert dtype is not None and shape is not None
        return torch.empty(shape, dtype=dtype, device=self.device)

    def send_obj(self, obj: Union[KVTensorMeta, KVPatchMeta]) -> None:
        self.meta_group.send_obj(obj, dst=self.peer_rank)

    def recv_obj(self) -> Union[KVTensorMeta, KVPatchMeta]:
        obj = self.meta_group.recv_obj(src=self.peer_rank)
        assert isinstance(obj, (KVTensorMeta, KVPatchMeta)), "The object should be a KVTensorMeta or KVPatchMeta"
        return obj

    def send_data(self, tensor: torch.Tensor) -> None:
        if self.is_use_nccl:
            assert self._nccl is not None, "The nccl communicator should be initialized"
            dev_tensor = tensor.to(self.device)
            self._nccl.send(dev_tensor, dst=self.peer_rank)
        else:
            self.data_group.send_obj(tensor, dst=self.peer_rank)

    def recv_data(self, dtype: torch.dtype, shape: torch.Size) -> torch.Tensor:
        """Data-plane receive: allocate buffer and perform NCCL recv.

        Args:
            metadata: dict with dtype/shape (from recv_metadata_once).
        Returns:
            Received tensor on local GPU.
        """
        if self.is_use_nccl:
            assert self._nccl is not None, "The nccl communicator should be initialized"
            buf = self._prepare_recv_buffer(dtype, shape)
            self._nccl.recv(buf, src=self.peer_rank)
        else:
            buf = self.data_group.recv_obj(src=self.peer_rank)
            assert isinstance(buf, torch.Tensor), "The object should be a torch.Tensor"

        return buf

    # ===================== request/response helpers (token range) =====================


    def barrier(self) -> None:
        self.meta_group.barrier()
        self.data_group.barrier()

    def close(self) -> None:
        # StatelessProcessGroup has no explicit close API for TCPStore; rely on GC.
        # PyNcclCommunicator also cleans up on GC; no explicit destroy required here.
        try:
            self.meta_group.barrier()
            self.data_group.barrier()
        except Exception:
            pass

class DynamicKVSynchronizer():

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
        model_executable: torch.nn.Module,
    ):

        # 使用专门的 LayerKVConnector 配置
        self._vllm_config = config
        self.config = config.layer_kv_connector_config
        self.kv_helper = kv_helper(config)
        self.kv_synchronizer_helper = kv_helper(config)
        self.model_executable = model_executable
        self.kv_synchronizer_helper.model_executable = model_executable
        logger.info("Initializing DynamicKVSynchronizer with config %s",
                    self.config)
        self.rank = rank
        self.local_rank = local_rank
        # 每个 peer_rank 对应两个方向的控制/数据通道
        # send: 本 rank -> peer_rank； recv: peer_rank -> 本 rank
        self._pair_pipes_send: Dict[int, PairPipe] = {}
        self._pair_pipes_recv: Dict[int, PairPipe] = {}
        self.buffers: Dict[int, KVPatchBuffer] = {}
        # Pydantic adapters for control metadata
        self._control_adapter = TypeAdapter(Union[KVTensorMeta,
                                                  KVPatchMeta])

        pp_size = int(self._vllm_config.parallel_config.pipeline_parallel_size)
        # 目前这个是每一个synchronizer对应一个buffer，后续可能可以共享buffer
        self.kv_cache_transfer_in_process: dict[int, bool] = {} 
        self.last_patch_ids: Dict[int, int] = {} # rank -> patch_id
        for peer in range(pp_size):
            if peer == self.rank:
                continue
            self.kv_cache_transfer_in_process[peer] = False
            self.last_patch_ids[peer] = 0

        self.sending_synchronizer_lock = threading.Lock()
        self.recv_synchronizer_lock = threading.Lock()
        self.kv_patch_sending = False
        # Eagerly initialize NCCL pipes for all peers (both directions).
        # This avoids first-use latency and surfaces connectivity issues early.
        try:
            self._initialize_all_pipes()
        except Exception:
            logger.exception("Eager NCCL link initialization failed; will retry lazily on demand.")

    def _initialize_all_pipes(self) -> None:
        """Initialize send/recv pipes to all peer ranks upfront.

        This creates both control-plane stores and data-plane communicators
        for each peer in both directions, and allocates receive buffers.
        Also marks peers as having no pending patches initially.
        """
        # Determine world size from vLLM parallel configuration (PP ranks).
        pp_size = int(self._vllm_config.parallel_config.pipeline_parallel_size)

        for peer in range(pp_size):
            if peer == self.rank:
                continue
            try:
                # Deterministic complementary ordering per pair to avoid
                # rendezvous deadlocks: smaller rank does SEND then RECV;
                # larger rank does RECV then SEND.
                if self.rank < peer:
                    self._ensure_pipe_and_buffer(peer, 'send')
                    self._ensure_pipe_and_buffer(peer, 'recv')
                else:
                    self._ensure_pipe_and_buffer(peer, 'recv')
                    self._ensure_pipe_and_buffer(peer, 'send')
                # Mark peer as initially having no pending patches.
            except Exception:
                logger.exception(f"Failed to initialize NCCL pipes with peer {peer}")
                raise

    def _pair_key(self, peer_rank: int) -> Tuple[int, int]:
        me = self.rank
        return (me, peer_rank) if me < peer_rank else (peer_rank, me)

    def _pair_port(self, peer_rank: int, direction: Literal['send', 'recv']) -> int:
        """Deterministically derive a TCP port for (self.rank, peer_rank).

        Use a symmetric mapping so both sides compute the same port.
        If torch.distributed is initialized, prefer the true world size.
        Otherwise, fall back to a symmetric function that only depends on the
        two ranks (works before process group init).
        """
        base = self.config.kv_port
        a, b = self._pair_key(peer_rank)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                ws = torch.distributed.get_world_size()
            except Exception:
                ws = max(a, b) + 1
        else:
            ws = max(a, b) + 1
        pair_id = a * ws + b
        # 为每一对 (a,b) 预留两个端口：0 -> a->b; 1 -> b->a
        # 如果我向对端发送（self->peer），dir_bit 取决于 self 和 a/b 的关系
        if direction == 'send':
            dir_bit = 0 if self.rank == a else 1
        else:  # 'recv': peer -> self
            # 源是 peer_rank
            dir_bit = 0 if peer_rank == a else 1
        return base + 2 * pair_id + dir_bit

    def _get_rank_ip(self, rank: int) -> str:
        # 优先使用 rank_to_ip，否则回退到 kv_ip 或 127.0.0.1
        host = None
        mapping = getattr(self.config, 'rank_to_ip', {}) or {}
        assert rank in mapping
        return mapping[rank]

    def _ensure_pipe_and_buffer(self, peer_rank: int, direction: Literal['send','recv']) -> "PairPipe":
        lock = self.sending_synchronizer_lock if direction == 'send' else self.recv_synchronizer_lock
        with lock:
            # 如果已有则直接返回
            if direction == 'send' and peer_rank in self._pair_pipes_send:
                return self._pair_pipes_send[peer_rank]
            if direction == 'recv' and peer_rank in self._pair_pipes_recv:
                return self._pair_pipes_recv[peer_rank]
            logger.info(f"initilizing {direction} from rank {self.rank} to rank {peer_rank}")
            logger.info(f"----------------debug start to initialize {direction} from rank {self.rank} to rank {peer_rank}")
            # 设备检查
            base_device = self._vllm_config.device_config.device
            assert isinstance(base_device, torch.device)
            assert base_device.type == "cuda"
            device = torch.device("cuda", self.local_rank)

            # 确定本通道的“源”端（作为 TCPStore server 的一侧）
            src_rank = self.rank if direction == 'send' else peer_rank
            dst_rank = peer_rank if direction == 'send' else self.rank
            # 在 2-rank 组内：src -> pair_rank 0, dst -> pair_rank 1
            pair_rank = 0 if self.rank == src_rank else 1
            port = self._pair_port(peer_rank, direction)
            server_ip = self._get_rank_ip(src_rank)

            pipe = PairPipe(local_rank=self.local_rank,
                             host=server_ip,
                             port=port,
                             pair_rank=pair_rank,
                             store_timeout_s=self.config.store_timeout_s,
                             device=device)
            max_kv_patch_buffer_size = int(os.environ.get('MAX_KV_PATCH_BUFFER_SIZE', 5000))
            if direction == 'send':
                self._pair_pipes_send[peer_rank] = pipe
                self.buffers[peer_rank] = KVPatchBuffer(max_kv_patch_buffer_size)
            else:
                self._pair_pipes_recv[peer_rank] = pipe
            return pipe

    def _send_meta_to_rank(self, rank: int, meta: Union[KVTensorMeta, KVPatchMeta]) -> None:
        pipe = self._ensure_pipe_and_buffer(rank, 'send')
        pipe = self._pair_pipes_send[rank]
        pipe.send_obj(meta)

    def _send_data_to_rank(self, rank: int, kv_cache: torch.Tensor) -> None:
        pipe = self._ensure_pipe_and_buffer(rank, 'send')
        pipe = self._pair_pipes_send[rank]
        pipe.send_data(kv_cache)


    def _recv_metadata_from_rank(self, rank: int) -> Union[KVTensorMeta, KVPatchMeta]:
        """Block to receive one metadata entry from peer.

        Returns a dict with keys: "dtype", "shape", "layer_id".
        """
        pipe = self._ensure_pipe_and_buffer(rank, 'recv')
        pipe = self._pair_pipes_recv[rank]
        return self._control_adapter.validate_python(pipe.recv_obj())

    def _recv_data_from_rank(self, rank: int,
                               dtype: torch.dtype,
                               shape: torch.Size,
                               ) -> torch.Tensor:
        """Given previously received metadata, receive tensor payload via NCCL.

        Args:
            rank: peer global rank.
            metadata: dict returned by recv_kv_metadata_from_rank.
        Returns:
            Received tensor on local GPU.
        """
        pipe = self._ensure_pipe_and_buffer(rank, 'recv')
        pipe = self._pair_pipes_recv[rank]
        return pipe.recv_data(dtype, shape)

    def get_recv_pipe(self, rank: int) -> PairPipe:
        self._ensure_pipe_and_buffer(rank, 'recv')
        return self._pair_pipes_recv[rank]

    def get_send_pipe(self, rank: int) -> PairPipe:
        self._ensure_pipe_and_buffer(rank, 'send')
        return self._pair_pipes_send[rank]

    def start_kv_tensor_transfer_sync(self, rank_to_layers_ids: dict[int, list[int]], kv_caches: list[torch.Tensor],
                              start_layer_id: int) -> None:
        """Send layers' KV cache to a peer rank.
            In each of the migration process, this function should only be called once.
        Args:
            rank: peer global rank to send to.
            kv_cache: tensor to send (GPU or CPU tensor; will be moved to local GPU).
            start_layer_id: which layer this KV cache belongs to (for receiver to demux).
            layer_ids: which layers this KV cache belongs to (for receiver to demux).
        """
        assert all(transfer_in_process == False for transfer_in_process in self.kv_cache_transfer_in_process.values()), "In each of the migration process, this function should only be called once."
        assert all(patch_id == 0 for patch_id in self.last_patch_ids.values()), "The patch id of the rank should be 0."

        for rank, layer_ids in rank_to_layers_ids.items():
            for layer_id in layer_ids:
                logger.info(f"rank {self.rank} send kv cache to rank {rank} for layer {layer_id}")
                local_layer_id = layer_id - start_layer_id
                kv_cache = kv_caches[local_layer_id]
                # 第一次访问时默认置为 False，避免 KeyError
                logger.info(f"rank {self.rank} send kv tensor meta to rank {rank} for layer {layer_id}")
                self._send_meta_to_rank(rank, KVTensorMeta(
                    type='kv_tensor',
                    layer_to_be_received=set(layer_ids),
                    layer_id=int(layer_id),
                    num_tokens=int(kv_cache.size(0)),
                    dtype=kv_cache.dtype,
                    shape=kv_cache.shape))
                logger.info(f"rank {self.rank} send kv tensor data to rank {rank} for layer {layer_id}")
                self._send_data_to_rank(rank, kv_cache)
                self.kv_cache_transfer_in_process[rank] = True
            # Only to tell the receiver the kv patch have been
            self._send_meta_to_rank(rank, KVPatchMeta(
                type='kv_patch_meta',
                id=0,
                layer_ids=layer_ids,
                num_tokens=int(kv_cache.size(0)),
                slot_mapping_dtype=torch.int64,
                slot_mapping_shape=torch.Size([]),
                kv_payload_dtype=torch.int64,
                kv_payload_shape=torch.Size([]))
            )

        return None
        
    def start_kv_tensor_transfer_async(self, rank_to_layers_ids: dict[int, list[int]], kv_caches: list[torch.Tensor],
                              start_layer_id: int) -> None:
        """Send layers' KV cache to a peer rank.
            In each of the migration process, this function should only be called once.
        Args:
            rank: peer global rank to send to.
            kv_cache: tensor to send (GPU or CPU tensor; will be moved to local GPU).
            start_layer_id: which layer this KV cache belongs to (for receiver to demux).
            layer_ids: which layers this KV cache belongs to (for receiver to demux).
        """
        assert all(transfer_in_process == False for transfer_in_process in self.kv_cache_transfer_in_process.values()), "In each of the migration process, this function should only be called once."
        assert all(patch_id == 0 for patch_id in self.last_patch_ids.values()), "The patch id of the rank should be 0."

        for rank, layer_ids in rank_to_layers_ids.items():
            for layer_id in layer_ids:
                logger.info(f"rank {self.rank} send kv cache to rank {rank} for layer {layer_id}")
                local_layer_id = layer_id - start_layer_id
                kv_cache = kv_caches[local_layer_id]
                # 第一次访问时默认置为 False，避免 KeyError
                logger.info(f"rank {self.rank} send kv tensor meta to rank {rank} for layer {layer_id}")
                self._send_meta_to_rank(rank, KVTensorMeta(
                    type='kv_tensor',
                    layer_to_be_received=set(layer_ids),
                    layer_id=int(layer_id),
                    num_tokens=int(kv_cache.size(0)),
                    dtype=kv_cache.dtype,
                    shape=kv_cache.shape))
                logger.info(f"rank {self.rank} send kv tensor data to rank {rank} for layer {layer_id}")
                self._send_data_to_rank(rank, kv_cache)
                self.kv_cache_transfer_in_process[rank] = True
            
        # 在发送完kv tensor之后开始发送kv patch
        def patch_sending_thread(rank: int):
            while True:
                kv_patch = self.buffers[rank].pop_patch()
                self.kv_patch_sending = True
                self._send_meta_to_rank(rank, kv_patch.meta)
                self._send_data_to_rank(rank, kv_patch.slot_mapping)
                self._send_data_to_rank(rank, kv_patch.kv_payload)
                if kv_patch.meta.type == "kv_patch_finished":
                    self.kv_patch_sending = False
                    self.kv_cache_transfer_in_process[rank] = False
                    break

        for rank in rank_to_layers_ids:
            threading.Thread(target=patch_sending_thread, args=(rank,), daemon=True).start()


    def add_patch_to_send_buffer(self, rank: int, kv_patch: KVPatch) -> None:
        self.buffers[rank].add_patch(kv_patch)
        self.last_patch_ids[rank] += 1

    def get_last_patch_id(self, rank: int) -> int:
        return self.last_patch_ids[rank]

    def send_kv_cache_patch(self, 
            target_rank: int,
            kv_patches: KVPatch) -> None:

        assert self.kv_cache_transfer_in_process[target_rank] == True, "The kv cache transfer in process of the rank should be True."
        if target_rank not in self.last_patch_ids:
            self.last_patch_ids[target_rank] = 0
        self.last_patch_ids[target_rank] += 1
        meta, kv_payload, slot_mapping = kv_patches.meta, kv_patches.kv_payload, kv_patches.slot_mapping
        self._send_meta_to_rank(target_rank, meta)
        self._send_data_to_rank(target_rank, slot_mapping)
        self._send_data_to_rank(target_rank, kv_payload)

        if meta.type == "kv_patch_finished":
            self.kv_cache_transfer_in_process[target_rank] = False

    def finish_kv_cache_transfer(self) -> None:
        for rank in self.last_patch_ids:
            meta = KVPatchMeta(
                type='kv_patch_finished',
                id=int(self.last_patch_ids[rank]),
                layer_ids=[],
                num_tokens=0,
                slot_mapping_dtype=torch.int64,
                slot_mapping_shape=torch.Size([]),
                kv_payload_dtype=torch.int64,
                kv_payload_shape=torch.Size([]),
            )
            self.buffers[rank].add_patch(KVPatch(meta, torch.empty(0, device='cpu'), torch.empty(0, device='cpu')))
        self.last_patch_ids = {}

    def recv_controller(self, rank: int) -> Union[KVTensorMeta, KVPatchMeta]:
        pipe = self._ensure_pipe_and_buffer(rank, 'recv')
        pipe = self._pair_pipes_recv[rank]
        meta_obj = pipe.recv_obj()
        return self._control_adapter.validate_python(meta_obj)

    def get_one_patch(self, rank: int) -> KVPatch:
        # Ensure the recv pipe and buffer exist before popping
        kv_patch = self.buffers[rank].pop_patch()
        return kv_patch

    def put_kv_patch_to_buffer(self, rank: int, kv_caches: list[torch.Tensor], layer_ids: list[int], start_layer_id: int, slot_mapping: torch.Tensor) -> None:
        time_start = time.time()
        kv_patch = self.kv_synchronizer_helper.extract_kv_patch_from_kv_cache(self.last_patch_ids[rank], kv_caches, layer_ids, start_layer_id, slot_mapping)
        time_after_extract_kv_patch = time.time()
        # logger.info(f"debug: ---------------------extract kv patch time: {time_after_extract_kv_patch - time_start:.2f} seconds")
        assert kv_patch is not None
        self.last_patch_ids[rank] += 1
        self.buffers[rank].add_patch(kv_patch)

    def apply_one_patch(self, kv_caches: list[torch.Tensor], start_layer_id: int, meta: KVPatchMeta, kv_payload: torch.Tensor, slot_mapping: torch.Tensor) -> int:
        keys = kv_payload[0]
        values = kv_payload[1]
        logger.info(f"apply one patch with id {meta.id}, num_tokens: {meta.num_tokens}")
        # slot_mapping 已经被裁剪过，只包含有效的 token
        assert slot_mapping.size(0) == meta.num_tokens, f"slot_mapping size {slot_mapping.size(0)} should match num_tokens {meta.num_tokens}"
        for layer_id, key, value in zip(meta.layer_ids, keys, values):
            local_layer_id = layer_id - start_layer_id
            kv_cache = kv_caches[local_layer_id]
            # 使用裁剪后的 slot_mapping，从 0 到 num_tokens
            self.kv_helper.put_kv_to_cache(self.model_executable, key, value, layer_id, kv_cache, slot_mapping, 0, meta.num_tokens)
        return meta.id
    def is_kv_patch_sending(self) -> bool:
        return self.kv_patch_sending
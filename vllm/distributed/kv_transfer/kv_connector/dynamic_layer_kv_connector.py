# SPDX-License-Identifier: Apache-2.0
"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The SimpleConnector transfers KV caches between prefill vLLM worker (KV cache
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Union, Callable, Dict, Tuple, cast, List, Literal
import threading
import time
from contextlib import contextmanager
import os

import torch
from pydantic import BaseModel, TypeAdapter, ConfigDict

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    model_aware_kv_ops_helper as kv_helper,
    kv_synchronizer_helper)
from vllm.logger import init_logger
from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)

# ===================== Pydantic models for control metadata =====================
class KVPatchMeta(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal['kv_patch_meta', 'kv_patch_finished']
    id: int
    layer_ids: List[int]
    num_tokens: int
    slot_mapping_dtype: torch.dtype
    slot_mapping_shape: torch.Size
    kv_payload_dtype: torch.dtype
    kv_payload_shape: torch.Size

class KVTensorMeta(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal['kv_tensor']
    layer_id: int
    num_tokens: int
    dtype: torch.dtype
    shape: torch.Size

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
        self.kv_synchronizer_helper = kv_synchronizer_helper()
        self.model_executable = model_executable
        logger.info("Initializing DynamicKVSynchronizer with config %s",
                    self.config)
        self.rank = rank
        self.local_rank = local_rank
        # 每个 peer_rank 对应两个方向的控制/数据通道
        # send: 本 rank -> peer_rank； recv: peer_rank -> 本 rank
        self._pair_pipes_send: Dict[int, _PairPipe] = {}
        self._pair_pipes_recv: Dict[int, _PairPipe] = {}

        # Pydantic adapters for control metadata
        self._control_adapter = TypeAdapter(Union[KVTensorMeta,
                                                  KVPatchMeta])

        # 目前这个是每一个synchronizer对应一个buffer，后续可能可以共享buffer
        self.buffers: Dict[int, KVPatchBuffer] = {}
        self.all_patch_applied: Dict[int, bool] = defaultdict(bool)
        self.kv_cache_transfer_in_process: dict[int, bool] = defaultdict(bool)
        self.patch_ids: Dict[int, int] = {} # rank -> patch_id
        self.last_applied_patch_ids: Dict[int, int] = defaultdict(int) # rank -> last_applied_patch_id
        self.sending_synchronizer_lock = threading.Lock()
        self.recv_synchronizer_lock = threading.Lock()

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

    def _ensure_pipe_and_buffer(self, peer_rank: int, direction: Literal['send','recv']) -> "_PairPipe":
        lock = self.sending_synchronizer_lock if direction == 'send' else self.recv_synchronizer_lock
        with lock:
            # 如果已有则直接返回
            if direction == 'send' and peer_rank in self._pair_pipes_send:
                return self._pair_pipes_send[peer_rank]
            if direction == 'recv' and peer_rank in self._pair_pipes_recv:
                return self._pair_pipes_recv[peer_rank]
            logger.info(f"initilizing {direction} from rank {self.rank} to rank {peer_rank}")
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

            pipe = _PairPipe(local_rank=self.local_rank,
                             host=server_ip,
                             port=port,
                             pair_rank=pair_rank,
                             store_timeout_s=self.config.store_timeout_s,
                             device=device)

            if direction == 'send':
                self._pair_pipes_send[peer_rank] = pipe
            else:
                self._pair_pipes_recv[peer_rank] = pipe
                # 仅在接收方向初始化缓冲区
                if peer_rank not in self.buffers:
                    buffer_size = int(os.environ.get("KV_BUFFER_SIZE", "3000"))
                    self.buffers[peer_rank] = KVPatchBuffer(buffer_size)
            return pipe

    def _send_meta_to_rank(self, rank: int, meta: Union[KVTensorMeta, KVPatchMeta]) -> None:
        pipe = self._ensure_pipe_and_buffer(rank, 'send')
        pipe.send_obj(meta)

    def _send_data_to_rank(self, rank: int, kv_cache: torch.Tensor) -> None:
        pipe = self._ensure_pipe_and_buffer(rank, 'send')
        pipe.send_data(kv_cache)


    def _recv_metadata_from_rank(self, rank: int) -> Union[KVTensorMeta, KVPatchMeta]:
        """Block to receive one metadata entry from peer.

        Returns a dict with keys: "dtype", "shape", "layer_id".
        """
        pipe = self._ensure_pipe_and_buffer(rank, 'recv')
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
        return pipe.recv_data(dtype, shape)

    def is_recv_buffer_all_empty(self) -> bool:
        return all(len(kv_buffer.buffer) == 0 for kv_buffer in self.buffers.values())

    def start_kv_cache_transfer(self, rank_to_layers_ids: dict[int, list[int]], kv_caches: list[torch.Tensor],
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
        assert all(patch_id == 0 for patch_id in self.patch_ids.values()), "The patch id of the rank should be 0."

        for rank, layer_ids in rank_to_layers_ids.items():
            for layer_id in layer_ids:
                logger.info(f"rank {self.rank} send kv cache to rank {rank} for layer {layer_id}")
                local_layer_id = layer_id - start_layer_id
                kv_cache = kv_caches[local_layer_id]
                # 第一次访问时默认置为 False，避免 KeyError
                logger.info(f"rank {self.rank} send kv tensor meta to rank {rank} for layer {layer_id}")
                self._send_meta_to_rank(rank, KVTensorMeta(
                    type='kv_tensor',
                    layer_id=int(layer_id),
                    num_tokens=int(kv_cache.size(0)),
                    dtype=kv_cache.dtype,
                    shape=kv_cache.shape))
                logger.info(f"rank {self.rank} send kv tensor data to rank {rank} for layer {layer_id}")
                self._send_data_to_rank(rank, kv_cache)
                self.kv_cache_transfer_in_process[rank] = True


    def send_kv_cache_patch(self, 
            rank_to_layers_ids: dict[int, list[int]], 
            slot_mapping: torch.Tensor, 
            kv_caches: list[torch.Tensor], 
            start_layer_id: int) -> None:
        for rank, layer_ids in rank_to_layers_ids.items():
            assert self.kv_cache_transfer_in_process[rank] == True, "The kv cache transfer in process of the rank should be True."
            if rank not in self.patch_ids:
                self.patch_ids[rank] = 0
            self._send_kv_cache_patch_to_rank(rank, layer_ids, start_layer_id, slot_mapping, kv_caches)
            self.patch_ids[rank] += 1

    # ===================== stage-aggregated (multi-layer) send/recv =====================
    def _send_kv_cache_patch_to_rank(self,
                         rank: int,
                         layer_ids: list[int],
                         start_layer_id: int,
                         slot_mapping: torch.Tensor,
                         kv_caches: Optional[list[torch.Tensor]] = None,
                         ) -> None:
        """Aggregate multiple layers' KV for this stage and send in one shot.

        Prefer passing `kv_caches` and `dest_slot_mapping` only; this method
        will extract per-layer K/V slices from the kv caches. `K_all`/`V_all`
        are accepted for backward compatibility.

        Args:
            peer_rank: destination global rank to send to.
            batch_id: monotonically increasing id for ordering.
            layer_ids: layers contained in this batch; if `kv_caches` is used,
                it should be aligned with this order.
            dest_slot_mapping: int64 tensor [T] with destination slot indices.
            K_all, V_all: optional prebuilt tensors [L, T, H, D].
            hidden: optional tensor [T, hidden_size].
            kv_caches: optional list of per-layer kv_cache tensors; when
                provided, K/V will be extracted internally using
                `slot_mapping`.
        """
        assert kv_caches is not None, (
            "Provide either (K_all,V_all) or kv_caches.")
        # Derive shapes using model config and helper flags.

        num_heads, head_size = self.kv_helper.get_model_args(self.model_executable)
        # Peek the first cache to get dtype/device.
        k0, v0 = self.kv_helper.get_kv_from_cache(kv_caches[0], num_heads,
                                                  head_size)

        # Preallocate KV tensor [2, L, T, H, D] and fill per layer for speed.
        L = len(layer_ids)
        T = int(slot_mapping.numel())

        # Ensure index lives on same device for fast gather.
        if slot_mapping.device != k0.device:
            slot_mapping = slot_mapping.to(k0.device, non_blocking=True)

        KV_shape = (2, int(L), int(T), int(num_heads), int(head_size))
        # Prefer K dtype; cast V if needed when filling
        KV_all = torch.empty(KV_shape, dtype=k0.dtype, device=k0.device)

        for idx, layer_id in enumerate(layer_ids):
            local_layer_id = int(layer_id - start_layer_id)
            kv_cache = kv_caches[local_layer_id]
            key_cache, value_cache = self.kv_helper.get_kv_from_cache(
                kv_cache, num_heads, head_size)

            assert slot_mapping.device == key_cache.device

            # slot_mapping may contain PAD_SLOT_ID (-1) entries. PyTorch
            # index_select does not accept negative indices. We therefore:
            # 1) Gather only valid indices (>= 0 and < cache_len).
            # 2) Create output buffers of shape [T, H, D].
            # 3) Scatter gathered rows to valid positions; fill invalid with 0.
            cache_len = int(key_cache.size(0))
            valid_mask = (slot_mapping >= 0) & (slot_mapping < cache_len)
            num_valid = int(valid_mask.sum().item())

            # Preallocate outputs on device
            selected_k = torch.zeros((T, int(num_heads), int(head_size)),
                                     dtype=key_cache.dtype,
                                     device=key_cache.device)
            selected_v = torch.zeros((T, int(num_heads), int(head_size)),
                                     dtype=value_cache.dtype,
                                     device=value_cache.device)

            if num_valid > 0:
                valid_pos = torch.nonzero(valid_mask, as_tuple=False).flatten()
                valid_idx = slot_mapping.index_select(0, valid_pos)
                gathered_k = torch.index_select(key_cache, 0, valid_idx)
                gathered_v = torch.index_select(value_cache, 0, valid_idx)
                # Scatter back to their original positions
                selected_k.index_copy_(0, valid_pos, gathered_k)
                selected_v.index_copy_(0, valid_pos, gathered_v)
            else:
                # All positions are PAD; keep zeros to be ignored by receiver
                pass

            if num_valid != T:
                # Some positions are PAD or out-of-range; receiver will ignore
                # these via PAD_SLOT_ID in slot_mapping.
                missing = T - num_valid
                logger.debug(
                    "rank %d layer %d: %d/%d slots are PAD or invalid; filled with zeros",
                    self.rank, layer_id, missing, T)

            assert selected_k.device == KV_all.device and selected_k.dtype == KV_all.dtype
            assert selected_v.device == KV_all.device and selected_v.dtype == KV_all.dtype

            # 按 layer_ids 在批次中的顺序写入 KV_all 第二维
            KV_all[0, idx] = selected_k
            KV_all[1, idx] = selected_v

        assert KV_all is not None
        assert KV_all.dim() == 5 and KV_all.size(0) == 2, (
            "KV_all must have shape [2, L, T, H, D]")
        _, L2, T2, H, D = KV_all.shape
        assert int(L2) == len(layer_ids) and int(T2) == T

        meta = KVPatchMeta(
            type='kv_patch_meta',
            id=int(self.patch_ids[rank]),
            layer_ids=list(map(int, layer_ids)),
            num_tokens=int(T),
            slot_mapping_dtype=slot_mapping.dtype,
            slot_mapping_shape=slot_mapping.shape,
            kv_payload_dtype=KV_all.dtype,
            kv_payload_shape=KV_all.shape,
        )
        self._send_meta_to_rank(rank, meta)
        self._send_data_to_rank(rank, slot_mapping)
        self._send_data_to_rank(rank, KV_all)

    def finish_kv_cache_transfer(self) -> None:
        for rank in self.patch_ids:
            meta = KVPatchMeta(
                type='kv_patch_finished',
                id=int(self.patch_ids[rank]),
                layer_ids=[],
                num_tokens=0,
                slot_mapping_dtype=torch.int64,
                slot_mapping_shape=torch.Size([]),
                kv_payload_dtype=torch.int64,
                kv_payload_shape=torch.Size([]),
            )
            self._send_meta_to_rank(rank, meta)
            self.kv_cache_transfer_in_process[rank] = False
        self.patch_ids = {}

    def recv_controller(self, rank: int) -> Union[KVTensorMeta, KVPatchMeta]:
        pipe = self._ensure_pipe_and_buffer(rank, 'recv')
        meta_obj = pipe.recv_obj()
        return self._control_adapter.validate_python(meta_obj)

    def kv_patch_handler(self, rank: int,
                                 meta: KVPatchMeta) -> None:
        """Given control metadata, receive tensor payload via NCCL.

        Returns (dest_slot_mapping, K_all, V_all, hidden)
        """
        if meta.type == "kv_patch_meta":
            logger.info(f"debug: recv kv patch meta from rank {rank}")
            pipe = self._ensure_pipe_and_buffer(rank, 'recv')

            slot_mapping = pipe.recv_data(meta.slot_mapping_dtype, meta.slot_mapping_shape)
            kv_payload = pipe.recv_data(meta.kv_payload_dtype, meta.kv_payload_shape)
            assert kv_payload.dim() == 5 and kv_payload.size(0) == 2
            self.all_patch_applied[rank] = False
            self.buffers[rank].add_patch(meta, kv_payload, slot_mapping)
        elif meta.type == "kv_patch_finished":
            logger.info(f"debug: recv kv patch finished from rank {rank}")
            # 创建完全空的CPU tensor，避免调用GPU
            self.buffers[rank].buffer.append((meta, torch.empty(0, device='cpu'), torch.empty(0, device='cpu')))
        else:
            assert False, f"Unexpected message type: {meta.type}"

    def update_last_applied_patch_id(self, rank: int, patch_id: int) -> None:
        if patch_id != 0:
            assert patch_id == self.last_applied_patch_ids[rank] + 1, "The patch id should be the next id of the last applied patch"
        self.last_applied_patch_ids[rank] = patch_id

    def get_one_patch(self, rank: int) -> tuple[KVPatchMeta, torch.Tensor, torch.Tensor]:
        # Ensure the recv pipe and buffer exist before popping
        self._ensure_pipe_and_buffer(rank, 'recv')
        meta, kv_payload, slot_mapping = self.buffers[rank].pop_patch()
        if meta.type == "kv_patch_finished":
            assert len(self.buffers[rank].buffer) == 0, "The buffer should be empty after all patches are applied"
            self.all_patch_applied[rank] = True
        return meta, kv_payload, slot_mapping

    def apply_one_patch(self, kv_caches: list[torch.Tensor], start_layer_id: int, meta: KVPatchMeta, kv_payload: torch.Tensor, slot_mapping: torch.Tensor) -> int:
        keys = kv_payload[0]
        values = kv_payload[1]
        logger.info(f"apply one patch with id {meta.id}")
        for layer_id, key, value in zip(meta.layer_ids, keys, values):
            local_layer_id = layer_id - start_layer_id
            kv_cache = kv_caches[local_layer_id]
            self.kv_helper.put_kv_to_cache(self.model_executable, key, value, layer_id, kv_cache, slot_mapping, 0, slot_mapping.size(0))
        return meta.id
    
    def is_all_patch_applied(self) -> bool:
        return all(self.all_patch_applied.values())


class KVPatchBuffer:
    def __init__(self, size: int):
        self.buffer: list[tuple[KVPatchMeta, torch.Tensor, torch.Tensor]] = []
        # 总容量（token 数）与剩余可用容量（token 数）
        self.capacity = int(size)
        self.size = int(size)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def add_patch(self, meta: KVPatchMeta, kv_payload: torch.Tensor, slot_mapping: torch.Tensor) -> bool:
        # 校验 kv_payload 的维度: 期望 [2, L, T, H, D]
        assert kv_payload.dim() == 5, (
            f"kv_payload must be 5D [2, L, T, H, D], got {kv_payload.dim()}D with shape {tuple(kv_payload.shape)}")
        assert kv_payload.size(0) == 2, (
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
                last_id = self.buffer[-1][0].id
                assert meta.id == last_id + 1, "The id of the patch is not the next id of the last patch"

            self.buffer.append((meta, kv_payload, slot_mapping))
            self.size -= token_num
            # 通知可能等待容量的消费者/生产者
            logger.info(f"notify the consumers/producers to apply the patch")
            self._cv.notify_all()
            return True

    def pop_patch(self) -> tuple[KVPatchMeta, torch.Tensor, torch.Tensor]:
        with self._cv:
            # 当缓冲区为空时阻塞等待
            while not self.buffer:
                self._cv.wait()
            meta, kv_payload, slot_mapping = self.buffer.pop(0)
            token_num = int(slot_mapping.size(0))
            self.size += token_num
            # 通知可能等待容量的生产者
            self._cv.notify_all()
            return meta, kv_payload, slot_mapping


class _PairPipe:
    """A minimal 1:1 NCCL pipe between two actors (world_size=2).

    - Uses a StatelessProcessGroup(TCPStore) for rendezvous + metadata.
    - Uses PyNcclCommunicator for GPU tensor p2p send/recv.
    - Sender/receiver roles are symmetric; peer is 1 - pair_rank.
    """

    def __init__(self, local_rank: int, host: str, port: int, pair_rank: int, store_timeout_s: int = 300, device: Optional[torch.device] = None):
        assert pair_rank in (0, 1)
        self.pair_rank = pair_rank
        self.peer_rank = 1 - pair_rank
        # 发送/接收张量所使用的设备
        self.device = device if device is not None else torch.device("cuda", local_rank)
        logger.info(f"initilized {pair_rank} rank to with port {port} and ip {host}")
        # Rendezvous store (metadata/control plane)
        self.group = StatelessProcessGroup.create(host=host,
                                                  port=port,
                                                  rank=pair_rank,
                                                  world_size=2,
                                                  store_timeout=store_timeout_s)
        # Ensure both sides are ready
        self.group.barrier()

        # NCCL data-plane communicator
        self._nccl = PyNcclCommunicator(group=self.group, device=local_rank)

    def _prepare_recv_buffer(self, dtype: torch.dtype, shape: torch.Size) -> torch.Tensor:
        assert dtype is not None and shape is not None
        return torch.empty(shape, dtype=dtype, device=self.device)

    def send_obj(self, obj: Union[KVTensorMeta, KVPatchMeta]) -> None:
        self.group.send_obj(obj, dst=self.peer_rank)

    def recv_obj(self) -> Union[KVTensorMeta, KVPatchMeta]:
        obj = self.group.recv_obj(src=self.peer_rank)
        assert isinstance(obj, (KVTensorMeta, KVPatchMeta)), "The object should be a KVTensorMeta or KVPatchMeta"
        return obj

    def send_data(self, tensor: torch.Tensor) -> None:
        dev_tensor = tensor.to(self.device)
        self._nccl.send(dev_tensor, dst=self.peer_rank)

    def recv_data(self, dtype: torch.dtype, shape: torch.Size) -> torch.Tensor:
        """Data-plane receive: allocate buffer and perform NCCL recv.

        Args:
            metadata: dict with dtype/shape (from recv_metadata_once).
        Returns:
            Received tensor on local GPU.
        """
        buf = self._prepare_recv_buffer(dtype, shape)
        self._nccl.recv(buf, src=self.peer_rank)
        return buf

    # ===================== request/response helpers (token range) =====================


    def barrier(self) -> None:
        self.group.barrier()

    def close(self) -> None:
        # StatelessProcessGroup has no explicit close API for TCPStore; rely on GC.
        # PyNcclCommunicator also cleans up on GC; no explicit destroy required here.
        try:
            self.group.barrier()
        except Exception:
            pass

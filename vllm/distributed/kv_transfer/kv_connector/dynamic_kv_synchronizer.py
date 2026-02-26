# SPDX-License-Identifier: Apache-2.0
"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The SimpleConnector transfers KV caches between prefill vLLM worker (KV cache
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
from typing import TYPE_CHECKING, Generator, Optional, Union, Dict, Tuple, Literal
import threading
import os

import bitarray
from httpx import patch
from responses import start
import torch
from pydantic import TypeAdapter
from vllm import _custom_ops as ops

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.dynamic_utils import (
    FlexiKVTensorMeta,
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
            logger.info(f"[operation]: add kv patch to buffer, the size of the buffer is {self.size}, the token num is {token_num}, kv patch tensor size: {kv_payload.shape}")
            # 当剩余容量不足时阻塞等待
            while self.size < token_num:
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

class KVSlotMapping:
    def __init__(self, size: int):
        self.bitmap = bitarray.bitarray(size)
        self.bitmap.setall(0)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self.is_finished = False
        self.stored_tokens = 0

    def add_slot_mappings(self, slot_mapping: list[int], is_finished: bool, num_total_new_tokens: int) -> None:
        with self._cv:
            trimmed_slot_mapping = []
            for ele in slot_mapping:
                if ele < 0:
                    break
                trimmed_slot_mapping.append(ele)

            assert num_total_new_tokens == len(trimmed_slot_mapping), f"The num_total_new_tokens {num_total_new_tokens} is not equal to the length of trimmed_slot_mapping {len(trimmed_slot_mapping)}"

            self.bitmap[trimmed_slot_mapping] = 1
            self.is_finished = is_finished
            self.stored_tokens += num_total_new_tokens
            logger.info(f"[num tokens]: sender side add len(trimmed_slot_mapping): {len(trimmed_slot_mapping)} to synchronizer, total stored tokens: {self.stored_tokens}")
            self._cv.notify_all()

    def get_all_slot_mappings(self) -> Tuple[list[int], bool, int]:
        with self._cv:
            while self.bitmap.count() == 0:
                self._cv.wait()
            slot_mapping = list(self.bitmap.search(1))
            stored_tokens = self.stored_tokens
            is_finished = self.is_finished

            self.bitmap.setall(0)
            self.stored_tokens = 0
        return slot_mapping, is_finished, stored_tokens

class PairPipe:
    """A minimal 1:1 NCCL pipe between two actors (world_size=2).

    - Uses a StatelessProcessGroup(TCPStore) for rendezvous + metadata.
    - Uses PyNcclCommunicator for GPU tensor p2p send/recv.
    - Sender/receiver roles are symmetric; peer is 1 - pair_rank.
    """

    def __init__(self, local_rank: int, host: str, port: int, pair_rank: int, 
                 store_timeout_s: int = 30000, device: Optional[torch.device] = None,
                 nccl_lock: Optional[threading.Lock] = None):
        assert pair_rank in (0, 1)
        self.pair_rank = pair_rank
        self.peer_rank = 1 - pair_rank
        # 发送/接收张量所使用的设备
        self.device = device if device is not None else torch.device("cuda", local_rank)
        logger.info(f"initilized {pair_rank} rank to with port {port} and ip {host}")
        
        # NCCL lock to prevent deadlock between KV synchronizer and Ray compiled_dag
        self._nccl_lock = nccl_lock
        
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
        # Separate signal group for kv_patch ready signals
        # This avoids protocol conflict where "ack" could be received by recv_meta()
        self.signal_group = StatelessProcessGroup.create(host=host,
                                                  port=port+200,
                                                  rank=pair_rank,
                                                  world_size=2,
                                                  store_timeout=store_timeout_s)
        # Ensure both sides are ready
        self.meta_group.barrier()
        self.data_group.barrier()
        self.signal_group.barrier()


        self.is_use_nccl = not os.getenv("KV_SYNC_USE_CPU", "0") == "1"
        logger.info(f"[debug]: is_use_nccl: {self.is_use_nccl}")
        if self.is_use_nccl:
        # NCCL data-plane communicator
            self._nccl = PyNcclCommunicator(group=self.meta_group, device=local_rank)
            logger.info(f"[PairPipe.__init__] NCCL created: pair_rank={pair_rank}, peer_rank={self.peer_rank}, nccl_rank={self._nccl.rank}, port={port}, host={host}")
            # 创建专用的 CUDA stream 用于 KV 传输，避免与模型计算的 stream 冲突
            self._kv_transfer_stream = torch.cuda.Stream(device=self.device)
        else:
            self._nccl = None
            self._kv_transfer_stream = None

    def _prepare_recv_buffer(self, dtype: torch.dtype, shape: torch.Size) -> torch.Tensor:
        assert dtype is not None and shape is not None
        return torch.empty(shape, dtype=dtype, device=self.device)

    def send_meta(self, obj: Union[KVTensorMeta, KVPatchMeta, FlexiKVTensorMeta]) -> None:
        self.meta_group.send_obj(obj, dst=self.peer_rank)

    def recv_meta(self) -> Union[KVTensorMeta, KVPatchMeta, FlexiKVTensorMeta]:
        obj = self.meta_group.recv_obj(src=self.peer_rank)
        assert isinstance(obj, (KVTensorMeta, KVPatchMeta, FlexiKVTensorMeta)), "The object should be a KVTensorMeta, KVPatchMeta, or FlexiKVTensorMeta"
        return obj

    def wait_ready_for_kv_patch(self) -> None:
        logger.info(f"[PairPipe.wait_ready_for_kv_patch] Waiting for ready signal from peer {self.peer_rank}")
        obj = self.signal_group.recv_obj(src=self.peer_rank)
        assert obj == "ack", f"Expected 'ack' but got {obj}"
        return

    def notify_ready_for_kv_patch(self) -> None:
        logger.info(f"[PairPipe.notify_ready_for_kv_patch] Notifying peer {self.peer_rank} that ready")
        self.signal_group.send_obj("ack", dst=self.peer_rank)
        return

    def send_data(self, tensor: torch.Tensor, stream=None, wait_for_ack: bool = False) -> None:
        """发送数据，可以指定使用的 CUDA stream，并可选择等待接收方确认
        
        Protocol for deadlock prevention:
        1. Send meta (already done by caller via send_obj)
        2. If wait_for_ack: wait for receiver's ACK (receiver has acquired lock)
        3. Acquire NCCL lock
        4. NCCL send
        5. Release lock
        
        This ensures that when we do NCCL send, the receiver is already holding
        the lock and waiting for our data, so no other NCCL operation can interfere.
        
        Args:
            tensor: 要发送的 tensor
            stream: 可选的 CUDA stream，如果为 None 则使用默认行为
            wait_for_ack: 如果为 True，先等待接收方的 ACK 确认再发送数据
        """
        logger.info(f"[PairPipe.send_data] Starting send to peer {self.peer_rank}, tensor_shape={tensor.shape}, is_use_nccl={self.is_use_nccl}, wait_for_ack={wait_for_ack}")
        
        # 如果需要等待确认，先接收对方的 ACK（表示对方已经 acquire lock 并准备好接收）
        
        if self.is_use_nccl:
            assert self._nccl is not None, "The nccl communicator should be initialized"
            dev_tensor = tensor.to(self.device)
            logger.info(f"[PairPipe.send_data] Calling NCCL send, nccl_rank={self._nccl.rank}, peer_rank={self.peer_rank}")
            # Acquire NCCL lock to prevent deadlock with Ray compiled_dag's NCCL operations
            # if wait_for_ack:
                # logger.info(f"[PairPipe.send_data] Waiting for ACK from peer {self.peer_rank} before sending")
                # ack = self.meta_group.recv_obj(src=self.peer_rank)
                # assert ack == "SYC+ACK", f"Expected ACK, got {ack}"
                # self.meta_group.send_obj("ACK", dst=self.peer_rank)

                # logger.info(f"[PairPipe.send_data] Received ACK from peer {self.peer_rank}")
            self._nccl.send(dev_tensor, dst=self.peer_rank, stream=stream)
            # Synchronize within the lock to ensure the NCCL operation completes
            # if stream is not None:
            #     stream.synchronize()
            # else:
            #     torch.cuda.synchronize(device=dev_tensor.device)
            #     logger.info(f"[PairPipe.send_data] NCCL lock released after send")
            logger.info(f"[PairPipe.send_data] NCCL send completed")
        else:
            self.data_group.send_obj(tensor, dst=self.peer_rank)

    def recv_data(self, dtype: torch.dtype, shape: torch.Size, stream=None, send_ack: bool = False) -> torch.Tensor:
        """Data-plane receive: allocate buffer and perform NCCL recv.

        Protocol for deadlock prevention:
        1. Receive meta (already done by caller via recv_obj)
        2. If send_ack: acquire NCCL lock FIRST
        3. If send_ack: send ACK to sender (sender is waiting for this before sending)
        4. NCCL recv (lock is held, preventing other NCCL ops)
        5. Release lock
        
        This ensures that when we do NCCL recv, we hold the lock and the sender
        knows we are ready. The sender will then acquire lock and send.
        
        Args:
            dtype: 数据类型
            shape: 数据形状
            send_ack: 如果为 True，在 NCCL recv 之前先 acquire lock 并发送 ACK
        Returns:
            Received tensor on local GPU.
        """
        logger.info(f"[PairPipe.recv_data] Starting recv from peer {self.peer_rank}, shape={shape}, dtype={dtype}, is_use_nccl={self.is_use_nccl}, send_ack={send_ack}")
        if self.is_use_nccl:
            assert self._nccl is not None, "The nccl communicator should be initialized"
            buf = self._prepare_recv_buffer(dtype, shape)
            logger.info(f"[PairPipe.recv_data] Buffer prepared, calling NCCL recv, nccl_rank={self._nccl.rank}, peer_rank={self.peer_rank}")
            # Protocol: acquire lock BEFORE sending ACK, then recv within lock
            logger.info(f"[PairPipe.recv_data] Acquiring NCCL lock before sending ACK")
            time_start = time.time()
            # Send ACK while holding lock - this tells sender we're ready
            from vllm.v1.utils import human_readable_duration
            logger.info(f"[PairPipe.recv_data] Acquiring NCCL lock using {human_readable_duration(time_start - time.time())} and sending ACK to peer {self.peer_rank} (lock held)")
            # self.meta_group.send_obj("SYC+ACK", dst=self.peer_rank)
            # ack = self.meta_group.recv_obj(src=self.peer_rank)
            # assert ack == "ACK", f"Expected ACK, got {ack}"
                
            # Now recv - sender will acquire lock and send after receiving our ACK
            self._nccl.recv(buf, src=self.peer_rank, stream=stream)
            # NCCL recv is async - must synchronize within the lock
            if stream is not None:
                stream.synchronize()
            else:
                torch.cuda.synchronize(device=buf.device)
            logger.info(f"[PairPipe.recv_data] NCCL lock released after recv")
            # else:
            #     # No ACK needed, but still use lock for safety
            #     logger.info(f"[PairPipe.recv_data] Acquiring NCCL lock for recv (no ACK)")
            #     self._nccl.recv(buf, src=self.peer_rank, stream=stream)
            #     if stream is not None:
            #         stream.synchronize()
            #     else:
            #         torch.cuda.synchronize(device=buf.device)
            #     logger.info(f"[PairPipe.recv_data] NCCL lock released after recv")
            logger.info(f"[PairPipe.recv_data] NCCL recv completed and synchronized")
        else:
            buf = self.data_group.recv_obj(src=self.peer_rank)
            assert isinstance(buf, torch.Tensor), "The object should be a torch.Tensor"
            # For non-NCCL mode, send ACK after recv if needed
            if send_ack:
                self.meta_group.send_obj("ACK", dst=self.peer_rank)

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
        device: torch.device,
        config: VllmConfig,
        model_executable: torch.nn.Module,
        nccl_lock: threading.Lock
    ):

        # 使用专门的 LayerKVConnector 配置
        self.vllm_config = config
        self.config = config.layer_kv_connector_config
        self.kv_helper = kv_helper(config)
        self.num_heads, self.head_size = self.kv_helper.get_model_args(model_executable)
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
        self.slot_mappings: Dict[int, KVSlotMapping] = {}

        # Pydantic adapters for control metadata
        self._control_adapter = TypeAdapter(Union[KVTensorMeta,
                                                  KVPatchMeta,
                                                  FlexiKVTensorMeta])

        pp_size = int(self.vllm_config.parallel_config.pipeline_parallel_size)
        # 目前这个是每一个synchronizer对应一个buffer，后续可能可以共享buffer
        self.kv_cache_transfer_in_process: dict[int, bool] = {} 
        self.last_patch_ids: Dict[int, int] = {} # rank -> patch_id
        for peer in range(pp_size):
            if peer == self.rank:
                continue
            self.kv_cache_transfer_in_process[peer] = False
            self.last_patch_ids[peer] = 0

        # Initialize device
        self.device = device
        self.sending_synchronizer_lock = threading.Lock()
        self.recv_synchronizer_lock = threading.Lock()
        
        # NCCL lock to prevent deadlock between KV synchronizer and Ray compiled_dag
        # This lock ensures that only one NCCL operation can be executed at a time
        # in the same process. If not provided, create a local one.
        self._nccl_lock = nccl_lock
        # Eagerly initialize NCCL pipes for all peers (both directions).
        # This avoids first-use latency and surfaces connectivity issues early.
        try:
            self._initialize_all_pipes()
        except Exception:
            logger.exception("Eager NCCL link initialization failed; will retry lazily on demand.")

        self.kv_caches = []
        self.key_cache_list = []
        self.value_cache_list = []
        self.key_cache_ptrs = []
        self.value_cache_ptrs = []
    
    def get_nccl_lock(self) -> threading.Lock:
        """Return the NCCL lock for external use (e.g., Ray compiled_dag)."""
        return self._nccl_lock

    def create_slot_mappings(self, num_block: int) -> None:
        """Initialize slot mapping for each peer rank.

        Args:
            num_block: Maximum number of token blocks that can be stored.
        """
        pp_size = int(self.vllm_config.parallel_config.pipeline_parallel_size)

        for peer in range(pp_size):
            if peer == self.rank:
                continue
            self.slot_mappings[peer] = KVSlotMapping(num_block)

    def _initialize_all_pipes(self) -> None:
        """Initialize send/recv pipes to all peer ranks upfront.

        This creates both control-plane stores and data-plane communicators
        for each peer in both directions, and allocates receive buffers.
        Also marks peers as having no pending patches initially.
        """
        # Determine world size from vLLM parallel configuration (PP ranks).
        pp_size = int(self.vllm_config.parallel_config.pipeline_parallel_size)

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
                             device=self.device,
                             nccl_lock=self._nccl_lock)
            max_kv_patch_buffer_size = int(os.environ.get('MAX_KV_PATCH_BUFFER_SIZE', 5000))
            if direction == 'send':
                self._pair_pipes_send[peer_rank] = pipe
                self.buffers[peer_rank] = KVPatchBuffer(max_kv_patch_buffer_size)
            else:
                self._pair_pipes_recv[peer_rank] = pipe
            return pipe

    def _send_meta_to_rank(self, rank: int, meta: Union[KVTensorMeta, KVPatchMeta, FlexiKVTensorMeta]) -> None:
        pipe = self._ensure_pipe_and_buffer(rank, 'send')
        pipe = self._pair_pipes_send[rank]
        pipe.send_meta(meta)

    def _send_data_to_rank(self, rank: int, kv_cache: torch.Tensor, synchronize: bool = True, wait_for_ack: bool = False) -> None:
        """发送 KV 数据到指定 rank
        
        Args:
            rank: 目标 rank
            kv_cache: 要发送的 KV cache tensor
            synchronize: 是否等待 CUDA 操作完成（本地同步）
            wait_for_ack: 是否等待接收方确认（远程同步）
        """
        pipe = self._ensure_pipe_and_buffer(rank, 'send')
        pipe = self._pair_pipes_send[rank]
        
        # 使用专用的 KV 传输 stream（如果有），避免与模型计算 stream 冲突
        if pipe.is_use_nccl and pipe._kv_transfer_stream is not None:
            # 使用专用 stream 发送
            pipe.send_data(kv_cache, stream=pipe._kv_transfer_stream, wait_for_ack=wait_for_ack)
            # 如果需要同步，只等待这个专用 stream
            if synchronize:
                pipe._kv_transfer_stream.synchronize()
        else:
            pipe.send_data(kv_cache, wait_for_ack=wait_for_ack)
            # 如果需要同步但没有专用 stream，等待整个设备
            if synchronize:
                torch.cuda.synchronize(device=kv_cache.device)


    def _recv_metadata_from_rank(self, rank: int) -> Union[KVTensorMeta, KVPatchMeta]:
        """Block to receive one metadata entry from peer.

        Returns a dict with keys: "dtype", "shape", "layer_id".
        """
        pipe = self._ensure_pipe_and_buffer(rank, 'recv')
        pipe = self._pair_pipes_recv[rank]
        return self._control_adapter.validate_python(pipe.recv_meta())

    def _recv_data_from_rank(self, rank: int,
                               dtype: torch.dtype,
                               shape: torch.Size,
                               send_ack: bool = False) -> torch.Tensor:
        """Given previously received metadata, receive tensor payload via NCCL.

        Args:
            rank: peer global rank
            dtype: tensor 数据类型
            shape: tensor 形状
            send_ack: 是否在接收完成后发送 ACK 确认
        Returns:
            Received tensor on local GPU.
        """
        with self.device:
            torch.cuda.set_device(self.device)
            pipe = self._ensure_pipe_and_buffer(rank, 'recv')
            pipe = self._pair_pipes_recv[rank]
            return pipe.recv_data(dtype, shape, stream=pipe._kv_transfer_stream, send_ack=send_ack)

    def get_recv_pipe(self, rank: int) -> PairPipe:
        self._ensure_pipe_and_buffer(rank, 'recv')
        return self._pair_pipes_recv[rank]

    def get_send_pipe(self, rank: int) -> PairPipe:
        self._ensure_pipe_and_buffer(rank, 'send')
        return self._pair_pipes_send[rank]
    
    # ########################################## #
    # Sender side functions                      #
    # ########################################## #
    def send_kv_tensor_sync(self, rank_to_layers_ids: dict[int, list[int]], kv_caches: list[torch.Tensor],
                              start_layer_id: int, slot_mapping: Optional[torch.Tensor] = None) -> None:
        """Send layers' KV cache to a peer rank (synchronous version).
            In each of the migration process, this function should only be called once.
            
            This function supports both regular and flexi KV cache formats.
            For flexi format, slot_mapping is required to gather KV data from pointer-based storage.
            
        Args:
            rank_to_layers_ids: Mapping of target rank to layer IDs to send.
            kv_caches: list of KV cache tensors (used for regular format).
            start_layer_id: which layer this KV cache belongs to (for receiver to demux).
            slot_mapping: Optional tensor of slot indices. Required for flexi format.
                         If None for flexi format, no data is transferred (only signals are sent).
        """
        assert all(transfer_in_process == False for transfer_in_process in self.kv_cache_transfer_in_process.values()), "In each of the migration process, this function should only be called once."
        assert all(patch_id == 0 for patch_id in self.last_patch_ids.values()), "The patch id of the rank should be 0."
        
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        torch.cuda.synchronize()
        
        for rank, layer_ids in rank_to_layers_ids.items():
            if is_flexi:
                self._flexi_send_kv_tensors_sync(rank, layer_ids, start_layer_id, slot_mapping)
            else:
                self._regular_send_kv_tensors_sync(rank, layer_ids, kv_caches, start_layer_id)
            # 在这里发送是为了符合目前的协议设计
            self.wait_for_kv_patch(rank) 
            self.send_sync_finished_to_rank(rank, layer_ids)
        return None

    def _regular_send_kv_tensors_sync(self, rank: int, layer_ids: list[int], 
                                       kv_caches: list[torch.Tensor], start_layer_id: int) -> None:
        """Send KV tensors using regular (non-flexi) format."""
        for layer_id in layer_ids:
            logger.info(f"[debug]: rank {self.rank} send kv cache to rank {rank} for layer {layer_id}")
            local_layer_id = layer_id - start_layer_id
            logger.info(f"[debug]:rank {self.rank} local_layer_id: {local_layer_id}")
            kv_cache = kv_caches[local_layer_id]
            logger.info(f"[debug]: rank {self.rank} send kv tensor meta to rank {rank} for layer {layer_id}")
            self._send_meta_to_rank(rank, KVTensorMeta(
                type='kv_tensor',
                layer_to_be_received=set(layer_ids),
                layer_id=int(layer_id),
                num_tokens=int(kv_cache.size(0)),
                dtype=kv_cache.dtype,
                shape=kv_cache.shape))
            logger.info(f"[debug]: rank {self.rank} send kv tensor data to rank {rank} for layer {layer_id}")
            self._send_data_to_rank(rank, kv_cache, synchronize=False, wait_for_ack=True)
            self.kv_cache_transfer_in_process[rank] = True

    def _flexi_send_kv_tensors_sync(self, rank: int, layer_ids: list[int], 
                                     start_layer_id: int, slot_mapping: Optional[torch.Tensor]) -> None:
        """Send KV tensors using flexi format with pointer-based storage.
        
        For sync migration (e.g., set_pp_config), slot_mapping is empty because
        there are no running requests. We send empty KV tensors and the receiver
        will create fresh, empty KV caches for the new layers.
        """
        # Get page metadata for gathering KV data
        assert len(self.key_cache_ptrs) > 0, "key_cache_ptrs not initialized for flexi mode"
        assert len(self.value_cache_ptrs) > 0, "value_cache_ptrs not initialized for flexi mode"
        
        # Get dimensions from config
        num_heads = self.num_heads
        head_dim = self.head_size
        
        # For sync migration, slot_mapping should be empty (no running requests)
        # We send empty tensors and receiver will create fresh KV caches
        slot_mapping_tensor = torch.empty(0, dtype=torch.int64, device='cuda')
        
        logger.info(f"[flexi sync]: rank {self.rank} sending to rank {rank}, layers={layer_ids}, num_tokens=0 (empty KV for sync)")
        
        for layer_id in layer_ids:
            # Create empty KV tensor - receiver will create fresh KV cache
            kv_out = torch.empty(2, 0, num_heads, head_dim, dtype=torch.float16, device='cuda')
            
            kv_tensor_meta = FlexiKVTensorMeta(
                type='kv_tensor',
                layer_to_be_received=set(layer_ids),
                layer_id=int(layer_id),
                num_tokens=0,
                slot_mapping_dtype=slot_mapping_tensor.dtype,
                slot_mapping_shape=slot_mapping_tensor.shape,
                kv_payload_dtype=kv_out.dtype,
                kv_payload_shape=kv_out.shape)
            
            # Use send_kv_tensor_to_rank interface
            self.send_kv_tensor_to_rank(rank, kv_tensor_meta, kv_out, slot_mapping_tensor)


    # ########################################## #
    # Sender side functions                      #
    # ########################################## #

    def get_kv_patch(self, rank: int, start_layer_id: int, layer_ids: list[int], kv_meta: torch.Tensor) -> Generator[KVPatch, None, None]:
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        if is_flexi:
            return self._flexi_get_patch(rank,
                                            layer_ids=layer_ids,
                                            kv_cache_meta=kv_meta,
                                            key_cache_ptrs=self.key_cache_ptrs,
                                            value_cache_ptrs=self.value_cache_ptrs,
                                            start_layer_id=start_layer_id)
        else:
            return self._get_patch(rank,
                                    layer_ids,
                                    self.kv_caches,
                                    start_layer_id)

    def _buffer_get_kv_patch(self, rank: int) -> Generator[KVPatch, None, None]:
        # 在发送完kv tensor之后开始发送kv patch
        while True:
            kv_patch = self.buffers[rank].pop_patch()
            yield kv_patch
            if kv_patch.meta.type == "kv_patch_finished": 
                self.kv_cache_transfer_in_process[rank] = False
                self.kv_patch_sending = False
                self.last_patch_ids[rank] = 0
                break


    # 在发送完kv tensor之后开始发送kv patch
    def _flexi_get_patch(self, rank: int, 
                        layer_ids: list[int],
                        kv_cache_meta: torch.Tensor, 
                        key_cache_ptrs: list[int],
                        value_cache_ptrs: list[int],
                        start_layer_id: int) -> Generator[KVPatch, None, None]:
        while True:
            logger.info(f"start to get slot mapping for rank {rank}")
            slot_mapping, is_finished, stored_tokens = self.slot_mappings[rank].get_all_slot_mappings()
            slot_mapping_dev = torch.tensor(slot_mapping, device=kv_cache_meta.device, dtype=torch.int64)
            block_size, num_head, head_dim = kv_cache_meta.shape
            # 为了匹配receiver端的期望，使用shape: [2, num_layers, num_tokens, num_heads, head_dim]
            # 第0维是K/V区分，第1维是layers
            kv_out = torch.empty(2, len(layer_ids), slot_mapping_dev.size(0), num_head, head_dim, dtype=kv_cache_meta.dtype, device=kv_cache_meta.device)
            logger.info(f"layer_ids: {layer_ids}, kv_out shape: {kv_out.shape}, slot mapping shape: {slot_mapping_dev.shape}, start_layer_id: {start_layer_id}")
            for idx, layer_id in enumerate(layer_ids):
                local_layer_id = layer_id - start_layer_id
                key_cache_ptr = key_cache_ptrs[local_layer_id]
                value_cache_ptr = value_cache_ptrs[local_layer_id]
                # 填充到 kv_out[0][idx] (keys) 和 kv_out[1][idx] (values)
                ops.flexi_gather_pages(key_cache_ptr, value_cache_ptr, slot_mapping_dev, kv_out[0][idx], kv_out[1][idx], block_size)
            yield KVPatch(
                KVPatchMeta(
                    type='kv_patch_meta' if not is_finished else "kv_patch_finished",
                    id=self.last_patch_ids[rank],
                    layer_ids=layer_ids,
                    num_tokens=stored_tokens,
                    slot_mapping_dtype=torch.int64,
                    slot_mapping_shape=slot_mapping_dev.shape,
                    kv_payload_dtype=kv_out.dtype,
                    kv_payload_shape=kv_out.shape,
                ),
                kv_out,
                slot_mapping_dev
            )
            self.last_patch_ids[rank] += 1
            if is_finished: 
                self.kv_cache_transfer_in_process[rank] = False
                self.kv_patch_sending = False
                self.last_patch_ids[rank] = 0
                break
    # 在发送完kv tensor之后开始发送kv patch
    def _get_patch(self, rank: int, 
                        layer_ids: list[int],
                        kv_cache: list[torch.Tensor], 
                        start_layer_id: int) -> Generator[KVPatch, None, None]:
        while True:
            patch_id = self.last_patch_ids[rank]
            logger.info(f"start to get slot mapping for rank {rank}")
            slot_mapping, is_finished, stored_tokens = self.slot_mappings[rank].get_all_slot_mappings()
            slot_mapping_dev = torch.tensor(slot_mapping, device=kv_cache[0].device, dtype=torch.int64)

            kv_patch = self.kv_helper.extract_kv_patch_from_kv_cache(
                patch_id=patch_id,
                kv_caches=kv_cache,
                layer_ids=layer_ids,
                start_layer_id=start_layer_id,
                slot_mapping=slot_mapping_dev,
                is_finished=is_finished
            )
            kv_patch.meta.num_tokens = stored_tokens
            yield kv_patch
            self.last_patch_ids[rank] += 1
            if is_finished: 
                self.kv_cache_transfer_in_process[rank] = False
                self.kv_patch_sending = False
                self.last_patch_ids[rank] = 0
                break

    def get_kv_tensor_from_cache(self, layer_ids: list[int],layer_id: int, start_layer_id: int, kv_cache_meta: torch.Tensor, slot_mapping: Optional[torch.Tensor] = None) -> Tuple[Union[KVTensorMeta, FlexiKVTensorMeta], torch.Tensor]:
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        if is_flexi:
            assert slot_mapping is not None, "slot_mapping should be provided when using flexi flash attention"
            return self._flexi_get_kv_tensor(
                                        layer_id=layer_id,
                                        layer_ids=layer_ids,
                                        kv_cache_meta=kv_cache_meta,
                                        key_cache_ptrs=self.key_cache_ptrs,
                                        value_cache_ptrs=self.value_cache_ptrs,
                                        slot_mapping=slot_mapping,
                                        start_layer_id=start_layer_id)
        else:
            return self._regular_get_kv_tensor(layer_id, layer_ids, self.kv_caches, start_layer_id)

    def _regular_get_kv_tensor(self, layer_id: int, layer_ids: list[int], kv_caches: list[torch.Tensor],
                            start_layer_id: int) -> Tuple[KVTensorMeta, torch.Tensor]:
        local_layer_id = layer_id - start_layer_id
        kv_cache = kv_caches[local_layer_id]
        # 第一次访问时默认置为 False，避免 KeyError
        return KVTensorMeta(
            type='kv_tensor',
            layer_to_be_received=set(layer_ids),
            layer_id=int(layer_id),
            num_tokens=0,
            dtype=kv_cache.dtype,
            shape=kv_cache.shape), kv_cache

    def _flexi_get_kv_tensor(self, layer_id: int, 
                            layer_ids: list[int],
                            kv_cache_meta: torch.Tensor, 
                            key_cache_ptrs: list[int],
                            value_cache_ptrs: list[int],
                            slot_mapping: torch.Tensor,
                            start_layer_id: int) -> Tuple[FlexiKVTensorMeta, torch.Tensor]:
        local_layer_id = layer_id - start_layer_id
        key_cache_ptr = key_cache_ptrs[local_layer_id]
        value_cache_ptr = value_cache_ptrs[local_layer_id]
        block_size, num_head, head_dim = kv_cache_meta.shape
        kv_out = torch.empty(2, slot_mapping.size(0), num_head, head_dim, dtype=kv_cache_meta.dtype, device=kv_cache_meta.device)
        logger.info(f"kv out device: {kv_out.device}, slot mapping device: {slot_mapping.device}")
        ops.flexi_gather_pages(key_cache_ptr, value_cache_ptr, slot_mapping, kv_out[0], kv_out[1], block_size)
        # 第一次访问时默认置为 False，避免 KeyError
        return FlexiKVTensorMeta(
            type='kv_tensor',
            layer_to_be_received=set(layer_ids),
            layer_id=int(layer_id),
            num_tokens=slot_mapping.size(0),
            kv_payload_dtype=kv_out.dtype,
            kv_payload_shape=kv_out.shape,
            slot_mapping_dtype=slot_mapping.dtype,
            slot_mapping_shape=slot_mapping.shape), kv_out



    def send_sync_finished_to_rank(self, rank: int, layer_ids: list[int]) -> None:
        """Send sync_finished signal to receiver indicating sync migration is complete.
        
        This signal tells the receiver to skip the kv_patch receiving phase
        and complete the migration immediately after binding KV tensors.
        
        Args:
            rank: Target rank to send the signal to.
            layer_ids: List of layer IDs that were transferred.
        """
        logger.info(f"[sync]: rank {self.rank} sending sync_finished to rank {rank}")
        self._send_meta_to_rank(rank, KVPatchMeta(
            type='sync_finished',
            id=0,
            layer_ids=layer_ids,
            num_tokens=0,
            slot_mapping_dtype=torch.int64,
            slot_mapping_shape=torch.Size([0]),
            kv_payload_dtype=torch.int64,
            kv_payload_shape=torch.Size([0]))
        )
        # Mark transfer as complete
        self.kv_cache_transfer_in_process[rank] = False

    # 在发送完kv tensor之后，sender如果直接发送send，那么根据我们的parallel nccl发送机制，需要先获取nccl锁再发送。然而receiver可能还没有准备好接收，这会导致nccl锁被占用很长时间，从而导致pipeline ranks之间的通行被block
    # 因此，这里的funtion时sender先等待receiver的ready信号，再获取nccl锁发送
    def wait_for_kv_patch(self, rank: int) -> None:
        send_pipe = self._ensure_pipe_and_buffer(rank, 'recv')
        send_pipe.wait_ready_for_kv_patch()

    def notify_for_kv_patch(self, rank: int) -> None:
        recv_pipe = self._ensure_pipe_and_buffer(rank, 'send')
        recv_pipe.notify_ready_for_kv_patch()

    def send_kv_tensor_to_rank(self, rank: int, kv_tensor_meta: Union[KVTensorMeta, FlexiKVTensorMeta], kv_tensor: torch.Tensor, slot_mapping: Optional[torch.Tensor] = None) -> None:
        """Send a single KV tensor to a peer rank with deadlock-free protocol.
        
        Protocol:
        1. Acquire nccl_lock
        2. Send meta
        3. Wait for ACCEPT/REJECT from receiver
        4. If ACCEPT: NCCL send, then release lock
        5. If REJECT: release lock, backoff, retry from step 1
        """
        self.kv_cache_transfer_in_process[rank] = True
        # Ensure pipe exists before using it
        self._ensure_pipe_and_buffer(rank, 'send')
        pipe = self._pair_pipes_send[rank]
        
        retry_delay = 0.005  # 5ms fixed retry delay
        
        while True:
            # Step 1: Acquire lock
            self._nccl_lock.acquire()
            
            try:
                # Step 2: Send meta
                self._send_meta_to_rank(rank, kv_tensor_meta)
                logger.info(f"[send_kv_tensor_to_rank]: sent meta to rank {rank}, waiting for ACCEPT/REJECT")
                
                # Step 3: Wait for ACCEPT/REJECT
                response = pipe.signal_group.recv_obj(src=pipe.peer_rank)
                
                if response == "ACCEPT":
                    # Step 4a: Receiver has lock, do NCCL send
                    logger.info(f"[send_kv_tensor_to_rank]: received ACCEPT from rank {rank}")
                    if isinstance(kv_tensor_meta, KVTensorMeta):
                        self._send_data_to_rank(rank, kv_tensor, wait_for_ack=False)
                    else:
                        assert slot_mapping is not None, "slot_mapping should be provided when sending FlexiKVTensorMeta"
                        self._send_data_to_rank(rank, slot_mapping, wait_for_ack=False)
                        logger.info(f"[send_kv_tensor_to_rank]: sent slot mapping to rank {rank}")
                        self._send_data_to_rank(rank, kv_tensor, wait_for_ack=False)
                        logger.info(f"[send_kv_tensor_to_rank]: sent kv tensor to rank {rank}")
                    break  # Success
                elif response == "REJECT":
                    # Step 4b: Retry after fixed delay
                    logger.info(f"[send_kv_tensor_to_rank]: received REJECT from rank {rank}, retry in 5ms")
                    self._nccl_lock.release()
                    time.sleep(retry_delay)
                    continue
                else:
                    raise RuntimeError(f"Unexpected response: {response}")
            except Exception as e:
                self._nccl_lock.release()
                raise e
        
        self._nccl_lock.release()
        logger.info(f"[send_kv_tensor_to_rank]: completed sending to rank {rank}")
    
    def send_kv_patch_to_rank(self, 
            target_rank: int,
            kv_patches: KVPatch) -> None:
        """Send KV patch to target rank with deadlock-free protocol.
        
        Protocol:
        1. Acquire nccl_lock
        2. Send meta
        3. Wait for ACCEPT/REJECT from receiver
        4. If ACCEPT: NCCL send, then release lock
        5. If REJECT: release lock, backoff, retry from step 1
        """
        assert self.kv_cache_transfer_in_process[target_rank] == True, "The kv cache transfer in process of the rank should be True."
        meta, kv_payload, slot_mapping = kv_patches.meta, kv_patches.kv_payload, kv_patches.slot_mapping
        time_start = time.time()
        # Ensure pipe exists (should already exist from send_kv_tensor_to_rank, but be safe)
        self._ensure_pipe_and_buffer(target_rank, 'send')
        pipe = self._pair_pipes_send[target_rank]
        
        retry_delay = 0.005  # 5ms fixed retry delay
        
        while True:
            # Step 1: Acquire lock
            self._nccl_lock.acquire()
            time_lock_acquired = time.time()
            
            try:
                # Step 2: Send meta
                self._send_meta_to_rank(target_rank, meta)
                logger.info(f"sent meta to rank {target_rank}, patch id: {meta.id}, waiting for ACCEPT/REJECT")
                
                # Step 3: Wait for ACCEPT/REJECT
                response = pipe.signal_group.recv_obj(src=pipe.peer_rank)
                
                if response == "ACCEPT":
                    # Step 4a: Receiver has lock, do NCCL send
                    logger.info(f"received ACCEPT from rank {target_rank}, sending data")
                    self._send_data_to_rank(target_rank, slot_mapping, wait_for_ack=False)
                    logger.info(f"sent slot mapping to rank {target_rank}, patch id: {meta.id}")
                    self._send_data_to_rank(target_rank, kv_payload, wait_for_ack=False)
                    logger.info(f"sent kv payload to rank {target_rank}, patch id: {meta.id}, shape:{kv_payload.shape}")
                    break  # Success, exit loop
                elif response == "REJECT":
                    # Step 4b: Retry after fixed delay
                    logger.info(f"received REJECT from rank {target_rank}, retry in 5ms")
                    self._nccl_lock.release()
                    time.sleep(retry_delay)
                    continue  # Retry
                else:
                    raise RuntimeError(f"Unexpected response: {response}")
            except Exception as e:
                self._nccl_lock.release()
                raise e
        
        # Release lock after successful send
        self._nccl_lock.release()
        
        logger.info(f"send_kv_patch_to_rank completed: time to acquire lock: {time_lock_acquired - time_start}, total time: {time.time() - time_start}")
        if meta.type == "kv_patch_finished":
            self.kv_cache_transfer_in_process[target_rank] = False

    # def start_kv_tensor_transfer_async(self, 
    #                                    rank_to_layers_ids: dict[int, list[int]], 
    #                                    kv_caches: list[torch.Tensor],
    #                                     start_layer_id: int) -> None:
    #     """Send layers' KV cache to a peer rank.
    #         In each of the migration process, this function should only be called once.
    #     Args:
    #         rank: peer global rank to send to.
    #         kv_cache: tensor to send (GPU or CPU tensor; will be moved to local GPU).
    #         start_layer_id: which layer this KV cache belongs to (for receiver to demux).
    #         layer_ids: which layers this KV cache belongs to (for receiver to demux).
    #     """
    #     assert all(transfer_in_process == False for transfer_in_process in self.kv_cache_transfer_in_process.values()), "In each of the migration process, this function should only be called once."
    #     assert all(patch_id == 0 for patch_id in self.last_patch_ids.values()), "The patch id of the rank should be 0."
    #     for rank, layer_ids in rank_to_layers_ids.items():
    #         for layer_id in layer_ids:
    #             time_start = time.time()
    #             local_layer_id = layer_id - start_layer_id
    #             kv_cache = kv_caches[local_layer_id]
    #             # 第一次访问时默认置为 False，避免 KeyError
    #             logger.info(f"[debug]: start to send kv tensor meta to rank {rank} for layer {layer_id}, kv_cache shape: {kv_cache.shape}")
    #             self._send_meta_to_rank(rank, KVTensorMeta(
    #                 type='kv_tensor',
    #                 layer_to_be_received=set(layer_ids),
    #                 layer_id=int(layer_id),
    #                 num_tokens=int(kv_cache.size(0)),
    #                 dtype=kv_cache.dtype,
    #                 shape=kv_cache.shape))

    #             logger.info(f"[timeline]: send kv tensor meta to rank {rank} for layer {layer_id}, time taken: {time.time() - time_start}")
    #             # 使用同步发送，确保数据传输完成后再继续
    #             self._send_data_to_rank(rank, kv_cache, synchronize=True)
    #             logger.info(f"[timeline]: send kv tensor payload to rank {rank} for layer {layer_id}, time taken: {time.time() - time_start}, data_size: {kv_cache.numel() * kv_cache.element_size() / 1024 ** 2:.2f}MB")
    #             self.kv_cache_transfer_in_process[rank] = True
    #             del kv_cache
    #     del kv_caches
    #     logger.info(f"[debug]: finished sending kv tensor, start to send kv patch")
    #     # 在发送完kv tensor之后开始发送kv patch
    #     while True:
    #         for rank in rank_to_layers_ids.keys():
    #             time_start = time.time()
    #             kv_patch = self.buffers[rank].pop_patch()
    #             self._send_meta_to_rank(rank, kv_patch.meta)
    #             self._send_data_to_rank(rank, kv_patch.slot_mapping, synchronize=True, wait_for_ack=True)
    #             self._send_data_to_rank(rank, kv_patch.kv_payload, synchronize=True, wait_for_ack=True)
    #             self.kv_patch_sending = True
    #             patch_payload_size = kv_patch.kv_payload.numel() * kv_patch.kv_payload.element_size()
    #             patch_slot_mapping_size = kv_patch.slot_mapping.numel() * kv_patch.slot_mapping.element_size()
    #             logger.info(f"[timeline]: send kv patch to rank {rank}, time taken: {time.time() - time_start}, data_size: {patch_slot_mapping_size / 1024 ** 2:.2f}MB + {patch_payload_size / 1024 ** 2:.2f}MB")
    #             if kv_patch.meta.type == "kv_patch_finished":
    #                 logger.info(f"[operation]: rank {self.rank} finish the kv patch sending to rank {rank}")
    #                 self.kv_cache_transfer_in_process[rank] = False
    #                 self.kv_patch_sending = False
    #                 break

    # def start_kv_tensor_transfer_async_without_buffer(self, 
    #                                          rank_to_layers_ids: dict[int, list[int]], 
    #                                          kv_cache_meta: torch.Tensor, 
    #                                          key_cache_ptrs: list[int],
    #                                          value_cache_ptrs: list[int],
    #                                          slot_mapping: list[int],
    #                           start_layer_id: int) -> None:
    #     """Send layers' KV cache to a peer rank.
    #         In each of the migration process, this function should only be called once.
    #         Different from the buffered version, this function directly gathers the kv cache from the memory pointers 
    #         and sends them without using intermediate buffers.
    #         Slot mappings is updated by the worker threads, which essentially replace the role of buffer.
    #         The good thing about slot mapping is that is is implemented with dirty page, which avoids duplicated token ids.
    #     Args:
    #         rank: peer global rank to send to.
    #         kv_cache: tensor to send (GPU or CPU tensor; will be moved to local GPU).
    #         start_layer_id: which layer this KV cache belongs to (for receiver to demux).
    #         layer_ids: which layers this KV cache belongs to (for receiver to demux).
    #         slot_mapping: the slot mapping to gather the kv cache.
    #     """
    #     assert all(transfer_in_process == False for transfer_in_process in self.kv_cache_transfer_in_process.values()), "In each of the migration process, this function should only be called once."
    #     assert all(patch_id == 0 for patch_id in self.last_patch_ids.values()), "The patch id of the rank should be 0."
    #     slot_mapping_dev = torch.tensor(slot_mapping)

    #     for rank, layer_ids in rank_to_layers_ids.items():
    #         for layer_id in layer_ids:
    #             time_start = time.time()
    #             local_layer_id = layer_id - start_layer_id
    #             key_cache_ptr = key_cache_ptrs[local_layer_id]
    #             value_cache_ptr = value_cache_ptrs[local_layer_id]
    #             block_size, num_head, head_dim = kv_cache_meta.shape
    #             kv_out = torch.empty(2, len(slot_mapping), num_head, head_dim, dtype=kv_cache_meta.dtype)
    #             ops.flexi_gather_pages(key_cache_ptr, value_cache_ptr, slot_mapping_dev, kv_out[0], kv_out[1], block_size)
    #             # 第一次访问时默认置为 False，避免 KeyError
    #             logger.info(f"[debug]: start to send kv tensor meta to rank {rank} for layer {layer_id}, kv_cache block shape: {kv_cache_meta[0].shape}")

    #             self._send_meta_to_rank(rank, FlexiKVTensorMeta(
    #                 type='kv_tensor',
    #                 layer_to_be_received=set(layer_ids),
    #                 layer_id=int(layer_id),
    #                 num_tokens=int(len(slot_mapping)),
    #                 dtype=kv_out.dtype,
    #                 shape=kv_out.shape,
    #                 slot_mapping=slot_mapping_dev))

    #             # 使用同步发送，确保数据传输完成后再继续
    #             self._send_data_to_rank(rank, kv_out, synchronize=True)
    #             self.kv_cache_transfer_in_process[rank] = True
    #     logger.info(f"[debug]: finished sending kv tensor, start to send kv patch")
    #     # 在发送完kv tensor之后开始发送kv patch
    #     while True:
    #         for rank, layer_ids in rank_to_layers_ids.items():
    #             time_start = time.time()
    #             slot_mapping, is_finished = self.slot_mappings[rank].get_all_slot_mappings()

    #             block_size, num_head, head_dim = kv_cache_meta.shape
    #             kv_out = torch.empty(len(layer_ids), 2, len(slot_mapping), num_head, head_dim, dtype=kv_cache_meta.dtype)
    #             for layer_id in layer_ids:
    #                 local_layer_id = layer_id - start_layer_id
    #                 key_cache_ptr = key_cache_ptrs[local_layer_id]
    #                 value_cache_ptr = value_cache_ptrs[local_layer_id]
    #                 ops.flexi_gather_pages(key_cache_ptr, value_cache_ptr, slot_mapping_dev, kv_out[layer_id][0], kv_out[layer_id][1], block_size)

    #             self.last_patch_ids[rank] += 1
    #             kv_patch = KVPatch(
    #                 KVPatchMeta(
    #                     type='kv_patch_meta' if not is_finished else "kv_patch_finished",
    #                     id=self.last_patch_ids[rank],
    #                     layer_ids=layer_ids,
    #                     num_tokens=int(len(slot_mapping)),
    #                     slot_mapping_dtype=torch.int64,
    #                     slot_mapping_shape=torch.Size([len(slot_mapping)]),
    #                     kv_payload_dtype=kv_out.dtype,
    #                     kv_payload_shape=kv_out.shape,
    #                 ),
    #                 kv_out,
    #                 torch.tensor(slot_mapping, dtype=torch.int64)
    #             )
    #             self._send_meta_to_rank(rank, kv_patch.meta)
    #             self._send_data_to_rank(rank, kv_patch.slot_mapping, synchronize=True, wait_for_ack=True)
    #             self._send_data_to_rank(rank, kv_patch.kv_payload, synchronize=True, wait_for_ack=True)
    #             self.kv_patch_sending = True
    #             patch_payload_size = kv_patch.kv_payload.numel() * kv_patch.kv_payload.element_size()
    #             patch_slot_mapping_size = kv_patch.slot_mapping.numel() * kv_patch.slot_mapping.element_size()
    #             logger.info(f"[timeline]: send kv patch to rank {rank}, time taken: {time.time() - time_start}, data_size: {patch_slot_mapping_size / 1024 ** 2:.2f}MB + {patch_payload_size / 1024 ** 2:.2f}MB")
    #             if kv_patch.meta.type == "kv_patch_finished":
    #                 logger.info(f"[operation]: rank {self.rank} finish the kv patch sending to rank {rank}")
    #                 self.kv_cache_transfer_in_process[rank] = False
    #                 self.kv_patch_sending = False
    #                 self.last_patch_ids[rank] = 0
    #                 break


    def add_patch_to_send_buffer(self, rank: int, kv_patch: KVPatch) -> None:
        self.buffers[rank].add_patch(kv_patch)
        self.last_patch_ids[rank] += 1

    def finish_kv_cache_transfer(self) -> None:
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        for rank in self.last_patch_ids:
            if is_flexi:
                self.slot_mappings[rank].add_slot_mappings([], is_finished=True)
            else:
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

    def add_new_tokens_to_kv_synchronizer(self, rank: int, slot_mapping: torch.Tensor, is_finished: bool, num_total_new_tokens: int) -> None:
        time_start = time.time()
        # is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        # if is_flexi:
        self.slot_mappings[rank].add_slot_mappings(slot_mapping.tolist(), is_finished=is_finished, num_total_new_tokens=num_total_new_tokens)
        # else:
        #     kv_patch = self.kv_synchronizer_helper.extract_kv_patch_from_kv_cache(self.last_patch_ids[rank], kv_caches, layer_ids, start_layer_id, slot_mapping, is_finished)
        #     time_after_extract_kv_patch = time.time()
        #     logger.info(f"debug: ---------------------extract kv patch time: {time_after_extract_kv_patch - time_start:.2f} seconds")
        #     assert kv_patch is not None
        #     self.last_patch_ids[rank] += 1
        #     self.buffers[rank].add_patch(kv_patch)

    # ########################################## #
    # Receiver side functions                      #
    # ########################################## #
    def get_last_patch_id(self, rank: int) -> int:
        return self.last_patch_ids[rank]

    def recv_controller(self, rank: int) -> Union[KVTensorMeta, KVPatchMeta, FlexiKVTensorMeta]:
        pipe = self._ensure_pipe_and_buffer(rank, 'recv')
        pipe = self._pair_pipes_recv[rank]
        meta_obj = pipe.recv_meta()
        return self._control_adapter.validate_python(meta_obj)

    def recv_kv_tensor(self, from_rank: int, meta: Union[KVTensorMeta, FlexiKVTensorMeta]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Receive KV tensor from sender with deadlock-free protocol.
        
        Protocol:
        1. Meta already received by caller
        2. Try to acquire nccl_lock (non-blocking)
        3. If failed: send REJECT, receive new meta (sender will retry), go to step 2
        4. If acquired: send ACCEPT, do NCCL recv, release lock
        """
        time_start = time.time()
        pipe = self._pair_pipes_recv[from_rank]
        
        current_meta: Union[KVTensorMeta, FlexiKVTensorMeta] = meta
        while True:
            # Step 2: Try to acquire lock (non-blocking)
            lock_acquired = self._nccl_lock.acquire(blocking=False)
            
            if not lock_acquired:
                # Step 3: Lock held by others - send REJECT immediately
                logger.info(f"recv_kv_tensor: lock held by others, sending REJECT")
                pipe.signal_group.send_obj("REJECT", dst=pipe.peer_rank)
                # Sender will backoff and resend meta, we need to receive it
                new_meta = self.recv_controller(from_rank)
                assert isinstance(new_meta, (KVTensorMeta, FlexiKVTensorMeta)), f"Expected KVTensorMeta or FlexiKVTensorMeta, got {type(new_meta)}"
                current_meta = new_meta
                continue
            
            try:
                # Step 4: Lock acquired - send ACCEPT and recv
                time_lock_acquired = time.time()
                pipe.signal_group.send_obj("ACCEPT", dst=pipe.peer_rank)
                logger.info(f"recv_kv_tensor: lock acquired in {time_lock_acquired - time_start}s, sent ACCEPT")
                
                if isinstance(current_meta, FlexiKVTensorMeta):
                    slot_mapping = self._recv_data_from_rank(from_rank, current_meta.slot_mapping_dtype, current_meta.slot_mapping_shape, send_ack=False)
                    logger.info(f"recv_kv_tensor: received slot mapping, took {time.time() - time_start}s")
                    kv_payload = self._recv_data_from_rank(from_rank, current_meta.kv_payload_dtype, current_meta.kv_payload_shape, send_ack=False)
                    logger.info(f"recv_kv_tensor: received kv payload, took {time.time() - time_start}s")
                else:
                    kv_payload = self._recv_data_from_rank(from_rank, current_meta.dtype, current_meta.shape, send_ack=False)
                    slot_mapping = None
                break  # Success, exit loop
            finally:
                self._nccl_lock.release()
        
        logger.info(f"recv_kv_tensor completed: lock wait: {time_lock_acquired - time_start}, total: {time.time() - time_start}")
        
        if isinstance(current_meta, FlexiKVTensorMeta):
            assert slot_mapping is not None, "slot_mapping should not be None for FlexiKVTensorMeta"
            assert kv_payload.dim() == 4 and kv_payload.size(0) == 2
            return slot_mapping, kv_payload
        else:
            assert kv_payload.dim() == 5 and kv_payload.size(0) == 2, f"kv_payload shape {kv_payload.shape} is not correct"
            return kv_payload

    def recv_kv_patch(self, from_rank: int, meta: KVPatchMeta) -> Tuple[torch.Tensor, torch.Tensor]:
        """Receive KV patch from sender with deadlock-free protocol.
        
        Protocol:
        1. Meta already received by caller
        2. Try to acquire nccl_lock (non-blocking)
        3. If failed: send REJECT, receive new meta (sender will retry), go to step 2
        4. If acquired: send ACCEPT, do NCCL recv, release lock
        """
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        time_start = time.time()
        pipe = self._pair_pipes_recv[from_rank]
        
        current_meta = meta
        while True:
            # Step 2: Try to acquire lock (non-blocking)
            lock_acquired = self._nccl_lock.acquire(blocking=False)
            
            if not lock_acquired:
                # Step 3: Lock held by others - send REJECT immediately
                logger.info(f"recv_kv_patch: lock held by others, sending REJECT")
                pipe.signal_group.send_obj("REJECT", dst=pipe.peer_rank)
                # Sender will backoff and resend meta, we need to receive it
                new_meta = self.recv_controller(from_rank)
                assert isinstance(new_meta, KVPatchMeta), f"Expected KVPatchMeta, got {type(new_meta)}"
                current_meta = new_meta
                continue
            
            try:
                # Step 4: Lock acquired - send ACCEPT and recv
                time_lock_acquired = time.time()
                pipe.signal_group.send_obj("ACCEPT", dst=pipe.peer_rank)
                logger.info(f"recv_kv_patch: lock acquired in {time_lock_acquired - time_start}s, sent ACCEPT")
                
                slot_mapping = self._recv_data_from_rank(from_rank, current_meta.slot_mapping_dtype, current_meta.slot_mapping_shape, False)
                kv_payload = self._recv_data_from_rank(from_rank, current_meta.kv_payload_dtype, current_meta.kv_payload_shape, False)
                break  # Success, exit loop
            finally:
                self._nccl_lock.release()
        
        logger.info(f"recv_kv_patch completed: lock wait: {time_lock_acquired - time_start}, total: {time.time() - time_start}")
        
        # For kv_patch_finished with empty data, skip the dimension assertion
        if current_meta.type == 'kv_patch_finished' and kv_payload.numel() == 0:
            logger.info(f"[recv_kv_patch]: received empty kv_patch_finished from rank {from_rank}")
            return slot_mapping, kv_payload
        
        if is_flexi:
            assert kv_payload.dim() == 5 and kv_payload.size(0) == 2, f"kv_payload shape {kv_payload.shape} is not correct"
        else:
            assert kv_payload.dim() == 5 and kv_payload.size(0) == 2, f"kv_payload shape {kv_payload.shape} is not correct"

        return slot_mapping, kv_payload

    # def put_kv_patch_to_buffer(self, rank: int, kv_caches: list[torch.Tensor], layer_ids: list[int], start_layer_id: int, slot_mapping: torch.Tensor) -> None:
    #     time_start = time.time()
    #     kv_patch = self.kv_synchronizer_helper.extract_kv_patch_from_kv_cache(self.last_patch_ids[rank], kv_caches, layer_ids, start_layer_id, slot_mapping)
    #     time_after_extract_kv_patch = time.time()
    #     logger.info(f"debug: ---------------------extract kv patch time: {time_after_extract_kv_patch - time_start:.2f} seconds")
    #     assert kv_patch is not None
    #     self.last_patch_ids[rank] += 1

    #     self.buffers[rank].add_patch(kv_patch)
    def get_one_patch_from_buffer(self, rank: int) -> KVPatch:
        # Ensure the recv pipe and buffer exist before popping
        kv_patch = self.buffers[rank].pop_patch()
        return kv_patch

    def apply_one_patch_to_kv_cache(self, start_layer_id: int, meta: KVPatchMeta, kv_payload: torch.Tensor,  slot_mapping: torch.Tensor, page_meta: Optional[torch.Tensor] = None,) -> int:
        keys = kv_payload[0]
        values = kv_payload[1]
        logger.info(f"apply one patch with id {meta.id}, num_tokens: {meta.num_tokens}")
        # slot_mapping 已经被裁剪过，只包含有效的 token
        is_flexi = self.vllm_config.dynamic_config.use_flexi_kv
        # assert slot_mapping.size(0) == meta.num_tokens, f"slot_mapping size {slot_mapping.size(0)} should match num_tokens {meta.num_tokens}"
        if slot_mapping.size(0) != meta.num_tokens:
            logger.info(f"Warning: slot_mapping size {slot_mapping.size(0)} does not match num_tokens {meta.num_tokens}, proceed anyway.")
        # The operations below are executed in the default stream and will be properly ordered.
        with self.device:
            logger.info(f"[listen loop] apply kv patch to kv cache on device {self.device}")
            for layer_id, key, value in zip(meta.layer_ids, keys, values):
                local_layer_id = layer_id - start_layer_id
                if is_flexi:
                    assert page_meta is not None, "page_meta should be provided when using flexi flash attention"
                    key_cache_ptr = self.key_cache_ptrs[local_layer_id]
                    value_cache_ptr = self.value_cache_ptrs[local_layer_id]
                    self.kv_helper.flexi_put_kv_to_cache(
                        model_executable=self.model_executable,
                        page_meta=page_meta,
                        keys=key,
                        values=value,
                        key_cache_ptr=key_cache_ptr,
                        value_cache_ptr=value_cache_ptr,
                        layer=layer_id,
                        slot_mapping=slot_mapping,
                    )
                else:
                    kv_cache = self.kv_caches[local_layer_id]
                    # 使用裁剪后的 slot_mapping，从 0 到 num_tokens
                    self.kv_helper.put_kv_to_cache(self.model_executable, key, value, layer_id, kv_cache, slot_mapping, 0, slot_mapping.size(0))
        return meta.id

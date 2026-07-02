# SPDX-License-Identifier: Apache-2.0
"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The SimpleConnector transfers KV caches between prefill vLLM worker (KV cache
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Literal,
                    Optional, Tuple, Union)
import threading
import os
from contextlib import nullcontext

import bitarray
from httpx import patch
from responses import start
import torch
from pydantic import TypeAdapter
from vllm import _custom_ops as ops

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.dynamic_utils import (
    FlexiKVTensorMeta,
    KVCachedSparseKVTensorMeta,
    kv_synchronizer_helper as kv_helper,
    KVPatch,
    KVPatchMeta,
    KVTensorMeta,
)
from vllm.kvcached_integration import (
    guard_kvcached_vmm_slots_mapped,
    use_flexi_kv_for_runtime,
    use_kvcached_backend,
)
from vllm.logger import init_logger
from vllm.distributed.utils import StatelessProcessGroup
import time
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.utils import current_stream
if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


def _autoscaling_kvcached_debug_enabled() -> bool:
    return os.environ.get("VLLM_AUTOSCALING_KVCACHED_DEBUG",
                          "").lower() in {"1", "true", "yes", "on"}


def _autoscaling_kvcached_debug_verbose() -> bool:
    return os.environ.get("VLLM_AUTOSCALING_KVCACHED_DEBUG_VERBOSE",
                          "").lower() in {"1", "true", "yes", "on"}


def _blocks_from_slots(slot_mapping: Union[torch.Tensor, list[int]],
                       block_size: Optional[int]) -> list[int]:
    if block_size is None or block_size <= 0:
        return []
    if isinstance(slot_mapping, torch.Tensor):
        if slot_mapping.numel() == 0:
            return []
        slots = slot_mapping.detach().cpu().tolist()
    else:
        slots = slot_mapping
    return sorted({
        int(slot) // int(block_size)
        for slot in slots
        if int(slot) >= 0
    })


def _shape_tuple(shape: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _kv_signal_identity(
    meta: Union[KVTensorMeta, KVPatchMeta, FlexiKVTensorMeta,
                KVCachedSparseKVTensorMeta],
) -> dict[str, Any]:
    if isinstance(meta, KVPatchMeta):
        return {
            "phase": "kv_patch",
            "meta_type": meta.type,
            "patch_id": int(meta.id),
            "layer_ids": tuple(int(layer_id) for layer_id in meta.layer_ids),
            "num_tokens": int(meta.num_tokens),
            "slot_mapping_shape": _shape_tuple(meta.slot_mapping_shape),
            "kv_payload_shape": _shape_tuple(meta.kv_payload_shape),
        }
    if isinstance(meta, KVTensorMeta):
        return {
            "phase": "kv_tensor",
            "meta_type": meta.type,
            "layer_id": int(meta.layer_id),
            "layers": tuple(sorted(int(layer_id)
                                   for layer_id in meta.layer_to_be_received)),
            "num_tokens": int(meta.num_tokens),
            "kv_payload_shape": _shape_tuple(meta.shape),
        }
    return {
        "phase": "kv_tensor",
        "meta_type": meta.type,
        "layer_id": int(meta.layer_id),
        "layers": tuple(sorted(int(layer_id)
                               for layer_id in meta.layer_to_be_received)),
        "num_tokens": int(meta.num_tokens),
        "slot_mapping_shape": _shape_tuple(meta.slot_mapping_shape),
        "kv_payload_shape": _shape_tuple(meta.kv_payload_shape),
    }


def _kv_transfer_signal(
    status: Literal["ACCEPT", "REJECT"],
    meta: Union[KVTensorMeta, KVPatchMeta, FlexiKVTensorMeta,
                KVCachedSparseKVTensorMeta],
) -> dict[str, Any]:
    signal = {
        "type": "KV_TRANSFER_SIGNAL",
        "status": status,
    }
    signal.update(_kv_signal_identity(meta))
    return signal


def _kv_signal_matches(
    response: Any,
    status: Literal["ACCEPT", "REJECT"],
    meta: Union[KVTensorMeta, KVPatchMeta, FlexiKVTensorMeta,
                KVCachedSparseKVTensorMeta],
) -> bool:
    if response == status:
        return True
    if not isinstance(response, dict):
        return False
    if response.get("type") != "KV_TRANSFER_SIGNAL":
        return False
    if response.get("status") != status:
        return False
    identity = _kv_signal_identity(meta)
    return all(response.get(key) == value for key, value in identity.items())


def _kv_patch_ready_signal() -> dict[str, str]:
    return {
        "type": "KV_TRANSFER_SIGNAL",
        "status": "READY",
        "phase": "kv_patch_ready",
    }


def _kv_patch_ready_matches(response: Any) -> bool:
    return (response == "ack"
            or (isinstance(response, dict)
                and response.get("type") == "KV_TRANSFER_SIGNAL"
                and response.get("status") == "READY"
                and response.get("phase") == "kv_patch_ready"))


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

class KVSlotSnapshot:
    """A materialized KV migration bitmap snapshot.

    KVCacheD unmap invalidates both pending bitmaps and active snapshots. The
    sender releases the snapshot after it finishes gathering the referenced KV
    payload, so an unmap either trims the snapshot before gather starts or waits
    for the gather-side page lifetime lock to be released.
    """

    def __init__(
        self,
        owner: "KVSlotMapping",
        snapshot_id: int,
        slot_mapping: list[int],
        is_finished: bool,
        stored_tokens: int,
        trace_info: dict[str, list[int]],
    ) -> None:
        self._owner = owner
        self.snapshot_id = snapshot_id
        self.slot_mapping = slot_mapping
        self.is_finished = is_finished
        self.stored_tokens = stored_tokens
        self.trace_info = trace_info
        self.original_slot_count = len(slot_mapping)
        self.dropped_slots = 0
        self._released = False

    def replace_slots(self, slots: list[int]) -> None:
        dropped = max(0, len(self.slot_mapping) - len(slots))
        self.slot_mapping = slots
        self.dropped_slots += dropped

    def clear_slots_for_blocks(
        self,
        block_ids: list[int],
        block_size: int,
    ) -> dict[str, int]:
        if not self.slot_mapping or not block_ids:
            return {
                "cleared_slots": 0,
                "remaining_slots": len(self.slot_mapping),
            }

        block_id_set = {int(block_id) for block_id in block_ids}
        kept_slots: list[int] = []
        cleared_slots = 0
        for slot in self.slot_mapping:
            if int(slot) < 0:
                kept_slots.append(slot)
                continue
            if int(slot) // int(block_size) in block_id_set:
                cleared_slots += 1
            else:
                kept_slots.append(slot)

        if cleared_slots:
            self.slot_mapping = kept_slots
            self.dropped_slots += cleared_slots

        return {
            "cleared_slots": cleared_slots,
            "remaining_slots": len(self.slot_mapping),
        }

    def meta_num_tokens(self) -> int:
        if self.dropped_slots:
            return len(self.slot_mapping)
        return self.stored_tokens

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._owner.release_snapshot(self.snapshot_id)


class KVSlotMapping:
    def __init__(self, size: int):
        self.bitmap = bitarray.bitarray(size)
        self.bitmap.setall(0)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self.is_finished = False
        self.stored_tokens = 0
        self.trace_infos: list[dict[str, Any]] = []
        self._next_snapshot_id = 0
        self._active_snapshots: dict[int, KVSlotSnapshot] = {}

    def resize(self, size: int) -> None:
        with self._cv:
            old_size = len(self.bitmap)
            if size == old_size:
                return

            new_bitmap = bitarray.bitarray(size)
            new_bitmap.setall(0)
            copy_len = min(old_size, size)
            if copy_len:
                new_bitmap[:copy_len] = self.bitmap[:copy_len]
            if size < old_size and self.bitmap[size:].any():
                logger.warning(
                    "Dropping pending KV slot mappings outside resized "
                    "capacity: old_size=%s new_size=%s",
                    old_size, size)
            self.bitmap = new_bitmap
            self._cv.notify_all()

    def add_slot_mappings(
        self,
        slot_mapping: list[int],
        is_finished: bool,
        num_total_new_tokens: int,
        trace_info: Optional[dict[str, Any]] = None,
    ) -> None:
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
            if trace_info is not None:
                self.trace_infos.append(dict(trace_info))
            logger.info(f"[num tokens]: sender side add len(trimmed_slot_mapping): {len(trimmed_slot_mapping)} to synchronizer, total stored tokens: {self.stored_tokens}")
            self._cv.notify_all()

    def _trace_summary(self) -> dict[str, list[int]]:
        summary: dict[str, list[int]] = {}
        keys = (
            "scheduler_step_id",
            "scheduler_output_version",
            "scheduler_request_free_epoch",
            "scheduler_block_free_epoch",
        )
        for key in keys:
            values = sorted({
                int(info[key])
                for info in self.trace_infos
                if key in info and info[key] is not None
            })
            if values:
                summary[f"{key}s"] = values
        return summary

    def wait_for_slot_mappings(self) -> None:
        with self._cv:
            while self.bitmap.count() == 0 and not self.is_finished:
                self._cv.wait()

    def pop_slot_mapping_snapshot(self) -> Optional[KVSlotSnapshot]:
        with self._cv:
            if self.bitmap.count() == 0 and not self.is_finished:
                return None
            slot_mapping = list(self.bitmap.search(1))
            stored_tokens = self.stored_tokens
            is_finished = self.is_finished
            trace_info = self._trace_summary()

            self.bitmap.setall(0)
            self.stored_tokens = 0
            self.is_finished = False
            self.trace_infos.clear()

            snapshot_id = self._next_snapshot_id
            self._next_snapshot_id += 1
            snapshot = KVSlotSnapshot(
                self,
                snapshot_id,
                slot_mapping,
                is_finished,
                stored_tokens,
                trace_info,
            )
            if slot_mapping:
                self._active_snapshots[snapshot_id] = snapshot
            return snapshot

    def get_all_slot_mapping_snapshot(self) -> KVSlotSnapshot:
        while True:
            self.wait_for_slot_mappings()
            snapshot = self.pop_slot_mapping_snapshot()
            if snapshot is not None:
                return snapshot

    def release_snapshot(self, snapshot_id: int) -> None:
        with self._cv:
            self._active_snapshots.pop(snapshot_id, None)

    def get_all_slot_mappings(self) -> Tuple[list[int], bool, int, dict[str, list[int]]]:
        snapshot = self.get_all_slot_mapping_snapshot()
        try:
            return (
                list(snapshot.slot_mapping),
                snapshot.is_finished,
                snapshot.stored_tokens,
                snapshot.trace_info,
            )
        finally:
            snapshot.release()

    def clear_slots_for_blocks(
        self,
        block_ids: list[int],
        block_size: int,
    ) -> dict[str, int]:
        if block_size <= 0 or not block_ids:
            return {"blocks": 0, "cleared_slots": 0}
        unique_blocks = sorted({
            int(block_id)
            for block_id in block_ids
            if int(block_id) >= 0
        })
        cleared_slots = 0
        active_cleared_slots = 0
        with self._cv:
            bitmap_len = len(self.bitmap)
            for block_id in unique_blocks:
                start = block_id * block_size
                if start >= bitmap_len:
                    continue
                end = min(start + block_size, bitmap_len)
                old_count = self.bitmap[start:end].count()
                if old_count == 0:
                    continue
                zeros = bitarray.bitarray(end - start)
                zeros.setall(0)
                self.bitmap[start:end] = zeros
                cleared_slots += old_count
            for snapshot in list(self._active_snapshots.values()):
                stats = snapshot.clear_slots_for_blocks(unique_blocks,
                                                        block_size)
                active_cleared_slots += int(stats.get("cleared_slots", 0))
            if cleared_slots or active_cleared_slots:
                self._cv.notify_all()
        return {
            "blocks": len(unique_blocks),
            "cleared_slots": cleared_slots + active_cleared_slots,
            "pending_cleared_slots": cleared_slots,
            "active_cleared_slots": active_cleared_slots,
            "active_snapshots": len(self._active_snapshots),
        }

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

    def send_meta(self, obj: Union[KVTensorMeta, KVPatchMeta,
                                   FlexiKVTensorMeta,
                                   KVCachedSparseKVTensorMeta]) -> None:
        self.meta_group.send_obj(obj, dst=self.peer_rank)

    def recv_meta(self) -> Union[KVTensorMeta, KVPatchMeta,
                                 FlexiKVTensorMeta,
                                 KVCachedSparseKVTensorMeta]:
        obj = self.meta_group.recv_obj(src=self.peer_rank)
        assert isinstance(
            obj,
            (KVTensorMeta, KVPatchMeta, FlexiKVTensorMeta,
             KVCachedSparseKVTensorMeta)), (
                 "The object should be a KVTensorMeta, KVPatchMeta, "
                 "FlexiKVTensorMeta, or KVCachedSparseKVTensorMeta")
        return obj

    def wait_ready_for_kv_patch(self) -> None:
        logger.info(f"[PairPipe.wait_ready_for_kv_patch] Waiting for ready signal from peer {self.peer_rank}")
        while True:
            obj = self.signal_group.recv_obj(src=self.peer_rank)
            if _kv_patch_ready_matches(obj):
                return
            logger.warning(
                "[PairPipe.wait_ready_for_kv_patch] Ignoring stale signal "
                "while waiting for patch-ready from peer %s: %s",
                self.peer_rank, obj)
        return

    def notify_ready_for_kv_patch(self) -> None:
        logger.info(f"[PairPipe.notify_ready_for_kv_patch] Notifying peer {self.peer_rank} that ready")
        self.signal_group.send_obj(_kv_patch_ready_signal(),
                                   dst=self.peer_rank)
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
            logger.info(f"[PairPipe.send_data] Calling NCCL send, nccl_rank={self._nccl.rank}, peer_rank={self.peer_rank}")
            # Acquire NCCL lock to prevent deadlock with Ray compiled_dag's NCCL operations
            # if wait_for_ack:
                # logger.info(f"[PairPipe.send_data] Waiting for ACK from peer {self.peer_rank} before sending")
                # ack = self.meta_group.recv_obj(src=self.peer_rank)
                # assert ack == "SYC+ACK", f"Expected ACK, got {ack}"
                # self.meta_group.send_obj("ACK", dst=self.peer_rank)
                # logger.info(f"[PairPipe.send_data] Received ACK from peer {self.peer_rank}")
            if stream is not None:
                with torch.cuda.device(self.device), torch.cuda.stream(stream):
                    dev_tensor = tensor.to(self.device)
                    self._nccl.send(dev_tensor,
                                    dst=self.peer_rank,
                                    stream=stream)
            else:
                dev_tensor = tensor.to(self.device)
                self._nccl.send(dev_tensor, dst=self.peer_rank, stream=stream)
            # Synchronize within the lock to ensure the NCCL operation completes
            # if stream is not None:
            #     stream.synchronize()
            # else:
            #     current_stream().synchronize()
            #     logger.info(f"[PairPipe.send_data] NCCL lock released after send")
            logger.info(f"[PairPipe.send_data] NCCL send completed")
        else:
            self.data_group.send_obj(tensor, dst=self.peer_rank)

    def recv_data(self,
                  dtype: torch.dtype,
                  shape: torch.Size,
                  stream=None,
                  send_ack: bool = False,
                  synchronize: bool = True,
                  recv_buffer: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        time_start = time.time()
        if self.is_use_nccl:
            assert self._nccl is not None, "The nccl communicator should be initialized"
            if recv_buffer is None:
                buf = self._prepare_recv_buffer(dtype, shape)
            else:
                assert tuple(recv_buffer.shape) == tuple(shape), (
                    f"recv buffer shape {recv_buffer.shape} does not match {shape}")
                assert recv_buffer.dtype == dtype, (
                    f"recv buffer dtype {recv_buffer.dtype} does not match {dtype}")
                buf = recv_buffer
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
            # NCCL recv is async. Keep synchronization stream-scoped while
            # holding the shared PP/KV NCCL lock.
            if synchronize:
                if stream is not None:
                    stream.synchronize()
                else:
                    current_stream().synchronize()
                logger.info(f"[PairPipe.recv_data] NCCL recv stream synchronized")
            # else:
            #     # No ACK needed, but still use lock for safety
            #     logger.info(f"[PairPipe.recv_data] Acquiring NCCL lock for recv (no ACK)")
            #     self._nccl.recv(buf, src=self.peer_rank, stream=stream)
            #     if stream is not None:
            #         stream.synchronize()
            #     else:
            #         current_stream().synchronize()
            #     logger.info(f"[PairPipe.recv_data] NCCL lock released after recv")
            logger.info(f"[PairPipe.recv_data] NCCL recv enqueued")
        else:
            buf = self.data_group.recv_obj(src=self.peer_rank)
            assert isinstance(buf, torch.Tensor), "The object should be a torch.Tensor"
            # For non-NCCL mode, send ACK after recv if needed
            if send_ack:
                self.meta_group.send_obj("ACK", dst=self.peer_rank)
        logger.info(f"[PairPipe.recv_data] Finished recv from peer {self.peer_rank}, tensor_shape={buf.shape}, duration={time.time() - time_start:.2f}s")
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
                                                  FlexiKVTensorMeta,
                                                  KVCachedSparseKVTensorMeta])

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
        self._kvcached_page_lifetime_lock = threading.RLock()
        self._kvcached_invalidated_block_ids: set[int] = set()
        
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
        self.kv_cache_start_layer = 0

    def _sync_kvcached_stream(self, reason: str) -> None:
        if not use_kvcached_backend():
            return
        sync_start = time.time()
        try:
            current_stream().synchronize()
        except Exception:
            logger.exception(
                "KVCacheD stream sync failed after %s trace_ns=%s",
                reason, time.time_ns())
            raise
        logger.info(
            "KVCacheD synchronized CUDA stream after %s, took %.6fs",
            reason, time.time() - sync_start)
    
    def get_nccl_lock(self) -> threading.Lock:
        """Return the NCCL lock for external use (e.g., Ray compiled_dag)."""
        return self._nccl_lock

    def kvcached_page_lifetime_context(self) -> threading.RLock:
        return self._kvcached_page_lifetime_lock

    def _kvcached_offsets_to_pages_and_blocks(
        self,
        offsets: list[int],
    ) -> tuple[list[int], list[int]]:
        geometry = getattr(self, "_kvcached_debug_geometry", None)
        if not offsets or not geometry:
            return [], []

        map_page_size = max(1, int(geometry["physical_group_page_size"]))
        blocks_per_page = max(1, int(geometry["blocks_per_physical_page"]))

        page_ids: set[int] = set()
        if geometry.get("layer_group_layout"):
            group_total_span = max(1, int(geometry["group_total_span"]))
            for offset in offsets:
                page_ids.add((int(offset) % group_total_span) //
                             map_page_size)
        else:
            for offset in offsets:
                page_ids.add(int(offset) // map_page_size)

        sorted_pages = sorted(page_ids)
        block_ids = [
            page_id * blocks_per_page + block_offset
            for page_id in sorted_pages
            for block_offset in range(blocks_per_page)
        ]
        return sorted_pages, block_ids

    def notify_kvcached_map_offsets(
        self,
        offsets: list[int],
        group_id: int = 0,
        reason: str = "",
    ) -> dict[str, Any]:
        """Mark KVCacheD pages as mapped again after a page allocation."""
        page_ids, block_ids = self._kvcached_offsets_to_pages_and_blocks(
            offsets)
        if not block_ids:
            return {}
        with self.kvcached_page_lifetime_context():
            before = len(self._kvcached_invalidated_block_ids)
            self._kvcached_invalidated_block_ids.difference_update(block_ids)
            cleared_blocks = before - len(self._kvcached_invalidated_block_ids)

        if cleared_blocks:
            logger.info(
                "[KVCACHED_MIGRATION_BITMAP_TRACE] phase=post_map_restore "
                "rank=%s group_id=%s reason=%s offsets=%d pages=%d "
                "cleared_blocks=%d sample_pages=%s sample_blocks=%s",
                self.rank, group_id, reason, len(offsets), len(page_ids),
                cleared_blocks, page_ids[:64], block_ids[:64])
        return {
            "pages": len(page_ids),
            "blocks": len(block_ids),
            "cleared_blocks": cleared_blocks,
        }

    def filter_slots_for_kvcached_live_blocks(
        self,
        slots: list[int],
        block_size: int,
        reason: str = "",
    ) -> list[int]:
        """Drop slots whose pages were invalidated by the unmap path."""
        if not slots or not use_kvcached_backend():
            return list(slots)
        block_size = int(block_size)
        if block_size <= 0:
            return list(slots)

        with self.kvcached_page_lifetime_context():
            invalidated = set(self._kvcached_invalidated_block_ids)
            if not invalidated:
                return list(slots)
            kept = [
                int(slot) for slot in slots
                if int(slot) < 0 or int(slot) // block_size not in invalidated
            ]

        dropped = len(slots) - len(kept)
        if dropped:
            dropped_blocks = sorted({
                int(slot) // block_size
                for slot in slots
                if int(slot) >= 0
                and int(slot) // block_size in invalidated
            })
            logger.info(
                "[KVCACHED_MIGRATION_BITMAP_TRACE] "
                "phase=invalidated_slot_filter rank=%s reason=%s "
                "original_slots=%d kept_slots=%d dropped_slots=%d "
                "sample_blocks=%s",
                self.rank, reason, len(slots), len(kept), dropped,
                dropped_blocks[:64])
        return kept

    def kvcached_patch_apply_ranges_for_live_blocks_locked(
        self,
        slot_mapping: torch.Tensor,
        block_size: int,
        reason: str = "",
    ) -> tuple[list[tuple[int, int]], int]:
        """Return contiguous token ranges that skip invalidated blocks.

        Caller must hold the KVCacheD page lifetime lock until the selected
        slots have been written and the write stream has been synchronized.
        """
        full_range = (
            [(0, int(slot_mapping.numel()))]
            if slot_mapping is not None and slot_mapping.numel() > 0 else [])
        if (not use_kvcached_backend() or slot_mapping is None
                or slot_mapping.numel() == 0):
            return full_range, 0
        block_size = int(block_size)
        if block_size <= 0:
            return full_range, 0

        invalidated = set(self._kvcached_invalidated_block_ids)
        if not invalidated:
            return full_range, 0
        slot_values = [
            int(slot) for slot in slot_mapping.detach().cpu().tolist()
        ]
        keep_indices = [
            idx for idx, slot in enumerate(slot_values)
            if slot < 0 or slot // block_size not in invalidated
        ]

        dropped = len(slot_values) - len(keep_indices)
        if dropped <= 0:
            return full_range, 0

        ranges: list[tuple[int, int]] = []
        range_start: Optional[int] = None
        prev_idx: Optional[int] = None
        for idx in keep_indices:
            if range_start is None:
                range_start = idx
                prev_idx = idx
            elif prev_idx is not None and idx == prev_idx + 1:
                prev_idx = idx
            else:
                ranges.append((range_start, int(prev_idx) + 1))
                range_start = idx
                prev_idx = idx
        if range_start is not None and prev_idx is not None:
            ranges.append((range_start, int(prev_idx) + 1))

        dropped_blocks = sorted({
            slot // block_size
            for slot in slot_values
            if slot >= 0 and slot // block_size in invalidated
        })
        logger.info(
            "[KVCACHED_MIGRATION_BITMAP_TRACE] "
            "phase=incoming_patch_apply_skip rank=%s reason=%s "
            "original_slots=%d kept_slots=%d dropped_slots=%d "
            "ranges=%d sample_blocks=%s",
            self.rank, reason, len(slot_values),
            sum(end - start for start, end in ranges), dropped, len(ranges),
            dropped_blocks[:64])
        return ranges, dropped

    def create_slot_mappings(self, num_block: int) -> None:
        """Initialize slot mapping for each peer rank.

        Args:
            num_block: Maximum number of token blocks that can be stored.
        """
        pp_size = int(self.vllm_config.parallel_config.pipeline_parallel_size)

        with self.kvcached_page_lifetime_context():
            for peer in range(pp_size):
                if peer == self.rank:
                    continue
                existing = self.slot_mappings.get(peer)
                if existing is None:
                    self.slot_mappings[peer] = KVSlotMapping(num_block)
                else:
                    existing.resize(num_block)

    def drop_slots_for_kvcached_unmap_offsets(
        self,
        offsets: list[int],
        group_id: int = 0,
        reason: str = "",
    ) -> dict[str, Any]:
        """Remove KVCacheD pages that are about to be unmapped from bitmaps."""
        if not offsets:
            return {}
        geometry = getattr(self, "_kvcached_debug_geometry", None)
        if not geometry:
            return {}
        block_size = max(1, int(geometry["block_size"]))
        page_ids, block_ids = self._kvcached_offsets_to_pages_and_blocks(
            offsets)
        total_cleared = 0
        per_peer: dict[int, dict[str, int]] = {}
        with self.kvcached_page_lifetime_context():
            self._kvcached_invalidated_block_ids.update(block_ids)
            for peer, slot_mapping in self.slot_mappings.items():
                stats = slot_mapping.clear_slots_for_blocks(
                    block_ids, block_size)
                if stats.get("cleared_slots", 0):
                    per_peer[int(peer)] = stats
                    total_cleared += int(stats["cleared_slots"])

        if total_cleared:
            logger.info(
                "[KVCACHED_MIGRATION_BITMAP_TRACE] phase=pre_unmap_clear "
                "rank=%s group_id=%s reason=%s offsets=%d pages=%d "
                "blocks=%d cleared_slots=%d peers=%s "
                "sample_pages=%s sample_blocks=%s",
                self.rank, group_id, reason, len(offsets), len(page_ids),
                len(block_ids), total_cleared, per_peer, page_ids[:64],
                block_ids[:64])
        return {
            "pages": len(page_ids),
            "blocks": len(block_ids),
            "cleared_slots": total_cleared,
            "peers": per_peer,
        }

    def drop_slots_for_kvcached_unmapped_offsets(
        self,
        offsets: list[int],
        group_id: int = 0,
        reason: str = "",
    ) -> dict[str, Any]:
        return self.drop_slots_for_kvcached_unmap_offsets(
            offsets,
            group_id=group_id,
            reason=reason,
        )

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

    def _send_meta_to_rank(
        self, rank: int,
        meta: Union[KVTensorMeta, KVPatchMeta, FlexiKVTensorMeta,
                    KVCachedSparseKVTensorMeta]) -> None:
        pipe = self._ensure_pipe_and_buffer(rank, 'send')
        pipe = self._pair_pipes_send[rank]
        pipe.send_meta(meta)

    def _send_data_to_rank(self, rank: int, kv_cache: torch.Tensor, synchronize: bool = True, wait_for_ack: bool = False, wait_current_stream: bool = True) -> None:
        """发送 KV 数据到指定 rank
        
        Args:
            rank: 目标 rank
            kv_cache: 要发送的 KV cache tensor
            synchronize: 是否等待 CUDA 操作完成（本地同步）
            wait_for_ack: 是否等待接收方确认（远程同步）
        """
        pipe = self._ensure_pipe_and_buffer(rank, 'send')
        pipe = self._pair_pipes_send[rank]
        
        if pipe.is_use_nccl and pipe._kv_transfer_stream is not None:
            transfer_stream = pipe._kv_transfer_stream
            if kv_cache.is_cuda and wait_current_stream:
                with torch.cuda.device(kv_cache.device):
                    transfer_stream.wait_stream(current_stream())
            pipe.send_data(kv_cache,
                           stream=transfer_stream,
                           wait_for_ack=wait_for_ack)
            if synchronize:
                transfer_stream.synchronize()
        else:
            pipe.send_data(kv_cache, wait_for_ack=wait_for_ack)
            if synchronize:
                current_stream().synchronize()

    def _send_slot_mapping_to_rank(self, rank: int,
                                   slot_mapping: torch.Tensor,
                                   synchronize: bool = True,
                                   wait_current_stream: bool = True) -> None:
        """Send slot mapping over the same NCCL data path as KV payload."""
        self._send_data_to_rank(rank, slot_mapping, synchronize=synchronize,
                                wait_for_ack=False,
                                wait_current_stream=wait_current_stream)
        logger.info("[slot_mapping_nccl]: sent slot mapping to rank %s, "
                    "shape=%s", rank, tuple(slot_mapping.shape))

    def _recv_slot_mapping_from_rank(self, rank: int,
                                     dtype: torch.dtype,
                                     shape: torch.Size,
                                     synchronize: bool = True,
                                     recv_buffer: Optional[torch.Tensor] = None
                                     ) -> torch.Tensor:
        pipe = self._ensure_pipe_and_buffer(rank, 'recv')
        pipe = self._pair_pipes_recv[rank]
        slot_mapping = self._recv_data_from_rank(rank, dtype, shape,
                                                 send_ack=False,
                                                 synchronize=synchronize,
                                                 recv_buffer=recv_buffer)
        assert tuple(slot_mapping.shape) == tuple(shape), (
            f"slot_mapping shape {slot_mapping.shape} does not match {shape}")
        logger.info("[slot_mapping_nccl]: received slot mapping from rank %s, "
                    "shape=%s", rank, tuple(slot_mapping.shape))
        return slot_mapping

    def _recv_metadata_from_rank(
        self, rank: int
    ) -> Union[KVTensorMeta, KVPatchMeta, FlexiKVTensorMeta,
               KVCachedSparseKVTensorMeta]:
        """Block to receive one metadata entry from peer.

        Returns a dict with keys: "dtype", "shape", "layer_id".
        """
        pipe = self._ensure_pipe_and_buffer(rank, 'recv')
        pipe = self._pair_pipes_recv[rank]
        return self._control_adapter.validate_python(pipe.recv_meta())

    def _recv_data_from_rank(self, rank: int,
                               dtype: torch.dtype,
                               shape: torch.Size,
                               send_ack: bool = False,
                               synchronize: bool = True,
                               recv_buffer: Optional[torch.Tensor] = None
                               ) -> torch.Tensor:
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
            stream = pipe._kv_transfer_stream if pipe.is_use_nccl else None
            return pipe.recv_data(dtype, shape,
                                  stream=stream,
                                  send_ack=send_ack,
                                  synchronize=synchronize,
                                  recv_buffer=recv_buffer)

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
        
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
        current_stream().synchronize()
        
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
        
        For async_fast migration, slot_mapping contains the token indices whose KV cache
        needs to be transferred. We gather KV data using flexi_gather_pages and send it.
        
        For sync migration without running requests, slot_mapping may be empty/None,
        in which case we send empty KV tensors.
        """
        # Get page metadata for gathering KV data
        assert len(self.key_cache_ptrs) > 0, "key_cache_ptrs not initialized for flexi mode"
        assert len(self.value_cache_ptrs) > 0, "value_cache_ptrs not initialized for flexi mode"
        
        # Get dimensions from config
        num_heads = self.num_heads
        head_dim = self.head_size
        block_size = self.vllm_config.cache_config.block_size
        
        # Convert slot_mapping to tensor if provided
        if slot_mapping is not None and len(slot_mapping) > 0:
            if isinstance(slot_mapping, torch.Tensor):
                slot_mapping_tensor = slot_mapping.to(device='cuda', dtype=torch.int64)
            else:
                slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int64, device='cuda')
            num_tokens = slot_mapping_tensor.size(0)
        else:
            slot_mapping_tensor = torch.empty(0, dtype=torch.int64, device='cuda')
            num_tokens = 0
        
        logger.info(f"[flexi sync]: rank {self.rank} sending to rank {rank}, layers={layer_ids}, num_tokens={num_tokens}")
        
        for layer_id in layer_ids:
            local_layer_id = layer_id - start_layer_id
            
            if num_tokens > 0:
                # Allocate KV output tensor: [2, num_tokens, num_heads, head_dim]
                # dim 0 = K/V, dim 1 = tokens
                kv_out = torch.empty(2, num_tokens, num_heads, head_dim, dtype=torch.float16, device='cuda')
                
                # Gather KV data from pointer-based storage
                key_cache_ptr = self.key_cache_ptrs[local_layer_id]
                value_cache_ptr = self.value_cache_ptrs[local_layer_id]
                ops.flexi_gather_pages(key_cache_ptr, value_cache_ptr, slot_mapping_tensor, kv_out[0], kv_out[1], block_size)
            else:
                # No tokens to transfer - send empty KV tensor
                kv_out = torch.empty(2, 0, num_heads, head_dim, dtype=torch.float16, device='cuda')
            
            kv_tensor_meta = FlexiKVTensorMeta(
                type='kv_tensor',
                layer_to_be_received=set(layer_ids),
                layer_id=int(layer_id),
                num_tokens=num_tokens,
                slot_mapping_dtype=slot_mapping_tensor.dtype,
                slot_mapping_shape=slot_mapping_tensor.shape,
                kv_payload_dtype=kv_out.dtype,
                kv_payload_shape=kv_out.shape)
            
            # Use send_kv_tensor_to_rank interface
            self.send_kv_tensor_to_rank(rank, kv_tensor_meta, kv_out, slot_mapping_tensor)


    # ########################################## #
    # Sender side functions                      #
    # ########################################## #

    def get_kv_patch(
            self,
            rank: int,
            start_layer_id: int,
            layer_ids: list[int],
            kv_meta: torch.Tensor,
            cuda_op_lock=None,
            key_cache_ptrs: Optional[list[int]] = None,
            value_cache_ptrs: Optional[list[int]] = None,
            kv_caches: Optional[list[torch.Tensor]] = None,
            live_block_ids: Optional[set[int]] = None,
            live_block_ids_getter: Optional[Callable[[], set[int]]] = None,
            block_size: Optional[int] = None,
    ) -> Generator[KVPatch, None, None]:
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
        if is_flexi:
            if key_cache_ptrs is None:
                key_cache_ptrs = self.key_cache_ptrs
            if value_cache_ptrs is None:
                value_cache_ptrs = self.value_cache_ptrs
            return self._flexi_get_patch(rank,
                                            layer_ids=layer_ids,
                                            kv_cache_meta=kv_meta,
                                            key_cache_ptrs=key_cache_ptrs,
                                            value_cache_ptrs=value_cache_ptrs,
                                            start_layer_id=start_layer_id,
                                            cuda_op_lock=cuda_op_lock,
                                            block_size=block_size)
        else:
            if kv_caches is None:
                kv_caches = self.kv_caches
            return self._get_patch(rank,
                                    layer_ids,
                                    kv_caches,
                                    start_layer_id,
                                    cuda_op_lock=cuda_op_lock,
                                    live_block_ids=live_block_ids,
                                    live_block_ids_getter=live_block_ids_getter,
                                    block_size=block_size)

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
                        start_layer_id: int,
                        cuda_op_lock=None,
                        block_size: Optional[int] = None,
                        ) -> Generator[KVPatch, None, None]:
        while True:
            logger.info(f"start to get slot mapping for rank {rank}")
            meta_block_size, num_head, head_dim = kv_cache_meta.shape
            patch_block_size = (
                int(block_size) if block_size is not None
                and int(block_size) > 0 else int(meta_block_size))
            snapshot: Optional[KVSlotSnapshot] = None
            kv_patch: Optional[KVPatch] = None
            if use_kvcached_backend():
                self.slot_mappings[rank].wait_for_slot_mappings()
                lock_context = self.kvcached_page_lifetime_context()
            else:
                lock_context = (cuda_op_lock
                                if cuda_op_lock is not None else nullcontext())
            with lock_context:
                try:
                    if use_kvcached_backend():
                        snapshot = (
                            self.slot_mappings[rank]
                            .pop_slot_mapping_snapshot())
                        if snapshot is None:
                            continue
                        slot_mapping = snapshot.slot_mapping
                        is_finished = snapshot.is_finished
                        stored_tokens = snapshot.stored_tokens
                        scheduler_trace = snapshot.trace_info
                    else:
                        slot_mapping, is_finished, stored_tokens, scheduler_trace = (
                            self.slot_mappings[rank].get_all_slot_mappings())
                    slot_mapping_dev = torch.tensor(slot_mapping,
                                                    device=kv_cache_meta.device,
                                                    dtype=torch.int64)
                    # 为了匹配receiver端的期望，使用shape: [2, num_layers, num_tokens, num_heads, head_dim]
                    # 第0维是K/V区分，第1维是layers
                    kv_out = torch.empty(2, len(layer_ids),
                                         slot_mapping_dev.size(0), num_head,
                                         head_dim, dtype=kv_cache_meta.dtype,
                                         device=kv_cache_meta.device)
                    logger.info(f"layer_ids: {layer_ids}, kv_out shape: {kv_out.shape}, slot mapping shape: {slot_mapping_dev.shape}, start_layer_id: {start_layer_id}")
                    if slot_mapping_dev.numel() > 0:
                        if use_kvcached_backend():
                            guard_kvcached_vmm_slots_mapped(
                                "migration_flexi_patch_gather_snapshot:"
                                f"target={rank}:patch={self.last_patch_ids[rank]}:"
                                f"trace={scheduler_trace}",
                                self,
                                slot_mapping,
                                patch_block_size,
                                layer_ids=[
                                    int(layer_id) for layer_id in layer_ids
                                ],
                                rank=self.rank,
                                trace_info=scheduler_trace,
                            )
                        for idx, layer_id in enumerate(layer_ids):
                            local_layer_id = layer_id - start_layer_id
                            if (local_layer_id < 0
                                    or local_layer_id >= len(key_cache_ptrs)):
                                raise IndexError(
                                    "KV patch sender layer index out of range: "
                                    f"layer_id={layer_id}, "
                                    f"start_layer_id={start_layer_id}, "
                                    f"local_layer_id={local_layer_id}, "
                                    f"num_key_cache_ptrs={len(key_cache_ptrs)}")
                            key_cache_ptr = key_cache_ptrs[local_layer_id]
                            value_cache_ptr = value_cache_ptrs[local_layer_id]
                            if key_cache_ptr == 0 or value_cache_ptr == 0:
                                raise RuntimeError(
                                    "KV patch sender resolved an empty cache "
                                    "pointer for non-empty slot mapping: "
                                    f"layer_id={layer_id}, "
                                    f"start_layer_id={start_layer_id}, "
                                    f"local_layer_id={local_layer_id}, "
                                    f"num_key_cache_ptrs={len(key_cache_ptrs)}")
                            # 填充到 kv_out[0][idx] (keys) 和 kv_out[1][idx] (values)
                            ops.flexi_gather_pages(key_cache_ptr, value_cache_ptr, slot_mapping_dev, kv_out[0][idx], kv_out[1][idx], patch_block_size)
                    self._sync_kvcached_stream("flexi KV patch gather")
                    meta_num_tokens = (
                        snapshot.meta_num_tokens()
                        if snapshot is not None else stored_tokens)
                    kv_patch = KVPatch(
                        KVPatchMeta(
                            type=('kv_patch_meta' if not is_finished else
                                  "kv_patch_finished"),
                            id=self.last_patch_ids[rank],
                            layer_ids=layer_ids,
                            num_tokens=meta_num_tokens,
                            slot_mapping_dtype=torch.int64,
                            slot_mapping_shape=slot_mapping_dev.shape,
                            kv_payload_dtype=kv_out.dtype,
                            kv_payload_shape=kv_out.shape,
                            scheduler_trace=scheduler_trace,
                        ),
                        kv_out,
                        slot_mapping_dev,
                    )
                finally:
                    if snapshot is not None:
                        snapshot.release()
            assert kv_patch is not None
            yield kv_patch
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
                        start_layer_id: int,
                        cuda_op_lock=None,
                        live_block_ids: Optional[set[int]] = None,
                        live_block_ids_getter: Optional[Callable[[], set[int]]] = None,
                        block_size: Optional[int] = None) -> Generator[KVPatch, None, None]:
        while True:
            patch_id = self.last_patch_ids[rank]
            logger.info(f"start to get slot mapping for rank {rank}")
            snapshot: Optional[KVSlotSnapshot] = None
            if use_kvcached_backend():
                self.slot_mappings[rank].wait_for_slot_mappings()
                lock_context = self.kvcached_page_lifetime_context()
            else:
                lock_context = (cuda_op_lock
                                if cuda_op_lock is not None else nullcontext())
            empty_patch: Optional[KVPatch] = None
            kv_patch: Optional[KVPatch] = None
            is_finished = False

            with lock_context:
                try:
                    if use_kvcached_backend():
                        snapshot = (
                            self.slot_mappings[rank]
                            .pop_slot_mapping_snapshot())
                        if snapshot is None:
                            continue
                        slot_mapping = snapshot.slot_mapping
                        is_finished = snapshot.is_finished
                        stored_tokens = snapshot.stored_tokens
                        scheduler_trace = snapshot.trace_info
                    else:
                        slot_mapping, is_finished, stored_tokens, scheduler_trace = (
                            self.slot_mappings[rank].get_all_slot_mappings())

                    original_slot_count = len(slot_mapping)
                    live_blocks_for_patch = (
                        live_block_ids_getter()
                        if live_block_ids_getter is not None else live_block_ids)
                    if (slot_mapping and live_blocks_for_patch is not None
                            and block_size is not None and block_size > 0):
                        original_slot_mapping = list(slot_mapping)
                        filtered_slot_mapping = [
                            slot for slot in slot_mapping
                            if int(slot) // int(block_size) in
                            live_blocks_for_patch
                        ]
                        dropped_slots = (original_slot_count -
                                         len(filtered_slot_mapping))
                        if dropped_slots:
                            dropped_slot_mapping = [
                                slot for slot in original_slot_mapping
                                if int(slot) // int(block_size) not in
                                live_blocks_for_patch
                            ]
                            kept_blocks = _blocks_from_slots(
                                filtered_slot_mapping, block_size)
                            dropped_blocks = _blocks_from_slots(
                                dropped_slot_mapping, block_size)
                            logger.info(
                                "Filtered KV patch slots against live request blocks: "
                                "rank=%s patch_id=%s original_slots=%s kept_slots=%s "
                                "dropped_slots=%s live_blocks=%s stored_tokens=%s",
                                rank, patch_id, original_slot_count,
                                len(filtered_slot_mapping), dropped_slots,
                                len(live_blocks_for_patch), stored_tokens)
                            if _autoscaling_kvcached_debug_enabled():
                                if _autoscaling_kvcached_debug_verbose():
                                    logger.warning(
                                        "[KVCACHED_MIGRATION_PATCH_TRACE] "
                                        "sender_rank=%s target_rank=%s patch_id=%s "
                                        "phase=filtered original_slots=%s kept_slots=%s "
                                        "dropped_slots=%s block_size=%s "
                                        "kept_blocks=%s dropped_blocks=%s "
                                        "stored_tokens=%s is_finished=%s",
                                        self.rank, rank, patch_id,
                                        original_slot_count,
                                        len(filtered_slot_mapping),
                                        dropped_slots, block_size, kept_blocks,
                                        dropped_blocks, stored_tokens,
                                        is_finished)
                                else:
                                    logger.warning(
                                        "[KVCACHED_MIGRATION_PATCH_TRACE] "
                                        "sender_rank=%s target_rank=%s patch_id=%s "
                                        "phase=filtered original_slots=%s kept_slots=%s "
                                        "dropped_slots=%s block_size=%s "
                                        "kept_blocks_count=%s sample_kept_blocks=%s "
                                        "dropped_blocks_count=%s sample_dropped_blocks=%s "
                                        "stored_tokens=%s is_finished=%s",
                                        self.rank, rank, patch_id,
                                        original_slot_count,
                                        len(filtered_slot_mapping),
                                        dropped_slots, block_size,
                                        len(kept_blocks), kept_blocks[:64],
                                        len(dropped_blocks),
                                        dropped_blocks[:64], stored_tokens,
                                        is_finished)
                            slot_mapping = filtered_slot_mapping
                            if snapshot is not None:
                                snapshot.replace_slots(filtered_slot_mapping)

                    if is_finished and not slot_mapping:
                        dtype = torch.bfloat16
                        if kv_cache:
                            dtype = kv_cache[0][0].dtype
                        slot_mapping_dev = torch.empty(0,
                                                       device=self.device,
                                                       dtype=torch.int64)
                        kv_out = torch.empty(
                            2,
                            len(layer_ids),
                            0,
                            self.num_heads,
                            self.head_size,
                            dtype=dtype,
                            device=self.device)
                        empty_patch = KVPatch(
                            KVPatchMeta(
                                type='kv_patch_finished',
                                id=patch_id,
                                layer_ids=layer_ids,
                                num_tokens=0,
                                slot_mapping_dtype=torch.int64,
                                slot_mapping_shape=slot_mapping_dev.shape,
                                kv_payload_dtype=kv_out.dtype,
                                kv_payload_shape=kv_out.shape,
                                scheduler_trace=scheduler_trace,
                            ),
                            kv_out,
                            slot_mapping_dev,
                        )
                    elif not slot_mapping:
                        dtype = torch.bfloat16
                        device = self.device
                        if kv_cache:
                            dtype = kv_cache[0][0].dtype
                            device = kv_cache[0][0].device
                        slot_mapping_dev = torch.empty(0,
                                                       device=device,
                                                       dtype=torch.int64)
                        kv_out = torch.empty(
                            2,
                            len(layer_ids),
                            0,
                            self.num_heads,
                            self.head_size,
                            dtype=dtype,
                            device=device)
                        empty_patch = KVPatch(
                            KVPatchMeta(
                                type='kv_patch_meta',
                                id=patch_id,
                                layer_ids=layer_ids,
                                num_tokens=0,
                                slot_mapping_dtype=torch.int64,
                                slot_mapping_shape=slot_mapping_dev.shape,
                                kv_payload_dtype=kv_out.dtype,
                                kv_payload_shape=kv_out.shape,
                                scheduler_trace=scheduler_trace,
                            ),
                            kv_out,
                            slot_mapping_dev,
                        )
                    else:
                        if (_autoscaling_kvcached_debug_enabled()
                                and block_size is not None and block_size > 0):
                            patch_blocks = _blocks_from_slots(slot_mapping,
                                                              block_size)
                            if _autoscaling_kvcached_debug_verbose():
                                logger.warning(
                                    "[KVCACHED_MIGRATION_GATHER_TRACE] rank=%s "
                                    "target=%s phase=patch_gather patch_id=%s "
                                    "slot_count=%s block_size=%s blocks=%s "
                                    "stored_tokens=%s is_finished=%s scheduler_trace=%s",
                                    self.rank, rank, patch_id,
                                    len(slot_mapping), block_size,
                                    patch_blocks, stored_tokens, is_finished,
                                    scheduler_trace)
                            else:
                                logger.warning(
                                    "[KVCACHED_MIGRATION_GATHER_TRACE] rank=%s "
                                    "target=%s phase=patch_gather patch_id=%s "
                                    "slot_count=%s block_size=%s blocks_count=%s "
                                    "sample_blocks=%s stored_tokens=%s is_finished=%s "
                                    "scheduler_trace=%s",
                                    self.rank, rank, patch_id,
                                    len(slot_mapping), block_size,
                                    len(patch_blocks), patch_blocks[:64],
                                    stored_tokens, is_finished,
                                    scheduler_trace)

                        if (use_kvcached_backend() and block_size is not None
                                and block_size > 0):
                            guard_kvcached_vmm_slots_mapped(
                                "migration_patch_gather_snapshot:"
                                f"target={rank}:patch={patch_id}:"
                                f"trace={scheduler_trace}",
                                self,
                                slot_mapping,
                                int(block_size),
                                layer_ids=[
                                    int(layer_id) for layer_id in layer_ids
                                ],
                                rank=self.rank,
                                trace_info=scheduler_trace,
                            )
                        slot_mapping_dev = torch.tensor(slot_mapping,
                                                        dtype=torch.int64)
                        kv_patch = self.kv_helper.extract_kv_patch_from_kv_cache(
                            patch_id=patch_id,
                            kv_caches=kv_cache,
                            layer_ids=layer_ids,
                            start_layer_id=start_layer_id,
                            slot_mapping=slot_mapping_dev,
                            is_finished=is_finished,
                            slot_mapping_already_valid=True,
                        )
                        self._sync_kvcached_stream("regular KV patch gather")
                        kv_patch.meta.num_tokens = (
                            snapshot.meta_num_tokens()
                            if snapshot is not None else stored_tokens)
                        kv_patch.meta.scheduler_trace = scheduler_trace
                finally:
                    if snapshot is not None:
                        snapshot.release()
            if empty_patch is not None:
                yield empty_patch
                self.last_patch_ids[rank] += 1
                if is_finished:
                    self.kv_cache_transfer_in_process[rank] = False
                    self.kv_patch_sending = False
                    self.last_patch_ids[rank] = 0
                    break
                continue
            assert kv_patch is not None
            yield kv_patch
            self.last_patch_ids[rank] += 1
            if is_finished: 
                self.kv_cache_transfer_in_process[rank] = False
                self.kv_patch_sending = False
                self.last_patch_ids[rank] = 0
                break

    def get_kv_tensor_from_cache(
            self,
            layer_ids: list[int],
            layer_id: int,
            start_layer_id: int,
            kv_cache_meta: torch.Tensor,
            slot_mapping: Optional[torch.Tensor] = None,
            key_cache_ptrs: Optional[list[int]] = None,
            value_cache_ptrs: Optional[list[int]] = None,
            logical_num_blocks: Optional[int] = None,
            kv_caches: Optional[list[torch.Tensor]] = None,
            slot_mapping_already_valid: bool = False,
    ) -> Tuple[Union[KVTensorMeta, FlexiKVTensorMeta,
                     KVCachedSparseKVTensorMeta], torch.Tensor]:
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
        if is_flexi:
            assert slot_mapping is not None, "slot_mapping should be provided when using flexi flash attention"
            if key_cache_ptrs is None:
                key_cache_ptrs = self.key_cache_ptrs
            if value_cache_ptrs is None:
                value_cache_ptrs = self.value_cache_ptrs
            return self._flexi_get_kv_tensor(
                                        layer_id=layer_id,
                                        layer_ids=layer_ids,
                                        kv_cache_meta=kv_cache_meta,
                                        key_cache_ptrs=key_cache_ptrs,
                                        value_cache_ptrs=value_cache_ptrs,
                                        slot_mapping=slot_mapping,
                                        start_layer_id=start_layer_id)
        else:
            if kv_caches is None:
                kv_caches = self.kv_caches
            return self._regular_get_kv_tensor(
                layer_id, layer_ids, kv_caches, start_layer_id,
                slot_mapping=slot_mapping,
                logical_num_blocks=logical_num_blocks,
                slot_mapping_already_valid=slot_mapping_already_valid)

    def _regular_get_kv_tensor(self, layer_id: int, layer_ids: list[int], kv_caches: list[torch.Tensor],
                            start_layer_id: int,
                            slot_mapping: Optional[torch.Tensor] = None,
                            logical_num_blocks: Optional[int] = None,
                            slot_mapping_already_valid: bool = False,
                            ) -> Tuple[Union[KVTensorMeta,
                                             KVCachedSparseKVTensorMeta],
                                       torch.Tensor]:
        local_layer_id = layer_id - start_layer_id
        kv_cache = kv_caches[local_layer_id]
        if (use_kvcached_backend() and kv_cache.dim() == 5
                and kv_cache.size(0) == 2 and slot_mapping is not None):
            valid_slots = (slot_mapping if slot_mapping_already_valid else
                           slot_mapping[slot_mapping >= 0])
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
            if valid_slots.device != key_cache.device:
                valid_slots = valid_slots.to(key_cache.device,
                                             non_blocking=True)
            if valid_slots.numel() > 0:
                slot_min = int(valid_slots.min().item())
                slot_max = int(valid_slots.max().item())
                flat_capacity = self.kv_helper._flat_capacity_for_cache(
                    key_cache)
                if (_autoscaling_kvcached_debug_enabled()
                        and use_kvcached_backend()):
                    block_size_for_trace = int(key_cache.size(1))
                    slot_blocks = _blocks_from_slots(valid_slots,
                                                     block_size_for_trace)
                    if _autoscaling_kvcached_debug_verbose():
                        logger.warning(
                            "[KVCACHED_MIGRATION_GATHER_TRACE] rank=%s "
                            "target=unknown phase=initial_snapshot layer=%s "
                            "slot_count=%s slots_min=%s slots_max=%s "
                            "block_size=%s blocks=%s",
                            self.rank, layer_id, int(valid_slots.numel()),
                            slot_min, slot_max, block_size_for_trace,
                            slot_blocks)
                    else:
                        logger.warning(
                            "[KVCACHED_MIGRATION_GATHER_TRACE] rank=%s "
                            "target=unknown phase=initial_snapshot layer=%s "
                            "slot_count=%s slots_min=%s slots_max=%s "
                            "block_size=%s blocks_count=%s sample_blocks=%s",
                            self.rank, layer_id, int(valid_slots.numel()),
                            slot_min, slot_max, block_size_for_trace,
                            len(slot_blocks), slot_blocks[:64])
                if use_kvcached_backend():
                    guard_kvcached_vmm_slots_mapped(
                        f"migration_initial_snapshot:layer={layer_id}",
                        self,
                        valid_slots,
                        int(key_cache.size(1)),
                        layer_ids=[int(layer_id)],
                        rank=self.rank,
                    )
                if slot_max >= flat_capacity:
                    raise ValueError(
                        "KVCacheD sparse migration slot mapping is out of "
                        "bounds: "
                        f"layer={layer_id}, slot_min={slot_min}, "
                        f"slot_max={slot_max}, flat_capacity={flat_capacity}")
                gathered_k = self.kv_helper._select_kv_slots(
                    key_cache, valid_slots)
                gathered_v = self.kv_helper._select_kv_slots(
                    value_cache, valid_slots)
                kv_payload = torch.stack((gathered_k, gathered_v),
                                         dim=0).contiguous()
            else:
                kv_payload = torch.empty(
                    2,
                    0,
                    int(kv_cache.size(-2)),
                    int(kv_cache.size(-1)),
                    dtype=kv_cache.dtype,
                    device=kv_cache.device)
            self._sync_kvcached_stream("KVCacheD sparse KV tensor snapshot")
            logger.info(
                "KVCacheD migration sends sparse live-slot tensor for "
                "layer %s: live_tokens=%s cache_shape=%s payload_shape=%s",
                layer_id, valid_slots.numel(), tuple(kv_cache.shape),
                tuple(kv_payload.shape))
            return KVCachedSparseKVTensorMeta(
                type='kv_tensor_sparse',
                layer_to_be_received=set(layer_ids),
                layer_id=int(layer_id),
                num_tokens=int(valid_slots.numel()),
                slot_mapping_dtype=valid_slots.dtype,
                slot_mapping_shape=valid_slots.shape,
                kv_payload_dtype=kv_payload.dtype,
                kv_payload_shape=kv_payload.shape), kv_payload
        if (use_kvcached_backend() and kv_cache.dim() == 5
                and kv_cache.size(0) == 2):
            send_num_blocks = kv_cache.size(1)
            if logical_num_blocks is not None:
                send_num_blocks = min(logical_num_blocks, send_num_blocks)
            if 0 < send_num_blocks <= kv_cache.size(1):
                if send_num_blocks != kv_cache.size(1):
                    logger.info(
                        "KVCacheD migration sends logical KV window for layer "
                        "%s: blocks %s -> %s",
                        layer_id, kv_cache.size(1), send_num_blocks)
                logger.info(
                    "KVCacheD migration materializes sender snapshot for "
                    "layer %s: shape=%s, send_blocks=%s",
                    layer_id, tuple(kv_cache.shape), send_num_blocks)
                kv_cache = kv_cache[:, :send_num_blocks, ...].contiguous()
                self._sync_kvcached_stream("regular KV tensor snapshot")
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
        if local_layer_id < 0 or local_layer_id >= len(key_cache_ptrs):
            raise IndexError(
                "KV sender layer index out of range: "
                f"layer_id={layer_id}, start_layer_id={start_layer_id}, "
                f"local_layer_id={local_layer_id}, "
                f"num_key_cache_ptrs={len(key_cache_ptrs)}")
        key_cache_ptr = key_cache_ptrs[local_layer_id]
        value_cache_ptr = value_cache_ptrs[local_layer_id]
        block_size, num_head, head_dim = kv_cache_meta.shape
        kv_out = torch.empty(2, slot_mapping.size(0), num_head, head_dim, dtype=kv_cache_meta.dtype, device=kv_cache_meta.device)
        logger.info(f"kv out device: {kv_out.device}, slot mapping device: {slot_mapping.device}")
        if slot_mapping.numel() > 0:
            if key_cache_ptr == 0 or value_cache_ptr == 0:
                raise RuntimeError(
                    "KV sender resolved an empty cache pointer for non-empty "
                    "slot mapping: "
                    f"layer_id={layer_id}, start_layer_id={start_layer_id}, "
                    f"local_layer_id={local_layer_id}, "
                    f"num_key_cache_ptrs={len(key_cache_ptrs)}")
            ops.flexi_gather_pages(key_cache_ptr, value_cache_ptr,
                                   slot_mapping, kv_out[0], kv_out[1],
                                   block_size)
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

    def send_kv_tensor_to_rank(self,
            rank: int,
            kv_tensor_meta: Union[KVTensorMeta, FlexiKVTensorMeta,
                                  KVCachedSparseKVTensorMeta],
            kv_tensor: torch.Tensor,
            slot_mapping: Optional[torch.Tensor] = None,
            cuda_op_lock=None) -> None:
        """Send a single KV tensor to a peer rank with deadlock-free protocol.
        
        Protocol:
        1. Send meta without holding the local nccl_lock.
        2. Wait for ACCEPT/REJECT from receiver.
        3. If ACCEPT: acquire nccl_lock, send each NCCL payload and synchronize it before
           enqueueing the next one, then release lock. The same lock is used
           by PP NCCL, so this keeps KV and PP collectives mutually exclusive
           on the local GPU.
        4. If REJECT: backoff, retry from step 1.
        """
        self.kv_cache_transfer_in_process[rank] = True
        # Ensure pipe exists before using it
        self._ensure_pipe_and_buffer(rank, 'send')
        pipe = self._pair_pipes_send[rank]
        nccl_lock = pipe._nccl_lock or self._nccl_lock
        slot_mapping_to_send = slot_mapping
        if not isinstance(kv_tensor_meta, KVTensorMeta):
            assert slot_mapping is not None, (
                "slot_mapping should be provided when sending "
                "slot-mapped KV tensor metadata")
        
        retry_delay = 0.005  # 5ms fixed retry delay
        
        while True:
            should_retry = False
            # Keep the control-plane handshake outside the local NCCL lock.
            # The receiver may wait for its local forward pass to drain before
            # accepting, and holding this rank's PP/KV lock during that wait can
            # block foreground PP progress.
            self._send_meta_to_rank(rank, kv_tensor_meta)
            logger.info(f"[send_kv_tensor_to_rank]: sent meta to rank {rank}, waiting for ACCEPT/REJECT")

            while True:
                response = pipe.signal_group.recv_obj(src=pipe.peer_rank)
                if (_kv_signal_matches(response, "ACCEPT", kv_tensor_meta)
                        or _kv_signal_matches(response, "REJECT",
                                              kv_tensor_meta)):
                    break
                logger.warning(
                    "[send_kv_tensor_to_rank]: ignoring stale or mismatched "
                    "signal from rank %s for current meta %s: %s",
                    rank, _kv_signal_identity(kv_tensor_meta), response)

            if _kv_signal_matches(response, "ACCEPT", kv_tensor_meta):
                logger.info(f"[send_kv_tensor_to_rank]: received ACCEPT from rank {rank}")
                execution_context = (cuda_op_lock if cuda_op_lock is not None
                                     else nullcontext())
                with execution_context:
                    nccl_lock.acquire()
                    try:
                        if isinstance(kv_tensor_meta, KVTensorMeta):
                            self._send_data_to_rank(rank, kv_tensor,
                                                    wait_for_ack=False)
                        else:
                            assert slot_mapping_to_send is not None
                            self._send_slot_mapping_to_rank(
                                rank,
                                slot_mapping_to_send,
                                wait_current_stream=False)
                            logger.info(f"[send_kv_tensor_to_rank]: sent slot mapping to rank {rank}")
                            self._send_data_to_rank(rank, kv_tensor,
                                                    wait_for_ack=False,
                                                    wait_current_stream=False)
                        logger.info(f"[send_kv_tensor_to_rank]: sent kv tensor to rank {rank}")
                    finally:
                        nccl_lock.release()
                break  # Success
            elif _kv_signal_matches(response, "REJECT", kv_tensor_meta):
                logger.info(f"[send_kv_tensor_to_rank]: received REJECT from rank {rank}, retry in 5ms")
                should_retry = True
            else:
                raise RuntimeError(f"Unexpected response: {response}")
            if should_retry:
                time.sleep(retry_delay)
                continue
        logger.info(f"[send_kv_tensor_to_rank]: completed sending to rank {rank}")
    
    def send_kv_patch_to_rank(self, 
            target_rank: int,
            kv_patches: KVPatch,
            cuda_op_lock=None) -> None:
        """Send KV patch to target rank with deadlock-free protocol.
        
        Protocol:
        1. Send meta without holding the local nccl_lock.
        2. Wait for ACCEPT/REJECT from receiver.
        3. If ACCEPT: acquire nccl_lock, send each NCCL payload and synchronize it before
           enqueueing the next one, then release lock. The same lock is used
           by PP NCCL, so this keeps KV and PP collectives mutually exclusive
           on the local GPU.
        4. If REJECT: backoff, retry from step 1.
        """
        assert self.kv_cache_transfer_in_process[target_rank] == True, "The kv cache transfer in process of the rank should be True."
        meta, kv_payload, slot_mapping = kv_patches.meta, kv_patches.kv_payload, kv_patches.slot_mapping
        slot_mapping_to_send = slot_mapping
        time_start = time.time()
        # Ensure pipe exists (should already exist from send_kv_tensor_to_rank, but be safe)
        self._ensure_pipe_and_buffer(target_rank, 'send')
        pipe = self._pair_pipes_send[target_rank]
        nccl_lock = pipe._nccl_lock or self._nccl_lock
        
        retry_delay = 0.005  # 5ms fixed retry delay
        time_lock_acquired = time_start
        
        while True:
            should_retry = False
            self._send_meta_to_rank(target_rank, meta)
            logger.info(f"sent meta to rank {target_rank}, patch id: {meta.id}, waiting for ACCEPT/REJECT")

            while True:
                response = pipe.signal_group.recv_obj(src=pipe.peer_rank)
                if (_kv_signal_matches(response, "ACCEPT", meta)
                        or _kv_signal_matches(response, "REJECT", meta)):
                    break
                logger.warning(
                    "send_kv_patch_to_rank: ignoring stale or mismatched "
                    "signal from rank %s for current meta %s: %s",
                    target_rank, _kv_signal_identity(meta), response)

            if _kv_signal_matches(response, "ACCEPT", meta):
                logger.info(f"received ACCEPT from rank {target_rank}, sending data")
                execution_context = (cuda_op_lock if cuda_op_lock is not None
                                     else nullcontext())
                with execution_context:
                    nccl_lock.acquire()
                    time_lock_acquired = time.time()
                    try:
                        self._send_slot_mapping_to_rank(
                            target_rank,
                            slot_mapping_to_send,
                            wait_current_stream=False)
                        logger.info(f"sent slot mapping to rank {target_rank}, patch id: {meta.id}")
                        self._send_data_to_rank(target_rank, kv_payload,
                                                wait_for_ack=False,
                                                wait_current_stream=False)
                        logger.info(f"sent kv payload to rank {target_rank}, patch id: {meta.id}, shape:{kv_payload.shape}")
                    finally:
                        nccl_lock.release()
                self._wait_kv_patch_applied_ack(target_rank, meta.id)
                break  # Success, exit loop
            elif _kv_signal_matches(response, "REJECT", meta):
                logger.info(f"received REJECT from rank {target_rank}, retry in 5ms")
                should_retry = True
            else:
                raise RuntimeError(f"Unexpected response: {response}")
            if should_retry:
                time.sleep(retry_delay)
                continue  # Retry

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
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
        for rank in self.last_patch_ids:
            if is_flexi:
                with self.kvcached_page_lifetime_context():
                    self.slot_mappings[rank].add_slot_mappings(
                        [], is_finished=True, num_total_new_tokens=0)
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

    def add_new_tokens_to_kv_synchronizer(
        self,
        rank: int,
        slot_mapping: Union[torch.Tensor, list[int]],
        is_finished: bool,
        num_total_new_tokens: int,
        trace_info: Optional[dict[str, Any]] = None,
    ) -> None:
        slot_mapping_list = (slot_mapping.tolist()
                             if isinstance(slot_mapping, torch.Tensor)
                             else slot_mapping)
        with self.kvcached_page_lifetime_context():
            self.slot_mappings[rank].add_slot_mappings(
                slot_mapping_list,
                is_finished=is_finished,
                num_total_new_tokens=num_total_new_tokens,
                trace_info=trace_info)


    # ########################################## #
    # Receiver side functions                      #
    # ########################################## #
    def get_last_patch_id(self, rank: int) -> int:
        return self.last_patch_ids[rank]

    def recv_controller(
        self, rank: int
    ) -> Union[KVTensorMeta, KVPatchMeta, FlexiKVTensorMeta,
               KVCachedSparseKVTensorMeta]:
        pipe = self._ensure_pipe_and_buffer(rank, 'recv')
        pipe = self._pair_pipes_recv[rank]
        meta_obj = pipe.recv_meta()
        return self._control_adapter.validate_python(meta_obj)

    def recv_kv_tensor(self,
            from_rank: int,
            meta: Union[KVTensorMeta, FlexiKVTensorMeta,
                        KVCachedSparseKVTensorMeta],
            cuda_op_lock=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Receive KV tensor from sender with deadlock-free protocol.
        
        Protocol:
        1. Meta already received by caller
        2. Try to acquire nccl_lock (non-blocking)
        3. If failed: send REJECT, receive new meta (sender will retry), go to step 2
        4. If acquired: send ACCEPT, receive each NCCL payload and
           synchronize it before receiving the next one, then release lock.
           The same lock is used by PP NCCL, so this keeps KV and PP
           collectives mutually exclusive on the local GPU.
        """
        time_start = time.time()
        pipe = self._pair_pipes_recv[from_rank]
        nccl_lock = pipe._nccl_lock or self._nccl_lock
        
        current_meta: Union[KVTensorMeta, FlexiKVTensorMeta,
                            KVCachedSparseKVTensorMeta] = meta
        while True:
            execution_context = (cuda_op_lock if cuda_op_lock is not None
                                 else nullcontext())
            with execution_context:
                # Drain local forward before accepting a large receive. The
                # sender keeps its local NCCL lock free while waiting for our
                # response, so foreground PP on the sender can still progress.
                lock_acquired = nccl_lock.acquire(blocking=False)

                if lock_acquired:
                    try:
                        if isinstance(current_meta, (FlexiKVTensorMeta,
                                                     KVCachedSparseKVTensorMeta)):
                            slot_mapping_buffer = pipe._prepare_recv_buffer(
                                current_meta.slot_mapping_dtype,
                                current_meta.slot_mapping_shape)
                            kv_payload_buffer = pipe._prepare_recv_buffer(
                                current_meta.kv_payload_dtype,
                                current_meta.kv_payload_shape)
                        else:
                            slot_mapping_buffer = None
                            kv_payload_buffer = pipe._prepare_recv_buffer(
                                current_meta.dtype, current_meta.shape)

                        # Step 4: Lock acquired and buffers prepared. Only now
                        # tell the sender it can enqueue NCCL sends.
                        time_lock_acquired = time.time()
                        pipe.signal_group.send_obj(
                            _kv_transfer_signal("ACCEPT", current_meta),
                            dst=pipe.peer_rank)
                        logger.info(
                            "recv_kv_tensor: lock acquired in %ss, sent ACCEPT",
                            time_lock_acquired - time_start)

                        if isinstance(current_meta, (FlexiKVTensorMeta,
                                                     KVCachedSparseKVTensorMeta)):
                            slot_mapping = self._recv_slot_mapping_from_rank(
                                from_rank, current_meta.slot_mapping_dtype,
                                current_meta.slot_mapping_shape,
                                recv_buffer=slot_mapping_buffer)
                            logger.info(f"recv_kv_tensor: received slot mapping, took {time.time() - time_start}s")
                            kv_payload = self._recv_data_from_rank(
                                from_rank, current_meta.kv_payload_dtype,
                                current_meta.kv_payload_shape, send_ack=False,
                                synchronize=True,
                                recv_buffer=kv_payload_buffer)
                            logger.info(f"recv_kv_tensor: received kv payload, took {time.time() - time_start}s")
                        else:
                            kv_payload = self._recv_data_from_rank(
                                from_rank, current_meta.dtype,
                                current_meta.shape, send_ack=False,
                                recv_buffer=kv_payload_buffer)
                            slot_mapping = None
                    finally:
                        nccl_lock.release()
                    break  # Success, exit loop

            if not lock_acquired:
                # Step 3: Lock held by others - send REJECT immediately
                logger.info(f"recv_kv_tensor: lock held by others, sending REJECT")
                pipe.signal_group.send_obj(
                    _kv_transfer_signal("REJECT", current_meta),
                    dst=pipe.peer_rank)
                # Sender will backoff and resend meta, we need to receive it
                new_meta = self.recv_controller(from_rank)
                assert isinstance(
                    new_meta,
                    (KVTensorMeta, FlexiKVTensorMeta,
                     KVCachedSparseKVTensorMeta)), (
                         "Expected KVTensorMeta, FlexiKVTensorMeta, or "
                         f"KVCachedSparseKVTensorMeta, got {type(new_meta)}")
                current_meta = new_meta
                continue

        logger.info(f"recv_kv_tensor completed: lock wait: {time_lock_acquired - time_start}, total: {time.time() - time_start}")
        
        if isinstance(current_meta, (FlexiKVTensorMeta,
                                     KVCachedSparseKVTensorMeta)):
            assert slot_mapping is not None, (
                "slot_mapping should not be None for slot-mapped KV tensor")
            assert kv_payload.dim() == 4 and kv_payload.size(0) == 2
            return slot_mapping, kv_payload
        else:
            assert kv_payload.dim() == 5 and kv_payload.size(0) == 2, f"kv_payload shape {kv_payload.shape} is not correct"
            return kv_payload

    def recv_kv_patch(self,
            from_rank: int,
            meta: KVPatchMeta,
            cuda_op_lock=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Receive KV patch from sender with deadlock-free protocol.
        
        Protocol:
        1. Meta already received by caller
        2. Try to acquire nccl_lock (non-blocking)
        3. If failed: send REJECT, receive new meta (sender will retry), go to step 2
        4. If acquired: send ACCEPT, receive each NCCL payload and
           synchronize it before receiving the next one, then release lock.
           The same lock is used by PP NCCL, so this keeps KV and PP
           collectives mutually exclusive on the local GPU.
        """
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
        time_start = time.time()
        pipe = self._pair_pipes_recv[from_rank]
        nccl_lock = pipe._nccl_lock or self._nccl_lock
        
        current_meta = meta
        while True:
            execution_context = (cuda_op_lock if cuda_op_lock is not None
                                 else nullcontext())
            with execution_context:
                # Step 2: Try to acquire lock (non-blocking)
                lock_acquired = nccl_lock.acquire(blocking=False)

                if lock_acquired:
                    try:
                        slot_mapping_buffer = pipe._prepare_recv_buffer(
                            current_meta.slot_mapping_dtype,
                            current_meta.slot_mapping_shape)
                        kv_payload_buffer = pipe._prepare_recv_buffer(
                            current_meta.kv_payload_dtype,
                            current_meta.kv_payload_shape)

                        # Step 4: Lock acquired and buffers prepared. Only now
                        # tell the sender it can enqueue NCCL sends.
                        time_lock_acquired = time.time()
                        pipe.signal_group.send_obj(
                            _kv_transfer_signal("ACCEPT", current_meta),
                            dst=pipe.peer_rank)
                        logger.info(
                            "recv_kv_patch: lock acquired in %ss, sent ACCEPT",
                            time_lock_acquired - time_start)

                        slot_mapping = self._recv_slot_mapping_from_rank(
                            from_rank, current_meta.slot_mapping_dtype,
                            current_meta.slot_mapping_shape,
                            recv_buffer=slot_mapping_buffer)
                        kv_payload = self._recv_data_from_rank(
                            from_rank, current_meta.kv_payload_dtype,
                            current_meta.kv_payload_shape, False,
                            synchronize=True,
                            recv_buffer=kv_payload_buffer)
                    finally:
                        nccl_lock.release()
                    break  # Success, exit loop

            if not lock_acquired:
                # Step 3: Lock held by others - send REJECT immediately
                logger.info(f"recv_kv_patch: lock held by others, sending REJECT")
                pipe.signal_group.send_obj(
                    _kv_transfer_signal("REJECT", current_meta),
                    dst=pipe.peer_rank)
                # Sender will backoff and resend meta, we need to receive it
                new_meta = self.recv_controller(from_rank)
                assert isinstance(new_meta, KVPatchMeta), f"Expected KVPatchMeta, got {type(new_meta)}"
                current_meta = new_meta
                continue

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

    def notify_kv_patch_applied(
        self,
        from_rank: int,
        patch_id: int,
    ) -> None:
        pipe = self._pair_pipes_recv[from_rank]
        pipe.signal_group.send_obj({
            "type": "PATCH_APPLIED",
            "id": int(patch_id),
            "sender_rank": int(from_rank),
            "target_rank": int(self.rank),
        }, dst=pipe.peer_rank)

    def _wait_kv_patch_applied_ack(
        self,
        target_rank: int,
        patch_id: int,
    ) -> None:
        pipe = self._pair_pipes_send[target_rank]
        while True:
            response = pipe.signal_group.recv_obj(src=pipe.peer_rank)
            if (isinstance(response, dict)
                    and response.get("type") == "PATCH_APPLIED"
                    and int(response.get("id", -1)) == int(patch_id)
                    and int(response.get("sender_rank", self.rank)) == int(self.rank)
                    and int(response.get("target_rank", target_rank)) == int(target_rank)):
                break
            logger.warning(
                "Ignoring stale KV patch apply acknowledgement while waiting "
                "for target_rank=%s patch_id=%s: %s",
                target_rank, patch_id, response)
        logger.info(
            "received PATCH_APPLIED from rank %s for patch id %s",
            target_rank, patch_id)

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

    def apply_one_patch_to_kv_cache(
            self,
            start_layer_id: int,
            meta: KVPatchMeta,
            kv_payload: torch.Tensor,
            slot_mapping: torch.Tensor,
            page_meta: Optional[torch.Tensor] = None,
            key_cache_ptrs: Optional[list[int]] = None,
            value_cache_ptrs: Optional[list[int]] = None,
    ) -> int:
        logger.info(f"apply one patch with id {meta.id}, num_tokens: {meta.num_tokens}")
        is_flexi = use_flexi_kv_for_runtime(self.vllm_config)
        patch_block_size: Optional[int] = None
        if use_kvcached_backend() and slot_mapping.numel() > 0:
            if is_flexi:
                if page_meta is not None and page_meta.dim() > 0:
                    patch_block_size = int(page_meta.size(0))
            elif meta.layer_ids:
                first_local_layer_id = int(meta.layer_ids[0]) - start_layer_id
                if (0 <= first_local_layer_id
                        < len(getattr(self, "kv_caches", []))):
                    first_kv_cache = self.kv_caches[first_local_layer_id]
                    if first_kv_cache.dim() >= 3:
                        patch_block_size = int(first_kv_cache.size(2))
        keys = kv_payload[0]
        values = kv_payload[1]
        # assert slot_mapping.size(0) == meta.num_tokens, f"slot_mapping size {slot_mapping.size(0)} should match num_tokens {meta.num_tokens}"
        if slot_mapping.size(0) != meta.num_tokens:
            logger.info(f"Warning: slot_mapping size {slot_mapping.size(0)} does not match num_tokens {meta.num_tokens}, proceed anyway.")

        should_hold_page_lifetime = (
            use_kvcached_backend() and slot_mapping.numel() > 0
            and patch_block_size is not None and int(patch_block_size) > 0)
        patch_page_context = (
            self.kvcached_page_lifetime_context()
            if should_hold_page_lifetime else nullcontext())
        if is_flexi:
            if key_cache_ptrs is None:
                key_cache_ptrs = self.key_cache_ptrs
            if value_cache_ptrs is None:
                value_cache_ptrs = self.value_cache_ptrs

        # The operations below are executed in the default stream and will be properly ordered.
        with self.device:
            logger.info(f"[listen loop] apply kv patch to kv cache on device {self.device}")
            patch_trace_logged = False
            scheduler_trace = getattr(meta, "scheduler_trace", {})
            with patch_page_context:
                apply_ranges = ([(0, int(slot_mapping.numel()))]
                                if slot_mapping.numel() > 0 else [])
                if should_hold_page_lifetime:
                    apply_ranges, _ = (
                        self.kvcached_patch_apply_ranges_for_live_blocks_locked(
                            slot_mapping,
                            int(patch_block_size),
                            reason=(
                                f"patch_apply:patch={meta.id}:"
                                f"layers={meta.layer_ids}")))
                applied_num_tokens = sum(
                    end - start for start, end in apply_ranges)
                if applied_num_tokens == 0:
                    return 0

                for layer_id, key, value in zip(meta.layer_ids, keys, values):
                    local_layer_id = layer_id - start_layer_id
                    layer_block_size = patch_block_size
                    if is_flexi:
                        assert page_meta is not None, "page_meta should be provided when using flexi flash attention"
                        if (local_layer_id < 0
                                or local_layer_id >= len(key_cache_ptrs)):
                            raise IndexError(
                                "KV receiver patch layer index out of range: "
                                f"layer_id={layer_id}, "
                                f"start_layer_id={start_layer_id}, "
                                f"local_layer_id={local_layer_id}, "
                                f"num_key_cache_ptrs={len(key_cache_ptrs)}")
                        key_cache_ptr = key_cache_ptrs[local_layer_id]
                        value_cache_ptr = value_cache_ptrs[local_layer_id]
                        if key_cache_ptr == 0 or value_cache_ptr == 0:
                            raise RuntimeError(
                                "KV receiver patch resolved an empty cache "
                                "pointer: "
                                f"layer_id={layer_id}, "
                                f"start_layer_id={start_layer_id}, "
                                f"local_layer_id={local_layer_id}, "
                                f"num_key_cache_ptrs={len(key_cache_ptrs)}")
                    else:
                        if (local_layer_id < 0
                                or local_layer_id >= len(self.kv_caches)):
                            raise IndexError(
                                "KV receiver patch layer index out of range: "
                                f"layer_id={layer_id}, "
                                f"start_layer_id={start_layer_id}, "
                                f"local_layer_id={local_layer_id}, "
                                f"num_kv_caches={len(self.kv_caches)}, "
                                f"patch_layer_ids={meta.layer_ids}")
                        kv_cache = self.kv_caches[local_layer_id]
                        if layer_block_size is None and kv_cache.dim() >= 3:
                            layer_block_size = int(kv_cache.size(2))

                    if (_autoscaling_kvcached_debug_enabled()
                            and use_kvcached_backend()
                            and not patch_trace_logged
                            and not is_flexi
                            and kv_cache.dim() == 5
                            and apply_ranges):
                        block_size_for_trace = int(layer_block_size)
                        if len(apply_ranges) == 1:
                            trace_slot_mapping = slot_mapping[
                                apply_ranges[0][0]:apply_ranges[0][1]]
                        else:
                            trace_slot_mapping = torch.cat([
                                slot_mapping[start:end]
                                for start, end in apply_ranges
                                if end > start
                            ])
                        patch_blocks = _blocks_from_slots(
                            trace_slot_mapping, block_size_for_trace)
                        if _autoscaling_kvcached_debug_verbose():
                            logger.warning(
                                "[KVCACHED_MIGRATION_APPLY_TRACE] "
                                "rank=%s phase=patch_apply patch_id=%s "
                                "layer_ids=%s slot_count=%s block_size=%s "
                                "blocks=%s slots_min=%s slots_max=%s "
                                "meta_num_tokens=%s scheduler_trace=%s",
                                self.rank, meta.id, meta.layer_ids,
                                int(trace_slot_mapping.numel()),
                                block_size_for_trace,
                                patch_blocks,
                                int(trace_slot_mapping.min().item()),
                                int(trace_slot_mapping.max().item()),
                                meta.num_tokens, scheduler_trace)
                        else:
                            logger.warning(
                                "[KVCACHED_MIGRATION_APPLY_TRACE] "
                                "rank=%s phase=patch_apply patch_id=%s "
                                "slot_count=%s block_size=%s blocks_count=%s "
                                "sample_blocks=%s slots_min=%s slots_max=%s "
                                "meta_num_tokens=%s scheduler_trace=%s",
                                self.rank, meta.id,
                                int(trace_slot_mapping.numel()),
                                block_size_for_trace, len(patch_blocks),
                                patch_blocks[:64],
                                int(trace_slot_mapping.min().item()),
                                int(trace_slot_mapping.max().item()),
                                meta.num_tokens, scheduler_trace)
                        patch_trace_logged = True
                    for range_start, range_end in apply_ranges:
                        if range_end <= range_start:
                            continue
                        layer_slot_mapping = slot_mapping[range_start:range_end]
                        if (use_kvcached_backend()
                                and layer_slot_mapping.numel() > 0
                                and layer_block_size is not None):
                            guard_kvcached_vmm_slots_mapped(
                                f"migration_patch_apply:patch={meta.id}:"
                                f"layer={layer_id}:trace={scheduler_trace}",
                                self,
                                layer_slot_mapping,
                                int(layer_block_size),
                                layer_ids=[int(layer_id)],
                                rank=self.rank,
                                trace_info=scheduler_trace,
                            )
                        if is_flexi:
                            self.kv_helper.flexi_put_kv_to_cache(
                                model_executable=self.model_executable,
                                page_meta=page_meta,
                                keys=key[range_start:range_end],
                                values=value[range_start:range_end],
                                key_cache_ptr=key_cache_ptr,
                                value_cache_ptr=value_cache_ptr,
                                layer=layer_id,
                                slot_mapping=layer_slot_mapping,
                            )
                        else:
                            self.kv_helper.put_kv_to_cache(
                                self.model_executable,
                                key[range_start:range_end],
                                value[range_start:range_end], layer_id,
                                kv_cache, layer_slot_mapping, 0,
                                layer_slot_mapping.size(0))
                    if use_kvcached_backend():
                        self._sync_kvcached_stream(
                            f"KV patch apply layer {layer_id}")
        return 0 if applied_num_tokens is None else applied_num_tokens

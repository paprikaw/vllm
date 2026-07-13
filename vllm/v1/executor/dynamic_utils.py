# SPDX-License-Identifier: Apache-2.0

import os
import sched
import traceback
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from vllm.distributed.parallel_state import (
    get_pp_group,
    set_pp_group_active_ranks,
)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.executor.ray_utils import RayWorkerWrapper
from vllm.v1.utils import human_readable_duration
from vllm.v1.worker.dynamic_gpu_worker import DynamicGPUWorker, DynamicGPUModelRunner
import torch
from vllm.v1.core.sched.dynamic_scheduler import create_from_dynamic_scheduler_output
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
import time
import vllm.envs as envs
from vllm.utils import current_stream

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)
PG_WAIT_TIMEOUT = 1800
_RDT_KV_NCCL_LOCK = None


@dataclass
class PPNCCLIntermediateMetadata:
    tensors: dict[str, tuple[tuple[int, ...], str]]
    send_time: float
    pp_nccl_seq: int = -1
    pp_config_fingerprint: str = ""
    active_ranks: tuple[int, ...] = ()
    request_ids: tuple[str, ...] = ()
    total_num_scheduled_tokens: int = 0


def _dynamic_pp_nccl_transport_enabled() -> bool:
    if _dynamic_pp_rdt_transport_enabled():
        return False
    return os.getenv("VLLM_DYNAMIC_PP_NCCL_TRANSPORT", "0") == "1"


def _dynamic_pp_rdt_transport_enabled() -> bool:
    transport = os.getenv("VLLM_DYNAMIC_PP_RDT_TRANSPORT", "").lower()
    return transport in ("nccl", "1", "true")


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _dynamic_pp_lifetime_trace_enabled() -> bool:
    return (_truthy_env("VLLM_DYNAMIC_PP_LIFETIME_TRACE")
            or _truthy_env("VLLM_AUTOSCALING_KVCACHED_DEBUG"))


def _short_req_ids(req_ids) -> list[str]:
    return [req_id[-8:] if isinstance(req_id, str) else str(req_id)
            for req_id in (req_ids or ())]


def _log_pp_lifetime_trace(phase: str,
                           worker: Optional[DynamicGPUWorker],
                           scheduler_output: Optional[DynamicSchedulerOutput],
                           **extra) -> None:
    if not _dynamic_pp_lifetime_trace_enabled():
        return
    rank = getattr(worker, "rank", None)
    if scheduler_output is None:
        logger.warning(
            "[PP_LIFETIME_TRACE] phase=%s trace_ns=%s rank=%s extra=%s",
            phase, time.time_ns(), rank, extra)
        return
    logger.warning(
        "[PP_LIFETIME_TRACE] phase=%s trace_ns=%s rank=%s step_id=%s "
        "sched_ver=%s pp_seq=%s req_ids=%s finished_req_ids=%s "
        "total_tokens=%s migration=%s sync=%s sender_list=%s "
        "receiver_list=%s extra=%s",
        phase, time.time_ns(), rank,
        getattr(scheduler_output, "scheduler_step_id", -1),
        getattr(scheduler_output, "current_scheduler_output_version", -1),
        getattr(scheduler_output, "pp_nccl_seq", -1),
        _short_req_ids(getattr(scheduler_output, "num_scheduled_tokens", {})
                       .keys()),
        _short_req_ids(getattr(scheduler_output, "finished_req_ids", ()) or ()),
        getattr(scheduler_output, "total_num_scheduled_tokens", 0),
        getattr(scheduler_output, "migration_in_process", False),
        getattr(scheduler_output, "is_sync_after_migration", False),
        getattr(scheduler_output, "sender_list", None),
        getattr(scheduler_output, "receiver_list", None), extra)


def _vllm_ray_rdt_rank_to_device() -> list[int]:
    raw = os.getenv("VLLM_RAY_RDT_RANK_TO_DEVICE", "")
    if not raw:
        return []
    return [int(item) for item in raw.split(",") if item != ""]


def _vllm_ray_rdt_device_for_rank(rank: int) -> int:
    mapping = _vllm_ray_rdt_rank_to_device()
    if mapping and 0 <= rank < len(mapping):
        return mapping[rank]
    return int(os.getenv("VLLM_RAY_RDT_LOCAL_DEVICE_INDEX", "0"))


def _vllm_ray_rdt_sync_after_transfer() -> bool:
    return os.getenv("VLLM_RAY_RDT_SYNC_AFTER_TRANSFER", "1") == "1"


def _vllm_ray_rdt_serialize_with_kv_nccl_lock() -> bool:
    return os.getenv("VLLM_RAY_RDT_SERIALIZE_WITH_KV_NCCL_LOCK",
                     "1") == "1"


def set_vllm_ray_rdt_kv_nccl_lock(nccl_lock) -> None:
    global _RDT_KV_NCCL_LOCK
    _RDT_KV_NCCL_LOCK = nccl_lock


def _vllm_ray_rdt_kv_nccl_lock_context():
    if (_RDT_KV_NCCL_LOCK is not None
            and _vllm_ray_rdt_serialize_with_kv_nccl_lock()):
        return _RDT_KV_NCCL_LOCK
    return nullcontext()


def _cuda_device_index(device: torch.device) -> int:
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


def _ray_collective_p2p_comm_key(my_rank: int, my_gpu_index: int,
                                 peer_rank: int, peer_gpu_index: int) -> str:
    if my_rank < peer_rank:
        lower_key = f"{my_rank}_{my_gpu_index}"
        higher_key = f"{peer_rank}_{peer_gpu_index}"
    elif my_rank > peer_rank:
        lower_key = f"{peer_rank}_{peer_gpu_index}"
        higher_key = f"{my_rank}_{my_gpu_index}"
    else:
        raise RuntimeError(
            "Ray RDT p2p transfer cannot synchronize a self-send stream.")
    return f"{lower_key}:{higher_key}"


def _sync_ray_collective_p2p_stream(group, *, local_gpu_index: int,
                                    peer_rank: int, peer_gpu_index: int,
                                    op_name: str) -> None:
    comm_key = _ray_collective_p2p_comm_key(group.rank, local_gpu_index,
                                            peer_rank, peer_gpu_index)
    streams = (getattr(group, "_dev_streams_map", None) or {}).get(comm_key)
    if not streams:
        raise RuntimeError(
            "Cannot locate Ray RDT NCCL stream for "
            f"{op_name}: comm_key={comm_key}, local_rank={group.rank}, "
            f"local_gpu={local_gpu_index}, peer_rank={peer_rank}, "
            f"peer_gpu={peer_gpu_index}")
    for stream in streams:
        stream.synchronize()


def _vllm_ray_rdt_send(self, communicator_name: str, obj_id: str,
                       dst_rank: int):
    from ray._private.worker import global_worker
    import ray.util.collective as collective
    from ray.experimental.gpu_object_manager.gpu_object_store import (
        COLLECTIVE_BACKEND_TO_TORCH_DEVICE,
    )
    from ray.util.collective.types import Backend

    gpu_object_store = global_worker.gpu_object_manager._gpu_object_store
    assert gpu_object_store.has_object(
        obj_id), f"obj_id={obj_id} not found in GPU object store"
    tensors = gpu_object_store.get_object(obj_id)

    group = collective.get_group_handle(communicator_name)
    backend = group.backend()
    device = COLLECTIVE_BACKEND_TO_TORCH_DEVICE[backend]
    dst_gpu_index = _vllm_ray_rdt_device_for_rank(dst_rank)

    if backend == Backend.NCCL:
        stream_keys = set()
        with _vllm_ray_rdt_kv_nccl_lock_context():
            for tensor in tensors:
                if tensor.device.type != device.type:
                    raise ValueError(
                        f"tensor device {tensor.device} does not match "
                        f"device {device}")
                torch.cuda.set_device(tensor.device)
                collective.send_multigpu(
                    tensor,
                    dst_rank,
                    dst_gpu_index,
                    group_name=communicator_name,
                )
                stream_keys.add((_cuda_device_index(tensor.device), dst_rank,
                                 dst_gpu_index))
            if tensors and _vllm_ray_rdt_sync_after_transfer():
                for local_gpu_index, peer_rank, peer_gpu_index in stream_keys:
                    _sync_ray_collective_p2p_stream(
                        group,
                        local_gpu_index=local_gpu_index,
                        peer_rank=peer_rank,
                        peer_gpu_index=peer_gpu_index,
                        op_name="send")
    else:
        for tensor in tensors:
            if tensor.device.type != device.type:
                raise ValueError(
                    f"tensor device {tensor.device} does not match device "
                    f"{device}")
            collective.send(tensor, dst_rank, group_name=communicator_name)


def _vllm_ray_rdt_recv(
    self,
    communicator_name: str,
    obj_id: str,
    src_rank: int,
    tensor_meta: list[tuple["torch.Size", "torch.dtype"]],
):
    from ray._private.worker import global_worker
    import ray.util.collective as collective
    from ray.experimental.gpu_object_manager.gpu_object_store import (
        COLLECTIVE_BACKEND_TO_TORCH_DEVICE,
    )
    from ray.util.collective.types import Backend

    group = collective.get_group_handle(communicator_name)
    backend = group.backend()

    gpu_object_store = global_worker.gpu_object_manager.gpu_object_store
    tensors = []
    if backend == Backend.NCCL:
        dst_gpu_index = _vllm_ray_rdt_device_for_rank(group.rank)
        src_gpu_index = _vllm_ray_rdt_device_for_rank(src_rank)
        target_device = torch.device(f"cuda:{dst_gpu_index}")
        with _vllm_ray_rdt_kv_nccl_lock_context():
            torch.cuda.set_device(target_device)
            for shape, dtype in tensor_meta:
                tensor = torch.zeros(shape, dtype=dtype, device=target_device)
                collective.recv_multigpu(
                    tensor,
                    src_rank,
                    src_gpu_index,
                    group_name=communicator_name,
                )
                tensors.append(tensor)
            if tensors and _vllm_ray_rdt_sync_after_transfer():
                _sync_ray_collective_p2p_stream(
                    group,
                    local_gpu_index=dst_gpu_index,
                    peer_rank=src_rank,
                    peer_gpu_index=src_gpu_index,
                    op_name="recv")
    else:
        device = COLLECTIVE_BACKEND_TO_TORCH_DEVICE[backend]
        for shape, dtype in tensor_meta:
            tensor = torch.zeros(shape, dtype=dtype, device=device)
            collective.recv(tensor, src_rank, group_name=communicator_name)
            tensors.append(tensor)
    gpu_object_store.add_object(obj_id, tensors)


def install_vllm_ray_rdt_gpu_object_patch() -> None:
    if os.getenv("VLLM_RAY_RDT_MULTIGPU_PATCH", "1") != "1":
        return
    from ray.experimental.gpu_object_manager import gpu_object_store

    gpu_object_store.__ray_send__ = _vllm_ray_rdt_send
    gpu_object_store.__ray_recv__ = _vllm_ray_rdt_recv


def _sync_dynamic_pp_nccl_before_return() -> bool:
    return os.getenv("VLLM_DYNAMIC_PP_NCCL_SYNC_BEFORE_RETURN", "0") == "1"


def _dynamic_pp_explicit_nccl_transfer_enabled() -> bool:
    return (_dynamic_pp_nccl_transport_enabled()
            or _dynamic_pp_rdt_transport_enabled())


def _dynamic_pp_nccl_blocking_edge_sync_enabled() -> bool:
    return _truthy_env("VLLM_DYNAMIC_PP_NCCL_BLOCKING_EDGE_SYNC")


def _dynamic_pp_nccl_debug_enabled() -> bool:
    return os.getenv("VLLM_DYNAMIC_PP_NCCL_DEBUG", "0") == "1"


def _dynamic_pp_nccl_edge_trace_enabled() -> bool:
    return (_dynamic_pp_nccl_debug_enabled()
            or _truthy_env("VLLM_DYNAMIC_PP_NCCL_EDGE_TRACE"))


def _dynamic_pp_nccl_seq_sentinel_enabled() -> bool:
    return _truthy_env("VLLM_DYNAMIC_PP_NCCL_SEQ_SENTINEL")


_PP_NCCL_SEQ_SENTINEL = "__pp_nccl_seq_sentinel__"


def _stream_debug_id() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    try:
        return int(current_stream().cuda_stream)
    except Exception:
        return None


def _log_pp_nccl_edge_trace(phase: str,
                            worker: DynamicGPUWorker,
                            metadata: PPNCCLIntermediateMetadata,
                            pp_group,
                            **extra) -> None:
    if not _dynamic_pp_nccl_edge_trace_enabled():
        return
    logger.warning(
        "[PP_NCCL_EDGE_TRACE] phase=%s trace_ns=%s worker_rank=%s "
        "pp_rank=%s group_ranks=%s active_ranks=%s routing_ranks=%s "
        "first_rank=%s last_rank=%s prev_rank=%s next_rank=%s "
        "seq=%s config=%s request_ids=%s total_tokens=%s stream=%s "
        "extra=%s",
        phase, time.time_ns(), getattr(worker, "rank", None),
        getattr(pp_group, "rank", None), list(getattr(pp_group, "ranks", [])),
        getattr(pp_group, "active_ranks", None),
        list(getattr(pp_group, "routing_ranks", [])),
        getattr(pp_group, "first_rank", None),
        getattr(pp_group, "last_rank", None),
        getattr(pp_group, "prev_rank", None),
        getattr(pp_group, "next_rank", None), metadata.pp_nccl_seq,
        metadata.pp_config_fingerprint, _short_req_ids(metadata.request_ids),
        metadata.total_num_scheduled_tokens, _stream_debug_id(), extra)


def _pending_pp_nccl_send_limit(pp_group, tensors_per_batch: int) -> int:
    env_limit = os.getenv("VLLM_DYNAMIC_PP_NCCL_PENDING_SEND_LIMIT")
    if env_limit is not None:
        return max(1, int(env_limit))
    active_depth = len(pp_group.routing_ranks)
    return max(16, active_depth * active_depth * max(1, tensors_per_batch))


def _dtype_to_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _name_to_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(f"Unsupported PP NCCL tensor dtype: {name}")
    return dtype


def _rank_to_group_index(pp_group, global_rank: int) -> int:
    try:
        return pp_group.ranks.index(global_rank)
    except ValueError as exc:
        raise ValueError(
            f"Rank {global_rank} is not in PP group {pp_group.ranks}") from exc


def _sync_cuda_stream(stream: Optional[torch.cuda.Stream] = None) -> None:
    if not torch.cuda.is_available():
        return
    if stream is None:
        stream = current_stream()
    stream.synchronize()


def _sync_worker_inference_stream(worker: DynamicGPUWorker) -> None:
    _sync_cuda_stream(getattr(worker, "inference_stream", None))


@contextmanager
def _worker_inference_stream(worker: DynamicGPUWorker):
    stream = getattr(worker, "inference_stream", None)
    if stream is None or not torch.cuda.is_available():
        yield
        return

    # torch.cuda.stream() does not update vllm.utils._current_stream, but
    # pynccl reads vllm.utils.current_stream() when choosing its CUDA stream.
    # Keep the cached vLLM stream and the actual torch stream aligned so PP
    # NCCL send/recv is ordered with the model forward kernels.
    import vllm.utils as vllm_utils

    previous_cached_stream = getattr(vllm_utils, "_current_stream", None)
    previous_torch_stream = torch.cuda.current_stream(device=stream.device)
    torch.cuda.set_stream(stream)
    vllm_utils._current_stream = stream
    try:
        yield
    finally:
        torch.cuda.set_stream(previous_torch_stream)
        vllm_utils._current_stream = previous_cached_stream


def _tensor_debug_stats(tensor: torch.Tensor) -> tuple[float, float, float]:
    if tensor.numel() == 0:
        return 0.0, 0.0, 0.0
    tensor_float = tensor.detach().float()
    return (
        float(tensor_float.sum().item()),
        float(torch.linalg.vector_norm(tensor_float).item()),
        float(tensor_float.abs().mean().item()),
    )


def _metadata_from_scheduler_output(
    scheduler_output: Optional[DynamicSchedulerOutput],
    tensors: dict[str, tuple[tuple[int, ...], str]],
    send_time: float,
) -> PPNCCLIntermediateMetadata:
    if scheduler_output is None:
        return PPNCCLIntermediateMetadata(tensors=tensors, send_time=send_time)
    return PPNCCLIntermediateMetadata(
        tensors=tensors,
        send_time=send_time,
        pp_nccl_seq=getattr(scheduler_output, "pp_nccl_seq", -1),
        pp_config_fingerprint=getattr(
            scheduler_output, "pp_nccl_config_fingerprint", ""),
        active_ranks=getattr(scheduler_output, "pp_nccl_active_ranks", ()) or (),
        request_ids=getattr(scheduler_output, "pp_nccl_request_ids", ()) or (),
        total_num_scheduled_tokens=getattr(
            scheduler_output, "total_num_scheduled_tokens", 0),
    )


def _send_intermediate_tensors_nccl(
    worker: DynamicGPUWorker,
    tensors: IntermediateTensors,
    scheduler_output: Optional[DynamicSchedulerOutput] = None,
) -> PPNCCLIntermediateMetadata:
    pp_group = get_pp_group()
    dst = _rank_to_group_index(pp_group, pp_group.next_rank)
    metadata = _metadata_from_scheduler_output(
        scheduler_output,
        {
            name: (tuple(tensor.shape), _dtype_to_name(tensor.dtype))
            for name, tensor in tensors.tensors.items()
        },
        time.time(),
    )
    send_items = list(tensors.tensors.items())
    if _dynamic_pp_nccl_seq_sentinel_enabled():
        metadata.tensors[_PP_NCCL_SEQ_SENTINEL] = (
            (3, ), _dtype_to_name(torch.float32))
        sentinel = torch.tensor(
            [float(metadata.pp_nccl_seq),
             float(worker.rank),
             float(pp_group.next_rank)],
            dtype=torch.float32,
            device=worker.device)
        send_items.append((_PP_NCCL_SEQ_SENTINEL, sentinel))
    debug_enabled = _dynamic_pp_nccl_debug_enabled()
    _log_pp_nccl_edge_trace("send_enter",
                            worker,
                            metadata,
                            pp_group,
                            dst_index=dst,
                            tensor_names=list(metadata.tensors))
    if debug_enabled:
        last_by_dst = getattr(worker, "_pp_nccl_last_send_seq_by_dst", None)
        if last_by_dst is None:
            last_by_dst = {}
            worker._pp_nccl_last_send_seq_by_dst = last_by_dst
        last_seq = last_by_dst.get(pp_group.next_rank)
        if last_seq is not None and metadata.pp_nccl_seq <= last_seq:
            raise AssertionError(
                "PP NCCL send seq is not monotonically increasing: "
                f"rank={worker.rank}, dst={pp_group.next_rank}, "
                f"seq={metadata.pp_nccl_seq}, last_seq={last_seq}")
        last_by_dst[pp_group.next_rank] = metadata.pp_nccl_seq
    pending = getattr(worker, "_pending_pp_nccl_sends", None)
    if pending is None:
        pending = []
        worker._pending_pp_nccl_sends = pending
    nccl_lock = getattr(worker, "_nccl_lock", None)
    lock_context = nccl_lock if nccl_lock is not None else nullcontext()
    with lock_context:
        for name, tensor in send_items:
            send_tensor = tensor.contiguous()
            if debug_enabled:
                tensor_sum, tensor_norm, tensor_mean_abs = (
                    _tensor_debug_stats(send_tensor))
                logger.info(
                    "[pp_nccl_debug] send seq=%s edge=%s->%s "
                    "config=%s active=%s reqs=%s tensor=%s shape=%s "
                    "dtype=%s sum=%.6e norm=%.6e mean_abs=%.6e",
                    metadata.pp_nccl_seq, worker.rank, pp_group.next_rank,
                    metadata.pp_config_fingerprint, metadata.active_ranks,
                    metadata.request_ids, name, tuple(send_tensor.shape),
                    send_tensor.dtype, tensor_sum, tensor_norm,
                    tensor_mean_abs)
            pp_group.send(send_tensor, dst=dst)
            pending.append(send_tensor)
        _log_pp_nccl_edge_trace("send_ops_enqueued",
                                worker,
                                metadata,
                                pp_group,
                                dst_index=dst,
                                pending_sends=len(pending))
        if _dynamic_pp_nccl_blocking_edge_sync_enabled():
            _sync_worker_inference_stream(worker)
            send_done_phase = "send_after_sync"
        else:
            send_done_phase = "send_after_enqueue"
        _log_pp_nccl_edge_trace(send_done_phase,
                                worker,
                                metadata,
                                pp_group,
                                dst_index=dst,
                                pending_sends=len(pending))
    keep_limit = _pending_pp_nccl_send_limit(pp_group, len(metadata.tensors))
    if len(pending) > keep_limit:
        del pending[:-keep_limit]
    logger.debug("[forward]: rank %s enqueued PP NCCL send to rank %s "
                 "for tensors %s, seq=%s", worker.rank, pp_group.next_rank,
                 list(metadata.tensors.keys()), metadata.pp_nccl_seq)
    return metadata


def _recv_intermediate_tensors_nccl(
    metadata: PPNCCLIntermediateMetadata,
    worker: DynamicGPUWorker,
) -> IntermediateTensors:
    pp_group = get_pp_group()
    src = _rank_to_group_index(pp_group, pp_group.prev_rank)
    tensors = {}
    sentinel_tensor: Optional[torch.Tensor] = None
    debug_enabled = _dynamic_pp_nccl_debug_enabled()
    _log_pp_nccl_edge_trace("recv_enter",
                            worker,
                            metadata,
                            pp_group,
                            src_index=src,
                            tensor_names=list(metadata.tensors))
    if debug_enabled:
        last_by_src = getattr(worker, "_pp_nccl_last_recv_seq_by_src", None)
        if last_by_src is None:
            last_by_src = {}
            worker._pp_nccl_last_recv_seq_by_src = last_by_src
        last_seq = last_by_src.get(pp_group.prev_rank)
        if last_seq is not None and metadata.pp_nccl_seq <= last_seq:
            raise AssertionError(
                "PP NCCL recv seq is not monotonically increasing: "
                f"rank={worker.rank}, src={pp_group.prev_rank}, "
                f"seq={metadata.pp_nccl_seq}, last_seq={last_seq}")
        last_by_src[pp_group.prev_rank] = metadata.pp_nccl_seq
    nccl_lock = getattr(worker, "_nccl_lock", None)
    lock_context = nccl_lock if nccl_lock is not None else nullcontext()
    pending_recvs = getattr(worker, "_pending_pp_nccl_recvs", None)
    if pending_recvs is None:
        pending_recvs = []
        worker._pending_pp_nccl_recvs = pending_recvs
    with lock_context:
        for name, (shape, dtype_name) in metadata.tensors.items():
            tensor = pp_group.recv(torch.Size(shape),
                                   _name_to_dtype(dtype_name),
                                   src=src)
            if name == _PP_NCCL_SEQ_SENTINEL:
                sentinel_tensor = tensor
            else:
                tensors[name] = tensor
            pending_recvs.append(tensor)
        _log_pp_nccl_edge_trace("recv_ops_enqueued",
                                worker,
                                metadata,
                                pp_group,
                                src_index=src,
                                tensor_names=list(tensors))
        if _dynamic_pp_nccl_blocking_edge_sync_enabled():
            _sync_worker_inference_stream(worker)
            _log_pp_nccl_edge_trace("recv_after_sync",
                                    worker,
                                    metadata,
                                    pp_group,
                                    src_index=src,
                                    tensor_names=list(tensors))
            if sentinel_tensor is not None:
                seq_value, src_value, dst_value = [
                    int(round(value))
                    for value in sentinel_tensor.detach().cpu().tolist()
                ]
                expected = (metadata.pp_nccl_seq, pp_group.prev_rank,
                            worker.rank)
                observed = (seq_value, src_value, dst_value)
                if observed != expected:
                    raise AssertionError(
                        "PP NCCL seq sentinel mismatch: "
                        f"expected={expected}, observed={observed}, "
                        f"metadata_active={metadata.active_ranks}, "
                        f"metadata_reqs={metadata.request_ids}")
            if debug_enabled:
                for name, tensor in tensors.items():
                    tensor_sum, tensor_norm, tensor_mean_abs = (
                        _tensor_debug_stats(tensor))
                    logger.info(
                        "[pp_nccl_debug] recv_post_sync seq=%s edge=%s->%s "
                        "config=%s active=%s reqs=%s tensor=%s shape=%s "
                        "dtype=%s sum=%.6e norm=%.6e mean_abs=%.6e",
                        metadata.pp_nccl_seq, pp_group.prev_rank, worker.rank,
                        metadata.pp_config_fingerprint, metadata.active_ranks,
                        metadata.request_ids, name, tuple(tensor.shape),
                        tensor.dtype, tensor_sum, tensor_norm,
                        tensor_mean_abs)
        else:
            _log_pp_nccl_edge_trace("recv_after_enqueue",
                                    worker,
                                    metadata,
                                    pp_group,
                                    src_index=src,
                                    tensor_names=list(tensors))
    keep_limit = _pending_pp_nccl_send_limit(pp_group, len(metadata.tensors))
    if len(pending_recvs) > keep_limit:
        del pending_recvs[:-keep_limit]
    logger.debug("[forward]: rank %s enqueued PP NCCL recv from rank %s "
                 "for tensors %s, seq=%s", pp_group.rank, pp_group.prev_rank,
                 list(tensors.keys()), metadata.pp_nccl_seq)
    return IntermediateTensors(tensors)


@contextmanager
def _batch_pp_routing(pp_layer_config):
    pp_group = get_pp_group()
    previous_active_ranks = (list(pp_group.active_ranks)
                             if pp_group.active_ranks is not None else None)
    batch_active_ranks = [
        rank for rank, (start, end) in enumerate(pp_layer_config)
        if end >= start
    ]
    if _dynamic_pp_nccl_edge_trace_enabled():
        logger.warning(
            "[PP_ROUTING_TRACE] phase=enter trace_ns=%s pp_rank=%s "
            "previous_active=%s batch_active=%s routing_before=%s "
            "pp_layer_config=%s",
            time.time_ns(), getattr(pp_group, "rank", None),
            previous_active_ranks, batch_active_ranks,
            list(getattr(pp_group, "routing_ranks", [])), pp_layer_config)
    set_pp_group_active_ranks(batch_active_ranks)
    try:
        yield
    finally:
        set_pp_group_active_ranks(previous_active_ranks)
        if _dynamic_pp_nccl_edge_trace_enabled():
            logger.warning(
                "[PP_ROUTING_TRACE] phase=exit trace_ns=%s pp_rank=%s "
                "restored_active=%s routing_after=%s",
                time.time_ns(), getattr(pp_group, "rank", None),
                previous_active_ranks,
                list(getattr(pp_group, "routing_ranks", [])))

try:
    import ray
    from ray.util import placement_group_table
    from ray.util.placement_group import PlacementGroup
    try:
        from ray._private.state import available_resources_per_node
    except ImportError:
        # Ray 2.9.x doesn't expose `available_resources_per_node`
        from ray._private.state import state as _state
        available_resources_per_node = _state._available_resources_per_node

    class DynamicRayWorkerWrapper(RayWorkerWrapper):
        """Ray wrapper for vllm.worker.Worker, allowing Worker to be
        lazily initialized after Ray sets CUDA_VISIBLE_DEVICES."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        def update_environment_variables(self, envs_list) -> None:
            super().update_environment_variables(envs_list)
            from vllm.kvcached_integration import (
                maybe_apply_kvcached_vllm_patches,
            )
            maybe_apply_kvcached_vllm_patches("Ray worker env update")

        def reset_ray_compiled_dag_nccl_lock(self) -> None:
            try:
                from ray.experimental.channel.nccl_group import set_global_nccl_lock
            except ImportError:
                logger.debug(
                    "Ray does not expose set_global_nccl_lock; skipping "
                    "compiled DAG NCCL lock reset.")
                return
            set_global_nccl_lock(None)

        def init_device(self):
            super().init_device()
            if _dynamic_pp_rdt_transport_enabled():
                assert self.worker is not None
                assert self.worker.device is not None
                os.environ["VLLM_RAY_RDT_LOCAL_DEVICE_INDEX"] = str(
                    self.worker.device.index)
                install_vllm_ray_rdt_gpu_object_patch()
                logger.info(
                    "Installed Ray RDT GPU object multigpu patch on %s",
                    self.worker.device)

        def _set_ray_rdt_torch_device(self) -> torch.device:
            self.setup_device_if_necessary()
            assert self.worker is not None
            assert self.worker.device is not None
            device = torch.device(self.worker.device)
            if _dynamic_pp_rdt_transport_enabled():
                from ray.experimental.channel import ChannelContext
                ChannelContext.get_current().set_torch_device(device)
            return device

        def prewarm_ray_rdt_send(self, src_rank: int, dst_rank: int,
                                 numel: int = 1) -> torch.Tensor:
            device = self._set_ray_rdt_torch_device()
            with torch.cuda.device(device):
                tensor = torch.ones(numel, dtype=torch.float32, device=device)
            logger.info(
                "[pp_rdt_prewarm] rank %s produced warmup tensor for edge "
                "%s->%s on %s", self.rpc_rank, src_rank, dst_rank, device)
            return tensor

        def prewarm_ray_rdt_recv(self, tensor: torch.Tensor, src_rank: int,
                                 dst_rank: int) -> float:
            device = self._set_ray_rdt_torch_device()
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Ray RDT prewarm expected a torch.Tensor, got "
                                f"{type(tensor)!r}")
            if tensor.is_cuda:
                torch.cuda.set_device(tensor.device)
            checksum = float(tensor.sum().item())
            logger.info(
                "[pp_rdt_prewarm] rank %s received warmup tensor for edge "
                "%s->%s on %s, tensor_device=%s, checksum=%.1f",
                self.rpc_rank, src_rank, dst_rank, device, tensor.device,
                checksum)
            return checksum

        def activate_pp_ranks_for_autoscaling_chain(
            self,
            active_ranks: list[int],
            generation: int,
            upstream_marker: Optional[dict] = None,
        ) -> dict:
            self.setup_device_if_necessary()
            assert isinstance(self.worker, DynamicGPUWorker)
            start = time.time()
            self.worker.set_active_pp_ranks(active_ranks)
            marker = {
                "rank": self.rpc_rank,
                "generation": generation,
                "active_ranks": list(active_ranks),
                "upstream": upstream_marker,
            }
            logger.info(
                "[autoscaling active ranks] rank %s activated PP ranks %s "
                "for generation %s via actor chain in %s",
                self.rpc_rank, active_ranks, generation,
                human_readable_duration(time.time() - start))
            return marker

        def execute_model_ray(
            self,
            scheduler_output: Union["DynamicSchedulerOutput",
                                    Tuple["DynamicSchedulerOutput",
                                          "IntermediateTensors", float],
                                    "ModelRunnerOutput"],
            upstream_barrier: Optional[object] = None,
        ) -> Union["ModelRunnerOutput", Tuple["DynamicSchedulerOutput",
                                              "IntermediateTensors", float]]:
            # This method is used by Ray Compiled Graph to execute the model,
            # and it needs a special logic of self.setup_device_if_necessary()
            del upstream_barrier
            time_recv = time.time()  # 记录接收时间
            assert isinstance(self.worker, DynamicGPUWorker)
            assert self.worker.inference_stream is not None, "high_priority_stream is not initialized"
            
            # Use high priority stream for model execution with proper synchronization
            # The high priority stream ensures compute kernels are scheduled with higher priority,
            # but we must synchronize before returning results to prevent data races.
            
            try:
                self.setup_device_if_necessary()
                if _dynamic_pp_rdt_transport_enabled():
                    from ray.experimental.channel import ChannelContext
                    ChannelContext.get_current().set_torch_device(
                        torch.device(self.worker.device))
                # Autoscaling KV migration uses a separate migration stream.
                # Do not route normal inference through the coarse foreground
                # gate here; correctness is enforced by stream syncs, the NCCL
                # lock, and the narrower forward_lock scopes below.
                with _worker_inference_stream(self.worker):
                    assert self.worker is not None, "Worker is not initialized"
                    assert isinstance(self.worker, DynamicGPUWorker), "Worker is not a DynamicGPUWorker"
                    assert isinstance(self.worker.model_runner, DynamicGPUModelRunner), "Model runner is not a DynamicGPUModelRunner"
                    _log_pp_lifetime_trace(
                        "task_enter_raw",
                        self.worker,
                        None,
                        input_type=type(scheduler_output).__name__)

                    upstream_send_time = None
                    if isinstance(scheduler_output, tuple):
                        if len(scheduler_output) == 3:
                            scheduler_output, intermediate_tensors, upstream_send_time = scheduler_output
                        else:
                            scheduler_output, intermediate_tensors = scheduler_output
                        if isinstance(intermediate_tensors,
                                      PPNCCLIntermediateMetadata):
                            upstream_send_time = (
                                intermediate_tensors.send_time or None)
                            assert isinstance(scheduler_output,
                                              DynamicSchedulerOutput)
                            _log_pp_lifetime_trace(
                                "before_pp_recv",
                                self.worker,
                                scheduler_output,
                                metadata_seq=intermediate_tensors.pp_nccl_seq,
                                metadata_active_ranks=intermediate_tensors
                                .active_ranks)
                            with _batch_pp_routing(
                                    scheduler_output.pp_layer_config):
                                intermediate_tensors = (
                                    _recv_intermediate_tensors_nccl(
                                        intermediate_tensors, self.worker))
                            _log_pp_lifetime_trace("after_pp_recv",
                                                   self.worker,
                                                   scheduler_output)
                    else:
                        scheduler_output, intermediate_tensors = (
                            scheduler_output, None)

                    if not isinstance(scheduler_output,
                                      DynamicSchedulerOutput):
                        logger.warning(
                            "[forward]: rank %s received %s instead of "
                            "DynamicSchedulerOutput; returning it unchanged",
                            self.rpc_rank, type(scheduler_output).__name__)
                        return scheduler_output

                    _log_pp_lifetime_trace("task_enter", self.worker,
                                           scheduler_output)
                    time_before_lock = time.time()
                    inference_stream_synced = False
                    self.worker.prepare_autoscaling_request_states_from_sync_batch(
                        scheduler_output)
                    self.worker.async_migration_before_execute_callback(
                        scheduler_output)
                    _log_pp_lifetime_trace(
                        "after_before_execute_callback",
                        self.worker,
                        scheduler_output)
                    time_after_before_execute_callback = time.time()
                    with self.worker.model_runner.forward_lock:
                        logger.debug(
                            "[forward]: rank %s acquired forward_lock for "
                            "executing model, took %s",
                            self.rpc_rank,
                            human_readable_duration(time.time() -
                                                    time_before_lock))
                        assert isinstance(
                            scheduler_output, DynamicSchedulerOutput), (
                                "Scheduler output is not a "
                                f"DynamicSchedulerOutput:{type(scheduler_output)}")

                        with _batch_pp_routing(scheduler_output.pp_layer_config):
                            try:
                                _log_pp_lifetime_trace("before_forward",
                                                       self.worker,
                                                       scheduler_output)
                                logger.debug(
                                    "forwarding from layer%s to layer%s",
                                    scheduler_output.pp_layer_config[
                                        self.worker.rank][0],
                                    scheduler_output.pp_layer_config[
                                        self.worker.rank][1])
                                output = self.worker.model_runner.execute_model(
                                    create_from_dynamic_scheduler_output(
                                        scheduler_output),
                                    scheduler_output.pp_layer_config[
                                        self.rpc_rank],
                                    intermediate_tensors)
                                _log_pp_lifetime_trace(
                                    "after_forward",
                                    self.worker,
                                    scheduler_output,
                                    output_type=type(output).__name__)
                            except Exception as e:
                                print(traceback.format_exc())
                                print(f"scheduler_output: {scheduler_output}")
                                print("error is raised within the compiled ray "
                                      f"DAG graph, error: {e}")
                                time.sleep(1)
                                raise e
                        time_after_execute = time.time()

                        assert (
                            len(self.worker.model_runner.input_batch.block_table
                                .block_tables) == 1)
                        sender_list = scheduler_output.sender_list
                        sender_active = (
                            scheduler_output.migration_in_process
                            and sender_list is not None
                            and self.worker.rank in sender_list)
                        # Old-topology batches can finish after sender state is
                        # installed. Sync under forward_lock before migration
                        # threads can gather the just-written KV cache pages.
                        sender_active = sender_active or bool(
                            self.worker.rank_to_layers_ids)
                        if sender_active:
                            sync_start = time.time()
                            _sync_worker_inference_stream(self.worker)
                            inference_stream_synced = True
                            _log_pp_lifetime_trace(
                                "after_sender_pre_migration_sync",
                                self.worker,
                                scheduler_output)
                            logger.debug(
                                "[forward]: rank %s synchronized CUDA stream "
                                "under forward_lock before KV sender can read "
                                "cache, took %s",
                                self.rpc_rank,
                                human_readable_duration(time.time() -
                                                        sync_start))

                    self.worker.async_migration_after_execute_callback(
                        scheduler_output)
                    _log_pp_lifetime_trace("after_execute_callback",
                                           self.worker, scheduler_output)
                    time_after_execute_callback = time.time()

                    sender_list = scheduler_output.sender_list
                    if (scheduler_output.migration_in_process
                            and sender_list is not None
                            and self.worker.rank in sender_list
                            and not inference_stream_synced):
                        sync_start = time.time()
                        _sync_worker_inference_stream(self.worker)
                        inference_stream_synced = True
                        _log_pp_lifetime_trace(
                            "after_sender_post_execute_sync",
                            self.worker,
                            scheduler_output)
                        logger.debug(
                            "[forward]: rank %s synchronized CUDA stream "
                            "before KV sender can read cache, took %s",
                            self.rpc_rank,
                            human_readable_duration(time.time() -
                                                    sync_start))

                    if isinstance(output, IntermediateTensors):
                        send_time = time.time()
                        if (_dynamic_pp_explicit_nccl_transfer_enabled()
                                and not envs.VLLM_USE_RAY_COMPILED_DAG):
                            with _batch_pp_routing(
                                    scheduler_output.pp_layer_config):
                                _log_pp_lifetime_trace("before_pp_send",
                                                       self.worker,
                                                       scheduler_output)
                                metadata = _send_intermediate_tensors_nccl(
                                    self.worker, output, scheduler_output)
                                _log_pp_lifetime_trace(
                                    "after_pp_send",
                                    self.worker,
                                    scheduler_output,
                                    metadata_seq=metadata.pp_nccl_seq)
                            output = (scheduler_output, metadata)
                        else:
                            output = (scheduler_output, output, send_time)

                    self.worker.async_migration_after_pp_transfer_callback(
                        scheduler_output)
                    _log_pp_lifetime_trace("after_pp_transfer_callback",
                                           self.worker, scheduler_output)

                    if upstream_send_time is not None:
                        comm_time = (time_recv - upstream_send_time) * 1000
                        if intermediate_tensors is not None:
                            hidden_states = intermediate_tensors.tensors[
                                "hidden_states"]
                            residual = intermediate_tensors.tensors["residual"]
                            hidden_size_mb = (
                                hidden_states.numel()
                                * hidden_states.element_size()
                                / 1024 / 1024)
                            residual_size_mb = (
                                residual.numel() * residual.element_size()
                                / 1024 / 1024)
                            total_size_mb = hidden_size_mb + residual_size_mb
                            if comm_time > 0.001:
                                bandwidth_str = (
                                    f"{total_size_mb / (comm_time / 1000):.2f} MB/s")
                            else:
                                bandwidth_str = "N/A (comm_time too small)"
                            logger.debug(
                                "[forward]: rank %s Communication time from "
                                "upstream: %.2f ms, hidden_states dtype: %s, "
                                "shape: %s, size: %.2f MB, residual dtype: %s, "
                                "shape: %s, size: %.2f MB, total data volume: "
                                "%.2f MB, bandwidth: %s",
                                self.rpc_rank, comm_time, hidden_states.dtype,
                                hidden_states.shape, hidden_size_mb,
                                residual.dtype, residual.shape,
                                residual_size_mb, total_size_mb,
                                bandwidth_str)
                        else:
                            logger.debug(
                                "[forward]: rank %s Communication time from "
                                "upstream: %.2f ms, no received data",
                                self.rpc_rank, comm_time)
                    before_sync = time.time()
                    will_sync_before_return = (
                        not _dynamic_pp_explicit_nccl_transfer_enabled()
                        or envs.VLLM_USE_RAY_COMPILED_DAG
                        or _sync_dynamic_pp_nccl_before_return()
                        or not isinstance(output, tuple)
                        or not isinstance(output[1],
                                          PPNCCLIntermediateMetadata))
                    _log_pp_lifetime_trace(
                        "before_return_sync_gate",
                        self.worker,
                        scheduler_output,
                        will_sync=will_sync_before_return,
                        inference_stream_synced=inference_stream_synced,
                        output_type=type(output).__name__)
                    if (not _dynamic_pp_explicit_nccl_transfer_enabled()
                            or envs.VLLM_USE_RAY_COMPILED_DAG
                            or _sync_dynamic_pp_nccl_before_return()
                            or not isinstance(output, tuple)
                            or not isinstance(output[1],
                                              PPNCCLIntermediateMetadata)):
                        if not inference_stream_synced:
                            _sync_worker_inference_stream(self.worker)
                            _log_pp_lifetime_trace("after_return_sync",
                                                   self.worker,
                                                   scheduler_output)
                    logger.debug(
                        "[forward]: forwarding from layer%s to layer%s, "
                        "before_execute_callback=%.2fs, execute=%.2fs, "
                        "after_execute_callback=%.2fs, "
                        "inference_stream_synchronize=%.2fs, total=%.2fs",
                        scheduler_output.pp_layer_config[self.worker.rank][0],
                        scheduler_output.pp_layer_config[self.worker.rank][1],
                        time_after_before_execute_callback - time_recv,
                        time_after_execute -
                        time_after_before_execute_callback,
                        time_after_execute_callback - time_after_execute,
                        time.time() - before_sync,
                        time.time() - time_recv)
                    _log_pp_lifetime_trace("before_task_return", self.worker,
                                           scheduler_output)
                    return output
            except Exception as e:
                _log_pp_lifetime_trace("task_exception",
                                       getattr(self, "worker", None),
                                       None,
                                       error=repr(e))
                print(traceback.format_exc())
                print(f"error is raised within the compiled ray DAG graph, error: {e}")
                time.sleep(1)
                raise e

except ImportError as e:
    ray = None  # type: ignore
    ray_import_err = e
    RayWorkerWrapper = None  # type: ignore

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

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)
PG_WAIT_TIMEOUT = 1800


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
    return os.getenv("VLLM_DYNAMIC_PP_NCCL_TRANSPORT", "0") == "1"


def _sync_dynamic_pp_nccl_before_return() -> bool:
    return os.getenv("VLLM_DYNAMIC_PP_NCCL_SYNC_BEFORE_RETURN", "0") == "1"


def _dynamic_pp_nccl_debug_enabled() -> bool:
    return os.getenv("VLLM_DYNAMIC_PP_NCCL_DEBUG", "0") == "1"


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


def _sync_current_cuda_stream() -> None:
    if torch.cuda.is_available():
        # NCCL may enqueue work on an internal/non-current stream. Keep the
        # shared PP/KV NCCL lock held until all local CUDA work is complete.
        torch.cuda.synchronize()


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
    debug_enabled = _dynamic_pp_nccl_debug_enabled()
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
        for name, tensor in tensors.tensors.items():
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
        _sync_current_cuda_stream()
    keep_limit = _pending_pp_nccl_send_limit(pp_group, len(metadata.tensors))
    if len(pending) > keep_limit:
        del pending[:-keep_limit]
    logger.info("[forward]: rank %s enqueued PP NCCL send to rank %s "
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
    debug_enabled = _dynamic_pp_nccl_debug_enabled()
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
    with lock_context:
        for name, (shape, dtype_name) in metadata.tensors.items():
            tensors[name] = pp_group.recv(torch.Size(shape),
                                          _name_to_dtype(dtype_name),
                                          src=src)
        _sync_current_cuda_stream()
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
                    tensor.dtype, tensor_sum, tensor_norm, tensor_mean_abs)
    logger.info("[forward]: rank %s enqueued PP NCCL recv from rank %s "
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
    set_pp_group_active_ranks(batch_active_ranks)
    try:
        yield
    finally:
        set_pp_group_active_ranks(previous_active_ranks)

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

        def reset_ray_compiled_dag_nccl_lock(self) -> None:
            from ray.experimental.channel.nccl_group import set_global_nccl_lock
            set_global_nccl_lock(None)

        def execute_model_ray(
            self,
            scheduler_output: Union["DynamicSchedulerOutput",
                                    Tuple["DynamicSchedulerOutput",
                                          "IntermediateTensors", float]],
        ) -> Union["ModelRunnerOutput", Tuple["DynamicSchedulerOutput",
                                              "IntermediateTensors", float]]:
            # This method is used by Ray Compiled Graph to execute the model,
            # and it needs a special logic of self.setup_device_if_necessary()
            time_recv = time.time()  # 记录接收时间
            assert isinstance(self.worker, DynamicGPUWorker)
            assert self.worker.inference_stream is not None, "high_priority_stream is not initialized"
            
            # Use high priority stream for model execution with proper synchronization
            # The high priority stream ensures compute kernels are scheduled with higher priority,
            # but we must synchronize before returning results to prevent data races.
            
            try:
                self.setup_device_if_necessary()
                with self.worker.inference_stream:
                    assert self.worker is not None, "Worker is not initialized"
                    assert isinstance(self.worker, DynamicGPUWorker), "Worker is not a DynamicGPUWorker"
                    assert isinstance(self.worker.model_runner, DynamicGPUModelRunner), "Model runner is not a DynamicGPUModelRunner"

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
                            with _batch_pp_routing(
                                    scheduler_output.pp_layer_config):
                                intermediate_tensors = (
                                    _recv_intermediate_tensors_nccl(
                                        intermediate_tensors, self.worker))
                    else:
                        scheduler_output, intermediate_tensors = (
                            scheduler_output, None)

                    time_before_lock = time.time()
                    inference_stream_synced = False
                    self.worker.async_migration_before_execute_callback(
                        scheduler_output)
                    time_after_before_execute_callback = time.time()
                    with self.worker.model_runner.forward_lock:
                        logger.info(
                            f"[forward]: rank {self.rpc_rank} Acquired "
                            "forward_lock for executing model, took "
                            f"{human_readable_duration(time.time() - time_before_lock)}")
                        assert isinstance(
                            scheduler_output, DynamicSchedulerOutput), (
                                "Scheduler output is not a "
                                f"DynamicSchedulerOutput:{type(scheduler_output)}")

                        with _batch_pp_routing(scheduler_output.pp_layer_config):
                            try:
                                logger.info(
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

                    self.worker.async_migration_after_execute_callback(
                        scheduler_output)
                    time_after_execute_callback = time.time()

                    sender_list = scheduler_output.sender_list
                    if (scheduler_output.migration_in_process
                            and sender_list is not None
                            and self.worker.rank in sender_list):
                        sync_start = time.time()
                        torch.cuda.synchronize(device=self.worker.device)
                        inference_stream_synced = True
                        logger.info(
                            "[forward]: rank %s synchronized CUDA device "
                            "under forward_lock before KV sender "
                            "can read cache, took %s",
                            self.rpc_rank,
                            human_readable_duration(time.time() -
                                                    sync_start))

                    if isinstance(output, IntermediateTensors):
                        send_time = time.time()
                        if (_dynamic_pp_nccl_transport_enabled()
                                and not envs.VLLM_USE_RAY_COMPILED_DAG):
                            with _batch_pp_routing(
                                    scheduler_output.pp_layer_config):
                                metadata = _send_intermediate_tensors_nccl(
                                    self.worker, output, scheduler_output)
                            output = (scheduler_output, metadata)
                        else:
                            output = (scheduler_output, output, send_time)

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
                            logger.info(
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
                            logger.info(
                                "[forward]: rank %s Communication time from "
                                "upstream: %.2f ms, no received data",
                                self.rpc_rank, comm_time)
                    before_sync = time.time()
                    if (not _dynamic_pp_nccl_transport_enabled()
                            or envs.VLLM_USE_RAY_COMPILED_DAG
                            or _sync_dynamic_pp_nccl_before_return()
                            or not isinstance(output, tuple)
                            or not isinstance(output[1],
                                              PPNCCLIntermediateMetadata)):
                        if not inference_stream_synced:
                            torch.cuda.synchronize(device=self.worker.device)
                    logger.info(f"""
                    [forward]: forwarding from layer{scheduler_output.pp_layer_config[self.worker.rank][0]} to layer{scheduler_output.pp_layer_config[self.worker.rank][1]},
                    [forward]: before execute callback time: {time_after_before_execute_callback - time_recv:.2f} seconds,
                    [forward]: execute time: {time_after_execute - time_after_before_execute_callback:.2f} seconds,
                    [forward]: after execute callback time: {time_after_execute_callback - time_after_execute:.2f} seconds,
                    [forward]: inference stream synchronize time: {time.time() - before_sync:.2f} seconds,
                    [forward]: total time: {time.time() - time_recv:.2f} seconds
                    """)
                    return output
            except Exception as e:
                print(traceback.format_exc())
                print(f"error is raised within the compiled ray DAG graph, error: {e}")
                time.sleep(1)
                raise e

except ImportError as e:
    ray = None  # type: ignore
    ray_import_err = e
    RayWorkerWrapper = None  # type: ignore

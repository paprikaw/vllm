# SPDX-License-Identifier: Apache-2.0

import os
import sched
import traceback
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Dict, Tuple, Union

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


def _dynamic_pp_nccl_transport_enabled() -> bool:
    return os.getenv("VLLM_DYNAMIC_PP_NCCL_TRANSPORT", "0") == "1"


def _sync_dynamic_pp_nccl_before_return() -> bool:
    return os.getenv("VLLM_DYNAMIC_PP_NCCL_SYNC_BEFORE_RETURN", "0") == "1"


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


def _send_intermediate_tensors_nccl(
    worker: DynamicGPUWorker,
    tensors: IntermediateTensors,
) -> PPNCCLIntermediateMetadata:
    pp_group = get_pp_group()
    dst = _rank_to_group_index(pp_group, pp_group.next_rank)
    metadata = PPNCCLIntermediateMetadata(
        tensors={
            name: (tuple(tensor.shape), _dtype_to_name(tensor.dtype))
            for name, tensor in tensors.tensors.items()
        },
        send_time=time.time(),
    )
    pending = getattr(worker, "_pending_pp_nccl_sends", None)
    if pending is None:
        pending = []
        worker._pending_pp_nccl_sends = pending
    for tensor in tensors.tensors.values():
        send_tensor = tensor.contiguous()
        pp_group.send(send_tensor, dst=dst)
        pending.append(send_tensor)
    keep_limit = _pending_pp_nccl_send_limit(pp_group, len(metadata.tensors))
    if len(pending) > keep_limit:
        del pending[:-keep_limit]
    logger.info("[forward]: rank %s enqueued PP NCCL send to rank %s "
                "for tensors %s", worker.rank, pp_group.next_rank,
                list(metadata.tensors.keys()))
    return metadata


def _recv_intermediate_tensors_nccl(
    metadata: PPNCCLIntermediateMetadata,
) -> IntermediateTensors:
    pp_group = get_pp_group()
    src = _rank_to_group_index(pp_group, pp_group.prev_rank)
    tensors = {}
    for name, (shape, dtype_name) in metadata.tensors.items():
        tensors[name] = pp_group.recv(torch.Size(shape),
                                      _name_to_dtype(dtype_name),
                                      src=src)
    logger.info("[forward]: rank %s enqueued PP NCCL recv from rank %s "
                "for tensors %s", pp_group.rank, pp_group.prev_rank,
                list(tensors.keys()))
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

                    # 提取上游发送时间（如果有）
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
                            intermediate_tensors = (
                                _recv_intermediate_tensors_nccl(
                                    intermediate_tensors))
                    else:
                        scheduler_output, intermediate_tensors = scheduler_output, None

                    # DIAG: Log receiver-side consistency check with NaN detection
                    # if intermediate_tensors is not None:
                        # _diag_total = scheduler_output.total_num_scheduled_tokens
                        # _diag_shapes = {k: v.shape for k, v in intermediate_tensors.tensors.items()}
                        # # Check for NaN in received tensors
                        # _recv_nan_info = {}
                        # for k, v in intermediate_tensors.tensors.items():
                        #     _has_nan = torch.isnan(v).any().item()
                        #     _nan_count = torch.isnan(v).sum().item() if _has_nan else 0
                        #     _recv_nan_info[k] = {"has_nan": _has_nan, "nan_count": _nan_count}
                        #     if _has_nan:
                        #         logger.error(f"[DIAG_RECV_NAN] rank={self.rpc_rank} "
                        #                    f"tensor '{k}' has {_nan_count} NaN values! "
                        #                    f"shape={v.shape} dtype={v.dtype}")
                        # logger.info(f"[DIAG_RECV] rank={self.rpc_rank} "
                        #            f"sched_total_tokens={_diag_total} "
                        #            f"tensor_shapes={_diag_shapes} "
                        #            f"pp_layer_config={scheduler_output.pp_layer_config}")
                        # if any(v.shape[0] != _diag_total for v in intermediate_tensors.tensors.values()):
                        #     logger.error(f"[DIAG_RECV] MISMATCH DETECTED at receiver! "
                        #                 f"sched_total={_diag_total} but tensor dim0={[v.shape[0] for v in intermediate_tensors.tensors.values()]}")

                    # DEBUG: Generate unique step_id for cross-rank tracking
                    # Use sorted req_ids to ensure deterministic step_id
                    # all_sched_req_ids = sorted(scheduler_output.num_scheduled_tokens.keys())
                    # step_id = hash(tuple(all_sched_req_ids)) % 100000  # Short hash for readability
                    # total_tokens = scheduler_output.total_num_scheduled_tokens
                    # new_req_ids = [r.request_id for r in scheduler_output.scheduled_new_reqs]
                    # cached_req_ids = [r.req_id for r in scheduler_output.scheduled_cached_reqs]
                    # finished_req_ids = list(scheduler_output.finished_req_ids)
                
                    # # Log comprehensive info for step tracking
                    # logger.info(f"[PP_STEP] rank={self.rpc_rank} step_id={step_id} "
                    #            f"total_tokens={total_tokens} num_reqs={len(all_sched_req_ids)} "
                    #            f"new_reqs={len(new_req_ids)} cached_reqs={len(cached_req_ids)} "
                    #            f"finished_reqs={len(finished_req_ids)} "
                    #            f"has_intermediate={intermediate_tensors is not None}")
                    # # Log per-request token counts for detailed tracking
                    # req_token_summary = [(req_id[-8:], scheduler_output.num_scheduled_tokens[req_id]) 
                    #                      for req_id in all_sched_req_ids[:10]]
                    # logger.info(f"[PP_STEP] rank={self.rpc_rank} step_id={step_id} "
                    #            f"req_tokens(last8chars,tokens)={req_token_summary}")

                    # Wait for KV cache resize BEFORE acquiring forward_lock to avoid deadlock
                    # do_resize thread needs forward_lock to complete resize_kv_cache
                    # with self.worker.inference_stream:
                    # self.worker.wait_for_resize_done()
                    time_before_lock = time.time()
                    with self.worker.model_runner.forward_lock:
                        logger.info(f"[forward]: rank {self.rpc_rank} Acquired forward_lock for executing model, took {human_readable_duration(time.time() - time_before_lock)}")
                        assert isinstance(scheduler_output, DynamicSchedulerOutput), f"Scheduler output is not a DynamicSchedulerOutput:{type(scheduler_output)}"

                        routing_context = (
                            _batch_pp_routing(scheduler_output.pp_layer_config)
                            if envs.VLLM_USE_RAY_COMPILED_DAG
                            else nullcontext()
                        )
                        with routing_context:
                            # 计算通信时间（如果有上游数据）
                            self.worker.async_migration_before_execute_callback(scheduler_output)
                            time_after_before_execute_callback = time.time()

                            # Execute model in high priority stream
                            exec_start = time.time()
                            assert self.worker.inference_stream is not None, "high_priority_stream is not initialized"
                            # with torch.cuda.stream(self.worker.inference_stream):
                            try:
                                logger.info(f"forwarding from layer{scheduler_output.pp_layer_config[self.worker.rank][0]} to layer{scheduler_output.pp_layer_config[self.worker.rank][1]}")
                                output = self.worker.model_runner.execute_model(
                                create_from_dynamic_scheduler_output(scheduler_output),
                                scheduler_output.pp_layer_config[self.rpc_rank],
                                intermediate_tensors)
                            except Exception as e:
                                print(traceback.format_exc())
                                print(f"scheduler_output: {scheduler_output}")
                                print(f"error is raised within the compiled ray DAG graph, error: {e}")
                                time.sleep(1)
                                raise e
                        time_after_execute = time.time()

                        assert(len(self.worker.model_runner.input_batch.block_table.block_tables) == 1) # Only for consistent shape of attention

                        self.worker.async_migration_after_execute_callback(scheduler_output)
                        time_after_execute_callback = time.time()

                    # 在发送给下游前，打包时间戳
                    if isinstance(output, IntermediateTensors):
                        send_time = time.time()  # 记录发送时间
                        # _diag_total = scheduler_output.total_num_scheduled_tokens
                        # _diag_shapes = {k: v.shape for k, v in output.tensors.items()}
                        # # Check for NaN in tensors being sent
                        # _send_nan_info = {}
                        # for k, v in output.tensors.items():
                        #     _has_nan = torch.isnan(v).any().item()
                        #     _nan_count = torch.isnan(v).sum().item() if _has_nan else 0
                        #     _send_nan_info[k] = {"has_nan": _has_nan, "nan_count": _nan_count}
                        #     if _has_nan:
                        #         logger.error(f"[DIAG_SEND_NAN] rank={self.rpc_rank} "
                        #                    f"Sending tensor '{k}' with {_nan_count} NaN values! "
                        #                    f"shape={v.shape} dtype={v.dtype}")
                        # logger.info(f"[DIAG_SEND] rank={self.rpc_rank} "
                        #            f"sched_total_tokens={_diag_total} "
                        #            f"tensor_shapes={_diag_shapes} "
                        #            f"nan_check={_send_nan_info} "
                        #            f"pp_layer_config={scheduler_output.pp_layer_config}")
                        # if any(v.shape[0] != _diag_total for v in output.tensors.values()):
                        #     logger.error(f"[DIAG_SEND] MISMATCH DETECTED at sender! "
                        #                 f"sched_total={_diag_total} but tensor dim0={[v.shape[0] for v in output.tensors.values()]}")
                        if (_dynamic_pp_nccl_transport_enabled()
                                and not envs.VLLM_USE_RAY_COMPILED_DAG):
                            metadata = _send_intermediate_tensors_nccl(
                                self.worker, output)
                            output = (scheduler_output, metadata)
                        else:
                            output = (scheduler_output, output, send_time)

                    if upstream_send_time is not None:
                        comm_time = (time_recv - upstream_send_time) * 1000  # ms
                        if intermediate_tensors is not None:
                            # 计算数据量（正确处理数据类型）
                            hidden_states = intermediate_tensors.tensors["hidden_states"]
                            residual = intermediate_tensors.tensors["residual"]

                            # 使用 numel() 获取元素数量，element_size() 获取每个元素的字节数
                            hidden_size_mb = hidden_states.numel() * hidden_states.element_size() / 1024 / 1024
                            residual_size_mb = residual.numel() * residual.element_size() / 1024 / 1024
                            total_size_mb = hidden_size_mb + residual_size_mb

                            # Avoid division by zero when comm_time is too small
                            if comm_time > 0.001:  # > 1 microsecond
                                bandwidth_str = f"{total_size_mb / (comm_time / 1000):.2f} MB/s"
                            else:
                                bandwidth_str = "N/A (comm_time too small)"

                            logger.info(f"[forward]: rank {self.rpc_rank} Communication time from upstream: {comm_time:.2f} ms, "
                                       f"hidden_states dtype: {hidden_states.dtype}, shape: {hidden_states.shape}, size: {hidden_size_mb:.2f} MB, "
                                       f"residual dtype: {residual.dtype}, shape: {residual.shape}, size: {residual_size_mb:.2f} MB, "
                                       f"total data volume: {total_size_mb:.2f} MB, "
                                       f"bandwidth: {bandwidth_str}")
                        else:
                            logger.info(f"[forward]: rank {self.rpc_rank} Communication time from upstream: {comm_time:.2f} ms, no received data")
                    before_sync = time.time()
                    if (not _dynamic_pp_nccl_transport_enabled()
                            or envs.VLLM_USE_RAY_COMPILED_DAG
                            or _sync_dynamic_pp_nccl_before_return()
                            or not isinstance(output, tuple)
                            or not isinstance(output[1],
                                              PPNCCLIntermediateMetadata)):
                        self.worker.inference_stream.synchronize()
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

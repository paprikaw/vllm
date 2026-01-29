# SPDX-License-Identifier: Apache-2.0

import os
import sched
import traceback
from typing import TYPE_CHECKING, Dict, Tuple, Union

from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.executor.ray_utils import RayWorkerWrapper
from vllm.v1.worker.dynamic_gpu_worker import DynamicGPUWorker, DynamicGPUModelRunner
import torch
from vllm.v1.core.sched.dynamic_scheduler import create_from_dynamic_scheduler_output
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
import time

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)
PG_WAIT_TIMEOUT = 1800

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
                    else:
                        scheduler_output, intermediate_tensors = scheduler_output, None

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
                    self.worker.wait_for_resize_done()

                    with self.worker.model_runner.forward_lock:
                        assert isinstance(scheduler_output, DynamicSchedulerOutput), f"Scheduler output is not a DynamicSchedulerOutput:{type(scheduler_output)}"
                        # 计算通信时间（如果有上游数据）
                        # logger.info(f"[perf_analysis] rank {self.rpc_rank}: About to call async_migration_before_execute_callback()")
                        callback_start = time.time()
                        self.worker.async_migration_before_execute_callback(scheduler_output)
                        time_after_before_execute_callback = time.time()
                        callback_time = time_after_before_execute_callback - callback_start
                        # logger.info(f"[perf_analysis] rank {self.rpc_rank}: async_migration_before_execute_callback() took {callback_time:.4f}s")
                        # self.worker.sync_migration_before_execute_callback(scheduler_output.new_kv_cache_block_num)

                        # Execute model in high priority stream
                        # logger.info(f"[perf_analysis] rank {self.rpc_rank}: About to execute_model()")
                        exec_start = time.time()
                        assert self.worker.inference_stream is not None, "high_priority_stream is not initialized"
                        # with torch.cuda.stream(self.worker.inference_stream):
                        try:
                            logger.info(f"forwarding from layer{scheduler_output.pp_layer_config[self.worker.rank][0]} to layer{scheduler_output.pp_layer_config[self.worker.rank][1]}")
                            model_exec_start = time.time()
                            output = self.worker.model_runner.execute_model(
                            create_from_dynamic_scheduler_output(scheduler_output), 
                            scheduler_output.pp_layer_config[self.rpc_rank],
                            intermediate_tensors)
                            model_exec_end = time.time()
                            model_exec_cpu_time = model_exec_end - model_exec_start
                        except Exception as e:
                            print(traceback.format_exc())
                            print(f"scheduler_output: {scheduler_output}")
                            print(f"error is raised within the compiled ray DAG graph, error: {e}")
                            time.sleep(1)
                            raise e
                
                    # # CRITICAL: Synchronize the high priority stream before using results
                    # # This ensures all computations are complete before we access the output tensors
                    # sync_start = time.time()
                    # torch.cuda.synchronize(self.worker.device)
                    # sync_time = time.time() - sync_start

                    time_after_execute = time.time()
                    exec_time = time_after_execute - exec_start
                    assert(len(self.worker.model_runner.input_batch.block_table.block_tables) == 1) # Only for consistent shape of attention
                    after_callback_start = time.time()
                    self.worker.async_migration_after_execute_callback(scheduler_output)

                    time_after_execute_callback = time.time()
                    after_callback_time = time_after_execute_callback - after_callback_start

                    # 在发送给下游前，打包时间戳
                    if isinstance(output, IntermediateTensors):
                        send_time = time.time()  # 记录发送时间
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
                    logger.info(f"""
                    [forward]: forwarding from layer{scheduler_output.pp_layer_config[self.worker.rank][0]} to layer{scheduler_output.pp_layer_config[self.worker.rank][1]},
                    [forward]: before execute callback time: {time_after_before_execute_callback - time_recv:.2f} seconds,
                    [forward]: execute time: {time_after_execute - time_after_before_execute_callback:.2f} seconds,
                    [forward]: after execute callback time: {time_after_execute_callback - time_after_execute:.2f} seconds
                    [forward]: total time: {time_after_execute_callback - time_recv:.2f} seconds
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

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
            try:
                self.setup_device_if_necessary()
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



                assert isinstance(scheduler_output, DynamicSchedulerOutput), f"Scheduler output is not a DynamicSchedulerOutput:{type(scheduler_output)}"

                # 计算通信时间（如果有上游数据）

                logger.info(f"[forward]: received scheduler output, is_sync_after_migration: {scheduler_output.is_sync_after_migration}, total_migration_tokens: {scheduler_output.total_migration_tokens}")

                self.worker.async_migration_before_execute_callback(scheduler_output.total_migration_tokens)
                time_after_before_execute_callback = time.time()
                # self.worker.sync_migration_before_execute_callback(scheduler_output.new_kv_cache_block_num)
                
                try:
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
                self.worker.async_migration_after_execute_callback(scheduler_output.is_sync_after_migration, scheduler_output.new_kv_cache_block_num)

                time_after_execute_callback = time.time()
                
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
                        
                        logger.info(f"[forward]: rank {self.rpc_rank} Communication time from upstream: {comm_time:.2f} ms, "
                                   f"hidden_states dtype: {hidden_states.dtype}, shape: {hidden_states.shape}, size: {hidden_size_mb:.2f} MB, "
                                   f"residual dtype: {residual.dtype}, shape: {residual.shape}, size: {residual_size_mb:.2f} MB, "
                                   f"total data volume: {total_size_mb:.2f} MB, "
                                   f"bandwidth: {total_size_mb / (comm_time / 1000):.2f} MB/s")
                    else:
                        logger.info(f"[forward]: rank {self.rpc_rank} Communication time from upstream: {comm_time:.2f} ms, no received data")
                logger.info(f"""
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

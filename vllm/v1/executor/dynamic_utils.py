# SPDX-License-Identifier: Apache-2.0

import os
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
                                          "IntermediateTensors"]],
        ) -> Union["ModelRunnerOutput", Tuple["DynamicSchedulerOutput",
                                              "IntermediateTensors"]]:
            # This method is used by Ray Compiled Graph to execute the model,
            # and it needs a special logic of self.setup_device_if_necessary()
            time_start = time.time()
            try:
                self.setup_device_if_necessary()
                assert self.worker is not None, "Worker is not initialized"
                assert isinstance(self.worker, DynamicGPUWorker), "Worker is not a DynamicGPUWorker"
                assert isinstance(self.worker.model_runner, DynamicGPUModelRunner), "Model runner is not a DynamicGPUModelRunner"
                if isinstance(scheduler_output, tuple):
                    scheduler_output, intermediate_tensors = scheduler_output
                else:
                    scheduler_output, intermediate_tensors = scheduler_output, None

                assert isinstance(scheduler_output, DynamicSchedulerOutput), f"Scheduler output is not a DynamicSchedulerOutput:{type(scheduler_output)}"

                # self.worker.kv_synchronize_before_execute_callback(
                #     scheduler_output.is_sync_after_migration)
                self.worker.sync_migration_before_execute_callback(scheduler_output.new_kv_cache_block_num)
                time_after_before_execute_callback = time.time()
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
                # self.worker.kv_synchronize_after_execute_callback(scheduler_output.is_sync_after_migration, scheduler_output.new_kv_cache_block_num)

                time_after_execute_callback = time.time()
                if isinstance(output, IntermediateTensors):
                    output = scheduler_output, output
                # logger.info(f"finished the results:{output}")
                # logger.info(f"""
                # before execute callback time: {time_after_before_execute_callback - time_start:.2f} seconds,
                # execute time: {time_after_execute - time_after_before_execute_callback:.2f} seconds,
                # after execute callback time: {time_after_execute_callback - time_after_execute:.2f} seconds
                # total time: {time_after_execute_callback - time_start:.2f} seconds
                # """)
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

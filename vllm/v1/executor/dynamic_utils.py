# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Dict, Tuple, Union

from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.executor.ray_utils import RayWorkerWrapper
from vllm.v1.worker.dynamic_gpu_worker import DynamicGPUWorker
from vllm.v1.core.sched.dynamic_scheduler import create_from_dynamic_scheduler_output

if TYPE_CHECKING:
    from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
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
            self.setup_device_if_necessary()
            assert self.worker is not None, "Worker is not initialized"
            assert isinstance(self.worker, DynamicGPUWorker), "Worker is not a DynamicGPUWorker"
            if isinstance(scheduler_output, tuple):
                scheduler_output, intermediate_tensors = scheduler_output
            else:
                scheduler_output, intermediate_tensors = scheduler_output, None

            try:
                output = self.worker.model_runner.execute_model(
                create_from_dynamic_scheduler_output(scheduler_output), 
                scheduler_output.pp_layer_config[self.rpc_rank], 
                intermediate_tensors)
            except Exception as e:
                logger.exception("Exception occurred during execute_model")
                raise e
            if isinstance(output, IntermediateTensors):
                output = scheduler_output, output
            logger.info(f"finished the results:{output}")
            return output

except ImportError as e:
    ray = None  # type: ignore
    ray_import_err = e
    RayWorkerWrapper = None  # type: ignore

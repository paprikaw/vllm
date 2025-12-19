# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheConfig
import os
from collections import defaultdict
from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor
from vllm.executor.ray_distributed_executor import RayWorkerMetaData
from vllm.v1.executor.dynamic_utils import DynamicRayWorkerWrapper
from vllm.v1.core.sched.dynamic_scheduler import DynamicSchedulerOutput
import vllm.envs as envs
from vllm.executor.ray_utils import (RayWorkerWrapper, 
                                     ray)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import (get_distributed_init_method,
                        get_ip, get_open_port)
from bitarray import bitarray
if ray is not None:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
else:
    ActorHandle = None
if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup
from vllm.v1.executor.abstract import ModelRunnerOutput
from vllm.v1.executor.abstract import Future
from vllm.v1.executor.ray_distributed_executor import FutureWrapper
from vllm.v1.utils import WorkerMemInfo
from vllm.v1.worker.utils import KVBufferStatus
logger = init_logger(__name__)

class DynamicRayDistributedExecutor(RayDistributedExecutor):


    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        # Copied from vllm.executor.ray_distributed_executor.RayDistributedExecutor._init_workers_ray
        # We need to override this for our own DynamicRayWorkerWrapper

        num_gpus = envs.VLLM_RAY_PER_WORKER_GPUS

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: Optional[DynamicRayWorkerWrapper] = None
        # The remaining workers are the actual ray actors.
        self.workers: List[DynamicRayWorkerWrapper] = []

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        self.pp_tp_workers: List[List[DynamicRayWorkerWrapper]] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)

        logger.info("use_ray_spmd_worker: %s", self.use_ray_spmd_worker)

        # Create the workers.
        bundle_indices: List[int]
        if envs.VLLM_RAY_BUNDLE_INDICES:
            # Use the bundle indices specified by the user.
            bundle_indices = list(
                map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
            assert len(bundle_indices) == self.parallel_config.world_size, \
            ("VLLM_RAY_BUNDLE_INDICES must have the same size"
            f" as the world size, but got {bundle_indices=} "
            f"and {self.parallel_config.world_size=}")
            assert len(set(bundle_indices)) == len(bundle_indices), \
            ("VLLM_RAY_BUNDLE_INDICES cannot have duplicate values,"
            f" but got {bundle_indices=}")
        else:
            # use the first N bundles that have GPU resources.
            bundle_indices = []
            for bundle_id, bundle in enumerate(placement_group.bundle_specs):
                if bundle.get(current_platform.ray_device_key, 0):
                    bundle_indices.append(bundle_id)
            bundle_indices = bundle_indices[:self.parallel_config.world_size]

        worker_metadata: List[RayWorkerMetaData] = []
        driver_ip = get_ip()
        for rank, bundle_id in enumerate(bundle_indices):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            if current_platform.ray_device_key == "GPU":
                # NV+AMD GPUs, and Intel XPUs
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=num_gpus,
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(DynamicRayWorkerWrapper).remote(vllm_config=self.vllm_config,
                                           rpc_rank=rank)
            else:
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    resources={current_platform.ray_device_key: num_gpus},
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(DynamicRayWorkerWrapper).remote(vllm_config=self.vllm_config,
                                           rpc_rank=rank)
            worker_metadata.append(
                RayWorkerMetaData(worker=worker, created_rank=rank))

        worker_ips = ray.get([
            each.worker.get_node_ip.remote()  # type: ignore[attr-defined]
            for each in worker_metadata
        ])

        for each, ip in zip(worker_metadata, worker_ips):
            each.ip = ip

        if not self.use_ray_spmd_worker:
            for i, each in enumerate(worker_metadata):
                # find and remove the dummy worker from the list
                worker = each.worker
                worker_ip = each.ip
                if self.driver_dummy_worker is None and worker_ip == driver_ip:
                    # If the worker is on the same node as the driver, we use it
                    # as the resource holder for the driver process.
                    self.driver_dummy_worker = worker
                    self.driver_worker = DynamicRayWorkerWrapper(
                        vllm_config=self.vllm_config, rpc_rank=0)
                    worker_metadata.pop(i)
                    break

        if not self.use_ray_spmd_worker and self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node."
                f"Driver IP: {driver_ip}, worker IPs: {worker_ips}."
                "Consider adjusting the Ray placement group or running "
                "the driver on a GPU node.")

        ip_counts: Dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1
        def sort_by_driver_then_worker_ip(item: RayWorkerMetaData):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = item.ip
            return (0 if ip == driver_ip else 1, ip_counts[ip], ip)

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        sorted_worker_metadata = sorted(worker_metadata,
                                        key=sort_by_driver_then_worker_ip)
        start_rank = 0 if self.use_ray_spmd_worker else 1
        for i, item in enumerate(sorted_worker_metadata):
            item.adjusted_rank = i + start_rank
        self.workers = [item.worker for item in sorted_worker_metadata]
        rerank_mapping = {
            item.created_rank: item.adjusted_rank
            for item in sorted_worker_metadata
        }
        self._run_workers("adjust_rank", rerank_mapping)

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = []
        for worker in [self.driver_dummy_worker] + self.workers:
            if worker is None:
                # driver_dummy_worker can be None when using ray spmd worker.
                continue
            worker_node_and_gpu_ids.append(
                ray.get(worker.get_node_and_gpu_ids.remote()) \
            ) # type: ignore

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_gpus = defaultdict(list)  # node id -> list of gpu ids

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            # `gpu_ids` can be a list of strings or integers.
            # convert them to integers for consistency.
            # NOTE: gpu_ids can be larger than 9 (e.g. 16 GPUs),
            # string sorting is not sufficient.
            # see https://github.com/vllm-project/vllm/issues/5590
            gpu_ids = [int(x) for x in gpu_ids]
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        all_ips = set(worker_ips + [driver_ip])
        n_ips = len(all_ips)
        n_nodes = len(node_workers)

        if n_nodes != n_ips:
            raise RuntimeError(
                f"Every node should have a unique IP address. Got {n_nodes}"
                f" nodes with node ids {list(node_workers.keys())} and "
                f"{n_ips} unique IP addresses {all_ips}. Please check your"
                " network configuration. If you set `VLLM_HOST_IP`"
                " environment variable, make sure it is unique for"
                " each node.")

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [{
            current_platform.device_control_env_var:
            ",".join(map(str, node_gpus[node_id])),
        } for (node_id, _) in worker_node_and_gpu_ids]

        # Environment variables to copy from driver to workers
        env_vars_to_copy = [
            v for v in envs.environment_variables
            if v not in self.WORKER_SPECIFIC_ENV_VARS
            and v not in self.non_carry_over_env_vars
        ]

        env_vars_to_copy.extend(current_platform.additional_env_vars)

        # Copy existing env vars to each worker's args
        for args in all_args_to_update_environment_variables:
            # TODO: refactor platform-specific env vars
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]

        logger.info("non_carry_over_env_vars from config: %s",
                    self.non_carry_over_env_vars)
        logger.info(
            "Copying the following environment variables to workers: %s",
            [v for v in env_vars_to_copy if v in os.environ])
        logger.info(
            "If certain env vars should NOT be copied to workers, add them to "
            "%s file", self.non_carry_over_env_vars_file)

        self._env_vars_for_all_workers = (
            all_args_to_update_environment_variables)

        self._run_workers("update_environment_variables",
                          self._get_env_vars_to_be_updated())

        if len(node_gpus) == 1:
            # in single node case, we don't need to get the IP address.
            # the loopback address is sufficient
            # NOTE: a node may have several IP addresses, one for each
            # network interface. `get_ip()` might return any of them,
            # while they might not work for communication inside the node
            # if the network setup is complicated. Using the loopback address
            # solves this issue, as it always works for communication inside
            # the node.
            driver_ip = "127.0.0.1"
        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port())

        # Initialize the actual workers inside worker wrapper.
        all_kwargs = []
        for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids):
            local_rank = node_workers[node_id].index(rank)
            kwargs = dict(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=(not self.parallel_config)
                or (rank % self.parallel_config.tensor_parallel_size == 0),
            )
            all_kwargs.append(kwargs)
        self._run_workers("init_worker", all_kwargs)

        self._run_workers("init_device")
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)

        if self.use_ray_spmd_worker:
            for pp_rank in range(self.parallel_config.pipeline_parallel_size):
                self.pp_tp_workers.append([])
                for tp_rank in range(
                        self.parallel_config.tensor_parallel_size):
                    # PP=2, TP=4
                    # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                    rank = (pp_rank * self.parallel_config.tensor_parallel_size
                            ) + tp_rank
                    assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                    assert pp_rank < len(self.pp_tp_workers)
                    self.pp_tp_workers[pp_rank].append(self.workers[rank])

        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[DynamicRayWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[DynamicRayWorkerWrapper] = []

        # Enforce rank order for correct rank to return final output.
        for index, worker in enumerate(self.workers):
            # The driver worker is rank 0 and not in self.workers.
            rank = index + 1
            if rank % self.parallel_config.tensor_parallel_size == 0:
                self.tp_driver_workers.append(worker)
            else:
                self.non_driver_workers.append(worker)


    def _compiled_cpu_ray_dag(self, enable_asyncio: bool):
        assert self.parallel_config.use_ray
        self._check_ray_cgraph_installation()
        from ray.dag import InputNode, MultiOutputNode


        # Enlarge the default value of "RAY_CGRAPH_get_timeout" to 300 seconds
        # (it is 10 seconds by default). This is a Ray environment variable to
        # control the timeout of getting result from a compiled graph execution,
        # i.e., the distributed execution that includes model forward runs and
        # intermediate tensor communications, in the case of vllm.
        os.environ.setdefault("RAY_CGRAPH_get_timeout", "300")  # noqa: SIM112
        logger.info("debug start compile cpu ray dag")
        logger.info("RAY_CGRAPH_get_timeout is set to %s",
                    os.environ["RAY_CGRAPH_get_timeout"])  # noqa: SIM112

        with InputNode() as input_data:
            # Example DAG: PP=2, TP=4
            #
            # For V0:
            # ExecuteModelRequest -> 0 -> (ExecuteModelReq, IntermediateTensors) -> 4 -> SamplerOutput   # noqa: E501
            # ExecuteModelRequest -> 1 -> (ExecuteModelReq, IntermediateTensors) -> 5 -> SamplerOutput   # noqa: E501
            # ExecuteModelRequest -> 2 -> (ExecuteModelReq, IntermediateTensors) -> 6 -> SamplerOutput   # noqa: E501
            # ExecutaModelRequest -> 3 -> (ExecuteModelReq, IntermediateTensors) -> 7 -> SamplerOutput   # noqa: E501
            #
            # For V1:
            # SchedulerOutput -> 0 -> (SchedulerOutput, IntermediateTensors) -> 4 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 1 -> (SchedulerOutput, IntermediateTensors) -> 5 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 2 -> (SchedulerOutput, IntermediateTensors) -> 6 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 3 -> (SchedulerOutput, IntermediateTensors) -> 7 -> ModelRunnerOutput   # noqa: E501

            # Build two branches in one DAG: default(SHM) and NCCL.
            # Both branches share the same workers and execution steps,
            # but only the NCCL branch specifies 'nccl' transport between PP stages.
            outputs_shm = [input_data for _ in self.pp_tp_workers[0]]
            for pp_rank, tp_group in enumerate(self.pp_tp_workers):
                # Advance SHM branch (explicitly use auto -> SHM/object-store on-node)
                if self.use_v1:
                    outputs_shm = [
                        worker.execute_model_ray.bind(  # type: ignore[attr-defined]
                            outputs_shm[i]) for i, worker in enumerate(tp_group)
                    ]
                else:
                    outputs_shm = [
                        worker.execute_model_spmd.bind(  # type: ignore[attr-defined]
                            outputs_shm[i]) for i, worker in enumerate(tp_group)
                    ]
                last_pp_rank = len(self.pp_tp_workers) - 1
                if pp_rank < last_pp_rank:
                    outputs_shm = [
                        output.with_tensor_transport(transport="shm")
                        for output in outputs_shm
                    ]
            # Combine branch outputs; we will pick one branch at runtime.
            forward_dag = MultiOutputNode(outputs_shm)
            # Record per-branch output size for selection later.
            self._cpu_tp_group_size = len(self.pp_tp_workers[-1])

        return forward_dag.experimental_compile(
            enable_asyncio=enable_asyncio,
            _overlap_gpu_communication=envs.
            VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM)

    def _init_executor(self) -> None:
        super()._init_executor()
        self.cpu_forward_dag = None

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        # 在这里我们也事先构建cpu的dag，减少migratin期间切换到cpu dag的延迟
        # if self.cpu_forward_dag is None:  # type: ignore
        #     self.cpu_forward_dag = self._compiled_cpu_ray_dag(enable_asyncio=False)
        return super().execute_model(scheduler_output)

    def execute_cpu_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        """
        Copied from vllm.executor.ray_distributed_executor.RayDistributedExecutor.execute_model
        The only difference is we also initialize the cpu_forward_dag
        Execute the model on the Ray workers.

        Args:
            scheduler_output: The scheduler output to execute.

        Returns:
            The model runner output.
        """
        # Build the compiled DAG for the first time.
        if self.cpu_forward_dag is None:
            self.cpu_forward_dag = self._compiled_cpu_ray_dag(enable_asyncio=False)
        refs = self.cpu_forward_dag.execute(scheduler_output)  # type: ignore
        # Select branch at runtime: default(SHM) uses the first block; NCCL uses the second.
        tp_group = getattr(self, "_cpu_tp_group_size", len(self.pp_tp_workers[-1]))
        channel = os.environ.get("VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "shm").lower()
        start_idx = 0 if channel == "shm" else tp_group
        # When PP is not used, we block here until the result is available.
        if self.max_concurrent_batches == 1:
            return refs[start_idx].get()

        # When PP is used, we return a FutureWrapper immediately so that
        # the scheduler can yield to the next batch.
        return FutureWrapper(refs[start_idx])

    def initialize_kv_cache_for_layers(self, rank: int, 
                                        kv_cache_specs: dict[str, KVCacheSpec], 
                                        kv_cache_size: int, 
                                        kv_cache_num_blocks: int,
                                        layers: Tuple[int, int]) -> None:
        self.collective_rpc("initialize_kv_cache_for_layers", args=(rank, kv_cache_specs, kv_cache_size, kv_cache_num_blocks, layers))

    def get_kv_cache_spec_for_layers(self, rank: int, layer_range: Tuple[int, int]) -> dict[str, KVCacheSpec]:
        output = self.collective_rpc("get_kv_cache_spec_for_layers", args=(rank, layer_range))
        return output[rank]

    def add_layers(self, rank: int, layers_list: list[Tuple[int, int]]):
        self.collective_rpc("add_layers", args=(rank, layers_list))
    
    def async_add_layers(self, rank: int, layers_list: list[Tuple[int, int]]):
        # fire-and-forget 异步发起，每个 worker 内部用线程执行
        self.collective_rpc("async_add_layers", args=(rank, layers_list))
    
    def start_kv_cache_migration_async(self, src_to_plan: dict[int, dict[int, list[int]]], slot_mapping: Optional[list[int]]):
        # 调用 worker 侧的同名方法，仅在 source_rank 上发送，其余 rank 不做事
        self.collective_rpc("start_kv_cache_migration_async", args=(src_to_plan,slot_mapping))

    def get_applied_token_num(self, receiver_list: list[int]) -> list[list[int]]:
        """Return KV patch buffer status per rank.
        """
        return self.collective_rpc("get_applied_token_num", args=(receiver_list,))

    def start_kv_cache_migration_sync(self, src_to_plan: dict[int, dict[int, list[Tuple[int, int]]]], dst_to_layer_ids: dict[int, list[Tuple[int, int]]]):
        # 调用 worker 侧的同名方法，仅在 source_rank 上发送，其余 rank 不做事
        self.collective_rpc("start_kv_cache_migration_sync", args=(src_to_plan, dst_to_layer_ids,))

    def get_kv_buffer_status(self) -> list[KVBufferStatus]:
        output = self.collective_rpc("get_kv_buffer_status")
        return output

    def remove_layers(self, rank: int, layers_list: list[Tuple[int, int]]):
        self.collective_rpc("remove_layers", args=(rank, layers_list))

    def get_current_available_memory(self) -> List[int]:
        return self.collective_rpc("get_current_available_memory")

    def get_workers_mem_info(self) -> List[WorkerMemInfo]:
        """Return per worker (layer_size, free_memory, single_kv_tensor_size).

        - layer_size: bytes of a single layer's weights (recorded by model).
        - free_memory: current free GPU memory in bytes.
        - single_kv_tensor_size: bytes of one layer's KV cache tensor.
        """
        return self.collective_rpc("get_mem_info")
    
    def compact_kv_cache(self, compacted_length: int, bitmap: bitarray) -> None:
        self.collective_rpc("compact_kv_cache", args=(compacted_length, bitmap))

    def resize_kv_cache(self, new_length: int) -> None:
        self.collective_rpc("resize_kv_cache", args=(new_length,))

    def release_kv_cache_for_layers(self, rank: int, layers_list: list[Tuple[int, int]]):
        self.collective_rpc("release_kv_cache_for_layers", args=(rank, layers_list))

    def release_kv_cache(self) -> None:
        self.collective_rpc("release_kv_cache")

    def reinitialize_kv_cache(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        assert len(kv_cache_configs) == self.parallel_config.world_size, "kv_cache_configs must have the same length as world_size"
        self.collective_rpc("reinitialize_kv_cache", args=(kv_cache_configs,))

    def dynamic_initialize_from_config(self, kv_cache_configs: list[KVCacheConfig], num_blocks: int) -> None:
        self.collective_rpc("dynamic_initialize_from_config", args=(kv_cache_configs, num_blocks))
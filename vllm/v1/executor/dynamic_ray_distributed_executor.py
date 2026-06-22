# SPDX-License-Identifier: Apache-2.0
import asyncio
from concurrent.futures import Future as StdFuture
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheConfig
import msgspec
import os
import time
from collections import defaultdict
import torch
from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor
from vllm.executor.ray_distributed_executor import RayWorkerMetaData
from vllm.executor.msgspec_utils import encode_hook
from vllm.v1.executor.dynamic_utils import (
    DynamicRayWorkerWrapper,
    PPNCCLIntermediateMetadata,
    install_vllm_ray_rdt_gpu_object_patch,
)
from vllm.v1.core.sched.dynamic_scheduler import DynamicSchedulerOutput
import vllm.envs as envs
from vllm.executor.ray_utils import (RayWorkerWrapper, initialize_ray_cluster,
                                     ray)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.utils import (get_distributed_init_method,
                        get_ip, get_open_port, make_async)
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


def _dynamic_pp_rdt_transport() -> str:
    transport = os.getenv("VLLM_DYNAMIC_PP_RDT_TRANSPORT", "").lower()
    if transport in ("1", "true"):
        return "nccl"
    if transport in ("", "0", "false", "none", "object_store"):
        return ""
    return transport


def _dynamic_pp_rdt_nccl_enabled() -> bool:
    return _dynamic_pp_rdt_transport() == "nccl"


def _dynamic_pp_rdt_prewarm_enabled() -> bool:
    value = os.getenv("VLLM_DYNAMIC_PP_RDT_PREWARM", "0").lower()
    return value in ("1", "true", "yes", "on")


class RayObjectRefFuture(StdFuture):
    """Future wrapper for a plain Ray ObjectRef.

    Ray Compiled DAG refs expose `.get()`, while regular Ray ObjectRefs are
    consumed with `ray.get()`. The v1 engine only needs `result()`.
    """

    def __init__(self, ref, refs=None):
        super().__init__()
        self.ref = ref
        self.refs = list(refs) if refs is not None else [ref]

    def result(self, timeout=None):
        if timeout is not None:
            raise NotImplementedError("timeout is not supported")
        return ray.get(self.ref)


class DynamicRayDistributedExecutor(RayDistributedExecutor):

    @property
    def max_concurrent_batches(self) -> int:
        override = os.getenv("VLLM_DYNAMIC_PP_MAX_CONCURRENT_BATCHES")
        if override:
            return max(1, int(override))
        active_ranks = getattr(self, "active_pp_ranks", None)
        if active_ranks:
            return len(active_ranks)
        return self.parallel_config.pipeline_parallel_size


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
        self.active_pp_ranks: Optional[list[int]] = None
        self._autoscaling_active_pp_ranks_generation: int = -1
        self._autoscaling_active_pp_ranks_ref = None
        self._autoscaling_request_state_sync_pending: Optional[
            tuple[int, ...]] = None

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)
        if _dynamic_pp_rdt_nccl_enabled():
            if current_platform.ray_device_key != "GPU":
                raise ValueError(
                    "VLLM_DYNAMIC_PP_RDT_TRANSPORT=nccl requires GPU Ray "
                    f"actors, got {current_platform.ray_device_key}.")
            ray_remote_kwargs["enable_tensor_transport"] = True
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
        capture_child_tasks = not _dynamic_pp_rdt_nccl_enabled()
        for rank, bundle_id in enumerate(bundle_indices):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=capture_child_tasks,
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

        # Build a mapping from node_id to worker IP for setting VLLM_HOST_IP per worker
        # worker_node_and_gpu_ids contains (node_id, gpu_ids) for each worker in order
        # We can directly map node_id -> IP using the worker_ips we collected earlier
        node_id_to_ip: Dict[str, str] = {}
        
        # Collect IPs for all workers including driver
        # For SPMD mode: self.workers contains all workers, worker_ips has their IPs
        # For non-SPMD mode: driver_dummy_worker is separate
        if self.use_ray_spmd_worker:
            # worker_ips corresponds to worker_metadata (before sorting)
            # worker_node_and_gpu_ids corresponds to self.workers (after sorting)
            # We need to build the mapping from node_id to IP
            for i, (node_id, _) in enumerate(worker_node_and_gpu_ids):
                if i < len(sorted_worker_metadata):
                    node_id_to_ip[node_id] = sorted_worker_metadata[i].ip
        else:
            # First worker is driver on driver_ip
            if worker_node_and_gpu_ids:
                node_id_to_ip[worker_node_and_gpu_ids[0][0]] = driver_ip
            # Rest are from sorted_worker_metadata
            for i, (node_id, _) in enumerate(worker_node_and_gpu_ids[1:], start=0):
                if i < len(sorted_worker_metadata):
                    node_id_to_ip[node_id] = sorted_worker_metadata[i].ip
        
        logger.info("Node ID to IP mapping: %s", node_id_to_ip)

        # Set environment variables for the driver and workers.
        # Include per-worker VLLM_HOST_IP based on the node's IP
        rank_to_local_rank = [
            node_workers[node_id].index(rank)
            for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids)
        ]
        rdt_rank_to_device = ",".join(str(rank)
                                      for rank in rank_to_local_rank)
        all_args_to_update_environment_variables = []
        for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids):
            worker_ip = node_id_to_ip.get(node_id, driver_ip)
            args = {
                current_platform.device_control_env_var:
                ",".join(map(str, node_gpus[node_id])),
                'VLLM_HOST_IP': worker_ip,  # Set per-worker IP
                'VLLM_RAY_RDT_RANK_TO_DEVICE': rdt_rank_to_device,
                'VLLM_RAY_RDT_LOCAL_DEVICE_INDEX':
                str(rank_to_local_rank[rank]),
            }
            all_args_to_update_environment_variables.append(args)
            logger.info("Worker on node %s will use VLLM_HOST_IP=%s", node_id, worker_ip)

        # Environment variables to copy from driver to workers
        env_vars_to_copy = [
            v for v in envs.environment_variables
            if v not in self.WORKER_SPECIFIC_ENV_VARS
            and v not in self.non_carry_over_env_vars
        ]

        env_vars_to_copy.extend(current_platform.additional_env_vars)
        
        # Add NCCL/GLOO network interface env vars (critical for multi-node)
        nccl_env_vars = [
            'NCCL_SOCKET_IFNAME',
            'GLOO_SOCKET_IFNAME', 
            'NCCL_IB_DISABLE',
            'NCCL_NET_GDR_LEVEL',
            'NCCL_P2P_LEVEL',
            'NCCL_SHM_DISABLE',
            'VLLM_DYNAMIC_PP_NCCL_TRANSPORT',
            'VLLM_DYNAMIC_PP_RDT_TRANSPORT',
        ]
        for var in nccl_env_vars:
            if var not in env_vars_to_copy:
                env_vars_to_copy.append(var)

        # Critical env vars that MUST be copied to workers even if not in os.environ
        # These use vLLM's default values from envs.py
        critical_env_vars_with_defaults = {
            'RAY_DEDUP_LOGS': '0',  # Disable Ray log deduplication for debugging
        }

        # Copy existing env vars to each worker's args
        for args in all_args_to_update_environment_variables:
            # TODO: refactor platform-specific env vars
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]
                elif name in critical_env_vars_with_defaults:
                    # Use vLLM's default value for critical env vars not set in os.environ
                    args[name] = critical_env_vars_with_defaults[name]

        # Collect which env vars are actually being copied
        copied_vars = [v for v in env_vars_to_copy if v in os.environ]
        copied_with_defaults = [v for v in critical_env_vars_with_defaults 
                                if v not in os.environ and v in env_vars_to_copy]

        logger.info("non_carry_over_env_vars from config: %s",
                    self.non_carry_over_env_vars)
        logger.info(
            "Copying the following environment variables to workers: %s",
            copied_vars)
        if copied_with_defaults:
            logger.info(
                "Using vLLM default values for env vars not in os.environ: %s",
                {v: critical_env_vars_with_defaults[v] for v in copied_with_defaults})
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
        dynamic_config = self.vllm_config.dynamic_config
        if getattr(dynamic_config, "pipeline_autoscaling_enabled", False):
            active_ranks = self._active_ranks_from_partition(
                dynamic_config.pp_layer_partition)
            self._run_workers("set_active_pp_ranks", active_ranks)
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
                    rank = self.parallel_config.get_rank_for_pipeline_stage(
                        pp_rank, tp_rank)
                    assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                    assert pp_rank < len(self.pp_tp_workers)
                    self.pp_tp_workers[pp_rank].append(self.workers[rank])
            if getattr(dynamic_config, "pipeline_autoscaling_enabled", False):
                self._set_active_pp_ranks_local(active_ranks)

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

    def _active_ranks_from_partition(
        self,
        pp_layer_partition: Optional[str],
    ) -> list[int]:
        if pp_layer_partition is None:
            return list(range(self.parallel_config.pipeline_parallel_size))
        parts = [int(x.strip()) for x in pp_layer_partition.split(",")]
        if len(parts) > self.parallel_config.pipeline_parallel_size:
            raise ValueError(
                "pp_layer_partition has more entries than "
                f"pipeline_parallel_size: {pp_layer_partition}")
        return [rank for rank, num_layers in enumerate(parts)
                if num_layers > 0]

    def _set_active_pp_ranks_local(self, active_ranks: Optional[list[int]]) -> None:
        self.active_pp_ranks = list(active_ranks) if active_ranks else None
        if not self.use_ray_spmd_worker or not active_ranks:
            return
        if self.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "pipeline autoscaling currently supports TP=1 only")
        if self.use_ray_compiled_dag:
            if getattr(self, "forward_dag", None) is not None:
                logger.info("Tearing down old compiled PP worker chain before "
                            "switching active ranks")
                self.forward_dag.teardown()
            if getattr(self, "cpu_forward_dag", None) is not None:
                logger.info("Tearing down old compiled CPU PP worker chain before "
                            "switching active ranks")
                self.cpu_forward_dag.teardown()
            if self.workers:
                ray.get([
                    worker.reset_ray_compiled_dag_nccl_lock.remote()
                    for worker in self.workers
                ])
        self.pp_tp_workers = [[self.workers[rank]] for rank in active_ranks]
        self.forward_dag = None
        self.cpu_forward_dag = None
        logger.info("Updated active PP worker chain to ranks %s", active_ranks)

    def commit_active_pp_ranks_local(
        self,
        active_ranks: Optional[list[int]],
    ) -> None:
        """Commit the driver-side PP actor chain after worker routing is ready."""
        start = time.time()
        previous_active = set(self.active_pp_ranks or [])
        next_active = set(active_ranks or [])
        if next_active - previous_active:
            self._autoscaling_request_state_sync_pending = tuple(
                active_ranks or [])
            logger.info(
                "[autoscaling request states] will carry request states on "
                "first target PP batch for active_ranks=%s previous=%s",
                active_ranks, self.active_pp_ranks)
        self._set_active_pp_ranks_local(active_ranks)
        logger.info(
            "[autoscaling active ranks] committed local PP worker chain to "
            "ranks %s in %.3fs", active_ranks, time.time() - start)

    def _start_autoscaling_active_pp_ranks_chain(
        self,
        scheduler_output,
    ) -> None:
        active_ranks = getattr(scheduler_output,
                               "autoscaling_activate_pp_ranks", None)
        if active_ranks is None:
            return
        generation = getattr(
            scheduler_output, "autoscaling_activate_pp_ranks_generation", -1)
        if generation <= self._autoscaling_active_pp_ranks_generation:
            return
        start = time.time()
        worker_ranks = list(range(len(self.workers)))
        marker = None
        for rank in worker_ranks:
            marker = self.workers[
                rank].activate_pp_ranks_for_autoscaling_chain.remote(
                    active_ranks, generation, marker)
        self._autoscaling_active_pp_ranks_generation = generation
        self._autoscaling_active_pp_ranks_ref = marker
        logger.info(
            "[autoscaling active ranks] started actor-chain activation for "
            "generation=%s active_ranks=%s worker_ranks=%s in %.3fs",
            generation, active_ranks, worker_ranks, time.time() - start)

    def wait_for_autoscaling_active_pp_ranks(
        self,
        generation: int,
    ) -> None:
        ref = self._autoscaling_active_pp_ranks_ref
        if ref is None or generation > self._autoscaling_active_pp_ranks_generation:
            logger.info(
                "[autoscaling active ranks] no actor-chain activation to wait "
                "for generation=%s; latest_generation=%s",
                generation, self._autoscaling_active_pp_ranks_generation)
            return
        start = time.time()
        ray.get(ref)
        logger.info(
            "[autoscaling active ranks] actor-chain activation ready for "
            "generation=%s in %.3fs", generation, time.time() - start)

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
        dynamic_config = self.vllm_config.dynamic_config
        use_dynamic_actor_chain = (
            envs.VLLM_USE_V1
            and getattr(dynamic_config, "pipeline_autoscaling_enabled", False)
            and not envs.VLLM_USE_RAY_COMPILED_DAG
        )
        if not use_dynamic_actor_chain:
            super()._init_executor()
            self.cpu_forward_dag = None
            return

        self.forward_dag = None
        # V1 normally forces Ray Compiled DAG. Autoscaling needs the PP chain
        # to be selected at runtime, so keep the SPMD actor pool but avoid
        # compiling a static DAG topology.
        os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "1"
        os.environ["VLLM_USE_RAY_COMPILED_DAG"] = "0"
        self.use_ray_compiled_dag = False
        self.use_ray_spmd_worker = True

        assert self.uses_ray
        initialize_ray_cluster(self.parallel_config)
        placement_group = self.parallel_config.placement_group

        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        self._init_workers_ray(placement_group)
        self.input_encoder = msgspec.msgpack.Encoder(enc_hook=encode_hook)
        self.output_decoder = msgspec.msgpack.Decoder(
            Optional[List[SamplerOutput]])
        self.use_v1 = envs.VLLM_USE_V1
        self.pp_locks: Optional[List[asyncio.Lock]] = None
        self.cpu_forward_dag = None

        self._init_dynamic_pp_rdt_transport()

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        if not self.use_ray_compiled_dag and self.use_ray_spmd_worker:
            return self._execute_model_dynamic_actor_chain(scheduler_output)
        return super().execute_model(scheduler_output)

    def _execute_model_dynamic_actor_chain(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        self._start_autoscaling_active_pp_ranks_chain(scheduler_output)
        active_ranks = self._active_ranks_for_scheduler_output(
            scheduler_output)
        if not active_ranks:
            active_ranks = getattr(self, "active_pp_ranks", None)
        self._sync_request_states_before_target_chain(
            scheduler_output, active_ranks)
        if self.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "dynamic Ray actor PP chain currently supports TP=1 only")

        if _dynamic_pp_rdt_nccl_enabled():
            if getattr(self, "_dynamic_pp_rdt_group", None) is None:
                self._init_dynamic_pp_rdt_transport()
            self._annotate_pp_nccl_debug_metadata(
                scheduler_output, active_ranks)
            ref_or_value = scheduler_output
            refs = []
            last_index = len(active_ranks) - 1
            for index, rank in enumerate(active_ranks):
                method = self.workers[rank].execute_model_ray
                if index < last_index:
                    ref_or_value = method.options(
                        tensor_transport="nccl").remote(ref_or_value)
                else:
                    ref_or_value = method.remote(ref_or_value)
                refs.append(ref_or_value)
            if self.max_concurrent_batches == 1:
                return ray.get(ref_or_value)
            return RayObjectRefFuture(ref_or_value, refs=refs)

        if os.getenv("VLLM_DYNAMIC_PP_NCCL_TRANSPORT", "0") == "1":
            self._annotate_pp_nccl_debug_metadata(
                scheduler_output, active_ranks)
            metadata = self._make_pp_nccl_intermediate_metadata(
                scheduler_output)
            refs = []
            for index, rank in enumerate(active_ranks):
                worker_input = (scheduler_output if index == 0 else
                                (scheduler_output, metadata))
                refs.append(self.workers[rank].execute_model_ray.remote(
                    worker_input))
            self._last_pp_nccl_refs = refs
            final_ref = refs[-1]
            if self.max_concurrent_batches == 1:
                return ray.get(final_ref)
            return RayObjectRefFuture(final_ref, refs=refs)

        ref_or_value = scheduler_output
        for rank in active_ranks:
            ref_or_value = self.workers[rank].execute_model_ray.remote(
                ref_or_value)

        if self.max_concurrent_batches == 1:
            return ray.get(ref_or_value)
        return RayObjectRefFuture(ref_or_value)

    def _init_dynamic_pp_rdt_transport(self) -> None:
        transport = _dynamic_pp_rdt_transport()
        if not transport:
            return
        if transport != "nccl":
            raise ValueError(
                "Unsupported VLLM_DYNAMIC_PP_RDT_TRANSPORT="
                f"{transport!r}; Ray 2.49 dynamic PP path supports 'nccl'.")
        if self.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "dynamic PP RDT transport currently supports TP=1 only")
        if getattr(self, "_dynamic_pp_rdt_group", None) is not None:
            self._prewarm_dynamic_pp_rdt_edges()
            return

        install_vllm_ray_rdt_gpu_object_patch()

        from ray.experimental.collective import (create_collective_group,
                                                get_collective_groups)

        existing_groups = get_collective_groups([self.workers[0]],
                                                backend="nccl")
        if existing_groups:
            if len(existing_groups) > 1:
                raise RuntimeError(
                    "Ray RDT found multiple NCCL collective groups for "
                    "dynamic PP workers; expected exactly one.")
            self._dynamic_pp_rdt_group = existing_groups[0]
            logger.info("Reusing Ray RDT NCCL collective group %s",
                        self._dynamic_pp_rdt_group.name)
            self._prewarm_dynamic_pp_rdt_edges()
            return

        group_name = f"vllm_dynamic_pp_rdt_nccl_{id(self)}"
        logger.info("Creating Ray RDT NCCL collective group %s for %d workers",
                    group_name, len(self.workers))
        self._dynamic_pp_rdt_group = create_collective_group(
            self.workers, backend="nccl", name=group_name)
        self._prewarm_dynamic_pp_rdt_edges()

    def _dynamic_pp_rdt_prewarm_ranks(self) -> list[int]:
        dynamic_config = self.vllm_config.dynamic_config
        ranks = getattr(dynamic_config, "autoscaling_candidate_ranks", None)
        if not ranks:
            ranks = list(range(self.parallel_config.pipeline_parallel_size))
        ranks = [int(rank) for rank in ranks]
        return sorted(dict.fromkeys(ranks))

    def _dynamic_pp_rdt_prewarm_edges(self) -> list[tuple[int, int]]:
        ranks = self._dynamic_pp_rdt_prewarm_ranks()
        if len(ranks) < 2:
            return []
        mode = os.getenv("VLLM_DYNAMIC_PP_RDT_PREWARM_MODE",
                         "adjacent").lower()
        if mode == "all":
            return [(src, dst) for i, src in enumerate(ranks)
                    for dst in ranks[i + 1:]]
        if mode not in ("adjacent", "pipeline"):
            raise ValueError("Unsupported VLLM_DYNAMIC_PP_RDT_PREWARM_MODE="
                             f"{mode!r}; expected 'adjacent' or 'all'.")
        return list(zip(ranks, ranks[1:]))

    def _prewarm_dynamic_pp_rdt_edges(self) -> None:
        if not _dynamic_pp_rdt_prewarm_enabled():
            return
        if getattr(self, "_dynamic_pp_rdt_prewarmed", False):
            return
        if getattr(self, "_dynamic_pp_rdt_group", None) is None:
            return

        edges = self._dynamic_pp_rdt_prewarm_edges()
        if not edges:
            self._dynamic_pp_rdt_prewarmed = True
            return

        start = time.time()
        logger.info("Prewarming Ray RDT NCCL PP edges: %s", edges)
        for src_rank, dst_rank in edges:
            edge_start = time.time()
            src_ref = self.workers[src_rank].prewarm_ray_rdt_send.options(
                tensor_transport="nccl").remote(src_rank, dst_rank)
            dst_ref = self.workers[dst_rank].prewarm_ray_rdt_recv.remote(
                src_ref, src_rank, dst_rank)
            checksum = ray.get(dst_ref)
            logger.info(
                "Prewarmed Ray RDT NCCL PP edge %s->%s in %.3fs "
                "(checksum=%.1f)", src_rank, dst_rank,
                time.time() - edge_start, checksum)
        self._dynamic_pp_rdt_prewarmed = True
        logger.info("Finished Ray RDT NCCL PP prewarm for %d edges in %.3fs",
                    len(edges), time.time() - start)

    def _sync_request_states_before_target_chain(
        self,
        scheduler_output: DynamicSchedulerOutput,
        active_ranks: Optional[list[int]],
    ) -> None:
        dynamic_config = self.vllm_config.dynamic_config
        if not getattr(dynamic_config, "pipeline_autoscaling_enabled", False):
            return
        if not active_ranks:
            return
        current_active = getattr(self, "active_pp_ranks", None)
        if current_active is None:
            current_active = list(range(
                self.parallel_config.pipeline_parallel_size))

        # During no-drain autoscaling, scheduler_output.pp_layer_config can
        # route a target-config batch through newly activated ranks before
        # set_active_pp_ranks() commits the final routing set. Those ranks
        # have loaded weights/KV but may not have seen the original
        # scheduled_new_reqs, so import request states before their first
        # target-chain forward.
        pending_active = getattr(
            self, "_autoscaling_request_state_sync_pending", None)
        should_sync = pending_active == tuple(active_ranks)
        if not should_sync and not (set(active_ranks) - set(current_active)):
            return

        sync_key = tuple(active_ranks)
        if getattr(self, "_autoscaling_request_state_sync_key", None) == sync_key:
            return

        scheduler_output.autoscaling_request_state_sync = True
        self._autoscaling_request_state_sync_key = sync_key
        self._autoscaling_request_state_sync_pending = None
        logger.info(
            "Marked target PP actor chain %s to carry autoscaling request "
            "states in-band",
            active_ranks)

    def _active_ranks_for_scheduler_output(self, scheduler_output) -> list[int]:
        pp_layer_config = getattr(scheduler_output, "pp_layer_config", None)
        if pp_layer_config is None:
            return list(range(self.parallel_config.pipeline_parallel_size))
        return [
            rank for rank, (start, end) in enumerate(pp_layer_config)
            if end >= start
        ]

    def _annotate_pp_nccl_debug_metadata(
        self,
        scheduler_output: DynamicSchedulerOutput,
        active_ranks: list[int],
    ) -> None:
        seq = getattr(self, "_pp_nccl_seq_counter", 0)
        self._pp_nccl_seq_counter = seq + 1
        scheduler_output.pp_nccl_seq = seq
        scheduler_output.pp_nccl_active_ranks = tuple(active_ranks)
        scheduler_output.pp_nccl_config_fingerprint = "|".join(
            f"{start}-{end}" for start, end in scheduler_output.pp_layer_config)

        request_ids: list[str] = []
        for req in scheduler_output.scheduled_new_reqs:
            req_id = getattr(req, "request_id", None)
            if req_id is not None:
                request_ids.append(req_id)
        for req in scheduler_output.scheduled_cached_reqs:
            req_id = getattr(req, "req_id", None)
            if req_id is not None:
                request_ids.append(req_id)
        if not request_ids:
            request_ids.extend(scheduler_output.num_scheduled_tokens.keys())
        scheduler_output.pp_nccl_request_ids = tuple(request_ids)

    def _make_pp_nccl_intermediate_metadata(
        self,
        scheduler_output,
    ) -> PPNCCLIntermediateMetadata:
        hidden_size = getattr(self.vllm_config.model_config.hf_config,
                              "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.vllm_config.model_config,
                                  "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size for PP NCCL transport")
        dtype = self.vllm_config.model_config.dtype
        if not isinstance(dtype, torch.dtype):
            dtype = getattr(torch, str(dtype).removeprefix("torch."))
        dtype_name = str(dtype).removeprefix("torch.")
        shape = (scheduler_output.total_num_scheduled_tokens, hidden_size)
        return PPNCCLIntermediateMetadata(
            tensors={
                "hidden_states": (shape, dtype_name),
                "residual": (shape, dtype_name),
            },
            send_time=0.0,
            pp_nccl_seq=getattr(scheduler_output, "pp_nccl_seq", -1),
            pp_config_fingerprint=getattr(
                scheduler_output, "pp_nccl_config_fingerprint", ""),
            active_ranks=getattr(scheduler_output,
                                 "pp_nccl_active_ranks", ()) or (),
            request_ids=getattr(scheduler_output,
                                "pp_nccl_request_ids", ()) or (),
            total_num_scheduled_tokens=getattr(
                scheduler_output, "total_num_scheduled_tokens", 0),
        )

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

    def sync_add_layers(self, rank: int, layers_list: list[Tuple[int, int]]):
        """Synchronously add layers on the target rank.

        This is an explicit alias of `add_layers()` for call sites that need
        to distinguish it from `async_add_layers()`.
        """
        self.add_layers(rank, layers_list)
    
    def async_add_layers(self, rank: int, layers_list: list[Tuple[int, int]]):
        # fire-and-forget 异步发起，每个 worker 内部用线程执行
        self.collective_rpc("async_add_layers", args=(rank, layers_list))

    def wait_for_all_async_add_layers(self):
        """Wait for all async_add_layers to complete on all workers."""
        self.collective_rpc("wait_for_async_add_layers")
    
    def start_kv_cache_migration_async(
            self,
            pp_layer_config: list[Tuple[int, int]],
            src_to_plan: dict[int, dict[int, list[int]]],
            slot_mapping: Optional[list[int]],
            logical_num_blocks: Optional[int] = None):
        # 调用 worker 侧的同名方法，仅在 source_rank 上发送，其余 rank 不做事
        self.collective_rpc(
            "start_kv_cache_migration_async",
            args=(pp_layer_config, src_to_plan, slot_mapping,
                  logical_num_blocks))

    def get_is_kv_resizing_done(self) -> list[bool]:
        return self.collective_rpc("get_is_kv_resizing_done")

    def get_async_migration_state(self) -> list[dict[str, Any]]:
        start = time.time()
        states = self.collective_rpc("get_async_migration_state")
        logger.info("[autoscaling rpc timing] get_async_migration_state took %.3fs",
                    time.time() - start)
        return states

    def finalize_async_migration_after_sync(
        self,
        sender_list: list[int],
        receiver_list: list[int],
        total_migration_tokens: int,
        new_kv_cache_block_num: int,
    ) -> None:
        start = time.time()
        self.collective_rpc(
            "finalize_async_migration_after_sync",
            args=(sender_list, receiver_list, total_migration_tokens,
                  new_kv_cache_block_num))
        logger.info(
            "[autoscaling rpc timing] finalize_async_migration_after_sync took "
            "%.3fs, senders=%s receivers=%s total_tokens=%s new_kv_blocks=%s",
            time.time() - start, sender_list, receiver_list,
            total_migration_tokens, new_kv_cache_block_num)

    def finish_idle_async_kv_cache_transfer(
        self,
        sender_list: list[int],
    ) -> None:
        start = time.time()
        self.collective_rpc(
            "finish_idle_async_kv_cache_transfer",
            args=(sender_list,))
        logger.info(
            "[autoscaling rpc timing] finish_idle_async_kv_cache_transfer "
            "took %.3fs, senders=%s",
            time.time() - start, sender_list)

    def get_applied_token_num(self, receiver_list: list[int]) -> list[list[int]]:
        """Return KV patch buffer status per rank.
        """
        return self.collective_rpc("get_applied_token_num", args=(receiver_list,))

    def start_kv_cache_migration_sync(self, pp_layer_config: list[Tuple[int, int]], src_to_plan: dict[int, dict[int, list[Tuple[int, int]]]], 
                                       dst_to_layer_ids: dict[int, list[Tuple[int, int]]],
                                       slot_mapping: Optional[list[int]] = None):
        # 调用 worker 侧的同名方法，仅在 source_rank 上发送，其余 rank 不做事
        self.collective_rpc("start_kv_cache_migration_sync", args=(pp_layer_config, src_to_plan, dst_to_layer_ids, slot_mapping))

    def start_kv_cache_migration_async_fast(self, pp_layer_config: list[Tuple[int, int]], 
                                             src_to_plan: dict[int, dict[int, list[Tuple[int, int]]]], 
                                             dst_to_layer_ids: dict[int, list[Tuple[int, int]]],
                                             slot_mapping: Optional[list[int]] = None):
        """Start async_fast KV cache migration.
        
        This is similar to sync migration but used after async weight loading.
        It performs one-shot KV transfer without the continuous patch phase.
        """
        self.collective_rpc("start_kv_cache_migration_async_fast", args=(pp_layer_config, src_to_plan, dst_to_layer_ids, slot_mapping))

    def get_kv_buffer_status(self) -> list[KVBufferStatus]:
        output = self.collective_rpc("get_kv_buffer_status")
        return output

    def remove_layers(self, rank: int, layers_list: list[Tuple[int, int]]):
        self.collective_rpc("remove_layers", args=(rank, layers_list))

    def remove_layers_and_release_kv_cache_for_layers(
        self,
        rank: int,
        layers_list: list[Tuple[int, int]],
    ):
        self.collective_rpc(
            "remove_layers_and_release_kv_cache_for_layers",
            args=(rank, layers_list))

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

    def resize_kv_cache(self, new_length: int, ranks: Optional[list[int]] = None) -> None:
        if ranks is not None:
            ray_get_reqs = []
            for rank in ranks:
                ray_get_reqs.append(
                    self.workers[rank].execute_method.remote(
                        "resize_kv_cache", new_length))
            ray.get(ray_get_reqs)
            return
        self.collective_rpc("resize_kv_cache", args=(new_length,))

    def start_resize_kv_cache_async(self, new_length: int) -> None:
        self.collective_rpc("start_resize_kv_cache_async", args=(new_length,))

    def release_kv_cache_for_layers(self, rank: int, layers_list: list[Tuple[int, int]]):
        self.collective_rpc("release_kv_cache_for_layers", args=(rank, layers_list))

    def release_kv_cache(self) -> None:
        self.collective_rpc("release_kv_cache")

    def reinitialize_kv_cache(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        assert len(kv_cache_configs) == self.parallel_config.world_size, "kv_cache_configs must have the same length as world_size"
        self.collective_rpc("reinitialize_kv_cache", args=(kv_cache_configs,))

    def dynamic_initialize_from_config(self, kv_cache_configs: list[KVCacheConfig], num_blocks: int) -> None:
        self.collective_rpc("dynamic_initialize_from_config", args=(kv_cache_configs, num_blocks))

    def set_env_var(self, key: str, value: str) -> None:
        """Broadcast an environment variable update to all workers."""
        self.collective_rpc("set_env_var", args=(key, value))

    def set_log_stop_time(self, enabled: bool) -> None:
        """Enable or disable STOP_TIME logging on all workers.
        
        Used to disable logging during set_pp_config to avoid polluting
        migration metrics with initialization overhead.
        """
        self.collective_rpc("set_log_stop_time", args=(enabled,))

    def set_active_pp_ranks(self, active_ranks: Optional[list[int]]) -> None:
        """Set the active PP routing subset on all workers."""
        total_start = time.time()
        collective_start = time.time()
        self.collective_rpc("set_active_pp_ranks", args=(active_ranks,))
        logger.info("[autoscaling rpc timing] set_active_pp_ranks collective took "
                    "%.3fs for active_ranks=%s",
                    time.time() - collective_start, active_ranks)
        local_start = time.time()
        self._set_active_pp_ranks_local(active_ranks)
        logger.info("[autoscaling rpc timing] set_active_pp_ranks local took %.3fs",
                    time.time() - local_start)
        logger.info("[autoscaling rpc timing] set_active_pp_ranks total took %.3fs",
                    time.time() - total_start)

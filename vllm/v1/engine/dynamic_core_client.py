# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import queue
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from threading import Thread
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import msgspec
import zmq
import zmq.asyncio

from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils import (get_open_port, get_open_zmq_inproc_path,
                        get_open_zmq_ipc_path, get_tcp_uri, make_zmq_socket)
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.executor.abstract import Executor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr
from vllm.v1.utils import CoreEngineProcManager 
from .core_client import MPClient, BackgroundResources, CoreEngine
from .dynamic_core import DynamicEngineCoreProc
from .core_client import AnyFuture, _R
from vllm.dynamic_config import MigrationConfig

class DynamicMPClient(MPClient):
    """
    DynamicMPClient: base client for multi-proc DynamicEngineCore.
    We use this class to adapt MPClient to DynamicEngineCore.
    The Only difference here is that we use DynamicEngineCoreProc in __init__ 
    instead of EngineCoreProc.
    """
    def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        dynamic_config: MigrationConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        self.vllm_config = vllm_config
        self.dynamic_config = dynamic_config
        # Serialization setup.
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineCoreOutputs)

        # ZMQ setup.
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # This will ensure resources created so far are closed
        # when the client is garbage collected, even if an
        # exception is raised mid-construction.
        self.resources = BackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)
        success = False
        try:
            parallel_config = vllm_config.parallel_config
            local_engine_count = parallel_config.data_parallel_size_local
            start_index = parallel_config.data_parallel_rank
            local_start_index = parallel_config.data_parallel_rank_local

            # SPMD mode is where there is an LLM instance per DP rank and
            # one core engine per LLM, see
            # examples/offline_inference/data_parallel.py.
            spmd_mode = local_start_index is not None
            if spmd_mode:
                assert local_engine_count == 1
                self.core_engines = [
                    CoreEngine(index=local_start_index, local=True)
                ]
            else:
                assert start_index == 0
                local_start_index = 0
                self.core_engines = [
                    CoreEngine(index=i, local=(i < local_engine_count))
                    for i in range(parallel_config.data_parallel_size)
                ]

            input_address, output_address = self._get_zmq_addresses(
                parallel_config, spmd_mode)

            # Create input and output sockets.
            self.input_socket = self.resources.input_socket = make_zmq_socket(
                self.ctx, input_address, zmq.ROUTER, bind=True)

            self.resources.output_socket = make_zmq_socket(
                self.ctx, output_address, zmq.constants.PULL)
            # Start local engines.
            if local_engine_count:
                # In server mode, start_index and local_start_index will
                # both be 0.
                self.resources.local_engine_manager = CoreEngineProcManager(
                    DynamicEngineCoreProc.run_engine_core, 
                    vllm_config=vllm_config,
                    dynamic_config=dynamic_config,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    input_address=input_address,
                    on_head_node=True,
                    local_engine_count=local_engine_count,
                    start_index=start_index,
                    local_start_index=local_start_index)

            self.core_engine = self.core_engines[0]

            # Wait for engine core process(es) to start.
            self._wait_for_engine_startup(output_address, parallel_config)

            self.utility_results: dict[int, AnyFuture] = {}

            # Request objects which may contain pytorch-allocated tensors
            # that we need to keep references to until zmq is done with the
            # underlying data.
            self.pending_messages = deque[tuple[zmq.MessageTracker, Any]]()

            success = True
        finally:
            if not success:
                self._finalizer()

def _process_utility_output(output: UtilityOutput,
                            utility_results: dict[int, AnyFuture]):
    """Set the result from a utility method in the waiting future"""
    future = utility_results.pop(output.call_id)
    if output.failure_message is not None:
        future.set_exception(Exception(output.failure_message))
    else:
        future.set_result(output.result)


class DynamicAsyncMPClient(DynamicMPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    def __init__(self, vllm_config: VllmConfig, executor_class: type[Executor],
                 log_stats: bool, dynamic_config: MigrationConfig):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            dynamic_config=dynamic_config,
        )

        self.outputs_queue = asyncio.Queue[Union[EngineCoreOutputs,
                                                 Exception]]()
        try:
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            asyncio.get_running_loop()
            self._ensure_output_queue_task()
        except RuntimeError:
            pass

    def _ensure_output_queue_task(self):
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_handler: Optional[Callable[[AsyncMPClient, EngineCoreOutputs],
                                          Awaitable[None]]] = getattr(
                                              self.__class__,
                                              "process_engine_outputs", None)
        _self_ref = weakref.ref(self) if output_handler else None
        output_socket = resources.output_socket
        assert output_socket is not None

        async def process_outputs_socket():
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output,
                                                utility_results)
                        continue

                    if output_handler is not None:
                        assert _self_ref is not None
                        _self = _self_ref()
                        if not _self:
                            # Client has been garbage collected, abort.
                            return
                        await output_handler(_self, outputs)

                    if outputs.outputs or outputs.scheduler_stats:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask")

    async def get_output_async(self) -> EngineCoreOutputs:
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    def _send_input(self,
                    request_type: EngineCoreRequestType,
                    request: Any,
                    engine: Optional[CoreEngine] = None) -> Awaitable[Any]:
        self.ensure_alive()
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine, request)

    def _send_input_message(self, message: tuple[bytestr,
                                                 ...], engine: CoreEngine,
                            objects: Any) -> Awaitable[Any]:
        """
        objects is a reference to retain until zmq is finished with the
        buffers, in case they were extracted from tensors in the request.
        """
        self.ensure_alive()
        self.free_pending_messages()

        msg = (engine.identity, ) + message
        if not objects or len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            return self.input_socket.send_multipart(msg, copy=False)

        future: asyncio.Future[zmq.MessageTracker]
        future = self.input_socket.send_multipart(msg, copy=False, track=True)

        def add_pending(f: asyncio.Future[zmq.MessageTracker]):
            with contextlib.suppress(BaseException):
                self.add_pending_message(f.result(), objects)

        future.add_done_callback(add_pending)
        return future

    async def call_utility_async(self, method: str, *args) -> Any:
        return await self._call_utility_async(method,
                                              *args,
                                              engine=self.core_engine)

    async def _call_utility_async(self, method: str, *args,
                                  engine: CoreEngine) -> Any:
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        message = (EngineCoreRequestType.UTILITY.value, *self.encoder.encode(
            (call_id, method, args)))
        await self._send_input_message(message, engine, args)
        self._ensure_output_queue_task()
        return await future

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def profile_async(self, is_start: bool = True) -> None:
        await self.call_utility_async("profile", is_start)

    async def reset_mm_cache_async(self) -> None:
        await self.call_utility_async("reset_mm_cache")

    async def reset_prefix_cache_async(self) -> None:
        await self.call_utility_async("reset_prefix_cache")

    async def set_pp_config_async(
        self,
        pp_layer_config: list,
        alternative_configs: Optional[Dict] = None,
        migration_steps: Optional[list[int]] = None,
        migration_mode: Optional[str] = None,
        weight_loading_mode: Optional[str] = None,
        weight_chunk_size_mb: Optional[float] = None,
        fixed_num_gpu_blocks: Optional[int] = None
    ) -> None:
        await self.call_utility_async("set_pp_config", pp_layer_config,
                                      alternative_configs, migration_steps,
                                      migration_mode,
                                      weight_loading_mode,
                                      weight_chunk_size_mb,
                                      fixed_num_gpu_blocks)

    async def get_engine_state_async(self) -> dict[str, Any]:
        return await self.call_utility_async("get_engine_state")

    async def sleep_async(self, level: int = 1) -> None:
        await self.call_utility_async("sleep", level)

    async def wake_up_async(self, tags: Optional[list[str]] = None) -> None:
        await self.call_utility_async("wake_up", tags)

    async def is_sleeping_async(self) -> bool:
        return await self.call_utility_async("is_sleeping")

    async def execute_dummy_batch_async(self) -> None:
        await self.call_utility_async("execute_dummy_batch")

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        return await self.call_utility_async("add_lora", lora_request)

    async def remove_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("remove_lora", lora_id)

    async def list_loras_async(self) -> set[int]:
        return await self.call_utility_async("list_loras")

    async def pin_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("pin_lora", lora_id)

    async def save_sharded_state_async(self,
                                       path: str,
                                       pattern: Optional[str] = None,
                                       max_size: Optional[int] = None) -> None:
        await self.call_utility_async("save_sharded_state", path, pattern,
                                      max_size)

    async def collective_rpc_async(
            self,
            method: Union[str, Callable[..., _R]],
            timeout: Optional[float] = None,
            args: tuple = (),
            kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        return await self.call_utility_async("collective_rpc", method, timeout,
                                             args, kwargs)
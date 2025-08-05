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
from typing import Any, Callable, Optional, TypeVar, Union

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
        executor_class: type[Executor],
        log_stats: bool,
    ):
        self.vllm_config = vllm_config
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
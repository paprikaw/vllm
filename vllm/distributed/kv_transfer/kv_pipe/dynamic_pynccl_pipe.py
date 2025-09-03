# SPDX-License-Identifier: Apache-2.0
"""
    This module implements a PyNccl pipe for sending and receiving
    Optional[torch.Tensor] between distributed ranks with advanced
    communication features.

    Key Features:
    - Supports sending and receiving tensors with metadata
    - Handles both CUDA and CPU device communications
    - Implements a non-blocking tensor transfer mechanism
    - Manages buffer size and provides backpressure control
    - Supports distributed process groups with configurable parameters
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Union

import torch

from vllm.config import KVTransferConfig
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe

logger = init_logger(__name__)

Metadata = dict[str, Optional[Union[torch.Tensor, int, torch.dtype, torch.Size]]]


class DynamicPyNcclPipe(PyNcclPipe):

    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 device: Optional[str] = None,
                 port_offset: int = 0):
        super().__init__(local_rank, config, device, port_offset)
    def _make_metadata(self, tensor: Optional[torch.Tensor], rank: int) -> Metadata:
        """
        Copied from PyNcclPipe._make_metadata
        Only difference is that we add the rank to the metadata
        Create the metadata as a dictionary based on the input tensor.

        Args:
            tensor: The input tensor or None if no tensor is provided.

        Returns:
            metadata: A dictionary with the following keys:
                - "dtype": The data type of the tensor or None.
                - "shape": The shape of the tensor or None.
        """
        if tensor is None:
            return {"dtype": None, "shape": None, "rank": rank}
        else:
            return {"dtype": tensor.dtype, "shape": tensor.shape, "rank": rank}

    def _send_impl(self, tensor: Optional[torch.Tensor], source_rank: int, target_rank: int) -> None:
        """
        Copied from PyNcclPipe._send_impl
        Only difference is that we add the rank to the metadata
        The actual implementation of sending the tensor and its metadata to the
        target rank.

        Args:
            tensor: The input tensor to be sent, or `None` if no tensor is
                being sent.
        """
        metadata = self._make_metadata(tensor, source_rank)
        self._send_metadata(metadata)
        if tensor is not None:
            self.device_send_func(tensor.to(self.device),
                                  target_rank)


    def _recv_impl_with_callback(self, callback: Callable[[torch.Tensor], None]) -> Optional[torch.Tensor]:
        """
        The actual implementation of receiving a tensor and its metadata from
        the target rank.

        Returns:
            buffer: The received tensor, or `None` if no tensor is received.
        """
        metadata = self._recv_metadata()
        if metadata["dtype"] is None:
            return None
        buffer = self._prepare_recv_buffer(metadata)
        self.device_recv_func(buffer, self.target_rank_for_recv)
        callback(buffer)

    def send_tensor(self, tensor: Optional[torch.Tensor], source_rank: int, target_rank: int) -> None:
        """
        Copied from PyNcclPipe.send_tensor
        Only difference is that we add the source_rank and target_rank to the metadata
        Sends a tensor and its metadata to the destination rank in a
        non-blocking way.

        Args:
            tensor: The tensor to send, or `None` if no tensor is being sent.
        """
        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)

        if tensor is not None:
            tensor_size = tensor.element_size() * tensor.numel()
        else:
            tensor_size = 0

        self.block_if_full()

        with self.buffer_size_lock:
            self.buffer_size += tensor_size

        self.transport_thread.submit(self.send_tensor_wrapper, tensor,
                                     tensor_size, source_rank, target_rank)

    def send_tensor_wrapper(self, tensor: Optional[torch.Tensor],
                            tensor_size: int, source_rank: int, target_rank: int) -> None:
        """
        Copied from PyNcclPipe.send_tensor_wrapper
        Wrapper for _send_impl to handle exceptions and update buffer size.
        """
        try:
            self._send_impl(tensor, source_rank, target_rank)

            with self.buffer_size_lock:
                self.buffer_size -= tensor_size
        except Exception as e:
            logger.error("[rank%d]: Exception when trying to send %s, msg: %s",
                         torch.distributed.get_rank(), str(tensor), str(e))
            import traceback
            traceback.print_exc()


    def recv_tensor_with_callback(self, callback: Callable[[torch.Tensor], None]) -> None:
        """
        Receives a tensor and its metadata from the source rank. Blocking call.

        Args:
            tensor: The received tensor, or `None` if no tensor is received.
        """
        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)

        self._recv_impl_with_callback(callback)
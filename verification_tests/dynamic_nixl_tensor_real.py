#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-process smoke for DirectNixlTensorTransport.

This validates the transport with the real NIXL Python package, separate
processes, and optionally CUDA tensors.  It keeps the control plane tiny by
using multiprocessing queues that mimic PairPipe.signal_group.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass
class QueueSignalGroup:
    rank: int
    queues: list

    def send_obj(self, obj, dst: int):
        self.queues[dst].put(obj)

    def recv_obj(self, src: int):
        return self.queues[self.rank].get(timeout=30)


@dataclass
class QueuePipe:
    peer_rank: int
    signal_group: QueueSignalGroup


def _wait_for_ready(ready_q, count: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    ready = 0
    while ready < count and time.monotonic() < deadline:
        try:
            item = ready_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if item != "ready":
            raise RuntimeError(f"Unexpected readiness item: {item!r}")
        ready += 1
    if ready != count:
        raise TimeoutError(f"Only {ready}/{count} processes became ready")


def _sender(args, signal_queues, ready_q, result_q):
    if args.sender_cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.sender_cuda_visible_devices

    import torch

    DirectNixlTensorTransport = _load_transport_module(
        args.root_dir).DirectNixlTensorTransport

    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    transport = DirectNixlTensorTransport(rank=0,
                                          local_rank=0,
                                          device=device,
                                          rank_to_ip={
                                              0: args.ip,
                                              1: args.ip,
                                          },
                                          base_port=args.base_port,
                                          timeout_s=args.timeout)
    pipe = QueuePipe(peer_rank=1,
                     signal_group=QueueSignalGroup(rank=0,
                                                   queues=signal_queues))
    tensor = torch.arange(args.numel, dtype=torch.float32,
                          device=device).reshape(args.rows, args.cols)
    ready_q.put("ready")
    transport.send_tensor(1, tensor, pipe)
    result_q.put(("sender", float(tensor.sum().item())))
    if args.quick_exit:
        result_q.close()
        result_q.join_thread()
        os._exit(0)


def _receiver(args, signal_queues, ready_q, result_q):
    if args.receiver_cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.receiver_cuda_visible_devices

    import torch

    DirectNixlTensorTransport = _load_transport_module(
        args.root_dir).DirectNixlTensorTransport

    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    transport = DirectNixlTensorTransport(rank=1,
                                          local_rank=0,
                                          device=device,
                                          rank_to_ip={
                                              0: args.ip,
                                              1: args.ip,
                                          },
                                          base_port=args.base_port,
                                          timeout_s=args.timeout)
    pipe = QueuePipe(peer_rank=0,
                     signal_group=QueueSignalGroup(rank=1,
                                                   queues=signal_queues))
    ready_q.put("ready")
    tensor = transport.recv_tensor(0, torch.float32,
                                   torch.Size([args.rows, args.cols]), pipe)
    expected = torch.arange(args.numel, dtype=torch.float32,
                            device=device).reshape(args.rows, args.cols)
    if not torch.equal(tensor, expected):
        max_diff = float((tensor - expected).abs().max().item())
        raise RuntimeError(f"received tensor mismatch, max_diff={max_diff}")
    result_q.put(("receiver", float(tensor.sum().item())))
    if args.quick_exit:
        result_q.close()
        result_q.join_thread()
        os._exit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir",
                        default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=32000)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--sender-cuda-visible-devices")
    parser.add_argument("--receiver-cuda-visible-devices")
    parser.add_argument("--quick-exit", action="store_true")
    args = parser.parse_args()
    args.numel = args.rows * args.cols
    return args


def _load_transport_module(root_dir: str):
    """Load the transport file without importing the full vllm package.

    The A100 NIXL smoke environment can have a newer transformers package than
    this checkout expects, and importing vllm.__init__ may fail before the
    transport is exercised.  This verifier needs only vllm.logger, so provide a
    tiny logger stub and load the target file directly.
    """
    logger_mod = ModuleType("vllm.logger")

    def init_logger(name):
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s %(name)s: %(message)s")
        return logging.getLogger(name)

    logger_mod.init_logger = init_logger
    vllm_mod = ModuleType("vllm")
    vllm_mod.__path__ = [str(Path(root_dir) / "vllm")]
    sys.modules.setdefault("vllm", vllm_mod)
    sys.modules["vllm.logger"] = logger_mod

    module_path = (Path(root_dir) / "vllm" / "distributed" / "kv_transfer" /
                   "kv_connector" / "dynamic_nixl_tensor.py")
    spec = importlib.util.spec_from_file_location(
        "dynamic_nixl_tensor_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load transport module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    ctx = mp.get_context("spawn")
    signal_queues = [ctx.Queue(), ctx.Queue()]
    ready_q = ctx.Queue()
    result_q = ctx.Queue()
    sender = ctx.Process(target=_sender,
                         args=(args, signal_queues, ready_q, result_q),
                         name="direct-nixl-sender")
    receiver = ctx.Process(target=_receiver,
                           args=(args, signal_queues, ready_q, result_q),
                           name="direct-nixl-receiver")
    receiver.start()
    sender.start()
    _wait_for_ready(ready_q, 2, args.timeout)
    sender.join(args.timeout)
    receiver.join(args.timeout)
    if sender.is_alive() or receiver.is_alive():
        sender.terminate()
        receiver.terminate()
        raise TimeoutError("direct NIXL smoke processes did not exit")
    if sender.exitcode != 0 or receiver.exitcode != 0:
        raise RuntimeError(
            "direct NIXL smoke failed: "
            f"sender={sender.exitcode} receiver={receiver.exitcode}")

    results = dict(result_q.get(timeout=2) for _ in range(2))
    if results["sender"] != results["receiver"]:
        raise RuntimeError(f"checksum mismatch: {results}")
    print(f"direct NIXL tensor smoke passed: checksum={results['receiver']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

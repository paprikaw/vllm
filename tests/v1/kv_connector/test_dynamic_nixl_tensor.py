# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from dataclasses import dataclass

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector import dynamic_nixl_tensor
from vllm.distributed.kv_transfer.kv_connector.dynamic_nixl_tensor import (
    DirectNixlTensorTransport,
    NixlRemoteTensor,
)


class FakeNixlConfig:

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeNixlAgent:
    agents: dict[str, "FakeNixlAgent"] = {}
    descs: dict[str, list[torch.Tensor]] = {}
    current_registered: dict[str, set[int]] = {}
    desc_counter = 0
    fail_initialize_once_for: set[str] = set()
    require_descriptor_metadata = True

    def __init__(self, name: str, config: FakeNixlConfig):
        self.name = name
        self.config = config
        self.remote_metadata: set[str] = set()
        self.remote_desc_metadata: dict[str, set[int]] = {}
        self.removed_agents: list[str] = []
        self.done_tags: set[tuple[str, bytes]] = set()
        self.handles: dict[str, dict] = {}
        self.register_count = 0
        FakeNixlAgent.agents[name] = self

    def get_plugin_list(self):
        return ["fake"]

    def send_local_metadata(self, host: str, port: int):
        self.last_sent_metadata = (host, port)

    def fetch_remote_metadata(self, name: str, host: str, port: int):
        if name in FakeNixlAgent.agents:
            self.remote_metadata.add(name)
            self.remote_desc_metadata.setdefault(name, set()).update(
                FakeNixlAgent.current_registered.get(name, set()))

    def check_remote_metadata(self, name: str, descs=None):
        return name in self.remote_metadata

    def remove_remote_agent(self, name: str):
        self.remote_metadata.discard(name)
        self.remote_desc_metadata.pop(name, None)
        self.removed_agents.append(name)

    def register_memory(self, tensor: torch.Tensor):
        self.register_count += 1
        FakeNixlAgent.current_registered.setdefault(self.name, set()).add(
            id(tensor))
        return ("reg", id(tensor))

    def deregister_memory(self, reg_descs):
        FakeNixlAgent.current_registered.setdefault(self.name, set()).discard(
            reg_descs[1])

    def get_xfer_descs(self, tensors: list[torch.Tensor]):
        return list(tensors)

    def get_serialized_descs(self, descs):
        FakeNixlAgent.desc_counter += 1
        token = f"desc-{FakeNixlAgent.desc_counter}"
        FakeNixlAgent.descs[token] = list(descs)
        return token.encode("ascii")

    def deserialize_descs(self, payload: bytes):
        return FakeNixlAgent.descs[payload.decode("ascii")]

    def initialize_xfer(self, op, local_descs, remote_descs, peer_name, tag):
        assert op == "READ"
        fail_key = f"{self.name}->{peer_name}"
        if fail_key in FakeNixlAgent.fail_initialize_once_for:
            FakeNixlAgent.fail_initialize_once_for.remove(fail_key)
            raise RuntimeError("NIXL_ERR_NOT_FOUND")
        if FakeNixlAgent.require_descriptor_metadata:
            known_descs = self.remote_desc_metadata.get(peer_name, set())
            if not all(id(remote) in known_descs for remote in remote_descs):
                raise RuntimeError("NIXL_ERR_NOT_FOUND")
        handle = f"handle-{len(self.handles) + 1}"
        self.handles[handle] = {
            "local": local_descs,
            "remote": remote_descs,
            "peer": peer_name,
            "tag": tag,
            "state": "INIT",
        }
        return handle

    def transfer(self, handle):
        entry = self.handles[handle]
        for local, remote in zip(entry["local"], entry["remote"]):
            local.copy_(remote)
        entry["state"] = "DONE"
        FakeNixlAgent.agents[entry["peer"]].done_tags.add((self.name,
                                                           entry["tag"]))
        return "DONE"

    def check_xfer_state(self, handle):
        return self.handles[handle]["state"]

    def release_xfer_handle(self, handle):
        self.handles.pop(handle, None)

    def check_remote_xfer_done(self, peer_name, tag, tag_is_prefix=False):
        if tag_is_prefix:
            return any(remote == peer_name and done_tag.startswith(tag)
                       for remote, done_tag in self.done_tags)
        return (peer_name, tag) in self.done_tags


@pytest.fixture(autouse=True)
def fake_nixl(monkeypatch):
    FakeNixlAgent.agents.clear()
    FakeNixlAgent.descs.clear()
    FakeNixlAgent.current_registered.clear()
    FakeNixlAgent.desc_counter = 0
    FakeNixlAgent.fail_initialize_once_for.clear()
    FakeNixlAgent.require_descriptor_metadata = True
    monkeypatch.setattr(dynamic_nixl_tensor, "_load_nixl_runtime",
                        lambda: (FakeNixlAgent, FakeNixlConfig))


class FakeSignalGroup:

    def __init__(self, rank: int, queues: list[queue.Queue]):
        self.rank = rank
        self.queues = queues

    def send_obj(self, obj, dst: int):
        self.queues[dst].put(obj)

    def recv_obj(self, src: int):
        return self.queues[self.rank].get(timeout=2)


@dataclass
class FakePipe:
    peer_rank: int
    signal_group: FakeSignalGroup


def _transport(rank: int) -> DirectNixlTensorTransport:
    return DirectNixlTensorTransport(rank=rank,
                                     local_rank=rank,
                                     device=torch.device("cpu"),
                                     rank_to_ip={
                                         0: "127.0.0.1",
                                         1: "127.0.0.1",
                                     },
                                     base_port=62000,
                                     timeout_s=1.0)


def _pp_transport(rank: int) -> DirectNixlTensorTransport:
    return DirectNixlTensorTransport(rank=rank,
                                     local_rank=rank,
                                     device=torch.device("cpu"),
                                     rank_to_ip={
                                         0: "127.0.0.1",
                                         1: "127.0.0.1",
                                     },
                                     base_port=63000,
                                     timeout_s=1.0,
                                     agent_prefix="vllm-dynamic-pp-rank")


def test_kv_tensor_read_over_signal_descriptor_protocol():
    sender = _transport(0)
    receiver = _transport(1)
    queues = [queue.Queue(), queue.Queue()]
    send_pipe = FakePipe(peer_rank=1,
                         signal_group=FakeSignalGroup(rank=0, queues=queues))
    recv_pipe = FakePipe(peer_rank=0,
                         signal_group=FakeSignalGroup(rank=1, queues=queues))
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    result: dict[str, torch.Tensor] = {}

    send_thread = threading.Thread(target=sender.send_tensor,
                                   args=(1, source, send_pipe))
    recv_thread = threading.Thread(
        target=lambda: result.setdefault(
            "tensor",
            receiver.recv_tensor(0, source.dtype, source.shape, recv_pipe)))

    send_thread.start()
    recv_thread.start()
    send_thread.join(timeout=2)
    recv_thread.join(timeout=2)

    assert not send_thread.is_alive()
    assert not recv_thread.is_alive()
    assert torch.equal(result["tensor"], source)


def test_pp_remote_tensor_read_uses_metadata_endpoint_and_refreshes_peer():
    source_transport = _transport(0)
    reader = _transport(1)
    source = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)

    remote = source_transport.prepare_read_source(1, source)
    out = reader.read_remote_tensor(peer_rank=0,
                                    remote_tensor=remote,
                                    dtype=source.dtype,
                                    shape=source.shape)

    assert torch.equal(out, source)

    reader.ensure_peer_at(0, "127.0.0.1", 62000)
    reader.ensure_peer_at(0, "127.0.0.2", 62000)
    assert FakeNixlAgent.agents[reader.agent_name].removed_agents == [
        "vllm-dynamic-kv-rank-0"
    ]


def test_pp_remote_tensor_read_supports_custom_agent_prefix_and_retry():
    source_transport = _pp_transport(0)
    reader = _pp_transport(1)
    source = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64)
    remote = source_transport.prepare_read_source(1, source)

    assert remote.agent_name == "vllm-dynamic-pp-rank-0"

    FakeNixlAgent.fail_initialize_once_for.add(
        "vllm-dynamic-pp-rank-1->vllm-dynamic-pp-rank-0")
    out = reader.read_remote_tensor(peer_rank=0,
                                    remote_tensor=remote,
                                    dtype=source.dtype,
                                    shape=source.shape)

    assert torch.equal(out, source)
    assert FakeNixlAgent.agents[reader.agent_name].removed_agents == [
        "vllm-dynamic-pp-rank-0"
    ]


def test_remote_tensor_metadata_mismatch_fails_before_copy():
    source_transport = _transport(0)
    reader = _transport(1)
    source = torch.ones((2, 2), dtype=torch.float32)
    remote = source_transport.prepare_read_source(1, source)
    bad_remote = NixlRemoteTensor(xfer_id=remote.xfer_id,
                                  agent_name=remote.agent_name,
                                  host=remote.host,
                                  port=remote.port,
                                  desc_bytes=remote.desc_bytes,
                                  shape=(2, 3),
                                  dtype=remote.dtype)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        reader.read_remote_tensor(peer_rank=0,
                                  remote_tensor=bad_remote,
                                  dtype=source.dtype,
                                  shape=source.shape)

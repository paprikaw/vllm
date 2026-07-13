import json
import asyncio
import sys
from types import ModuleType

import pytest


fake_vmm_ops = ModuleType("kvcached.vmm_ops")
fake_vmm_ops.kv_tensors_created = lambda **kwargs: True
fake_vmm_ops.map_to_kv_tensors = lambda offsets, **kwargs: True
fake_vmm_ops.unmap_from_kv_tensors = lambda offsets, **kwargs: True
sys.modules.setdefault("kvcached.vmm_ops", fake_vmm_ops)

from kvcached import tp_ipc_util


@pytest.fixture(autouse=True)
def reset_worker_mapping_state():
    with tp_ipc_util._WORKER_MAPPING_LOCK:
        tp_ipc_util._WORKER_OFFSET_FILTER = None
        tp_ipc_util._WORKER_MAPPED_OFFSETS.clear()
    yield
    with tp_ipc_util._WORKER_MAPPING_LOCK:
        tp_ipc_util._WORKER_OFFSET_FILTER = None
        tp_ipc_util._WORKER_MAPPED_OFFSETS.clear()


def test_worker_tcp_address_uses_pp_rank_mapping(monkeypatch):
    monkeypatch.setenv("KVCACHED_PP_RANK_TO_IP", json.dumps({
        0: "10.0.0.1",
        1: "10.0.0.2",
    }))
    monkeypatch.setenv("KVCACHED_WORKER_IPC_PORT_BASE", "21000")
    monkeypatch.setenv("KVCACHED_WORKER_IPC_PORT_STRIDE", "8")

    assert tp_ipc_util.get_worker_tcp_address(0, 0) == ("10.0.0.1", 21000)
    assert tp_ipc_util.get_worker_tcp_address(3, 1) == ("10.0.0.2", 21011)


def test_worker_tcp_address_requires_every_pp_rank(monkeypatch):
    monkeypatch.setenv("KVCACHED_PP_RANK_TO_IP", '{"0": "10.0.0.1"}')
    monkeypatch.setenv("KVCACHED_WORKER_IPC_PORT_BASE", "21000")

    with pytest.raises(RuntimeError, match="PP rank 1"):
        tp_ipc_util.get_worker_tcp_address(0, 1)


def test_worker_ipc_transport_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("KVCACHED_WORKER_IPC_TRANSPORT", "rdma")

    with pytest.raises(ValueError, match="expected 'unix' or 'tcp'"):
        tp_ipc_util._use_tcp_worker_ipc()


def test_broadcast_map_rolls_back_successful_workers(monkeypatch):
    monkeypatch.setenv("KVCACHED_NUM_PP_STAGES", "2")
    calls = []

    async def fake_send(rank, message, pp_rank=0):
        calls.append((pp_rank, message["cmd"]))
        if message["cmd"] == "map_to_kv_tensors" and pp_rank == 1:
            return {"status": "error", "message": "out of memory"}
        return {"status": "success"}

    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", fake_send)

    with pytest.raises(RuntimeError, match="pp1/tp0 failed to map"):
        asyncio.run(tp_ipc_util._broadcast_map_to_kv_tensors(
            1, [0, 1024], pp_rank=-1, group_id=0))

    assert calls == [
        (0, "map_to_kv_tensors"),
        (1, "map_to_kv_tensors"),
        (0, "unmap_from_kv_tensors"),
    ]


def test_worker_map_filters_offsets_to_owned_layer_groups(monkeypatch):
    mapped = []
    monkeypatch.setattr(
        tp_ipc_util,
        "map_to_kv_tensors",
        lambda offsets, **kwargs: mapped.append(list(offsets)) or True,
    )
    tp_ipc_util.register_worker_offset_filter(
        lambda offsets, group_id: [offset for offset in offsets
                                   if offset < 200])

    assert tp_ipc_util._map_worker_offsets(
        [300, 100, 0, 200], group_id=0, apply_filter=True)

    assert mapped == [[0, 100]]
    assert tp_ipc_util._WORKER_MAPPED_OFFSETS[0] == {0, 100}


def test_worker_unmap_uses_recorded_offsets_after_ownership_changes(
        monkeypatch):
    unmapped = []
    monkeypatch.setattr(tp_ipc_util, "map_to_kv_tensors",
                        lambda offsets, **kwargs: True)
    monkeypatch.setattr(
        tp_ipc_util,
        "unmap_from_kv_tensors",
        lambda offsets, **kwargs: unmapped.append(list(offsets)) or True,
    )
    monkeypatch.setattr(tp_ipc_util, "_synchronize_cuda_device", lambda: None)
    tp_ipc_util.register_worker_offset_filter(
        lambda offsets, group_id: [offset for offset in offsets
                                   if offset < 200])
    assert tp_ipc_util._map_worker_offsets(
        [0, 100, 200, 300], group_id=0, apply_filter=True)
    tp_ipc_util.register_worker_offset_filter(
        lambda offsets, group_id: [offset for offset in offsets
                                   if offset >= 200])

    assert tp_ipc_util._unmap_worker_offsets(
        [0, 100, 200, 300], group_id=0)

    assert unmapped == [[0, 100]]
    assert tp_ipc_util._WORKER_MAPPED_OFFSETS[0] == set()


def test_migration_map_bypasses_current_ownership_filter(monkeypatch):
    mapped = []
    monkeypatch.setattr(
        tp_ipc_util,
        "map_to_kv_tensors",
        lambda offsets, **kwargs: mapped.append(list(offsets)) or True,
    )
    tp_ipc_util.register_worker_offset_filter(
        lambda offsets, group_id: [offset for offset in offsets
                                   if offset < 200])

    assert tp_ipc_util.map_worker_offsets_for_migration(
        [200, 300], group_id=0)

    assert mapped == [[200, 300]]
    assert tp_ipc_util._WORKER_MAPPED_OFFSETS[0] == {200, 300}


def test_failed_worker_map_does_not_update_mapping_registry(monkeypatch):
    monkeypatch.setattr(tp_ipc_util, "map_to_kv_tensors",
                        lambda offsets, **kwargs: False)

    assert not tp_ipc_util._map_worker_offsets(
        [0, 100], group_id=0, apply_filter=False)

    assert tp_ipc_util._WORKER_MAPPED_OFFSETS[0] == set()

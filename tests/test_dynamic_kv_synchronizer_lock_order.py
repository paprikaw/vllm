import threading
from types import SimpleNamespace

import torch

from vllm.distributed.kv_transfer.kv_connector.dynamic_kv_synchronizer import (
    DynamicKVSynchronizer,
    _kv_transfer_signal,
)
from vllm.distributed.kv_transfer.kv_connector.dynamic_utils import (
    KVPatch,
    KVPatchMeta,
)


class _SignalGroup:
    def __init__(self, responses):
        self._responses = iter(responses)

    def recv_obj(self, src):
        del src
        return next(self._responses)


def test_patch_sender_holds_local_nccl_lock_before_handshake_and_data():
    meta = KVPatchMeta(
        type="kv_patch_finished",
        id=0,
        layer_ids=[4, 5, 6, 7],
        num_tokens=1,
        slot_mapping_dtype=torch.int64,
        slot_mapping_shape=torch.Size([1]),
        kv_payload_dtype=torch.float16,
        kv_payload_shape=torch.Size([2, 4, 1, 1, 1]),
    )
    local_lock = threading.Lock()
    pipe = SimpleNamespace(
        _nccl_lock=local_lock,
        peer_rank=1,
        signal_group=_SignalGroup([
            _kv_transfer_signal("REJECT", meta),
            _kv_transfer_signal("ACCEPT", meta),
        ]),
    )
    sync = object.__new__(DynamicKVSynchronizer)
    sync.kv_cache_transfer_in_process = {1: True}
    sync._pair_pipes_send = {1: pipe}
    sync._nccl_lock = None
    sync._ensure_pipe_and_buffer = lambda rank, direction: pipe
    events = []

    def assert_locked(event):
        assert local_lock.locked()
        events.append(event)

    sync._send_meta_to_rank = lambda rank, value: assert_locked("meta")
    sync._send_slot_mapping_to_rank = (
        lambda rank, value, **kwargs: assert_locked("slots"))
    sync._send_data_to_rank = (
        lambda rank, value, **kwargs: assert_locked("payload"))
    sync._wait_kv_patch_applied_ack = (
        lambda rank, patch_id: events.append("ack"))

    sync.send_kv_patch_to_rank(
        1,
        KVPatch(
            meta,
            torch.zeros(2, 4, 1, 1, 1, dtype=torch.float16),
            torch.zeros(1, dtype=torch.int64),
        ),
    )

    assert events == ["meta", "meta", "slots", "payload", "ack"]
    assert not local_lock.locked()
    assert not sync.kv_cache_transfer_in_process[1]

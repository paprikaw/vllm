from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm import kvcached_integration


def _runner(layer_ids: range) -> SimpleNamespace:
    layer_names = [
        f"model.layers.{layer_id}.self_attn.attn"
        for layer_id in layer_ids
    ]
    forward_context = {name: object() for name in layer_names}
    return SimpleNamespace(
        _kvcached_debug_geometry={
            "layer_group_layout": True,
            "layer_group_granularity": 4,
            "group_total_span": 1024,
            "group_id": 0,
        },
        _kvcached_layer_names=layer_names,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context=forward_context)),
    )


def test_owned_groups_follow_current_pp_layers():
    assert kvcached_integration._kvcached_owned_group_indices(
        _runner(range(36))) == set(range(9))


def test_owned_groups_reject_partial_stacking_group():
    with pytest.raises(RuntimeError, match="incomplete_groups"):
        kvcached_integration._kvcached_owned_group_indices(
            _runner(range(35)))


def test_release_unmaps_only_groups_no_longer_owned(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "kvcached.tp_ipc_util.unmap_worker_offsets_for_layer_groups",
        lambda groups, span, group_id=0: calls.append(
            (set(groups), span, group_id)) or True,
    )
    runner = _runner(range(32))

    kvcached_integration.release_kvcached_layer_groups(
        runner, set(range(32, 36)))

    assert calls == [({8}, 1024, 0)]


def test_pre_unmap_uses_forward_then_nccl_lock_order(monkeypatch):
    events = []
    callbacks = {}

    @contextmanager
    def recorded(name):
        events.append(f"enter:{name}")
        try:
            yield
        finally:
            events.append(f"exit:{name}")

    monkeypatch.setattr(
        "kvcached.tp_ipc_util.register_post_map_callback",
        lambda callback: callbacks.setdefault("post", callback),
    )
    monkeypatch.setattr(
        "kvcached.tp_ipc_util.register_pre_unmap_callback",
        lambda callback: callbacks.setdefault("pre", callback),
    )
    monkeypatch.setattr(
        "kvcached.tp_ipc_util.register_worker_offset_filter",
        lambda callback: callbacks.setdefault("filter", callback),
    )
    synchronizer = SimpleNamespace(
        drop_slots_for_kvcached_unmap_offsets=(
            lambda offsets, **kwargs: events.append("drop")),
        kvcached_page_lifetime_context=lambda: recorded("page"),
        _kvcached_pre_unmap_context=recorded("forward"),
        get_nccl_lock=lambda: recorded("nccl"),
    )

    kvcached_integration._register_kvcached_migration_unmap_hook(
        _runner(range(36)), synchronizer)
    with callbacks["pre"]([0], 0):
        events.append("unmap")

    assert events == [
        "enter:forward",
        "enter:page",
        "drop",
        "enter:nccl",
        "unmap",
        "exit:nccl",
        "exit:page",
        "exit:forward",
    ]

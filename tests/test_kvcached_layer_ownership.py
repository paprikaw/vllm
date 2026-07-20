from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
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


def test_register_migrated_layer_group_is_atomic():
    runner = _runner(range(20, 40))
    for layer_id in range(12, 20):
        runner.vllm_config.compilation_config.static_forward_context[
            f"model.layers.{layer_id}.self_attn.attn"] = object()
    kvcached_integration.remember_kvcached_layer_names(
        runner, list(range(12, 20)))

    assert kvcached_integration._kvcached_owned_group_indices(
        runner) == set(range(3, 10))


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


def test_release_rebinds_active_views_before_unmap(monkeypatch):
    events = []

    class RecordedLock:

        def __enter__(self):
            events.append("lock_enter")

        def __exit__(self, exc_type, exc, tb):
            events.append("lock_exit")

    runner = _runner(range(32))
    runner._kvcached_kv_tensors = [
        SimpleNamespace(shape=(2, 44872, 32, 8, 128))
    ]
    runner._kvcached_dynamic_kv_synchronizer = object()
    runner.forward_lock = RecordedLock()
    monkeypatch.setattr(
        kvcached_integration,
        "_rebind_kvcached_tensor_views",
        lambda runner, synchronizer, blocks: events.append(
            ("rebind", blocks)),
    )
    monkeypatch.setattr(
        "kvcached.tp_ipc_util.unmap_worker_offsets_for_layer_groups",
        lambda groups, span, group_id=0: events.append(
            ("unmap", set(groups), span, group_id)) or True,
    )

    kvcached_integration.release_kvcached_layer_groups(
        runner, set(range(32, 36)))

    assert events == [
        "lock_enter",
        ("rebind", 44872),
        "lock_exit",
        ("unmap", {8}, 1024, 0),
    ]


def test_deferred_release_retries_on_failure(monkeypatch):
    runner = _runner(range(32))
    calls = []
    kvcached_integration.defer_kvcached_layer_groups(
        runner, set(range(32, 36)))
    monkeypatch.setattr(
        kvcached_integration,
        "release_kvcached_layer_groups",
        lambda runner, layers: calls.append(set(layers))
        or (_ for _ in ()).throw(RuntimeError("unmap failed")),
    )

    with pytest.raises(RuntimeError, match="unmap failed"):
        kvcached_integration.release_deferred_kvcached_layer_groups(runner)

    assert calls == [set(range(32, 36))]
    assert runner._kvcached_deferred_release_layers == set(range(32, 36))


def test_autoscaling_state_restore_maps_all_live_blocks(monkeypatch):
    from vllm.v1.worker import dynamic_gpu_model_runner as runner_module

    calls = []
    block_table = SimpleNamespace(
        num_blocks_per_row=np.array([3, 2], dtype=np.int32),
        block_table_np=np.array([
            [3, 2, 1, -1],
            [9, 8, -1, -1],
        ], dtype=np.int32),
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            block_table=[block_table],
            num_reqs=2,
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(block_size=32))
            ]),
        model=SimpleNamespace(get_sched_layers=lambda: (12, 24)),
    )
    scheduler_output = SimpleNamespace(
        autoscaling_request_state_sync=True,
        scheduler_step_id=17,
        current_scheduler_output_version=23,
        scheduler_request_free_epoch=5,
        scheduler_block_free_epoch=7,
    )
    monkeypatch.setattr(runner_module, "use_kvcached_backend", lambda: True)
    monkeypatch.setattr(runner_module, "get_pp_group",
                        lambda: SimpleNamespace(rank=1))
    monkeypatch.setattr(
        runner_module,
        "guard_kvcached_vmm_slots_mapped",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    method = getattr(
        runner_module.DynamicGPUModelRunner,
        "_ensure_autoscaling_live_kvcached_blocks_mapped")
    method(runner, scheduler_output)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "autoscaling_restored_live_blocks"
    assert args[2] == [32, 64, 96, 256, 288]
    assert args[3] == 32
    assert kwargs["layer_ids"] == list(range(12, 24))
    assert kwargs["rank"] == 1
    assert kwargs["group_id"] == 0
    assert kwargs["ensure_mapped"] is True
    assert kwargs["trace_info"] == {
        "scheduler_step_id": 17,
        "scheduler_output_version": 23,
        "scheduler_request_free_epoch": 5,
        "scheduler_block_free_epoch": 7,
    }


def test_dynamic_scheduler_conversion_preserves_autoscaling_handoff():
    from vllm.v1.core.sched.dynamic_scheduler import (
        create_from_dynamic_scheduler_output)

    dynamic_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
        kv_connector_metadata=None,
        scheduler_step_id=17,
        current_scheduler_output_version=23,
        scheduler_request_free_epoch=5,
        scheduler_block_free_epoch=7,
        autoscaling_request_state_sync=True,
    )

    scheduler_output = create_from_dynamic_scheduler_output(dynamic_output)

    assert scheduler_output.scheduler_step_id == 17
    assert scheduler_output.current_scheduler_output_version == 23
    assert scheduler_output.scheduler_request_free_epoch == 5
    assert scheduler_output.scheduler_block_free_epoch == 7
    assert scheduler_output.autoscaling_request_state_sync is True


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
        "kvcached.tp_ipc_util.register_pre_unmap_drain_callback",
        lambda callback: callbacks.setdefault("drain", callback),
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
    with callbacks["drain"]([0], 0):
        events.append("sync")
        with callbacks["pre"]([0], 0):
            events.append("unmap")

    assert events == [
        "enter:forward",
        "sync",
        "enter:page",
        "drop",
        "enter:nccl",
        "unmap",
        "exit:nccl",
        "exit:page",
        "exit:forward",
    ]

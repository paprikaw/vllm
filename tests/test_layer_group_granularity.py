# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types

import pytest
import torch

from kvcached.utils import PAGE_SIZE


class _FakeInternalPage:

    @staticmethod
    def get_num_blocks(page_size, block_mem_size):
        return page_size // block_mem_size


class _FakePageAllocator:

    init_args = None
    init_kwargs = None

    def __init__(self, *args, **kwargs):
        type(self).init_args = args
        type(self).init_kwargs = kwargs

    def set_should_use_worker_ipc_callback(self, callback):
        pass

    def start_prealloc_thread(self):
        pass

    def stop_prealloc_thread(self):
        pass

    def get_resize_target(self):
        return -1

    def get_num_inuse_pages(self):
        return 0

    def get_num_free_pages(self):
        return 0

    def get_avail_physical_pages(self):
        return 0

    def get_num_reserved_pages(self):
        return 0


@pytest.fixture()
def kv_cache_manager_module(monkeypatch):
    _FakePageAllocator.init_args = None
    _FakePageAllocator.init_kwargs = None
    fake_vmm_ops = types.ModuleType("kvcached.vmm_ops")
    fake_vmm_ops.PageAllocator = _FakePageAllocator
    fake_vmm_ops.InternalPage = _FakeInternalPage
    fake_vmm_ops.kv_tensors_created = lambda group_id=0: True
    fake_vmm_ops.map_to_kv_tensors = lambda offsets, group_id=0: True
    fake_vmm_ops.unmap_from_kv_tensors = lambda offsets, group_id=0: True
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", fake_vmm_ops)
    module = importlib.import_module("kvcached.kv_cache_manager")
    monkeypatch.setattr(module, "PageAllocator", _FakePageAllocator)
    monkeypatch.setattr(module, "InternalPage", _FakeInternalPage)
    monkeypatch.setattr(module.KVCacheManager, "_post_init",
                        lambda self: self._post_init_done.set())
    return module


def test_physical_block_size_derives_layer_group_granularity(
        kv_cache_manager_module):
    manager = kv_cache_manager_module.KVCacheManager(
        num_blocks=16,
        block_size=16,
        cell_size=1024,
        num_layers=8,
        world_size=1,
        physical_block_size=PAGE_SIZE // 4,
    )

    assert manager.layer_group_granularity == 4
    assert manager.num_layer_groups == 2
    assert _FakePageAllocator.init_kwargs["layer_group_granularity"] == 4


def test_physical_block_larger_than_page_attention_block_uses_slice_packing(
        kv_cache_manager_module):
    physical_block_size = PAGE_SIZE // 4
    manager = kv_cache_manager_module.KVCacheManager(
        num_blocks=16,
        block_size=128,
        cell_size=1024,
        num_layers=8,
        world_size=1,
        num_kv_buffers=2,
        physical_block_size=physical_block_size,
    )

    per_kv_block = 128 * 1024
    assert manager.block_level_layer_groups is True
    assert manager.layer_group_granularity == 4
    assert manager.layer_group_layout == "slice_packed"
    assert manager.page_attention_block_mem_size == per_kv_block
    assert manager.physical_block_size == physical_block_size
    assert manager.block_mem_size == per_kv_block
    assert manager.page_size == physical_block_size
    assert manager.physical_layer_slice_size == physical_block_size
    assert manager.physical_group_page_size == PAGE_SIZE * 2
    assert manager.packed_block_stride_bytes == per_kv_block * 2 * 4
    assert manager.blocks_per_physical_page == 4
    assert _FakePageAllocator.init_args[1] == 16 * per_kv_block
    assert _FakePageAllocator.init_args[2] == physical_block_size
    assert _FakePageAllocator.init_kwargs["map_page_size"] == PAGE_SIZE * 2


def test_physical_block_equal_to_page_attention_block_is_block_aligned(
        kv_cache_manager_module):
    physical_block_size = PAGE_SIZE // 4
    manager = kv_cache_manager_module.KVCacheManager(
        num_blocks=16,
        block_size=128,
        cell_size=4096,
        num_layers=8,
        world_size=1,
        num_kv_buffers=2,
        physical_block_size=physical_block_size,
    )

    per_kv_block = 128 * 4096
    assert manager.block_level_layer_groups is True
    assert manager.layer_group_granularity == 4
    assert manager.layer_group_layout == "block_aligned"
    assert manager.page_attention_block_mem_size == per_kv_block
    assert manager.physical_block_size == physical_block_size
    assert manager.block_mem_size == per_kv_block
    assert manager.page_size == physical_block_size
    assert manager.physical_layer_slice_size == physical_block_size
    assert manager.physical_group_page_size == PAGE_SIZE * 2
    assert manager.packed_block_stride_bytes == per_kv_block * 2 * 4
    assert manager.blocks_per_physical_page == 1
    assert _FakePageAllocator.init_args[1] == 16 * per_kv_block
    assert _FakePageAllocator.init_args[2] == physical_block_size
    assert _FakePageAllocator.init_kwargs["map_page_size"] == PAGE_SIZE * 2


def test_layer_group_granularity_must_divide_layers(kv_cache_manager_module):
    with pytest.raises(ValueError, match="must be divisible"):
        kv_cache_manager_module.KVCacheManager(
            num_blocks=16,
            block_size=16,
            cell_size=1024,
            num_layers=10,
            world_size=1,
            physical_block_size=PAGE_SIZE // 4,
        )


class _CapturedCreateArgs(Exception):

    def __init__(self, args, kwargs):
        super().__init__("captured create_kv_tensors")
        self.args = args
        self.kwargs = kwargs


def test_vllm_alloc_kv_cache_derives_slice_packing_from_physical_block(
        monkeypatch):
    fake_vmm_ops = types.ModuleType("kvcached.vmm_ops")
    fake_vmm_ops.PageAllocator = _FakePageAllocator
    fake_vmm_ops.InternalPage = _FakeInternalPage
    fake_vmm_ops.create_kv_tensors = lambda *args, **kwargs: []
    fake_vmm_ops.init_kvcached = lambda *args, **kwargs: None
    fake_vmm_ops.shutdown_kvcached = lambda *args, **kwargs: None
    fake_vmm_ops.kv_tensors_created = lambda group_id=0: True
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", fake_vmm_ops)
    monkeypatch.delitem(
        sys.modules, "kvcached.integration.vllm.interfaces", raising=False)
    mod = importlib.import_module("kvcached.integration.vllm.interfaces")
    monkeypatch.setattr(mod, "_kvcached_initialized", True, raising=False)
    monkeypatch.setattr(mod, "_contiguous_layout", True, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class _Props:
        total_memory = 80 * 1024**3

    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda dev=None: _Props())

    def _fake_create_kv_tensors(*args, **kwargs):
        raise _CapturedCreateArgs(args, kwargs)

    monkeypatch.setattr(mod, "create_kv_tensors", _fake_create_kv_tensors)

    with pytest.raises(_CapturedCreateArgs) as excinfo:
        mod.alloc_kv_cache(
            (2, 16, 128, 4, 128),
            128,
            torch.float16,
            "cuda:0",
            16,
            physical_block_size=PAGE_SIZE // 4,
        )

    kwargs = excinfo.value.kwargs
    assert kwargs["layer_group_granularity"] == 4
    assert kwargs["contiguous_page_size"] == PAGE_SIZE * 2
    assert kwargs["contiguous_total_size"] == 4 * 4 * PAGE_SIZE * 2


def test_vllm_alloc_kv_cache_derives_block_alignment_from_physical_block(
        monkeypatch):
    fake_vmm_ops = types.ModuleType("kvcached.vmm_ops")
    fake_vmm_ops.PageAllocator = _FakePageAllocator
    fake_vmm_ops.InternalPage = _FakeInternalPage
    fake_vmm_ops.create_kv_tensors = lambda *args, **kwargs: []
    fake_vmm_ops.init_kvcached = lambda *args, **kwargs: None
    fake_vmm_ops.shutdown_kvcached = lambda *args, **kwargs: None
    fake_vmm_ops.kv_tensors_created = lambda group_id=0: True
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", fake_vmm_ops)
    monkeypatch.delitem(
        sys.modules, "kvcached.integration.vllm.interfaces", raising=False)
    mod = importlib.import_module("kvcached.integration.vllm.interfaces")
    monkeypatch.setattr(mod, "_kvcached_initialized", True, raising=False)
    monkeypatch.setattr(mod, "_contiguous_layout", True, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class _Props:
        total_memory = 80 * 1024**3

    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda dev=None: _Props())

    def _fake_create_kv_tensors(*args, **kwargs):
        raise _CapturedCreateArgs(args, kwargs)

    monkeypatch.setattr(mod, "create_kv_tensors", _fake_create_kv_tensors)

    with pytest.raises(_CapturedCreateArgs) as excinfo:
        mod.alloc_kv_cache(
            (2, 16, 128, 16, 128),
            128,
            torch.float16,
            "cuda:0",
            16,
            physical_block_size=PAGE_SIZE // 4,
        )

    kwargs = excinfo.value.kwargs
    assert kwargs["layer_group_granularity"] == 4
    assert kwargs["contiguous_page_size"] == PAGE_SIZE * 2
    assert kwargs["contiguous_total_size"] == 4 * 16 * PAGE_SIZE * 2

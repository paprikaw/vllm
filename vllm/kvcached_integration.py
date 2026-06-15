# SPDX-License-Identifier: Apache-2.0
"""Optional KVCacheD integration bootstrap for this vLLM workspace.

The local dynamic/VMM KV cache implementation remains the default. Set
``VLLM_KVCACHE_BACKEND=kvcached`` to apply KVCacheD's native vLLM patches.
"""

from __future__ import annotations

import os
import sys
import importlib
import importlib.util
import inspect
import time
from pathlib import Path
from typing import Any, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

_APPLIED = False
_FAILED = False

_KVCACHED_BACKENDS = {"kvcached", "native", "elastic", "1", "true"}
_VMM_BACKENDS = {
    "",
    "vmm",
    "vm",
    "dynamic",
    "baseline",
    "off",
    "none",
    "disabled",
    "0",
    "false",
}


def get_kv_cache_backend() -> str:
    """Return the selected KV cache backend for this workspace."""
    backend = os.getenv("VLLM_KVCACHE_BACKEND")
    if backend is None and "KVCACHED_VLLM_INTEGRATION_MODE" in os.environ:
        backend = os.getenv("KVCACHED_VLLM_INTEGRATION_MODE")
    if backend is None:
        enabled = os.getenv("ENABLE_KVCACHED", "false").strip().lower()
        backend = "kvcached" if enabled in ("true", "1") else "vmm"
    return backend.strip().lower()


def use_kvcached_backend() -> bool:
    return get_kv_cache_backend() in _KVCACHED_BACKENDS


def use_vmm_backend() -> bool:
    return get_kv_cache_backend() in _VMM_BACKENDS


def use_vm_backend() -> bool:
    """Backward-compatible alias for older local scripts."""
    return use_vmm_backend()


def use_flexi_kv_for_runtime(vllm_config: Any) -> bool:
    """Return the effective KV kernel family for the active backend.

    The VMM baseline keeps using the experimental flexi/direct attention
    kernels. KVCacheD exposes normal VMM-backed tensors, so it must stay on
    the regular FlashAttention tensor path even when the experiment config's
    attention_kernel is ``direct`` for VMM comparisons.
    """
    if use_kvcached_backend():
        return False
    dynamic_config = getattr(vllm_config, "dynamic_config", None)
    return bool(getattr(dynamic_config, "use_flexi_kv", False))


def use_direct_ptr_for_runtime(vllm_config: Any) -> bool:
    """Return whether direct pointer tables are valid for this backend."""
    dynamic_config = getattr(vllm_config, "dynamic_config", None)
    return (use_flexi_kv_for_runtime(vllm_config)
            and bool(getattr(dynamic_config, "use_direct_ptr", False)))


def is_dynamic_migration_enabled(vllm_config: Any) -> bool:
    """Return whether this config should use the dynamic migration engine."""
    dynamic_config = getattr(vllm_config, "dynamic_config", None)
    if dynamic_config is None:
        return False

    is_migration = getattr(dynamic_config, "is_migration", False)
    if callable(is_migration):
        is_migration = is_migration()
    if bool(is_migration):
        return True

    return bool(
        getattr(dynamic_config, "deployment_config_path", None)
        or (getattr(dynamic_config, "alternative_configs", None) is not None
            and getattr(dynamic_config, "migration_steps", None) is not None))


def commit_kvcached_physical_resize_if_needed(
    scheduler: Any,
    reason: str,
) -> bool:
    """Commit a deferred KVCacheD physical resize target.

    Dynamic migration still computes the same target KV block capacity used by
    the VMM baseline. For KVCacheD, that target controls the physical page pool:
    live blocks do not need to be copied/compacted, and this helper trims
    reserved pages so CUDA free memory really increases before weight loading.
    """
    if not use_kvcached_backend():
        return False
    kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
    block_pool = getattr(kv_cache_manager, "block_pool", None)
    commit = getattr(block_pool, "commit_physical_resize", None)
    if commit is None:
        return False
    committed = bool(commit(reason=reason))
    if committed:
        target = getattr(block_pool, "num_gpu_blocks", None)
        if target is not None and hasattr(kv_cache_manager, "num_gpu_blocks"):
            kv_cache_manager.num_gpu_blocks = target
    return committed


def _prepend_kvcached_home() -> None:
    kvcached_home = os.getenv("KVCACHED_HOME")
    if not kvcached_home:
        return
    path = Path(kvcached_home).expanduser().resolve()
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _layer_group_granularity(vllm_config: Any = None) -> int:
    dynamic_config = getattr(vllm_config, "dynamic_config", None)
    value = getattr(dynamic_config, "layer_group_granularity", None)
    if value is None:
        value = os.getenv("KVCACHED_LAYER_GROUP_GRANULARITY", "1")
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _call_with_supported_kwargs(cls: type, *args: Any, **kwargs: Any) -> Any:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError, AttributeError):
        sig = inspect.signature(cls)
    params = sig.parameters
    accepts_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD
                         for param in params.values())
    supported = kwargs if accepts_kwargs else {
        k: v
        for k, v in kwargs.items() if k in params
    }
    return cls(*args, **supported)


def _init_kvcached_scheduler_state(async_sched: bool = True) -> None:
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )
        tp_size = int(get_tensor_model_parallel_world_size())
    except Exception:
        tp_size = 1

    from kvcached.integration.vllm import interfaces as kvi

    kvi.init_kvcached(
        tp_rank=0,
        world_size=tp_size,
        pp_rank=0,
        is_worker=False,
        async_sched=async_sched,
    )


def _enhance_elastic_block_pool_for_dynamic_resize(
    ElasticBlockPool: type,
) -> None:
    """Add the small DynamicKVCacheManager surface to KVCacheD's block pool."""
    if getattr(ElasticBlockPool, "__kvcached_dynamic_resize__", False):
        return

    original_init = ElasticBlockPool.__init__

    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.blocks = self.kv_block_pool
        self._pending_physical_num_gpu_blocks = self.num_gpu_blocks
        self._pending_kvcached_block_moves: dict[int, int] = {}

    def _ensure_block_metadata(self, target_num_blocks: int) -> None:
        if len(self.kv_block_pool) < target_num_blocks:
            block_cls = type(self.kv_block_pool[0])
            start = len(self.kv_block_pool)
            self.kv_block_pool.extend(
                block_cls(idx) for idx in range(start, target_num_blocks))
        self.blocks = self.kv_block_pool

    def _resize_kvcached_pool(self, target_num_blocks: int) -> bool:
        assert target_num_blocks > 0
        block_mem_size = getattr(self.kv_cache_manager, "block_mem_size")
        return bool(self.kv_cache_manager.resize(
            target_num_blocks * block_mem_size))

    def _page_id_for_block(self, block_id: int) -> int:
        idx_dict = self.kv_cache_manager.page_allocator.group_indices_by_page(
            [block_id], self.kv_cache_manager.block_mem_size)
        assert len(idx_dict) == 1
        return next(iter(idx_dict))

    def _take_kvcached_page(self, page_id: int) -> Any:
        manager = self.kv_cache_manager
        if page_id in manager.full_pages:
            return manager.full_pages.pop(page_id)
        if page_id in manager.avail_pages:
            return manager.avail_pages.pop(page_id)
        raise RuntimeError(
            f"KVCacheD page {page_id} missing while committing block moves")

    def _put_kvcached_page(self, page: Any, pages_to_free: list[int]) -> None:
        manager = self.kv_cache_manager
        if page.empty():
            manager.num_avail_blocks -= page.num_free_blocks()
            pages_to_free.append(page.page_id)
        elif page.full():
            manager.full_pages[page.page_id] = page
        else:
            manager.avail_pages[page.page_id] = page

    def _commit_pending_block_moves(self) -> int:
        moves = getattr(self, "_pending_kvcached_block_moves", {})
        if not moves:
            return 0

        manager = self.kv_cache_manager
        alloc_targets_by_page: dict[int, set[int]] = {}
        free_sources_by_page: dict[int, set[int]] = {}
        for source, target in moves.items():
            alloc_targets_by_page.setdefault(
                _page_id_for_block(self, target), set()).add(target)
            free_sources_by_page.setdefault(
                _page_id_for_block(self, source), set()).add(source)

        pages_to_free: list[int] = []
        for page_id in sorted(
                set(alloc_targets_by_page) | set(free_sources_by_page)):
            page = _take_kvcached_page(self, page_id)
            current_free = set(page.get_free_blocks())
            alloc_targets = alloc_targets_by_page.get(page_id, set())
            free_sources = free_sources_by_page.get(page_id, set())

            missing_targets = alloc_targets - current_free
            if missing_targets:
                raise RuntimeError(
                    "KVCacheD compact metadata mismatch: target blocks are "
                    f"not free: {sorted(missing_targets)}")
            already_free_sources = free_sources & current_free
            if already_free_sources:
                raise RuntimeError(
                    "KVCacheD compact metadata mismatch: source blocks are "
                    f"already free: {sorted(already_free_sources)}")

            desired_free = (current_free - alloc_targets) | free_sources
            if page.num_free_blocks() > 0:
                page.alloc(page.num_free_blocks())
            if desired_free:
                page.free_batch(sorted(desired_free))
            manager.num_avail_blocks += len(desired_free) - len(current_free)
            _put_kvcached_page(self, page, pages_to_free)

        if pages_to_free:
            manager.page_allocator.free_pages(pages_to_free)
        num_moves = len(moves)
        moves.clear()
        logger.info("KVCacheD committed %s pending compact block moves",
                    num_moves)
        return num_moves

    def shrink_block_pool(self, new_block_num: int) -> None:
        assert new_block_num <= self.num_gpu_blocks
        if new_block_num == self.num_gpu_blocks:
            self._pending_physical_num_gpu_blocks = new_block_num
            return
        if use_kvcached_backend():
            self.num_gpu_blocks = new_block_num
            self._pending_physical_num_gpu_blocks = new_block_num
            logger.info(
                "KVCacheD deferred physical shrink to %s blocks without "
                "logical KV block compaction", new_block_num)
            return
        for block in self.kv_block_pool[new_block_num:]:
            assert block.ref_cnt == 0, "block_id should be free before shrink"
            if hasattr(block, "block_hash"):
                assert block.block_hash is None, (
                    "block_id should not have a hash before shrink")
        self.kv_block_pool = self.kv_block_pool[:new_block_num]
        self.blocks = self.kv_block_pool
        self.num_gpu_blocks = new_block_num
        self._pending_physical_num_gpu_blocks = new_block_num
        logger.info(
            "KVCacheD deferred physical shrink to %s blocks until worker "
            "KV compaction completes", new_block_num)

    def extend_block_pool(self, new_block_num: int) -> None:
        assert new_block_num >= self.num_gpu_blocks
        if new_block_num == self.num_gpu_blocks:
            return
        if not _resize_kvcached_pool(self, new_block_num):
            raise RuntimeError(
                f"KVCacheD physical extend failed: target_blocks={new_block_num}")
        _ensure_block_metadata(self, new_block_num)
        self.kv_cache_manager.num_blocks = new_block_num
        self.kv_cache_manager.mem_size = (
            new_block_num * self.kv_cache_manager.block_mem_size)
        self.num_gpu_blocks = new_block_num
        self._pending_physical_num_gpu_blocks = new_block_num

    def commit_physical_resize(self, reason: str = "") -> bool:
        num_moves = _commit_pending_block_moves(self)
        target = getattr(self, "_pending_physical_num_gpu_blocks",
                         self.num_gpu_blocks)
        if target == getattr(self.kv_cache_manager, "num_blocks", target):
            before_stats = self.kv_cache_manager.stats()
            try:
                self.kv_cache_manager.trim()
            except Exception:
                logger.exception("KVCacheD trim failed during %s", reason)
                raise
            after_stats = self.kv_cache_manager.stats()
            logger.info(
                "KVCacheD physical trim committed%s: target_blocks=%s "
                "stats_before=%s stats_after=%s pending_moves=%s",
                f" ({reason})" if reason else "", target, before_stats,
                after_stats, num_moves)
            return True

        before_stats = self.kv_cache_manager.stats()
        ok = _resize_kvcached_pool(self, target)
        if not ok:
            raise RuntimeError(
                "KVCacheD physical resize failed after worker compaction: "
                f"target_blocks={target}, reason={reason}, "
                f"stats_before={before_stats}")
        self.kv_cache_manager.num_blocks = target
        self.kv_cache_manager.mem_size = (
            target * self.kv_cache_manager.block_mem_size)
        self.kv_cache_manager.trim()
        after_stats = self.kv_cache_manager.stats()
        logger.info(
            "KVCacheD physical resize committed%s: target_blocks=%s "
            "stats_before=%s stats_after=%s",
            f" ({reason})" if reason else "", target, before_stats,
            after_stats)
        return True

    def migrate_block(
        self,
        block_id_from: int,
        block_id_to: int,
    ) -> None:
        _ensure_block_metadata(self, max(block_id_from, block_id_to) + 1)
        blk_from = self.kv_block_pool[block_id_from]
        blk_to = self.kv_block_pool[block_id_to]
        assert blk_from.ref_cnt != 0, "source block is not in use"
        assert blk_to.ref_cnt == 0, "target block is in use"

        if getattr(self, "enable_prefix_cache", False):
            key = self._block_id_to_key.pop(block_id_from, None)
            if key is not None:
                self._cached_blocks.pop(key, None)
                self._cached_blocks[key] = blk_to
                self._block_id_to_key[block_id_to] = key
            self._evictable_blocks.pop(block_id_to, None)

        if hasattr(blk_to, "block_hash"):
            blk_to.block_hash = getattr(blk_from, "block_hash", None)
        if hasattr(blk_from, "reset_hash"):
            blk_from.reset_hash()
        elif hasattr(blk_from, "block_hash"):
            blk_from.block_hash = None
        self._pending_kvcached_block_moves[block_id_from] = block_id_to
        blk_to.ref_cnt = blk_from.ref_cnt
        blk_from.ref_cnt = 0

    ElasticBlockPool.__init__ = _patched_init
    ElasticBlockPool.shrink_block_pool = shrink_block_pool
    ElasticBlockPool.extend_block_pool = extend_block_pool
    ElasticBlockPool.commit_physical_resize = commit_physical_resize
    ElasticBlockPool.migrate_block = migrate_block
    setattr(ElasticBlockPool, "__kvcached_dynamic_resize__", True)


def _infer_attention_type_from_dynamic_config(kv_cache_config: Any) -> str:
    kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    return "MLA" if getattr(kv_cache_spec, "use_mla", False) else "MHA"


def _ptr_lists_from_kvcached_tensor(
    kv_tensor: Any,
    num_blocks: int,
) -> tuple[list[int], list[int]]:
    """Build FlexiAttention K/V block pointer lists from a KVCacheD tensor."""
    shape = tuple(kv_tensor.shape)
    if len(shape) < 3:
        raise ValueError(f"Unsupported KVCacheD tensor shape: {shape}")
    if shape[0] == 2:
        key_blocks = kv_tensor[0]
        value_blocks = kv_tensor[1]
    elif len(shape) > 1 and shape[1] == 2:
        key_blocks = kv_tensor[:, 0]
        value_blocks = kv_tensor[:, 1]
    else:
        raise ValueError(f"Unsupported KVCacheD MHA tensor shape: {shape}")

    return (
        [int(key_blocks[idx].data_ptr()) for idx in range(num_blocks)],
        [int(value_blocks[idx].data_ptr()) for idx in range(num_blocks)],
    )


def _slice_kvcached_tensor(kv_tensor: Any, num_blocks: int) -> Any:
    """Return a logical KV tensor view exposing ``num_blocks`` blocks."""
    shape = tuple(kv_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"Unsupported KVCacheD tensor shape: {shape}")
    if shape[0] == 2:
        return kv_tensor[:, :num_blocks, ...]
    if len(shape) > 1 and shape[1] == 2:
        return kv_tensor[:num_blocks, ...]
    return kv_tensor[:num_blocks, ...]


def _wait_for_kvcached_tensors_created(group_id: int = 0,
                                       timeout_s: float = 10.0) -> None:
    """Wait until KVCacheD has finished registering its virtual tensors."""
    from kvcached.vmm_ops import kv_tensors_created

    deadline = time.time() + timeout_s
    while not kv_tensors_created(group_id=group_id):
        if time.time() >= deadline:
            raise TimeoutError(
                f"KVCacheD tensors were not created within {timeout_s:.1f}s")
        time.sleep(0.001)


def _rebind_kvcached_tensor_views(
    runner: Any,
    kv_synchronizer: Any,
    num_blocks: int,
) -> None:
    """Refresh regular FlashAttention KV tensor views after KVCacheD resize."""
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.utils import dynamic_bind_kv_cache

    kv_tensors = getattr(runner, "_kvcached_kv_tensors", None)
    if not kv_tensors:
        raise RuntimeError("KVCacheD KV tensors are not initialized")

    forward_context = runner.vllm_config.compilation_config.static_forward_context
    layer_names = list(getattr(runner, "_kvcached_layer_names", []))
    if not layer_names:
        layer_names = sorted(forward_context, key=extract_layer_index)

    kv_views = {
        layer_name: _slice_kvcached_tensor(kv_tensors[idx], num_blocks)
        for idx, layer_name in enumerate(layer_names)
    }
    runner.kv_caches.clear()
    kv_synchronizer.kv_caches.clear()
    dynamic_bind_kv_cache(kv_views, forward_context, runner.kv_caches,
                          kv_synchronizer)


def _rebind_kvcached_flexi_ptrs(runner: Any, num_blocks: int) -> None:
    """Refresh FlexiAttention pointer lists after a KVCacheD resize."""
    import torch

    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.utils import create_ptr_tensor_from_list

    kv_tensors = getattr(runner, "_kvcached_kv_tensors", None)
    if not kv_tensors:
        raise RuntimeError("KVCacheD flexi KV tensors are not initialized")
    if runner.device is not None:
        torch.cuda.set_device(runner.device)

    forward_context = runner.vllm_config.compilation_config.static_forward_context
    layer_names = list(getattr(runner, "_kvcached_layer_names", []))
    if not layer_names:
        layer_names = sorted(forward_context, key=extract_layer_index)

    key_caches: list[list[int]] = []
    value_caches: list[list[int]] = []
    key_ptrs: list[int] = []
    value_ptrs: list[int] = []
    k_ptr_tensors: list[Any] = []
    v_ptr_tensors: list[Any] = []

    for local_idx, layer_name in enumerate(layer_names):
        key_list, value_list = _ptr_lists_from_kvcached_tensor(
            kv_tensors[local_idx], num_blocks)
        k_tensor = create_ptr_tensor_from_list(key_list, runner.device)
        v_tensor = create_ptr_tensor_from_list(value_list, runner.device)

        key_caches.append(key_list)
        value_caches.append(value_list)
        key_ptrs.append(int(k_tensor.data_ptr()))
        value_ptrs.append(int(v_tensor.data_ptr()))
        k_ptr_tensors.append(k_tensor)
        v_ptr_tensors.append(v_tensor)

        attn = forward_context[layer_name]
        attn.key_dev_ptr = int(k_tensor.data_ptr())
        attn.value_dev_ptr = int(v_tensor.data_ptr())
        attn.num_blocks = num_blocks
        attn.page_meta = runner.page_meta

    runner.key_caches = key_caches
    runner.value_caches = value_caches
    runner.key_cache_ptrs = key_ptrs
    runner.value_cache_ptrs = value_ptrs
    runner.k_ptr_tensors = k_ptr_tensors
    runner.v_ptr_tensors = v_ptr_tensors

    kv_synchronizer = getattr(runner, "_kvcached_dynamic_kv_synchronizer", None)
    if kv_synchronizer is not None:
        kv_synchronizer.key_cache_list = list(key_caches)
        kv_synchronizer.value_cache_list = list(value_caches)
        kv_synchronizer.key_cache_ptrs = list(key_ptrs)
        kv_synchronizer.value_cache_ptrs = list(value_ptrs)


def _apply_dynamic_migration_kvcached_patches(
    ElasticBlockPool: type,
    _get_kv_cache_params: Any,
    _get_max_cached_blocks: Any,
) -> None:
    """Patch the local Dynamic* classes to use KVCacheD for migration resize."""
    _enhance_elastic_block_pool_for_dynamic_resize(ElasticBlockPool)

    dyn_mgr_mod = importlib.import_module("vllm.v1.core.dynamic_kv_cache_manager")
    DynamicKVCacheManager = dyn_mgr_mod.DynamicKVCacheManager
    if not getattr(DynamicKVCacheManager.__init__,
                   "__kvcached_dynamic_mgr__", False):
        original_dyn_mgr_init = DynamicKVCacheManager.__init__

        def _patched_dynamic_mgr_init(self, *args: Any, **kwargs: Any) -> None:
            original_dyn_mgr_init(self, *args, **kwargs)
            if not use_kvcached_backend():
                return
            kv_cache_config = args[0] if args else kwargs.get("kv_cache_config")
            if kv_cache_config is None:
                logger.warning(
                    "KVCacheD dynamic manager patch could not find "
                    "kv_cache_config")
                return

            kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
            attention_type = _infer_attention_type_from_dynamic_config(
                kv_cache_config)
            cell_size, num_kv_buffers = _get_kv_cache_params(
                kv_cache_spec, self.block_size, attention_type=attention_type)
            num_layers = len(kv_cache_config.kv_cache_groups[0].layer_names)
            if num_layers == 0:
                num_layers = len(getattr(kv_cache_config, "tensors", {}))

            _init_kvcached_scheduler_state(async_sched=True)
            self.block_pool = _call_with_supported_kwargs(
                ElasticBlockPool,
                self.num_gpu_blocks,
                self.block_size,
                cell_size=cell_size,
                num_layers=num_layers,
                enable_caching=False,
                num_kv_buffers=num_kv_buffers,
                max_cached_blocks=_get_max_cached_blocks(self.block_size),
                layer_group_granularity=_layer_group_granularity(),
            )
            self.single_type_manager.block_pool = self.block_pool
            if hasattr(self.single_type_manager, "_null_block"):
                self.single_type_manager._null_block = self.block_pool.null_block
            logger.info(
                "KVCacheD dynamic KVCacheManager installed "
                "(num_blocks=%s, num_layers=%s)",
                self.num_gpu_blocks, num_layers)

        setattr(_patched_dynamic_mgr_init, "__kvcached_dynamic_mgr__", True)
        DynamicKVCacheManager.__init__ = _patched_dynamic_mgr_init

    dyn_runner_mod = importlib.import_module(
        "vllm.v1.worker.dynamic_gpu_model_runner")
    DynamicGPUModelRunner = dyn_runner_mod.DynamicGPUModelRunner
    if not getattr(DynamicGPUModelRunner.dynamic_initialize_kv_cache,
                   "__kvcached_dynamic_tensor__", False):
        original_tensor_init = DynamicGPUModelRunner.dynamic_initialize_kv_cache

        def _patched_dynamic_tensor_init(
            self,
            kv_cache_config: Any,
            kv_synchronizer: Any,
            num_blocks: int,
        ) -> None:
            if not use_kvcached_backend():
                return original_tensor_init(
                    self, kv_cache_config, kv_synchronizer, num_blocks)

            import torch

            from vllm.v1.kv_cache_interface import AttentionSpec

            from kvcached.integration.vllm import interfaces as kvi

            if len(kv_cache_config.kv_cache_groups) != 1:
                raise NotImplementedError(
                    "KVCacheD dynamic tensor path supports one KV group")
            self.kv_cache_config = kv_cache_config
            self.initialize_attn_backend(kv_cache_config)
            initial_start_layer = getattr(self.model.model, "start_layer", 0)
            self.set_kv_cache_start_layer(
                initial_start_layer, "kvcached_dynamic_tensor_initialize")
            kv_synchronizer.kv_cache_start_layer = initial_start_layer

            kv_cache_group = kv_cache_config.kv_cache_groups[0]
            kv_cache_spec = kv_cache_group.kv_cache_spec
            assert isinstance(kv_cache_spec, AttentionSpec)
            self.kv_cache_dtype = kv_cache_spec.dtype
            kv_cache_shape = self.attn_backends[0].get_kv_cache_shape(
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
            )
            self.kv_cache_shape = kv_cache_shape
            assert num_blocks == kv_cache_config.num_blocks

            tp_rank = int(getattr(self, "tp_rank", 0))
            world_size = int(
                self.vllm_config.parallel_config.tensor_parallel_size)
            pp_rank = int(getattr(self, "rank", 0))
            if self.device is not None:
                torch.cuda.set_device(self.device)
            kvi.init_kvcached(
                tp_rank=tp_rank,
                world_size=world_size,
                pp_rank=pp_rank,
                is_worker=True,
                device=str(self.device),
                async_sched=True,
            )
            if not kv_cache_group.layer_names:
                self._kvcached_kv_tensors = []
                self._kvcached_layer_names = []
                self._kvcached_dynamic_kv_synchronizer = kv_synchronizer
                self.kv_caches = []
                kv_synchronizer.kv_caches = []
                logger.info(
                    "KVCacheD dynamic tensor KV cache initialized with no "
                    "local layers (num_blocks=%s)", num_blocks)
                return
            kv_tensors = kvi.alloc_kv_cache(
                kv_cache_shape,
                kv_cache_spec.block_size,
                kv_cache_spec.dtype,
                self.device.type,
                len(kv_cache_group.layer_names),
                attention_type=_infer_attention_type_from_dynamic_config(
                    kv_cache_config),
                kv_layout="NHD",
                layer_group_granularity=_layer_group_granularity(
                    getattr(self, "vllm_config", None)),
            )
            _wait_for_kvcached_tensors_created()
            self._kvcached_kv_tensors = kv_tensors
            self._kvcached_layer_names = list(kv_cache_group.layer_names)
            self._kvcached_dynamic_kv_synchronizer = kv_synchronizer
            _rebind_kvcached_tensor_views(self, kv_synchronizer, num_blocks)
            logger.info(
                "KVCacheD dynamic tensor KV cache initialized "
                "(num_blocks=%s, num_layers=%s)",
                num_blocks, len(kv_cache_group.layer_names))

        setattr(_patched_dynamic_tensor_init,
                "__kvcached_dynamic_tensor__", True)
        DynamicGPUModelRunner.dynamic_initialize_kv_cache = (
            _patched_dynamic_tensor_init)

    if not getattr(DynamicGPUModelRunner.dynamic_initialize_kv_cache_flexi,
                   "__kvcached_dynamic_flexi__", False):
        original_flexi_init = DynamicGPUModelRunner.dynamic_initialize_kv_cache_flexi

        def _patched_dynamic_flexi_init(
            self,
            kv_cache_config: Any,
            kv_synchronizer: Any,
            num_blocks: int,
        ) -> None:
            if not use_kvcached_backend():
                return original_flexi_init(
                    self, kv_cache_config, kv_synchronizer, num_blocks)

            import torch

            from vllm.utils import get_kv_cache_torch_dtype
            from vllm.v1.kv_cache_interface import AttentionSpec

            from kvcached.integration.vllm import interfaces as kvi

            if len(kv_cache_config.kv_cache_groups) != 1:
                raise NotImplementedError(
                    "KVCacheD dynamic flexi path supports one KV group")
            self.kv_cache_config = kv_cache_config
            self.initialize_attn_backend(kv_cache_config)
            initial_start_layer = getattr(self.model.model, "start_layer", 0)
            self.set_kv_cache_start_layer(
                initial_start_layer, "kvcached_dynamic_initialize")
            kv_synchronizer.kv_cache_start_layer = initial_start_layer

            kv_cache_group = kv_cache_config.kv_cache_groups[0]
            kv_cache_spec = kv_cache_group.kv_cache_spec
            assert isinstance(kv_cache_spec, AttentionSpec)
            self.kv_cache_dtype = kv_cache_spec.dtype
            kv_cache_shape = self.attn_backends[0].get_kv_cache_shape(
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
            )
            self.kv_cache_shape = kv_cache_shape
            block_shape = kv_cache_shape[2:]
            block_token_num, num_heads, head_size = block_shape
            kv_dtype = get_kv_cache_torch_dtype(
                self.kv_cache_dtype, self.model_config.dtype)
            self.page_meta = torch.zeros(
                (block_token_num, num_heads, head_size),
                dtype=kv_dtype,
                device=self.device,
            )

            tp_rank = int(getattr(self, "tp_rank", 0))
            world_size = int(
                self.vllm_config.parallel_config.tensor_parallel_size)
            pp_rank = int(getattr(self, "rank", 0))
            if self.device is not None:
                torch.cuda.set_device(self.device)
            kvi.init_kvcached(
                tp_rank=tp_rank,
                world_size=world_size,
                pp_rank=pp_rank,
                is_worker=True,
                device=str(self.device),
                async_sched=True,
            )
            kv_tensors = kvi.alloc_kv_cache(
                kv_cache_shape,
                kv_cache_spec.block_size,
                kv_cache_spec.dtype,
                self.device.type,
                len(kv_cache_group.layer_names),
                attention_type=_infer_attention_type_from_dynamic_config(
                    kv_cache_config),
                kv_layout="NHD",
                layer_group_granularity=_layer_group_granularity(
                    getattr(self, "vllm_config", None)),
            )
            _wait_for_kvcached_tensors_created()
            self._kvcached_kv_tensors = kv_tensors
            self._kvcached_layer_names = list(kv_cache_group.layer_names)
            self._kvcached_dynamic_kv_synchronizer = kv_synchronizer
            self.grouped_handles = []
            self.key_handles = [[] for _ in kv_cache_group.layer_names]
            self.value_handles = [[] for _ in kv_cache_group.layer_names]
            self.vmm_aligned_bytes = 0
            self.vmm_combined_mode = False
            self.vmm_bytes_per_tensor = 0
            self.vmm_bytes_per_kv = 0

            _rebind_kvcached_flexi_ptrs(self, num_blocks)
            if use_direct_ptr_for_runtime(self.vllm_config):
                self.commit_ptr_tables(
                    self.k_ptr_tensors,
                    self.v_ptr_tensors,
                    is_first_time=True,
                )
            logger.info(
                "KVCacheD dynamic flexi KV cache initialized "
                "(num_blocks=%s, num_layers=%s)",
                num_blocks, len(kv_cache_group.layer_names))

        setattr(_patched_dynamic_flexi_init, "__kvcached_dynamic_flexi__", True)
        DynamicGPUModelRunner.dynamic_initialize_kv_cache_flexi = (
            _patched_dynamic_flexi_init)

    dyn_worker_mod = importlib.import_module("vllm.v1.worker.dynamic_gpu_worker")
    DynamicGPUWorker = dyn_worker_mod.DynamicGPUWorker
    if not getattr(DynamicGPUWorker._resize_kv_cache,
                   "__kvcached_dynamic_tensor_resize__", False):
        original_tensor_resize = DynamicGPUWorker._resize_kv_cache

        def _patched_tensor_resize(self, new_length: int) -> None:
            if (not use_kvcached_backend()
                    or not getattr(self.model_runner, "_kvcached_kv_tensors",
                                   None)):
                return original_tensor_resize(self, new_length)
            assert new_length > 0
            self.block_num = new_length
            logger.info(
                "KVCacheD dynamic tensor KV resize recorded physical target "
                "%s blocks without shrinking virtual tensor views",
                new_length)

        setattr(_patched_tensor_resize,
                "__kvcached_dynamic_tensor_resize__", True)
        DynamicGPUWorker._resize_kv_cache = _patched_tensor_resize

    if not getattr(DynamicGPUWorker._resize_kv_cache_impl,
                   "__kvcached_dynamic_resize__", False):
        original_resize_impl = DynamicGPUWorker._resize_kv_cache_impl

        def _patched_resize_impl(self, new_length: int) -> None:
            if (not use_kvcached_backend()
                    or not use_flexi_kv_for_runtime(
                        getattr(self, "vllm_config", None))
                    or not getattr(self.model_runner, "_kvcached_kv_tensors",
                                   None)):
                return original_resize_impl(self, new_length)
            start_time = time.time()
            if self.device is not None:
                import torch
                torch.cuda.set_device(self.device)
            old_length = self.block_num
            if new_length == old_length:
                logger.info("KVCacheD KV cache length already %s", new_length)
                return
            self.block_num = new_length
            with self.model_runner.forward_lock:
                _rebind_kvcached_flexi_ptrs(self.model_runner, new_length)
            self.dynamic_kv_synchronizer.create_slot_mappings(
                new_length * self.block_size)
            elapsed = time.time() - start_time
            logger.info(
                "KVCacheD dynamic KV resize rebound flexi pointers "
                "from %s to %s blocks in %.3fs",
                old_length, new_length, elapsed)

        setattr(_patched_resize_impl, "__kvcached_dynamic_resize__", True)
        DynamicGPUWorker._resize_kv_cache_impl = _patched_resize_impl


def _apply_legacy_vllm_kvcached_patches() -> None:
    """Patch this workspace's pre-KVCacheCoordinator vLLM layout.

    This checkout reports a 0.9 dev version but still uses the older direct
    KVCacheManager + GPUModelRunner.initialize_kv_cache structure. KVCacheD's
    native patches already contain the right old GPUModelRunner hook; the KV
    manager replacement is small enough to adapt here by structure.
    """
    from kvcached.integration.vllm.patches import (
        ElasticBlockPoolPatch,
        GPUModelRunnerPatch,
        _get_kv_cache_params,
        _get_max_cached_blocks,
    )

    block_pool_mod = importlib.import_module("vllm.v1.core.block_pool")
    if not hasattr(block_pool_mod, "ElasticBlockPool"):
        ElasticBlockPoolPatch().inject_elastic_block_pool(block_pool_mod)
    ElasticBlockPool = getattr(block_pool_mod, "ElasticBlockPool")
    _enhance_elastic_block_pool_for_dynamic_resize(ElasticBlockPool)

    gpu_model_runner_mod = importlib.import_module(
        "vllm.v1.worker.gpu_model_runner")
    gpu_model_runner_patch = GPUModelRunnerPatch()
    gpu_model_runner_patch.initialize_version_info()
    gpu_model_runner_patch.patch_model_runner_init(
        gpu_model_runner_mod.GPUModelRunner)
    gpu_model_runner_patch.patch_initialize_kv_cache(
        gpu_model_runner_mod.GPUModelRunner)

    kvcache_manager_mod = importlib.import_module(
        "vllm.v1.core.kv_cache_manager")
    KVCacheManager = kvcache_manager_mod.KVCacheManager
    if getattr(KVCacheManager.__init__, "__kvcached_workspace_legacy__", False):
        return

    original_init = KVCacheManager.__init__

    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if not use_kvcached_backend():
            return

        kv_cache_config = getattr(self, "kv_cache_config", None)
        if kv_cache_config is None:
            if args:
                kv_cache_config = args[0]
            else:
                kv_cache_config = kwargs.get("kv_cache_config")
        if kv_cache_config is None:
            logger.warning("KVCacheD legacy patch could not find kv_cache_config")
            return

        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        attention_type = "MLA" if getattr(kv_cache_spec, "use_mla", False) else "MHA"
        cell_size, num_kv_buffers = _get_kv_cache_params(
            kv_cache_spec, self.block_size, attention_type=attention_type)
        num_layers = len(getattr(kv_cache_config, "tensors", {}))

        # Keep legacy prefix-cache semantics conservative for this first
        # integration: the old KVCacheD patch also disables it on vLLM 0.8.
        enable_caching = False
        _init_kvcached_scheduler_state(async_sched=True)
        self.block_pool = _call_with_supported_kwargs(
            ElasticBlockPool,
            self.num_gpu_blocks,
            self.block_size,
            cell_size=cell_size,
            num_layers=num_layers,
            enable_caching=enable_caching,
            num_kv_buffers=num_kv_buffers,
            max_cached_blocks=_get_max_cached_blocks(self.block_size),
            layer_group_granularity=_layer_group_granularity(
                getattr(self, "vllm_config", None)),
        )
        if hasattr(self, "single_type_manager"):
            self.single_type_manager.block_pool = self.block_pool
            if hasattr(self.single_type_manager, "_null_block"):
                self.single_type_manager._null_block = self.block_pool.null_block
        logger.info("KVCacheD legacy KVCacheManager patch installed "
                    "(num_blocks=%s, num_layers=%s, num_kv_buffers=%s)",
                    self.num_gpu_blocks, num_layers, num_kv_buffers)

    setattr(_patched_init, "__kvcached_workspace_legacy__", True)
    KVCacheManager.__init__ = _patched_init

    _apply_dynamic_migration_kvcached_patches(
        ElasticBlockPool, _get_kv_cache_params, _get_max_cached_blocks)


def maybe_apply_kvcached_vllm_patches(reason: Optional[str] = None) -> bool:
    """Apply KVCacheD's native vLLM patches once when requested."""
    global _APPLIED, _FAILED

    if not use_kvcached_backend():
        return False
    if _APPLIED:
        return True
    if _FAILED:
        return False

    _prepend_kvcached_home()
    os.environ.setdefault("ENABLE_KVCACHED", "1")
    os.environ.setdefault("KVCACHED_AUTOPATCH", "1")
    os.environ["KVCACHED_VLLM_INTEGRATION_MODE"] = "native"

    try:
        import vllm as vllm_module
        from kvcached.integration.patch_base import (
            PatchManager,
            log_patch_results,
        )
        from kvcached.integration.vllm.nixl_compat import NixlConnectorPatch
        from kvcached.integration.vllm.patches import (
            VLLM_ALL_RANGE,
            VLLM_V8_RANGE,
            VLLM_V9_PLUS_RANGE,
            ElasticBlockPoolPatch,
            EngineCorePatch,
            GPUModelRunnerPatch,
            GPUWorkerPatch,
            KVCacheCoordinatorPatch,
            KVCacheManagerPatch,
        )

        # Reference vLLM so partially-initialized module importers keep the
        # package registered before the patch manager imports vLLM submodules.
        assert vllm_module is not None

        has_coordinator = _has_module("vllm.v1.core.kv_cache_coordinator")
        has_new_gpu_allocator = hasattr(
            importlib.import_module("vllm.v1.worker.gpu_model_runner").
            GPUModelRunner, "_allocate_kv_cache_tensors")

        patch_manager = PatchManager("vllm")
        patch_entries = [
            (NixlConnectorPatch(), VLLM_ALL_RANGE),
            (ElasticBlockPoolPatch(), VLLM_ALL_RANGE),
            (EngineCorePatch(), VLLM_ALL_RANGE),
            (GPUWorkerPatch(), VLLM_ALL_RANGE),
        ]
        if has_coordinator and has_new_gpu_allocator:
            patch_entries.extend([
                (GPUModelRunnerPatch(), VLLM_ALL_RANGE),
                (KVCacheCoordinatorPatch(), VLLM_V9_PLUS_RANGE),
                (KVCacheManagerPatch(), VLLM_V8_RANGE),
            ])
        patch_manager.register_patches_with_versions(patch_entries)
        results = patch_manager.apply_all_patches()
        log_patch_results("vllm", results)
        block_pool_mod = importlib.import_module("vllm.v1.core.block_pool")
        if not hasattr(block_pool_mod, "ElasticBlockPool"):
            ElasticBlockPoolPatch().inject_elastic_block_pool(block_pool_mod)
        from kvcached.integration.vllm.patches import (
            _get_kv_cache_params,
            _get_max_cached_blocks,
        )
        _apply_dynamic_migration_kvcached_patches(
            getattr(block_pool_mod, "ElasticBlockPool"),
            _get_kv_cache_params,
            _get_max_cached_blocks,
        )
        if not (has_coordinator and has_new_gpu_allocator):
            _apply_legacy_vllm_kvcached_patches()
        _APPLIED = True
        logger.info("Applied KVCacheD native vLLM patches%s",
                    f" ({reason})" if reason else "")
        return True
    except Exception as exc:
        _FAILED = True
        logger.exception(
            "Failed to apply KVCacheD native vLLM patches. Set "
            "KVCACHED_HOME or PYTHONPATH to the KVCacheD checkout if it is "
            "not installed. Error: %s", exc)
        raise

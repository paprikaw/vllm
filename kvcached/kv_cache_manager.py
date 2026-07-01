# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
kvcached Memory Manager

This module implements a hierarchical memory management system for KV cache:
- Pages: Large memory chunks (e.g., 2MB) that are mapped/unmapped to physical memory
- Blocks: Smaller units within pages that are allocated to store KV cache data
"""

from __future__ import annotations

import functools
import os
import threading
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from kvcached.locks import NoOpLock
from kvcached.tp_ipc_util import (
    _enter_pre_unmap_callbacks,
    _notify_post_map_callbacks,
    broadcast_kv_tensors_created,
)
from kvcached.utils import (
    CONTIGUOUS_LAYOUT,
    DEFAULT_IPC_NAME,
    LAYER_STACKING,
    PAGE_PREALLOC_ENABLED,
    PAGE_SIZE,
    SANITY_CHECK,
    get_physical_block_size,
    get_kvcached_logger,
)
from kvcached.vmm_ops import kv_tensors_created

try:
    import kvcached.vmm_ops as kvcached_cpp
    PageAllocator = kvcached_cpp.PageAllocator
    InternalPage: Any = kvcached_cpp.InternalPage
except ImportError as e:
    raise ImportError(f"Failed to import kvcached.vmm_ops. Please ensure the C++ extension is built properly. err: {e}")

logger = get_kvcached_logger()

KV_TENSOR_WAIT_TIMEOUT: float = 10.0  # seconds
PREALLOC_THREAD_TIMEOUT: float = 2.0  # seconds


def _autoscaling_debug_enabled() -> bool:
    return os.environ.get("VLLM_AUTOSCALING_KVCACHED_DEBUG", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _autoscaling_debug_verbose() -> bool:
    return os.environ.get("VLLM_AUTOSCALING_KVCACHED_DEBUG_VERBOSE",
                          "").lower() in {
                              "1",
                              "true",
                              "yes",
                              "on",
                          }


def synchronized(method):
    """
    A helper decorator to synchronize access to a method.
    """

    @functools.wraps(method)
    def synchronized_method(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return synchronized_method


class KVCacheManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        cell_size: int,
        num_layers: int,
        world_size: int = 1,
        pp_rank: int = 0,
        async_sched: bool = False,
        reserve_null_block: bool = False,
        num_kv_buffers: int = 2,
        group_id: int = 0,
        physical_block_size: Optional[int] = None,
    ):
        """
        Args:
            num_blocks: Number of blocks.
            block_size: Size of each block in bytes.
            cell_size: Size of each cell in bytes.
            num_layers: Number of layers.
            world_size: Tensor parallel world size within a pipeline stage.
            async_sched: Whether asynchronous scheduling is enabled.
            reserve_null_block: Whether to reserve the first block as null block
                for padding tokens. This is required by SGLang which assumes the
                first block is always reserved as padded tokens.
            num_kv_buffers: Number of KV buffers per layer (2 for MHA K+V,
                1 for MLA combined KV).
            group_id: KV cache group identifier for hybrid attention models.
                Different groups have independent FTensors and page spaces.
            physical_block_size: Layer-stacking K+V physical block size in
                bytes. Only valid when KVCACHED_LAYER_STACKING=true.
        """
        self.num_blocks = num_blocks
        self.kv_tensor_block_mem_size = block_size * cell_size
        self.num_layers = num_layers
        self.num_kv_buffers = num_kv_buffers
        self.layer_stacking_enabled = LAYER_STACKING
        env_physical_block_size = get_physical_block_size()
        if physical_block_size is None:
            physical_block_size = env_physical_block_size
        self.page_attention_block_mem_size = (
            self.kv_tensor_block_mem_size * self.num_kv_buffers)
        self.vrm_block_size = self.page_attention_block_mem_size
        self.reserve_null_block = reserve_null_block
        self.group_id = group_id

        if self.layer_stacking_enabled:
            if CONTIGUOUS_LAYOUT:
                raise ValueError(
                    "KVCACHED_LAYER_STACKING is independent of the original "
                    "KVCacheD contiguous layout. Set "
                    "KVCACHED_CONTIGUOUS_LAYOUT=false when enabling layer "
                    "stacking.")
            if physical_block_size is None:
                physical_block_size = PAGE_SIZE
            physical_block_size = int(physical_block_size)
            if physical_block_size <= 0:
                raise ValueError(
                    "physical_block_size must be positive, got "
                    f"{physical_block_size}")
            if physical_block_size > PAGE_SIZE:
                raise ValueError(
                    "physical_block_size must be no larger than PAGE_SIZE "
                    f"({PAGE_SIZE}), got {physical_block_size}")
            if PAGE_SIZE % physical_block_size != 0:
                raise ValueError(
                    "PAGE_SIZE must be divisible by physical_block_size: "
                    f"PAGE_SIZE={PAGE_SIZE}, "
                    f"physical_block_size={physical_block_size}")
            if physical_block_size % self.num_kv_buffers != 0:
                raise ValueError(
                    "physical_block_size must be divisible by num_kv_buffers: "
                    f"physical_block_size={physical_block_size}, "
                    f"num_kv_buffers={self.num_kv_buffers}")
            self.physical_block_size = physical_block_size
            self.physical_block_slice_size = (
                physical_block_size // self.num_kv_buffers)
            self.layer_group_granularity = PAGE_SIZE // physical_block_size
            if self.num_layers % self.layer_group_granularity != 0:
                raise ValueError(
                    f"num_layers ({self.num_layers}) must be divisible by "
                    "derived layer_group_granularity "
                    f"({self.layer_group_granularity})")
            if self.physical_block_size < self.vrm_block_size:
                raise ValueError(
                    "physical_block_size "
                    f"({self.physical_block_size}) is smaller than "
                    "one K+V PageAttention block "
                    f"({self.vrm_block_size})")
            if self.physical_block_size % self.vrm_block_size != 0:
                raise ValueError(
                    "physical_block_size "
                    f"({self.physical_block_size}) must be divisible by "
                    "K+V PageAttention block size "
                    f"({self.vrm_block_size})")
            self.num_layer_groups = (
                self.num_layers // self.layer_group_granularity)
            self.block_level_layer_groups = True
            # Keep allocator accounting in the same unit as vLLM's
            # PageAttention block: K+V bytes for one logical block in one
            # layer. The per-K/per-V slice size is still tracked separately for
            # tensor layout, but PageAllocator page ids and resize limits should
            # not use the half-block K-only unit.
            self.block_mem_size = self.page_attention_block_mem_size
            self.page_size = self.physical_block_size
            self.physical_layer_slice_size = self.physical_block_size
            self.blocks_per_physical_page = (
                self.physical_block_size // self.vrm_block_size)
            self.layer_group_layout = (
                "block_aligned" if self.blocks_per_physical_page == 1
                else "slice_packed")
            self.physical_group_page_size = (
                self.physical_block_size * self.layer_group_granularity)
            self.packed_block_stride_bytes = (
                self.page_attention_block_mem_size *
                self.layer_group_granularity)
        else:
            if physical_block_size is not None:
                raise ValueError(
                    "KVCACHED_PHYSICAL_BLOCK_SIZE_* / dynamic_config."
                    "physical_block_size is only valid when "
                    "KVCACHED_LAYER_STACKING=true.")
            self.physical_block_size = PAGE_SIZE
            if self.physical_block_size % self.num_kv_buffers != 0:
                raise ValueError(
                    "physical_block_size must be divisible by num_kv_buffers: "
                    f"physical_block_size={self.physical_block_size}, "
                    f"num_kv_buffers={self.num_kv_buffers}")
            self.physical_block_slice_size = (
                self.physical_block_size // self.num_kv_buffers)
            self.layer_group_granularity = 1
            self.num_layer_groups = self.num_layers
            self.block_level_layer_groups = False
            if CONTIGUOUS_LAYOUT:
                self.block_mem_size = self.kv_tensor_block_mem_size
                self.page_size = self.physical_block_slice_size
                self.page_allocator_num_kv_buffers = self.num_kv_buffers
                self.physical_group_page_size = (
                    self.physical_block_size * self.num_layers)
            else:
                # Non-contiguous default layout keeps one FTensor per layer,
                # but maps a physical page as one K+V block-interleaved unit.
                self.block_mem_size = self.page_attention_block_mem_size
                self.page_size = self.physical_block_size
                self.page_allocator_num_kv_buffers = 1
                self.physical_group_page_size = self.physical_block_size
            self.packed_block_stride_bytes = (
                self.page_attention_block_mem_size)
            self.blocks_per_physical_page = (
                self.physical_block_size // self.vrm_block_size)
            self.physical_layer_slice_size = self.page_size
            self.layer_group_layout = (
                "upstream_contiguous" if CONTIGUOUS_LAYOUT else "per_layer")

        # This is the logical allocator size in vLLM PageAttention units:
        # K+V bytes for one block in one layer. Per-buffer K/V slice sizes are
        # only used for tensor layout.
        self.mem_size = self.num_blocks * self.block_mem_size
        self.world_size = world_size
        self.pp_rank = pp_rank
        self.page_allocator = PageAllocator(
            self.num_layers,
            self.mem_size,
            self.page_size,
            self.world_size,
            pp_rank=self.pp_rank,
            async_sched=async_sched,
            contiguous_layout=CONTIGUOUS_LAYOUT,
            layer_group_layout=self.block_level_layer_groups,
            enable_page_prealloc=PAGE_PREALLOC_ENABLED,
            num_kv_buffers=getattr(self, "page_allocator_num_kv_buffers",
                                   self.num_kv_buffers),
            group_id=self.group_id,
            ipc_name=DEFAULT_IPC_NAME,
            layer_group_granularity=self.layer_group_granularity,
            map_page_size=self.physical_group_page_size,
        )
        # Register should_use_worker_ipc callback so C++ PageAllocator knows
        # when the allocator should use worker IPC even with world_size == 1
        # (e.g. vLLM V1 EngineCore + worker in separate processes).
        try:
            from kvcached.integration.vllm.interfaces import (
                should_use_worker_ipc,
            )
            self.page_allocator.set_should_use_worker_ipc_callback(
                should_use_worker_ipc)
            use_worker_ipc = should_use_worker_ipc()
        except ImportError:
            use_worker_ipc = False

        try:
            from kvcached.tp_ipc_util import (
                broadcast_map_to_kv_tensors,
                broadcast_unmap_from_kv_tensors,
            )

            # Wrap Python functions to match C++ callback signature.
            def map_callback(world_size: int, offsets: List[int], pp_rank: int = 0, group_id: int = 0) -> None:
                broadcast_map_to_kv_tensors(world_size, offsets, pp_rank, group_id)

            def unmap_callback(world_size: int, offsets: List[int], pp_rank: int = 0, group_id: int = 0) -> None:
                broadcast_unmap_from_kv_tensors(world_size, offsets, pp_rank, group_id)

            self.page_allocator.set_broadcast_map_callback(map_callback)
            self.page_allocator.set_broadcast_unmap_callback(unmap_callback)

            logger.info(
                "Set up broadcast callbacks for multi-process "
                "(world_size=%d, use_worker_ipc=%s)",
                self.world_size, use_worker_ipc)
        except ImportError as e:
            logger.warning(
                "Failed to import tp_ipc_util module: %s. Broadcast "
                "callbacks will not be available.", e)
        except Exception as e:
            logger.warning(
                "Failed to set up broadcast callbacks: %s. Falling back to "
                "single-process mode.", e)

        self.num_avail_blocks = 0  # Only count free blocks in avail_pages
        self.avail_pages: Dict[int, InternalPage] = {}
        self.full_pages: Dict[int, InternalPage] = {}

        self.reserved_blocks: List[int] = []
        self.null_block: Optional[list[int]] = None

        self.in_shrink: bool = False
        self.target_num_blocks: Optional[int] = None
        self.defer_physical_free_pages: bool = False
        self.deferred_free_page_ids: List[int] = []
        self.deferred_free_page_id_set: set[int] = set()
        self.deferred_free_page_blocks: Dict[int, List[int]] = {}
        self._closed: bool = False
        # NOTE: we use a no-op lock for sync scheduling to avoid overhead
        self._lock = threading.RLock() if async_sched else NoOpLock()

        # Event used to signal that _post_init() has finished.
        self._post_init_done = threading.Event()
        # Launch _post_init in the background; it will block until KV tensors
        # exist, then complete the remaining setup (reserve null block, start
        # pre-alloc thread) and finally set the event.
        threading.Thread(target=self._post_init, daemon=True).start()

    def _post_init(self):
        if self.null_block is not None:
            return

        def _check_kv_tensors_created():
            try:
                from kvcached.integration.vllm.interfaces import should_use_worker_ipc
                vllm_remote = should_use_worker_ipc()
            except ImportError:
                vllm_remote = False

            if self.world_size > 1 or vllm_remote:
                return broadcast_kv_tensors_created(
                    self.world_size, self.pp_rank,
                    group_id=self.group_id)
            else:
                return kv_tensors_created(group_id=self.group_id)

        try:
            total_wait = 0.0
            while not _check_kv_tensors_created():
                if total_wait >= KV_TENSOR_WAIT_TIMEOUT:
                    raise TimeoutError("KV tensors not created after "
                                       f"{KV_TENSOR_WAIT_TIMEOUT} seconds")
                time.sleep(0.001)  # 1ms
                total_wait += 0.001
            # KV tensors created now
            # Possibly reserve the first block as null block for padding tokens
            self._reserve_null_block()

            if not self._closed:
                self.page_allocator.start_prealloc_thread()
                self._maybe_wait_for_sync_prealloc()
        except Exception as e:
            logger.error(
                f"Error during KVCacheManager post-initialization: {e}")
            # Set the event even on error to unblock waiting threads
            raise
        finally:
            self._post_init_done.set()

    def _wait_post_init(self):
        if not self._post_init_done.is_set():
            self._post_init_done.wait()

    def _page_ids_to_unmap_offsets(self, page_ids: List[int]) -> List[int]:
        if not page_ids:
            return []
        pages = [int(page_id) for page_id in page_ids]
        if self.layer_stacking_enabled:
            group_total_span = self.num_blocks * self.physical_group_page_size
            return [
                group_idx * group_total_span
                + page_id * self.physical_group_page_size
                for group_idx in range(self.num_layer_groups)
                for page_id in pages
            ]
        if CONTIGUOUS_LAYOUT:
            return [
                page_id * self.physical_group_page_size
                for page_id in pages
            ]
        return [page_id * self.page_size for page_id in pages]

    def _direct_pre_unmap_context(self, page_ids: List[int]):
        offsets = self._page_ids_to_unmap_offsets(page_ids)
        if not offsets:
            return nullcontext()
        return _enter_pre_unmap_callbacks(offsets, self.group_id)

    def _notify_pages_mapped(self, page_ids: List[int]) -> None:
        offsets = self._page_ids_to_unmap_offsets(page_ids)
        if offsets:
            _notify_post_map_callbacks(offsets, self.group_id)

    def free_pages_with_pre_unmap(self, page_ids: List[int]) -> None:
        if not page_ids:
            return
        pages = [int(page_id) for page_id in page_ids]
        with self._direct_pre_unmap_context(pages):
            self.page_allocator.free_pages(pages)

    def _log_allocator_state(self, label: str) -> None:
        reserved_pages = self.page_allocator.get_num_reserved_pages()
        inuse_pages = self.page_allocator.get_num_inuse_pages()
        free_pages = self.page_allocator.get_num_free_pages()
        budget_free_pages = self.page_allocator.get_num_budget_free_pages()
        mapped_pages = self.page_allocator.get_num_mapped_pages()
        physical_page_limit = self.page_allocator.get_physical_page_limit()
        total_pages = self.page_allocator.get_num_total_pages()
        avail_physical_pages = self.page_allocator.get_avail_physical_pages()
        if self.block_level_layer_groups:
            prealloc_bytes = (
                reserved_pages * self.num_layer_groups *
                self.physical_group_page_size)
            used_bytes = (
                inuse_pages * self.num_layer_groups *
                self.physical_group_page_size)
        else:
            prealloc_bytes = (
                reserved_pages * self.num_layers * self.page_size *
                getattr(self, "page_allocator_num_kv_buffers",
                        self.num_kv_buffers))
            used_bytes = (
                inuse_pages * self.num_layers * self.page_size *
                getattr(self, "page_allocator_num_kv_buffers",
                        self.num_kv_buffers))
        logger.info(
            "KVCacheD allocator state [%s]: reserved_pages=%d, "
            "inuse_pages=%d, free_pages=%d, budget_free_pages=%d, "
            "mapped_pages=%d, physical_page_limit=%d, total_pages=%d, "
            "avail_physical_pages=%d, prealloc_bytes=%.2f GB, "
            "used_bytes=%.2f GB",
            label, reserved_pages, inuse_pages, free_pages,
            budget_free_pages, mapped_pages, physical_page_limit, total_pages,
            avail_physical_pages, prealloc_bytes / (1024**3),
            used_bytes / (1024**3))

    def _maybe_wait_for_sync_prealloc(self) -> None:
        target_raw = os.getenv("KVCACHED_SYNC_PREALLOC_TARGET_PAGES")
        if not target_raw:
            return

        target_pages = int(target_raw)
        if target_pages <= 0:
            self._log_allocator_state("sync-prealloc-disabled")
            return

        timeout_s = float(os.getenv("KVCACHED_SYNC_PREALLOC_TIMEOUT_SEC",
                                    "60"))
        poll_s = float(os.getenv("KVCACHED_SYNC_PREALLOC_POLL_SEC", "0.05"))
        deadline = time.monotonic() + timeout_s
        self._log_allocator_state("sync-prealloc-start")
        logger.info(
            "Waiting for KVCacheD preallocation: target_reserved_pages=%d, "
            "timeout=%.2fs", target_pages, timeout_s)

        while time.monotonic() < deadline:
            reserved_pages = self.page_allocator.get_num_reserved_pages()
            if reserved_pages >= target_pages:
                self._log_allocator_state("sync-prealloc-ready")
                return
            time.sleep(poll_s)

        self._log_allocator_state("sync-prealloc-timeout")
        raise TimeoutError(
            "KVCacheD preallocation did not reach "
            f"{target_pages} reserved pages within {timeout_s:.2f}s")

    def _reserve_null_block(self) -> None:
        """
        Reserve the first block as null block for padding tokens.
        """
        if self.reserve_null_block:
            self.null_block = self._alloc(1, _skip_wait=True)
            if self.null_block != [0]:
                logger.error(f"Failed to reserve null block, got {self.null_block}")
                raise RuntimeError("Failed to reserve null block at index 0")
        else:
            self.null_block = None


    def alloc(self, need_size: int) -> Optional[List[int]]:
        return self._alloc(need_size)

    @synchronized
    def _alloc(self,
               need_size: int,
               _skip_wait: bool = False) -> Optional[List[int]]:
        if not _skip_wait:
            # Normal callers must wait until background initialisation is
            # finished and then perform the usual capacity check.
            self._wait_post_init()

        new_mem_size = self.page_allocator.get_resize_target()
        if new_mem_size > 0:
            self.resize(new_mem_size)

        if self.available_size() < need_size:
            logger.warning(f"available_size()={self.available_size()} < "
                           f"need_size={need_size}")
            return None

        ret_index = []
        page: Optional[InternalPage] = None

        remaining_need = need_size

        if self.reserved_blocks:  # Try to allocate from reserved blocks first
            num_from_reserved = min(len(self.reserved_blocks), remaining_need)
            # ret_index is empty before so we directly assign it
            ret_index = self.reserved_blocks[:num_from_reserved]
            self.reserved_blocks = self.reserved_blocks[num_from_reserved:]
            remaining_need -= num_from_reserved

        while remaining_need > 0:  # Allocate the remaining blocks from pages
            if not self.avail_pages:
                try:
                    page = self.page_allocator.alloc_page()
                except RuntimeError as exc:
                    logger.warning(
                        "KVCacheD page allocation failed: %s. "
                        "Rolling back %d partially allocated blocks.",
                        exc, len(ret_index))
                    if ret_index:
                        self.free(ret_index)
                    return None
                self._notify_pages_mapped([page.page_id])
                page.init(self.block_mem_size)
                if hasattr(page, "cap_blocks"):
                    page.cap_blocks(self.num_blocks)
                # A page may have zero usable blocks when block_mem_size is
                # large (e.g. HYBRID_LINEAR) and every aligned block would
                # straddle the page boundary. Park it in full_pages so it's
                # not re-handed-out but stays lookupable by free().
                if page.num_free_blocks() == 0:
                    self.full_pages[page.page_id] = page
                    continue
                self.num_avail_blocks += page.num_free_blocks()
            else:
                _, page = self.avail_pages.popitem()
            num_from_page = min(page.num_free_blocks(), remaining_need)
            alloced_index = page.alloc(num_from_page)
            ret_index.extend(alloced_index)
            if page.full():
                self.full_pages[page.page_id] = page
            else:
                self.avail_pages[page.page_id] = page

            self.num_avail_blocks -= num_from_page
            remaining_need -= num_from_page

        return ret_index

    @synchronized
    def free(self, indices: List[int]):
        self._wait_post_init()

        if len(indices) == 0:
            return  # Nothing to free

        if SANITY_CHECK:
            for idx in indices:
                if idx in self.reserved_blocks:
                    raise ValueError(f"Freed index {idx} is in "
                                     " reserved_blocks, which is not allowed.")

        idx_dict = self.page_allocator.group_indices_by_page(
            indices, self.block_mem_size)
        if _autoscaling_debug_enabled():
            page_ids = sorted(idx_dict.keys())
            logger.warning(
                "[KVCACHED_FREE_DEBUG] free blocks=%d sample_blocks=%s "
                "pages=%d sample_pages=%s defer_physical=%s in_shrink=%s",
                len(indices), indices[:32], len(page_ids), page_ids[:32],
                self.defer_physical_free_pages, self.in_shrink)

        pages_to_free: List[int] = []
        page_blocks_for_trace: Dict[int, List[int]] = {}
        for page_id, idxs in idx_dict.items():
            # Find the page - it must be in either full_pages or avail_pages
            page = None
            if page_id in self.full_pages:
                page = self.full_pages.pop(page_id)
            elif page_id in self.avail_pages:
                page = self.avail_pages.pop(page_id)
            else:
                if SANITY_CHECK:
                    # This is a serious error - the page should exist
                    raise ValueError(
                        f"Page {page_id} not found in avail_pages or full_pages. "
                        f"This indicates a serious state inconsistency.")
                else:
                    logger.error(
                        f"Page {page_id} not found in avail_pages or full_pages. "
                        f"Skipping to avoid crash, but this indicates a serious bug."
                    )
                    continue

            self.num_avail_blocks += len(idxs)
            page.free_batch(idxs)

            if page.empty():
                pages_to_free.append(page.page_id)
                page_blocks_for_trace[page.page_id] = list(idxs)
                self.num_avail_blocks -= page.num_free_blocks()
            else:
                self.avail_pages[page_id] = page

        if pages_to_free and _autoscaling_debug_enabled():
            traced_pages = sorted(pages_to_free)
            traced_blocks = {
                int(page_id): [
                    int(block)
                    for block in page_blocks_for_trace.get(page_id, [])
                ]
                for page_id in traced_pages
            }
            if _autoscaling_debug_verbose():
                logger.warning(
                    "[KVCACHED_UNMAP_TRACE] phase=free_candidate "
                    "defer_physical=%s in_shrink=%s pages=%s page_blocks=%s "
                    "block_mem_size=%s page_size=%s blocks_per_page=%s "
                    "group_id=%s pp_rank=%s",
                    self.defer_physical_free_pages, self.in_shrink,
                    traced_pages, traced_blocks, self.block_mem_size,
                    self.page_size, self.blocks_per_physical_page,
                    self.group_id, self.pp_rank)
            else:
                logger.warning(
                    "[KVCACHED_UNMAP_TRACE] phase=free_candidate "
                    "defer_physical=%s in_shrink=%s pages_count=%s "
                    "sample_pages=%s sample_page_blocks=%s "
                    "block_mem_size=%s page_size=%s blocks_per_page=%s "
                    "group_id=%s pp_rank=%s",
                    self.defer_physical_free_pages, self.in_shrink,
                    len(traced_pages), traced_pages[:64],
                    {page: traced_blocks[page]
                     for page in traced_pages[:16]}, self.block_mem_size,
                    self.page_size, self.blocks_per_physical_page,
                    self.group_id, self.pp_rank)

        if pages_to_free and self.defer_physical_free_pages:
            new_pages = [
                page_id for page_id in pages_to_free
                if page_id not in self.deferred_free_page_id_set
            ]
            self.deferred_free_page_ids.extend(new_pages)
            self.deferred_free_page_id_set.update(new_pages)
            for page_id in new_pages:
                self.deferred_free_page_blocks[page_id] = list(
                    page_blocks_for_trace.get(page_id, []))
            logger.info(
                "KVCacheD deferred physical free of %d pages during "
                "migration (total_deferred=%d)",
                len(new_pages), len(self.deferred_free_page_ids))
        elif pages_to_free:
            self.free_pages_with_pre_unmap(pages_to_free)
            if _autoscaling_debug_enabled():
                logger.warning(
                    "[KVCACHED_UNMAP_TRACE] phase=physical_free_done "
                    "trace_ns=%s defer_physical=False pages=%s page_blocks=%s "
                    "group_id=%s pp_rank=%s",
                    time.time_ns(),
                    sorted(pages_to_free),
                    {
                        int(page_id): [
                            int(block) for block in
                            page_blocks_for_trace.get(page_id, [])
                        ]
                        for page_id in sorted(pages_to_free)
                    },
                    self.group_id, self.pp_rank)

        if self.in_shrink:
            assert self.target_num_blocks is not None
            if self._get_num_alloced_blocks() <= self.target_num_blocks:
                self.page_allocator.resize(self.target_num_blocks *
                                           self.block_mem_size)
                self.in_shrink = False
                self.target_num_blocks = None

    @synchronized
    def try_to_reserve(self, need_size: int) -> bool:
        self._wait_post_init()
        if self.available_size() < need_size:
            return False
        reserved = self.alloc(need_size)
        if reserved is None:
            logger.warning("Failed to reserve blocks.")
            return False
        self.reserved_blocks.extend(reserved)
        return True

    @synchronized
    def free_reserved(self):
        if self.reserved_blocks:
            self.free(self.reserved_blocks)
            self.reserved_blocks.clear()

    @synchronized
    def resize(self, new_mem_size: int):
        """
        Reset the limit of the K or V tensor in one layer.
        new_mem_size: the memory size of the K or V tensor in one layer
        """
        self._wait_post_init()
        assert new_mem_size > 0, "new_mem_size must be positive"
        if self.page_allocator.resize(new_mem_size):
            if self.in_shrink:
                self.in_shrink = False
                self.target_num_blocks = None
            return True  # Successfully resized.
        # Failed to resize due to too many in-use blocks.
        assert (len(self.reserved_blocks) == 0
                ), "Reserved blocks must be freed before resizing."
        # NOTE: we can support resizing with reserved blocks, but we want to
        # enforce this check for now to ensure correctness.
        self.in_shrink = True
        self.target_num_blocks = new_mem_size // self.block_mem_size
        self.free_reserved()
        return False

    @synchronized
    def trim(self) -> None:
        """
        Trim the reserved pages to free up physical memory.
        """
        self._wait_post_init()
        if self.defer_physical_free_pages:
            logger.info(
                "KVCacheD skipping trim while migration physical frees are "
                "deferred")
            return
        self.page_allocator.trim()

    @synchronized
    def begin_defer_physical_free(self) -> None:
        self._wait_post_init()
        self.defer_physical_free_pages = True
        logger.info("KVCacheD began deferring physical page frees")

    @synchronized
    def end_defer_physical_free(self) -> None:
        self._wait_post_init()
        pages_to_free = list(self.deferred_free_page_ids)
        page_blocks = {
            int(page_id): [
                int(block)
                for block in self.deferred_free_page_blocks.get(page_id, [])
            ]
            for page_id in pages_to_free
        }
        self.deferred_free_page_ids.clear()
        self.deferred_free_page_id_set.clear()
        self.deferred_free_page_blocks.clear()
        self.defer_physical_free_pages = False
        if not pages_to_free:
            logger.info("KVCacheD ended physical free deferral: no pages")
            return
        logger.info(
            "KVCacheD ending physical free deferral: freeing %d pages",
            len(pages_to_free))
        if _autoscaling_debug_enabled():
            if _autoscaling_debug_verbose():
                logger.warning(
                    "[KVCACHED_UNMAP_TRACE] phase=deferred_release pages=%s "
                    "page_blocks=%s block_mem_size=%s page_size=%s "
                    "blocks_per_page=%s group_id=%s pp_rank=%s",
                    pages_to_free, page_blocks, self.block_mem_size,
                    self.page_size, self.blocks_per_physical_page,
                    self.group_id, self.pp_rank)
            else:
                logger.warning(
                    "[KVCACHED_UNMAP_TRACE] phase=deferred_release "
                    "pages_count=%s sample_pages=%s sample_page_blocks=%s "
                    "block_mem_size=%s page_size=%s blocks_per_page=%s "
                    "group_id=%s pp_rank=%s",
                    len(pages_to_free), pages_to_free[:64],
                    {page: page_blocks[page]
                     for page in pages_to_free[:16]}, self.block_mem_size,
                    self.page_size, self.blocks_per_physical_page,
                    self.group_id, self.pp_rank)
        self.free_pages_with_pre_unmap(pages_to_free)
        if _autoscaling_debug_enabled():
            logger.warning(
                "[KVCACHED_UNMAP_TRACE] phase=physical_free_done "
                "trace_ns=%s defer_physical=True pages=%s page_blocks=%s "
                "group_id=%s pp_rank=%s",
                time.time_ns(), pages_to_free, page_blocks, self.group_id,
                self.pp_rank)

    def shutdown(self) -> None:
        """Stop background allocator activity before KV tensors are destroyed."""
        if self._closed:
            return
        self._closed = True
        try:
            self._post_init_done.wait(PREALLOC_THREAD_TIMEOUT)
            self.page_allocator.stop_prealloc_thread()
        except Exception as e:
            logger.debug("Failed to stop KVCacheManager cleanly: %s", e)

    def close(self) -> None:
        self.shutdown()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    @synchronized
    def available_size(self) -> int:
        avail_blocks = self.num_avail_blocks + len(self.reserved_blocks)
        if self.in_shrink:
            blocks_from_free_pages = 0
        else:
            budget_free_pages = self.page_allocator.get_num_budget_free_pages()
            physical_free_pages = self.page_allocator.get_avail_physical_pages(
            ) + self.page_allocator.get_num_reserved_pages()
            free_pages = min(budget_free_pages, physical_free_pages)
            blocks_from_free_pages = free_pages * InternalPage.get_num_blocks(
                self.page_size, self.block_mem_size)
        return avail_blocks + blocks_from_free_pages

    @synchronized
    def get_mapped_memory_size(self, unit='bytes') -> float:
        """Get memory usage in specified unit (bytes, kb, mb, gb)."""
        if self.block_level_layer_groups:
            memory_bytes = (self.page_allocator.get_num_inuse_pages() *
                            self.num_layer_groups *
                            self.physical_group_page_size)
        else:
            memory_bytes = (self.page_allocator.get_num_inuse_pages() *
                            self.num_layer_groups *
                            self.layer_group_granularity * self.page_size *
                            getattr(self, "page_allocator_num_kv_buffers",
                                    self.num_kv_buffers))

        if unit == 'bytes':
            return memory_bytes
        elif unit == 'kb':
            return memory_bytes / 1024
        elif unit == 'mb':
            return memory_bytes / (1024**2)
        elif unit == 'gb':
            return memory_bytes / (1024**3)
        else:
            raise ValueError(f"Unknown unit: {unit}")

    def _page_capacity_blocks(self, page_id: int, block_limit: int) -> int:
        start, end = InternalPage.get_block_range(page_id, self.page_size,
                                                  self.block_mem_size)
        capped_start = max(start, 0)
        capped_end = min(end, block_limit)
        return max(capped_end - capped_start, 0)

    def _page_free_blocks_below_limit(self, page: InternalPage,
                                      block_limit: int) -> int:
        return sum(1 for block_id in page.get_free_blocks()
                   if 0 <= block_id < block_limit)

    def _fragmentation_stats(
        self,
        target_blocks: Optional[int] = None,
    ) -> dict[str, Any]:
        """Compare logical PageAttention free blocks with whole free pages."""
        block_limit = self.num_blocks if target_blocks is None else int(
            target_blocks)
        block_limit = max(block_limit, 0)
        page_limit = (block_limit * self.block_mem_size + self.page_size -
                      1) // self.page_size
        page_limit = min(page_limit, self.page_allocator.get_num_total_pages())
        bytes_per_logical_block = (
            self.page_attention_block_mem_size * self.num_layers)

        active_pages = set(self.full_pages) | set(self.avail_pages)
        live_blocks = 0
        partial_page_free_blocks = 0

        for page_id, page in self.full_pages.items():
            live_blocks += self._page_capacity_blocks(page_id, block_limit)

        for page_id, page in self.avail_pages.items():
            capacity = self._page_capacity_blocks(page_id, block_limit)
            free_blocks = self._page_free_blocks_below_limit(page, block_limit)
            live_blocks += max(capacity - free_blocks, 0)
            partial_page_free_blocks += free_blocks

        reserved_available_blocks = sum(
            1 for block_id in self.reserved_blocks
            if 0 <= block_id < block_limit)
        logical_live_blocks = max(live_blocks - reserved_available_blocks, 0)
        logical_available_blocks = max(block_limit - logical_live_blocks, 0)

        physical_whole_available_blocks = 0
        for page_id in range(page_limit):
            if page_id in active_pages:
                continue
            physical_whole_available_blocks += self._page_capacity_blocks(
                page_id, block_limit)
        physical_whole_available_blocks = min(
            physical_whole_available_blocks, logical_available_blocks)

        internal_fragmentation_blocks = max(
            logical_available_blocks - physical_whole_available_blocks, 0)
        logical_available_bytes = (
            logical_available_blocks * bytes_per_logical_block)
        physical_whole_available_bytes = (
            physical_whole_available_blocks * bytes_per_logical_block)
        internal_fragmentation_bytes = (
            internal_fragmentation_blocks * bytes_per_logical_block)
        internal_fragmentation_ratio = (
            internal_fragmentation_bytes / logical_available_bytes
            if logical_available_bytes > 0 else 0.0)

        return {
            "fragmentation_target_blocks": block_limit,
            "fragmentation_target_pages": page_limit,
            "fragmentation_bytes_per_logical_block":
            bytes_per_logical_block,
            "fragmentation_logical_live_blocks": logical_live_blocks,
            "fragmentation_logical_available_blocks":
            logical_available_blocks,
            "fragmentation_logical_available_bytes":
            logical_available_bytes,
            "fragmentation_physical_whole_available_blocks":
            physical_whole_available_blocks,
            "fragmentation_physical_whole_available_bytes":
            physical_whole_available_bytes,
            "fragmentation_internal_blocks": internal_fragmentation_blocks,
            "fragmentation_internal_bytes": internal_fragmentation_bytes,
            "fragmentation_internal_ratio": internal_fragmentation_ratio,
            "fragmentation_partial_page_free_blocks":
            partial_page_free_blocks,
            "fragmentation_reserved_available_blocks":
            reserved_available_blocks,
        }

    @synchronized
    def stats(self, target_blocks: Optional[int] = None) -> dict[str, Any]:
        """Return lightweight allocator statistics for debugging/tests."""
        blocks_per_allocator_page = InternalPage.get_num_blocks(
            self.page_size, self.block_mem_size)
        page_allocator_num_kv_buffers = getattr(
            self, "page_allocator_num_kv_buffers", self.num_kv_buffers)
        if self.block_level_layer_groups:
            physical_maps_per_allocator_page = self.num_layer_groups
            bytes_per_allocator_page = (
                self.num_layer_groups * self.physical_group_page_size)
        elif CONTIGUOUS_LAYOUT:
            physical_maps_per_allocator_page = 1
            bytes_per_allocator_page = self.physical_group_page_size
        else:
            physical_maps_per_allocator_page = (
                self.num_layers * page_allocator_num_kv_buffers)
            bytes_per_allocator_page = (
                self.num_layers * page_allocator_num_kv_buffers *
                self.page_size)
        num_mapped_allocator_pages = self.page_allocator.get_num_mapped_pages()
        stats = {
            "num_blocks": self.num_blocks,
            "block_mem_size": self.block_mem_size,
            "kv_tensor_block_mem_size": self.kv_tensor_block_mem_size,
            "page_attention_block_mem_size": self.page_attention_block_mem_size,
            "num_layers": self.num_layers,
            "num_kv_buffers": self.num_kv_buffers,
            "page_allocator_num_kv_buffers": page_allocator_num_kv_buffers,
            "vrm_block_size": self.vrm_block_size,
            "layer_stacking_enabled": self.layer_stacking_enabled,
            "contiguous_layout": CONTIGUOUS_LAYOUT,
            "layer_group_granularity": self.layer_group_granularity,
            "layer_group_layout": self.layer_group_layout,
            "num_layer_groups": self.num_layer_groups,
            "block_level_layer_groups": self.block_level_layer_groups,
            "physical_block_size": self.physical_block_size,
            "physical_block_slice_size": self.physical_block_slice_size,
            "page_size": self.page_size,
            "physical_group_page_size": self.physical_group_page_size,
            "physical_layer_slice_size": self.physical_layer_slice_size,
            "packed_block_stride_bytes": self.packed_block_stride_bytes,
            "blocks_per_physical_page": self.blocks_per_physical_page,
            "blocks_per_allocator_page": blocks_per_allocator_page,
            "physical_maps_per_allocator_page":
            physical_maps_per_allocator_page,
            "bytes_per_allocator_page": bytes_per_allocator_page,
            "available_size": self.available_size(),
            "mapped_memory_bytes": self.get_mapped_memory_size("bytes"),
            "num_total_pages": self.page_allocator.get_num_total_pages(),
            "physical_page_limit": self.page_allocator.get_physical_page_limit(),
            "num_mapped_pages": num_mapped_allocator_pages,
            "num_mapped_allocator_pages": num_mapped_allocator_pages,
            "estimated_physical_map_ops": (
                num_mapped_allocator_pages *
                physical_maps_per_allocator_page),
            "num_inuse_pages": self.page_allocator.get_num_inuse_pages(),
            "num_free_pages": self.page_allocator.get_num_free_pages(),
            "num_budget_free_pages":
            self.page_allocator.get_num_budget_free_pages(),
            "num_reserved_pages": self.page_allocator.get_num_reserved_pages(),
            "num_avail_blocks": self.num_avail_blocks,
            "num_full_pages": len(self.full_pages),
            "num_avail_pages": len(self.avail_pages),
            "num_reserved_blocks": len(self.reserved_blocks),
        }
        stats.update(self._fragmentation_stats(target_blocks))
        return stats

    @synchronized
    def clear(self):
        """
        Free all allocated blocks and reset the allocator to initial state.
        """

        self._wait_post_init()

        # Stop the prealloc thread first — it runs on the PageAllocator's
        # lock and can grab pages between our trim/reset/reserve steps,
        # causing the null-block reservation to get a non-zero block.
        self.page_allocator.stop_prealloc_thread()

        # Clear reserved blocks
        self.free_reserved()

        # Free all blocks from avail_pages and full_pages
        pages_to_free: List[int] = []
        for page in self.avail_pages.values():
            pages_to_free.append(page.page_id)
        for page in self.full_pages.values():
            pages_to_free.append(page.page_id)
        if pages_to_free:
            self.free_pages_with_pre_unmap(pages_to_free)
        self.avail_pages.clear()
        self.full_pages.clear()

        # Trim the page allocator to free up reserved pages
        self.trim()

        # Reset the page allocator's free list to its original sorted order.
        # After free_pages + trim, freed pages are appended to the END of
        # free_page_list, so the order is scrambled.  This matters because
        # _reserve_null_block() pops from the LEFT and expects to get page 0
        # (which yields block 0 — the null block SGLang requires).
        self.page_allocator.reset_free_page_order()

        self.target_num_blocks = None
        self.in_shrink = False
        self.num_avail_blocks = 0

        # Possibly reserve the first block as null block for padding tokens
        self._reserve_null_block()

        # Restart the prealloc thread now that null block is safely reserved.
        if not self._closed:
            self.page_allocator.start_prealloc_thread()

    # Private methods
    @synchronized
    def _get_num_alloced_blocks(self) -> int:
        # Blocks from fully allocated pages
        blocks_from_full_pages = len(self.full_pages) * InternalPage.get_num_blocks(
            self.page_size, self.block_mem_size)
        # Blocks from partially allocated pages. num_avail_blocks is the number
        # of free blocks in the partially allocated pages so the number of
        # allocated blocks is the total number of blocks in the partially
        # allocated pages minus the number of free blocks.
        blocks_from_avail_pages = len(self.avail_pages) * InternalPage.get_num_blocks(
            self.page_size, self.block_mem_size) - self.num_avail_blocks
        # Blocks from reserved blocks
        blocks_from_reserved_blocks = len(self.reserved_blocks)
        return (blocks_from_full_pages + blocks_from_avail_pages +
                blocks_from_reserved_blocks)

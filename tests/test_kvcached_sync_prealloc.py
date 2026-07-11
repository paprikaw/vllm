import pytest

from kvcached.kv_cache_manager import KVCacheManager


class _FakePageAllocator:

    def __init__(self, reserved_pages: int):
        self.reserved_pages = reserved_pages
        self.refill_disabled = False

    def get_num_reserved_pages(self) -> int:
        return self.reserved_pages

    def disable_prealloc_refill(self) -> None:
        self.refill_disabled = True


def _make_manager(reserved_pages: int):
    manager = object.__new__(KVCacheManager)
    manager.page_allocator = _FakePageAllocator(reserved_pages)
    labels = []
    manager._log_allocator_state = labels.append
    return manager, labels


def test_one_shot_prealloc_disables_refill_after_target(monkeypatch):
    monkeypatch.setenv("KVCACHED_PREALLOC_MODE", "one_shot")
    monkeypatch.setenv("KVCACHED_SYNC_PREALLOC_TARGET_PAGES", "8")
    manager, labels = _make_manager(reserved_pages=8)

    manager._maybe_wait_for_sync_prealloc()

    assert manager.page_allocator.refill_disabled
    assert labels == ["sync-prealloc-start", "sync-prealloc-one-shot-ready"]


def test_continuous_prealloc_keeps_refill_enabled(monkeypatch):
    monkeypatch.setenv("KVCACHED_PREALLOC_MODE", "continuous")
    monkeypatch.setenv("KVCACHED_SYNC_PREALLOC_TARGET_PAGES", "8")
    manager, labels = _make_manager(reserved_pages=8)

    manager._maybe_wait_for_sync_prealloc()

    assert not manager.page_allocator.refill_disabled
    assert labels == ["sync-prealloc-start", "sync-prealloc-ready"]


def test_one_shot_prealloc_requires_sync_target(monkeypatch):
    monkeypatch.setenv("KVCACHED_PREALLOC_MODE", "one_shot")
    monkeypatch.delenv("KVCACHED_SYNC_PREALLOC_TARGET_PAGES", raising=False)
    manager, _ = _make_manager(reserved_pages=0)

    with pytest.raises(ValueError, match="requires"):
        manager._maybe_wait_for_sync_prealloc()


def test_one_shot_prealloc_requires_positive_target(monkeypatch):
    monkeypatch.setenv("KVCACHED_PREALLOC_MODE", "one_shot")
    monkeypatch.setenv("KVCACHED_SYNC_PREALLOC_TARGET_PAGES", "0")
    manager, _ = _make_manager(reserved_pages=0)

    with pytest.raises(ValueError, match="positive"):
        manager._maybe_wait_for_sync_prealloc()


def test_prealloc_mode_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("KVCACHED_PREALLOC_MODE", "sometimes")
    manager, _ = _make_manager(reserved_pages=0)

    with pytest.raises(ValueError, match="continuous.*one_shot"):
        manager._maybe_wait_for_sync_prealloc()

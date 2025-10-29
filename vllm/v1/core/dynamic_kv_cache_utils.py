from typing import Callable
from .kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

def compact_cache(migrate_func: Callable[[int, int], None], is_used: Callable[[int], bool], compacted_length: int, kv_cache_length: int):
    left, right = 0, kv_cache_length - 1
    while left < right:
        while left < right and is_used(left):
            left += 1
        while left < right and not is_used(right): # block_pool.blocks[right].ref_cnt == 0:
            right -= 1
        if left < right:
            migrate_func(right, left)
            left += 1
            right -= 1
        if right < compacted_length:
            break

def compact_cache_with_record(migrate_func: Callable[[int, int, dict[int, int]], None], is_used: Callable[[int], bool], compacted_length: int, kv_cache_length: int, migrate_record: dict[int, int]):
    left, right = 0, kv_cache_length - 1
    while left < right:
        while left < right and is_used(left):
            left += 1
        while left < right and not is_used(right): # block_pool.blocks[right].ref_cnt == 0:
            right -= 1
        if left < right:
            migrate_func(right, left, migrate_record)
            left += 1
            right -= 1
        if right < compacted_length:
            break
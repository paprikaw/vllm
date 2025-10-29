from vllm.v1.core.dynamic_block_pool import DynamicBlockPool
from .kv_cache_manager import KVCacheManager, KVCacheConfig, BlockPool, BlockHashType, PrefixCacheStats
from vllm.utils import sha256
from collections import defaultdict
from vllm.v1.core.single_type_kv_cache_manager import (
    get_manager_for_kv_cache_spec)
from vllm.v1.core.dynamic_single_type_kv_cache_manager import DynamicFullAttentionManager
from .dynamic_kv_cache_utils import compact_cache_with_record
from bitarray import bitarray
from vllm.logger import init_logger
logger = init_logger(__name__)




class DynamicKVCacheManager(KVCacheManager):
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        assert len(kv_cache_config.kv_cache_groups) == 1, (
            "KVCacheManager does not support hybrid models with more than 1 "
            "kv cache group")
        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = kv_cache_spec.block_size
        self.num_gpu_blocks = kv_cache_config.num_blocks
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.caching_hash_fn = sha256 if caching_hash_algo == "sha256" else hash
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        # FIXME: make prefix cache stats conditional on log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.block_pool = DynamicBlockPool(self.num_gpu_blocks, enable_caching,
                                    enable_kv_cache_events)

        self.single_type_manager = DynamicFullAttentionManager(
            kv_cache_spec=kv_cache_spec,
            block_pool=self.block_pool,
            use_eagle=self.use_eagle,
            num_kv_cache_groups=1,
            caching_hash_fn=self.caching_hash_fn,
        )
        
        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        # 这个不需要更改
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHashType]] = defaultdict(list)
    
    def _migrate_block(self, block_id: int, to_block_id: int, migrate_record: dict[int, int]) -> None:
        self.single_type_manager.update_requests_to_new_block_id(block_id, to_block_id)
        self.block_pool.migrate_block(block_id, to_block_id)
        migrate_record[block_id] = to_block_id

    def get_bitmap(self) -> bitarray:
        bitmap = bitarray(self.block_pool.num_gpu_blocks)
        for block in self.block_pool.blocks:
            if block.ref_cnt != 0:
                assert block.block_id is not None
                bitmap[block.block_id] = 1
            else:
                assert block.block_hash is None
        return bitmap

    def compact_kv_cache(self, compacted_length: int) -> None:
        """Compact the KV cache by moving all used blocks to the leftmost
        `compact length` portion of the cache.
        """
        total = self.block_pool.num_gpu_blocks
        assert 0 < compacted_length <= total, "invalid compacted_length"

        # 需要满足：可裁掉的容量 <= 当前空闲块数
        free_blocks = self.block_pool.get_num_free_blocks()
        need_free = total - compacted_length
        assert free_blocks >= need_free, \
            f"Not enough free space to compact: free={free_blocks}, need={need_free}"

        def is_used(idx):
            return self.block_pool.blocks[idx].ref_cnt != 0
        migrate_record: dict[int, int] = {}
        compact_cache_with_record(
            self._migrate_block, 
            is_used,
            compacted_length, 
            total,
            migrate_record)
        logger.info(f"[debug]: scheduler migrate_record: {migrate_record}")
        logger.info(f"before kv cache shrinking, the kv cache utilization is {self.block_pool.get_usage()}")
        # left, right = 0, len(self.block_pool.blocks) - 1
        # while left < right:
        #     while left < right and self.block_pool.blocks[left].ref_cnt != 0:
        #         left += 1
        #     while left < right and self.block_pool.blocks[right].ref_cnt == 0:
        #         right -= 1
        #     if left < right:
        #         self._migrate_block(right, left)
        #         left += 1
        #         right -= 1
        #     if right < compacted_length:
        #         break
        
        self.block_pool.shrink_block_pool(compacted_length)
        self.num_gpu_blocks = compacted_length
        logger.info(f"after kv cache shrinking, the kv cache utilization is {self.block_pool.get_usage()}")
        assert self.block_pool.num_gpu_blocks == compacted_length
        return
    
    def extend_kv_cache(self, extended_length: int) -> None:
        """ Extend the KV cache to a certain length
        """
        self.block_pool.extend_block_pool(extended_length)
        assert self.block_pool.num_gpu_blocks == extended_length
        return
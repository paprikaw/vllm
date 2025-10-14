from numpy import block
from .block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.logger import init_logger

logger = init_logger(__name__)

class DynamicBlockPool(BlockPool):
    def migrate_block(self, block_id_from: int, block_id_to: int) -> None:
        """
        将某个哈希 h 的“归属槽位”从 block_id_from 切换到 block_id_to。
        不搬运任何 KV 内容；仅更新哈希桶映射和块元数据，使得：
          - cached_block_hash_to_block[h] 中的 bucket 从 from -> to
          - blocks[to].block_hash = h
          - blocks[from].block_hash = None

        前置假设：
          - blocks[from].block_hash 是一个有效的满块哈希（非 None）
          - blocks[to].block_hash is None（目标槽位目前为空或非满块）
          - 两边 ref_cnt == 0（避免正在使用中的块被“改名”）
        """
        # with self._lock:   # 若有并发，解开这行
        # 1) 基本合法性和边界检查
        n = len(self.blocks)
        assert 0 <= block_id_from < n, f"block_id_from out of range: {block_id_from}"
        assert 0 <= block_id_to  < n, f"block_id_to out of range: {block_id_to}"
        assert block_id_from != block_id_to, "from/to cannot be the same"

        blk_from = self.blocks[block_id_from]
        blk_to   = self.blocks[block_id_to]

        # 建议在迁移前确保两边都不在使用中（ref_cnt == 0）
        assert blk_from.ref_cnt != 0, "block_id_from is in use (ref_cnt > 0)"
        assert blk_to.ref_cnt == 0, "block_id_to is in use (ref_cnt > 0)"

        if self.enable_caching:
            assert blk_from.block_hash is not None, "block_id_from should have a hash (must be a full block)"
            assert blk_to.block_hash is None, "block_id_to should not have a hash (must be empty or non-full)"
            bucket = self.cached_block_hash_to_block.get(blk_from.block_hash)
            assert bucket is not None, "hash bucket missing in cached_block_hash_to_block"
            assert block_id_from in bucket, "source id missing in hash bucket"
            assert block_id_to not in bucket, "target id already exists in hash bucket"

            # 2) 迁移哈希桶键（只改“归属槽位”，不改对象内容）
            popped = bucket.pop(block_id_from, None)
            assert popped is not None, "unexpected: lost source KVCacheBlock in bucket during pop"
            # 语义：该哈希现在由 block_id_to 槽位承载，所以映射值应指向 to 槽位的块对象
            bucket[block_id_to] = blk_to

            # 3) 迁移块数据
            blk_to.block_hash = blk_from.block_hash
            blk_from.reset_hash()
        else:
            assert len(self.cached_block_hash_to_block) == 0, "cached_block_hash_to_block should be empty when prefix caching is disabled"
            assert blk_from.block_hash is None, "block_id_from should not have a hash when prefix caching is disabled"
            assert blk_to.block_hash is None, "block_id_to should not have a hash when prefix caching is disabled"


        blk_to.ref_cnt = blk_from.ref_cnt
        blk_from.ref_cnt = 0
        # 3) 将blk_to从free list中删除
        self.free_block_queue.remove(blk_to)
        self.free_block_queue.append(blk_from)

        return 

    def shrink_block_pool(self, new_block_num: int) -> None:
        """
        根据当前最新的kv block大小，丢弃free_block_queue中所有id超过highest_block_id的块
        """
        assert new_block_num < len(self.blocks), "highest_block_id out of range"
        logger.info(f"before shrink_block_pool, num of blocks: {len(self.blocks)}")
        for block in self.blocks[new_block_num:]:
            assert block.ref_cnt == 0, "block_id should be free"
            assert block.block_hash is None, "block_id should not have a hash"
            self.free_block_queue.remove(block)
        self.blocks = self.blocks[:new_block_num]
        self.num_gpu_blocks = new_block_num
        logger.info(f"after shrink_block_pool, num of blocks: {len(self.blocks)}")

    def extend_block_pool(self, new_block_num: int) -> None:
        """
        根据当前最新的kv block大小，为free_block_queue新增从当前最大id到new_block_num-1的块
        """
        logger.info(f"expand the block pool from {len(self.blocks)} to {new_block_num}")
        old_block_num = len(self.blocks)
        assert new_block_num > old_block_num, "new_block_num should be larger than current block num"
        self.blocks.extend([KVCacheBlock(idx) for idx in range(old_block_num, new_block_num)])

        for block in self.blocks[old_block_num:]:
            self.free_block_queue.append(block)
        self.num_gpu_blocks = new_block_num
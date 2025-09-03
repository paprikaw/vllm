from collections import defaultdict

from vllm.v1.kv_cache_interface import KVCacheSpec
from .single_type_kv_cache_manager import FullAttentionManager, BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.utils import cdiv
from vllm.logger import init_logger

logger = init_logger(__name__)

class DynamicFullAttentionManager(FullAttentionManager):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.block_to_reqs: defaultdict[int, set[str]] = defaultdict(set)


    def save_new_computed_blocks(
            self, request_id: str,
            new_computed_blocks: list[KVCacheBlock]) -> None:
        """
        Copied directly from parent class.
        We add logics to keep track of which block is associated with which request.
        This will be used for block migration.
        """
        if request_id not in self.num_cached_block:
            # A new request.
            req_blocks = self.req_to_blocks[request_id]
            assert len(req_blocks) == 0
            req_blocks.extend(new_computed_blocks)
            self.num_cached_block[request_id] = len(new_computed_blocks)
            for block in new_computed_blocks:
                self.block_to_reqs[block.block_id].add(request_id)
        else:
            # A running request. Should not have new computed blocks.
            assert len(new_computed_blocks) == 0
        self.validate_mappings()
        
    def allocate_new_blocks(self, request_id: str,
                            num_tokens: int) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens` 
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
            self.validate_mappings()
        else:
            new_blocks = self.block_pool.get_new_blocks(
                num_new_blocks * self.num_kv_cache_groups)
            req_blocks.extend(new_blocks)
            for block in new_blocks:
                self.block_to_reqs[block.block_id].add(request_id)
            self.validate_mappings()
            return new_blocks

    def free(self, request_id: str) -> None:
        self.validate_mappings()
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are
        # freed first.
        ordered_blocks = reversed(req_blocks)


        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)
        # For the block, we need to clear the map from block to req
        for block in req_blocks:
            self.block_to_reqs[block.block_id].remove(request_id)
        self.validate_mappings()

    def update_requests_to_new_block_id(self, block_id: int, to_block_id: int) -> None:
        """
        Update the block that is stored in the block list of a request.
        """
        assert self.block_pool.blocks[to_block_id].ref_cnt == 0, "target block should be free"
        assert self.block_pool.blocks[to_block_id].block_hash is None, "target block should be free"
            # Copy to avoid modifying while iterating
        request_ids = list(self.block_to_reqs[block_id])
        if not request_ids:
            return 
        new_blk = self.block_pool.blocks[to_block_id]
        self.validate_mappings()
        # 1. 通过block_id找到对应的request
        # 2. 将对应request中的block_id替换成新的block()
        # 3. 调换block_to_reqs中
        for request_id in request_ids:
            blocks = self.req_to_blocks[request_id]
            idx = next((i for i, blk in enumerate(blocks) if blk.block_id == block_id), None)
            assert idx is not None, f"Block {block_id} not found in request {request_id}, blocks in the request:{blocks}"
            blocks[idx] = new_blk
        self.block_to_reqs[to_block_id] = self.block_to_reqs[block_id]
        self.block_to_reqs.pop(block_id)
    def validate_mappings(self):
        """
        Validate the mappings between block_to_reqs and req_to_blocks.
        """
        is_consistent = True
        # 检查 block_to_reqs 中的每个映射是否在 req_to_blocks 中存在
        for block_id, request_ids in self.block_to_reqs.items():
            for request_id in request_ids:
                if request_id not in self.req_to_blocks:
                    logger.error(f"Inconsistency: request_id {request_id} in block_to_reqs[{block_id}] but not in req_to_blocks")
                    is_consistent = False
                    break 
                
                blocks = self.req_to_blocks[request_id]
                if not any(blk.block_id == block_id for blk in blocks):
                    logger.error(f"Inconsistency: block_id {block_id} in block_to_reqs[{request_id}] but not in req_to_blocks[{request_id}]")
                    is_consistent = False
        if not is_consistent:
            raise ValueError("Inconsistency: block_to_reqs and req_to_blocks are not consistent")
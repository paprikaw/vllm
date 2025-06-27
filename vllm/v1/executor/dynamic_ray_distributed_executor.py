# SPDX-License-Identifier: Apache-2.0
from typing import Tuple

from vllm.v1.executor.ray_distributed_executor import (
    RayDistributedExecutor)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheSpec


logger = init_logger(__name__)


class DynamicRayDistributedExecutor(RayDistributedExecutor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_kv_cache_for_layers(self, rank: int, 
                                        kv_cache_specs: dict[str, KVCacheSpec], 
                                        kv_cache_size: int, 
                                        kv_cache_num_blocks: int) -> None:
        self.collective_rpc("initialize_kv_cache_for_layers", args=(rank, kv_cache_specs, kv_cache_size, kv_cache_num_blocks))

    def get_kv_cache_spec_for_layers(self, rank: int, layer_range: Tuple[int, int]) -> dict[str, KVCacheSpec]:
        output = self.collective_rpc("get_kv_cache_spec_for_layers", args=(rank, layer_range))
        return output[rank]

    def add_layers(self, rank: int, layers: Tuple[int, int]):
        self.collective_rpc("add_layers", args=(rank, layers))
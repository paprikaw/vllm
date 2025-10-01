from dataclasses import dataclass
from typing import List, Dict


@dataclass
class KVCacheSnapshotEntry:
    request_id: str
    num_computed_tokens: int
    # block_ids per KV group, typically length == num_kv_cache_groups
    block_ids: List[List[int]]


@dataclass
class KVCacheSnapshot:
    # Mapping request_id -> entry for O(1) lookup
    entries_by_id: Dict[str, KVCacheSnapshotEntry]

    def get(self, request_id: str) -> KVCacheSnapshotEntry | None:
        return self.entries_by_id.get(request_id)

    def request_ids(self) -> List[str]:
        return list(self.entries_by_id.keys())


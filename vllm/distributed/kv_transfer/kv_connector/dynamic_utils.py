# SPDX-License-Identifier: Apache-2.0
"""
KV cache helper for store.
"""
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import model_aware_kv_ops_helper
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.logger import init_logger
from typing import Dict, Optional, Union, TypedDict, List, Literal
from pydantic import BaseModel, ConfigDict

# Avoid circular import by defining metadata type locally instead of importing
# from dynamic_layer_kv_connector
class KVSynchronizerMetadata(TypedDict, total=False):
    type: str
    dtype: torch.dtype
    shape: torch.Size
    batch_id: int
    layer_ids: List[int]
    num_tokens: int
    layer_id: Optional[int]

logger = init_logger(__name__)

class KVPatchMeta(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal['kv_patch_meta', 'kv_patch_finished']
    id: int
    layer_ids: List[int]
    num_tokens: int
    slot_mapping_dtype: torch.dtype
    slot_mapping_shape: torch.Size
    kv_payload_dtype: torch.dtype
    kv_payload_shape: torch.Size

class KVTensorMeta(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal['kv_tensor']
    layer_to_be_received: set[int]
    layer_id: int
    num_tokens: int
    dtype: torch.dtype
    shape: torch.Size

class KVPatch:
    def __init__(self, meta: KVPatchMeta, kv_payload: torch.Tensor, slot_mapping: torch.Tensor):
        self.meta = meta
        self.kv_payload = kv_payload
        self.slot_mapping = slot_mapping



class kv_synchronizer_helper(model_aware_kv_ops_helper):
    @staticmethod
    def make_metadata_for_layer_tensor(tensor: torch.Tensor,
                       layer_id: Optional[int]) -> KVSynchronizerMetadata:
        return {"dtype": tensor.dtype, "shape": tensor.shape, "layer_id": layer_id}

    @staticmethod
    def make_metadata_for_tensor(tensor: torch.Tensor) -> KVSynchronizerMetadata:
        return {"dtype": tensor.dtype, "shape": tensor.shape}

    def make_metadata_for_stage_batch(self, batch_id: int, layer_ids: list[int], num_tokens: int, tensor: torch.Tensor) -> KVSynchronizerMetadata:
        return {"type": "kv_stage_batch", "dtype": tensor.dtype, "shape": tensor.shape, "batch_id": batch_id, "layer_ids": layer_ids, "num_tokens": num_tokens}

    def extract_kv_patch_from_kv_cache(self, patch_id: int, kv_caches: list[torch.Tensor], layer_ids: list[int], start_layer_id: int, slot_mapping: torch.Tensor) -> KVPatch:
        """Aggregate multiple layers' KV for this stage and send in one shot.

        Prefer passing `kv_caches` and `dest_slot_mapping` only; this method
        will extract per-layer K/V slices from the kv caches. `K_all`/`V_all`
        are accepted for backward compatibility.

        Args:
            peer_rank: destination global rank to send to.
            batch_id: monotonically increasing id for ordering.
            layer_ids: layers contained in this batch; if `kv_caches` is used,
                it should be aligned with this order.
            dest_slot_mapping: int64 tensor [T] with destination slot indices.
            K_all, V_all: optional prebuilt tensors [L, T, H, D].
            hidden: optional tensor [T, hidden_size].
            kv_caches: optional list of per-layer kv_cache tensors; when
                provided, K/V will be extracted internally using
                `slot_mapping`.
        """
        assert kv_caches is not None, (
            "Provide either (K_all,V_all) or kv_caches.")
        # Derive shapes using model config and helper flags.
        assert self.model_executable is not None, "The model_executable should be set"
        num_heads, head_size = self.get_model_args(self.model_executable)
        # Peek the first cache to get dtype/device.
        k0, v0 = self.get_kv_from_cache(kv_caches[0], num_heads,
                                                  head_size)

        # Ensure index lives on same device for fast gather.
        if slot_mapping.device != k0.device:
            slot_mapping = slot_mapping.to(k0.device, non_blocking=True)
        
        # 首先裁剪 slot_mapping，只保留有效部分（>= 0）
        valid_mask = (slot_mapping >= 0)
        num_valid = int(valid_mask.sum().item())
        
        assert num_valid != 0, "No valid slots in slot_mapping, skipping patch creation"
        
        # 只保留有效的 slot_mapping
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        trimmed_slot_mapping = slot_mapping.index_select(0, valid_indices)
        
        logger.info(f"debug-------------------- Original T: {slot_mapping.numel()}, Valid T: {num_valid}")
        
        # Preallocate KV tensor [2, L, T_valid, H, D] 只分配有效token的空间
        L = len(layer_ids)
        T_valid = num_valid
        KV_shape = (2, int(L), int(T_valid), int(num_heads), int(head_size))
        KV_all = torch.empty(KV_shape, dtype=k0.dtype, device=k0.device)

        for idx, layer_id in enumerate(layer_ids):
            local_layer_id = int(layer_id - start_layer_id)
            kv_cache = kv_caches[local_layer_id]
            key_cache, value_cache = self.get_kv_from_cache(
                kv_cache, num_heads, head_size)

            assert trimmed_slot_mapping.device == key_cache.device

            # 直接用裁剪后的 slot_mapping 提取 KV
            gathered_k = torch.index_select(key_cache, 0, trimmed_slot_mapping)
            gathered_v = torch.index_select(value_cache, 0, trimmed_slot_mapping)

            assert gathered_k.device == KV_all.device and gathered_k.dtype == KV_all.dtype
            assert gathered_v.device == KV_all.device and gathered_v.dtype == KV_all.dtype

            # 按 layer_ids 在批次中的顺序写入 KV_all 第二维
            KV_all[0, idx] = gathered_k
            KV_all[1, idx] = gathered_v

        assert KV_all is not None
        assert KV_all.dim() == 5 and KV_all.size(0) == 2, (
            "KV_all must have shape [2, L, T, H, D]")
        _, L2, T2, H, D = KV_all.shape
        assert int(L2) == len(layer_ids) and int(T2) == T_valid

        meta = KVPatchMeta(
            type='kv_patch_meta',
            id=int(patch_id),
            layer_ids=list(map(int, layer_ids)),
            num_tokens=int(T_valid),
            slot_mapping_dtype=trimmed_slot_mapping.dtype,
            slot_mapping_shape=trimmed_slot_mapping.shape,
            kv_payload_dtype=KV_all.dtype,
            kv_payload_shape=KV_all.shape,
        )
        return KVPatch(meta, KV_all, trimmed_slot_mapping)
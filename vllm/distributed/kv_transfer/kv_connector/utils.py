# SPDX-License-Identifier: Apache-2.0
"""
KV cache helper for store.
"""
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class model_aware_kv_ops_helper:

    def __init__(self, config: VllmConfig):
        self.is_deepseek_mla = config.model_config.is_deepseek_mla
        self.use_mla_opt = not envs.VLLM_MLA_DISABLE
        self.tp_size = config.parallel_config.tensor_parallel_size

    def get_model_args(self, model_executable: torch.nn.Module):

        model_config = model_executable.model.config
        self.model_executable = model_executable
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads

        # Deepseek's MLA (Multi-head Latent Attention) uses two different
        # kv_cache shapes based on whether VLLM_MLA_DISABLE is set to 0.
        # When VLLM_MLA_DISABLE=0 (default), forward absorb is applied,
        # resulting in a kv_cache shape of [num_blks, blk_size, 1,
        # kv_lora_rank + qk_rope_head_dim].
        # When VLLM_MLA_DISABLE=1, standard FA is used instead, leading
        # to a kv_cache shape of [2, num_blks, blk_size,
        # num_key_value_heads / tp, qk_nope_head_dim + qk_rope_head_dim].
        # For more details, see vllm/attention/backends/mla/common.py.
        if self.is_deepseek_mla and self.use_mla_opt:
            head_size = model_config.kv_lora_rank + \
                model_config.qk_rope_head_dim
            num_heads = 1
        elif self.is_deepseek_mla and not self.use_mla_opt:
            head_size = model_config.qk_nope_head_dim + \
                model_config.qk_rope_head_dim
        else:
            head_size = getattr(model_config, "head_dim", None)
            if head_size is None:
                head_size = int(hidden_size // num_attention_heads)

        return num_heads, head_size

    def get_kv_from_cache(self, kv_cache, num_heads, head_size):
        if self.is_deepseek_mla and self.use_mla_opt:
            key_cache = kv_cache.reshape(-1, num_heads, head_size)
            value_cache = kv_cache.reshape(-1, num_heads, head_size)
        else:
            key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
            value_cache = kv_cache[1].reshape(-1, num_heads, head_size)
        return key_cache, value_cache

    def put_kv_to_cache(self, model_executable: torch.nn.Module, keys, values,
                        layer, kv_cache, slot_mapping, start_pos, end_pos):

        model_config = model_executable.model.config

        # Resolve layer module if an integer layer id is provided.
        # layer can be either a module or a global layer index.
        layer_module = layer
        if isinstance(layer, int):
            # DynamicModelBase.model is DynamicModel which has .layers
            layer_module = model_executable.model.layers[layer]

        if self.is_deepseek_mla and self.use_mla_opt:
            assert False
            layer_module.self_attn.attn = layer_module.self_attn.mla_attn
            k_c_normed_k_pe = keys.squeeze(1)
            k_c_normed = k_c_normed_k_pe[:, :model_config.kv_lora_rank]
            k_pe = k_c_normed_k_pe[:, model_config.kv_lora_rank:]
            ops.concat_and_cache_mla(
                k_c_normed.to(kv_cache.device),
                k_pe.to(kv_cache.device),
                kv_cache,
                slot_mapping[start_pos:end_pos],
                layer_module.self_attn.attn.kv_cache_dtype,
                layer_module.self_attn.attn._k_scale,
            )
        else:
            # When migrating KV cache between ranks, keys/values may already
            # be in the storage dtype of the destination kv_cache (e.g. fp8 as
            # uint8/float8). In such case, calling reshape_and_cache_flash()
            # would re-quantize already-quantized values and corrupt data.
            # To avoid this, if the input dtype matches the kv cache storage
            # dtype, directly scatter-copy into the flattened cache.
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            tgt_slot = slot_mapping[start_pos:end_pos]
            if tgt_slot.numel() > 0:
                slot_min = int(tgt_slot.min().item())
                slot_max = int(tgt_slot.max().item())
                flat_capacity = int(key_cache.reshape(
                    -1, key_cache.shape[-2], key_cache.shape[-1]).size(0))
                logger.info(
                    "KV patch apply slot range: layer=%s slots=[%s,%s] "
                    "flat_capacity=%s",
                    layer, slot_min, slot_max, flat_capacity)
                if slot_min < 0 or slot_max >= flat_capacity:
                    raise IndexError(
                        "KV patch apply slot mapping is out of bounds: "
                        f"layer={layer}, slot_min={slot_min}, "
                        f"slot_max={slot_max}, flat_capacity={flat_capacity}")
            # tgt_slot = slot_mapping[start_pos:end_pos]
            # Fast path: direct copy when dtypes already match storage.
            # if keys.dtype == key_cache.dtype and values.dtype == value_cache.dtype:
            #     # keys/values shape: [T, H, D]; flatten caches to [T_total, H, D]
            #     H = key_cache.shape[-2]
            #     D = key_cache.shape[-1]
            #     flat_k = key_cache.reshape(-1, H, D)
            #     flat_v = value_cache.reshape(-1, H, D)
            #     # Ensure indices live on same device
            #     if tgt_slot.device != flat_k.device:
            #         tgt_slot = tgt_slot.to(flat_k.device, non_blocking=True)
            #     flat_k.index_copy_(0, tgt_slot, keys)
            #     flat_v.index_copy_(0, tgt_slot, values)
            # else:
                # Fallback: normal path (expects float{16,32} inputs).
            ops.reshape_and_cache_flash(
                keys.to(key_cache.device),
                values.to(value_cache.device),
                key_cache,
                value_cache,
                slot_mapping[start_pos:end_pos],
                layer_module.self_attn.attn.kv_cache_dtype,
                layer_module.self_attn.attn._k_scale,
                layer_module.self_attn.attn._v_scale,
            )
    def flexi_put_kv_to_cache(self, model_executable: torch.nn.Module, page_meta, keys, values,key_cache_ptr, value_cache_ptr, layer, slot_mapping):
        # Resolve layer module if an integer layer id is provided.
        # layer can be either a module or a global layer index.
        layer_module = layer
        if isinstance(layer, int):
            # DynamicModelBase.model is DynamicModel which has .layers
            layer_module = model_executable.model.layers[layer]

        if self.is_deepseek_mla and self.use_mla_opt:
            assert False
        else:
            # When migrating KV cache between ranks, keys/values may already
            # be in the storage dtype of the destination kv_cache (e.g. fp8 as
            # uint8/float8). In such case, calling reshape_and_cache_flash()
            # would re-quantize already-quantized values and corrupt data.
            # To avoid this, if the input dtype matches the kv cache storage
            # dtype, directly scatter-copy into the flattened cache.
            # tgt_slot = slot_mapping[start_pos:end_pos]
            # Fast path: direct copy when dtypes already match storage.
            # if keys.dtype == key_cache.dtype and values.dtype == value_cache.dtype:
            #     # keys/values shape: [T, H, D]; flatten caches to [T_total, H, D]
            #     H = key_cache.shape[-2]
            #     D = key_cache.shape[-1]
            #     flat_k = key_cache.reshape(-1, H, D)
            #     flat_v = value_cache.reshape(-1, H, D)
            #     # Ensure indices live on same device
            #     if tgt_slot.device != flat_k.device:
            #         tgt_slot = tgt_slot.to(flat_k.device, non_blocking=True)
            #     flat_k.index_copy_(0, tgt_slot, keys)
            #     flat_v.index_copy_(0, tgt_slot, values)
            # else:
                # Fallback: normal path (expects float{16,32} inputs).
            ops.flexi_reshape_and_cache_flash(key=keys,
                                              value=values,
                                              key_cache_ptr=key_cache_ptr,
                                              value_cache_ptr=value_cache_ptr,
                                              key_cache_meta=page_meta,
                                              value_cache_meta=page_meta,
                                              slot_mapping=slot_mapping,
                                              kv_cache_dtype=layer_module.self_attn.attn.kv_cache_dtype,
                                              k_scale=layer_module.self_attn.attn._k_scale,
                                              v_scale=layer_module.self_attn.attn._v_scale,
            )

# SPDX-License-Identifier: Apache-2.0
import threading
import copy
import gc
import time
import weakref
from typing import TYPE_CHECKING, Optional, Union, Tuple
import sys
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from vllm.attention.layer import Attention
from vllm.attention import AttentionType
from vllm.config import (get_layers_from_vllm_config)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils import (LazyLoader)
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheSpec, KVCacheConfig,
                                        SlidingWindowSpec)
from vllm.v1.utils import extract_layer_index
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.utils import bind_kv_cache
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.model_executor.models.dynamic_qwen3 import DynamicQwen3ForCausalLM
from vllm.model_executor.model_loader.dynamic_qwen3_loader import CustomModelLoader
from collections import defaultdict
from vllm.config import set_current_vllm_config
from threading import Lock
from vllm.v1.spec_decode.eagle import EagleProposer
from bitarray import bitarray
from vllm.v1.core.dynamic_kv_cache_utils import compact_cache
from vllm.distributed.kv_transfer.kv_connector.dynamic_kv_synchronizer import DynamicKVSynchronizer

if TYPE_CHECKING:
    import xgrammar as xgr

    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


class DynamicGPUModelRunner(GPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_lock = Lock()
    # def load_model(self)->float:
    #     super().load_model()
    #     return self.model_memory_usage
    def initialize_kv_cache_for_layers(self, 
            kv_cache_specs: dict[str, KVCacheSpec],
            kv_cache_size: int,
            kv_cache_num_blocks: int,
            layers: Tuple[int, int],
            ) -> None:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"before initialize_kv_cache_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        assert len(self.attn_backends) == 1, "Only one attention backend is supported for now"
        kv_caches: dict[str, torch.Tensor] = {} 
        logger.info(f"kv_cache_specs: {kv_cache_specs}")
        for layer_name, kv_cache_spec in kv_cache_specs.items():
                if extract_layer_index(layer_name) not in range(layers[0], layers[1]+1):
                    continue
                assert kv_cache_size % kv_cache_spec.page_size_bytes == 0
                num_blocks = kv_cache_size // kv_cache_spec.page_size_bytes
                assert num_blocks >= kv_cache_num_blocks
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = self.attn_backends[0].get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    logger.info(f"layer_name: {layer_name}, kv_cache_shape: {kv_cache_shape}, dtype: {dtype}")
                    kv_caches[layer_name] = torch.zeros(kv_cache_shape,
                                                        dtype=dtype,
                                                        device=self.device)
                    self.bind_layer_kv_tensor(extract_layer_index(layer_name), kv_caches[layer_name])
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")
                # Added the kv cache spec to kv cache config.
                assert len(self.kv_cache_config.kv_cache_groups) == 1
                self.kv_cache_config.kv_cache_groups[0].layer_names.append(layer_name)
        if self.speculative_config and self.speculative_config.use_eagle():
            raise NotImplementedError("Eagle is not supported for dynamic weights")

        self._update_start_layer()
        if has_kv_transfer_group():
            raise NotImplementedError("KV transfer group is not supported for dynamic weights")

        logger.info(f"after initialize_kv_cache_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        return

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        layer_config: Tuple[int, int],
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        with self.forward_lock:
            # logger.info("start to execute model in gpu model runner")
            if not isinstance(self.model, DynamicQwen3ForCausalLM):
                raise AssertionError(f"model is not a DynamicQwen3ForCausalLM: {self.model.__class__.__name__}")
            self.model.set_sched_layers(layer_config[0], layer_config[1])
            result = super().execute_model(scheduler_output, intermediate_tensors)
            return result

    def profile_run(self) -> None: 
        """
        Profile run can be run during the model is running,
        Needed to acquire a lock
        """
        with self.forward_lock:
            super().profile_run()

    def get_kv_cache_spec_for_layers(self, layer_range: Tuple[int, int]) -> dict[str, KVCacheSpec]:
        """
        Get the KV cache spec for the given layers.
        Copy from get_kv_cache_spec() but only return the KV cache spec for the given layers.
        """
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            if extract_layer_index(layer_name) not in range(layer_range[0], layer_range[1]+1):
                continue
            # TODO: Support other attention modules, e.g., cross-attention
            if attn_module.attn_type == AttentionType.DECODER:
                if attn_module.sliding_window is not None:
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        sliding_window=attn_module.sliding_window,
                        use_mla=use_mla)
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        use_mla=use_mla)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")
        return kv_cache_spec

    def get_layer_name_for_index(self, layer_index: int) -> str:
        """Return the layer name in forward_context for a given global layer index.

        Raises KeyError if not found or ambiguous.
        """
        forward_context = self.vllm_config.compilation_config.static_forward_context
        candidates = [name for name in forward_context.keys()
                      if extract_layer_index(name) == layer_index]
        if not candidates:
            raise KeyError(f"No layer name found for index {layer_index} in forward_context")
        if len(candidates) > 1:
            raise KeyError(f"Multiple layer names found for index {layer_index}: {candidates}")
        return candidates[0]

    def bind_layer_kv_tensor(self, layer_index: int, kv_tensor: torch.Tensor) -> None:
        """Bind a single layer's KV tensor to runner caches and forward context.

        - 更新本 runner 的 `self.kv_caches`
        - 将 forward context 中对应 Attention 的 `kv_cache[ve]` 指向该张量
        - 如有必要，补齐 kv_cache_config 的 layer_names，确保后续 attn_metadata 构建覆盖到该层

        线程安全：内部获取 forward_lock。
        """
        if not isinstance(self.model, DynamicQwen3ForCausalLM):
            raise AssertionError(
                f"model is not a DynamicQwen3ForCausalLM: {self.model.__class__.__name__}")
        # Determine local index in runner kv cache list
        start_layer: int = self.model.model.start_layer
        end_layer: int = self.model.model.end_layer
        assert layer_index >= start_layer and layer_index < end_layer, f"Layer {layer_index} outside of current model range [{start_layer}, {end_layer}]"

        local_index = layer_index - start_layer
        assert local_index < len(self.kv_caches), f"Local index {local_index} is out of range, kv_tensor length: {len(self.kv_caches)}"
        logger.info(f"bind kv tensor for {layer_index}, local_index={local_index}, kv_tensor length: {len(self.kv_caches)}")
        # Basic sanity: non-empty tensor
        assert isinstance(kv_tensor, torch.Tensor) and kv_tensor.numel() > 0, (
            f"Binding empty KV tensor for layer {layer_index}")
        self.kv_caches[local_index] = kv_tensor
        # Bind to forward context
        layer_name: str = self.get_layer_name_for_index(layer_index)
        fctx: dict[str, "Attention"] = \
            self.vllm_config.compilation_config.static_forward_context
        if layer_name not in fctx:
            raise KeyError(
                f"No attention layer named {layer_name} in forward_context.")

        attn_module = fctx[layer_name]
        attn_module.kv_cache = [kv_tensor]

        # 3) 补齐 kv_cache_config 的 layer_names，保证后续 attn_metadata 覆盖
        group = self.kv_cache_config.kv_cache_groups[0]
        if layer_name not in group.layer_names:
            # 按 layer_index 位置插入，保持有序
            insert_idx = len(group.layer_names)
            target_idx = extract_layer_index(layer_name)
            for i, name in enumerate(group.layer_names):
                if extract_layer_index(name) > target_idx:
                    insert_idx = i
                    break
            group.layer_names.insert(insert_idx, layer_name)


    def has_layer(self, layer_index: int) -> bool:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"debug: ---------------------has_layer: {layer_index} in range {self.model.model.start_layer} to {self.model.model.end_layer}")
        return layer_index in range(self.model.model.start_layer, self.model.model.end_layer)

    def add_layers(self, layers_list: list[Tuple[int, int]]) -> None:
        if not isinstance(self.model, DynamicQwen3ForCausalLM):
            raise AssertionError(f"model is not a DynamicQwen3ForCausalLM: {self.model.__class__.__name__}")
        available_memory = torch.cuda.mem_get_info()[0]
        num_added_layers = sum([layer[1] - layer[0] + 1 for layer in layers_list])
        assert available_memory > num_added_layers * self.model.get_layer_weight_size(), \
            f"Available memory: {available_memory} is not enough for {num_added_layers} layers"

        model: DynamicQwen3ForCausalLM = self.model
        loader = CustomModelLoader(self.vllm_config.load_config)
        with set_current_vllm_config(self.vllm_config):
            for layers in layers_list:
                assert len(layers) == 2
                loader.load_qwen3_layers(self.vllm_config, self.model_config, layers, model)

    def reinitialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Copied from initialize_kv_cache() but only called when reinitialize the kv cache.
        The only difference is that we don't need to initialize the attention backend again.
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")
        self.kv_cache_config = kv_cache_config
        kv_caches: dict[str, torch.Tensor] = {}
        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                assert tensor_config.size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                assert num_blocks >= kv_cache_config.num_blocks
                # Handle known KV cache spec types explicitly for clarity.
                if isinstance(kv_cache_spec, (FullAttentionSpec, SlidingWindowSpec)):
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    kv_caches[layer_name] = torch.zeros(
                        kv_cache_shape, dtype=dtype, device=self.device)
                elif isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    kv_caches[layer_name] = torch.zeros(
                        kv_cache_shape, dtype=dtype, device=self.device)
                else:
                    raise ValueError(
                        f"Unknown KV cache spec type: {type(kv_cache_spec).__name__}.")

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)
        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def get_single_kv_tensor_size(self) -> int:
        """Return the size in bytes of a single layer's KV cache tensor.

        If KV caches are not initialized yet, returns 0.
        """
        assert hasattr(self, "kv_caches") and self.kv_caches and isinstance(self.kv_caches[0], torch.Tensor) 

        t: torch.Tensor = self.kv_caches[0]
        return int(t.numel() * t.element_size())

    def remove_layers(self, layers_list: list[Tuple[int, int]]) -> None:
        if not isinstance(self.model, DynamicQwen3ForCausalLM):
            raise AssertionError(f"model is not a DynamicQwen3ForCausalLM: {self.model.__class__.__name__}")
        logger.info(f"before delete_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        for layers in layers_list:
            logger.info(f"Deleting layers {layers}")
            self.model.delete_layers(layers)

        deleted_layers = set(layer for layers in layers_list for layer in range(layers[0], layers[1]+1))
        logger.info(f"deleted layer set {deleted_layers}")
        # Delete the layer from forward context
        self.vllm_config.compilation_config.static_forward_context = {
            layer_name: attn_module for layer_name, attn_module in self.vllm_config.compilation_config.static_forward_context.items()
            if extract_layer_index(layer_name) not in deleted_layers
        }
        for i in range(len(self.kv_caches)):
            logger.info(f"kv_caches[{i}] shape: {self.kv_caches[i].shape}")
        # gc.collect()
        # torch.cuda.empty_cache()

    
    def release_kv_cache_for_layers(self, layers_list: list[Tuple[int, int]]) -> None:
        '''
        目前release_kv_cache依赖于model.start_layers来进行kv cache list的删除。
        因此只能先release kv cache再删除layers
        '''
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"before release_kv_cache_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

        # Delete the kv cache from kv_cache list
        start_layer = self.model.model.start_layer
        self.kv_caches = [
            kv_cache for idx, kv_cache in enumerate(self.kv_caches) 
            if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
        ]

        # Delete layer name from kv_cache_config
        deleted_layers = set(layer for layers in layers_list for layer in range(layers[0], layers[1]+1))
        group = self.kv_cache_config.kv_cache_groups[0]
        group.layer_names = [
            name for name in group.layer_names
            if extract_layer_index(name) not in deleted_layers
        ]

        # logger.info(f"after release_kv_cache_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def release_kv_cache(self) -> None:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"before release_kv_cache: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

        # 清除引用
        self.kv_caches = []
        self.kv_cache_config.kv_cache_groups[0].layer_names = []
        forward_context = self.vllm_config.compilation_config.static_forward_context 
        for _, attn_module in forward_context.items():
            attn_module.kv_cache = [torch.tensor([]) for _ in range(self.vllm_config.parallel_config.pipeline_parallel_size)]
        # 强制回收 + 清空缓存
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"after release_kv_cache: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def compact_kv_cache(self, compacted_length: int, bitmap: bitarray) -> None:
        time_start = time.time()
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"start to compact kv cache for layers {self.model.model.start_layer} to {self.model.model.end_layer}")
        num_blocks = len(bitmap)
        assert num_blocks == len(self.kv_caches[0][0]), f"bitmap length mismatch: num_blocks: {num_blocks} != kv_cache_tensor_length: {len(self.kv_caches[0][0])}"

        def is_used(idx):
            return bitmap[idx]
        compact_cache(self._migrate_block, is_used, compacted_length, num_blocks)
        time_end = time.time()
        logger.info(f"compacted kv cache in {time_end - time_start} seconds")

    def resize_kv_cache(self, new_length: int) -> None:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)

        logger.info(f"resizing kv cache from {len(self.kv_caches[0][0])} to {new_length}")
        logger.info(f"before resize kv cache, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        time_start = time.time()
        forward_context = self.vllm_config.compilation_config.static_forward_context
        kv, kv_length, T, H, Dh = self.kv_caches[0].shape
        logger.info(f"num of kv tensors{len(self.kv_caches)}")
        for layer_name, attn_module in forward_context.items():
            logger.info(f"resizing kv cache for layer {layer_name}")
            idx = extract_layer_index(layer_name) - self.model.model.start_layer
            cache = self.kv_caches[idx]
            # 使用 zeros 而不是 empty 来避免未初始化的数据导致错误生成EOS
            tmp_cache = torch.zeros((kv, new_length, T, H, Dh), device=self.device, dtype=cache.dtype)
            logger.info(f"tmp_cache shape: {tmp_cache.shape}, cache shape: {cache.shape}")
            if new_length > kv_length:
                # 只复制旧的有效部分，新增的部分已经是0了
                tmp_cache[:, :kv_length, ...].copy_(cache[:, :kv_length, ...])
            else:
                tmp_cache[:, :new_length, ...].copy_(cache[:, :new_length, ...])

            self.kv_caches[idx] = tmp_cache
            attn_module.kv_cache = [tmp_cache]
        time_end = time.time()
        logger.info(f"resize kv cache in {time_end - time_start} seconds")
        # torch.cuda.empty_cache()
        # time_tensor_end = time.time()
        # # logger.info(f"allocate a 1GB tensor in {time_tensor_end - time_end} seconds")
        # logger.info(f"after empty cache, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        # logger.info(f"empty cache in {time_tensor_end - time_end} seconds")
    def _migrate_block(self, new_block_id: int, old_block_id: int):
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        assert len(self.kv_caches) != 0
        for cache in self.kv_caches:
            # loop for k and v tensor
            for i in range(2):
                cache[i][new_block_id].copy_(cache[i][old_block_id])
                # 这里我们只将旧的搬迁到新的，我们不对旧的block数据制0
    def _update_start_layer(self) -> None:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        forward_context: dict[str, "Attention"] = self.vllm_config.compilation_config.static_forward_context
        new_start = min(extract_layer_index(layer_name) for layer_name in forward_context.keys())
        start_layer = self.model.model.start_layer

        # Sync model start if left-padded
        if new_start != start_layer:
            self.model.model.start_layer = new_start

# def bind_kv_cache_for_layers(
#     start_layer_index: int,
#     kv_caches: dict[str, torch.Tensor],
#     forward_context: dict[str, "Attention"],
#     runner_kv_caches: list[torch.Tensor],
# ) -> int:
#     """
#     Bind the allocated KV cache of layers to ModelRunner and forward context.
#     """
#     # Bind kv_caches to ModelRunner (support discrete additions with padding)
#     # Convert kv_caches dict to a mapping indexed by global layer index.
#     index2name = defaultdict(list)
#     for layer_name in kv_caches:
#         layer_index = extract_layer_index(layer_name)
#         index2name[layer_index].append(layer_name)

#     # Extend the current runner's kv cache list
#     if index2name:
#         min_index, max_index = min(index2name.keys()), max(index2name.keys())
#         if min_index < start_layer_index:
#             pad_left = start_layer_index - min_index
#             runner_kv_caches[:0] = [torch.tensor([])] * pad_left
#             start_layer_index = min_index

#         max_local_index = max_index - start_layer_index
#         if max_local_index >= len(runner_kv_caches):
#             pad_right = max_local_index + 1 - len(runner_kv_caches)
#             runner_kv_caches.extend([torch.tensor([])] * pad_right)

#     for layer_index in sorted(index2name.keys()):
#         layer_names = index2name[layer_index]
#         if len(layer_names) > 1:
#             # One typical case is encoder-decoder model, e.g., bart.
#             # The cross attention and self attention in the same decoder layer
#             # has different layer_name but the same layer_index.
#             raise NotImplementedError
#         layer_name = layer_names[0]
#         local_index = layer_index - start_layer_index
#         runner_kv_caches[local_index] = kv_caches[layer_name]

#     # Bind kv_caches to forward context
#     for layer_name, kv_cache in kv_caches.items():
#         # NOTE: Use list because of v0 PP virtual engine.
#         forward_context[layer_name].kv_cache = [kv_cache]
#     # Return possibly updated start_layer_index (if left-padded)
#     return start_layer_index

def _debug_tensor_referrers(tensor, max_referrers=10):
    """打印引用该 tensor 的对象及可能的变量名或容器索引。"""
    logger.info("=" * 60)
    logger.info(f"🧠 Tensor device: {tensor.device}, shape: {tensor.shape}, dtype: {tensor.dtype}")
    logger.info(f"📦 Tensor refcount: {sys.getrefcount(tensor) - 1}")

    referrers = gc.get_referrers(tensor)
    logger.info(f"🔎 Found {len(referrers)} referrers for tensor (showing up to {max_referrers}):")

    for i, ref in enumerate(referrers[:max_referrers]):
        logger.info(f"{i}: type={type(ref)}, id={id(ref)}")

        # 如果是 dict，例如 locals()/globals() 或模块字段
        if isinstance(ref, dict):
            for k, v in ref.items():
                if v is tensor:
                    logger.info(f"    ↳ 📌 Found in dict key: {repr(k)}")

        # 如果是 list/tuple/set 等容器
        elif isinstance(ref, (list, tuple, set)):
            for idx, item in enumerate(ref):
                if item is tensor:
                    logger.info(f"    ↳ 🧩 Found in {type(ref).__name__} at index {idx}")

        # 如果是某个实例对象
        elif hasattr(ref, "__class__"):
            logger.info(f"    ↳ 🧷 Possibly from instance of: {ref.__class__.__name__}")
    logger.info("=" * 60)

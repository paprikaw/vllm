# SPDX-License-Identifier: Apache-2.0
from hmac import new
import threading
import copy
import gc
import time
import weakref
from typing import TYPE_CHECKING, Optional, Union, Tuple
import sys
import humanize
from matplotlib.pylab import dtype
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from vllm._custom_ops import flexi_reshape_and_cache_flash
from vllm.attention.dynamic_layer import FlexiAttention
from vllm.attention.layer import Attention
from vllm.attention import AttentionType
from vllm.config import (get_layers_from_vllm_config)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.dynamic_utils import FlexiKVTensorMeta
from vllm.distributed.parallel_state import get_pp_group
from vllm.sampling_params import SamplingType
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.v1.utils import dynamic_bind_kv_cache, dynamic_flexi_bind_kv_cache, human_readable_size, human_readable_duration
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils import (LazyLoader)
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheSpec, KVCacheConfig,
                                        SlidingWindowSpec)
from vllm.v1.utils import extract_layer_index
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.utils import dynamic_bind_kv_cache, dynamic_bind_single_kv_tensor, dynamic_flexi_bind_single_kv_tensor
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.forward_context import get_forward_context
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.model_executor.models.dynamic_qwen3 import DynamicQwen3ForCausalLM
from vllm.model_executor.model_loader.dynamic_qwen3_loader import CustomModelLoader
from collections import defaultdict
from vllm.config import set_current_vllm_config
from threading import Lock
from vllm.v1.spec_decode.eagle import EagleProposer
from bitarray import bitarray
from vllm.v1.core.dynamic_kv_cache_utils import compact_cache_with_record
from vllm.distributed.kv_transfer.kv_connector.dynamic_kv_synchronizer import DynamicKVSynchronizer
from vllm.v1.worker.utils import get_flexi_kv_cache
from vllm.vllm_flash_attn.flash_attn_interface import prepare_flexi_kv_ptrs, free_flexi_kv_ptrs

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
        self.key_caches: list[list[torch.Tensor]] = []
        self.value_caches: list[list[torch.Tensor]] = []
        self.key_cache_ptrs: list[int] = []
        self.value_cache_ptrs: list[int] = []
    # def load_model(self)->float:
    #     super().load_model()
    #     return self.model_memory_usage
    def initialize_kv_cache_for_layers(self, 
            kv_cache_specs: dict[str, KVCacheSpec],
            kv_cache_size: int,
            kv_cache_num_blocks: int,
            kv_synchronizer: DynamicKVSynchronizer,
            layers: Tuple[int, int],
            ) -> None:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"before initialize_kv_cache_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        assert len(self.attn_backends) == 1, "Only one attention backend is supported for now"
        kv_caches: dict[str, torch.Tensor] = {} 
        logger.info(f"kv_cache_specs: {kv_cache_specs}")
        for layer_name, kv_cache_spec in kv_cache_specs.items():
                layer_index = extract_layer_index(layer_name)
                if layer_index not in range(layers[0], layers[1]+1):
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
                    dynamic_bind_single_kv_tensor(
                        layer_index=layer_index,
                        start_layer=self.model.model.start_layer,
                        end_layer=self.model.model.end_layer,
                        forward_context=self.vllm_config.compilation_config.static_forward_context,
                        kv_synchronizer=kv_synchronizer,
                        runner=self,
                        kv_tensor=kv_caches[layer_name],
                    )
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
            try:
                result = super().execute_model(scheduler_output, intermediate_tensors)
            except Exception as e:
                logger.exception(f"Error in execute_model: {e}")
                for request in scheduler_output.scheduled_new_reqs:
                    for block_id in request.block_ids:
                        logger.info(f"request {request.request_id} block id list: {block_id}")
                time.sleep(1)
                raise
            return result

    def profile_run(self) -> None: 
        """
        Profile run can be run during the model is running,
        Needed to acquire a lock
        """
        with self.forward_lock:
            super().profile_run()

    @torch.inference_mode()
    def initialize_intermediate_states(
        self,
    ) -> None:

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        num_tokens = self.max_num_tokens
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = max_num_reqs if num_tokens >= max_num_reqs else num_tokens
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        if not get_pp_group().is_first_rank:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = (
                    self.model.make_empty_intermediate_tensors(
                        batch_size=self.max_num_tokens,
                        dtype=self.model_config.dtype,
                        device=self.device))

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


    # def bind_layer_kv_tensor(self, layer_index: int, kv_tensor: torch.Tensor) -> None:
    #     """Bind a single layer's KV tensor to runner caches and forward context.

    #     - 更新本 runner 的 `self.kv_caches`
    #     - 将 forward context 中对应 Attention 的 `kv_cache[ve]` 指向该张量
    #     - 如有必要，补齐 kv_cache_config 的 layer_names，确保后续 attn_metadata 构建覆盖到该层

    #     线程安全：内部获取 forward_lock。
    #     """
    #     if not isinstance(self.model, DynamicQwen3ForCausalLM):
    #         raise AssertionError(
    #             f"model is not a DynamicQwen3ForCausalLM: {self.model.__class__.__name__}")
    #     # Determine local index in runner kv cache list
    #     start_layer: int = self.model.model.start_layer
    #     end_layer: int = self.model.model.end_layer
    #     assert layer_index >= start_layer and layer_index < end_layer, f"Layer {layer_index} outside of current model range [{start_layer}, {end_layer}]"

    #     local_index = layer_index - start_layer
    #     assert local_index < len(self.kv_caches), f"Local index {local_index} is out of range, kv_tensor length: {len(self.kv_caches)}"
    #     logger.info(f"bind kv tensor for {layer_index}, local_index={local_index}, kv_tensor length: {len(self.kv_caches)}")
    #     # Basic sanity: non-empty tensor
    #     assert isinstance(kv_tensor, torch.Tensor) and kv_tensor.numel() > 0, (
    #         f"Binding empty KV tensor for layer {layer_index}")
    #     self.kv_caches[local_index] = kv_tensor
    #     # Bind to forward context
    #     layer_name: str = self.get_layer_name_for_index(layer_index)
    #     fctx: dict[str, "Attention"] = \
    #         self.vllm_config.compilation_config.static_forward_context
    #     if layer_name not in fctx:
    #         raise KeyError(
    #             f"No attention layer named {layer_name} in forward_context.")

    #     attn_module = fctx[layer_name]
    #     attn_module.kv_cache = [kv_tensor]

    #     # 3) 补齐 kv_cache_config 的 layer_names，保证后续 attn_metadata 覆盖
    #     group = self.kv_cache_config.kv_cache_groups[0]
    #     if layer_name not in group.layer_names:
    #         # 按 layer_index 位置插入，保持有序
    #         insert_idx = len(group.layer_names)
    #         target_idx = extract_layer_index(layer_name)
    #         for i, name in enumerate(group.layer_names):
    #             if extract_layer_index(name) > target_idx:
    #                 insert_idx = i
    #                 break
    #         group.layer_names.insert(insert_idx, layer_name)

    # def flexi_bind_layer_kv_tensor(self, layer_index: int, kv_tensor: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    #     """Bind a single layer's KV tensor to runner caches and forward context.

    #     - 更新本 runner 的 `self.kv_caches`
    #     - 将 forward context 中对应 Attention 的 `kv_cache[ve]` 指向该张量
    #     - 如有必要，补齐 kv_cache_config 的 layer_names，确保后续 attn_metadata 构建覆盖到该层

    #     线程安全：内部获取 forward_lock。
    #     """
    #     if not isinstance(self.model, DynamicQwen3ForCausalLM):
    #         raise AssertionError(
    #             f"model is not a DynamicQwen3ForCausalLM: {self.model.__class__.__name__}")
    #     # Determine local index in runner kv cache list
    #     start_layer: int = self.model.model.start_layer
    #     end_layer: int = self.model.model.end_layer
    #     assert layer_index >= start_layer and layer_index < end_layer, f"Layer {layer_index} outside of current model range [{start_layer}, {end_layer}]"
    #     local_index = layer_index - start_layer
    #     assert len(self.key_caches) == len(self.value_caches) == len(self.key_cache_ptrs) == len(self.value_cache_ptrs), "key_caches and value_caches length mismatch"
    #     assert local_index < len(self.key_caches), f"Local index {local_index} is out of range, key_cache length: {len(self.key_caches)}"
    #     logger.info(f"bind kv tensor for {layer_index}, local_index={local_index}, kv_tensor length: {len(self.key_caches)}")

    #     key_cache_list, value_cache_list, key_cache_ptr, value_cache_ptr = self._get_flexi_kv_cache_from_gathered_kv_tensor(slot_mapping, kv_tensor)

    #     self.key_caches[local_index] = key_cache_list
    #     self.value_caches[local_index] = value_cache_list
    #     self.key_cache_ptrs[local_index] = key_cache_ptr
    #     self.value_cache_ptrs[local_index] = value_cache_ptr

    #     # Bind to forward context
    #     layer_name: str = self.get_layer_name_for_index(layer_index)
    #     fctx: dict[str, "Attention"] = \
    #         self.vllm_config.compilation_config.static_forward_context
    #     if layer_name not in fctx:
    #         raise KeyError(
    #             f"No attention layer named {layer_name} in forward_context.")
    #     attn_module = fctx[layer_name]
    #     assert isinstance(attn_module, FlexiAttention), f"Attention module for layer {layer_name} is not FlexiAttention"
    #     attn_module.key_cache = key_cache_list
    #     attn_module.value_cache = value_cache_list
    #     attn_module.key_dev_ptr = key_cache_ptr
    #     attn_module.value_dev_ptr = value_cache_ptr

    #     # 3) 补齐 kv_cache_config 的 layer_names，保证后续 attn_metadata 覆盖
    #     group = self.kv_cache_config.kv_cache_groups[0]
    #     if layer_name not in group.layer_names:
    #         # 按 layer_index 位置插入，保持有序
    #         insert_idx = len(group.layer_names)
    #         target_idx = extract_layer_index(layer_name)
    #         for i, name in enumerate(group.layer_names):
    #             if extract_layer_index(name) > target_idx:
    #                 insert_idx = i
    #                 break
            # group.layer_names.insert(insert_idx, layer_name)

    def has_layer(self, layer_index: int) -> bool:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"debug: ---------------------has_layer: {layer_index} in range {self.model.model.start_layer} to {self.model.model.end_layer}")
        return layer_index in range(self.model.model.start_layer, self.model.model.end_layer)

    def add_layers(self, layers_list: list[Tuple[int, int]]) -> None:
        if not isinstance(self.model, DynamicQwen3ForCausalLM):
            raise AssertionError(f"model is not a DynamicQwen3ForCausalLM: {self.model.__class__.__name__}")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        available_memory = torch.cuda.mem_get_info()[0]
        num_added_layers = sum([layer[1] - layer[0] + 1 for layer in layers_list])
        time_wait = 0
        while available_memory < num_added_layers * self.model.get_layer_weight_size() and time_wait < 10:
            time_wait += 1
            time.sleep(0.1)
            gc.collect()
            torch.cuda.empty_cache()
            available_memory = torch.cuda.mem_get_info()[0]
            logger.info(f"waiting for available memory to be enough, time_wait: {time_wait}, available_memory: {available_memory}")
        assert available_memory > num_added_layers * self.model.get_layer_weight_size(), \
            f"Available memory: {human_readable_size(available_memory)} is not enough for {num_added_layers} layers"

        model: DynamicQwen3ForCausalLM = self.model
        loader = CustomModelLoader(self.vllm_config.load_config)
        with set_current_vllm_config(self.vllm_config):
            for layers in layers_list:
                assert len(layers) == 2
                loader.load_qwen3_layers(self.vllm_config, self.model_config, layers, model)

    def reinitialize_kv_cache(self, kv_cache_config: KVCacheConfig, kv_synchronizer: DynamicKVSynchronizer) -> None:
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

        dynamic_bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
            kv_synchronizer=kv_synchronizer)
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
        logger.info(f"after delete_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def dynamic_initialize_kv_cache(self, 
                                    kv_cache_config: KVCacheConfig, 
                                    dynamic_kv_synchronizer: DynamicKVSynchronizer,
                                    num_blocks: int
                                    ) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")
        self.kv_cache_config = kv_cache_config
        self.initialize_attn_backend(kv_cache_config)

        kv_caches: dict[str, torch.Tensor] = {}

        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                assert num_blocks >= kv_cache_config.num_blocks
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    kv_caches[layer_name] = torch.zeros(kv_cache_shape,
                                                        dtype=dtype,
                                                        device=self.device)
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")
                logger.info(f"kv_caches shape: {kv_cache_shape}")
        
        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        dynamic_bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
            dynamic_kv_synchronizer
            )
        del kv_caches
        logger.info(f"debug---------------- init kv cache done, kv block num: {len(self.kv_caches[0][0])}")
        if has_kv_transfer_group():
            assert False
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def dynamic_initialize_kv_cache_flexi(self, kv_cache_config: KVCacheConfig, kv_synchronizer: DynamicKVSynchronizer, num_blocks: int) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")
        self.kv_cache_config = kv_cache_config
        self.initialize_attn_backend(kv_cache_config)

        kv_caches: dict[str, torch.Tensor] = {} 
        key_cache_ptrs: dict[str, int] = {}
        value_cache_ptrs: dict[str, int] = {}
        key_caches: dict[str, list[torch.Tensor]] = {}
        value_caches: dict[str, list[torch.Tensor]] = {}
        assert len(kv_cache_config.kv_cache_groups) == 1, "Only one KV cache group is supported."
        assert len(self.attn_backends) == 1, "Only one attention backend is supported."

        kv_cache_group = kv_cache_config.kv_cache_groups[0]
        kv_cache_spec = kv_cache_group.kv_cache_spec
        assert isinstance(kv_cache_spec, AttentionSpec), "KV cache spec must be an AttentionSpec."
        self.kv_cache_dtype = kv_cache_spec.dtype
        kv_cache_shape = self.attn_backends[0].get_kv_cache_shape(
            num_blocks, kv_cache_spec.block_size,
            kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
        self.kv_cache_shape = kv_cache_shape
        assert num_blocks >= kv_cache_config.num_blocks
        assert len(kv_cache_shape) == 5 # (2, nkvblocks, blockdim, n_head, headdim)
        block_shape = kv_cache_shape[2:]

        start_time = time.time()
        for layer_name in kv_cache_group.layer_names:
            key_caches[layer_name], value_caches[layer_name] = get_flexi_kv_cache(
                kv_cache_shape[1], block_shape, self.kv_cache_dtype, self.device)
            key_cache_ptrs[layer_name], value_cache_ptrs[layer_name] = prepare_flexi_kv_ptrs(key_caches[layer_name], value_caches[layer_name])
        
        logger.info(f"time to intialize kv blocks:{human_readable_duration(time.time() - start_time)}")
        logger.info(f"key cache shape: {key_caches[kv_cache_group.layer_names[0]][0].shape}, value cache shape:{value_caches[kv_cache_group.layer_names[0]][0].shape}")
        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)
        
        dynamic_flexi_bind_kv_cache(
            key_caches,
            value_caches,
            key_cache_ptrs,
            value_cache_ptrs,
            self.vllm_config.compilation_config.static_forward_context,
            kv_synchronizer,
            self.key_caches,
            self.value_caches,
            self.key_cache_ptrs,
            self.value_cache_ptrs
        )
        del kv_caches
        if has_kv_transfer_group():
            assert False
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def flexi_release_kv_cache_for_layers(self, layers_list: list[Tuple[int, int]]) -> None:
        '''
        目前release_kv_cache依赖于model.start_layers来进行kv cache list的删除。
        因此只能先release kv cache再删除layers
        '''
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"before release_kv_cache_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

        # Delete the kv cache from kv_cache list
        start_layer = self.model.model.start_layer
        self.key_caches = [
            key_cache for idx, key_cache in enumerate(self.key_caches) 
            if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
        ]
        self.value_caches = [
            value_cache for idx, value_cache in enumerate(self.value_caches)
            if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
        ]
        self.key_cache_ptrs = [
            ptr for idx, ptr in enumerate(self.key_cache_ptrs)
            if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
        ]
        self.value_cache_ptrs = [
            ptr for idx, ptr in enumerate(self.value_cache_ptrs)
            if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
        ]

        # Delete layer name from kv_cache_config
        deleted_layers = set(layer for layers in layers_list for layer in range(layers[0], layers[1]+1))
        group = self.kv_cache_config.kv_cache_groups[0]
        group.layer_names = [
            name for name in group.layer_names
            if extract_layer_index(name) not in deleted_layers
        ]
        forward_context = self.vllm_config.compilation_config.static_forward_context
        deleted_layer_names = [name for name in forward_context.keys() if extract_layer_index(name) in deleted_layers]

        for layer_name in deleted_layer_names:
            del forward_context[layer_name]
        logger.info(f"after release_kv_cache_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, model runner's kv cache length: {len(self.kv_caches)}")

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
        forward_context = self.vllm_config.compilation_config.static_forward_context
        deleted_layer_names = []
        for layer_name, attn_module in forward_context.items():
            if extract_layer_index(layer_name) in deleted_layers:
                deleted_layer_names.append(layer_name)
                attn_module.kv_cache = [torch.tensor([])]

        for layer_name in deleted_layer_names:
            del forward_context[layer_name]
        logger.info(f"after release_kv_cache_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, model runner's kv cache length: {len(self.kv_caches)}")

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
        migrate_record: dict[int, int] = {}
        if self.vllm_config.dynamic_config.enable_flexi_flash_attn:
            compact_cache_with_record(self._migrate_block_by_swapping_ptrs, is_used, compacted_length, num_blocks, migrate_record)
            # After swapping tensor references in Python lists, we must update the GPU pointer arrays
            # because prepare_flexi_kv_ptrs caches the data_ptr() of each tensor on GPU
            forward_context = self.vllm_config.compilation_config.static_forward_context
            for layer_name, attn_module in forward_context.items():
                idx = extract_layer_index(layer_name) - self.model.model.start_layer
                # Free old GPU pointer arrays to avoid memory leak
                old_k_ptrs, old_v_ptrs = self.key_cache_ptrs[idx], self.value_cache_ptrs[idx]
                free_flexi_kv_ptrs(old_k_ptrs, old_v_ptrs)
                # Allocate new GPU pointer arrays with updated tensor addresses
                self.key_cache_ptrs[idx], self.value_cache_ptrs[idx] = prepare_flexi_kv_ptrs(
                    self.key_caches[idx], self.value_caches[idx])
                assert isinstance(attn_module, FlexiAttention)
                attn_module.key_dev_ptr = self.key_cache_ptrs[idx]
                attn_module.value_dev_ptr = self.value_cache_ptrs[idx]
            logger.info(f"Updated GPU pointer arrays after compact")
        else:
            compact_cache_with_record(self._migrate_block_by_copy_data, is_used, compacted_length, num_blocks, migrate_record)
        logger.info(f"[debug]: migrate_record: {migrate_record}")

        # 遍历CachedRequestState更新block_ids
        for req_state in self.requests.values():
            for i, block_ids in enumerate(req_state.block_ids):
                for j, block_id in enumerate(block_ids):
                    if block_id in migrate_record:
                        req_state.block_ids[i][j] = migrate_record[block_id]
        # 遍历InputBatch中的block_table更新block_ids
        for block_table in self.input_batch.block_table:
            for row in block_table.block_table_np:
                for idx, block_id in enumerate(row):
                    if block_id in migrate_record:
                        row[idx] = migrate_record[block_id]
        
        # # ===== 关键修复：将更新后的 block table 同步到 GPU =====
        # # block_table_np 的修改已自动同步到 block_table_cpu (numpy view)
        # # 但必须显式调用 commit() 将 CPU tensor 拷贝到 GPU tensor
        # num_reqs = len(self.requests)
        # logger.info(f"[debug]: committing block table updates to GPU for {num_reqs} requests")
        # self.input_batch.block_table.commit(num_reqs)
        # logger.info(f"[debug]: block table committed to GPU")

        torch.cuda.synchronize()
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
                tmp_cache[:, :kv_length, ...].copy_(cache[:, :kv_length, ...], non_blocking=True)
            else:
                tmp_cache[:, :new_length, ...].copy_(cache[:, :new_length, ...], non_blocking=True)
            self.kv_caches[idx] = tmp_cache
            attn_module.kv_cache = [tmp_cache]
        time_end = time.time()
        logger.info(f"resized kv cache in {human_readable_duration(time_end - time_start)} seconds")
        logger.info(f"resized kv cache ,available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")

    def flexi_resize_kv_cache(self, new_length: int) -> None:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"resizing kv cache from {len(self.kv_caches[0][0])} to {new_length}")
        logger.info(f"before resize kv cache, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        time_start = time.time()
        forward_context = self.vllm_config.compilation_config.static_forward_context
        T, H, Dh = self.key_caches[0][0].shape
        cache_length = len(self.key_caches[0])
        logger.info(f"num of kv tensors{cache_length}")

        if new_length < cache_length:
            for layer_name, attn_module in forward_context.items():
                logger.info(f"resizing kv cache for layer {layer_name}")
                idx = extract_layer_index(layer_name) - self.model.model.start_layer
                key_cache = self.key_caches[idx][:new_length]
                value_cache = self.value_caches[idx][:new_length]
                self.key_caches[idx] = key_cache
                self.value_caches[idx] = value_cache
                # Free old GPU pointer arrays before allocating new ones
                old_k_ptrs, old_v_ptrs = self.key_cache_ptrs[idx], self.value_cache_ptrs[idx]
                free_flexi_kv_ptrs(old_k_ptrs, old_v_ptrs)
                self.key_cache_ptrs[idx], self.value_cache_ptrs[idx] = prepare_flexi_kv_ptrs(key_cache, value_cache)
                assert isinstance(attn_module, FlexiAttention)
                attn_module.key_cache = key_cache
                attn_module.value_cache = value_cache
                attn_module.key_dev_ptr = self.key_cache_ptrs[idx]
                attn_module.value_dev_ptr = self.value_cache_ptrs[idx]
        elif new_length > cache_length:
            extended_kv_cache_shape = (T, H, Dh)
            for layer_name, attn_module in forward_context.items():
                new_allocated_block_num = new_length - cache_length
                new_allocated_key_cache, new_allocated_value_cache = get_flexi_kv_cache( new_allocated_block_num, extended_kv_cache_shape, self.kv_cache_dtype, self.device)
                idx = extract_layer_index(layer_name) - self.model.model.start_layer
                key_cache = self.key_caches[idx]
                value_cache = self.value_caches[idx]
                key_cache.extend(new_allocated_key_cache)
                value_cache.extend(new_allocated_value_cache)
                # Free old GPU pointer arrays before allocating new ones
                old_k_ptrs, old_v_ptrs = self.key_cache_ptrs[idx], self.value_cache_ptrs[idx]
                free_flexi_kv_ptrs(old_k_ptrs, old_v_ptrs)
                self.key_cache_ptrs[idx], self.value_cache_ptrs[idx] = prepare_flexi_kv_ptrs(key_cache, value_cache)
                assert isinstance(attn_module, FlexiAttention)
                attn_module.key_cache = key_cache
                attn_module.value_cache = value_cache
                attn_module.key_dev_ptr = self.key_cache_ptrs[idx]
                attn_module.value_dev_ptr = self.value_cache_ptrs[idx]
            
        time_end = time.time()
        logger.info(f"resized kv cache in {human_readable_duration(time_end - time_start)} seconds")
        logger.info(f"resized kv cache ,available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")

    def _migrate_block_by_copy_data(self, old_block_id: int, new_block_id: int, migrate_record: dict[int, int]):
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        assert len(self.kv_caches) != 0
        for cache in self.kv_caches:
            # loop for k and v tensor
            for i in range(2):
                cache[i][new_block_id].copy_(cache[i][old_block_id])
                # 这里我们只将旧的搬迁到新的，我们不对旧的block数据制0
        migrate_record[old_block_id] = new_block_id

    def _migrate_block_by_swapping_ptrs(self, old_block_id: int, new_block_id: int, migrate_record: dict[int, int]):
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        assert len(self.kv_caches) != 0
        for key_cache in self.key_caches:
            key_cache[ new_block_id], key_cache[ old_block_id] = key_cache[ old_block_id], key_cache[ new_block_id]
        for value_cache in self.value_caches:
            value_cache[ new_block_id], value_cache[ old_block_id] = value_cache[ old_block_id], value_cache[ new_block_id]
        migrate_record[old_block_id] = new_block_id

    def _update_start_layer(self) -> None:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        forward_context: dict[str, "Attention"] = self.vllm_config.compilation_config.static_forward_context
        new_start = min(extract_layer_index(layer_name) for layer_name in forward_context.keys())
        start_layer = self.model.model.start_layer

        # Sync model start if left-padded
        if new_start != start_layer:
            self.model.model.start_layer = new_start
    
    def get_flexi_kv_cache_from_gathered_kv_tensor(self, slot_mapping: torch.Tensor, 
                                 gathered_kv_tensor: torch.Tensor,
                                 block_num: int) -> Tuple[list[torch.Tensor], list[torch.Tensor], int, int]:
        '''
        Based on current kv cache list shape and dtype, we allocate a new kv cache list
        and extract the data from gathered_kv_tensor to the new kv cache list.
        after that, we also flush the new kv cache to GPU ptrs.
        '''
        kv_cache_shape = self.kv_cache_shape
        assert len(kv_cache_shape) == 5 # (2, nkvblocks, blockdim, n_head, headdim)
        block_shape = kv_cache_shape[2:]
        kv_dtype = self.kv_cache_dtype

        key_cache, value_cache = get_flexi_kv_cache(block_num, block_shape, kv_dtype, self.device)
        key_cache_list_ptr, value_cache_list_ptr = prepare_flexi_kv_ptrs(key_cache, value_cache) 
        logger.info(f"gathered_kv_tensor shape: {gathered_kv_tensor.shape}, kv_cache_shape: {kv_cache_shape}, block_shape:{block_shape}, block_num:{block_num}, key_cache shape:{key_cache[0].shape}, value_cache shape:{value_cache[0].shape}, key_cache length:{len(key_cache)}, value_cache length:{len(value_cache)}")
        logger.info(f"key_cache_list_ptr {key_cache_list_ptr}, value cache list ptr:{value_cache_list_ptr}")
        dummy_scale = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        assert gathered_kv_tensor.shape[-2:] == kv_cache_shape[-2:], f"gathered_kv_tensor shape {gathered_kv_tensor.shape} mismatch kv_cache_shape {kv_cache_shape}"

        flexi_reshape_and_cache_flash(gathered_kv_tensor[0], gathered_kv_tensor[1], key_cache_list_ptr,value_cache_list_ptr,  key_cache[0], value_cache[0], slot_mapping,"auto", dummy_scale, dummy_scale)

        return key_cache, value_cache, key_cache_list_ptr, value_cache_list_ptr


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

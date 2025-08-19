# SPDX-License-Identifier: Apache-2.0

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
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")
                # Added the kv cache spec to kv cache config.
                assert len(self.kv_cache_config.kv_cache_groups) == 1
                self.kv_cache_config.kv_cache_groups[0].layer_names.append(layer_name)
        if self.speculative_config and self.speculative_config.use_eagle():
            raise NotImplementedError("Eagle is not supported for dynamic weights")
        start_layer = self.model.model.start_layer
        bind_kv_cache_for_layers(
            start_layer,
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
            layers)

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
            return super().execute_model(scheduler_output, intermediate_tensors)

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
        logger.info(f"forward context:{self.vllm_config.compilation_config.static_forward_context}")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"after delete_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    
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

        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"after release_kv_cache_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

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

def bind_kv_cache_for_layers(
    start_layer_index: int,
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, "Attention"],
    runner_kv_caches: list[torch.Tensor],
    layers: Tuple[int, int],
) -> None:
    """
    Bind the allocated KV cache of layers to ModelRunner and forward context.
    """
    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) != 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        layer_index = extract_layer_index(layer_name)
        assert layer_index in range(layers[0], layers[1]+1), f"layer_index: {layer_index} not in range({layers[0]}, {layers[1]+1})"
        index2name[layer_index].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.
            raise NotImplementedError
        layer_name = layer_names[0]
        runner_kv_caches.insert(layer_index - start_layer_index, kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].kv_cache = [kv_cache]

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
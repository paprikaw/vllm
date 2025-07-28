from collections.abc import Iterable
from typing import Optional, Union, Tuple
import gc

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_pp_group
from vllm.logger import init_logger

from vllm.sequence import IntermediateTensors

from .qwen2 import Qwen2Model
from .qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM, Qwen3Model
from threading import Lock
from .utils import is_pp_missing_parameter
from vllm.model_executor.utils import extract_layer_index
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.config import VllmConfig
from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import LayerFn
from vllm.model_executor.models.utils import maybe_offload_to_cpu
from vllm.config import set_current_vllm_config
logger = init_logger(__name__)

@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class DynamicQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = DynamicQwen3Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
    def add_layers(self, layers: Tuple[int, int], decoder_layer_type: type[nn.Module] = Qwen3DecoderLayer) -> None:
        self.model.add_layers(layers, decoder_layer_type)

    def set_sched_layers(self, 
                         start_layer: int, 
                         end_layer: int) -> None:
        self.model.set_sched_layers(start_layer, end_layer)

    def get_sched_layers(self) -> Tuple[int, int]:
        return self.model.get_sched_layers()

    def delete_layers(self, layers: Tuple[int, int]):
        self.model.delete_layers(layers)

    def forward(self, 
                input_ids: torch.Tensor, 
                positions: torch.Tensor, 
                intermediate_tensors: Optional[IntermediateTensors] = None, 
                inputs_embeds: Optional[torch.Tensor] = None) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    # def load_layer_weights(self, weights: Iterable[tuple[str, torch.Tensor]], layers: Tuple[int, int])->set[str]:
    #     return self.model.load_layer_weights(weights, layers)
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

class DynamicQwen3Model(Qwen3Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix)
        self.sched_start_layer = self.start_layer
        self.sched_end_layer = self.end_layer
        self.model_lock = Lock()
        self.cache_config = vllm_config.cache_config
        # self.vllm_config = vllm_config
        # self.layer_module = vllm_config.compilation_config.static_forward_context
        # logger.info(f"Initlized Dyanmic Qwen3 Model, layers: {self.layers}")
        # logger.info(f"Initlized Dyanmic Qwen3 Model, vllm_config: {self.vllm_config.compilation_config.static_forward_context}")


    def add_layers(self, 
                    layers: Tuple[int, int], 
                    decoder_layer_type: type[nn.Module] = Qwen3DecoderLayer,
                    ) -> None:
        # First update model's layers
        assert layers[0] <= layers[1], "layers[0] must be less than layers[1]"
        assert layers[1] == self.start_layer - 1 or layers[0] == self.start_layer + 1, "layers must be adjacent to old_layers"
        if layers[1] == self.start_layer:
            self.start_layer = layers[0]
        if layers[0] == self.end_layer:
            self.end_layer = layers[1]
        # with set_current_vllm_config(self.vllm_config):
        new_module = add_layers(
            self.layers, 
            layers,
            (self.start_layer, self.end_layer),
            lambda prefix: decoder_layer_type(config=self.config,
                                              cache_config=self.cache_config,
                                              quant_config=self.quant_config,
                                              prefix=prefix),
            prefix=f"{self.prefix}.layers",
            )
        # logger.info(f"after add_layers, vllm_config static_forward_context: {self.vllm_config.compilation_config.static_forward_context}")
        old_layers = self.layers
        with self.model_lock:
            self.layers = new_module
        del old_layers
        gc.collect()
        torch.cuda.empty_cache()

    def delete_layers(self, layers: Tuple[int, int]):
        """Delete the layer module and its parameters at the given index."""
        with self.model_lock:
            for layer_idx in range(layers[0], layers[1]+1):
                if not (0 <= layer_idx < len(self.layers)):
                    raise IndexError(f"Layer index {layer_idx} out of range.")
                # # 1. 删除子模块引用
                layer = self.layers[layer_idx]
                assert layer is not None, "Layer is None"
                assert layer is not PPMissingLayer, "Layer is PPMissingLayer"
                # for name, _ in list(layer.named_parameters(recurse=True)):
                #     # 删除每个参数
                #     delattr(layer, name.split(".")[-1])
                # for name, _ in list(layer.named_children()):
                #     delattr(layer, name)
                self.layers[layer_idx] = PPMissingLayer()  # 占位符
                logger.info(f"Layer {layer_idx} deleted successfully.")
                del layer
                # 2. 显式从 _modules 中删除（可选但更保险）
                # 由于 nn.ModuleList 自动注册子模块，这一步确保彻底清除
                # prefix = f"layers.{layer_idx}"
                # keys_to_delete = (k for k in self._modules if k.startswith(prefix))
                # for key in keys_to_delete:
                #     self._modules.pop(key)
            # 3. 强制垃圾回收（释放 CPU/GPU 内存）
            gc.collect()
            torch.cuda.empty_cache()
        return
    def set_sched_layers(self, 
                         start_layer: int, 
                         end_layer: int) -> None:
        self.sched_start_layer = start_layer
        self.sched_end_layer = end_layer + 1

    def get_sched_layers(self) -> Tuple[int, int]:
        return self.sched_start_layer, self.sched_end_layer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        assert self.sched_start_layer != -1 and self.sched_end_layer != -1, "Please set sched_layers first"

        with self.model_lock:
            if get_pp_group().is_first_rank:
                if inputs_embeds is not None:
                    hidden_states = inputs_embeds
                else:
                    hidden_states = self.get_input_embeddings(input_ids)
                residual = None
            else:
                assert intermediate_tensors is not None
                hidden_states = intermediate_tensors["hidden_states"]
                residual = intermediate_tensors["residual"]
            logger.warning(f"Forwarding with layers:{self.sched_start_layer} to {self.sched_end_layer}")
            for layer in self.layers[self.sched_start_layer:self.sched_end_layer]:
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    residual,
                )
            if not get_pp_group().is_last_rank:
                return IntermediateTensors({
                    "hidden_states": hidden_states,
                    "residual": residual
                })
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # def load_layer_weights(self, weights: Iterable[tuple[str, torch.Tensor]], layers: Tuple[int, int])->set[str]:
    #     # Mostly copied from Qwen2Model.load_layer_weights
    #     # We added logic to only load weights for partial of the model layers.
    #     stacked_params_mapping = [
    #         # (param_name, shard_name, shard_id)
    #         ("qkv_proj", "q_proj", "q"),
    #         ("qkv_proj", "k_proj", "k"),
    #         ("qkv_proj", "v_proj", "v"),
    #         ("gate_up_proj", "gate_proj", 0),
    #         ("gate_up_proj", "up_proj", 1),
    #     ]
    #     params_dict = dict(self.named_parameters(remove_duplicate=False))
    #     loaded_params: set[str] = set()
    #     logger.info(f"Starting to load weights for layers:{layers}")
    #     for name, loaded_weight in weights:
    #         assert "layers" not in name, "layers should not be in the name"
    #         layer_index = extract_layer_index(name)
    #         assert layer_index in range(layers[0], layers[1]+1), \
    #              f"layer_index:{layer_index} not in range({layers[0]}, {layers[1]})"

    #         logger.info(f"Found layer:{name}")
    #         if "rotary_emb.inv_freq" in name:
    #             continue
    #         if (self.quant_config is not None and
    #             (scale_name := self.quant_config.get_cache_scale(name))):
    #             # Loading kv cache quantization scales
    #             param = params_dict[scale_name]
    #             weight_loader = getattr(param, "weight_loader",
    #                                     default_weight_loader)
    #             loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
    #                              loaded_weight[0])
    #             weight_loader(param, loaded_weight)
    #             loaded_params.add(scale_name)
    #             continue
    #         for (param_name, weight_name, shard_id) in stacked_params_mapping:
    #             if weight_name not in name:
    #                 continue
    #             name = name.replace(weight_name, param_name)
    #             # Skip loading extra bias for GPTQ models.
    #             if name.endswith(".bias") and name not in params_dict:
    #                 continue
    #             if is_pp_missing_parameter(name, self):
    #                 continue
    #             param = params_dict[name]
    #             weight_loader = param.weight_loader
    #             weight_loader(param, loaded_weight, shard_id)
    #             break
    #         else:
    #             # Skip loading extra bias for GPTQ models.
    #             if name.endswith(".bias") and name not in params_dict:
    #                 continue
    #             # Remapping the name of FP8 kv-scale.
    #             name = maybe_remap_kv_scale_name(name, params_dict)
    #             if name is None:
    #                 continue
    #             if is_pp_missing_parameter(name, self):
    #                 continue
    #             if name not in params_dict:
    #                 raise ValueError(f"Parameter {name} not found in params_dict: {params_dict.keys()}")
    #             param = params_dict[name]
    #             weight_loader = getattr(param, "weight_loader",
    #                                     default_weight_loader)
    #             weight_loader(param, loaded_weight)
    #         loaded_params.add(name)
    #     return loaded_params 

def add_layers(
    module: torch.nn.ModuleList,
    added_layers: Tuple[int, int],
    old_layers: Tuple[int, int],
    layer_fn: LayerFn,
    prefix: str,
) -> torch.nn.ModuleList:
    """
    Replace layers in `old_layers` range with new layers defined by `added_layers`,
    and use `PPMissingLayer()` as placeholders for the rest.

    The ranges are inclusive: [start, end]
    - model: original model's ModuleList
    - added_layers: range of new layers to insert
    - layer_fn: factory function to generate a new layer
    - prefix: prefix for naming the new layers
    """

    num_layers = len(module)
    new_module = torch.nn.ModuleList()
    assert added_layers[0] <= added_layers[1], "added_layers[0] must be less than added_layers[1]"
    assert added_layers[1] == old_layers[0] - 1 or added_layers[0] == old_layers[1] + 1, "added_layers must be adjacent to old_layers"

    for idx in range(num_layers):
        if old_layers[0] <= idx <= old_layers[1]:
            new_module.append(module[idx])
        elif added_layers[0] <= idx <= added_layers[1]:
            new_module.append(maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}")))
        else:
            new_module.append(module[idx])
    return new_module
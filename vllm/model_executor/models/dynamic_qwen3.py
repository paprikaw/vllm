from collections.abc import Iterable
from typing import Optional, Union, Tuple
import gc

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger

from vllm.sequence import IntermediateTensors

from .qwen2 import Qwen2Model
from .qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM
from threading import Lock
from .utils import is_pp_missing_parameter
from vllm.model_executor.utils import extract_layer_index
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sched_start_layer = -1 
        self.sched_end_layer = -1 
        self.model_lock = Lock()

    def add_layers(self, 
                    layers: Tuple[int, int], 
                    decoder_layer_type: type[nn.Module] = Qwen3DecoderLayer,
                    ) -> None:
        from vllm.model_executor.models.utils import add_layers

        # First update model's layers
        assert layers[0] <= layers[1], "layers[0] must be less than layers[1]"
        assert layers[1] == self.model.start_layer - 1 or layers[0] == self.model.start_layer + 1, "layers must be adjacent to old_layers"
        if layers[1] == self.start_layer:
            self.model.start_layer = layers[0]
        if layers[0] == self.end_layer:
            self.model.end_layer = layers[1]

        new_module = add_layers(
            self.model.layers, 
            layers,
            (self.model.start_layer, self.model.end_layer),
            lambda prefix: decoder_layer_type(config=self.config,
                                              cache_config=self.cache_config,
                                              quant_config=self.quant_config,
                                              prefix=prefix),
            prefix=f"{self.prefix}.layers",
            )
        old_layers = self.model.layers
        with self.model_lock:
            self.model.layers = new_module
        del old_layers
        gc.collect()
        torch.cuda.empty_cache()

    def set_sched_layers(self, 
                         start_layer: int, 
                         end_layer: int) -> None:
        self.sched_start_layer = start_layer
        self.sched_end_layer = end_layer

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
            logger.info(f"Forwarding with layers:{self.model.start_layer} to {self.model.end_layer}")
            for layer in self.model.layers[self.sched_start_layer:self.sched_end_layer]:
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
            hidden_states, _ = self.model.norm(hidden_states, residual)
        return hidden_states

    def load_layer_weights(self, weights: Iterable[tuple[str, torch.Tensor]], layers: Tuple[int, int])->set[str]:
        # Mostly copied from Qwen2Model.load_layer_weights
        # We added logic to only load weights for partial of the model layers.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        logger.info(f"Starting to load weights for layers:{layers}")
        # 这里的fuse操作有可能导致GPU显存不足。
        # 我们不能假设模型加载的最小显存开支就等于模型权重大小。
        for name, loaded_weight in weights:
            layer_index = extract_layer_index(name)
            logger.info(f"Found layer:{name}")
            if layer_index not in range(layers[0], layers[1]+1):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params 
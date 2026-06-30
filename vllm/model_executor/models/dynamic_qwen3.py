from collections.abc import Iterable
from typing import Optional, Union, Tuple
import gc

from huggingface_hub import delete_space_secret
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_pp_group
from vllm.kv_allocator import ForegroundBackgroundGate
from vllm.logger import init_logger

from vllm.sequence import IntermediateTensors
from vllm.v1.utils import human_readable_duration

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
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import LayerFn
from vllm.model_executor.models.utils import maybe_offload_to_cpu
from vllm.config import set_current_vllm_config
from .dynamic_model_base import DynamicModelBase
import time
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
class DynamicQwen3ForCausalLM(Qwen3ForCausalLM, DynamicModelBase):
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

        is_autoscaling = vllm_config.dynamic_config.pipeline_autoscaling_enabled
        if get_pp_group().is_last_rank or is_autoscaling:
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
        self.layer_weight_size = -1

    def add_fbgate(self, fbgate: ForegroundBackgroundGate) -> None:
        self.model.add_fbgate(fbgate)

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

    def detach_layers_for_later_release(self, layers: Tuple[int, int]
                                        ) -> list[object]:
        return self.model.detach_layers_for_later_release(layers)

    def release_deleted_layers(self,
                               old_layers: list[object],
                               empty_cuda_cache: bool = False) -> None:
        self.model.release_deleted_layers(
            old_layers, empty_cuda_cache=empty_cuda_cache)

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
        # 在weights读出的时候，记录layer 0的weight权重大小。
        # 只记录layer 0，因为我们假设所有weight的大小相同。
        def weight_size_record_generator(weights: Iterable[tuple[str, torch.Tensor]]):
            # 如果已经记录过layer weight size，就不再记录，但仍需要yield所有weights
            # 🔴 修复：在generator中使用return会导致generator立即结束，不yield任何值！
            # 正确做法：无论是否已记录，都要yield所有权重
            if self.layer_weight_size != -1:
                # 已经记录过，直接yield所有权重，不再记录大小
                yield from weights
                return

            first_layer_index = -1
            layer_weight_accumulator = 0
            
            for name, weight in weights:
                # 只记录第一个出现的 layer 的权重大小，假设所有 layer 权重大小一致
                layer_idx = extract_layer_index(name) if "layers" in name else None
                
                if layer_idx is not None:
                    if first_layer_index == -1:
                        # 发现第一个layer，开始记录
                        first_layer_index = layer_idx
                        layer_weight_accumulator = 0
                        logger.info(f"开始记录layer {first_layer_index}的权重大小")
                    
                    if layer_idx == first_layer_index:
                        # 累加同一个layer的所有weight
                        weight_size = weight.numel() * weight.element_size()
                        layer_weight_accumulator += weight_size
                
                yield name, weight
            self.layer_weight_size = layer_weight_accumulator
            # 处理只有一个layer的情况（迭代结束时还没有保存）
            assert first_layer_index != -1 and self.layer_weight_size != -1

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weight_size_record_generator(weights))

    def get_layer_weight_size(self) -> int:
        if self.layer_weight_size == -1:
            raise ValueError("Layer weight size not recorded")
        return self.layer_weight_size

class DynamicQwen3Model(Qwen3Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix)
        if (vllm_config.dynamic_config.pipeline_autoscaling_enabled
                and isinstance(self.norm, PPMissingLayer)):
            self.norm = RMSNorm(self.config.hidden_size,
                                eps=self.config.rms_norm_eps)
        self.sched_start_layer = self.start_layer
        self.sched_end_layer = self.end_layer
        self.model_lock = Lock()
        self.cache_config = vllm_config.cache_config
        self.vllm_config = vllm_config  # 保存vllm_config用于后续add_layers
        self.fbgate: Optional[ForegroundBackgroundGate] = None
        # self.layer_module = vllm_config.compilation_config.static_forward_context
        # logger.info(f"Initlized Dyanmic Qwen3 Model, layers: {self.layers}")
        # logger.info(f"Initlized Dyanmic Qwen3 Model, vllm_config: {self.vllm_config.compilation_config.static_forward_context}")

    def add_fbgate(self, fbgate: ForegroundBackgroundGate) -> None:
        self.fbgate = fbgate

    def add_layers(self, 
                    layers: Tuple[int, int], 
                    decoder_layer_type: type[nn.Module] = Qwen3DecoderLayer,
                    ) -> None:
        # First update model's layers
        assert layers[0] <= layers[1], "layers[0] must be less than layers[1]"
        old_start_layer = self.start_layer
        old_end_layer = self.end_layer
        old_layers_empty = old_start_layer >= old_end_layer
        assert (old_layers_empty or layers[1] == old_start_layer - 1
                or layers[0] == old_end_layer), (
                    f"layers must be adjacent to old_layers, layers: {layers}, "
                    f"start_layer: {old_start_layer}, end_layer: {old_end_layer}")
        # 使用set_current_vllm_config上下文管理器，确保新层初始化时能访问到正确的vllm_config
        # 这与initialize_model中的逻辑保持一致
        with set_current_vllm_config(self.vllm_config):
            new_module = add_layers(
                self.layers, 
                layers,
                (old_start_layer, old_end_layer - 1),
                lambda prefix: decoder_layer_type(config=self.config,
                                                  cache_config=self.cache_config,
                                                  quant_config=self.quant_config,
                                                  prefix=prefix),
                prefix=f"{self.prefix}.layers",
                )
        with self.model_lock:
            self.layers = new_module
            if old_layers_empty:
                self.start_layer = layers[0]
                self.end_layer = layers[1] + 1
            else:
                if layers[1] == old_start_layer - 1:
                    self.start_layer = layers[0]
                if layers[0] == old_end_layer:
                    self.end_layer = layers[1] + 1
        
        # 🔴 关键修复：清除get_pp_missing_layer_names的缓存
        # add_layers后，原来的PPMissingLayer变成了真实的层，但缓存仍保存旧的missing列表
        # 必须清除缓存，否则is_pp_missing_parameter会错误地跳过新增层的权重加载
        from vllm.model_executor.models.utils import _model_to_pp_missing_layer_names
        # 清除当前模型（DynamicQwen3Model/Qwen2Model）的缓存
        # 这是Qwen2Model.load_weights()中is_pp_missing_parameter检查的对象
        _model_to_pp_missing_layer_names.pop(id(self), None)
        logger.info(f"[DEBUG] Cleared PP missing layer cache after add_layers for model id={id(self)}")
        
        # gc.collect()
        # torch.cuda.empty_cache()

    def detach_layers_for_later_release(self, layers: Tuple[int, int]
                                        ) -> list[object]:
        """Detach layer modules from execution and return references to free."""
        # Adapt input layers to the model's layers open internal representation
        deleted_start_layer, deleted_end_layer = layers[0], layers[1] + 1
        old_start_layer, old_end_layer = self.start_layer, self.end_layer
        logger.info(f"deleted_start_layer: {deleted_start_layer}, deleted_end_layer: {deleted_end_layer}")
        assert deleted_start_layer < deleted_end_layer, "layers[0] must be less than layers[1]"
        assert deleted_start_layer >= old_start_layer and deleted_end_layer <= old_end_layer, f"layers must be in the range of start_layer and end_layer, old start_layer: {old_start_layer}, old end_layer: {old_end_layer}, deleted_start_layer{deleted_start_layer}, deleted_end_layer:{deleted_end_layer}"
        assert deleted_start_layer == old_start_layer or deleted_end_layer == old_end_layer, f"model layers must be continuous after delete layers, old start_layer: {old_start_layer}, old end_layer: {old_end_layer}, deleted_start_layer{deleted_start_layer}, deleted_end_layer:{deleted_end_layer}"

        old_layers = []
        with self.model_lock:
            time_start = time.time()
            for layer_idx in range(deleted_start_layer, deleted_end_layer):
                layer = self.layers[layer_idx]
                assert layer is not None, "Layer is None"
                assert layer is not PPMissingLayer, "Layer is PPMissingLayer"
                # Collect old layer reference before replacing
                old_layers.append(layer)
                self.layers[layer_idx] = PPMissingLayer()  # 占位符
                logger.info(f"Layer {layer_idx} deleted successfully.")

            # Keep the model's visible routing state consistent with the layer
            # list mutation. The actual memory release below is intentionally
            # outside model_lock so inference does not wait on GC/empty_cache.
            if deleted_start_layer == old_start_layer:
                self.start_layer = deleted_end_layer
            if deleted_end_layer == old_end_layer:
                self.end_layer = deleted_start_layer
            self.sync_sched_layers()
            logger.info(f"Deleted layers took {human_readable_duration(time.time() - time_start)}")

        return old_layers

    def release_deleted_layers(self,
                               old_layers: list[object],
                               empty_cuda_cache: bool = False) -> None:
        release_start = time.time()
        free_before = torch.cuda.memory_allocated()
        for layer in old_layers:
            del layer
        old_layers.clear()
        del old_layers
        if empty_cuda_cache:
            gc.collect()
            torch.cuda.empty_cache()
        free_after = torch.cuda.memory_allocated()
        logger.info(f"[delete_layers] Freed {(free_before - free_after) / 1024**3:.2f} GB of GPU memory for model weights in {human_readable_duration(time.time() - release_start)}")

    def delete_layers(self, layers: Tuple[int, int]):
        """Delete the layer module and its parameters at the given index."""
        old_layers = self.detach_layers_for_later_release(layers)
        self.release_deleted_layers(old_layers, empty_cuda_cache=True)
        logger.info(f"after delete_layers, start_layer: {self.start_layer}, end_layer: {self.end_layer}")
        return
    def set_sched_layers(self, 
                         start_layer: int, 
                         end_layer: int) -> None:
        self.sched_start_layer = start_layer
        self.sched_end_layer = end_layer + 1

    def get_sched_layers(self) -> Tuple[int, int]:
        return self.sched_start_layer, self.sched_end_layer

    def sync_sched_layers(self) -> None:
        self.sched_start_layer = self.start_layer
        self.sched_end_layer = self.end_layer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        assert self.sched_start_layer != -1 and self.sched_end_layer != -1, "Please set sched_layers first"

        # 从环境变量读取每个 layer 的超时时间（秒）
        time_start = time.time()
        with self.model_lock:
            logger.debug("getting model lock taking %s",
                         human_readable_duration(time.time() - time_start))
            with set_current_vllm_config(self.vllm_config):
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
                # logger.info(f"forwarding model with layers: {self.sched_start_layer} to {self.sched_end_layer}, total layers: {len(self.layers)}")
                # if self.sched_start_layer < 12:
                #     start = 0
                #     end = 12
                # else:
                #     start = 40
                #     end = 41

                logger.debug(f"forwarding model with layers: {self.sched_start_layer} to {self.sched_end_layer}, total layers: {len(self.layers)}")
                forwarding_start_time = time.time()
                for layer_idx, layer in enumerate(self.layers[self.sched_start_layer:self.sched_end_layer], start=self.sched_start_layer):
                    layer_start_time = time.time()
                    try:
                        # 使用 threading.Timer 实现超时检测
                        result = [None, None]  # 用于存储结果
                        
                        h, r = layer(positions, hidden_states, residual)
                        result[0], result[1] = h, r

                        hidden_states, residual = result[0], result[1]
                        
                    except TimeoutError as e:
                        logger.error(f"Layer {layer_idx} execution timeout: {e}")
                        raise
                    except Exception as e:
                        logger.error(f"Error in layer {layer_idx}: {e}")
                        raise
                    logger.debug(f"after Layer forwarding took {human_readable_duration(time.time() - layer_start_time)}, layer {layer_idx}")
                logger.debug(
                    "after forwarding took %s",
                    human_readable_duration(time.time() -
                                            forwarding_start_time))
                if not get_pp_group().is_last_rank:
                    return IntermediateTensors({
                        "hidden_states": hidden_states,
                        "residual": residual
                    })
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        '''
        Copied from qwen2.py
        We will use fbgate here to load weights
        '''
        from vllm.model_executor.layers.linear import RowParallelLinear
        # assert self.fbgate is not None, "fbgate is not set, please call add_fbgate first"
        if self.fbgate is None:
            logger.info("fbgate is not set")
        else:
            logger.info("fbgate is set")
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        # Create a mapping from parameter name to its parent module
        # This is needed to call chunk_weight_loader on the module, not the parameter
        param_to_module = {}
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                param_to_module[full_param_name] = module
        
        loaded_params: set[str] = set()
        # 这里的fuse操作有可能导致GPU显存不足。
        # 我们不能假设模型加载的最小显存开支就等于模型权重大小。
        def weight_load(name, loaded_weight):
            from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
            if "rotary_emb.inv_freq" in name:
               return 
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                assert False
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                return 
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                   continue  # 修复：应该continue而不是return
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                   continue 
                if is_pp_missing_parameter(name, self):
                   continue 
                param = params_dict[name]
                weight_loader = param.weight_loader
                
                # Check if this is a v2 weight loader (used by FP8 quantization)
                # v2 loaders now also accept fbgate parameter as keyword argument
                is_v2_loader = hasattr(weight_loader, "__name__") and "v2" in weight_loader.__name__
                
                if isinstance(param_to_module[name], MergedColumnParallelLinear):
                    time_start = time.time()
                    if is_v2_loader:
                        # v2 loader: (param, loaded_weight, shard_id, fbgate=fbgate)
                        weight_loader(param, loaded_weight, shard_id, fbgate=self.fbgate)
                    else:
                        # v1 loader: (fbgate, param, loaded_weight, shard_id)
                        weight_loader(self.fbgate, param, loaded_weight, shard_id)
                    logger.info(f"stacked params, loaded weight using weight loader: {weight_loader} for param: {name}, took {human_readable_duration(time.time() - time_start)}")
                    break

                if self.fbgate is not None:
                    with self.fbgate.background():
                        time_start = time.time()
                        weight_loader(param, loaded_weight, shard_id)
                        logger.info(f"stacked params, loaded weight using weight loader: {weight_loader} for param: {name}, took {human_readable_duration(time.time() - time_start)}")
                else:
                    time_start = time.time()
                    weight_loader(param, loaded_weight, shard_id)
                    logger.info(f"stacked params, loaded weight using weight loader: {weight_loader} for param: {name}, took {human_readable_duration(time.time() - time_start)}")
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    return 
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    return 
                if is_pp_missing_parameter(name, self):
                    return 
                param = params_dict[name]
                
                # Check if the parameter's parent module has chunk_weight_loader
                # chunk_weight_loader = getattr(parent_module, "chunk_weight_loader", None) if parent_module else None
                weight_loader = getattr(param, "weight_loader",
                        default_weight_loader)
                
                # Check if this is a v2 weight loader (used by FP8 quantization)
                # v2 loaders now also accept fbgate parameter as keyword argument
                is_v2_loader = hasattr(weight_loader, "__name__") and "v2" in weight_loader.__name__
                
                assert not isinstance(param_to_module[name], MergedColumnParallelLinear)
                if isinstance(param_to_module[name], RowParallelLinear):
                    time_start = time.time()
                    if is_v2_loader:
                        # v2 loader: (param, loaded_weight, fbgate=fbgate)
                        weight_loader(param, loaded_weight, fbgate=self.fbgate)
                    else:
                        # v1 loader: (fbgate, param, loaded_weight)
                        weight_loader(self.fbgate, param, loaded_weight)
                    logger.info(f"loaded weight using weight loader: {weight_loader} for param: {name}, took {human_readable_duration(time.time() - time_start)}")
                    loaded_params.add(name)
                    return
                if self.fbgate is not None:
                    with self.fbgate.background():
                        time_start = time.time()
                        weight_loader(param, loaded_weight)
                        logger.info(f"loaded weight using weight loader: {weight_loader} for param: {name}, took {human_readable_duration(time.time() - time_start)}")
                else:
                    start_time = time.time()
                    weight_loader(param, loaded_weight)
                    logger.info(f"loaded weight using weight loader: {weight_loader} for param: {name}, took {human_readable_duration(time.time() - start_time)}")
            # logger.info(f"Loaded weight for {name} took {human_readable_duration(time.time() - time_start)}")
            loaded_params.add(name)
        logger.info(f"start to load weights inside dynamic qwen3")
        for name, loaded_weight in weights:
            # fbgate is now handled at the iterator level in default_loader.py
            # so we don't need to wrap it here again
            time_start = time.time()
            weight_load(name, loaded_weight)
            logger.info(f"[Weight Loading] Loaded weight for {name} took {human_readable_duration(time.time() - time_start)}, with stream id: {torch.cuda.current_stream()}")
            
        return loaded_params


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
    old_layers_empty = old_layers[0] > old_layers[1]
    assert (old_layers_empty or added_layers[1] == old_layers[0] - 1
            or added_layers[0] == old_layers[1] + 1), (
                f"added_layers must be adjacent to old_layers, "
                f"added_layers:{added_layers}, old_layers:{old_layers}")

    for idx in range(num_layers):
        if not old_layers_empty and old_layers[0] <= idx <= old_layers[1]:
            new_module.append(module[idx])
        elif added_layers[0] <= idx <= added_layers[1]:
            new_module.append(maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}")))
        else:
            new_module.append(module[idx])
    return new_module

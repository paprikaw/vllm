"""Dynamic Llama model supporting runtime PP reconfiguration.

Extends LlamaForCausalLM / LlamaModel with the same dynamic pipeline
parallelism pattern used by DynamicQwen3 (add/delete layers, scheduled
layer execution, fbgate-aware weight loading).

Note: FlexiAttention support is now in LlamaAttention directly (in llama.py),
so we can use LlamaDecoderLayer without modification.
"""
from collections.abc import Iterable
from typing import Optional, Tuple, Union
import gc
import time

import torch
from torch import nn
from threading import Lock

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.kv_allocator import ForegroundBackgroundGate
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.utils import extract_layer_index
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.sequence import IntermediateTensors
from vllm.v1.utils import human_readable_duration

from .dynamic_model_base import DynamicModelBase, add_layers
from .llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaModel
from .utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    maybe_prefix)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# DynamicLlamaForCausalLM – CausalLM wrapper
# ---------------------------------------------------------------------------

@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class DynamicLlamaForCausalLM(LlamaForCausalLM, DynamicModelBase):
    """LlamaForCausalLM extended with dynamic PP reconfiguration support."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Bypass LlamaForCausalLM.__init__ – set up everything ourselves
        # so we can swap in DynamicLlamaModel.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        # Use DynamicLlamaModel instead of plain LlamaModel
        self.model = DynamicLlamaModel(vllm_config=vllm_config,
                                       prefix=maybe_prefix(prefix, "model"))

        use_dynamic_communication = (
            vllm_config.dynamic_config.dynamic_communication_enabled)
        if get_pp_group().is_last_rank or use_dynamic_communication:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.layer_weight_size = -1

    # -- Dynamic delegation methods -----------------------------------------

    def add_fbgate(self, fbgate: ForegroundBackgroundGate) -> None:
        self.model.add_fbgate(fbgate)

    def add_layers(self, layers: Tuple[int, int],
                   decoder_layer_type: type[nn.Module] = LlamaDecoderLayer
                   ) -> None:
        self.model.add_layers(layers, decoder_layer_type)

    def set_sched_layers(self, start_layer: int, end_layer: int) -> None:
        self.model.set_sched_layers(start_layer, end_layer)

    def get_sched_layers(self) -> Tuple[int, int]:
        return self.model.get_sched_layers()

    def delete_layers(self, layers: Tuple[int, int]):
        self.model.delete_layers(layers)

    # -- Forward: delegate to inner model -----------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)

    # -- Weight loading with size recording ---------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]
                     ) -> set[str]:
        def weight_size_record_generator(
                weights: Iterable[tuple[str, torch.Tensor]]):
            if self.layer_weight_size != -1:
                yield from weights
                return

            first_layer_index = -1
            layer_weight_accumulator = 0

            for name, weight in weights:
                layer_idx = (extract_layer_index(name)
                             if "layers" in name else None)
                if layer_idx is not None:
                    if first_layer_index == -1:
                        first_layer_index = layer_idx
                        layer_weight_accumulator = 0
                        logger.info("开始记录layer %d的权重大小",
                                    first_layer_index)
                    if layer_idx == first_layer_index:
                        layer_weight_accumulator += (weight.numel()
                                                     * weight.element_size())
                yield name, weight

            self.layer_weight_size = layer_weight_accumulator
            assert first_layer_index != -1 and self.layer_weight_size != -1

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight
            in weight_size_record_generator(weights))

    def get_layer_weight_size(self) -> int:
        if self.layer_weight_size == -1:
            raise ValueError("Layer weight size not recorded")
        return self.layer_weight_size


# ---------------------------------------------------------------------------
# DynamicLlamaModel – inner transformer model
# ---------------------------------------------------------------------------

class DynamicLlamaModel(LlamaModel):
    """LlamaModel extended with dynamic PP reconfiguration support."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if (vllm_config.dynamic_config.dynamic_communication_enabled
                and isinstance(self.norm, PPMissingLayer)):
            self.norm = RMSNorm(self.config.hidden_size,
                                eps=self.config.rms_norm_eps)
        # LlamaModel doesn't store prefix / cache_config – add them
        self.prefix = prefix
        self.cache_config = vllm_config.cache_config
        self.vllm_config = vllm_config

        # Scheduled layer range (may differ from start/end during migration)
        self.sched_start_layer = self.start_layer
        self.sched_end_layer = self.end_layer

        self.model_lock = Lock()
        self.fbgate: Optional[ForegroundBackgroundGate] = None

    # -- fbgate -------------------------------------------------------------

    def add_fbgate(self, fbgate: ForegroundBackgroundGate) -> None:
        self.fbgate = fbgate

    # -- Layer management ---------------------------------------------------

    def add_layers(
        self,
        layers: Tuple[int, int],
        decoder_layer_type: type[nn.Module] = LlamaDecoderLayer,
    ) -> None:
        assert layers[0] <= layers[1], "layers[0] must be <= layers[1]"
        old_start_layer = self.start_layer
        old_end_layer = self.end_layer
        old_layers_empty = old_start_layer >= old_end_layer
        assert (old_layers_empty or layers[1] == old_start_layer - 1
                or layers[0] == old_end_layer), \
            (f"layers must be adjacent to old_layers, layers: {layers}, "
             f"start_layer: {old_start_layer}, end_layer: {old_end_layer}")

        with set_current_vllm_config(self.vllm_config):
            new_module = add_layers(
                self.layers,
                layers,
                (old_start_layer, old_end_layer - 1),
                lambda prefix: decoder_layer_type(
                    config=self.config,
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

        # Clear PP missing-layer cache so new layers can receive weights
        from vllm.model_executor.models.utils import (
            _model_to_pp_missing_layer_names)
        _model_to_pp_missing_layer_names.pop(id(self), None)
        logger.info("[DEBUG] Cleared PP missing layer cache after "
                    "add_layers for model id=%d", id(self))

    def delete_layers(self, layers: Tuple[int, int]):
        """Delete layers in the *inclusive* range and free GPU memory."""
        deleted_start, deleted_end = layers[0], layers[1] + 1
        old_start, old_end = self.start_layer, self.end_layer
        logger.info("deleted_start_layer: %d, deleted_end_layer: %d",
                    deleted_start, deleted_end)
        assert deleted_start < deleted_end
        assert (deleted_start >= old_start and deleted_end <= old_end), \
            (f"layers out of range: old [{old_start}, {old_end}), "
             f"delete [{deleted_start}, {deleted_end})")
        assert (deleted_start == old_start or deleted_end == old_end), \
            (f"layers must be continuous after delete: old [{old_start}, "
             f"{old_end}), delete [{deleted_start}, {deleted_end})")

        old_layers = []
        with self.model_lock:
            t0 = time.time()
            for idx in range(deleted_start, deleted_end):
                layer = self.layers[idx]
                assert layer is not None and layer is not PPMissingLayer
                old_layers.append(layer)
                self.layers[idx] = PPMissingLayer()
                logger.info("Layer %d deleted successfully.", idx)
            logger.info("Deleted layers took %s",
                        human_readable_duration(time.time() - t0))

        mem_before = torch.cuda.memory_allocated()
        for layer in old_layers:
            del layer
        old_layers.clear()
        del old_layers
        gc.collect()
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated()
        logger.info("[delete_layers] Freed %.2f GB of GPU memory",
                    (mem_before - mem_after) / 1024**3)

        if deleted_start == old_start:
            self.start_layer = deleted_end
        if deleted_end == old_end:
            self.end_layer = deleted_start

        self.sync_sched_layers()
        logger.info("after delete_layers, start_layer: %d, end_layer: %d",
                    self.start_layer, self.end_layer)

    # -- Scheduled layer control --------------------------------------------

    def set_sched_layers(self, start_layer: int, end_layer: int) -> None:
        self.sched_start_layer = start_layer
        self.sched_end_layer = end_layer + 1

    def get_sched_layers(self) -> Tuple[int, int]:
        return self.sched_start_layer, self.sched_end_layer

    def sync_sched_layers(self) -> None:
        self.sched_start_layer = self.start_layer
        self.sched_end_layer = self.end_layer

    # -- Forward with model_lock and scheduled layers -----------------------

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        assert (self.sched_start_layer != -1
                and self.sched_end_layer != -1), \
            "Please set sched_layers first"

        t0 = time.time()
        with self.model_lock:
            logger.debug("getting model lock taking %s",
                         human_readable_duration(time.time() - t0))
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

                logger.debug(
                    "forwarding model with layers: %d to %d, total: %d",
                    self.sched_start_layer, self.sched_end_layer,
                    len(self.layers))
                fwd_t0 = time.time()
                
                # Diagnostic: Track NaN propagation through layers
                _nan_first_detected_layer = -1
                
                # Fine-grained timing counters
                _layer_times = []
                _total_layer_time = 0.0
                
                for layer_idx, layer in enumerate(
                        self.layers[self.sched_start_layer:
                                    self.sched_end_layer],
                        start=self.sched_start_layer):
                    layer_t0 = time.time()
                    
                    # Check input for NaN before layer
                    try:
                        input_has_nan = torch.isnan(
                            hidden_states).any().item()
                    except Exception:
                        # This scalar check synchronizes the inference stream,
                        # so it is the first reliable attribution point for an
                        # asynchronous fault from the preceding layer (or the
                        # PP receive for the first local layer).
                        logger.exception(
                            "CUDA failure surfaced before layer %d "
                            "(scheduled range [%d, %d))",
                            layer_idx,
                            self.sched_start_layer,
                            self.sched_end_layer,
                        )
                        raise
                    
                    try:
                        hidden_states, residual = layer(
                            positions, hidden_states, residual)
                    except Exception as e:
                        logger.error("Error in layer %d: %s", layer_idx, e)
                        raise
                    
                    # # Check output for NaN after layer
                    # output_has_nan = torch.isnan(hidden_states).any().item()
                    
                    # # Log if NaN first appears in this layer
                    # if output_has_nan and not input_has_nan and _nan_first_detected_layer == -1:
                    #     _nan_first_detected_layer = layer_idx
                    #     device = hidden_states.device
                    #     gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
                    #     capability = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
                    #     logger.error(f"[LAYER_NAN] NaN FIRST APPEARED in layer {layer_idx}! "
                    #                 f"device={device} gpu={gpu_name} SM={capability[0]}{capability[1]} "
                    #                 f"hidden_shape={hidden_states.shape} "
                    #                 f"nan_count={torch.isnan(hidden_states).sum().item()}")
                    
                    layer_elapsed = time.time() - layer_t0
                    _layer_times.append(layer_elapsed)
                    _total_layer_time += layer_elapsed
                    
                    if layer_idx % 10 == 0 or layer_elapsed > 0.1:
                        logger.debug(
                            "[LAYER_TIMING] layer %d took %s (cumulative: %s)",
                            layer_idx, human_readable_duration(layer_elapsed),
                            human_readable_duration(_total_layer_time))
                
                # Log layer timing summary
                if _layer_times:
                    avg_time = sum(_layer_times) / len(_layer_times)
                    max_time = max(_layer_times)
                    max_layer = _layer_times.index(max_time) + self.sched_start_layer
                    logger.debug(
                        "[LAYER_SUMMARY] %d layers: total=%s, avg=%s, max=%s (layer %d)",
                        len(_layer_times), human_readable_duration(_total_layer_time),
                        human_readable_duration(avg_time), human_readable_duration(max_time), max_layer)
                
                logger.debug("after forwarding took %s",
                             human_readable_duration(time.time() - fwd_t0))

                if not get_pp_group().is_last_rank:
                    return IntermediateTensors({
                        "hidden_states": hidden_states,
                        "residual": residual,
                    })
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # -- Weight loading with fbgate support ---------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]
                     ) -> set[str]:
        """Custom load_weights adapted from LlamaModel with fbgate support."""
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear, RowParallelLinear)

        if self.fbgate is None:
            logger.info("fbgate is not set")
        else:
            logger.info("fbgate is set")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        # Map param names to their parent modules (needed for fbgate path)
        param_to_module: dict[str, nn.Module] = {}
        for module_name, module in self.named_modules():
            for pname, _ in module.named_parameters(recurse=False):
                full = f"{module_name}.{pname}" if module_name else pname
                param_to_module[full] = module

        loaded_params: set[str] = set()

        def _load_one(name: str, loaded_weight: torch.Tensor):
            if "rotary_emb.inv_freq" in name:
                return
            # Models trained with ColossalAI may have these
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                return

            if (self.quant_config is not None
                    and (scale_name :=
                         self.quant_config.get_cache_scale(name))):
                param = params_dict[scale_name]
                wl = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0
                                 else loaded_weight[0])
                wl(param, loaded_weight)
                loaded_params.add(scale_name)
                return

            # FP8 kv-scale remap (before stacked-params matching)
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    return

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader

                is_v2 = (hasattr(weight_loader, "__name__")
                         and "v2" in weight_loader.__name__)
                parent = param_to_module.get(name)

                if isinstance(parent, MergedColumnParallelLinear):
                    t = time.time()
                    if is_v2:
                        weight_loader(param, loaded_weight, shard_id,
                                      fbgate=self.fbgate)
                    else:
                        weight_loader(self.fbgate, param, loaded_weight,
                                      shard_id)
                    logger.info(
                        "stacked params loaded for %s, took %s",
                        name, human_readable_duration(time.time() - t))
                    break

                if self.fbgate is not None:
                    with self.fbgate.background():
                        weight_loader(param, loaded_weight, shard_id)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Non-stacked parameter
                if name.endswith(".bias") and name not in params_dict:
                    return
                if is_pp_missing_parameter(name, self):
                    return
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)

                is_v2 = (hasattr(weight_loader, "__name__")
                         and "v2" in weight_loader.__name__)
                parent = param_to_module.get(name)

                if isinstance(parent, RowParallelLinear):
                    t = time.time()
                    if is_v2:
                        weight_loader(param, loaded_weight,
                                      fbgate=self.fbgate)
                    else:
                        weight_loader(self.fbgate, param, loaded_weight)
                    logger.info("loaded %s, took %s",
                                name,
                                human_readable_duration(time.time() - t))
                    loaded_params.add(name)
                    return

                if self.fbgate is not None:
                    with self.fbgate.background():
                        weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight)

            loaded_params.add(name)

        logger.info("start to load weights inside dynamic llama")
        for name, loaded_weight in weights:
            t = time.time()
            _load_one(name, loaded_weight)
            logger.info(
                "[Weight Loading] Loaded %s took %s, stream=%s",
                name, human_readable_duration(time.time() - t),
                torch.cuda.current_stream())
        return loaded_params

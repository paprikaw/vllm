from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.utils import extract_layer_index
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models.dynamic_qwen3 import DynamicQwen3ForCausalLM
from vllm.logger import init_logger
from typing import cast, Tuple, Generator, Iterable, Optional
from tqdm.auto import tqdm

from .weight_utils import _BAR_FORMAT
import time

import torch
from torch import nn
from vllm.config import LoadConfig, LoadFormat, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.weight_utils import enable_tqdm
from safetensors import safe_open
from vllm.platforms import current_platform
from vllm.model_executor.layers.linear import QKVCrossParallelLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.attention import Attention
from vllm.v1.utils import human_readable_duration
logger = init_logger(__name__)

class CustomModelLoader(DefaultModelLoader):

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        # Dictionary to store preloaded weights: {weight_name: tensor}
        self._preloaded_weights: dict[str, torch.Tensor] = {}
        self._weights_preloaded = False

    def _preload_all_weights(
            self, 
            model_or_path: str,
            revision: Optional[str],
            fall_back_to_pt: bool,
            allow_patterns_overrides: Optional[list[str]]
    ) -> None:
        """Preload all weights from safetensors files into memory once."""
        if self._weights_preloaded:
            logger.info("Weights already preloaded, skipping...")
            return
        
        logger.info("Starting to preload all model weights into memory...")
        time_start = time.time()
        
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            model_or_path, revision, fall_back_to_pt, allow_patterns_overrides)
        
        assert use_safetensors and \
            self.load_config.load_format != LoadFormat.FASTSAFETENSORS and \
            current_platform.is_cuda()
        
        total_size = 0
        for st_file in tqdm(
                hf_weights_files,
                desc="Preloading safetensors checkpoint shards",
                disable=not enable_tqdm(self.load_config.use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
        ):
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():
                    # Load tensor into memory
                    param = f.get_tensor(name)
                    self._preloaded_weights[name] = param
                    total_size += param.numel() * param.element_size()
        
        self._weights_preloaded = True
        logger.info(f"[timeline]: Preloaded {len(self._preloaded_weights)} weights, "
                   f"total size: {total_size / (1024**3):.2f} GB, "
                   f"time taken: {human_readable_duration(time.time() - time_start)}")

    def _get_layer_weights_iterator(
            self, source: DefaultModelLoader.Source, 
            layers: Tuple[int, int]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        #ss Preload all weights on first call
        assert self._weights_preloaded
            # self._preload_all_weights(
            #     source.model_or_path, 
            #     source.revision, 
            #     source.fall_back_to_pt,
            #     source.allow_patterns_overrides
            # )
        logger.info(f"using preloaded weights")
        weights_iterator = safetensors_layer_weights_iterator(
            self._preloaded_weights,
            layers
        )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor)
                for (name, tensor) in weights_iterator)

    def get_layer_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
        layers: Tuple[int, int],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # 获取 fbgate (如果存在)
        # fbgate = None
        # if hasattr(model, 'model') and hasattr(model.model, 'fbgate'):
        #     fbgate = model.model.fbgate
        #     if fbgate is not None:
        #         logger.info("Found fbgate in model.model, will use it for weight loading")

        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load",
                                    True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides",
                                             None),
        )
        yield from self._get_layer_weights_iterator(primary_weights, layers)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_layer_weights_iterator(source, layers)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get all weights for initial model loading (not layer-specific)."""
        # Preload weights on first call
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        
        # Trigger preloading if not already done
        if not self._weights_preloaded:
            self._preload_all_weights(
                primary_weights.model_or_path,
                primary_weights.revision,
                primary_weights.fall_back_to_pt,
                primary_weights.allow_patterns_overrides
            )
        
        # Yield all preloaded weights (not filtered by layer)
        for name, param in self._preloaded_weights.items():
            yield primary_weights.prefix + name, param
        
        # Handle secondary weights if any (rare case)
        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        # For secondary weights, we currently don't support them in preloading
        # This is a rare case and can be added if needed
        if secondary_weights:
            logger.warning("Secondary weights detected but not supported in preloading mode")

    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        """Load the complete model with weight preloading."""
        from vllm.model_executor.model_loader.utils import (
            initialize_model, process_weights_after_loading, set_default_torch_dtype)
        
        device_config = vllm_config.device_config
        target_device = torch.device(device_config.device)
        
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                logger.info(f"loading model on device: {torch.cuda.current_device()}")
                model = initialize_model(vllm_config=vllm_config, model_config=model_config)
            
            logger.info(f"initialize model weights: {model}")
            weights_to_load = {name for name, _ in model.named_parameters()}
            
            # Use get_all_weights which triggers preloading
            loaded_weights = model.load_weights(
                self.get_all_weights(model_config, model))
            
            self.counter_after_loading_weights = time.perf_counter()
            logger.info(
                "Loading weights took %.2f seconds",
                self.counter_after_loading_weights - self.counter_before_loading_weights)
            
            # Validate loaded weights for non-quantized models
            if model_config.quantization is None and loaded_weights is not None:
                weights_not_loaded = weights_to_load - loaded_weights
                if weights_not_loaded:
                    raise ValueError(
                        "Following weights were not initialized from "
                        f"checkpoint: {weights_not_loaded}")

            process_weights_after_loading(model, model_config, target_device)

        return model.eval()

    def load_qwen3_layers(self, vllm_config: VllmConfig,
                   model_config: ModelConfig,
                   layers: Tuple[int, int],
                   model: DynamicQwen3ForCausalLM,
                   device: torch.device
                   ) -> None:
        assert isinstance(model, DynamicQwen3ForCausalLM), "model must be a DynamicQwen3ForCausalLM instance"
        time_start = time.time()
        logger.info(f"[timeline]: start to load layers {layers}")
        # 创建低优先级 stream（priority 值越大优先级越低）
        # 注意：这主要影响 GPU kernel 执行，对 CPU I/O 无效
        with set_default_torch_dtype(model_config.dtype): 
            # 添加设备上下文管理器，与 vllm 正常初始化逻辑保持一致
            # 注意：这里使用 target_device，确保整个加载过程在正确的设备上
            with device:
                torch.cuda.set_device(device)
                # 使用 with 语句确保所有 CUDA 操作都在低优先级 stream 上执行
                # 再次显式设置当前设备，确保后续操作在正确的设备上
                logger.info(f"before weight loading, explicitly set device to current device: {torch.cuda.current_device()}")
                logger.info(f"before weight loading, gpu occupied: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
                # 只收集属于指定层范围的参数名，更易读
                model.add_layers(layers)
                logger.info(f"[timeline]: after add layers but not weigths, time taken: {human_readable_duration(time.time() - time_start)}")
                logger.info(f"[debug]: Loading weights for layers {layers}")
                weights_to_load = {
                    name
                    for name, _ in model.named_parameters()
                    if "layers" in name and extract_layer_index(name) in range(layers[0], layers[1]+1)
                }
                logger.info(f"[debug]: Weights to load: {weights_to_load}")
                loaded_weights = model.load_weights(
                    self.get_layer_weights(model_config, model, layers)) 
                logger.info(f"[debug]: after weight loading, gpu occupied: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
                logger.info(f"[timeline]: after weight loading, time taken: {human_readable_duration(time.time() - time_start)}")
                if model_config.quantization is None and loaded_weights is not None:
                    weights_not_loaded = weights_to_load - loaded_weights
                    if weights_not_loaded:
                        raise ValueError(
                            "Following weights were not initialized from "
                            f"checkpoint: {weights_not_loaded}")
                process_layer_weights_after_loading(model, model_config, device, layers)
                logger.info(f"[timeline]: after process layer weights after loading, time taken: {human_readable_duration(time.time() - time_start)}")
        return

########################################################
# Utils Functions #####################################
########################################################

def safetensors_layer_weights_iterator(
    preloaded_weights: dict[str, torch.Tensor],
    layer: Tuple[int, int],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the preloaded weights, filtering by layer range."""
    logger.info(f"[debug]: Iterating preloaded weights for layers {layer}")
    
    for name, param in preloaded_weights.items():
        # Filter weights: only yield weights belonging to the specified layer range
        if "layers" not in name:
            continue
        
        layer_idx = extract_layer_index(name)
        if layer_idx is None or layer_idx not in range(layer[0], layer[1]+1):
            continue
        
        logger.debug(f"[debug]: Yielding preloaded weight {name} for layer {layer_idx}")
        yield name, param

def process_layer_weights_after_loading(model: nn.Module, model_config: ModelConfig,
                                  target_device: torch.device, layers: Tuple[int, int]) -> None:
    from vllm.model_executor.utils import extract_layer_index
    for name, module in model.named_modules():
        if name == "model.layers" or "layers" not in name or extract_layer_index(name) not in range(layers[0], layers[1]+1):
            continue
        logger.debug(f"[debug]: processing weights after loading for module: {name}")
        if isinstance(module, QKVCrossParallelLinear):
            logger.debug(f"[debug]: processing weights after loading for QKVCrossParallelLinear: {module}")
            # NOTE(Isotr0py): special case for cross QKV layer because
            # q and kv proj aren't registered as submodules intentionally
            module.process_weights_after_loading()
            continue
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            logger.debug(f"[debug]: processing weights after loading for quant method: {quant_method}")
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    # Currently only used by MLA.
    # NOTE: This intentionally happens after other modules so we can easily
    # decompress the weights for MLA.
    for name, module in model.named_modules():
        if name == "model.layers" or "layers" not in name or extract_layer_index(name) not in range(layers[0], layers[1]+1):
            continue
        if isinstance(module, Attention) and \
            hasattr(module, "process_weights_after_loading"):
            logger.debug(f"[debug]: processing weights after loading for Attention: {module}")
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            module.process_weights_after_loading(model_config.dtype)
from collections import defaultdict
import os
import time
from typing import cast, Tuple, Generator, Iterable, Optional

from safetensors import safe_open
import torch
from torch import nn
from tqdm.auto import tqdm

from .weight_utils import _BAR_FORMAT
from vllm.attention import Attention
from vllm.config import LoadConfig, LoadFormat, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVCrossParallelLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    device_loading_context, set_default_torch_dtype)
from vllm.model_executor.model_loader.weight_utils import enable_tqdm
from vllm.model_executor.models.dynamic_model_base import DynamicModelBase
from vllm.model_executor.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.v1.utils import human_readable_duration
logger = init_logger(__name__)

class CustomModelLoader(DefaultModelLoader):

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        # CPU weight cache flag - will be set from vllm_config in load_model/load_dynamic_layers
        self._use_cpu_cache: Optional[bool] = None
        # Dictionary to store preloaded weights: {weight_name: tensor}
        self._preloaded_weights: dict[str, torch.Tensor] = {}
        self._weights_preloaded = False
        # Store prepared weights info for disk-based loading
        self._prepared_weights_info: Optional[tuple] = None
        self._pin_cpu_weight_cache = (
            os.getenv("VLLM_PIN_CPU_WEIGHT_CACHE", "1").lower()
            in ("1", "true", "yes", "on"))

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
                    # Load tensor from safetensors mmap first.
                    # We then materialize a real CPU copy so the cache does not
                    # keep file-backed mmap tensors that can still fault later
                    # during model.load_weights().
                    param = f.get_tensor(name)
                    # Use madvise to prefetch/populate mmap pages before the copy.
                    import ctypes
                    ptr = param.data_ptr()
                    nbytes = param.numel() * param.element_size()
                    # Align to page boundary
                    page_size = 4096
                    aligned_ptr = ptr & ~(page_size - 1)
                    aligned_len = nbytes + (ptr - aligned_ptr)
                    libc = ctypes.CDLL("libc.so.6", use_errno=True)
                    # MADV_POPULATE_READ = 22 (Linux 5.14+), fallback to MADV_WILLNEED = 3
                    MADV_POPULATE_READ = 22
                    MADV_WILLNEED = 3
                    ret = libc.madvise(ctypes.c_void_p(aligned_ptr), 
                                       ctypes.c_size_t(aligned_len), 
                                       MADV_POPULATE_READ)
                    if ret != 0 and ctypes.get_errno() == 22:  # EINVAL - not supported
                        libc.madvise(ctypes.c_void_p(aligned_ptr),
                                     ctypes.c_size_t(aligned_len),
                                     MADV_WILLNEED)
                    try:
                        materialized_param = torch.empty_like(
                            param,
                            device="cpu",
                            pin_memory=self._pin_cpu_weight_cache)
                    except RuntimeError:
                        if self._pin_cpu_weight_cache:
                            logger.exception(
                                "Failed to allocate pinned CPU weight cache "
                                "tensor for %s; falling back to pageable CPU "
                                "memory for the remaining weights",
                                name)
                            self._pin_cpu_weight_cache = False
                        materialized_param = torch.empty_like(
                            param, device="cpu")
                    materialized_param.copy_(param)
                    self._preloaded_weights[name] = materialized_param
                    total_size += materialized_param.numel() * materialized_param.element_size()
                    del param
        
        self._weights_preloaded = True
        logger.info(f"[timeline]: Preloaded {len(self._preloaded_weights)} weights into materialized CPU cache, "
               f"total size: {total_size / (1024**3):.2f} GB, "
               f"pinned={self._pin_cpu_weight_cache}, "
               f"time taken: {human_readable_duration(time.time() - time_start)}")

    def _get_layer_weights_iterator(
            self, source: DefaultModelLoader.Source, 
            layers: Tuple[int, int]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format.
        
        When cpu_cache=False: Only loads specified layers from disk, no caching.
        When cpu_cache=True: Uses preloaded weights from CPU memory cache.
        """
        
        if self._use_cpu_cache:
            # Use preloaded weights from CPU cache
            assert self._weights_preloaded, "Weights must be preloaded when CPU cache is enabled"
            logger.info(f"[cpu-cache]: Fetching layers {layers} from CPU memory cache")
            weights_iterator = safetensors_layer_weights_iterator(
                self._preloaded_weights,
                layers
            )
        else:
            # Load weights directly from disk - ONLY specified layers, no caching
            logger.info(f"[disk-mode]: Loading ONLY layers {layers} directly from disk (no caching)")
            hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
                source.model_or_path, source.revision, source.fall_back_to_pt,
                source.allow_patterns_overrides)
            assert use_safetensors and self.load_config.load_format != LoadFormat.FASTSAFETENSORS
            weights_iterator = safetensors_layer_weights_iterator_from_disk(
                hf_weights_files,
                layers,
                self.load_config.use_tqdm_on_load
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
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        
        if self._use_cpu_cache:
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
        else:
            # Load directly from disk without CPU cache
            logger.info("Loading all weights directly from disk (CPU cache disabled)")
            yield from self._get_weights_iterator(primary_weights)
        
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
        
        # Set CPU weight cache flag from dynamic config
        self._use_cpu_cache = vllm_config.dynamic_config.enable_cpu_weight_cache
        logger.info(f"CustomModelLoader using CPU weight cache: {self._use_cpu_cache}")
        
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

    def load_dynamic_layers(self, vllm_config: VllmConfig,
                   model_config: ModelConfig,
                   layers: Tuple[int, int],
                   model: DynamicModelBase,
                   device: torch.device
                   ) -> None:
        assert isinstance(model, DynamicModelBase), "model must be a DynamicModelBase instance"
        
        # Always update CPU weight cache flag from dynamic config
        # This ensures config changes between sweep experiments take effect
        new_cpu_cache_setting = vllm_config.dynamic_config.enable_cpu_weight_cache
        if self._use_cpu_cache != new_cpu_cache_setting:
            logger.info(f"CustomModelLoader CPU weight cache changed: {self._use_cpu_cache} -> {new_cpu_cache_setting}")
            self._use_cpu_cache = new_cpu_cache_setting
        
        time_start = time.time()
        logger.info(f"[timeline]: start to load layers {layers}")
        logger.info(
            "[dynamic-load]: begin layers=%s layer_count=%d cpu_cache=%s model_range_before=(%d,%d)",
            layers,
            layers[1] - layers[0] + 1,
            self._use_cpu_cache,
            model.model.start_layer,
            model.model.end_layer,
        )
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
                free_before_add, total_gpu_memory = torch.cuda.mem_get_info()
                # 只收集属于指定层范围的参数名，更易读
                add_structure_start = time.time()
                model.add_layers(layers)
                logger.info(
                    "[dynamic-load]: add_layers_structure layers=%s took=%s model_range_after=(%d,%d) free_gpu_before=%.2fGB free_gpu_after=%.2fGB total_gpu=%.2fGB",
                    layers,
                    human_readable_duration(time.time() - add_structure_start),
                    model.model.start_layer,
                    model.model.end_layer,
                    free_before_add / 1024 ** 3,
                    torch.cuda.mem_get_info()[0] / 1024 ** 3,
                    total_gpu_memory / 1024 ** 3,
                )
                logger.info(f"[timeline]: after add layers but not weigths, time taken: {human_readable_duration(time.time() - time_start)}")
                logger.info(f"[debug]: Loading weights for layers {layers}")
                weights_to_load = {
                    name
                    for name, _ in model.named_parameters()
                    if "layers" in name and extract_layer_index(name) in range(layers[0], layers[1]+1)
                }
                sample_weights = sorted(weights_to_load)[:5]
                logger.info(
                    "[dynamic-load]: weights_to_load_summary layers=%s count=%d sample=%s",
                    layers,
                    len(weights_to_load),
                    sample_weights,
                )

                layer_weight_stats: dict[int, dict[str, float]] = defaultdict(
                    lambda: {"tensor_count": 0, "bytes": 0.0})
                total_weight_tensors = 0
                total_weight_bytes = 0.0
                iterator_start = time.time()
                first_tensor_latency_s: Optional[float] = None

                def traced_layer_weights() -> Generator[tuple[str, torch.Tensor], None, None]:
                    nonlocal total_weight_tensors, total_weight_bytes, first_tensor_latency_s
                    for name, tensor in self.get_layer_weights(model_config, model, layers):
                        now = time.time()
                        if first_tensor_latency_s is None:
                            first_tensor_latency_s = now - iterator_start
                        tensor_bytes = float(tensor.numel() * tensor.element_size())
                        total_weight_tensors += 1
                        total_weight_bytes += tensor_bytes
                        layer_idx = extract_layer_index(name)
                        if layer_idx is not None:
                            layer_weight_stats[layer_idx]["tensor_count"] += 1
                            layer_weight_stats[layer_idx]["bytes"] += tensor_bytes
                        yield name, tensor

                load_weights_start = time.time()
                loaded_weights = model.load_weights(traced_layer_weights()) 
                per_layer_summary = ", ".join(
                    f"L{layer_idx}:{int(stats['tensor_count'])}t/{stats['bytes'] / 1024 ** 3:.2f}GB"
                    for layer_idx, stats in sorted(layer_weight_stats.items())
                )
                logger.info(
                    "[dynamic-load]: load_weights_summary layers=%s source=%s yielded_tensors=%d yielded_bytes=%.2fGB first_tensor_latency=%s load_weights_took=%s per_layer=[%s]",
                    layers,
                    "cpu_cache" if self._use_cpu_cache else "disk",
                    total_weight_tensors,
                    total_weight_bytes / 1024 ** 3,
                    human_readable_duration(first_tensor_latency_s) if first_tensor_latency_s is not None else "n/a",
                    human_readable_duration(time.time() - load_weights_start),
                    per_layer_summary,
                )
                logger.info(f"[debug]: after weight loading, gpu occupied: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
                if model_config.quantization is None and loaded_weights is not None:
                    weights_not_loaded = weights_to_load - loaded_weights
                    if weights_not_loaded:
                        raise ValueError(
                            "Following weights were not initialized from "
                            f"checkpoint: {weights_not_loaded}")
                process_after_loading_start = time.time()
                process_layer_weights_after_loading(model, model_config, device, layers)
                free_after_process, _ = torch.cuda.mem_get_info()
                logger.info(
                    "[dynamic-load]: process_after_loading_summary layers=%s took=%s free_gpu_after=%.2fGB total_dynamic_load=%s",
                    layers,
                    human_readable_duration(time.time() - process_after_loading_start),
                    free_after_process / 1024 ** 3,
                    human_readable_duration(time.time() - time_start),
                )
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


def safetensors_layer_weights_iterator_from_disk(
    hf_weights_files: list[str],
    layer: Tuple[int, int],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over safetensors files and yield weights ONLY for specified layer range.
    
    This function reads weights directly from disk without any CPU caching.
    Used when enable_cpu_weight_cache is disabled.
    
    IMPORTANT: Only weights for the specified layer range are loaded into memory.
    Other weights are skipped entirely (no disk I/O for them).
    """
    logger.info(f"[disk-loading]: Loading ONLY layers {layer[0]}-{layer[1]} from disk (no caching)")
    
    loaded_count = 0
    skipped_count = 0
    total_loaded_bytes = 0
    time_start = time.time()
    
    for st_file in tqdm(
            hf_weights_files,
            desc="Loading safetensors from disk",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
    ):
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():
                # Filter weights: only yield weights belonging to the specified layer range
                if "layers" not in name:
                    skipped_count += 1
                    continue
                
                layer_idx = extract_layer_index(name)
                if layer_idx is None or layer_idx not in range(layer[0], layer[1]+1):
                    skipped_count += 1
                    continue
                
                # Load ONLY this specific tensor from disk
                param = f.get_tensor(name)
                total_loaded_bytes += param.numel() * param.element_size()
                loaded_count += 1
                logger.debug(f"[disk-loading]: Loaded weight {name} for layer {layer_idx}")
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

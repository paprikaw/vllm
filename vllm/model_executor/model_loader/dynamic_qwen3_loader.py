from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.utils import extract_layer_index
from vllm.model_executor.model_loader.utils import set_default_torch_dtype, process_layer_weights_after_loading
from typing import Tuple
import torch


class CustomModelLoader(DefaultModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def load_qwen3_layers(self, vllm_config: VllmConfig,
                   model_config: ModelConfig,
                   layers: Tuple[int, int],
                   model,
                   ) -> None:
        device_config = vllm_config.device_config
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype): 
            # 只收集属于指定层范围的参数名，更易读
            model.add_layers(layers)
            weights_to_load = {
                name
                for name, _ in model.named_parameters()
                if extract_layer_index(name) in range(layers[0], layers[1]+1)
            }
            loaded_weights = model.load_layer_weights(
                self.get_all_weights(model_config, model),
                layers) 
            if model_config.quantization is None and loaded_weights is not None:
                weights_not_loaded = weights_to_load - loaded_weights
                if weights_not_loaded:
                    raise ValueError(
                        "Following weights were not initialized from "
                        f"checkpoint: {weights_not_loaded}")
            process_layer_weights_after_loading(model, model_config, target_device, layers)
        return
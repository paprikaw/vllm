from typing import Dict, Tuple, Optional
import json

# class PPLayerConfigs:
#     pp_layer_configs: Dict[str, list[Tuple[int,int]]]

#     def __init__(self, pp_layer_configs: Optional[Dict[str, list[Tuple[int,int]]]] = None):
#         if pp_layer_configs is not None:
#             self.pp_layer_configs = pp_layer_configs
#         else:
#             self.pp_layer_configs = {}
#         self.request_id_config_map = {}

#     def set_pp_layer_config(self, key: str, pp_layer_config: list[Tuple[int,int]]):
#         self.pp_layer_configs[key] = pp_layer_config

#     def get_pp_layer_config(self, key: str) -> list[Tuple[int,int]]:
#         return self.pp_layer_configs[key]

#     def delete_pp_layer_config(self, key: str):
#         del self.pp_layer_configs[key]
    
#     @staticmethod
#     def get_pp_layer_configs_from_json_files(path: str) -> 'PPLayerConfigs':
#         with open(path, "r") as f:
#             configs = json.load(f)
#         pp_layer_configs = {}
#         for key, config in configs.items():
#             if not (isinstance(config, list) and all(isinstance(x, int) for x in config)):
#                 raise ValueError("Value must be a list of integers")
#             current_config = []
#             start = 0
#             for num_layers in config:
#                 end = start + num_layers - 1
#                 current_config.append((start, end))
#                 start = end + 1
#             pp_layer_configs[key] = current_config
#         return PPLayerConfigs(pp_layer_configs)
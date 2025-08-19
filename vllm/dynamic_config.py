from typing import Dict, Tuple, Optional, Any, Iterable
import json

from pydantic import BaseModel, Field, field_validator, model_validator
from .config import config
from dataclasses import dataclass
import yaml


# ===== 工具函数：把 [a,b,c,...] 这样的“层数列表”转成 [(0,a-1),(a,a+b-1), ...]
def _counts_to_ranges(counts: Iterable[int]) -> list[Tuple[int, int]]:
    start = 0
    out: list[Tuple[int, int]] = []
    for n in counts:
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"layer count must be positive int, got {n!r}")
        end = start + n - 1
        out.append((start, end))
        start = end + 1
    return out

class PPLayerConfigs(BaseModel):
    # 目标规范：每个 key 对应一个“(start,end) 的列表”
    pp_layer_configs: Dict[str, list[Tuple[int, int]]] = Field(default_factory=dict)

    # 允许传入多种原始格式，在这里统一规范化
    @field_validator('pp_layer_configs', mode='before')
    @classmethod
    def normalize_pp_layer_configs(cls, v: Any) -> Dict[str, list[Tuple[int, int]]]:
        """
        支持的输入：
        1) {"0": [8,56], "1": [24,40]}                 # 解释为“层数列表”，转成 ranges
        2) {"0": [(0,7),(8,63)], "1": [(0,23),(24,63)]} # 已是 ranges
        3) {"0": (0,7), "1": (24,63)}                   # 单个 range，自动包成 list
        4) {"0": [[0,7],[8,63]], ...}                   # list[list] -> list[tuple]
        """
        if not isinstance(v, dict):
            raise TypeError("pp_layer_configs must be a dict")

        norm: Dict[str, list[Tuple[int, int]]] = {}
        for k, raw in v.items():
            # case 3: 单个二元 tuple -> 包一层 list
            if isinstance(raw, tuple) and len(raw) == 2 and all(isinstance(x, int) for x in raw):
                norm[k] = [raw]  # type: ignore[list-item]
                continue

            # case 1: list[int] -> 解释为“层数列表”，转 ranges
            if isinstance(raw, list) and all(isinstance(x, int) for x in raw):
                norm[k] = _counts_to_ranges(raw)
                continue

            # case 2/4: list[tuple]/list[list] -> 转成 list[tuple]
            if isinstance(raw, list) and all(
                (isinstance(x, (tuple, list)) and len(x) == 2 and all(isinstance(y, int) for y in x))
                for x in raw
            ):
                norm[k] = [ (int(a), int(b)) for a,b in raw ]  # 强制成 tuple
                continue

            raise TypeError(
                f"pp_layer_configs[{k!r}] invalid format: {raw!r}. "
                "Expected list[int] (counts) or list[tuple[int,int]] or tuple[int,int]."
            )
        return norm

    # 额外校验：确保每个 (start,end) 合法且不重叠、start<=end
    @model_validator(mode='after')
    def validate_ranges(self) -> 'PPLayerConfigs':
        for k, ranges in self.pp_layer_configs.items():
            # 单调不交叉检查（按 start 排序）
            ranges_sorted = sorted(ranges, key=lambda t: t[0])
            last_end = -1
            for (s, e) in ranges_sorted:
                if s > e:
                    raise ValueError(f"{k}: invalid range ({s},{e}), start must <= end")
                if s <= last_end:
                    raise ValueError(f"{k}: ranges overlap or not strictly increasing near {s},{e}")
                last_end = e
        return self

    # 便捷方法（你原来的接口）
    def set_pp_layer_config(self, key: str, pp_layer_config: list[Tuple[int,int]]):
        self.pp_layer_configs[key] = pp_layer_config

    def get_pp_layer_config(self, key: str) -> list[Tuple[int,int]]:
        return self.pp_layer_configs[key]

    def is_key_exist(self, key: str) -> bool:
        return key in self.pp_layer_configs

    def delete_pp_layer_config(self, key: str):
        del self.pp_layer_configs[key]


class DynamicConfig(BaseModel):
    alternative_configs: PPLayerConfigs
    migration_steps: list[int] = Field(default_factory=list)
    """
    Configurations that is used when switching between different pipeline configurations.
    """

    # 允许 alternative_configs 以多种形式传入，并在这里包一层/转成 PPLayerConfigs
    @field_validator('alternative_configs', mode='before')
    @classmethod
    def coerce_alternative_configs(cls, v: Any) -> Any:
        """
        支持：
        - 已是 PPLayerConfigs 实例
        - {"pp_layer_configs": {...}}  # 规范形式
        - 直接给一个 dict（当作 pp_layer_configs）
        """
        if isinstance(v, PPLayerConfigs):
            return v
        if isinstance(v, dict):
            if 'pp_layer_configs' in v:
                return v  # 交给 PPLayerConfigs 自己的 validator 处理
            else:
                # 直接把这层当成 pp_layer_configs
                return {'pp_layer_configs': v}
        raise TypeError("alternative_configs must be a dict or PPLayerConfigs")

    # init_layer_config: list[Tuple[int,int]] = None # type: ignore
    # """
    # Initial layer configuration, this is not used
    # """

    # @classmethod
    # def from_json_files(cls, config_path: str) -> 'DynamicConfig':
    #     with open(config_path, "r") as f:
    #         configs = json.load(f)
    #     # assert "init_layer_config" in configs, "init_layer_config must be specified in the config file"
    #     assert "alternative_configs" in configs, "alternative_configs must be specified in the config file"
    #     assert "migration_steps" in configs, "migration_steps must be specified in the config file"
    #     assert isinstance(configs["alternative_configs"], dict)
    #     assert 
    #     alternative_configs = {}
    #     for key, config in configs["alternative_configs"].items():
    #         if not (isinstance(config, list) and all(isinstance(x, int) for x in config)):
    #             raise ValueError("Value must be a list of integers")
    #         current_config = []
    #         start = 0
    #         for num_layers in config:
    #             end = start + num_layers - 1
    #             current_config.append((start, end))
    #             start = end + 1
    #         alternative_configs[key] = current_config
    #     alternative_configs = PPLayerConfigs(alternative_configs)
        # init_config = configs["init_layer_config"]
        # assert isinstance(init_config, list) and all(isinstance(x, Tuple) for x in init_config), "init_layer_config must be a list of tuples"

    @classmethod
    def load_config(cls, path: str):
        with open(path, "r") as f:
            return cls.model_validate(yaml.safe_load(f))
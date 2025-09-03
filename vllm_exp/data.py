#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import typer
from pydantic import BaseModel, Field, model_validator, field_validator, ValidationInfo
from rich.console import Console


app = typer.Typer(no_args_is_help=True)
C = Console()

# =========================
# Config models
# =========================

def collect_variables(cfg: Config) -> Dict[str, Any]:
    """汇总本轮 run 的所有可参与命名的因子"""
    return {
        # 组合维度
        "delay": cfg.network.delays,
        "request_rate": cfg.benchmark.sweep_request_rates,
        "start_pp_layer_partition": cfg.vllm.start_pp_layer_partitions,

        # 模型/系统（Config）
        "max_model_len": cfg.vllm.max_model_len,
        "pipeline_parallel_size": cfg.vllm.pipeline_parallel_size,
        "gpu_memory_utilization": cfg.vllm.gpu_memory_utilization,
        "chunked_prefill": cfg.vllm.chunked_prefill,
        "enable_cuda_graph": cfg.vllm.enable_cuda_graph,
        "enable_nsight": cfg.vllm.enable_nsight,
        "model_name": cfg.model.name,
        "model_path": cfg.model.path,

        # Migration相关
        "is_migration": cfg.migration.is_migration,
        "alternative_configs": cfg.migration.alternative_configs,

        # Benchmark相关
        "input_output_lens": cfg.benchmark.input_output_lens,
    }

def collect_aliases() -> Dict[str, str]:
    return {
        "request_rate": "rr",
        "input_output_lens": "io",
        "delay": "delay",
        "start_pp_layer_partition": "pp",
        "is_migration": "mig",
    }

class ModelCfg(BaseModel):
    path: str = "/root/.cache/huggingface/Qwen3-32B-AWQ"
    name: str = "Qwen3-32B-AWQ"

class VllmCfg(BaseModel):
    pipeline_parallel_size: int = 2
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    chunked_prefill: bool = True
    enable_cuda_graph: bool = False
    enable_nsight: bool = False
    port: int = 8000
    ray_port: int = 6379
    start_pp_layer_partitions: list[str] = ["8,56"]

class MigrationCfg(BaseModel):
    is_migration: bool = False 
    is_compact_kv: bool = False
    alternative_configs: dict[str, list[int]] = {}
    migration_steps: list[int] = []
    compact_steps: list[int] = []

class BenchCfg(BaseModel):
    running_num_requests: list[int] = []
    data_num_requests: list[int] = []
    pattern_batch_size: int = 250
    sweep_request_rates: list[float] = []
    running_request_rates: list[float] = []
    profile: bool = False
    input_output_lens: list[list[int]]

    @field_validator("running_request_rates")
    @classmethod
    def _check_running_request_rate_len(cls, v: List[float], info: ValidationInfo):
        # 在 v2 里，用 info.data 获取已解析字段
        num_requests = info.data.get("running_num_requests") or []
        if v and not num_requests:
            raise ValueError("当提供 request_rates_list_compact 时，必须同时提供 num_requests。")
        if v and num_requests and len(v) != len(num_requests):
            raise ValueError(
                f"request_rates_list_compact 长度 {len(v)} 必须与 num_requests 长度 {len(num_requests)} 一致"
            )
        return v

    @field_validator("input_output_lens")
    @classmethod
    def _check_io_lens_len(cls, v: List[List[int]], info: ValidationInfo):
        num_requests = info.data.get("data_num_requests") or []
        if v and not num_requests:
            raise ValueError("当提供 input_output_lens 时，必须同时提供 num_requests。")
        if v and num_requests and len(v) != len(num_requests):
            raise ValueError(
                f"input_output_lens 长度 {len(v)} 必须与 num_requests 长度 {len(num_requests)} 一致"
            )
        return v


class NetworkCfg(BaseModel):
    delays: List[float] = [0]

class PathPolicy(BaseModel):
    variables: List[str] = []                   # 本轮作为“变量”的键


class Config(BaseModel):
    project: str
    type: str
    model: ModelCfg = ModelCfg()
    vllm: VllmCfg
    benchmark: BenchCfg
    network: NetworkCfg = NetworkCfg()
    path_policy: Optional[PathPolicy] = Field(None, exclude=True)
    migration: MigrationCfg
    envs: Dict[str, str] = {}

class MultiConfig(BaseModel):
    projects: List[Config]
    envs: Dict[str, str] = {}
    is_log_cover: bool = False
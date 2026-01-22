#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
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
        "enable_flexi_flash_attn": cfg.vllm.enable_flexi_flash_attn,
        "block_size": cfg.vllm.block_size,
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
        "enable_flexi_flash_attn": "flexi"
    }

class ModelCfg(BaseModel):
    path: str = "/root/.cache/huggingface/Qwen3-32B-AWQ"
    name: str = "Qwen3-32B-AWQ"

class VllmCfg(BaseModel):
    pipeline_parallel_size: int = 2
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8000
    chunked_prefill: bool = True
    enable_cuda_graph: bool = False
    enable_nsight: bool = False
    enable_flexi_flash_attn: bool = False
    block_size: Optional[int] = None  # KV cache block size (1, 8, 16, 32, 64, 128), None means use vLLM default
    head_addr: str = "head"
    port: int = 8000
    ray_port: int = 6379
    start_pp_layer_partitions: list[str] = ["8,56"]

class MigrationCfg(BaseModel):
    is_migration: bool = False 
    is_compact_kv: bool = False
    alternative_configs: dict[str, list[int]] = {}
    migration_steps: list[int] = []
    compact_steps: list[int] = []
    tester_start_step: Optional[int] = None
    memory_stress_tester: Optional[Dict[str, Any]] = None

class WarmupBenchCfg(BaseModel):
    """Optional warmup stage config for vllm_exp.

    This mirrors the request-related fields used by the formal benchmark.
    The warmup stage is executed (and logged separately) only when enabled.
    """

    enabled: bool = False
    running_num_requests: list[int] = []
    data_num_requests: list[int] = []
    sweep_request_rates: list[float] = []
    running_request_rates: list[float] = []
    input_output_lens: list[list[int]] = []

    @field_validator("running_request_rates")
    @classmethod
    def _check_running_request_rate_len(cls, v: List[float], info: ValidationInfo):
        num_requests = info.data.get("running_num_requests") or []
        if v and not num_requests:
            raise ValueError("当提供 warmup.running_request_rates 时，必须同时提供 warmup.running_num_requests。")
        if v and num_requests and len(v) != len(num_requests):
            raise ValueError(
                f"warmup.running_request_rates 长度 {len(v)} 必须与 warmup.running_num_requests 长度 {len(num_requests)} 一致"
            )
        return v

    @field_validator("input_output_lens")
    @classmethod
    def _check_io_lens_len(cls, v: List[List[int]], info: ValidationInfo):
        num_requests = info.data.get("data_num_requests") or []
        if v and not num_requests:
            raise ValueError("当提供 warmup.input_output_lens 时，必须同时提供 warmup.data_num_requests。")
        if v and num_requests and len(v) != len(num_requests):
            raise ValueError(
                f"warmup.input_output_lens 长度 {len(v)} 必须与 warmup.data_num_requests 长度 {len(num_requests)} 一致"
            )
        return v

class BenchCfg(BaseModel):
    running_num_requests: list[int] = []
    data_num_requests: list[int] = []
    pattern_batch_size: int = 150
    sweep_request_rates: list[float] = []
    running_request_rates: list[float] = []
    profile: bool = False
    input_output_lens: list[list[int]]
    # Whether to print each request's generated output in benchmark logs
    print_outputs: bool = False
    # Path to benchmark script
    benchmark_script_path: str = "/root/vllm_workbench/vllm/benchmarks/benchmark_serving.py"
    # Burstiness factor for request generation (default 1.0 = Poisson process)
    # Higher values (e.g., 100) result in more uniform/constant request rate
    burstiness: float = 100.0
    # Optional metrics output path when running via vllm_exp
    metrics_file_name: Optional[str] = None

    # Optional warmup stage (separate logs + metrics)
    warmup: Optional[WarmupBenchCfg] = None

    @field_validator("running_request_rates")
    @classmethod
    def _check_running_request_rate_len(cls, v: List[float], info: ValidationInfo):
        num_requests = info.data.get("running_num_requests") or []
        if v and not num_requests:
            raise ValueError("当提供 running_request_rates 时，必须同时提供 running_num_requests。")
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
            raise ValueError("当提供 input_output_lens 时，必须同时提供 data_num_requests。")
        if v and num_requests and len(v) != len(num_requests):
            raise ValueError(
                f"input_output_lens 长度 {len(v)} 必须与 num_requests 长度 {len(num_requests)} 一致"
            )
        return v


class NetworkCfg(BaseModel):
    delays: List[float] = [0]
    # 可选：pipeline 并行各 rank 的可达 IP，供 KV synchronizer 双向通道使用
    rank_to_ip: Dict[int, str] = {}
    # 可选：指定每个 rank 对应的 Ray 节点（hostname 或 IP），用于跨节点 worker 部署
    # 示例：{0: "node1", 1: "node2"} 表示 rank 0 部署在 node1，rank 1 部署在 node2
    rank_to_node: Dict[int, str] = {}

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


# =========================
# Sweep Test Config Models (New unified format)
# =========================

from typing import ClassVar


class RequestStageConfig(BaseModel):
    """Configuration for a single request stage.
    
    Defines the request rate and input/output lengths for a stage.
    """
    request_rate: float
    input_lens: int
    output_lens: int


class SweepBenchmarkConfig(BaseModel):
    """Configuration for a single benchmark sweep item.
    
    Uses indexed dict format for pp_layer_config and requests,
    where keys represent the request index at which configuration changes.
    
    Example:
        num_total_requests: 400
        pp_layer_config:
          0: "32, 32"
          100: "12, 52"
        requests:
          0:
            request_rate: 1.8
            input_lens: 800
            output_lens: 64
          100:
            request_rate: 1.5
            input_lens: 1024
            output_lens: 128
    """
    
    # Naming aliases for file naming - use property names directly
    NAMING_ALIASES: ClassVar[Dict[str, str]] = {
        "num_total_requests": "n_req",
        "start_pp_partition": "pp",
        "start_request_rate": "rr",
        "has_migration": "mig",
        "start_input_lens": "in",
        "start_output_lens": "out",
    }
    
    num_total_requests: int
    """Total number of requests for this benchmark configuration."""
    
    pp_layer_config: Dict[int, str]
    """Pipeline layer partition configs indexed by request number.
    Example: {0: "32,32", 100: "12,52"} means use "32,32" for requests 0-99,
    then switch to "12,52" at request 100. Multiple configs imply migration."""
    
    requests: Dict[int, RequestStageConfig]
    """Request configurations indexed by request number.
    Example: {0: {request_rate: 1.8, input_lens: 800, output_lens: 64}, ...}"""
    
    @property
    def has_migration(self) -> bool:
        """Returns True if there are multiple pp_layer_config entries."""
        return len(self.pp_layer_config) > 1
    
    @property
    def start_pp_partition(self) -> str:
        """Returns the initial pp_layer_partition (at index 0)."""
        sorted_keys = sorted(self.pp_layer_config.keys())
        return self.pp_layer_config[sorted_keys[0]].replace(" ", "")
    
    @property
    def start_request_rate(self) -> float:
        """Returns the initial request rate."""
        sorted_keys = sorted(self.requests.keys())
        return self.requests[sorted_keys[0]].request_rate
    
    @property
    def start_input_lens(self) -> int:
        """Returns the initial input length."""
        sorted_keys = sorted(self.requests.keys())
        return self.requests[sorted_keys[0]].input_lens
    
    @property
    def start_output_lens(self) -> int:
        """Returns the initial output length."""
        sorted_keys = sorted(self.requests.keys())
        return self.requests[sorted_keys[0]].output_lens
    
    def get_request_rate_dict(self) -> Dict[int, float]:
        """Convert requests to simple request_rate dict."""
        return {idx: cfg.request_rate for idx, cfg in self.requests.items()}
    
    def get_input_output_lens(self) -> List[List[int]]:
        """Build input_output_lens list from requests stages."""
        sorted_keys = sorted(self.requests.keys())
        return [[self.requests[idx].input_lens, self.requests[idx].output_lens] 
                for idx in sorted_keys]


class SweepConfig(BaseModel):
    """Configuration for sweep variables.
    
    All fields are lists. The experiment framework generates
    Cartesian product of all non-None variables.
    """
    
    # Naming aliases for file naming - defined at Config level
    NAMING_ALIASES: ClassVar[Dict[str, str]] = {
        "enable_flexi_flash_attn": "flexi",
        "gpu_memory_utilization": "gpu_util",
        "block_size": "blk",
    }
    
    enable_flexi_flash_attn: Optional[List[bool]] = None
    benchmark_config: Optional[List[SweepBenchmarkConfig]] = None
    gpu_memory_utilization: Optional[List[float]] = None
    block_size: Optional[List[Optional[int]]] = None
    
    def get_sweep_axes(self) -> Dict[str, List[Any]]:
        """Return all non-None sweep axes for Cartesian product."""
        axes = {}
        if self.enable_flexi_flash_attn is not None:
            axes['enable_flexi_flash_attn'] = self.enable_flexi_flash_attn
        if self.benchmark_config is not None:
            axes['benchmark_config'] = self.benchmark_config
        if self.gpu_memory_utilization is not None:
            axes['gpu_memory_utilization'] = self.gpu_memory_utilization
        if self.block_size is not None:
            axes['block_size'] = self.block_size
        return axes


class StaticVllmCfg(BaseModel):
    """Static vLLM configuration (non-sweep parameters)."""
    pipeline_parallel_size: int = 2
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8000
    chunked_prefill: bool = True
    enable_cuda_graph: bool = False
    enable_nsight: bool = False
    enable_flexi_flash_attn: bool = False
    block_size: Optional[int] = None
    head_addr: str = "head"
    port: int = 8000


class StaticBenchCfg(BaseModel):
    """Static benchmark configuration (non-sweep parameters)."""
    pattern_batch_size: int = 150
    profile: bool = False
    print_outputs: bool = False
    benchmark_script_path: str = "/root/vllm_workbench/vllm/benchmarks/benchmark_serving.py"
    burstiness: float = 100.0
    warmup: Optional[WarmupBenchCfg] = None


class StaticConfig(BaseModel):
    """Static configuration that applies to all sweep experiments."""
    model: ModelCfg = ModelCfg()
    network: NetworkCfg = NetworkCfg()
    vllm: StaticVllmCfg = StaticVllmCfg()
    benchmark: StaticBenchCfg = StaticBenchCfg()


class SweepTestConfig(BaseModel):
    """Single-project sweep test configuration.
    
    This is the top-level config for sweep_test experiments.
    No multi-project support - one config file = one project.
    """
    project: str
    type: str = "sweep_test"
    static_config: StaticConfig
    sweep_config: SweepConfig
    envs: Dict[str, str] = {}
    is_log_cover: bool = False


# =========================
# Spec Data Types (for passing to vLLM/benchmark)
# =========================

def extract_naming_vars(sweep_combo: Dict[str, Any]) -> Dict[str, Any]:
    """Extract vars_mapping from sweep combination using Config-defined aliases.
    
    Uses NAMING_ALIASES defined in SweepConfig and SweepBenchmarkConfig
    to build the vars_mapping dict for file naming.
    
    Args:
        sweep_combo: A single sweep combination dict from Cartesian product
    
    Returns:
        Dict mapping alias -> value for file naming
    """
    result = {}
    
    # Extract from SweepConfig direct fields
    for field_name, alias in SweepConfig.NAMING_ALIASES.items():
        if field_name in sweep_combo:
            result[alias] = sweep_combo[field_name]
    
    # Extract from benchmark_config (SweepBenchmarkConfig)
    bench_cfg: Optional[SweepBenchmarkConfig] = sweep_combo.get('benchmark_config')
    if bench_cfg:
        for field_name, alias in SweepBenchmarkConfig.NAMING_ALIASES.items():
            result[alias] = getattr(bench_cfg, field_name)
    
    return result


@dataclass
class VllmServerSpec:
    """Specification for starting a vLLM server.
    
    This contains all parameters needed to start vLLM.
    Naming aliases are defined in Config classes (SweepConfig, SweepBenchmarkConfig),
    not here in Spec.
    """
    # Model
    model_path: str
    model_name: str
    
    # Server settings
    pipeline_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    block_size: Optional[int]
    head_addr: str
    port: int
    
    # Features
    enable_flexi_flash_attn: bool = False
    chunked_prefill: bool = False
    enable_cuda_graph: bool = False
    enable_nsight: bool = False
    
    # Migration/Partition
    pp_layer_partition: str = ""
    pp_layer_config: Dict[int, str] = field(default_factory=dict)
    
    # Alternative configs for migration (pre-parsed)
    # Format: {"pp_layer_configs": {"0": [[0,31],[32,63]], ...}}
    alternative_configs: Dict[str, Any] = field(default_factory=dict)
    # Migration steps: request indices where config changes
    migration_steps: List[int] = field(default_factory=list)
    
    # Network
    rank_to_ip: Dict[int, str] = field(default_factory=dict)
    rank_to_node: Dict[int, str] = field(default_factory=dict)
    
    # Benchmark-related (for dynamic config)
    pattern_batch_size: int = 150
    
    # Paths (set during spec generation)
    metrics_csv_path: Optional[str] = None
    server_raw_log_path: Optional[str] = None
    
    def to_dynamic_cfg(self) -> Dict[str, Any]:
        """Generate dynamic config dict for -D flag."""
        cfg = {
            "enable_flexi_flash_attn": self.enable_flexi_flash_attn,
            "pp_layer_partition": self.pp_layer_partition,
            "pattern_batch_size": self.pattern_batch_size,
            "metrics_csv_path": self.metrics_csv_path,
            "alternative_configs": self.alternative_configs,
            "migration_steps": self.migration_steps,
        }
        # Only include rank_to_ip if non-empty
        if self.rank_to_ip:
            cfg["rank_to_ip"] = {str(k): v for k, v in self.rank_to_ip.items()}
        return cfg


@dataclass
class BenchmarkSpec:
    """Specification for running a benchmark.
    
    This contains all parameters needed to run benchmark.
    Naming aliases are defined in Config classes (SweepConfig, SweepBenchmarkConfig),
    not here in Spec.
    """
    # Server connection
    base_url: str
    model_path: str
    model_name: str
    benchmark_script_path: str
    benchmark_config_path: str
    metrics_file_path: str
    benchmark_log_path: str
    
    # Request config
    num_total_requests: int = 0
    request_rate: Dict[int, float] = field(default_factory=dict)
    input_output_lens: List[List[int]] = field(default_factory=list)
    pattern_batch_size: int = 150
    burstiness: float = 100.0
    
    # Features
    print_outputs: bool = False
    profile: bool = False
    
    # Optional warmup
    warmup: Optional[WarmupBenchCfg] = None

    def build_benchmark_config(self) -> Dict[str, Any]:
        """Assemble benchmark config payload to be written to disk."""
        # Build running_num_requests and data_num_requests from request_rate dict
        # running_num_requests: list of request counts per stage
        # data_num_requests: list of data generation counts per stage
        running_num_requests = []
        data_num_requests = []
        running_request_rates = []
        
        sorted_keys = sorted(self.request_rate.keys())
        for i, key in enumerate(sorted_keys):
            if i == 0:
                # First stage starts at 0
                count = sorted_keys[i+1] if i+1 < len(sorted_keys) else self.num_total_requests
            else:
                # Subsequent stages
                next_key = sorted_keys[i+1] if i+1 < len(sorted_keys) else self.num_total_requests
                count = next_key - key
            
            running_num_requests.append(count)
            data_num_requests.append(count)
            running_request_rates.append(self.request_rate[key])
        
        return {
            "num_total_requests": self.num_total_requests,
            "request_rate": {str(k): v for k, v in self.request_rate.items()},
            "running_num_requests": running_num_requests,
            "data_num_requests": data_num_requests,
            "running_request_rates": running_request_rates,
            "input_output_lens": self.input_output_lens,
            "pattern_batch_size": self.pattern_batch_size,
            "profile": self.profile,
            "print_outputs": self.print_outputs,
            "burstiness": self.burstiness,
            "warmup": self.warmup.model_dump() if self.warmup else None,
            "metrics_file_name": self.metrics_file_path,
        }

    def write_benchmark_config(self) -> None:
        """Persist benchmark config to benchmark_config_path."""
        config_file_path = Path(self.benchmark_config_path)
        config_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file_path, "w", encoding="utf-8") as f:
            json.dump(self.build_benchmark_config(), f, indent=2, ensure_ascii=False)

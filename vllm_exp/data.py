#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator
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
    # Number of times to repeat the benchmark (default 1 = single run)
    # After each repetition, reset_pipeline is called to restore initial config
    repetition: int = 1

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
        "repetition": "rep",
    }
    
    num_total_requests: int
    """Total number of requests for this benchmark configuration."""
    
    repetition: int = 1
    """Number of times to repeat the benchmark. After each repetition,
    reset_pipeline is called to return to the initial pp configuration."""
    
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


class SweepBenchmarkParams(BaseModel):
    """Individual benchmark parameters for parameter-sweep mode.
    
    All parameters here are lists for sweeping. Shared parameters like
    num_total_requests and repetition should be defined in static_config.benchmark.
    
    Example:
        sweep_config:
          benchmark:
            pp_layer_configs: ["32, 32", "20, 44"]
            request_rates: [1.0, 2.0]
            input_lens: [500, 800]
            output_lens: [100, 200]
        
        static_config:
          benchmark:
            num_total_requests: 100
            repetition: 2
    """
    pp_layer_configs: Optional[List[str]] = None
    """List of pp_layer_partition strings. Each becomes a separate experiment."""
    
    request_rates: Optional[List[float]] = None
    """List of request rates to sweep."""
    
    input_lens: Optional[List[int]] = None
    """List of input lengths to sweep."""
    
    output_lens: Optional[List[int]] = None
    """List of output lengths to sweep."""


class SweepVllmParams(BaseModel):
    """Individual vLLM parameters for parameter-sweep mode."""
    enable_flexi_flash_attn: Optional[List[bool]] = None


class SweepConfig(BaseModel):
    """Configuration for sweep variables.
    
    Supports two modes:
    1. benchmark_config mode: Provide complete SweepBenchmarkConfig list
    2. Parameter sweep mode: Provide individual parameters via 'benchmark' and 'vllm'
       which will be combined via Cartesian product
    
    Example (benchmark_config mode):
        sweep_config:
          benchmark_config:
            - num_total_requests: 100
              pp_layer_config: {0: "32, 32"}
              requests: {0: {request_rate: 1.0, input_lens: 500, output_lens: 100}}
    
    Example (parameter sweep mode):
        sweep_config:
          vllm:
            enable_flexi_flash_attn: [true, false]
          benchmark:
            pp_layer_configs: ["32, 32", "20, 44"]
            request_rates: [1.0, 2.0]
            input_lens: [500, 800]
            output_lens: [100, 200]
    """
    
    # Naming aliases for file naming - only for actual sweep parameters
    NAMING_ALIASES: ClassVar[Dict[str, str]] = {
        "enable_flexi_flash_attn": "flexi",
    }
    
    # Mode 1: Complete benchmark_config list
    benchmark_config: Optional[List[SweepBenchmarkConfig]] = None
    
    # Mode 2: Individual sweep parameters
    vllm: Optional[SweepVllmParams] = None
    benchmark: Optional[SweepBenchmarkParams] = None
    
    # Note: gpu_memory_utilization and block_size are NOT sweep parameters.
    # They should only be defined in static_config.vllm.
    # Only enable_flexi_flash_attn can be swept via sweep_config.vllm.
    
    def get_sweep_axes(
        self,
        static_benchmark_cfg: Optional['StaticBenchCfg'] = None
    ) -> Dict[str, List[Any]]:
        """Return all non-None sweep axes for Cartesian product.
        
        Handles both benchmark_config mode and parameter sweep mode.
        
        Args:
            static_benchmark_cfg: Static benchmark config for shared parameters
                                  (num_total_requests, repetition) in parameter-sweep mode.
        """
        axes = {}
        
        # enable_flexi_flash_attn can only come from sweep_config.vllm
        if self.vllm is not None and self.vllm.enable_flexi_flash_attn is not None:
            axes['enable_flexi_flash_attn'] = self.vllm.enable_flexi_flash_attn
        
        # Mode 1: benchmark_config provided directly
        if self.benchmark_config is not None:
            axes['benchmark_config'] = self.benchmark_config
        # Mode 2: Generate benchmark_config from individual parameters
        elif self.benchmark is not None:
            assert static_benchmark_cfg is not None
            num_requests = static_benchmark_cfg.num_total_requests
            repetition = static_benchmark_cfg.repetition
            axes['benchmark_config'] = self._generate_benchmark_configs_from_params(
                num_total_requests=num_requests,
                repetition=repetition
            )
        
        return axes
    
    def _generate_benchmark_configs_from_params(
        self,
        num_total_requests: int = 100,
        repetition: int = 1
    ) -> List[SweepBenchmarkConfig]:
        """Generate SweepBenchmarkConfig list from individual benchmark parameters.
        
        Creates Cartesian product of pp_layer_configs, request_rates, input_lens, output_lens.
        num_total_requests and repetition are passed from static_config.benchmark.
        
        Args:
            num_total_requests: Shared across all combinations (from static_config.benchmark)
            repetition: Shared across all combinations (from static_config.benchmark)
        """
        import itertools
        
        if self.benchmark is None:
            return []
        
        params = self.benchmark
        
        # Validate required parameters - all must be explicitly provided
        missing_params = []
        if not params.pp_layer_configs:
            missing_params.append("pp_layer_configs")
        if not params.request_rates:
            missing_params.append("request_rates")
        if not params.input_lens:
            missing_params.append("input_lens")
        if not params.output_lens:
            missing_params.append("output_lens")
        
        if missing_params:
            raise ValueError(
                f"Missing required sweep_config.benchmark parameters: {missing_params}. "
                "All parameters must be explicitly provided as lists."
            )
        
        # At this point all params are guaranteed to be non-None
        pp_configs: List[str] = params.pp_layer_configs  # type: ignore[assignment]
        request_rates: List[float] = params.request_rates  # type: ignore[assignment]
        input_lens: List[int] = params.input_lens  # type: ignore[assignment]
        output_lens: List[int] = params.output_lens  # type: ignore[assignment]
        
        configs = []
        
        # Generate Cartesian product
        for pp_config, rr, in_len, out_len in itertools.product(
            pp_configs, request_rates, input_lens, output_lens
        ):
            # Build SweepBenchmarkConfig
            config = SweepBenchmarkConfig(
                num_total_requests=num_total_requests,
                repetition=repetition,
                pp_layer_config={0: pp_config.replace(" ", "")},
                requests={
                    0: RequestStageConfig(
                        request_rate=rr,
                        input_lens=in_len,
                        output_lens=out_len,
                    )
                }
            )
            configs.append(config)
        
        return configs


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
    """Static benchmark configuration (non-sweep parameters).
    
    Contains shared parameters that apply to all sweep combinations.
    These should NOT be redefined in sweep_config to avoid conflicts.
    """
    pattern_batch_size: int = 150
    profile: bool = False
    print_outputs: bool = False
    benchmark_script_path: str = "/root/vllm_workbench/vllm/benchmarks/benchmark_serving.py"
    burstiness: float = 100.0
    warmup: Optional[WarmupBenchCfg] = None
    
    # Shared parameters for parameter-sweep mode
    num_total_requests: int = 100
    """Number of total requests (shared across all sweep combinations)."""
    
    repetition: int = 1
    """Number of repetitions per experiment (shared across all sweep combinations)."""


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
    
    Configuration Conflict Rules:
    1. If using parameter-sweep mode (sweep_config.benchmark), shared params
       like num_total_requests and repetition must come from static_config.benchmark
    2. If using benchmark_config mode, repetition/num_total_requests are per-config
    3. vLLM params: if defined in sweep_config, they become sweep axes;
       static_config provides fallback values
    """
    project: str
    type: str = "sweep_test"
    static_config: StaticConfig
    sweep_config: SweepConfig
    envs: Dict[str, str] = {}
    is_log_cover: bool = False
    
    @model_validator(mode='after')
    def validate_no_conflicts(self) -> 'SweepTestConfig':
        """Validate that there are no configuration conflicts between modes."""
        errors = []
        
        # Check for mode conflicts: benchmark_config and benchmark should be mutually exclusive
        if (self.sweep_config.benchmark_config is not None 
            and self.sweep_config.benchmark is not None):
            errors.append(
                "Configuration conflict: 'sweep_config.benchmark_config' and "
                "'sweep_config.benchmark' are mutually exclusive. "
                "Use benchmark_config for complete configs or benchmark for parameter sweep."
            )
        
        # Note: enable_flexi_flash_attn can appear in:
        # - static_config.vllm.enable_flexi_flash_attn (single value, fallback)
        # - sweep_config.vllm.enable_flexi_flash_attn (list for sweeping)
        # When sweep_config.vllm.enable_flexi_flash_attn is provided, it overrides static_config.
        # No conflict check needed - sweep takes precedence over static.
        
        # Note: gpu_memory_utilization and block_size can ONLY appear in static_config.vllm
        # They are not sweep parameters.
        
        if errors:
            raise ValueError("\n".join(errors))
        
        return self
    
    def get_sweep_axes(self) -> Dict[str, List[Any]]:
        """Convenience method to get sweep axes with static config applied."""
        return self.sweep_config.get_sweep_axes(
            static_benchmark_cfg=self.static_config.benchmark
        )


# =========================
# Experiment Config (merged from static + sweep via Cartesian product)
# =========================

@dataclass
class ExpModelConfig:
    """Model configuration for a single experiment."""
    path: str
    name: str


@dataclass
class ExpNetworkConfig:
    """Network configuration for a single experiment."""
    rank_to_ip: Dict[int, str] = field(default_factory=dict)
    rank_to_node: Dict[int, str] = field(default_factory=dict)


@dataclass
class ExpVllmConfig:
    """vLLM server configuration for a single experiment.
    
    Merged from:
    - static_config.vllm (base values)
    - sweep_config.vllm (overrides like enable_flexi_flash_attn)
    """
    pipeline_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    block_size: Optional[int]
    head_addr: str
    port: int
    enable_flexi_flash_attn: bool
    chunked_prefill: bool
    enable_cuda_graph: bool
    enable_nsight: bool
    
    # Pipeline partition (from sweep benchmark_config)
    pp_layer_partition: str  # Initial partition, e.g. "32,32"
    pp_layer_config: Dict[int, str] = field(default_factory=dict)  # {0: "32,32", 100: "20,44"}
    
    @property
    def has_migration(self) -> bool:
        """Returns True if there are multiple pp_layer_config entries."""
        return len(self.pp_layer_config) > 1
    
    @property
    def base_url(self) -> str:
        return f"http://{self.head_addr}:{self.port}"
    
    def get_alternative_configs(self) -> Dict[str, Any]:
        """Parse pp_layer_config into alternative_configs format.
        
        Input format: {0: "32,32", 100: "12,52"}
        Output format: {"pp_layer_configs": {"0": [[0,31],[32,63]], ...}}
        """
        sorted_keys = sorted(self.pp_layer_config.keys())
        alternative_configs_inner = {}
        
        for i, req_idx in enumerate(sorted_keys):
            pp_str = self.pp_layer_config[req_idx].replace(" ", "")
            layers = [int(x.strip()) for x in pp_str.split(",")]
            ranges = []
            start = 0
            for num_layers in layers:
                end = start + num_layers - 1
                ranges.append([start, end])
                start = end + 1
            alternative_configs_inner[str(i)] = ranges
        
        return {"pp_layer_configs": alternative_configs_inner}
    
    def get_migration_steps(self) -> List[int]:
        """Get request indices where configuration changes."""
        return [k for k in sorted(self.pp_layer_config.keys()) if k > 0]


@dataclass
class ExpBenchmarkConfig:
    """Benchmark configuration for a single experiment.
    
    Merged from:
    - static_config.benchmark (shared params like pattern_batch_size, burstiness)
    - sweep benchmark_config (experiment-specific params like num_total_requests, request_rate)
    """
    # From sweep benchmark_config (required fields)
    num_total_requests: int
    repetition: int
    request_rate: float  # Initial request rate
    input_lens: int  # Initial input length
    output_lens: int  # Initial output length
    
    # From sweep benchmark_config (with defaults)
    request_rate_dict: Dict[int, float] = field(default_factory=dict)  # {0: 1.0, 100: 2.0}
    input_output_lens: List[List[int]] = field(default_factory=list)  # [[500, 100], [800, 200]]
    
    # From static_config.benchmark
    pattern_batch_size: int = 150
    burstiness: float = 100.0
    print_outputs: bool = False
    profile: bool = False
    benchmark_script_path: str = ""
    warmup: Optional[WarmupBenchCfg] = None


@dataclass
class ExperimentConfig:
    """A complete experiment configuration merged from static and sweep configs.
    
    This class represents a single experiment from the Cartesian product of
    static_config × sweep_config. It contains nested sub-configs for clarity.
    
    Structure:
    - model: ExpModelConfig (from static_config.model)
    - network: ExpNetworkConfig (from static_config.network)
    - vllm: ExpVllmConfig (merged from static_config.vllm + sweep_config.vllm + sweep benchmark_config.pp_layer_config)
    - benchmark: ExpBenchmarkConfig (merged from static_config.benchmark + sweep benchmark_config)
    """
    
    # Naming aliases for file naming
    NAMING_ALIASES: ClassVar[Dict[str, str]] = {
        "enable_flexi_flash_attn": "flexi",
        "pp_layer_partition": "pp",
        "request_rate": "rr",
        "has_migration": "mig",
        "input_lens": "in",
        "output_lens": "out",
        "num_total_requests": "n_req",
        "repetition": "rep",
    }
    
    model: ExpModelConfig
    network: ExpNetworkConfig
    vllm: ExpVllmConfig
    benchmark: ExpBenchmarkConfig
    
    def get_naming_vars(self) -> Dict[str, Any]:
        """Generate vars_mapping for file naming using NAMING_ALIASES."""
        return {
            "flexi": self.vllm.enable_flexi_flash_attn,
            "pp": self.vllm.pp_layer_partition,
            "rr": self.benchmark.request_rate,
            "mig": self.vllm.has_migration,
            "in": self.benchmark.input_lens,
            "out": self.benchmark.output_lens,
            "n_req": self.benchmark.num_total_requests,
            "rep": self.benchmark.repetition,
        }
    
    @classmethod
    def iter_from_sweep_test_config(
        cls,
        sweep_test_cfg: 'SweepTestConfig',
    ) -> 'Iterator[ExperimentConfig]':
        """Iterate over all ExperimentConfigs from Cartesian product of sweep axes.
        
        This is the main entry point for generating experiment configurations.
        It combines static_config with each sweep combination to produce
        ExperimentConfig instances.
        
        Args:
            sweep_test_cfg: The complete sweep test configuration containing
                           static_config and sweep_config
        
        Yields:
            ExperimentConfig for each combination in the Cartesian product
        
        Example:
            for exp_cfg in ExperimentConfig.iter_from_sweep_test_config(sweep_test_cfg):
                # exp_cfg is a complete ExperimentConfig ready for execution
                server_spec = exp_cfg.to_vllm_server_spec()
                bench_spec = exp_cfg.to_benchmark_spec(...)
        """
        import itertools
        
        static_cfg = sweep_test_cfg.static_config
        sweep_axes = sweep_test_cfg.get_sweep_axes()
        
        if not sweep_axes:
            return
        
        # Build Cartesian product of all sweep axes
        # sweep_axes: {"enable_flexi_flash_attn": [True, False], "benchmark_config": [cfg1, cfg2, ...]}
        axis_names = list(sweep_axes.keys())
        axis_values = [sweep_axes[name] for name in axis_names]
        
        for combo in itertools.product(*axis_values):
            # combo is a tuple of values, one per axis
            combo_dict = dict(zip(axis_names, combo))
            
            # Extract sweep parameters from combination
            bench_cfg: SweepBenchmarkConfig = combo_dict['benchmark_config']
            enable_flexi: Optional[bool] = combo_dict.get('enable_flexi_flash_attn')
            
            # Create ExperimentConfig from this combination
            yield cls._create_from_combo(static_cfg, bench_cfg, enable_flexi)
    
    @classmethod
    def _create_from_combo(
        cls,
        static_cfg: 'StaticConfig',
        bench_cfg: SweepBenchmarkConfig,
        enable_flexi_flash_attn: Optional[bool] = None,
    ) -> 'ExperimentConfig':
        """Create a single ExperimentConfig from static config and one sweep combination.
        
        Internal method used by iter_from_sweep_test_config.
        
        Args:
            static_cfg: Static configuration
            bench_cfg: Sweep benchmark configuration (from sweep axes)
            enable_flexi_flash_attn: Override from sweep axis (if sweeping), otherwise use static
        """
        # Resolve enable_flexi_flash_attn: sweep overrides static
        flexi = enable_flexi_flash_attn if enable_flexi_flash_attn is not None else static_cfg.vllm.enable_flexi_flash_attn
        
        # Get initial values from bench_cfg
        sorted_pp_keys = sorted(bench_cfg.pp_layer_config.keys())
        sorted_req_keys = sorted(bench_cfg.requests.keys())
        
        initial_pp = bench_cfg.pp_layer_config[sorted_pp_keys[0]].replace(" ", "")
        initial_req_cfg = bench_cfg.requests[sorted_req_keys[0]]
        
        # Build sub-configs
        model_cfg = ExpModelConfig(
            path=static_cfg.model.path,
            name=static_cfg.model.name,
        )
        
        network_cfg = ExpNetworkConfig(
            rank_to_ip=static_cfg.network.rank_to_ip,
            rank_to_node=static_cfg.network.rank_to_node,
        )
        
        vllm_cfg = ExpVllmConfig(
            pipeline_parallel_size=static_cfg.vllm.pipeline_parallel_size,
            gpu_memory_utilization=static_cfg.vllm.gpu_memory_utilization,
            max_model_len=static_cfg.vllm.max_model_len,
            block_size=static_cfg.vllm.block_size,
            head_addr=static_cfg.vllm.head_addr,
            port=static_cfg.vllm.port,
            enable_flexi_flash_attn=flexi,
            chunked_prefill=static_cfg.vllm.chunked_prefill,
            enable_cuda_graph=static_cfg.vllm.enable_cuda_graph,
            enable_nsight=static_cfg.vllm.enable_nsight,
            pp_layer_partition=initial_pp,
            pp_layer_config={k: v.replace(" ", "") for k, v in bench_cfg.pp_layer_config.items()},
        )
        
        benchmark_cfg = ExpBenchmarkConfig(
            num_total_requests=bench_cfg.num_total_requests,
            repetition=bench_cfg.repetition,
            request_rate=initial_req_cfg.request_rate,
            request_rate_dict={idx: cfg.request_rate for idx, cfg in bench_cfg.requests.items()},
            input_lens=initial_req_cfg.input_lens,
            output_lens=initial_req_cfg.output_lens,
            input_output_lens=bench_cfg.get_input_output_lens(),
            pattern_batch_size=static_cfg.benchmark.pattern_batch_size,
            burstiness=static_cfg.benchmark.burstiness,
            print_outputs=static_cfg.benchmark.print_outputs,
            profile=static_cfg.benchmark.profile,
            benchmark_script_path=static_cfg.benchmark.benchmark_script_path,
            warmup=static_cfg.benchmark.warmup,
        )
        
        return cls(
            model=model_cfg,
            network=network_cfg,
            vllm=vllm_cfg,
            benchmark=benchmark_cfg,
        )


# =========================
# Spec Data Types (for passing to vLLM/benchmark processes)
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
    
    # Repetition config
    repetition: int = 1
    """Number of times to repeat the benchmark. After each repetition,
    reset_pipeline is called to return to the initial pp configuration."""
    
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
            "repetition": self.repetition,
        }

    def write_benchmark_config(self) -> None:
        """Persist benchmark config to benchmark_config_path."""
        config_file_path = Path(self.benchmark_config_path)
        config_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file_path, "w", encoding="utf-8") as f:
            json.dump(self.build_benchmark_config(), f, indent=2, ensure_ascii=False)

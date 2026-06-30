#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator
import json
import os
from pathlib import Path
import typer
from pydantic import BaseModel, Field, model_validator, field_validator, ValidationInfo
from rich.console import Console


app = typer.Typer(no_args_is_help=True)
C = Console()


def _parse_size_env(name: str, scale: int) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {raw}. Must be an integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got: {value}")
    return value * scale


def get_page_attention_block_size_bytes_from_env() -> Optional[int]:
    """Return the requested K+V PageAttention block size in bytes.

    This is a semantic experiment knob. It is converted to vLLM's token
    block_size at launch time using the selected model's KV-cache shape.
    """
    for prefix in ("VLLM_PAGE_ATTENTION_BLOCK_SIZE",
                   "KVCACHED_PAGE_ATTENTION_BLOCK_SIZE"):
        for suffix, scale in (("BYTES", 1), ("KB", 1024),
                              ("MB", 1024 * 1024)):
            parsed = _parse_size_env(f"{prefix}_{suffix}", scale)
            if parsed is not None:
                return parsed
    return None


def _dtype_size_bytes(dtype_name: Optional[str]) -> int:
    if not dtype_name:
        return 2
    normalized = dtype_name.lower().replace("torch.", "")
    if normalized in ("float16", "fp16", "half", "bfloat16", "bf16"):
        return 2
    if normalized in ("float32", "fp32"):
        return 4
    if normalized in ("float8", "fp8", "float8_e4m3fn", "float8_e5m2"):
        return 1
    raise ValueError(
        f"Unsupported torch_dtype for PageAttention block derivation: {dtype_name}"
    )


def _load_model_config(model_path: str) -> Dict[str, Any]:
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.exists():
        raise ValueError(
            "Cannot derive vLLM token block_size from PageAttention block "
            f"size because model config is missing: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Model config is not a JSON object: {config_path}")
    return data


def derive_token_block_size_from_page_attention_block(
    *,
    model_path: str,
    explicit_block_size: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    """Resolve vLLM token block_size from a K+V PageAttention byte target."""
    page_attention_block_size = get_page_attention_block_size_bytes_from_env()
    if page_attention_block_size is None:
        return explicit_block_size, None

    model_cfg = _load_model_config(model_path)
    num_attention_heads = model_cfg.get("num_attention_heads")
    num_kv_heads = model_cfg.get("num_key_value_heads", num_attention_heads)
    head_dim = model_cfg.get("head_dim")
    if head_dim is None:
        hidden_size = model_cfg.get("hidden_size")
        if hidden_size is None or num_attention_heads is None:
            raise ValueError(
                "Cannot derive head_dim from model config; expected head_dim "
                "or hidden_size + num_attention_heads.")
        head_dim = int(hidden_size) // int(num_attention_heads)
    if num_kv_heads is None:
        raise ValueError(
            "Cannot derive PageAttention block size: model config lacks "
            "num_key_value_heads and num_attention_heads.")

    dtype_size = _dtype_size_bytes(model_cfg.get("torch_dtype"))
    bytes_per_token = 2 * int(num_kv_heads) * int(head_dim) * dtype_size
    if page_attention_block_size % bytes_per_token != 0:
        raise ValueError(
            "Requested K+V PageAttention block size is not an integer number "
            "of tokens: "
            f"page_attention_block_size={page_attention_block_size} bytes, "
            f"bytes_per_token={bytes_per_token}")
    derived_block_size = page_attention_block_size // bytes_per_token
    if derived_block_size <= 0:
        raise ValueError(
            "Derived token block_size must be positive, got "
            f"{derived_block_size}")
    if derived_block_size % 16 != 0:
        raise ValueError(
            "Derived token block_size must be divisible by 16 for the "
            f"FlashAttention backend, got {derived_block_size}.")
    if explicit_block_size is not None and explicit_block_size != derived_block_size:
        raise ValueError(
            "Do not set token block_size together with "
            "VLLM_PAGE_ATTENTION_BLOCK_SIZE_* unless they match. "
            f"explicit block_size={explicit_block_size}, derived "
            f"block_size={derived_block_size} from "
            f"{page_attention_block_size} bytes.")
    return int(derived_block_size), int(page_attention_block_size)

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
        "attention_kernel": cfg.vllm.attention_kernel,
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
        "attention_kernel": "kernel"
    }

class ModelCfg(BaseModel):
    path: str = "/root/.cache/huggingface/Qwen3-32B-AWQ"
    name: str = "Qwen3-32B-AWQ"

class VllmCfg(BaseModel):
    pipeline_parallel_size: int = 2
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8000
    max_num_batched_tokens: Optional[int] = None  # Max tokens per batch for chunked prefill (default: 2048 when chunked_prefill=True)
    max_num_seqs: Optional[int] = None  # Max concurrent sequences; keep unset to use vLLM default.
    chunked_prefill: bool = True
    enable_cuda_graph: bool = False
    enable_nsight: bool = False
    attention_kernel: str = "direct"  # "flash", "flexi", or "direct"
    block_size: Optional[int] = None  # KV cache block size (1, 8, 16, 32, 64, 128, 256, 512), None means use vLLM default. V0 only supports up to 32.
    head_addr: str = "head"
    port: int = 8000
    ray_port: int = 6379
    start_pp_layer_partitions: list[str] = ["8,56"]
    disable_memory_overhead_monitor: bool = False

class MigrationCfg(BaseModel):
    is_migration: bool = False 
    is_compact_kv: bool = False
    alternative_configs: dict[str, list[int]] = {}
    migration_steps: list[int] = []
    compact_steps: list[int] = []
    tester_start_step: Optional[int] = None
    memory_stress_tester: Optional[Dict[str, Any]] = None
    migration_mode: str = "async"  # "async" or "sync" - determines which migration method to use
    weight_loading_mode: str = "async"
    """Weight loading mode for migration: 'async' or 'sync'."""

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
    num_total_requests: Optional[int] = None
    """Total number of requests to sample from dataset. Used by sharegpt/burstgpt datasets."""
    running_num_requests: list[int] = []
    data_num_requests: list[int] = []
    pattern_batch_size: int = 150
    sweep_request_rates: list[float] = []
    running_request_rates: list[float] = []
    profile: bool = False
    input_output_lens: list[list[int]]
    arrival_trace: Optional[Dict[str, Any]] = None
    """Optional native arrival trace replay config passed to benchmark_serving."""
    # Whether to print each request's generated output in benchmark logs
    print_outputs: bool = False
    # Path to benchmark script
    benchmark_script_path: str = "/root/vllm_workbench/vllm/benchmarks/benchmark_serving.py"
    # Burstiness factor for request generation (default 1.0 = Poisson process)
    # Higher values (e.g., 100) result in more uniform/constant request rate
    burstiness: float = 100.0
    # Dataset configuration
    dataset_name: str = "pattern"
    """Dataset type: 'pattern' (default), 'sharegpt', 'burstgpt', 'random', 'sonnet'."""
    dataset_path: Optional[str] = None
    """Path to dataset file. Required for: sharegpt (JSON), burstgpt (CSV), sonnet (TXT).
    Default paths: ShareGPT=/home/bxb1/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
                   BurstGPT=/home/bxb1/data/datasets/BurstGPT_without_fails_2.csv"""
    sharegpt_output_len: Optional[int] = None
    """Output length for ShareGPT dataset. If None, uses actual completion length."""
    # Optional metrics output path when running via vllm_exp
    metrics_file_name: Optional[str] = None
    # Number of times to repeat the benchmark (default 1 = single run)
    # After each repetition, set_pp_config is called to restore initial config
    repetition: int = 1
    restart_server_between_repetitions: bool = False
    """Whether to restart the vLLM server between benchmark repetitions.
    When False, repetitions run on the same server process and reset PP config via API."""
    
    # Pipeline config for repetition reset (used when repetition > 1)
    initial_pp_config: Optional[List[List[int]]] = None
    """Initial PP layer config to restore between repetitions.
    Format: [[start, end], [start, end], ...] per rank."""
    alternative_configs: Optional[Dict[str, Any]] = None
    """Migration target configs. Format: {"pp_layer_configs": {"0": [...], "1": [...]}}.
    Passed to set_pp_config when resetting between repetitions."""
    migration_steps: Optional[List[int]] = None
    """Request indices at which migration is triggered.
    Passed to set_pp_config when resetting between repetitions."""
    migration_mode: Optional[str] = None
    """Migration mode: "async", "async_fast", or "sync"."""
    weight_loading_mode: Optional[str] = None
    """Weight loading mode for migration reset: "async" or "sync"."""

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


class PipelineStagePlacement(BaseModel):
    """Placement for one logical PP stage or global worker rank.

    For the current VMXpert experiments tensor_parallel_size is 1, so
    pp_stage and rank are usually the same value. Keeping both fields lets a
    config describe non-identity stage-to-rank mappings when vLLM is launched
    with a matching pipeline_stage_to_rank map.
    """

    pp_stage: Optional[int] = None
    rank: Optional[int] = None
    node: str
    ip: Optional[str] = None

    @model_validator(mode="after")
    def validate_stage_or_rank(self) -> "PipelineStagePlacement":
        if self.pp_stage is None and self.rank is None:
            raise ValueError("network.placements entries must define pp_stage or rank")
        return self


class NetworkCfg(BaseModel):
    delays: List[float] = [0]
    # 可选：pipeline 并行各 rank 的可达 IP，供 KV synchronizer 双向通道使用
    rank_to_ip: Dict[int, str] = {}
    # 可选：指定每个 rank 对应的 Ray 节点（hostname 或 IP），用于跨节点 worker 部署
    # 示例：{0: "node1", 1: "node2"} 表示 rank 0 部署在 node1，rank 1 部署在 node2
    rank_to_node: Dict[int, str] = {}
    # 可选：指定逻辑 PP stage 对应的 global rank。TP=1 时 rank 就是 GPU worker rank。
    # 示例：{0: 0, 1: 1, 2: 2, 3: 3}
    pipeline_stage_to_rank: Dict[int, int] = {}
    # 可选：更可读的 placement 写法，会自动展开为 rank_to_node/rank_to_ip。
    # 示例：
    # placements:
    #   - {pp_stage: 0, node: node-a, ip: node-a}
    #   - {pp_stage: 1, node: node-a, ip: node-a}
    #   - {pp_stage: 2, node: node-b, ip: node-b}
    #   - {pp_stage: 3, node: node-b, ip: node-b}
    placements: List[PipelineStagePlacement] = []

    def resolve_worker_placement(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int = 1,
    ) -> tuple[Dict[int, str], Dict[int, str], Dict[int, int]]:
        """Resolve high-level placement into rank_to_ip/node and stage_to_rank."""
        rank_to_ip = dict(self.rank_to_ip)
        rank_to_node = dict(self.rank_to_node)
        stage_to_rank = dict(self.pipeline_stage_to_rank)

        for placement in self.placements:
            rank = placement.rank
            if placement.pp_stage is not None:
                default_rank = placement.pp_stage * tensor_parallel_size
                if rank is None:
                    rank = default_rank
                stage_to_rank.setdefault(placement.pp_stage, rank)
            assert rank is not None
            rank_to_node[rank] = placement.node
            if placement.ip is not None:
                rank_to_ip[rank] = placement.ip

        if stage_to_rank:
            expected_stages = set(range(pipeline_parallel_size))
            actual_stages = set(stage_to_rank)
            if actual_stages != expected_stages:
                raise ValueError(
                    "network.pipeline_stage_to_rank/placements must cover every "
                    f"PP stage 0..{pipeline_parallel_size - 1}; got {sorted(actual_stages)}"
                )
            ranks = list(stage_to_rank.values())
            if len(set(ranks)) != len(ranks):
                raise ValueError(f"Duplicate ranks in pipeline_stage_to_rank: {stage_to_rank}")
            world_size = pipeline_parallel_size * tensor_parallel_size
            invalid_ranks = [rank for rank in ranks if rank < 0 or rank >= world_size]
            if invalid_ranks:
                raise ValueError(
                    f"pipeline_stage_to_rank contains ranks outside world size {world_size}: "
                    f"{invalid_ranks}"
                )
            if tensor_parallel_size > 1:
                invalid_bases = [
                    rank for rank in ranks if rank % tensor_parallel_size != 0
                ]
                if invalid_bases:
                    raise ValueError(
                        "With tensor_parallel_size > 1, pipeline_stage_to_rank values "
                        f"must be TP-group base ranks; got {invalid_bases}"
                    )

        return rank_to_ip, rank_to_node, stage_to_rank

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
    set_pp_config is called to return to the initial pp configuration."""
    
    pp_layer_config: Dict[int, str]
    """Pipeline layer partition configs indexed by request number.
    Example: {0: "32,32", 100: "12,52"} means use "32,32" for requests 0-99,
    then switch to "12,52" at request 100. Multiple configs imply migration."""
    
    requests: Dict[int, RequestStageConfig]
    """Request configurations indexed by request number.
    Example: {0: {request_rate: 1.8, input_lens: 800, output_lens: 64}, ...}"""

    arrival_trace: Optional[Dict[str, Any]] = None
    """Optional native arrival trace replay config for benchmark_serving."""
    
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
    
    Supports two benchmark sweep styles:
    1. Flat parameter sweep: use request_rates/input_lens/output_lens lists
    2. Segmented request sweep: use requests with staged request definitions

    Shared parameters like num_total_requests and repetition usually come from
    static_config.benchmark, but can also be overridden per sweep block.
    
        Example:
                sweep_config:
                    benchmark:
                        pp_layer_configs: ["32, 32", "20, 44"]
                        request_rates: [1.0, 2.0]
                        input_lens: [500, 800]
                        output_lens: [100, 200]

        Example with segmented requests:
                sweep_config:
                    benchmark:
                        pp_layer_configs: ["32, 32", "20, 44"]
                        requests:
                            0:
                                request_rate: 1.8
                                input_lens: 800
                                output_lens: 64
                            100:
                                request_rate: 1.5
                                input_lens: 1024
                                output_lens: 128
        
        static_config:
          benchmark:
            num_total_requests: 100
            repetition: 2
    """

    num_total_requests: Optional[int] = None
    """Optional override for total requests in this benchmark sweep block."""

    repetition: Optional[int] = None
    """Optional override for repetition in this benchmark sweep block."""

    pp_layer_configs: Optional[List[str]] = None
    """List of pp_layer_partition strings. Each becomes a separate experiment."""
    
    request_rates: Optional[List[float]] = None
    """List of request rates to sweep."""
    
    input_lens: Optional[List[int]] = None
    """List of input lengths to sweep."""
    
    output_lens: Optional[List[int]] = None
    """List of output lengths to sweep."""

    requests: Optional[Dict[int, RequestStageConfig] | List[Dict[int, RequestStageConfig]]] = None
    """Segmented request schedule(s) for sweeping pp_layer_configs.

    Can be either:
    - A single staged request dict: {0: {...}, 100: {...}}
    - A list of staged request dicts, each becoming an additional sweep axis item
    """

    @model_validator(mode='after')
    def validate_request_sweep_mode(self) -> 'SweepBenchmarkParams':
        """Validate flat-vs-segmented benchmark sweep mode usage."""
        has_segmented_requests = self.requests is not None
        has_flat_request_sweep = any(
            value is not None
            for value in (self.request_rates, self.input_lens, self.output_lens)
        )

        if has_segmented_requests and has_flat_request_sweep:
            raise ValueError(
                "sweep_config.benchmark 不能同时定义 'requests' 和 "
                "'request_rates'/'input_lens'/'output_lens'。"
            )

        if not has_segmented_requests and not all(
            value is not None
            for value in (self.request_rates, self.input_lens, self.output_lens)
        ):
            raise ValueError(
                "sweep_config.benchmark 必须提供以下两种配置方式之一：\n"
                "1. request_rates + input_lens + output_lens\n"
                "2. requests（分段请求配置）"
            )

        return self


class SweepVllmParams(BaseModel):
    """Individual vLLM parameters for parameter-sweep mode."""
    attention_kernel: Optional[List[str]] = None
    """List of attention kernels to sweep: 'flash', 'flexi', 'direct'."""
    weight_chunk_size_mb: Optional[List[float]] = None
    """List of weight chunk sizes (MB) for sweep. Affects weight loading during migration."""
    migration_approach: Optional[List[str]] = None
    """List of migration approaches to sweep: 'sync', 'async', or 'async_fast'."""
    weight_loading_mode: Optional[List[str]] = None
    """List of weight loading modes to sweep: 'sync' or 'async'."""
    fixed_num_gpu_blocks: Optional[List[int]] = None
    """List of fixed KV cache block counts to sweep. -1 means auto. Positive value fixes the block count."""
    block_size: Optional[List[int]] = None
    """List of KV cache block sizes to sweep (1, 8, 16, 32, 64, 128, 256, 512). V0 only supports up to 32."""
    use_vmm: Optional[List[bool]] = None
    """List of use_vmm values to sweep. True=VMM, False=cudaMallocAsync (has memory leak, for testing)."""
    enable_kv_resize: Optional[List[bool]] = None
    """List of enable_kv_resize values to sweep. True=allow resize during migration, False=disable."""
    enable_cpu_weight_cache: Optional[List[bool]] = None
    """List of enable_cpu_weight_cache values to sweep. True=preload weights to CPU, False=load from disk on demand."""


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
            attention_kernel: [flash, flexi, direct]
          benchmark:
            pp_layer_configs: ["32, 32", "20, 44"]
            request_rates: [1.0, 2.0]
            input_lens: [500, 800]
            output_lens: [100, 200]
    """
    
    # Naming aliases for file naming - only for actual sweep parameters
    NAMING_ALIASES: ClassVar[Dict[str, str]] = {
        "attention_kernel": "kernel",
        "block_size": "blk",
        "use_vmm": "vmm",
        "weight_loading_mode": "wl",
    }
    
    # Mode 1: Complete benchmark_config list
    benchmark_config: Optional[List[SweepBenchmarkConfig]] = None
    
    # Mode 2: Individual sweep parameters
    vllm: Optional[SweepVllmParams] = None
    benchmark: Optional[SweepBenchmarkParams] = None
    
    # Note: gpu_memory_utilization and block_size are NOT sweep parameters.
    # They should only be defined in static_config.vllm.
    # Commonly swept vLLM parameters include attention_kernel,
    # weight_chunk_size_mb, and weight_loading_mode.
    
    def get_sweep_axes(
        self,
        static_benchmark_cfg: Optional['StaticBenchCfg'] = None
    ) -> Dict[str, List[Any]]:
        """Return all non-None sweep axes for Cartesian product.
        
        Handles three modes:
        1. benchmark_config mode: Complete SweepBenchmarkConfig list provided directly
        2. Parameter sweep mode: Generate from sweep_config.benchmark parameters
        3. Fixed-benchmark mode: Use static_config.benchmark's pp_layer_config + requests
        
        Args:
            static_benchmark_cfg: Static benchmark config for shared parameters
                                  (num_total_requests, repetition) in parameter-sweep mode,
                                  or complete benchmark config in fixed-benchmark mode.
        """
        axes = {}
        
        # vLLM sweep parameters
        if self.vllm is not None:
            if self.vllm.attention_kernel is not None:
                axes['attention_kernel'] = self.vllm.attention_kernel
            if self.vllm.weight_chunk_size_mb is not None:
                axes['weight_chunk_size_mb'] = self.vllm.weight_chunk_size_mb
            if self.vllm.migration_approach is not None:
                axes['migration_approach'] = self.vllm.migration_approach
            if self.vllm.weight_loading_mode is not None:
                axes['weight_loading_mode'] = self.vllm.weight_loading_mode
            if self.vllm.fixed_num_gpu_blocks is not None:
                axes['fixed_num_gpu_blocks'] = self.vllm.fixed_num_gpu_blocks
            if self.vllm.block_size is not None:
                axes['block_size'] = self.vllm.block_size
            if self.vllm.use_vmm is not None:
                axes['use_vmm'] = self.vllm.use_vmm
            if self.vllm.enable_kv_resize is not None:
                axes['enable_kv_resize'] = self.vllm.enable_kv_resize
            if self.vllm.enable_cpu_weight_cache is not None:
                axes['enable_cpu_weight_cache'] = self.vllm.enable_cpu_weight_cache
        
        # Mode 1: benchmark_config provided directly in sweep_config
        if self.benchmark_config is not None:
            axes['benchmark_config'] = self.benchmark_config
        # Mode 2: Generate benchmark_config from individual sweep parameters
        elif self.benchmark is not None:
            assert static_benchmark_cfg is not None
            num_requests = static_benchmark_cfg.num_total_requests
            repetition = static_benchmark_cfg.repetition
            axes['benchmark_config'] = self._generate_benchmark_configs_from_params(
                num_total_requests=num_requests,
                repetition=repetition
            )
        # Mode 3: Fixed-benchmark mode - use static_config.benchmark's pp_layer_config + requests
        elif static_benchmark_cfg is not None and static_benchmark_cfg.has_fixed_benchmark:
            # Create a single benchmark_config from static_config.benchmark
            axes['benchmark_config'] = [static_benchmark_cfg.to_sweep_benchmark_config()]
        
        return axes
    
    def _generate_benchmark_configs_from_params(
        self,
        num_total_requests: int = 100,
        repetition: int = 1
    ) -> List[SweepBenchmarkConfig]:
        """Generate SweepBenchmarkConfig list from individual benchmark parameters.
        
        Creates benchmark configs from either:
        1. Cartesian product of pp_layer_configs, request_rates, input_lens, output_lens
        2. Cartesian product of pp_layer_configs and staged requests definitions
        num_total_requests and repetition default to static_config.benchmark, but can be
        overridden in sweep_config.benchmark.
        
        Args:
            num_total_requests: Shared across all combinations (from static_config.benchmark)
            repetition: Shared across all combinations (from static_config.benchmark)
        """
        import itertools
        
        if self.benchmark is None:
            return []
        
        params = self.benchmark
        num_total_requests = params.num_total_requests or num_total_requests
        repetition = params.repetition or repetition
        
        # Validate required parameters - all must be explicitly provided
        missing_params = []
        if not params.pp_layer_configs:
            missing_params.append("pp_layer_configs")
        
        if missing_params:
            raise ValueError(
                f"Missing required sweep_config.benchmark parameters: {missing_params}. "
                "pp_layer_configs must be explicitly provided."
            )
        
        pp_configs: List[str] = params.pp_layer_configs  # type: ignore[assignment]
        configs = []

        if params.requests is not None:
            request_variants = self._normalize_segmented_request_variants(params.requests)

            for pp_config, request_cfg in itertools.product(pp_configs, request_variants):
                configs.append(
                    SweepBenchmarkConfig(
                        num_total_requests=num_total_requests,
                        repetition=repetition,
                        pp_layer_config={0: pp_config.replace(" ", "")},
                        requests=request_cfg,
                    )
                )

            return configs

        # At this point flat sweep params are guaranteed to be non-None by validator
        request_rates: List[float] = params.request_rates  # type: ignore[assignment]
        input_lens: List[int] = params.input_lens  # type: ignore[assignment]
        output_lens: List[int] = params.output_lens  # type: ignore[assignment]
        
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

    @staticmethod
    def _normalize_segmented_request_variants(
        requests: Dict[int, RequestStageConfig] | List[Dict[int, RequestStageConfig]]
    ) -> List[Dict[int, RequestStageConfig]]:
        """Normalize segmented request definitions into a list of variants."""
        if isinstance(requests, dict):
            return [dict(sorted(requests.items()))]
        return [dict(sorted(request_cfg.items())) for request_cfg in requests]


class StaticVllmCfg(BaseModel):
    """Static vLLM configuration (non-sweep parameters)."""
    pipeline_parallel_size: int = 2
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8000
    max_num_batched_tokens: Optional[int] = None  # Max tokens per batch for chunked prefill (default: 2048 when chunked_prefill=True)
    max_num_seqs: Optional[int] = None  # Max concurrent sequences; keep unset to use vLLM default.
    chunked_prefill: bool = True
    enable_cuda_graph: bool = False
    enable_nsight: bool = False
    attention_kernel: str = "direct"  # "flash", "flexi", or "direct"
    block_size: Optional[int] = None
    """Legacy token block size. Prefer VLLM_PAGE_ATTENTION_BLOCK_SIZE_* for new experiments."""
    head_addr: str = "head"
    port: int = 8000
    weight_chunk_size_mb: float = 10.0
    """Weight chunk size in MB for chunked weight loading during migration. Default 10 MB."""
    migration_approach: str = "async"
    """Migration approach: 'sync', 'async', or 'async_fast'. Default 'async'."""
    weight_loading_mode: str = "async"
    """Weight loading mode during migration: 'sync' or 'async'. Default 'async'."""
    fixed_num_gpu_blocks: int = -1
    """Fixed number of GPU KV cache blocks per layer. Default -1 (auto).
    When positive, uses this exact block count and prevents changes during migration."""
    use_vmm: bool = True
    """Use CUDA VMM for KV cache allocation. Default True.
    When False, uses cudaMallocAsync (has memory leak, for testing memory leak size)."""
    enable_kv_resize: bool = True
    """Whether to allow KV cache resize during migration. Default True.
    When False, disables KV cache resize during migration (compact/shrink/expand).
    Note: set_pp_config can still resize when changing PP config."""
    log_kv_memory_stats: bool = False
    """Whether to log detailed KV cache memory statistics. Default False.
    When True, calculates actual_kv_memory_bytes and allocated_kv_memory_bytes.
    May have slight performance impact on schedule()."""
    enable_cpu_weight_cache: bool = True
    """Whether to preload all weights into CPU memory at startup. Default True.
    When True, weights are preloaded to CPU pinned memory for faster GPU loading.
    When False, weights are loaded from disk on-demand during migration."""
    disable_memory_overhead_monitor: bool = False
    """Disable VMXpert memory overhead monitoring during migration."""
    pipeline_autoscaling_enabled: bool = False
    """Enable experimental TP=1/DP=1 pipeline autoscaling."""
    autoscaling_candidate_ranks: Optional[List[int]] = None
    """Candidate PP ranks that may become active during autoscaling."""
    autoscaling_sequence: Optional[List[Dict[str, Any]]] = None
    """Optional explicit multi-step autoscaling sequence."""
    autoscaling_policy: Optional[Dict[str, Any]] = None
    """Optional runtime autoscaling policy, e.g. KV pressure threshold."""


class StaticBenchCfg(BaseModel):
    """Static benchmark configuration (non-sweep parameters).
    
    Contains shared parameters that apply to all sweep combinations.
    These should NOT be redefined in sweep_config to avoid conflicts.
    
    Supports two modes:
    1. Parameter-sweep mode: Define only shared params (num_total_requests, repetition),
       and let sweep_config.benchmark define the sweep axes.
    2. Fixed-benchmark mode: Define pp_layer_config and requests here to fix the
       benchmark configuration. Then sweep only vLLM parameters like attention_kernel.
    
    Example (fixed-benchmark mode):
        static_config:
          benchmark:
            num_total_requests: 100
            repetition: 1
            pp_layer_config:
              0: "32, 32"
              50: "20, 44"  # Migration at request 50
            requests:
              0:
                request_rate: 2.0
                input_lens: 500
                output_lens: 32
        sweep_config:
          vllm:
            attention_kernel: [flash, flexi, direct]
    """
    pattern_batch_size: int = 150
    profile: bool = False
    print_outputs: bool = False
    save_result: bool = False
    save_detailed: bool = False
    ignore_eos: bool = False
    benchmark_script_path: str = "/root/vllm_workbench/vllm/benchmarks/benchmark_serving.py"
    burstiness: float = 100.0
    warmup: Optional[WarmupBenchCfg] = None
    # Dataset configuration
    dataset_name: str = "pattern"
    """Dataset type: 'pattern' (default), 'sharegpt', 'burstgpt', 'random', 'sonnet'."""
    dataset_path: Optional[str] = None
    """Path to dataset file. Required for: sharegpt (JSON), burstgpt (CSV), sonnet (TXT).
    Default paths: ShareGPT=/home/bxb1/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
                   BurstGPT=/home/bxb1/data/datasets/BurstGPT_without_fails_2.csv"""
    sharegpt_output_len: Optional[int] = None
    """Output length for ShareGPT dataset. If None, uses actual completion length."""
    
    # Shared parameters for parameter-sweep mode
    num_total_requests: int = 100
    """Number of total requests (shared across all sweep combinations)."""
    
    repetition: int = 1
    """Number of repetitions per experiment (shared across all sweep combinations)."""
    restart_server_between_repetitions: bool = False
    """Whether to restart the vLLM server between repetitions of the same experiment.
    When False, repetitions reuse the same server process and rely on set_pp_config reset."""
    
    # Optional: Fixed benchmark configuration (for fixed-benchmark mode)
    pp_layer_config: Optional[Dict[int, str]] = None
    """Pipeline layer partition configs indexed by request number.
    If defined, this becomes a fixed benchmark config and sweep_config.benchmark
    should not be used. Example: {0: "32,32", 50: "20,44"} means migration at request 50."""
    
    requests: Optional[Dict[int, RequestStageConfig]] = None
    """Request configurations indexed by request number.
    Required when pp_layer_config is defined. Example: {0: {request_rate: 2.0, ...}}."""
    
    @property
    def has_fixed_benchmark(self) -> bool:
        """Returns True if this config defines a fixed benchmark (pp_layer_config + requests)."""
        return self.pp_layer_config is not None and self.requests is not None
    
    def to_sweep_benchmark_config(self) -> 'SweepBenchmarkConfig':
        """Convert to SweepBenchmarkConfig when using fixed-benchmark mode."""
        if not self.has_fixed_benchmark:
            raise ValueError("Cannot convert to SweepBenchmarkConfig: pp_layer_config or requests not defined")
        return SweepBenchmarkConfig(
            num_total_requests=self.num_total_requests,
            repetition=self.repetition,
            pp_layer_config=self.pp_layer_config,  # type: ignore
            requests=self.requests,  # type: ignore
        )


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
    3. If using fixed-benchmark mode (static_config.benchmark has pp_layer_config + requests),
       sweep_config.benchmark and sweep_config.benchmark_config should not be used
    4. vLLM params: if defined in sweep_config, they become sweep axes;
       static_config provides fallback values
    """
    project: str
    type: str = "sweep_test"
    static_config: StaticConfig
    sweep_config: Optional[SweepConfig] = None
    sweep_configs: Optional[List[SweepConfig]] = None
    envs: Dict[str, str] = {}
    is_log_cover: bool = False
    overwrite: bool = True  # If False, skip experiments that have already succeeded
    
    @model_validator(mode='after')
    def validate_no_conflicts(self) -> 'SweepTestConfig':
        """Validate that there are no configuration conflicts between modes."""
        errors = []
        
        # Must provide either sweep_config or sweep_configs (not both)
        if self.sweep_config is not None and self.sweep_configs is not None:
            errors.append(
                "Configuration conflict: 'sweep_config' and 'sweep_configs' are mutually exclusive. "
                "Use 'sweep_config' for a single sweep or 'sweep_configs' for a list of sweeps."
            )
        if self.sweep_config is None and self.sweep_configs is None:
            errors.append(
                "Either 'sweep_config' or 'sweep_configs' must be provided."
            )
        
        # Validate each sweep config (singular or each item in the list)
        configs_to_validate = []
        if self.sweep_config is not None:
            configs_to_validate = [self.sweep_config]
        elif self.sweep_configs is not None:
            configs_to_validate = self.sweep_configs
        
        for idx, sc in enumerate(configs_to_validate):
            prefix = f"sweep_configs[{idx}]" if self.sweep_configs is not None else "sweep_config"
            
            # Check for mode conflicts: benchmark_config and benchmark should be mutually exclusive
            if (sc.benchmark_config is not None and sc.benchmark is not None):
                errors.append(
                    f"Configuration conflict in {prefix}: 'benchmark_config' and "
                    "'benchmark' are mutually exclusive. "
                    "Use benchmark_config for complete configs or benchmark for parameter sweep."
                )
            
            # Check for fixed-benchmark mode conflicts
            if self.static_config.benchmark.has_fixed_benchmark:
                if sc.benchmark is not None:
                    errors.append(
                        f"Configuration conflict in {prefix}: 'static_config.benchmark' has "
                        "pp_layer_config and requests defined, which enables fixed-benchmark mode. "
                        "'benchmark' should not be used in this mode. "
                        "Only sweep vLLM parameters like attention_kernel, weight_chunk_size_mb, or weight_loading_mode."
                    )
                if sc.benchmark_config is not None:
                    errors.append(
                        f"Configuration conflict in {prefix}: 'static_config.benchmark' has "
                        "pp_layer_config and requests defined, which enables fixed-benchmark mode. "
                        "'benchmark_config' should not be used in this mode."
                    )
        
        # Note: attention_kernel can appear in:
        # - static_config.vllm.attention_kernel (single value, fallback)
        # - sweep_config.vllm.attention_kernel (list for sweeping)
        # When sweep_config.vllm.attention_kernel is provided, it overrides static_config.
        # No conflict check needed - sweep takes precedence over static.
        
        # Note: gpu_memory_utilization and block_size can ONLY appear in static_config.vllm
        # They are not sweep parameters.
        
        if errors:
            raise ValueError("\n".join(errors))
        
        return self
    
    def get_sweep_axes(self) -> Dict[str, List[Any]]:
        """Convenience method to get sweep axes with static config applied.
        
        Returns the sweep axes for the first (or only) sweep config.
        For multi-sweep support, use get_all_sweep_axes() instead.
        """
        configs = self.get_resolved_sweep_configs()
        if not configs:
            return {}
        return configs[0].get_sweep_axes(
            static_benchmark_cfg=self.static_config.benchmark
        )
    
    def get_resolved_sweep_configs(self) -> List[SweepConfig]:
        """Return the list of SweepConfig objects to iterate over.
        
        Handles both sweep_config (singular) and sweep_configs (list) modes.
        """
        if self.sweep_configs is not None:
            return self.sweep_configs
        elif self.sweep_config is not None:
            return [self.sweep_config]
        return []
    
    def get_all_sweep_axes(self) -> List[Dict[str, List[Any]]]:
        """Get sweep axes for all sweep configs.
        
        Returns a list of sweep axes dicts, one per sweep_config.
        """
        return [
            sc.get_sweep_axes(static_benchmark_cfg=self.static_config.benchmark)
            for sc in self.get_resolved_sweep_configs()
        ]


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
    pipeline_stage_to_rank: Dict[int, int] = field(default_factory=dict)


@dataclass
class ExpVllmConfig:
    """vLLM server configuration for a single experiment.
    
    Merged from:
    - static_config.vllm (base values)
    - sweep_config.vllm (overrides like attention_kernel, weight_chunk_size_mb)
    """
    pipeline_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    max_num_batched_tokens: Optional[int]  # Max tokens per batch for chunked prefill
    max_num_seqs: Optional[int]
    block_size: Optional[int]
    page_attention_block_size_bytes: Optional[int]
    head_addr: str
    port: int
    attention_kernel: str  # "flash", "flexi", or "direct"
    chunked_prefill: bool
    enable_cuda_graph: bool
    enable_nsight: bool
    pp_layer_partition: str  # Initial partition, e.g. "32,32"
    
    # Fields with defaults must come after required fields
    weight_chunk_size_mb: float = 10.0
    """Weight chunk size in MB for chunked weight loading during migration."""
    migration_approach: str = "async"
    """Migration approach: 'sync', 'async', or 'async_fast'."""
    weight_loading_mode: str = "async"
    """Weight loading mode during migration: 'sync' or 'async'."""
    fixed_num_gpu_blocks: int = -1
    """Fixed number of GPU KV cache blocks per layer. Default -1 (auto)."""
    use_vmm: bool = True
    """Use CUDA VMM for KV cache allocation. Default True.
    When False, uses cudaMallocAsync (has memory leak, for testing)."""
    enable_kv_resize: bool = True
    """Whether to allow KV cache resize during migration. Default True.
    When False, disables resize during migration (compact/shrink/expand)."""
    log_kv_memory_stats: bool = False
    """Whether to log detailed KV cache memory statistics. Default False.
    When True, calculates actual_kv_memory_bytes and allocated_kv_memory_bytes."""
    disable_memory_overhead_monitor: bool = False
    """Disable memory overhead monitoring during migration. Default False.
    Set to True to avoid performance impact from GC and memory measurements."""
    enable_cpu_weight_cache: bool = True
    """Whether to preload all weights into CPU memory at startup. Default True.
    When True, weights are preloaded to CPU pinned memory for faster GPU loading.
    When False, weights are loaded from disk on-demand during migration."""
    pipeline_autoscaling_enabled: bool = False
    autoscaling_candidate_ranks: Optional[List[int]] = None
    autoscaling_sequence: Optional[List[Dict[str, Any]]] = None
    autoscaling_policy: Optional[Dict[str, Any]] = None
    pp_layer_config: Dict[int, str] = field(default_factory=dict)  # {0: "32,32", 100: "20,44"}
    
    @property
    def has_migration(self) -> bool:
        """Returns True if there are multiple pp_layer_config entries."""
        return len(self.pp_layer_config) > 1
    
    @property
    def target_pp_partition(self) -> Optional[str]:
        """Returns the target pp_layer_partition for migration (last non-initial config).
        
        Returns None if there's no migration (only initial config exists).
        """
        if not self.has_migration:
            return None
        sorted_keys = sorted(self.pp_layer_config.keys())
        # Return the last config (final target), skip the initial (index 0)
        return self.pp_layer_config[sorted_keys[-1]].replace(" ", "")
    
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
    
    def get_initial_pp_config(self) -> List[List[int]]:
        """Get initial PP config in [[start, end], ...] format.
        
        Used for resetting pipeline configuration between benchmark repetitions.
        """
        pp_str = self.pp_layer_partition.replace(" ", "")
        layers = [int(x.strip()) for x in pp_str.split(",")]
        ranges = []
        start = 0
        for num_layers in layers:
            end = start + num_layers - 1
            ranges.append([start, end])
            start = end + 1
        return ranges


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
    arrival_trace: Optional[Dict[str, Any]] = None
    
    # From static_config.benchmark
    pattern_batch_size: int = 150
    burstiness: float = 100.0
    restart_server_between_repetitions: bool = False
    print_outputs: bool = False
    profile: bool = False
    save_result: bool = False
    save_detailed: bool = False
    ignore_eos: bool = False
    benchmark_script_path: str = ""
    warmup: Optional[WarmupBenchCfg] = None
    # Dataset configuration
    dataset_name: str = "pattern"
    """Dataset type: 'pattern' (default), 'sharegpt', 'burstgpt', 'random', 'sonnet'."""
    dataset_path: Optional[str] = None
    """Path to dataset file. Required for: sharegpt (JSON), burstgpt (CSV), sonnet (TXT)."""
    sharegpt_output_len: Optional[int] = None
    """Output length for ShareGPT dataset. If None, uses actual completion length."""


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
        "attention_kernel": "kernel",
        "pp_layer_partition": "pp",
        "request_rate": "rr",
        "has_migration": "mig",
        "input_lens": "in",
        "output_lens": "out",
        "num_total_requests": "n_req",
        "repetition": "rep",
        "weight_chunk_size_mb": "chunk",
        "migration_approach": "mig_mode",
        "weight_loading_mode": "wl",
        "block_size": "blk",
        "enable_cpu_weight_cache": "cpu_cache",
    }
    
    model: ExpModelConfig
    network: ExpNetworkConfig
    vllm: ExpVllmConfig
    benchmark: ExpBenchmarkConfig
    
    def get_naming_vars(self) -> Dict[str, Any]:
        """Generate vars_mapping for file naming using NAMING_ALIASES."""
        vars_dict = {
            "kernel": self.vllm.attention_kernel,
            "pp": self.vllm.pp_layer_partition,
            "rr": self.benchmark.request_rate,
            "mig": self.vllm.has_migration,
            "in": self.benchmark.input_lens,
            "out": self.benchmark.output_lens,
            "n_req": self.benchmark.num_total_requests,
            "rep": self.benchmark.repetition,
        }
        vars_dict["chunk"] = self.vllm.weight_chunk_size_mb
        vars_dict["mig_mode"] = self.vllm.migration_approach
        vars_dict["wl"] = self.vllm.weight_loading_mode
        if self.vllm.page_attention_block_size_bytes is not None:
            vars_dict["pab_kb"] = self.vllm.page_attention_block_size_bytes // 1024
        elif self.vllm.block_size is not None:
            vars_dict["blk"] = self.vllm.block_size
        if self.vllm.fixed_num_gpu_blocks != -1:
            vars_dict["fixed_blocks"] = self.vllm.fixed_num_gpu_blocks
        # Add vmm to naming when it's part of the sweep
        vars_dict["vmm"] = self.vllm.use_vmm
        # Add kv_resize to naming to distinguish experiments with/without resize
        vars_dict["kv_resize"] = self.vllm.enable_kv_resize
        # Add cpu_cache to naming to distinguish experiments with/without CPU weight preloading
        vars_dict["cpu_cache"] = self.vllm.enable_cpu_weight_cache
        # Add target_pp to naming when migration exists - distinguishes different migration targets
        if self.vllm.target_pp_partition is not None:
            vars_dict["tgt_pp"] = self.vllm.target_pp_partition
        return vars_dict
    
    @classmethod
    def iter_from_sweep_test_config(
        cls,
        sweep_test_cfg: 'SweepTestConfig',
    ) -> 'Iterator[tuple[int, ExperimentConfig]]':
        """Iterate over all ExperimentConfigs from Cartesian product of sweep axes.
        
        This is the main entry point for generating experiment configurations.
        It combines static_config with each sweep combination to produce
        ExperimentConfig instances.
        
        Returns tuples of (sweep_config_index, ExperimentConfig) to allow
        grouping experiments by their source sweep_config block for server
        restart isolation.
        
        Supports both sweep_config (singular) and sweep_configs (list) modes.
        When sweep_configs is used, experiments from each SweepConfig are
        yielded sequentially.
        
        Args:
            sweep_test_cfg: The complete sweep test configuration containing
                           static_config and sweep_config/sweep_configs
        
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
        
        # Iterate over all resolved sweep configs (handles both singular and list)
        # Each sweep_config block gets its own index for server restart isolation
        for sweep_config_index, sweep_config in enumerate(sweep_test_cfg.get_resolved_sweep_configs()):
            sweep_axes = sweep_config.get_sweep_axes(
                static_benchmark_cfg=static_cfg.benchmark
            )
            
            if not sweep_axes:
                continue
            
            # Build Cartesian product of all sweep axes
            # IMPORTANT: Ensure parameters requiring server restart are OUTERMOST loops
            # because switching these requires server restart.
            # Parameters requiring restart: attention_kernel, block_size
            # We reorder axes so that restart-requiring params come first.
            axis_names = list(sweep_axes.keys())
            
            # Reorder: put restart-requiring params first (outermost loops)
            # Order: block_size -> attention_kernel (block_size outermost)
            restart_params = ['block_size', 'attention_kernel']
            for param in restart_params:
                if param in axis_names:
                    axis_names.remove(param)
                    axis_names.insert(0, param)
            
            axis_values = [sweep_axes[name] for name in axis_names]
            
            for combo in itertools.product(*axis_values):
                # combo is a tuple of values, one per axis
                combo_dict = dict(zip(axis_names, combo))
                
                # Extract sweep parameters from combination
                bench_cfg: SweepBenchmarkConfig = combo_dict['benchmark_config']
                attn_kernel: Optional[str] = combo_dict.get('attention_kernel')
                weight_chunk_size: Optional[float] = combo_dict.get('weight_chunk_size_mb')
                mig_approach: Optional[str] = combo_dict.get('migration_approach')
                weight_loading_mode: Optional[str] = combo_dict.get('weight_loading_mode')
                fixed_blocks: Optional[int] = combo_dict.get('fixed_num_gpu_blocks')
                block_sz: Optional[int] = combo_dict.get('block_size')
                vmm_flag: Optional[bool] = combo_dict.get('use_vmm')
                kv_resize_flag: Optional[bool] = combo_dict.get('enable_kv_resize')
                cpu_cache_flag: Optional[bool] = combo_dict.get('enable_cpu_weight_cache')
                
                # Create ExperimentConfig from this combination
                # Yield (sweep_config_index, experiment_config) tuple for server restart isolation
                yield (sweep_config_index, cls._create_from_combo(static_cfg, bench_cfg, attn_kernel, weight_chunk_size, mig_approach, weight_loading_mode, fixed_blocks, block_sz, vmm_flag, kv_resize_flag, cpu_cache_flag))
    
    @classmethod
    def _create_from_combo(
        cls,
        static_cfg: 'StaticConfig',
        bench_cfg: SweepBenchmarkConfig,
        attention_kernel: Optional[str] = None,
        weight_chunk_size_mb: Optional[float] = None,
        migration_approach: Optional[str] = None,
        weight_loading_mode: Optional[str] = None,
        fixed_num_gpu_blocks: Optional[int] = None,
        block_size: Optional[int] = None,
        use_vmm: Optional[bool] = None,
        enable_kv_resize: Optional[bool] = None,
        enable_cpu_weight_cache: Optional[bool] = None,
    ) -> 'ExperimentConfig':
        """Create a single ExperimentConfig from static config and one sweep combination.
        
        Internal method used by iter_from_sweep_test_config.
        
        Args:
            static_cfg: Static configuration
            bench_cfg: Sweep benchmark configuration (from sweep axes)
            attention_kernel: Override from sweep axis (if sweeping), otherwise use static
            weight_chunk_size_mb: Override from sweep axis (if sweeping), otherwise use static
            migration_approach: Override from sweep axis (if sweeping), otherwise use static
            weight_loading_mode: Override from sweep axis (if sweeping), otherwise use static
            fixed_num_gpu_blocks: Override from sweep axis (if sweeping), otherwise use static
            block_size: Override from sweep axis (if sweeping), otherwise use static
            use_vmm: Override from sweep axis (if sweeping), otherwise use static
            enable_kv_resize: Override from sweep axis (if sweeping), otherwise use static
            enable_cpu_weight_cache: Override from sweep axis (if sweeping), otherwise use static
        """
        # Resolve sweep overrides: sweep values override static values
        kernel = attention_kernel if attention_kernel is not None else static_cfg.vllm.attention_kernel
        chunk_size = weight_chunk_size_mb if weight_chunk_size_mb is not None else static_cfg.vllm.weight_chunk_size_mb
        mig_approach = migration_approach if migration_approach is not None else static_cfg.vllm.migration_approach
        weight_loading = weight_loading_mode if weight_loading_mode is not None else static_cfg.vllm.weight_loading_mode
        fixed_blocks = fixed_num_gpu_blocks if fixed_num_gpu_blocks is not None else static_cfg.vllm.fixed_num_gpu_blocks
        raw_blk_size = block_size if block_size is not None else static_cfg.vllm.block_size
        blk_size, page_attention_block_size_bytes = (
            derive_token_block_size_from_page_attention_block(
                model_path=static_cfg.model.path,
                explicit_block_size=raw_blk_size,
            ))
        vmm = use_vmm if use_vmm is not None else static_cfg.vllm.use_vmm
        kv_resize = enable_kv_resize if enable_kv_resize is not None else static_cfg.vllm.enable_kv_resize
        cpu_cache = enable_cpu_weight_cache if enable_cpu_weight_cache is not None else static_cfg.vllm.enable_cpu_weight_cache
        
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
        
        rank_to_ip, rank_to_node, pipeline_stage_to_rank = (
            static_cfg.network.resolve_worker_placement(
                static_cfg.vllm.pipeline_parallel_size
            )
        )

        network_cfg = ExpNetworkConfig(
            rank_to_ip=rank_to_ip,
            rank_to_node=rank_to_node,
            pipeline_stage_to_rank=pipeline_stage_to_rank,
        )
        
        vllm_cfg = ExpVllmConfig(
            pipeline_parallel_size=static_cfg.vllm.pipeline_parallel_size,
            gpu_memory_utilization=static_cfg.vllm.gpu_memory_utilization,
            max_model_len=static_cfg.vllm.max_model_len,
            max_num_batched_tokens=static_cfg.vllm.max_num_batched_tokens,
            max_num_seqs=static_cfg.vllm.max_num_seqs,
            block_size=blk_size,
            page_attention_block_size_bytes=page_attention_block_size_bytes,
            head_addr=static_cfg.vllm.head_addr,
            port=static_cfg.vllm.port,
            attention_kernel=kernel,
            chunked_prefill=static_cfg.vllm.chunked_prefill,
            enable_cuda_graph=static_cfg.vllm.enable_cuda_graph,
            enable_nsight=static_cfg.vllm.enable_nsight,
            weight_chunk_size_mb=chunk_size,
            migration_approach=mig_approach,
            weight_loading_mode=weight_loading,
            fixed_num_gpu_blocks=fixed_blocks,
            use_vmm=vmm,
            enable_kv_resize=kv_resize,
            enable_cpu_weight_cache=cpu_cache,
            log_kv_memory_stats=static_cfg.vllm.log_kv_memory_stats,
            disable_memory_overhead_monitor=static_cfg.vllm.disable_memory_overhead_monitor,
            pipeline_autoscaling_enabled=static_cfg.vllm.pipeline_autoscaling_enabled,
            autoscaling_candidate_ranks=static_cfg.vllm.autoscaling_candidate_ranks,
            autoscaling_sequence=static_cfg.vllm.autoscaling_sequence,
            autoscaling_policy=static_cfg.vllm.autoscaling_policy,
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
            arrival_trace=bench_cfg.arrival_trace,
            pattern_batch_size=static_cfg.benchmark.pattern_batch_size,
            burstiness=static_cfg.benchmark.burstiness,
            restart_server_between_repetitions=static_cfg.benchmark.restart_server_between_repetitions,
            print_outputs=static_cfg.benchmark.print_outputs,
            profile=static_cfg.benchmark.profile,
            save_result=static_cfg.benchmark.save_result,
            save_detailed=static_cfg.benchmark.save_detailed,
            ignore_eos=static_cfg.benchmark.ignore_eos,
            benchmark_script_path=static_cfg.benchmark.benchmark_script_path,
            warmup=static_cfg.benchmark.warmup,
            dataset_name=static_cfg.benchmark.dataset_name,
            dataset_path=static_cfg.benchmark.dataset_path,
            sharegpt_output_len=static_cfg.benchmark.sharegpt_output_len,
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
    max_num_batched_tokens: Optional[int]
    max_num_seqs: Optional[int]
    block_size: Optional[int]
    page_attention_block_size_bytes: Optional[int]
    head_addr: str
    port: int
    
    # Features
    attention_kernel: str = "direct"  # "flash", "flexi", or "direct"
    chunked_prefill: bool = False
    enable_cuda_graph: bool = False
    enable_nsight: bool = False
    weight_chunk_size_mb: float = 10.0
    """Weight chunk size in MB for chunked weight loading during migration."""
    migration_approach: str = "async"
    """Migration approach: 'sync', 'async', or 'async_fast'."""
    weight_loading_mode: str = "async"
    """Weight loading mode during migration: 'sync' or 'async'."""
    fixed_num_gpu_blocks: int = -1
    """Fixed number of GPU KV cache blocks per layer. Default -1 (auto)."""
    use_vmm: bool = True
    """Use CUDA VMM for KV cache allocation. Default True.
    When False, uses cudaMallocAsync (has memory leak, for testing)."""
    enable_kv_resize: bool = True
    """Whether to allow KV cache resize during migration. Default True.
    When False, disables resize during migration (compact/shrink/expand)."""
    log_kv_memory_stats: bool = False
    """Whether to log detailed KV cache memory statistics. Default False.
    When True, calculates actual_kv_memory_bytes and allocated_kv_memory_bytes."""
    enable_cpu_weight_cache: bool = True
    """Whether to preload all weights into CPU memory at startup. Default True.
    When True, weights are preloaded to CPU pinned memory for faster GPU loading.
    When False, weights are loaded from disk on-demand during migration."""
    pipeline_autoscaling_enabled: bool = False
    autoscaling_candidate_ranks: Optional[List[int]] = None
    autoscaling_sequence: Optional[List[Dict[str, Any]]] = None
    autoscaling_policy: Optional[Dict[str, Any]] = None
    
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
    pipeline_stage_to_rank: Dict[int, int] = field(default_factory=dict)
    
    # Benchmark-related (for dynamic config)
    pattern_batch_size: int = 150
    
    # Paths (set during spec generation)
    metrics_csv_path: Optional[str] = None
    server_raw_log_path: Optional[str] = None
    
    def to_dynamic_cfg(self) -> Dict[str, Any]:
        """Generate dynamic config dict for -D flag."""
        cfg = {
            "attention_kernel": self.attention_kernel,
            "pp_layer_partition": self.pp_layer_partition,
            "pattern_batch_size": self.pattern_batch_size,
            "metrics_csv_path": self.metrics_csv_path,
            "alternative_configs": self.alternative_configs,
            "migration_steps": self.migration_steps,
            "migration_mode": self.migration_approach,
            "weight_loading_mode": self.weight_loading_mode,
            "fixed_num_gpu_blocks": self.fixed_num_gpu_blocks,
            "use_vmm": self.use_vmm,
            "enable_kv_resize": self.enable_kv_resize,
            "log_kv_memory_stats": self.log_kv_memory_stats,
            "enable_cpu_weight_cache": self.enable_cpu_weight_cache,
            "pipeline_autoscaling_enabled": self.pipeline_autoscaling_enabled,
        }
        if self.autoscaling_candidate_ranks is not None:
            cfg["autoscaling_candidate_ranks"] = self.autoscaling_candidate_ranks
        if self.autoscaling_sequence is not None:
            cfg["autoscaling_sequence"] = self.autoscaling_sequence
        if self.autoscaling_policy is not None:
            cfg["autoscaling_policy"] = self.autoscaling_policy
        # Only include rank_to_ip if non-empty
        if self.rank_to_ip:
            cfg["rank_to_ip"] = {str(k): v for k, v in self.rank_to_ip.items()}
        if self.pipeline_stage_to_rank:
            cfg["pipeline_stage_to_rank"] = {
                str(k): v for k, v in self.pipeline_stage_to_rank.items()
            }
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
    arrival_trace: Optional[Dict[str, Any]] = None
    pattern_batch_size: int = 150
    burstiness: float = 100.0
    
    # Repetition config
    repetition: int = 1
    """Number of times to repeat the benchmark. After each repetition,
    set_pp_config is called to return to the initial pp configuration."""
    restart_server_between_repetitions: bool = False
    """Whether to restart the vLLM server between repetitions instead of reusing one server."""
    
    # Pipeline config for repetition reset
    initial_pp_config: Optional[List[List[int]]] = None
    """Initial PP layer config to restore between repetitions.
    Format: [[start, end], [start, end], ...] per rank."""
    alternative_configs: Optional[Dict[str, Any]] = None
    """Migration target configs. Format: {"pp_layer_configs": {"0": [...], "1": [...]}}"""
    migration_steps: Optional[List[int]] = None
    """Request indices at which migration is triggered."""
    migration_mode: Optional[str] = None
    """Migration mode: 'sync', 'async', or 'async_fast'. Passed to set_pp_config between repetitions."""
    weight_loading_mode: Optional[str] = None
    """Weight loading mode: 'sync' or 'async'. Passed to set_pp_config between repetitions."""
    
    # Features
    print_outputs: bool = False
    profile: bool = False
    save_result: bool = False
    save_detailed: bool = False
    ignore_eos: bool = False
    
    # Optional warmup
    warmup: Optional[WarmupBenchCfg] = None
    
    # Dataset configuration
    dataset_name: str = "pattern"
    """Dataset type: 'pattern' (default), 'sharegpt', 'burstgpt', 'random', 'sonnet'."""
    dataset_path: Optional[str] = None
    """Path to dataset file. Required for: sharegpt (JSON), burstgpt (CSV), sonnet (TXT)."""
    sharegpt_output_len: Optional[int] = None
    """Output length for ShareGPT dataset. If None, uses actual completion length."""

    def build_benchmark_config(
        self,
        repetition_override: Optional[int] = None,
        metrics_file_path_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble benchmark config payload to be written to disk."""
        repetition = self.repetition if repetition_override is None else repetition_override
        metrics_file_path = (
            self.metrics_file_path
            if metrics_file_path_override is None else metrics_file_path_override
        )

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
            "arrival_trace": self.arrival_trace,
            "pattern_batch_size": self.pattern_batch_size,
            "profile": self.profile,
            "print_outputs": self.print_outputs,
            "save_result": self.save_result,
            "save_detailed": self.save_detailed,
            "ignore_eos": self.ignore_eos,
            "burstiness": self.burstiness,
            "warmup": self.warmup.model_dump() if self.warmup else None,
            "metrics_file_name": metrics_file_path,
            "repetition": repetition,
            # Pipeline config for repetition reset
            "initial_pp_config": self.initial_pp_config,
            "alternative_configs": self.alternative_configs,
            "migration_steps": self.migration_steps,
            "migration_mode": self.migration_mode,
            "weight_loading_mode": self.weight_loading_mode,
        }

    def write_benchmark_config(
        self,
        benchmark_config_path: Optional[str] = None,
        repetition_override: Optional[int] = None,
        metrics_file_path_override: Optional[str] = None,
    ) -> None:
        """Persist benchmark config to benchmark_config_path."""
        config_file_path = Path(
            self.benchmark_config_path if benchmark_config_path is None else benchmark_config_path
        )
        config_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file_path, "w", encoding="utf-8") as f:
            json.dump(
                self.build_benchmark_config(
                    repetition_override=repetition_override,
                    metrics_file_path_override=metrics_file_path_override,
                ),
                f,
                indent=2,
                ensure_ascii=False,
            )

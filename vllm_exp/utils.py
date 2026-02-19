#!/usr/bin/env python3
from __future__ import annotations
import atexit
import datetime
import itertools
import json
import os
import signal
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Iterable, Callable
from io import FileIO
import requests
import typer
import yaml
from pydantic import BaseModel
from rich.console import Console
from collections import OrderedDict

from .data import Config, MultiConfig

app = typer.Typer(no_args_is_help=True)
C = Console()

# =========================
# Config models
# =========================

@dataclass
class ServerRunSpec():
    start_pp_layer_partition: str
    is_migration: bool

@dataclass
class BenchmarkRunSpec():
    request_rate: float
    input_output_len: list[list[int]]

@dataclass
class TcRunSpec():
    delay: float


def load_config(path: str) -> MultiConfig:
    with open(path, "r") as f:
        return MultiConfig.model_validate(yaml.safe_load(f))

def wait_ready(base_url: str, timeout_s: int = 180) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=3)
            if r.ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False

def start_vllm(cfg: Config,  spec: ServerRunSpec, log_dir: Path, log_file_name: Optional[str] = None) -> subprocess.Popen:
    extra_env = {}
    # 配置vllm所需的环境变量
    extra_env["VLLM_PP_LAYER_PARTITION"] = spec.start_pp_layer_partition
    if cfg.migration.is_migration:
        extra_env["TEST_MIGRATION"] = "1"
    else:
        extra_env["TEST_MIGRATION"] = "0"

    env = os.environ.copy()
    env.update(extra_env)

    # 启动服务
    serve_args = [
        "vllm", "serve", cfg.model.path,
        "--pipeline-parallel-size", str(cfg.vllm.pipeline_parallel_size),
        "--gpu-memory-utilization", str(cfg.vllm.gpu_memory_utilization),
        "--max-model-len", str(cfg.vllm.max_model_len),
        "--served-model-name", cfg.model.name,
        "--distributed-executor-backend", "ray",
        "--disable-log-requests",
        "--no-enable-prefix-caching",
        "--scheduler-cls", "vllm.v1.core.sched.dynamic_scheduler.DynamicScheduler",
        "--worker-cls", "vllm.v1.worker.dynamic_gpu_worker.DynamicGPUWorker",
    ]
    
    # Build dynamic config - pass all parameters through -D flag (no file needed)
    alternative_configs_dict = {"pp_layer_configs": cfg.migration.alternative_configs}
    dynamic_cfg = json.dumps({
        "attention_kernel": cfg.vllm.attention_kernel,
        "pp_layer_partition": spec.start_pp_layer_partition,
        "alternative_configs": alternative_configs_dict,
        "migration_steps": cfg.migration.migration_steps,
        "migration_mode": cfg.migration.migration_mode,
        "allow_resize": cfg.migration.allow_resize,
    })
    serve_args.extend(["-D", dynamic_cfg])
    
    if cfg.vllm.chunked_prefill:
        serve_args.append("--enable-chunked-prefill")
    if not cfg.vllm.enable_cuda_graph:
        serve_args.append("--enforce-eager")
    if cfg.vllm.enable_nsight:
        serve_args.append("--ray-workers-use-nsight")

    if log_file_name is None or log_file_name == "":
        log_file_name = "server.log"
    else:
        log_file_name = f"server-{log_file_name}.log"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / log_file_name
    log_fd = open(log_file_path, "wb", buffering=0)
    
    proc = subprocess.Popen(
        serve_args,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid
    )
    return proc

def start_benchmark(cfg: Config, spec: BenchmarkRunSpec, log_dir: Path, log_file_name: Optional[str] = None) -> bool:
    base_url = f"http://head:{cfg.vllm.port}"
    ok = False

    if not wait_ready(base_url, 180):
        C.print("[red]ERROR[/] vLLM not ready in time")
        return False

    bench_args = [
        "python3", "/root/vllm_workbench/vllm/benchmarks/benchmark_serving.py",
        "--num-prompts", str(cfg.benchmark.num_requests),
        "--request-rate", str(spec.request_rate),
        "--backend", "openai-chat",
        "--model", cfg.model.path,
        "--endpoint", "/v1/chat/completions",
        "--base-url", base_url,
        "--dataset-name", "pattern",
        "--served-model-name", cfg.model.name,
        "--goodput", "tpot:300", "ttft:5000",
        "--temperature", "0",
        "--seed", "42",
        "--pattern-batch-size", str(cfg.benchmark.pattern_batch_size),
    ]
    if cfg.benchmark.profile:
        bench_args.append("--profile")
    if log_file_name is None or log_file_name == "":
        log_file_name = "benchmark.log"
    else:
        log_file_name = f"benchmark-{log_file_name}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    bench_fd = open(log_dir / log_file_name, "wb", buffering=0)
    # Dump workload config to configuration files
    config_file_path = cfg.envs["BENCHMARK_CONFIG_PATH"]
    Path(config_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_file_path, "w") as f:
        json.dump({
            "input_output_lens": cfg.benchmark.input_output_lens
            }, f)
    ret = subprocess.run(
        bench_args,
        stdout=bench_fd,
        stderr=subprocess.STDOUT,
        env={**os.environ}
    )
    ok = (ret.returncode == 0)
    return ok


# def start_benchmark(cfg: Config, log_fd, spec: BenchmarkRunSpec, extra_env: Dict[str,str]) -> subprocess.Popen:


def stop_tree(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

def start_tc(spec: TcRunSpec):
    # 配置tc所需的DELAY
    env = os.environ.copy()
    env["DELAY"] = str(spec.delay)
    os.environ.update(env)

    subprocess.run(["bash", "/root/vllm_workbench/tc.sh"], check=True)

def clean_metrics_directory(file_dir: Path):
    if file_dir and os.path.exists(file_dir):
        print(f"Cleaning directory {file_dir} before first write")
        for file_in_dir in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_in_dir)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Deleted directory: {file_path}")
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")

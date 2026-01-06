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
from dataclasses import dataclass, asdict, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Iterable, Callable
from io import FileIO
import requests
from sklearn import metrics
import typer
import yaml
from pydantic import BaseModel
from rich.console import Console
from collections import OrderedDict

from .data import Config, PathPolicy   
from .log import LogManager

app = typer.Typer(no_args_is_help=True)
C = Console()


def generate_analysis_report(logm: LogManager, vars: Optional[dict[str, Any]] = None):
    """
    Automatically generate analysis files in the log directory after experiment completes.
    Uses logm.get_path_with_log_type to construct filenames consistently.
    """
    try:
        log_dir = logm.get_dir()
        
        # Find all server log files
        server_logs = list(log_dir.glob("server-*.log"))
        
        if not server_logs:
            C.print(f"[yellow]Warning: No server logs found in {log_dir}[/]")
            return
        
        C.print(f"[bold cyan]Generating analysis reports in {log_dir}[/]")
        
        # If vars is provided, only analyze the corresponding server log
        if vars is not None:
            server_log_path = logm.get_path_with_log_type("server", "log", vars)
            if not server_log_path.exists():
                C.print(f"[yellow]Warning: Server log not found: {server_log_path}[/]")
                return
            server_logs = [server_log_path]
        
        # Process each server log
        for log_file in sorted(server_logs):
            # Use logm.get_path_with_log_type to construct analysis filename
            # This ensures consistency with server log naming
            analysis_path = logm.get_path_with_log_type("analysis", "txt", vars)
            
            C.print(f"  Analyzing {log_file.name} -> {analysis_path.name}")
            
            # Run analysis on this server log
            with open(analysis_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("VLLM EXPERIMENT ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Source Log: {log_file.name}\n")
                f.write(f"Log Directory: {log_dir}\n")
                f.write(f"Generated at: {datetime.datetime.now().isoformat()}\n\n")
                
                # Run the log analyzer
                analyzer_script = Path(__file__).parent / "log_analysing_tools" / "analyze_log_metrics.py"
                
                if analyzer_script.exists():
                    import sys
                    from io import StringIO
                    
                    # Capture stdout
                    old_stdout = sys.stdout
                    sys.stdout = captured_output = StringIO()
                    
                    try:
                        # Import and run analyzer
                        sys.path.insert(0, str(analyzer_script.parent))
                        from analyze_log_metrics import LogMetricsAnalyzer
                        
                        analyzer = LogMetricsAnalyzer(str(log_file))
                        analyzer.analyze_file()
                        analyzer.generate_report(top_n=50)
                        
                        # Get captured output
                        output = captured_output.getvalue()
                        f.write(output)
                        
                    except Exception as e:
                        f.write(f"Error analyzing {log_file.name}: {e}\n")
                        import traceback
                        f.write(traceback.format_exc())
                    finally:
                        sys.stdout = old_stdout
                else:
                    f.write(f"Warning: Analyzer script not found at {analyzer_script}\n")
            
            C.print(f"  [green]✓[/] Generated: {analysis_path.name}")
        
        C.print(f"[bold green]✓ All analysis reports generated successfully[/]")
        
    except Exception as e:
        C.print(f"[red]Error generating analysis report: {e}[/]")
        import traceback
        traceback.print_exc()

# =========================
# Config models
# =========================

@dataclass
class ServerRunSpec():
    start_pp_layer_partition: str
    is_migration: bool

@dataclass
class BenchmarkRunSpec():
    input_output_len: list[list[int]]
    request_rate: float = 0
    running_request_rate_list: list[float] = field(default_factory=list)

@dataclass
class TcRunSpec():
    delay: float

class MultiConfig(BaseModel):
    projects: List[Config]
## Utils 
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


def get_path_policy_from_var_keys(var_keys: list[str]) -> PathPolicy:
    return PathPolicy(variables=var_keys)

## Functions to start vllm and benchmark ##
def start_vllm(cfg: Config,  spec: ServerRunSpec, logm: LogManager, vars: Optional[dict[str, Any]] = None) -> subprocess.Popen:
    extra_env = {}
    # 配置vllm所需的环境变量
    extra_env["VLLM_PP_LAYER_PARTITION"] = spec.start_pp_layer_partition
    if cfg.migration.is_migration:
        extra_env["TEST_MIGRATION"] = "1"
        extra_env[f"PATTERN_BATCH_SIZE"] = str(cfg.benchmark.pattern_batch_size)
    else:
        extra_env["TEST_MIGRATION"] = "0"
    
    if cfg.migration.is_compact_kv:
        extra_env["TEST_KV_COMPACT"] = "1"
    else:
        extra_env["TEST_KV_COMPACT"] = "0"

    env = os.environ.copy()
    # 传递 LayerKV 双向通道所需的 rank->ip 映射（JSON 字符串）
    if cfg.network.rank_to_ip:
        import json as _json
        env["VLLM_LAYERKV_RANK_TO_IP"] = _json.dumps(cfg.network.rank_to_ip)
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
    # Add dynamic config for flexi flash attention
    dynamic_cfg = json.dumps({
        "enable_flexi_flash_attn": cfg.vllm.enable_flexi_flash_attn,
        "tester_start_step": cfg.migration.tester_start_step,
        "memory_stress_tester": cfg.migration.memory_stress_tester,
        })
    serve_args.extend(["-D", dynamic_cfg])
    
    if cfg.vllm.chunked_prefill:
        serve_args.append("--enable-chunked-prefill")
    if not cfg.vllm.enable_cuda_graph:
        serve_args.append("--enforce-eager")
    if cfg.vllm.enable_nsight:
        serve_args.append("--ray-workers-use-nsight")

    metrics_path = logm.get_path_with_log_type("timestamp_metrics", "csv", vars)
    if metrics_path.exists():
        os.remove(metrics_path)
    env["VLLM_METRICS_CSV_PATH"] = str(metrics_path)

    path = logm.get_path_with_log_type("server", "log", vars)
    # 如果file存在的话
    if path.exists():
        os.remove(path)
    log_fd = open(path, "wb", buffering=0)

    # Dump vllm dynamic deployment config to configuration files (always write if path is provided)
    deployment_config_path = os.environ.get("DEPLOYMENT_CONFIG_PATH")
    C.print(deployment_config_path)
    if deployment_config_path is not None:
        config_file_path = Path(deployment_config_path)
        config_file_path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = {
            "alternative_configs": {"pp_layer_configs": cfg.migration.alternative_configs},
            "migration_steps": cfg.migration.migration_steps,
            "compact_steps": cfg.migration.compact_steps,
        }
        with open(config_file_path, "w") as f:
            json.dump(config_dict, f)
    proc = subprocess.Popen(
        serve_args,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid
    )
    return proc

def start_benchmark(cfg: Config, spec: BenchmarkRunSpec, logm: LogManager, vars: Optional[dict[str, Any]] = None) -> bool:
    # Use localhost instead of 'head' for local development
    base_url = f"http://{cfg.vllm.head_addr}:{cfg.vllm.port}"
    ok = False
    if not wait_ready(base_url, 300):
        C.print("[red]ERROR[/] vLLM not ready in time")
        return False

    benchmark_config_path = os.environ.get("BENCHMARK_CONFIG_PATH")
    assert benchmark_config_path is not None
    bench_args = [
        "python3", cfg.benchmark.benchmark_script_path,
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
        "--benchmark-config", str(benchmark_config_path),
    ]
    C.print(f"start to run benchmark with args: {bench_args}")
    if cfg.benchmark.print_outputs:
        bench_args.append("--print-outputs")

    if cfg.benchmark.profile:
        bench_args.append("--profile")

    # Dump workload config to configuration files
    config_file_path = Path(benchmark_config_path)
    config_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file_path, "w", encoding="utf-8") as f:
        json.dump(cfg.benchmark.model_dump(), f, indent=2, ensure_ascii=False)
    
    # 规定metrics文件的命名方式
    metrics_file_name = logm.get_path_with_log_type("request_metrics", "csv", vars)
    if metrics_file_name.exists():
        os.remove(metrics_file_name)
    envs = os.environ.copy()
    envs.update({"METRICS_FILE_NAME": str(metrics_file_name)})

    log_path = logm.get_path_with_log_type("benchmark", "log", vars)
    if log_path.exists():
        os.remove(log_path)
    bench_fd = open(log_path, "wb", buffering=0)
    C.print(f"[bold cyan] log_path: {log_path}")
    ret = subprocess.run(
        bench_args,
        stdout=bench_fd,
        stderr=subprocess.STDOUT,
        env=envs
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


def partition_and_request_rate(cfg: Config, logm: LogManager):
    ok = False
    proc = None
    cfg.path_policy = get_path_policy_from_var_keys(["start_pp_layer_partition", "request_rate"]) 
    try:
        # 在这里，我们先sweep属于server的变量维度
        server_grid = itertools.product(
            cfg.network.delays,
            cfg.vllm.start_pp_layer_partitions,
        )
        for delay, part in server_grid:
            # Start tc
            tc_spec = TcRunSpec(delay)
            start_tc(tc_spec)

            # Start vllm
            spec = ServerRunSpec(part, cfg.migration.is_migration)
            vllm_vars_mapping = {"start_pp_layer_partition": spec.start_pp_layer_partition}
            logm.extend_path_with_vars(vllm_vars_mapping, move_to_end=True)
            proc = start_vllm(cfg, spec, logm)

            # Start Looping different configuration of benchmark
            for request_rate in cfg.benchmark.sweep_request_rates:
                bench_spec = BenchmarkRunSpec(cfg.benchmark.input_output_lens, request_rate)
                benchmark_vars_mapping = {"request_rate": bench_spec.request_rate}
                ok = start_benchmark(cfg, bench_spec, logm, benchmark_vars_mapping)
                logm.pop_vars_from_path(["request_rate"])
                if not ok:
                    break

            stop_tree(proc)
            logm.pop_vars_from_path(["start_pp_layer_partition"])

    except Exception as e:
        C.print(f"[red]ERROR[/] {e}")
        ok = False
    finally:
        if proc:
            stop_tree(proc)
        time.sleep(3)
        C.print(f"[bold cyan] Running is finished")
        # Generate analysis report for all server logs
        generate_analysis_report(logm, None)

def one_off_test(cfg: Config, logm: LogManager):
    cfg.path_policy = get_path_policy_from_var_keys([])
    ok = False
    proc = None
    # benchmark每发送num_requests次请求，都对应一个request_rate_list_compact中的request_rate
    assert len(cfg.vllm.start_pp_layer_partitions) == 1 
    vars_mapping = None
    try:
        # Start vllm
        spec = ServerRunSpec(cfg.vllm.start_pp_layer_partitions[0], cfg.migration.is_migration)
        vars_mapping = {"start_pp_layer_partition": spec.start_pp_layer_partition,
                        "enable_flexi_flash_attn": cfg.vllm.enable_flexi_flash_attn}
        logm.write_constants_meta(vars_mapping)
        proc = start_vllm(cfg=cfg, spec=spec, logm=logm, vars=vars_mapping)
        bench_spec = BenchmarkRunSpec(input_output_len=cfg.benchmark.input_output_lens, running_request_rate_list=cfg.benchmark.running_request_rates)

        # 使用同样的方式对benchmark的文件命名
        ok = start_benchmark(cfg=cfg, spec=bench_spec, logm=logm, vars=vars_mapping)
        if not ok:
            stop_tree(proc)
            raise Exception("Benchmark failed")
        stop_tree(proc)
    except Exception as e:
        C.print(f"[red]ERROR[/] {e}")
        ok = False
    finally:
        if proc:
            stop_tree(proc)
        time.sleep(3)
        C.print(f"[bold cyan] Running is finished")
        # Generate analysis report
        generate_analysis_report(logm, vars_mapping)

def test_migration_with_different_pp(cfg: Config, logm: LogManager):
    cfg.path_policy = get_path_policy_from_var_keys(["start_pp_layer_partition", "is_migration"])
    logm.extend_base_dir_with_vars({"is_migration": cfg.migration.is_migration})
    logm.write_constants_meta()
    ok = False
    proc = None
    # 我们暂时只使用一个request rate进行测试
    assert len(cfg.benchmark.sweep_request_rates) == 1 
    try:
        # 在这里，我们先sweep属于server的变量维度
        server_grid = itertools.product(
            cfg.network.delays,
            cfg.vllm.start_pp_layer_partitions,
        )
        for delay, part in server_grid:
            # Start tc
            tc_spec = TcRunSpec(delay)
            start_tc(tc_spec)

            # Start vllm
            spec = ServerRunSpec(part, cfg.migration.is_migration)
            vars_mapping = {"start_pp_layer_partition": spec.start_pp_layer_partition}
            proc = start_vllm(cfg=cfg, spec=spec, logm=logm, vars=vars_mapping)
            request_rate = cfg.benchmark.sweep_request_rates[0] 
            bench_spec = BenchmarkRunSpec(cfg.benchmark.input_output_lens, request_rate=request_rate)

            # 使用同样的方式对benchmark的文件命名
            ok = start_benchmark(cfg=cfg, spec=bench_spec, logm=logm, vars=vars_mapping)
            if not ok:
                break
            stop_tree(proc)
    except Exception as e:
        C.print(f"[red]ERROR[/] {e}")
        ok = False
    finally:
        if proc:
            stop_tree(proc)
        time.sleep(3)
        C.print(f"[bold cyan] Running is finished")
        # Generate analysis report for all server logs
        generate_analysis_report(logm, None)

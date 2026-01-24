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
import threading
import csv
from dataclasses import dataclass, asdict, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Iterable, Callable, Union
from io import FileIO
import requests
from sklearn import metrics
import typer
import yaml
from pydantic import BaseModel
from rich.console import Console
from collections import OrderedDict

from .data import Config, PathPolicy   
from .log import LogManager, SweepLogManager

app = typer.Typer(no_args_is_help=True)
C = Console()


def generate_analysis_report(logm: Union[LogManager, SweepLogManager], vars: Optional[dict[str, Any]] = None):
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


_PHASE_MARKER_TOKEN = "=== VLLM_EXP_PHASE_BOUNDARY ==="
_BENCH_PHASE_BOUNDARY_TOKEN = "=== VLLM_EXP_BENCH_PHASE_BOUNDARY ==="


def _phase_marker_line(phase_from: str, phase_to: str, boundary_ts: float) -> bytes:
    # Keep it simple and unique; make it easy to grep.
    return (f"{_PHASE_MARKER_TOKEN} from={phase_from} to={phase_to} boundary_ts={boundary_ts:.6f}\n").encode("utf-8")


class _ProcStdoutCapture:
    """Capture a subprocess stdout stream into a single raw log file.

    We also support injecting a marker line, protected by a lock, to avoid
    interleaving/corrupt writes.
    """

    def __init__(self, proc: subprocess.Popen, raw_log_path: Path):
        if proc.stdout is None:
            raise ValueError("proc.stdout is None; start process with stdout=PIPE")
        self._proc = proc
        self._raw_log_path = raw_log_path
        self._raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Unbuffered binary file to preserve original output bytes.
        self._fd = open(self._raw_log_path, "wb", buffering=0)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="vllm_exp_stdout_capture", daemon=True)
        self._thread.start()

    @property
    def raw_log_path(self) -> Path:
        return self._raw_log_path

    def write_marker(self, marker_line: bytes) -> None:
        with self._lock:
            self._fd.write(marker_line)
            try:
                self._fd.flush()
            except Exception:
                pass

    def _run(self) -> None:
        # Stream read loop.
        try:
            while True:
                chunk = self._proc.stdout.readline()  # type: ignore[union-attr]
                if not chunk:
                    break
                with self._lock:
                    self._fd.write(chunk)
        finally:
            try:
                with self._lock:
                    self._fd.flush()
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._thread.join(timeout=10)
        except Exception:
            pass
        try:
            with self._lock:
                self._fd.flush()
                self._fd.close()
        except Exception:
            pass


def _split_server_log_by_marker(raw_path: Path, warmup_path: Path, main_path: Path) -> bool:
    """Split a raw server log into warmup/main parts.

    Returns True if marker was found and split happened.
    The marker line itself is not included in either output file.
    """
    marker_bytes = _PHASE_MARKER_TOKEN.encode("utf-8")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    warmup_path.parent.mkdir(parents=True, exist_ok=True)
    main_path.parent.mkdir(parents=True, exist_ok=True)

    found = False
    # Always overwrite outputs to keep behavior deterministic.
    with open(raw_path, "rb") as r, open(warmup_path, "wb") as w_warm, open(main_path, "wb") as w_main:
        out = w_warm
        while True:
            line = r.readline()
            if not line:
                break
            if marker_bytes in line:
                found = True
                out = w_main
                continue
            out.write(line)
    return found


def _split_timestamp_metrics_csv(raw_path: Path, boundary_ts: float, warmup_path: Path, main_path: Path) -> None:
    """Split vLLM timestamp metrics CSV by timestamp column.

    The output CSVs contain only valid CSV rows (no markers).
    """
    warmup_path.parent.mkdir(parents=True, exist_ok=True)
    main_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        return

    with open(raw_path, "r", newline="", encoding="utf-8", errors="ignore") as r, \
            open(warmup_path, "w", newline="") as w_warm, \
            open(main_path, "w", newline="") as w_main:
        reader = csv.reader(r)
        warm_writer = csv.writer(w_warm)
        main_writer = csv.writer(w_main)

        header: Optional[list[str]] = None
        for row in reader:
            if not row:
                continue
            # Header row
            if row[0] == "timestamp":
                header = row
                continue
            if header is None:
                # If header is missing for some reason, synthesize the minimal known header.
                header = ["timestamp"]
            try:
                ts = float(row[0])
            except Exception:
                # Skip malformed lines.
                continue
            if ts < boundary_ts:
                # Ensure header exists once
                if w_warm.tell() == 0:
                    warm_writer.writerow(header)
                warm_writer.writerow(row)
            else:
                if w_main.tell() == 0:
                    main_writer.writerow(header)
                main_writer.writerow(row)
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


def wait_ready_or_fail(
    base_url: str,
    proc: subprocess.Popen,
    log_path: Path,
    timeout_s: int = 300,
) -> bool:
    """Wait for server ready, checking for early process termination.
    
    Returns True if server is ready, False if timeout or process died.
    Prints error info if process dies early.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        # Check if process died
        if proc.poll() is not None:
            C.print(f"[red]ERROR[/] Server terminated early (exit={proc.returncode})")
            if log_path.exists():
                with open(log_path, "r") as f:
                    for line in f.readlines()[-20:]:
                        print(line.rstrip())
            return False
        # Check if ready
        try:
            if requests.get(f"{base_url}/v1/models", timeout=3).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    C.print("[red]ERROR[/] Server not ready in time")
    return False


def get_path_policy_from_var_keys(var_keys: list[str]) -> PathPolicy:
    return PathPolicy(variables=var_keys)

## Functions to start vllm and benchmark ##
def start_vllm(cfg: Config,  spec: ServerRunSpec, logm: LogManager, vars: Optional[dict[str, Any]] = None) -> subprocess.Popen:
    env = os.environ.copy()
    # 传递 LayerKV 双向通道所需的 rank->ip 映射（JSON 字符串）
    if cfg.network.rank_to_ip:
        import json as _json
        env["VLLM_LAYERKV_RANK_TO_IP"] = _json.dumps(cfg.network.rank_to_ip)

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
    # Add block_size if specified
    if cfg.vllm.block_size is not None:
        serve_args.extend(["--block-size", str(cfg.vllm.block_size)])
    
    # Add dynamic config - pass all parameters through dynamic_cfg (no file needed)
    alternative_configs_dict = {"pp_layer_configs": cfg.migration.alternative_configs}
    dynamic_cfg = json.dumps({
        "enable_flexi_flash_attn": cfg.vllm.enable_flexi_flash_attn,
        "tester_start_step": cfg.migration.tester_start_step,
        "memory_stress_tester": cfg.migration.memory_stress_tester,
        "pp_layer_partition": spec.start_pp_layer_partition,
        "pattern_batch_size": cfg.benchmark.pattern_batch_size,
        "alternative_configs": alternative_configs_dict,
        "migration_steps": cfg.migration.migration_steps,
        })
    serve_args.extend(["-D", dynamic_cfg])
    
    if cfg.vllm.chunked_prefill:
        serve_args.append("--enable-chunked-prefill")
    if not cfg.vllm.enable_cuda_graph:
        serve_args.append("--enforce-eager")
    if cfg.vllm.enable_nsight:
        serve_args.append("--ray-workers-use-nsight")
    # 传递 Ray worker 节点部署映射（用于跨节点 pipeline parallelism）
    if cfg.network.rank_to_node:
        import json as _json
        serve_args.extend(["--ray-rank-to-node", _json.dumps(cfg.network.rank_to_node)])

    metrics_path = logm.get_path_with_log_type("timestamp_metrics", "csv", vars)
    if metrics_path.exists():
        os.remove(metrics_path)
    env["VLLM_METRICS_CSV_PATH"] = str(metrics_path)

    path = logm.get_path_with_log_type("server", "log", vars)
    # 如果file存在的话
    if path.exists():
        os.remove(path)
    log_fd = open(path, "wb", buffering=0)

    proc = subprocess.Popen(
        serve_args,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid
    )
    return proc


def start_vllm_with_raw_logging(cfg: Config, spec: ServerRunSpec, logm: LogManager,
                               vars: Optional[dict[str, Any]] = None) -> tuple[subprocess.Popen, _ProcStdoutCapture, Path]:
    """Start vLLM with stdout captured to a single raw log file.

    Returns (proc, capture, timestamp_metrics_raw_path).
    """
    env = os.environ.copy()
    if cfg.network.rank_to_ip:
        import json as _json
        env["VLLM_LAYERKV_RANK_TO_IP"] = _json.dumps(cfg.network.rank_to_ip)

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
    # Add block_size if specified
    if cfg.vllm.block_size is not None:
        serve_args.extend(["--block-size", str(cfg.vllm.block_size)])
    
    # Add dynamic config - pass all parameters through dynamic_cfg (no file needed)
    alternative_configs_dict = {"pp_layer_configs": cfg.migration.alternative_configs}
    dynamic_cfg = json.dumps({
        "enable_flexi_flash_attn": cfg.vllm.enable_flexi_flash_attn,
        "tester_start_step": cfg.migration.tester_start_step,
        "memory_stress_tester": cfg.migration.memory_stress_tester,
        "pp_layer_partition": spec.start_pp_layer_partition,
        "pattern_batch_size": cfg.benchmark.pattern_batch_size,
        "alternative_configs": alternative_configs_dict,
        "migration_steps": cfg.migration.migration_steps,
    })
    serve_args.extend(["-D", dynamic_cfg])

    if cfg.vllm.chunked_prefill:
        serve_args.append("--enable-chunked-prefill")
    if not cfg.vllm.enable_cuda_graph:
        serve_args.append("--enforce-eager")
    if cfg.vllm.enable_nsight:
        serve_args.append("--ray-workers-use-nsight")
    # 传递 Ray worker 节点部署映射（用于跨节点 pipeline parallelism）
    if cfg.network.rank_to_node:
        import json as _json
        serve_args.extend(["--ray-rank-to-node", _json.dumps(cfg.network.rank_to_node)])

    # Timestamp metrics: write a raw CSV, then split into warmup/main later.
    metrics_raw_path = logm.get_path_with_log_type("timestamp_metrics_raw", "csv", vars)
    if metrics_raw_path.exists():
        os.remove(metrics_raw_path)
    env["VLLM_METRICS_CSV_PATH"] = str(metrics_raw_path)

    raw_log_path = logm.get_path_with_log_type("server_raw", "log", vars)
    if raw_log_path.exists():
        os.remove(raw_log_path)

    proc = subprocess.Popen(
        serve_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
        bufsize=0,
    )
    capture = _ProcStdoutCapture(proc, raw_log_path)
    return proc, capture, metrics_raw_path

def start_benchmark(
    cfg: Config,
    spec: BenchmarkRunSpec,
    logm: LogManager,
    vars: Optional[dict[str, Any]] = None,
    *,
    log_basename: str = "benchmark",
    request_metrics_basename: str = "request_metrics",
    benchmark_cfg_override: Optional[dict[str, Any]] = None,
    on_stdout_line: Optional[Callable[[str], None]] = None,
) -> bool:
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
        cfg_payload = benchmark_cfg_override if benchmark_cfg_override is not None else cfg.benchmark.model_dump()
        json.dump(cfg_payload, f, indent=2, ensure_ascii=False)
    
    # 规定metrics文件的命名方式
    metrics_file_name = logm.get_path_with_log_type(request_metrics_basename, "csv", vars)
    if metrics_file_name.exists():
        os.remove(metrics_file_name)
    envs = os.environ.copy()
    envs.update({"METRICS_FILE_NAME": str(metrics_file_name)})

    log_path = logm.get_path_with_log_type(log_basename, "log", vars)
    if log_path.exists():
        os.remove(log_path)
    C.print(f"[bold cyan] log_path: {log_path}")

    if on_stdout_line is None:
        bench_fd = open(log_path, "wb", buffering=0)
        ret = subprocess.run(
            bench_args,
            stdout=bench_fd,
            stderr=subprocess.STDOUT,
            env=envs,
        )
        ok = (ret.returncode == 0)
    else:
        # Stream stdout so we can detect phase boundary markers in real time.
        proc = subprocess.Popen(
            bench_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=envs,
            bufsize=0,
        )
        assert proc.stdout is not None
        with open(log_path, "wb", buffering=0) as bench_fd:
            for raw in iter(proc.stdout.readline, b""):
                bench_fd.write(raw)
                try:
                    line = raw.decode("utf-8", errors="ignore").rstrip("\n")
                    on_stdout_line(line)
                except Exception:
                    pass
        retcode = proc.wait()
        ok = (retcode == 0)
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
        proc, capture, timestamp_metrics_raw_path = start_vllm_with_raw_logging(cfg=cfg, spec=spec, logm=logm, vars=vars_mapping)

        # Single benchmark run. If warmup is enabled, benchmark script will
        # submit warmup stages then immediately submit main stages (no pause)
        # and print a boundary marker line. We detect that marker live and
        # inject a corresponding server marker into server_raw.

        boundary_ts: Optional[float] = None
        warmup_enabled = cfg.benchmark.warmup is not None and cfg.benchmark.warmup.enabled

        # Prepare per-phase request_metrics outputs (benchmark script writes valid CSV only).
        if warmup_enabled:
            req_warm = logm.get_path_with_log_type("request_metrics_warmup", "csv", vars_mapping)
            req_main = logm.get_path_with_log_type("request_metrics", "csv", vars_mapping)
            if req_warm.exists():
                os.remove(req_warm)
            if req_main.exists():
                os.remove(req_main)
            os.environ["METRICS_FILE_NAME_WARMUP"] = str(req_warm)
            os.environ["METRICS_FILE_NAME_MAIN"] = str(req_main)

        def _on_bench_line(line: str) -> None:
            nonlocal boundary_ts
            if boundary_ts is not None:
                return
            if _BENCH_PHASE_BOUNDARY_TOKEN in line:
                # Expected format: "=== VLLM_EXP_BENCH_PHASE_BOUNDARY === boundary_ts=..."
                try:
                    parts = line.split("boundary_ts=")
                    ts = float(parts[-1].strip())
                except Exception:
                    ts = time.time()
                boundary_ts = ts
                capture.write_marker(_phase_marker_line("warmup", "main", boundary_ts))

        bench_spec = BenchmarkRunSpec(
            input_output_len=cfg.benchmark.input_output_lens,
            running_request_rate_list=cfg.benchmark.running_request_rates,
        )
        # Keep a single benchmark log file. It will contain the boundary marker
        # line and both warmup/main summary blocks.
        bench_log_basename = "benchmark"
        ok = start_benchmark(
            cfg=cfg,
            spec=bench_spec,
            logm=logm,
            vars=vars_mapping,
            log_basename=bench_log_basename,
            request_metrics_basename="request_metrics",
            benchmark_cfg_override=cfg.benchmark.model_dump(),
            on_stdout_line=_on_bench_line if warmup_enabled else None,
        )
        benchmark_failed = not ok
        if benchmark_failed:
            C.print("[yellow]WARNING:[/] Benchmark failed, but will attempt to process logs")
        
        stop_tree(proc)

        # Ensure raw capture thread is done.
        capture.close()

        # Split server raw log into warmup/main (marker excluded from outputs)
        # Even if benchmark failed, we should still try to split the logs that were captured.
        server_raw_path = logm.get_path_with_log_type("server_raw", "log", vars_mapping)
        server_warmup_path = logm.get_path_with_log_type("server_warmup", "log", vars_mapping)
        server_main_path = logm.get_path_with_log_type("server", "log", vars_mapping)
        if server_warmup_path.exists():
            os.remove(server_warmup_path)
        if server_main_path.exists():
            os.remove(server_main_path)
        split_ok = _split_server_log_by_marker(server_raw_path, server_warmup_path, server_main_path)
        if not split_ok:
            # No boundary marker found (e.g., warmup disabled): keep compatibility by
            # making server-*.log contain the full raw log for downstream analysis.
            if server_warmup_path.exists():
                os.remove(server_warmup_path)
            if server_main_path.exists():
                os.remove(server_main_path)
            shutil.copyfile(server_raw_path, server_main_path)

        # Split timestamp metrics CSV by boundary_ts if warmup was enabled and marker inserted.
        if boundary_ts is not None and split_ok:
            ts_warmup_path = logm.get_path_with_log_type("timestamp_metrics_warmup", "csv", vars_mapping)
            ts_main_path = logm.get_path_with_log_type("timestamp_metrics", "csv", vars_mapping)
            if ts_warmup_path.exists():
                os.remove(ts_warmup_path)
            if ts_main_path.exists():
                os.remove(ts_main_path)
            _split_timestamp_metrics_csv(timestamp_metrics_raw_path, boundary_ts, ts_warmup_path, ts_main_path)
        else:
            # No warmup boundary: keep compatibility by producing a main timestamp_metrics from raw.
            ts_main_path = logm.get_path_with_log_type("timestamp_metrics", "csv", vars_mapping)
            if ts_main_path.exists():
                os.remove(ts_main_path)
            if timestamp_metrics_raw_path.exists():
                shutil.copyfile(timestamp_metrics_raw_path, ts_main_path)

        # After attempting to process all logs, raise exception if benchmark failed
        if benchmark_failed:
            raise Exception("Benchmark failed (logs have been split if possible)")

        # Note: benchmark log is not split into separate files by design.
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
            server_spec = ServerRunSpec(part, cfg.migration.is_migration)
            vars_mapping = {"start_pp_layer_partition": server_spec.start_pp_layer_partition}
            proc = start_vllm(cfg=cfg, spec=server_spec, logm=logm, vars=vars_mapping)
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


# =========================
# Sweep Test Implementation
# =========================

from .data import (
    SweepTestConfig, SweepBenchmarkConfig, SweepConfig, StaticConfig,
    VllmServerSpec, BenchmarkSpec, WarmupBenchCfg, extract_naming_vars,
    ExperimentConfig,
)


# =========================
# Spec Data Classes for Generator Results
# =========================

@dataclass
class SweepExperimentSpec:
    """A single experiment specification generated from sweep config.
    
    Contains everything needed to run one experiment.
    """
    vllm_spec: VllmServerSpec
    bench_spec: BenchmarkSpec
    exp_config: ExperimentConfig  # The merged experiment config
    vars_mapping: Dict[str, Any]
    experiment_index: int
    total_experiments: int


# =========================
# Spec Generator
# =========================

def generate_experiment_specs(
    sweep_test_cfg: SweepTestConfig,
    logm: SweepLogManager,
) -> Iterable[SweepExperimentSpec]:
    """Generate experiment specs from SweepTestConfig.
    
    Uses ExperimentConfig.iter_from_sweep_test_config() to iterate over
    all combinations of static_config × sweep_config, then generates
    VllmServerSpec and BenchmarkSpec from each ExperimentConfig.
    
    Args:
        sweep_test_cfg: Complete sweep test configuration
        logm: Log manager for generating file paths
    
    Yields:
        SweepExperimentSpec containing specs and metadata for each experiment
    """
    # First pass to count total experiments
    exp_configs = list(ExperimentConfig.iter_from_sweep_test_config(sweep_test_cfg))
    total = len(exp_configs)
    
    for i, exp_cfg in enumerate(exp_configs):
        # Get vars_mapping from ExperimentConfig
        vars_mapping = exp_cfg.get_naming_vars()
        
        # Generate log paths using logm
        metrics_csv_path = str(logm.get_path_with_log_type("timestamp_metrics_raw", "csv", vars_mapping))
        server_raw_log_path = str(logm.get_path_with_log_type("server_raw", "log", vars_mapping))
        benchmark_log_path = str(logm.get_path_with_log_type("benchmark", "log", vars_mapping))
        benchmark_config_path = str(logm.get_path_with_log_type("benchmark_config", "json", vars_mapping))
        metrics_file_path = str(logm.get_path_with_log_type("request_metrics", "csv", vars_mapping))
        
        # Build VllmServerSpec from ExperimentConfig
        vllm_spec = VllmServerSpec(
            model_path=exp_cfg.model.path,
            model_name=exp_cfg.model.name,
            pipeline_parallel_size=exp_cfg.vllm.pipeline_parallel_size,
            gpu_memory_utilization=exp_cfg.vllm.gpu_memory_utilization,
            max_model_len=exp_cfg.vllm.max_model_len,
            block_size=exp_cfg.vllm.block_size,
            head_addr=exp_cfg.vllm.head_addr,
            port=exp_cfg.vllm.port,
            enable_flexi_flash_attn=exp_cfg.vllm.enable_flexi_flash_attn,
            chunked_prefill=exp_cfg.vllm.chunked_prefill,
            enable_cuda_graph=exp_cfg.vllm.enable_cuda_graph,
            enable_nsight=exp_cfg.vllm.enable_nsight,
            pp_layer_partition=exp_cfg.vllm.pp_layer_partition,
            pp_layer_config=exp_cfg.vllm.pp_layer_config,
            alternative_configs=exp_cfg.vllm.get_alternative_configs(),
            migration_steps=exp_cfg.vllm.get_migration_steps(),
            rank_to_ip=exp_cfg.network.rank_to_ip,
            rank_to_node=exp_cfg.network.rank_to_node,
            pattern_batch_size=exp_cfg.benchmark.pattern_batch_size,
            metrics_csv_path=metrics_csv_path,
            server_raw_log_path=server_raw_log_path,
        )
        
        # Build BenchmarkSpec from ExperimentConfig
        bench_spec = BenchmarkSpec(
            base_url=exp_cfg.vllm.base_url,
            model_path=exp_cfg.model.path,
            model_name=exp_cfg.model.name,
            benchmark_script_path=exp_cfg.benchmark.benchmark_script_path,
            benchmark_config_path=benchmark_config_path,
            metrics_file_path=metrics_file_path,
            benchmark_log_path=benchmark_log_path,
            num_total_requests=exp_cfg.benchmark.num_total_requests,
            request_rate=exp_cfg.benchmark.request_rate_dict,
            input_output_lens=exp_cfg.benchmark.input_output_lens,
            pattern_batch_size=exp_cfg.benchmark.pattern_batch_size,
            burstiness=exp_cfg.benchmark.burstiness,
            repetition=exp_cfg.benchmark.repetition,
            print_outputs=exp_cfg.benchmark.print_outputs,
            profile=exp_cfg.benchmark.profile,
            warmup=exp_cfg.benchmark.warmup,
        )
        
        yield SweepExperimentSpec(
            vllm_spec=vllm_spec,
            bench_spec=bench_spec,
            exp_config=exp_cfg,
            vars_mapping=vars_mapping,
            experiment_index=i,
            total_experiments=total,
        )


# =========================
# Start Functions (Spec -> Process)
# =========================

def start_vllm_for_sweep(
    spec: VllmServerSpec,
) -> tuple[subprocess.Popen, _ProcStdoutCapture, Path]:
    """Start vLLM server from spec.
    
    All parameters including log paths are contained in the spec.
    No additional parsing or path generation needed.
    
    Args:
        spec: VllmServerSpec with all parameters including paths
    
    Returns:
        (process, stdout_capture, metrics_path)
    
    Raises:
        ValueError: If metrics_csv_path or server_raw_log_path is not set in spec
    """
    # Validate required paths are set
    if not spec.metrics_csv_path:
        raise ValueError("spec.metrics_csv_path must be set before calling start_vllm_for_sweep")
    if not spec.server_raw_log_path:
        raise ValueError("spec.server_raw_log_path must be set before calling start_vllm_for_sweep")
    
    env = os.environ.copy()
    
    serve_args = [
        "vllm", "serve", spec.model_path,
        "--pipeline-parallel-size", str(spec.pipeline_parallel_size),
        "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
        "--max-model-len", str(spec.max_model_len),
        "--served-model-name", spec.model_name,
        "--distributed-executor-backend", "ray",
        "--disable-log-requests",
        "--no-enable-prefix-caching",
        "--scheduler-cls", "vllm.v1.core.sched.dynamic_scheduler.DynamicScheduler",
        "--worker-cls", "vllm.v1.worker.dynamic_gpu_worker.DynamicGPUWorker",
    ]
    
    if spec.block_size:
        serve_args.extend(["--block-size", str(spec.block_size)])
    if spec.chunked_prefill:
        serve_args.append("--enable-chunked-prefill")
    if not spec.enable_cuda_graph:
        serve_args.append("--enforce-eager")
    if spec.enable_nsight:
        serve_args.append("--ray-workers-use-nsight")
    if spec.rank_to_node:
        serve_args.extend(["--ray-rank-to-node", json.dumps(spec.rank_to_node)])
    # Generate dynamic config from spec (includes all settings like rank_to_ip, metrics_csv_path, etc.)
    dynamic_cfg = json.dumps(spec.to_dynamic_cfg())
    serve_args.extend(["-D", dynamic_cfg])
    
     
    # Clear existing metrics file
    metrics_raw_path = Path(spec.metrics_csv_path)
    if metrics_raw_path.exists():
        os.remove(metrics_raw_path)

    # Clear existing raw log file
    raw_log_path = Path(spec.server_raw_log_path)
    if raw_log_path.exists():
        os.remove(raw_log_path)

    C.print(f"[bold cyan]Starting vLLM with pp_layer_config: {spec.pp_layer_config}[/]")
    
    proc = subprocess.Popen(
        serve_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
        bufsize=0,
    )
    capture = _ProcStdoutCapture(proc, raw_log_path)
    return proc, capture, metrics_raw_path


def start_benchmark_for_sweep(
    spec: BenchmarkSpec,
) -> bool:
    """Start benchmark from spec.
    
    Args:
        spec: BenchmarkSpec with all parameters
        logm: Log manager for file paths
        vars: Variables for file naming
    
    Returns:
        True if benchmark succeeded
    """
    if not wait_ready(spec.base_url, 300):
        C.print("[red]ERROR[/] vLLM not ready in time")
        return False
    
    bench_args = [
        "python3", spec.benchmark_script_path,
        "--request-rate", "0",  # Use config-based rates
        "--backend", "openai-chat",
        "--model", spec.model_path,
        "--endpoint", "/v1/chat/completions",
        "--base-url", spec.base_url,
        "--dataset-name", "pattern",
        "--served-model-name", spec.model_name,
        "--goodput", "tpot:300", "ttft:5000",
        "--temperature", "0",
        "--seed", "42",
        "--pattern-batch-size", str(spec.pattern_batch_size),
        "--benchmark-config", str(spec.benchmark_config_path),
    ]
    
    if spec.print_outputs:
        bench_args.append("--print-outputs")
    if spec.profile:
        bench_args.append("--profile")
    
    # Write benchmark config (includes metrics path) via spec helper
    spec.write_benchmark_config()
    
    C.print(f"[bold cyan]Starting benchmark with {spec.num_total_requests} requests[/]")
    C.print(f"[bold cyan]Request rate config: {spec.request_rate}[/]")
    
    # Prepare metrics output path defined in spec
    metrics_file_path = Path(spec.metrics_file_path)
    if metrics_file_path.exists():
        os.remove(metrics_file_path)
    
    log_path = Path(spec.benchmark_log_path)
    if log_path.exists():
        os.remove(log_path)
    
    bench_fd = open(log_path, "wb", buffering=0)
    ret = subprocess.run(
        bench_args,
        stdout=bench_fd,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    return ret.returncode == 0


# =========================
# Experiment Runner
# =========================

def _run_single_sweep_experiment(
    vllm_spec: VllmServerSpec,
    bench_spec: BenchmarkSpec,
    logm: SweepLogManager,
    vars_mapping: Dict[str, Any],
) -> bool:
    """Run a single sweep experiment.
    
    Args:
        vllm_spec: VllmServerSpec for starting vLLM
        bench_spec: BenchmarkSpec for running benchmark
        logm: Log manager
        vars_mapping: Variables for file naming
    
    Returns:
        True if experiment succeeded
    """
    ok = False
    proc = None
    capture = None
    
    try:
        logm.write_constants_meta(vars_mapping)
        
        # Start vLLM (spec contains all paths - validated in start_vllm_for_sweep)
        proc, capture, metrics_raw_path = start_vllm_for_sweep(vllm_spec)
        # server_raw_log_path is guaranteed non-None after start_vllm_for_sweep validation
        assert vllm_spec.server_raw_log_path is not None
        server_raw_path = Path(vllm_spec.server_raw_log_path)
        
        # Wait for server ready (with early termination detection)
        if not wait_ready_or_fail(bench_spec.base_url, proc, server_raw_path):
            return False
        
        C.print("[green]vLLM server is ready[/]")
        
        # Start benchmark
        ok = start_benchmark_for_sweep(bench_spec)
        
        if not ok:
            C.print("[yellow]WARNING:[/] Benchmark failed, but will process logs")
        
        stop_tree(proc)
        capture.close()
        
        # Process logs - copy raw to main (server_raw_path already defined above)
        server_main_path = logm.get_path_with_log_type("server", "log", vars_mapping)
        if server_main_path.exists():
            os.remove(server_main_path)
        if server_raw_path.exists():
            shutil.copyfile(server_raw_path, server_main_path)
        
        # Copy metrics (use path from spec)
        ts_main_path = logm.get_path_with_log_type("timestamp_metrics", "csv", vars_mapping)
        if ts_main_path.exists():
            os.remove(ts_main_path)
        if metrics_raw_path.exists():
            shutil.copyfile(metrics_raw_path, ts_main_path)
        
    except Exception as e:
        C.print(f"[red]ERROR[/] {e}")
        import traceback
        traceback.print_exc()
        ok = False
    finally:
        if proc:
            stop_tree(proc)
        if capture:
            capture.close()
        time.sleep(3)
        generate_analysis_report(logm, vars_mapping)
    
    return ok


def sweep_test(cfg: SweepTestConfig, logm: SweepLogManager):
    """Run sweep test experiments.
    
    Uses generator to produce experiment specs from config,
    then runs each experiment sequentially.
    """
    all_ok = True
    experiment_count = 0
    
    for exp in generate_experiment_specs(cfg, logm):
        experiment_count += 1
        C.print(f"\n[bold magenta]{'='*60}[/]")
        C.print(f"[bold magenta]Experiment {exp.experiment_index + 1}/{exp.total_experiments}[/]")
        C.print(f"[bold magenta]vars: {exp.vars_mapping}[/]")
        C.print(f"[bold magenta]{'='*60}[/]\n")
        
        ok = _run_single_sweep_experiment(
            exp.vllm_spec, exp.bench_spec, logm, exp.vars_mapping
        )
        if not ok:
            all_ok = False
            C.print(f"[yellow]Experiment {exp.experiment_index + 1}/{exp.total_experiments} failed, finished...[/]")
            return
    
    if experiment_count == 0:
        C.print("[red]ERROR: No valid experiments generated (benchmark_config required)[/]")
        return
    
    status = "All succeeded" if all_ok else "Some failed"
    C.print(f"\n[bold cyan]Sweep test completed: {experiment_count} experiments, {status}[/]")

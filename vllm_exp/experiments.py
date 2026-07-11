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
import sys
import time
import threading
import csv
import statistics
from urllib.parse import urlsplit, urlunsplit
from dataclasses import dataclass, asdict, field, replace
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


def _format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unset"
    if num_bytes % (1024 * 1024) == 0:
        return f"{num_bytes // (1024 * 1024)}MB"
    if num_bytes % 1024 == 0:
        return f"{num_bytes // 1024}KB"
    return f"{num_bytes}B"


def _log_page_attention_block_geometry(spec: "VllmServerSpec") -> None:
    if spec.page_attention_block_size_bytes is None:
        if spec.block_size:
            C.print(
                "[bold cyan]Using explicit vLLM token block_size: "
                f"{spec.block_size}[/]")
        return
    C.print(
        "[bold cyan]Derived PageAttention geometry: "
        f"K+V page_attention_block_size="
        f"{_format_bytes(spec.page_attention_block_size_bytes)}, "
        f"vLLM token block_size={spec.block_size}[/]")


def _apply_page_attention_block_env(env: dict[str, str],
                                    spec: "VllmServerSpec") -> None:
    if spec.page_attention_block_size_bytes is None:
        return
    env["VLLM_PAGE_ATTENTION_BLOCK_SIZE_BYTES"] = str(
        spec.page_attention_block_size_bytes)
    if spec.block_size is not None:
        env["VLLM_DERIVED_TOKEN_BLOCK_SIZE"] = str(spec.block_size)


def generate_analysis_report(logm: Union[LogManager, SweepLogManager], vars: Optional[dict[str, Any]] = None):
    """
    Automatically generate analysis files in the log directory after experiment completes.
    Uses logm.get_path_with_log_type to construct filenames consistently.
    
    For sweep_test mode (SweepLogManager with vars), this generates an analysis report
    for the specific experiment's server.log in its subdirectory.
    """
    try:
        # If vars is provided, use get_path_with_log_type to find the exact server log
        # This works correctly for both LogManager and SweepLogManager
        if vars is not None:
            server_log_path = logm.get_path_with_log_type("server", "log", vars)
            if not server_log_path.exists():
                C.print(f"[yellow]Warning: Server log not found: {server_log_path}[/]")
                return
            server_logs = [server_log_path]
            log_dir = server_log_path.parent
        else:
            # No vars: search for server logs in base directory
            log_dir = logm.get_dir()
            server_logs = list(log_dir.glob("server-*.log"))
            if not server_logs:
                # Also try server.log (used in sweep_test subdirectories)
                server_logs = list(log_dir.glob("**/server.log"))
            
            if not server_logs:
                C.print(f"[yellow]Warning: No server logs found in {log_dir}[/]")
                return
        
        C.print(f"[bold cyan]Generating analysis reports in {log_dir}[/]")
        
        # Process each server log
        for log_file in sorted(server_logs):
            # Use logm.get_path_with_log_type to construct analysis filename
            # This ensures consistency with server log naming
            analysis_path = logm.get_path_with_log_type("analysis", "txt", vars)
            
            C.print(f"  Analyzing {log_file.name} -> {analysis_path}")
            
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


def _prepend_repo_to_pythonpath(env: Dict[str, str]) -> None:
    repo_root = str(Path(__file__).resolve().parents[1])
    pythonpath = env.get("PYTHONPATH")
    if pythonpath:
        if pythonpath.split(os.pathsep)[0] == repo_root:
            return
        env["PYTHONPATH"] = os.pathsep.join([repo_root, pythonpath])
    else:
        env["PYTHONPATH"] = repo_root


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


class DynamicLogRedirector:
    """Dynamically redirect subprocess output to different log files in real-time.
    
    Unlike _ProcStdoutCapture which writes to a single file, this class allows
    switching the output destination at runtime. This is useful for sweep_test
    where a single server instance serves multiple experiments, each requiring
    its own log file.
    
    Additionally, this class supports writing to a global log file that captures
    all output, useful for real-time monitoring during long-running experiments.
    
    Usage:
        redirector = DynamicLogRedirector(proc, global_log_path=Path("/path/to/global.log"))
        redirector.switch_log_file(Path("/path/to/exp1.log"))
        # ... run experiment 1 ...
        redirector.switch_log_file(Path("/path/to/exp2.log"))
        # ... run experiment 2 ...
        redirector.close()
    """

    def __init__(self, proc: subprocess.Popen, initial_log_path: Optional[Path] = None,
                 global_log_path: Optional[Path] = None):
        """Initialize the dynamic log redirector.
        
        Args:
            proc: Subprocess with stdout=PIPE
            initial_log_path: Optional initial log file path for per-experiment logs.
                              If not provided, output is discarded until switch_log_file is called.
            global_log_path: Optional path for a global log file that captures ALL output.
                             This file is written to continuously and is useful for monitoring.
        """
        if proc.stdout is None:
            raise ValueError("proc.stdout is None; start process with stdout=PIPE")
        self._proc = proc
        self._current_log_path: Optional[Path] = None
        self._current_fd: Optional[FileIO] = None
        self._global_log_path: Optional[Path] = global_log_path
        self._global_fd: Optional[FileIO] = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(
            target=self._run, 
            name="vllm_exp_dynamic_log_redirector", 
            daemon=True
        )
        
        # Open global log file if provided
        if global_log_path:
            global_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._global_fd = open(global_log_path, "wb", buffering=0)
            startup_marker = f"=== GLOBAL_LOG_START: {datetime.datetime.now().isoformat()} ===\n"
            self._global_fd.write(startup_marker.encode('utf-8'))
            C.print(f"[bold cyan]Global server log: {global_log_path}[/]")
        
        # Set initial log file if provided
        if initial_log_path:
            self._switch_log_file_internal(initial_log_path)
        
        self._thread.start()

    @property
    def current_log_path(self) -> Optional[Path]:
        """Get the current log file path."""
        with self._lock:
            return self._current_log_path
    
    @property
    def global_log_path(self) -> Optional[Path]:
        """Get the global log file path."""
        return self._global_log_path

    def _switch_log_file_internal(self, new_log_path: Path) -> None:
        """Internal method to switch log file without lock (caller must hold lock)."""
        # Close current file if open
        if self._current_fd is not None:
            try:
                self._current_fd.flush()
                self._current_fd.close()
            except Exception:
                pass
        
        # Open new file
        new_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._current_fd = open(new_log_path, "wb", buffering=0)
        self._current_log_path = new_log_path
        
        # Write a marker to indicate log switch
        marker = f"=== LOG_SWITCH: Switched to {new_log_path} at {datetime.datetime.now().isoformat()} ===\n"
        marker_bytes = marker.encode('utf-8')
        self._current_fd.write(marker_bytes)
        
        # Also write marker to global log
        if self._global_fd is not None:
            self._global_fd.write(marker_bytes)

    def switch_log_file(self, new_log_path: Path) -> None:
        """Switch to a new log file for subsequent output.
        
        This method is thread-safe and can be called while the subprocess
        is producing output. The switch happens atomically.
        
        Args:
            new_log_path: Path to the new log file. Parent directories will be created.
        """
        with self._lock:
            C.print(f"[bold cyan]Switching log output to: {new_log_path}[/]")
            self._switch_log_file_internal(new_log_path)

    def write_marker(self, marker_line: bytes) -> None:
        """Write a marker line to the current log file.
        
        Args:
            marker_line: Bytes to write as a marker.
        """
        with self._lock:
            if self._current_fd is not None:
                self._current_fd.write(marker_line)
                try:
                    self._current_fd.flush()
                except Exception:
                    pass
            # Also write to global log
            if self._global_fd is not None:
                self._global_fd.write(marker_line)
                try:
                    self._global_fd.flush()
                except Exception:
                    pass

    def _run(self) -> None:
        """Stream read loop - runs in background thread."""
        try:
            while self._running:
                chunk = self._proc.stdout.readline()  # type: ignore[union-attr]
                if not chunk:
                    break
                with self._lock:
                    # Write to per-experiment log
                    if self._current_fd is not None:
                        self._current_fd.write(chunk)
                    # Also write to global log
                    if self._global_fd is not None:
                        self._global_fd.write(chunk)
        finally:
            try:
                with self._lock:
                    if self._current_fd is not None:
                        self._current_fd.flush()
                    if self._global_fd is not None:
                        self._global_fd.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Close the redirector and release resources."""
        self._running = False
        try:
            self._thread.join(timeout=10)
        except Exception:
            pass
        try:
            with self._lock:
                if self._current_fd is not None:
                    self._current_fd.flush()
                    self._current_fd.close()
                    self._current_fd = None
                if self._global_fd is not None:
                    # Write closing marker
                    closing_marker = f"=== GLOBAL_LOG_END: {datetime.datetime.now().isoformat()} ===\n"
                    self._global_fd.write(closing_marker.encode('utf-8'))
                    self._global_fd.flush()
                    self._global_fd.close()
                    self._global_fd = None
        except Exception:
            pass


# =========================
# Experiment Success Checking
# =========================

import re

def check_experiment_completed(logm: SweepLogManager, vars_mapping: Dict[str, Any]) -> tuple[bool, str]:
    """Check if an experiment has already completed successfully.
    
    This function checks multiple criteria to determine if an experiment
    was successful:
    1. benchmark.log exists and contains successful completion markers
    2. request_metrics.csv exists and has data
    3. The number of successful requests matches the expected count
    
    Args:
        logm: Log manager for path generation
        vars_mapping: Variables for file naming
    
    Returns:
        Tuple of (is_completed, reason_message)
    """
    # Get paths
    benchmark_log_path = logm.get_path_with_log_type("benchmark", "log", vars_mapping)
    metrics_path = logm.get_path_with_log_type("request_metrics", "csv", vars_mapping)
    
    # Check 1: benchmark.log must exist
    if not benchmark_log_path.exists():
        return False, "benchmark.log not found"
    
    # Check 2: Parse benchmark.log for success indicators
    try:
        with open(benchmark_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return False, f"Failed to read benchmark.log: {e}"
    
    # Check for fatal errors in the log
    fatal_error_patterns = [
        r"ERROR.*Connection refused",
        r"ERROR.*Cannot connect to host",
        r"Traceback \(most recent call last\)",
        r"Exception:",
        r"Failed to start benchmark",
    ]
    
    for pattern in fatal_error_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return False, f"Found error pattern in benchmark.log: {pattern}"
    
    # Check for "Successful requests:" line
    success_match = re.search(r'Successful requests:\s+(\d+)', content)
    if not success_match:
        return False, "No 'Successful requests:' line found in benchmark.log"
    
    successful_requests = int(success_match.group(1))
    if successful_requests == 0:
        return False, "Successful requests count is 0"
    
    # Check for "Benchmark Result" section (indicates completion)
    if "Benchmark Result" not in content:
        return False, "'Benchmark Result' section not found (benchmark may have been interrupted)"
    
    # Check 3: request_metrics.csv must exist and have data
    if not metrics_path.exists():
        return False, "request_metrics.csv not found"
    
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # At least header + 1 data row
        if len(lines) < 2:
            return False, "request_metrics.csv has no data rows"
    except Exception as e:
        return False, f"Failed to read request_metrics.csv: {e}"
    
    # All checks passed
    return True, f"Experiment completed successfully ({successful_requests} requests)"


def check_experiment_completed_for_spec(
    logm: SweepLogManager, 
    vars_mapping: Dict[str, Any],
    bench_spec: 'BenchmarkSpec'
) -> tuple[bool, str]:
    """Enhanced check that also verifies the expected request count.
    
    Args:
        logm: Log manager for path generation
        vars_mapping: Variables for file naming
        bench_spec: Benchmark spec to get expected request count
    
    Returns:
        Tuple of (is_completed, reason_message)
    """
    is_completed, reason = check_experiment_completed(logm, vars_mapping)
    
    if not is_completed:
        return False, reason
    
    # Additional check: verify request count matches expectation
    benchmark_log_path = logm.get_path_with_log_type("benchmark", "log", vars_mapping)
    try:
        with open(benchmark_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        success_match = re.search(r'Successful requests:\s+(\d+)', content)
        if success_match:
            actual_count = int(success_match.group(1))
            expected_count = bench_spec.num_total_requests
            
            if actual_count < expected_count:
                return False, f"Only {actual_count}/{expected_count} requests completed"
    except Exception:
        pass  # If we can't verify, trust the basic check
    
    return True, reason


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


def _extract_timestamp_metrics_by_time_range(
    global_metrics_path: Path,
    output_path: Path,
    start_ts: float,
    end_ts: float,
) -> bool:
    """Extract timestamp metrics for a specific time range from global metrics CSV.
    
    Args:
        global_metrics_path: Path to the global metrics CSV file
        output_path: Path to write the extracted metrics
        start_ts: Start timestamp (inclusive)
        end_ts: End timestamp (inclusive)
    
    Returns:
        True if any rows were extracted
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not global_metrics_path.exists():
        return False
    
    rows_written = 0
    with open(global_metrics_path, "r", newline="", encoding="utf-8", errors="ignore") as r, \
            open(output_path, "w", newline="") as w_out:
        reader = csv.reader(r)
        writer = csv.writer(w_out)
        
        header: Optional[list[str]] = None
        for row in reader:
            if not row:
                continue
            # Header row
            if row[0] == "timestamp":
                header = row
                continue
            if header is None:
                header = ["timestamp"]
            try:
                ts = float(row[0])
            except Exception:
                continue
            # Check if timestamp is within the range
            if start_ts <= ts <= end_ts:
                if w_out.tell() == 0:
                    writer.writerow(header)
                writer.writerow(row)
                rows_written += 1
    
    return rows_written > 0


def _get_time_range_from_request_metrics(request_metrics_path: Path) -> tuple[Optional[float], Optional[float]]:
    """Get the time range of an experiment from its request_metrics.csv.
    
    The start time is the earliest request send time (timestamp column).
    The end time is the latest request completion time (timestamp + e2el).
    
    Args:
        request_metrics_path: Path to the request_metrics.csv file
    
    Returns:
        Tuple of (start_ts, end_ts) or (None, None) if file doesn't exist or is empty
    """
    if not request_metrics_path.exists():
        return None, None
    
    start_ts = None
    end_ts = None
    
    # CSV columns: timestamp, datetime, tpots, ttfts, e2els, repetition
    # e2els is the end-to-end latency in seconds
    E2EL_COL_INDEX = 4
    
    with open(request_metrics_path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] == "timestamp":
                continue
            try:
                ts = float(row[0])
                # Calculate completion time: send_time + e2el
                e2el = float(row[E2EL_COL_INDEX]) if len(row) > E2EL_COL_INDEX else 0.0
                completion_ts = ts + e2el
                
                if start_ts is None or ts < start_ts:
                    start_ts = ts
                # Use completion time for end_ts to capture all generation activity
                if end_ts is None or completion_ts > end_ts:
                    end_ts = completion_ts
            except Exception:
                continue
    
    return start_ts, end_ts

## Utils 
def load_config(path: str) -> MultiConfig:
    with open(path, "r") as f:
        return MultiConfig.model_validate(yaml.safe_load(f))

def _server_ready_timeout(default_s: int) -> int:
    raw_timeout = os.getenv("VLLM_EXP_SERVER_READY_TIMEOUT")
    if raw_timeout is None:
        return default_s
    try:
        return max(1, int(raw_timeout))
    except ValueError:
        C.print("[yellow]Ignoring invalid VLLM_EXP_SERVER_READY_TIMEOUT="
                f"{raw_timeout!r}; using {default_s}s[/]")
        return default_s


def _normalize_loopback_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.hostname != "localhost":
        return base_url
    netloc = "127.0.0.1"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query,
                       parsed.fragment))


def _server_log_has_startup_complete(log_path: Optional[Path]) -> bool:
    if log_path is None or not log_path.exists():
        return False
    try:
        with open(log_path, "r", errors="ignore") as f:
            return "Application startup complete" in f.read()
    except OSError:
        return False


def wait_ready(
    base_url: str,
    timeout_s: int = 180,
    startup_log_path: Optional[Path] = None,
) -> bool:
    base_url = _normalize_loopback_base_url(base_url)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=3)
            if r.ok and (startup_log_path is None or
                         _server_log_has_startup_complete(startup_log_path)):
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
    base_url = _normalize_loopback_base_url(base_url)
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
            if (requests.get(f"{base_url}/v1/models", timeout=3).ok
                    and _server_log_has_startup_complete(log_path)):
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
    _prepend_repo_to_pythonpath(env)
    rank_to_ip, rank_to_node, pipeline_stage_to_rank = cfg.network.resolve_worker_placement(
        cfg.vllm.pipeline_parallel_size
    )
    # 传递 LayerKV 双向通道所需的 rank->ip 映射（JSON 字符串）
    if rank_to_ip:
        import json as _json
        env["VLLM_LAYERKV_RANK_TO_IP"] = _json.dumps(rank_to_ip)

    # 启动服务
    serve_args = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", cfg.model.path,
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
        "attention_kernel": cfg.vllm.attention_kernel,
        "tester_start_step": cfg.migration.tester_start_step,
        "memory_stress_tester": cfg.migration.memory_stress_tester,
        "pp_layer_partition": spec.start_pp_layer_partition,
        "pattern_batch_size": cfg.benchmark.pattern_batch_size,
        "alternative_configs": alternative_configs_dict,
        "migration_steps": cfg.migration.migration_steps,
        "disable_memory_overhead_monitor": cfg.vllm.disable_memory_overhead_monitor,
        })
    serve_args.extend(["-D", dynamic_cfg])
    
    if cfg.vllm.chunked_prefill:
        serve_args.append("--enable-chunked-prefill")
    if cfg.vllm.max_num_batched_tokens is not None:
        serve_args.extend(["--max-num-batched-tokens", str(cfg.vllm.max_num_batched_tokens)])
    if not cfg.vllm.enable_cuda_graph:
        serve_args.append("--enforce-eager")
    if cfg.vllm.enable_nsight:
        serve_args.append("--ray-workers-use-nsight")
    # 传递 Ray worker 节点部署映射（用于跨节点 pipeline parallelism）
    if rank_to_node:
        import json as _json
        serve_args.extend(["--ray-rank-to-node", _json.dumps(rank_to_node)])
    if pipeline_stage_to_rank:
        import json as _json
        serve_args.extend(["--pipeline-stage-to-rank", _json.dumps(pipeline_stage_to_rank)])

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
    rank_to_ip, rank_to_node, pipeline_stage_to_rank = cfg.network.resolve_worker_placement(
        cfg.vllm.pipeline_parallel_size
    )
    if rank_to_ip:
        import json as _json
        env["VLLM_LAYERKV_RANK_TO_IP"] = _json.dumps(rank_to_ip)

    serve_args = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", cfg.model.path,
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
        "attention_kernel": cfg.vllm.attention_kernel,
        "tester_start_step": cfg.migration.tester_start_step,
        "memory_stress_tester": cfg.migration.memory_stress_tester,
        "pp_layer_partition": spec.start_pp_layer_partition,
        "pattern_batch_size": cfg.benchmark.pattern_batch_size,
        "alternative_configs": alternative_configs_dict,
        "migration_steps": cfg.migration.migration_steps,
        "disable_memory_overhead_monitor": cfg.vllm.disable_memory_overhead_monitor,
    })
    serve_args.extend(["-D", dynamic_cfg])

    if cfg.vllm.chunked_prefill:
        serve_args.append("--enable-chunked-prefill")
    if cfg.vllm.max_num_batched_tokens is not None:
        serve_args.extend(["--max-num-batched-tokens", str(cfg.vllm.max_num_batched_tokens)])
    if not cfg.vllm.enable_cuda_graph:
        serve_args.append("--enforce-eager")
    if cfg.vllm.enable_nsight:
        serve_args.append("--ray-workers-use-nsight")
    # 传递 Ray worker 节点部署映射（用于跨节点 pipeline parallelism）
    if rank_to_node:
        import json as _json
        serve_args.extend(["--ray-rank-to-node", _json.dumps(rank_to_node)])
    if pipeline_stage_to_rank:
        import json as _json
        serve_args.extend(["--pipeline-stage-to-rank", _json.dumps(pipeline_stage_to_rank)])

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
    
    # Get dataset configuration
    dataset_name = getattr(cfg.benchmark, 'dataset_name', 'pattern')
    dataset_path = getattr(cfg.benchmark, 'dataset_path', None)
    sharegpt_output_len = getattr(cfg.benchmark, 'sharegpt_output_len', None)
    
    bench_args = [
        "python3", cfg.benchmark.benchmark_script_path,
        "--request-rate", str(spec.request_rate),
        "--backend", "openai-chat",
        "--model", cfg.model.path,
        "--endpoint", "/v1/chat/completions",
        "--base-url", base_url,
        "--dataset-name", dataset_name,
        "--served-model-name", cfg.model.name,
        "--goodput", "tpot:300", "ttft:5000",
        "--temperature", "0",
        "--seed", "42",
        "--pattern-batch-size", str(cfg.benchmark.pattern_batch_size),
        "--benchmark-config", str(benchmark_config_path),
    ]
    
    # Add dataset-specific arguments
    if dataset_path:
        bench_args.extend(["--dataset-path", dataset_path])
    if dataset_name == "sharegpt" and sharegpt_output_len is not None:
        bench_args.extend(["--sharegpt-output-len", str(sharegpt_output_len)])
    
    if getattr(cfg.benchmark, "ignore_eos", False):
        bench_args.append("--ignore-eos")

    C.print(f"start to run benchmark with args: {bench_args}")

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
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
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
                        "attention_kernel": cfg.vllm.attention_kernel}
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
    sweep_config_index: int = 0  # Index of the sweep_config block this experiment belongs to


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
    # iter_from_sweep_test_config now returns (sweep_config_index, exp_cfg) tuples
    exp_configs_with_index = list(ExperimentConfig.iter_from_sweep_test_config(sweep_test_cfg))
    total = len(exp_configs_with_index)
    
    for i, (sweep_config_index, exp_cfg) in enumerate(exp_configs_with_index):
        # Get vars_mapping from ExperimentConfig
        vars_mapping = exp_cfg.get_naming_vars()
        
        # Generate log paths using logm
        metrics_csv_path = str(logm.get_path_with_log_type("timestamp_metrics_raw", "csv", vars_mapping))
        server_raw_log_path = str(logm.get_path_with_log_type("server_raw", "log", vars_mapping))
        benchmark_log_path = str(logm.get_path_with_log_type("benchmark", "log", vars_mapping))
        benchmark_config_path = str(logm.get_path_with_log_type("benchmark_config", "json", vars_mapping))
        metrics_file_path = str(logm.get_path_with_log_type("request_metrics", "csv", vars_mapping))
        print(f"chunk size: {exp_cfg.vllm.weight_chunk_size_mb} MB")
        print(f"max_num_batched_tokens: {exp_cfg.vllm.max_num_batched_tokens}")
        # Build VllmServerSpec from ExperimentConfig
        vllm_spec = VllmServerSpec(
            model_path=exp_cfg.model.path,
            model_name=exp_cfg.model.name,
            pipeline_parallel_size=exp_cfg.vllm.pipeline_parallel_size,
            gpu_memory_utilization=exp_cfg.vllm.gpu_memory_utilization,
            max_model_len=exp_cfg.vllm.max_model_len,
            max_num_batched_tokens=exp_cfg.vllm.max_num_batched_tokens,
            max_num_seqs=exp_cfg.vllm.max_num_seqs,
            block_size=exp_cfg.vllm.block_size,
            page_attention_block_size_bytes=exp_cfg.vllm.page_attention_block_size_bytes,
            head_addr=exp_cfg.vllm.head_addr,
            port=exp_cfg.vllm.port,
            attention_kernel=exp_cfg.vllm.attention_kernel,
            chunked_prefill=exp_cfg.vllm.chunked_prefill,
            enable_cuda_graph=exp_cfg.vllm.enable_cuda_graph,
            enable_nsight=exp_cfg.vllm.enable_nsight,
            weight_chunk_size_mb=exp_cfg.vllm.weight_chunk_size_mb,
            migration_approach=exp_cfg.vllm.migration_approach,
            weight_loading_mode=exp_cfg.vllm.weight_loading_mode,
            fixed_num_gpu_blocks=exp_cfg.vllm.fixed_num_gpu_blocks,
            use_vmm=exp_cfg.vllm.use_vmm,
            enable_kv_resize=exp_cfg.vllm.enable_kv_resize,
            log_kv_memory_stats=exp_cfg.vllm.log_kv_memory_stats,
            enable_cpu_weight_cache=exp_cfg.vllm.enable_cpu_weight_cache,
            dynamic_communication_enabled=exp_cfg.vllm.dynamic_communication_enabled,
            autoscaling_candidate_ranks=exp_cfg.vllm.autoscaling_candidate_ranks,
            autoscaling_sequence=exp_cfg.vllm.autoscaling_sequence,
            autoscaling_policy=exp_cfg.vllm.autoscaling_policy,
            pp_layer_partition=exp_cfg.vllm.pp_layer_partition,
            pp_layer_config=exp_cfg.vllm.pp_layer_config,
            alternative_configs=exp_cfg.vllm.get_alternative_configs(),
            migration_steps=exp_cfg.vllm.get_migration_steps(),
            rank_to_ip=exp_cfg.network.rank_to_ip,
            rank_to_node=exp_cfg.network.rank_to_node,
            pipeline_stage_to_rank=exp_cfg.network.pipeline_stage_to_rank,
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
            arrival_trace=exp_cfg.benchmark.arrival_trace,
            pattern_batch_size=exp_cfg.benchmark.pattern_batch_size,
            burstiness=exp_cfg.benchmark.burstiness,
            repetition=exp_cfg.benchmark.repetition,
            restart_server_between_repetitions=exp_cfg.benchmark.restart_server_between_repetitions,
            print_outputs=exp_cfg.benchmark.print_outputs,
            profile=exp_cfg.benchmark.profile,
            save_result=exp_cfg.benchmark.save_result,
            save_detailed=exp_cfg.benchmark.save_detailed,
            ignore_eos=exp_cfg.benchmark.ignore_eos,
            warmup=exp_cfg.benchmark.warmup,
            # Pipeline config for repetition reset
            initial_pp_config=exp_cfg.vllm.get_initial_pp_config(),
            alternative_configs=exp_cfg.vllm.get_alternative_configs(),
            migration_steps=exp_cfg.vllm.get_migration_steps(),
            migration_mode=exp_cfg.vllm.migration_approach,
            weight_loading_mode=exp_cfg.vllm.weight_loading_mode,
            # Dataset configuration
            dataset_name=exp_cfg.benchmark.dataset_name,
            dataset_path=exp_cfg.benchmark.dataset_path,
            sharegpt_output_len=exp_cfg.benchmark.sharegpt_output_len,
        )
        
        yield SweepExperimentSpec(
            vllm_spec=vllm_spec,
            bench_spec=bench_spec,
            exp_config=exp_cfg,
            vars_mapping=vars_mapping,
            experiment_index=i,
            total_experiments=total,
            sweep_config_index=sweep_config_index,
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
    _prepend_repo_to_pythonpath(env)
    
    # Set weight chunk size environment variable
    env["VLLM_WEIGHT_CHUNK_SIZE_MB"] = str(spec.weight_chunk_size_mb)
    _apply_page_attention_block_env(env, spec)
    
    serve_args = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", spec.model_path,
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
    
    _log_page_attention_block_geometry(spec)
    if spec.block_size:
        serve_args.extend(["--block-size", str(spec.block_size)])
    if spec.chunked_prefill:
        serve_args.append("--enable-chunked-prefill")
    if spec.max_num_batched_tokens is not None:
        serve_args.extend(["--max-num-batched-tokens", str(spec.max_num_batched_tokens)])
    if spec.max_num_seqs is not None:
        serve_args.extend(["--max-num-seqs", str(spec.max_num_seqs)])
    if not spec.enable_cuda_graph:
        serve_args.append("--enforce-eager")
    if spec.enable_nsight:
        serve_args.append("--ray-workers-use-nsight")
    if spec.rank_to_node:
        serve_args.extend(["--ray-rank-to-node", json.dumps(spec.rank_to_node)])
    if spec.pipeline_stage_to_rank:
        serve_args.extend([
            "--pipeline-stage-to-rank",
            json.dumps(spec.pipeline_stage_to_rank),
        ])
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
    C.print(f"[dim]serve_args: {' '.join(serve_args)}[/]")
    
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
    repetition_override: Optional[int] = None,
    benchmark_config_path_override: Optional[str] = None,
    benchmark_log_path_override: Optional[str] = None,
    metrics_file_path_override: Optional[str] = None,
) -> bool:
    """Start benchmark from spec.
    
    Args:
        spec: BenchmarkSpec with all parameters
        logm: Log manager for file paths
        vars: Variables for file naming
    
    Returns:
        True if benchmark succeeded
    """
    base_url = _normalize_loopback_base_url(spec.base_url)
    if not wait_ready(base_url, 300):
        C.print("[red]ERROR[/] vLLM not ready in time")
        return False
    
    # Get dataset configuration from spec
    dataset_name = getattr(spec, 'dataset_name', 'pattern')
    dataset_path = getattr(spec, 'dataset_path', None)
    sharegpt_output_len = getattr(spec, 'sharegpt_output_len', None)
    
    bench_args = [
        "python3", spec.benchmark_script_path,
        "--request-rate", "0",  # Use config-based rates
        "--backend", "openai-chat",
        "--model", spec.model_path,
        "--endpoint", "/v1/chat/completions",
        "--base-url", base_url,
        "--dataset-name", dataset_name,
        "--served-model-name", spec.model_name,
        "--goodput", "tpot:300", "ttft:5000",
        "--temperature", "0",
        "--seed", "42",
        "--pattern-batch-size", str(spec.pattern_batch_size),
        "--benchmark-config", str(
            spec.benchmark_config_path
            if benchmark_config_path_override is None else benchmark_config_path_override
        ),
    ]
    
    # Add dataset-specific arguments
    if dataset_path:
        bench_args.extend(["--dataset-path", dataset_path])
    if dataset_name == "sharegpt" and sharegpt_output_len is not None:
        bench_args.extend(["--sharegpt-output-len", str(sharegpt_output_len)])
    
    # Keep benchmark logs free of per-request generated text so the benchmark
    # driver has the same I/O behavior as the legacy baseline launcher.
    if spec.profile:
        bench_args.append("--profile")
    if spec.save_result or spec.save_detailed:
        bench_args.append("--save-result")
        bench_args.extend([
            "--result-dir",
            str(Path(spec.benchmark_log_path).parent),
        ])
    if spec.save_detailed:
        bench_args.append("--save-detailed")
    if spec.ignore_eos:
        bench_args.append("--ignore-eos")
    
    # Write benchmark config (includes metrics path) via spec helper
    spec.write_benchmark_config(
        benchmark_config_path=benchmark_config_path_override,
        repetition_override=repetition_override,
        metrics_file_path_override=metrics_file_path_override,
    )
    
    C.print(f"[bold cyan]Starting benchmark with {spec.num_total_requests} requests[/]")
    C.print(f"[bold cyan]Request rate config: {spec.request_rate}[/]")
    
    # Prepare metrics output path defined in spec
    metrics_file_path = Path(
        spec.metrics_file_path if metrics_file_path_override is None else metrics_file_path_override
    )
    if metrics_file_path.exists():
        os.remove(metrics_file_path)
    
    log_path = Path(
        spec.benchmark_log_path if benchmark_log_path_override is None else benchmark_log_path_override
    )
    if log_path.exists():
        os.remove(log_path)
    
    bench_fd = open(log_path, "wb", buffering=0)
    ret = subprocess.run(
        bench_args,
        stdout=bench_fd,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    bench_fd.close()
    
    # Validate benchmark results beyond just exit code
    if ret.returncode != 0:
        C.print(f"[red]ERROR[/] Benchmark exited with code {ret.returncode}")
        return False
    
    # Check if benchmark actually completed any requests successfully
    try:
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        # Look for "Successful requests:" line
        import re
        match = re.search(r'Successful requests:\s+(\d+)', log_content)
        if match:
            successful_requests = int(match.group(1))
            if successful_requests == 0:
                C.print("[red]ERROR[/] Benchmark completed 0 successful requests")
                return False
            C.print(f"[green]Benchmark completed with {successful_requests} successful requests[/]")
        else:
            C.print("[yellow]WARNING[/] Could not parse successful requests count from benchmark log")
            
        # Check for fatal benchmark errors without scanning generated text.
        # With --print-outputs, model output can legitimately contain phrases
        # like "Connection refused"; only explicit benchmark error lines count.
        fatal_connection_error = any(
            (
                line.startswith("ERROR")
                or ": ERROR:" in line
                or "aiohttp.client_exceptions" in line
                or line.startswith("Traceback")
            )
            and ("Cannot connect to host" in line or "Connection refused" in line)
            for line in log_content.splitlines()
        )
        if fatal_connection_error:
            C.print("[red]ERROR[/] Benchmark failed to connect to server")
            return False
            
    except Exception as e:
        C.print(f"[yellow]WARNING[/] Could not validate benchmark log: {e}")
    
    return True


def _wait_after_benchmark_if_requested() -> None:
    """Keep the server alive briefly after benchmark completion if requested."""
    raw_wait_s = os.environ.get("VLLM_EXP_POST_BENCHMARK_WAIT_S", "").strip()
    if not raw_wait_s:
        return

    try:
        wait_s = float(raw_wait_s)
    except ValueError:
        C.print(f"[yellow]WARNING:[/] Ignoring invalid VLLM_EXP_POST_BENCHMARK_WAIT_S={raw_wait_s!r}")
        return

    if wait_s <= 0:
        return

    C.print(f"[bold cyan]Waiting {wait_s:.1f}s after benchmark before stopping server[/]")
    time.sleep(wait_s)


def _append_text_file(src: Path, dst: Path, banner: Optional[str] = None) -> None:
    """Append a text file to another text file, optionally with a banner."""
    if not src.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "a", encoding="utf-8", errors="ignore") as dst_f:
        if banner is not None:
            dst_f.write(f"\n{'=' * 20} {banner} {'=' * 20}\n")
        with open(src, "r", encoding="utf-8", errors="ignore") as src_f:
            shutil.copyfileobj(src_f, dst_f)
        dst_f.write("\n")


def _extract_benchmark_result_summary(src: Path) -> str:
    """Extract the compact benchmark result section from a benchmark log."""
    if not src.exists():
        return ""

    content = src.read_text(encoding="utf-8", errors="ignore")
    match = re.search(
        r"------------ Benchmark Result \(Total\) ------------.*?--------------------------------------------------",
        content,
        re.DOTALL,
    )
    if match:
        return match.group(0).strip()
    return ""


@dataclass
class _BenchmarkAggregateMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    mean_tpot_ms: float
    mean_itl_ms: float
    mean_e2el_ms: Optional[float]
    mean_normalized_latency_ms: float


def _parse_benchmark_aggregate_metrics(src: Path) -> Optional[_BenchmarkAggregateMetrics]:
    """Parse key benchmark result numbers from one repetition log."""
    if not src.exists():
        return None

    content = src.read_text(encoding="utf-8", errors="ignore")

    patterns: dict[str, tuple[type, str]] = {
        "completed": (int, r"Successful requests:\s+(\d+)"),
        "total_input": (int, r"Total input tokens:\s+(\d+)"),
        "total_output": (int, r"Total generated tokens:\s+(\d+)"),
        "request_throughput": (float, r"Request throughput \(req/s\):\s+([0-9.]+)"),
        "output_throughput": (float, r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
        "total_token_throughput": (float, r"Total Token throughput \(tok/s\):\s+([0-9.]+)"),
        "mean_ttft_ms": (float, r"Mean TTFT \(ms\):\s+([0-9.]+)"),
        "mean_tpot_ms": (float, r"Mean TPOT \(ms\):\s+([0-9.]+)"),
        "mean_itl_ms": (float, r"Mean ITL \(ms\):\s+([0-9.]+)"),
        "mean_normalized_latency_ms": (float, r"Mean Normalized Latency \(ms/tok\):\s+([0-9.]+)"),
    }
    optional_patterns: dict[str, tuple[type, str]] = {
        "mean_e2el_ms": (float, r"Mean E2EL \(ms\):\s+([0-9.]+)"),
    }

    values: dict[str, Any] = {}
    for key, (cast, pattern) in patterns.items():
        match = re.search(pattern, content)
        if match is None:
            return None
        values[key] = cast(match.group(1))

    for key, (cast, pattern) in optional_patterns.items():
        match = re.search(pattern, content)
        values[key] = cast(match.group(1)) if match is not None else None

    return _BenchmarkAggregateMetrics(**values)


def _build_benchmark_final_summary(
    repetition: int,
    metrics_list: list[_BenchmarkAggregateMetrics],
) -> str:
    """Build final aggregate summary text matching benchmark_serving output style."""
    if not metrics_list:
        return ""

    throughputs = [m.request_throughput for m in metrics_list]
    output_throughputs = [m.output_throughput for m in metrics_list]
    total_token_throughputs = [m.total_token_throughput for m in metrics_list]
    mean_ttfts = [m.mean_ttft_ms for m in metrics_list]
    mean_tpots = [m.mean_tpot_ms for m in metrics_list]
    mean_itls = [m.mean_itl_ms for m in metrics_list]
    mean_e2els = [m.mean_e2el_ms for m in metrics_list if m.mean_e2el_ms is not None]
    mean_normalized_latencies = [m.mean_normalized_latency_ms for m in metrics_list]

    total_completed = sum(m.completed for m in metrics_list)
    total_input = sum(m.total_input for m in metrics_list)
    total_output = sum(m.total_output for m in metrics_list)

    def _mean(values: list[float]) -> float:
        return statistics.fmean(values)

    def _median(values: list[float]) -> float:
        return statistics.median(values)

    def _std(values: list[float]) -> float:
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    lines = [
        "",
        f"{'═' * 70}",
        f"{'═' * 70}",
        f"  FINAL SUMMARY: {repetition} REPETITIONS COMPLETED",
        f"{'═' * 70}",
        f"{'═' * 70}",
        "",
        f"┌{'─' * 68}┐",
        f"│ {'Rep':>4} │ {'Completed':>10} │ {'Throughput':>12} │ {'Mean TTFT':>12} │ {'Mean TPOT':>12} │",
        f"├{'─' * 68}┤",
    ]

    for idx, metric in enumerate(metrics_list, 1):
        lines.append(
            f"│ {idx:>4} │ {metric.completed:>10} │ {metric.request_throughput:>10.2f}/s │ {metric.mean_ttft_ms:>10.2f}ms │ {metric.mean_tpot_ms:>10.2f}ms │"
        )

    lines.extend([
        f"└{'─' * 68}┘",
        "",
        f"{'─' * 70}",
        f"  AGGREGATE STATISTICS ACROSS {repetition} REPETITIONS",
        f"{'─' * 70}",
        f"  Total Requests:     {total_completed}",
        f"  Total Input:        {total_input} tokens",
        f"  Total Output:       {total_output} tokens",
        "",
        "  REQUEST THROUGHPUT (req/s):",
        f"    Average:          {_mean(throughputs):.2f}",
        f"    Median:           {_median(throughputs):.2f}",
        f"    Std Dev:          {_std(throughputs):.2f}",
        f"    Min:              {min(throughputs):.2f}",
        f"    Max:              {max(throughputs):.2f}",
        "",
        "  OUTPUT THROUGHPUT (tokens/s):",
        f"    Average:          {_mean(output_throughputs):.2f}",
        f"    Median:           {_median(output_throughputs):.2f}",
        f"    Std Dev:          {_std(output_throughputs):.2f}",
        "",
        "  TOTAL TOKEN THROUGHPUT (tokens/s):",
        f"    Average:          {_mean(total_token_throughputs):.2f}",
        f"    Median:           {_median(total_token_throughputs):.2f}",
        f"    Std Dev:          {_std(total_token_throughputs):.2f}",
        "",
        "  MEAN TTFT ACROSS REPETITIONS (ms):",
        f"    Average:          {_mean(mean_ttfts):.2f}",
        f"    Median:           {_median(mean_ttfts):.2f}",
        f"    Std Dev:          {_std(mean_ttfts):.2f}",
        f"    Min:              {min(mean_ttfts):.2f}",
        f"    Max:              {max(mean_ttfts):.2f}",
        "",
        "  MEAN TPOT ACROSS REPETITIONS (ms):",
        f"    Average:          {_mean(mean_tpots):.2f}",
        f"    Median:           {_median(mean_tpots):.2f}",
        f"    Std Dev:          {_std(mean_tpots):.2f}",
        f"    Min:              {min(mean_tpots):.2f}",
        f"    Max:              {max(mean_tpots):.2f}",
        "",
        "  MEAN ITL ACROSS REPETITIONS (ms):",
        f"    Average:          {_mean(mean_itls):.2f}",
        f"    Median:           {_median(mean_itls):.2f}",
        f"    Std Dev:          {_std(mean_itls):.2f}",
        "",
        "  MEAN NORMALIZED LATENCY ACROSS REPETITIONS (ms/tok):",
        f"    Average:          {_mean(mean_normalized_latencies):.2f}",
        f"    Median:           {_median(mean_normalized_latencies):.2f}",
        f"    Std Dev:          {_std(mean_normalized_latencies):.2f}",
        f"    Min:              {min(mean_normalized_latencies):.2f}",
        f"    Max:              {max(mean_normalized_latencies):.2f}",
        "",
        f"{'═' * 70}",
    ])

    if mean_e2els:
        normalized_idx = lines.index("  MEAN NORMALIZED LATENCY ACROSS REPETITIONS (ms/tok):")
        lines[normalized_idx:normalized_idx] = [
            "  MEAN E2EL ACROSS REPETITIONS (ms):",
            f"    Average:          {_mean(mean_e2els):.2f}",
            f"    Median:           {_median(mean_e2els):.2f}",
            f"    Std Dev:          {_std(mean_e2els):.2f}",
            "",
        ]

    return "\n".join(lines)


def _append_benchmark_final_summary(
    dst: Path,
    repetition: int,
    metrics_list: list[_BenchmarkAggregateMetrics],
) -> None:
    """Append final aggregate statistics to benchmark.log."""
    summary = _build_benchmark_final_summary(repetition, metrics_list)
    if not summary:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "a", encoding="utf-8", errors="ignore") as dst_f:
        dst_f.write(summary)
        dst_f.write("\n")


def _append_csv_file(
    src: Path,
    dst: Path,
    row_transform: Optional[Callable[[list[str]], list[str]]] = None,
) -> None:
    """Append CSV rows from src to dst while preserving only one header."""
    if not src.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst_has_data = dst.exists() and dst.stat().st_size > 0

    with open(src, "r", newline="", encoding="utf-8", errors="ignore") as src_f:
        reader = csv.reader(src_f)
        rows = list(reader)

    if not rows:
        return

    with open(dst, "a", newline="", encoding="utf-8") as dst_f:
        writer = csv.writer(dst_f)
        for row_idx, row in enumerate(rows):
            if not row:
                continue
            is_header = row_idx == 0 and row[0] == "timestamp"
            if is_header and dst_has_data:
                continue
            out_row = row if is_header or row_transform is None else row_transform(row)
            writer.writerow(out_row)
            dst_has_data = True


def _build_request_metrics_row_transform(repetition_index: int) -> Callable[[list[str]], list[str]]:
    """Build a transformer that rewrites the repetition column."""
    def _transform(row: list[str]) -> list[str]:
        out_row = list(row)
        while len(out_row) < 6:
            out_row.append("")
        out_row[5] = str(repetition_index)
        return out_row

    return _transform


def _run_single_sweep_experiment_with_server_restart_between_repetitions(
    vllm_spec: VllmServerSpec,
    bench_spec: BenchmarkSpec,
    logm: SweepLogManager,
    vars_mapping: Dict[str, Any],
) -> bool:
    """Run one experiment by restarting the server between repetitions."""
    ok = True

    if not vllm_spec.server_raw_log_path or not vllm_spec.metrics_csv_path:
        raise ValueError("vllm_spec 缺少 server_raw_log_path 或 metrics_csv_path")

    aggregate_server_raw_path = Path(vllm_spec.server_raw_log_path)
    aggregate_timestamp_metrics_path = Path(vllm_spec.metrics_csv_path)
    benchmark_log_path = Path(bench_spec.benchmark_log_path)
    request_metrics_path = Path(bench_spec.metrics_file_path)
    benchmark_config_path = Path(bench_spec.benchmark_config_path)

    for final_path in (
        aggregate_server_raw_path,
        aggregate_timestamp_metrics_path,
        benchmark_log_path,
        request_metrics_path,
    ):
        if final_path.exists():
            os.remove(final_path)

    try:
        logm.write_constants_meta(vars_mapping)
        aggregate_metrics: list[_BenchmarkAggregateMetrics] = []

        for rep in range(1, bench_spec.repetition + 1):
            proc = None
            capture = None

            rep_server_raw_path = aggregate_server_raw_path.with_name(
                f"{aggregate_server_raw_path.name}.rep{rep}"
            )
            rep_timestamp_metrics_path = aggregate_timestamp_metrics_path.with_name(
                f"{aggregate_timestamp_metrics_path.name}.rep{rep}"
            )
            rep_benchmark_log_path = benchmark_log_path.with_name(
                f"{benchmark_log_path.name}.rep{rep}"
            )
            rep_request_metrics_path = request_metrics_path.with_name(
                f"{request_metrics_path.name}.rep{rep}"
            )
            rep_benchmark_config_path = benchmark_config_path.with_name(
                f"{benchmark_config_path.name}.rep{rep}"
            )

            for rep_path in (
                rep_server_raw_path,
                rep_timestamp_metrics_path,
                rep_benchmark_log_path,
                rep_request_metrics_path,
                rep_benchmark_config_path,
            ):
                if rep_path.exists():
                    os.remove(rep_path)

            rep_vllm_spec = replace(
                vllm_spec,
                server_raw_log_path=str(rep_server_raw_path),
                metrics_csv_path=str(rep_timestamp_metrics_path),
            )

            C.print(
                f"[bold cyan]Running repetition {rep}/{bench_spec.repetition} with server restart[/]"
            )

            try:
                proc, capture, _ = start_vllm_for_sweep(rep_vllm_spec)
                if not wait_ready_or_fail(bench_spec.base_url, proc, rep_server_raw_path):
                    return False

                ok = start_benchmark_for_sweep(
                    bench_spec,
                    repetition_override=1,
                    benchmark_config_path_override=str(rep_benchmark_config_path),
                    benchmark_log_path_override=str(rep_benchmark_log_path),
                    metrics_file_path_override=str(rep_request_metrics_path),
                )
                if not ok:
                    C.print("[yellow]WARNING:[/] Benchmark failed during repetition-restart mode")
                    return False
            finally:
                if proc:
                    stop_tree(proc)
                if capture:
                    capture.close()

            _append_text_file(
                rep_server_raw_path,
                aggregate_server_raw_path,
                banner=f"server repetition {rep}/{bench_spec.repetition}",
            )
            _append_text_file(
                rep_benchmark_log_path,
                benchmark_log_path,
                banner=f"benchmark repetition {rep}/{bench_spec.repetition}",
            )

            parsed_metric = _parse_benchmark_aggregate_metrics(rep_benchmark_log_path)
            if parsed_metric is None:
                C.print(f"[yellow]WARNING:[/] Failed to parse benchmark metrics from {rep_benchmark_log_path}")
            else:
                aggregate_metrics.append(parsed_metric)

            _append_csv_file(
                rep_request_metrics_path,
                request_metrics_path,
                row_transform=_build_request_metrics_row_transform(rep),
            )
            _append_csv_file(rep_timestamp_metrics_path, aggregate_timestamp_metrics_path)

            if rep_benchmark_log_path.exists():
                os.remove(rep_benchmark_log_path)

        if len(aggregate_metrics) == bench_spec.repetition:
            _append_benchmark_final_summary(
                benchmark_log_path,
                repetition=bench_spec.repetition,
                metrics_list=aggregate_metrics,
            )
        else:
            C.print("[yellow]WARNING:[/] Aggregate summary skipped because some repetition metrics could not be parsed")

        server_main_path = logm.get_path_with_log_type("server", "log", vars_mapping)
        if server_main_path.exists():
            os.remove(server_main_path)
        if aggregate_server_raw_path.exists():
            shutil.copyfile(aggregate_server_raw_path, server_main_path)

        ts_main_path = logm.get_path_with_log_type("timestamp_metrics", "csv", vars_mapping)
        if ts_main_path.exists():
            os.remove(ts_main_path)
        if aggregate_timestamp_metrics_path.exists():
            shutil.copyfile(aggregate_timestamp_metrics_path, ts_main_path)

    except Exception as e:
        C.print(f"[red]ERROR[/] {e}")
        import traceback
        traceback.print_exc()
        ok = False
    finally:
        time.sleep(3)
        generate_analysis_report(logm, vars_mapping)

    return ok


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
    if bench_spec.restart_server_between_repetitions and bench_spec.repetition > 1:
        return _run_single_sweep_experiment_with_server_restart_between_repetitions(
            vllm_spec,
            bench_spec,
            logm,
            vars_mapping,
        )

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
        
        _wait_after_benchmark_if_requested()

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
    
    If cfg.overwrite is False, experiments that have already completed
    successfully will be skipped.
    """
    all_ok = True
    experiment_count = 0
    skipped_count = 0
    
    for exp in generate_experiment_specs(cfg, logm):
        experiment_count += 1
        C.print(f"\n[bold magenta]{'='*60}[/]")
        C.print(f"[bold magenta]Experiment {exp.experiment_index + 1}/{exp.total_experiments}[/]")
        C.print(f"[bold magenta]vars: {exp.vars_mapping}[/]")
        C.print(f"[bold magenta]{'='*60}[/]\n")
        
        # Check if we should skip this experiment (overwrite=False mode)
        if not cfg.overwrite:
            is_completed, reason = check_experiment_completed_for_spec(
                logm, exp.vars_mapping, exp.bench_spec
            )
            if is_completed:
                C.print(f"[bold green]SKIPPING:[/] {reason}")
                skipped_count += 1
                continue
            else:
                C.print(f"[cyan]Running (not skipped): {reason}[/]")
        
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
    if skipped_count > 0:
        C.print(f"\n[bold cyan]Sweep test completed: {experiment_count} experiments ({skipped_count} skipped), {status}[/]")
    else:
        C.print(f"\n[bold cyan]Sweep test completed: {experiment_count} experiments, {status}[/]")


# =========================
# Single Server Mode Helper Functions
# =========================

def parse_pp_layer_partition(pp_partition_str: str) -> list:
    """Parse pp_layer_partition string to list of [start, end] pairs.
    
    Args:
        pp_partition_str: String like "32,32" meaning 32 layers on each rank.
    
    Returns:
        List of [start, end] pairs, e.g. [[0, 31], [32, 63]]
    """
    parts = [int(x.strip()) for x in pp_partition_str.split(",")]
    result = []
    current_start = 0
    for num_layers in parts:
        result.append([current_start, current_start + num_layers - 1])
        current_start += num_layers
    return result


async def call_set_pp_config(
    base_url: str,
    pp_layer_config: list,
    alternative_configs: Optional[Dict[int, Any]] = None,
    migration_steps: Optional[list[int]] = None,
    migration_mode: Optional[str] = None,
    weight_loading_mode: Optional[str] = None,
    weight_chunk_size_mb: Optional[float] = None,
    fixed_num_gpu_blocks: Optional[int] = None,
    timeout: float = 120.0
) -> bool:
    """Call the set_pp_config API endpoint.
    
    Args:
        base_url: Base URL of the vLLM server (e.g., http://localhost:8000)
        pp_layer_config: Target configuration as list of [start, end] pairs per rank.
        alternative_configs: Optional dict of migration targets. Keys are config indices,
                            values are pp_layer_config lists.
        migration_steps: Optional list of request indices at which to trigger migration.
        migration_mode: Optional migration mode ('sync', 'async', or 'async_fast').
        weight_loading_mode: Optional weight loading mode ('sync' or 'async').
        fixed_num_gpu_blocks: Optional fixed KV cache block count (-1 to disable).
        timeout: Request timeout in seconds
    
    Returns:
        True if successful, False otherwise
    """
    import aiohttp
    base_url = _normalize_loopback_base_url(base_url)
    url = f"{base_url}/set_pp_config"
    payload: Dict[str, Any] = {"pp_layer_config": pp_layer_config}
    
    # Add optional migration configuration
    if alternative_configs is not None:
        # Convert keys to string for JSON (dict keys must be strings)
        payload["alternative_configs"] = {str(k): v for k, v in alternative_configs.items()}
    if migration_steps is not None:
        payload["migration_steps"] = migration_steps
    if migration_mode is not None:
        payload["migration_mode"] = migration_mode
    if weight_loading_mode is not None:
        payload["weight_loading_mode"] = weight_loading_mode
    if weight_chunk_size_mb is not None:
        payload["weight_chunk_size_mb"] = weight_chunk_size_mb
    if fixed_num_gpu_blocks is not None:
        payload["fixed_num_gpu_blocks"] = fixed_num_gpu_blocks
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status == 200:
                    C.print(f"[green]PP config set successfully to {pp_layer_config}[/]")
                    if alternative_configs:
                        C.print(f"[green]  Migration configs: {alternative_configs}[/]")
                        C.print(f"[green]  Migration steps: {migration_steps}[/]")
                    else:
                        C.print(f"[green]  No migration configured for this experiment[/]")
                    return True
                else:
                    try:
                        error_body = await response.json()
                        error_msg = error_body.get('error', 'Unknown error')
                    except Exception:
                        error_msg = await response.text()
                    C.print(f"[red]set_pp_config failed with status {response.status}[/]")
                    C.print(f"[red]Error: {error_msg}[/]")
                    return False
    except Exception as e:
        C.print(f"[red]Failed to set pp config: {e}[/]")
        return False


def call_set_pp_config_sync(
    base_url: str,
    pp_layer_config: list,
    alternative_configs: Optional[Dict[int, Any]] = None,
    migration_steps: Optional[list[int]] = None,
    migration_mode: Optional[str] = None,
    weight_loading_mode: Optional[str] = None,
    weight_chunk_size_mb: Optional[float] = None,
    fixed_num_gpu_blocks: Optional[int] = None,
    timeout: float = 120.0
) -> bool:
    """Synchronous wrapper for call_set_pp_config.
    
    Args:
        base_url: Base URL of the vLLM server
        pp_layer_config: Target configuration as list of [start, end] pairs per rank.
        alternative_configs: Optional dict of migration targets.
        migration_steps: Optional list of request indices for migration triggers.
        migration_mode: Optional migration mode ('sync', 'async', or 'async_fast').
        weight_loading_mode: Optional weight loading mode ('sync' or 'async').
        fixed_num_gpu_blocks: Optional fixed KV cache block count (-1 to disable).
        timeout: Request timeout in seconds
    
    Returns:
        True if successful, False otherwise
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        call_set_pp_config(base_url, pp_layer_config, alternative_configs, migration_steps, migration_mode, weight_loading_mode, weight_chunk_size_mb, fixed_num_gpu_blocks, timeout)
    )


def start_vllm_single_instance(
    spec: VllmServerSpec,
    initial_log_path: Optional[Path] = None,
    global_log_path: Optional[Path] = None,
) -> tuple[subprocess.Popen, DynamicLogRedirector, Path]:
    """Start a single vLLM server instance with dynamic log redirection.
    
    This is used for single-server sweep_test mode where one server
    serves multiple experiments with different PP configurations.
    
    Args:
        spec: VllmServerSpec with all parameters including paths
        initial_log_path: Optional initial log file path for per-experiment logs.
        global_log_path: Optional path for a global log file that captures ALL output.
                         This is useful for real-time monitoring during experiments.
    
    Returns:
        (process, log_redirector, metrics_path)
    
    Raises:
        ValueError: If metrics_csv_path is not set in spec
    """
    if not spec.metrics_csv_path:
        raise ValueError("spec.metrics_csv_path must be set before calling start_vllm_single_instance")
    
    env = os.environ.copy()
    _prepend_repo_to_pythonpath(env)
    
    # Set weight chunk size environment variable
    env["VLLM_WEIGHT_CHUNK_SIZE_MB"] = str(spec.weight_chunk_size_mb)
    _apply_page_attention_block_env(env, spec)
    
    serve_args = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", spec.model_path,
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
    
    _log_page_attention_block_geometry(spec)
    if spec.block_size:
        serve_args.extend(["--block-size", str(spec.block_size)])
    if spec.chunked_prefill:
        serve_args.append("--enable-chunked-prefill")
    if spec.max_num_batched_tokens is not None:
        serve_args.extend(["--max-num-batched-tokens", str(spec.max_num_batched_tokens)])
    if spec.max_num_seqs is not None:
        serve_args.extend(["--max-num-seqs", str(spec.max_num_seqs)])
    if not spec.enable_cuda_graph:
        serve_args.append("--enforce-eager")
    if spec.enable_nsight:
        serve_args.append("--ray-workers-use-nsight")
    if spec.rank_to_node:
        serve_args.extend(["--ray-rank-to-node", json.dumps(spec.rank_to_node)])
    if spec.pipeline_stage_to_rank:
        serve_args.extend([
            "--pipeline-stage-to-rank",
            json.dumps(spec.pipeline_stage_to_rank),
        ])
    
    # Generate dynamic config from spec
    dynamic_cfg = json.dumps(spec.to_dynamic_cfg())
    serve_args.extend(["-D", dynamic_cfg])
    
    # Clear existing metrics file
    metrics_raw_path = Path(spec.metrics_csv_path)
    if metrics_raw_path.exists():
        os.remove(metrics_raw_path)
    
    # Clear existing global log file if provided
    if global_log_path and global_log_path.exists():
        os.remove(global_log_path)

    C.print(f"[bold cyan]Starting vLLM single instance with initial pp_layer_config: {spec.pp_layer_config}[/]")
    
    proc = subprocess.Popen(
        serve_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
        bufsize=0,
    )
    
    # Use DynamicLogRedirector with both per-experiment and global log
    log_redirector = DynamicLogRedirector(proc, initial_log_path, global_log_path)
    return proc, log_redirector, metrics_raw_path


def _run_single_experiment_on_running_server(
    vllm_spec: VllmServerSpec,
    bench_spec: BenchmarkSpec,
    logm: SweepLogManager,
    vars_mapping: Dict[str, Any],
    log_redirector: DynamicLogRedirector,
    is_first_experiment: bool = False,
    global_metrics_path: Optional[Path] = None,
) -> bool:
    """Run a single experiment on an already-running server.
    
    This is used in single-server mode where the server is already running.
    The function switches the PP config if needed and runs the benchmark.
    
    Args:
        vllm_spec: VllmServerSpec for this experiment
        bench_spec: BenchmarkSpec for running benchmark
        logm: Log manager
        vars_mapping: Variables for file naming
        log_redirector: DynamicLogRedirector for switching log files
        is_first_experiment: If True, skip set_pp_config (server already has correct config)
        global_metrics_path: Path to the global metrics CSV file (shared by all experiments)
    
    Returns:
        True if experiment succeeded
    """
    ok = False
    
    try:
        logm.write_constants_meta(vars_mapping)
        
        # Switch log file for this experiment
        assert vllm_spec.server_raw_log_path is not None
        server_raw_path = Path(vllm_spec.server_raw_log_path)
        log_redirector.switch_log_file(server_raw_path)
        
        # Always set PP config (even for first experiment) to update migration configuration
        # Parse pp_layer_partition to get target config (initial config for this experiment)
        target_pp_config = parse_pp_layer_partition(vllm_spec.pp_layer_partition)
        C.print(f"[bold cyan]Setting PP config to: {target_pp_config}[/]")
        
        # Build alternative_configs and migration_steps from vllm_spec
        # vllm_spec.alternative_configs format: {"pp_layer_configs": {"0": [[0,31],[32,63]], ...}}
        # API expects: {0: [[0,31],[32,63]], 1: [[0,19],[20,63]], ...}
        alternative_configs = None
        migration_steps = None
        
        if vllm_spec.alternative_configs and "pp_layer_configs" in vllm_spec.alternative_configs:
            inner = vllm_spec.alternative_configs["pp_layer_configs"]
            alternative_configs = {int(k): v for k, v in inner.items()}
            C.print(f"[dim]  Alternative configs: {alternative_configs}[/]")
        
        if vllm_spec.migration_steps:
            migration_steps = list(vllm_spec.migration_steps)
            C.print(f"[dim]  Migration steps: {migration_steps}[/]")
        else:
            C.print(f"[dim]  No migration for this experiment[/]")
        
        # Determine migration mode from spec
        migration_mode = getattr(vllm_spec, 'migration_approach', None)
        if migration_mode:
            C.print(f"[dim]  Migration mode: {migration_mode}[/]")

        weight_loading_mode = getattr(vllm_spec, 'weight_loading_mode', None)
        if weight_loading_mode:
            C.print(f"[dim]  Weight loading mode: {weight_loading_mode}[/]")
        
        # Pass weight_chunk_size_mb from spec
        exp_chunk_size = getattr(vllm_spec, 'weight_chunk_size_mb', None)
        if exp_chunk_size is not None:
            C.print(f"[dim]  Weight chunk size: {exp_chunk_size} MB[/]")
        
        # Pass fixed_num_gpu_blocks from spec
        exp_fixed_blocks = getattr(vllm_spec, 'fixed_num_gpu_blocks', None)
        if exp_fixed_blocks is not None:
            C.print(f"[dim]  Fixed num GPU blocks: {exp_fixed_blocks}[/]")
        
        if not call_set_pp_config_sync(
            bench_spec.base_url,
            target_pp_config,
            alternative_configs=alternative_configs,
            migration_steps=migration_steps,
            migration_mode=migration_mode,
            weight_loading_mode=weight_loading_mode,
            weight_chunk_size_mb=exp_chunk_size,
            fixed_num_gpu_blocks=exp_fixed_blocks
        ):
            C.print("[red]ERROR: Failed to set PP config[/]")
            return False
        
        # Wait a bit for config change to take effect
        time.sleep(1)
        
        # Run benchmark
        ok = start_benchmark_for_sweep(bench_spec)
        
        if not ok:
            C.print("[yellow]WARNING:[/] Benchmark failed")
        
        _wait_after_benchmark_if_requested()

        # Process logs - copy raw to main
        server_main_path = logm.get_path_with_log_type("server", "log", vars_mapping)
        if server_main_path.exists():
            os.remove(server_main_path)
        
        # Give log redirector time to flush
        time.sleep(0.5)
        if server_raw_path.exists():
            shutil.copyfile(server_raw_path, server_main_path)
        
        # Copy metrics from global metrics path
        # Extract only the metrics for this experiment's time range based on request_metrics timestamps
        if global_metrics_path and global_metrics_path.exists():
            ts_main_path = logm.get_path_with_log_type("timestamp_metrics", "csv", vars_mapping)
            if ts_main_path.exists():
                os.remove(ts_main_path)
            
            # Get time range from request_metrics
            request_metrics_path = Path(bench_spec.metrics_file_path)
            start_ts, end_ts = _get_time_range_from_request_metrics(request_metrics_path)
            
            if start_ts is not None and end_ts is not None:
                # Add some buffer (10 seconds before and after) to capture all relevant metrics
                buffer_seconds = 10.0
                extracted = _extract_timestamp_metrics_by_time_range(
                    global_metrics_path, ts_main_path,
                    start_ts - buffer_seconds, end_ts + buffer_seconds
                )
                if not extracted:
                    C.print(f"[yellow]WARNING: No timestamp metrics found for time range {start_ts:.2f} - {end_ts:.2f}[/]")
            else:
                # Fallback: copy the whole file if we can't determine time range
                C.print("[yellow]WARNING: Could not determine time range from request_metrics, copying full metrics[/]")
                shutil.copyfile(global_metrics_path, ts_main_path)
        
    except Exception as e:
        C.print(f"[red]ERROR[/] {e}")
        import traceback
        traceback.print_exc()
        ok = False
    finally:
        time.sleep(1)
        generate_analysis_report(logm, vars_mapping)
    
    return ok


def sweep_test_single_server(cfg: SweepTestConfig, logm: SweepLogManager):
    """Run sweep test with a single server instance per sweep_config block.
    
    Unlike sweep_test which starts/stops the server for each experiment,
    this function starts the server once per sweep_config block (each - vllm:
    entry in sweep_configs list) and switches PP configurations between 
    experiments within that block using the set_pp_config API.
    
    IMPORTANT: Different sweep_config blocks (different - vllm: entries in 
    sweep_configs) will cause a server restart to ensure complete isolation
    between different experiment configurations. Within the same
    sweep_config block, parameters that cannot be updated through the
    runtime API (for example `enable_kv_resize`) also force a restart.
    
    This provides:
    1. Complete isolation between different sweep_config blocks (server restart)
    2. Faster experiment iteration within the same sweep_config block
    3. Separate log files for each experiment via DynamicLogRedirector
    4. Dynamic PP config switching via set_pp_config API
    
    If cfg.overwrite is False, experiments that have already completed
    successfully will be skipped.
    
    Args:
        cfg: SweepTestConfig
        logm: SweepLogManager for log file management
    """
    all_ok = True
    experiment_count = 0
    skipped_count = 0
    
    try:
        # Pre-generate all experiment specs
        all_experiments = list(generate_experiment_specs(cfg, logm))
        
        if not all_experiments:
            C.print("[red]ERROR: No valid experiments generated (benchmark_config required)[/]")
            return
        
        # Pre-check which experiments need to run (for overwrite=False mode)
        experiments_to_run = []
        if not cfg.overwrite:
            C.print("[bold cyan]Checking which experiments need to run (overwrite=False)...[/]")
            for exp in all_experiments:
                is_completed, reason = check_experiment_completed_for_spec(
                    logm, exp.vars_mapping, exp.bench_spec
                )
                if is_completed:
                    C.print(f"  [green]SKIP:[/] {exp.vars_mapping} - {reason}")
                    skipped_count += 1
                else:
                    C.print(f"  [cyan]NEED:[/] {exp.vars_mapping} - {reason}")
                    experiments_to_run.append(exp)
            
            if not experiments_to_run:
                C.print(f"\n[bold green]All {len(all_experiments)} experiments already completed, nothing to run![/]")
                return
            
            C.print(f"\n[bold cyan]Will run {len(experiments_to_run)} experiments (skipping {skipped_count} completed)[/]")
        else:
            experiments_to_run = all_experiments
        
        restart_between_repetition_experiments = [
            exp for exp in experiments_to_run
            if exp.bench_spec.restart_server_between_repetitions and exp.bench_spec.repetition > 1
        ]
        grouped_experiments_to_run = [
            exp for exp in experiments_to_run
            if not (exp.bench_spec.restart_server_between_repetitions and exp.bench_spec.repetition > 1)
        ]

        C.print(f"[bold cyan]Starting single-server sweep test with {len(experiments_to_run)} experiments[/]")
        if restart_between_repetition_experiments:
            C.print(
                "[bold yellow]"
                f"{len(restart_between_repetition_experiments)} experiments require server restart between repetitions; "
                "they will run in standalone mode.[/]"
            )
        
        # Group experiments by parameters that require server restart:
        # - sweep_config_index (each - vllm: block gets its own server)
        # - attention_kernel (switching kernel requires restart)
        # - block_size (changing KV cache block size requires restart)
        # - migration_approach (user requires different migration modes to run on separate servers)
        # - enable_cpu_weight_cache (changing CPU weight cache mode requires restart)
        # - enable_kv_resize (not switchable through set_pp_config API)
        
        def get_server_group_key(exp: SweepExperimentSpec) -> tuple:
            """Return a tuple of parameters that require server restart when changed."""
            return (
                exp.sweep_config_index,
                exp.vllm_spec.attention_kernel,
                exp.vllm_spec.block_size,
                exp.vllm_spec.migration_approach,
                exp.vllm_spec.use_vmm,
                exp.vllm_spec.enable_cpu_weight_cache,
                exp.vllm_spec.enable_kv_resize,
            )
        
        # Group experiments by server restart parameters while preserving first-seen order.
        # This is more robust than itertools.groupby because experiments with the same
        # restart key may not always be adjacent in the sweep order.
        server_groups_map: dict[tuple, list[SweepExperimentSpec]] = {}
        for exp in grouped_experiments_to_run:
            group_key = get_server_group_key(exp)
            server_groups_map.setdefault(group_key, []).append(exp)
        server_groups = list(server_groups_map.items())
        
        C.print(f"[bold cyan]Experiments grouped by server config: {len(server_groups)} groups[/]")
        for group_key, group_exps in server_groups:
            sweep_idx, kernel, blk_size, migration_mode, use_vmm, cpu_cache, kv_resize = group_key
            C.print(
                f"  sweep[{sweep_idx}] kernel={kernel} block_size={blk_size} "
                f"migration_mode={migration_mode} vmm={use_vmm} cpu_cache={cpu_cache} "
                f"kv_resize={kv_resize}: "
                f"{len(group_exps)} experiments"
            )
        
        # Run experiments group by group, restarting server for each group
        for group_key, group_experiments in server_groups:
            sweep_idx, kernel_val, block_size_val, migration_mode_val, use_vmm_val, cpu_cache_val, kv_resize_val = group_key
            proc = None
            log_redirector = None
            
            try:
                C.print(f"\n[bold blue]{'='*60}[/]")
                C.print(
                    f"[bold blue]Starting server for sweep[{sweep_idx}] kernel={kernel_val} "
                    f"block_size={block_size_val} migration_mode={migration_mode_val} "
                    f"vmm={use_vmm_val} cpu_cache={cpu_cache_val} kv_resize={kv_resize_val}[/]"
                )
                C.print(f"[bold blue]This group has {len(group_experiments)} experiments[/]")
                C.print(f"[bold blue]{'='*60}[/]\n")
                
                # Get the first experiment in this group to start the server
                first_exp = group_experiments[0]
                
                # Create a global metrics path for the server (per server group)
                blk_str = f"_blk{block_size_val}" if block_size_val else ""
                mig_mode_str = f"_mig{migration_mode_val}"
                vmm_str = f"_vmm{1 if use_vmm_val else 0}"
                cpu_str = f"_cpu{1 if cpu_cache_val else 0}"
                kv_resize_str = f"_kvresize{1 if kv_resize_val else 0}"
                group_suffix = f"sweep{sweep_idx}_{kernel_val}{blk_str}{mig_mode_str}{vmm_str}{cpu_str}{kv_resize_str}"
                global_metrics_path = logm.get_dir() / f"global_metrics_raw_{group_suffix}.csv"
                first_exp.vllm_spec.metrics_csv_path = str(global_metrics_path)
                
                # Create initial log path (per-experiment)
                initial_log_path = Path(first_exp.vllm_spec.server_raw_log_path) if first_exp.vllm_spec.server_raw_log_path else None
                
                # Create global log path for this server group
                global_log_path = logm.get_dir() / f"server_global_{group_suffix}.log"
                
                # Start the server for this group
                proc, log_redirector, metrics_path = start_vllm_single_instance(
                    first_exp.vllm_spec,
                    initial_log_path=initial_log_path,
                    global_log_path=global_log_path
                )
                
                # Wait for server to be ready
                C.print(f"[bold cyan]Waiting for server to be ready (sweep[{sweep_idx}], kernel={kernel_val}, block_size={block_size_val}, vmm={use_vmm_val}, kv_resize={kv_resize_val})...[/]")
                C.print(f"[bold cyan]Monitor server status: tail -f {global_log_path}[/]")
                if not wait_ready(
                        first_exp.bench_spec.base_url,
                        _server_ready_timeout(300),
                        startup_log_path=global_log_path):
                    C.print(f"[red]ERROR: vLLM server not ready in time (sweep[{sweep_idx}])[/]")
                    continue
                
                C.print(f"[green]vLLM server is ready (sweep[{sweep_idx}], kernel={kernel_val}, block_size={block_size_val}, vmm={use_vmm_val}, kv_resize={kv_resize_val})[/]")
                
                # Run each experiment in this server group
                for i, exp in enumerate(group_experiments):
                    experiment_count += 1
                    is_first = (i == 0)
                    
                    C.print(f"\n[bold magenta]{'='*60}[/]")
                    C.print(f"[bold magenta]Experiment {experiment_count}/{len(experiments_to_run)} (sweep[{sweep_idx}], kernel={kernel_val}, blk={block_size_val}, vmm={use_vmm_val}, kv_resize={kv_resize_val})[/]")
                    C.print(f"[bold magenta]vars: {exp.vars_mapping}[/]")
                    C.print(f"[bold magenta]PP partition: {exp.vllm_spec.pp_layer_partition}[/]")
                    C.print(f"[bold magenta]{'='*60}[/]\n")
                    
                    ok = _run_single_experiment_on_running_server(
                        exp.vllm_spec,
                        exp.bench_spec,
                        logm,
                        exp.vars_mapping,
                        log_redirector,
                        is_first_experiment=is_first,
                        global_metrics_path=global_metrics_path,
                    )
                    
                    if not ok:
                        all_ok = False
                        C.print(f"[yellow]Experiment {experiment_count}/{len(experiments_to_run)} failed[/]")
                        # Continue with next experiment instead of stopping
                        continue
                
            finally:
                # Clean up server for this sweep_config group before starting next group
                if proc:
                    C.print(f"[bold cyan]Stopping vLLM server (sweep_config[{sweep_idx}])...[/]")
                    stop_tree(proc)
                if log_redirector:
                    log_redirector.close()
                time.sleep(3)

        for exp in restart_between_repetition_experiments:
            experiment_count += 1
            C.print(f"\n[bold magenta]{'='*60}[/]")
            C.print(
                f"[bold magenta]Standalone repetition-restart experiment {experiment_count}/{len(experiments_to_run)}[/]"
            )
            C.print(f"[bold magenta]vars: {exp.vars_mapping}[/]")
            C.print(f"[bold magenta]{'='*60}[/]\n")

            ok = _run_single_sweep_experiment(
                exp.vllm_spec,
                exp.bench_spec,
                logm,
                exp.vars_mapping,
            )

            if not ok:
                all_ok = False
                C.print(
                    f"[yellow]Standalone repetition-restart experiment {experiment_count}/{len(experiments_to_run)} failed[/]"
                )
        
        status = "All succeeded" if all_ok else "Some failed"
        if skipped_count > 0:
            C.print(f"\n[bold cyan]Single-server sweep test completed: {experiment_count} run, {skipped_count} skipped, {status}[/]")
        else:
            C.print(f"\n[bold cyan]Single-server sweep test completed: {experiment_count} experiments, {status}[/]")
        
    except Exception as e:
        C.print(f"[red]ERROR in sweep_test_single_server: {e}[/]")
        import traceback
        traceback.print_exc()

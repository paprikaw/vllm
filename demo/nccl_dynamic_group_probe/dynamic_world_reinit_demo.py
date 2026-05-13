#!/usr/bin/env python3
"""Launch a new GPU process at runtime and rebuild the distributed world.

This demo intentionally does not pre-launch the added GPU rank. The initial
world is started with N old ranks. Rank 0 later spawns a fresh Python process
on an unused GPU. At a safe point, the old ranks destroy the old default
process group and all ranks initialize a new default process group with
world_size=N+1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class RuntimeState:
    version: int
    world_name: str
    world_size: int
    rank: int
    local_rank: int
    logical_stage: int
    state_numel: int


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("old-rank", "new-worker"),
                        default="old-rank")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spawn-sec", type=float, default=2.0)
    parser.add_argument("--switch-sec", type=float, default=6.0)
    parser.add_argument("--total-sec", type=float, default=12.0)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--compute-repeats", type=int, default=1)
    parser.add_argument("--collective-mb", type=int, default=4)
    parser.add_argument("--state-mb", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="float16")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--new-master-addr", default="127.0.0.1")
    parser.add_argument("--new-master-port", type=int, default=29631)
    parser.add_argument("--new-rdzv-backend", choices=("file", "tcp"),
                        default="file")
    parser.add_argument("--new-world-size", type=int, default=-1)
    parser.add_argument("--new-rank", type=int, default=-1)
    parser.add_argument("--new-local-rank", type=int, default=-1)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def tensor_numel_for_mb(mb: int, dtype: torch.dtype) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return max(1, mb * 1024 * 1024 // element_size)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None,
                "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def allocate_state(
    *,
    state: RuntimeState,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.cuda.set_device(state.local_rank)
    n = max(256, int(math.sqrt(state.state_numel)))
    marker = torch.full((state.state_numel,), float(state.version),
                        device="cuda", dtype=dtype)
    x = torch.randn((n, n), device="cuda", dtype=dtype)
    w = torch.randn((n, n), device="cuda", dtype=dtype)
    return marker, x, w


def run_iteration(
    *,
    state: RuntimeState,
    x: torch.Tensor,
    w: torch.Tensor,
    comm: torch.Tensor,
    compute_repeats: int,
    phase: str,
    iteration: int,
    t0: float,
    writer: JsonlWriter,
) -> torch.Tensor:
    wall_start = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(state.local_rank)
    start_event.record(stream)
    for _ in range(compute_repeats):
        x = torch.mm(x, w)
        x = x * 0.999 + 0.001
    end_event.record(stream)
    end_event.synchronize()

    collective_start = time.perf_counter()
    dist.all_reduce(comm)
    torch.cuda.synchronize(state.local_rank)
    wall_end = time.perf_counter()

    writer.write({
        "type": "iteration",
        "phase": phase,
        "iteration": iteration,
        "rank": state.rank,
        "local_rank": state.local_rank,
        "state": asdict(state),
        "rel_start_sec": wall_start - t0,
        "rel_end_sec": wall_end - t0,
        "wall_ms": (wall_end - wall_start) * 1000.0,
        "cuda_compute_ms": start_event.elapsed_time(end_event),
        "collective_ms": (wall_end - collective_start) * 1000.0,
    })
    return x


def init_nccl_world(
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    init_method: str | None,
    timeout: timedelta,
) -> float:
    torch.cuda.set_device(local_rank)
    start = time.perf_counter()
    kwargs: dict[str, Any] = {
        "backend": "nccl",
        "rank": rank,
        "world_size": world_size,
        "timeout": timeout,
    }
    if init_method is not None:
        kwargs["init_method"] = init_method
    try:
        dist.init_process_group(
            **kwargs,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    except TypeError:
        dist.init_process_group(**kwargs)
    return time.perf_counter() - start


def spawn_new_worker(args: argparse.Namespace, old_world_size: int) -> subprocess.Popen:
    new_rank = old_world_size
    new_world_size = old_world_size + 1
    new_local_rank = old_world_size
    log_path = args.out_dir / "spawned_new_rank.log"
    env = os.environ.copy()
    for key in (
        "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
        "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_RUN_ID",
        "MASTER_ADDR", "MASTER_PORT",
    ):
        env.pop(key, None)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode", "new-worker",
        "--out-dir", str(args.out_dir),
        "--matrix-size", str(args.matrix_size),
        "--compute-repeats", str(args.compute_repeats),
        "--collective-mb", str(args.collective_mb),
        "--state-mb", str(args.state_mb),
        "--dtype", args.dtype,
        "--timeout-sec", str(args.timeout_sec),
        "--total-sec", str(args.total_sec),
        "--new-master-addr", args.new_master_addr,
        "--new-master-port", str(args.new_master_port),
        "--new-rdzv-backend", args.new_rdzv_backend,
        "--new-world-size", str(new_world_size),
        "--new-rank", str(new_rank),
        "--new-local-rank", str(new_local_rank),
    ]
    log_fh = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(command, env=env, stdout=log_fh,
                            stderr=subprocess.STDOUT)


def sentinel_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / "new_worker_ready.json", out_dir / "new_world_go"


def new_world_init_method(args: argparse.Namespace) -> str:
    if args.new_rdzv_backend == "tcp":
        return f"tcp://{args.new_master_addr}:{args.new_master_port}"
    path = (args.out_dir / "new_world_rendezvous").resolve()
    return f"file://{path}"


def summarize(out_dir: Path) -> None:
    events: dict[str, list[dict[str, Any]]] = {}
    iterations: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("rank_*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                item = json.loads(line)
                if item.get("type") == "iteration":
                    iterations.append(item)
                else:
                    events.setdefault(item.get("type", "unknown"), []).append(item)

    def max_value(event_type: str, key: str) -> float | None:
        values = [item.get(key) for item in events.get(event_type, [])]
        values = [value for value in values if value is not None]
        return max(values) if values else None

    per_rank: dict[str, dict[str, dict[str, Any]]] = {}
    for item in iterations:
        by_phase = per_rank.setdefault(str(item["rank"]), {})
        metrics = by_phase.setdefault(item["phase"], {
            "wall_ms": [],
            "collective_ms": [],
            "cuda_compute_ms": [],
        })
        metrics["wall_ms"].append(item["wall_ms"])
        metrics["collective_ms"].append(item["collective_ms"])
        metrics["cuda_compute_ms"].append(item["cuda_compute_ms"])

    summary = {
        "events": {
            "old_world_init_sec": max_value("old_world_initialized",
                                            "duration_sec"),
            "new_process_spawn_to_ready_sec": max_value(
                "new_worker_ready", "spawn_to_ready_sec"),
            "safe_point_pause_sec": max_value("safe_point_exit", "pause_sec"),
            "new_world_init_sec": max_value("new_world_initialized",
                                            "duration_sec"),
            "new_world_first_collective_sec": max_value(
                "new_world_first_collective", "duration_sec"),
            "new_state_install_sec": max_value("new_state_installed",
                                               "duration_sec"),
        },
        "iteration_stats": {
            rank: {
                phase: {
                    metric: stats(values)
                    for metric, values in metrics.items()
                }
                for phase, metrics in phases.items()
            }
            for rank, phases in per_rank.items()
        },
        "event_counts": {key: len(value) for key, value in events.items()},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(json.dumps(summary["events"], indent=2, sort_keys=True), flush=True)
    for rank, phases in sorted(summary["iteration_stats"].items(),
                               key=lambda item: int(item[0])):
        old_wall = phases.get("old", {}).get("wall_ms", {})
        new_wall = phases.get("new", {}).get("wall_ms", {})
        print(
            f"rank {rank}: old_count={old_wall.get('count')} "
            f"new_count={new_wall.get('count')} "
            f"old_median={old_wall.get('median')} "
            f"new_median={new_wall.get('median')}",
            flush=True,
        )


def old_rank_main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ready_path, go_path = sentinel_paths(args.out_dir)
    if int(os.environ.get("RANK", "0")) == 0:
        ready_path.unlink(missing_ok=True)
        go_path.unlink(missing_ok=True)
        (args.out_dir / "new_world_rendezvous").unlink(missing_ok=True)

    old_rank = int(os.environ["RANK"])
    old_world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    timeout = timedelta(seconds=args.timeout_sec)
    writer = JsonlWriter(args.out_dir / f"rank_{old_rank}.jsonl")
    dtype = dtype_from_name(args.dtype)

    old_init_sec = init_nccl_world(
        rank=old_rank,
        world_size=old_world_size,
        local_rank=local_rank,
        init_method=None,
        timeout=timeout,
    )
    writer.write({
        "type": "old_world_initialized",
        "rank": old_rank,
        "world_size": old_world_size,
        "local_rank": local_rank,
        "hostname": os.uname().nodename,
        "device": torch.cuda.get_device_name(local_rank),
        "duration_sec": old_init_sec,
        "args": vars(args) | {"out_dir": str(args.out_dir)},
    })

    state_numel = tensor_numel_for_mb(args.state_mb, dtype)
    state = RuntimeState(
        version=0,
        world_name="old",
        world_size=old_world_size,
        rank=old_rank,
        local_rank=local_rank,
        logical_stage=old_rank,
        state_numel=state_numel,
    )
    marker, x, w = allocate_state(state=state, dtype=dtype)
    comm = torch.ones(tensor_numel_for_mb(args.collective_mb, dtype),
                      device="cuda", dtype=dtype)
    continue_flag = torch.ones(1, device="cuda", dtype=torch.int32)
    writer.write({"type": "old_state_installed", "rank": old_rank,
                  "state": asdict(state), "checksum": float(marker[0])})

    dist.barrier()
    t0 = time.perf_counter()
    child: subprocess.Popen | None = None
    spawn_rel_sec: float | None = None
    iteration = 0
    while True:
        if old_rank == 0 and child is None and time.perf_counter() - t0 >= args.spawn_sec:
            spawn_rel_sec = time.perf_counter() - t0
            child = spawn_new_worker(args, old_world_size)
            writer.write({
                "type": "new_worker_spawned",
                "rank": old_rank,
                "rel_sec": spawn_rel_sec,
                "pid": child.pid,
            })

        should_switch = time.perf_counter() - t0 >= args.switch_sec
        continue_flag.fill_(0 if should_switch else 1)
        dist.all_reduce(continue_flag, op=dist.ReduceOp.MIN)
        if int(continue_flag.item()) == 0:
            break

        x = run_iteration(
            state=state,
            x=x,
            w=w,
            comm=comm,
            compute_repeats=args.compute_repeats,
            phase="old",
            iteration=iteration,
            t0=t0,
            writer=writer,
        )
        iteration += 1

    safe_enter = time.perf_counter()
    writer.write({
        "type": "safe_point_enter",
        "rank": old_rank,
        "rel_sec": safe_enter - t0,
        "old_state": asdict(state),
    })
    dist.barrier()
    dist.destroy_process_group()
    writer.write({
        "type": "old_world_destroyed",
        "rank": old_rank,
        "rel_sec": time.perf_counter() - t0,
    })

    if old_rank == 0:
        go_path.write_text(json.dumps({
            "rel_sec": time.perf_counter() - t0,
            "new_world_size": old_world_size + 1,
            "new_master_addr": args.new_master_addr,
            "new_master_port": args.new_master_port,
            "new_rdzv_backend": args.new_rdzv_backend,
        }), encoding="utf-8")
        writer.write({
            "type": "new_world_go_written",
            "rank": old_rank,
            "rel_sec": time.perf_counter() - t0,
        })

    new_world_size = old_world_size + 1
    new_init_method = new_world_init_method(args)
    new_init_start = time.perf_counter()
    new_init_sec = init_nccl_world(
        rank=old_rank,
        world_size=new_world_size,
        local_rank=local_rank,
        init_method=new_init_method,
        timeout=timeout,
    )
    writer.write({
        "type": "new_world_initialized",
        "rank": old_rank,
        "world_size": new_world_size,
        "rel_sec": time.perf_counter() - t0,
        "duration_sec": new_init_sec,
        "from_safe_enter_sec": time.perf_counter() - safe_enter,
        "init_started_rel_sec": new_init_start - t0,
    })

    first_collective_start = time.perf_counter()
    probe = torch.full((1,), float(old_rank + 1), device="cuda", dtype=dtype)
    dist.all_reduce(probe)
    torch.cuda.synchronize(local_rank)
    writer.write({
        "type": "new_world_first_collective",
        "rank": old_rank,
        "duration_sec": time.perf_counter() - first_collective_start,
        "value": float(probe.item()),
    })

    state_install_start = time.perf_counter()
    state = RuntimeState(
        version=1,
        world_name="new",
        world_size=new_world_size,
        rank=old_rank,
        local_rank=local_rank,
        logical_stage=old_rank,
        state_numel=state_numel,
    )
    marker, x, w = allocate_state(state=state, dtype=dtype)
    writer.write({
        "type": "new_state_installed",
        "rank": old_rank,
        "state": asdict(state),
        "duration_sec": time.perf_counter() - state_install_start,
        "checksum": float(marker[0]),
    })
    writer.write({
        "type": "safe_point_exit",
        "rank": old_rank,
        "rel_sec": time.perf_counter() - t0,
        "pause_sec": time.perf_counter() - safe_enter,
    })

    while True:
        continue_flag.fill_(1 if time.perf_counter() - t0 < args.total_sec else 0)
        dist.all_reduce(continue_flag, op=dist.ReduceOp.MIN)
        if int(continue_flag.item()) == 0:
            break
        x = run_iteration(
            state=state,
            x=x,
            w=w,
            comm=comm,
            compute_repeats=args.compute_repeats,
            phase="new",
            iteration=iteration,
            t0=t0,
            writer=writer,
        )
        iteration += 1

    writer.write({"type": "rank_finished", "rank": old_rank,
                  "rel_sec": time.perf_counter() - t0})
    dist.barrier()
    dist.destroy_process_group()
    writer.close()

    if old_rank == 0:
        if child is not None:
            child.wait(timeout=args.timeout_sec)
        summarize(args.out_dir)


def new_worker_main(args: argparse.Namespace) -> None:
    if args.new_rank < 0 or args.new_local_rank < 0 or args.new_world_size <= 0:
        raise ValueError("new-worker mode requires new rank/world/local rank")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ready_path, go_path = sentinel_paths(args.out_dir)
    writer = JsonlWriter(args.out_dir / f"rank_{args.new_rank}.jsonl")
    dtype = dtype_from_name(args.dtype)
    timeout = timedelta(seconds=args.timeout_sec)
    spawn_start = time.perf_counter()

    torch.cuda.set_device(args.new_local_rank)
    state_numel = tensor_numel_for_mb(args.state_mb, dtype)
    pre_state = RuntimeState(
        version=0,
        world_name="prewarm",
        world_size=args.new_world_size,
        rank=args.new_rank,
        local_rank=args.new_local_rank,
        logical_stage=args.new_rank,
        state_numel=state_numel,
    )
    marker, _, _ = allocate_state(state=pre_state, dtype=dtype)
    torch.cuda.synchronize(args.new_local_rank)
    ready_payload = {
        "type": "new_worker_ready",
        "rank": args.new_rank,
        "local_rank": args.new_local_rank,
        "hostname": os.uname().nodename,
        "device": torch.cuda.get_device_name(args.new_local_rank),
        "spawn_to_ready_sec": time.perf_counter() - spawn_start,
        "checksum": float(marker[0]),
    }
    ready_path.write_text(json.dumps(ready_payload), encoding="utf-8")
    writer.write(ready_payload)

    while not go_path.exists():
        time.sleep(0.001)
    t0 = time.perf_counter()
    writer.write({
        "type": "new_worker_go_seen",
        "rank": args.new_rank,
        "go_payload": json.loads(go_path.read_text(encoding="utf-8")),
    })

    init_method = new_world_init_method(args)
    new_init_sec = init_nccl_world(
        rank=args.new_rank,
        world_size=args.new_world_size,
        local_rank=args.new_local_rank,
        init_method=init_method,
        timeout=timeout,
    )
    writer.write({
        "type": "new_world_initialized",
        "rank": args.new_rank,
        "world_size": args.new_world_size,
        "duration_sec": new_init_sec,
    })

    first_collective_start = time.perf_counter()
    probe = torch.full((1,), float(args.new_rank + 1),
                       device="cuda", dtype=dtype)
    dist.all_reduce(probe)
    torch.cuda.synchronize(args.new_local_rank)
    writer.write({
        "type": "new_world_first_collective",
        "rank": args.new_rank,
        "duration_sec": time.perf_counter() - first_collective_start,
        "value": float(probe.item()),
    })

    state_install_start = time.perf_counter()
    state = RuntimeState(
        version=1,
        world_name="new",
        world_size=args.new_world_size,
        rank=args.new_rank,
        local_rank=args.new_local_rank,
        logical_stage=args.new_rank,
        state_numel=state_numel,
    )
    marker, x, w = allocate_state(state=state, dtype=dtype)
    writer.write({
        "type": "new_state_installed",
        "rank": args.new_rank,
        "state": asdict(state),
        "duration_sec": time.perf_counter() - state_install_start,
        "checksum": float(marker[0]),
    })

    comm = torch.ones(tensor_numel_for_mb(args.collective_mb, dtype),
                      device="cuda", dtype=dtype)
    continue_flag = torch.ones(1, device="cuda", dtype=torch.int32)
    iteration = 0
    while True:
        continue_flag.fill_(1)
        dist.all_reduce(continue_flag, op=dist.ReduceOp.MIN)
        if int(continue_flag.item()) == 0:
            break
        x = run_iteration(
            state=state,
            x=x,
            w=w,
            comm=comm,
            compute_repeats=args.compute_repeats,
            phase="new",
            iteration=iteration,
            t0=t0,
            writer=writer,
        )
        iteration += 1

    writer.write({"type": "rank_finished", "rank": args.new_rank,
                  "rel_sec": time.perf_counter() - t0})
    dist.barrier()
    dist.destroy_process_group()
    writer.close()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.mode == "old-rank":
        old_rank_main(args)
    else:
        new_worker_main(args)


if __name__ == "__main__":
    main()

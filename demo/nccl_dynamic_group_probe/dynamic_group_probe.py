#!/usr/bin/env python3
"""Measure workload perturbation from dynamic NCCL process-group creation."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--initial-size", type=int, default=-1,
                        help="Number of global ranks in the initial group. "
                        "Default: world_size - 1.")
    parser.add_argument("--add-rank", type=int, default=-1,
                        help="Rank excluded initially and included later. "
                        "Default: world_size - 1.")
    parser.add_argument("--trigger-sec", type=float, default=8.0)
    parser.add_argument("--total-sec", type=float, default=20.0)
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--compute-repeats", type=int, default=3)
    parser.add_argument("--collective-every", type=int, default=1,
                        help="Run one initial-group all_reduce every N "
                        "iterations. Set 0 to disable.")
    parser.add_argument("--collective-mb", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="float16")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--expanded-collective-mb", type=int, default=64)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


class JsonlWriter:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._fh = path.open("w", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def tensor_numel_for_mb(mb: int, dtype: torch.dtype) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return max(1, mb * 1024 * 1024 // element_size)


def run_compute_loop(
    *,
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    initial_group: dist.ProcessGroup,
    stop_event: threading.Event,
    t0: float,
    writer: JsonlWriter,
) -> None:
    dtype = dtype_from_name(args.dtype)
    torch.cuda.set_device(local_rank)
    stream = torch.cuda.Stream(device=local_rank)
    n = args.matrix_size

    with torch.cuda.device(local_rank):
        x = torch.randn((n, n), device="cuda", dtype=dtype)
        w = torch.randn((n, n), device="cuda", dtype=dtype)
        comm_numel = tensor_numel_for_mb(args.collective_mb, dtype)
        comm = torch.ones(comm_numel, device="cuda", dtype=dtype)
        continue_flag = torch.ones(1, device="cuda", dtype=torch.int32)

    for _ in range(args.warmup_iters):
        with torch.cuda.stream(stream):
            x = torch.mm(x, w)
            scale = x.norm().clamp_min(1.0)
            x = x / scale
        stream.synchronize()
        if args.collective_every:
            dist.all_reduce(comm, group=initial_group)
    torch.cuda.synchronize(local_rank)

    iteration = 0
    deadline = t0 + args.total_sec
    while True:
        if args.collective_every:
            should_continue = (
                not stop_event.is_set()
                and time.perf_counter() < deadline
            )
            continue_flag.fill_(1 if should_continue else 0)
            dist.all_reduce(continue_flag, op=dist.ReduceOp.MIN,
                            group=initial_group)
            if int(continue_flag.item()) == 0:
                break
        elif stop_event.is_set() or time.perf_counter() >= deadline:
            break

        wall_start = time.perf_counter()
        rel_start = wall_start - t0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(stream):
            start_event.record(stream)
            for _ in range(args.compute_repeats):
                x = torch.mm(x, w)
                # Keep values bounded without moving data to the host.
                x = x * 0.999 + 0.001
            end_event.record(stream)
        end_event.synchronize()

        collective_ms = None
        if args.collective_every and iteration % args.collective_every == 0:
            collective_start = time.perf_counter()
            dist.all_reduce(comm, group=initial_group)
            torch.cuda.synchronize(local_rank)
            collective_ms = (time.perf_counter() - collective_start) * 1000.0

        wall_end = time.perf_counter()
        writer.write({
            "type": "iteration",
            "rank": rank,
            "local_rank": local_rank,
            "iteration": iteration,
            "rel_start_sec": rel_start,
            "rel_end_sec": wall_end - t0,
            "wall_ms": (wall_end - wall_start) * 1000.0,
            "cuda_compute_ms": start_event.elapsed_time(end_event),
            "initial_collective_ms": collective_ms,
        })
        iteration += 1


def summarize_run(out_dir: Path, dynamic_start: float, dynamic_end: float) -> None:
    iterations: list[dict[str, Any]] = []
    events: dict[str, list[dict[str, Any]]] = {}
    metadata: list[dict[str, Any]] = []

    for path in sorted(out_dir.glob("rank_*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                item = json.loads(line)
                if item.get("type") == "iteration":
                    iterations.append(item)
                elif item.get("type") == "metadata":
                    metadata.append(item)
                else:
                    events.setdefault(item.get("type", "unknown"), []).append(item)

    stats: dict[str, dict[str, dict[str, Any]]] = {}
    for item in iterations:
        rank_key = str(item["rank"])
        if item["rel_end_sec"] < dynamic_start:
            phase = "before"
        elif item["rel_start_sec"] > dynamic_end:
            phase = "after"
        else:
            phase = "during"

        rank_stats = stats.setdefault(rank_key, {})
        phase_items = rank_stats.setdefault(phase, {
            "wall_ms": [],
            "cuda_compute_ms": [],
            "initial_collective_ms": [],
        })
        phase_items["wall_ms"].append(item["wall_ms"])
        phase_items["cuda_compute_ms"].append(item["cuda_compute_ms"])
        if item["initial_collective_ms"] is not None:
            phase_items["initial_collective_ms"].append(item["initial_collective_ms"])

    summarized_stats: dict[str, dict[str, dict[str, Any]]] = {}
    for rank, phases in stats.items():
        summarized_stats[rank] = {}
        for phase, metrics in phases.items():
            summarized_stats[rank][phase] = {
                metric: summarize_values(values)
                for metric, values in metrics.items()
            }

    def first_duration(event_type: str, duration_key: str) -> float | None:
        values = [event.get(duration_key) for event in events.get(event_type, [])]
        values = [value for value in values if value is not None]
        return max(values) if values else None

    summary = {
        "metadata": metadata,
        "dynamic_window": {
            "start_sec": dynamic_start,
            "end_sec": dynamic_end,
            "duration_sec": dynamic_end - dynamic_start,
        },
        "events": {
            "initial_new_group_sec": first_duration(
                "initial_group_created", "duration_sec"),
            "dynamic_new_group_sec": first_duration(
                "expanded_group_created", "duration_sec"),
            "expanded_first_all_reduce_sec": first_duration(
                "expanded_first_all_reduce", "duration_sec"),
        },
        "iteration_stats": summarized_stats,
        "raw_event_counts": {key: len(value) for key, value in events.items()},
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(json.dumps(summary["events"], indent=2, sort_keys=True), flush=True)
    for rank, phases in sorted(summarized_stats.items(), key=lambda item: int(item[0])):
        before = phases.get("before", {}).get("wall_ms", {})
        during = phases.get("during", {}).get("wall_ms", {})
        after = phases.get("after", {}).get("wall_ms", {})
        print(
            f"rank {rank}: wall_ms median before={before.get('median')} "
            f"during={during.get('median')} after={after.get('median')} "
            f"during_max={during.get('max')}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCCL probing")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(args.out_dir / f"rank_{rank}.jsonl")

    timeout = timedelta(seconds=args.timeout_sec)
    try:
        dist.init_process_group(
            backend="nccl",
            timeout=timeout,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    except TypeError:
        dist.init_process_group(backend="nccl", timeout=timeout)

    initial_size = args.initial_size if args.initial_size > 0 else world_size - 1
    add_rank = args.add_rank if args.add_rank >= 0 else world_size - 1
    if initial_size <= 0 or initial_size >= world_size:
        raise ValueError("initial_size must be in [1, world_size - 1]")
    if add_rank < 0 or add_rank >= world_size:
        raise ValueError("add_rank must be a valid global rank")

    initial_ranks = list(range(initial_size))
    if add_rank in initial_ranks:
        raise ValueError("add_rank must not be part of the initial group")
    expanded_ranks = sorted(initial_ranks + [add_rank])
    active = rank in initial_ranks

    writer.write({
        "type": "metadata",
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "local_world_size": local_world_size,
        "hostname": os.uname().nodename,
        "device": torch.cuda.get_device_name(local_rank),
        "initial_ranks": initial_ranks,
        "expanded_ranks": expanded_ranks,
        "args": vars(args) | {"out_dir": str(args.out_dir)},
    })

    dist.barrier()
    initial_start = time.perf_counter()
    initial_group = dist.new_group(ranks=initial_ranks, backend="nccl",
                                   timeout=timeout)
    initial_end = time.perf_counter()
    writer.write({
        "type": "initial_group_created",
        "rank": rank,
        "duration_sec": initial_end - initial_start,
        "is_member": active,
    })

    if active:
        dtype = dtype_from_name(args.dtype)
        probe = torch.ones(tensor_numel_for_mb(1, dtype), device="cuda", dtype=dtype)
        dist.all_reduce(probe, group=initial_group)
        torch.cuda.synchronize(local_rank)
        writer.write({"type": "initial_group_first_all_reduce", "rank": rank})

    dist.barrier()
    t0 = time.perf_counter()
    stop_event = threading.Event()
    compute_thread: threading.Thread | None = None
    if active:
        compute_thread = threading.Thread(
            target=run_compute_loop,
            kwargs={
                "args": args,
                "rank": rank,
                "local_rank": local_rank,
                "initial_group": initial_group,
                "stop_event": stop_event,
                "t0": t0,
                "writer": writer,
            },
            daemon=False,
        )
        compute_thread.start()

    sleep_for = max(0.0, args.trigger_sec - (time.perf_counter() - t0))
    time.sleep(sleep_for)

    dynamic_start = time.perf_counter() - t0
    expanded_start = time.perf_counter()
    expanded_group = dist.new_group(ranks=expanded_ranks, backend="nccl",
                                    timeout=timeout)
    expanded_end = time.perf_counter()
    writer.write({
        "type": "expanded_group_created",
        "rank": rank,
        "rel_start_sec": dynamic_start,
        "rel_end_sec": expanded_end - t0,
        "duration_sec": expanded_end - expanded_start,
        "is_member": rank in expanded_ranks,
    })

    if rank in expanded_ranks:
        dtype = dtype_from_name(args.dtype)
        numel = tensor_numel_for_mb(args.expanded_collective_mb, dtype)
        expanded_tensor = torch.full((numel,), float(rank + 1),
                                     device="cuda", dtype=dtype)
        coll_start_rel = time.perf_counter() - t0
        coll_start = time.perf_counter()
        dist.all_reduce(expanded_tensor, group=expanded_group)
        torch.cuda.synchronize(local_rank)
        coll_end = time.perf_counter()
        writer.write({
            "type": "expanded_first_all_reduce",
            "rank": rank,
            "rel_start_sec": coll_start_rel,
            "rel_end_sec": coll_end - t0,
            "duration_sec": coll_end - coll_start,
        })

    dynamic_end = time.perf_counter() - t0
    writer.write({
        "type": "dynamic_window_finished",
        "rank": rank,
        "rel_start_sec": dynamic_start,
        "rel_end_sec": dynamic_end,
    })

    remaining = args.total_sec - (time.perf_counter() - t0)
    if remaining > 0:
        time.sleep(remaining)
    stop_event.set()
    if compute_thread is not None:
        compute_thread.join()

    writer.write({"type": "rank_finished", "rank": rank,
                  "rel_end_sec": time.perf_counter() - t0})
    writer.close()

    dist.barrier()
    windows = [None for _ in range(world_size)]
    dist.all_gather_object(windows, (dynamic_start, dynamic_end))
    global_dynamic_start = min(window[0] for window in windows if window)
    global_dynamic_end = max(window[1] for window in windows if window)

    if rank == 0:
        summarize_run(args.out_dir, global_dynamic_start, global_dynamic_end)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

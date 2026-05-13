#!/usr/bin/env python3
"""Async NCCL group creation plus safe-point state switch demo.

All ranks are launched up front in the default world. The initial NCCL group
uses a subset of ranks. A background thread creates an expanded NCCL group while
active ranks continue communicating on the old group. At an iteration boundary,
all ranks enter a safe point, switch their local runtime state to the expanded
group, and continue communication with the new group.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
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
    group_name: str
    active_ranks: list[int]
    logical_stage: int | None
    participates: bool
    state_numel: int


class JsonlWriter:
    def __init__(self, path: Path):
        self._fh = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


class AsyncGroup:
    def __init__(self):
        self.ready = threading.Event()
        self.group: dist.ProcessGroup | None = None
        self.error: BaseException | None = None
        self.started_rel_sec: float | None = None
        self.finished_rel_sec: float | None = None
        self.duration_sec: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--initial-size", type=int, default=-1)
    parser.add_argument("--add-rank", type=int, default=-1)
    parser.add_argument("--async-create-sec", type=float, default=2.0)
    parser.add_argument("--switch-not-before-sec", type=float, default=6.0)
    parser.add_argument("--total-sec", type=float, default=12.0)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--compute-repeats", type=int, default=2)
    parser.add_argument("--collective-mb", type=int, default=8)
    parser.add_argument("--state-mb", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="float16")
    parser.add_argument("--timeout-sec", type=int, default=180)
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
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


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


def create_group_async(
    *,
    future: AsyncGroup,
    ranks: list[int],
    timeout: timedelta,
    t0: float,
    writer: JsonlWriter,
    rank: int,
) -> None:
    try:
        future.started_rel_sec = time.perf_counter() - t0
        start = time.perf_counter()
        writer.write({
            "type": "async_new_group_started",
            "rank": rank,
            "rel_sec": future.started_rel_sec,
            "ranks": ranks,
        })
        future.group = dist.new_group(ranks=ranks, backend="nccl",
                                      timeout=timeout)
        end = time.perf_counter()
        future.finished_rel_sec = end - t0
        future.duration_sec = end - start
        writer.write({
            "type": "async_new_group_finished",
            "rank": rank,
            "rel_sec": future.finished_rel_sec,
            "duration_sec": future.duration_sec,
        })
    except BaseException as exc:  # noqa: BLE001
        future.error = exc
        writer.write({
            "type": "async_new_group_error",
            "rank": rank,
            "error": repr(exc),
            "rel_sec": time.perf_counter() - t0,
        })
    finally:
        future.ready.set()


def allocate_stage_state(
    *,
    state: RuntimeState,
    dtype: torch.dtype,
    local_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.cuda.set_device(local_rank)
    n = int(math.sqrt(state.state_numel))
    n = max(256, n)
    state_tensor = torch.full((state.state_numel,), float(state.version),
                              device="cuda", dtype=dtype)
    x = torch.randn((n, n), device="cuda", dtype=dtype)
    w = torch.randn((n, n), device="cuda", dtype=dtype)
    return state_tensor, x, w


def run_one_iteration(
    *,
    group: dist.ProcessGroup,
    state: RuntimeState,
    x: torch.Tensor,
    w: torch.Tensor,
    comm: torch.Tensor,
    compute_repeats: int,
    phase: str,
    iteration: int,
    rank: int,
    local_rank: int,
    t0: float,
    writer: JsonlWriter,
) -> torch.Tensor:
    wall_start = time.perf_counter()
    rel_start = wall_start - t0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(local_rank)
    start_event.record(stream)
    for _ in range(compute_repeats):
        x = torch.mm(x, w)
        x = x * 0.999 + 0.001
    end_event.record(stream)
    end_event.synchronize()

    collective_start = time.perf_counter()
    dist.all_reduce(comm, group=group)
    torch.cuda.synchronize(local_rank)
    wall_end = time.perf_counter()

    writer.write({
        "type": "iteration",
        "rank": rank,
        "local_rank": local_rank,
        "phase": phase,
        "iteration": iteration,
        "state": asdict(state),
        "rel_start_sec": rel_start,
        "rel_end_sec": wall_end - t0,
        "wall_ms": (wall_end - wall_start) * 1000.0,
        "cuda_compute_ms": start_event.elapsed_time(end_event),
        "collective_ms": (wall_end - collective_start) * 1000.0,
    })
    return x


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

    def max_duration(event_type: str, key: str = "duration_sec") -> float | None:
        values = [item.get(key) for item in events.get(event_type, [])]
        values = [value for value in values if value is not None]
        return max(values) if values else None

    per_rank: dict[str, dict[str, dict[str, Any]]] = {}
    for item in iterations:
        rank_stats = per_rank.setdefault(str(item["rank"]), {})
        phase_stats = rank_stats.setdefault(item["phase"], {
            "wall_ms": [],
            "collective_ms": [],
            "cuda_compute_ms": [],
        })
        phase_stats["wall_ms"].append(item["wall_ms"])
        phase_stats["collective_ms"].append(item["collective_ms"])
        phase_stats["cuda_compute_ms"].append(item["cuda_compute_ms"])

    summarized = {
        rank: {
            phase: {metric: stats(values) for metric, values in metrics.items()}
            for phase, metrics in phases.items()
        }
        for rank, phases in per_rank.items()
    }

    summary = {
        "events": {
            "initial_group_create_sec": max_duration("initial_group_created"),
            "async_expanded_group_create_sec": max_duration(
                "async_new_group_finished"),
            "safe_point_pause_sec": max_duration("safe_point_exit",
                                                "pause_sec"),
            "expanded_first_collective_sec": max_duration(
                "expanded_first_collective"),
        },
        "iteration_stats": summarized,
        "event_counts": {key: len(value) for key, value in events.items()},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(json.dumps(summary["events"], indent=2, sort_keys=True), flush=True)
    for rank in sorted(summarized, key=lambda value: int(value)):
        old_phase = summarized[rank].get("old", {}).get("wall_ms", {})
        new_phase = summarized[rank].get("new", {}).get("wall_ms", {})
        print(
            f"rank {rank}: old_median={old_phase.get('median')} "
            f"new_median={new_phase.get('median')} "
            f"old_count={old_phase.get('count')} "
            f"new_count={new_phase.get('count')}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    timeout = timedelta(seconds=args.timeout_sec)

    try:
        dist.init_process_group(
            backend="nccl",
            timeout=timeout,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    except TypeError:
        dist.init_process_group(backend="nccl", timeout=timeout)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(args.out_dir / f"rank_{rank}.jsonl")

    initial_size = args.initial_size if args.initial_size > 0 else world_size - 1
    add_rank = args.add_rank if args.add_rank >= 0 else world_size - 1
    initial_ranks = list(range(initial_size))
    expanded_ranks = sorted(initial_ranks + [add_rank])
    if add_rank in initial_ranks:
        raise ValueError("add_rank must not be in the initial group")
    if expanded_ranks != list(range(world_size)):
        raise ValueError("this demo expects expanded_ranks to cover world")

    dtype = dtype_from_name(args.dtype)
    comm = torch.ones(tensor_numel_for_mb(args.collective_mb, dtype),
                      device="cuda", dtype=dtype)
    continue_flag = torch.ones(1, device="cuda", dtype=torch.int32)

    writer.write({
        "type": "metadata",
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
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
    initial_duration = time.perf_counter() - initial_start
    writer.write({
        "type": "initial_group_created",
        "rank": rank,
        "duration_sec": initial_duration,
        "is_member": rank in initial_ranks,
    })

    active = rank in initial_ranks
    state_numel = tensor_numel_for_mb(args.state_mb, dtype)
    state = RuntimeState(
        version=0,
        group_name="old",
        active_ranks=initial_ranks,
        logical_stage=initial_ranks.index(rank) if active else None,
        participates=active,
        state_numel=state_numel,
    )
    state_tensor: torch.Tensor | None = None
    x: torch.Tensor | None = None
    w: torch.Tensor | None = None
    if active:
        state_tensor, x, w = allocate_stage_state(
            state=state, dtype=dtype, local_rank=local_rank)
        writer.write({"type": "state_installed", "rank": rank,
                      "state": asdict(state), "checksum": float(state_tensor[0])})
    else:
        writer.write({"type": "state_idle", "rank": rank,
                      "state": asdict(state)})

    dist.barrier()
    t0 = time.perf_counter()

    future = AsyncGroup()
    sleep_for = max(0.0, args.async_create_sec - (time.perf_counter() - t0))
    time.sleep(sleep_for)
    creator = threading.Thread(
        target=create_group_async,
        kwargs={
            "future": future,
            "ranks": expanded_ranks,
            "timeout": timeout,
            "t0": t0,
            "writer": writer,
            "rank": rank,
        },
        daemon=False,
    )
    creator.start()

    iteration = 0
    if active:
        assert x is not None and w is not None
        while True:
            local_should_switch = (
                time.perf_counter() - t0 >= args.switch_not_before_sec
                and future.ready.is_set()
            )
            continue_flag.fill_(0 if local_should_switch else 1)
            dist.all_reduce(continue_flag, op=dist.ReduceOp.MIN,
                            group=initial_group)
            if int(continue_flag.item()) == 0:
                break
            x = run_one_iteration(
                group=initial_group,
                state=state,
                x=x,
                w=w,
                comm=comm,
                compute_repeats=args.compute_repeats,
                phase="old",
                iteration=iteration,
                rank=rank,
                local_rank=local_rank,
                t0=t0,
                writer=writer,
            )
            iteration += 1
    else:
        while (time.perf_counter() - t0 < args.switch_not_before_sec
               or not future.ready.is_set()):
            time.sleep(0.001)

    safe_enter = time.perf_counter()
    writer.write({
        "type": "safe_point_enter",
        "rank": rank,
        "rel_sec": safe_enter - t0,
        "old_state": asdict(state),
    })

    creator.join()
    if future.error is not None:
        raise RuntimeError(f"async group creation failed: {future.error!r}")
    if future.group is None:
        raise RuntimeError("async group did not return a process group")

    # This is the controlled switch point. No old-group collectives are in
    # flight. All ranks rendezvous, validate the expanded communicator, and
    # then install the new local state.
    dist.barrier()
    first_collective_start = time.perf_counter()
    expanded_probe = torch.full((1,), float(rank + 1), device="cuda", dtype=dtype)
    dist.all_reduce(expanded_probe, group=future.group)
    torch.cuda.synchronize(local_rank)
    first_collective_sec = time.perf_counter() - first_collective_start
    writer.write({
        "type": "expanded_first_collective",
        "rank": rank,
        "duration_sec": first_collective_sec,
        "value": float(expanded_probe.item()),
    })

    state = RuntimeState(
        version=1,
        group_name="expanded",
        active_ranks=expanded_ranks,
        logical_stage=expanded_ranks.index(rank),
        participates=True,
        state_numel=state_numel,
    )
    state_tensor, x, w = allocate_stage_state(
        state=state, dtype=dtype, local_rank=local_rank)
    safe_exit = time.perf_counter()
    writer.write({
        "type": "safe_point_exit",
        "rank": rank,
        "rel_sec": safe_exit - t0,
        "pause_sec": safe_exit - safe_enter,
        "new_state": asdict(state),
        "checksum": float(state_tensor[0]),
    })

    while True:
        continue_flag.fill_(1 if time.perf_counter() - t0 < args.total_sec else 0)
        dist.all_reduce(continue_flag, op=dist.ReduceOp.MIN,
                        group=future.group)
        if int(continue_flag.item()) == 0:
            break
        x = run_one_iteration(
            group=future.group,
            state=state,
            x=x,
            w=w,
            comm=comm,
            compute_repeats=args.compute_repeats,
            phase="new",
            iteration=iteration,
            rank=rank,
            local_rank=local_rank,
            t0=t0,
            writer=writer,
        )
        iteration += 1

    writer.write({"type": "rank_finished", "rank": rank,
                  "rel_sec": time.perf_counter() - t0})
    writer.close()
    dist.barrier()

    if rank == 0:
        summarize(args.out_dir)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

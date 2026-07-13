#!/usr/bin/env python3
"""Summarize the short single-migration KV backend comparison."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np


RUN_ROOT = Path(
    "/data/gpfs/projects/punim2715/vllm_workbench/burstgpt_runs/"
    "single_migration_backend_compare_20260712"
)
OUTPUT_DIR = Path(
    "/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/"
    "experiments/single_migration_backend_compare_20260712"
)

RUNS = {
    "KVCacheD PreMap": RUN_ROOT / "parallel_req62/premap",
    "KVCacheD On-demand": RUN_ROOT / "parallel_req62/ondemand",
}
NATIVE_FAILED_ROOT = RUN_ROOT / "native_req62_solo"
COLORS = {
    "KVCacheD PreMap": "#d97706",
    "KVCacheD On-demand": "#16845b",
    "Native VMM": "#c83e39",
}


def find_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} below {root}, got {matches}")
    return matches[0]


def parse_number(log: str, label: str) -> float:
    match = re.search(rf"^{re.escape(label)}\s+([0-9.]+)\s*$", log, re.M)
    if match is None:
        raise RuntimeError(f"Missing benchmark field: {label}")
    return float(match.group(1))


def parse_migration_epoch(log: str) -> float:
    match = re.search(
        r"INFO\s+(\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}).*"
        r"Using async migration mode",
        log,
    )
    if match is None:
        raise RuntimeError("Missing migration start timestamp")
    value = datetime.strptime(
        f"2026-{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=ZoneInfo("Australia/Melbourne"))
    return value.timestamp()


def read_successful_run(name: str, root: Path) -> dict[str, object]:
    request_path = find_one(root, "request_metrics.csv")
    benchmark_path = find_one(root, "benchmark.log")
    analysis_path = find_one(root, "analysis.txt")
    server_path = find_one(root, "server.log")

    with request_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    e2e_ms = np.asarray([float(row["e2els"]) * 1000 for row in rows])
    ttft_ms = np.asarray([float(row["ttfts"]) * 1000 for row in rows])
    tpot_ms = np.asarray([float(row["tpots"]) * 1000 for row in rows])

    benchmark_log = benchmark_path.read_text(encoding="utf-8")
    analysis_log = analysis_path.read_text(encoding="utf-8")
    server_log = server_path.read_text(encoding="utf-8")
    migration_match = re.search(
        r"\[TIMELINE\] MIGRATION PROCESS TOTAL TIME.*?Event 1:\s*([0-9.]+)s",
        analysis_log,
        re.S,
    )
    if migration_match is None:
        raise RuntimeError(f"Missing migration duration for {name}")

    start_epoch = float(timestamps.min())
    return {
        "name": name,
        "status": "completed",
        "successful_requests": int(parse_number(
            benchmark_log, "Successful requests:")),
        "benchmark_duration_s": parse_number(
            benchmark_log, "Benchmark duration (s):"),
        "request_throughput_rps": parse_number(
            benchmark_log, "Request throughput (req/s):"),
        "request_goodput_rps": parse_number(
            benchmark_log, "Request goodput (req/s):"),
        "total_token_throughput_tps": parse_number(
            benchmark_log, "Total Token throughput (tok/s):"),
        "mean_ttft_ms": float(ttft_ms.mean()),
        "mean_tpot_ms": float(tpot_ms.mean()),
        "mean_e2e_ms": float(e2e_ms.mean()),
        "median_e2e_ms": float(np.median(e2e_ms)),
        "p99_e2e_ms": float(np.percentile(e2e_ms, 99)),
        "max_e2e_ms": float(e2e_ms.max()),
        "migration_duration_s": float(migration_match.group(1)),
        "migration_replay_s": parse_migration_epoch(server_log) - start_epoch,
        "replay_s": (timestamps - start_epoch).tolist(),
        "e2e_ms": e2e_ms.tolist(),
        "request_metrics": str(request_path),
        "benchmark_log": str(benchmark_path),
        "server_log": str(server_path),
    }


def write_summary_csv(results: list[dict[str, object]]) -> None:
    fields = [
        "name",
        "status",
        "successful_requests",
        "benchmark_duration_s",
        "request_throughput_rps",
        "request_goodput_rps",
        "total_token_throughput_tps",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_e2e_ms",
        "median_e2e_ms",
        "p99_e2e_ms",
        "max_e2e_ms",
        "migration_duration_s",
    ]
    with (OUTPUT_DIR / "backend_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})


def plot(results: list[dict[str, object]]) -> None:
    completed = [row for row in results if row["status"] == "completed"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "Single migration backend comparison (36,44 to 48,32 at request 62)",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    for row in completed:
        name = str(row["name"])
        ax.plot(
            row["replay_s"], row["e2e_ms"], marker="o", markersize=2.5,
            linewidth=1.3, alpha=0.88, color=COLORS[name], label=name,
        )
        ax.axvline(
            float(row["migration_replay_s"]), color=COLORS[name],
            linestyle="--", linewidth=1.2, alpha=0.8,
        )
    ax.set_title("Per-request end-to-end latency")
    ax.set_xlabel("Replay time (s)")
    ax.set_ylabel("E2E latency (ms)")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    labels = [str(row["name"]).replace("KVCacheD ", "") for row in completed]
    x = np.arange(len(labels))
    width = 0.24
    for offset, metric, title in [
        (-width, "mean_e2e_ms", "Mean"),
        (0, "median_e2e_ms", "Median"),
        (width, "p99_e2e_ms", "P99"),
    ]:
        values = [float(row[metric]) for row in completed]
        ax.bar(x + offset, values, width, label=title)
    ax.set_xticks(x, labels)
    ax.set_title("E2E latency summary")
    ax.set_ylabel("Latency (ms), lower is better")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    throughput = [float(row["total_token_throughput_tps"]) for row in completed]
    colors = [COLORS[str(row["name"])] for row in completed]
    bars = ax.bar(labels, throughput, color=colors, width=0.55)
    ax.set_title("End-to-end token throughput")
    ax.set_ylabel("Total tokens/s, higher is better")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, throughput):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5,
                f"{value:.2f}", ha="center", va="bottom")
    ax.text(
        0.98, 0.05,
        "Native VMM: FAILED during first non-empty KV migration NCCL send",
        transform=ax.transAxes, ha="right", va="bottom", color=COLORS["Native VMM"],
        bbox={"facecolor": "white", "edgecolor": COLORS["Native VMM"], "pad": 6},
    )

    ax = axes[1, 1]
    migration = [float(row["migration_duration_s"]) for row in completed]
    bars = ax.bar(labels, migration, color=colors, width=0.55)
    ax.set_title("Migration process duration")
    ax.set_ylabel("Seconds, lower is better")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, migration):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03,
                f"{value:.2f}s", ha="center", va="bottom")
    ax.text(
        0.98, 0.93, "Native VMM: no completed migration",
        transform=ax.transAxes, ha="right", va="top", color=COLORS["Native VMM"],
    )

    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT_DIR / f"single_migration_backend_comparison.{suffix}",
                    dpi=180 if suffix == "png" else None)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [read_successful_run(name, root) for name, root in RUNS.items()]
    results.append({
        "name": "Native VMM",
        "status": "failed",
        "successful_requests": 59,
        "failure": (
            "NCCL unhandled cuda error during the first non-empty slot-mapping "
            "send; reproduced in a solo rerun"
        ),
        "server_log": str(find_one(NATIVE_FAILED_ROOT, "server_raw.log")),
    })
    write_summary_csv(results)
    plot(results)
    with (OUTPUT_DIR / "comparison_data.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    for row in results:
        print(row["name"], row["status"], row.get("mean_e2e_ms", "N/A"))


if __name__ == "__main__":
    main()

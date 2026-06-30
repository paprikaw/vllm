#!/usr/bin/env python3
"""Generate a short BurstGPT autoscaling experiment from ShareGPT calibration.

The generated config keeps the fixed experiment baseline knobs intact while
choosing runtime-specific request stages and PP layouts from workload size.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any

import yaml
from transformers import AutoTokenizer


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


PP_BY_RANKS = {
    1: "64,0,0,0",
    2: "32,32,0,0",
    3: "16,16,32,0",
    4: "16,16,16,16",
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def calibrate_from_sharegpt(
    sharegpt_path: Path,
    tokenizer,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    data = json.loads(sharegpt_path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    sample = rng.sample(data, min(sample_size, len(data)))

    prompt_lens: list[int] = []
    output_lens: list[int] = []
    total_lens: list[int] = []
    turns: list[int] = []
    for item in sample:
        conv = item.get("conversations") or []
        turns.append(len(conv))
        if len(conv) < 2:
            continue
        prompt = conv[0].get("value", "")
        output = conv[1].get("value", "")
        prompt_len = len(tokenizer(prompt).input_ids)
        output_len = max(1, len(tokenizer(output).input_ids))
        prompt_lens.append(prompt_len)
        output_lens.append(output_len)
        total_lens.append(prompt_len + output_len)

    capacity_unit = max(1.0, percentile(total_lens, 75))
    return {
        "sample_size": len(sample),
        "turns": {
            "p50": percentile(turns, 50),
            "p90": percentile(turns, 90),
            "p95": percentile(turns, 95),
        },
        "prompt_tokens": {
            "p50": percentile(prompt_lens, 50),
            "p75": percentile(prompt_lens, 75),
            "p90": percentile(prompt_lens, 90),
        },
        "output_tokens": {
            "p50": percentile(output_lens, 50),
            "p75": percentile(output_lens, 75),
            "p90": percentile(output_lens, 90),
        },
        "total_tokens": {
            "p50": percentile(total_lens, 50),
            "p75": percentile(total_lens, 75),
            "p90": percentile(total_lens, 90),
            "p95": percentile(total_lens, 95),
        },
        "capacity_unit_tokens_per_second": capacity_unit,
    }


def read_burst_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                if row["Model"] != "GPT-4":
                    continue
                request_tokens = int(float(row["Request tokens"]))
                response_tokens = int(float(row["Response tokens"]))
                timestamp = float(row["Timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if response_tokens <= 0:
                continue
            rows.append({
                **row,
                "_timestamp": timestamp,
                "_request_tokens": request_tokens,
                "_response_tokens": response_tokens,
                "_total_tokens": request_tokens + response_tokens,
            })
    rows.sort(key=lambda r: r["_timestamp"])
    return rows


def window_stats(rows: list[dict[str, Any]], start: int, size: int) -> dict[str, Any]:
    window = rows[start:start + size]
    duration = max(1.0, window[-1]["_timestamp"] - window[0]["_timestamp"])
    observed_rate = max(0.1, (len(window) - 1) / duration)
    mean_input = mean(r["_request_tokens"] for r in window)
    mean_output = mean(r["_response_tokens"] for r in window)
    mean_total = mean(r["_total_tokens"] for r in window)
    return {
        "start": start,
        "rows": window,
        "observed_rate": observed_rate,
        "mean_input": mean_input,
        "mean_output": mean_output,
        "mean_total": mean_total,
        "observed_token_rate": observed_rate * mean_total,
    }


def select_burst_stages(
    rows: list[dict[str, Any]],
    stage_size: int,
    max_rate: float,
    max_total_tokens: int,
) -> list[dict[str, Any]]:
    rows = [r for r in rows if r["_total_tokens"] <= max_total_tokens]
    stride = stage_size
    candidates = [
        window_stats(rows, start, stage_size)
        for start in range(0, max(0, len(rows) - stage_size), stride)
    ]
    candidates = [
        c for c in candidates
        if all(r["_total_tokens"] <= max_total_tokens for r in c["rows"])
    ]
    candidates.sort(key=lambda c: c["observed_token_rate"])
    if len(candidates) < 4:
        raise ValueError("Not enough BurstGPT windows after filtering")

    picks = [
        candidates[int(0.20 * (len(candidates) - 1))],
        candidates[int(0.55 * (len(candidates) - 1))],
        candidates[int(0.92 * (len(candidates) - 1))],
        candidates[int(0.35 * (len(candidates) - 1))],
    ]

    stages: list[dict[str, Any]] = []
    for stage in picks:
        clipped_rate = min(max_rate, max(0.4, stage["observed_rate"]))
        stage = dict(stage)
        stage["request_rate"] = round(clipped_rate, 3)
        stage["score"] = stage["request_rate"] * stage["mean_total"]
        stages.append(stage)
    return stages


def ranks_for_score(score: float, capacity_unit: float) -> int:
    return max(1, min(4, math.ceil(score / max(1.0, capacity_unit))))


def clean_burst_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Timestamp": row["Timestamp"],
        "Model": row["Model"],
        "Request tokens": row["Request tokens"],
        "Response tokens": row["Response tokens"],
        "Total tokens": row["Total tokens"],
        "Log Type": row.get("Log Type", "Conversation log"),
    }


def write_ordered_burst_subset(path: Path, stages: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Timestamp",
        "Model",
        "Request tokens",
        "Response tokens",
        "Total tokens",
        "Log Type",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stage in stages:
            for row in stage["rows"]:
                writer.writerow(clean_burst_row(row))


def benchmark_item(
    label: str,
    pp_by_stage: dict[int, str],
    requests: dict[int, dict[str, Any]],
    total_requests: int,
) -> dict[str, Any]:
    return {
        "label": label,
        "num_total_requests": total_requests,
        "repetition": 1,
        "pp_layer_config": pp_by_stage,
        "requests": requests,
    }


def make_benchmark_items(
    requests: dict[int, dict[str, Any]],
    total_requests: int,
    auto_pp: dict[int, str],
) -> list[dict[str, Any]]:
    static_items = [
        benchmark_item(
            f"static_pp{rank}",
            {0: PP_BY_RANKS[rank]},
            requests,
            total_requests,
        )
        for rank in (4, 1, 3, 2)
    ]
    auto_item = benchmark_item("autoscale_sharegpt_burst", auto_pp, requests,
                               total_requests)
    return static_items + [auto_item]


def build_config(
    *,
    project: str,
    node: str,
    model_path: str,
    dataset_path: Path,
    total_requests: int,
    benchmark_items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "project": project,
        "type": "sweep_test",
        "is_log_cover": True,
        "overwrite": True,
        "envs": {
            "RAY_DEDUP_LOGS": "0",
            "NCCL_DEBUG": "INFO",
            "NCCL_CUMEM_HOST_ENABLE": "0",
            "VLLM_NCCL_SO_PATH": "/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/NCCL/2.21.5-CUDA-12.4.1/lib/libnccl.so.2.21.5",
            "RAY_CGRAPH_get_timeout": "10000",
            "KV_SYNC_USE_CPU": "0",
            "VLLM_USE_CPU_MODEL": "0",
            "CUDA_LAUNCH_BLOCKING": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:32",
            "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM": "0",
            "VLLM_USE_RAY_COMPILED_DAG": "0",
            "VLLM_USE_RAY_SPMD_WORKER": "1",
            "VLLM_DYNAMIC_PP_NCCL_TRANSPORT": "0",
            "VLLM_DYNAMIC_PP_RDT_TRANSPORT": "nccl",
            "VLLM_DYNAMIC_PP_RDT_PREWARM": "1",
            "VLLM_DYNAMIC_PP_RDT_PREWARM_MODE": "adjacent",
            "VLLM_RAY_RDT_MULTIGPU_PATCH": "1",
            "VLLM_RAY_RDT_SYNC_AFTER_TRANSFER": "1",
            "VLLM_RAY_RDT_SERIALIZE_WITH_KV_NCCL_LOCK": "1",
            "VLLM_DYNAMIC_PP_MAX_CONCURRENT_BATCHES": "1",
            "NCCL_SOCKET_IFNAME": "bond0.3027",
            "GLOO_SOCKET_IFNAME": "bond0.3027",
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_DISABLE_CUTLASS_FP8": "1",
            "VLLM_PAGE_ATTENTION_BLOCK_SIZE_KB": "128",
            "VLLM_BURSTGPT_ORDERED": "1",
        },
        "static_config": {
            "model": {
                "path": model_path,
                "name": "Qwen3-32B-FP8",
            },
            "network": {
                "placements": [
                    {"pp_stage": idx, "node": node, "ip": node}
                    for idx in range(4)
                ],
            },
            "vllm": {
                "head_addr": "localhost",
                "pipeline_parallel_size": 4,
                "pipeline_autoscaling_enabled": True,
                "autoscaling_candidate_ranks": [0, 1, 2, 3],
                "gpu_memory_utilization": 0.85,
                "max_model_len": 1024,
                "max_num_batched_tokens": 512,
                "chunked_prefill": True,
                "enable_cuda_graph": False,
                "enable_nsight": False,
                "weight_chunk_size_mb": 160,
                "log_kv_memory_stats": True,
                "disable_memory_overhead_monitor": False,
            },
            "benchmark": {
                "benchmark_script_path": "/data/gpfs/projects/punim2715/vllm_workbench/vllm/benchmarks/benchmark_serving.py",
                "num_total_requests": total_requests,
                "repetition": 1,
                "burstiness": 100.0,
                "print_outputs": True,
                "profile": False,
                "dataset_name": "burstgpt",
                "dataset_path": str(dataset_path),
                "warmup": {"enabled": False},
            },
        },
        "sweep_configs": [{
            "vllm": {
                "attention_kernel": ["direct"],
                "fixed_num_gpu_blocks": [-1],
                "weight_chunk_size_mb": [160],
                "migration_approach": ["async"],
                "weight_loading_mode": ["async"],
                "use_vmm": [True],
                "enable_kv_resize": [True],
                "enable_cpu_weight_cache": [True],
            },
            "benchmark_config": copy.deepcopy(benchmark_items),
        }],
    }


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(config, Dumper=NoAliasDumper, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharegpt", type=Path,
                        default=Path("/data/gpfs/projects/punim2715/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"))
    parser.add_argument("--burstgpt", type=Path,
                        default=Path("/data/gpfs/projects/punim2715/datasets/BurstGPT_without_fails_2.csv"))
    parser.add_argument("--model-path", default="/home/bxb1/data/huggingface/Qwen3-32B-FP8")
    parser.add_argument("--node", default="spartan-gpgpu005")
    parser.add_argument("--stage-size", type=int, default=6)
    parser.add_argument("--sharegpt-sample-size", type=int, default=512)
    parser.add_argument("--max-rate", type=float, default=2.0)
    parser.add_argument("--max-total-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-config", type=Path,
                        default=Path("vllm_exp/configs/generated/l40_burstgpt_sharegpt_autoscale.yaml"))
    parser.add_argument("--output-split-dir", type=Path,
                        default=Path("vllm_exp/configs/generated/l40_burstgpt_sharegpt_runs"))
    parser.add_argument("--output-dataset", type=Path,
                        default=Path("vllm_exp/generated_workloads/burstgpt_l40_short_ordered.csv"))
    parser.add_argument("--output-report", type=Path,
                        default=Path("vllm_exp/generated_workloads/burstgpt_l40_autoscale_design_report.md"))
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    calibration = calibrate_from_sharegpt(
        args.sharegpt, tokenizer, args.sharegpt_sample_size, args.seed)
    burst_rows = read_burst_rows(args.burstgpt)
    stages = select_burst_stages(burst_rows, args.stage_size, args.max_rate,
                                 args.max_total_tokens)
    write_ordered_burst_subset(args.output_dataset, stages)

    capacity_unit = calibration["capacity_unit_tokens_per_second"]
    stage_boundaries = [i * args.stage_size for i in range(len(stages))]
    requests: dict[int, dict[str, Any]] = {}
    auto_pp: dict[int, str] = {}
    stage_report_rows: list[dict[str, Any]] = []
    last_pp = None
    for boundary, stage in zip(stage_boundaries, stages):
        ranks = ranks_for_score(stage["score"], capacity_unit)
        pp = PP_BY_RANKS[ranks]
        requests[boundary] = {
            "request_rate": float(stage["request_rate"]),
            "input_lens": int(round(stage["mean_input"])),
            "output_lens": int(round(stage["mean_output"])),
        }
        if last_pp != pp:
            auto_pp[boundary] = pp
            last_pp = pp
        stage_report_rows.append({
            "boundary": boundary,
            "rate": stage["request_rate"],
            "mean_input": stage["mean_input"],
            "mean_output": stage["mean_output"],
            "mean_total": stage["mean_total"],
            "score": stage["score"],
            "ranks": ranks,
            "pp": pp,
        })

    total_requests = args.stage_size * len(stages)
    benchmark_items = make_benchmark_items(requests, total_requests, auto_pp)
    config = build_config(
        project="l40_burstgpt_sharegpt_autoscale_short",
        node=args.node,
        model_path=args.model_path,
        dataset_path=args.output_dataset.resolve(),
        total_requests=total_requests,
        benchmark_items=benchmark_items,
    )

    write_yaml(args.output_config, config)

    split_paths: list[Path] = []
    for item in benchmark_items:
        label = item["label"]
        split_config = build_config(
            project=f"l40_burstgpt_sharegpt_{label}_short",
            node=args.node,
            model_path=args.model_path,
            dataset_path=args.output_dataset.resolve(),
            total_requests=total_requests,
            benchmark_items=[item],
        )
        split_path = args.output_split_dir / f"{label}.yaml"
        write_yaml(split_path, split_config)
        split_paths.append(split_path)

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ShareGPT-Calibrated BurstGPT Autoscaling Design",
        "",
        "## Rule",
        "",
        "Use ShareGPT first-turn token distribution to estimate one active-rank capacity.",
        f"`capacity_unit = ShareGPT total_tokens p75 = {capacity_unit:.1f} tokens/s at 1 rps`.",
        "For each BurstGPT stage, compute `score = request_rate * mean_total_tokens`.",
        "Choose `active_ranks = ceil(score / capacity_unit)`, clamped to `[1, 4]`.",
        "",
        "## ShareGPT Calibration",
        "",
        f"- sample_size: {calibration['sample_size']}",
        f"- total_tokens p50/p75/p90/p95: {calibration['total_tokens']['p50']:.1f} / {calibration['total_tokens']['p75']:.1f} / {calibration['total_tokens']['p90']:.1f} / {calibration['total_tokens']['p95']:.1f}",
        f"- output_tokens p50/p75/p90: {calibration['output_tokens']['p50']:.1f} / {calibration['output_tokens']['p75']:.1f} / {calibration['output_tokens']['p90']:.1f}",
        "",
        "## BurstGPT Stages",
        "",
        "| boundary | req/s | mean in | mean out | score | active ranks | pp |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in stage_report_rows:
        lines.append(
            f"| {row['boundary']} | {row['rate']:.3f} | {row['mean_input']:.1f} | "
            f"{row['mean_output']:.1f} | {row['score']:.1f} | "
            f"{row['ranks']} | `{row['pp']}` |")
    lines.extend([
        "",
        "## Generated Artifacts",
        "",
        f"- config: `{args.output_config.resolve()}`",
        f"- split config dir: `{args.output_split_dir.resolve()}`",
        f"- ordered BurstGPT subset: `{args.output_dataset.resolve()}`",
    ])
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote config: {args.output_config}")
    print(f"Wrote split configs: {args.output_split_dir}")
    print(f"Wrote dataset: {args.output_dataset}")
    print(f"Wrote report: {args.output_report}")


if __name__ == "__main__":
    main()

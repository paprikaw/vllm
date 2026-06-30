#!/usr/bin/env python3
"""Generate BurstGPT workloads for KV-pressure autoscaling experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any

import yaml

from vllm_exp.tools.generate_sharegpt_burst_autoscale import (
    NoAliasDumper,
    PP_BY_RANKS,
    calibrate_from_sharegpt,
    load_tokenizer,
)


PRESSURE_PP_BY_RANKS = {
    1: PP_BY_RANKS[1],
    2: PP_BY_RANKS[2],
    3: PP_BY_RANKS[3],
    4: PP_BY_RANKS[4],
}


def read_burst_rows(path: Path, max_total_tokens: int) -> list[dict[str, Any]]:
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
            total_tokens = request_tokens + response_tokens
            if response_tokens <= 0 or total_tokens > max_total_tokens:
                continue
            rows.append({
                **row,
                "_timestamp": timestamp,
                "_request_tokens": request_tokens,
                "_response_tokens": response_tokens,
                "_total_tokens": total_tokens,
            })
    return rows


def choose_rows_for_stage(
    rows: list[dict[str, Any]],
    stage_size: int,
    *,
    min_output_tokens: int,
    max_input_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = [
        r for r in rows
        if r["_response_tokens"] >= min_output_tokens
        and r["_request_tokens"] <= max_input_tokens
    ]
    if not candidates:
        raise ValueError(
            f"No BurstGPT rows for min_output_tokens={min_output_tokens}, "
            f"max_input_tokens={max_input_tokens}")

    # Prefer longer outputs while keeping a little timestamp diversity.
    candidates.sort(key=lambda r: (-r["_response_tokens"], r["_timestamp"]))
    top = candidates[:max(stage_size * 12, stage_size)]
    rng = random.Random(seed)
    if len(top) >= stage_size:
        return rng.sample(top, stage_size)
    return [rng.choice(top) for _ in range(stage_size)]


def clean_burst_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Timestamp": row["Timestamp"],
        "Model": row["Model"],
        "Request tokens": row["Request tokens"],
        "Response tokens": row["Response tokens"],
        "Total tokens": row["Total tokens"],
        "Log Type": row.get("Log Type", "Conversation log"),
    }


def write_ordered_burst_subset(path: Path,
                               stages: list[dict[str, Any]]) -> None:
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
        "requests": copy.deepcopy(requests),
    }


def build_config(
    *,
    project: str,
    node: str,
    model_path: str,
    dataset_path: Path,
    total_requests: int,
    benchmark_items: list[dict[str, Any]],
    autoscaling_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    vllm_static = {
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
    }
    if autoscaling_policy is not None:
        vllm_static["autoscaling_policy"] = autoscaling_policy

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
            "vllm": vllm_static,
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


def pressure_estimate(stage: dict[str, Any],
                      sharegpt_capacity_unit: float) -> float:
    total = stage["mean_input"] + stage["mean_output"]
    output_hold_factor = max(1.0, stage["mean_output"] / 256.0)
    return stage["request_rate"] * total * output_hold_factor / max(
        1.0, sharegpt_capacity_unit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharegpt", type=Path,
                        default=Path("/data/gpfs/projects/punim2715/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"))
    parser.add_argument("--burstgpt", type=Path,
                        default=Path("/data/gpfs/projects/punim2715/datasets/BurstGPT_without_fails_2.csv"))
    parser.add_argument("--model-path",
                        default="/home/bxb1/data/huggingface/Qwen3-32B-FP8")
    parser.add_argument("--node", default="spartan-gpgpu005")
    parser.add_argument("--stage-size", type=int, default=64)
    parser.add_argument("--sharegpt-sample-size", type=int, default=512)
    parser.add_argument("--max-total-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--output-config", type=Path,
                        default=Path("vllm_exp/configs/generated/l40_burstgpt_kv_pressure_autoscale.yaml"))
    parser.add_argument("--output-split-dir", type=Path,
                        default=Path("vllm_exp/configs/generated/l40_burstgpt_kv_pressure_runs"))
    parser.add_argument("--output-dataset", type=Path,
                        default=Path("vllm_exp/generated_workloads/burstgpt_l40_kv_pressure_ordered.csv"))
    parser.add_argument("--output-report", type=Path,
                        default=Path("vllm_exp/generated_workloads/burstgpt_l40_kv_pressure_design_report.md"))
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    calibration = calibrate_from_sharegpt(
        args.sharegpt, tokenizer, args.sharegpt_sample_size, args.seed)
    burst_rows = read_burst_rows(args.burstgpt, args.max_total_tokens)

    ladder_specs = [
        {"name": "pressure_1gpu", "request_rate": 2.0, "min_output": 500, "max_input": 360},
        {"name": "pressure_2gpu", "request_rate": 4.0, "min_output": 650, "max_input": 280},
        {"name": "pressure_3gpu", "request_rate": 6.0, "min_output": 760, "max_input": 220},
        {"name": "pressure_4gpu", "request_rate": 8.0, "min_output": 860, "max_input": 180},
    ]

    stages: list[dict[str, Any]] = []
    for idx, spec in enumerate(ladder_specs):
        rows = choose_rows_for_stage(
            burst_rows,
            args.stage_size,
            min_output_tokens=spec["min_output"],
            max_input_tokens=spec["max_input"],
            seed=args.seed + idx,
        )
        stage = {
            **spec,
            "rows": rows,
            "mean_input": mean(r["_request_tokens"] for r in rows),
            "mean_output": mean(r["_response_tokens"] for r in rows),
        }
        stage["mean_total"] = stage["mean_input"] + stage["mean_output"]
        stage["pressure_score"] = pressure_estimate(
            stage, calibration["capacity_unit_tokens_per_second"])
        stages.append(stage)

    write_ordered_burst_subset(args.output_dataset, stages)

    requests: dict[int, dict[str, Any]] = {}
    for idx, stage in enumerate(stages):
        boundary = idx * args.stage_size
        requests[boundary] = {
            "request_rate": float(stage["request_rate"]),
            "input_lens": int(round(stage["mean_input"])),
            "output_lens": int(round(stage["mean_output"])),
        }

    total_requests = args.stage_size * len(stages)
    static_items = [
        benchmark_item(f"static_pp{rank}", {0: PRESSURE_PP_BY_RANKS[rank]},
                       requests, total_requests)
        for rank in (1, 2, 3, 4)
    ]

    # The large unreachable keys expose all target configs as alternatives
    # without causing request-index migrations. The runtime policy decides when
    # to increment from config index 0 -> 1 -> 2 -> 3.
    autoscale_item = benchmark_item(
        "autoscale_kv_pressure",
        {
            0: PRESSURE_PP_BY_RANKS[1],
            100000: PRESSURE_PP_BY_RANKS[2],
            200000: PRESSURE_PP_BY_RANKS[3],
            300000: PRESSURE_PP_BY_RANKS[4],
        },
        requests,
        total_requests,
    )
    policy = {
        "type": "kv_pressure_incremental",
        "scale_up_threshold": args.threshold,
        "min_request_interval": max(4, args.stage_size // 2),
        "min_requests_before_scale": args.stage_size,
        "max_config_index": 3,
    }

    combined = build_config(
        project="l40_burstgpt_kv_pressure_autoscale",
        node=args.node,
        model_path=args.model_path,
        dataset_path=args.output_dataset.resolve(),
        total_requests=total_requests,
        benchmark_items=static_items + [autoscale_item],
        autoscaling_policy=policy,
    )
    write_yaml(args.output_config, combined)

    split_paths: list[Path] = []
    for item in static_items:
        cfg = build_config(
            project=f"l40_burstgpt_kv_pressure_{item['label']}",
            node=args.node,
            model_path=args.model_path,
            dataset_path=args.output_dataset.resolve(),
            total_requests=total_requests,
            benchmark_items=[item],
            autoscaling_policy=None,
        )
        path = args.output_split_dir / f"{item['label']}.yaml"
        write_yaml(path, cfg)
        split_paths.append(path)

    cfg = build_config(
        project="l40_burstgpt_kv_pressure_autoscale",
        node=args.node,
        model_path=args.model_path,
        dataset_path=args.output_dataset.resolve(),
        total_requests=total_requests,
        benchmark_items=[autoscale_item],
        autoscaling_policy=policy,
    )
    path = args.output_split_dir / "autoscale_kv_pressure.yaml"
    write_yaml(path, cfg)
    split_paths.append(path)

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BurstGPT KV-Pressure Autoscaling Design",
        "",
        "## Rule",
        "",
        "Runtime policy: read scheduler KV cache utilization on request arrival.",
        f"If utilization is at least {args.threshold:.2f}, increment one PP config: 1->2->3->4 active ranks.",
        "The policy is scale-up only for this experiment, so the workload must keep pressure high enough to cover all ranks.",
        "",
        "## ShareGPT Calibration",
        "",
        f"- capacity_unit_tokens_per_second: {calibration['capacity_unit_tokens_per_second']:.1f}",
        f"- total_tokens p50/p75/p90/p95: {calibration['total_tokens']['p50']:.1f} / {calibration['total_tokens']['p75']:.1f} / {calibration['total_tokens']['p90']:.1f} / {calibration['total_tokens']['p95']:.1f}",
        "",
        "## BurstGPT Pressure Ladder",
        "",
        "| boundary | stage | req/s | mean input | mean output | mean total | estimated pressure score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for idx, stage in enumerate(stages):
        lines.append(
            f"| {idx * args.stage_size} | {stage['name']} | "
            f"{stage['request_rate']:.2f} | {stage['mean_input']:.1f} | "
            f"{stage['mean_output']:.1f} | {stage['mean_total']:.1f} | "
            f"{stage['pressure_score']:.2f} |")
    lines.extend([
        "",
        "## Generated Artifacts",
        "",
        f"- combined config: `{args.output_config.resolve()}`",
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

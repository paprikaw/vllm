#!/usr/bin/env python3
"""Generate native-timestamp BurstGPT autoscaling configs."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


PP_BY_RANKS = {
    1: "64,0,0,0",
    2: "32,32,0,0",
    3: "16,32,16,0",
    4: "16,16,16,16",
}


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def load_burstgpt_window(
    path: Path,
    *,
    window_start: float,
    window_duration_s: float,
    max_total_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                if row["Model"] != "GPT-4":
                    continue
                timestamp = float(row["Timestamp"])
                request_tokens = int(float(row["Request tokens"]))
                response_tokens = int(float(row["Response tokens"]))
                total_tokens = int(float(row.get(
                    "Total tokens", request_tokens + response_tokens)))
            except (KeyError, TypeError, ValueError):
                continue
            if not (window_start <= timestamp < window_start + window_duration_s):
                continue
            if response_tokens <= 0 or total_tokens > max_total_tokens:
                continue
            rows.append({
                "Timestamp": row["Timestamp"],
                "Model": row["Model"],
                "Request tokens": str(request_tokens),
                "Response tokens": str(response_tokens),
                "Total tokens": str(total_tokens),
                "Log Type": row.get("Log Type", "Conversation log"),
                "_timestamp": timestamp,
                "_request_tokens": request_tokens,
                "_response_tokens": response_tokens,
                "_total_tokens": total_tokens,
            })
    rows.sort(key=lambda row: row["_timestamp"])
    if len(rows) < 2:
        raise ValueError("BurstGPT window must contain at least two requests.")
    return rows


def write_burstgpt_subset(path: Path, rows: list[dict[str, Any]]) -> None:
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
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def peak_window_rps(rows: list[dict[str, Any]],
                    time_scale_factor: float,
                    window_s: float) -> float:
    if not rows or window_s <= 0:
        return 0.0
    timestamps = [
        (row["_timestamp"] - rows[0]["_timestamp"]) / time_scale_factor
        for row in rows
    ]
    peak = 0
    right = 0
    for left, start in enumerate(timestamps):
        while right < len(timestamps) and timestamps[right] < start + window_s:
            right += 1
        peak = max(peak, right - left)
    return peak / window_s


def benchmark_item(
    *,
    label: str,
    pp_layer_config: dict[int, str],
    total_requests: int,
    request_rate: float,
    mean_input: float,
    mean_output: float,
    arrival_trace: dict[str, Any] | None,
) -> dict[str, Any]:
    item = {
        "label": label,
        "num_total_requests": total_requests,
        "repetition": 1,
        "pp_layer_config": pp_layer_config,
        "requests": {
            0: {
                "request_rate": request_rate,
                "input_lens": int(round(mean_input)),
                "output_lens": int(round(mean_output)),
            }
        },
    }
    if arrival_trace is not None:
        item["arrival_trace"] = copy.deepcopy(arrival_trace)
    return item


def build_config(
    *,
    project: str,
    node: str,
    model_path: str,
    dataset_path: Path,
    total_requests: int,
    benchmark_items: list[dict[str, Any]],
    autoscaling_policy: dict[str, Any] | None,
    max_model_len: int,
    max_num_batched_tokens: int,
    post_benchmark_wait_s: float,
) -> dict[str, Any]:
    vllm_static: dict[str, Any] = {
        "head_addr": "localhost",
        "pipeline_parallel_size": 4,
        "pipeline_autoscaling_enabled": True,
        "autoscaling_candidate_ranks": [0, 1, 2, 3],
        "gpu_memory_utilization": 0.85,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
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
            "VLLM_EXP_POST_BENCHMARK_WAIT_S": str(post_benchmark_wait_s),
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
                "dataset_path": str(dataset_path.resolve()),
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


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(payload, Dumper=NoAliasDumper, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--burstgpt", type=Path,
                        default=Path("/data/gpfs/projects/punim2715/datasets/BurstGPT_without_fails_2.csv"))
    parser.add_argument("--model-path",
                        default="/home/bxb1/data/huggingface/Qwen3-32B-FP8")
    parser.add_argument("--node", default="spartan-gpgpu068")
    parser.add_argument("--window-start", type=float, default=5539015.0)
    parser.add_argument("--window-duration-s", type=float, default=900.0)
    parser.add_argument("--time-scale-factor", type=float, default=8.0)
    parser.add_argument("--four-gpu-capacity-rps", type=float, default=None,
                        help=("If set, derive time_scale_factor from a "
                              "measured static-4GPU capacity instead of using "
                              "--time-scale-factor directly."))
    parser.add_argument("--capacity-headroom", type=float, default=0.85,
                        help=("Fraction of measured 4GPU capacity to target "
                              "when --four-gpu-capacity-rps is provided."))
    parser.add_argument("--max-replay-seconds", type=float, default=180.0)
    parser.add_argument("--max-total-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--scale-up-threshold", type=float, default=0.55)
    parser.add_argument("--scale-down-threshold", type=float, default=0.20)
    parser.add_argument("--post-benchmark-wait-s", type=float, default=45.0)
    parser.add_argument("--output-config", type=Path,
                        default=Path("vllm_exp/configs/generated/a100_burstgpt_native_autoscale.yaml"))
    parser.add_argument("--output-split-dir", type=Path,
                        default=Path("vllm_exp/configs/generated/a100_burstgpt_native_runs"))
    parser.add_argument("--output-dataset", type=Path,
                        default=Path("vllm_exp/generated_workloads/burstgpt_a100_native_900s.csv"))
    parser.add_argument("--output-report", type=Path,
                        default=Path("vllm_exp/generated_workloads/burstgpt_a100_native_900s_design_report.md"))
    args = parser.parse_args()

    rows = load_burstgpt_window(
        args.burstgpt,
        window_start=args.window_start,
        window_duration_s=args.window_duration_s,
        max_total_tokens=args.max_total_tokens,
    )
    write_burstgpt_subset(args.output_dataset, rows)

    timestamps = [row["_timestamp"] for row in rows]
    native_span = timestamps[-1] - timestamps[0]
    total_requests = len(rows)
    native_avg_rps = total_requests / native_span
    capacity_target_rps = None
    if args.four_gpu_capacity_rps is not None:
        if args.four_gpu_capacity_rps <= 0:
            raise ValueError("--four-gpu-capacity-rps must be positive.")
        if not (0 < args.capacity_headroom <= 1.0):
            raise ValueError("--capacity-headroom must be in (0, 1].")
        capacity_target_rps = args.four_gpu_capacity_rps * args.capacity_headroom
        args.time_scale_factor = capacity_target_rps / native_avg_rps

    replay_span = native_span / args.time_scale_factor
    if replay_span > args.max_replay_seconds:
        raise ValueError(
            f"replay span {replay_span:.3f}s exceeds {args.max_replay_seconds:.3f}s")

    mean_input = mean(row["_request_tokens"] for row in rows)
    mean_output = mean(row["_response_tokens"] for row in rows)
    arrival_trace = {
        "mode": "burstgpt_timestamp",
        "model_filter": "GPT-4",
        "source_window_start": args.window_start,
        "source_window_duration_s": args.window_duration_s,
        "time_scale_factor": args.time_scale_factor,
        "max_total_tokens": args.max_total_tokens,
        "max_replay_seconds": args.max_replay_seconds,
        "preserve_temporal_pattern": True,
    }
    autoscaling_policy = {
        "type": "kv_pressure_incremental",
        "scale_up_threshold": args.scale_up_threshold,
        "scale_down_threshold": args.scale_down_threshold,
        "min_requests_before_scale": 32,
        "min_request_interval": 32,
        "scale_down_min_request_interval": 0,
        "scale_down_stable_checks": 2,
        "scale_down_poll_interval_s": 5.0,
        "min_config_index": 0,
        "max_config_index": 3,
    }

    static_items = [
        benchmark_item(
            label=f"static_pp{rank}",
            pp_layer_config={0: PP_BY_RANKS[rank]},
            total_requests=total_requests,
            request_rate=0.0,
            mean_input=mean_input,
            mean_output=mean_output,
            arrival_trace=arrival_trace,
        )
        for rank in (1, 2, 3, 4)
    ]
    autoscale_item = benchmark_item(
        label="autoscale_native_burstgpt",
        pp_layer_config={
            0: PP_BY_RANKS[1],
            100000: PP_BY_RANKS[2],
            200000: PP_BY_RANKS[3],
            300000: PP_BY_RANKS[4],
        },
        total_requests=total_requests,
        request_rate=0.0,
        mean_input=mean_input,
        mean_output=mean_output,
        arrival_trace=arrival_trace,
    )

    combined = build_config(
        project="a100_burstgpt_native_autoscale",
        node=args.node,
        model_path=args.model_path,
        dataset_path=args.output_dataset,
        total_requests=total_requests,
        benchmark_items=static_items + [autoscale_item],
        autoscaling_policy=autoscaling_policy,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        post_benchmark_wait_s=args.post_benchmark_wait_s,
    )
    write_yaml(args.output_config, combined)

    split_paths: list[Path] = []
    for item in static_items:
        cfg = build_config(
            project=f"a100_burstgpt_native_{item['label']}",
            node=args.node,
            model_path=args.model_path,
            dataset_path=args.output_dataset,
            total_requests=total_requests,
            benchmark_items=[item],
            autoscaling_policy=None,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            post_benchmark_wait_s=args.post_benchmark_wait_s,
        )
        path = args.output_split_dir / f"{item['label']}.yaml"
        write_yaml(path, cfg)
        split_paths.append(path)

    cfg = build_config(
        project="a100_burstgpt_native_autoscale",
        node=args.node,
        model_path=args.model_path,
        dataset_path=args.output_dataset,
        total_requests=total_requests,
        benchmark_items=[autoscale_item],
        autoscaling_policy=autoscaling_policy,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        post_benchmark_wait_s=args.post_benchmark_wait_s,
    )
    path = args.output_split_dir / "autoscale_native_burstgpt.yaml"
    write_yaml(path, cfg)
    split_paths.append(path)

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        "\n".join([
            "# A100 Native BurstGPT Autoscaling Design",
            "",
            f"- source trace: `{args.burstgpt}`",
            f"- selected native window: [{args.window_start:.0f}, {args.window_start + args.window_duration_s:.0f})",
            f"- requests: {total_requests}",
            f"- native span: {native_span:.3f}s",
            f"- native average RPS: {native_avg_rps:.3f}",
            f"- time scale factor: {args.time_scale_factor:.3f}",
            f"- replay span: {replay_span:.3f}s",
            f"- replay average RPS: {total_requests / replay_span:.3f}",
            f"- replay peak 10s RPS: {peak_window_rps(rows, args.time_scale_factor, 10.0):.3f}",
            f"- replay peak 30s RPS: {peak_window_rps(rows, args.time_scale_factor, 30.0):.3f}",
            f"- replay peak 60s RPS: {peak_window_rps(rows, args.time_scale_factor, 60.0):.3f}",
            f"- max replay seconds: {args.max_replay_seconds:.3f}",
            f"- four_gpu_capacity_rps: {args.four_gpu_capacity_rps if args.four_gpu_capacity_rps is not None else 'not set'}",
            f"- capacity_headroom: {args.capacity_headroom:.3f}",
            f"- capacity_target_rps: {capacity_target_rps if capacity_target_rps is not None else 'not set'}",
            "- capacity regulation: preserve BurstGPT timestamps and scale only RPS to fit the measured 4GPU serving capacity.",
            f"- total input tokens: {sum(row['_request_tokens'] for row in rows)}",
            f"- total output tokens: {sum(row['_response_tokens'] for row in rows)}",
            f"- max total tokens/request: {max(row['_total_tokens'] for row in rows)}",
            f"- max_model_len: {args.max_model_len}",
            f"- max_num_batched_tokens: {args.max_num_batched_tokens}",
            f"- scale_up_threshold: {args.scale_up_threshold:.3f}",
            f"- scale_down_threshold: {args.scale_down_threshold:.3f}",
            f"- post_benchmark_wait_s: {args.post_benchmark_wait_s:.3f}",
            "",
            "## Generated Artifacts",
            "",
            f"- combined config: `{args.output_config.resolve()}`",
            f"- split config dir: `{args.output_split_dir.resolve()}`",
            f"- ordered BurstGPT subset: `{args.output_dataset.resolve()}`",
            f"- split configs: {', '.join(str(path.resolve()) for path in split_paths)}",
        ]) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote config: {args.output_config}")
    print(f"Wrote split configs: {args.output_split_dir}")
    print(f"Wrote dataset: {args.output_dataset}")
    print(f"Wrote report: {args.output_report}")
    print(f"Replay span: {replay_span:.3f}s ({total_requests} requests)")


if __name__ == "__main__":
    main()

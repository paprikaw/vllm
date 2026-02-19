# SPDX-License-Identifier: Apache-2.0
r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""

import argparse
import asyncio
import gc
import json
import os
import random
import shutil
import time
import warnings
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Tuple
import csv

import numpy as np
from pydantic import BaseModel
from vllm_exp.data import BenchCfg 
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
)

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from benchmark_dataset import (
    AIMODataset,
    ASRDataset,
    BurstGPTDataset,
    ConversationDataset,
    HuggingFaceDataset,
    InstructCoderDataset,
    MTBenchDataset,
    NextEditPredictionDataset,
    RandomDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    VisionArenaDataset,
)
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from dynamic_benchmark_dataset import PatternDataset
MILLISECONDS_TO_SECONDS_CONVERSION = 1000

_BENCH_PHASE_BOUNDARY_TOKEN = "=== VLLM_EXP_BENCH_PHASE_BOUNDARY ==="



@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[SampleRequest, None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a SampleRequest.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    input_requests: Iterable[SampleRequest] = iter(input_requests)

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
    metrics_file_name: Optional[str] = None,
    repetition_index: Optional[int] = None,
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    all_itls: list[list[float]] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        # Skip Exception objects that were added as placeholders
        if isinstance(outputs[i], Exception):
            actual_output_lens.append(0)
            continue
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(
                        outputs[i].generated_text, add_special_tokens=False
                    ).input_ids
                )
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            tpots.append(tpot)
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            all_itls.append(outputs[i].itl)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    # 将以上这些metrics作为csv输出到文件中
    file_name = metrics_file_name
    if file_name is None:
        try:
            file_name = os.environ["METRICS_FILE_NAME"]
        except KeyError:
            file_name = None
    if file_name:
        print(f"Writing metrics to {file_name}")
    
    if file_name:
        # 从 outputs 中提取时间戳（仅记录成功请求，避免行数/字段不一致）
        from datetime import datetime
        success_outputs = [o for o in outputs if o.success]
        timestamps = [output.timestamp for output in success_outputs]
        datetimes = [
            datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            for ts in timestamps
        ]

        # When repetition_index is provided:
        #   - repetition_index == 1: write mode with header (first repetition)
        #   - repetition_index > 1: append mode without header
        # When repetition_index is None: write mode with header (single run)
        if repetition_index is not None and repetition_index > 1:
            # Append mode for subsequent repetitions
            write_mode = "a"
            write_header = False
        else:
            # Write mode for first repetition or single run
            write_mode = "w"
            write_header = True
        
        with open(file_name, write_mode, newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "datetime", "tpots", "ttfts", "e2els", "repetition"])
            rep_val = repetition_index if repetition_index is not None else 1
            for ts, dt, t, tf, e in zip(timestamps, datetimes, tpots, ttfts, e2els):
                writer.writerow([ts, dt, t, tf, e, rep_val])
    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
    )

    return metrics, actual_output_lens


async def set_pp_config(
    base_url: str,
    pp_layer_config: list,
    alternative_configs: Optional[dict] = None,
    migration_steps: Optional[list] = None,
    migration_mode: Optional[str] = None,
    timeout: float = 120.0
) -> bool:
    """Set pipeline configuration to a specific target config.
    
    The server will skip if already at the target configuration.
    Note: set_pp_config always uses sync migration internally for the
    reset/restore operation. The migration_mode parameter only updates
    the server's migration_mode setting for subsequent migrations
    triggered by the migration_thread during the benchmark run.
    
    Args:
        base_url: Base URL of the vLLM server (e.g., http://localhost:8000)
        pp_layer_config: Target configuration as list of [start, end] pairs per rank.
                         Example: [[0, 39], [40, 63]] for 2 ranks.
        migration_mode: Optional migration mode ('sync' or 'async') to set on the
                       server for use by migration_thread. Does NOT affect the
                       set_pp_config reset migration itself (always sync).
        timeout: Request timeout in seconds
    
    Returns:
        True if set was successful, False otherwise
    """
    import aiohttp
    set_url = f"{base_url}/set_pp_config"
    payload = {"pp_layer_config": pp_layer_config}
    if alternative_configs is not None:
        payload["alternative_configs"] = alternative_configs
    if migration_steps is not None:
        payload["migration_steps"] = migration_steps
    if migration_mode is not None:
        payload["migration_mode"] = migration_mode
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(set_url, json=payload, 
                                    timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status == 200:
                    print(f"  PP config set successfully to {pp_layer_config}")
                    return True
                else:
                    try:
                        error_body = await response.json()
                        error_msg = error_body.get('error', 'Unknown error')
                    except Exception:
                        error_msg = await response.text()
                    print(f"  Warning: set_pp_config failed with status {response.status}")
                    print(f"  Error details: {error_msg}")
                    return False
    except Exception as e:
        print(f"  Warning: Failed to set pp config: {e}")
        return False


def _print_repetition_summary(rep: int, metrics: BenchmarkMetrics) -> None:
    """Print detailed summary for a single repetition."""
    duration = metrics.completed / metrics.request_throughput if metrics.request_throughput > 0 else 0
    
    print(f"\n{'─' * 60}")
    print(f"REPETITION {rep} SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Requests:           {metrics.completed} completed")
    print(f"  Duration:           {duration:.2f}s")
    print(f"  Total Input:        {metrics.total_input} tokens")
    print(f"  Total Output:       {metrics.total_output} tokens")
    print()
    print(f"  Throughput:")
    print(f"    Request:          {metrics.request_throughput:.2f} req/s")
    print(f"    Output:           {metrics.output_throughput:.2f} tokens/s")
    print(f"    Total Token:      {metrics.total_token_throughput:.2f} tokens/s")
    print()
    print(f"  TTFT (Time To First Token):")
    print(f"    Mean:             {metrics.mean_ttft_ms:.2f} ms")
    print(f"    Median:           {metrics.median_ttft_ms:.2f} ms")
    print(f"    Std Dev:          {metrics.std_ttft_ms:.2f} ms")
    print()
    print(f"  TPOT (Time Per Output Token):")
    print(f"    Mean:             {metrics.mean_tpot_ms:.2f} ms")
    print(f"    Median:           {metrics.median_tpot_ms:.2f} ms")
    print(f"    Std Dev:          {metrics.std_tpot_ms:.2f} ms")
    print()
    print(f"  ITL (Inter-Token Latency):")
    print(f"    Mean:             {metrics.mean_itl_ms:.2f} ms")
    print(f"    Median:           {metrics.median_itl_ms:.2f} ms")
    print(f"    Std Dev:          {metrics.std_itl_ms:.2f} ms")
    print()
    print(f"  E2EL (End-to-End Latency):")
    print(f"    Mean:             {metrics.mean_e2el_ms:.2f} ms")
    print(f"    Median:           {metrics.median_e2el_ms:.2f} ms")
    print(f"    Std Dev:          {metrics.std_e2el_ms:.2f} ms")


def _print_final_summary(repetition: int, all_metrics_list: list[BenchmarkMetrics]) -> None:
    """Print final summary with avg, median, std dev across all repetitions."""
    n = len(all_metrics_list)
    
    # Extract metrics arrays
    throughputs = [m.request_throughput for m in all_metrics_list]
    output_throughputs = [m.output_throughput for m in all_metrics_list]
    total_token_throughputs = [m.total_token_throughput for m in all_metrics_list]
    
    mean_ttfts = [m.mean_ttft_ms for m in all_metrics_list]
    mean_tpots = [m.mean_tpot_ms for m in all_metrics_list]
    mean_itls = [m.mean_itl_ms for m in all_metrics_list]
    mean_e2els = [m.mean_e2el_ms for m in all_metrics_list]
    
    total_completed = sum(m.completed for m in all_metrics_list)
    total_input = sum(m.total_input for m in all_metrics_list)
    total_output = sum(m.total_output for m in all_metrics_list)
    
    print(f"\n{'═' * 70}")
    print(f"{'═' * 70}")
    print(f"  FINAL SUMMARY: {repetition} REPETITIONS COMPLETED")
    print(f"{'═' * 70}")
    print(f"{'═' * 70}")
    
    # Per-repetition breakdown
    print(f"\n┌{'─' * 68}┐")
    print(f"│ {'Rep':>4} │ {'Completed':>10} │ {'Throughput':>12} │ {'Mean TTFT':>12} │ {'Mean TPOT':>12} │")
    print(f"├{'─' * 68}┤")
    for i, m in enumerate(all_metrics_list, 1):
        print(f"│ {i:>4} │ {m.completed:>10} │ {m.request_throughput:>10.2f}/s │ {m.mean_ttft_ms:>10.2f}ms │ {m.mean_tpot_ms:>10.2f}ms │")
    print(f"└{'─' * 68}┘")
    
    # Aggregate statistics
    print(f"\n{'─' * 70}")
    print(f"  AGGREGATE STATISTICS ACROSS {repetition} REPETITIONS")
    print(f"{'─' * 70}")
    print(f"  Total Requests:     {total_completed}")
    print(f"  Total Input:        {total_input} tokens")
    print(f"  Total Output:       {total_output} tokens")
    
    # Throughput statistics
    print(f"\n  REQUEST THROUGHPUT (req/s):")
    print(f"    Average:          {np.mean(throughputs):.2f}")
    print(f"    Median:           {np.median(throughputs):.2f}")
    print(f"    Std Dev:          {np.std(throughputs):.2f}")
    print(f"    Min:              {np.min(throughputs):.2f}")
    print(f"    Max:              {np.max(throughputs):.2f}")
    
    print(f"\n  OUTPUT THROUGHPUT (tokens/s):")
    print(f"    Average:          {np.mean(output_throughputs):.2f}")
    print(f"    Median:           {np.median(output_throughputs):.2f}")
    print(f"    Std Dev:          {np.std(output_throughputs):.2f}")
    
    print(f"\n  TOTAL TOKEN THROUGHPUT (tokens/s):")
    print(f"    Average:          {np.mean(total_token_throughputs):.2f}")
    print(f"    Median:           {np.median(total_token_throughputs):.2f}")
    print(f"    Std Dev:          {np.std(total_token_throughputs):.2f}")
    
    # Latency statistics
    print(f"\n  MEAN TTFT ACROSS REPETITIONS (ms):")
    print(f"    Average:          {np.mean(mean_ttfts):.2f}")
    print(f"    Median:           {np.median(mean_ttfts):.2f}")
    print(f"    Std Dev:          {np.std(mean_ttfts):.2f}")
    print(f"    Min:              {np.min(mean_ttfts):.2f}")
    print(f"    Max:              {np.max(mean_ttfts):.2f}")
    
    print(f"\n  MEAN TPOT ACROSS REPETITIONS (ms):")
    print(f"    Average:          {np.mean(mean_tpots):.2f}")
    print(f"    Median:           {np.median(mean_tpots):.2f}")
    print(f"    Std Dev:          {np.std(mean_tpots):.2f}")
    print(f"    Min:              {np.min(mean_tpots):.2f}")
    print(f"    Max:              {np.max(mean_tpots):.2f}")
    
    print(f"\n  MEAN ITL ACROSS REPETITIONS (ms):")
    print(f"    Average:          {np.mean(mean_itls):.2f}")
    print(f"    Median:           {np.median(mean_itls):.2f}")
    print(f"    Std Dev:          {np.std(mean_itls):.2f}")
    
    print(f"\n  MEAN E2EL ACROSS REPETITIONS (ms):")
    print(f"    Average:          {np.mean(mean_e2els):.2f}")
    print(f"    Median:           {np.median(mean_e2els):.2f}")
    print(f"    Std Dev:          {np.std(mean_e2els):.2f}")
    
    print(f"\n{'═' * 70}")


async def run_repetition_benchmark(
    repetition: int,
    base_url: str,
    run_single_benchmark_func,
    initial_pp_config: Optional[list] = None,
    alternative_configs: Optional[dict] = None,
    migration_steps: Optional[list] = None,
    migration_mode: Optional[str] = None,
    **kwargs
) -> tuple[BenchmarkMetrics, list[int]]:
    """Run benchmark multiple times with pipeline config reset between repetitions.
    
    Args:
        repetition: Number of times to repeat the benchmark
        base_url: Base URL of the vLLM server for set_pp_config API
        run_single_benchmark_func: Function that runs a single benchmark iteration
        initial_pp_config: Initial pipeline config to restore between repetitions.
                          If None, skip the reset (server will handle same-config skip).
        alternative_configs: Optional migration target configs for set_pp_config.
        migration_steps: Optional migration trigger points (request indices).
        migration_mode: Optional migration mode ('sync' or 'async') for migration_thread.
        **kwargs: Arguments to pass to run_single_benchmark_func
    
    Returns:
        Aggregated metrics from all repetitions and output lens
    """
    print(f"\n{'═' * 70}")
    print(f"  STARTING BENCHMARK WITH {repetition} REPETITIONS")
    print(f"{'═' * 70}")
    
    all_metrics_list = []
    all_output_lens = []
    
    for rep in range(1, repetition + 1):
        print(f"\n{'#' * 70}")
        print(f"###################### REPETITION {rep}/{repetition} ######################")
        print(f"{'#' * 70}")
        
        # Run single benchmark (add base_url and repetition_index to kwargs for run_multi_stage_benchmark)
        kwargs_with_base_url = {**kwargs, 'base_url': base_url, 'repetition_index': rep}
        metrics, output_lens = await run_single_benchmark_func(**kwargs_with_base_url)
        all_metrics_list.append(metrics)
        all_output_lens.extend(output_lens)
        
        # Print detailed summary for this repetition
        _print_repetition_summary(rep, metrics)
        
        # Reset pipeline config if not the last repetition
        if rep < repetition:
            if initial_pp_config is not None:
                print(f"\n  Resetting pipeline configuration to {initial_pp_config} for next repetition...")
                success = await set_pp_config(
                    base_url,
                    initial_pp_config,
                    alternative_configs=alternative_configs,
                    migration_steps=migration_steps,
                    migration_mode=migration_mode
                )
                if not success:
                    print(f"  Warning: Pipeline config reset failed, continuing anyway...")
                # Small delay after reset to ensure server is ready
                await asyncio.sleep(1.0)
            else:
                print(f"\n  No initial_pp_config provided, skipping pipeline reset...")
    
    # Print final summary with statistics across all repetitions
    _print_final_summary(repetition, all_metrics_list)
    
    # Aggregate metrics from all repetitions
    if len(all_metrics_list) == 1:
        return all_metrics_list[0], all_output_lens
    
    # Calculate aggregated metrics
    total_completed = sum(m.completed for m in all_metrics_list)
    total_input = sum(m.total_input for m in all_metrics_list)
    total_output = sum(m.total_output for m in all_metrics_list)
    
    # Use average for rate metrics
    avg_request_throughput = np.mean([m.request_throughput for m in all_metrics_list])
    avg_request_goodput = np.mean([m.request_goodput for m in all_metrics_list])
    avg_output_throughput = np.mean([m.output_throughput for m in all_metrics_list])
    avg_total_token_throughput = np.mean([m.total_token_throughput for m in all_metrics_list])
    
    # For latency metrics, compute proper statistics across all repetitions
    avg_mean_ttft = np.mean([m.mean_ttft_ms for m in all_metrics_list])
    avg_median_ttft = np.median([m.mean_ttft_ms for m in all_metrics_list])
    avg_std_ttft = np.std([m.mean_ttft_ms for m in all_metrics_list])
    
    avg_mean_tpot = np.mean([m.mean_tpot_ms for m in all_metrics_list])
    avg_median_tpot = np.median([m.mean_tpot_ms for m in all_metrics_list])
    avg_std_tpot = np.std([m.mean_tpot_ms for m in all_metrics_list])
    
    avg_mean_itl = np.mean([m.mean_itl_ms for m in all_metrics_list])
    avg_median_itl = np.median([m.mean_itl_ms for m in all_metrics_list])
    avg_std_itl = np.std([m.mean_itl_ms for m in all_metrics_list])
    
    avg_mean_e2el = np.mean([m.mean_e2el_ms for m in all_metrics_list])
    avg_median_e2el = np.median([m.mean_e2el_ms for m in all_metrics_list])
    avg_std_e2el = np.std([m.mean_e2el_ms for m in all_metrics_list])
    
    # For percentiles, take the average across repetitions
    avg_percentiles_ttft = [(p, np.mean([m.percentiles_ttft_ms[i][1] for m in all_metrics_list])) 
                           for i, (p, _) in enumerate(all_metrics_list[0].percentiles_ttft_ms)]
    avg_percentiles_tpot = [(p, np.mean([m.percentiles_tpot_ms[i][1] for m in all_metrics_list])) 
                           for i, (p, _) in enumerate(all_metrics_list[0].percentiles_tpot_ms)]
    avg_percentiles_itl = [(p, np.mean([m.percentiles_itl_ms[i][1] for m in all_metrics_list])) 
                          for i, (p, _) in enumerate(all_metrics_list[0].percentiles_itl_ms)]
    avg_percentiles_e2el = [(p, np.mean([m.percentiles_e2el_ms[i][1] for m in all_metrics_list])) 
                           for i, (p, _) in enumerate(all_metrics_list[0].percentiles_e2el_ms)]
    
    aggregated_metrics = BenchmarkMetrics(
        completed=total_completed,
        total_input=total_input,
        total_output=total_output,
        request_throughput=float(avg_request_throughput),
        request_goodput=float(avg_request_goodput),
        output_throughput=float(avg_output_throughput),
        total_token_throughput=float(avg_total_token_throughput),
        mean_ttft_ms=float(avg_mean_ttft),
        median_ttft_ms=float(avg_median_ttft),
        std_ttft_ms=float(avg_std_ttft),
        percentiles_ttft_ms=avg_percentiles_ttft,
        mean_tpot_ms=float(avg_mean_tpot),
        median_tpot_ms=float(avg_median_tpot),
        std_tpot_ms=float(avg_std_tpot),
        percentiles_tpot_ms=avg_percentiles_tpot,
        mean_itl_ms=float(avg_mean_itl),
        median_itl_ms=float(avg_median_itl),
        std_itl_ms=float(avg_std_itl),
        percentiles_itl_ms=avg_percentiles_itl,
        mean_e2el_ms=float(avg_mean_e2el),
        median_e2el_ms=float(avg_median_e2el),
        std_e2el_ms=float(avg_std_e2el),
        percentiles_e2el_ms=avg_percentiles_e2el,
    )
    
    return aggregated_metrics, all_output_lens


async def run_multi_stage_benchmark(
    request_func,
    input_requests: list[SampleRequest],
    request_rate_list: list[float],
    running_num_requests: list[int],
    burstiness: float,
    disable_tqdm: bool,
    max_concurrency: Optional[int],
    lora_modules: Optional[Iterable[str]],
    model_id: str,
    model_name: str,
    api_url: str,
    logprobs: Optional[int],
    ignore_eos: bool,
    extra_body: Optional[dict],
    profile: bool,
    base_url: str,
    test_prompt: str,
    test_prompt_len: int,
    test_output_len: int,
    test_mm_content: Optional[dict],
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
    print_outputs: bool,
    warmup_stage_count: int = 0,
    repetition_index: Optional[int] = None,
):
    """
    执行多阶段基准测试，每个阶段使用不同的请求速率和请求数量
    
    Args:
        repetition_index: Current repetition number (1-indexed). If None, treated as single run.
    """
    
    # Execute stages sequentially: wait warmup fully completes before main.
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    warmup_start_perf = time.perf_counter()

    # 计算每个阶段的起始索引
    start_indices = [0]
    for num_req in running_num_requests[:-1]:
        start_indices.append(start_indices[-1] + num_req)

    warmup_end_wall_ts: Optional[float] = None
    warmup_end_perf: Optional[float] = None

    tasks: list[asyncio.Task] = []
    for stage_idx, (request_rate, num_req) in enumerate(zip(request_rate_list, running_num_requests)):
        print(f"\n{'='*20} Stage {stage_idx + 1} {'='*20}")
        print(f"Request rate: {request_rate} req/s")
        print(f"Number of requests: {num_req}")

        start_idx = start_indices[stage_idx]
        end_idx = start_idx + num_req
        stage_requests = input_requests[start_idx:end_idx]

        stage_lora_modules = None
        if lora_modules:
            stage_lora_modules = iter(
                [random.choice(list(lora_modules)) for _ in range(len(stage_requests))]
            )

        stage_tasks: list[asyncio.Task] = []
        async for request in get_request(stage_requests, request_rate, burstiness):
            prompt, prompt_len, output_len, mm_content = (
                request.prompt,
                request.prompt_len,
                request.expected_output_len,
                request.multi_modal_data,
            )

            req_model_id, req_model_name = model_id, model_name
            if stage_lora_modules:
                req_lora_module = next(stage_lora_modules)
                req_model_id, req_model_name = req_lora_module, req_lora_module

            request_func_input = RequestFuncInput(
                model=req_model_id,
                model_name=req_model_name,
                prompt=prompt,
                api_url=api_url,
                prompt_len=prompt_len,
                output_len=output_len,
                logprobs=logprobs,
                multi_modal_content=mm_content,
                ignore_eos=ignore_eos,
                extra_body=extra_body,
            )
            t = asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
            stage_tasks.append(t)
            tasks.append(t)

        # Wait for this stage to fully finish before continuing.
        if stage_tasks:
            await asyncio.gather(*stage_tasks, return_exceptions=True)

        # Emit boundary marker after warmup stage fully completes.
        if warmup_stage_count > 0 and (stage_idx + 1) == warmup_stage_count:
            warmup_end_wall_ts = time.time()
            warmup_end_perf = time.perf_counter()
            # IMPORTANT: stdout is piped when invoked by vllm_exp; without an
            # explicit flush, Python may buffer this line and delay the
            # warmup→main boundary detection until process termination.
            print(
                f"{_BENCH_PHASE_BOUNDARY_TOKEN} boundary_ts={warmup_end_wall_ts:.6f}",
                flush=True,
            )

    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks, return_exceptions=True)

    # Verify outputs length matches expectations
    expected_total = sum(running_num_requests)
    if len(outputs) != expected_total:
        print(f"WARNING: Expected {expected_total} outputs but got {len(outputs)}")
        print(f"Tasks created: {len(tasks)}, Outputs received: {len(outputs)}")
        print(f"Running num requests per stage: {running_num_requests}")
        # Pad with None if outputs are missing to prevent IndexError
        while len(outputs) < expected_total:
            outputs.append(Exception("Task result missing"))

    # Optionally print per-request outputs, grouped by stage
    if print_outputs:
        for s_idx, num_req in enumerate(running_num_requests):
            s = start_indices[s_idx]
            e = s + num_req
            stage_total_output_tokens = 0
            for i in range(s, e):
                if i >= len(outputs):
                    print(f"[Stage {s_idx + 1}] Request {i - s}: MISSING (index {i} >= {len(outputs)})")
                    continue
                out = outputs[i]
                rel_i = i - s
                # Handle both exceptions and RequestFuncOutput objects
                if isinstance(out, Exception):
                    print(f"[Stage {s_idx + 1}] Request {rel_i}: EXCEPTION: {out}")
                elif out.success:
                    # Accumulate output token count with tokenizer fallback
                    output_len = len(
                        tokenizer(out.generated_text, add_special_tokens=False).input_ids
                    )
                    print(f"[Stage {s_idx + 1}] Request {rel_i}: output len: {output_len}, output text: {out.generated_text}")
                    stage_total_output_tokens += output_len
                else:
                    print(f"[Stage {s_idx + 1}] Request {rel_i}: ERROR: {out.error}")
            print(f"[Stage {s_idx + 1}] Total generated tokens: {stage_total_output_tokens}")
        
    if pbar is not None:
        pbar.close()

    total_duration = time.perf_counter() - warmup_start_perf

    # Compute full-run metrics
    # If no warmup, write to metrics_file_name. Otherwise, write to separate phase files.
    full_run_metrics_file = "" if warmup_stage_count > 0 else os.environ.get("METRICS_FILE_NAME")
    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=total_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
        # Write request_metrics for full run if warmup is disabled.
        # Otherwise, warmup/main are written to separate phase files.
        metrics_file_name=full_run_metrics_file,
        repetition_index=repetition_index,
    )
        
    def _print_latency_block(m: BenchmarkMetrics) -> None:
        def process_one_metric(
            metric_attribute_name: str,
            metric_name: str,
            metric_header: str,
        ) -> None:
            """Print stats for one latency metric."""
            if metric_attribute_name not in selected_percentile_metrics:
                return
            print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
            print(
                "{:<40} {:<10.2f}".format(
                    f"Mean {metric_name} (ms):",
                    getattr(m, f"mean_{metric_attribute_name}_ms"),
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    f"Median {metric_name} (ms):",
                    getattr(m, f"median_{metric_attribute_name}_ms"),
                )
            )
            for p, value in getattr(m, f"percentiles_{metric_attribute_name}_ms"):
                p_word = str(int(p)) if int(p) == p else str(p)
                print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))

        process_one_metric("ttft", "TTFT", "Time to First Token")
        process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
        process_one_metric("itl", "ITL", "Inter-token Latency")
        process_one_metric("e2el", "E2EL", "End-to-end Latency")

    # Print summary for full run
    print("{s:{c}^{n}}".format(s=" Benchmark Result (Total) ", n=50, c="-"))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Duration (s):", total_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    if goodput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))
    _print_latency_block(metrics)
        
    # Optionally compute and print warmup/main split metrics.
    warmup_metrics = None
    main_metrics = None
    warmup_count = 0
    if warmup_stage_count > 0:
        warmup_count = sum(running_num_requests[:warmup_stage_count])
        warmup_inputs = input_requests[:warmup_count]
        warmup_outputs = outputs[:warmup_count]
        main_inputs = input_requests[warmup_count:]
        main_outputs = outputs[warmup_count:]

        # Determine phase durations based on each phase's actual request window
        # (start_time .. start_time+latency) to make split stats stable even if
        # warmup/main overlap in flight.
        def _phase_duration(phase_outputs: list[RequestFuncOutput]) -> float:
            oks = [o for o in phase_outputs if o.success]
            if not oks:
                return 0.0
            start = min(o.start_time for o in oks)
            end = max(o.start_time + o.latency for o in oks)
            return max(0.0, end - start)

        warmup_dur = _phase_duration(warmup_outputs)
        main_dur = _phase_duration(main_outputs)

        warmup_file = os.environ.get("METRICS_FILE_NAME_WARMUP")
        main_file = os.environ.get("METRICS_FILE_NAME_MAIN") or os.environ.get("METRICS_FILE_NAME")

        warmup_metrics, _ = calculate_metrics(
            input_requests=warmup_inputs,
            outputs=warmup_outputs,
            dur_s=warmup_dur,
            tokenizer=tokenizer,
            selected_percentile_metrics=selected_percentile_metrics,
            selected_percentiles=selected_percentiles,
            goodput_config_dict=goodput_config_dict,
            metrics_file_name=warmup_file,
            repetition_index=repetition_index,
        )
        main_metrics, _ = calculate_metrics(
            input_requests=main_inputs,
            outputs=main_outputs,
            dur_s=main_dur,
            tokenizer=tokenizer,
            selected_percentile_metrics=selected_percentile_metrics,
            selected_percentiles=selected_percentiles,
            goodput_config_dict=goodput_config_dict,
            metrics_file_name=main_file,
            repetition_index=repetition_index,
        )

        def _print_block(title: str, m: BenchmarkMetrics, dur: float):
            print("{s:{c}^{n}}".format(s=title, n=50, c="="))
            print("{:<40} {:<10}".format("Successful requests:", m.completed))
            print("{:<40} {:<10.2f}".format("Duration (s):", dur))
            print("{:<40} {:<10}".format("Total input tokens:", m.total_input))
            print("{:<40} {:<10}".format("Total generated tokens:", m.total_output))
            print("{:<40} {:<10.2f}".format("Request throughput (req/s):", m.request_throughput))
            if goodput_config_dict:
                print("{:<40} {:<10.2f}".format("Request goodput (req/s):", m.request_goodput))
            print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", m.output_throughput))
            print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", m.total_token_throughput))
            _print_latency_block(m)

        _print_block(" Warmup Benchmark Result ", warmup_metrics, warmup_dur)
        _print_block(" Main Benchmark Result ", main_metrics, main_dur)

    # Return structured result for callers.
    result = {
        "total": {
            "duration": total_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "request_throughput": metrics.request_throughput,
            "request_goodput": metrics.request_goodput if goodput_config_dict else None,
            "output_throughput": metrics.output_throughput,
            "total_token_throughput": metrics.total_token_throughput,
        },
        "warmup": None,
        "main": None,
        "warmup_boundary_ts": warmup_end_wall_ts,
        "warmup_request_count": warmup_count,
    }
    if warmup_metrics is not None:
        result["warmup"] = {
            "completed": warmup_metrics.completed,
            "total_input_tokens": warmup_metrics.total_input,
            "total_output_tokens": warmup_metrics.total_output,
            "request_throughput": warmup_metrics.request_throughput,
            "request_goodput": warmup_metrics.request_goodput if goodput_config_dict else None,
            "output_throughput": warmup_metrics.output_throughput,
            "total_token_throughput": warmup_metrics.total_token_throughput,
        }
    if main_metrics is not None:
        result["main"] = {
            "completed": main_metrics.completed,
            "total_input_tokens": main_metrics.total_input,
            "total_output_tokens": main_metrics.total_output,
            "request_throughput": main_metrics.request_throughput,
            "request_goodput": main_metrics.request_goodput if goodput_config_dict else None,
            "output_throughput": main_metrics.output_throughput,
            "total_token_throughput": main_metrics.total_token_throughput,
        }
        
    print("-" * 50)
    
    # 停止分析器（如果启用）
    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")
    
    # # 计算总体指标
    # total_duration = sum(result["duration"] for result in stage_results)
    # total_completed = sum(result["completed"] for result in stage_results)
    # total_input_tokens = sum(result["total_input_tokens"] for result in stage_results)
    # total_output_tokens = sum(result["total_output_tokens"] for result in stage_results)
    
    # # 打印总体结果摘要
    # print("\n" + "="*60)
    # print("MULTI-STAGE BENCHMARK SUMMARY")
    # print("="*60)
    # print(f"{'Stage':<6} {'Rate':<10} {'Requests':<10} {'Duration':<12} {'Throughput':<15} {'Success':<10}")
    # print("-" * 60)
    
    # for result in stage_results:
    #     print(f"{result['stage']:<6} {result['request_rate']:<10.1f} {result['num_requests']:<10} "
    #           f"{result['duration']:<12.2f} {result['request_throughput']:<15.2f} {result['completed']:<10}")
    
    # print("-" * 60)
    # print(f"{'Total':<6} {'-':<10} {sum(r['num_requests'] for r in stage_results):<10} "
    #       f"{total_duration:<12.2f} {total_completed/total_duration:<15.2f} {total_completed:<10}")
    
    # 返回总体结果
    
    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest],
    logprobs: Optional[int],
    request_rate: float,
    compact_kv_request_rate_list: list[float],
    running_num_requests: list[int],
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: Optional[int],
    lora_modules: Optional[Iterable[str]],
    extra_body: Optional[dict],
    print_outputs: bool,
    warmup_stage_count: int = 0,
    repetition: int = 1,
    initial_pp_config: Optional[list] = None,
    alternative_configs: Optional[dict] = None,
    migration_steps: Optional[list] = None,
    migration_mode: Optional[str] = None,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0].prompt,
        input_requests[0].prompt_len,
        input_requests[0].expected_output_len,
        input_requests[0].multi_modal_data,
    )

    assert test_mm_content is None or isinstance(test_mm_content, dict)
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
        extra_body=extra_body,
    )

    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")

    if lora_modules:
        # For each input request, choose a LoRA module at random.
        lora_modules = iter(
            [random.choice(lora_modules) for _ in range(len(input_requests))]
        )

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    # 检查是否使用多阶段基准测试
    print(f"compact list {compact_kv_request_rate_list}")
    print(f"running requests number:{running_num_requests}")
    if compact_kv_request_rate_list and running_num_requests:
        assert len(compact_kv_request_rate_list) == len(running_num_requests), "request_rate_list and num_requests must have the same length"
        for i, (rate, num_req) in enumerate(zip(compact_kv_request_rate_list, running_num_requests)):
            print(f"  Stage {i+1}: {num_req} requests at {rate} req/s")
        
        # If repetition > 1, wrap in run_repetition_benchmark
        if repetition > 1:
            return await run_repetition_benchmark(
                repetition=repetition,
                base_url=base_url,
                run_single_benchmark_func=run_multi_stage_benchmark,
                initial_pp_config=initial_pp_config,
                alternative_configs=alternative_configs,
                migration_steps=migration_steps,
                migration_mode=migration_mode,
                request_func=request_func,
                input_requests=input_requests,
                request_rate_list=compact_kv_request_rate_list,
                running_num_requests=running_num_requests,
                burstiness=burstiness,
                disable_tqdm=disable_tqdm,
                max_concurrency=max_concurrency,
                lora_modules=lora_modules,
                model_id=model_id,
                model_name=model_name,
                api_url=api_url,
                logprobs=logprobs,
                ignore_eos=ignore_eos,
                extra_body=extra_body,
                profile=profile,
                test_prompt=test_prompt,
                test_prompt_len=test_prompt_len,
                test_output_len=test_output_len,
                test_mm_content=test_mm_content,
                tokenizer=tokenizer,
                selected_percentile_metrics=selected_percentile_metrics,
                selected_percentiles=selected_percentiles,
                goodput_config_dict=goodput_config_dict,
                print_outputs=print_outputs,
                warmup_stage_count=warmup_stage_count,
            )
        
        # 执行多阶段基准测试 (single run)
        return await run_multi_stage_benchmark(
            request_func=request_func,
            input_requests=input_requests,
            request_rate_list=compact_kv_request_rate_list,
            running_num_requests=running_num_requests,
            burstiness=burstiness,
            disable_tqdm=disable_tqdm,
            max_concurrency=max_concurrency,
            lora_modules=lora_modules,
            model_id=model_id,
            model_name=model_name,
            api_url=api_url,
            logprobs=logprobs,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
            profile=profile,
            base_url=base_url,
            test_prompt=test_prompt,
            test_prompt_len=test_prompt_len,
            test_output_len=test_output_len,
            test_mm_content=test_mm_content,
            tokenizer=tokenizer,
            selected_percentile_metrics=selected_percentile_metrics,
            selected_percentiles=selected_percentiles,
            goodput_config_dict=goodput_config_dict,
            print_outputs=print_outputs,
            warmup_stage_count=warmup_stage_count,
        )
    
    # 原有的单阶段基准测试逻辑
    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate, burstiness):
        prompt, prompt_len, output_len, mm_content = (
            request.prompt,
            request.prompt_len,
            request.expected_output_len,
            request.multi_modal_data,
        )
        print(f"input, output length: {prompt_len}, {output_len}")
        req_model_id, req_model_name = model_id, model_name
        if lora_modules:
            req_lora_module = next(lora_modules)
            req_model_id, req_model_name = req_lora_module, req_lora_module

        request_func_input = RequestFuncInput(
            model=req_model_id,
            model_name=req_model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            logprobs=logprobs,
            multi_modal_content=mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks, return_exceptions=True)

    # Optionally print per-request outputs
    if print_outputs:
        total_output_tokens = 0
        for idx, out in enumerate(outputs):
            # Handle both exceptions and RequestFuncOutput objects
            if isinstance(out, Exception):
                print(f"Request {idx}: EXCEPTION: {out}")
            elif out.success:
                print(f"Request {idx}: {out.generated_text}")
                # Accumulate output token count with tokenizer fallback
                output_len = out.output_tokens
                if not output_len:
                    output_len = len(
                        tokenizer(out.generated_text, add_special_tokens=False).input_ids
                    )
                total_output_tokens += output_len
            else:
                print(f"Request {idx}: ERROR: {out.error}")
        print(f"Total generated tokens: {total_output_tokens}")

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    if goodput_config_dict:
        print(
            "{:<40} {:<10.2f}".format(
                "Request goodput (req/s):", metrics.request_goodput
            )
        )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput:": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. "
                )
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative."
                )
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            'Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds."
        ) from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any], file_name: str
) -> None:
    metrics = [
        "median_ttft_ms",
        "mean_ttft_ms",
        "std_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p99_tpot_ms",
        "median_itl_ms",
        "mean_itl_ms",
        "std_itl_ms",
        "p99_itl_ms",
    ]
    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]] for k in metrics},
        extra_info={
            k: results[k]
            for k in results
            if k not in metrics and k not in ignored_metrics
        },
    )
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )
        # 读取基准测试配置
    if args.benchmark_config:
        # 检查配置文件是否存在
        if not os.path.exists(args.benchmark_config):
            raise ValueError(f"Benchmark config file not found: {args.benchmark_config}")
        with open(args.benchmark_config, 'r') as f:
            benchmark_config = BenchCfg.model_validate(json.load(f))
        # Propagate metrics output path from config if provided (for vllm_exp integration)
        if getattr(benchmark_config, "metrics_file_name", None):
            os.environ["METRICS_FILE_NAME"] = benchmark_config.metrics_file_name
        
        # Override args.burstiness with value from config file if present
        if hasattr(benchmark_config, 'burstiness'):
            args.burstiness = benchmark_config.burstiness
            print(f"Using burstiness from config file: {args.burstiness}")

    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required."
        )

    if args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        # For the "sonnet" dataset, formatting depends on the backend.
        if args.backend == "openai-chat":
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=False,
            )
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset."
            )
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=True,
            )

    elif args.dataset_name == "hf":
        # all following datasets are implemented from the
        # HuggingFaceDataset base class
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = VisionArenaDataset
            args.hf_split = "train"
            args.hf_subset = None
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = InstructCoderDataset
            args.hf_split = "train"
        elif args.dataset_path in MTBenchDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = MTBenchDataset
            args.hf_split = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ConversationDataset
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_class = AIMODataset
            args.hf_split = "train"
        elif args.dataset_path in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS:  # noqa: E501
            dataset_class = NextEditPredictionDataset
            args.hf_split = "train"
        elif args.dataset_path in ASRDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ASRDataset
            args.hf_split = "train"
        else:
            supported_datasets = set(
                [
                    dataset_name
                    for cls in HuggingFaceDataset.__subclasses__()
                    for dataset_name in cls.SUPPORTED_DATASET_PATHS
                ]
            )
            raise ValueError(
                f"Unsupported dataset path: {args.dataset_path}. "
                "Huggingface dataset only supports dataset_path"
                f" from one of following: {supported_datasets}. "
                "Please consider contributing if you would "
                "like to add support for additional dataset formats."
            )

        if dataset_class.IS_MULTIMODAL and backend not in [
            "openai-chat",
            "openai-audio",
        ]:
            # multi-modal benchmark is only available on OpenAI Chat backend.
            raise ValueError(
                "Multi-modal content is only supported on 'openai-chat' and "
                "'openai-audio' backend."
            )
        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
        )

    elif args.dataset_name == "pattern":
        def _to_pairs(v: list[list[int]]) -> list[Tuple[int, int]]:
            pairs: list[Tuple[int, int]] = []
            assert len(v) != 0
            for ele in v:
                assert isinstance(ele, list)
                assert len(ele) == 2
                pairs.append((ele[0], ele[1]))
            return pairs

        pattern_ds = PatternDataset(dataset_path=args.dataset_path)

        # If warmup is enabled in benchmark config, build a single combined
        # request list (warmup first, then main) so the benchmark can run once.
        if getattr(benchmark_config, "warmup", None) is not None and benchmark_config.warmup.enabled:
            warm = benchmark_config.warmup
            warm_pairs = _to_pairs(warm.input_output_lens)
            main_pairs = _to_pairs(benchmark_config.input_output_lens)

            warm_requests = pattern_ds.pattern_sample(
                tokenizer=tokenizer,
                num_requests=warm.data_num_requests,
                input_output_len=warm_pairs,
                prefix_len=args.pattern_prefix_len,
                range_ratio=args.pattern_range_ratio,
            )
            main_requests = pattern_ds.pattern_sample(
                tokenizer=tokenizer,
                num_requests=benchmark_config.data_num_requests,
                input_output_len=main_pairs,
                prefix_len=args.pattern_prefix_len,
                range_ratio=args.pattern_range_ratio,
            )
            input_requests = warm_requests + main_requests
        else:
            input_output_len = _to_pairs(benchmark_config.input_output_lens)
            input_requests = pattern_ds.pattern_sample(
                tokenizer=tokenizer,
                num_requests=benchmark_config.data_num_requests,
                input_output_len=input_output_len,
                prefix_len=args.pattern_prefix_len,
                range_ratio=args.pattern_range_ratio,
            )
    else:
        # For datasets that follow a similar structure, use a mapping.
        dataset_mapping = {
            "sharegpt": lambda: ShareGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                output_len=args.sharegpt_output_len,
            ),
            "burstgpt": lambda: BurstGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(tokenizer=tokenizer, num_requests=args.num_prompts),
            "random": lambda: RandomDataset(dataset_path=args.dataset_path).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                range_ratio=args.random_range_ratio,
            ),
        }

        try:
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err
    goodput_config_dict = check_goodput_args(args)

    # Collect the sampling parameters.
    sampling_params = {
        k: v
        for k, v in {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "temperature": args.temperature,
        }.items()
        if v is not None
    }

    # Sampling parameters are only supported by openai-compatible backend.
    if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
        raise ValueError(
            "Sampling parameters are only supported by openai-compatible backends."
        )

    if "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.0  # Default to greedy decoding.

    # Avoid GC processing "static" data - reduce pause times.
    gc.collect()
    gc.freeze()


    # Combine warmup+main stage definitions (request rates + running request counts)
    # into a single multi-stage run when warmup is enabled.
    compact_rates = benchmark_config.running_request_rates
    running_nums = benchmark_config.running_num_requests
    warmup_stage_count = 0
    repetition = getattr(benchmark_config, "repetition", 1)
    
    # Get pipeline config for repetition reset
    initial_pp_config = getattr(benchmark_config, "initial_pp_config", None)
    alternative_configs = getattr(benchmark_config, "alternative_configs", None)
    migration_steps = getattr(benchmark_config, "migration_steps", None)
    migration_mode = getattr(benchmark_config, "migration_mode", None)
    
    if getattr(benchmark_config, "warmup", None) is not None and benchmark_config.warmup.enabled:
        warm = benchmark_config.warmup
        warmup_stage_count = len(warm.running_num_requests)
        compact_rates = (warm.running_request_rates or []) + (benchmark_config.running_request_rates or [])
        running_nums = (warm.running_num_requests or []) + (benchmark_config.running_num_requests or [])
        # Allow separate request_metrics outputs for warmup/main.
        # (Files are provided by the launcher via env vars.)
        os.environ.setdefault("METRICS_FILE_NAME_MAIN", os.environ.get("METRICS_FILE_NAME", ""))

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            request_rate=args.request_rate,
            compact_kv_request_rate_list=compact_rates,
            running_num_requests=running_nums,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency,
            lora_modules=args.lora_modules,
            extra_body=sampling_params,
            print_outputs=args.print_outputs,
            warmup_stage_count=warmup_stage_count,
            repetition=repetition,
            initial_pp_config=initial_pp_config,
            alternative_configs=alternative_configs,
            migration_steps=migration_steps,
            migration_mode=migration_mode,
        )
    )

    # Save config and results to json
    if args.save_result or args.append_result:
        result_json: dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )
        # Traffic
        if args.benchmark_config:
            result_json["benchmark_config_file"] = args.benchmark_config
            # 从配置文件中读取实际值用于显示
            try:
                with open(args.benchmark_config, 'r') as f:
                    config_data = json.load(f)
                request_rates = config_data.get("request_rates")
                num_requests = config_data.get("num_requests")
                
                if request_rates is not None and num_requests is not None:
                    result_json["multi_stage"] = True
                    result_json["request_rate_list"] = request_rates
                    result_json["num_requests"] = num_requests
                    result_json["request_rate"] = "multi_stage"
                else:
                    result_json["multi_stage"] = False
                    result_json["request_rate"] = (
                        args.request_rate if args.request_rate < float("inf") else "inf"
                    )
            except:
                result_json["multi_stage"] = False
                result_json["request_rate"] = (
                    args.request_rate if args.request_rate < float("inf") else "inf"
                )
        else:
            result_json["multi_stage"] = False
            result_json["request_rate"] = (
                args.request_rate if args.request_rate < float("inf") else "inf"
            )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        if not args.save_detailed:
            # Remove fields with too many data points
            for field in [
                "input_lens",
                "output_lens",
                "ttfts",
                "itls",
                "generated_texts",
                "errors",
            ]:
                if field in result_json:
                    del result_json[field]

        # Save to file
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}"
            if args.max_concurrency is not None
            else ""
        )
        
        if args.benchmark_config:
            # 检查是否包含多阶段配置
            try:
                with open(args.benchmark_config, 'r') as f:
                    config_data = json.load(f)
                request_rates = config_data.get("request_rates", [])
                
                if request_rates:
                    # 多阶段基准测试文件名
                    rates_str = "_".join([f"{rate}qps" for rate in request_rates])
                    file_name = f"{backend}-multi_stage-{rates_str}{max_concurrency_str}-{base_model_id}-{current_dt}.json"
                else:
                    # 单阶段基准测试文件名
                    file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"
            except:
                # 如果读取配置文件失败，使用单阶段文件名
                file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"
        else:
            # 单阶段基准测试文件名
            file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"
        
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(
            file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
        ) as outfile:
            # Append a newline.
            if args.append_result and outfile.tell() != 0:
                outfile.write("\n")
            json.dump(result_json, outfile)
        save_to_pytorch_benchmark_format(args, result_json, file_name)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "burstgpt", "sonnet", "random", "hf", "pattern"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the sharegpt/sonnet dataset. "
        "Or the huggingface dataset ID if using HF dataset.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=(
            "Number of logprobs-per-token to compute & return as part of "
            "the request. If unspecified, then either (1) if beam search "
            "is disabled, no logprobs are computed & a single dummy "
            "logprob is returned for each token; or (2) if beam search "
            "is enabled 1 logprob per token is computed"
        ),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default=None,
        help="Path to JSON configuration file for benchmark. "
        "For multi-stage benchmark, the file should contain 'request_rates' and 'num_requests' arrays. "
        "Example: --benchmark-config config.json",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="When saving the results, whether to include per request "
        "information such as response, error, ttfs, tpots, etc.",
    )
    parser.add_argument(
        "--print-outputs",
        action="store_true",
        help="Print each request's generated output to benchmark stdout.",
    )
    parser.add_argument(
        "--append-result",
        action="store_true",
        help="Append the benchmark result to the existing json file.",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-separated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
        'Default value is "ttft,tpot,itl".',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-separated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99". '
        'Use "--percentile-metrics" to select metrics.',
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help='Specify service level objectives for goodput as "KEY:VALUE" '
        "pairs, where the key is a metric name, and the value is in "
        'milliseconds. Multiple "KEY:VALUE" pairs can be provided, '
        "separated by spaces. Allowed request level metric names are "
        '"ttft", "tpot", "e2el". For more context on the definition of '
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
    )

    # group for dataset specific arguments
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help="Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help="Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help="Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.",
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for random sampling. Must be in the range [0, 1) to define "
        "a symmetric sampling range"
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help=(
            "Number of fixed prefix tokens before the random context "
            "in a request. "
            "The total input length is the sum of `random-prefix-len` and "
            "a random "
            "context length sampled from [input_len * (1 - range_ratio), "
            "input_len * (1 + range_ratio)]."
        ),
    )

    pattern_group = parser.add_argument_group("pattern dataset options")
    pattern_group.add_argument(
        "--pattern-batch-size",
        type=int,
        default=60,
        help="Time interval (in seconds) to change the input/output length.",
    )

    pattern_group.add_argument(
        "--pattern-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for random sampling. Must be in the range [0, 1) to define "
        "a symmetric sampling range"
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )
    pattern_group.add_argument(
        "--pattern-prefix-len",
        type=int,
        default=0,
        help=(
            "Number of fixed prefix tokens before the random context "
            "in a request. "
            "The total input length is the sum of `random-prefix-len` and "
            "a random "
            "context length sampled from [input_len * (1 - range_ratio), "
            "input_len * (1 + range_ratio)]."
        ),
    )

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument(
        "--hf-subset", type=str, default=None, help="Subset of the HF dataset."
    )
    hf_group.add_argument(
        "--hf-split", type=str, default=None, help="Split of the HF dataset."
    )
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )

    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature sampling parameter. Only has effect on "
        "openai-compatible backends. If not specified, default to greedy "
        "decoding (i.e. temperature==0.0).",
    )

    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        choices=["auto", "slow", "mistral", "custom"],
        help='The tokenizer mode.\n\n* "auto" will use the '
        'fast tokenizer if available.\n* "slow" will '
        "always use the slow tokenizer. \n* "
        '"mistral" will always use the `mistral_common` tokenizer. \n*'
        '"custom" will use --tokenizer to select the preregistered tokenizer.',
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. "
        "If not specified, the model name will be the "
        "same as the ``--model`` argument. ",
    )

    parser.add_argument(
        "--lora-modules",
        nargs="+",
        default=None,
        help="A subset of LoRA module names passed in when "
        "launching the server. For each request, the "
        "script chooses a LoRA module at random.",
    )

    args = parser.parse_args()

    # 验证基准测试配置
    if args.benchmark_config:
        if not os.path.exists(args.benchmark_config):
            raise ValueError(f"Benchmark config file not found: {args.benchmark_config}")
        
        try:
            with open(args.benchmark_config, 'r') as f:
                config_data = json.load(f)
            
            # 检查是否包含多阶段基准测试配置
            request_rates = config_data.get("request_rates")
            num_requests = config_data.get("num_requests")
            
            if request_rates is not None and num_requests is not None:
                if len(request_rates) != len(num_requests):
                    raise ValueError(
                        f"Length of request_rates ({len(request_rates)}) must match "
                        f"length of num_requests ({len(num_requests)})"
                    )
                
                # 验证请求数量总和不超过总请求数
                total_stage_requests = sum(num_requests)
                if hasattr(args, 'num_prompts') and total_stage_requests > args.num_prompts:
                    print(f"Warning: Total requests in stages ({total_stage_requests}) exceeds "
                          f"total prompts ({args.num_prompts}). This may cause issues.")
                
                print(f"Multi-stage benchmark config validation passed: {len(request_rates)} stages configured")
            else:
                print("Benchmark config loaded (no multi-stage configuration found)")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading config file: {e}")

    main(args)

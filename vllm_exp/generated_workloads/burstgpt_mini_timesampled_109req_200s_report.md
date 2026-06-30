# BurstGPT Mini Time-Sampled Workload, 109 Requests / 200s

## Goal

Build a short BurstGPT mini workload that still follows the full high-window request trend, covers the dense early region, and stays within the measured 4GPU capacity target on average.

## Selection Rule

- source trace: `/data/gpfs/projects/punim2715/datasets/BurstGPT_without_fails_2.csv`
- model filter: `GPT-4`
- source window: `5539015.0` to `5539915.0`
- source requests after filtering: `423`
- source native span: `894.000s`
- measured 4GPU drain capacity: `0.642 req/s`
- headroom: `0.85`
- target duration: `200.0s`
- target requests: `round(0.642 * 0.85 * 200.0) = 109`
- sampling method: stratified deterministic random sampling by `45s` native-time bins
- seed: `5`
- replay time_scale_factor: `4.470000`

## Shape And Token Summary

| scope | requests | mean input | mean output | mean total | max total |
|---|---:|---:|---:|---:|---:|
| full high window | 423 | 1204.3 | 165.4 | 1369.7 | 2572 |
| mini sampled trace | 109 | 1201.5 | 165.4 | 1366.9 | 2448 |

- trend correlation over 20 replay-time bins: `0.992`
- replay average RPS: `0.545`
- replay peak RPS: 10s `1.800`, 30s `1.267`, 60s `1.117`
- full 10s-bin counts after 200s scaling: `[22, 52, 40, 41, 50, 40, 34, 1, 20, 2, 20, 6, 1, 20, 11, 13, 20, 12, 17, 1]`
- mini 10s-bin counts: `[6, 13, 10, 10, 13, 10, 9, 1, 5, 1, 5, 1, 1, 5, 3, 2, 6, 3, 4, 1]`

## Artifacts

- mini BurstGPT dataset: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/generated_workloads/burstgpt_mini_timesampled_109req_200s.csv`
- detailed selected request CSV: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/generated_workloads/burstgpt_mini_timesampled_109req_200s_detail.csv`
- comparison SVG: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/generated_workloads/burstgpt_mini_timesampled_109req_200s_comparison.svg`
- static PP4 validation config: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_mini_timesampled_kvcached_layerstack_runs/static_pp4_kvcached_layerstack_mini_timesampled_109req_200s.yaml`

## Static-4GPU Validation Result

- validation log directory: `/data/gpfs/projects/punim2715/vllm_workbench/logs/project-a100_burstgpt_mini_timesampled_kvcached_layerstack_static_pp4`
- config baseline check: passed
- booked node: `spartan-gpgpu138`, 4 x A100 80GB
- PP layout: `16,16,16,16`
- successful requests: `109 / 109`
- replay span from benchmark log: `200.000s`
- benchmark duration: `208.67s`
- request throughput: `0.52 req/s`
- TTFT: mean `0.387s`, p50 `0.332s`, p90 `0.873s`, p95 `0.954s`, p99 `1.071s`, max `1.072s`
- TPOT: mean `30.6ms`, p50 `25.6ms`, p90 `43.5ms`, p95 `59.3ms`, p99 `91.2ms`, max `116.5ms`
- E2EL: mean `4.438s`, p50 `1.799s`, p90 `11.356s`, p95 `12.780s`, p99 `15.180s`, max `15.778s`
- waiting queue: mean `0`, p99 `0`, max `0`, non-zero samples `0 / 91`
- KV pressure: mean `0.212%`, p95 `0.640%`, p99 `0.805%`, max `0.815%`
- fatal errors: none found for `EngineCore encountered`, `EngineDeadError`, `Traceback`, `IndexError`, or `CUDA error`

# BurstGPT Mini-Valid 109-Request Trace

- source trace: `/data/gpfs/projects/punim2715/datasets/BurstGPT_without_fails_2.csv`
- full high window: `GPT-4`, start `5539015.0`, duration `900.0s`, requests `423`, native span `894.000s`
- selected full-window index: `208` to `316`
- selected native timestamp range: `5539242.000` to `5539481.000`
- selected native span: `239.000s`
- target replay span: `200.000s`
- time_scale_factor: `1.195000`
- average replay RPS: `0.545000`
- replay peak RPS: 10s `2.200`, 30s `1.733`, 60s `1.067`

## Token Summary

| scope | requests | mean input | mean output | mean total | max total |
|---|---:|---:|---:|---:|---:|
| full high window | 423 | 1204.3 | 165.4 | 1369.7 | 2572 |
| selected mini trace | 109 | 1286.6 | 171.1 | 1457.7 | 2297 |

## Artifacts

- CSV: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/generated_workloads/burstgpt_mini_valid_109_requests_tokens.csv`
- SVG: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/generated_workloads/burstgpt_mini_valid_109_requests_tokens.svg`

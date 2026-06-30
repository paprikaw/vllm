# ShareGPT-Calibrated BurstGPT Autoscaling Design

## Rule

Use ShareGPT first-turn token distribution to estimate one active-rank capacity.
`capacity_unit = ShareGPT total_tokens p75 = 593.0 tokens/s at 1 rps`.
For each BurstGPT stage, compute `score = request_rate * mean_total_tokens`.
Choose `active_ranks = ceil(score / capacity_unit)`, clamped to `[1, 4]`.

## ShareGPT Calibration

- sample_size: 512
- total_tokens p50/p75/p90/p95: 369.0 / 593.0 / 817.8 / 1079.8
- output_tokens p50/p75/p90: 120.0 / 301.5 / 474.0

## BurstGPT Stages

| boundary | req/s | mean in | mean out | score | active ranks | pp |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.400 | 345.5 | 149.9 | 198.2 | 1 | `64,0,0,0` |
| 8 | 0.400 | 408.5 | 377.6 | 314.5 | 1 | `64,0,0,0` |
| 16 | 0.875 | 705.4 | 50.8 | 661.6 | 2 | `32,32,0,0` |
| 24 | 0.400 | 268.4 | 320.9 | 235.7 | 1 | `64,0,0,0` |

## Generated Artifacts

- config: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/l40_burstgpt_sharegpt_autoscale_stress.yaml`
- split config dir: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/l40_burstgpt_sharegpt_stress_runs`
- ordered BurstGPT subset: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/generated_workloads/burstgpt_l40_stress_ordered.csv`

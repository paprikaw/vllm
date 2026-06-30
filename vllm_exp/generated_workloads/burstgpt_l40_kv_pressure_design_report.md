# BurstGPT KV-Pressure Autoscaling Design

## Rule

Runtime policy: read scheduler KV cache utilization on request arrival.
If utilization is at least 0.80, increment one PP config: 1->2->3->4 active ranks.
The policy is scale-up only for this experiment, so the workload must keep pressure high enough to cover all ranks.

## ShareGPT Calibration

- capacity_unit_tokens_per_second: 593.0
- total_tokens p50/p75/p90/p95: 369.0 / 593.0 / 817.8 / 1079.8

## BurstGPT Pressure Ladder

| boundary | stage | req/s | mean input | mean output | mean total | estimated pressure score |
|---:|---|---:|---:|---:|---:|---:|
| 0 | pressure_1gpu | 2.00 | 64.0 | 835.3 | 899.3 | 9.90 |
| 64 | pressure_2gpu | 4.00 | 71.8 | 827.2 | 899.0 | 19.60 |
| 128 | pressure_3gpu | 6.00 | 70.6 | 842.1 | 912.7 | 30.38 |
| 192 | pressure_4gpu | 8.00 | 47.2 | 910.0 | 957.2 | 45.90 |

## Generated Artifacts

- combined config: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/l40_burstgpt_kv_pressure_autoscale.yaml`
- split config dir: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/l40_burstgpt_kv_pressure_runs`
- ordered BurstGPT subset: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/generated_workloads/burstgpt_l40_kv_pressure_ordered.csv`

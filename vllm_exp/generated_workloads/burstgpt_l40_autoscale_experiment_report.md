# L40 ShareGPT-Calibrated BurstGPT Autoscaling Report

## Objective

Use ShareGPT to design an autoscaling rule, replay a short BurstGPT workload on
the booked L40 node, and compare the autoscaling deployment against static
1/2/3/4 active-GPU deployments.

## Autoscaling Rule

ShareGPT first-turn conversations are sampled to estimate the one-active-rank
capacity. The rule uses total tokens as the workload signal:

```text
capacity_unit = ShareGPT total_tokens p75 = 593.0 tokens/s at 1 rps
score = request_rate * mean_total_tokens
active_ranks = ceil(score / capacity_unit), clamped to [1, 4]
```

The generated PP layouts are:

```text
1 rank: 64,0,0,0
2 ranks: 32,32,0,0
3 ranks: 16,16,32,0
4 ranks: 16,16,16,16
```

The 3-rank layout is intentionally VMM-valid: each non-empty active rank has a
layer count divisible by the current VMM layer-group granularity of 16.

## Workloads

Two 24-request BurstGPT replays were run:

| workload | request schedule | autoscale decisions |
|---|---|---|
| short | 0: 0.4 rps, 6: 0.4 rps, 12: 2.0 rps, 18: 0.4 rps | 1 rank -> 2 ranks -> 1 rank |
| stress24 | 0: 0.4 rps, 6: 0.4 rps, 12: 5.0 rps, 18: 0.4 rps | 1 rank -> 4 ranks -> 1 rank |

## Results

All successful runs completed 24/24 requests.

### Short Replay

| deployment | avg active ranks | p50 TTFT | p95 TTFT | p50 TPOT | p95 TPOT | p50 E2E | p95 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| static_pp1 | 1.00 | 0.1340s | 0.2579s | 0.0641s | 0.0660s | 15.847s | 30.672s |
| static_pp2 | 2.00 | 0.1362s | 0.2573s | 0.0669s | 0.0686s | 16.545s | 31.974s |
| static_pp3 | 3.00 | 0.1460s | 0.2810s | 0.0696s | 0.0714s | 17.259s | 33.285s |
| static_pp4 | 4.00 | 0.1432s | 0.3039s | 0.0724s | 0.0744s | 18.058s | 34.630s |
| autoscale | 1.25 | 0.1439s | 0.3581s | 0.0683s | 0.1038s | 15.839s | 33.681s |

Autoscale migration process total time: 2.3s and 3.3s.

### Stress24 Replay

| deployment | avg active ranks | p50 TTFT | p95 TTFT | p50 TPOT | p95 TPOT | p50 E2E | p95 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| static_pp1 | 1.00 | 0.1332s | 0.2737s | 0.0644s | 0.0670s | 15.856s | 30.781s |
| static_pp2 | 2.00 | 0.1398s | 0.2883s | 0.0674s | 0.0697s | 16.599s | 32.173s |
| static_pp3 | 3.00 | 0.1388s | 0.2736s | 0.0700s | 0.0726s | 17.256s | 33.465s |
| static_pp4 | 4.00 | 0.1503s | 0.3781s | 0.0728s | 0.0752s | 17.989s | 34.695s |
| autoscale | 1.75 | 0.1589s | 1.4004s | 0.0790s | 0.2139s | 16.369s | 37.562s |

Autoscale migration process total time: 2.1s and 10.5s.

## Conclusion

The autoscaling mechanism is functionally validated on the current L40 node: it
can make workload-dependent decisions from the BurstGPT replay and complete both
1-to-2-to-1 and 1-to-4-to-1 vertical scaling paths.

The current short replays do not demonstrate a performance win over static PP1.
The burst windows are only six requests long, and the request outputs in the
high-rate stage are short. Migration overhead is therefore too large relative to
the useful high-load window. This workload is a good smoke test for autoscaling
correctness, but not a strong performance showcase.

For a performance-winning scenario, use longer high-load windows, larger request
batches, longer prefills/decodes, or a larger model/context setting so that
static PP1 saturates and migration cost can be amortized over many more requests.

## Artifacts

| artifact | path |
|---|---|
| generator | `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/tools/generate_sharegpt_burst_autoscale.py` |
| short config | `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/l40_burstgpt_sharegpt_autoscale.yaml` |
| short split configs | `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/l40_burstgpt_sharegpt_runs` |
| stress24 config | `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/l40_burstgpt_sharegpt_autoscale_stress24.yaml` |
| stress24 split configs | `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/l40_burstgpt_sharegpt_stress24_runs` |
| short logs | `/data/gpfs/projects/punim2715/vllm_workbench/vllm/new-logs/l40_burstgpt_autoscale_split_20260623` |
| stress24 logs | `/data/gpfs/projects/punim2715/vllm_workbench/vllm/new-logs/l40_burstgpt_autoscale_stress24_20260623` |

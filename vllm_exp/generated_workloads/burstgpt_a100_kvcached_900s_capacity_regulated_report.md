# A100 KVCacheD Layer-Stacking BurstGPT Capacity Regulation

## Regulation Rule

- Paper reference: KunServe scales BurstGPT RPS to fit the serving capacity while preserving the temporal pattern, and keeps average memory demand below a target utilization.
- Local adaptation: preserve the selected BurstGPT timestamp window, but derive `time_scale_factor` from measured 4GPU drain capacity instead of using the old fixed factor `8.0`.
- measured 4GPU drain capacity from failed autoscale run: `0.642 req/s`
- headroom: `0.85`
- target average replay RPS: `0.5457 req/s`
- resulting time_scale_factor: `1.153`
- replay span: `775.151s` for `423` requests
- replay peak RPS: `3.000` over 10s, `1.733` over 30s, `1.367` over 60s

## Validation Order

1. Run static-4GPU KVCacheD first. The workload is valid only if static-4GPU has bounded TTFT/TPOT and drains without large backlog.
2. Then run autoscale from 1GPU to direct-to-4GPU and compare against static baselines.

## Static-4GPU Validation Result

- config: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_kvcached_layerstack_direct4_wait0_capacity_regulated_runs/static_pp4_kvcached_layerstack_capacity_regulated.yaml`
- log directory: `/data/gpfs/projects/punim2715/vllm_workbench/logs/project-a100_burstgpt_kvcached_layerstack_direct4_wait0_capacity_regulated_static_pp4`
- successful requests: `423 / 423`
- benchmark duration: `784.63s`
- TTFT: mean `1.353s`, p50 `1.393s`, p90 `2.022s`, p95 `2.340s`, p99 `2.953s`, max `3.192s`
- TPOT: mean `81.3ms`, p50 `62.6ms`, p90 `167.2ms`, p95 `208.8ms`, p99 `243.4ms`, max `335.0ms`
- E2EL: mean `9.927s`, p50 `4.076s`, p90 `23.277s`, p95 `26.527s`, p99 `32.071s`, max `34.721s`
- waiting queue: mean `0.070`, p99 `2.160`, max `4`, non-zero in `8 / 285` samples
- KV pressure: mean `0.599%`, p95 `2.222%`, p99 `2.571%`, max `2.575%`
- fatal errors: none found for `EngineCore encountered`, `EngineDeadError`, `Traceback`, `IndexError`, or `CUDA error`

## Generated Artifacts

- autoscale config: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_kvcached_layerstack_direct4_wait0_capacity_regulated.yaml`
- split config dir: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_kvcached_layerstack_direct4_wait0_capacity_regulated_runs`
- static-4GPU validation config: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_kvcached_layerstack_direct4_wait0_capacity_regulated_runs/static_pp4_kvcached_layerstack_capacity_regulated.yaml`
- regulated dataset: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/generated_workloads/burstgpt_a100_native_900s_capacity_regulated.csv`

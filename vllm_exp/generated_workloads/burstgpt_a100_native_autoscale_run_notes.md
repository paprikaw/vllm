# A100 Native BurstGPT Autoscale Run Notes

## Workload

- Source trace: `/data/gpfs/projects/punim2715/datasets/BurstGPT_without_fails_2.csv`
- Model filter: `GPT-4`
- Native high window: `[5539015, 5539915)`
- Requests: `423`
- Native span: `894s`
- Replay span: `111.75s` with `time_scale_factor=8`
- Token totals: `509437` input, `69962` output
- Max request total tokens: `2572`
- Model: `/home/bxb1/data/huggingface/Qwen3-32B-FP8`
- Testbed: `spartan-gpgpu068`, 4 x A100 80GB

## Validated Baseline Checks

- Config guard passed for:
  `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_runs/autoscale_native_burstgpt.yaml`
- Config guard passed for:
  `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_early_runs/autoscale_native_burstgpt.yaml`
- Static PP4 config guard passed for:
  `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_runs/static_pp4.yaml`

## Runs

### Autoscale, 640s Window, Threshold 0.80

- Log dir: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/new-logs/project-a100_burstgpt_native_autoscale`
- Requests: `359/359` successful
- Benchmark duration: `117.52s`
- Scale event: `1 -> 2` at request `304`, `kv_pressure=0.8119`
- Migration process time: `19.5s`
- Caveat: background KV resize on rank 1 hit CUDA OOM after migration. The benchmark completed, but this is not a clean run.

### Autoscale, 900s Window, Threshold 0.55

- Log dir: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/new-logs-burstgpt-900/project-a100_burstgpt_native_autoscale`
- Requests: `339/423` successful
- Benchmark duration: `112.11s`
- Scale event: `1 -> 2` at request `269`, `kv_pressure=0.6035`
- Request-state sync payload: `payload_states=320`, `req_ids=320`
- Migration process time: `25.6s`
- Failure: `AssertionError: ('placeholder', 0, 'out_of_band_tensors', [])`
- Interpretation: target PP actor-chain received a Ray tensor placeholder without the corresponding out-of-band tensor buffer.

### Static PP4, 900s Window

- Log dir: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/new-logs-burstgpt-static-900/project-a100_burstgpt_native_static_pp4`
- Run was manually interrupted after exceeding the 180s wall-time constraint.
- Partial progress observed: about `166/423` completed after roughly `3m53s` in the benchmark progress log.
- This is not a valid baseline result; it only shows that the 900s trace is very heavy even for fixed PP4 with the current serving configuration.

### Autoscale, 900s Window, Threshold 0.35

- Log dir: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/new-logs-burstgpt-early-900/project-a100_burstgpt_native_autoscale`
- Requests: `325/423` successful
- Benchmark duration: `112.10s`
- Scale event: `1 -> 2` at request `148`, `kv_pressure=0.3949`
- Request-state sync payload: `payload_states=309`, `req_ids=309`
- Migration process time: `29.9s`
- Failure: `AssertionError: ('placeholder', 0, 'out_of_band_tensors', [])`
- Interpretation: lowering the threshold did not avoid the same target PP actor-chain/Ray tensor transport failure.

## Current Conclusion

The native BurstGPT timestamp replay path and config generation are working.
The high 900s native window is useful for stressing autoscaling, but the current
implementation cannot produce a clean paper result because 1-to-2 autoscaling
under this burst repeatedly kills EngineCore with the same Ray out-of-band tensor
deserialization failure.

Before running the final autoscale-vs-static matrix, debug the autoscaling target
PP actor-chain path around request-state sync and hidden-state tensor transport:

- `vllm/v1/executor/dynamic_ray_distributed_executor.py`
- `vllm/v1/worker/dynamic_gpu_worker.py`
- `vllm/v1/engine/dynamic_core.py`


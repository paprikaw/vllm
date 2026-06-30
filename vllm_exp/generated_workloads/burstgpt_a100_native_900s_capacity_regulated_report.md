# A100 Native BurstGPT Autoscaling Design

- source trace: `/data/gpfs/projects/punim2715/datasets/BurstGPT_without_fails_2.csv`
- selected native window: [5539015, 5539915)
- requests: 423
- native span: 894.000s
- native average RPS: 0.473
- time scale factor: 1.153
- replay span: 775.151s
- replay average RPS: 0.546
- replay peak 10s RPS: 3.000
- replay peak 30s RPS: 1.733
- replay peak 60s RPS: 1.367
- max replay seconds: 900.000
- four_gpu_capacity_rps: 0.642
- capacity_headroom: 0.850
- capacity_target_rps: 0.5457
- capacity regulation: preserve BurstGPT timestamps and scale only RPS to fit the measured 4GPU serving capacity.
- total input tokens: 509437
- total output tokens: 69962
- max total tokens/request: 2572
- max_model_len: 32768
- max_num_batched_tokens: 8192
- scale_up_threshold: 0.300
- scale_down_threshold: 0.260
- post_benchmark_wait_s: 30.000

## Generated Artifacts

- combined config: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_capacity_regulated.yaml`
- split config dir: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_capacity_regulated_runs`
- ordered BurstGPT subset: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/generated_workloads/burstgpt_a100_native_900s_capacity_regulated.csv`
- split configs: /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_capacity_regulated_runs/static_pp1.yaml, /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_capacity_regulated_runs/static_pp2.yaml, /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_capacity_regulated_runs/static_pp3.yaml, /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_capacity_regulated_runs/static_pp4.yaml, /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_capacity_regulated_runs/autoscale_native_burstgpt.yaml

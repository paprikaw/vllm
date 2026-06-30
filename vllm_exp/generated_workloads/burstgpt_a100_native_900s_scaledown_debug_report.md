# A100 Native BurstGPT Autoscaling Design

- source trace: `/data/gpfs/projects/punim2715/datasets/BurstGPT_without_fails_2.csv`
- selected native window: [5539015, 5539915)
- requests: 423
- native span: 894.000s
- time scale factor: 8.000
- replay span: 111.750s
- max replay seconds: 180.000
- total input tokens: 509437
- total output tokens: 69962
- max total tokens/request: 2572
- max_model_len: 32768
- max_num_batched_tokens: 8192
- scale_up_threshold: 0.350
- scale_down_threshold: 0.260
- post_benchmark_wait_s: 30.000

## Generated Artifacts

- combined config: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_scaledown_debug.yaml`
- split config dir: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_scaledown_debug_runs`
- ordered BurstGPT subset: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/generated_workloads/burstgpt_a100_native_900s_scaledown_debug.csv`
- split configs: /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_scaledown_debug_runs/static_pp1.yaml, /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_scaledown_debug_runs/static_pp2.yaml, /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_scaledown_debug_runs/static_pp3.yaml, /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_scaledown_debug_runs/static_pp4.yaml, /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_scaledown_debug_runs/autoscale_native_burstgpt.yaml

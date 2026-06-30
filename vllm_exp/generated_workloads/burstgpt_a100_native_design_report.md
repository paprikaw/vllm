# A100 Native BurstGPT Autoscaling Design

- source trace: `/data/gpfs/projects/punim2715/datasets/BurstGPT_without_fails_2.csv`
- selected native window: [5539015, 5539655)
- requests: 359
- native span: 633.000s
- time scale factor: 8.000
- replay span: 79.125s
- max replay seconds: 180.000
- total input tokens: 440162
- total output tokens: 47536
- max total tokens/request: 2448
- max_model_len: 32768
- max_num_batched_tokens: 8192

## Generated Artifacts

- combined config: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_autoscale.yaml`
- split config dir: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_runs`
- ordered BurstGPT subset: `/data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/generated_workloads/burstgpt_a100_native_640s.csv`
- split configs: /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_runs/static_pp1.yaml, /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_runs/static_pp2.yaml, /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_runs/static_pp3.yaml, /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_runs/static_pp4.yaml, /data/gpfs/projects/punim2715/vllm_workbench/vllm/vllm_exp/configs/generated/a100_burstgpt_native_runs/autoscale_native_burstgpt.yaml

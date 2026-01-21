cd /home/bxb1/vllm_workbench/vllm 
python3 -m vllm_exp.run --config vllm_exp/configs/migration_test_A100.yaml --log-dir logs/
python3 -m vllm_exp.run --config vllm_exp/configs/migration_test_A100_2.yaml  --log-dir logs/
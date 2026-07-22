from vllm_exp.data import ExperimentConfig, SweepTestConfig


def test_max_num_batched_tokens_is_a_resolved_sweep_axis():
    config = SweepTestConfig.model_validate({
        "project": "batch-token-test",
        "type": "sweep_test",
        "static_config": {
            "vllm": {
                "max_num_batched_tokens": 8128,
                "max_num_seqs": 2048,
            },
            "benchmark": {
                "num_total_requests": 1,
                "pp_layer_config": {0: "1,1"},
                "requests": {
                    0: {
                        "request_rate": 1.0,
                        "input_lens": 1,
                        "output_lens": 1,
                    }
                },
            },
        },
        "sweep_config": {
            "vllm": {
                "max_num_batched_tokens": [2048, 4000, 8128],
            }
        },
    })

    experiments = list(ExperimentConfig.iter_from_sweep_test_config(config))
    assert [experiment.vllm.max_num_batched_tokens
            for _, experiment in experiments] == [2048, 4000, 8128]
    assert [experiment.get_naming_vars()["mbt"]
            for _, experiment in experiments] == [2048, 4000, 8128]

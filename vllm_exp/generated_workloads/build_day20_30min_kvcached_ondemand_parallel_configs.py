#!/usr/bin/env python3
"""Build isolated KVCacheD on-demand configs for the Day20 30-minute grid."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / (
    "burstgpt_llama_day20_30min_dynamic300_"
    "kvcached_ondemand_6c104f5.yaml"
)
OUTPUT_DIR = ROOT / "configs" / "day20_30min_kvcached_ondemand_parallel"

RUNS = [
    {
        "label": "prefix180-diagnostic",
        "pp": {0: "36,44"},
        "a100": "172.26.93.138",
        "l40s": "172.26.93.47",
        "port": 8106,
        "layerkv_port": 19106,
        "num_total_requests": 234,
        "max_replay_seconds": 185,
    },
    {
        "label": "dynamic300-a100x2",
        "pp": {0: "36,44", 804: "48,32", 1081: "52,28"},
        "a100": "172.26.93.138",
        "l40s": "172.26.93.138",
        "port": 8110,
        "layerkv_port": 19110,
    },
    {
        "label": "dynamic300",
        "pp": {0: "36,44", 804: "48,32", 1081: "52,28"},
        "a100": "172.26.93.138",
        "l40s": "172.26.93.47",
        "port": 8100,
        "layerkv_port": 19100,
    },
    {
        "label": "static36-44",
        "pp": {0: "36,44"},
        "a100": "172.26.93.138",
        "l40s": "172.26.93.47",
        "port": 8101,
        "layerkv_port": 19101,
    },
    {
        "label": "static40-40",
        "pp": {0: "40,40"},
        "a100": "172.26.93.138",
        "l40s": "172.26.93.47",
        "port": 8102,
        "layerkv_port": 19102,
    },
    {
        "label": "static44-36",
        "pp": {0: "44,36"},
        "a100": "172.26.93.141",
        "l40s": "172.26.93.49",
        "port": 8103,
        "layerkv_port": 19103,
    },
    {
        "label": "static48-32",
        "pp": {0: "48,32"},
        "a100": "172.26.93.141",
        "l40s": "172.26.93.49",
        "port": 8104,
        "layerkv_port": 19104,
    },
    {
        "label": "static52-28",
        "pp": {0: "52,28"},
        "a100": "172.26.93.141",
        "l40s": "172.26.93.49",
        "port": 8105,
        "layerkv_port": 19105,
    },
]


def build_config(template: dict, run: dict) -> dict:
    config = deepcopy(template)
    label = run["label"]
    temp_root = f"/home/bxb1/data/tmp/vllm_exp_day20_kvcached_{label}"

    config["project"] = (
        f"burstgpt-llama-day20-30min-kvcached-ondemand-{label}-6c104f5"
    )
    config["envs"].update({
        "BENCHMARK_CONFIG_PATH": f"{temp_root}/benchmark_config.json",
        "DEPLOYMENT_CONFIG_PATH": f"{temp_root}/vllm_config.json",
        "VLLM_LAYERKV_PORT": str(run["layerkv_port"]),
        "KVCACHED_IPC_NAME": (
            f"burstgpt_day20_30min_ondemand_{label}_6c104f5_20260712"
        ),
        "KVCACHED_PAGE_PREALLOC_ENABLED": "false",
        "KVCACHED_MIN_RESERVED_PAGES": "0",
        "KVCACHED_MAX_RESERVED_PAGES": "0",
        "KVCACHED_WORKER_IPC_TRANSPORT": "tcp",
        "KVCACHED_PP_RANK_TO_IP": json.dumps({
            0: run["a100"],
            1: run["l40s"],
        }),
        "KVCACHED_WORKER_IPC_PORT_BASE": str(run["layerkv_port"] + 1000),
    })

    config["static_config"]["network"] = {
        "rank_to_ip": {0: run["a100"], 1: run["l40s"]},
        "rank_to_node": {0: run["a100"], 1: run["l40s"]},
    }
    config["static_config"]["vllm"].update({
        "port": run["port"],
        "disable_memory_overhead_monitor": True,
        "dynamic_communication_enabled": False,
        "pipeline_autoscaling_enabled": False,
    })

    benchmark = config["sweep_configs"][0]["benchmark_config"][0]
    benchmark["pp_layer_config"] = run["pp"]
    if "num_total_requests" in run:
        benchmark["num_total_requests"] = run["num_total_requests"]
    if "max_replay_seconds" in run:
        benchmark["arrival_trace"]["max_replay_seconds"] = run[
            "max_replay_seconds"
        ]
    return config


def main() -> None:
    with TEMPLATE.open(encoding="utf-8") as handle:
        template = yaml.safe_load(handle)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        output = OUTPUT_DIR / f"{run['label']}.yaml"
        with output.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                build_config(template, run),
                handle,
                sort_keys=False,
                width=100,
            )
        print(output)


if __name__ == "__main__":
    main()

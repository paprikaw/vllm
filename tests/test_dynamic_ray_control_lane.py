import ast
from pathlib import Path


def _assert_method_uses_migration_control_group(method_name: str):
    repo_root = Path(__file__).parents[1]
    source = (
        repo_root / "vllm/v1/executor/dynamic_utils.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )
    decorator = next(
        node
        for node in method.decorator_list
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "method"
    )
    concurrency_group = next(
        keyword.value
        for keyword in decorator.keywords
        if keyword.arg == "concurrency_group"
    )
    assert isinstance(concurrency_group, ast.Constant)
    assert concurrency_group.value == "migration_control"


def test_layer_load_launcher_uses_migration_control_group():
    _assert_method_uses_migration_control_group(
        "launch_async_add_layers_control")


def test_layer_load_barrier_uses_migration_control_group():
    _assert_method_uses_migration_control_group(
        "wait_for_async_add_layers_control")


def test_memory_info_uses_migration_control_group():
    _assert_method_uses_migration_control_group("get_mem_info_control")


def test_executor_submits_memory_info_to_control_method():
    repo_root = Path(__file__).parents[1]
    source = (
        repo_root
        / "vllm/v1/executor/dynamic_ray_distributed_executor.py"
    ).read_text(encoding="utf-8")
    assert "worker.get_mem_info_control.remote()" in source
    assert 'collective_rpc("get_mem_info")' not in source


def test_control_memory_info_is_read_only_and_ungated():
    repo_root = Path(__file__).parents[1]
    source = (
        repo_root / "vllm/v1/worker/dynamic_gpu_worker.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_mem_info"
    )
    method_source = ast.get_source_segment(source, method)
    assert method_source is not None
    assert "if control_plane:" in method_source
    assert "return collect_mem_info()" in method_source
    assert "if not control_plane:" in method_source
    wrapper_source = (
        repo_root / "vllm/v1/executor/dynamic_utils.py"
    ).read_text(encoding="utf-8")
    assert "self.worker.get_mem_info(control_plane=True)" in wrapper_source


def test_kv_migration_setup_stays_on_forward_plane():
    repo_root = Path(__file__).parents[1]
    executor_source = (
        repo_root
        / "vllm/v1/executor/dynamic_ray_distributed_executor.py"
    ).read_text(encoding="utf-8")
    wrapper_source = (
        repo_root / "vllm/v1/executor/dynamic_utils.py"
    ).read_text(encoding="utf-8")
    assert 'self.collective_rpc(\n            "start_kv_cache_migration_async"' in (
        executor_source
    )
    assert "start_kv_cache_migration_async_control" not in wrapper_source


def test_async_layer_loading_precedes_migration_setup():
    repo_root = Path(__file__).parents[1]
    source = (
        repo_root / "vllm/v1/engine/dynamic_core.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "change_model_configuration_by_kv_transfer_async"
    )
    calls = {
        node.func.attr: node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {
            "start_kv_cache_migration_async",
            "launch_async_add_layers",
        }
    }
    assert calls["launch_async_add_layers"] < calls[
        "start_kv_cache_migration_async"
    ]


def test_explicit_step_id_fence_is_removed():
    repo_root = Path(__file__).parents[1]
    engine_source = (
        repo_root / "vllm/v1/engine/dynamic_core.py"
    ).read_text(encoding="utf-8")
    worker_source = (
        repo_root / "vllm/v1/worker/dynamic_gpu_worker.py"
    ).read_text(encoding="utf-8")
    assert "forward_cut_step_id" not in engine_source
    assert "_wait_for_forward_cut" not in worker_source

#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
import typer
import yaml
from rich.console import Console

from .data import Config, SweepTestConfig
from .log import LogManager
from .utils import load_config, clean_metrics_directory


app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False, pretty_exceptions_show_locals=False)
C = Console()


def load_sweep_test_config(path: str) -> SweepTestConfig:
    """Load sweep test configuration from YAML file."""
    with open(path, "r") as f:
        return SweepTestConfig.model_validate(yaml.safe_load(f))


# =========================
# CLI: Legacy multi-project sweep
# =========================

@app.command()
def sweep(
    config: str = typer.Option(..., help="Path to config.yaml (supports multiple projects)"),
    log_dir: str = typer.Option("/root/vllm_workbench/logs", help="Base log dir"),
):
    """Run legacy multi-project experiments."""
    multi_cfg = load_config(config)
    C.print(multi_cfg)

    os.environ.update(multi_cfg.envs)
    for cfg in multi_cfg.projects:
        C.rule(f"[bold green]Project: {cfg.project}[/]")
        base_dir = Path(log_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        logm = LogManager(base_dir, cfg)

        os.environ.update(cfg.envs)

        if cfg.type == "partition_and_request_rate":
            from .experiments import partition_and_request_rate
            partition_and_request_rate(cfg, logm)
        elif cfg.type == "test_migration_with_different_pp":
            from .experiments import test_migration_with_different_pp
            test_migration_with_different_pp(cfg, logm)
        elif cfg.type == "one_off_test":
            from .experiments import one_off_test
            one_off_test(cfg, logm)
        else:
            raise ValueError(f"Unknown project type: {cfg.type}")


# =========================
# CLI: New sweep_test (single project)
# =========================

@app.command()
def sweep_test(
    config: str = typer.Option(..., help="Path to sweep_test config.yaml"),
    log_dir: str = typer.Option("/root/vllm_workbench/logs", help="Base log dir"),
    single_server: bool = typer.Option(False, "--single-server", "-s", 
                                        help="Use single-server mode: start server once and switch PP configs between experiments"),
):
    """Run sweep test experiment with the new unified configuration format.
    
    This is the recommended way to run parametric experiments.
    One config file = one project with sweep variables.
    
    Modes:
    - Default: Start/stop server for each experiment
    - --single-server: Start server once, switch PP configs via API
    """
    cfg = load_sweep_test_config(config)
    mode_str = "Single-Server Mode" if single_server else "Default Mode"
    C.rule(f"[bold green]Sweep Test ({mode_str}): {cfg.project}[/]")
    
    # Set environment variables
    os.environ.update(cfg.envs)
    
    # Create log manager
    base_dir = Path(log_dir) / f"project-{cfg.project}"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a simple log manager for sweep tests
    from .log import SweepLogManager
    logm = SweepLogManager(base_dir, cfg)
    
    # Run sweep test
    if single_server:
        from .experiments import sweep_test_single_server
        sweep_test_single_server(cfg, logm)
    else:
        from .experiments import sweep_test as run_sweep_test
        run_sweep_test(cfg, logm)


if __name__ == "__main__":
    app()

#!/usr/bin/env python3
from __future__ import annotations
import atexit
import datetime
import itertools
import json
import os
import signal
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass, asdict
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Iterable, Callable
from io import FileIO
import requests
import typer
import yaml
from pydantic import BaseModel
from rich.console import Console
from collections import OrderedDict

from .data import Config
from .log import LogManager
from .utils import load_config, clean_metrics_directory


app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False, pretty_exceptions_show_locals=False)
C = Console()

# =========================
# CLI: sweep over multiple projects
# =========================

@app.command()
def sweep(
    config: str = typer.Option(..., help="Path to config.yaml (supports multiple projects)"),
    log_dir: str = typer.Option("/root/vllm_workbench/logs", help="Base log dir"),
):
    multi_cfg = load_config(config)
    C.print(multi_cfg)

    # 清理metrics目录

    os.environ.update(multi_cfg.envs)
    for cfg in multi_cfg.projects:
        C.rule(f"[bold green]Project: {cfg.project}[/]")
        base_dir = Path(log_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        logm = LogManager(base_dir, cfg)
        clean_metrics_directory(logm.get_dir())

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

if __name__ == "__main__":
    app()

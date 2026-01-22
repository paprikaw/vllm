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


from .data import Config, collect_variables, collect_aliases, SweepTestConfig
app = typer.Typer(no_args_is_help=True)
C = Console()

class PathPolicy(BaseModel):
    variables: List[str] = []                   # 本轮作为“变量”的键
    include_flags: List[str] = []               # 也纳入变量的布尔/开关键

# =========================
# Path planning (variables/constants)
# =========================


def _norm_val(v: Any) -> str:
    """将值标准化为路径友好字符串"""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        v = round(v, 6)
    s = str(v)
    return s.replace("/", "_").replace(" ", "")

class PathPlanner:
    """
    负责：
      - constants 指纹化（const-<hash>）
      - 维护“当前活动变量”集合（有序、可覆盖、可删除）
      - 基于当前活动变量生成路径片段与签名
    不关心 server/bench 概念。
    """
    def __init__(self, base_dir: Path, cfg: "Config"):
        self.base_dir = base_dir
        self.cfg = cfg
        self.policy = cfg.path_policy or PathPolicy()
        self.factors = collect_variables(cfg)
        self.aliases = collect_aliases()

        # constants = 所有因子 - policy 声明的变量键（只是为了 const hash 的稳定）
        base_var_keys = list(self.policy.variables or [])
        base_var_keys = [k for k in base_var_keys if k in self.factors]
        self.const_items = {k: v for k, v in self.factors.items() if k not in base_var_keys}
        self.var_keys = set(base_var_keys)

        const_json = json.dumps(self.const_items, sort_keys=True, default=str)
        self.const_hash = sha1(const_json.encode("utf-8")).hexdigest()[:10]
        self.timestamp = datetime.datetime.now().isoformat()
        # “当前活动变量”：你通过 extend/remove 进行维护
        self.path_vars: "OrderedDict[str, Any]" = OrderedDict()
        self.const_dir_vars: "OrderedDict[str, Any]" = OrderedDict()


    # ---------- 常量目录 ----------
    def constants_dir(self) -> Path:
        if len(self.const_dir_vars) == 0:
            path =  self.base_dir / f"project-{self.cfg.project}"
        else:
            path = self.base_dir / f"project-{self.cfg.project}" / "-".join(f"{self._k(k)}={self._encode_val(v)}" for k, v in self.const_dir_vars.items())
        return path

    def write_constants_meta(self, path: Path):
        meta = {
            "constants": self.const_items,
        }
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------- 迭代式变量维护 ----------
    def extend_path_with_vars(self, vars_mapping: Dict[str, Any], *, move_to_end: bool = True) -> None:
        """
        将 mapping 中的变量加入/覆盖到当前活动变量集合。
        若变量已存在且 move_to_end=True，会把该键移动到末尾，保持“最近加入”的顺序。
        """
        for k, v in vars_mapping.items():
            if k not in self.var_keys:
                raise ValueError(f"Variable {k} is not declared in path policy.")
            if k in self.path_vars and move_to_end:
                # 先删除后插入，以更新顺序
                del self.path_vars[k]
            self.path_vars[k] = v

    def extend_base_dir_with_vars(self, vars_mapping: Dict[str, Any], *, move_to_end: bool = True) -> None:
        """
        将 mapping 中的变量加入/覆盖到base dir变量集合。
        若变量已存在且 move_to_end=True，会把该键移动到末尾，保持“最近加入”的顺序。
        """
        for k, v in vars_mapping.items():
            if k not in self.var_keys:
                raise ValueError(f"Variable {k} is not declared in path policy.")
            if k in self.const_dir_vars and move_to_end:
                # 先删除后插入，以更新顺序
                del self.const_dir_vars[k]
            self.const_dir_vars[k] = v
    # def combo_path_with_filename_vars(self, vars_mapping: Optional[Dict[str, Any]] = None, prefix: Optional[str] = None, suffix: Optional[str] = None) -> Path:
    #     """
    #     使用的combo dir和最终出现在log文件名中的vars来构建最终的path
    #     """
    #     root = self.constants_dir()
    #     if vars_mapping is None:
    #         vars_mapping = {}

    #     for k, _ in vars_mapping.items():
    #         if k not in self.var_keys:
    #             raise ValueError(f"Variable {k} is not declared in path policy.")
    #     path_parts = [f"{self._k(k)}={self._encode_val(v)}" for k, v in self.path_vars.items()]
    #     name_parts = [f"{self._k(k)}={self._encode_val(v)}" for k, v in vars_mapping.items()]

    #     if prefix is not None:
    #         name_parts = [prefix] + name_parts 
    #     if suffix is not None:
    #         name_parts.append(suffix)

    #     for dir_name in path_parts:
    #         root = root / dir_name
    #     path = root / "-".join(name_parts)
    #     return path

    def get_filename(self, vars_mapping: Optional[Dict[str, Any]] = None, prefix: Optional[str] = None, suffix: Optional[str] = None)-> str:
        if vars_mapping is None:
            vars_mapping = {}
        name_parts = [f"{{{self._k(k)}={self._encode_val(v)}}}" for k, v in vars_mapping.items()]

        if prefix is not None:
            name_parts = [prefix] + name_parts 
        if suffix is not None:
            name_parts.append(suffix)
        return "-".join(name_parts)

    def get_dir(self) -> Path:
        root = self.constants_dir()
        path_parts = [f"{self._k(k)}={self._encode_val(v)}" for k, v in self.path_vars.items()]
        for dir in path_parts:
            root = root / dir
        return root

    def remove_vars(self, keys: Iterable[str]) -> None:
        """从当前path的变量中删除这些键（若不存在则忽略）。"""
        for k in keys:
            assert k in self.path_vars
            self.path_vars.pop(k, None)

    def clear_vars(self) -> None:
        """清空当前path变量。"""
        self.path_vars.clear()

    def active_vars(self) -> Dict[str, Any]:
        """返回当前活动变量的浅拷贝（保持顺序）。"""
        return dict(self.path_vars)

    # ---------- 路径与签名 ----------
    def _k(self, key: str) -> str:
        return self.aliases.get(key, key)

    def _encode_val(self, v: Any) -> str:
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, float):
            return f"{v:.6g}"
        if isinstance(v, (list, tuple)):
            if all(isinstance(x, int) for x in v):
                return "x".join(str(x) for x in v)   # 128x256
            return "+".join(self._encode_val(x) for x in v)
        if isinstance(v, dict):
            items = sorted(v.items(), key=lambda kv: kv[0])
            return ",".join(f"{self._encode_val(k)}={self._encode_val(val)}" for k, val in items)
        return str(v).replace("/", "_").replace(" ", "")

class LogManager:
    """
    非耦合、极简：
      - 只负责固定写入点 current/*；
      - archive(success) 基于 planner.combo_dir() + planner.signature() 归档；
      - 提供 open_stream/open_server/open_bench；
      - 不保存 bindings，全部交由 planner 的 extend/remove 控制。
    """

    def __init__(self, base_dir: Path, cfg: "Config"):
        self.base_dir = base_dir
        self.cfg = cfg
        self.planner = PathPlanner(base_dir, cfg)

        self._fds: Dict[str, Any] = {}
        self._dirty: bool = False

        # atexit.register(self._atexit_try_archive_failed)

    def write_constants_meta(self, vars: Optional[Dict[str, Any]] = None):
        const_dir = self.planner.constants_dir()
        C.print(f"[bold cyan] Writing constants meta to {const_dir}")
        const_dir.mkdir(parents=True, exist_ok=True)
        const_path = self.get_path_with_log_type("constants", "json", vars)
        if const_path.exists():
            os.remove(const_path)
        self.planner.write_constants_meta(const_path)
    # # ---- 日志流 ----
    # def open_stream_with_combo(self, vars_mapping: Optional[Dict[str, Any]] = None, prefix: Optional[str] = None, suffix: Optional[str] = None) -> FileIO:
    #     # 关闭句柄
    #     for f in list(self._fds.values()):
    #         try:
    #             if f and not f.closed: f.flush(); f.close()
    #         except Exception: pass
    #     self._fds.clear()

    #     combo_path = self.planner.combo_path_with_filename_vars(vars_mapping, prefix, suffix)
    #     combo_path.parent.mkdir(parents=True, exist_ok=True)
    #     f = open(combo_path, "ab", buffering=0)
    #     self._fds[str(combo_path)] = f
    #     self._dirty = True

    #     return f
 
    def get_filename_with_vars(self, vars_mapping: Optional[Dict[str, Any]] = None, prefix: Optional[str] = None, suffix: Optional[str] = None) -> str:
        return self.planner.get_filename(vars_mapping=vars_mapping, prefix=prefix, suffix=suffix)

    def get_dir(self) -> Path:
        return self.planner.get_dir()
    # def write_meta(self, extra: Optional[Dict[str, Any]] = None):
    #     meta = {
    #         "project": self.cfg.project,
    #         "model": self.cfg.model.model_dump(),
    #         "vllm": self.cfg.vllm.model_dump(),
    #         "benchmark": self.cfg.benchmark.model_dump(),
    #         "bindings": self.planner.active_vars(),
    #         "started_at": datetime.datetime.now().isoformat(),
    #     }
    #     if extra: meta.update(extra)
    #     self.meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    #     self._dirty = True

    # # ---- 归档 ----
    # def archive(self, success: bool) -> Path:
        

    #     # 移动/复制 current/*
    #     for fn in self.current_dir.glob("*"):
    #         if fn.is_file():
    #             self._move_or_copy(fn, dest / fn.name)

    #     # latest
    #     latest = dest.parent / "latest"
    #     try:
    #         if latest.exists() or latest.is_symlink(): latest.unlink()
    #         latest.symlink_to(dest.name)
    #     except Exception: pass

    #     # 截断 current/*
    #     for fn in self.current_dir.glob("*"):
    #         if fn.is_file(): self._truncate_file(fn)

    #     self._dirty = False
    #     return dest

    # # ---- atexit 兜底 ----
    # def _atexit_try_archive_failed(self):
    #     if not self._dirty: return
    #     try:
    #         self.archive(False)
    #     except Exception:
    #         pass

    # ---- utils ----
    def extend_path_with_vars(self, vars_mapping: Dict[str, Any], *, move_to_end: bool = True) -> None:
        self.planner.extend_path_with_vars(vars_mapping, move_to_end=move_to_end)
    def extend_base_dir_with_vars(self, vars_mapping: Dict[str, Any], *, move_to_end: bool = True) -> None:
        self.planner.extend_base_dir_with_vars(vars_mapping, move_to_end=move_to_end)
    def pop_vars_from_path(self, keys: Iterable[str]) -> None:
        self.planner.remove_vars(keys)


    def get_path_with_log_type(self, basename: str, type: str, vars: Optional[Dict[str, Any]] = None) -> Path:
        log_dir = self.get_dir()
        log_file_name_var_part: str = self.get_filename_with_vars(vars)
        filename = ""
        if log_file_name_var_part is None or log_file_name_var_part == "":
            filename = f"{basename}.{type}"
        else:
            filename = f"{basename}-{log_file_name_var_part}.{type}"
        log_dir.mkdir(parents=True, exist_ok=True)

        return log_dir / filename


class SweepLogManager:
    """Simplified log manager for sweep_test experiments.
    
    Provides a clean interface for managing log file paths based on
    sweep variable values, without the complexity of the legacy LogManager.
    """
    
    def __init__(self, base_dir: Path, cfg: SweepTestConfig):
        self.base_dir = base_dir
        self.cfg = cfg
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_dir(self) -> Path:
        return self.base_dir
    
    def write_constants_meta(self, vars_mapping: Optional[Dict[str, Any]] = None):
        """Write metadata about the experiment constants."""
        meta = {
            "project": self.cfg.project,
            "type": self.cfg.type,
            "static_config": self.cfg.static_config.model_dump(),
            "sweep_config": self.cfg.sweep_config.model_dump(),
            "timestamp": datetime.datetime.now().isoformat(),
            "experiment_vars": vars_mapping or {},
        }
        meta_path = self.get_path_with_log_type("constants", "json", vars_mapping)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    def get_path_with_log_type(self, basename: str, file_type: str, vars: Optional[Dict[str, Any]] = None) -> Path:
        """Generate a log file path based on basename, type, and variables.
        
        Variables are used to create a subdirectory, keeping filenames clean.
        
        Example: get_path_with_log_type("server", "log", {"flexi": True, "pp": "32,32"})
        Returns: base_dir / "test-{flexi=1}-{pp=32-32}" / "server.log"
        """
        if vars:
            var_parts = [f"{{{k}={self._encode_val(v)}}}" for k, v in vars.items()]
            subdir_name = f"test-{'-'.join(var_parts)}"
            test_dir = self.base_dir / subdir_name
            test_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{basename}.{file_type}"
            return test_dir / filename
        else:
            filename = f"{basename}.{file_type}"
            return self.base_dir / filename
    
    def _encode_val(self, v: Any) -> str:
        """Encode a value as a path-safe string."""
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, float):
            return f"{v:.6g}"
        if isinstance(v, (list, tuple)):
            if all(isinstance(x, int) for x in v):
                return "x".join(str(x) for x in v)
            return "+".join(self._encode_val(x) for x in v)
        return str(v).replace("/", "_").replace(" ", "").replace(",", "-")
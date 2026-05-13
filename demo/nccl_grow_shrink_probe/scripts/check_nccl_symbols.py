#!/usr/bin/env python3
import ctypes
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lib_path = root / "vendor" / "nvidia" / "nccl" / "lib" / "libnccl.so.2"
header_path = root / "vendor" / "nvidia" / "nccl" / "include" / "nccl.h"

lib = ctypes.CDLL(str(lib_path))
version = ctypes.c_int()
rc = lib.ncclGetVersion(ctypes.byref(version))
print(f"lib: {lib_path}")
print(f"ncclGetVersion rc={rc} version_int={version.value}")

for sym in [
    "ncclCommGrow",
    "ncclCommShrink",
    "ncclCommGetUniqueId",
    "ncclCommInitRankScalable",
    "ncclCommFinalize",
    "ncclCommRevoke",
]:
    print(f"{sym}: {hasattr(lib, sym)}")

with open(header_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#define NCCL_MAJOR") or line.startswith(
            "#define NCCL_MINOR"
        ) or line.startswith("#define NCCL_PATCH"):
            print(line.strip())

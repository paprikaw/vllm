#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize dynamic NCCL-vs-NIXL expand/shrink logs.

This is intentionally conservative: it reports metrics and red flags, but it
does not declare success unless all expected run directories exist and no
obvious correctness/runtime failures are found.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ERROR_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in [
        r"\btraceback\b",
        r"\bruntimeerror\b",
        r"\bassertionerror\b",
        r"\bnan\b",
        r"\binf\b",
        r"shape mismatch",
        r"dtype mismatch",
        r"descriptor",
        r"gibberish",
        r"jibish",
        r"乱码",
    ]
]

METRIC_PATTERNS = [
    re.compile(r"(throughput[^:=]*[:=]\s*[-+0-9.eE]+[^\n]*)",
               re.IGNORECASE),
    re.compile(r"(requests?/s[^:=]*[:=]\s*[-+0-9.eE]+[^\n]*)",
               re.IGNORECASE),
    re.compile(r"(migration[^:=]*(?:latency|time)[^:=]*[:=]\s*[-+0-9.eE]+[^\n]*)",
               re.IGNORECASE),
    re.compile(r"(ttft[^:=]*[:=]\s*[-+0-9.eE]+[^\n]*)",
               re.IGNORECASE),
]


def iter_text_files(run_dir: Path):
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {
                ".log", ".out", ".err", ".txt", ".json", ".jsonl"
        }:
            yield path


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception as exc:
        return f"<failed to read {path}: {exc}>"


def summarize_run(run_dir: Path) -> dict:
    text_blobs = []
    files = list(iter_text_files(run_dir))
    for path in files:
        text_blobs.append((path, read_text(path)))

    red_flags = []
    metrics = []
    for path, text in text_blobs:
        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in ERROR_PATTERNS):
                red_flags.append(f"{path}:{line_no}: {line[:240]}")
            for pattern in METRIC_PATTERNS:
                match = pattern.search(line)
                if match:
                    metrics.append(f"{path}:{line_no}: {match.group(1)}")

    return {
        "exists": run_dir.exists(),
        "files": len(files),
        "red_flags": red_flags,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_log_dir", type=Path)
    args = parser.parse_args()

    expected = [
        "expand-nccl",
        "expand-nixl",
        "shrink-nccl",
        "shrink-nixl",
    ]
    failed = False
    for name in expected:
        run_dir = args.base_log_dir / name
        summary = summarize_run(run_dir)
        print(f"=== {name} ===")
        print(f"dir={run_dir}")
        print(f"exists={summary['exists']} files={summary['files']}")
        if not summary["exists"] or summary["files"] == 0:
            failed = True
        if summary["metrics"]:
            print("metrics:")
            for metric in summary["metrics"][:40]:
                print(f"  {metric}")
        else:
            print("metrics: none found")
        if summary["red_flags"]:
            failed = True
            print("red_flags:")
            for red_flag in summary["red_flags"][:80]:
                print(f"  {red_flag}")
        else:
            print("red_flags: none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

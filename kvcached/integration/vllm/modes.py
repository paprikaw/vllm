# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Runtime mode selection for the vLLM integration."""

from __future__ import annotations

import os

from kvcached.integration.patch_base import enable_kvcached

VLLM_INTEGRATION_MODE_ENV = "KVCACHED_VLLM_INTEGRATION_MODE"

_NATIVE_MODES = {"native", "elastic", "kvcached"}
_VM_MODES = {"vm", "baseline", "off", "none", "disabled", "false", "0"}


def get_vllm_integration_mode() -> str:
    """Return the requested vLLM integration mode.

    ``native``/``elastic`` enables kvcached's existing vLLM patches
    (ElasticBlockPool, KV tensor allocation, worker init, NIXL compatibility).
    ``vm``/``off`` leaves vLLM's own implementation untouched so benchmark
    runs can compare against the existing VM/dynamic-resizing path.
    """
    return os.getenv(VLLM_INTEGRATION_MODE_ENV, "native").strip().lower()


def is_vllm_native_mode() -> bool:
    return get_vllm_integration_mode() in _NATIVE_MODES


def is_vllm_vm_mode() -> bool:
    return get_vllm_integration_mode() in _VM_MODES


def enable_vllm_kvcached_native() -> bool:
    """Whether kvcached should take over vLLM's KV cache integration."""
    return enable_kvcached() and is_vllm_native_mode()


def describe_vllm_integration_mode() -> str:
    mode = get_vllm_integration_mode()
    if mode in _NATIVE_MODES:
        return "native"
    if mode in _VM_MODES:
        return "vm"
    return f"unknown:{mode}"

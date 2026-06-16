# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from unittest import mock


_torch_mock = mock.MagicMock()
_torch_mock.__version__ = "2.6.0"
sys.modules.setdefault("torch", _torch_mock)
sys.modules.setdefault("torch.cuda", _torch_mock.cuda)
sys.modules.setdefault("torch.utils", _torch_mock.utils)
sys.modules.setdefault("torch.utils.cpp_extension", _torch_mock.utils.cpp_extension)
sys.modules.setdefault("posix_ipc", mock.MagicMock())

_wrapt_importer_mock = types.ModuleType("wrapt.importer")
_wrapt_importer_mock.when_imported = lambda _name: (lambda fn: fn)
_wrapt_mock = types.ModuleType("wrapt")
_wrapt_mock.importer = _wrapt_importer_mock
sys.modules.setdefault("wrapt", _wrapt_mock)
sys.modules.setdefault("wrapt.importer", _wrapt_importer_mock)


def test_vllm_integration_mode_defaults_to_native(monkeypatch):
    monkeypatch.delenv("KVCACHED_VLLM_INTEGRATION_MODE", raising=False)
    monkeypatch.setenv("ENABLE_KVCACHED", "1")

    modes = importlib.import_module("kvcached.integration.vllm.modes")

    assert modes.get_vllm_integration_mode() == "native"
    assert modes.describe_vllm_integration_mode() == "native"
    assert modes.enable_vllm_kvcached_native() is True


def test_vllm_integration_mode_can_keep_existing_vm_path(monkeypatch):
    monkeypatch.setenv("ENABLE_KVCACHED", "1")
    monkeypatch.setenv("KVCACHED_VLLM_INTEGRATION_MODE", "vm")

    modes = importlib.import_module("kvcached.integration.vllm.modes")

    assert modes.describe_vllm_integration_mode() == "vm"
    assert modes.is_vllm_vm_mode() is True
    assert modes.enable_vllm_kvcached_native() is False


def test_vllm_autopatch_skips_native_patches_in_vm_mode(monkeypatch):
    monkeypatch.setenv("ENABLE_KVCACHED", "1")
    monkeypatch.setenv("KVCACHED_AUTOPATCH", "1")
    monkeypatch.setenv("KVCACHED_VLLM_INTEGRATION_MODE", "vm")

    autopatch = importlib.import_module("kvcached.integration.vllm.autopatch")

    class _UnexpectedPatchManager:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("PatchManager should not be created in vm mode")

    monkeypatch.setattr(autopatch, "PatchManager", _UnexpectedPatchManager)
    autopatch._patch_vllm(types.ModuleType("vllm"))


def test_nixl_compat_uses_mode_aware_enable_alias(monkeypatch):
    monkeypatch.setenv("ENABLE_KVCACHED", "1")
    monkeypatch.setenv("KVCACHED_VLLM_INTEGRATION_MODE", "off")

    nixl_compat = importlib.import_module("kvcached.integration.vllm.nixl_compat")

    assert nixl_compat.enable_kvcached() is False

    monkeypatch.setenv("KVCACHED_VLLM_INTEGRATION_MODE", "native")
    assert nixl_compat.enable_kvcached() is True

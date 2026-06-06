# SPDX-License-Identifier: Apache-2.0
"""Direct NIXL tensor transport for dynamic KV migration.

This module intentionally keeps the existing DynamicKVSynchronizer control
protocol out of the data path.  Each worker owns one NIXL agent and dynamically
adds remote rank agents as peers appear.  Individual transfers exchange only
serialized tensor descriptors over the existing pair control pipe.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _resolve_nixl_host(host: str) -> str:
    """Resolve hostnames before passing endpoints to the NIXL listener API."""
    try:
        return socket.getaddrinfo(host, None, family=socket.AF_INET)[0][4][0]
    except socket.gaierror as exc:
        raise RuntimeError(f"Failed to resolve NIXL host {host!r}") from exc


@dataclass
class NixlRemoteTensor:
    xfer_id: str
    agent_name: str
    host: str
    port: int
    desc_bytes: bytes
    shape: tuple[int, ...]
    dtype: str


@dataclass
class _RegistrationHandle:
    descs: Any


def _load_nixl_runtime():
    try:
        from nixl import nixl_agent, nixl_agent_config

        return nixl_agent, nixl_agent_config
    except Exception as exc:
        raise RuntimeError(
            "Direct NIXL tensor transport requires the top-level nixl Python "
            "API (`from nixl import nixl_agent, nixl_agent_config`). Install "
            "a NIXL runtime that provides this API before setting "
            "VLLM_DYNAMIC_KV_TRANSPORT=nixl.") from exc


class DirectNixlTensorTransport:
    """One local NIXL agent with dynamically added remote rank peers."""

    def __init__(
        self,
        *,
        rank: int,
        local_rank: int,
        device: torch.device,
        rank_to_ip: dict[int, str],
        base_port: int,
        backend: str = "UCX",
        timeout_s: float = 30.0,
        agent_prefix: str = "vllm-dynamic-kv-rank",
    ) -> None:
        self.rank = int(rank)
        self.local_rank = int(local_rank)
        self.device = device
        self.rank_to_ip = {
            int(rank): _resolve_nixl_host(str(host))
            for rank, host in rank_to_ip.items()
        }
        self.base_port = int(base_port)
        self.backend = backend
        self.timeout_s = float(timeout_s)
        self.agent_prefix = agent_prefix
        self.agent_name = self._agent_name(self.rank)
        self.listen_port = self.base_port + self.rank
        self._lock = threading.Lock()
        self._remote_peers: set[int] = set()
        self._remote_peer_endpoints: dict[int, tuple[str, int]] = {}

        nixl_agent, nixl_agent_config = _load_nixl_runtime()
        config = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=True,
            listen_port=self.listen_port,
            backends=[backend],
        )
        self.agent = nixl_agent(self.agent_name, config)
        self._pending_sources: list[tuple[str, str, torch.Tensor, Any, float]] = []
        self._pending_source_limit = int(
            os.getenv("VLLM_DYNAMIC_NIXL_PENDING_SOURCE_LIMIT", "8"))
        logger.info(
            "Initialized direct NIXL tensor agent name=%s rank=%s port=%s "
            "backend=%s plugins=%s",
            self.agent_name,
            self.rank,
            self.listen_port,
            backend,
            getattr(self.agent, "get_plugin_list", lambda: [])(),
        )

    def _agent_name(self, rank: int) -> str:
        return f"{self.agent_prefix}-{rank}"

    def _peer_endpoint(self, peer_rank: int) -> tuple[str, int]:
        if peer_rank not in self.rank_to_ip:
            raise RuntimeError(
                f"Missing rank_to_ip entry for NIXL peer rank {peer_rank}")
        return self.rank_to_ip[peer_rank], self.base_port + peer_rank

    def _wait_until(self, description: str, predicate) -> Any:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.01)
        raise TimeoutError(description)

    def ensure_peer(self, peer_rank: int) -> str:
        peer_rank = int(peer_rank)
        host, port = self._peer_endpoint(peer_rank)
        return self.ensure_peer_at(peer_rank, host, port)

    def ensure_peer_at(self, peer_rank: int, host: str, port: int) -> str:
        peer_rank = int(peer_rank)
        endpoint = (host, int(port))
        if (peer_rank in self._remote_peers
                and self._remote_peer_endpoints.get(peer_rank) == endpoint):
            return self._agent_name(peer_rank)

        peer_name = self._agent_name(peer_rank)
        with self._lock:
            if (peer_rank in self._remote_peers
                    and self._remote_peer_endpoints.get(peer_rank) == endpoint):
                return peer_name
            if peer_rank in self._remote_peers:
                self.agent.remove_remote_agent(peer_name)
                self._remote_peers.discard(peer_rank)
                self._remote_peer_endpoints.pop(peer_rank, None)
            logger.info(
                "Adding direct NIXL peer rank=%s name=%s endpoint=%s:%s",
                peer_rank,
                peer_name,
                host,
                port,
            )
            self.agent.send_local_metadata(host, port)
            self.agent.fetch_remote_metadata(peer_name, host, port)
            self._wait_until(
                f"NIXL metadata for peer rank {peer_rank} did not arrive",
                lambda: self.agent.check_remote_metadata(peer_name),
            )
            self._remote_peers.add(peer_rank)
            self._remote_peer_endpoints[peer_rank] = endpoint
        return peer_name

    def refresh_peer_at(self, peer_rank: int, host: str, port: int) -> str:
        peer_rank = int(peer_rank)
        peer_name = self._agent_name(peer_rank)
        with self._lock:
            if peer_rank in self._remote_peers:
                try:
                    self.agent.remove_remote_agent(peer_name)
                finally:
                    self._remote_peers.discard(peer_rank)
                    self._remote_peer_endpoints.pop(peer_rank, None)
        return self.ensure_peer_at(peer_rank, host, port)

    def _local_endpoint(self) -> tuple[str, int]:
        if self.rank not in self.rank_to_ip:
            raise RuntimeError(
                f"Missing rank_to_ip entry for local NIXL rank {self.rank}")
        return self.rank_to_ip[self.rank], self.listen_port

    def _cleanup_completed_sources(self) -> None:
        if not self._pending_sources:
            return
        still_pending: list[tuple[str, str, torch.Tensor, Any, float]] = []
        for peer_name, xfer_id, tensor, reg_descs, created_at in self._pending_sources:
            try:
                done = self.agent.check_remote_xfer_done(
                    peer_name,
                    xfer_id.encode("utf-8"),
                    tag_is_prefix=False,
                )
            except Exception:
                done = False
            if done:
                self._deregister_tensor(reg_descs)
            else:
                still_pending.append(
                    (peer_name, xfer_id, tensor, reg_descs, created_at))
        self._pending_sources = still_pending

    def _retain_source_until_read_done(
        self,
        *,
        peer_name: str,
        xfer_id: str,
        tensor: torch.Tensor,
        reg_descs: Any,
    ) -> None:
        self._cleanup_completed_sources()
        self._pending_sources.append(
            (peer_name, xfer_id, tensor, reg_descs, time.monotonic()))
        if len(self._pending_sources) <= self._pending_source_limit:
            return

        peer_name, xfer_id, tensor, reg_descs, _ = self._pending_sources.pop(0)
        self._wait_until(
            f"NIXL source transfer {xfer_id} did not complete before the "
            "pending source limit was reached",
            lambda: self.agent.check_remote_xfer_done(
                peer_name,
                xfer_id.encode("utf-8"),
                tag_is_prefix=False,
            ),
        )
        self._deregister_tensor(reg_descs)
        del tensor

    def _register_tensor(self, tensor: torch.Tensor) -> _RegistrationHandle:
        return _RegistrationHandle(self.agent.register_memory(tensor))

    def _deregister_tensor(self, reg_descs: Any) -> None:
        if isinstance(reg_descs, _RegistrationHandle):
            reg_descs = reg_descs.descs
        try:
            self.agent.deregister_memory(reg_descs)
        except Exception:
            logger.exception("Failed to deregister NIXL tensor memory")

    def _tensor_desc_bytes(self, tensor: torch.Tensor) -> bytes:
        descs = self.agent.get_xfer_descs([tensor])
        return self.agent.get_serialized_descs(descs)

    def _deserialize_descs(self, payload: bytes) -> Any:
        return self.agent.deserialize_descs(payload)

    def send_tensor(self, peer_rank: int, tensor: torch.Tensor, pipe) -> None:
        """Expose tensor descriptors to peer and wait for READ completion."""
        peer_name = self.ensure_peer(peer_rank)
        xfer_id = uuid.uuid4().hex

        if tensor.is_cuda:
            torch.cuda.current_stream(tensor.device).synchronize()

        if tensor.numel() == 0:
            pipe.signal_group.send_obj(
                {
                    "type": "nixl_empty_tensor",
                    "xfer_id": xfer_id,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                },
                dst=pipe.peer_rank,
            )
            ack = pipe.signal_group.recv_obj(src=pipe.peer_rank)
            if not isinstance(ack, dict) or ack.get("xfer_id") != xfer_id:
                raise RuntimeError(f"Unexpected NIXL empty-tensor ack: {ack}")
            return

        send_tensor = tensor.contiguous()
        reg_descs = self._register_tensor(send_tensor)
        try:
            desc_bytes = self._tensor_desc_bytes(send_tensor)
            pipe.signal_group.send_obj(
                {
                    "type": "nixl_tensor_desc",
                    "xfer_id": xfer_id,
                    "desc_bytes": desc_bytes,
                    "shape": tuple(send_tensor.shape),
                    "dtype": str(send_tensor.dtype),
                    "peer_name": self.agent_name,
                },
                dst=pipe.peer_rank,
            )
            ack = pipe.signal_group.recv_obj(src=pipe.peer_rank)
            if (not isinstance(ack, dict) or ack.get("type") != "nixl_done"
                    or ack.get("xfer_id") != xfer_id):
                raise RuntimeError(f"Unexpected NIXL completion ack: {ack}")
            if ack.get("state") != "DONE":
                raise RuntimeError(f"NIXL transfer failed: {ack}")
            logger.info(
                "Direct NIXL send completed rank=%s peer=%s xfer_id=%s "
                "shape=%s dtype=%s",
                self.rank,
                peer_rank,
                xfer_id,
                tuple(send_tensor.shape),
                send_tensor.dtype,
            )
        finally:
            self._deregister_tensor(reg_descs)
            # Keep the source tensor alive until after deregistration and ack.
            del send_tensor
            del peer_name

    def prepare_read_source(self, peer_rank: int,
                            tensor: torch.Tensor) -> NixlRemoteTensor:
        """Register a source tensor and return metadata for a remote READ."""
        # The downstream PP stage may initialize its NIXL agent only after it
        # receives this metadata through Ray, so the source side must not block
        # on fetching downstream metadata here.
        peer_name = self._agent_name(peer_rank)
        xfer_id = uuid.uuid4().hex
        if tensor.is_cuda:
            torch.cuda.current_stream(tensor.device).synchronize()
        source_tensor = tensor.contiguous()
        reg_descs = self._register_tensor(source_tensor)
        try:
            host, port = self._local_endpoint()
            remote_tensor = NixlRemoteTensor(
                xfer_id=xfer_id,
                agent_name=self.agent_name,
                host=host,
                port=port,
                desc_bytes=self._tensor_desc_bytes(source_tensor),
                shape=tuple(source_tensor.shape),
                dtype=str(source_tensor.dtype),
            )
            self._retain_source_until_read_done(
                peer_name=peer_name,
                xfer_id=xfer_id,
                tensor=source_tensor,
                reg_descs=reg_descs,
            )
            return remote_tensor
        except Exception:
            self._deregister_tensor(reg_descs)
            raise

    def read_remote_tensor(
        self,
        *,
        peer_rank: int,
        remote_tensor: NixlRemoteTensor,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> torch.Tensor:
        """Read a tensor described in Ray metadata from a remote PP stage."""
        peer_name = self.ensure_peer_at(peer_rank, remote_tensor.host,
                                        remote_tensor.port)
        if remote_tensor.agent_name != peer_name:
            raise RuntimeError(
                "NIXL remote tensor agent mismatch: "
                f"expected={peer_name}, got={remote_tensor.agent_name}")
        if tuple(remote_tensor.shape) != tuple(shape):
            raise RuntimeError(
                "NIXL remote tensor shape mismatch: "
                f"meta={tuple(shape)}, desc={remote_tensor.shape}")
        if remote_tensor.dtype != str(dtype):
            raise RuntimeError(
                "NIXL remote tensor dtype mismatch: "
                f"meta={dtype}, desc={remote_tensor.dtype}")

        out = torch.empty(shape, dtype=dtype, device=self.device)
        if out.numel() == 0:
            return out

        reg_descs: Optional[Any] = None
        xfer_handle: Optional[Any] = None
        try:
            reg_descs = self._register_tensor(out)
            local_descs = self.agent.get_xfer_descs([out])
            remote_descs = self._deserialize_descs(remote_tensor.desc_bytes)
            tag = remote_tensor.xfer_id.encode("utf-8")
            try:
                xfer_handle = self.agent.initialize_xfer(
                    "READ",
                    local_descs,
                    remote_descs,
                    peer_name,
                    tag,
                )
            except Exception as exc:
                if "NIXL_ERR_NOT_FOUND" not in repr(exc):
                    raise
                logger.warning(
                    "NIXL PP READ initialize_xfer failed with %r; "
                    "refreshing peer metadata once rank=%s peer=%s "
                    "endpoint=%s:%s xfer_id=%s",
                    exc,
                    self.rank,
                    peer_rank,
                    remote_tensor.host,
                    remote_tensor.port,
                    remote_tensor.xfer_id,
                )
                peer_name = self.refresh_peer_at(peer_rank,
                                                 remote_tensor.host,
                                                 remote_tensor.port)
                remote_descs = self._deserialize_descs(
                    remote_tensor.desc_bytes)
                xfer_handle = self.agent.initialize_xfer(
                    "READ",
                    local_descs,
                    remote_descs,
                    peer_name,
                    tag,
                )
            state = self.agent.transfer(xfer_handle)
            if state == "ERR":
                raise RuntimeError(
                    f"Posting NIXL PP READ failed for peer rank {peer_rank}")
            self._wait_until(
                f"NIXL PP READ from peer rank {peer_rank} did not complete",
                lambda: self.agent.check_xfer_state(xfer_handle) == "DONE",
            )
            if out.is_cuda:
                torch.cuda.current_stream(out.device).synchronize()
            return out
        finally:
            if xfer_handle is not None:
                try:
                    self.agent.release_xfer_handle(xfer_handle)
                except Exception:
                    logger.exception("Failed to release NIXL PP xfer handle")
            if reg_descs is not None:
                self._deregister_tensor(reg_descs)

    def recv_tensor(
        self,
        peer_rank: int,
        dtype: torch.dtype,
        shape: torch.Size,
        pipe,
    ) -> torch.Tensor:
        """Receive a tensor by READing the peer's exposed NIXL descriptors."""
        peer_name = self.ensure_peer(peer_rank)
        descriptor_msg = pipe.signal_group.recv_obj(src=pipe.peer_rank)
        if not isinstance(descriptor_msg, dict):
            raise RuntimeError(
                f"Expected NIXL descriptor message, got {descriptor_msg}")

        xfer_id = descriptor_msg.get("xfer_id")
        expected_shape = tuple(shape)
        if tuple(descriptor_msg.get("shape", ())) != expected_shape:
            raise RuntimeError(
                "NIXL tensor shape mismatch: "
                f"meta={expected_shape}, desc={descriptor_msg.get('shape')}")
        if descriptor_msg.get("dtype") != str(dtype):
            raise RuntimeError(
                "NIXL tensor dtype mismatch: "
                f"meta={dtype}, desc={descriptor_msg.get('dtype')}")

        if descriptor_msg.get("type") == "nixl_empty_tensor":
            out = torch.empty(shape, dtype=dtype, device=self.device)
            pipe.signal_group.send_obj(
                {
                    "type": "nixl_done",
                    "xfer_id": xfer_id,
                    "state": "DONE",
                },
                dst=pipe.peer_rank,
            )
            return out

        if descriptor_msg.get("type") != "nixl_tensor_desc":
            raise RuntimeError(f"Unexpected NIXL descriptor: {descriptor_msg}")

        out = torch.empty(shape, dtype=dtype, device=self.device)
        if out.numel() == 0:
            return out

        reg_descs: Optional[Any] = None
        xfer_handle: Optional[Any] = None
        try:
            reg_descs = self._register_tensor(out)
            local_descs = self.agent.get_xfer_descs([out])
            remote_descs = self._deserialize_descs(descriptor_msg["desc_bytes"])
            tag = str(xfer_id).encode("utf-8")
            host, port = self._peer_endpoint(peer_rank)
            try:
                xfer_handle = self.agent.initialize_xfer(
                    "READ",
                    local_descs,
                    remote_descs,
                    peer_name,
                    tag,
                )
            except Exception as exc:
                if "NIXL_ERR_NOT_FOUND" not in repr(exc):
                    raise
                logger.warning(
                    "NIXL KV READ initialize_xfer failed with %r; "
                    "refreshing peer metadata once rank=%s peer=%s "
                    "xfer_id=%s",
                    exc,
                    self.rank,
                    peer_rank,
                    xfer_id,
                )
                peer_name = self.refresh_peer_at(peer_rank, host, port)
                remote_descs = self._deserialize_descs(
                    descriptor_msg["desc_bytes"])
                xfer_handle = self.agent.initialize_xfer(
                    "READ",
                    local_descs,
                    remote_descs,
                    peer_name,
                    tag,
                )
            state = self.agent.transfer(xfer_handle)
            if state == "ERR":
                raise RuntimeError(
                    f"Posting NIXL READ failed for peer rank {peer_rank}")
            self._wait_until(
                f"NIXL READ from peer rank {peer_rank} did not complete",
                lambda: self.agent.check_xfer_state(xfer_handle) == "DONE",
            )
            if out.is_cuda:
                torch.cuda.current_stream(out.device).synchronize()
            pipe.signal_group.send_obj(
                {
                    "type": "nixl_done",
                    "xfer_id": xfer_id,
                    "state": "DONE",
                },
                dst=pipe.peer_rank,
            )
            logger.info(
                "Direct NIXL recv completed rank=%s peer=%s xfer_id=%s "
                "shape=%s dtype=%s",
                self.rank,
                peer_rank,
                xfer_id,
                tuple(out.shape),
                out.dtype,
            )
            return out
        except Exception as exc:
            pipe.signal_group.send_obj(
                {
                    "type": "nixl_done",
                    "xfer_id": xfer_id,
                    "state": "ERR",
                    "error": repr(exc),
                },
                dst=pipe.peer_rank,
            )
            raise
        finally:
            if xfer_handle is not None:
                try:
                    self.agent.release_xfer_handle(xfer_handle)
                except Exception:
                    logger.exception("Failed to release NIXL transfer handle")
            if reg_descs is not None:
                self._deregister_tensor(reg_descs)

    def remove_peer(self, peer_rank: int) -> None:
        peer_rank = int(peer_rank)
        peer_name = self._agent_name(peer_rank)
        with self._lock:
            if peer_rank not in self._remote_peers:
                return
            try:
                self.agent.remove_remote_agent(peer_name)
            finally:
                self._remote_peers.discard(peer_rank)
                self._remote_peer_endpoints.pop(peer_rank, None)

    def close(self) -> None:
        for peer_rank in list(self._remote_peers):
            self.remove_peer(peer_rank)

"""
SPDX-License-Identifier: Apache-2.0

DynamicStageKVConnector: stage-level aggregated KV transfer between two ranks.

This connector aggregates per-stage KV updates from multiple layers into a
single batch and transfers them asynchronously to a peer rank. It reuses the
PairPipe used by DynamicLayerKVConnector for a 1:1 NCCL data plane and a TCP
store control plane.

Protocol (per batch):
- Control-plane: send one metadata object with type="kv_stage_batch" that
  includes: batch_id, layer_ids, num_tokens, has_hidden, and optional shapes.
- Data-plane: then send the following tensors in order using NCCL:
  1) dest_slot_mapping: int64[num_tokens]
  2) K_all: float[L, T, H, D]
  3) V_all: float[L, T, H, D]
  4) optional hidden: float[T, hidden_size]

Receiver expects this exact order and applies reordering/installation outside
this class.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

# Reuse the lightweight 1:1 pipe for rendezvous + NCCL from the dynamic layer
# connector implementation.
from vllm.distributed.kv_transfer.kv_connector.dynamic_layer_kv_connector import (  # noqa: E501
    PairPipe)


logger = init_logger(__name__)


class DynamicStageKVConnector:

    def __init__(self, rank: int, local_rank: int, config: VllmConfig):
        self.rank = rank
        self.local_rank = local_rank
        self.config = config.layer_kv_connector_config

        # Map (min_rank, max_rank) -> _PairPipe
        self._pair_pipes: Dict[Tuple[int, int], PairPipe] = {}

    def _pair_key(self, peer_rank: int) -> Tuple[int, int]:
        me = self.rank
        return (me, peer_rank) if me < peer_rank else (peer_rank, me)

    def _pair_port(self, peer_rank: int) -> int:
        """Deterministically derive a TCP port for (self.rank, peer_rank)."""
        base = self.config.kv_port
        a, b = self._pair_key(peer_rank)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                ws = torch.distributed.get_world_size()
            except Exception:
                ws = max(a, b) + 1
        else:
            ws = max(a, b) + 1
        return base + a * ws + b

    def _ensure_pipe(self, peer_rank: int) -> PairPipe:
        key = self._pair_key(peer_rank)
        if key in self._pair_pipes:
            return self._pair_pipes[key]

        a, b = key
        pair_rank = 0 if self.rank == a else 1
        port = self._pair_port(peer_rank)

        pipe = PairPipe(local_rank=self.local_rank,
                         host=self.config.kv_ip,
                         port=port,
                         pair_rank=pair_rank,
                         store_timeout_s=self.config.store_timeout_s)
        self._pair_pipes[key] = pipe
        return pipe

    # ========================= Sender APIs =========================
    def send_stage_batch(self,
                         peer_rank: int,
                         batch_id: int,
                         layer_ids: List[int],
                         dest_slot_mapping: torch.Tensor,
                         K_all: torch.Tensor,
                         V_all: torch.Tensor,
                         hidden: Optional[torch.Tensor] = None) -> None:
        """Send one aggregated stage batch to a peer.

        Args:
            peer_rank: destination global rank.
            batch_id: monotonically increasing id for ordering.
            layer_ids: list of layer indices contained in this batch, in the
                same order as the first dimension of K_all/V_all.
            dest_slot_mapping: int64 tensor [T] with destination slot indices.
            K_all, V_all: float tensors shaped [L, T, H, D].
            hidden: optional tensor [T, hidden_size].
        """
        pipe = self._ensure_pipe(peer_rank)

        assert dest_slot_mapping.dtype in (torch.int64, torch.long)
        assert K_all.dim() == 4 and V_all.dim() == 4, (
            "K_all/V_all must have shape [L, T, H, D]")
        assert K_all.shape == V_all.shape
        L, T, H, D = K_all.shape
        assert L == len(layer_ids), "layer_ids must match K_all/V_all[0]"

        has_hidden = hidden is not None

        ctrl_meta = {
            "type": "kv_stage_batch",
            "batch_id": int(batch_id),
            "layer_ids": list(map(int, layer_ids)),
            "num_tokens": int(T),
            "has_hidden": bool(has_hidden),
            "kv_shape": (int(L), int(T), int(H), int(D)),
        }

        # Control-plane metadata first.
        pipe._send_obj(ctrl_meta)  # type: ignore[attr-defined]

        # Data-plane tensors in fixed order.
        pipe.send(dest_slot_mapping)
        pipe.send(K_all)
        pipe.send(V_all)
        if has_hidden:
            pipe.send(hidden)

    # ========================= Receiver APIs =========================
    def recv_stage_batch_metadata(self, peer_rank: int) -> Optional[Dict[str, Union[int, bool, list, tuple]]]:
        """Blocking receive of one stage-batch control metadata.

        Returns None if a non-stage message is received (caller may ignore).
        """
        pipe = self._ensure_pipe(peer_rank)
        meta = pipe._recv_obj()  # type: ignore[attr-defined]
        if not isinstance(meta, dict) or meta.get("type") != "kv_stage_batch":
            logger.debug("Ignored non kv_stage_batch control message: %s", meta)
            return None
        return meta  # contains: batch_id, layer_ids, num_tokens, has_hidden, kv_shape

    def recv_stage_batch_tensors(self, peer_rank: int, meta: Dict[str, Union[int, bool, list, tuple]]
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Given control metadata, receive the corresponding tensor payloads.

        Returns: (dest_slot_mapping, K_all, V_all, hidden)
        """
        pipe = self._ensure_pipe(peer_rank)

        # 1) dest_slot_mapping
        t_meta = pipe.recv_metadata()
        dest_slot_mapping = pipe.recv_data(t_meta)

        # 2) K_all
        t_meta = pipe.recv_metadata()
        K_all = pipe.recv_data(t_meta)

        # 3) V_all
        t_meta = pipe.recv_metadata()
        V_all = pipe.recv_data(t_meta)

        hidden: Optional[torch.Tensor] = None
        if bool(meta.get("has_hidden", False)):
            t_meta = pipe.recv_metadata()
            hidden = pipe.recv_data(t_meta)

        return dest_slot_mapping, K_all, V_all, hidden

    def close(self) -> None:
        for pipe in self._pair_pipes.values():
            try:
                pipe.close()
            except Exception:
                pass


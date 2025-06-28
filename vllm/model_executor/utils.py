# SPDX-License-Identifier: Apache-2.0
"""Utils for model executor."""
from typing import Any, Optional

import torch


def set_random_seed(seed: int) -> None:
    from vllm.platforms import current_platform
    current_platform.seed_everything(seed)


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        from vllm.platforms import current_platform
        if current_platform.is_tpu() and key == "weight_loader":
            value = _make_synced_weight_loader(value)
        setattr(weight, key, value)


def _make_synced_weight_loader(original_weight_loader):

    def _synced_weight_loader(param, *args, **kwargs):
        original_weight_loader(param, *args, **kwargs)
        torch._sync(param)

    return _synced_weight_loader

def extract_layer_index(layer_name: str) -> int:
    """
    Copied from vllm.model_executor.models.utils.extract_layer_index
    to avoid circular import.
    Extract the layer index from the module name.
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError
    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    assert len(int_vals) == 1, (f"layer name {layer_name} should"
                                " only contain one integer")
    return int_vals[0]
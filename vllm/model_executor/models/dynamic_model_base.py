"""Base interface for dynamic models supporting PP reconfiguration.

Dynamic models inherit from this marker class to indicate they support
runtime pipeline parallelism reconfiguration (add/delete layers, 
scheduled layer execution, fbgate-aware weight loading).

CausalLM wrapper (e.g. DynamicLlamaForCausalLM) must implement:
    - set_sched_layers(start, end)
    - get_sched_layers() -> (start, end)
    - add_layers(layers)
    - delete_layers(layers)
    - get_layer_weight_size() -> int
    - add_fbgate(fbgate)
    - forward(input_ids, positions, intermediate_tensors, inputs_embeds)
    - load_weights(weights) -> set[str]

Inner model (.model) must expose:
    - start_layer, end_layer (int)
    - layers (nn.ModuleList)
    - config (HF config)
    - add_fbgate(fbgate)
"""

from typing import Tuple

import torch
from torch import nn

from vllm.model_executor.models.utils import (
    LayerFn, PPMissingLayer, maybe_offload_to_cpu)
from vllm.logger import init_logger

logger = init_logger(__name__)


class DynamicModelBase:
    """Marker base class for dynamic models supporting PP reconfiguration."""
    pass


def add_layers(
    module: torch.nn.ModuleList,
    added_layers: Tuple[int, int],
    old_layers: Tuple[int, int],
    layer_fn: LayerFn,
    prefix: str,
) -> torch.nn.ModuleList:
    """
    Build a new ModuleList by merging existing layers with newly created ones.

    Ranges are inclusive: [start, end].
    - module:       current ModuleList (num_hidden_layers long)
    - added_layers: inclusive range of new layers to insert
    - old_layers:   inclusive range of existing real layers
    - layer_fn:     factory ``fn(prefix) -> nn.Module``
    - prefix:       naming prefix for the new layers
    """
    num_layers = len(module)
    new_module = torch.nn.ModuleList()
    assert added_layers[0] <= added_layers[1], \
        "added_layers[0] must be <= added_layers[1]"
    old_layers_empty = old_layers[0] > old_layers[1]
    assert (old_layers_empty
            or added_layers[1] == old_layers[0] - 1
            or added_layers[0] == old_layers[1] + 1), \
        (f"added_layers must be adjacent to old_layers, "
         f"added_layers:{added_layers}, old_layers:{old_layers}")

    for idx in range(num_layers):
        if not old_layers_empty and old_layers[0] <= idx <= old_layers[1]:
            new_module.append(module[idx])
        elif added_layers[0] <= idx <= added_layers[1]:
            new_module.append(
                maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}")))
        else:
            new_module.append(module[idx])
    return new_module

from typing import List, Tuple

def get_new_layer_config_with_migration_action(rank_from: int, rank_to: int, num_layers: int, current_layer_config: List[Tuple[int,int]]) -> Tuple[Tuple[int,int], List[Tuple[int,int]]]:
        # Migration is only supported for adjacent ranks at the moment
        assert abs(rank_from - rank_to) == 1
        rank_from_layer = current_layer_config[rank_from]
        rank_to_layer = current_layer_config[rank_to]

        # Check if rank_from has enough layers 
        assert rank_from_layer[1] - rank_from_layer[0] + 1 >= num_layers

        # Decide the layer range to added
        next_layer_config: List[Tuple[int,int]]
        if rank_from < rank_to:
            layers = (rank_from_layer[1] - num_layers + 1, rank_from_layer[1])
            rank_to_layer = (layers[0], rank_to_layer[1])
            rank_from_layer = (rank_from_layer[0], layers[0] - 1)
            next_layer_config = [rank_from_layer, rank_to_layer]
        else:
            layers = (rank_from_layer[0], rank_from_layer[0] + num_layers - 1)
            rank_to_layer = (rank_to_layer[0], layers[1])
            rank_from_layer = (layers[1] + 1, rank_from_layer[1])
            next_layer_config = [rank_to_layer, rank_from_layer]
        return layers, next_layer_config
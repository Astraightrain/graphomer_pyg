import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


class ShortestPathGenerator:
    def __init__(self, max_num_nodes: int):
        self.max_num_nodes = max_num_nodes

    def __call__(self, data: Data):
        G = to_networkx(data)
        shortest_path = dict(nx.all_pairs_shortest_path_length(G))
        dist_matrix = np.full((self.max_num_nodes, self.max_num_nodes), -1)
        np.fill_diagonal(dist_matrix, 0)
        for source, paths in shortest_path.items():
            for target, dist in paths.items():
                dist_matrix[source, target] = dist
        data.hop = torch.from_numpy(dist_matrix)
        return data

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch
from typing import List, Optional
from graphomer.datamodule.featurizer.features import (
    ATOM_FEATURES_DIM,
    BOND_FEATURES_DIM,
)
from torch import Tensor
from einops import rearrange


class DiscreteEmbedding(nn.Module):
    def __init__(self, d_model: int = 768, d_features: List[int] = None):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(d_feature + 1, d_model) for d_feature in d_features]
        )  # + 1 for virtual node
        self._init_weights()

    def _init_weights(self):
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, x: Tensor):
        if x.dim() != 2 or x.size(1) != len(self.embeddings):
            raise ValueError("Input tensor must have shape (batch_size, num_features)")

        embeddings = [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)]
        stacked_embeddings = torch.stack(embeddings, dim=0)
        return torch.sum(stacked_embeddings, dim=0)


class ContinuousEmbedding(nn.Module):
    def __init__(self, d_model: int = 768, d_features: List[int] = None):
        self.embeddings = nn.ModuleList([nn.Linear(d_feature, d_model) for d_feature in d_features])
        self._init_weights()

    def _init_weights(self):
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, x: Tensor):
        if x.dim() != 2 or x.size(1) != len(self.embeddings):
            raise ValueError("Input tensor must have shape (batch_size, num_features)")

        embeddings = [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)]
        stacked_embeddings = torch.stack(embeddings, dim=0)
        return torch.sum(stacked_embeddings, dim=0)


class GraphEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        atom_features_dim: List[int] = None,
        bond_features_dim: List[int] = None,
        max_in_degree: int = 10,
        max_out_degree: int = 10,
        cont_features_dim: Optional[List] = None,
    ):
        super().__init__()
        # Atom and Bond embedding
        self.atom_embedding = DiscreteEmbedding(d_model, atom_features_dim)
        self.bond_embedding = DiscreteEmbedding(d_model, bond_features_dim)

        # Centrality embedding
        self.z_in = DiscreteEmbedding(d_model, [max_in_degree])
        self.z_out = DiscreteEmbedding(d_model, [max_out_degree])

    def forward(self, batch: Batch):
        batch.x = self.atom_embedding(batch.x)
        batch.edge_attr = self.bond_embedding(batch.edge_attr)
        z_in = self.z_in(batch.degree)
        z_out = self.z_out(batch.degree)  # same degree for undirected graph
        batch.x = batch.x + z_in + z_out

        return batch


class VirtualEmbedding(nn.Module):
    def __init__(self, d_model: int = 768, num_tasks: int = 1, depth: int = 1):
        super().__init__()
        self.virtual_x = nn.Embedding(1, d_model)
        self.virtual_edge_attr = nn.Embedding(1, d_model)

    def forward(self, batch: Batch):
        x, mask = to_dense_batch(batch.x, batch.batch)
        b, n, d = x.size()  # [batch size, max_num_nodes, d_model]
        # virtual node index => 0
        x_with_vn = torch.zeros(b, n + 1, d, dtype=torch.float, device=batch.x.device)
        x_with_vn[:, 1:, :] = x
        x_with_vn[:, 0, :] = self.virtual_x(torch.tensor([0], device=batch.x.device))

        edge_adj_matrix = to_dense_adj(batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        edge_adj_with_vn = torch.zeros(b, n + 1, n + 1, d, dtype=torch.float, device=batch.edge_index.device)
        edge_adj_with_vn[:, 1:, 1:, :] = edge_adj_matrix
        edge_adj_with_vn[:, 0, 1:, :] = self.virtual_edge_attr(torch.tensor([0], device=batch.edge_index.device))
        edge_adj_with_vn[:, 1:, 0, :] = self.virtual_edge_attr(torch.tensor([0], device=batch.edge_index.device))

        batch.x = x_with_vn
        batch.edge_attr = edge_adj_with_vn
        batch.mask = mask

        return batch


if __name__ == "__main__":
    from graphomer.batchmodule import GraphFeaturizer
    from graphomer.batchmodule.featurizer.features import (
        ATOM_FEATURES_DIM,
        BOND_FEATURES_DIM,
    )

    featurizer = GraphFeaturizer()
    atom_features_dim = list(ATOM_FEATURES_DIM.values())
    bond_features_dim = list(BOND_FEATURES_DIM.values())
    graph_encoder = GraphEmbedding(
        d_model=128,
        atom_features_dim=atom_features_dim,
        bond_features_dim=bond_features_dim,
    )

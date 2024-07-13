import torch.nn as nn
from .layers import GraphomerBlock
from .embeddings import GraphEmbedding
from torch_geometric.data import Data


class Graphomer(nn.Module):

    def __init__(
        self,
        d_model=128,
        num_heads=4,
        d_ff=128,
        num_layers=4,
        dropout=0.1,
        attn_dropout=0.1,
        max_num_nodes=128,
        max_num_edges=256,
        atom_features_dim=[118, 18, 6, 4, 11, 11, 9, 5, 5, 2, 2],
        bond_features_dim=[4, 6, 2],
        max_in_degree=10,
        max_out_degree=10,
    ):
        super().__init__()
        self.atom_features_dim = atom_features_dim
        self.bond_features_dim = bond_features_dim

        self.embeddings = GraphEmbedding(
            d_model, atom_features_dim, bond_features_dim, max_in_degree=max_in_degree, max_out_degree=max_out_degree
        )

        self.virtual_embeddings = VirtualNodeEmbedding(d_model)

        self.layers = nn.ModuleList(
            [GraphomerBlock(d_model, num_heads, d_ff, dropout, attn_dropout) for _ in range(num_layers)]
        )

    def forward(self, batch: Data):
        batch = self.embeddings(batch)
        x = batch.x
        for layer in self.layers:
            x = layer(x)
        return x

import torch
import torch.nn as nn
from torch_geometric.data import Data


class DiscreteEmbedding(nn.Module):
    def __init__(self, d_model, d_features):
        super(DiscreteEmbedding, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(d_feature, d_model) for d_feature in d_features
        ])
        self._init_weights()

    def _init_weights(self):
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        
    def forward(self, x):
        if x.dim() != 2 or x.size(1) != len(self.embeddings):
            raise ValueError("Input tensor must have shape (batch_size, num_features)")
            
        embeddings = [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)]
        stacked_embeddings = torch.stack(embeddings, dim=0)
        return torch.sum(stacked_embeddings, dim=0)
    
class GraphEmbedding(nn.Module):
    def __init__(self, d_model, atom_features_dim, bond_features_dim):
        super().__init__()
        self.atom_embedding = DiscreteEmbedding(d_model, atom_features_dim)
        self.bond_embedding = DiscreteEmbedding(d_model, bond_features_dim)

    def forward(self, data: Data):
        data.x = self.atom_embedding(data.x)
        data.edge_attr = self.bond_embedding(data.edge_attr)
        return data


if __name__=="__main__":
    from graphomer.datamodule import GraphFeaturizer
    from graphomer.datamodule.featurizer.features import ATOM_FEATURES_DIM, BOND_FEATURES_DIM
    featurizer = GraphFeaturizer()
    atom_features_dim = list(ATOM_FEATURES_DIM.values())
    bond_features_dim = list(BOND_FEATURES_DIM.values())
    graph_encoder = GraphEmbedding(d_model=128, atom_features_dim=atom_features_dim, bond_features_dim=bond_features_dim)
    
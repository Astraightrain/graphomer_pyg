from torch_geometric.data import Dataset
import pandas as pd
from .featurizer import GraphFeaturizer
from .shortest_path import ShortestPathGenerator
import torch


class PCQM4MDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.featurizer = GraphFeaturizer()
        self.shortest_path = ShortestPathGenerator(max_num_nodes=29)
        self.dataset = pd.read_csv(root)

    def process(self, idx):
        smiles = self.dataset["smiles"][idx]
        data = self.featurizer(smiles)
        data = self.shortest_path(data)
        data.y = torch.tensor(self.dataset["homolumogap"][idx])
        return data

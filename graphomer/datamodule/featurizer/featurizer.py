import abc
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit.Chem.rdchem import Atom, Bond, Mol
from typing import Callable, Optional

from .features import VALID_ATOM_FEATURES, VALID_BOND_FEATURES, ATOMNUM2GROUP, ATOMNUM2PERIOD
from .utils import  safe_index, rdkit_parser

# Base class for featurizers
class FeaturizerBase(abc.ABC):
    @abc.abstractmethod
    def featurize(self, mol: Mol) -> Data:
        pass

    def __init__(self):
        self._parser_func = rdkit_parser
        self._preprocess_func = None

    @property
    def parse_input(self) -> Callable[[str], Mol]:
        return self._parser_func

    @parse_input.setter
    def parse_input(self, parser_func: Optional[Callable[[str], Mol]] = None):
        if parser_func and not callable(parser_func):
            raise ValueError(f"Parser function must be callable, got {type(parser_func)}")
        self._parser_func = parser_func or rdkit_parser

    @property
    def preprocess_input(self) -> Optional[Callable[[Mol], Mol]]:
        return self._preprocess_func

    @preprocess_input.setter
    def preprocess_input(self, preprocess_func: Optional[Callable[[Mol], Mol]] = None):
        if preprocess_func and not callable(preprocess_func):
            raise ValueError(f"Preprocess function must be callable, got {type(preprocess_func)}")
        self._preprocess_func = preprocess_func

    def __call__(self, smiles_or_path: str) -> Data:
        mol = self.parse_input(smiles_or_path)
        if self.preprocess_input:
            mol = self.preprocess_input(mol)
        return self.featurize(mol)

    def __repr__(self) -> str:
        attributes = sorted((k, v) for k, v in self.__dict__.items() if not k.startswith("_"))
        fields = [f"{k}={v}" for k, v in attributes]
        return f"{self.__class__.__name__}(" + ", ".join(fields) + ")"

# Molecule featurizer class
class GraphFeaturizer(FeaturizerBase):
    def featurize(self, mol: Mol) -> Data:
        return self._mol_to_graph(mol)

    @staticmethod
    def atom_featurizer(atom: Atom) -> np.ndarray:
        atomic_num = atom.GetAtomicNum()
        features = [
            safe_index(VALID_ATOM_FEATURES["atom_num"], atomic_num),
            safe_index(VALID_ATOM_FEATURES["group"], ATOMNUM2GROUP[atomic_num]),
            safe_index(VALID_ATOM_FEATURES["period"], ATOMNUM2PERIOD[atomic_num]),
            safe_index(VALID_ATOM_FEATURES["chirality"], str(atom.GetChiralTag())),
            safe_index(VALID_ATOM_FEATURES["degree"], atom.GetTotalDegree()),
            safe_index(VALID_ATOM_FEATURES["formal_charge"], atom.GetFormalCharge()),
            safe_index(VALID_ATOM_FEATURES["num_H"], atom.GetTotalNumHs()),
            safe_index(VALID_ATOM_FEATURES["num_radical"], atom.GetNumRadicalElectrons()),
            safe_index(VALID_ATOM_FEATURES["hybridization"], str(atom.GetHybridization())),
            safe_index(VALID_ATOM_FEATURES["is_aromatic"], atom.GetIsAromatic()),
            safe_index(VALID_ATOM_FEATURES["is_in_ring"], atom.IsInRing())
        ]
        return np.array(features, dtype=np.int64)

    @staticmethod
    def bond_featurizer(bond: Bond) -> np.ndarray:
        features = [
            safe_index(VALID_BOND_FEATURES["bond_type"], str(bond.GetBondType())),
            safe_index(VALID_BOND_FEATURES["stereo"], str(bond.GetStereo())),
            safe_index(VALID_BOND_FEATURES["is_conjugated"], bond.GetIsConjugated())
        ]
        return np.array(features, dtype=np.int64)

    @staticmethod
    def _mol_to_graph(mol: Mol) -> Data:
        atom_features = []
        atom_idx_map = {}

        for idx, atom in enumerate(mol.GetAtoms()):
            atom_features.append(GraphFeaturizer.atom_featurizer(atom))
            atom_idx_map[atom.GetIdx()] = idx

        x = np.stack(atom_features, axis=0)
        
        edges = []
        edge_features = []

        for bond in mol.GetBonds():
            i = atom_idx_map[bond.GetBeginAtomIdx()]
            j = atom_idx_map[bond.GetEndAtomIdx()]
            edge_feat = GraphFeaturizer.bond_featurizer(bond)

            edges.extend([(i, j), (j, i)])
            edge_features.extend([edge_feat, edge_feat])

        edge_index = np.array(edges, dtype=np.int64).T
        edge_attr = np.array(edge_features, dtype=np.int64)

        return Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr)
        )

if __name__ == "__main__":
    smiles = 'c1ccccc1'
    featurizer = GraphFeaturizer()
    print(featurizer(smiles))

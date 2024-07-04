from pathlib import Path
from typing import Union
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

# Utility function to safely get index
def safe_index(lst, element):
    try:
        return lst.index(element)
    except ValueError:
        return len(lst) - 1

# RDKit parser
def rdkit_parser(smiles_or_path: Union[str, Path]) -> Mol:
    path = Path(smiles_or_path)
    if path.exists():
        raise NotImplementedError(f"File input is not supported: {path}")
    mol = Chem.MolFromSmiles(smiles_or_path)
    if not mol:
        raise ValueError(f"Invalid SMILES string: {smiles_or_path}")
    return mol

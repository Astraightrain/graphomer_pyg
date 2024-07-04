import os
from collections import defaultdict
import pandas as pd

# Load periodic table data
PERIODIC_TABLE = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "periodic_table.csv")
)

# Maps for atom group and period based on atomic number
ATOMNUM2GROUP = defaultdict(
    lambda: -1, {k: v for k, v in PERIODIC_TABLE[["atomic_num", "group"]].values}
)
ATOMNUM2PERIOD = defaultdict(
    lambda: -1, {k: v for k, v in PERIODIC_TABLE[["atomic_num", "period"]].values}
)

# Define valid features for atoms and bonds
VALID_ATOM_FEATURES = {
    "atom_num": list(range(1, 119)),
    "group": list(range(1, 19)),
    "period": list(range(1, 7)),
    "chirality": ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"],
    "degree": list(range(11)),
    "formal_charge": list(range(-5, 6)),
    "num_H": list(range(9)),
    "num_radical": list(range(5)),
    "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2"],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True]
}

VALID_BOND_FEATURES = {
    "bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    "stereo": ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"],
    "is_conjugated": [False, True]
}

ATOM_FEATURES_DIM = {k: len(v) for k, v in VALID_ATOM_FEATURES.items()}
BOND_FEATURES_DIM = {k: len(v) for k, v in VALID_BOND_FEATURES.items()}

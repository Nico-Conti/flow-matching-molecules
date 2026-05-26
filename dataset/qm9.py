import numpy as np
from rdkit import RDLogger
from torch_molecule.datasets import load_qm9 as _tm_load_qm9

from .featurize import QM9_ATOMS
from .filtering import featurize_dataset

RDLogger.DisableLog("rdApp.*")

# EDM-six DFT targets (covers FreeGress mu/HOMO + MolGuidance/VI-VFM full set).
# Quantum-chemical properties shipped with QM9; NOT RDKit-recomputable.
QM9_TARGETS_DEFAULT = ("mu", "alpha", "homo", "lumo", "gap", "cv")


def load_qm9(local_dir="data", targets=QM9_TARGETS_DEFAULT, apply_filter=False, limit=None):
    # torch_molecule's loader (fetches the HF-mirrored QM9): neutral SMILES + DFT target columns.
    ds = _tm_load_qm9(local_dir=local_dir, target_cols=list(targets))
    smiles = list(ds.data)
    y_all = np.asarray(ds.target, dtype="float32")
    if limit:
        smiles, y_all = smiles[:limit], y_all[:limit]

    # charge_aware=True unified with ZINC; QM9 is neutral so no charged tokens
    # are expected (any that appear fall out as drop_vocab). Round-trip report-only.
    Xs, Es, kept_idx, stats = featurize_dataset(
        smiles, QM9_ATOMS, charge_aware=True, uncharge=False, apply_filter=apply_filter)

    return {
        "X": Xs,
        "E": Es,
        "y": y_all[kept_idx],
        "smiles": [smiles[k] for k in kept_idx],
        "atom_vocab": QM9_ATOMS,
        "targets": tuple(targets),
        "stats": stats,
    }

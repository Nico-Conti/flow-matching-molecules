import os
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, QED
from torch_molecule.datasets import load_zinc250k

from .featurize import ZINC_ATOMS
from .filtering import featurize_dataset

RDLogger.DisableLog("rdApp.*")

ZINC_TARGETS_DEFAULT = ("logP", "qed", "SAS")


def _sascorer():
    # rdkit's bundled Contrib SA_Score (ships fpscores.pkl.gz); no vendoring.
    from rdkit.Chem import RDConfig
    sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_dir not in sys.path:
        sys.path.append(sa_dir)
    import sascorer
    return sascorer


def _compute_targets(smiles, targets):
    sa = _sascorer() if "SAS" in targets else None
    fns = {
        "logP": Crippen.MolLogP,
        "qed": QED.qed,
        "SAS": (lambda m: sa.calculateScore(m)) if sa else None,
    }
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([fns[t](m) for t in targets])
    return np.asarray(rows, dtype="float32")


def load_zinc(local_dir="data", targets=ZINC_TARGETS_DEFAULT, apply_filter=True,
              uncharge=False, limit=None):
    # torch_molecule ZINC-250k (HuggingFace): ~250k SMILES, no targets.
    ds = load_zinc250k(local_dir=local_dir)
    smiles = list(ds.data)
    if limit:
        smiles = smiles[:limit]

    # keep N+/O- (charge_aware=True); round-trip filter applied.
    Xs, Es, kept_idx, stats = featurize_dataset(
        smiles, ZINC_ATOMS, charge_aware=True, uncharge=uncharge, apply_filter=apply_filter)

    kept_smiles = [smiles[k] for k in kept_idx]
    return {
        "X": Xs,
        "E": Es,
        "y": _compute_targets(kept_smiles, targets),
        "smiles": kept_smiles,
        "atom_vocab": ZINC_ATOMS,
        "targets": tuple(targets),
        "stats": stats,
    }

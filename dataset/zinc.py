import os
import urllib.request

import pandas as pd
from rdkit import RDLogger

from .featurize import ZINC_ATOMS
from .filtering import featurize_dataset

RDLogger.DisableLog("rdApp.*")


def download(url: str, dest: str) -> str:
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest

ZINC_CSV = ("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
            "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")
ZINC_TARGETS_DEFAULT = ("logP", "qed")


def load_zinc(local_dir="data", targets=ZINC_TARGETS_DEFAULT, apply_filter=True,
              uncharge=False, limit=None):

    path = download(ZINC_CSV, os.path.join(local_dir, "zinc", "zinc250k.csv"))
    df = pd.read_csv(path)
    df["smiles"] = df["smiles"].str.strip()           # rows carry a trailing newline
    if limit:
        df = df.iloc[:limit]
    smiles = df["smiles"].tolist()

    Xs, Es, kept_idx, stats = featurize_dataset(
        smiles, ZINC_ATOMS, charge_aware=True, uncharge=uncharge, apply_filter=apply_filter)

    y = df.iloc[kept_idx][list(targets)].to_numpy(dtype="float32")
    return {
        "X": Xs,
        "E": Es,
        "y": y,
        "smiles": [smiles[k] for k in kept_idx],
        "atom_vocab": ZINC_ATOMS,
        "targets": tuple(targets),
        "stats": stats,
    }

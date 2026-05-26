import os
import urllib.request
import zipfile

import pandas as pd
from rdkit import Chem, RDLogger

from .featurize import QM9_ATOMS
from .filtering import featurize_dataset

RDLogger.DisableLog("rdApp.*")


def download(url: str, dest: str) -> str:
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def download_and_extract(url: str, dest_zip: str, extract_dir: str, marker: str) -> str:
    download(url, dest_zip)
    if not os.path.exists(marker):
        with zipfile.ZipFile(dest_zip) as z:
            z.extractall(extract_dir)
    return marker

QM9_ZIP = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
QM9_UNCHARACTERIZED = "https://ndownloader.figshare.com/files/3195404"
QM9_TARGETS_DEFAULT = ("mu", "homo")


# Skips uncharactirized molecules (following digress code)
def _load_skip(path: str):
    with open(path) as f:
        return {int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]}


def load_qm9(local_dir="data", targets=QM9_TARGETS_DEFAULT, apply_filter=False, limit=None):
    root = os.path.join(local_dir, "qm9")
    download_and_extract(QM9_ZIP, os.path.join(root, "qm9.zip"), root,
                         marker=os.path.join(root, "gdb9.sdf"))
    skip = _load_skip(download(QM9_UNCHARACTERIZED, os.path.join(root, "uncharacterized.txt")))

    props = pd.read_csv(os.path.join(root, "gdb9.sdf.csv"))
    suppl = Chem.SDMolSupplier(os.path.join(root, "gdb9.sdf"), removeHs=True, sanitize=True)

    # We converto mol to smile since sdf is connection-table format  
    smiles, sdf_rows = [], []
    for i, mol in enumerate(suppl):
        if i in skip or mol is None:
            continue
        smiles.append(Chem.MolToSmiles(mol))
        sdf_rows.append(i)
        if limit and len(smiles) >= limit:
            break

    Xs, Es, kept_idx, stats = featurize_dataset(
        smiles, QM9_ATOMS, charge_aware=False, uncharge=False, apply_filter=apply_filter)

    rows = [sdf_rows[k] for k in kept_idx]
    y = props.iloc[rows][list(targets)].to_numpy(dtype="float32")
    return {
        "X": Xs,
        "E": Es,
        "y": y,
        "smiles": [smiles[k] for k in kept_idx],
        "atom_vocab": QM9_ATOMS,
        "targets": tuple(targets),
        "stats": stats,
    }

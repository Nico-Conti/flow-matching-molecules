#!/usr/bin/env python3
"""Evaluate released DeFoG MOSES graphs with the thesis's existing FCD metric."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import pickle
import sys
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm

SAMPLES_URL = (
    "https://drive.switch.ch/index.php/s/MG7y2EZoithAywE/download"
    "?path=%2Fmoses&files=generated_samples.pkl"
)
MOSES_URL = (
    "https://media.githubusercontent.com/media/molecularsets/moses/"
    "master/moses/dataset/data/{split}.csv.gz"
)
ATOMS = ("C", "N", "S", "O", "F", "Cl", "Br", "H")
BONDS = (None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
         Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)


def download(url, path):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        expected = int(response.headers.get("Content-Length", 0))
        with tqdm(total=expected or None, unit="B", unit_scale=True,
                  desc=path.name) as bar:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                bar.update(len(chunk))
            if expected and out.tell() != expected:
                raise IOError(f"Incomplete download: {path}")
    partial.replace(path)
    return path


def _storage_on_cpu(data):
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


class _TensorUnpickler(pickle.Unpickler):
    # Public samples are ordinary pickle lists of tensor pairs, not torch.save files.
    # Restrict globals and remap serialized CUDA storage for CPU-only evaluation.
    def find_class(self, module, name):
        allowed = {
            ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            ("torch.storage", "_load_from_bytes"): _storage_on_cpu,
            ("collections", "OrderedDict"): collections.OrderedDict,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported sample pickle global: {module}.{name}")
        return allowed[module, name]


def load_samples(path):
    with path.open("rb") as stream:
        graphs = _TensorUnpickler(stream).load()
    if not isinstance(graphs, (list, tuple)):
        raise ValueError("Expected a list of (atom_indices, bond_indices) graphs")
    return graphs


def select_samples(graphs, n_samples, offset):
    if n_samples <= 0 or offset < 0 or offset + n_samples > len(graphs):
        raise ValueError(f"Requested [{offset}:{offset + n_samples}] from {len(graphs)} graphs")
    return graphs[offset:offset + n_samples]


def decode_graph(graph):
    atoms, edges = (torch.as_tensor(x).cpu() for x in graph)
    if atoms.ndim != 1 or edges.shape != (len(atoms), len(atoms)) or len(atoms) == 0:
        raise ValueError("Expected class indices with shapes (N,) and (N,N), not one-hot tensors")
    if (atoms.is_floating_point() or edges.is_floating_point()
            or atoms.min() < 0 or atoms.max() >= len(ATOMS) or edges.min() < 0):
        raise ValueError("Invalid atom/bond class indices in DeFoG sample")
    mol = Chem.RWMol()
    for atom in atoms.tolist():
        mol.AddAtom(Chem.Atom(ATOMS[atom]))
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            bond = int(edges[i, j])
            # Same convention as upstream: classes >=5 are virtual, hence no bond.
            if 0 < bond < len(BONDS):
                mol.AddBond(i, j, BONDS[bond])
    try:
        Chem.SanitizeMol(mol)
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        largest = max(fragments, default=mol, key=lambda fragment: fragment.GetNumAtoms())
        return Chem.MolToSmiles(largest)
    except ValueError:
        return None


def load_reference(path):
    import pandas as pd
    from dataset.featurize import MOSES_ATOMS
    from dataset.filtering import sanitize_smiles_dataset

    raw = pd.read_csv(path)["SMILES"].tolist()
    # Same cleaning/representation checks as load_moses(apply_filter=False),
    # without calculating unused molecular property labels or accessing HF.
    clean, _, stats = sanitize_smiles_dataset(
        raw, MOSES_ATOMS, charge_aware=False, apply_filter=False)
    return clean, {"n_raw": len(raw), "n_reference": len(clean), "cleaning": stats}


def score_smiles(smiles, references, device):
    from evaluate import _fcd

    if len({s for s in smiles if s is not None}) < 2:
        raise ValueError("FCD requires at least two unique valid generated molecules")
    scores = {}
    for split, reference in references.items():
        if len(reference) < 2:
            raise ValueError(f"FCD requires at least two reference molecules for {split}")
        print(f"Computing FCD/{split} against {len(reference):,} references on {device} ...", flush=True)
        score = _fcd(smiles, reference, device)
        if not math.isfinite(score):
            raise ValueError(f"Non-finite FCD/{split}: {score}")
        scores[split] = score
    return scores


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, help="Local released generated_samples.pkl; otherwise download it")
    parser.add_argument("--n-samples", type=int, default=25000, help="Generated graphs before invalid removal (default: 25000)")
    parser.add_argument("--offset", type=int, default=0, help="Start index for another batch/fold (default: 0)")
    parser.add_argument("--device", default="auto", help="FCD device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/defog_moses")
    parser.add_argument("--test-csv", type=Path, help="Local official random Test CSV/CSV.gz")
    parser.add_argument("--test-scaffolds-csv", type=Path, help="Local official TestSF CSV/CSV.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/defog_moses_fcd" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if args.n_samples < 2 or args.offset < 0:
        parser.error("--n-samples must be >=2 and --offset must be >=0")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment; use a GPU host or --device cpu")
    RDLogger.DisableLog("rdApp.*")
    sample_path = args.samples or download(SAMPLES_URL, args.cache_dir / "generated_samples.pkl")
    print(f"Loading {sample_path} on CPU ...", flush=True)
    graphs = load_samples(sample_path)
    n_available = len(graphs)
    selected = select_samples(graphs, args.n_samples, args.offset)
    del graphs
    smiles = [decode_graph(graph) for graph in tqdm(selected, desc="Decoding DeFoG", unit="mol")]
    del selected
    valid = [s for s in smiles if s is not None]
    counts = {"n_available": n_available, "n_generated": len(smiles),
              "n_valid": len(valid), "n_unique": len(set(valid)),
              "validity": len(valid) / len(smiles),
              "uniqueness": len(set(valid)) / len(valid) if valid else 0.0}
    print(json.dumps(counts, indent=2), flush=True)
    if counts["n_unique"] < 2:
        raise ValueError("FCD requires at least two unique valid generated molecules")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "generated_smiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "SMILES"])
        writer.writerows(enumerate(smiles, start=args.offset))

    references, ref_info = {}, {}
    for label, split, supplied in [("Test", "test", args.test_csv),
                                   ("TestSF", "test_scaffolds", args.test_scaffolds_csv)]:
        url = MOSES_URL.format(split=split)
        path = supplied or download(url, args.cache_dir / f"{split}.csv.gz")
        print(f"Preparing {label} references from {path} ...", flush=True)
        references[label], stats = load_reference(path)
        ref_info[label] = {"path": str(path.resolve()), "url": None if supplied else url,
                           "sha256": file_sha256(path), **stats}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "samples": {"path": str(sample_path.resolve()), "sha256": file_sha256(sample_path),
                    "url": None if args.samples else SAMPLES_URL, "offset": args.offset},
        "protocol": {"scorer": "evaluate._fcd", "generated_duplicates": "removed",
                     "decoder": "DeFoG aromatic bonds; strict sanitize; largest fragment; no repair",
                     "reference_cleaning": "thesis sanitize_smiles_dataset(apply_filter=False)",
                     "checkpoint_provenance": "public retrained release; not original paper samples"},
        "device": device,
        "versions": {name: version(name) for name in ("torch", "rdkit", "fcd_torch", "numpy", "scipy")},
        "counts": counts, "references": ref_info,
        "fcd": score_smiles(smiles, references, device),
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["fcd"], indent=2))
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()

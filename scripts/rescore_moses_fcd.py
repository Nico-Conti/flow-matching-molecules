#!/usr/bin/env python3
"""Rescore saved MOSES SMILES with fresh HF statistics or the official MOSES package."""
import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from rdkit import Chem

from eval_defog_moses_fcd import file_sha256, load_hf_reference

REFERENCE_SPECS = [
    ("Test", "nico8771/moses_test", 176_074),
    ("TestSF", "nico8771/moses_test_scaffolds_ours", 176_225),
]


def load_smiles_csv(path, expected_generated):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "SMILES" not in reader.fieldnames:
            raise ValueError(f"Missing SMILES column in {path}")
        smiles = [(row["SMILES"] or "").strip() or None for row in reader]
    if len(smiles) != expected_generated:
        raise ValueError(f"Found {len(smiles):,} generated rows; expected {expected_generated:,}")
    if any(Chem.MolFromSmiles(s) is None for s in smiles if s is not None):
        raise ValueError("Nonempty invalid SMILES found; expected valid strings or blank invalid rows")
    return smiles


def select_smiles(smiles, n_samples, seed):
    if n_samples is None:
        return smiles
    if not 2 <= n_samples <= len(smiles):
        raise ValueError(f"n-samples must be between 2 and {len(smiles):,}")
    indices = sorted(random.Random(seed).sample(range(len(smiles)), n_samples))
    return [smiles[i] for i in indices]


def load_official_moses():
    import importlib.util
    import types
    import pandas as pd

    if importlib.util.find_spec("moses") is None:
        raise RuntimeError("Install the scorer first: python -m pip install --no-deps molsets==0.3.1")
    fixes = []
    if "rdkit.six" not in sys.modules and importlib.util.find_spec("rdkit.six") is None:
        shim = types.ModuleType("rdkit.six")
        shim.iteritems = lambda mapping: iter(mapping.items())
        sys.modules["rdkit.six"] = shim
        fixes.append("rdkit.six.iteritems -> iter(mapping.items()) for SA score")
    needs_append = not hasattr(pd.DataFrame, "append")
    if needs_append:
        # MOSES uses append only to combine its two bundled SMARTS filter tables.
        pd.DataFrame.append = lambda self, other, sort=False: pd.concat([self, other], sort=sort)
        fixes.append("DataFrame.append -> pandas.concat during MOSES filter-table import")
    try:
        import moses
    finally:
        if needs_append:
            del pd.DataFrame.append
    return moses, fixes


def score_official_moses(smiles, device, batch_size, n_jobs):
    moses, fixes = load_official_moses()
    print("Calling moses.get_all_metrics with default bundled references/statistics; all metrics may take several minutes ...", flush=True)
    raw = moses.get_all_metrics(gen=smiles, device=device, batch_size=batch_size, n_jobs=n_jobs)
    scores = {"Test": float(raw["FCD/Test"]), "TestSF": float(raw["FCD/TestSF"])}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError(f"Non-finite official MOSES FCD: {scores}")
    metrics = {key: float(value) if math.isfinite(float(value)) else None for key, value in raw.items()}
    return scores, metrics, fixes


def score_fresh(valid, references, device, batch_size, n_jobs):
    from fcd_torch import FCD

    scorer = FCD(device=device, batch_size=batch_size, n_jobs=n_jobs)
    print(f"Computing fresh ChemNet statistics for {len(valid):,} valid generated molecules (duplicates retained) ...", flush=True)
    pgen = scorer.precalc(valid)
    scores = {}
    for label, smiles in references.items():
        print(f"Computing fresh {label} reference statistics for {len(smiles):,} molecules ...", flush=True)
        pref = scorer.precalc(smiles)
        score = float(scorer(pref=pref, pgen=pgen))
        if not math.isfinite(score):
            raise ValueError(f"Non-finite FCD/{label}: {score}")
        scores[label] = score
        print(f"FCD/{label}: {score:.8f}", flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles-csv", type=Path, required=True)
    parser.add_argument("--expected-generated", type=int, default=25000)
    parser.add_argument("--n-samples", type=int,
                        help="Random subset size before invalid removal; default: all CSV rows")
    parser.add_argument("--seed", type=int, default=0, help="Random subset seed (default: 0)")
    parser.add_argument("--scorer", choices=("fresh", "moses"), default="fresh",
                        help="fresh: full HF references, recompute statistics; moses: official get_all_metrics defaults")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_generated < 2 or args.batch_size < 1 or args.n_jobs < 1:
        parser.error("expected-generated must be >=2; batch-size and n-jobs must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use a GPU host or --device cpu")
    smiles = load_smiles_csv(args.smiles_csv, args.expected_generated)
    n_available = len(smiles)
    smiles = select_smiles(smiles, args.n_samples, args.seed)
    if args.n_samples is not None:
        print(f"Selected {len(smiles):,}/{n_available:,} generated rows without replacement, seed={args.seed}", flush=True)
    valid = [s for s in smiles if s is not None]
    if len(valid) < 2:
        raise ValueError("FCD requires at least two valid molecules")
    counts = {"n_generated": len(smiles), "n_valid": len(valid), "n_unique": len(set(valid)),
              "validity": len(valid) / len(smiles), "uniqueness": len(set(valid)) / len(valid)}
    print(json.dumps(counts, indent=2), flush=True)
    references, reference_info = {}, {}
    official_metrics, compatibility = None, []
    if args.scorer == "moses":
        scores, official_metrics, compatibility = score_official_moses(
            smiles, args.device, args.batch_size, args.n_jobs)
        data_dir = Path(sys.modules["moses"].__file__).parent / "dataset/data"
        for label, split in [("Test", "test"), ("TestSF", "test_scaffolds")]:
            csv_path = data_dir / f"{split}.csv.gz"
            stats_path = data_dir / f"{split}_stats.npz"
            reference_info[label] = {"csv_path": str(csv_path), "csv_sha256": file_sha256(csv_path),
                                     "stats_path": str(stats_path), "stats_sha256": file_sha256(stats_path)}
    else:
        for label, repo_id, expected in REFERENCE_SPECS:
            reference, info = load_hf_reference(repo_id)
            if len(reference) != expected:
                raise ValueError(f"{repo_id}: {len(reference):,} references; expected full split of {expected:,}")
            references[label], reference_info[label] = reference, info
        scores = score_fresh(valid, references, device=args.device,
                             batch_size=args.batch_size, n_jobs=args.n_jobs)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "samples": {"path": str(args.smiles_csv.resolve()), "sha256": file_sha256(args.smiles_csv),
                    "n_available": n_available,
                    "selection": "all rows" if args.n_samples is None else "uniform rows without replacement; original order",
                    "selection_seed": args.seed if args.n_samples is not None else None},
        "protocol": {"scorer": "moses.get_all_metrics" if args.scorer == "moses" else "fcd_torch.FCD",
                     "generated_duplicates": "retained",
                     "reference_source": "MOSES package defaults" if args.scorer == "moses" else
                         "full HF MOSES splits; stored SMILES without graph preprocessing",
                     "reference_statistics": "MOSES package bundled statistics" if args.scorer == "moses" else
                         "fresh; no statistics cache read or written",
                     "compatibility": compatibility},
        "device": args.device, "batch_size": args.batch_size, "n_jobs": args.n_jobs,
        "versions": {name: version(name) for name in ("torch", "rdkit", "fcd_torch", "numpy", "scipy")},
        "counts": counts, "references": reference_info, "fcd": scores,
    }
    if official_metrics is not None:
        report["versions"]["molsets"] = version("molsets")
        report["moses_metrics"] = official_metrics
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(scores, indent=2))
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()

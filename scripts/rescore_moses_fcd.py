#!/usr/bin/env python3
"""Rescore saved MOSES SMILES against full HF references with fresh ChemNet statistics."""
import argparse
import csv
import json
import math
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
    valid = [s for s in smiles if s is not None]
    if len(valid) < 2:
        raise ValueError("FCD requires at least two valid molecules")
    counts = {"n_generated": len(smiles), "n_valid": len(valid), "n_unique": len(set(valid)),
              "validity": len(valid) / len(smiles), "uniqueness": len(set(valid)) / len(valid)}
    print(json.dumps(counts, indent=2), flush=True)
    references, reference_info = {}, {}
    for label, repo_id, expected in REFERENCE_SPECS:
        reference, info = load_hf_reference(repo_id)
        if len(reference) != expected:
            raise ValueError(f"{repo_id}: {len(reference):,} references; expected full split of {expected:,}")
        references[label], reference_info[label] = reference, info
    scores = score_fresh(valid, references, device=args.device,
                         batch_size=args.batch_size, n_jobs=args.n_jobs)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "samples": {"path": str(args.smiles_csv.resolve()), "sha256": file_sha256(args.smiles_csv)},
        "protocol": {"scorer": "fcd_torch.FCD", "generated_duplicates": "retained",
                     "reference_source": "full HF MOSES splits; stored SMILES without graph preprocessing",
                     "reference_statistics": "fresh; no statistics cache read or written"},
        "device": args.device, "batch_size": args.batch_size, "n_jobs": args.n_jobs,
        "versions": {name: version(name) for name in ("torch", "rdkit", "fcd_torch", "numpy", "scipy")},
        "counts": counts, "references": reference_info, "fcd": scores,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(scores, indent=2))
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()

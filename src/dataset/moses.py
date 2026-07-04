from rdkit import RDLogger

from .featurize import MOSES_ATOMS, N_BOND_CLASSES
from .filtering import sanitize_smiles_dataset
from .zinc import _compute_targets, _push  # shared RDKit targets + HF upload

RDLogger.DisableLog("rdApp.*")

MOSES_TARGETS_DEFAULT = ("logP", "qed", "SAS")
MOSES_SPLITS = ("train", "test", "test_scaffolds")
# Official MOSES splits (git-LFS media endpoint); column "SMILES".
MOSES_URL = ("https://media.githubusercontent.com/media/molecularsets/moses/"
             "master/moses/dataset/data/{split}.csv.gz")


def load_moses(split="train", targets=MOSES_TARGETS_DEFAULT, apply_filter=False,
               limit=None, use_cache=True, repo_id=None, push=False):
    # MOSES is neutral (no charges), featurized over 7 atom types with kekulized bonds
    # (4-class). Splits are the official train/test/test_scaffolds. use_cache pulls cleaned
    # (smiles, y) from repo_id; on a miss it downloads the official gzip and builds. Targets
    # are RDKit-recomputed (MOSES ships none) on the sanitized SMILES.
    if split not in MOSES_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {MOSES_SPLITS}")
    repo_id = repo_id or f"nico8771/moses_{split}"

    raw, stats = None, None
    if use_cache:
        try:
            from datasets import load_dataset
            dd = load_dataset(repo_id)
            raw = dd[next(iter(dd))]
            stats = {"source": f"hub:{repo_id}", "rows": raw.num_rows}
        except Exception:
            raw = None

    if raw is None:
        import pandas as pd
        smiles = pd.read_csv(MOSES_URL.format(split=split), compression="gzip")["SMILES"].tolist()
        if limit:
            smiles = smiles[:limit]

        clean, _kept_idx, stats = sanitize_smiles_dataset(
            smiles, MOSES_ATOMS, charge_aware=False, apply_filter=apply_filter)

        from datasets import Dataset
        raw = Dataset.from_dict({
            "smiles": clean,
            "y": [list(map(float, r)) for r in _compute_targets(clean, targets)],
        })
        if push:
            _push(raw, repo_id, moses_card(repo_id, split, stats, targets))
            stats = {**stats, "pushed_to": repo_id}

    if limit and raw.num_rows > limit:
        raw = raw.select(range(limit))
    return {"ds": raw, "atom_vocab": MOSES_ATOMS, "charge_aware": False,
            "n_bond_classes": N_BOND_CLASSES,
            "targets": tuple(targets), "split": split, "stats": stats}


def push_moses(moses, repo_id=None, token=None):
    repo_id = repo_id or f"nico8771/moses_{moses['split']}"
    _push(moses["ds"], repo_id,
          moses_card(repo_id, moses["split"], moses["stats"], moses["targets"]), token)
    return repo_id


def moses_card(repo_id, split, stats, targets):
    tgt = ", ".join(f"`{t}`" for t in targets)
    vocab = ", ".join(f"`{a}`" for a in MOSES_ATOMS)
    n_kept = (stats or {}).get("kept", 0) + (stats or {}).get("kept_no_roundtrip", 0)
    drops = "\n".join(
        f"| `{k}` | {v:,} |" for k, v in sorted((stats or {}).items())
        if k.startswith(("drop_", "kept"))) or "| (from cache) | |"
    return f"""---
license: other
pretty_name: {repo_id.split('/')[-1]}
tags:
- chemistry
- molecules
- graph-generation
- flow-matching
size_categories:
- 1M<n<10M
---

# {repo_id} — cleaned MOSES ({split} split)

Each row is a molecule as **canonical SMILES** plus RDKit-recomputed targets. MOSES is
neutral by construction, so molecules are featurized over **7 atom types** with **no
formal charges**, and **bonds are kekulized** into 4 classes (aromatic rings become
alternating single/double).

> Source: official MOSES `{split}.csv.gz` (molecularsets/moses). Code:
> <https://github.com/Nico-Conti/flow-matching-molecules> (`dataset/`).

## Schema

| column | type | description |
|---|---|---|
| `smiles` | string | canonical, single-fragment SMILES (post-sanitize) |
| `y` | list[float] | RDKit targets, columns = {tgt} |

## Pipeline

1. **Parse** with RDKit; unparseable dropped.
2. **Standardize** — remove stereochemistry, sanitize (MOSES is neutral; *no* Uncharger).
3. **Featurize** over atom vocab ({vocab}); atoms outside the vocab dropped.
4. **Round-trip check** — `smiles -> (X, E) -> mol -> smiles`.

Bonds use {N_BOND_CLASSES} classes (none / single / double / triple).
Targets ({tgt}) are recomputed from the sanitized SMILES with RDKit.

### Drop / keep counts (this build)

| outcome | count |
|---|---|
{drops}

Kept: **{n_kept:,}** molecules.
"""

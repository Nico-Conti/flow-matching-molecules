import os
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, QED
from torch_molecule.datasets import load_zinc250k

from .featurize import ZINC_ATOMS, N_BOND_CLASSES
from .filtering import sanitize_smiles_dataset

RDLogger.DisableLog("rdApp.*")

ZINC_TARGETS_DEFAULT = ("logP", "qed", "SAS")
ZINC_REPO_DEFAULT = "nico8771/zinc_neutral"


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
              uncharge=True, limit=None, use_cache=True, repo_id=ZINC_REPO_DEFAULT,
              push=False):
    # use_cache=True pulls cleaned (smiles, y) from repo_id (public) and skips the
    # torch_molecule download + sanitize/round-trip pass; on a miss it builds.
    # push=True uploads the cleaned dataset + card after a build (needs HF_TOKEN).
    # Returns {"ds", "atom_vocab", "targets", "stats"}; ds = HF Dataset(smiles, y).
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
        ds = load_zinc250k(local_dir=local_dir)
        smiles = list(ds.data)
        if limit:
            smiles = smiles[:limit]
        # Neutralize (uncharge=True) and featurize element-only over the 9-type
        # vocab; charges are recovered at decode, not stored. Round-trip applied.
        clean, kept_idx, stats = sanitize_smiles_dataset(
            smiles, ZINC_ATOMS, charge_aware=False, uncharge=uncharge, apply_filter=apply_filter)

        from datasets import Dataset
        raw = Dataset.from_dict({
            "smiles": clean,
            # targets recomputed from the cleaned canonical SMILES.
            "y": [list(map(float, r)) for r in _compute_targets(clean, targets)],
        })
        if push:
            _push(raw, repo_id, zinc_card(repo_id, stats, targets))
            stats = {**stats, "pushed_to": repo_id}

    if limit and raw.num_rows > limit:
        raw = raw.select(range(limit))
    return {"ds": raw, "atom_vocab": ZINC_ATOMS, "charge_aware": False,
            "targets": tuple(targets), "stats": stats}


def push_zinc(zinc, repo_id=ZINC_REPO_DEFAULT, token=None):
    _push(zinc["ds"], repo_id, zinc_card(repo_id, zinc["stats"], zinc["targets"]), token)
    return repo_id


def _push(raw, repo_id, card, token=None):
    # Upload data then overwrite the auto-card with our README.
    from dotenv import load_dotenv, find_dotenv
    from huggingface_hub import HfApi
    load_dotenv(find_dotenv(usecwd=True))
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("pushing requires HF_TOKEN (set it in .env)")
    raw.push_to_hub(repo_id, token=token)
    HfApi().upload_file(
        path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
        repo_id=repo_id, repo_type="dataset", token=token,
        commit_message="Add pipeline dataset card")


def zinc_card(repo_id, stats, targets):
    tgt = ", ".join(f"`{t}`" for t in targets)
    vocab = ", ".join(f"`{a}`" for a in ZINC_ATOMS)
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
- 100K<n<1M
---

# {repo_id} — cleaned ZINC-250k

Each row is a molecule as **canonical SMILES** plus RDKit-recomputed targets. Molecules
are **neutralized** (RDKit `Uncharger`) and featurized over 9 neutral atom types; formal
charges are recovered at decode time (DeFoG/DiGress partial-charge build), not stored.

> Source: torch_molecule ZINC-250k (HuggingFace). Code:
> <https://github.com/Nico-Conti/flow-matching-molecules> (`dataset/`).

## Schema

| column | type | description |
|---|---|---|
| `smiles` | string | canonical, kekulizable, single-fragment SMILES (post-sanitize) |
| `y` | list[float] | RDKit targets, columns = {tgt} |

## Pipeline

1. **Parse** with RDKit; unparseable dropped.
2. **Standardize** — remove stereochemistry, neutralize (`Uncharger`), sanitize.
3. **Kekulize** over atom vocab ({vocab}); atoms outside the vocab dropped.
4. **Round-trip check** — `smiles -> (X, E) -> mol -> smiles` must reproduce the
   canonical molecule as a single fragment (**filter applied**).

Bonds use {N_BOND_CLASSES} classes (none / single / double / triple). Targets
({tgt}) are recomputed from the canonical SMILES with RDKit.

### Drop / keep counts (this build)

| outcome | count |
|---|---|
{drops}

Kept: **{n_kept:,}** molecules.
"""

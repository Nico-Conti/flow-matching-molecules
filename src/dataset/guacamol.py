import urllib.request

from rdkit import RDLogger

from .featurize import GUACAMOL_ATOMS, N_BOND_CLASSES
from .filtering import sanitize_smiles_dataset
from .zinc import _compute_targets, _push  # shared RDKit targets + HF upload

RDLogger.DisableLog("rdApp.*")

GUACAMOL_TARGETS_DEFAULT = ("logP", "qed", "SAS")
# Official GuacaMol splits (Figshare, ChEMBL-24-derived); one SMILES per line.
GUACAMOL_URL = "https://ndownloader.figshare.com/files/{fid}"
GUACAMOL_FILES = {"train": "13612760", "valid": "13612766", "test": "13612757"}


def load_guacamol(split="train", targets=GUACAMOL_TARGETS_DEFAULT, apply_filter=False,
                  limit=None, use_cache=True, repo_id=None, push=False):
    # GuacaMol has formal charges and a 12-element vocab, featurized element-only (charges
    # recovered at decode via the partial-charge build, like ZINC/DeFoG) with kekulized bonds
    # (4-class). Keep-all (representation-agnostic SMILES). use_cache pulls cleaned (smiles, y)
    # from repo_id; on a miss it downloads the official split and builds. Targets RDKit-recomputed.
    if split not in GUACAMOL_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {tuple(GUACAMOL_FILES)}")
    repo_id = repo_id or f"nico8771/guacamol_{split}"

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
        with urllib.request.urlopen(GUACAMOL_URL.format(fid=GUACAMOL_FILES[split])) as r:
            smiles = [ln.split()[0] for ln in r.read().decode().splitlines() if ln.strip()]
        if limit:
            smiles = smiles[:limit]

        clean, _kept_idx, stats = sanitize_smiles_dataset(
            smiles, GUACAMOL_ATOMS, charge_aware=False, apply_filter=apply_filter)

        from datasets import Dataset
        raw = Dataset.from_dict({
            "smiles": clean,
            "y": [list(map(float, r)) for r in _compute_targets(clean, targets)],
        })
        if push:
            _push(raw, repo_id, guacamol_card(repo_id, split, stats, targets))
            stats = {**stats, "pushed_to": repo_id}

    if limit and raw.num_rows > limit:
        raw = raw.select(range(limit))
    return {"ds": raw, "atom_vocab": GUACAMOL_ATOMS, "charge_aware": False,
            "n_bond_classes": N_BOND_CLASSES,
            "targets": tuple(targets), "split": split, "stats": stats}


def push_guacamol(guacamol, repo_id=None, token=None):
    repo_id = repo_id or f"nico8771/guacamol_{guacamol['split']}"
    _push(guacamol["ds"], repo_id,
          guacamol_card(repo_id, guacamol["split"], guacamol["stats"], guacamol["targets"]), token)
    return repo_id


def guacamol_card(repo_id, split, stats, targets):
    tgt = ", ".join(f"`{t}`" for t in targets)
    vocab = ", ".join(f"`{a}`" for a in GUACAMOL_ATOMS)
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

# {repo_id} — cleaned GuacaMol ({split} split)

Each row is a molecule as **canonical SMILES** plus RDKit-recomputed targets. Keep-all:
GuacaMol (ChEMBL-24-derived) has formal charges; molecules are featurized **element-only**
over 12 atom types (charges recovered at decode via the partial-charge build, like
ZINC/DeFoG) with **kekulized bonds** (4-class). A small fraction of graphs don't strictly
round-trip (charges + fused rings; cf. DiGress App. F.3) but are **kept** (not dropped) —
the model still learns them.

> Source: official GuacaMol `guacamol_v1_{split}.smiles` (Figshare, BenevolentAI). Code:
> <https://github.com/Nico-Conti/flow-matching-molecules> (`dataset/`).

## Schema

| column | type | description |
|---|---|---|
| `smiles` | string | canonical, single-fragment SMILES (post-sanitize, charges intact) |
| `y` | list[float] | RDKit targets, columns = {tgt} |

## Pipeline

1. **Parse** with RDKit; unparseable dropped.
2. **Standardize** — remove stereochemistry, sanitize (charges left intact).
3. **Featurize** element-only over atom vocab ({vocab}); atoms outside the vocab dropped.
4. **Keep-all** — every sanitized, in-vocab molecule is kept; the round-trip check is
   recorded as a stat (`kept` / `kept_no_roundtrip`), not a filter.

Bonds use {N_BOND_CLASSES} classes (none / single / double / triple). Targets ({tgt}) are
recomputed with RDKit.

### Drop / keep counts (this build)

| outcome | count |
|---|---|
{drops}

Kept: **{n_kept:,}** molecules.
"""

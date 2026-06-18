import os

import numpy as np
from rdkit import RDLogger
from torch_molecule.datasets import load_qm9 as _tm_load_qm9

from .featurize import QM9_ATOMS, N_BOND_CLASSES
from .filtering import sanitize_smiles_dataset

RDLogger.DisableLog("rdApp.*")

QM9_TARGETS_DEFAULT = ("mu", "alpha", "homo", "lumo", "gap", "cv")
QM9_REPO_DEFAULT = "nico8771/qm9_clean"

HOMO_LUMO_GAP = ("homo", "lumo", "gap")
HARTREE_TO_EV = 27.211324570273  # keep == properties.HARTREE_TO_EV
QM9_UNITS_RAW = {
    "mu": "Debye", "alpha": "Bohr³",
    "homo": "Hartree", "lumo": "Hartree", "gap": "Hartree",
    "cv": "cal·mol⁻¹·K⁻¹",
}


def load_qm9(local_dir="data", targets=QM9_TARGETS_DEFAULT, apply_filter=False, limit=None,
             use_cache=True, repo_id=QM9_REPO_DEFAULT, push=False, to_ev=True):
    # use_cache=True pulls cleaned (smiles, y) from repo_id (public) and skips the
    # torch_molecule download + sanitize/round-trip pass; on a miss it builds.
    # push=True uploads the cleaned dataset + card after a build (needs HF_TOKEN).
    # to_ev=True converts homo/lumo/gap Hartree->eV at load (stored HF data stays Hartree).
    # Returns {"ds", "atom_vocab", "targets", "units", "stats"}; ds = HF Dataset(smiles, y).
    # Featurization to dense (X, E) happens lazily in MoleculeDataset.
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
        ds = _tm_load_qm9(local_dir=local_dir, target_cols=list(targets))
        smiles = list(ds.data)
        y_all = np.asarray(ds.target, dtype="float32")
        if limit:
            smiles, y_all = smiles[:limit], y_all[:limit]

        clean, kept_idx, stats = sanitize_smiles_dataset(
            smiles, QM9_ATOMS, charge_aware=True, apply_filter=apply_filter)

        from datasets import Dataset
        raw = Dataset.from_dict({
            "smiles": clean,
            "y": [list(map(float, r)) for r in y_all[kept_idx]],
        })
        if push:
            _push(raw, repo_id, qm9_card(repo_id, stats, targets))
            stats = {**stats, "pushed_to": repo_id}

    if limit and raw.num_rows > limit:
        raw = raw.select(range(limit))

    units = dict(QM9_UNITS_RAW)
    if to_ev:
        factors = [HARTREE_TO_EV if t in HOMO_LUMO_GAP else 1.0 for t in targets]
        if any(f != 1.0 for f in factors):
            raw = raw.map(lambda ex: {"y": [v * f for v, f in zip(ex["y"], factors)]})
            units.update({t: "eV" for t in targets if t in HOMO_LUMO_GAP})

    return {"ds": raw, "atom_vocab": QM9_ATOMS, "targets": tuple(targets),
            "units": {t: units[t] for t in targets if t in units}, "stats": stats}


def push_qm9(qm9, repo_id=QM9_REPO_DEFAULT, token=None):
    _push(qm9["ds"], repo_id, qm9_card(repo_id, qm9["stats"], qm9["targets"]), token)
    return repo_id


def push_qm9_card(repo_id=QM9_REPO_DEFAULT, targets=QM9_TARGETS_DEFAULT, token=None):
    # Card-only refresh: inject the units table into the live README, no data re-push.
    from dotenv import load_dotenv, find_dotenv
    from huggingface_hub import HfApi, hf_hub_download
    load_dotenv(find_dotenv(usecwd=True))
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("pushing requires HF_TOKEN (set it in .env)")
    path = hf_hub_download(repo_id, "README.md", repo_type="dataset", token=token)
    with open(path, encoding="utf-8") as f:
        md = _inject_units(f.read(), targets)
    HfApi().upload_file(
        path_or_fileobj=md.encode("utf-8"), path_in_repo="README.md",
        repo_id=repo_id, repo_type="dataset", token=token,
        commit_message="Document target units (Debye / Hartree / Bohr³ / cal·mol⁻¹·K⁻¹)")
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


def _units_section(targets):
    rows = "\n".join(f"| `{t}` | {QM9_UNITS_RAW.get(t, '?')} |" for t in targets)
    return ("### Units (as stored — raw QM9)\n\n"
            "| target | unit |\n|---|---|\n" + rows + "\n\n"
            "Energy targets (`homo`, `lumo`, `gap`) are in **Hartree**; multiply by "
            "27.2113 for eV (matches FreeGress/DiGress reported MAE). "
            "`load_qm9(to_ev=True)` does this conversion at load time.\n\n")


def _inject_units(md, targets):
    # Insert/replace the Units section without touching the rest of the card.
    sec = _units_section(targets)
    marker = "### Units (as stored"
    if marker in md:
        i = md.index(marker)
        j = md.find("\n## ", i)
        return md[:i] + sec + (md[j + 1:] if j != -1 else "")
    if "## Pipeline" in md:
        return md.replace("## Pipeline", sec + "## Pipeline", 1)
    return md.rstrip() + "\n\n" + sec


def qm9_card(repo_id, stats, targets):
    tgt = ", ".join(f"`{t}`" for t in targets)
    vocab = ", ".join(f"`{a}`" for a in QM9_ATOMS)
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

# {repo_id} — cleaned QM9

Each row is saved as a neutrally charged molecule using **canonical SMILES** plus the **EDM-six DFT targets**.

> Source: torch_molecule QM9 (HuggingFace mirror). Code:
> <https://github.com/Nico-Conti/flow-matching-molecules> (`dataset/`).

## Schema

| column | type | description |
|---|---|---|
| `smiles` | string | canonical, kekulizable, single-fragment SMILES (post-sanitize) |
| `y` | list[float] | DFT targets, columns = {tgt} |

{_units_section(targets)}## Pipeline

1. **Parse** with RDKit; unparseable dropped.
2. **Standardize** — remove stereochemistry, sanitize (QM9 is neutral).
3. **Kekulize** over atom vocab ({vocab}); atoms outside the vocab dropped.
4. **Round-trip check** — `smiles -> (X, E) -> mol -> smiles`.

Bonds use {N_BOND_CLASSES} classes (none / single / double / triple). The six DFT
properties are shipped with QM9 and are **not** RDKit-recomputable.

### Drop / keep counts (this build)

| outcome | count |
|---|---|
{drops}

Kept: **{n_kept:,}** molecules.
"""

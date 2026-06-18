import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .featurize import tensor_to_mol, largest_fragment, QM9_ATOMS

PSI4_METHOD = "b3lyp/6-31G*"
HARTREE_TO_EV = 27.211324570273  
AU_TO_DEBYE = 2.5417464519


def embed_geometry(mol, seed=0xC0FFEE, max_iters=200):
    # 2D mol -> explicit-H 3D conformer + MMFF cleanup.
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
    except Exception:
        return None
    return mol


def _psi4_geometry(mol):
    import psi4
    conf = mol.GetConformer()
    lines = ["0 1"]  # neutral closed-shell singlet
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    lines += ["units angstrom", "no_reorient", "no_com"]
    return psi4.geometry("\n".join(lines))


def psi4_properties(mol, method=PSI4_METHOD, optimize=False,
                    memory="2 GB", threads=1, scratch=None):
    import psi4
    psi4.core.be_quiet()
    psi4.set_memory(memory)
    psi4.set_num_threads(threads)
    if scratch:
        psi4.core.IOManager.shared_object().set_default_path(scratch)

    geom = _psi4_geometry(mol)
    psi4.set_options({"reference": "rks"})
    try:
        if optimize:
            psi4.optimize(method, molecule=geom)
        _, wfn = psi4.energy(method, molecule=geom, return_wfn=True)
    except Exception:
        psi4.core.clean()
        return None

    eps = np.asarray(wfn.epsilon_a().to_array())
    homo = float(eps[wfn.nalpha() - 1]) * HARTREE_TO_EV  # nalpha = n doubly-occ
    dipole = np.asarray(wfn.variable("SCF DIPOLE"))
    mu = float(np.linalg.norm(dipole) * AU_TO_DEBYE)
    psi4.core.clean()
    return {"mu": mu, "homo": homo}


def compute_targets(mol, **kw):
    geom = embed_geometry(mol)
    if geom is None:
        return None
    return psi4_properties(geom, **kw)


def targets_from_graph(X, E, atom_vocab=QM9_ATOMS, repair=False, **kw):
    mol, _ = tensor_to_mol(X, E, atom_vocab=atom_vocab, repair=repair)
    mol = largest_fragment(mol)
    if mol is None:
        return None
    return compute_targets(mol, **kw)


def property_mae(graphs, y_targets, target_cols=("mu", "homo"),
                 atom_vocab=QM9_ATOMS, repair=False, progress=False, **kw):
    # y_targets: [N, len(target_cols)] conditioning values, column-aligned to
    # target_cols. Molecules that fail to decode/embed/converge are skipped.
    y_targets = np.asarray(y_targets, dtype="float64")
    pairs = list(zip(graphs, y_targets))
    if progress:
        from tqdm.auto import tqdm
        pairs = tqdm(pairs, desc="psi4", unit="mol")

    errs = {c: [] for c in target_cols}
    n_ok = 0
    for (X, E), y in pairs:
        props = targets_from_graph(X, E, atom_vocab=atom_vocab, repair=repair, **kw)
        if props is None:
            continue
        n_ok += 1
        for j, c in enumerate(target_cols):
            errs[c].append(abs(props[c] - float(y[j])))

    out = {f"mae_{c}": (float(np.mean(errs[c])) if errs[c] else float("nan"))
           for c in target_cols}
    out["n_evaluated"] = n_ok
    out["n_total"] = len(graphs)
    out["coverage"] = n_ok / len(graphs) if graphs else 0.0
    return out

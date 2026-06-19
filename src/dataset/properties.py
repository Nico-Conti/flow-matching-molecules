import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .featurize import tensor_to_mol, largest_fragment, QM9_ATOMS

PSI4_METHOD = "b3lyp/6-31G*"
PYSCF_XC, PYSCF_BASIS = "b3lyp", "6-31g*"   
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
        return "dft_error"

    eps = np.asarray(wfn.epsilon_a().to_array())
    homo = float(eps[wfn.nalpha() - 1]) * HARTREE_TO_EV  # nalpha = n doubly-occ
    dipole = np.asarray(wfn.variable("SCF DIPOLE"))
    mu = float(np.linalg.norm(dipole) * AU_TO_DEBYE)
    psi4.core.clean()
    return {"mu": mu, "homo": homo}


def pyscf_properties(mol, xc=PYSCF_XC, basis=PYSCF_BASIS, optimize=False, auto_spin=True):
    # auto_spin=True (default, FreeGress-like): actual charge + spin=2S (=Nα−Nβ), UKS if open-shell.
    # auto_spin=False: force neutral closed-shell singlet RKS (QM9 convention).
    from pyscf import gto, dft
    conf = mol.GetConformer()
    atom = [(a.GetSymbol(), tuple(conf.GetAtomPosition(a.GetIdx())))
            for a in mol.GetAtoms()]
    if auto_spin:
        charge = Chem.GetFormalCharge(mol)
        spin = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())   # 2S, not multiplicity
    else:
        charge, spin = 0, 0

    n_elec = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
    if (n_elec - spin) % 2:                            # charge/spin can't form this electron count
        return "parity"
    try:
        m = gto.M(atom=atom, basis=basis, charge=charge, spin=spin,
                  unit="Angstrom", verbose=0)
        ks = dft.UKS if spin else dft.RKS
        mf = ks(m); mf.xc = xc
        if optimize:                                   # needs `pip install geometric`
            from pyscf.geomopt.geometric_solver import optimize as geom_opt
            m = geom_opt(mf); mf = ks(m); mf.xc = xc
        mf.kernel()
    except Exception:
        return "dft_error"
    if not mf.converged:
        return "not_converged"
    occ, eps = mf.mo_occ, mf.mo_energy
    if spin:                                           # UKS: alpha HOMO (matches FreeGress)
        occ, eps = occ[0], eps[0]
    homo = float(eps[occ > 0][-1]) * HARTREE_TO_EV     # Hartree -> eV
    mu = float(np.linalg.norm(mf.dip_moment(unit="Debye", verbose=0)))
    return {"mu": mu, "homo": homo}


def compute_targets(mol, engine="pyscf", seed=0xC0FFEE, **kw):
    geom = embed_geometry(mol, seed=seed)
    if geom is None:
        return "embed"
    if engine == "pyscf":
        return pyscf_properties(geom, **kw)
    if engine == "psi4":
        return psi4_properties(geom, **kw)
    raise ValueError(f"unknown engine {engine!r}; expected 'pyscf' or 'psi4'")


def targets_from_graph(X, E, atom_vocab=QM9_ATOMS, repair=False, seed=0xC0FFEE, **kw):
    mol, _ = tensor_to_mol(X, E, atom_vocab=atom_vocab, repair=repair)
    mol = largest_fragment(mol)
    if mol is None:
        return "decode"
    return compute_targets(mol, seed=seed, **kw)


def property_mae(graphs, y_targets, target_cols=("mu", "homo"),
                 atom_vocab=QM9_ATOMS, repair=False, seed=0xC0FFEE,
                 progress=False, **kw):
    # y_targets: [N, len(target_cols)] conditioning values, column-aligned to
    # target_cols. Failures are skipped and tallied by reason in out["failures"]:
    # decode / embed / parity / not_converged / dft_error.
    from collections import Counter
    y_targets = np.asarray(y_targets, dtype="float64")
    pairs = list(zip(graphs, y_targets))
    if progress:
        from tqdm.auto import tqdm
        pairs = tqdm(pairs, desc="dft", unit="mol")

    errs = {c: [] for c in target_cols}
    fails = Counter()
    n_ok = 0
    for (X, E), y in pairs:
        props = targets_from_graph(X, E, atom_vocab=atom_vocab, repair=repair,
                                   seed=seed, **kw)
        if not isinstance(props, dict):
            fails[props] += 1
            continue
        n_ok += 1
        for j, c in enumerate(target_cols):
            errs[c].append(abs(props[c] - float(y[j])))

    out = {f"mae_{c}": (float(np.mean(errs[c])) if errs[c] else float("nan"))
           for c in target_cols}
    out["n_evaluated"] = n_ok
    out["n_total"] = len(graphs)
    out["coverage"] = n_ok / len(graphs) if graphs else 0.0
    out["failures"] = dict(fails)
    return out

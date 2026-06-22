import os
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, QED

from .featurize import (tensor_to_mol, build_mol_partial_charges,
                        largest_fragment, QM9_ATOMS, ZINC_ATOMS)


RDKIT_FNS = {"logP": Crippen.MolLogP, "qed": QED.qed}
DFT_PROPS = ("mu", "homo")

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


def _score_one(graph, cfg):
    # top-level (picklable) so it can run in a worker process. cfg holds the
    # decode/score config; returns a props dict or a failure-reason string.
    X, E = graph
    if cfg["partial_charges"]:
        mol = build_mol_partial_charges(X, E, atom_vocab=cfg["atom_vocab"])
    else:
        mol, _ = tensor_to_mol(X, E, atom_vocab=cfg["atom_vocab"], repair=cfg["repair"])
    mol = largest_fragment(mol)
    if mol is None:
        return "decode"
    out = {}
    try:
        for c in cfg["target_cols"]:
            if c in RDKIT_FNS:
                out[c] = float(RDKIT_FNS[c](mol))
    except Exception:
        return "prop_error"
    if cfg["needs_dft"]:
        props = compute_targets(mol, seed=cfg["seed"], **cfg["dft_kw"])  # dict or failure str
        if not isinstance(props, dict):
            return props
        out.update({c: props[c] for c in cfg["target_cols"] if c in DFT_PROPS})
    return out


_blas_limiter = None


def _worker_init():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    global _blas_limiter
    try:                          
        from threadpoolctl import threadpool_limits
        _blas_limiter = threadpool_limits(limits=1)
    except Exception:
        pass
    try:
        from pyscf import lib
        lib.num_threads(1)
    except Exception:
        pass


def _mae_over_graphs(graphs, y_targets, target_cols, cfg, desc, progress, n_jobs=1):
    y_targets = np.asarray(y_targets, dtype="float64")
    n = len(graphs)

    if n_jobs > 1 and cfg["needs_dft"]:        # parallel DFT (independent per molecule)
        import functools, multiprocessing as mp
        worker = functools.partial(_score_one, cfg=cfg)
        with mp.get_context("fork").Pool(n_jobs, initializer=_worker_init) as pool:
            it = pool.imap(worker, graphs, chunksize=1)   # imap preserves order
            if progress:
                from tqdm.auto import tqdm
                it = tqdm(it, total=n, desc=desc, unit="mol")
            props_list = list(it)
    else:
        it = graphs
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(graphs, total=n, desc=desc, unit="mol")
        props_list = [_score_one(g, cfg) for g in it]

    errs = {c: [] for c in target_cols}
    fails = Counter()
    n_ok = 0
    for props, y in zip(props_list, y_targets):
        if not isinstance(props, dict):
            fails[props] += 1
            continue
        n_ok += 1
        for j, c in enumerate(target_cols):
            errs[c].append(abs(props[c] - float(y[j])))

    out = {f"mae_{c}": (float(np.mean(errs[c])) if errs[c] else float("nan"))
           for c in target_cols}
    out["n_evaluated"] = n_ok
    out["n_total"] = n
    out["coverage"] = n_ok / n if graphs else 0.0
    out["failures"] = dict(fails)
    return out


def property_mae(graphs, y_targets, target_cols=("homo",),
                 atom_vocab=QM9_ATOMS, repair=False, partial_charges=False, seed=1,
                 progress=False, n_jobs=None, **dft_kw):
    unknown = [c for c in target_cols if c not in RDKIT_FNS and c not in DFT_PROPS]
    if unknown:
        raise ValueError(f"unknown target columns {unknown}; "
                         f"known: {tuple(RDKIT_FNS) + DFT_PROPS}")
    needs_dft = any(c in DFT_PROPS for c in target_cols)
    if n_jobs is None:                         # env knob: DFT_JOBS=64 ... (1 = serial)
        n_jobs = int(os.environ.get("DFT_JOBS", "1"))

    cfg = {"target_cols": tuple(target_cols), "atom_vocab": atom_vocab,
           "repair": repair, "partial_charges": partial_charges,
           "needs_dft": needs_dft, "seed": seed, "dft_kw": dft_kw}
    desc = "dft" if needs_dft else "rdkit"
    return _mae_over_graphs(graphs, y_targets, target_cols, cfg, desc, progress, n_jobs)

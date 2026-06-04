import re

import numpy as np
import torch
from rdkit import Chem
from torch_molecule.utils.graph.graph_to_smiles import correct_mol

# Max neutral valence per atomic number (DeFoG/DiGress). Used by the
# partial-charge decoder to detect over-valence-by-one.
_ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


QM9_ATOMS = ("C", "N", "O", "F")
ZINC_ATOMS = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I")

# Bond class index: 0 = no bond, 1 = single, 2 = double, 3 = triple.
N_BOND_CLASSES = 4
_BOND_TO_IDX = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
}
_IDX_TO_BOND = {i: b for b, i in _BOND_TO_IDX.items()}


def _atom_token(atom, charge_aware: bool) -> str:
    s = atom.GetSymbol()
    if not charge_aware:
        return s
    c = atom.GetFormalCharge()
    if c == 0:
        return s
    return s + ("+" if c == 1 else "-" if c == -1 else f"{c:+d}")


def _token_to_atom(token: str):
    if token.endswith("+"):
        atom = Chem.Atom(token[:-1])
        atom.SetFormalCharge(1)
    elif token.endswith("-"):
        atom = Chem.Atom(token[:-1])
        atom.SetFormalCharge(-1)
    else:
        atom = Chem.Atom(token)
    return atom


def smiles_to_tensor(smiles: str, atom_vocab=QM9_ATOMS, charge_aware: bool = True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    # Kekulize: alternating single/double bonds, aromatic flags cleared.
    Chem.Kekulize(mol, clearAromaticFlags=True)

    idx = {s: i for i, s in enumerate(atom_vocab)}
    n = mol.GetNumAtoms()
    X = torch.zeros((n, len(atom_vocab)), dtype=torch.float32)
    for atom in mol.GetAtoms():
        tok = _atom_token(atom, charge_aware)
        if tok not in idx:
            raise ValueError(f"atom token {tok!r} not in vocab {atom_vocab}")
        X[atom.GetIdx(), idx[tok]] = 1.0

    E = torch.zeros((n, n, N_BOND_CLASSES), dtype=torch.float32)
    E[:, :, 0] = 1.0  # default: no bond
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        k = _BOND_TO_IDX[bond.GetBondType()]
        for a, b in ((i, j), (j, i)):
            E[a, b, 0] = 0.0
            E[a, b, k] = 1.0
    return X, E


def tensor_to_mol(X, E, atom_vocab=QM9_ATOMS, repair=True):

    X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    E = E.detach().cpu().numpy() if isinstance(E, torch.Tensor) else np.asarray(E)
    atom_idx = X.argmax(-1)
    bond_idx = E.argmax(-1)
    n = X.shape[0]

    rw = Chem.RWMol()
    for i in range(n):
        rw.AddAtom(_token_to_atom(atom_vocab[int(atom_idx[i])]))
    for i in range(n):
        for j in range(i + 1, n):
            k = int(bond_idx[i, j])
            if k > 0:
                rw.AddBond(i, j, _IDX_TO_BOND[k])

    # Try naive sanitize first.
    try:
        mol = Chem.Mol(rw)
        Chem.SanitizeMol(mol)
        return mol, False
    except Exception:
        if not repair:
            return None, False

    # try torch_molecule's repair pipeline. Aggressive first
    # (with fragment fusion), then plain valency repair. A fresh RWMol copy
    # per call because correct_mol mutates its input in place;
    for connection in (True, False):
        candidate = Chem.RWMol(rw)
        candidate.UpdatePropertyCache(strict=False)
        try:
            mol_conn, _no_correct = correct_mol(candidate, connection=connection)
        except Exception:
            continue
        if mol_conn is not None:
            return mol_conn, True
    return None, False


def _check_valency(mol):
    # (ok, [atom_idx, valence]) — parses RDKit's over-valence message like DeFoG.
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        sub = str(e)[str(e).find("#"):]
        return False, list(map(int, re.findall(r"\d+", sub)))


def build_mol_partial_charges(X, E, atom_vocab=QM9_ATOMS):
    X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    E = E.detach().cpu().numpy() if isinstance(E, torch.Tensor) else np.asarray(E)
    atom_idx = X.argmax(-1)
    bond_idx = E.argmax(-1)
    n = X.shape[0]

    rw = Chem.RWMol()
    for i in range(n):
        rw.AddAtom(_token_to_atom(atom_vocab[int(atom_idx[i])]))
    for i in range(n):
        for j in range(i + 1, n):
            k = int(bond_idx[i, j])
            if k == 0:
                continue
            rw.AddBond(i, j, _IDX_TO_BOND[k])
            ok, ids = _check_valency(rw)
            if ok:
                continue
            idx, v = ids[0], ids[1]
            an = rw.GetAtomWithIdx(idx).GetAtomicNum()
            if an in (7, 8, 16) and (v - _ATOM_VALENCY[an]) == 1:
                rw.GetAtomWithIdx(idx).SetFormalCharge(1)

    try:
        mol = Chem.Mol(rw)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def largest_fragment(mol):
    # Returns largest fragment of unconnected mols (could happen during gen that we generated a unconnected node, following other approaches we only take the max one)
    if mol is None:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return None
    return max(frags, default=mol, key=lambda m: m.GetNumAtoms())

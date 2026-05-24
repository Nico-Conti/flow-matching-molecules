import numpy as np
from rdkit import Chem


QM9_ATOMS = ("C", "N", "O", "F")
ZINC_ATOMS = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "N+", "O-")

# Bond class index: 0 = no bond, 1 = single, 2 = double, 3 = triple.
# Molecules are Kekulized before featurization, so aromatic bonds become
# single/double and there is no separate aromatic class.
N_BOND_CLASSES = 4
_BOND_TO_IDX = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
}
_IDX_TO_BOND = {i: b for b, i in _BOND_TO_IDX.items()}


def _atom_token(atom) -> str:
    """Atom identity = element symbol + formal-charge sign: 'N', 'N+', 'O-'."""
    c = atom.GetFormalCharge()
    s = atom.GetSymbol()
    if c == 0:
        return s
    return s + ("+" if c == 1 else "-" if c == -1 else f"{c:+d}")


def _token_to_atom(token: str):
    """Inverse of _atom_token: an RDKit Atom with the right element + formal charge."""
    if token.endswith("+"):
        atom = Chem.Atom(token[:-1])
        atom.SetFormalCharge(1)
    elif token.endswith("-"):
        atom = Chem.Atom(token[:-1])
        atom.SetFormalCharge(-1)
    else:
        atom = Chem.Atom(token)
    return atom


def smiles_to_tensor(smiles: str, atom_vocab=QM9_ATOMS):
    """SMILES -> (X, E) dense one-hot arrays for a single molecule (unpadded).

    X: (N, len(atom_vocab))  one-hot atom types
    E: (N, N, 4)             one-hot bond types, symmetric, diagonal = no-bond

    Atom identity includes formal charge (e.g. 'N+', 'O-'); raises ValueError for
    unparseable SMILES or any atom token outside ``atom_vocab``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    
    # Kekulize to get alternating single/double bonds, and clear aromatic flags 
    Chem.Kekulize(mol, clearAromaticFlags=True)

    idx = {s: i for i, s in enumerate(atom_vocab)}
    n = mol.GetNumAtoms()
    X = np.zeros((n, len(atom_vocab)), dtype=np.float32)
    for atom in mol.GetAtoms():
        tok = _atom_token(atom)
        if tok not in idx:
            raise ValueError(f"atom token {tok!r} not in vocab {atom_vocab}")
        X[atom.GetIdx(), idx[tok]] = 1.0

    E = np.zeros((n, n, N_BOND_CLASSES), dtype=np.float32)
    E[:, :, 0] = 1.0  # default: no bond
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        k = _BOND_TO_IDX[bond.GetBondType()]
        for a, b in ((i, j), (j, i)):
            E[a, b, 0] = 0.0
            E[a, b, k] = 1.0
    return X, E


def tensor_to_mol(X, E, atom_vocab=QM9_ATOMS):
    """(X, E) one-hot arrays -> sanitized RDKit Mol, or None if invalid. """
    
    X = np.asarray(X)
    E = np.asarray(E)
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

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol

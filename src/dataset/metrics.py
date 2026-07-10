from rdkit import Chem

from .featurize import (
    tensor_to_mol, build_mol_partial_charges, largest_fragment, QM9_ATOMS,
)


def vun(gen_smiles, train_smiles):
    n = len(gen_smiles)
    valid = [s for s in gen_smiles if s is not None]
    unique = set(valid)
    novel = unique - set(train_smiles)
    return {
        "validity": len(valid) / n if n else 0.0,
        "uniqueness": len(unique) / len(valid) if valid else 0.0,
        "novelty": len(novel) / len(unique) if unique else 0.0,
        "n_generated": n,
        "n_valid": len(valid),
        "n_unique": len(unique),
        "n_novel": len(novel),
    }


def vun_from_graphs(graphs, train_smiles, atom_vocab=QM9_ATOMS, repair=False,
                    return_smiles=False, partial_charges=False, progress=False):

    gen, n_repaired = [], 0
    it = graphs
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(graphs, desc="decoding", unit="mol")
    for X, E in it:
        if partial_charges:
            mol, was_repaired = build_mol_partial_charges(X, E, atom_vocab=atom_vocab), False
        else:
            mol, was_repaired = tensor_to_mol(X, E, atom_vocab=atom_vocab, repair=repair)
        if was_repaired:
            n_repaired += 1
        mol = largest_fragment(mol)
        gen.append(Chem.MolToSmiles(mol) if mol is not None else None)
    out = vun(gen, train_smiles)
    out["repair_rate"] = n_repaired / len(graphs) if graphs else 0.0
    if return_smiles:
        return out, gen
    return out


def validity_counts(graphs, atom_vocab=QM9_ATOMS, partial_charges=False):
    # (n_valid, n_total): how many graphs decode to a sanitizable molecule. Same decode
    # path as vun_from_graphs but validity-only, so it's rank-decomposable — DDP workers
    # each score a shard and the counts sum to a global validity fraction.
    n_valid = 0
    for X, E in graphs:
        if partial_charges:
            mol = build_mol_partial_charges(X, E, atom_vocab=atom_vocab)
        else:
            mol, _ = tensor_to_mol(X, E, atom_vocab=atom_vocab, repair=False)
        if largest_fragment(mol) is not None:
            n_valid += 1
    return n_valid, len(graphs)

from rdkit import Chem

from .featurize import tensor_to_mol, largest_fragment, QM9_ATOMS


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


def vun_from_graphs(graphs, train_smiles, atom_vocab=QM9_ATOMS):

    # repair_rate is the fraction of generations where correct_mol successfully
    # rescued the molecule from naive-Sanitize failure.

    gen, n_repaired = [], 0
    for X, E in graphs:
        mol, was_repaired = tensor_to_mol(X, E, atom_vocab=atom_vocab)
        if was_repaired:
            n_repaired += 1
        mol = largest_fragment(mol)
        gen.append(Chem.MolToSmiles(mol) if mol is not None else None)
    out = vun(gen, train_smiles)
    out["repair_rate"] = n_repaired / len(graphs) if graphs else 0.0
    return out

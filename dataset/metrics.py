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
    gen = []
    for X, E in graphs:
        mol = largest_fragment(tensor_to_mol(X, E, atom_vocab=atom_vocab))
        gen.append(Chem.MolToSmiles(mol) if mol is not None else None)
    return vun(gen, train_smiles)

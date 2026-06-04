from collections import Counter

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .featurize import smiles_to_tensor, tensor_to_mol

_UNCHARGER = rdMolStandardize.Uncharger()


def clean_mol(mol, uncharge: bool = False):
    # RemoveStereochemistry, optionally neutralize charges, sanitize.
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    if uncharge:
        mol = _UNCHARGER.uncharge(mol)
    Chem.SanitizeMol(mol)
    return mol


def sanitize_smiles_dataset(smiles_list, atom_vocab, charge_aware=True, uncharge=False,
                            apply_filter=True):
    clean, kept_idx, stats = [], [], Counter()
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s.strip()) if isinstance(s, str) else s
        if m is None:
            stats["drop_parse"] += 1
            continue
        try:
            m = clean_mol(m, uncharge)
            s_clean = Chem.MolToSmiles(m)
        except Exception:
            stats["drop_sanitize"] += 1
            continue
        try:
            X, E = smiles_to_tensor(s_clean, atom_vocab=atom_vocab, charge_aware=charge_aware)
        except ValueError:
            stats["drop_vocab"] += 1
            continue
        except Exception:
            stats["drop_kekulize"] += 1
            continue

        m2, _ = tensor_to_mol(X, E, atom_vocab=atom_vocab, repair=False)
        roundtrips = (
            m2 is not None
            and Chem.MolToSmiles(m2) == s_clean
            and len(Chem.GetMolFrags(m2)) == 1
        )
        if apply_filter and not roundtrips:
            stats["drop_roundtrip"] += 1
            continue

        stats["kept" if roundtrips else "kept_no_roundtrip"] += 1
        clean.append(s_clean)
        kept_idx.append(i)
    return clean, kept_idx, dict(stats)

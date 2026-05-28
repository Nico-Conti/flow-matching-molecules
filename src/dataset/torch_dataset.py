import torch
from torch.utils.data import Dataset

from .featurize import smiles_to_tensor


class MoleculeDataset(Dataset):
    def __init__(self, ds, atom_vocab, charge_aware=True):
        self.ds = ds
        self.atom_vocab = atom_vocab
        self.charge_aware = charge_aware

    @classmethod
    def from_loader(cls, d):
        return cls(d["ds"], d["atom_vocab"])

    def __len__(self):
        return self.ds.num_rows

    def __getitem__(self, i):
        row = self.ds[i]
        X, E = smiles_to_tensor(row["smiles"], atom_vocab=self.atom_vocab,
                                charge_aware=self.charge_aware)
        y = torch.as_tensor(row["y"], dtype=torch.float32)
        return X, E, y


def collate_dense(batch):        
    # Xs, Es, ys = [], [], []
    # for x, e, y in batch:
    #     Xs.append(x)
    #     Es.append(e)
    #     ys.append(y)
    Xs, Es, ys = zip(*batch) #Does the same as above

    B = len(Xs)
    n = max(x.shape[0] for x in Xs)
    a = Xs[0].shape[-1]
    b = Es[0].shape[-1]

    X = torch.zeros(B, n, a)
    E = torch.zeros(B, n, n, b)
    E[:, :, :, 0] = 1.0

    mask = torch.zeros(B, n, dtype=torch.bool)
    for k, (x, e) in enumerate(zip(Xs, Es)):
        ni = x.shape[0]
        X[k, :ni, :] = x
        E[k, :ni, :ni, :] = e
        mask[k, :ni] = True

    y = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in ys])
    return {"X": X, "E": E, "y": y, "mask": mask}


def unbatch(X, E, mask):
    out = []
    for b in range(X.shape[0]):
        n = int(mask[b].sum())
        out.append((X[b, :n], E[b, :n, :n]))
    return out

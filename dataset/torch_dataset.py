import torch
from torch.utils.data import Dataset


class MoleculeDataset(Dataset):
    def __init__(self, X, E, y, smiles=None, atom_vocab=None):
        assert len(X) == len(E) == len(y)
        self.X, self.E = X, E
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.smiles = smiles
        self.atom_vocab = atom_vocab

    @classmethod
    def from_loader(cls, d):
        return cls(d["X"], d["E"], d["y"], d.get("smiles"), d.get("atom_vocab"))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.E[i], self.y[i]


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

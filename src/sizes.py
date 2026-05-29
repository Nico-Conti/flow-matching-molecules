import torch


class SizeSampler:
    """Sample graph sizes n from the empirical training atom-count histogram."""

    def __init__(self, counts):
        counts = torch.as_tensor(counts, dtype=torch.float)
        self.counts = counts
        self.max_n = int(counts.numel() - 1)
        self.probs = counts / counts.sum()

    @classmethod
    def _counts_from_sizes(cls, sizes, max_n=None):
        m = int(max_n) if max_n is not None else max(sizes)
        counts = torch.zeros(m + 1)
        for s in sizes:
            counts[s] += 1.0
        return cls(counts)

    @classmethod
    def from_smiles(cls, smiles, max_n=None):
        """Build the histogram from canonical SMILES with a parse-only pass.
        Counts heavy atoms via RDKit (mol.GetNumAtoms()), which equals
        smiles_to_tensor's X.shape[0] — without allocating the N x N edge
        tensor. Preferred over from_dataset: ~no startup cost on full ZINC."""
        from rdkit import Chem
        sizes = [Chem.MolFromSmiles(s).GetNumAtoms() for s in smiles]
        return cls._counts_from_sizes(sizes, max_n=max_n)

    @classmethod
    def from_dataset(cls, ds, max_n=None):
        """Build the histogram by reading node counts from a MoleculeDataset.
        One-time pass: triggers smiles_to_tensor per row (slow on full QM9/ZINC
        — it builds and discards the full graph tensors). Prefer from_smiles
        when canonical SMILES are already on hand."""
        sizes = [int(ds[i][0].shape[0]) for i in range(len(ds))]
        return cls._counts_from_sizes(sizes, max_n=max_n)

    def sample(self, batch):
        idx = torch.multinomial(self.probs, int(batch), replacement=True)
        return idx.tolist()

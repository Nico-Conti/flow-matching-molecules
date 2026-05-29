import torch

from dataset.torch_dataset import unbatch
from dataset.metrics import vun_from_graphs
from flow import sample


@torch.no_grad()
def evaluate(model, size_sampler, train_smiles, atom_vocab, k_X, k_E,
             n_samples=1000, batch=256, steps=100, device="cpu", repair=False):
    graphs = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch, remaining)
        n_list = size_sampler.sample(b)
        Xoh, Eoh, mask = sample(model, n_list, k_X, k_E, steps=steps, device=device)
        graphs.extend(unbatch(Xoh.cpu(), Eoh.cpu(), mask.cpu()))
        remaining -= b
    return vun_from_graphs(graphs, train_smiles, atom_vocab, repair=repair)

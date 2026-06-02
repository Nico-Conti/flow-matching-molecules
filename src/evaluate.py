import torch

from dataset.torch_dataset import unbatch
from dataset.metrics import vun_from_graphs
from flow import sample
from seeding import set_seed


def _fcd(gen_smiles, ref_smiles, device):
    # FCD on UNIQUE valid generated SMILES vs the reference set (matches CatFlow).
    from fcd_torch import FCD
    gen_unique = list({s for s in gen_smiles if s is not None})
    if not gen_unique or not ref_smiles:
        return float("inf")
    scorer = FCD(device=device, n_jobs=2)
    return float(scorer(gen_unique, list(ref_smiles)))


@torch.no_grad()
def evaluate(model, size_sampler, train_smiles, atom_vocab, k_X, k_E,
             n_samples=1000, batch=256, steps=100, t_end=1.0, device="cpu",
             repair=False, fcd_ref=None, fcd_device=None, seed=None):
    if seed is not None:
        set_seed(seed)
    graphs = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch, remaining)
        n_list = size_sampler.sample(b)
        Xoh, Eoh, mask = sample(model, n_list, k_X, k_E, steps=steps,
                                t_end=t_end, device=device)
        graphs.extend(unbatch(Xoh.cpu(), Eoh.cpu(), mask.cpu()))
        remaining -= b

    if fcd_ref is None:
        return vun_from_graphs(graphs, train_smiles, atom_vocab, repair=repair)

    out, gen = vun_from_graphs(graphs, train_smiles, atom_vocab, repair=repair,
                               return_smiles=True)
    out["fcd"] = _fcd(gen, fcd_ref, fcd_device or device)
    return out

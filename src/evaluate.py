import torch

from dataset.torch_dataset import unbatch
from dataset.metrics import vun_from_graphs
from methods import get_method
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
             repair=False, fcd_ref=None, fcd_device=None, seed=None,
             eta=0.0, distortion="identity",
             method="fm_graph", partial_charges=False, progress=False):
    if isinstance(method, str):
        method = get_method(method)
    if seed is not None:
        set_seed(seed)
    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(total=n_samples, desc="sampling", unit="mol")
    graphs = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch, remaining)
        n_list = size_sampler.sample(b)
        Xoh, Eoh, mask = method.sample(model, n_list, k_X, k_E, steps=steps,
                                       t_end=t_end, device=device,
                                       eta=eta, distortion=distortion)
        graphs.extend(unbatch(Xoh.cpu(), Eoh.cpu(), mask.cpu()))
        remaining -= b
        if bar is not None:
            bar.update(b)
    if bar is not None:
        bar.close()

    if fcd_ref is None:
        return vun_from_graphs(graphs, train_smiles, atom_vocab, repair=repair,
                               partial_charges=partial_charges, progress=progress)

    out, gen = vun_from_graphs(graphs, train_smiles, atom_vocab, repair=repair,
                               return_smiles=True, partial_charges=partial_charges,
                               progress=progress)
    if progress:
        print("computing FCD ...", flush=True)
    out["fcd"] = _fcd(gen, fcd_ref, fcd_device or device)
    return out


@torch.no_grad()
def evaluate_property_targeting(model, size_sampler, atom_vocab, k_X, k_E, targets,
                                cond_cols=("homo",), s_list=(0.0, 1.0, 3.0, 5.0),
                                n_per_target=10, steps=100, t_end=1.0, device="cpu",
                                eta=0.0, distortion="identity",
                                method="fm_graph", repair=False, seed=None,
                                optimize=False, progress=True):

    from dataset.properties import property_mae
    if isinstance(method, str):
        method = get_method(method)
    targets = torch.as_tensor(targets, dtype=torch.float32).view(-1, len(cond_cols))

    results = {}
    for s in s_list:
        if seed is not None:
            set_seed(seed)
        graphs, ys = [], []
        for tvec in targets:
            n_list = size_sampler.sample(n_per_target)
            cond = tvec.view(1, -1).repeat(n_per_target, 1).to(device)
            Xoh, Eoh, mask = method.sample(model, n_list, k_X, k_E, steps=steps,
                                           t_end=t_end, device=device, cond=cond, s=s,
                                           eta=eta, distortion=distortion)
            graphs.extend(unbatch(Xoh.cpu(), Eoh.cpu(), mask.cpu()))
            ys.extend([tvec.tolist()] * n_per_target)
        mae = property_mae(graphs, ys, target_cols=tuple(cond_cols),
                           atom_vocab=atom_vocab, repair=repair, optimize=optimize,
                           progress=progress)
        results[s] = mae
        print(f"s={s}: {mae}")
    return results

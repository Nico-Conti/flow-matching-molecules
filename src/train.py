import os
from contextlib import contextmanager, nullcontext

import torch
from torch.utils.data import DataLoader, random_split

from model import TimeConditionedGraphTransformer
from methods import get_method
from sizes import SizeSampler
from seeding import set_seed


class EMA:

    def __init__(self, parameters, decay=0.999):
        self.decay = decay
        self.params = list(parameters)
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, parameters=None):
        params = self.params if parameters is None else list(parameters)
        for p, s in zip(params, self.shadow):
            p.data.copy_(s.data)

    @contextmanager
    def average_parameters(self):
        backup = [p.detach().clone() for p in self.params]
        self.copy_to()
        try:
            yield
        finally:
            for p, b in zip(self.params, backup):
                p.data.copy_(b)


def _cond_from_batch(batch, cond_idx):
    # (bs, len(cond_idx)) property targets, or None if not conditioning.
    if cond_idx is None or "y" not in batch:
        return None
    return batch["y"][:, cond_idx]


def train_step(model, method, batch, optimizer, lambda_E=1.0, grad_clip=None,
               cond=None, p_uncond=0.0, accum_steps=1, last_micro=True):
    # accum_steps>1: one micro-batch. /K keeps the accumulated gradient a mean, so lr transfers.
    model.train()
    loss, parts = method.loss(model, batch, lambda_E=lambda_E, cond=cond, p_uncond=p_uncond)
    (loss / accum_steps).backward()
    if last_micro:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
    return {
        "loss": float(loss.detach()),
        "loss_x": float(parts["loss_x"]),
        "loss_e": float(parts["loss_e"]),
    }


@torch.no_grad()
def _val_loss(model, method, val_loader, lambda_E, device, ema=None, cond_idx=None):
    model.eval()
    keep = ("X", "E", "mask", "y") if cond_idx is not None else ("X", "E", "mask")
    def _run():
        total, n_batches = 0.0, 0
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items() if k in keep}
            cond = _cond_from_batch(batch, cond_idx)
            loss, _ = method.loss(model, batch, lambda_E=lambda_E, cond=cond, p_uncond=0.0)
            total += float(loss.detach())
            n_batches += 1
        return total / max(n_batches, 1)
    if ema is not None:
        with ema.average_parameters():
            return _run()
    return _run()


@torch.no_grad()
def _sample_validity(model, method, ema, size_sampler, k_X, k_E, n, steps, eta,
                     distortion, atom_vocab, partial_charges, device, sample_seed,
                     batch=256):
    # Sample n molecules unconditionally under the EMA weights and count how many decode
    # to a valid molecule. Reseeds (distinct per epoch/rank) for a diverse draw, and
    # snapshots/restores the RNG so the check never perturbs the training stream.
    from dataset.torch_dataset import unbatch
    from dataset.metrics import validity_counts
    dev = torch.device(device)
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None
    torch.manual_seed(sample_seed)
    ctx = ema.average_parameters() if ema is not None else nullcontext()
    n_valid = n_total = 0
    with ctx:
        for start in range(0, n, batch):
            n_list = size_sampler.sample(min(batch, n - start))
            Xoh, Eoh, mask = method.sample(model, n_list, k_X, k_E, steps=steps,
                                           device=device, eta=eta, distortion=distortion)
            nv, nt = validity_counts(unbatch(Xoh.cpu(), Eoh.cpu(), mask.cpu()),
                                     atom_vocab, partial_charges)
            n_valid += nv
            n_total += nt
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, dev)
    return n_valid, n_total


def _resolve_rows(train_ds):
    # Walk nested Subsets back to the underlying HF dataset for this split.
    base, indices = train_ds, None
    while hasattr(base, "dataset"):
        indices = base.indices if indices is None else [base.indices[i] for i in indices]
        base = base.dataset
    return base.ds if indices is None else base.ds.select(indices)


def _collect_train_smiles(train_ds):
    return [r["smiles"] for r in _resolve_rows(train_ds)]


def _collect_train_targets(train_ds):
    # (N, n_targets) tensor of y for the split — reads ds["y"], no featurization.
    import numpy as np
    return torch.as_tensor(np.asarray(_resolve_rows(train_ds)["y"], dtype="float32"))


def build_split(dataset="qm9", subset=None, seed=0, val_frac=0.15, test_frac=0.10):
    from dataset.torch_dataset import MoleculeDataset

    g = torch.Generator().manual_seed(seed)

    # Datasets with official train/(valid)/test splits: train on `train` (carve val when no
    # official val exists), evaluate on the official `test`. Never random-re-split these —
    # it would leak test molecules into train and break FCD comparability.
    if dataset in ("moses", "guacamol"):
        if dataset == "moses":
            from dataset.moses import load_moses
            _load = load_moses
            d, d_val = _load(split="train"), None      # MOSES ships no val; carve from train
        else:
            from dataset.guacamol import load_guacamol
            _load = load_guacamol
            d, d_val = _load(split="train"), _load(split="valid")   # GuacaMol ships a val
        d_test = _load(split="test")
        atom_vocab = d["atom_vocab"]
        train_full = MoleculeDataset.from_loader(d)
        if subset is not None:
            keep = min(subset, len(train_full))
            train_full, _ = random_split(train_full, [keep, len(train_full) - keep], generator=g)
        if d_val is not None and subset is None:
            train_ds, val_ds = train_full, MoleculeDataset.from_loader(d_val)
        else:
            n_val = max(1, int(len(train_full) * val_frac))
            train_ds, val_ds = random_split(train_full, [len(train_full) - n_val, n_val], generator=g)
        return {
            "train_ds": train_ds, "val_ds": val_ds,
            "test_ds": MoleculeDataset.from_loader(d_test),
            "atom_vocab": atom_vocab, "k_X": len(atom_vocab), "k_E": d.get("n_bond_classes", 4),
            "targets": tuple(d["targets"]),
            "train_smiles": _collect_train_smiles(train_ds),
            "test_smiles": list(d_test["ds"]["smiles"]),
        }

    if dataset == "qm9":
        from dataset.qm9 import load_qm9
        d = load_qm9()
    elif dataset == "zinc":
        from dataset.zinc import load_zinc
        d = load_zinc()
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected 'qm9', 'zinc', 'moses', or 'guacamol'")
    full = MoleculeDataset.from_loader(d)
    atom_vocab = d["atom_vocab"]

    n_total = len(full)
    if subset is not None:
        keep = min(subset, n_total)
        full, _ = random_split(full, [keep, n_total - keep], generator=g)

    n_test = max(0, int(len(full) * test_frac))
    n_val = max(1, int(len(full) * val_frac))
    n_train = len(full) - n_val - n_test
    assert n_train > 0, f"split too aggressive: n_train={n_train} (val={n_val}, test={n_test})"
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=g)

    return {
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "atom_vocab": atom_vocab, "k_X": len(atom_vocab), "k_E": d.get("n_bond_classes", 4),
        "targets": tuple(d["targets"]),
        "train_smiles": _collect_train_smiles(train_ds),
        "test_smiles": _collect_train_smiles(test_ds) if n_test > 0 else [],
    }


def train(devices=1, **hparams):
    """Train a model and return it. Hyperparameters are keyword arguments forwarded
    to `_train` (see it for the full list and defaults), e.g.
    train(dataset="moses", method="defog", epochs=300, batch_size=128).

    Effective batch = batch_size * devices * accum_steps.

    devices > 1 data-parallelizes across that many GPUs on one node: one worker
    process per GPU, gradients averaged automatically by DDP. A multi-GPU run's
    output is the rank-0 checkpoint (save_path / push_repo); train() returns None.
    The machine is shared: 1 GPU is the budget, 2 only for a short period and with an
    e-mail motivating it. Nothing here enforces that; set devices by hand.
    """
    if devices > 1:
        torch.multiprocessing.spawn(_spawn_entry, args=(devices, hparams), nprocs=devices)
        return None
    return _train(rank=0, world_size=1, **hparams)


def _spawn_entry(rank, world_size, hparams):
    # mp.spawn passes args positionally and can't unpack a dict into **hparams, so
    # this thin wrapper does the unpacking for the spawned worker.
    _train(rank, world_size, **hparams)


def _ddp_setup(rank, world_size):
    # Join this worker to the process group and bind it to its own GPU.
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def _make_train_loader(train_ds, batch_size, seed, rank, world_size):
    # Under DDP each rank trains on a disjoint shard (DistributedSampler); on a single
    # process we just shuffle. Returns (loader, sampler); sampler is None off-DDP.
    from dataset.torch_dataset import collate_dense
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=seed)
        loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                            collate_fn=collate_dense)
        return loader, sampler
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_dense, generator=gen)
    return loader, None


def _train(rank, world_size, epochs=50, batch_size=128, accum_steps=1, lr=5e-4, weight_decay=1e-12,
           lambda_E=1.0, ema_decay=0.999, use_ema=True, val_frac=0.15, test_frac=0.10,
           seed=0, device=None, subset=None, log_every=50, dataset="qm9",
           save_path=None, save_every=0, push_repo=None, resume=True,
           grad_clip=None, amsgrad=False, deterministic=False, method="fm_graph", n_layers=None,
           extra_features=None, rrwp_steps=12, dy=None,
           cond_cols=None, p_uncond=0.15, cond_emb=64,
           val_sample_every=1, n_val_samples=1000, val_sample_steps=500,
           val_sample_eta=0.0, val_sample_distortion="polydec"):

    from dataset.torch_dataset import collate_dense

    distributed = world_size > 1
    is_main = rank == 0
    torch.set_num_threads(max(1, 18 // world_size))   # our 25% of the 72 physical cores,
                                                      # split across ranks; else torch takes all 144
    def log(*a, **k):                        # only the main rank writes to stdout
        if is_main:
            print(*a, **k)

    if distributed:
        device = _ddp_setup(rank, world_size)
    else:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # The local checkpoint path is implicit from push_repo unless given.
    if save_path is None and push_repo is not None:
        from checkpoint import repo_to_path
        save_path = repo_to_path(push_repo)

    set_seed(seed, deterministic=deterministic)

    sp = build_split(dataset=dataset, subset=subset, seed=seed,
                     val_frac=val_frac, test_frac=test_frac)
    train_ds, val_ds = sp["train_ds"], sp["val_ds"]
    atom_vocab, k_X, k_E = sp["atom_vocab"], sp["k_X"], sp["k_E"]
    train_smiles, test_smiles = sp["train_smiles"], sp["test_smiles"]

    cond_idx = cond_dim = None
    if cond_cols:
        cond_idx = [sp["targets"].index(c) for c in cond_cols]
        cond_dim = len(cond_idx)

    train_loader, train_sampler = _make_train_loader(train_ds, batch_size, seed, rank, world_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_dense)

    # Build the size histogram from SMILES (parse-only) rather than from_dataset,
    # which would featurize the whole split just to read atom counts.
    size_sampler = SizeSampler.from_smiles(train_smiles)

    method_name = method
    method = get_method(method_name)

    arch = {} if n_layers is None else {"n_layers": n_layers}
    if dy is not None:                                     # global-stream width (cond + time + extra)
        arch["dy"] = dy
    if extra_features is not None:                         # needs max_n_nodes
        arch.update(extra_features=extra_features, rrwp_steps=rrwp_steps,
                    max_n_nodes=size_sampler.max_n)
    if cond_dim:
        arch.update(cond_dim=cond_dim, cond_emb=cond_emb)
    model = TimeConditionedGraphTransformer(k_X=k_X, k_E=k_E, **arch).to(device)
    if cond_dim:                                           # z-score stats from train split
        ytr = _collect_train_targets(train_ds)[:, cond_idx]
        model.set_cond_stats(ytr.mean(0).to(device), ytr.std(0).to(device))
        log(f"  conditioning on {cond_cols} (cols {cond_idx}); "
            f"mean {ytr.mean(0).tolist()}, std {ytr.std(0).tolist()}")

    # ddp_model wraps `model` so backward() averages gradients across ranks; everything
    # else (optimizer, EMA, checkpoint, validation) keeps using the unwrapped `model`.
    ddp_model = model
    if distributed:
        # find_unused_parameters: the model discards the global-y output (forward returns
        # only outX, outE), so the last layer's y-path + mlp_out_y receive no gradient.
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ema = EMA(model.parameters(), decay=ema_decay) if use_ema else None

    history = {"step": [], "epoch": [], "loss": [], "loss_x": [], "loss_e": [],
               "val_loss": [], "validity": []}

    from checkpoint import best_path as _best_path
    best_ckpt = _best_path(save_path) if save_path else None
    best_V = -1.0

    def _save(epoch, path, push=False, tag="checkpoint"):
        # Saves live training weights + EMA shadow + optimizer/scheduler state.
        from checkpoint import save_checkpoint, push_checkpoint_to_hf
        val = history["val_loss"][-1] if history["val_loss"] else float("nan")
        # report validity only if measured THIS epoch; else it'd be a stale value from the
        # last val_sample_every epoch (periodic saves are offset from validity epochs).
        measured = bool(val_sample_every) and epoch % val_sample_every == 0 and bool(history["validity"])
        vld = history["validity"][-1] if measured else None
        save_checkpoint(path, model, k_X=k_X, k_E=k_E,
                        atom_vocab=atom_vocab, size_sampler=size_sampler,
                        train_smiles=train_smiles, history=history,
                        ema_shadow=(ema.shadow if ema is not None else None),
                        optimizer=opt, scheduler=sched, epoch=epoch,
                        method=method_name,
                        extra={"dataset": dataset, "lambda_E": lambda_E,
                               "seed": seed, "best_validity": best_V,
                               "cond_cols": list(cond_cols) if cond_cols else None,
                               "p_uncond": p_uncond})
        msg = f"  {tag} saved -> {path} (epoch {epoch})"
        if push and push_repo:
            vstr = f", validity {vld:.4f}" if vld is not None else ""
            push_checkpoint_to_hf(path, push_repo,
                                  commit_message=f"{tag}: epoch {epoch}, val_loss {val:.4f}{vstr}")
            msg += f" + pushed to {push_repo}"
        print(msg)

    # Auto-resume: restore live weights, EMA, optimizer, scheduler, epoch, RNG. All ranks
    # load the same checkpoint; the main rank warms the (shared) HF cache first so the
    # ranks don't race the download.
    start_epoch = 0
    if resume and save_path:
        from checkpoint import resolve_checkpoint
        if distributed:
            import torch.distributed as dist
            if is_main:
                resolve_checkpoint(save_path, push_repo)
            dist.barrier()
        local = resolve_checkpoint(save_path, push_repo)
        if local is not None:
            ck = torch.load(local, map_location=device, weights_only=False)
            if ck.get("optimizer") is None or ck.get("epoch") is None:
                log(f"  found {local} but it is not resumable; starting fresh")
            else:
                model.load_state_dict(ck["state_dict"])
                if ema is not None and ck.get("ema_shadow") is not None:
                    for s, saved in zip(ema.shadow, ck["ema_shadow"]):
                        s.copy_(saved.to(device))
                opt.load_state_dict(ck["optimizer"])
                if ck.get("scheduler") is not None:
                    sched.load_state_dict(ck["scheduler"])
                if ck.get("history"):
                    history = ck["history"]
                    history.setdefault("validity", [])     # pre-validity checkpoints lack it
                if history.get("validity"):
                    best_V = max(history["validity"])
                if ck.get("rng_state") is not None:
                    try:
                        torch.set_rng_state(ck["rng_state"].cpu())
                    except Exception:
                        pass
                start_epoch = int(ck["epoch"]) + 1
                log(f"  resumed from {local} at epoch {start_epoch}")

    keep = ("X", "E", "mask", "y") if cond_idx is not None else ("X", "E", "mask")
    partial = (dataset == "zinc")          # partial-charge decode for validity, auto by dataset
    step = len(history["step"])
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:                      # reshuffle the shards each epoch
            train_sampler.set_epoch(epoch)
        n_micro = len(train_loader)
        micro = []
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items() if k in keep}
            cond = _cond_from_batch(batch, cond_idx)
            last_micro = (i + 1) % accum_steps == 0 or (i + 1) == n_micro   # short tail still steps
            comp = train_step(ddp_model, method, batch, opt, lambda_E=lambda_E,
                              grad_clip=grad_clip, cond=cond, p_uncond=p_uncond,
                              accum_steps=accum_steps, last_micro=last_micro)
            micro.append(comp)
            if not last_micro:                 # a step is one optimizer move, not one micro-batch
                continue
            comp = {k: sum(m[k] for m in micro) / len(micro) for k in comp}
            micro = []
            if ema is not None:
                ema.update()
            history["step"].append(step)
            history["epoch"].append(epoch)
            history["loss"].append(comp["loss"])
            history["loss_x"].append(comp["loss_x"])
            history["loss_e"].append(comp["loss_e"])
            if step % log_every == 0:
                log(f"epoch {epoch} step {step} "
                    f"loss {comp['loss']:.4f} "
                    f"loss_x {comp['loss_x']:.4f} "
                    f"loss_e {comp['loss_e']:.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e}")
            step += 1
        sched.step()

        # Generative validity on the EMA weights, computed on ALL ranks so the global
        # fraction can be all-reduced; this (not val_loss) drives best-ckpt selection —
        # val_loss and sample validity decouple here (see meetings/doctorand.md §5).
        V = None
        if val_sample_every and epoch % val_sample_every == 0:
            per_rank = -(-n_val_samples // world_size)          # ceil: 1000 over 4 -> 250
            nv, nt = _sample_validity(model, method, ema, size_sampler, k_X, k_E,
                                      n=per_rank, steps=val_sample_steps, eta=val_sample_eta,
                                      distortion=val_sample_distortion, atom_vocab=atom_vocab,
                                      partial_charges=partial, device=device,
                                      sample_seed=seed + 10000 + epoch * world_size + rank)
            if distributed:
                import torch.distributed as dist
                counts = torch.tensor([nv, nt], device=device)
                dist.all_reduce(counts)                         # sum valid/total over ranks
                nv, nt = int(counts[0]), int(counts[1])
            V = nv / max(nt, 1)

        # val_loss, logging, checkpoint selection and saving on the main rank only; the
        # other ranks skip them and block at the next epoch's first all-reduce.
        if is_main:
            val_loss = _val_loss(model, method, val_loader, lambda_E, device, ema=ema,
                                 cond_idx=cond_idx)
            history["val_loss"].append(val_loss)
            if V is not None:
                history["validity"].append(V)
            log(f"epoch {epoch} done — val_loss {val_loss:.4f}"
                + (f" — validity {V:.3f}" if V is not None else ""))
            if V is not None and save_path and V > best_V:
                best_V = V
                _save(epoch, best_ckpt, push=bool(push_repo), tag="best")
            if save_path and save_every and (epoch + 1) % save_every == 0:
                _save(epoch, save_path, push=bool(push_repo), tag="checkpoint")

    # Final checkpoint: save live weights + EMA shadow (load_checkpoint overlays
    # EMA for eval) before installing EMA into the returned model.
    if is_main and save_path:
        _save(epochs - 1, save_path, push=bool(push_repo), tag="final")

    # Install the EMA weights so sampling/evaluation on the returned model uses
    # them (the paper reports metrics under EMA). Training ran on live weights.
    if ema is not None:
        ema.copy_to(model.parameters())

    if distributed:
        import torch.distributed as dist
        dist.destroy_process_group()

    return model, history, size_sampler, train_smiles, atom_vocab, k_X, k_E, test_smiles

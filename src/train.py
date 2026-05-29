from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader, random_split

from model import FMModel
from flow import fm_loss
from sizes import SizeSampler


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


def train_step(model, batch, optimizer, lambda_E=1.0, grad_clip=None):
    model.train()
    optimizer.zero_grad()
    loss, parts = fm_loss(model, batch, lambda_E=lambda_E)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "loss_x": float(parts["loss_x"]),
        "loss_e": float(parts["loss_e"]),
    }


@torch.no_grad()
def _val_loss(model, val_loader, lambda_E, device, ema=None):
    model.eval()
    def _run():
        total, n_batches = 0.0, 0
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ("X", "E", "mask")}
            loss, _ = fm_loss(model, batch, lambda_E=lambda_E)
            total += float(loss.detach())
            n_batches += 1
        return total / max(n_batches, 1)
    if ema is not None:
        with ema.average_parameters():
            return _run()
    return _run()


def _collect_train_smiles(train_ds):
    base = train_ds
    indices = None
    while hasattr(base, "dataset"):
        indices = base.indices if indices is None else [base.indices[i] for i in indices]
        base = base.dataset
    rows = base.ds if indices is None else base.ds.select(indices)
    return [r["smiles"] for r in rows]


def train(epochs=50, batch_size=128, lr=5e-4, weight_decay=1e-12, lambda_E=1.0,
          ema_decay=0.999, use_ema=True, val_frac=0.15, test_frac=0.10,
          seed=0, device=None, subset=None, log_every=50, dataset="qm9",
          save_path=None, save_every=0, push_repo=None, resume=True,
          grad_clip=None):
    from dataset.torch_dataset import MoleculeDataset, collate_dense

    # The local checkpoint path is implicit from push_repo unless given.
    if save_path is None and push_repo is not None:
        from checkpoint import repo_to_path
        save_path = repo_to_path(push_repo)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(seed)

    if dataset == "qm9":
        from dataset.qm9 import load_qm9
        d = load_qm9()
    elif dataset == "zinc":
        from dataset.zinc import load_zinc
        d = load_zinc()
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected 'qm9' or 'zinc'")
    full = MoleculeDataset.from_loader(d)
    atom_vocab = d["atom_vocab"]
    k_X, k_E = len(atom_vocab), 4

    n_total = len(full)
    if subset is not None:
        keep = min(subset, n_total)
        full, _ = random_split(full, [keep, n_total - keep], generator=g)

    n_test = max(0, int(len(full) * test_frac))
    n_val = max(1, int(len(full) * val_frac))
    n_train = len(full) - n_val - n_test
    assert n_train > 0, f"split too aggressive: n_train={n_train} (val={n_val}, test={n_test})"
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_dense)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_dense)

    train_smiles = _collect_train_smiles(train_ds)
    test_smiles = _collect_train_smiles(test_ds) if n_test > 0 else []
    # Build the size histogram from SMILES (parse-only) rather than from_dataset,
    # which would featurize the whole split just to read atom counts.
    size_sampler = SizeSampler.from_smiles(train_smiles)

    model = FMModel(k_X=k_X, k_E=k_E).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ema = EMA(model.parameters(), decay=ema_decay) if use_ema else None

    history = {"step": [], "epoch": [], "loss": [], "loss_x": [], "loss_e": [],
               "val_loss": []}

    from checkpoint import best_path as _best_path
    best_ckpt = _best_path(save_path) if save_path else None
    best_val = float("inf")

    def _save(epoch, path, push=False, tag="checkpoint"):
        # Saves live training weights + EMA shadow + optimizer/scheduler state,
        # so the checkpoint is both eval-ready and resume-ready. Each push uses
        # a descriptive commit message, building a navigable HF commit history.
        from checkpoint import save_checkpoint, push_checkpoint_to_hf
        val = history["val_loss"][-1] if history["val_loss"] else float("nan")
        save_checkpoint(path, model, k_X=k_X, k_E=k_E,
                        atom_vocab=atom_vocab, size_sampler=size_sampler,
                        train_smiles=train_smiles, history=history,
                        ema_shadow=(ema.shadow if ema is not None else None),
                        optimizer=opt, scheduler=sched, epoch=epoch,
                        extra={"dataset": dataset, "lambda_E": lambda_E,
                               "seed": seed, "best_val": best_val})
        msg = f"  {tag} saved -> {path} (epoch {epoch})"
        if push and push_repo:
            push_checkpoint_to_hf(path, push_repo,
                                  commit_message=f"{tag}: epoch {epoch}, val_loss {val:.4f}")
            msg += f" + pushed to {push_repo}"
        print(msg)

    # Auto-resume: restore live weights, EMA, optimizer, scheduler, epoch, RNG.
    start_epoch = 0
    if resume and save_path:
        from checkpoint import resolve_checkpoint
        local = resolve_checkpoint(save_path, push_repo)
        if local is not None:
            ck = torch.load(local, map_location=device, weights_only=False)
            if ck.get("optimizer") is None or ck.get("epoch") is None:
                print(f"  found {local} but it is not resumable; starting fresh")
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
                if history.get("val_loss"):
                    best_val = min(history["val_loss"])
                if ck.get("rng_state") is not None:
                    try:
                        torch.set_rng_state(ck["rng_state"].cpu())
                    except Exception:
                        pass
                start_epoch = int(ck["epoch"]) + 1
                print(f"  resumed from {local} at epoch {start_epoch}")

    step = len(history["step"])
    for epoch in range(start_epoch, epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ("X", "E", "mask")}
            comp = train_step(model, batch, opt, lambda_E=lambda_E,
                              grad_clip=grad_clip)
            if ema is not None:
                ema.update()
            history["step"].append(step)
            history["epoch"].append(epoch)
            history["loss"].append(comp["loss"])
            history["loss_x"].append(comp["loss_x"])
            history["loss_e"].append(comp["loss_e"])
            if step % log_every == 0:
                print(f"epoch {epoch} step {step} "
                      f"loss {comp['loss']:.4f} "
                      f"loss_x {comp['loss_x']:.4f} "
                      f"loss_e {comp['loss_e']:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}")
            step += 1
        sched.step()
        val_loss = _val_loss(model, val_loader, lambda_E, device, ema=ema)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch} done — val_loss {val_loss:.4f}")
        if save_path and val_loss < best_val:
            best_val = val_loss
            _save(epoch, best_ckpt, push=bool(push_repo), tag="best")
        if save_path and save_every and (epoch + 1) % save_every == 0:
            _save(epoch, save_path, push=bool(push_repo), tag="checkpoint")

    # Final checkpoint: save live weights + EMA shadow (load_checkpoint overlays
    # EMA for eval) before installing EMA into the returned model.
    if save_path:
        _save(epochs - 1, save_path, push=bool(push_repo), tag="final")

    # Install the EMA weights so sampling/evaluation on the returned model uses
    # them (the paper reports metrics under EMA). Training ran on live weights.
    if ema is not None:
        ema.copy_to(model.parameters())

    return model, history, size_sampler, train_smiles, atom_vocab, k_X, k_E, test_smiles

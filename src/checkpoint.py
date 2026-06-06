import inspect
import os

import torch


CKPT_FORMAT = 2


def repo_to_path(push_repo, directory="checkpoints", ext=".pt"):
    # "nico8771/qm9_stage1_fm" -> "checkpoints/qm9_stage1_fm.pt"
    name = push_repo.rstrip("/").split("/")[-1]
    return os.path.join(directory, name + ext)


def best_path(save_path):
    # "checkpoints/qm9_stage1_fm.pt" -> "checkpoints/qm9_stage1_fm_best.pt"
    root, ext = os.path.splitext(save_path)
    return root + "_best" + ext


def _model_kwargs(model, k_X, k_E):
    # Prefer the model's recorded constructor args so overrides (n_layers, dims,
    # ...) round-trip; this is what lets a 12-layer ZINC checkpoint reload as 12.
    if hasattr(model, "init_kwargs"):
        kw = dict(model.init_kwargs)
        kw["k_X"], kw["k_E"] = int(k_X), int(k_E)
        return kw
    # Fallback for older models: snapshot the constructor defaults as they are NOW.
    sig = inspect.signature(type(model).__init__)
    kw = {name: p.default for name, p in sig.parameters.items()
          if p.default is not inspect.Parameter.empty}
    kw["k_X"], kw["k_E"] = int(k_X), int(k_E)
    if getattr(model, "max_n_nodes", None) is not None:
        kw["max_n_nodes"] = model.max_n_nodes
    return kw


def save_checkpoint(path, model, *, k_X, k_E, atom_vocab, size_sampler,
                    train_smiles, history=None, extra=None,
                    ema_shadow=None, optimizer=None, scheduler=None, epoch=None,
                    method="fm_graph"):
    # state_dict = live training weights; ema_shadow = EMA copy used for eval.
    # Both are kept so a checkpoint is eval-ready (EMA) and resume-ready (live
    # weights + optimizer state).
    payload = {
        "format": CKPT_FORMAT,
        "method": method,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_kwargs": _model_kwargs(model, k_X, k_E),
        "atom_vocab": list(atom_vocab),
        "size_counts": size_sampler.counts.detach().cpu(),
        "train_smiles": list(train_smiles),
        "history": history,
        "extra": dict(extra or {}),
        "ema_shadow": ([s.detach().cpu() for s in ema_shadow]
                       if ema_shadow is not None else None),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "rng_state": torch.get_rng_state(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path, device=None, eval_weights=True):
    # eval_weights=True: returned model carries EMA weights (what eval uses).
    # eval_weights=False: returned model carries the live training weights.
    from model import TimeConditionedGraphTransformer
    from sizes import SizeSampler

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model = TimeConditionedGraphTransformer(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])          # live weights + buffers
    shadow = ckpt.get("ema_shadow")
    if eval_weights and shadow is not None:            # overlay EMA onto params
        with torch.no_grad():
            for p, s in zip(model.parameters(), shadow):
                p.data.copy_(s.to(device))
    model.eval()

    k_X = ckpt["model_kwargs"]["k_X"]
    k_E = ckpt["model_kwargs"]["k_E"]
    return {
        "model": model,
        "method": ckpt.get("method", "fm_graph"),
        "size_sampler": SizeSampler(ckpt["size_counts"]),
        "train_smiles": ckpt["train_smiles"],
        "atom_vocab": tuple(ckpt["atom_vocab"]),
        "k_X": k_X,
        "k_E": k_E,
        "history": ckpt.get("history"),
        "extra": ckpt.get("extra", {}),
        "ema_shadow": shadow,
        "optimizer": ckpt.get("optimizer"),
        "scheduler": ckpt.get("scheduler"),
        "epoch": ckpt.get("epoch"),
        "rng_state": ckpt.get("rng_state"),
    }


def _hf_token(token=None):
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("pushing requires HF_TOKEN (set it in .env)")
    return token


def resolve_checkpoint(save_path, push_repo=None, token=None):
    # Local path to resume from: prefer an existing local file, else pull
    # basename(save_path) from the HF repo. Never raises -- None means start
    # fresh.
    if save_path and os.path.exists(save_path):
        return save_path
    if push_repo and save_path:
        try:
            from huggingface_hub import hf_hub_download
            tok = _hf_token(token)
            return hf_hub_download(repo_id=push_repo,
                                   filename=os.path.basename(save_path),
                                   repo_type="model", token=tok)
        except Exception:
            return None
    return None


def push_checkpoint_to_hf(path, repo_id, token=None, path_in_repo=None,
                          commit_message="Add model checkpoint", private=True):
    from huggingface_hub import HfApi
    token = _hf_token(token)
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", token=token, exist_ok=True,
                    private=private)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo or os.path.basename(path),
        repo_id=repo_id, repo_type="model", token=token,
        commit_message=commit_message)
    return repo_id


def load_checkpoint_from_hf(repo_id, filename, token=None, device=None,
                            eval_weights=True):
    from huggingface_hub import hf_hub_download
    token = _hf_token(token)
    local = hf_hub_download(repo_id=repo_id, filename=filename,
                            repo_type="model", token=token)
    return load_checkpoint(local, device=device, eval_weights=eval_weights)

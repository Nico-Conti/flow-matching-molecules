import torch
import torch.nn.functional as F

from model import mask_graph


# ---------------------------------------------------------------------------
# Continuous (Gaussian) flow matching on graphs: linear interpolation of one-hot
# node/edge features from a Gaussian prior, MSE-to-velocity training loss, and
# deterministic ODE (Euler) sampling with a final argmax back to one-hot.
# ---------------------------------------------------------------------------


def sample_xt(X1, E1, node_mask, t):
    X0 = torch.randn_like(X1)
    E0 = torch.randn_like(E1)
    E0 = 0.5 * (E0 + E0.transpose(1, 2))            # symmetric edge noise

    tX = t.view(-1, 1, 1)
    tE = t.view(-1, 1, 1, 1)
    Xt = (1 - tX) * X0 + tX * X1
    Et = (1 - tE) * E0 + tE * E1

    vX = X1 - X0
    vE = E1 - E0

    Xt, Et = mask_graph(Xt, Et, node_mask)
    vX, vE = mask_graph(vX, vE, node_mask)
    return Xt, Et, vX, vE


def fm_loss(model, batch, lambda_E=1.0):

    X1, E1, node_mask = batch["X"], batch["E"], batch["mask"]
    bs, n = X1.shape[0], X1.shape[1]
    t = torch.rand(bs, device=X1.device)

    Xt, Et, vX, vE = sample_xt(X1, E1, node_mask, t)
    pX, pE = model(Xt, Et, t, node_mask)

    x_mask = node_mask.unsqueeze(-1).to(X1.dtype)                       # bs,n,1
    e_pair = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2))          # bs,n,n
    diag = torch.eye(n, dtype=torch.bool, device=X1.device).view(1, n, n)
    e_mask = (e_pair & ~diag).unsqueeze(-1).to(X1.dtype)               # bs,n,n,1

    denomX = (x_mask.sum() * X1.shape[-1]).clamp_min(1.0)
    denomE = (e_mask.sum() * E1.shape[-1]).clamp_min(1.0)
    lossX = (((pX - vX) ** 2) * x_mask).sum() / denomX
    lossE = (((pE - vE) ** 2) * e_mask).sum() / denomE
    loss = lossX + lambda_E * lossE
    return loss, {"loss_x": lossX.detach(), "loss_e": lossE.detach()}


@torch.no_grad()
def sample(model, n_list, k_X, k_E, steps=100, t_end=1.0, device="cpu"):
    model.eval()
    bs = len(n_list)
    n = max(n_list)
    node_mask = torch.zeros(bs, n, dtype=torch.bool, device=device)
    for i, ni in enumerate(n_list):
        node_mask[i, :ni] = True

    X = torch.randn(bs, n, k_X, device=device)
    E = torch.randn(bs, n, n, k_E, device=device)
    E = 0.5 * (E + E.transpose(1, 2))
    X, E = mask_graph(X, E, node_mask)

    ts = torch.linspace(0, t_end, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(bs)
        dt = ts[i + 1] - ts[i]
        vX, vE = model(X, E, t, node_mask)
        X = X + dt * vX
        E = E + dt * vE
        E = 0.5 * (E + E.transpose(1, 2))
        X, E = mask_graph(X, E, node_mask)

    Xoh = F.one_hot(X.argmax(-1), k_X).to(X.dtype)
    Eoh = F.one_hot(E.argmax(-1), k_E).to(E.dtype)
    Xoh, Eoh = mask_graph(Xoh, Eoh, node_mask)
    return Xoh, Eoh, node_mask


class FMGraph:
    """Continuous Gaussian flow matching: linear interpolation of one-hot
    features, MSE-to-velocity loss, deterministic ODE Euler sampling."""

    name = "fm_graph"

    def loss(self, model, batch, lambda_E=1.0):
        return fm_loss(model, batch, lambda_E=lambda_E)

    def sample(self, model, n_list, k_X, k_E, steps=100, device="cpu", t_end=1.0):
        return sample(model, n_list, k_X, k_E, steps=steps, t_end=t_end,
                      device=device)

import torch
import torch.nn.functional as F

from model import mask_graph


# ---------------------------------------------------------------------------
# Discrete flow matching (DeFoG), minimal form: uniform prior, R^* rate matrix
# only (no detailed-balance / target-guidance terms).
# ---------------------------------------------------------------------------


def _sample_discrete(probX, probE, node_mask):
    bs, n, kX = probX.shape
    kE = probE.shape[-1]

    pX = probX.clone()
    pX[~node_mask] = 1.0 / kX                                  # valid rows only
    Xlab = pX.reshape(bs * n, kX).multinomial(1).reshape(bs, n)

    pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    diag = torch.eye(n, dtype=torch.bool, device=probE.device).unsqueeze(0)
    pE = probE.clone()
    pE[~pair_mask] = 1.0 / kE
    pE[diag.expand(bs, -1, -1)] = 1.0 / kE
    Elab = pE.reshape(bs * n * n, kE).multinomial(1).reshape(bs, n, n)
    Elab = torch.triu(Elab, diagonal=1)
    Elab = Elab + Elab.transpose(1, 2)                         # symmetric

    Xoh = F.one_hot(Xlab, kX).float()
    Eoh = F.one_hot(Elab, kE).float()
    Xoh, Eoh = mask_graph(Xoh, Eoh, node_mask)
    return Xoh, Eoh


def apply_noise(X1, E1, node_mask, t, kX, kE):
    tX = t.view(-1, 1, 1)
    tE = t.view(-1, 1, 1, 1)
    probX = tX * X1 + (1 - tX) / kX
    probE = tE * E1 + (1 - tE) / kE
    return _sample_discrete(probX, probE, node_mask)


def defog_loss(model, batch, lambda_E=1.0):
    X1, E1, node_mask = batch["X"], batch["E"], batch["mask"]
    bs, n = X1.shape[0], X1.shape[1]
    kX, kE = X1.shape[-1], E1.shape[-1]
    t = torch.rand(bs, device=X1.device)

    Xt, Et = apply_noise(X1, E1, node_mask, t, kX, kE)
    logX, logE = model(Xt, Et, t, node_mask)                   # read as logits

    e_pair = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    diag = torch.eye(n, dtype=torch.bool, device=X1.device).view(1, n, n)
    e_mask = e_pair & ~diag

    lossX = F.cross_entropy(logX[node_mask], X1.argmax(-1)[node_mask])
    lossE = F.cross_entropy(logE[e_mask], E1.argmax(-1)[e_mask])
    loss = lossX + lambda_E * lossE
    return loss, {"loss_x": lossX.detach(), "loss_e": lossE.detach()}


def _rstar(zt_label, x1_oh, p0, t):
    dt_p = x1_oh - p0                                          # d/dt p_t(. | x1)
    pt = t * x1_oh + (1 - t) * p0                              # p_t(. | x1)
    idx = zt_label.unsqueeze(-1)
    dt_p_at = dt_p.gather(-1, idx)
    pt_at = pt.gather(-1, idx)
    Z = (pt > 0).sum(-1, keepdim=True)
    R = F.relu(dt_p - dt_p_at) / (Z * pt_at).clamp_min(1e-9)
    R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return R.masked_fill(pt == 0, 0.0)


def _step_probs(R, zt_label, dt):
    """CTMC Euler step: off-diagonal mass = R * dt, diagonal absorbs the rest."""
    step = R * dt
    idx = zt_label.unsqueeze(-1)
    step.scatter_(-1, idx, 0.0)
    stay = (1.0 - step.sum(-1, keepdim=True)).clamp_min(0.0)
    step.scatter_(-1, idx, stay)
    return step


@torch.no_grad()
def sample(model, n_list, k_X, k_E, steps=100, device="cpu", **_):
    model.eval()
    bs = len(n_list)
    n = max(n_list)
    node_mask = torch.zeros(bs, n, dtype=torch.bool, device=device)
    for i, ni in enumerate(n_list):
        node_mask[i, :ni] = True

    p0X, p0E = 1.0 / k_X, 1.0 / k_E
    probX = torch.full((bs, n, k_X), p0X, device=device)
    probE = torch.full((bs, n, n, k_E), p0E, device=device)
    X, E = _sample_discrete(probX, probE, node_mask)           # z_0 ~ uniform

    for i in range(steps):
        t = i / steps
        dt = 1.0 / steps
        tt = torch.full((bs,), t, device=device)
        logX, logE = model(X, E, tt, node_mask)
        phatX = F.softmax(logX, dim=-1)
        phatE = F.softmax(logE, dim=-1)

        if i == steps - 1:                                     # last step: s = 1
            probX, probE = phatX, phatE
        else:
            x1X, x1E = _sample_discrete(phatX, phatE, node_mask)
            RX = _rstar(X.argmax(-1), x1X, p0X, t)
            RE = _rstar(E.argmax(-1), x1E, p0E, t)
            probX = _step_probs(RX, X.argmax(-1), dt)
            probE = _step_probs(RE, E.argmax(-1), dt)

        X, E = _sample_discrete(probX, probE, node_mask)

    return X, E, node_mask


class DeFoG:
    name = "defog"

    def loss(self, model, batch, lambda_E=1.0):
        return defog_loss(model, batch, lambda_E=lambda_E)

    def sample(self, model, n_list, k_X, k_E, steps=100, device="cpu", **kw):
        return sample(model, n_list, k_X, k_E, steps=steps, device=device)

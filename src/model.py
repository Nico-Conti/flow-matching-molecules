import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from features import extra_graph_features, extra_feature_dims


# ---------------------------------------------------------------------------
# Helpers (ported from catflow/models/layers.py and catflow/utils.py),
# decoupled from PyG and the PlaceHolder container.
# ---------------------------------------------------------------------------

def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


def mask_graph(X, E, node_mask):
    """Zero padded nodes in X and padded node-pairs in E. y is untouched."""
    x_mask = node_mask.unsqueeze(-1).to(X.dtype)   # bs, n, 1
    e_mask1 = x_mask.unsqueeze(2)                   # bs, n, 1, 1
    e_mask2 = x_mask.unsqueeze(1)                   # bs, 1, n, 1
    X = X * x_mask
    E = E * e_mask1 * e_mask2
    return X, E


class Xtoy(nn.Module):
    """Map node features to global features."""
    def __init__(self, dx, dy):
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


class Etoy(nn.Module):
    """Map edge features to global features."""
    def __init__(self, d, dy):
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


# ---------------------------------------------------------------------------
# Node/edge self-attention block (DiGress). FiLM: E->X, y->E, y->X.
# ---------------------------------------------------------------------------

class NodeEdgeBlock(nn.Module):
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx, self.de, self.dy = dx, de, dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(dx, dx)
        self.v = nn.Linear(dx, dx)

        self.e_add = nn.Linear(de, dx)
        self.e_mul = nn.Linear(de, dx)

        self.y_e_mul = nn.Linear(dy, dx)
        self.y_e_add = nn.Linear(dy, dx)
        self.y_x_mul = nn.Linear(dy, dx)
        self.y_x_add = nn.Linear(dy, dx)

        self.y_y = nn.Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        self.x_out = nn.Linear(dx, dx)
        self.e_out = nn.Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1).to(X.dtype)   # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)                  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)                  # bs, 1, n, 1

        Q = self.q(X) * x_mask
        K = self.k(X) * x_mask
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))
        Q = Q.unsqueeze(2)                             # bs, 1, n, n_head, df
        K = K.unsqueeze(1)                             # bs, n, 1, n_head, df

        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        E2 = self.e_add(E) * e_mask1 * e_mask2
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        Y = Y * (E1 + 1) + E2                          # bs, n, n, n_head, df

        newE = Y.flatten(start_dim=3)                  # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE
        newE = self.e_out(newE) * e_mask1 * e_mask2    # bs, n, n, de

        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head, df

        V = self.v(X) * x_mask
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                             # bs, 1, n, n_head, df

        weighted_V = (attn * V).sum(dim=2)
        weighted_V = weighted_V.flatten(start_dim=2)   # bs, n, dx

        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V
        newX = self.x_out(newX) * x_mask

        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = self.y_out(y + x_y + e_y)

        return newX, newE, new_y


class XEyTransformerLayer(nn.Module):
    def __init__(self, dx, de, dy, n_head, dim_ffX=256, dim_ffE=128, dim_ffy=128,
                 dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head)

        self.linX1 = nn.Linear(dx, dim_ffX)
        self.linX2 = nn.Linear(dim_ffX, dx)
        self.normX1 = nn.LayerNorm(dx, eps=layer_norm_eps)
        self.normX2 = nn.LayerNorm(dx, eps=layer_norm_eps)
        self.dropoutX1 = nn.Dropout(dropout)
        self.dropoutX2 = nn.Dropout(dropout)
        self.dropoutX3 = nn.Dropout(dropout)

        self.linE1 = nn.Linear(de, dim_ffE)
        self.linE2 = nn.Linear(dim_ffE, de)
        self.normE1 = nn.LayerNorm(de, eps=layer_norm_eps)
        self.normE2 = nn.LayerNorm(de, eps=layer_norm_eps)
        self.dropoutE1 = nn.Dropout(dropout)
        self.dropoutE2 = nn.Dropout(dropout)
        self.dropoutE3 = nn.Dropout(dropout)

        self.lin_y1 = nn.Linear(dy, dim_ffy)
        self.lin_y2 = nn.Linear(dim_ffy, dy)
        self.norm_y1 = nn.LayerNorm(dy, eps=layer_norm_eps)
        self.norm_y2 = nn.LayerNorm(dy, eps=layer_norm_eps)
        self.dropout_y1 = nn.Dropout(dropout)
        self.dropout_y2 = nn.Dropout(dropout)
        self.dropout_y3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, X, E, y, node_mask):
        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)
        X = self.normX1(X + self.dropoutX1(newX))
        E = self.normE1(E + self.dropoutE1(newE))
        y = self.norm_y1(y + self.dropout_y1(new_y))

        ffX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        X = self.normX2(X + self.dropoutX3(ffX))
        ffE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        E = self.normE2(E + self.dropoutE3(ffE))
        ffy = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        y = self.norm_y2(y + self.dropout_y3(ffy))
        return X, E, y


class GraphTransformer(nn.Module):
    def __init__(self, n_layers, input_dims, hidden_mlp_dims, hidden_dims,
                 output_dims, act_fn_in, act_fn_out):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]), act_fn_in)
        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]), act_fn_in)
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"], hidden_mlp_dims["y"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]), act_fn_in)

        self.tf_layers = nn.ModuleList([
            XEyTransformerLayer(dx=hidden_dims["dx"], de=hidden_dims["de"],
                                dy=hidden_dims["dy"], n_head=hidden_dims["n_head"],
                                dim_ffX=hidden_dims["dim_ffX"],
                                dim_ffE=hidden_dims["dim_ffE"],
                                dim_ffy=hidden_dims["dim_ffy"])
            for _ in range(n_layers)])

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]), act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], output_dims["X"]))
        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]), act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]))
        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]), act_fn_out,
            nn.Linear(hidden_mlp_dims["y"], output_dims["y"]))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n, device=X.device)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        X, E, y = self.mlp_in_X(X), new_E, self.mlp_in_y(y)
        X, E = mask_graph(X, E, node_mask)

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X) + X_to_out
        E = self.mlp_out_E(E) + E_to_out
        y = self.mlp_out_y(y) + y_to_out

        E = E * diag_mask
        E = 0.5 * (E + torch.transpose(E, 1, 2))
        X, E = mask_graph(X, E, node_mask)
        return X, E, y


# ---------------------------------------------------------------------------
# Time conditioning + Stage-1 velocity model wrapper.
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding of a scalar t in [0,1] -> (bs, dim)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        t = t.reshape(-1).float()                       # (bs,)
        half = self.dim // 2
        denom = max(half - 1, 1)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / denom)
        args = t[:, None] * freqs[None, :]              # (bs, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb                                      # (bs, dim)


class TimeConditionedGraphTransformer(nn.Module):
    def __init__(self, k_X, k_E, n_layers=5, dx=256, de=64, dy=128, n_head=8,
                 dim_ffX=256, dim_ffE=128, dim_ffy=256, time_dim=128,
                 hidden_mlp_X=256, hidden_mlp_E=128, hidden_mlp_y=256,
                 max_n_nodes=None, extra_features=None, rrwp_steps=12,
                 cond_dim=0, cond_emb=64):
        super().__init__()
        # Recorded so checkpoints rebuild the exact architecture even when
        # defaults are overridden (n_layers, dims, ...). See checkpoint._model_kwargs.
        self.init_kwargs = dict(
            k_X=k_X, k_E=k_E, n_layers=n_layers, dx=dx, de=de, dy=dy,
            n_head=n_head, dim_ffX=dim_ffX, dim_ffE=dim_ffE, dim_ffy=dim_ffy,
            time_dim=time_dim, hidden_mlp_X=hidden_mlp_X, hidden_mlp_E=hidden_mlp_E,
            hidden_mlp_y=hidden_mlp_y, max_n_nodes=max_n_nodes,
            extra_features=extra_features, rrwp_steps=rrwp_steps,
            cond_dim=cond_dim, cond_emb=cond_emb)
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.max_n_nodes = max_n_nodes
        self.extra_features = extra_features
        self.rrwp_steps = rrwp_steps

        # CFG conditioning: embed the (z-scored) property vector and concat to the
        # global y; cond=None or dropped rows use a learned null token. cond_dim=0
        # disables it entirely.
        self.cond_dim = cond_dim
        y_cond = 0
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, cond_emb), nn.ReLU(), nn.Linear(cond_emb, cond_emb))
            self.cond_null = nn.Parameter(torch.randn(cond_emb))
            self.register_buffer("cond_mean", torch.zeros(cond_dim))
            self.register_buffer("cond_std", torch.ones(cond_dim))
            y_cond = cond_emb

        if extra_features == "rrwp":
            assert max_n_nodes is not None, "extra_features='rrwp' requires max_n_nodes"
            ed = extra_feature_dims(rrwp_steps)
            input_dims = {"X": k_X + ed["X"], "E": k_E + ed["E"],
                          "y": time_dim + ed["y"] + y_cond}
        elif extra_features is None:
            y_extra = 1 if max_n_nodes is not None else 0
            input_dims = {"X": k_X, "E": k_E, "y": time_dim + y_extra + y_cond}
        else:
            raise ValueError(f"unknown extra_features {extra_features!r}")
        hidden_mlp_dims = {"X": hidden_mlp_X, "E": hidden_mlp_E, "y": hidden_mlp_y}
        hidden_dims = {"dx": dx, "de": de, "dy": dy, "n_head": n_head,
                       "dim_ffX": dim_ffX, "dim_ffE": dim_ffE, "dim_ffy": dim_ffy}
        output_dims = {"X": k_X, "E": k_E, "y": 1}
        self.net = GraphTransformer(n_layers, input_dims, hidden_mlp_dims,
                                    hidden_dims, output_dims, nn.ReLU(), nn.ReLU())

    @torch.no_grad()
    def set_cond_stats(self, mean, std):
        # Z-score stats for the conditioning properties (from the training split).
        self.cond_mean.copy_(torch.as_tensor(mean, dtype=self.cond_mean.dtype))
        self.cond_std.copy_(torch.as_tensor(std, dtype=self.cond_std.dtype).clamp_min(1e-6))

    def _cond_embedding(self, cond, drop, bs, device):
        # (bs, cond_emb): null token where cond is None or a row is dropped.
        null = self.cond_null.expand(bs, -1)
        if cond is None:
            return null
        emb = self.cond_mlp((cond - self.cond_mean) / self.cond_std)
        if drop is not None:
            emb = torch.where(drop.view(bs, 1), null, emb)
        return emb

    def forward(self, X, E, t, node_mask, cond=None, drop=None):
        y = self.time_emb(t)                            # (bs, time_dim)
        if self.extra_features == "rrwp":
            eX, eE, y_cyc = extra_graph_features(X, E, node_mask, self.rrwp_steps)
            n_norm = node_mask.sum(dim=1, keepdim=True).float() / self.max_n_nodes
            X = torch.cat([X, eX], dim=-1)
            E = torch.cat([E, eE], dim=-1)
            y = torch.cat([y, n_norm, y_cyc], dim=-1)   # time + n + cycles
        elif self.max_n_nodes is not None:
            n_norm = node_mask.sum(dim=1, keepdim=True).float() / self.max_n_nodes
            y = torch.cat([y, n_norm], dim=-1)          # (bs, time_dim + 1)
        if self.cond_dim > 0:
            y = torch.cat([y, self._cond_embedding(cond, drop, X.shape[0], X.device)], dim=-1)
        outX, outE, _ = self.net(X, E, y, node_mask)    # velocities or logits
        return outX, outE

import torch

# Ported directly from DeFoG's 'rrwp' extra-features mode (RRWP + cycle counts).
# Spectral (Laplacian eigen) features are omitted, as DeFoG does, for compute cost.

# Graph-level cycle features: 3-, 4-, 5-, 6-cycles.
Y_CYCLE_DIMS = 4


def extra_feature_dims(k):
    # 'rrwp' mode: +k to X and E; +1 (node count) +cycles to y.
    return {"X": k, "E": k, "y": 1 + Y_CYCLE_DIMS}


def soft_adjacency(E):
    # (bs,n,n,k_E) -> (bs,n,n): sum non-"none" bond channels. clamp>=0 keeps RRWP
    # well-posed for fm_graph's Gaussian E (negative entries blow up D^-1 A powers);
    # no-op for one-hot/simplex E (defog).
    return E[..., 1:].sum(-1).clamp(min=0.0)


def rrwp_features(A, k):
    # [I, M, ..., M^(k-1)] with M = D^-1 A; node feats = diagonal.
    bs, n, _ = A.shape
    deg = A.sum(-1)
    inv = torch.zeros_like(deg)
    nz = deg != 0
    inv[nz] = 1.0 / deg[nz]
    M = torch.diag_embed(inv) @ A
    eye = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0).expand(bs, -1, -1)
    powers = [eye]
    for _ in range(k - 1):
        powers.append(powers[-1] @ M)
    edge = torch.stack(powers, dim=-1)               # (bs,n,n,k)
    node = torch.diagonal(edge, dim1=1, dim2=2).transpose(1, 2)  # (bs,n,k)
    return node, edge


def _batch_trace(X):
    return torch.diagonal(X, dim1=-2, dim2=-1).sum(-1)


def _batch_diagonal(X):
    return torch.diagonal(X, dim1=-2, dim2=-1)


def _cycle_counts(A):
    # Graph-level 3-6 cycle counts (bs,4); port of DeFoG KNodeCycles (y only).
    A = A.float()
    d = A.sum(-1)
    k2 = A @ A
    k3 = k2 @ A
    k4 = k3 @ A
    k5 = k4 @ A
    k6 = k5 @ A

    c3 = _batch_diagonal(k3)
    k3y = (c3.sum(-1) / 6).unsqueeze(-1)

    diag_a4 = _batch_diagonal(k4)
    c4 = diag_a4 - d * (d - 1) - (A @ d.unsqueeze(-1)).sum(-1)
    k4y = (c4.sum(-1) / 8).unsqueeze(-1)

    diag_a5 = _batch_diagonal(k5)
    tri = _batch_diagonal(k3)
    c5 = diag_a5 - 2 * tri * d - (A @ tri.unsqueeze(-1)).sum(-1) + tri
    k5y = (c5.sum(-1) / 10).unsqueeze(-1)

    t1 = _batch_trace(k6)
    t2 = _batch_trace(k3 ** 2)
    t3 = (A * k2.pow(2)).sum(dim=[-2, -1])
    t4 = (_batch_diagonal(k2) * _batch_diagonal(k4)).sum(-1)
    t5 = _batch_trace(k4)
    t6 = _batch_trace(k3)
    t7 = _batch_diagonal(k2).pow(3).sum(-1)
    t8 = k3.sum(dim=[-2, -1])
    t9 = _batch_diagonal(k2).pow(2).sum(-1)
    t10 = _batch_trace(k2)
    c6 = (t1 - 3 * t2 + 9 * t3 - 6 * t4 + 6 * t5
          - 4 * t6 + 4 * t7 + 3 * t8 - 12 * t9 + 4 * t10)
    k6y = (c6 / 12).unsqueeze(-1)

    return torch.cat([k3y, k4y, k5y, k6y], dim=-1)   # (bs,4)


def cycle_features(A):
    # /10-scaled, clamped (matches DeFoG NodeCycleFeatures).
    return (_cycle_counts(A) / 10).clamp(max=1.0)


def extra_graph_features(X, E, node_mask, k):
    # RRWP node/edge + graph cycle feats; backbone prepends the node count to y.
    A = soft_adjacency(E)
    node, edge = rrwp_features(A, k)
    y_cyc = cycle_features(A)
    return node, edge, y_cyc

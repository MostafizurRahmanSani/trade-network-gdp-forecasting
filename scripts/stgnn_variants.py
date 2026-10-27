"""Configurable STGNN used for the selected architecture and for the one-factor ablation.

The variant is given by a spec string in the model name, e.g.  STGNNX-l1-evec-d2  ->
  l: message-passing layers | e: edge handling (mlp|linear|attn|concat|uniform|vec|vecmlp) | c: node update (gru|add|concat) | a: aggregation (mean|sum|max)
  g: temporal GRU layers | t: read-out (gru|last|mean) | k: skip connection from the raw last-year features (0/1) | d: dropout (x10) | n: L2 normalisation (0/1)
The selected STGNN is STGNNX-l1-evec-d2 (one layer, per-channel edge gate, dropout 0.2).
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import softmax as edge_softmax
from gdp_stgnn.models.spatiotemporal import _dense_adj

DEFAULT = dict(l=2, e="mlp", c="gru", a="mean", g=1, t="gru", k=0, d=0, n=1)


def parse(name):
    spec = dict(DEFAULT)
    for tok in name.split("-")[1:]:
        key, val = tok[0], tok[1:]
        spec[key] = int(val) if key in "lgkdn" else val
    return spec


class MPX(nn.Module):
    def __init__(self, h, edge_dim, e, c, a):
        super().__init__()
        self.e, self.c, self.a = e, c, a
        self.msg = nn.Sequential(nn.Linear(h + (edge_dim if e == "concat" else 0), h), nn.ReLU())
        if e.startswith("mlp"):
            x = int(e[3:]) if e[3:] else h // 2
            self.ew = nn.Sequential(nn.Linear(edge_dim, x), nn.ReLU(), nn.Linear(x, 1), nn.Sigmoid())
        elif e == "vec":
            self.ew = nn.Sequential(nn.Linear(edge_dim, h), nn.Sigmoid())
        elif e == "vecmlp":
            self.ew = nn.Sequential(nn.Linear(edge_dim, h), nn.ReLU(), nn.Linear(h, h), nn.Sigmoid())
        elif e == "linear":
            self.ew = nn.Sequential(nn.Linear(edge_dim, 1), nn.Sigmoid())
        elif e == "attn":
            self.att_s, self.att_d, self.att_e = nn.Linear(h, 1, bias=False), nn.Linear(h, 1, bias=False), nn.Linear(edge_dim, 1, bias=False)
        self.upd = nn.GRUCell(h, h) if c == "gru" else nn.Sequential(nn.Linear(h if c == "add" else 2 * h, h), nn.ReLU())

    def forward(self, x, ei, ea):
        src, dst = ei; N = x.size(0)
        if self.e == "concat":
            m = self.msg(torch.cat([x[src], ea], -1))
        else:
            m = self.msg(x[src])
        if self.e.startswith("mlp") or self.e in ("linear", "vec", "vecmlp"):
            m = m * self.ew(ea)
        elif self.e == "attn":
            alpha = edge_softmax(F.leaky_relu(self.att_s(x[src]) + self.att_d(x[dst]) + self.att_e(ea), 0.2).squeeze(-1), dst, num_nodes=N)
            m = m * alpha.unsqueeze(-1)
        out = torch.zeros(N, m.size(1), device=m.device)
        if self.e == "attn" or self.a == "sum":
            out.index_add_(0, dst, m)
        elif self.a == "mean":
            out.index_add_(0, dst, m); out = out / torch.bincount(dst, minlength=N).clamp(min=1).unsqueeze(1)
        else:
            out.scatter_reduce_(0, dst.unsqueeze(1).expand_as(m), m, reduce="amax", include_self=False)
        if self.c == "gru":
            return self.upd(out, x)
        return self.upd(x + out if self.c == "add" else torch.cat([x, out], -1))


class STGNNX(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim, spec):
        super().__init__()
        self.s = spec; h = hidden_dim
        self.pre = nn.Linear(in_dim, h)
        self.layers = nn.ModuleList([MPX(h, edge_dim, spec["e"], spec["c"], spec["a"]) for _ in range(spec["l"])])
        if spec["e"] == "uniform":
            self.layers = nn.ModuleList([MPX(h, edge_dim, "none", spec["c"], spec["a"]) for _ in range(spec["l"])])
        self.gru = nn.GRU(h, h, num_layers=spec["g"], batch_first=False) if spec["t"] == "gru" else None
        self.skip = nn.Linear(in_dim, h) if spec["k"] else None
        self.drop = nn.Dropout(spec["d"] / 10)
        self.post = nn.Linear(h, h); self.reg = nn.Linear(h, 1)

    def forward(self, xs, edge_indices, edge_attrs=None):
        outs = []
        for t in range(xs.shape[0]):
            x = F.relu(self.pre(xs[t]))
            for L in self.layers:
                x = L(x, edge_indices[t], edge_attrs[t] if edge_attrs is not None else None)
            if self.s["n"]:
                x = F.normalize(x, p=2, dim=1)
            outs.append(x)
        H = torch.stack(outs, 0)
        h = self.gru(H)[0][-1] if self.gru is not None else (H[-1] if self.s["t"] == "last" else H.mean(0))
        if self.skip is not None:
            h = h + F.relu(self.skip(xs[-1]))
        return self.reg(F.relu(self.post(self.drop(h))))


class TransformerBaseline(nn.Module):
    """Per-country Transformer encoder over the 6-year window (learned positional embedding, last-token read-out)."""

    def __init__(self, hidden_dim=32, horizon=1, input_dim=4, heads=2, layers=1, window=6):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(window, hidden_dim))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, heads, dim_feedforward=2 * hidden_dim, dropout=0.0, batch_first=True), layers)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        return self.fc(self.enc(self.inp(x) + self.pos)[:, -1, :])


class _GRN(nn.Module):
    """Gated residual network (Lim et al., 2021): LayerNorm(skip(x) + GLU(W2 ELU(W1 x)))."""

    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.fc1, self.fc2, self.gate = nn.Linear(d_in, d_hid), nn.Linear(d_hid, d_out), nn.Linear(d_out, 2 * d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity(); self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        return self.norm(self.skip(x) + F.glu(self.gate(self.fc2(F.elu(self.fc1(x))))))


class TFTBaseline(nn.Module):
    """Compact Temporal Fusion Transformer (per country, no static covariates): per-feature embeddings with a variable-selection network,
    LSTM encoder with gated skip, gated interpretable self-attention over the window, gated position-wise GRN, output from the last step."""

    def __init__(self, hidden_dim=32, horizon=1, input_dim=4, heads=2):
        super().__init__()
        h = hidden_dim; self.F = input_dim
        self.emb = nn.ModuleList([nn.Linear(1, h) for _ in range(input_dim)])
        self.vsn = _GRN(input_dim * h, h, input_dim)
        self.lstm = nn.LSTM(h, h, batch_first=True); self.g1, self.n1 = nn.Linear(h, 2 * h), nn.LayerNorm(h)
        self.attn = nn.MultiheadAttention(h, heads, batch_first=True); self.g2, self.n2 = nn.Linear(h, 2 * h), nn.LayerNorm(h)
        self.pw = _GRN(h, h, h); self.g3, self.n3 = nn.Linear(h, 2 * h), nn.LayerNorm(h)
        self.out = nn.Linear(h, horizon)

    def forward(self, x):
        e = torch.stack([self.emb[j](x[..., j:j + 1]) for j in range(self.F)], 2)
        w = torch.softmax(self.vsn(e.flatten(2)), -1).unsqueeze(-1)
        v = (w * e).sum(2)
        enc = self.n1(v + F.glu(self.g1(self.lstm(v)[0])))
        att = self.n2(enc + F.glu(self.g2(self.attn(enc, enc, enc, need_weights=False)[0])))
        dec = self.n3(att + F.glu(self.g3(self.pw(att))))
        return self.out(dec[:, -1])


class MTGNN(nn.Module):
    """Compact MTGNN (Wu et al., 2020): graph-learning layer (antisymmetric node-embedding adjacency, top-k sparsified), gated dilated temporal convolutions,
    mix-hop propagation with retained input (beta), residual connections. use_trade=True additionally propagates over the row-normalised trade graph
    of the last input year (MTGNN's 'predefined adjacency' option); use_trade=False is the learned-graph-only MTGNN."""

    def __init__(self, in_dim, hidden_dim, edge_dim=13, n_nodes=182, emb=16, topk=20, hops=2, beta=0.05, alpha=3.0, use_trade=False):
        super().__init__()
        h = hidden_dim; self.topk, self.hops, self.beta, self.alpha, self.use_trade = topk, hops, beta, alpha, use_trade
        self.start = nn.Linear(in_dim, h)
        self.E1, self.E2 = nn.Parameter(torch.randn(n_nodes, emb) * 0.1), nn.Parameter(torch.randn(n_nodes, emb) * 0.1)
        self.l1, self.l2 = nn.Linear(emb, emb), nn.Linear(emb, emb)
        self.tf = nn.ModuleList([nn.Conv1d(h, h, 2, dilation=d) for d in (1, 2)]); self.tg = nn.ModuleList([nn.Conv1d(h, h, 2, dilation=d) for d in (1, 2)])
        ng = 2 if use_trade else 1
        self.mix = nn.ModuleList([nn.Linear(h * (1 + hops * ng), h) for _ in range(2)]); self.norm = nn.ModuleList([nn.LayerNorm(h) for _ in range(2)])
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def adjacency(self, n):
        m1, m2 = torch.tanh(self.alpha * self.l1(self.E1[:n])), torch.tanh(self.alpha * self.l2(self.E2[:n]))
        a = torch.relu(torch.tanh(self.alpha * (m1 @ m2.T - m2 @ m1.T)))
        k = min(self.topk, n); idx = a.topk(k, dim=1).indices
        a = a * torch.zeros_like(a).scatter_(1, idx, 1.0)
        return a / a.sum(1, keepdim=True).clamp(min=1e-8)

    def forward(self, xs, edge_indices, edge_attrs=None):
        T, N, _ = xs.shape
        x = torch.relu(self.start(xs)).permute(1, 2, 0)
        As = [self.adjacency(N)]
        if self.use_trade: As.append(_dense_adj(edge_indices[-1], None, N, xs.device))
        for i in range(2):
            z = torch.tanh(self.tf[i](x)) * torch.sigmoid(self.tg[i](x))
            z = z.permute(0, 2, 1)
            outs = [z]
            for A in As:
                hk = z
                for _ in range(self.hops):
                    hk = self.beta * z + (1 - self.beta) * torch.einsum("ij,jtc->itc", A, hk); outs.append(hk)
            z = self.mix[i](torch.cat(outs, -1))
            x = self.norm[i](z + x[:, :, -z.size(1):].permute(0, 2, 1)).permute(0, 2, 1)
        return self.out(x[:, :, -1])


class BonnigGAT(nn.Module):
    """Spatial-only model of the public 'GNNs for GDP' repository (Bonnig 2021): two GATConv layers (hidden size h, one head, ReLU and dropout 0.5 in between),
    a linear read-out; use_edges=True passes the edge features to the attention (their model.py), False ignores them (their baseline.py).
    Differences from the original: next-year growth is the target, so the final ReLU on the output is dropped; one yearly snapshot (last input year) per sample."""

    def __init__(self, in_dim, hidden_dim, edge_dim=13, use_edges=False, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.use_edges, self.p = use_edges, dropout
        kw = dict(edge_dim=edge_dim) if use_edges else {}
        self.c1, self.c2 = GATConv(in_dim, hidden_dim, **kw), GATConv(hidden_dim, hidden_dim, **kw)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr=None):
        ea = edge_attr if self.use_edges else None
        x = F.dropout(F.relu(self.c1(x, edge_index, edge_attr=ea)), self.p, self.training)
        return self.lin(self.c2(x, edge_index, edge_attr=ea))

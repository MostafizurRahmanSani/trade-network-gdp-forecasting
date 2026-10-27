"""Spatio-temporal models: per-year graph encoder followed by a temporal GRU."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

from gdp_stgnn.models.message_passing import MessagePassingLayer


class STGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim=10):
        super().__init__()
        self.pre = nn.Linear(in_dim, hidden_dim)

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.post = nn.Linear(hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, xs, edge_indices, edge_attrs=None):
        T, N, _ = xs.shape
        spatial_out = []

        for t in range(T):
            xt = F.relu(self.pre(xs[t]))
            edge_index = edge_indices[t]

            edge_weight = None
            if edge_attrs is not None:
                ea = edge_attrs[t]
                edge_weight = self.edge_proj(ea).squeeze(-1)
                edge_weight = edge_weight / (edge_weight.max() + 1e-8)
                edge_weight = edge_weight.clamp(min=1e-8)

            h1 = F.relu(self.conv1(xt, edge_index, edge_weight))
            xt = torch.nan_to_num(xt + h1)

            h2 = F.relu(self.conv2(xt, edge_index, edge_weight))
            xt = torch.nan_to_num(xt + h2)

            spatial_out.append(xt)

        h = torch.stack(spatial_out, dim=0)
        h, _ = self.temporal_gru(h)
        h = h[-1]

        return self.regressor(F.relu(self.post(h)))


class STGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim=10, heads=4):
        super().__init__()
        self.pre = nn.Linear(in_dim, hidden_dim)
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads,
                             concat=False, dropout=0.1)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads,
                             concat=False, dropout=0.1)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.post = nn.Linear(hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, xs, edge_indices, edge_attrs=None):
        T, N, _ = xs.shape
        spatial_out = []

        for t in range(T):
            xt = F.relu(self.pre(xs[t]))
            edge_index = edge_indices[t]

            edge_weight = None
            if edge_attrs is not None:
                ea = edge_attrs[t]
                edge_weight = self.edge_proj(ea).squeeze(-1)
                edge_weight = edge_weight.clamp(min=1e-6)

            h1 = F.relu(self.conv1(xt, edge_index, edge_weight))
            xt = torch.nan_to_num(xt + h1)

            h2 = F.relu(self.conv2(xt, edge_index, edge_weight))
            xt = torch.nan_to_num(xt + h2)

            spatial_out.append(xt)

        h = torch.stack(spatial_out, dim=0)
        h, _ = self.temporal_gru(h)
        h_out = h[-1]

        return self.regressor(F.relu(self.post(h_out)))


def _dense_adj(edge_index, edge_weight, n, device):
    """Row-normalised dense adjacency (receiver rows) from an edge list; self-loops added."""
    A = torch.zeros(n, n, device=device)
    w = torch.ones(edge_index.size(1), device=device) if edge_weight is None else edge_weight
    A.index_put_((edge_index[1], edge_index[0]), w, accumulate=True)
    A = A + torch.eye(n, device=device)
    return A / A.sum(1, keepdim=True).clamp(min=1e-8)


class _EdgeWeights(nn.Module):
    def __init__(self, edge_dim, hidden_dim, uniform=False):
        super().__init__()
        self.uniform = uniform
        self.proj = nn.Sequential(nn.Linear(edge_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())

    def forward(self, ei, ea):
        if self.uniform or ea is None:
            return None
        return self.proj(ea).squeeze(-1)


class DCRNN(nn.Module):
    """Diffusion-convolution GRU (Li et al., 2018): forward and reverse random-walk diffusion, K=2 hops."""

    def __init__(self, in_dim, hidden_dim, edge_dim=10, K=2, uniform=False):
        super().__init__()
        self.K, self.h = K, hidden_dim
        self.ew = _EdgeWeights(edge_dim, hidden_dim, uniform)
        n_in = in_dim + hidden_dim
        self.W_gate = nn.Linear(n_in * (2 * K + 1), 2 * hidden_dim)
        self.W_cand = nn.Linear(n_in * (2 * K + 1), hidden_dim)
        self.regressor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def _diffuse(self, x, Af, Ab):
        outs, xf, xb = [x], x, x
        for _ in range(self.K):
            xf, xb = Af @ xf, Ab @ xb
            outs += [xf, xb]
        return torch.cat(outs, -1)

    def forward(self, xs, edge_indices, edge_attrs=None):
        T, N, _ = xs.shape; h = torch.zeros(N, self.h, device=xs.device)
        for t in range(T):
            ea = edge_attrs[t] if edge_attrs is not None else None
            w = self.ew(edge_indices[t], ea)
            Af = _dense_adj(edge_indices[t], w, N, xs.device); Ab = _dense_adj(edge_indices[t].flip(0), w, N, xs.device)
            z = torch.cat([xs[t], h], -1)
            r, u = torch.sigmoid(self.W_gate(self._diffuse(z, Af, Ab))).chunk(2, -1)
            c = torch.tanh(self.W_cand(self._diffuse(torch.cat([xs[t], r * h], -1), Af, Ab)))
            h = u * h + (1 - u) * c
        return self.regressor(h)


class GraphWaveNet(nn.Module):
    """Graph WaveNet / MTGNN-style (Wu et al., 2019/2020): dilated causal temporal convs + graph conv on the trade
    graph and a learned self-adaptive adjacency softmax(relu(E1 E2^T)). use_trade=False keeps only the adaptive graph."""

    def __init__(self, in_dim, hidden_dim, edge_dim=10, n_nodes=164, emb=8, use_trade=True, uniform=False):
        super().__init__()
        self.use_trade = use_trade
        self.ew = _EdgeWeights(edge_dim, hidden_dim, uniform)
        self.E1, self.E2 = nn.Parameter(torch.randn(n_nodes, emb) * 0.1), nn.Parameter(torch.randn(n_nodes, emb) * 0.1)
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.tconv1 = nn.Conv1d(hidden_dim, hidden_dim, 2, dilation=1)
        self.tconv2 = nn.Conv1d(hidden_dim, hidden_dim, 2, dilation=2)
        self.gc = nn.Linear(hidden_dim * (3 if use_trade else 2), hidden_dim)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, xs, edge_indices, edge_attrs=None):
        T, N, _ = xs.shape
        x = torch.relu(self.inp(xs)).permute(1, 2, 0)
        x = torch.relu(self.tconv1(x)); x = torch.relu(self.tconv2(x))
        h = x[:, :, -1]
        Aad = torch.softmax(torch.relu(self.E1[:N] @ self.E2[:N].T), -1)
        parts = [h, Aad @ h]
        if self.use_trade:
            w = self.ew(edge_indices[-1], edge_attrs[-1] if edge_attrs is not None else None)
            parts.append(_dense_adj(edge_indices[-1], w, N, xs.device) @ h)
        return self.out(torch.relu(self.gc(torch.cat(parts, -1))) + h)



"""Spatial-only baselines operating on a single yearly trade graph."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

from gdp_stgnn.models.message_passing import MessagePassingLayer


class GNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        edge_dim=10,
        agg="mean",
        comb="gru",
        norm=True
    ):
        super().__init__()

        self.comb = comb

        self.pre = nn.Linear(in_dim, hidden_dim)

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.layer1 = MessagePassingLayer(
            hidden_dim, hidden_dim, aggregation=agg,
            combination=comb, normalize=False
        )
        self.layer2 = MessagePassingLayer(
            hidden_dim, hidden_dim, aggregation=agg,
            combination=comb, normalize=True
        )

        self.post = nn.Linear(hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.pre(x))

        edge_weight = None
        if edge_attr is not None:
            edge_weight = self.edge_proj(edge_attr).squeeze(-1)

        h1 = self.layer1(x, edge_index, edge_weight)
        x  = h1 if self.comb == "gru" else x + h1

        h2 = self.layer2(x, edge_index, edge_weight)
        x  = h2 if self.comb == "gru" else x + h2

        x = F.relu(self.post(x))
        return self.regressor(x)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim=10):
        super().__init__()

        self.pre  = nn.Linear(in_dim, hidden_dim)

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.post       = nn.Linear(hidden_dim, hidden_dim)
        self.regressor  = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.pre(x))

        edge_weight = None
        if edge_attr is not None:
            edge_weight = self.edge_proj(edge_attr).squeeze(-1)
            edge_weight = edge_weight / (edge_weight.max() + 1e-8)
            edge_weight = edge_weight.clamp(min=1e-8)

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = torch.nan_to_num(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = torch.nan_to_num(x)

        x = F.relu(self.post(x))
        return self.regressor(x)


class GAT(nn.Module):
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
        self.post = nn.Linear(hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.pre(x))

        edge_weight = None
        if edge_attr is not None:
            edge_weight = self.edge_proj(edge_attr).squeeze(-1)
            edge_weight = edge_weight.clamp(min=1e-6)

        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        x  = torch.nan_to_num(x + h1)
        h2 = F.relu(self.conv2(x, edge_index, edge_weight))
        x  = torch.nan_to_num(x + h2)

        return self.regressor(F.relu(self.post(x)))

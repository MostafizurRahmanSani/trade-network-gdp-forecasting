"""Custom message-passing layer shared by GNN and STGNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassingLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        aggregation,
        combination,
        normalize=False
    ):
        super().__init__()

        self.aggregation = aggregation
        self.combination = combination
        self.normalize = normalize

        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )

        if combination == "gru":
            self.update_fn = nn.GRUCell(out_dim, out_dim)
        elif combination == "concat":
            self.update_fn = nn.Sequential(
                nn.Linear(in_dim+out_dim, out_dim),
                nn.ReLU()
            )
        elif combination == "add":
            self.update_fn = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU()
            )
        else:
            raise ValueError("Invalid Combination Type!")

    def aggregate(self, src, msg, num_nodes):
        out = torch.zeros(num_nodes, msg.size(1), device=msg.device)

        if self.aggregation == "sum":
            out.index_add_(0, src, msg)
        elif self.aggregation == "mean":
            out.index_add_(0, src, msg)
            deg = torch.bincount(src, minlength=num_nodes).clamp(min=1).unsqueeze(1)
            out = out / deg
        elif self.aggregation == "max":
            out.scatter_reduce_(0, src.unsqueeze(1).expand_as(msg), msg, reduce="amax")
        else:
            raise ValueError("Invalid Aggregation!")

        return out

    def forward(self, x, edge_index, edge_weight):
        src_idx, dst_idx = edge_index
        num_nodes = x.size(0)

        src_feat = x[src_idx]
        msg = self.msg_mlp(src_feat)

        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(1)

        agg = self.aggregate(dst_idx, msg, num_nodes)

        if self.combination == "gru":
            out = self.update_fn(agg, x)
        elif self.combination == "concat":
            out = self.update_fn(torch.cat([x, agg], dim=1))
        elif self.combination == "add":
            out = self.update_fn(x + agg)

        if self.normalize:
            out = F.normalize(out, p=2, dim=1)

        return out

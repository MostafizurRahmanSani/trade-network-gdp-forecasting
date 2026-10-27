"""Helpers for the yearly graph snapshots (torch_geometric Data objects)."""

import torch


def unpack_graphs(graphs):
    xs = torch.stack([g.x for g in graphs], dim=0)
    edge_indices = [g.edge_index for g in graphs]
    edge_attrs = [g.edge_attr for g in graphs]
    return xs, edge_indices, edge_attrs

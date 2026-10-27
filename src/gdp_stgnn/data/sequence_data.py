"""Per-country sequence datasets for the temporal-only baselines."""

import numpy as np

from gdp_stgnn.data.graph_data import unpack_graphs


def build_gru_dataset_from_stgnn(data, past_window, horizon):
    """Convert ST samples into (num_samples * N, PAST_WINDOW, F) sequences."""
    Xs, Ys = [], []

    for graphs, y in data:
        xs, _, _ = unpack_graphs(graphs)
        y_target = y

        xs = xs.permute(1, 0, 2)

        Xs.append(xs.cpu().numpy())
        Ys.append(y_target.cpu().numpy())

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)

    return X, Y

"""Training loop for the spatial-only baselines (GNN, GCN, GAT)."""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score


def train_gnn(model, name_prefix, hyperparams, data_train, data_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Training for '{name_prefix}' on {device} ---")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["learning_rate"],
        weight_decay=hyperparams.get("weight_decay", 0.0)
    )

    best_mae   = float("inf")
    best_state = None
    losses     = []

    for epoch in range(hyperparams["n_epochs"]):
        model.train()
        train_mse = 0.0
        train_mae = 0.0

        for graph, y in data_train:
            x          = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            edge_attr  = graph.edge_attr.to(device)
            y          = y.to(device)

            optimizer.zero_grad()
            pred = model(x, edge_index, edge_attr)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_mse += loss.item()
            train_mae += F.l1_loss(pred, y).item()

        train_mse /= len(data_train)
        train_mae /= len(data_train)

        if epoch % hyperparams["save_loss_interval"] == 0:
            model.eval()
            val_mse = 0.0
            val_mae = 0.0
            r2s     = []

            with torch.no_grad():
                for graph, y in data_val:
                    x          = graph.x.to(device)
                    edge_index = graph.edge_index.to(device)
                    edge_attr  = graph.edge_attr.to(device)
                    y          = y.to(device)

                    pred     = model(x, edge_index, edge_attr)
                    val_mse += F.mse_loss(pred, y).item()
                    val_mae += F.l1_loss(pred, y).item()
                    r2s.append(
                        r2_score(
                            y.cpu().numpy().reshape(-1),
                            pred.cpu().numpy().reshape(-1)
                        )
                    )

            val_mse /= len(data_val)
            val_mae /= len(data_val)
            val_r2   = np.mean(r2s)

            losses.append((epoch, train_mse, train_mae, val_mse, val_mae))

            if val_mae < best_mae:
                best_mae   = val_mae
                best_state = copy.deepcopy(model.state_dict())

            if epoch % hyperparams["print_interval"] == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Train MSE {train_mse:.4f} | Train MAE {train_mae:.4f} | "
                    f"Val MSE {val_mse:.4f} | Val MAE {val_mae:.4f} | R² {val_r2:.4f}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nBest model restored (Val MAE = {best_mae:.4f})")

    print(f"--- Finished Training for '{name_prefix}' ---")
    return model, losses

"""Training loop for the temporal-only baselines (LSTM, BiLSTM, RNN, GRU)."""

import copy

import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset


def train_lstm(model, name_prefix, hyperparams, X_train, Y_train, X_val, Y_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting LSTM Training for '{name_prefix}' on {device} ---")
    model = model.to(device)

    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t   = torch.FloatTensor(X_val).to(device)
    Y_val_t   = torch.FloatTensor(Y_val).to(device)

    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=hyperparams.get("batch_size", 32), shuffle=True)

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

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = F.mse_loss(pred, batch_Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_mse += loss.item()
            train_mae += F.l1_loss(pred, batch_Y).item()

        train_mse /= len(train_loader)
        train_mae /= len(train_loader)

        if epoch % hyperparams["save_loss_interval"] == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_mse  = F.mse_loss(val_pred, Y_val_t).item()
                val_mae  = F.l1_loss(val_pred, Y_val_t).item()
                val_r2   = r2_score(
                    Y_val_t.cpu().numpy().reshape(-1),
                    val_pred.cpu().numpy().reshape(-1)
                )

            losses.append((epoch, train_mse, train_mae, val_mse, val_mae))

            if val_mae < best_mae:
                best_mae   = val_mae
                best_state = copy.deepcopy(model.state_dict())

            if epoch % hyperparams["print_interval"] == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Train MSE {train_mse:.4f} | "
                    f"Train MAE {train_mae:.4f} | "
                    f"Val MSE {val_mse:.4f} | "
                    f"Val MAE {val_mae:.4f} | "
                    f"R² {val_r2:.4f}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nBest LSTM restored (Val MAE = {best_mae:.4f})")

    print(f"--- Finished LSTM Training for '{name_prefix}' ---")
    return model, losses

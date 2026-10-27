"""Temporal-only baselines operating on per-country feature sequences."""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, hidden_dim=32, horizon=1, input_dim=4, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class BiLSTM(nn.Module):
    def __init__(self, hidden_dim=32, horizon=1, input_dim=4, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim*2, horizon)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        h_forward  = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        out = torch.cat([h_forward, h_backward], dim=-1)
        return self.fc(out)


class RNN(nn.Module):
    def __init__(self, hidden_dim=32, horizon=1, input_dim=4, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'
        )
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class GRU(nn.Module):
    def __init__(self, hidden_dim=32, horizon=1, input_dim=4, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

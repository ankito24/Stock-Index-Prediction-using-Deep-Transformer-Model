"""
train.py
Training loop and evaluation metrics (Section 5.2 / 5.3 of the paper).

Hyper-parameters follow the paper:
- batch size = 16
- loss = MSE
- optimizer = Adam, lr = 0.0001
- epochs = 1000
- dropout = 0.1
- T (encoder input length) = 9, N (decoder input length) = 2
- embedding dimension d = 32
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import build_model


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model_name, X_train, y_train, X_test, y_test,
                 T=9, N=2, epochs=1000, batch_size=16, lr=1e-4,
                 device=None, verbose_every=100):
    """Train a single model and return trained model + loss history."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name, T=T, N=N).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    loss_history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)
        loss_history.append(epoch_loss)

        if verbose_every and (epoch + 1) % verbose_every == 0:
            print(f"[{model_name}] epoch {epoch+1}/{epochs} - train MSE loss: {epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train_t.to(device)).cpu().numpy()
        y_pred_test = model(X_test_t).cpu().numpy()

    return model, loss_history, y_pred_train, y_pred_test


# --------------------------------------------------------------------------
# Evaluation metrics (Eq. 7, 8, 9)
# --------------------------------------------------------------------------
def mae(y_pred, y_true):
    return float(np.mean(np.abs(y_pred - y_true)))


def mse(y_pred, y_true):
    return float(np.mean((y_pred - y_true) ** 2))


def mape(y_pred, y_true, eps=1e-8):
    return float(np.mean(np.abs((y_pred - y_true) / (y_true + eps))) * 100)


def evaluate(y_pred, y_true):
    return {
        "MAE": mae(y_pred, y_true),
        "MSE": mse(y_pred, y_true),
        "MAPE": mape(y_pred, y_true),
    }

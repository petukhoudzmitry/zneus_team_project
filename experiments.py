import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64], activation=nn.ReLU,
                 dropout=0.0, batchnorm=False, skip=False, bottleneck=False):
        super().__init__()
        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            if bottleneck and i == len(hidden_dims)//2:
                b_dim = max(8, int(prev*0.25))
                layers.append(nn.Linear(prev, b_dim))
                if batchnorm: layers.append(nn.BatchNorm1d(b_dim))
                layers.append(activation())
                if dropout > 0: layers.append(nn.Dropout(dropout))
                prev = b_dim
                continue
            layers.append(nn.Linear(prev, h))
            if batchnorm: layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        self.skip = skip
        self.input_proj = nn.Linear(input_dim, prev) if skip else None

    def forward(self, x):
        out = self.net(x)
        if self.skip and self.input_proj is not None:
            skip_out = self.input_proj(x)
            if skip_out.shape[-1] == out.shape[-1]:
                out = out + skip_out
        return self.head(out).squeeze(-1)


def make_dataloaders(X_train, y_train, X_val, y_val, batch_size=64, shuffle=True):
    X_train_t = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)
    X_val_t = torch.tensor(np.asarray(X_val), dtype=torch.float32)
    y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                            batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate_regression(model, loader, device='cpu'):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
            trues.append(yb.numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    mse = mean_squared_error(trues, preds)
    return {'mse': mse, 'rmse': mse ** 0.5}


def run_experiment(X_train, y_train, X_val, y_val, config, device='cpu', verbose=True):
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base_tag = config.get('tag')
    tag = base_tag if base_tag else f"exp_{timestamp}"
    tb_dir = os.path.join("runs", tag)

    writer = SummaryWriter(log_dir=tb_dir)
    try:
        writer.add_text("config", json.dumps(config, indent=2))
    except Exception:
        writer.add_text("config", str(config))

    torch.manual_seed(config.get('seed', 42))
    model = MLP(
        input_dim=X_train.shape[1],
        hidden_dims=config.get('hidden_dims', [128, 64]),
        activation=nn.ReLU if config.get('activation', 'relu') == 'relu' else nn.LeakyReLU,
        dropout=config.get('dropout', 0.0),
        batchnorm=config.get('batchnorm', False),
        skip=config.get('skip', False),
        bottleneck=config.get('bottleneck', False)
    ).to(device)

    criterion = nn.MSELoss()
    opt_name = config.get('optimizer', 'adam').lower()
    if opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3),
                               weight_decay=config.get('weight_decay', 0.0))
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.get('lr', 1e-2),
                              momentum=config.get('momentum', 0.9),
                              weight_decay=config.get('weight_decay', 0.0))

    train_loader, val_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=config.get('batch_size', 64)
    )

    best_val = float('inf')
    history = []
    start_time = time.time()
    for epoch in range(config.get('epochs', 50)):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_metrics = evaluate_regression(model, val_loader, device)

        train_loss = np.mean(losses)
        val_mse = val_metrics['mse']
        val_rmse = val_metrics['rmse']

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mse': val_mse,
            'val_rmse': val_rmse
        })

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={np.mean(losses):.4f}, val_rmse={val_metrics['rmse']:.4f}")
        if val_metrics['mse'] < best_val:
            best_val = val_metrics['mse']
            best_state = model.state_dict()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val_mse", val_mse, epoch)
        writer.add_scalar("Metrics/val_rmse", val_rmse, epoch)

    duration = time.time() - start_time

    os.makedirs("artifacts", exist_ok=True)

    safe_ts = timestamp
    history_path = os.path.join("artifacts", f"history_{tag}_{safe_ts}.pkl")
    model_path = os.path.join("artifacts", f"model_{tag}_{safe_ts}.pt")

    history_df = pd.DataFrame(history)
    try:
        history_df.to_pickle(history_path)
    except Exception as e:
        history_path = os.path.join("artifacts", f"history_{tag}_{safe_ts}.csv")
        history_df.to_csv(history_path, index=False)

    try:
        if best_state is not None:
            torch.save(best_state, model_path)
            model.load_state_dict(best_state)
        else:
            torch.save(model.state_dict(), model_path)
    except Exception as e:
        model_path = None

    try:
        writer.add_text("artifacts", f"history_path: {history_path}\nmodel_path: {model_path}\nduration_s: {duration:.2f}")
        writer.close()
    except Exception:
        pass

    writer.close()

    model.load_state_dict(best_state)
    return model, {'best_val_mse': best_val, 'history': pd.DataFrame(history), 'config': config}


def grid_search(X_train, y_train, X_val, y_val, param_grid, device='cpu'):
    import itertools
    results = []
    keys = list(param_grid.keys())
    for vals in itertools.product(*[param_grid[k] for k in keys]):
        cfg = {k: v for k, v in zip(keys, vals)}
        print("Running:", cfg)
        _, res = run_experiment(X_train, y_train, X_val, y_val, cfg, device=device, verbose=False)
        results.append({'config': cfg, 'best_val_mse': res['best_val_mse']})
    return pd.DataFrame(results)

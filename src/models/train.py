"""
Model Training Module for NASA C-MAPSS RUL Prediction.
Implements and evaluates:
- Model A: Linear / Ridge Regression baseline
- Model B: Random Forest Regressor
- Model C: XGBoost Regressor
- Model D: PyTorch Temporal LSTM
Supports both uncapped RUL and piecewise linear capped RUL (125 cycles).
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.models.evaluate import compute_regression_metrics

# --- PyTorch LSTM Definition ---
class CMAPSSSequenceDataset(Dataset):
    """Sequence dataset for temporal LSTM modeling with past-window lookback."""
    def __init__(self, X: np.ndarray, y: np.ndarray, units: np.ndarray, seq_len: int = 30):
        self.X_seqs = []
        self.y_targets = []
        self.units = []
        
        unique_units = np.unique(units)
        for u in unique_units:
            mask = (units == u)
            X_u = X[mask]
            y_u = y[mask]
            n_samples = len(X_u)
            
            for t in range(n_samples):
                if t + 1 < seq_len:
                    # Pad beginning with repetition of earliest observation
                    pad_len = seq_len - (t + 1)
                    pad = np.repeat(X_u[0:1], pad_len, axis=0)
                    seq = np.vstack([pad, X_u[:t + 1]])
                else:
                    seq = X_u[t + 1 - seq_len:t + 1]
                self.X_seqs.append(seq)
                self.y_targets.append(y_u[t])
                self.units.append(u)

        self.X_seqs = np.array(self.X_seqs, dtype=np.float32)
        self.y_targets = np.array(self.y_targets, dtype=np.float32)

    def __len__(self):
        return len(self.X_seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.X_seqs[idx]), torch.tensor(self.y_targets[idx])


class TurbofanLSTM(nn.Module):
    """2-Layer Recurrent LSTM for temporal sensor trajectory processing."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        # Readout from final sequence step
        final_step = out[:, -1, :]
        return self.fc(final_step).squeeze(-1)


def train_linear_baseline(X_train: np.ndarray, y_train: np.ndarray) -> Ridge:
    """Train Ridge linear regression baseline."""
    model = Ridge(alpha=10.0, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """Train Random Forest regressor."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    """Train XGBoost gradient boosting regressor."""
    model = XGBRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_units: np.ndarray,
    seq_len: int = 30,
    epochs: int = 20,
    batch_size: int = 64
) -> TurbofanLSTM:
    """Train PyTorch LSTM model on historical sequences."""
    dataset = CMAPSSSequenceDataset(X_train, y_train, train_units, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = X_train.shape[1]
    model = TurbofanLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
    return model


def predict_lstm(model: TurbofanLSTM, X: np.ndarray, units: np.ndarray, seq_len: int = 30) -> np.ndarray:
    """Generate predictions from trained LSTM model."""
    dataset = CMAPSSSequenceDataset(X, np.zeros(len(X)), units, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            out = model(X_batch)
            preds.extend(out.cpu().numpy().tolist())
    return np.array(preds)

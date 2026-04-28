"""
Regression Model Wrappers for return forecasting.

XGBoostPriceRegressor, RandomForestPriceRegressor, LSTMPriceRegressor.
All implement fit/predict/evaluate/save/load with MAE/RMSE/MAPE/DirectionalAccuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor as _SKLearnRF
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..utils.logger import get_logger

logger = get_logger(__name__)


def _directional_accuracy(y_true, y_pred, prev_close=None) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    # The prediction spec trains return targets. Keep a price-target fallback
    # so old direct-price artifacts can still be inspected if needed.
    if prev_close is not None and np.nanmedian(np.abs(y_true)) > 2.0:
        prev_close = np.asarray(prev_close, dtype=np.float32)
        actual_dir = np.sign(y_true - prev_close)
        pred_dir = np.sign(y_pred - prev_close)
    else:
        actual_dir = np.sign(y_true)
        pred_dir = np.sign(y_pred)

    valid = actual_dir != 0
    if valid.sum() == 0:
        return 0.5
    return float(np.mean(actual_dir[valid] == pred_dir[valid]))


def _compute_metrics(y_true, y_pred, prev_close=None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if np.nanmedian(np.abs(y_true)) <= 2.0:
        # For return targets, surface absolute percentage-point error.
        mape = float(mae * 100.0)
    else:
        nonzero = y_true != 0
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.sum() > 0 else 0.0
    da = _directional_accuracy(y_true, y_pred, prev_close)
    return {"mae": mae, "rmse": rmse, "mape": mape, "directional_accuracy": da}


class XGBoostPriceRegressor:
    def __init__(self, params=None):
        default = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                   "subsample": 0.8, "colsample_bytree": 0.8, "objective": "reg:squarederror",
                   "random_state": 42, "n_jobs": -1}
        if params:
            default.update(params)
        self.params = default
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model = xgb.XGBRegressor(**self.params)
        fit_kw = {}
        if X_val is not None and y_val is not None:
            fit_kw["eval_set"] = [(X_val, y_val)]
            fit_kw["verbose"] = False
        self.model.fit(X_train, y_train, **fit_kw)

    def predict(self, X):
        return np.asarray(self.model.predict(X), dtype=np.float32)

    def evaluate(self, X, y, prev_close=None):
        return _compute_metrics(y, self.predict(X), prev_close)

    def save(self, path):
        path = str(path)
        if not path.endswith(".json"):
            path += ".json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path):
        path = str(path)
        if not Path(path).exists() and not path.endswith(".json"):
            path += ".json"
        self.model = xgb.XGBRegressor(**self.params)
        self.model.load_model(path)


class RandomForestPriceRegressor:
    def __init__(self, params=None):
        default = {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5,
                   "min_samples_leaf": 2, "bootstrap": True, "oob_score": True,
                   "random_state": 42, "n_jobs": -1}
        if params:
            default.update(params)
        self.params = default
        self.model = None
        self.oob_error_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model = _SKLearnRF(**self.params)
        self.model.fit(X_train, y_train)
        if getattr(self.model, "oob_score_", None) is not None:
            self.oob_error_ = float(1.0 - self.model.oob_score_)

    def predict(self, X):
        return np.asarray(self.model.predict(X), dtype=np.float32)

    def evaluate(self, X, y, prev_close=None):
        return _compute_metrics(y, self.predict(X), prev_close)

    def save(self, path):
        path = str(path)
        if not path.endswith(".joblib"):
            path += ".joblib"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        path = str(path)
        if not Path(path).exists() and not path.endswith(".joblib"):
            path += ".joblib"
        self.model = joblib.load(path)


class _LSTMNet(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(out))).squeeze(-1)


class LSTMPriceRegressor:
    def __init__(self, params=None):
        self.params = {"units": 64, "layers": 2, "dropout": 0.2, "batch_size": 32,
                       "epochs": 80, "learning_rate": 0.001, "sequence_length": 60, "patience": 15}
        if params:
            self.params.update(params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = None
        self.is_fitted = False
        self._input_size = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_train.ndim == 2:
            X_train = X_train[:, np.newaxis, :]
        if X_val is not None and X_val.ndim == 2:
            X_val = X_val[:, np.newaxis, :]

        self._input_size = X_train.shape[2]
        self.network = _LSTMNet(self._input_size, self.params["units"],
                                self.params["layers"], self.params["dropout"]).to(self.device)
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(self.network.parameters(), lr=self.params["learning_rate"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train).to(self.device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.params["batch_size"], shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_v = torch.FloatTensor(X_val).to(self.device)
            y_v = torch.FloatTensor(y_val).to(self.device)
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_v, y_v), batch_size=self.params["batch_size"])

        best_val, patience_ctr, best_state = float("inf"), 0, None
        for epoch in range(self.params["epochs"]):
            self.network.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = criterion(self.network(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                opt.step()
            if val_loader:
                self.network.eval()
                vl = [criterion(self.network(xb), yb).item() for xb, yb in val_loader]
                avg_v = float(np.mean(vl))
                sched.step(avg_v)
                if avg_v < best_val:
                    best_val = avg_v
                    patience_ctr = 0
                    best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
                else:
                    patience_ctr += 1
                if patience_ctr >= self.params["patience"]:
                    break
        if best_state:
            self.network.load_state_dict(best_state)
        self.is_fitted = True

    def predict(self, X):
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        self.network.eval()
        with torch.no_grad():
            return self.network(torch.FloatTensor(X).to(self.device)).cpu().numpy().reshape(-1).astype(np.float32)

    def predict_with_uncertainty(self, X, n_samples: int = 30):
        """Run Monte Carlo dropout and return mean and standard deviation."""
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        tensor = torch.FloatTensor(X).to(self.device)
        preds = []
        self.network.train()
        with torch.no_grad():
            for _ in range(int(n_samples)):
                preds.append(self.network(tensor).cpu().numpy().reshape(-1))
        self.network.eval()
        samples = np.vstack(preds).astype(np.float32)
        return samples.mean(axis=0), samples.std(axis=0)

    def evaluate(self, X, y, prev_close=None):
        return _compute_metrics(y, self.predict(X), prev_close)

    def save(self, path):
        path = str(path)
        if not path.endswith(".pt"):
            path += ".pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.network.state_dict(),
                    "params": self.params, "input_size": self._input_size}, path)

    def load(self, path):
        path = str(path)
        if not Path(path).exists() and not path.endswith(".pt"):
            path += ".pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.params = ckpt.get("params", self.params)
        self._input_size = ckpt["input_size"]
        self.network = _LSTMNet(self._input_size, self.params["units"],
                                self.params["layers"], self.params["dropout"]).to(self.device)
        self.network.load_state_dict(ckpt["model_state_dict"])
        self.is_fitted = True


REGRESSOR_FILE_NAMES = {"xgboost": "model.json", "random_forest": "model.joblib", "lstm": "model.pt"}
REGRESSOR_FACTORIES = {"xgboost": XGBoostPriceRegressor, "random_forest": RandomForestPriceRegressor, "lstm": LSTMPriceRegressor}

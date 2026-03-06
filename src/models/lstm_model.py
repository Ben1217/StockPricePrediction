"""
PyTorch LSTM Model Implementation.
Replaces TensorFlow LSTM with PyTorch for stock price prediction.
Supports MC Dropout for uncertainty estimation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from .base_model import BaseModel
from ..utils.logger import get_logger
from ..utils.config_loader import get_config_value

logger = get_logger(__name__)


class LSTMNetwork(nn.Module):
    """PyTorch LSTM network architecture."""

    def __init__(self, input_size: int, hidden_size: int = 50,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(-1)


class LSTMModel(BaseModel):
    """PyTorch LSTM model for stock prediction with uncertainty estimation."""

    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            'units': 50,
            'layers': 2,
            'dropout': 0.2,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'sequence_length': 60,
            'patience': 15,
        }

        config_params = get_config_value('models.lstm.architecture', {})
        if config_params:
            default_params.update({
                'layers': len(config_params.get('layers', [])),
            })

        if params:
            default_params.update(params)

        super().__init__(name='LSTM', params=default_params)
        self.history = {'train_loss': [], 'val_loss': []}
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build(self, input_shape: Tuple[int, int] = None) -> None:
        """
        Build LSTM model architecture.

        Parameters
        ----------
        input_shape : tuple
            (sequence_length, n_features)
        """
        if input_shape is None:
            input_shape = (self.params['sequence_length'], 1)

        self.model = LSTMNetwork(
            input_size=input_shape[1],
            hidden_size=self.params['units'],
            num_layers=self.params['layers'],
            dropout=self.params['dropout'],
        ).to(self.device)

        logger.info(
            f"Built PyTorch LSTM: {self.params['layers']} layers, "
            f"{self.params['units']} units, device={self.device}"
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """Fit LSTM model."""
        # Ensure 3D input
        if len(X_train.shape) == 2:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            if X_val is not None:
                X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        if self.model is None:
            self.build(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Convert to tensors
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        batch_size = self.params['batch_size']
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.params.get('patience', 15)
        best_epoch = 0

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_v = torch.FloatTensor(X_val).to(self.device)
            y_v = torch.FloatTensor(y_val).to(self.device)
            val_ds = torch.utils.data.TensorDataset(X_v, y_v)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

        for epoch in range(self.params['epochs']):
            # Training
            self.model.train()
            train_losses = []
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            self.history['train_loss'].append(avg_train)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        preds = self.model(X_batch)
                        loss = criterion(preds, y_batch)
                        val_losses.append(loss.item())
                avg_val = np.mean(val_losses)
                self.history['val_loss'].append(avg_val)
                scheduler.step(avg_val)

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    patience_counter = 0
                    best_epoch = epoch + 1
                    # Save best weights
                    self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}, best epoch was {best_epoch}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}: train_loss={avg_train:.6f}, val_loss={avg_val:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}: train_loss={avg_train:.6f}")

        # Always restore best weights (whether early stopped or ran all epochs)
        if hasattr(self, '_best_state') and self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            logger.info(f"Restored best weights from epoch {best_epoch} (val_loss={best_val_loss:.6f})")

        self.is_fitted = True
        logger.info(f"LSTM fitted: {len(self.history['train_loss'])} epochs completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            preds = self.model(X_t).cpu().numpy()
        return preds

    def predict_with_uncertainty(
        self, X: np.ndarray, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MC Dropout prediction for uncertainty estimation.

        Parameters
        ----------
        X : np.ndarray
            Input features (samples, seq_len, features)
        n_samples : int
            Number of forward passes (MC samples)

        Returns
        -------
        tuple
            (mean_prediction, lower_bound, upper_bound) at 95% CI
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))

        X_t = torch.FloatTensor(X).to(self.device)

        # Enable dropout at inference time (MC Dropout)
        self.model.train()  # Keep dropout active
        all_preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds = self.model(X_t).cpu().numpy()
                all_preds.append(preds)

        self.model.eval()
        all_preds = np.array(all_preds)  # (n_samples, n_points)
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)

        lower_95 = mean_pred - 1.96 * std_pred
        upper_95 = mean_pred + 1.96 * std_pred

        return mean_pred, lower_95, upper_95

    def save(self, filepath: str) -> None:
        """Save PyTorch LSTM model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        save_path = filepath if filepath.endswith('.pt') else filepath + '.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'params': self.params,
            'history': self.history,
        }, save_path)
        logger.info(f"LSTM model saved to {save_path}")

    def load(self, filepath: str) -> None:
        """Load PyTorch LSTM model."""
        load_path = filepath
        if not Path(load_path).exists() and not load_path.endswith('.pt'):
            load_path = filepath + '.pt'
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.params = checkpoint.get('params', self.params)
        self.history = checkpoint.get('history', self.history)

        # Rebuild model with loaded params — need to know input_size
        # Try to infer from state_dict
        state = checkpoint['model_state_dict']
        input_size = state['lstm.weight_ih_l0'].shape[1]
        self.build(input_shape=(self.params['sequence_length'], input_size))
        self.model.load_state_dict(state)
        self.is_fitted = True
        logger.info(f"LSTM model loaded from {load_path}")

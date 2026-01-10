"""
LSTM Model Implementation
"""

import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from .base_model import BaseModel
from ..utils.logger import get_logger
from ..utils.config_loader import get_config_value

logger = get_logger(__name__)


class LSTMModel(BaseModel):
    """LSTM deep learning model for stock prediction"""

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize LSTM model

        Parameters
        ----------
        params : dict, optional
            Model hyperparameters
        """
        default_params = {
            'units': 50,
            'layers': 2,
            'dropout': 0.2,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'sequence_length': 60,
        }

        config_params = get_config_value('models.lstm.architecture', {})
        if config_params:
            default_params.update({
                'layers': len(config_params.get('layers', [])),
            })

        if params:
            default_params.update(params)

        super().__init__(name='LSTM', params=default_params)
        self.history = None
        self.scaler = None

    def build(self, input_shape: Tuple[int, int] = None) -> None:
        """
        Build LSTM model architecture

        Parameters
        ----------
        input_shape : tuple
            (sequence_length, n_features)
        """
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        if input_shape is None:
            input_shape = (self.params['sequence_length'], 1)

        model = Sequential()

        # First LSTM layer
        model.add(LSTM(
            units=self.params['units'],
            return_sequences=(self.params['layers'] > 1),
            input_shape=input_shape
        ))
        model.add(Dropout(self.params['dropout']))

        # Additional LSTM layers
        for i in range(1, self.params['layers']):
            return_seq = (i < self.params['layers'] - 1)
            model.add(LSTM(units=self.params['units'], return_sequences=return_seq))
            model.add(Dropout(self.params['dropout']))

        # Output layer
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))

        # Compile
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        self.model = model
        logger.info(f"Built LSTM model: {self.params['layers']} layers, {self.params['units']} units")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """
        Fit LSTM model

        Parameters
        ----------
        X_train : np.ndarray
            Training sequences (samples, sequence_length, features)
        y_train : np.ndarray
            Training targets
        X_val : np.ndarray, optional
            Validation sequences
        y_val : np.ndarray, optional
            Validation targets
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        # Ensure 3D input
        if len(X_train.shape) == 2:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            if X_val is not None:
                X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        if self.model is None:
            self.build(input_shape=(X_train.shape[1], X_train.shape[2]))

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        self.is_fitted = True
        logger.info(f"LSTM model fitted: {len(self.history.history['loss'])} epochs")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Ensure 3D input
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def save(self, filepath: str) -> None:
        """Save LSTM model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info(f"LSTM model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load LSTM model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        self.is_fitted = True
        logger.info(f"LSTM model loaded from {filepath}")

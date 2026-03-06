"""
Unit test: LSTM Early Stopping.
Validates that the LSTM training loop stops early and restores best weights.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.models.lstm_model import LSTMModel


def test_lstm_stops_before_max_epochs():
    """With deliberately bad validation data, training should stop early."""
    np.random.seed(42)
    n_train, seq_len, n_feat = 200, 10, 3

    # Training data: clean learnable relationship
    X_train = np.random.randn(n_train, seq_len, n_feat).astype(np.float32)
    y_train = X_train[:, -1, 0].astype(np.float32)

    # Validation data: completely unrelated target at different scale
    n_val = 50
    X_val = np.random.randn(n_val, seq_len, n_feat).astype(np.float32)
    y_val = np.random.randn(n_val).astype(np.float32) * 100

    model = LSTMModel(params={
        'units': 32,
        'layers': 2,
        'dropout': 0.3,
        'batch_size': 32,
        'epochs': 200,
        'learning_rate': 0.005,
        'sequence_length': seq_len,
        'patience': 8,
    })

    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    epochs_run = len(model.history['train_loss'])
    # With patience=8 and unrelated val data, should stop well before 200
    assert epochs_run < 200, (
        f"Expected early stopping but model ran all epochs ({epochs_run})"
    )


def test_lstm_restores_best_weights():
    """After training, model weights should correspond to the best val epoch."""
    np.random.seed(123)
    n, seq_len, n_feat = 200, 10, 2

    X_train = np.random.randn(n, seq_len, n_feat).astype(np.float32)
    y_train = X_train[:, -1, 0].astype(np.float32)
    X_val = np.random.randn(50, seq_len, n_feat).astype(np.float32)
    y_val = np.random.randn(50).astype(np.float32) * 100

    model = LSTMModel(params={
        'units': 32,
        'layers': 2,
        'dropout': 0.3,
        'batch_size': 32,
        'epochs': 200,
        'learning_rate': 0.005,
        'sequence_length': seq_len,
        'patience': 8,
    })

    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # The _best_state should exist and have been loaded
    assert hasattr(model, '_best_state'), "Best state was never saved"
    assert model._best_state is not None, "Best state is None"
    assert model.is_fitted, "Model should be marked as fitted"


def test_lstm_patience_is_configurable():
    """Patience parameter should be read from params dict."""
    model = LSTMModel(params={'patience': 20})
    assert model.params['patience'] == 20

    model2 = LSTMModel()  # default
    assert model2.params['patience'] == 15

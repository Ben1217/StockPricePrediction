"""
Base Model Class
Abstract interface for all prediction models
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import joblib
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all prediction models"""

    def __init__(self, name: str, params: Optional[Dict] = None, task: str = "classification"):
        """
        Initialize base model

        Parameters
        ----------
        name : str
            Model name
        params : dict, optional
            Model hyperparameters
        """
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.task = str(self.params.get("task", task)).lower()

    @abstractmethod
    def build(self) -> None:
        """Build the model architecture"""
        pass

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Fit the model to training data

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Parameters
        ----------
        X : np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Predictions
        """
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance

        Parameters
        ----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets

        Returns
        -------
        dict
            Evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
            roc_auc_score,
        )

        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        predictions = self.predict(X_test)

        if self.task == "classification":
            y_true = np.asarray(y_test).astype(int).reshape(-1)
            y_pred = np.asarray(predictions).astype(int).reshape(-1)

            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                'f1': float(f1_score(y_true, y_pred, zero_division=0)),
                'directional_accuracy': float(accuracy_score(y_true, y_pred)),
            }

            try:
                probabilities = self.predict_proba(X_test)
                positive_proba = np.asarray(probabilities)[:, -1]
                metrics['roc_auc'] = float(roc_auc_score(y_true, positive_proba))
            except Exception:
                metrics['roc_auc'] = 0.5

            logger.info(
                "%s evaluation: ACC=%.4f, F1=%.4f, ROC-AUC=%.4f",
                self.name,
                metrics['accuracy'],
                metrics['f1'],
                metrics['roc_auc'],
            )
            return metrics

        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions),
        }

        # Directional accuracy
        if len(y_test) > 1:
            actual_direction = np.sign(y_test[1:] - y_test[:-1])
            pred_direction = np.sign(predictions[1:] - predictions[:-1])
            metrics['directional_accuracy'] = np.mean(actual_direction == pred_direction)

        logger.info(f"{self.name} evaluation: RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.4f}")

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities when supported."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise NotImplementedError(f"{self.name} does not expose predict_proba")

    def save(self, filepath: str) -> None:
        """Save model to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model from file"""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")

    def get_params(self) -> Dict:
        """Get model parameters"""
        return self.params

    def set_params(self, **params) -> None:
        """Set model parameters"""
        self.params.update(params)

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

    def __init__(self, name: str, params: Optional[Dict] = None):
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
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        predictions = self.predict(X_test)

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

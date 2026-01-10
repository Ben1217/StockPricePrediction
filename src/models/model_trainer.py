"""
Model Trainer Module
Orchestrates model training, evaluation, and comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Type
from pathlib import Path
import json

from .base_model import BaseModel
from .xgboost_model import XGBoostModel
from .lstm_model import LSTMModel
from .random_forest_model import RandomForestModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


MODEL_REGISTRY = {
    'xgboost': XGBoostModel,
    'lstm': LSTMModel,
    'random_forest': RandomForestModel,
}


class ModelTrainer:
    """Orchestrates model training and evaluation"""

    def __init__(self, models_dir: str = "models/saved_models"):
        """
        Initialize trainer

        Parameters
        ----------
        models_dir : str
            Directory for saving models
        """
        self.models_dir = Path(models_dir)
        self.models: Dict[str, BaseModel] = {}
        self.results: Dict[str, Dict] = {}

    def create_model(self, model_type: str, params: Optional[Dict] = None) -> BaseModel:
        """
        Create a model instance

        Parameters
        ----------
        model_type : str
            Type of model ('xgboost', 'lstm', 'random_forest')
        params : dict, optional
            Model parameters

        Returns
        -------
        BaseModel
            Model instance
        """
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(MODEL_REGISTRY.keys())}")

        return MODEL_REGISTRY[model_type](params)

    def train_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict] = None,
        save: bool = True
    ) -> BaseModel:
        """
        Train a single model

        Parameters
        ----------
        model_type : str
            Type of model
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray, optional
            Validation data
        params : dict, optional
            Model parameters
        save : bool
            Whether to save the model

        Returns
        -------
        BaseModel
            Trained model
        """
        logger.info(f"Training {model_type} model...")

        model = self.create_model(model_type, params)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        self.models[model_type] = model

        if save:
            save_path = self.models_dir / model_type / f"{model_type}_model"
            model.save(str(save_path))

        return model

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_types: Optional[List[str]] = None,
        save: bool = True
    ) -> Dict[str, BaseModel]:
        """
        Train multiple models

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray, optional
            Validation data
        model_types : list, optional
            Models to train (default: all)
        save : bool
            Whether to save models

        Returns
        -------
        dict
            Dictionary of trained models
        """
        if model_types is None:
            model_types = ['xgboost', 'random_forest']  # Exclude LSTM by default (slower)

        for model_type in model_types:
            try:
                self.train_model(model_type, X_train, y_train, X_val, y_val, save=save)
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")

        return self.models

    def evaluate_all_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate all trained models

        Parameters
        ----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets

        Returns
        -------
        pandas.DataFrame
            Comparison of model metrics
        """
        results = []

        for name, model in self.models.items():
            try:
                metrics = model.evaluate(X_test, y_test)
                metrics['model'] = name
                results.append(metrics)
                self.results[name] = metrics
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('model')
            df = df.sort_values('rmse')

        logger.info(f"Evaluation complete for {len(results)} models")
        return df

    def get_best_model(self, metric: str = 'rmse') -> Optional[BaseModel]:
        """
        Get the best performing model

        Parameters
        ----------
        metric : str
            Metric to compare (lower is better for rmse, mse, mae)

        Returns
        -------
        BaseModel or None
            Best model
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return None

        # Find best model
        best_name = min(self.results, key=lambda x: self.results[x].get(metric, float('inf')))
        return self.models.get(best_name)

    def save_results(self, filepath: str) -> None:
        """Save evaluation results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {filepath}")

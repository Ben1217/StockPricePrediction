"""
XGBoost Model Implementation
"""

import numpy as np
from typing import Dict, Optional
import xgboost as xgb

from .base_model import BaseModel
from ..utils.logger import get_logger
from ..utils.config_loader import get_config_value

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost regression model for stock prediction"""

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize XGBoost model

        Parameters
        ----------
        params : dict, optional
            Model hyperparameters. Uses config defaults if not provided.
        """
        default_params = get_config_value('models.xgboost.params', {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42,
        })

        if params:
            default_params.update(params)

        super().__init__(name='XGBoost', params=default_params)
        self.feature_importance_ = None

    def build(self) -> None:
        """Build XGBoost model"""
        self.model = xgb.XGBRegressor(**self.params)
        logger.info(f"Built XGBoost model with params: {self.params}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 10,
        **kwargs
    ) -> None:
        """
        Fit XGBoost model

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation targets
        early_stopping_rounds : int
            Early stopping patience
        """
        if self.model is None:
            self.build()

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False

        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_

        logger.info(f"XGBoost model fitted on {X_train.shape[0]} samples")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: list = None) -> Dict[str, float]:
        """
        Get feature importance scores

        Parameters
        ----------
        feature_names : list, optional
            Names of features

        Returns
        -------
        dict
            Feature name -> importance score
        """
        if self.feature_importance_ is None:
            return {}

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance_))]

        importance = dict(zip(feature_names, self.feature_importance_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, filepath: str) -> None:
        """Save XGBoost model"""
        if not filepath.endswith('.json'):
            filepath = filepath + '.json'
        self.model.save_model(filepath)
        logger.info(f"XGBoost model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load XGBoost model"""
        self.build()
        self.model.load_model(filepath)
        self.is_fitted = True
        logger.info(f"XGBoost model loaded from {filepath}")

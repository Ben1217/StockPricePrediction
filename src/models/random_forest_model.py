"""
Random Forest Model Implementation
"""

import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestRegressor

from .base_model import BaseModel
from ..utils.logger import get_logger
from ..utils.config_loader import get_config_value

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest regression model for stock prediction"""

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize Random Forest model

        Parameters
        ----------
        params : dict, optional
            Model hyperparameters
        """
        default_params = get_config_value('models.random_forest.params', {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
        })

        if params:
            default_params.update(params)

        super().__init__(name='RandomForest', params=default_params)
        self.feature_importance_ = None

    def build(self) -> None:
        """Build Random Forest model"""
        self.model = RandomForestRegressor(**self.params)
        logger.info(f"Built Random Forest model with params: {self.params}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> None:
        """
        Fit Random Forest model

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        """
        if self.model is None:
            self.build()

        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_

        logger.info(f"Random Forest model fitted on {X_train.shape[0]} samples")

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

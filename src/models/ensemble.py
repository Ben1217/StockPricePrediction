"""
Ensemble Predictor Module.

Combines predictions from multiple models (LSTM, XGBoost, Random Forest)
using Sharpe-ratio-based dynamic weighting. Models that performed better
on recent validation data receive higher weight.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EnsemblePredictor:
    """
    Weighted ensemble that dynamically adjusts model weights
    based on rolling Sharpe Ratio performance.

    Parameters
    ----------
    model_names : list of str
        Names of the models in the ensemble (e.g. ['xgboost', 'random_forest', 'lstm']).
    temperature : float
        Softmax temperature — lower = more aggressive weighting toward
        the best model; higher = more uniform weights.
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        temperature: float = 1.0,
    ):
        self.model_names = model_names or ['xgboost', 'random_forest', 'lstm']
        self.temperature = temperature
        self._weights: Dict[str, float] = {
            name: 1.0 / len(self.model_names) for name in self.model_names
        }
        self._sharpe_history: Dict[str, List[float]] = {
            name: [] for name in self.model_names
        }

    @property
    def weights(self) -> Dict[str, float]:
        """Current ensemble weights."""
        return dict(self._weights)

    def update_weights(self, sharpe_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Recalculate ensemble weights from per-model Sharpe Ratios.

        Uses softmax(sharpe / temperature) so that models with higher
        Sharpe get proportionally larger weight.

        Parameters
        ----------
        sharpe_scores : dict
            {model_name: sharpe_ratio} for each model.

        Returns
        -------
        dict
            Updated weights {model_name: weight}.
        """
        for name, sharpe in sharpe_scores.items():
            if name in self._sharpe_history:
                self._sharpe_history[name].append(sharpe)

        # Use latest Sharpe for weighting
        scores = np.array([
            sharpe_scores.get(name, 0.0) for name in self.model_names
        ])

        # Softmax with temperature
        self._weights = self._softmax_weights(scores)

        logger.info(
            f"Ensemble weights updated: "
            + ", ".join(f"{n}={w:.3f}" for n, w in self._weights.items())
        )
        return dict(self._weights)

    def update_weights_from_rolling(
        self,
        model_predictions: Dict[str, np.ndarray],
        actual_returns: np.ndarray,
        window: int = 30,
    ) -> Dict[str, float]:
        """
        Compute rolling Sharpe for each model and update weights.

        Parameters
        ----------
        model_predictions : dict
            {model_name: predicted_returns_array} for each model.
        actual_returns : np.ndarray
            Actual returns over the same period.
        window : int
            Rolling window (trading days) for Sharpe calculation.

        Returns
        -------
        dict
            Updated weights.
        """
        sharpe_scores = {}
        for name in self.model_names:
            preds = model_predictions.get(name)
            if preds is None or len(preds) < window:
                sharpe_scores[name] = 0.0
                continue

            # Use last `window` days
            recent_preds = preds[-window:]
            recent_actual = actual_returns[-window:]

            # Strategy returns: go long when model predicts positive
            strategy = np.sign(recent_preds) * recent_actual

            if np.std(strategy) > 0:
                sharpe = float(np.mean(strategy) / np.std(strategy) * np.sqrt(252))
            else:
                sharpe = 0.0

            sharpe_scores[name] = sharpe

        return self.update_weights(sharpe_scores)

    def predict(
        self,
        model_predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Produce a weighted ensemble prediction.

        Parameters
        ----------
        model_predictions : dict
            {model_name: prediction_array} for each model.

        Returns
        -------
        np.ndarray
            Weighted average prediction.
        """
        predictions = []
        weights = []

        for name in self.model_names:
            pred = model_predictions.get(name)
            if pred is not None:
                predictions.append(pred)
                weights.append(self._weights.get(name, 0.0))

        if not predictions:
            raise ValueError("No predictions provided for any model")

        predictions = np.array(predictions)
        weights = np.array(weights)

        # Renormalise weights for available models
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)

        # Weighted average across models
        ensemble = np.average(predictions, axis=0, weights=weights)
        return ensemble

    def _softmax_weights(self, scores: np.ndarray) -> Dict[str, float]:
        """Compute softmax weights from scores with temperature scaling."""
        # Shift for numerical stability
        shifted = (scores - np.max(scores)) / max(self.temperature, 1e-6)
        exp_scores = np.exp(shifted)
        softmax = exp_scores / exp_scores.sum()

        return {name: float(w) for name, w in zip(self.model_names, softmax)}

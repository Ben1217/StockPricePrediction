"""
Market Regime Detection using Hidden Markov Models.

Classifies the market into 3 regimes:
  - Bull  (high positive mean return)
  - Bear  (negative mean return)
  - Sideways (low absolute mean return)

Uses daily returns and rolling realised volatility as observable inputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MarketRegimeDetector:
    """
    Hidden Markov Model-based market regime detector.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 3: Bull/Bear/Sideways).
    vol_window : int
        Rolling window for realised volatility feature (trading days).
    random_state : int
        Seed for reproducibility.
    """

    REGIME_NAMES = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}

    def __init__(
        self,
        n_states: int = 3,
        vol_window: int = 20,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.vol_window = vol_window
        self.random_state = random_state
        self.model = None
        self._state_order: Optional[np.ndarray] = None  # mapping from HMM state → sorted label
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> "MarketRegimeDetector":
        """
        Fit the HMM on historical returns.

        Parameters
        ----------
        returns : pd.Series
            Daily percentage returns (e.g. ``Close.pct_change()``).
            NaN rows are dropped automatically.

        Returns
        -------
        self
        """
        from hmmlearn.hmm import GaussianHMM

        features = self._build_features(returns)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type='diag',
            n_iter=200,
            random_state=self.random_state,
        )
        self.model.fit(features)

        # Determine label ordering so that:
        #   0 = lowest mean return  → Bear
        #   1 = middle              → Sideways
        #   2 = highest             → Bull
        mean_returns = self.model.means_[:, 0]  # first column = returns
        self._state_order = np.argsort(mean_returns)

        self.is_fitted = True
        logger.info(
            f"MarketRegimeDetector fitted: {self.n_states} states, "
            f"state mean returns = {mean_returns[self._state_order].tolist()}"
        )
        return self

    def predict(self, returns: pd.Series) -> np.ndarray:
        """
        Predict regime labels for each time step.

        Parameters
        ----------
        returns : pd.Series
            Daily percentage returns.

        Returns
        -------
        np.ndarray
            Array of regime labels (0=Bear, 1=Sideways, 2=Bull).
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        features = self._build_features(returns)
        raw_states = self.model.predict(features)

        # Map raw HMM states to ordered labels
        label_map = {old: new for new, old in enumerate(self._state_order)}
        ordered = np.array([label_map[s] for s in raw_states])
        return ordered

    def predict_proba(self, returns: pd.Series) -> np.ndarray:
        """
        Predict regime probabilities for each time step.

        Returns
        -------
        np.ndarray  shape (n_samples, n_states)
            Columns ordered as (Bear, Sideways, Bull).
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        features = self._build_features(returns)
        raw_proba = self.model.predict_proba(features)

        # Reorder columns to match (Bear, Sideways, Bull)
        return raw_proba[:, self._state_order]

    def get_regime_features(self, returns: pd.Series) -> pd.DataFrame:
        """
        Generate regime features ready to merge into a model feature matrix.

        Parameters
        ----------
        returns : pd.Series
            Daily percentage returns (must have a DatetimeIndex or
            matching index to the main DataFrame).

        Returns
        -------
        pd.DataFrame
            Columns: regime_label (int), regime_name (str),
            regime_probability (float — probability of the most likely state).
            Index matches the *valid* portion of ``returns`` (after
            dropping NaN from the volatility rolling window).
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before generating features")

        clean = returns.dropna()
        vol = clean.rolling(self.vol_window).std().dropna()
        valid_idx = vol.index

        labels = self.predict(clean)
        proba = self.predict_proba(clean)

        # Align to valid_idx (features start after vol_window)
        labels_aligned = labels[-len(valid_idx):]
        proba_aligned = proba[-len(valid_idx):]

        regime_prob = np.max(proba_aligned, axis=1)

        df = pd.DataFrame(
            {
                'regime_label': labels_aligned,
                'regime_name': [self.REGIME_NAMES.get(l, 'Unknown') for l in labels_aligned],
                'regime_probability': regime_prob,
            },
            index=valid_idx,
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_features(self, returns: pd.Series) -> np.ndarray:
        """
        Build the 2D observable feature matrix for the HMM.

        Features:
            1. Daily return
            2. Rolling realised volatility (std of returns)

        Returns
        -------
        np.ndarray  shape (n_valid_samples, 2)
        """
        clean = returns.dropna()
        vol = clean.rolling(self.vol_window).std()

        combined = pd.DataFrame({'return': clean, 'volatility': vol}).dropna()

        return combined.values

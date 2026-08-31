"""
Foundation Ensemble Direction Estimator.

Combines the tabular foundation model (TabPFN) and the candlestick foundation model
(Kronos) by averaging their out-of-sample next-day upward probability forecasts:

    P(up) = 0.5 * P_TabPFN(up) + 0.5 * P_Kronos(up)

Price forecast bands (5th, 50th, 95th percentiles) are propagated from Kronos
to provide a forecasted price range alongside the binary directional call.

Resilience & Graceful Degradation:
If one foundation model is unavailable (e.g., missing API token or uninstalled),
the ensemble gracefully adapts by weighting the active model 100% and logging
an informative diagnostic notice.

Public API:
    FoundationEnsemble(seed, ...) — DirectionEstimator subclass
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from .direction_models import DirectionEstimator, _ConstantProbabilityMixin
# Kept in step with the Kronos slot: an ensemble that silently used a 252-bar
# lookback while the standalone model used 128 would not be comparing the same
# model, and would cost four times as much per row.
from .kronos_direction import (
    DEFAULT_LOOKBACK as KRONOS_DEFAULT_LOOKBACK,
    DEFAULT_SAMPLE_COUNT as KRONOS_DEFAULT_SAMPLE_COUNT,
)

logger = get_logger(__name__)


class FoundationEnsemble(DirectionEstimator, _ConstantProbabilityMixin):
    """
    Ensemble model combining TabPFNDirection and KronosDirection.

    Implements the standard DirectionEstimator interface with dual model calls:
    1. TabPFN operates on the 46 engineered tabular features.
    2. Kronos operates on the raw OHLCV candlestick sequences.
    3. Probabilities are averaged.
    4. Kronos price bands are exposed in ``self.price_bands_``.
    """

    name = "foundation_ensemble"

    def __init__(
        self,
        seed: int = 42,
        tabpfn_weight: float = 0.5,
        kronos_weight: float = 0.5,
        sample_count: int = KRONOS_DEFAULT_SAMPLE_COUNT,
        lookback: int = KRONOS_DEFAULT_LOOKBACK,
        tabpfn_estimator: Optional[DirectionEstimator] = None,
        kronos_estimator: Optional[DirectionEstimator] = None,
    ):
        super().__init__(seed=seed)
        self.tabpfn_weight = float(tabpfn_weight)
        self.kronos_weight = float(kronos_weight)
        self.sample_count = int(sample_count)
        self.lookback = int(lookback)

        self.tabpfn_ = tabpfn_estimator
        self.kronos_ = kronos_estimator
        self.price_bands_: Optional[np.ndarray] = None
        self.degenerate_: bool = False
        self._ohlcv_all: Optional[pd.DataFrame] = None
        self._tabpfn_active: bool = True
        self._kronos_active: bool = True

        if self.tabpfn_ is None or self.kronos_ is None:
            self._init_submodels()

    def _init_submodels(self) -> None:
        """Instantiate submodels with graceful error handling."""
        from .tabpfn_direction import TabPFNDirection
        from .kronos_direction import KronosDirection

        if self.tabpfn_ is None:
            try:
                self.tabpfn_ = TabPFNDirection(seed=self.seed)
            except Exception as exc:
                logger.warning("TabPFN could not be initialized (%s); disabling TabPFN branch in ensemble", exc)
                self.tabpfn_ = None
                self._tabpfn_active = False

        if self.kronos_ is None:
            try:
                self.kronos_ = KronosDirection(
                    seed=self.seed,
                    sample_count=self.sample_count,
                    lookback=self.lookback,
                )
            except Exception as exc:
                logger.warning("Kronos could not be initialized (%s); disabling Kronos branch in ensemble", exc)
                self.kronos_ = None
                self._kronos_active = False

    def set_ohlcv_context(self, ohlcv: pd.DataFrame) -> None:
        """Pass the raw OHLCV bars through to Kronos."""
        self._ohlcv_all = ohlcv.copy()
        if self.kronos_ is not None and hasattr(self.kronos_, "set_ohlcv_context"):
            self.kronos_.set_ohlcv_context(ohlcv)

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.degenerate_ = len(np.unique(y)) < 2
        if self.degenerate_:
            logger.warning(
                "Training window for %s holds one class only; emitting its base rate",
                self.name,
            )
            self.fit_info_["degenerate_single_class"] = True
            return

        if self.tabpfn_ is None and self.kronos_ is None:
            self._init_submodels()

        if self._ohlcv_all is not None and self.kronos_ is not None and hasattr(self.kronos_, "set_ohlcv_context"):
            self.kronos_.set_ohlcv_context(self._ohlcv_all)

        # Fit TabPFN on engineered features
        if self.tabpfn_ is not None:
            try:
                self.tabpfn_.fit(X, y)
                self._tabpfn_active = True
            except Exception as exc:
                logger.warning(
                    "TabPFN fit encountered an error (%s); adapting ensemble to 100%% Kronos weighting. "
                    "(To enable TabPFN, set TABPFN_TOKEN from https://ux.priorlabs.ai)",
                    exc,
                )
                self._tabpfn_active = False

        # Fit Kronos (stores context and checks validity)
        if self.kronos_ is not None:
            try:
                self.kronos_.fit(X, y)
                self._kronos_active = True
            except Exception as exc:
                logger.warning("Kronos fit encountered an error (%s); adapting ensemble to 100%% TabPFN weighting", exc)
                self._kronos_active = False

        if not self._tabpfn_active and not self._kronos_active:
            raise RuntimeError("Both foundation models (TabPFN and Kronos) failed to fit in FoundationEnsemble.")

        self.fit_info_.update({
            "tabpfn_active": self._tabpfn_active,
            "kronos_active": self._kronos_active,
            "tabpfn_info": self.tabpfn_.fit_info_ if self._tabpfn_active and self.tabpfn_ else {},
            "kronos_info": self.kronos_.fit_info_ if self._kronos_active and self.kronos_ else {},
            "tabpfn_weight": self.tabpfn_weight if self._tabpfn_active else 0.0,
            "kronos_weight": self.kronos_weight if self._kronos_active else 0.0,
        })

    def _predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        if self.degenerate_ or (not self._tabpfn_active and not self._kronos_active):
            self.price_bands_ = None
            return self._constant(len(X), self.train_base_rate_)

        p_tabpfn: Optional[np.ndarray] = None
        if self._tabpfn_active and self.tabpfn_ is not None:
            try:
                p_tabpfn = self.tabpfn_.predict_proba_up(X)
            except Exception as exc:
                logger.warning("TabPFN predict_proba_up failed: %s", exc)
                p_tabpfn = None

        p_kronos: Optional[np.ndarray] = None
        if self._kronos_active and self.kronos_ is not None:
            try:
                p_kronos = self.kronos_.predict_proba_up(X)
                self.price_bands_ = getattr(self.kronos_, "price_bands_", None)
            except Exception as exc:
                logger.warning("Kronos predict_proba_up failed: %s", exc)
                p_kronos = None

        if p_tabpfn is not None and p_kronos is not None:
            total_weight = self.tabpfn_weight + self.kronos_weight
            w_tab = self.tabpfn_weight / total_weight if total_weight > 0 else 0.5
            w_kron = self.kronos_weight / total_weight if total_weight > 0 else 0.5
            return np.asarray(w_tab * p_tabpfn + w_kron * p_kronos, dtype=np.float64)
        elif p_kronos is not None:
            return np.asarray(p_kronos, dtype=np.float64)
        elif p_tabpfn is not None:
            return np.asarray(p_tabpfn, dtype=np.float64)
        else:
            return self._constant(len(X), self.train_base_rate_)

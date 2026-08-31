"""
TabPFN v2 direction classifier — tabular foundation model slot.

TabPFN is a Prior-Data Fitted Network: a transformer pre-trained on millions
of synthetic classification problems so that its forward pass on a new dataset
*is* Bayesian inference, without gradient updates. ``fit()`` stores the
training set as the in-context "prompt"; ``predict_proba()`` conditions on it
and returns calibrated posterior probabilities.

On a ~2 500-row x 46-feature direction dataset this lands squarely in the
regime where TabPFN's authors demonstrated it matches or exceeds tuned
gradient-boosted trees, so it is a legitimate contender — not a novelty.

Graceful degradation
--------------------
If ``tabpfn`` is not installed the module still imports; the class raises at
construction time with an actionable message, and the model-registry guard in
``direction_models.py`` keeps it out of ``MODEL_FACTORIES``.

Public API:
    TabPFNDirection(seed) — DirectionEstimator subclass
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ..utils.logger import get_logger
from .direction_models import DirectionEstimator, _ConstantProbabilityMixin

logger = get_logger(__name__)
load_dotenv()

_TABPFN_AVAILABLE = False
_TABPFN_IMPORT_ERROR: Optional[str] = None

try:
    from tabpfn import TabPFNClassifier  # type: ignore[import-untyped]
    _TABPFN_AVAILABLE = True
except ImportError as exc:
    _TABPFN_IMPORT_ERROR = str(exc)


class TabPFNDirection(DirectionEstimator, _ConstantProbabilityMixin):
    """
    TabPFN v2 classifier for next-day direction.

    Implements the same ``fit / predict_proba_up / predict`` interface as
    ``LogisticDirection`` and ``GradientBoostingDirection``, so the walk-forward
    harness, baselines, backtest, and leakage check all work unchanged.

    There is no hyperparameter tuning: the model architecture is the
    hyperparameter, chosen by pre-training on synthetic tasks. The training set
    is handed to ``fit()`` as context, and ``predict_proba()`` conditions on it.

    Device selection: CUDA when available, CPU otherwise. TabPFN's CPU path is
    slower but functional and sufficient for the dataset sizes here.
    """

    name = "tabpfn"

    def __init__(self, seed: int = 42, model: Optional[Any] = None):
        super().__init__(seed=seed)
        self.model_ = model
        self.degenerate_: bool = False
        if self.model_ is None and not _TABPFN_AVAILABLE:
            raise ImportError(
                f"TabPFN is not installed ({_TABPFN_IMPORT_ERROR}). "
                f"Install it with: pip install tabpfn>=2.0"
            )

    def _detect_device(self) -> str:
        """Pick the best available device. TabPFN supports cuda, mps, and cpu."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def _as_array(X: pd.DataFrame) -> np.ndarray:
        """
        DataFrame to float array, refusing to paper over non-finite values.

        The previous version called ``np.nan_to_num(..., nan=0.0)``. On a
        standardised feature, 0.0 is the *mean* — so a missing value would have
        been silently replaced by "perfectly average", which is a fabricated
        observation the model then treats as real. ``build_direction_dataset``
        already drops incomplete rows, so a non-finite value here means an
        upstream contract broke and should be loud.
        """
        values = X.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            bad = X.columns[~np.isfinite(values).all(axis=0)].tolist()
            raise ValueError(
                f"TabPFN received non-finite values in {bad}; the dataset builder "
                f"should have dropped those rows rather than passing them on"
            )
        return values

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.degenerate_ = len(np.unique(y)) < 2
        if self.degenerate_:
            logger.warning(
                "Training window for %s holds one class only; emitting its base rate",
                self.name,
            )
            self.fit_info_["degenerate_single_class"] = True
            return

        if self.model_ is None:
            device = self._detect_device()
            self.model_ = TabPFNClassifier(device=device, random_state=self.seed)
        else:
            device = getattr(self.model_, "device", "custom")

        x_values = self._as_array(X)
        self.model_.fit(x_values, y)
        self.fit_info_.update({
            "device": device,
            "n_features": int(x_values.shape[1]),
        })
        logger.info(
            "TabPFN fitted on %d rows × %d features (device=%s)",
            len(y), x_values.shape[1], device,
        )

    def _predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        if self.degenerate_ or self.model_ is None:
            return self._constant(len(X), self.train_base_rate_)

        proba = self.model_.predict_proba(self._as_array(X))
        # TabPFN returns (n_samples, n_classes). Class 1 is "up".
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float64)
        # Single-class fallback (should not happen after the degenerate guard).
        return self._constant(len(X), self.train_base_rate_)

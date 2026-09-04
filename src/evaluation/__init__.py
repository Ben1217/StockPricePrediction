"""
Evaluation, benchmarking and econometric validation (Addendum A).

The question this package answers is not "how accurate is the model" but "does
the model reject the martingale null" -- ``E[P(t+h) | F(t)] = P(t)`` -- and it
is built so the answer can come back "no" without anything breaking. Negative
results are deliverables here, not bugs.

Layout
------
``splitting``        purged and embargoed walk-forward folds, effective N,
                     held-out ticker split (A4)
``baselines``        the mandatory baseline suite, always-up included (A2)
``metrics``          directional and probabilistic scoring, EOBR (A3, A6)
``calibration``      Brier decomposition, PIT, reliability diagram, fold-honest
                     recalibration and the uncalibrated verdict (A3.2, A6.4-A6.6)
``testing``          Diebold-Mariano, McNemar, Benjamini-Hochberg (A5)
``economics``        paper-trading overlay, cost model, break-even bps (A7)
``cross_sectional``  IC, RankIC, IC-IR, quintile spread (A7.5)
``volatility``       OHLC realised-variance estimators and QLIKE (A8)
``snapshot``         frozen, hashed dataset snapshots (A10.1)
``leakage``          the ten-item A9 checklist, executed and recorded

Two conventions worth knowing before reading any of it
------------------------------------------------------
Scoring happens in **log-return space**; price levels are a display quantity
only (A1). And loss differentials are always ``L_model - L_baseline``, so a
negative mean differential means the model won (see :mod:`~src.evaluation.testing`).

Nothing here is imported eagerly: several submodules pull in scikit-learn or
scipy machinery that a plain API request has no use for, so import the
submodule you need rather than paying for all nine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time cost avoided at runtime
    from . import (
        baselines,
        calibration,
        cross_sectional,
        economics,
        leakage,
        metrics,
        snapshot,
        splitting,
        testing,
        volatility,
    )

__all__ = [
    "baselines",
    "calibration",
    "cross_sectional",
    "economics",
    "leakage",
    "metrics",
    "snapshot",
    "splitting",
    "testing",
    "volatility",
]

"""
Does the stationary feature set carry any signal at these horizons?

The bundles keep landing at or below coin-flip directional accuracy with negative
skill against a constant predictor. That is either a modelling problem or a data
problem, and the two need different fixes, so this measures which it is.

Three tests, all on the held-out segment of a chronological split:

  1. Permutation importance — shuffle one feature at a time and see whether the
     error moves. If nothing moves, the model is not using the features.

  2. Label-shuffle control — retrain on randomly permuted targets. Whatever
     score that reaches is the score attainable with zero signal. A real model
     must beat its own shuffled twin; if it does not, the apparent structure is
     noise the model memorised.

  3. Univariate rank correlation — Spearman between each feature and the forward
     return, with the sign-agreement rate. This sees monotone relationships the
     trees might be splitting away.

Usage:
    python scripts/feature_signal_audit.py --symbol GOOGL --horizon 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import download_stock_data  # noqa: E402
from src.features.feature_engineering import (  # noqa: E402
    build_regression_dataset,
    normalize_feature_config,
    split_dataset_chronologically,
)
from src.models.ensemble_training import DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE  # noqa: E402
from src.models.regression_models import REGRESSOR_FACTORIES  # noqa: E402


def load_split(symbol: str, horizon: int, lookback_days: int = 1825):
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    df = download_stock_data(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        raise SystemExit(f"no data for {symbol}")

    config = normalize_feature_config()
    dataset, feature_cols, target_col = build_regression_dataset(df, horizon=horizon, feature_config=config)
    split = split_dataset_chronologically(
        dataset, feature_columns=feature_cols, target_column=target_col,
        scaler_type="standard", test_size=DEFAULT_TEST_SIZE, val_size=DEFAULT_VAL_SIZE,
        embargo=horizon,
    )
    return split, feature_cols, target_col


def directional_accuracy(y_true, y_pred) -> float:
    actual, pred = np.sign(y_true), np.sign(y_pred)
    valid = actual != 0
    return float(np.mean(actual[valid] == pred[valid])) if valid.sum() else 0.5


def permutation_importance(model, X, y, feature_cols: List[str], n_repeats: int = 10, seed: int = 0) -> List[Dict]:
    rng = np.random.default_rng(seed)
    base = float(np.mean(np.abs(y - model.predict(X))))
    rows = []
    for j, name in enumerate(feature_cols):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(float(np.mean(np.abs(y - model.predict(Xp)))) - base)
        rows.append({"feature": name, "mae_increase": float(np.mean(deltas)),
                     "std": float(np.std(deltas))})
    rows.sort(key=lambda r: r["mae_increase"], reverse=True)
    return rows


def label_shuffle_control(factory, X_tr, y_tr, X_te, y_te, n_trials: int = 5, seed: int = 0) -> Dict:
    """Scores reachable with the signal destroyed — the real null for this data."""
    rng = np.random.default_rng(seed)
    maes, das = [], []
    for _ in range(n_trials):
        model = factory(None)
        model.fit(X_tr, rng.permutation(y_tr))
        pred = model.predict(X_te)
        maes.append(float(np.mean(np.abs(y_te - pred))))
        das.append(directional_accuracy(y_te, pred))
    return {"shuffled_mae_mean": float(np.mean(maes)), "shuffled_mae_std": float(np.std(maes)),
            "shuffled_da_mean": float(np.mean(das))}


def univariate_correlations(split, feature_cols: List[str], target_col: str) -> List[Dict]:
    train = split["train_frame"]
    rows = []
    for name in feature_cols:
        x = train[name].values.astype(np.float64)
        y = train[target_col].values.astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 30:
            continue
        rho, p = stats.spearmanr(x[ok], y[ok])
        rows.append({"feature": name, "spearman": float(rho), "p_value": float(p),
                     "abs_rho": abs(float(rho))})
    rows.sort(key=lambda r: r["abs_rho"], reverse=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="GOOGL")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--model", default="random_forest", choices=list(REGRESSOR_FACTORIES))
    args = ap.parse_args()

    split, feature_cols, target_col = load_split(args.symbol.upper(), args.horizon)
    X_tr, y_tr = split["X_train"], split["y_train"]
    X_te, y_te = split["X_test"], split["y_test"]

    factory = REGRESSOR_FACTORIES[args.model]
    model = factory(None)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    real_mae = float(np.mean(np.abs(y_te - pred)))
    real_da = directional_accuracy(y_te, pred)
    train_mean = float(np.mean(y_tr))
    baseline_mae = float(np.mean(np.abs(y_te - train_mean)))

    print(f"\n{'=' * 72}")
    print(f"FEATURE SIGNAL AUDIT — {args.symbol.upper()} h={args.horizon}d model={args.model}")
    print(f"{'=' * 72}")
    print(f"\nrows: train={len(X_tr)} val={len(split['X_val'])} test={len(X_te)}  features={len(feature_cols)}")

    print(f"\n[0] Baselines on the held-out test segment")
    print(f"    constant train-mean MAE : {baseline_mae:.6f}")
    print(f"    model MAE               : {real_mae:.6f}   "
          f"(skill {1 - real_mae / baseline_mae:+.4f})")
    print(f"    model directional acc   : {real_da:.2%}")

    print(f"\n[1] Label-shuffle control — what a zero-signal model reaches")
    ctrl = label_shuffle_control(factory, X_tr, y_tr, X_te, y_te)
    print(f"    shuffled-target MAE     : {ctrl['shuffled_mae_mean']:.6f} "
          f"(+/- {ctrl['shuffled_mae_std']:.6f})")
    print(f"    shuffled-target DA      : {ctrl['shuffled_da_mean']:.2%}")
    verdict = ("model does NOT beat its own shuffled twin — no usable signal"
               if real_mae >= ctrl["shuffled_mae_mean"] else
               "model beats the shuffled control")
    print(f"    -> {verdict}")

    print(f"\n[2] Permutation importance on the test segment (MAE increase when shuffled)")
    for row in permutation_importance(model, X_te, y_te, feature_cols)[:10]:
        flag = "" if row["mae_increase"] > row["std"] else "   (within noise)"
        print(f"    {row['feature']:<24} {row['mae_increase']:+.6f}{flag}")

    print(f"\n[3] Univariate Spearman vs forward {args.horizon}d return (train segment)")
    for row in univariate_correlations(split, feature_cols, target_col)[:10]:
        sig = "*" if row["p_value"] < 0.05 else " "
        print(f"    {row['feature']:<24} rho={row['spearman']:+.4f}  p={row['p_value']:.4f} {sig}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

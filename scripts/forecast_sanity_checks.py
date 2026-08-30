"""
Sanity checks for the return-regression forecast path.

Two questions, both answerable without a human squinting at a chart:

  1. Drawdown check — freeze the pipeline to a date just before a known
     drawdown and ask what it forecast. A model with a systematic upward bias
     predicts a rise into every drawdown it is shown.

  2. Bound check — is the forecast reachable from the RandomForest's training
     targets? Note the units: the RF target is a forward RETURN, so the bound
     that holds is on the predicted return, not on the price. A price above
     anything in the training window is expected and proves nothing, because
     the live price is an input, not something the tree emits.

Usage:
    python scripts/forecast_sanity_checks.py --symbol GOOGL --horizon 30
    python scripts/forecast_sanity_checks.py --symbol TSLA --freeze-date 2022-01-03
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.regression_models import REGRESSOR_FACTORIES, REGRESSOR_FILE_NAMES  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

BUNDLES = Path("models/bundles")


# ---------------------------------------------------------------------------
# Check 2 — is the forecast reachable from the RF's training targets?
# ---------------------------------------------------------------------------

def rf_output_bounds(symbol: str, horizon: int) -> Optional[Dict[str, float]]:
    """
    The interval a fitted RandomForest can emit, in RETURN units.

    A forest averages its trees, so its reachable range sits inside the min/max
    of the per-tree leaf values. Those leaves are means of training targets, so
    this is exactly "what the trees could have learned to say".
    """
    bdir = BUNDLES / symbol.upper() / "random_forest" / str(int(horizon))
    model_path = bdir / REGRESSOR_FILE_NAMES["random_forest"]
    if not model_path.exists():
        return None
    model = REGRESSOR_FACTORIES["random_forest"]()
    model.load(str(model_path))
    est = getattr(model, "model", model)
    trees = getattr(est, "estimators_", [])
    if not trees:
        return None
    leaves = np.concatenate([t.tree_.value.reshape(-1) for t in trees])
    return {"min_return": float(leaves.min()), "max_return": float(leaves.max())}


def check_prediction_within_rf_bounds(
    symbol: str,
    horizon: int,
    predicted_return: float,
    current_price: float,
) -> Dict[str, object]:
    """
    Verify the RF's predicted RETURN lies inside its reachable leaf range.

    Also reports the price comparison the naive version of this check would
    make, to show why that one cannot fail informatively.
    """
    bounds = rf_output_bounds(symbol, horizon)
    if bounds is None:
        return {"status": "skipped", "reason": f"no RF bundle for {symbol} h={horizon}"}

    lo, hi = bounds["min_return"], bounds["max_return"]
    within = lo - 1e-9 <= predicted_return <= hi + 1e-9
    implied_price = current_price * (1.0 + predicted_return)
    max_price = current_price * (1.0 + hi)

    return {
        "status": "pass" if within else "FAIL",
        "predicted_return": round(float(predicted_return), 6),
        "rf_reachable_return_range": [round(lo, 6), round(hi, 6)],
        "within_bounds": bool(within),
        "implied_price": round(float(implied_price), 2),
        "max_price_reachable_from_current": round(float(max_price), 2),
        "note": (
            "The bound is on the RETURN. The implied price scales with the live "
            "current price, so it can legitimately exceed every price in the "
            "training window — comparing the forecast price to a training price "
            "maximum tests nothing."
        ),
    }


# ---------------------------------------------------------------------------
# Check 1 — frozen-date drawdown test
# ---------------------------------------------------------------------------

def frozen_forecast(
    symbol: str,
    horizon: int,
    freeze_date: str,
    model_types: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Run the live predictor against history truncated at `freeze_date`.

    Nothing after the freeze date is visible to feature construction, so the
    forecast is what the deployed path would have produced on that morning. The
    realised forward return over the same window is then reported beside it.
    """
    os.environ.setdefault("QUANTVISION_ENFORCE_MODEL_SKILL", "false")
    from src.data.data_loader import download_stock_data
    from src.models.ensemble_predictor import EnsemblePricePredictor

    freeze = pd.Timestamp(freeze_date)
    # Enough history for the long-window indicators, plus the forward window to
    # score against.
    start = (freeze - pd.Timedelta(days=1825)).strftime("%Y-%m-%d")
    end = (freeze + pd.Timedelta(days=int(horizon * 2.2) + 30)).strftime("%Y-%m-%d")

    full = download_stock_data(symbol, start, end)
    if full is None or full.empty:
        return {"status": "error", "reason": "no data"}
    full = full.sort_index()
    if getattr(full.index, "tz", None) is not None:
        full.index = full.index.tz_localize(None)

    history = full.loc[full.index <= freeze]
    future = full.loc[full.index > freeze]
    if len(history) < 300:
        return {"status": "error", "reason": f"only {len(history)} rows before {freeze_date}"}

    forecast = EnsemblePricePredictor().predict(symbol=symbol, horizon=horizon, raw_df=history)
    if forecast is None:
        return {"status": "error", "reason": "predictor returned None (no servable bundles)"}

    price_at_freeze = float(history["Close"].iloc[-1])
    realised = None
    if len(future) >= horizon:
        realised = float(future["Close"].iloc[horizon - 1]) / price_at_freeze - 1.0
    elif len(future) > 0:
        realised = float(future["Close"].iloc[-1]) / price_at_freeze - 1.0

    predicted_change = forecast.expected_change_pct / 100.0
    result = {
        "status": "ok",
        "symbol": symbol,
        "horizon": horizon,
        "freeze_date": str(freeze.date()),
        "price_at_freeze": round(price_at_freeze, 2),
        "predicted_change_pct": round(forecast.expected_change_pct, 2),
        "predicted_direction": "UP" if predicted_change > 0 else "DOWN",
        "signal": forecast.signal,
        "path_type": forecast.path_type,
        "per_model_returns": {
            r.model_type: round(r.predicted_return, 6) for r in forecast.model_predictions
        },
    }
    if realised is not None:
        result.update({
            "realised_change_pct": round(realised * 100, 2),
            "realised_direction": "UP" if realised > 0 else "DOWN",
            "direction_correct": (predicted_change > 0) == (realised > 0),
        })
    return result


def run_drawdown_suite(symbol: str, horizon: int, freeze_dates: List[str]) -> List[Dict]:
    """Run the frozen check across several pre-drawdown dates."""
    results = []
    for date in freeze_dates:
        try:
            results.append(frozen_forecast(symbol, horizon, date))
        except Exception as exc:  # noqa: BLE001
            results.append({"status": "error", "freeze_date": date, "reason": str(exc)})
    return results


# Well-known drawdown onsets, usable as default freeze dates.
KNOWN_DRAWDOWN_ONSETS = [
    "2020-02-14",  # COVID crash
    "2022-01-03",  # 2022 rate-driven bear market
    "2022-08-16",  # second leg down
    "2025-02-19",  # 2025 drawdown
]


def _train_target_mean(symbol: str, horizon: int) -> Optional[float]:
    """The mean training target — the level a signal-free regressor decays toward."""
    meta_path = BUNDLES / symbol.upper() / "random_forest" / str(int(horizon)) / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        value = json.loads(meta_path.read_text()).get("train_target_mean")
        return float(value) if value is not None else None
    except Exception:
        return None


def live_bound_check(symbol: str, horizon: int) -> Dict[str, object]:
    """Run a current forecast and test the RF's predicted return against its leaf range."""
    os.environ.setdefault("QUANTVISION_ENFORCE_MODEL_SKILL", "false")
    from src.data.data_loader import download_stock_data
    from src.models.ensemble_predictor import EnsemblePricePredictor

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=1825)
    df = download_stock_data(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        return {"status": "error", "reason": "no data"}
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    forecast = EnsemblePricePredictor().predict(
        symbol=symbol, horizon=horizon, raw_df=df, model_types=["random_forest"]
    )
    if forecast is None or not forecast.model_predictions:
        return {"status": "error", "reason": "no servable RF bundle"}

    rf = forecast.model_predictions[0]
    return check_prediction_within_rf_bounds(
        symbol, horizon, rf.predicted_return, forecast.current_price
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Forecast sanity checks")
    parser.add_argument("--symbol", default="GOOGL")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--freeze-date", action="append", dest="freeze_dates")
    parser.add_argument("--skip-drawdown", action="store_true")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    freeze_dates = args.freeze_dates or KNOWN_DRAWDOWN_ONSETS

    print(f"\n{'=' * 72}\nFORECAST SANITY CHECKS — {symbol} h={args.horizon}d\n{'=' * 72}")

    if not args.skip_drawdown:
        print("\n[1] Frozen-date drawdown check")
        print("    Does the model forecast a rise into a known drawdown?\n")
        results = run_drawdown_suite(symbol, args.horizon, freeze_dates)
        upward = 0
        scored = 0
        for r in results:
            if r.get("status") != "ok":
                print(f"    {r.get('freeze_date', '?')}: {r.get('status')} — {r.get('reason')}")
                continue
            scored += 1
            if r["predicted_direction"] == "UP":
                upward += 1
            realised = r.get("realised_change_pct")
            realised_str = f"realised {realised:+.2f}%" if realised is not None else "realised n/a"
            mark = ""
            if "direction_correct" in r:
                mark = "  OK" if r["direction_correct"] else "  MISS"
            print(f"    {r['freeze_date']}: predicted {r['predicted_change_pct']:+.2f}% "
                  f"({r['predicted_direction']}), {realised_str}{mark}")
        if scored:
            missed = sum(1 for r in results
                         if r.get("status") == "ok" and r.get("direction_correct") is False)
            print(f"\n    -> forecast UP on {upward}/{scored} pre-drawdown dates; "
                  f"missed the direction on {missed}/{scored}")
            drift = _train_target_mean(symbol, args.horizon)
            if drift is not None:
                print(f"       RF training-target mean is {drift:+.2%} per {args.horizon}d — "
                      f"the drift the model regresses toward when features carry no signal.")
            if upward == scored:
                print("       Every pre-drawdown forecast points up: the bias is SYSTEMATIC.")
            elif upward == 0:
                print("       No upward lean on these dates.")
            else:
                print(f"       Leans up ({upward}/{scored}) but is not rigidly upward — "
                      "the model does emit negative forecasts.")

    print("\n[2] RandomForest bound check")
    print("    Is the forecast reachable from the RF's training targets?\n")
    bounds = rf_output_bounds(symbol, args.horizon)
    if bounds is None:
        print(f"    skipped — no RF bundle for {symbol} h={args.horizon}")
    else:
        print(f"    RF reachable return range: "
              f"[{bounds['min_return']:+.4f}, {bounds['max_return']:+.4f}]")
        print(f"    i.e. a {args.horizon}d return between "
              f"{bounds['min_return']:+.2%} and {bounds['max_return']:+.2%}")
        live = live_bound_check(symbol, args.horizon)
        if live.get("status") == "skipped" or live.get("status") == "error":
            print(f"    live check skipped: {live.get('reason')}")
        else:
            print(f"\n    live RF predicted return : {live['predicted_return']:+.6f}")
            print(f"    within reachable range   : {live['within_bounds']}  -> {live['status']}")
            print(f"    implied price            : {live['implied_price']}")
            print(f"    max price reachable now  : {live['max_price_reachable_from_current']}")
        print("\n    The forecast PRICE is current_price x (1 + that return), so it")
        print("    is not bounded by any price the model ever saw — only the")
        print("    return is. Check the return, not the price level.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

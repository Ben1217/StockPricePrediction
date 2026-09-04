"""
The experiment: does Kronos beat the existing models at next-day forecasting?

Runs every model over identical walk-forward folds on identical features and
targets, scores price and direction separately, and prints a head-to-head table.
The point is to answer the question with evidence, not to assume the newest
model wins -- so the majority-class base rate is reported next to every accuracy
figure, and the fold-to-fold spread next to every mean.

Usage
-----
    python scripts/unified_benchmark.py --symbols AAPL,MSFT --n-splits 3
    python scripts/unified_benchmark.py --models unified_xgboost,unified_kronos
    python scripts/unified_benchmark.py --interval 15m --test-size 200   # intraday
    python scripts/unified_benchmark.py --quick        # small folds, fast smoke test

Kronos is a transformer forward pass per test row. On CPU expect roughly ten
seconds a row, so a 3 x 63-row run takes around half an hour; --kronos-samples
trades Monte-Carlo precision for time. On CUDA it is minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ohlcv_cache import cached_download  # noqa: E402
from src.features.direction_features import build_direction_dataset  # noqa: E402
from src.models.unified_evaluation import (  # noqa: E402
    evaluate_unified_walk_forward,
    summarise_comparison,
)
from src.models.unified_models import (  # noqa: E402
    BENCHMARK_MODELS,
    DEFAULT_BENCHMARK_MODELS,
    DEFAULT_MODEL_PARAMS,
    build_unified_model,
    foundation_model_availability,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

ARTIFACTS_DIR = Path("artifacts")
LOOKBACK_DAYS = 1825



# yfinance will not serve intraday bars from five years ago. These are its
# published limits, and asking past them returns an empty frame rather than an
# error, so the ceiling is applied here instead of being discovered at runtime.
MAX_LOOKBACK_DAYS = {"1m": 7, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "60m": 729, "1h": 729}


def fetch_data(symbol: str, interval: str = "1d", lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Historical bars at the requested interval, from the on-disk cache when warm.

    The interval is a parameter rather than a constant because "the next
    timeframe" is whatever bar the model was trained on: daily bars predict the
    next trading day, 15-minute bars predict the next 15-minute candle. Nothing
    downstream is daily-specific -- the label is ``Close.shift(-horizon)`` and
    the indicators are window counts, both of which are bar-agnostic.
    """
    lookback_days = min(lookback_days, MAX_LOOKBACK_DAYS.get(interval, lookback_days))
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = end - pd.Timedelta(days=lookback_days)

    def _download() -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw

    frame = cached_download(
        symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), interval, _download
    )
    if frame is None or frame.empty:
        raise ValueError(f"No {interval} data returned for {symbol}")
    return frame.sort_index().ffill().dropna()


def run_symbol(
    symbol: str,
    model_names: List[str],
    *,
    test_size: int,
    n_splits: int,
    kronos_samples: int,
    kronos_lookback: int,
    interval: str = "1d",
    horizon: int = 1,
) -> List[Dict[str, Any]]:
    """Evaluate every requested model on one symbol, over the same folds."""
    logger.info("=== %s (%s bars, horizon %d) ===", symbol, interval, horizon)
    raw = fetch_data(symbol, interval=interval)
    dataset = build_direction_dataset(raw, horizon=horizon)

    X = dataset.features
    y_return = dataset.forward_return.to_numpy(dtype=np.float64)
    y_direction = dataset.labels.to_numpy(dtype=np.int8)

    # The close at the decision date: the price a forecast is made *from*, and
    # what the predicted return is applied to. It comes from the dataset's own
    # aligned OHLCV so it cannot drift out of step with the label.
    if dataset.ohlcv is not None and "Close" in dataset.ohlcv.columns:
        prev_close = dataset.ohlcv["Close"].to_numpy(dtype=np.float64)
    else:
        prev_close = raw["Close"].reindex(X.index).to_numpy(dtype=np.float64)

    logger.info(
        "%s: %d aligned rows, %d features, base rate %.4f",
        symbol,
        len(X),
        len(dataset.feature_columns),
        dataset.base_rate,
    )

    results: List[Dict[str, Any]] = []
    for name in model_names:
        # Hyperparameters come from DEFAULT_MODEL_PARAMS unless this is Kronos,
        # whose only knobs are the sampling budget the CLI sets.
        params = (
            {"sample_count": kronos_samples, "lookback": kronos_lookback}
            if name == "unified_kronos"
            else DEFAULT_MODEL_PARAMS.get(name)
        )

        try:
            model = build_unified_model(name, params)
        except Exception as exc:  # noqa: BLE001 - an unavailable optional model is not a failure
            logger.warning("Skipping %s: %s", name, exc)
            continue

        if getattr(model, "requires_ohlcv", False):
            model.set_ohlcv_context(raw)

        started = time.perf_counter()
        metrics = evaluate_unified_walk_forward(
            model,
            X,
            y_return,
            y_direction,
            prev_close,
            test_size=test_size,
            n_splits=n_splits,
            # Labels resolve `horizon` bars out. The splitter purges that many
            # rows off the training tail and embargoes as many again, and the
            # Diebold-Mariano lag is derived from it too -- so the horizon has
            # to travel with the call rather than only being stamped on the
            # result afterwards.
            horizon=horizon,
        )
        if not metrics:
            logger.warning("%s produced no results on %s", name, symbol)
            continue

        metrics["symbol"] = symbol
        metrics["interval"] = interval
        metrics["horizon"] = horizon
        metrics["fit_seconds"] = round(time.perf_counter() - started, 1)
        results.append(metrics)
        logger.info(
            "%s / %s: accuracy %.4f, MAPE %.4f%%, %.1fs",
            symbol,
            name,
            metrics.get("direction_accuracy", float("nan")),
            metrics.get("price_mape", float("nan")),
            metrics["fit_seconds"],
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

PRICE_COLUMNS = [
    "symbol",
    "model_name",
    "price_mae",
    "price_rmse",
    "price_mape",
    "price_r2",
    "price_r2_return",
]
DIRECTION_COLUMNS = [
    "symbol",
    "model_name",
    "base_rate",
    "direction_accuracy",
    "direction_precision",
    "direction_recall",
    "direction_f1",
    "direction_roc_auc",
    "direction_brier_score",
]


def _table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        return frame.to_string(index=False)


def _print_section(title: str, frame: pd.DataFrame, columns: List[str]) -> None:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return
    print(f"\n--- {title} ---")
    print(_table(frame[available]))


def _null_table(results: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """
    Each model against the two nulls it has to clear, pooled across folds.

    Every column here is a verdict rather than a score. ``r2_vs_rw`` below zero
    means the random walk predicted the price better; ``beats_rw`` is only true
    when Diebold-Mariano agrees the gap is real, so a model cannot claim the win
    on a favourable mean alone.
    """
    rows = []
    for result in results:
        evaluation = result.get("evaluation") or {}
        if not evaluation.get("available"):
            continue
        random_walk = evaluation.get("vs_random_walk") or {}
        dm = random_walk.get("diebold_mariano") or {}
        direction = evaluation.get("direction") or {}
        edge = evaluation.get("edge_vs_majority") or {}

        eobr = direction.get("eobr")
        r2 = random_walk.get("r2_vs_random_walk")
        differential = dm.get("mean_differential")
        p_value = dm.get("p_value")

        rows.append(
            {
                "symbol": result.get("symbol", ""),
                "model": result.get("model_name", ""),
                "n": evaluation.get("n"),
                "eobr_pp": None if eobr is None else round(eobr * 100, 2),
                "edge_p": edge.get("p_value_one_sided"),
                "r2_vs_rw": None if r2 is None else round(r2, 4),
                "dm_p": p_value,
                "beats_rw": bool(
                    differential is not None
                    and differential < 0
                    and p_value is not None
                    and p_value < 0.05
                ),
            }
        )
    return pd.DataFrame(rows) if rows else None


def print_report(results: List[Dict[str, Any]], output_dir: Path) -> None:
    if not results:
        logger.error("No results to report")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    # "per_fold" and "evaluation" hold nested structures. A DataFrame column of
    # dicts survives to_csv only as stringified Python, which reads back as
    # nothing useful, so they are kept out of the flat tables and live in the
    # JSON alone.
    NESTED = {"per_fold", "evaluation", "split_protocol"}
    aggregate = pd.DataFrame([{k: v for k, v in r.items() if k not in NESTED} for r in results])

    print("\n" + "=" * 78)
    print("UNIFIED BENCHMARK - walk-forward, mean across folds")
    print("=" * 78)
    _print_section("Price forecasting", aggregate, PRICE_COLUMNS)
    _print_section("Direction classification", aggregate, DIRECTION_COLUMNS)

    comparison = summarise_comparison(results)
    if comparison is not None:
        print("\n--- Head to head (ranked by directional accuracy) ---")
        print(_table(comparison))
        print(
            "\nedge_pp is accuracy minus the majority-class rate on the same test windows,\n"
            "in percentage points. At or below zero the model adds nothing over always\n"
            "guessing the more common direction. p_value is one-sided; n_required is how\n"
            "many test days an edge that size would need to clear 5% significance."
        )
        comparison.to_csv(output_dir / "benchmark_comparison.csv", index=False)

    nulls = _null_table(results)
    if nulls is not None:
        print("\n--- Against the nulls (pooled across folds) ---")
        print(_table(nulls))
        print(
            "\nr2_vs_rw is measured in log-return space against the random walk, which\n"
            "forecasts no change. Below zero the random walk predicted the price better.\n"
            "dm_p is Diebold-Mariano on the paired squared-loss differential; beats_rw is\n"
            "true only when the model is ahead AND that gap clears 5%."
        )
        nulls.to_csv(output_dir / "benchmark_nulls.csv", index=False)

    fold_rows = [
        {
            "symbol": r.get("symbol", ""),
            "model_name": r.get("model_name", ""),
            **{k: v for k, v in fold.items() if k != "predictions"},
        }
        for r in results
        for fold in r.get("per_fold", [])
    ]
    if fold_rows:
        folds = pd.DataFrame(fold_rows)
        print("\n--- Per fold ---")
        columns = [
            c
            for c in [
                "symbol",
                "model_name",
                "fold",
                "test_start",
                "test_end",
                "test_size",
                "base_rate",
                "direction_accuracy",
                "direction_f1",
                "price_mape",
            ]
            if c in folds.columns
        ]
        print(_table(folds[columns]))
        folds.to_csv(output_dir / "benchmark_per_fold.csv", index=False)

    aggregate.to_csv(output_dir / "benchmark_results.csv", index=False)
    with open(output_dir / "benchmark_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    logger.info("Wrote benchmark artifacts to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_models(requested: Optional[str]) -> List[str]:
    """
    Turn the --models flag into a runnable list.

    An optional foundation model whose package is not installed is dropped with
    a note rather than crashing the run: the comparison is still valid without
    it, and the brief marks TimesFM and Chronos as optional.
    """
    names = (
        [n.strip() for n in requested.split(",") if n.strip()]
        if requested
        else list(DEFAULT_BENCHMARK_MODELS)
    )

    resolved: List[str] = []
    for name in names:
        if name not in BENCHMARK_MODELS:
            raise SystemExit(f"Unknown model {name!r}. Known: {', '.join(BENCHMARK_MODELS)}")
        if name in ("unified_kronos", "unified_timesfm", "unified_chronos"):
            available, reason = foundation_model_availability(name)
            if not available:
                logger.warning("Skipping %s - not installed (%s)", name, reason)
                continue
        resolved.append(name)

    if not resolved:
        raise SystemExit("No runnable models were selected")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default="AAPL", help="Comma-separated tickers, e.g. AAPL,MSFT")
    parser.add_argument("--symbol", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--models", default=None, help=f"Subset of: {', '.join(BENCHMARK_MODELS)}")
    parser.add_argument("--test-size", type=int, default=63, help="Rows per test fold (63 = one quarter)")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of walk-forward folds")
    parser.add_argument("--kronos-samples", type=int, default=64, help="Monte Carlo draws per Kronos row")
    parser.add_argument("--kronos-lookback", type=int, default=128, help="Bars of context per Kronos row")
    parser.add_argument(
        "--interval",
        default="1d",
        choices=["1d", "1h", "60m", "30m", "15m", "5m"],
        help="Bar size. The forecast is always the next bar, so this sets the timeframe.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Bars ahead the label resolves. 1 is the intended case; larger values are embargoed by that many rows.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small, fast folds for a smoke test. Not a result worth reporting.",
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()

    if args.quick:
        args.test_size, args.n_splits, args.kronos_samples = 20, 2, 16

    symbols = (
        [args.symbol.strip().upper()]
        if args.symbol
        else [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    )
    if not symbols:
        parser.error("At least one symbol is required")

    model_names = resolve_models(args.models)
    logger.info("Benchmarking %s on %s", ", ".join(model_names), ", ".join(symbols))

    results: List[Dict[str, Any]] = []
    for symbol in symbols:
        try:
            results.extend(
                run_symbol(
                    symbol,
                    model_names,
                    test_size=args.test_size,
                    n_splits=args.n_splits,
                    kronos_samples=args.kronos_samples,
                    kronos_lookback=args.kronos_lookback,
                    interval=args.interval,
                    horizon=args.horizon,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not lose the others
            logger.error("Benchmark failed for %s: %s", symbol, exc, exc_info=True)

    print_report(results, args.output_dir)


if __name__ == "__main__":
    main()

"""
Next-day direction classifier: walk-forward evaluation and long/flat backtest.

Answers one question and refuses to answer it vaguely: does P(up tomorrow),
predicted from 46 engineered features or from raw OHLCV candlestick sequences,
beat the naive baselines out-of-sample after costs?

Every model emits one prediction per day, recomputed from that day's chart: a
direction with a probability, and a price *range* rather than a point. Nothing
here interpolates a single horizon-end number across the days in between.

Supports standard estimators (logistic, gradient boosting) as well as
foundation models:
  - TabPFN v2 (tabular foundation model for engineered features)
  - Kronos (candlestick foundation model for raw OHLCV sequences)
  - Foundation Ensemble (averaged probability of TabPFN + Kronos with price bands)

Examples:
    python scripts/direction_backtest.py --ticker AAPL --start 2015-01-01
    python scripts/direction_backtest.py --ticker MSFT --model tabpfn --cost-bps 10
    python scripts/direction_backtest.py --ticker SPY --model kronos
    python scripts/direction_backtest.py --ticker AAPL,MSFT,NVDA --all-models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.direction_backtest import DEFAULT_THRESHOLD_GRID  # noqa: E402
from src.data.direction_data import MIN_USABLE_ROWS, load_daily_bars  # noqa: E402
from src.features.direction_features import (  # noqa: E402
    DIRECTION_BASE_FEATURE_COLUMNS,
    DIRECTION_FEATURE_COLUMNS,
    build_direction_dataset,
)
from src.models.direction_models import ALL_FACTORIES, MODEL_FACTORIES  # noqa: E402
from src.models.direction_pipeline import (  # noqa: E402
    DEFAULT_COST_BPS,
    DEFAULT_MIN_TRAIN,
    DEFAULT_N_FOLDS,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    run_walk_forward,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_OUT_DIR = Path("data/direction_backtests")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="direction_backtest",
        description="Walk-forward next-day direction classifier with a costed long/flat backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker", "--tickers", dest="tickers", nargs="+", required=True,
        help="One or more symbols (e.g. AAPL, MSFT NVDA, or ^GSPC)",
    )
    parser.add_argument("--start", default=None, help="First bar, YYYY-MM-DD (default: 10 years back)")
    parser.add_argument("--end", default=None, help="Last bar, YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--model", default="logistic", choices=sorted(ALL_FACTORIES),
        help="Estimator to evaluate. Baselines are always scored alongside it",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Evaluate every real model in turn (baselines run inside each anyway)",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_N_FOLDS, help="Walk-forward folds")
    parser.add_argument(
        "--test-size", type=int, default=DEFAULT_TEST_SIZE,
        help="Rows per test window (63 = one trading quarter)",
    )
    parser.add_argument(
        "--min-train", type=int, default=DEFAULT_MIN_TRAIN,
        help="Minimum training rows for a fold to be kept",
    )
    parser.add_argument(
        "--embargo", type=int, default=None,
        help="Rows purged between train and test (default: the target horizon)",
    )
    parser.add_argument(
        "--cost-bps", type=float, default=DEFAULT_COST_BPS,
        help="Round-trip cost in basis points, charged on every active day",
    )
    parser.add_argument(
        "--horizon", type=int, default=1,
        help="Bars ahead the label resolves over. 1 is the supported case",
    )
    parser.add_argument(
        "--deadband", type=float, default=0.0,
        help="Drop rows whose forward return is inside +/- this many 20-day sigmas. "
             "Cuts label noise but changes the base rate, which is reported",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Fix the long/flat threshold instead of choosing it on validation",
    )
    parser.add_argument(
        "--threshold-objective", default="sharpe", choices=["sharpe", "total_return"],
        help="What the validation window maximises when choosing the threshold",
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.0,
        help="Annual rate subtracted before the Sharpe ratio, for both strategy and benchmark",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for every fit and draw")
    parser.add_argument(
        "--min-rows", type=int, default=MIN_USABLE_ROWS,
        help="Minimum usable daily bars required after cleaning",
    )
    parser.add_argument(
        "--allow-short-history", action="store_true",
        help="Proceed below --min-rows. The accuracy figures will not be separable from noise",
    )
    parser.add_argument(
        "--feature-set", default="full", choices=["full", "base"],
        help="'full' is the 46-column set including the chart-pattern block; 'base' is "
             "the original 19, for an ablation showing what the pattern columns bought",
    )
    parser.add_argument(
        "--kronos-samples", type=int, default=None,
        help="Kronos: autoregressive paths sampled per day. The Monte-Carlo error on "
             "P(up) is sqrt(0.25/N), and cost is linear in N",
    )
    parser.add_argument(
        "--kronos-lookback", type=int, default=None,
        help="Kronos: bars of context per prediction. Attention is quadratic in this, "
             "so halving it roughly quarters the runtime",
    )
    parser.add_argument(
        "--kronos-device", default=None,
        help="Kronos: force a torch device (cuda:0, mps, cpu). Auto-detected otherwise",
    )
    parser.add_argument("--no-cache", action="store_true", help="Force a fresh download")
    parser.add_argument("--skip-leakage-check", action="store_true", help="Skip the shuffled-label check")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Where reports are written")
    parser.add_argument("--no-write", action="store_true", help="Print the summary without writing files")
    return parser


def _format_optional(value: Optional[float], spec: str = "+.4f", missing: str = "n/a") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else missing


def print_summary(report: Dict[str, Any], predictions: Optional[Any] = None) -> None:
    """Human-readable digest. Every accuracy is printed with its interval."""
    config, data, pooled = report["config"], report.get("data", {}), report["pooled"]
    model_metrics = pooled["model"]
    backtest = pooled["backtest"]
    skill = model_metrics["skill"]

    print()
    print("=" * 78)
    print(f"  {data.get('ticker', '?')}  |  {config['model']}  |  next-day direction (h={config['horizon']})")
    print("=" * 78)

    print("\nDATA")
    print(f"  bars            {data.get('clean_rows', '?')} rows  "
          f"{data.get('first_bar', '?')} .. {data.get('last_bar', '?')}")
    print(f"  price basis     {data.get('price_basis', '?')}  "
          f"({data.get('dividend_events', 0)} dividends, {data.get('split_events', 0)} splits)")
    print(f"  content sha256  {str(data.get('content_sha256', ''))[:16]}")
    print(f"  dataset         {report['dataset']['n_rows']} labelled rows x "
          f"{report['dataset']['n_features']} features, base rate "
          f"{report['dataset']['base_rate']:.4f}")
    if report["dataset"]["rows_dropped_deadband"]:
        print(f"  deadband        dropped {report['dataset']['rows_dropped_deadband']} rows at "
              f"{report['dataset']['deadband_sigma_multiple']} sigma -> base rate shifted")

    print("\nWALK-FORWARD")
    if config.get("n_features"):
        print(f"  {config['n_features']} features")
    if config.get("model_kwargs"):
        print(f"  model options   {config['model_kwargs']}")
    print(f"  {config['n_folds_run']} expanding folds, {config['test_size']} test rows each, "
          f"embargo {config['embargo']}, min train {config['min_train']}")
    print(f"  {config['execution']}")
    print(f"  {'fold':<5}{'test window':<26}{'acc':>8}{'95% CI':>18}{'AUC':>8}{'thr':>7}{'net':>9}")
    for fold in report["folds"]:
        metrics = fold["model"]
        window = f"{fold['test_range'][0]}..{fold['test_range'][1]}"
        ci = f"[{metrics['accuracy_ci_low']:.3f},{metrics['accuracy_ci_high']:.3f}]"
        print(f"  {fold['fold']:<5}{window:<26}{metrics['accuracy']:>8.4f}{ci:>18}"
              f"{_format_optional(metrics['roc_auc'], '.3f'):>8}"
              f"{fold['threshold']['threshold']:>7.2f}"
              f"{100 * fold['backtest']['strategy']['total_return']:>8.2f}%")

    print(f"\nPOOLED OUT-OF-SAMPLE  ({pooled['n_test_rows']} days, "
          f"{pooled['test_range'][0]} .. {pooled['test_range'][1]})")
    print(f"  accuracy            {model_metrics['accuracy']:.4f}  "
          f"95% CI [{model_metrics['accuracy_ci_low']:.4f}, {model_metrics['accuracy_ci_high']:.4f}]")
    print(f"  balanced accuracy   {_format_optional(model_metrics['balanced_accuracy'], '.4f')}")
    print(f"  base rate (test)    {model_metrics['base_rate']:.4f}   "
          f"predicted up rate {model_metrics['predicted_up_rate']:.4f}")
    print(f"  AUC / Brier / logl  {_format_optional(model_metrics['roc_auc'], '.4f')} / "
          f"{model_metrics['brier_score']:.4f} / {model_metrics['log_loss']:.4f}")
    print(f"  MCC                 {model_metrics['mcc']:+.4f}")
    print(f"  precision/recall    up {_format_optional(model_metrics['class_up']['precision'], '.4f')}"
          f"/{_format_optional(model_metrics['class_up']['recall'], '.4f')}   "
          f"down {_format_optional(model_metrics['class_down']['precision'], '.4f')}"
          f"/{_format_optional(model_metrics['class_down']['recall'], '.4f')}")
    print(f"  skill vs constant   Brier {skill['brier_skill_score']:+.4f}   "
          f"log loss {skill['log_loss_skill_score']:+.4f}   "
          f"prediction std {skill['prediction_std']:.4f}")

    if predictions is not None and "price_lo_5" in predictions.columns:
        valid_bands = predictions.dropna(subset=["price_lo_5", "price_hi_95"])
        if not valid_bands.empty:
            last_p = valid_bands.iloc[-1]
            print("\nPRICE FORECAST RANGE (last out-of-sample bar)")
            print(f"  5th percentile (low)     ${last_p['price_lo_5']:.2f}")
            print(f"  50th percentile (median) ${last_p['price_median']:.2f}")
            print(f"  95th percentile (high)   ${last_p['price_hi_95']:.2f}")

    print("\nBASELINES (same folds, same training windows)")
    print(f"  {'baseline':<14}{'acc':>8}{'95% CI':>18}{'bal acc':>10}{'MCC':>9}")
    for name, metrics in pooled["baselines"].items():
        ci = f"[{metrics['accuracy_ci_low']:.3f},{metrics['accuracy_ci_high']:.3f}]"
        print(f"  {name:<14}{metrics['accuracy']:>8.4f}{ci:>18}"
              f"{_format_optional(metrics['balanced_accuracy'], '.4f'):>10}"
              f"{metrics['mcc']:>+9.4f}")
    edge = pooled["edge_vs_best_baseline"]
    print(f"\n  model vs best baseline ({pooled['best_baseline']}): "
          f"{edge['edge_pp']:+.2f}pp, SE {edge['standard_error_pp']:.2f}pp, "
          f"z={edge['z']:.2f}, one-sided p={edge['p_value_one_sided']:.4f}")
    print(f"  {'SIGNIFICANT' if edge['significant'] else 'NOT significant'} at alpha={edge['alpha']}"
          + (f" -- an edge this size needs ~{edge['n_required']} test days"
             if edge.get("n_required") else ""))

    print("\nBACKTEST (long/flat, costs charged per active day)")
    strategy, benchmark, breakeven = backtest["strategy"], backtest["benchmark"], backtest["breakeven"]
    print(f"  {'':<22}{'strategy':>14}{'buy & hold':>14}")
    for label, key, spec in [
        ("total return", "total_return", ".2%"),
        ("CAGR", "cagr", ".2%"),
        ("Sharpe (x sqrt252)", "sharpe", ".2f"),
        ("max drawdown", "max_drawdown", ".2%"),
        ("hit rate", "hit_rate", ".2%"),
        ("avg win", "avg_win", ".3%"),
        ("avg loss", "avg_loss", ".3%"),
        ("time in market %", "time_in_market_pct", ".1f"),
    ]:
        left = _format_optional(strategy.get(key), spec)
        right = _format_optional(benchmark.get(key), spec)
        print(f"  {label:<22}{left:>14}{right:>14}")
    print(f"  {'round trips/year':<22}{_format_optional(strategy.get('round_trips_per_year'), '.1f'):>14}"
          f"{'1 total':>14}")
    print(f"\n  cost charged        {breakeven['cost_charged_bps']:.1f} bps round trip")
    edge_per_trade = _format_optional(breakeven["mean_gross_return_per_trade_bps"], ".2f")
    print(f"  edge per trade      {edge_per_trade} bps gross")
    positive = breakeven["breakeven_cost_bps_positive"]
    print(f"  BREAKEVEN COST      {_format_optional(positive, '.2f')} bps "
          f"-- the edge dies above this")
    vs_benchmark = _format_optional(breakeven["breakeven_cost_bps_vs_buy_and_hold"], ".2f")
    print(f"  vs buy & hold at    {vs_benchmark} bps")

    band = pooled.get("price_band")
    if band and band.get("n"):
        source = (report["folds"][0].get("price_band") or {}).get("source", "?")
        print("\nPRICE BAND (5th / 50th / 95th percentile of tomorrow's close)")
        print(f"  source              {source}")
        print(f"  coverage            {band['coverage']:.1%} of actual closes landed inside, "
              f"against a nominal {band['nominal_coverage']:.0%}")
        gap = band["coverage_gap"]
        reading = ("well calibrated" if abs(gap) <= 0.05
                   else "too narrow, so the band is overconfident" if gap < 0
                   else "too wide to constrain anything")
        print(f"  calibration         {gap:+.1%} vs nominal -- {reading}")
        print(f"  mean width          {band['mean_relative_width']:.2%} of price")
        print(f"  pinball loss        {band['pinball_loss']:.4f}   "
              f"median bias {band['median_bias_relative']:+.3%}")

    leakage = report.get("leakage_check")
    if leakage:
        state = {True: "PASS", False: "FAIL", None: "INCONCLUSIVE"}[leakage["passed"]]
        print(f"\nLEAKAGE CHECK  {state}")
        print(f"  shuffled-label accuracy {leakage['mean_shuffled_accuracy']:.4f} vs "
              f"no-relationship expectation {leakage['mean_null_accuracy']:.4f}")
        print(f"  mean difference {leakage['mean_difference']:+.4f} +/- "
              f"{leakage['standard_error_of_mean']:.4f} over {leakage['n_fits']} fits"
              + (f", p={leakage['p_value_one_sided']:.4f}"
                 if leakage.get("p_value_one_sided") is not None else ""))
        if leakage["passed"] is False:
            print("  Permuted labels still predict the test window. Every number above is void.")

    verdict = report["verdict"]
    print("\nVERDICT")
    for name, passed in verdict["criteria"].items():
        print(f"  [{'x' if passed else ' '}] {name}")
    print(f"\n  {'SHIP' if verdict['ship'] else 'DO NOT SHIP'}: {verdict['summary']}")
    print("=" * 78)


def print_comparison_table(ticker: str, results: Dict[str, Any], cost_bps: float) -> None:
    """Print side-by-side comparison across evaluated models."""
    if len(results) <= 1:
        return

    print("\n" + "=" * 90)
    print(f"  MODEL COMPARISON SUMMARY FOR {ticker} (Pooled Out-of-Sample after {cost_bps:.1f} bps costs)")
    print("=" * 90)
    print(f"  {'model':<22}{'acc':>8}{'edge (pp)':>12}{'p-value':>10}{'breakeven':>14}{'net return':>13}{'ship':>7}")
    print("  " + "-" * 86)

    for name, res in results.items():
        pooled = res.report["pooled"]
        m = pooled["model"]
        edge = pooled["edge_vs_best_baseline"]
        bt = pooled["backtest"]
        breakeven = bt["breakeven"].get("breakeven_cost_bps_positive")
        total_ret = bt["strategy"].get("total_return")

        acc_str = f"{m['accuracy']:.4f}"
        edge_str = f"{edge['edge_pp']:+.2f}pp"
        p_val_str = f"{edge['p_value_one_sided']:.4f}" if edge.get("p_value_one_sided") is not None else "n/a"
        bk_str = f"{breakeven:.1f} bps" if breakeven is not None else "n/a"
        ret_str = f"{100 * total_ret:+.2f}%" if total_ret is not None else "n/a"
        ship_str = "YES" if res.ship else "NO"

        print(f"  {name:<22}{acc_str:>8}{edge_str:>12}{p_val_str:>10}{bk_str:>14}{ret_str:>13}{ship_str:>7}")

    print("=" * 90)


def write_outputs(out_dir: Path, ticker: str, model_name: str, result) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ticker.upper().replace('^', '')}_{model_name}"

    report_path = out_dir / f"{stem}_report.json"
    report_path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")

    equity_path = out_dir / f"{stem}_equity_curve.csv"
    result.equity_curve.to_csv(equity_path)

    predictions_path = out_dir / f"{stem}_predictions.csv"
    result.predictions.to_csv(predictions_path)

    return [report_path, equity_path, predictions_path]


def parse_tickers(raw_tickers: List[str]) -> List[str]:
    """Support space-separated and comma-separated ticker inputs."""
    tickers = []
    for item in raw_tickers:
        for symbol in item.split(","):
            cleaned = symbol.strip().upper()
            if cleaned and cleaned not in tickers:
                tickers.append(cleaned)
    return tickers


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ticker_list = parse_tickers(args.tickers)

    if not ticker_list:
        print("ERROR: No valid tickers provided.", file=sys.stderr)
        return 1

    all_shipped = True
    models = sorted(MODEL_FACTORIES) if args.all_models else [args.model]

    # Kronos options only mean anything to the slots that run Kronos; handing
    # them to the logistic constructor would be a TypeError, so they are
    # attached per model rather than globally.
    kronos_options = {
        key: value for key, value in (
            ("sample_count", args.kronos_samples),
            ("lookback", args.kronos_lookback),
            ("device", args.kronos_device),
        ) if value is not None
    }
    feature_columns = (
        DIRECTION_FEATURE_COLUMNS if args.feature_set == "full"
        else DIRECTION_BASE_FEATURE_COLUMNS
    )

    for ticker in ticker_list:
        print(f"\n>>> Running evaluation for ticker: {ticker} (10-year history window)")
        try:
            bars = load_daily_bars(
                ticker, args.start, args.end,
                use_cache=not args.no_cache,
                min_rows=args.min_rows,
                require_min_rows=not args.allow_short_history,
            )
            dataset = build_direction_dataset(
                bars.frame,
                horizon=args.horizon,
                deadband_sigma_multiple=args.deadband,
                feature_columns=feature_columns,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not build dataset for %s: %s", ticker, exc)
            print(f"\nERROR ({ticker}): {exc}\n", file=sys.stderr)
            all_shipped = False
            continue

        ticker_results: Dict[str, Any] = {}

        for model_name in models:
            try:
                result = run_walk_forward(
                    dataset,
                    model_name=model_name,
                    model_kwargs=(
                        kronos_options
                        if model_name in ("kronos", "foundation_ensemble") else {}
                    ),
                    n_folds=args.folds,
                    test_size=args.test_size,
                    embargo=args.embargo,
                    min_train=args.min_train,
                    cost_bps=args.cost_bps,
                    threshold_grid=DEFAULT_THRESHOLD_GRID,
                    threshold_objective=args.threshold_objective,
                    fixed_threshold=args.threshold,
                    seed=args.seed,
                    risk_free_rate=args.risk_free_rate,
                    data_meta=bars.meta,
                    run_leakage_check=not args.skip_leakage_check,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Walk-forward run failed for %s on %s: %s", model_name, ticker, exc)
                print(f"\nERROR ({ticker} / {model_name}): {exc}\n", file=sys.stderr)
                all_shipped = False
                continue

            print_summary(result.report, predictions=result.predictions)
            ticker_results[model_name] = result

            if not args.no_write:
                written = write_outputs(args.out_dir, ticker, model_name, result)
                print("\nWrote:")
                for path in written:
                    print(f"  {path}")
            all_shipped = all_shipped and result.ship

        if len(ticker_results) > 1:
            print_comparison_table(ticker, ticker_results, args.cost_bps)

    return 0 if all_shipped else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""
Bundle training script.

Builds symbol-aware model bundles under ``models/bundles/<SYMBOL>/<MODEL>/``
so prediction and backtest endpoints can load deterministic per-ticker
artifacts without relying on generic model files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bundle_training import (  # noqa: E402
    DEFAULT_BOOTSTRAP_MODELS,
    DEFAULT_BOOTSTRAP_SYMBOLS,
    DEFAULT_BUNDLE_HORIZONS,
    normalize_horizons,
    resolve_training_symbols,
    train_batch_model_bundles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train symbol-aware next-day direction bundles.")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--sp500", action="store_true", help="Train the full S&P 500 universe")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of symbols to process")
    parser.add_argument("--model-types", nargs="+", default=DEFAULT_BOOTSTRAP_MODELS)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_BUNDLE_HORIZONS)
    parser.add_argument("--lookback-days", type=int, default=756)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--skip-fresh-hours", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = resolve_training_symbols(
        args.symbols if args.symbols else DEFAULT_BOOTSTRAP_SYMBOLS,
        use_sp500=args.sp500,
    )
    if args.limit is not None:
        symbols = symbols[: max(0, int(args.limit))]

    model_types = [str(model_type) for model_type in args.model_types]
    horizons = normalize_horizons(args.horizons)

    print("\n" + "=" * 72)
    print("BATCH TRAINING NEXT-DAY DIRECTION BUNDLES")
    print("=" * 72)
    print(f"Symbols    : {', '.join(symbols[:12])}{' ...' if len(symbols) > 12 else ''}")
    print(f"Count      : {len(symbols)}")
    print(f"Models     : {', '.join(model_types)}")
    print(f"Horizons   : {horizons} (fixed by simplified pipeline)")
    print(f"Lookback   : {args.lookback_days} days")
    print(f"Test Size  : {args.test_size}")
    print(f"Skip Fresh : {args.skip_fresh_hours}h")

    result = train_batch_model_bundles(
        symbols=symbols,
        use_sp500=False,
        model_types=model_types,
        horizons=horizons,
        lookback_days=args.lookback_days,
        test_size=args.test_size,
        skip_fresh_hours=args.skip_fresh_hours,
        progress_callback=lambda completed, total, symbol, model_type, status: print(
            f"[{completed}/{total}] {symbol} / {model_type}: {status}"
        ),
    )

    print("\n" + "=" * 72)
    print("BATCH TRAINING COMPLETE")
    print("=" * 72)
    print(f"Successful runs: {result['success_count']}")
    print(f"Skipped runs   : {result['skipped_count']}")
    print(f"Failed runs    : {result['failure_count']}")
    failures = [run for run in result["runs"] if run["status"] == "failed"]
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure['symbol']} / {failure['model_type']}: {failure['message']}")

    print("\nBundles are stored under models/bundles/<SYMBOL>/<MODEL_TYPE>/")
    return 1 if result["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

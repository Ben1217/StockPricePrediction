from unittest.mock import patch

import pandas as pd

from src.features.feature_engineering import select_feature_columns
from src.models import bundle_training


def _sample_training_frame(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = pd.Series([100 + i * 0.2 for i in range(rows)], index=index)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": [1000 + i for i in range(rows)],
        },
        index=index,
    )


def test_train_batch_model_bundles_skips_fresh_runs():
    with (
        patch.object(bundle_training, "download_training_frame", return_value=_sample_training_frame()),
        patch.object(bundle_training, "is_bundle_fresh", side_effect=lambda symbol, model_type, **_: model_type == "xgboost"),
        patch.object(
            bundle_training,
            "train_model_bundles",
            return_value={"versions": ["bundle_v1"], "horizons": [1]},
        ) as mocked_train,
    ):
        result = bundle_training.train_batch_model_bundles(
            symbols=["AAPL"],
            model_types=["xgboost", "lstm"],
            horizons=[1],
            skip_fresh_hours=24,
        )

    assert result["success_count"] == 1
    assert result["skipped_count"] == 1
    assert result["failure_count"] == 0
    assert mocked_train.call_count == 1
    assert {run["status"] for run in result["runs"]} == {"completed", "skipped"}


def test_select_feature_columns_excludes_all_nan_long_window_features():
    """
    Long-window indicators that are entirely NaN on a short frame must be dropped,
    otherwise a bundle records a feature the model can never actually use.
    """
    frame = pd.DataFrame(
        {
            "Close": [10.0, 11.0, 12.0, 13.0],
            "Target": [0.1, 0.2, 0.3, 0.4],
            "SMA_50": [None, None, None, None],
            "RSI": [55.0, 56.0, 57.0, 58.0],
        }
    )

    feature_columns = select_feature_columns(frame, min_non_null=1)

    # The subject of this test: SMA_50 is all-NaN and must not be selected.
    assert "SMA_50" not in feature_columns
    # A populated indicator is kept.
    assert "RSI" in feature_columns
    # Raw OHLCV columns are intentionally retained — see CORE_REGRESSION_FEATURE_COLUMNS,
    # which declares Open/High/Low/Close/Volume as core model inputs. This assertion is
    # here to make that a deliberate contract rather than an accident.
    assert "Close" in feature_columns


def test_select_feature_columns_excludes_target_and_leakage_columns():
    """The label and forward-looking regression targets must never become features."""
    frame = pd.DataFrame(
        {
            "Close": [10.0, 11.0, 12.0, 13.0],
            "RSI": [55.0, 56.0, 57.0, 58.0],
            "Target": [0, 1, 0, 1],
            "Forward_Return": [0.01, -0.02, 0.03, 0.00],
            "Adj Close": [9.9, 10.9, 11.9, 12.9],
            "target_return_30d": [0.05, 0.06, 0.07, 0.08],
        }
    )

    feature_columns = select_feature_columns(frame, min_non_null=1)

    for leaked in ("Target", "Forward_Return", "Adj Close", "target_return_30d"):
        assert leaked not in feature_columns, f"{leaked} must not be used as a feature"
    assert feature_columns == ["Close", "RSI"]

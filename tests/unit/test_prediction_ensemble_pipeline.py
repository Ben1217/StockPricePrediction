import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.api.schemas.schemas import EnsemblePredictRequest, EnsembleTrainRequest
from src.features.feature_engineering import (
    build_regression_dataset,
    build_regression_feature_frame,
    transform_feature_frame,
)
from src.models.ensemble_predictor import (
    _build_forecast_points,
    _compute_weights,
    _metadata_is_return_regression,
    _spec_reliability_score,
)


def _sample_ohlcv(rows: int = 320) -> pd.DataFrame:
    index = pd.date_range("2022-01-03", periods=rows, freq="B")
    base = np.linspace(100.0, 180.0, rows)
    close = pd.Series(base + np.sin(np.arange(rows) / 8.0), index=index)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000 + np.arange(rows) * 100,
        },
        index=index,
    )


def test_regression_dataset_creates_shifted_multi_horizon_targets():
    df = _sample_ohlcv()

    dataset, feature_cols, target_col = build_regression_dataset(df, horizon=30)

    assert target_col == "target_return_30d"
    assert {
        "target_return_7d",
        "target_return_15d",
        "target_return_30d",
        "target_return_60d",
    }.issubset(dataset.columns)
    assert target_col not in feature_cols
    assert feature_cols == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Daily_Return",
        "SMA_20",
        "SMA_50",
        "RSI_14",
        "MACD",
        "Volatility",
    ]
    assert {
        "Returns",
        "Log_Returns",
        "SMA_200",
        "EMA_12",
        "EMA_26",
        "High_Low_Range",
        "Close_Lag_10",
    }.issubset(dataset.columns)

    first_idx = dataset.index[0]
    source_pos = df.index.get_loc(first_idx)
    expected = df["Adj Close"].iloc[source_pos + 30] / df["Adj Close"].iloc[source_pos] - 1
    assert dataset.loc[first_idx, "target_return_30d"] == pytest.approx(expected)


def test_regression_feature_frame_supports_legacy_bundle_columns():
    df = _sample_ohlcv()
    feature_frame = build_regression_feature_frame(df)

    legacy_feature_cols = [
        "Returns",
        "Log_Returns",
        "SMA_200",
        "EMA_12",
        "EMA_26",
        "MACD_Signal",
        "BB_High",
        "Volume_Ratio",
        "High_Low_Range",
        "Close_Lag_10",
        "Return_Lag_10",
        "DayOfWeek",
        "Month",
        "Quarter",
        "Price_Momentum",
        "Rolling_Volatility",
    ]

    aligned, X = transform_feature_frame(feature_frame, legacy_feature_cols, scaler=None)

    assert not aligned.empty
    assert X.shape[1] == len(legacy_feature_cols)


def test_ensemble_weights_use_fixed_spec_weights():
    bundles = {
        "xgboost": {"meta": {}},
        "random_forest": {"meta": {}},
        "lstm": {"meta": {}},
    }

    weights = _compute_weights(bundles, current_price=100.0)

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["lstm"] == pytest.approx(0.40)
    assert weights["xgboost"] == pytest.approx(0.35)
    assert weights["random_forest"] == pytest.approx(0.25)


def test_forecast_points_use_each_daily_prediction_value():
    points = _build_forecast_points(
        predicted_price=259.0,
        current_price=270.0,
        horizon=4,
        last_date=pd.Timestamp("2026-04-27"),
        avg_mape=1.0,
        weighted_rmse=1.0,
        spread_pct=0.0,
        recent_volatility=0.01,
        raw_predictions={"xgboost": 258.0, "random_forest": 260.0, "lstm": 259.0},
    )

    predictions = [point["predicted"] for point in points]

    assert len(points) == 4
    assert [point["date"] for point in points] == ["2026-04-28", "2026-04-29", "2026-04-30", "2026-05-01"]
    assert predictions[-1] == pytest.approx(259.0)
    assert len(set(predictions)) > 1
    assert predictions[0] > predictions[-1]


def test_reliability_flags_hard_gap_and_volatility_bounds():
    bundles = {
        "xgboost": {"meta": {"val_metrics": {"mae": 0.01}}},
        "random_forest": {"meta": {"val_metrics": {"mae": 0.01}}},
        "lstm": {"meta": {"val_metrics": {"mae": 0.01}}},
    }

    signal, reliability, reason, *_ = _spec_reliability_score(
        predictions={"lstm": 145.0, "xgboost": 146.0, "random_forest": 144.0},
        current_price=100.0,
        bundles=bundles,
        recent_volatility=0.02,
        horizon=30,
        ensemble_change_pct=45.0,
        confidence_width_pct=6.0,
    )
    assert signal == "Bullish"
    assert reliability == "Low"
    assert "gap" in reason.lower()

    _, reliability, reason, *_ = _spec_reliability_score(
        predictions={"lstm": 110.0, "xgboost": 110.5, "random_forest": 109.5},
        current_price=100.0,
        bundles=bundles,
        recent_volatility=0.005,
        horizon=30,
        ensemble_change_pct=10.0,
        confidence_width_pct=6.0,
    )
    assert reliability == "Low"
    assert "volatility" in reason.lower()


def test_legacy_price_regression_metadata_is_rejected():
    assert _metadata_is_return_regression({"target_type": "return_regression"})
    assert not _metadata_is_return_regression({"target_type": "price_regression", "objective": "future_close_price"})


def test_ensemble_schema_rejects_unsupported_horizons():
    with pytest.raises(ValidationError):
        EnsemblePredictRequest(symbol="AAPL", horizon=10)

    with pytest.raises(ValidationError):
        EnsembleTrainRequest(symbol="AAPL", horizons=[7, 10, 30])

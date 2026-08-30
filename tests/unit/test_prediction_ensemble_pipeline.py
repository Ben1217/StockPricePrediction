import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.api.schemas.schemas import EnsemblePredictRequest, EnsembleTrainRequest
from src.features.feature_engineering import (
     CORE_REGRESSION_FEATURE_COLUMNS,
    STATIONARY_REGRESSION_FEATURE_COLUMNS,
    build_regression_dataset,
    build_regression_feature_frame,
    split_dataset_chronologically,
    transform_feature_frame,
)
from src.models.ensemble_predictor import (
    _build_forecast_points,
    _compute_weights,
    _metadata_is_return_regression,
    _spec_reliability_score,
    bundle_skill_failure,
)
from src.models.ensemble_training import _baseline_skill


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
    assert feature_cols == STATIONARY_REGRESSION_FEATURE_COLUMNS
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
    points, _paths = _build_forecast_points(
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
    assert predictions[-1] == pytest.approx(259.0, abs=0.01)
    assert len(set(predictions)) > 1
    assert predictions[0] > predictions[-1]


def test_forecast_path_compounds_instead_of_interpolating():
    """
    The daily series must be a compounded curve, not a ruler drawn from today's
    price to the model's endpoint. Linear interpolation gives every step an
    identical increment, which is what made every forecast render as a straight
    ray regardless of ticker or horizon.
    """
    current, horizon = 100.0, 30
    points, _paths = _build_forecast_points(
        predicted_price=112.0,
        current_price=current,
        horizon=horizon,
        last_date=pd.Timestamp("2026-04-27"),
        avg_mape=2.0,
        weighted_rmse=3.0,
        spread_pct=1.0,
        recent_volatility=0.015,
        raw_predictions={"xgboost": 113.0, "random_forest": 111.0, "lstm": 112.0},
        seed=7,
    )
    predicted = np.array([point["predicted"] for point in points])
    increments = np.diff(predicted)

    assert predicted[-1] == pytest.approx(112.0, abs=0.01)
    # Interpolation would make every increment identical.
    assert not np.allclose(increments, increments[0], atol=1e-6)
    # Compounding makes each step larger than the one before it.
    assert increments[-1] > increments[0]
    # Every step must sit on the compound curve current * (1 + r) ** (t / horizon).
    expected = current * (1.12 ** (np.arange(1, horizon + 1) / horizon))
    assert predicted == pytest.approx(expected, abs=0.01)


def test_forecast_bands_widen_with_horizon_and_never_invert():
    points, _paths = _build_forecast_points(
        predicted_price=112.0,
        current_price=100.0,
        horizon=30,
        last_date=pd.Timestamp("2026-04-27"),
        avg_mape=2.0,
        weighted_rmse=3.0,
        spread_pct=1.0,
        recent_volatility=0.015,
        raw_predictions={"xgboost": 112.0},
        seed=7,
    )

    for point in points:
        assert point["lower_95"] <= point["lower_68"] <= point["predicted"]
        assert point["predicted"] <= point["upper_68"] <= point["upper_95"]

    widths = [point["upper_95"] - point["lower_95"] for point in points]
    assert widths[-1] > widths[0]


def test_forecast_scenario_paths_are_rough_and_anchored_to_today():
    """
    The fan-chart paths carry the volatility. The median line stays smooth, so the
    paths are the only place a viewer can read what the spread actually looks like.
    """
    current = 100.0
    points, paths = _build_forecast_points(
        predicted_price=112.0,
        current_price=current,
        horizon=30,
        last_date=pd.Timestamp("2026-04-27"),
        avg_mape=2.0,
        weighted_rmse=3.0,
        spread_pct=1.0,
        recent_volatility=0.015,
        raw_predictions={"xgboost": 112.0},
        seed=7,
    )

    assert paths, "expected scenario paths for the fan chart"
    for path in paths:
        assert len(path) == 31  # today's anchor plus one point per step
        assert path[0] == pytest.approx(current, abs=0.01)
        assert all(value > 0 for value in path)

    median_roughness = np.std(np.diff([point["predicted"] for point in points]))
    path_roughness = np.mean([np.std(np.diff(path)) for path in paths])
    assert path_roughness > median_roughness * 5

    terminals = [path[-1] for path in paths]
    assert max(terminals) > min(terminals)


def test_forecast_is_reproducible_for_the_same_inputs():
    """The chart must not move when a user refreshes the page."""
    kwargs = dict(
        predicted_price=112.0,
        current_price=100.0,
        horizon=15,
        last_date=pd.Timestamp("2026-04-27"),
        avg_mape=2.0,
        weighted_rmse=3.0,
        spread_pct=1.0,
        recent_volatility=0.015,
        raw_predictions={"xgboost": 112.0},
        seed=99,
    )
    first_points, first_paths = _build_forecast_points(**kwargs)
    second_points, second_paths = _build_forecast_points(**kwargs)

    assert first_points == second_points
    assert first_paths == second_paths


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


def test_regression_features_are_free_of_price_levels():
    """
    Return regressors must not see absolute price levels. A scaler fitted on a
    training window below present-day prices pushes every recent bar past the
    range the trees learned, where they can only repeat the forward return that
    window happened to contain.
    """
    price_levels = {"Open", "High", "Low", "Close", "SMA_20", "SMA_50", "MACD", "Volume"}
    assert price_levels.isdisjoint(STATIONARY_REGRESSION_FEATURE_COLUMNS)

    # A series scaled up by 10x must produce the same features, since none of them
    # carry the unit of the price.
    df = _sample_ohlcv()
    scaled = df.copy()
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        scaled[col] = scaled[col] * 10.0

    base, cols, _ = build_regression_dataset(df, horizon=30)
    lifted, lifted_cols, _ = build_regression_dataset(scaled, horizon=30)

    assert cols == lifted_cols == STATIONARY_REGRESSION_FEATURE_COLUMNS
    for col in cols:
        np.testing.assert_allclose(
            base[col].to_numpy(dtype=float),
            lifted[col].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"{col} changed when the price level was scaled",
        )


def test_regression_dataset_can_still_build_the_legacy_feature_set():
    """Bundles trained before the change reference level columns and must keep loading."""
    df = _sample_ohlcv()
    _dataset, feature_cols, _target = build_regression_dataset(
        df, horizon=30, feature_columns=CORE_REGRESSION_FEATURE_COLUMNS
    )
    assert feature_cols == CORE_REGRESSION_FEATURE_COLUMNS


def test_chronological_split_purges_overlapping_target_windows():
    """
    A 30-day forward return makes consecutive rows overlap by 29 days, so without
    a purge the last training rows resolve inside the validation window.
    """
    df = _sample_ohlcv(rows=600)
    dataset, feature_cols, target_col = build_regression_dataset(df, horizon=30)

    unpurged = split_dataset_chronologically(
        dataset, feature_columns=feature_cols, target_column=target_col, embargo=0
    )
    purged = split_dataset_chronologically(
        dataset, feature_columns=feature_cols, target_column=target_col, embargo=30
    )

    assert len(purged["train_frame"]) == len(unpurged["train_frame"]) - 30
    assert len(purged["val_frame"]) == len(unpurged["val_frame"]) - 30
    # Test set is never trimmed: it is the last segment and overlaps nothing later.
    assert len(purged["test_frame"]) == len(unpurged["test_frame"])

    train_end = purged["train_frame"].index[-1]
    val_start = purged["val_frame"].index[0]
    gap_rows = dataset.index.get_loc(val_start) - dataset.index.get_loc(train_end)
    assert gap_rows > 30


def test_baseline_skill_separates_a_real_model_from_a_constant():
    y_true = np.array([0.05, -0.03, 0.02, -0.01, 0.04])
    train_mean = 0.02

    informative = _baseline_skill(y_true, y_true * 0.9, train_mean)
    assert informative["skill_score"] > 0
    assert informative["prediction_std"] > 0

    collapsed = _baseline_skill(y_true, np.full_like(y_true, 0.13), train_mean)
    assert collapsed["skill_score"] < 0
    assert collapsed["prediction_std"] == 0.0


def test_skill_gate_refuses_unproven_and_failing_bundles(monkeypatch):
    monkeypatch.setenv("QUANTVISION_ENFORCE_MODEL_SKILL", "true")

    assert bundle_skill_failure({"passes_baseline": True}) is None

    unproven = bundle_skill_failure({"target_type": "return_regression"})
    assert unproven is not None and "no evidence" in unproven

    failing = bundle_skill_failure(
        {"passes_baseline": False, "skill": {"test": {"skill_score": -0.76, "prediction_std": 0.004}}}
    )
    assert failing is not None
    assert "does not beat a constant forecast" in failing
    assert "-0.7600" in failing

    monkeypatch.setenv("QUANTVISION_ENFORCE_MODEL_SKILL", "false")
    assert bundle_skill_failure({"passes_baseline": False}) is None


def test_ensemble_schema_rejects_unsupported_horizons():
    with pytest.raises(ValidationError):
        EnsemblePredictRequest(symbol="AAPL", horizon=10)

    with pytest.raises(ValidationError):
        EnsembleTrainRequest(symbol="AAPL", horizons=[7, 10, 30])

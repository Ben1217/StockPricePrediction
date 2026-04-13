from src.api.schemas.schemas import BacktestRequest, BootstrapTrainRequest, PredictRequest, TrainRequest


def test_train_request_defaults_cover_ui_horizons_and_one_step():
    req = TrainRequest()

    assert req.symbol == "^GSPC"
    assert req.horizons == [1]


def test_bootstrap_request_defaults_cover_supported_symbols_and_models():
    req = BootstrapTrainRequest()

    assert req.symbols[0] == "^GSPC"
    assert "AAPL" in req.symbols
    assert "SPY" not in req.symbols
    assert [model.value for model in req.model_types] == ["xgboost", "random_forest", "lstm"]
    assert req.horizons == [1]
    assert req.use_sp500 is False
    assert req.skip_fresh_hours == 24


def test_prediction_and_backtest_defaults_target_sp500_index():
    predict_req = PredictRequest()
    backtest_req = BacktestRequest()

    assert predict_req.symbol == "^GSPC"
    assert predict_req.horizon == 1
    assert backtest_req.symbol == "^GSPC"
    assert backtest_req.benchmark_symbol == "^GSPC"

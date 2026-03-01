"""
Prediction API routes — run inference with uncertainty bands.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import APIRouter, HTTPException

from src.api.schemas.schemas import PredictRequest, PredictResponse, ForecastPoint

router = APIRouter()


@router.post("", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Run model inference and return forecasts with uncertainty bands.
    Falls back to a statistical forecast if no trained model is available.
    """
    import yfinance as yf

    # Fetch recent data
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    df = yf.download(req.symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise HTTPException(404, f"No data for {req.symbol}")

    close = df["Close"].values.astype(float)
    current_price = float(close[-1])
    model_type = req.model_type.value
    model_info = {"type": model_type, "source": "live"}

    # Try to load trained model
    model_dir = Path("models/saved_models") / model_type
    trained_model = None
    if model_dir.exists():
        try:
            from src.models.model_trainer import ModelTrainer
            trainer = ModelTrainer()
            model = trainer.create_model(model_type)
            # Find model file
            for p in model_dir.iterdir():
                try:
                    model.load(str(p))
                    trained_model = model
                    model_info["source"] = "trained"
                    model_info["model_file"] = p.name
                    break
                except Exception:
                    continue
        except Exception:
            pass

    # Generate forecasts
    forecasts = []
    if trained_model is not None and model_type != "lstm":
        # Use trained model for next-step predictions iteratively
        from src.features.technical_indicators import add_all_technical_indicators
        from src.features.feature_engineering import create_features

        feat_df = create_features(df)
        feat_df = feat_df.dropna()
        feature_cols_file = Path("models/model_metadata/feature_columns.json")
        if feature_cols_file.exists():
            import json
            with open(feature_cols_file) as f:
                feature_cols = json.load(f)
            feature_cols = [c for c in feature_cols if c in feat_df.columns]
        else:
            feature_cols = [c for c in feat_df.columns
                           if c not in ["Open", "High", "Low", "Close", "Volume", "Adj Close", "Target"]]

        if feature_cols and not feat_df.empty:
            last_features = feat_df[feature_cols].iloc[-1:].values
            pred_price = current_price
            n = min(60, len(close))
            daily_vol = float(np.std(np.diff(close[-n:]) / close[-n:-1])) if n > 2 else 0.01

            for d in range(1, req.horizon + 1):
                try:
                    pred_return = float(trained_model.predict(last_features)[0])
                except Exception:
                    pred_return = 0.0
                pred_price = pred_price * (1 + pred_return)
                uncertainty = daily_vol * np.sqrt(d) * current_price
                dt = datetime.now() + timedelta(days=d)
                forecasts.append(ForecastPoint(
                    date=dt.strftime("%Y-%m-%d"),
                    predicted=round(pred_price, 2),
                    upper95=round(pred_price + 1.96 * uncertainty, 2),
                    lower95=round(pred_price - 1.96 * uncertainty, 2),
                    upper68=round(pred_price + uncertainty, 2),
                    lower68=round(pred_price - uncertainty, 2),
                ))
            model_info["method"] = "iterative_prediction"
    else:
        # Statistical fallback: drift + volatility cone
        n = min(252, len(close))
        returns = np.diff(close[-n:]) / close[-n:-1]
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))
        pred_price = current_price

        for d in range(1, req.horizon + 1):
            pred_price = pred_price * (1 + mu)
            uncertainty = sigma * np.sqrt(d) * current_price
            dt = datetime.now() + timedelta(days=d)
            forecasts.append(ForecastPoint(
                date=dt.strftime("%Y-%m-%d"),
                predicted=round(pred_price, 2),
                upper95=round(pred_price + 1.96 * uncertainty, 2),
                lower95=round(pred_price - 1.96 * uncertainty, 2),
                upper68=round(pred_price + uncertainty, 2),
                lower68=round(pred_price - uncertainty, 2),
            ))
        model_info["method"] = "statistical_drift"

    return PredictResponse(
        symbol=req.symbol,
        model_type=model_type,
        horizon=req.horizon,
        current_price=round(current_price, 2),
        forecasts=forecasts,
        model_info=model_info,
    )

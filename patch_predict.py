import sys

with open(r'D:\StockPricePrediction\StockPricePrediction\src\api\routes\predict.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''    bundle = load_model_bundle(model_type=model_type, symbol=symbol, horizon=NEXT_DAY_HORIZON)
    if bundle is None:
        message = f"No trained {model_type} bundle found for {symbol}"
        model_info = _build_model_info(
            symbol=symbol,
            requested_model=model_type,
            bundle=None,
            available=False,
            reason="missing_bundle",
            message=message,
        )
        return PredictResponse(
            symbol=symbol,
            model_type=model_type,
            horizon=requested_horizon,
            current_price=round(current_price, 2),
            direction=None,
            signal=None,
            confidence=None,
            probability_up=None,
            probability_down=None,
            expected_move=None,
            prediction_date=None,
            model_info=model_info,
            status="unavailable",
            model_available=False,
            reason="missing_bundle",
            message=message,
            can_train=True,
        )'''

replacement = '''    bundle = load_model_bundle(model_type=model_type, symbol=symbol, horizon=NEXT_DAY_HORIZON)

    if bundle is None:
        logger.info(f"Missing bundle for {symbol} ({model_type}). Attempting auto-retrain...")
        try:
            from src.models.ensemble_training import train_ensemble_for_symbol
            train_ensemble_for_symbol(
                symbol=symbol,
                horizons=[NEXT_DAY_HORIZON],
                model_types=[model_type],
                lookback_days=1825,
            )
            bundle = load_model_bundle(model_type=model_type, symbol=symbol, horizon=NEXT_DAY_HORIZON)
        except Exception as e:
            logger.error(f"Auto-retrain failed: {e}")

    if bundle is None:
        raise HTTPException(status_code=400, detail="Prediction model not available. Please train or load model bundle.")'''

if target.replace('\r\n', '\n') in code.replace('\r\n', '\n'):
    print('Found missing bundle return. Replacing...')
    code = code.replace('\r\n', '\n').replace(target.replace('\r\n', '\n'), replacement)
else:
    print('Target not found in code.')

target2 = '''    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index()'''

replacement2 = '''    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index().ffill().dropna()'''

if target2.replace('\r\n', '\n') in code.replace('\r\n', '\n'):
    print('Found ffill line. Replacing...')
    code = code.replace('\r\n', '\n').replace(target2.replace('\r\n', '\n'), replacement2)
else:
    print('Target 2 not found.')
    
with open(r'D:\StockPricePrediction\StockPricePrediction\src\api\routes\predict.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Done")

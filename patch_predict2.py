import sys

with open(r'D:\StockPricePrediction\StockPricePrediction\src\api\routes\predict.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''    try:
        probabilities = _predict_bundle_probabilities(bundle, feature_frame)
        prob_up = float(probability_up(probabilities)[0])
    except ValueError as exc:
        message = str(exc)
        model_info = _build_model_info(
            symbol=symbol,
            requested_model=model_type,
            bundle=bundle,
            available=False,
            reason="insufficient_inference_history",
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
            reason="insufficient_inference_history",
            message=message,
            can_train=True,
        )'''

replacement = '''    try:
        probabilities = _predict_bundle_probabilities(bundle, feature_frame)
        prob_up = float(probability_up(probabilities)[0])
    except ValueError as exc:
        # Legacy bundle feature mismatch. Auto-retrain!
        logger.info(f"Feature mismatch for {symbol} ({model_type}): {exc}. Attempting auto-retrain...")
        try:
            from src.models.ensemble_training import train_ensemble_for_symbol
            import shutil
            import os
            bundle_path = os.path.join("models", "bundles", symbol, model_type)
            if os.path.exists(bundle_path):
                shutil.rmtree(bundle_path)
            
            train_ensemble_for_symbol(
                symbol=symbol,
                horizons=[NEXT_DAY_HORIZON],
                model_types=[model_type],
                lookback_days=400,
            )
            bundle = load_model_bundle(model_type=model_type, symbol=symbol, horizon=NEXT_DAY_HORIZON)
            if bundle is None:
                raise ValueError("Retrained bundle failed to load")
            feature_frame = build_feature_frame(raw_df, feature_config=bundle.feature_config)
            probabilities = _predict_bundle_probabilities(bundle, feature_frame)
            prob_up = float(probability_up(probabilities)[0])
        except Exception as retry_exc:
            message = str(retry_exc)
            model_info = _build_model_info(
                symbol=symbol,
                requested_model=model_type,
                bundle=bundle,
                available=False,
                reason="insufficient_inference_history",
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
                reason="insufficient_inference_history",
                message=message,
                can_train=True,
            )'''

if target.replace('\r\n', '\n') in code.replace('\r\n', '\n'):
    print('Found ValueError catch block. Replacing...')
    code = code.replace('\r\n', '\n').replace(target.replace('\r\n', '\n'), replacement)
    with open(r'D:\StockPricePrediction\StockPricePrediction\src\api\routes\predict.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print('Done.')
else:
    print('Target not found in code.')

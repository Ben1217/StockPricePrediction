import sys

with open(r'D:\StockPricePrediction\StockPricePrediction\src\api\routes\predict.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''        try:
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
                lookback_days=1825,
            )'''

replacement = '''        try:
            from src.models.bundle_training import train_model_bundles
            import shutil
            import os
            bundle_path = os.path.join("models", "bundles", symbol, model_type)
            if os.path.exists(bundle_path):
                shutil.rmtree(bundle_path)
            
            train_model_bundles(
                symbol=symbol,
                model_type=model_type,
                horizons=[NEXT_DAY_HORIZON],
                lookback_days=1825,
            )'''

if target.replace('\r\n', '\n') in code.replace('\r\n', '\n'):
    print('Found ensemble retrain code. Replacing with bundle_training...')
    code = code.replace('\r\n', '\n').replace(target.replace('\r\n', '\n'), replacement)
    with open(r'D:\StockPricePrediction\StockPricePrediction\src\api\routes\predict.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print('Done.')
else:
    print('Target not found.')

import sys

with open(r'D:\StockPricePrediction\StockPricePrediction\src\api\routes\predict.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''            train_ensemble_for_symbol(
                symbol=symbol,
                horizons=[NEXT_DAY_HORIZON],
                model_types=[model_type],
                lookback_days=400,
            )'''

replacement = '''            train_ensemble_for_symbol(
                symbol=symbol,
                horizons=[NEXT_DAY_HORIZON],
                model_types=[model_type],
                lookback_days=1825,
            )'''

if target.replace('\r\n', '\n') in code.replace('\r\n', '\n'):
    print('Found 400. Replacing with 1825...')
    code = code.replace('\r\n', '\n').replace(target.replace('\r\n', '\n'), replacement)
    with open(r'D:\StockPricePrediction\StockPricePrediction\src\api\routes\predict.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print('Done.')
else:
    print('Target not found.')

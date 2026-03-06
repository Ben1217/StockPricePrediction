import traceback
import sys
from datetime import datetime, timedelta
import yfinance as yf
from src.features.technical_indicators import add_all_technical_indicators

def test():
    try:
        symbol = "AAPL"
        days = 120
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days + 200)).strftime("%Y-%m-%d")
        print(f"Fetching {symbol} from {start} to {end}")
        
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = add_all_technical_indicators(df)
        df = df.tail(days)
        df = df.replace({float("nan"): None, float("inf"): None, float("-inf"): None})

        indicator_cols = [
            c for c in df.columns
            if c not in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        ]
        data = []
        for dt, row in df.iterrows():
            entry = {"date": str(dt.date()) if hasattr(dt, "date") else str(dt)}
            for c in indicator_cols:
                v = row[c]
                entry[c] = round(float(v), 4) if v is not None else None
            data.append(entry)
            
        print("Success, created dict length:", len(data))
        from src.api.schemas.schemas import IndicatorResponse
        from fastapi.encoders import jsonable_encoder
        import json
        
        resp = IndicatorResponse(
            symbol=symbol,
            indicators=indicator_cols,
            data=data,
            count=len(data),
        )
        print("Pydantic validation passed")
        
        json_str = json.dumps(jsonable_encoder(resp))
        print("JSON encoding passed, length:", len(json_str))
        
    except Exception as e:
        traceback.print_exc()

import pandas as pd
test()

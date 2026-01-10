"""
Quick Model Training Script
Trains XGBoost and Random Forest models for signal generation
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import download_stock_data
from src.features.technical_indicators import add_all_technical_indicators
from src.models.model_trainer import ModelTrainer


def train_models_for_stock(symbol: str = 'SPY', lookback_days: int = 756):
    """
    Train ML models for a specific stock.
    
    Parameters
    ----------
    symbol : str
        Stock ticker
    lookback_days : int
        Days of historical data (default 756 = ~3 years)
    
    Returns
    -------
    dict
        Training results
    """
    print(f"\n{'='*60}")
    print(f"TRAINING MODELS FOR {symbol}")
    print(f"{'='*60}")
    
    # Step 1: Download data
    print(f"\n[1/5] Downloading {symbol} data...")
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    df = download_stock_data(symbol, start_date, end_date)
    
    if df.empty:
        print(f"[ERROR] Failed to download data for {symbol}")
        return None
    
    print(f"  Downloaded {len(df)} days of data")
    
    # Step 2: Add technical indicators
    print(f"\n[2/5] Adding technical indicators...")
    df = add_all_technical_indicators(df)
    df = df.dropna()  # Remove rows with NaN from indicator calculations
    print(f"  {len(df)} days after indicator calculation")
    
    # Step 3: Prepare features and target
    print(f"\n[3/5] Preparing features...")
    
    # Feature columns (all indicators)
    feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    # Create target: next day's return
    df['Target'] = df['Close'].pct_change().shift(-1)  # Next day return
    df = df.dropna()
    
    X = df[feature_cols].values
    y = df['Target'].values
    
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Samples: {len(X)}")
    
    # Step 4: Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Step 5: Train models
    print(f"\n[4/5] Training models...")
    trainer = ModelTrainer()
    
    # Train XGBoost and Random Forest (faster than LSTM)
    models = trainer.train_all_models(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        model_types=['xgboost', 'random_forest'],
        save=True
    )
    
    print(f"\n[5/5] Evaluating models...")
    results = trainer.evaluate_all_models(X_test, y_test)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(results)
    
    # Save feature columns for later use
    import json
    feature_path = Path('models/model_metadata/feature_columns.json')
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_path, 'w') as f:
        json.dump(feature_cols, f)
    
    print(f"\n[OK] Models saved to models/saved_models/")
    print(f"[OK] Feature columns saved to {feature_path}")
    
    return {
        'symbol': symbol,
        'samples': len(X),
        'features': len(feature_cols),
        'results': results
    }


def test_signal_on_stock(symbol: str):
    """
    Test signal generation on a specific stock.
    """
    print(f"\n{'='*60}")
    print(f"TESTING SIGNAL GENERATOR ON {symbol}")
    print(f"{'='*60}")
    
    # Download recent data
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    df = download_stock_data(symbol, start_date, end_date)
    df = add_all_technical_indicators(df)
    df = df.dropna()
    
    # Generate signal
    from src.signals.signal_generator import TradingSignalGenerator
    
    generator = TradingSignalGenerator(mode='TECHNICAL_ONLY')
    signal = generator.analyze_stock(symbol, df)
    
    print(f"\nSymbol: {signal['symbol']}")
    print(f"Mode: {signal['mode']}")
    print(f"Action: {signal['action']}")
    print(f"Confidence: {signal['confidence']:.1f}%")
    print(f"Current Price: ${signal['current_price']:.2f}")
    
    if signal['action'] != 'HOLD':
        print(f"Entry: ${signal.get('entry_price', 0):.2f}")
        print(f"Stop Loss: ${signal.get('stop_loss', 0):.2f}")
        print(f"Target: ${signal.get('target', 0):.2f}")
        print(f"Patterns: {signal['technical_details'].get('patterns', [])}")
    else:
        print(f"Reason: {signal.get('reason', 'No pattern detected')}")
    
    # Show trend scores
    tech = signal.get('technical_details', {})
    trend = tech.get('trend_score', {})
    print(f"\nTrend Scores:")
    print(f"  Uptrend: {trend.get('uptrend', 0):.0f}/100")
    print(f"  Downtrend: {trend.get('downtrend', 0):.0f}/100")
    
    return signal


if __name__ == "__main__":
    # Train models on SPY (market benchmark)
    results = train_models_for_stock('SPY')
    
    print("\n" + "="*60)
    print("TESTING SIGNALS ON TRENDING STOCKS")
    print("="*60)
    
    # Test on multiple stocks
    test_stocks = ['SPY', 'NVDA', 'TSLA', 'AAPL']
    
    for stock in test_stocks:
        try:
            test_signal_on_stock(stock)
        except Exception as e:
            print(f"\n[ERROR] {stock}: {e}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("""
Next steps:
1. Models are now saved in models/saved_models/
2. Restart the dashboard to use FULL MODE
3. Run: streamlit run src/dashboard/app.py
    """)

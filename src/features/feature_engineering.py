"""
Feature Engineering Module
Create features and prepare data for machine learning models
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .technical_indicators import add_all_technical_indicators
from .candlestick_patterns import detect_candlestick_patterns
from ..models.regime_detection import MarketRegimeDetector
from ..utils.logger import get_logger

logger = get_logger(__name__)


DEFAULT_FEATURE_CONFIG: Dict[str, object] = {
    "include_technical": True,
    "include_lags": True,
    "include_regime": False,
    "include_candlesticks": False,
    "lag_periods": [1, 2, 3, 5, 10],
}

MODEL_FEATURE_EXCLUDE_COLUMNS = {
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Adj Close",
    "Target",
    "Forward_Return",
}


def normalize_feature_config(feature_config: Optional[Dict] = None) -> Dict:
    """Return a complete feature configuration dictionary."""
    config = dict(DEFAULT_FEATURE_CONFIG)
    if feature_config:
        config.update(feature_config)

    lag_periods = config.get("lag_periods", DEFAULT_FEATURE_CONFIG["lag_periods"])
    config["lag_periods"] = [int(period) for period in lag_periods]
    return config


def create_features(
    df: pd.DataFrame,
    include_technical: bool = True,
    include_lags: bool = True,
    include_regime: bool = False,
    include_candlesticks: bool = False,
    lag_periods: List[int] = [1, 2, 3, 5, 10]
) -> pd.DataFrame:
    """
    Create all features for model training

    Parameters
    ----------
    df : pandas.DataFrame
        Raw stock data with OHLCV columns
    include_technical : bool
        Whether to add technical indicators
    include_lags : bool
        Whether to add lagged features
    lag_periods : list
        Periods for lag features

    Returns
    -------
    pandas.DataFrame
        Data with all features
    """
    data = df.copy()
    
    # Add technical indicators
    if include_technical:
        data = add_all_technical_indicators(data)
    
    # Calculate returns
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Price-based features
    data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close']
    data['Open_Close_Range'] = (data['Close'] - data['Open']) / data['Open']
    
    # Add lagged features
    if include_lags:
        for lag in lag_periods:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Return_Lag_{lag}'] = data['Daily_Return'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
    
    # Day of week (if datetime index)
    if isinstance(data.index, pd.DatetimeIndex):
        data['DayOfWeek'] = data.index.dayofweek
        data['Month'] = data.index.month
        data['Quarter'] = data.index.quarter
    
    # Market regime detection (HMM)
    if include_regime:
        try:
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 30:  # need minimum data for HMM
                detector = MarketRegimeDetector()
                detector.fit(returns)
                regime_df = detector.get_regime_features(returns)
                data = data.join(regime_df)
                logger.info(f"Added HMM regime features: {regime_df.columns.tolist()}")
        except Exception as e:
            logger.warning(f"Regime detection failed, skipping: {e}")

    # Candlestick pattern detection
    if include_candlesticks:
        try:
            cdl_df = detect_candlestick_patterns(data)
            data = data.join(cdl_df)
            logger.info(f"Added {len(cdl_df.columns)} candlestick pattern features")
        except Exception as e:
            logger.warning(f"Candlestick detection failed, skipping: {e}")

    logger.info(f"Created features: {len(data.columns)} total columns")
    
    return data


def build_feature_frame(df: pd.DataFrame, feature_config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Build a feature frame using the project's canonical feature pipeline.

    This helper keeps training and inference on the same feature logic.
    """
    config = normalize_feature_config(feature_config)
    return create_features(
        df,
        include_technical=bool(config["include_technical"]),
        include_lags=bool(config["include_lags"]),
        include_regime=bool(config["include_regime"]),
        include_candlesticks=bool(config["include_candlesticks"]),
        lag_periods=list(config["lag_periods"]),
    )


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw and derived market data without introducing future leakage.

    The cleaning is intentionally conservative:
    - keep only rows with valid OHLCV values
    - replace infinities created by indicator math
    - drop duplicated timestamps
    """
    data = df.copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data[~data.index.duplicated(keep="last")]

    required_cols = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in data.columns]
    if required_cols:
        data = data.dropna(subset=required_cols)
        for col in ["Open", "High", "Low", "Close"]:
            if col in data.columns:
                data = data[data[col] > 0]
        if "Volume" in data.columns:
            data = data[data["Volume"] >= 0]

    return data.sort_index()


def add_forward_return(df: pd.DataFrame, horizon: int = 1, column_name: str = "Forward_Return") -> pd.DataFrame:
    """Attach the realised forward return used by simple validation/backtests."""
    data = df.copy()
    data[column_name] = data["Close"].pct_change(horizon).shift(-horizon)
    return data


def create_target_variable(
    df: pd.DataFrame,
    target_type: str = 'return',
    horizon: int = 1,
    classification_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Create target variable for prediction

    Parameters
    ----------
    df : pandas.DataFrame
        Data with Close prices
    target_type : str
        Type of target: 'return', 'price', 'direction'
    horizon : int
        Prediction horizon (days ahead)
    classification_threshold : float
        Threshold for direction classification

    Returns
    -------
    pandas.DataFrame
        Data with target column added
    """
    data = df.copy()
    
    if target_type == 'return':
        data['Target'] = data['Close'].pct_change(horizon).shift(-horizon)
    elif target_type == 'price':
        data['Target'] = data['Close'].shift(-horizon)
    elif target_type == 'direction':
        future_return = data['Close'].pct_change(horizon).shift(-horizon)
        data['Target'] = (future_return > classification_threshold).astype(int)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    logger.info(f"Created target variable: {target_type}, horizon={horizon}")
    
    return data


def select_feature_columns(
    df: pd.DataFrame,
    target_column: str = "Target",
    exclude_columns: Optional[List[str]] = None,
    min_non_null: int = 30,
) -> List[str]:
    """Select numeric model features in a stable column order."""
    exclude = set(MODEL_FEATURE_EXCLUDE_COLUMNS)
    exclude.add(target_column)
    if exclude_columns:
        exclude.update(exclude_columns)

    return [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude and int(df[col].notna().sum()) >= int(min_non_null)
    ]


def build_supervised_dataset(
    df: pd.DataFrame,
    horizon: int = 1,
    target_type: str = "direction",
    feature_config: Optional[Dict] = None,
    feature_columns: Optional[List[str]] = None,
    target_column: str = "Target",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a clean supervised-learning dataset from raw OHLCV data.
    """
    dataset = clean_market_data(build_feature_frame(df, feature_config=feature_config))
    dataset = add_forward_return(dataset, horizon=horizon)
    dataset = create_target_variable(dataset, target_type=target_type, horizon=horizon)

    resolved_feature_columns = feature_columns or select_feature_columns(
        dataset, target_column=target_column
    )
    if not resolved_feature_columns:
        return dataset.iloc[0:0].copy(), []

    dataset = dataset.dropna(subset=resolved_feature_columns + [target_column]).copy()
    logger.info(
        "Built supervised dataset: %s rows, %s features, horizon=%s",
        len(dataset),
        len(resolved_feature_columns),
        horizon,
    )
    return dataset, resolved_feature_columns


def _create_scaler(scaler_type: Optional[str]):
    if scaler_type == "minmax":
        return MinMaxScaler()
    if scaler_type == "standard":
        return StandardScaler()
    return None


def split_dataset_chronologically(
    dataset: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = "Target",
    scaler_type: Optional[str] = "minmax",
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Dict[str, object]:
    """
    Chronologically split a supervised dataset into train/validation/test sets.
    """
    if dataset.empty:
        raise ValueError("Cannot split an empty dataset")
    if not feature_columns:
        raise ValueError("No feature columns available for splitting")

    data = dataset.dropna(subset=feature_columns + [target_column]).copy()
    n_rows = len(data)
    if n_rows < 3:
        raise ValueError("Need at least 3 clean rows for train/validation/test splitting")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be between 0 and 1")

    test_count = max(1, int(round(n_rows * test_size)))
    val_count = max(1, int(round(n_rows * val_size)))

    while n_rows - test_count - val_count < 1 and test_count > 1:
        test_count -= 1
    while n_rows - test_count - val_count < 1 and val_count > 1:
        val_count -= 1
    if n_rows - test_count - val_count < 1:
        raise ValueError("Split sizes leave no rows for training")

    train_end = n_rows - test_count - val_count
    val_end = n_rows - test_count

    train_frame = data.iloc[:train_end].copy()
    val_frame = data.iloc[train_end:val_end].copy()
    test_frame = data.iloc[val_end:].copy()

    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise ValueError("Chronological split produced an empty train, validation, or test set")

    X_train_raw = train_frame[feature_columns].values.astype(np.float32)
    X_val_raw = val_frame[feature_columns].values.astype(np.float32)
    X_test_raw = test_frame[feature_columns].values.astype(np.float32)
    y_train = train_frame[target_column].values.astype(np.float32)
    y_val = val_frame[target_column].values.astype(np.float32)
    y_test = test_frame[target_column].values.astype(np.float32)

    scaler = _create_scaler(scaler_type)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_val = scaler.transform(X_val_raw).astype(np.float32)
        X_test = scaler.transform(X_test_raw).astype(np.float32)
    else:
        X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw

    logger.info(
        "Chronological split complete: train=%s, val=%s, test=%s",
        len(train_frame),
        len(val_frame),
        len(test_frame),
    )

    return {
        "dataset": data,
        "train_frame": train_frame,
        "val_frame": val_frame,
        "test_frame": test_frame,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "scaler_type": scaler_type,
    }


def transform_feature_frame(
    feature_frame: pd.DataFrame,
    feature_columns: List[str],
    scaler=None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Align a feature frame to the saved training columns and optional scaler.
    """
    if not feature_columns:
        return feature_frame.iloc[0:0].copy(), np.empty((0, 0), dtype=np.float32)

    missing = [col for col in feature_columns if col not in feature_frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:5]}")

    aligned = feature_frame.dropna(subset=feature_columns).copy()
    if aligned.empty:
        return aligned, np.empty((0, len(feature_columns)), dtype=np.float32)

    X = aligned[feature_columns].values.astype(np.float32)
    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)
    return aligned, X


def prepare_features_for_model(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_column: str = 'Target',
    scaler_type: str = 'minmax',
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Prepare features for model training

    Parameters
    ----------
    df : pandas.DataFrame
        Data with features and target
    feature_columns : list, optional
        Columns to use as features (None = all numeric except target)
    target_column : str
        Name of target column
    scaler_type : str
        Type of scaler: 'minmax', 'standard', or None
    test_size : float
        Proportion for test set

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, scaler
    """
    data = df.copy()
    
    # Remove rows with NaN in target
    data = data.dropna(subset=[target_column])
    
    # Select features
    if feature_columns is None:
        exclude = [target_column, 'Target', 'Close', 'Open', 'High', 'Low']
        feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if col not in exclude]
    
    # Drop rows with NaN in features
    data = data.dropna(subset=feature_columns)
    
    X = data[feature_columns].values
    y = data[target_column].values
    
    # Time series split (no shuffling)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = None
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaler_type == 'standard':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    logger.info(f"Prepared data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test, scaler


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM model

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    sequence_length : int
        Length of each sequence

    Returns
    -------
    tuple
        X_sequences, y_sequences
    """
    X_seq, y_seq = [], []
    
    for i in range(sequence_length - 1, len(X)):
        X_seq.append(X[i-sequence_length+1:i+1])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)

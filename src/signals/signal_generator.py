"""
Trading Signal Generator Module

Combines traditional technical analysis patterns with ML predictions
to generate actionable trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PATTERN DETECTION FUNCTIONS - Tier 1 (Highest Priority)
# =============================================================================

def detect_base_breakout(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Detect base breakout buy signals (Tier 1 - 85% base confidence).
    
    A base breakout occurs when price consolidates in a tight range
    and then breaks out with volume confirmation.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with SMA_20 and ATR columns
    lookback : int
        Number of candles to look back for base formation
    
    Returns
    -------
    pd.DataFrame
        DataFrame with buy_signal, entry_price, stop_loss columns
    """
    signals = pd.DataFrame(index=df.index)
    signals['buy_signal'] = False
    signals['entry_price'] = np.nan
    signals['stop_loss'] = np.nan
    signals['pattern'] = ''
    signals['confidence'] = 0.0
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return signals
    
    # Calculate SMA_20 if not present
    if 'SMA_20' not in df.columns:
        df = df.copy()
        df['SMA_20'] = df['Close'].rolling(20).mean()
    
    # Calculate ATR if not present
    if 'ATR' not in df.columns:
        df = df.copy()
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
    
    for i in range(lookback + 5, len(df)):
        window = df.iloc[i-lookback:i]
        current = df.iloc[i]
        
        # Skip if missing data
        if pd.isna(current.get('SMA_20')) or pd.isna(current.get('ATR')):
            continue
        
        ma_20_current = current['SMA_20']
        ma_20_previous = df.iloc[i-1]['SMA_20']
        atr = current['ATR']
        
        # 1. Check for uptrend (price above rising 20MA)
        uptrend = (current['Close'] > ma_20_current and 
                   ma_20_current > ma_20_previous)
        
        if not uptrend:
            continue
        
        # 2. Identify base (consolidation)
        base_high = window['High'].max()
        base_low = window['Low'].min()
        base_range = base_high - base_low
        
        # Base should be tight (< 2x ATR)
        is_tight_base = base_range < (2.0 * atr)
        
        if not is_tight_base:
            continue
        
        # 3. Check for declining volume in base
        if len(window) >= 3:
            recent_vol = window['Volume'].iloc[-3:].values
            volume_declining = recent_vol[-1] < recent_vol[0]
        else:
            volume_declining = True
        
        # 4. Detect breakout (price closes above base high with volume)
        avg_volume = window['Volume'].mean()
        breakout = (current['Close'] > base_high and 
                    current['Volume'] > avg_volume * 1.2)
        
        # 5. Check if base is near 20MA
        near_ma = abs(base_low - ma_20_current) < (1.5 * atr)
        
        # ALL CONDITIONS MET
        if is_tight_base and breakout and near_ma:
            confidence = 85.0
            
            # Bonus confidence for volume confirmation
            if volume_declining and current['Volume'] > avg_volume * 1.5:
                confidence += 5.0
            
            # Bonus for strong uptrend
            if current['Close'] > df.iloc[i].get('SMA_200', 0):
                confidence += 5.0
            
            signals.loc[df.index[i], 'buy_signal'] = True
            signals.loc[df.index[i], 'entry_price'] = base_high * 1.002  # Just above
            signals.loc[df.index[i], 'stop_loss'] = base_low * 0.998     # Just below
            signals.loc[df.index[i], 'pattern'] = 'BASE_BREAKOUT'
            signals.loc[df.index[i], 'confidence'] = min(confidence, 95.0)
    
    return signals


# =============================================================================
# PATTERN DETECTION FUNCTIONS - Tier 2
# =============================================================================

def detect_pullback_buy(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Detect pullback buy setup (Tier 2 - 75% base confidence).
    
    A pullback occurs when price retraces to the 20 MA in an established uptrend,
    offering a lower-risk entry.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with SMA_20 and ATR columns
    lookback : int
        Number of candles to analyze
    
    Returns
    -------
    pd.DataFrame
        DataFrame with buy_signal, entry_price, stop_loss, confidence columns
    """
    signals = pd.DataFrame(index=df.index)
    signals['buy_signal'] = False
    signals['entry_price'] = np.nan
    signals['stop_loss'] = np.nan
    signals['pattern'] = ''
    signals['confidence'] = 0.0
    
    # Calculate indicators if needed
    df = df.copy()
    if 'SMA_20' not in df.columns:
        df['SMA_20'] = df['Close'].rolling(20).mean()
    if 'ATR' not in df.columns:
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
    
    for i in range(lookback + 5, len(df)):
        current = df.iloc[i]
        window = df.iloc[i-lookback:i]
        
        if pd.isna(current.get('SMA_20')) or pd.isna(current.get('ATR')):
            continue
        
        sma_20 = current['SMA_20']
        sma_20_prev = df.iloc[i-5]['SMA_20'] if i >= 5 else sma_20
        atr = current['ATR']
        
        # 1. Confirm uptrend (price above rising MA)
        uptrend = (current['Close'] > sma_20 and sma_20 > sma_20_prev)
        
        if not uptrend:
            continue
        
        # 2. Find last significant high
        recent_high = window['High'].max()
        
        # 3. Check for pullback (3 consecutive down candles)
        if i < 3:
            continue
        last_3 = df.iloc[i-3:i]
        consecutive_down = all(last_3['Close'].values < last_3['Open'].values)
        
        # 4. Calculate pullback depth (should be 40-60% retracement)
        pullback_low = df.iloc[i-3:i+1]['Low'].min()
        window_low = window.iloc[0]['Low']
        up_move = recent_high - window_low
        
        if up_move <= 0:
            continue
        
        pullback_depth = (recent_high - pullback_low) / up_move
        ideal_pullback = 0.3 <= pullback_depth <= 0.65
        
        # 5. Check if near 20 MA (within 0.5 ATR)
        distance_to_ma = abs(current['Low'] - sma_20)
        near_ma = distance_to_ma < (0.75 * atr)
        
        # 6. Entry bar characteristics
        current_range = current['High'] - current['Low']
        avg_range = (window['High'] - window['Low']).mean()
        narrow_range = current_range < avg_range * 0.8
        
        # Bottoming tail (lower wick > 2x body)
        body = abs(current['Close'] - current['Open'])
        lower_wick = min(current['Open'], current['Close']) - current['Low']
        has_bottoming_tail = lower_wick > (1.5 * body) if body > 0 else False
        
        # 7. Volume declining on pullback
        pullback_vol = df.iloc[i-3:i]['Volume'].mean()
        base_vol = window['Volume'].mean()
        volume_declining = pullback_vol < base_vol
        
        # Calculate confidence
        confidence = 75.0
        if ideal_pullback:
            confidence += 10.0
        if near_ma:
            confidence += 5.0
        if narrow_range or has_bottoming_tail:
            confidence += 5.0
        if volume_declining:
            confidence += 5.0
        
        # TRIGGER: Key conditions met
        if near_ma and (narrow_range or has_bottoming_tail) and confidence >= 75:
            signals.loc[df.index[i], 'buy_signal'] = True
            signals.loc[df.index[i], 'entry_price'] = current['High'] * 1.002
            signals.loc[df.index[i], 'stop_loss'] = current['Low'] * 0.998
            signals.loc[df.index[i], 'pattern'] = 'PULLBACK_BUY'
            signals.loc[df.index[i], 'confidence'] = min(confidence, 95.0)
    
    return signals


# =============================================================================
# PATTERN DETECTION FUNCTIONS - Tier 3
# =============================================================================

def detect_123_continuation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect 1-2-3 continuation pattern (Tier 3 - 70% base confidence).
    
    Pattern:
    1. Igniting bar: Wide-range green candle
    2. Resting bar: Small narrow candle (pause)
    3. Triggering bar: Breaks above previous highs
    
    Returns
    -------
    pd.DataFrame
        DataFrame with buy_signal, entry_price, stop_loss columns
    """
    signals = pd.DataFrame(index=df.index)
    signals['buy_signal'] = False
    signals['entry_price'] = np.nan
    signals['stop_loss'] = np.nan
    signals['pattern'] = ''
    signals['confidence'] = 0.0
    
    # Calculate ATR if needed
    df = df.copy()
    if 'ATR' not in df.columns:
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
    
    for i in range(3, len(df)):
        current = df.iloc[i]  # Potential triggering bar
        resting = df.iloc[i-1]  # Potential resting bar
        igniting = df.iloc[i-2]  # Potential igniting bar
        
        if pd.isna(current.get('ATR')):
            continue
        
        atr = current['ATR']
        
        # 1. IGNITING BAR: Wide-range green candle
        igniting_range = igniting['High'] - igniting['Low']
        igniting_is_green = igniting['Close'] > igniting['Open']
        igniting_is_wide = igniting_range > (1.5 * atr)
        
        if not (igniting_is_green and igniting_is_wide):
            continue
        
        # 2. RESTING BAR: Small narrow candle
        resting_range = resting['High'] - resting['Low']
        resting_is_narrow = resting_range < (0.7 * atr)
        resting_inside = (resting['High'] <= igniting['High'] and 
                          resting['Low'] >= igniting['Low'] * 0.99)
        
        if not (resting_is_narrow or resting_inside):
            continue
        
        # 3. TRIGGERING BAR: Breaks above previous highs
        combo_high = max(igniting['High'], resting['High'])
        triggering_breakout = current['Close'] > combo_high
        
        if not triggering_breakout:
            continue
        
        # Check volume confirmation
        avg_vol = df.iloc[i-10:i]['Volume'].mean() if i >= 10 else df.iloc[:i]['Volume'].mean()
        volume_spike = current['Volume'] > avg_vol * 1.2
        
        # Calculate confidence
        confidence = 70.0
        if volume_spike:
            confidence += 10.0
        if current['Close'] > current.get('SMA_20', 0):
            confidence += 5.0
        
        signals.loc[df.index[i], 'buy_signal'] = True
        signals.loc[df.index[i], 'entry_price'] = combo_high * 1.002
        signals.loc[df.index[i], 'stop_loss'] = resting['Low'] * 0.998
        signals.loc[df.index[i], 'pattern'] = '123_CONTINUATION'
        signals.loc[df.index[i], 'confidence'] = min(confidence, 95.0)
    
    return signals


# =============================================================================
# SHORT PATTERN DETECTION
# =============================================================================

def detect_base_breakdown(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Detect base breakdown sell/short signals.
    
    Mirror of base breakout - consolidation below declining 20 MA
    followed by breakdown with volume.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with sell_signal, entry_price, stop_loss columns
    """
    signals = pd.DataFrame(index=df.index)
    signals['sell_signal'] = False
    signals['entry_price'] = np.nan
    signals['stop_loss'] = np.nan
    signals['pattern'] = ''
    signals['confidence'] = 0.0
    
    df = df.copy()
    if 'SMA_20' not in df.columns:
        df['SMA_20'] = df['Close'].rolling(20).mean()
    if 'ATR' not in df.columns:
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
    
    for i in range(lookback + 5, len(df)):
        window = df.iloc[i-lookback:i]
        current = df.iloc[i]
        
        if pd.isna(current.get('SMA_20')) or pd.isna(current.get('ATR')):
            continue
        
        ma_20_current = current['SMA_20']
        ma_20_previous = df.iloc[i-1]['SMA_20']
        atr = current['ATR']
        
        # 1. Check for downtrend (price below declining 20MA)
        downtrend = (current['Close'] < ma_20_current and 
                     ma_20_current < ma_20_previous)
        
        if not downtrend:
            continue
        
        # 2. Identify base (consolidation)
        base_high = window['High'].max()
        base_low = window['Low'].min()
        base_range = base_high - base_low
        
        is_tight_base = base_range < (2.0 * atr)
        
        if not is_tight_base:
            continue
        
        # 3. Detect breakdown (price closes below base low with volume)
        avg_volume = window['Volume'].mean()
        breakdown = (current['Close'] < base_low and 
                     current['Volume'] > avg_volume * 1.2)
        
        # 4. Check if base is near 20MA
        near_ma = abs(base_high - ma_20_current) < (1.5 * atr)
        
        if is_tight_base and breakdown and near_ma:
            confidence = 85.0
            
            signals.loc[df.index[i], 'sell_signal'] = True
            signals.loc[df.index[i], 'entry_price'] = base_low * 0.998  # Just below
            signals.loc[df.index[i], 'stop_loss'] = base_high * 1.002   # Just above
            signals.loc[df.index[i], 'pattern'] = 'BASE_BREAKDOWN'
            signals.loc[df.index[i], 'confidence'] = min(confidence, 95.0)
    
    return signals


# =============================================================================
# TREND SCORING FUNCTIONS
# =============================================================================

def check_uptrend(df: pd.DataFrame) -> float:
    """
    Score strength of uptrend (0-100).
    
    Checks:
    1. Price above 20 MA (20 points)
    2. Price above 200 MA (15 points)
    3. 20 MA rising (15 points)
    4. Higher highs and higher lows (25 points)
    5. Volume confirmation (15 points)
    6. RSI favorable (10 points)
    
    Returns
    -------
    float
        Uptrend score from 0 to 100
    """
    if len(df) < 20:
        return 0.0
    
    current = df.iloc[-1]
    score = 0.0
    
    # Calculate indicators if needed
    df = df.copy()
    if 'SMA_20' not in df.columns:
        df['SMA_20'] = df['Close'].rolling(20).mean()
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = df['Close'].rolling(200).mean()
    if 'RSI' not in df.columns:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    current = df.iloc[-1]
    
    # 1. Price above 20 MA (20 points)
    if not pd.isna(current.get('SMA_20')) and current['Close'] > current['SMA_20']:
        score += 20
    
    # 2. Price above 200 MA (15 points)
    if not pd.isna(current.get('SMA_200')) and current['Close'] > current['SMA_200']:
        score += 15
    
    # 3. 20 MA rising (15 points)
    ma_20_slope = df['SMA_20'].iloc[-5:].diff().mean()
    if not pd.isna(ma_20_slope) and ma_20_slope > 0:
        score += 15
    
    # 4. Higher highs and higher lows (25 points)
    if len(df) >= 20:
        recent_highs = df['High'].iloc[-20:].rolling(5).max()
        recent_lows = df['Low'].iloc[-20:].rolling(5).min()
        
        if len(recent_highs) >= 5 and recent_highs.iloc[-1] > recent_highs.iloc[-5]:
            score += 12.5
        if len(recent_lows) >= 5 and recent_lows.iloc[-1] > recent_lows.iloc[-5]:
            score += 12.5
    
    # 5. Volume confirmation (15 points)
    up_days = df[df['Close'] > df['Open']]
    down_days = df[df['Close'] < df['Open']]
    
    up_volume = up_days['Volume'].iloc[-10:].mean() if len(up_days) >= 10 else 0
    down_volume = down_days['Volume'].iloc[-10:].mean() if len(down_days) >= 10 else 1
    
    if up_volume > down_volume * 1.1:
        score += 15
    
    # 6. RSI favorable (10 points)
    rsi = current.get('RSI', 50)
    if not pd.isna(rsi) and 40 < rsi < 70:
        score += 10
    
    return min(score, 100.0)


def check_downtrend(df: pd.DataFrame) -> float:
    """
    Score strength of downtrend (0-100).
    
    Mirror logic of check_uptrend.
    
    Returns
    -------
    float
        Downtrend score from 0 to 100
    """
    if len(df) < 20:
        return 0.0
    
    df = df.copy()
    if 'SMA_20' not in df.columns:
        df['SMA_20'] = df['Close'].rolling(20).mean()
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = df['Close'].rolling(200).mean()
    if 'RSI' not in df.columns:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    current = df.iloc[-1]
    score = 0.0
    
    # 1. Price below 20 MA
    if not pd.isna(current.get('SMA_20')) and current['Close'] < current['SMA_20']:
        score += 20
    
    # 2. Price below 200 MA
    if not pd.isna(current.get('SMA_200')) and current['Close'] < current['SMA_200']:
        score += 15
    
    # 3. 20 MA declining
    ma_20_slope = df['SMA_20'].iloc[-5:].diff().mean()
    if not pd.isna(ma_20_slope) and ma_20_slope < 0:
        score += 15
    
    # 4. Lower highs and lower lows
    if len(df) >= 20:
        recent_highs = df['High'].iloc[-20:].rolling(5).max()
        recent_lows = df['Low'].iloc[-20:].rolling(5).min()
        
        if len(recent_highs) >= 5 and recent_highs.iloc[-1] < recent_highs.iloc[-5]:
            score += 12.5
        if len(recent_lows) >= 5 and recent_lows.iloc[-1] < recent_lows.iloc[-5]:
            score += 12.5
    
    # 5. Volume confirmation
    down_days = df[df['Close'] < df['Open']]
    up_days = df[df['Close'] > df['Open']]
    
    down_volume = down_days['Volume'].iloc[-10:].mean() if len(down_days) >= 10 else 0
    up_volume = up_days['Volume'].iloc[-10:].mean() if len(up_days) >= 10 else 1
    
    if down_volume > up_volume * 1.1:
        score += 15
    
    # 6. RSI favorable for shorts
    rsi = current.get('RSI', 50)
    if not pd.isna(rsi) and 30 < rsi < 60:
        score += 10
    
    return min(score, 100.0)


# =============================================================================
# MAIN SIGNAL GENERATOR CLASS
# =============================================================================

@dataclass
class MLPrediction:
    """ML model prediction output"""
    predicted_price: float
    predicted_return: float
    confidence_score: float
    directional_signal: str  # 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'


class TradingSignalGenerator:
    """
    Complete trading signal generation combining technical and ML analysis.
    
    Supports two modes:
    - FULL: Technical + ML combined (when models are trained)
    - TECHNICAL_ONLY: Pure technical analysis (fallback)
    """
    
    # Signal modes
    MODE_FULL = 'FULL'
    MODE_TECHNICAL_ONLY = 'TECHNICAL_ONLY'
    
    # ML thresholds
    STRONG_BULLISH_THRESHOLD = 0.03   # 3% predicted gain
    BULLISH_THRESHOLD = 0.01          # 1% predicted gain
    BEARISH_THRESHOLD = -0.01         # -1% predicted loss
    STRONG_BEARISH_THRESHOLD = -0.03  # -3% predicted loss
    MIN_CONFIDENCE = 0.70             # 70% confidence minimum
    
    def __init__(self, ml_model=None, mode: str = None):
        """
        Initialize signal generator.
        
        Parameters
        ----------
        ml_model : object, optional
            Trained ML model with predict method
        mode : str, optional
            Force specific mode ('FULL' or 'TECHNICAL_ONLY')
        """
        self.ml_model = ml_model
        self._mode = mode
        
        if mode is None:
            self._mode = self.MODE_FULL if ml_model is not None else self.MODE_TECHNICAL_ONLY
    
    @property
    def mode(self) -> str:
        return self._mode
    
    def check_ml_models_available(self) -> Dict[str, bool]:
        """Check which ML models are available."""
        models_dir = Path('models/saved_models')
        
        return {
            'xgboost': (models_dir / 'xgboost').exists(),
            'lstm': (models_dir / 'lstm').exists(),
            'random_forest': (models_dir / 'random_forest').exists(),
            'scaler': Path('models/scalers/feature_scaler.pkl').exists()
        }
    
    def analyze_stock(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Main analysis function - returns complete trading signal.
        
        Parameters
        ----------
        symbol : str
            Stock ticker symbol
        df : pd.DataFrame
            OHLCV data
        
        Returns
        -------
        dict
            Complete signal with action, confidence, entry/stop/target
        """
        # Step 1: Technical Analysis
        tech_signals = self._get_technical_signals(df)
        
        # Step 2: ML Prediction (if available)
        ml_prediction = None
        if self._mode == self.MODE_FULL and self.ml_model is not None:
            ml_prediction = self._get_ml_prediction(df)
        
        # Step 3: Combine signals
        if ml_prediction is not None:
            combined = self._combine_signals(tech_signals, ml_prediction)
        else:
            combined = self._create_technical_only_signal(tech_signals)
        
        # Step 4: Add context
        combined['symbol'] = symbol
        combined['current_price'] = float(df.iloc[-1]['Close'])
        combined['timestamp'] = str(df.index[-1])
        combined['mode'] = self._mode
        combined['technical_details'] = tech_signals
        combined['ml_details'] = ml_prediction
        
        return combined
    
    def _get_technical_signals(self, df: pd.DataFrame) -> Dict:
        """Analyze technical indicators and patterns."""
        # Get trend scores
        uptrend_score = check_uptrend(df)
        downtrend_score = check_downtrend(df)
        
        signals = {
            'signal': 'NEUTRAL',
            'confidence': 0.0,
            'patterns': [],
            'entry_price': None,
            'stop_loss': None,
            'trend_score': {
                'uptrend': uptrend_score,
                'downtrend': downtrend_score
            }
        }
        
        # Detect patterns in priority order
        
        # Tier 1: Base Breakout (85% base)
        breakout = detect_base_breakout(df)
        if breakout.iloc[-1]['buy_signal'] and uptrend_score >= 60:
            signals['signal'] = 'BUY'
            signals['confidence'] = breakout.iloc[-1]['confidence']
            signals['patterns'].append('BASE_BREAKOUT')
            signals['entry_price'] = float(breakout.iloc[-1]['entry_price'])
            signals['stop_loss'] = float(breakout.iloc[-1]['stop_loss'])
            return signals
        
        # Tier 2: Pullback Buy (75% base)
        pullback = detect_pullback_buy(df)
        if pullback.iloc[-1]['buy_signal'] and uptrend_score >= 65:
            signals['signal'] = 'BUY'
            signals['confidence'] = pullback.iloc[-1]['confidence']
            signals['patterns'].append('PULLBACK_BUY')
            signals['entry_price'] = float(pullback.iloc[-1]['entry_price'])
            signals['stop_loss'] = float(pullback.iloc[-1]['stop_loss'])
            return signals
        
        # Tier 3: 1-2-3 Continuation (70% base)
        continuation = detect_123_continuation(df)
        if continuation.iloc[-1]['buy_signal'] and uptrend_score >= 70:
            signals['signal'] = 'BUY'
            signals['confidence'] = continuation.iloc[-1]['confidence']
            signals['patterns'].append('123_CONTINUATION')
            signals['entry_price'] = float(continuation.iloc[-1]['entry_price'])
            signals['stop_loss'] = float(continuation.iloc[-1]['stop_loss'])
            return signals
        
        # Short patterns
        breakdown = detect_base_breakdown(df)
        if breakdown.iloc[-1]['sell_signal'] and downtrend_score >= 60:
            signals['signal'] = 'SELL'
            signals['confidence'] = breakdown.iloc[-1]['confidence']
            signals['patterns'].append('BASE_BREAKDOWN')
            signals['entry_price'] = float(breakdown.iloc[-1]['entry_price'])
            signals['stop_loss'] = float(breakdown.iloc[-1]['stop_loss'])
            return signals
        
        return signals
    
    def _get_ml_prediction(self, df: pd.DataFrame) -> Optional[MLPrediction]:
        """Get ML model prediction."""
        if self.ml_model is None:
            return None
        
        try:
            # Prepare features (implement based on your feature engineering)
            features = df.iloc[-1:].copy()
            
            # Get prediction
            prediction = self.ml_model.predict(features)
            
            current_price = df.iloc[-1]['Close']
            predicted_price = prediction[0] if hasattr(prediction, '__iter__') else prediction
            predicted_return = (predicted_price - current_price) / current_price
            
            # Get confidence if available
            confidence = 0.75
            if hasattr(self.ml_model, 'predict_proba'):
                proba = self.ml_model.predict_proba(features)
                confidence = max(proba[0]) if hasattr(proba[0], '__iter__') else proba[0]
            
            # Classify directional signal
            if predicted_return > self.STRONG_BULLISH_THRESHOLD:
                direction = 'STRONG_BUY'
            elif predicted_return > self.BULLISH_THRESHOLD:
                direction = 'BUY'
            elif predicted_return < self.STRONG_BEARISH_THRESHOLD:
                direction = 'STRONG_SELL'
            elif predicted_return < self.BEARISH_THRESHOLD:
                direction = 'SELL'
            else:
                direction = 'HOLD'
            
            return MLPrediction(
                predicted_price=float(predicted_price),
                predicted_return=float(predicted_return),
                confidence_score=float(confidence),
                directional_signal=direction
            )
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return None
    
    def _combine_signals(self, tech: Dict, ml: MLPrediction) -> Dict:
        """Combine technical and ML signals."""
        tech_signal = tech['signal']
        tech_confidence = tech['confidence']
        
        ml_return = ml.predicted_return
        ml_confidence = ml.confidence_score
        
        # CASE 1: STRONG BUY - Both agree on strong upside
        if (tech_signal == 'BUY' and 
            ml_return > self.STRONG_BULLISH_THRESHOLD and 
            ml_confidence > self.MIN_CONFIDENCE and
            tech_confidence > 70):
            return {
                'action': 'STRONG_BUY',
                'confidence': (tech_confidence + ml_confidence * 100) / 2,
                'entry_price': tech['entry_price'],
                'stop_loss': tech['stop_loss'],
                'target': tech['entry_price'] * (1 + ml_return) if tech['entry_price'] else None,
                'position_multiplier': 1.0
            }
        
        # CASE 2: BUY - Technical setup + ML moderately positive
        elif (tech_signal == 'BUY' and 
              ml_return > self.BULLISH_THRESHOLD and 
              ml_confidence > self.MIN_CONFIDENCE):
            return {
                'action': 'BUY',
                'confidence': (tech_confidence + ml_confidence * 100) / 2,
                'entry_price': tech['entry_price'],
                'stop_loss': tech['stop_loss'],
                'target': tech['entry_price'] * (1 + ml_return * 0.8) if tech['entry_price'] else None,
                'position_multiplier': 0.75
            }
        
        # CASE 3: SELL - Technical bearish + ML predicts decline
        elif (tech_signal == 'SELL' and 
              ml_return < self.BEARISH_THRESHOLD and 
              ml_confidence > self.MIN_CONFIDENCE):
            return {
                'action': 'SELL',
                'confidence': (tech_confidence + ml_confidence * 100) / 2,
                'entry_price': tech['entry_price'],
                'stop_loss': tech['stop_loss'],
                'target': tech['entry_price'] * (1 + ml_return * 0.8) if tech['entry_price'] else None,
                'position_multiplier': 0.75
            }
        
        # CASE 4: STRONG SELL
        elif (tech_signal == 'SELL' and 
              ml_return < self.STRONG_BEARISH_THRESHOLD and 
              ml_confidence > self.MIN_CONFIDENCE and
              tech_confidence > 70):
            return {
                'action': 'STRONG_SELL',
                'confidence': (tech_confidence + ml_confidence * 100) / 2,
                'entry_price': tech['entry_price'],
                'stop_loss': tech['stop_loss'],
                'target': tech['entry_price'] * (1 + ml_return) if tech['entry_price'] else None,
                'position_multiplier': 1.0
            }
        
        # CASE 5: HOLD - Mixed or low confidence
        else:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'entry_price': None,
                'stop_loss': None,
                'target': None,
                'position_multiplier': 0.0,
                'reason': 'Mixed signals or insufficient confidence'
            }
    
    def _create_technical_only_signal(self, tech: Dict) -> Dict:
        """Create signal based only on technical analysis."""
        signal = tech['signal']
        confidence = tech['confidence']
        
        # Cap confidence lower for technical-only mode
        capped_confidence = min(confidence * 0.9, 80.0)
        
        if signal == 'BUY' and confidence >= 70:
            # Calculate target (2.5:1 risk/reward)
            entry = tech['entry_price']
            stop = tech['stop_loss']
            risk = abs(entry - stop) if entry and stop else 0
            target = entry + (risk * 2.5) if entry and risk > 0 else None
            
            return {
                'action': 'BUY',
                'confidence': capped_confidence,
                'entry_price': entry,
                'stop_loss': stop,
                'target': target,
                'position_multiplier': 0.75,  # Reduce for technical-only
                'warning': 'Technical-only signal (ML not available). Consider reduced position size.'
            }
        
        elif signal == 'SELL' and confidence >= 70:
            entry = tech['entry_price']
            stop = tech['stop_loss']
            risk = abs(entry - stop) if entry and stop else 0
            target = entry - (risk * 2.5) if entry and risk > 0 else None
            
            return {
                'action': 'SELL',
                'confidence': capped_confidence,
                'entry_price': entry,
                'stop_loss': stop,
                'target': target,
                'position_multiplier': 0.75,
                'warning': 'Technical-only signal (ML not available). Consider reduced position size.'
            }
        
        else:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'entry_price': None,
                'stop_loss': None,
                'target': None,
                'position_multiplier': 0.0,
                'reason': 'No clear pattern detected or insufficient trend strength'
            }

"""
Advanced Indicators Component
Dynamic indicator management with sub-charts
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.theme import COLORS


# =============================================================================
# AVAILABLE INDICATORS
# =============================================================================

INDICATOR_CATALOG = {
    # Overlay indicators (on main chart)
    'overlay': {
        'SMA': {
            'name': 'Simple Moving Average',
            'params': {'period': 20},
            'color': COLORS['accent_blue']
        },
        'EMA': {
            'name': 'Exponential Moving Average', 
            'params': {'period': 12},
            'color': COLORS['accent_orange']
        },
        'BB': {
            'name': 'Bollinger Bands',
            'params': {'period': 20, 'std': 2},
            'color': '#9C27B0'
        },
        'VWAP': {
            'name': 'VWAP',
            'params': {},
            'color': '#00BCD4'
        }
    },
    # Sub-chart indicators
    'subchart': {
        'RSI': {
            'name': 'Relative Strength Index',
            'params': {'period': 14},
            'range': [0, 100],
            'levels': [30, 70]
        },
        'MACD': {
            'name': 'MACD',
            'params': {'fast': 12, 'slow': 26, 'signal': 9},
            'range': None,
            'levels': [0]
        },
        'ATR': {
            'name': 'Average True Range',
            'params': {'period': 14},
            'range': None,
            'levels': []
        },
        'Volume': {
            'name': 'Volume',
            'params': {},
            'range': None,
            'levels': []
        },
        'OBV': {
            'name': 'On-Balance Volume',
            'params': {},
            'range': None,
            'levels': []
        },
        'Stochastic': {
            'name': 'Stochastic Oscillator',
            'params': {'period': 14},
            'range': [0, 100],
            'levels': [20, 80]
        }
    }
}


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def init_indicators_state():
    """Initialize indicators in session state."""
    if 'active_overlay_indicators' not in st.session_state:
        st.session_state.active_overlay_indicators = ['SMA']
    
    if 'active_subchart_indicators' not in st.session_state:
        st.session_state.active_subchart_indicators = ['RSI']
    
    if 'indicator_params' not in st.session_state:
        st.session_state.indicator_params = {}


def get_active_overlays() -> List[str]:
    """Get list of active overlay indicators."""
    init_indicators_state()
    return st.session_state.active_overlay_indicators


def get_active_subcharts() -> List[str]:
    """Get list of active sub-chart indicators."""
    init_indicators_state()
    return st.session_state.active_subchart_indicators


def add_indicator(indicator_id: str, is_overlay: bool = False):
    """Add an indicator to active list."""
    init_indicators_state()
    if is_overlay:
        if indicator_id not in st.session_state.active_overlay_indicators:
            st.session_state.active_overlay_indicators.append(indicator_id)
    else:
        if indicator_id not in st.session_state.active_subchart_indicators:
            st.session_state.active_subchart_indicators.append(indicator_id)


def remove_indicator(indicator_id: str, is_overlay: bool = False):
    """Remove an indicator from active list."""
    init_indicators_state()
    if is_overlay:
        if indicator_id in st.session_state.active_overlay_indicators:
            st.session_state.active_overlay_indicators.remove(indicator_id)
    else:
        if indicator_id in st.session_state.active_subchart_indicators:
            st.session_state.active_subchart_indicators.remove(indicator_id)


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def calculate_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average."""
    return df['Close'].rolling(window=period).mean()


def calculate_ema(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return df['Close'].ewm(span=period, adjust=False).mean()


def calculate_bollinger(df: pd.DataFrame, period: int = 20, std: int = 2):
    """Calculate Bollinger Bands."""
    sma = df['Close'].rolling(window=period).mean()
    std_dev = df['Close'].rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return sma, upper, lower


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD."""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)


def calculate_stochastic(df: pd.DataFrame, period: int = 14):
    """Calculate Stochastic Oscillator."""
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=3).mean()
    return k, d


# =============================================================================
# CHART BUILDING
# =============================================================================

def create_chart_with_indicators(
    df: pd.DataFrame,
    title: str = "",
    height: int = 600
) -> go.Figure:
    """
    Create a complete chart with all active indicators.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    title : str
        Chart title
    height : int
        Total chart height
    
    Returns
    -------
    go.Figure
        Complete chart with indicators
    """
    init_indicators_state()
    
    active_subcharts = get_active_subcharts()
    num_subcharts = len(active_subcharts)
    
    # Calculate row heights
    if num_subcharts == 0:
        row_heights = [1.0]
        rows = 1
    else:
        main_height = 0.6
        sub_height = 0.4 / num_subcharts
        row_heights = [main_height] + [sub_height] * num_subcharts
        rows = 1 + num_subcharts
    
    # Create subplots
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=[title] + active_subcharts if num_subcharts > 0 else [title]
    )
    
    # Add candlestick to main chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color=COLORS['accent_green'],
        decreasing_line_color=COLORS['accent_red']
    ), row=1, col=1)
    
    # Add overlay indicators
    for ind_id in get_active_overlays():
        ind_config = INDICATOR_CATALOG['overlay'].get(ind_id, {})
        params = st.session_state.indicator_params.get(ind_id, ind_config.get('params', {}))
        
        if ind_id == 'SMA':
            period = params.get('period', 20)
            sma = calculate_sma(df, period)
            fig.add_trace(go.Scatter(
                x=df.index, y=sma,
                name=f'SMA {period}',
                line=dict(color=ind_config['color'], width=1.5)
            ), row=1, col=1)
        
        elif ind_id == 'EMA':
            period = params.get('period', 12)
            ema = calculate_ema(df, period)
            fig.add_trace(go.Scatter(
                x=df.index, y=ema,
                name=f'EMA {period}',
                line=dict(color=ind_config['color'], width=1.5)
            ), row=1, col=1)
        
        elif ind_id == 'BB':
            period = params.get('period', 20)
            std = params.get('std', 2)
            sma, upper, lower = calculate_bollinger(df, period, std)
            
            fig.add_trace(go.Scatter(
                x=df.index, y=upper,
                name='BB Upper',
                line=dict(color=ind_config['color'], width=1, dash='dot'),
                showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=lower,
                name='BB Lower',
                line=dict(color=ind_config['color'], width=1, dash='dot'),
                fill='tonexty',
                fillcolor=f"{ind_config['color']}20",
                showlegend=True
            ), row=1, col=1)
        
        elif ind_id == 'VWAP':
            vwap = calculate_vwap(df)
            fig.add_trace(go.Scatter(
                x=df.index, y=vwap,
                name='VWAP',
                line=dict(color=ind_config['color'], width=1.5, dash='dash')
            ), row=1, col=1)
    
    # Add sub-chart indicators
    for i, ind_id in enumerate(active_subcharts):
        row = i + 2
        ind_config = INDICATOR_CATALOG['subchart'].get(ind_id, {})
        
        if ind_id == 'RSI':
            rsi = calculate_rsi(df)
            fig.add_trace(go.Scatter(
                x=df.index, y=rsi,
                name='RSI',
                line=dict(color=COLORS['accent_orange'], width=1.5)
            ), row=row, col=1)
            
            # Add levels
            fig.add_hline(y=70, line=dict(color=COLORS['accent_red'], dash='dash', width=1), row=row, col=1)
            fig.add_hline(y=30, line=dict(color=COLORS['accent_green'], dash='dash', width=1), row=row, col=1)
        
        elif ind_id == 'MACD':
            macd, signal, hist = calculate_macd(df)
            
            colors = [COLORS['accent_green'] if v >= 0 else COLORS['accent_red'] for v in hist]
            fig.add_trace(go.Bar(
                x=df.index, y=hist,
                name='MACD Hist',
                marker_color=colors,
                showlegend=False
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=macd,
                name='MACD',
                line=dict(color=COLORS['accent_blue'], width=1.5)
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=signal,
                name='Signal',
                line=dict(color=COLORS['accent_orange'], width=1.5)
            ), row=row, col=1)
        
        elif ind_id == 'ATR':
            atr = calculate_atr(df)
            fig.add_trace(go.Scatter(
                x=df.index, y=atr,
                name='ATR',
                line=dict(color=COLORS['accent_blue'], width=1.5),
                fill='tozeroy',
                fillcolor=f"{COLORS['accent_blue']}20"
            ), row=row, col=1)
        
        elif ind_id == 'Volume':
            colors = [COLORS['accent_green'] if df['Close'].iloc[j] >= df['Open'].iloc[j] 
                     else COLORS['accent_red'] for j in range(len(df))]
            fig.add_trace(go.Bar(
                x=df.index, y=df['Volume'],
                name='Volume',
                marker_color=colors
            ), row=row, col=1)
        
        elif ind_id == 'OBV':
            obv = calculate_obv(df)
            fig.add_trace(go.Scatter(
                x=df.index, y=obv,
                name='OBV',
                line=dict(color=COLORS['accent_blue'], width=1.5)
            ), row=row, col=1)
        
        elif ind_id == 'Stochastic':
            k, d = calculate_stochastic(df)
            fig.add_trace(go.Scatter(
                x=df.index, y=k,
                name='%K',
                line=dict(color=COLORS['accent_blue'], width=1.5)
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=d,
                name='%D',
                line=dict(color=COLORS['accent_orange'], width=1.5)
            ), row=row, col=1)
            
            fig.add_hline(y=80, line=dict(color=COLORS['accent_red'], dash='dash', width=1), row=row, col=1)
            fig.add_hline(y=20, line=dict(color=COLORS['accent_green'], dash='dash', width=1), row=row, col=1)
    
    # Update layout
    fig.update_layout(
        height=height,
        paper_bgcolor=COLORS['bg_primary'],
        plot_bgcolor=COLORS['bg_primary'],
        font=dict(color=COLORS['text_secondary'], size=11),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=10, r=10, t=40, b=30),
        xaxis_rangeslider_visible=False
    )
    
    # Update all axes
    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True, side='right', row=i, col=1)
    
    return fig


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_indicator_selector():
    """Render the indicator selector panel."""
    init_indicators_state()
    
    st.markdown(f"""
    <div style="color: {COLORS['text_primary']}; font-weight: 600; font-size: 14px; margin-bottom: 12px;">
        📊 Indicators
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<small style='color:{COLORS['text_muted']}'>Overlay</small>", unsafe_allow_html=True)
        for ind_id, config in INDICATOR_CATALOG['overlay'].items():
            is_active = ind_id in get_active_overlays()
            if st.checkbox(config['name'], value=is_active, key=f"ov_{ind_id}"):
                if not is_active:
                    add_indicator(ind_id, is_overlay=True)
            else:
                if is_active:
                    remove_indicator(ind_id, is_overlay=True)
    
    with col2:
        st.markdown(f"<small style='color:{COLORS['text_muted']}'>Sub-Charts</small>", unsafe_allow_html=True)
        for ind_id, config in INDICATOR_CATALOG['subchart'].items():
            is_active = ind_id in get_active_subcharts()
            if st.checkbox(config['name'], value=is_active, key=f"sc_{ind_id}"):
                if not is_active:
                    add_indicator(ind_id, is_overlay=False)
            else:
                if is_active:
                    remove_indicator(ind_id, is_overlay=False)


def render_indicator_toolbar():
    """Render compact indicator toolbar."""
    init_indicators_state()
    
    with st.expander("📊 Indicators", expanded=False):
        render_indicator_selector()
        
        # Quick presets
        st.markdown(f"<hr style='border-color:{COLORS['border']}'>", unsafe_allow_html=True)
        st.markdown(f"<small style='color:{COLORS['text_muted']}'>Quick Presets</small>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Trend", use_container_width=True, key="preset_trend"):
                st.session_state.active_overlay_indicators = ['SMA', 'EMA']
                st.session_state.active_subchart_indicators = ['MACD']
                st.rerun()
        
        with col2:
            if st.button("📊 Momentum", use_container_width=True, key="preset_momentum"):
                st.session_state.active_overlay_indicators = ['BB']
                st.session_state.active_subchart_indicators = ['RSI', 'Stochastic']
                st.rerun()
        
        with col3:
            if st.button("📉 Volume", use_container_width=True, key="preset_volume"):
                st.session_state.active_overlay_indicators = ['VWAP']
                st.session_state.active_subchart_indicators = ['Volume', 'OBV']
                st.rerun()

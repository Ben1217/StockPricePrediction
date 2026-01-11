"""
Multi-Timeframe Component
Quick timeframe switcher and multi-view
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.theme import COLORS


# =============================================================================
# TIMEFRAME DEFINITIONS
# =============================================================================

TIMEFRAMES = {
    '1d': {'label': '1D', 'period': '1d', 'interval': '1m', 'tooltip': '1 minute bars for today'},
    '5d': {'label': '5D', 'period': '5d', 'interval': '5m', 'tooltip': '5 minute bars for 5 days'},
    '1mo': {'label': '1M', 'period': '1mo', 'interval': '30m', 'tooltip': '30 minute bars for 1 month'},
    '3mo': {'label': '3M', 'period': '3mo', 'interval': '1d', 'tooltip': 'Daily bars for 3 months'},
    '6mo': {'label': '6M', 'period': '6mo', 'interval': '1d', 'tooltip': 'Daily bars for 6 months'},
    '1y': {'label': '1Y', 'period': '1y', 'interval': '1d', 'tooltip': 'Daily bars for 1 year'},
    '5y': {'label': '5Y', 'period': '5y', 'interval': '1wk', 'tooltip': 'Weekly bars for 5 years'},
    'max': {'label': 'MAX', 'period': 'max', 'interval': '1mo', 'tooltip': 'Monthly bars for all time'}
}

DEFAULT_TIMEFRAME = '1y'


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def init_timeframe_state():
    """Initialize timeframe state."""
    if 'selected_timeframe' not in st.session_state:
        st.session_state.selected_timeframe = DEFAULT_TIMEFRAME
    
    if 'timeframe_data_cache' not in st.session_state:
        st.session_state.timeframe_data_cache = {}


def get_selected_timeframe() -> str:
    """Get currently selected timeframe."""
    init_timeframe_state()
    return st.session_state.selected_timeframe


def set_timeframe(tf: str):
    """Set the active timeframe."""
    init_timeframe_state()
    st.session_state.selected_timeframe = tf


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=60)
def load_timeframe_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Load data for a specific timeframe.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    period : str
        yfinance period (1d, 5d, 1mo, etc.)
    interval : str
        yfinance interval (1m, 5m, 1h, 1d, etc.)
    
    Returns
    -------
    pd.DataFrame
        OHLCV data
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()


def get_data_for_timeframe(symbol: str, tf_key: str) -> pd.DataFrame:
    """Get data for a specific timeframe key."""
    init_timeframe_state()
    
    tf_config = TIMEFRAMES.get(tf_key, TIMEFRAMES[DEFAULT_TIMEFRAME])
    return load_timeframe_data(symbol, tf_config['period'], tf_config['interval'])


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_timeframe_switcher(symbol: str = None, on_change: callable = None):
    """
    Render the timeframe switcher bar.
    
    Parameters
    ----------
    symbol : str, optional
        Current symbol for data loading
    on_change : callable, optional
        Callback when timeframe changes
    """
    init_timeframe_state()
    
    current_tf = get_selected_timeframe()
    
    # Create button row
    cols = st.columns(len(TIMEFRAMES))
    
    for i, (tf_key, tf_config) in enumerate(TIMEFRAMES.items()):
        with cols[i]:
            is_active = tf_key == current_tf
            
            button_style = f"""
            <style>
            div[data-testid="stButton"] button[kind="secondary"][key="tf_{tf_key}"] {{
                background: {COLORS['accent_orange'] if is_active else 'transparent'};
                color: {'#000' if is_active else COLORS['text_muted']};
                border: {'none' if is_active else f"1px solid {COLORS['border']}"};
                font-weight: {'600' if is_active else '400'};
            }}
            </style>
            """
            
            if st.button(
                tf_config['label'],
                key=f"tf_{tf_key}",
                use_container_width=True,
                help=tf_config['tooltip']
            ):
                set_timeframe(tf_key)
                if on_change:
                    on_change(tf_key)
                st.rerun()


def render_timeframe_bar():
    """Render a compact timeframe bar using HTML."""
    init_timeframe_state()
    current_tf = get_selected_timeframe()
    
    buttons_html = ""
    for tf_key, tf_config in TIMEFRAMES.items():
        is_active = tf_key == current_tf
        bg = COLORS['accent_orange'] if is_active else 'transparent'
        color = '#000' if is_active else COLORS['text_muted']
        border = 'none' if is_active else f"1px solid {COLORS['border']}"
        weight = '600' if is_active else '400'
        
        buttons_html += f"""
        <button style="
            background: {bg};
            color: {color};
            border: {border};
            border-radius: 4px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: {weight};
            cursor: pointer;
            margin-right: 4px;
        " title="{tf_config['tooltip']}">{tf_config['label']}</button>
        """
    
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        padding: 8px 0;
    ">
        {buttons_html}
    </div>
    """, unsafe_allow_html=True)


def render_timeframe_tabs(symbol: str):
    """
    Render timeframe selection as tabs.
    
    Parameters
    ----------
    symbol : str
        Stock symbol for data loading
    
    Returns
    -------
    pd.DataFrame
        Data for the selected timeframe
    """
    init_timeframe_state()
    
    # Create tabs for timeframes
    tab_labels = [tf['label'] for tf in TIMEFRAMES.values()]
    tabs = st.tabs(tab_labels)
    
    selected_data = None
    
    for i, (tf_key, tf_config) in enumerate(TIMEFRAMES.items()):
        with tabs[i]:
            if st.session_state.selected_timeframe == tf_key or selected_data is None:
                data = get_data_for_timeframe(symbol, tf_key)
                if not data.empty:
                    selected_data = data
                    st.session_state.selected_timeframe = tf_key
    
    return selected_data if selected_data is not None else pd.DataFrame()

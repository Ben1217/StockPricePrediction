"""
Watchlist Component
Interactive watchlist for tracking multiple symbols
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional, Callable
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.theme import COLORS


# Default watchlist symbols
DEFAULT_WATCHLIST = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']


def get_watchlist_data(symbols: List[str]) -> List[Dict]:
    """
    Fetch current data for watchlist symbols.
    
    Parameters
    ----------
    symbols : list
        List of stock symbols
    
    Returns
    -------
    list
        List of dicts with symbol data
    """
    data = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='2d')
            
            if len(hist) >= 1:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) >= 2 else current
                change = ((current - prev) / prev) * 100 if prev > 0 else 0
                
                data.append({
                    'symbol': symbol,
                    'price': current,
                    'change': change,
                    'volume': hist['Volume'].iloc[-1] if 'Volume' in hist else 0
                })
        except Exception:
            # Add placeholder if fetch fails
            data.append({
                'symbol': symbol,
                'price': 0,
                'change': 0,
                'volume': 0,
                'error': True
            })
    
    return data


def init_watchlist_state():
    """Initialize watchlist in session state."""
    if 'watchlist_symbols' not in st.session_state:
        st.session_state.watchlist_symbols = DEFAULT_WATCHLIST.copy()
    
    if 'watchlist_data' not in st.session_state:
        st.session_state.watchlist_data = []


def add_to_watchlist(symbol: str) -> bool:
    """
    Add symbol to watchlist.
    
    Parameters
    ----------
    symbol : str
        Stock symbol to add
    
    Returns
    -------
    bool
        True if added, False if already exists
    """
    init_watchlist_state()
    symbol = symbol.upper().strip()
    
    if symbol and symbol not in st.session_state.watchlist_symbols:
        st.session_state.watchlist_symbols.append(symbol)
        return True
    return False


def remove_from_watchlist(symbol: str) -> bool:
    """
    Remove symbol from watchlist.
    
    Parameters
    ----------
    symbol : str
        Stock symbol to remove
    
    Returns
    -------
    bool
        True if removed, False if not found
    """
    init_watchlist_state()
    symbol = symbol.upper().strip()
    
    if symbol in st.session_state.watchlist_symbols:
        st.session_state.watchlist_symbols.remove(symbol)
        return True
    return False


def render_watchlist_panel(on_symbol_select: Optional[Callable] = None):
    """
    Render the complete watchlist panel.
    
    Parameters
    ----------
    on_symbol_select : callable, optional
        Callback when a symbol is clicked
    """
    init_watchlist_state()
    
    # Header with add button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div style="
            color: {COLORS['text_primary']};
            font-weight: 600;
            font-size: 14px;
            padding: 4px 0;
        ">📋 Watchlist</div>
        """, unsafe_allow_html=True)
    
    # Add symbol input
    with st.expander("➕ Add Symbol", expanded=False):
        new_symbol = st.text_input(
            "Symbol",
            key="watchlist_add_input",
            placeholder="e.g., META",
            label_visibility="collapsed"
        )
        if st.button("Add", key="watchlist_add_btn", use_container_width=True):
            if new_symbol:
                if add_to_watchlist(new_symbol):
                    st.success(f"Added {new_symbol.upper()}")
                    st.rerun()
                else:
                    st.warning(f"{new_symbol.upper()} already in watchlist")
    
    # Fetch data if needed (cache for 60 seconds)
    @st.cache_data(ttl=60)
    def cached_watchlist_data(symbols_tuple):
        return get_watchlist_data(list(symbols_tuple))
    
    watchlist_data = cached_watchlist_data(tuple(st.session_state.watchlist_symbols))
    
    # Render watchlist items
    st.markdown(f"""
    <div style="
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        overflow: hidden;
        margin-top: 8px;
    ">
    """, unsafe_allow_html=True)
    
    for i, item in enumerate(watchlist_data):
        symbol = item['symbol']
        price = item.get('price', 0)
        change = item.get('change', 0)
        has_error = item.get('error', False)
        
        change_color = COLORS['accent_green'] if change >= 0 else COLORS['accent_red']
        bg_color = COLORS['bg_secondary'] if i % 2 == 0 else COLORS['bg_primary']
        
        # Create columns for symbol button and stats
        cols = st.columns([2, 1.5, 0.5])
        
        with cols[0]:
            if st.button(
                symbol,
                key=f"wl_sym_{symbol}",
                use_container_width=True,
                help=f"Click to analyze {symbol}"
            ):
                if on_symbol_select:
                    on_symbol_select(symbol)
                else:
                    st.session_state['selected_symbol'] = symbol
                    st.session_state['ticker'] = symbol
        
        with cols[1]:
            if not has_error:
                st.markdown(f"""
                <div style="
                    display: flex;
                    flex-direction: column;
                    align-items: flex-end;
                    padding: 6px 0;
                    font-size: 12px;
                ">
                    <span style="color: {COLORS['text_primary']}; font-weight: 500;">${price:.2f}</span>
                    <span style="color: {change_color};">{change:+.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="color: {COLORS['text_muted']}; font-size: 11px; padding: 10px 0;">
                    Error
                </div>
                """, unsafe_allow_html=True)
        
        with cols[2]:
            if st.button("✕", key=f"wl_rm_{symbol}", help=f"Remove {symbol}"):
                remove_from_watchlist(symbol)
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Refresh button
    if st.button("🔄 Refresh", key="watchlist_refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


def render_mini_watchlist(max_items: int = 5, on_symbol_select: Optional[Callable] = None):
    """
    Render a compact mini watchlist for sidebar.
    
    Parameters
    ----------
    max_items : int
        Maximum items to show
    on_symbol_select : callable, optional
        Callback when symbol is clicked
    """
    init_watchlist_state()
    
    @st.cache_data(ttl=60)
    def cached_data(symbols_tuple):
        return get_watchlist_data(list(symbols_tuple))
    
    symbols = st.session_state.watchlist_symbols[:max_items]
    data = cached_data(tuple(symbols))
    
    st.markdown(f"""
    <div style="
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 12px;
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid {COLORS['border']};
        ">
            <span style="color: {COLORS['text_primary']}; font-weight: 600; font-size: 12px;">
                📋 Watchlist
            </span>
            <span style="color: {COLORS['text_muted']}; font-size: 11px;">
                {len(st.session_state.watchlist_symbols)} items
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    for item in data:
        symbol = item['symbol']
        change = item.get('change', 0)
        change_color = COLORS['accent_green'] if change >= 0 else COLORS['accent_red']
        
        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 0;
            cursor: pointer;
        " onclick="window.parent.postMessage({{type:'watchlist_select', symbol:'{symbol}'}}, '*')">
            <span style="color: {COLORS['text_secondary']}; font-size: 12px;">{symbol}</span>
            <span style="color: {change_color}; font-size: 11px; font-weight: 500;">{change:+.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

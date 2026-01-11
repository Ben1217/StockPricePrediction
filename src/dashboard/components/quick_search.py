"""
Quick Search Component
Ctrl+K modal with symbol autocomplete
"""

import streamlit as st
from typing import List, Callable, Optional
import yfinance as yf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.theme import COLORS


# =============================================================================
# POPULAR SYMBOLS
# =============================================================================

POPULAR_SYMBOLS = [
    ('SPY', 'SPDR S&P 500 ETF'),
    ('QQQ', 'Invesco QQQ Trust'),
    ('IWM', 'iShares Russell 2000'),
    ('DIA', 'SPDR Dow Jones'),
    ('AAPL', 'Apple Inc.'),
    ('MSFT', 'Microsoft Corp.'),
    ('GOOGL', 'Alphabet Inc.'),
    ('AMZN', 'Amazon.com Inc.'),
    ('NVDA', 'NVIDIA Corp.'),
    ('TSLA', 'Tesla Inc.'),
    ('META', 'Meta Platforms'),
    ('AMD', 'Advanced Micro Devices'),
    ('JPM', 'JPMorgan Chase'),
    ('V', 'Visa Inc.'),
    ('JNJ', 'Johnson & Johnson'),
    ('WMT', 'Walmart Inc.'),
    ('PG', 'Procter & Gamble'),
    ('XOM', 'Exxon Mobil'),
    ('BAC', 'Bank of America'),
    ('DIS', 'Walt Disney')
]

SECTOR_ETFS = [
    ('XLK', 'Technology'),
    ('XLF', 'Financials'),
    ('XLV', 'Healthcare'),
    ('XLE', 'Energy'),
    ('XLY', 'Consumer Discretionary'),
    ('XLP', 'Consumer Staples'),
    ('XLI', 'Industrials'),
    ('XLB', 'Materials'),
    ('XLU', 'Utilities'),
    ('XLRE', 'Real Estate')
]


# =============================================================================
# SYMBOL VALIDATION
# =============================================================================

@st.cache_data(ttl=3600)
def validate_symbol(symbol: str) -> bool:
    """Check if a symbol is valid."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return 'regularMarketPrice' in info or 'currentPrice' in info
    except:
        return False


def search_symbols(query: str, limit: int = 10) -> List[tuple]:
    """
    Search for symbols matching query.
    
    Parameters
    ----------
    query : str
        Search query
    limit : int
        Maximum results
    
    Returns
    -------
    list
        List of (symbol, name) tuples
    """
    query = query.upper().strip()
    if not query:
        return POPULAR_SYMBOLS[:limit]
    
    # Search in popular symbols
    matches = []
    for symbol, name in POPULAR_SYMBOLS + SECTOR_ETFS:
        if query in symbol or query.lower() in name.lower():
            matches.append((symbol, name))
    
    # If exact match, put it first
    exact_matches = [m for m in matches if m[0] == query]
    other_matches = [m for m in matches if m[0] != query]
    
    return (exact_matches + other_matches)[:limit]


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def init_search_state():
    """Initialize search state."""
    if 'search_open' not in st.session_state:
        st.session_state.search_open = False
    
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ''
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = POPULAR_SYMBOLS[:8]


def open_search():
    """Open the search modal."""
    init_search_state()
    st.session_state.search_open = True


def close_search():
    """Close the search modal."""
    init_search_state()
    st.session_state.search_open = False
    st.session_state.search_query = ''


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_search_modal(on_select: Optional[Callable] = None):
    """
    Render the quick search modal.
    
    Parameters
    ----------
    on_select : callable
        Callback when a symbol is selected, receives (symbol, name)
    """
    init_search_state()
    
    # Search modal container
    st.markdown(f"""
    <div style="
        background: {COLORS['bg_secondary']};
        border: 2px solid {COLORS['accent_orange']};
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        ">
            <span style="color: {COLORS['text_primary']}; font-weight: 600;">
                🔍 Quick Search
            </span>
            <span style="color: {COLORS['text_muted']}; font-size: 11px;">
                Press Esc to close
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search input
    query = st.text_input(
        "Search",
        value=st.session_state.search_query,
        placeholder="Type symbol or company name...",
        key="quick_search_input",
        label_visibility="collapsed"
    )
    
    if query != st.session_state.search_query:
        st.session_state.search_query = query
        st.session_state.search_results = search_symbols(query)
    
    # Results
    results = st.session_state.search_results
    
    if results:
        st.markdown(f"<small style='color:{COLORS['text_muted']}; margin-bottom: 8px; display: block;'>Results</small>", unsafe_allow_html=True)
        
        for symbol, name in results:
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if st.button(symbol, key=f"qs_{symbol}", use_container_width=True):
                    if on_select:
                        on_select(symbol, name)
                    close_search()
                    st.rerun()
            
            with col2:
                st.markdown(f"""
                <div style="
                    color: {COLORS['text_muted']};
                    font-size: 12px;
                    padding-top: 8px;
                ">{name}</div>
                """, unsafe_allow_html=True)
    
    # Categories
    st.markdown(f"<hr style='border-color:{COLORS['border']}; margin: 16px 0;'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<small style='color:{COLORS['text_muted']}'>📈 Popular</small>", unsafe_allow_html=True)
        for symbol, name in POPULAR_SYMBOLS[:5]:
            if st.button(f"{symbol}", key=f"pop_{symbol}", use_container_width=True):
                if on_select:
                    on_select(symbol, name)
                close_search()
                st.rerun()
    
    with col2:
        st.markdown(f"<small style='color:{COLORS['text_muted']}'>🏢 Sectors</small>", unsafe_allow_html=True)
        for symbol, name in SECTOR_ETFS[:5]:
            if st.button(f"{symbol} - {name}", key=f"sec_{symbol}", use_container_width=True):
                if on_select:
                    on_select(symbol, name)
                close_search()
                st.rerun()


def render_search_trigger():
    """Render a search trigger button."""
    init_search_state()
    
    if st.button("🔍 Search (Ctrl+K)", key="open_search", use_container_width=True):
        open_search()
        st.rerun()


def render_search_bar(on_select: Optional[Callable] = None):
    """Render an inline search bar."""
    init_search_state()
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Search Symbol",
            placeholder="🔍 Search symbol...",
            key="inline_search",
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("Go", key="search_go", use_container_width=True):
            if query and validate_symbol(query.upper()):
                if on_select:
                    on_select(query.upper(), "")
                st.rerun()
            else:
                st.error("Invalid symbol")
    
    # Show suggestions if typing
    if query:
        results = search_symbols(query, 5)
        if results:
            for symbol, name in results:
                if st.button(f"{symbol} - {name}", key=f"sug_{symbol}", use_container_width=True):
                    if on_select:
                        on_select(symbol, name)
                    st.rerun()

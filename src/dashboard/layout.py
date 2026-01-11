"""
Dashboard Layout Components
Professional trading platform layout with top bar, sidebar, and main content area
"""

import streamlit as st
from datetime import datetime
from typing import Optional, List, Dict
import yfinance as yf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dashboard.theme import COLORS


# =============================================================================
# TOP STATUS BAR
# =============================================================================

def get_market_status() -> Dict:
    """Get current market status and key indices."""
    try:
        spy = yf.Ticker('SPY')
        info = spy.info
        market_state = info.get('marketState', 'UNKNOWN')
        
        status_map = {
            'CLOSED': ('🔵', 'Market Closed', '#64B5F6'),
            'PRE': ('🟡', 'Pre-Market', '#FFD54F'),
            'POST': ('🟡', 'After-Hours', '#FFD54F'),
            'REGULAR': ('🟢', 'Market Open', '#81C784'),
        }
        
        emoji, text, color = status_map.get(market_state, ('⚪', 'Unknown', '#9E9E9E'))
        
        return {
            'emoji': emoji,
            'text': text,
            'color': color,
            'state': market_state
        }
    except Exception:
        return {
            'emoji': '⚪',
            'text': 'Unknown',
            'color': '#9E9E9E',
            'state': 'UNKNOWN'
        }


@st.cache_data(ttl=60)
def get_index_quotes() -> List[Dict]:
    """Get quick quotes for major indices."""
    indices = [
        ('SPY', 'S&P 500'),
        ('QQQ', 'Nasdaq'),
        ('DIA', 'Dow Jones'),
        ('VIX', 'VIX')
    ]
    
    quotes = []
    for symbol, name in indices:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='2d')
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                quotes.append({
                    'symbol': symbol,
                    'name': name,
                    'price': current,
                    'change': change,
                    'color': COLORS['accent_green'] if change >= 0 else COLORS['accent_red']
                })
        except Exception:
            pass
    
    return quotes


def render_top_bar(alert_count: int = 0):
    """
    Render the top status bar with market status, indices, and alerts.
    """
    market = get_market_status()
    quotes = get_index_quotes()
    current_time = datetime.now().strftime('%H:%M:%S')
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(180deg, {COLORS['bg_secondary']} 0%, {COLORS['bg_primary']} 100%);
        border-bottom: 1px solid {COLORS['border']};
        padding: 8px 16px;
        margin: -1rem -2rem 1rem -2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    ">
        <!-- Market Status -->
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="
                background: {market['color']}22;
                border: 1px solid {market['color']};
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 13px;
                color: {market['color']};
                font-weight: 600;
            ">
                {market['emoji']} {market['text']}
            </div>
            
            <!-- Index Quotes -->
            <div style="display: flex; gap: 16px;">
                {''.join([f'''
                <div style="color: {COLORS['text_secondary']}; font-size: 13px;">
                    <span style="color: {COLORS['text_muted']};">{q['symbol']}</span>
                    <span style="color: {q['color']}; font-weight: 600; margin-left: 4px;">
                        {q['change']:+.2f}%
                    </span>
                </div>
                ''' for q in quotes])}
            </div>
        </div>
        
        <!-- Right side: Time and Alerts -->
        <div style="display: flex; align-items: center; gap: 16px;">
            <span style="color: {COLORS['text_muted']}; font-size: 13px; font-family: 'Roboto Mono', monospace;">
                {current_time}
            </span>
            <div style="
                position: relative;
                cursor: pointer;
                padding: 4px 8px;
                border-radius: 4px;
                background: {'#FF5252' + '33' if alert_count > 0 else 'transparent'};
            ">
                🔔
                {f'<span style="position: absolute; top: -2px; right: -2px; background: #FF5252; color: white; border-radius: 50%; width: 16px; height: 16px; font-size: 10px; display: flex; align-items: center; justify-content: center;">{alert_count}</span>' if alert_count > 0 else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR COMPONENTS
# =============================================================================

def render_sidebar_header():
    """Render the sidebar header with logo."""
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 16px 0;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 16px;
    ">
        <h1 style="
            margin: 0;
            color: {COLORS['accent_orange']};
            font-size: 22px;
            font-weight: 700;
        ">📈 TradeView Pro</h1>
        <p style="
            margin: 4px 0 0 0;
            color: {COLORS['text_muted']};
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
        ">Professional Trading</p>
    </div>
    """, unsafe_allow_html=True)


def render_watchlist(watchlist: List[Dict], on_select=None):
    """
    Render the watchlist component in sidebar.
    
    Parameters
    ----------
    watchlist : list
        List of dicts with symbol, price, change
    on_select : callable, optional
        Callback when symbol is selected
    """
    st.markdown(f"""
    <div style="
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 16px;
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        ">
            <span style="
                color: {COLORS['text_primary']};
                font-weight: 600;
                font-size: 13px;
            ">📋 Watchlist</span>
            <span style="
                color: {COLORS['accent_orange']};
                font-size: 11px;
                cursor: pointer;
            ">+ Add</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render each watchlist item as a button
    for item in watchlist:
        symbol = item.get('symbol', '')
        price = item.get('price', 0)
        change = item.get('change', 0)
        change_color = COLORS['accent_green'] if change >= 0 else COLORS['accent_red']
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button(symbol, key=f"wl_{symbol}", use_container_width=True):
                if on_select:
                    on_select(symbol)
        with col2:
            st.markdown(f"""
            <div style="
                text-align: right;
                padding-top: 8px;
                color: {change_color};
                font-size: 12px;
            ">{change:+.2f}%</div>
            """, unsafe_allow_html=True)


def render_quick_stats(data: Dict):
    """Render quick stats panel in sidebar."""
    if not data:
        return
    
    st.markdown(f"""
    <div style="
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 16px;
    ">
        <div style="
            color: {COLORS['text_primary']};
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 10px;
        ">📊 Quick Stats</div>
        
        <div style="display: grid; gap: 8px;">
            <div style="display: flex; justify-content: space-between;">
                <span style="color: {COLORS['text_muted']}; font-size: 12px;">Price</span>
                <span style="color: {COLORS['text_primary']}; font-size: 12px; font-weight: 600;">
                    ${data.get('price', 0):.2f}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: {COLORS['text_muted']}; font-size: 12px;">Change</span>
                <span style="
                    color: {COLORS['accent_green'] if data.get('change', 0) >= 0 else COLORS['accent_red']};
                    font-size: 12px;
                    font-weight: 600;
                ">{data.get('change', 0):+.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: {COLORS['text_muted']}; font-size: 12px;">RSI</span>
                <span style="color: {COLORS['text_primary']}; font-size: 12px;">{data.get('rsi', 0):.1f}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: {COLORS['text_muted']}; font-size: 12px;">Signal</span>
                <span style="
                    color: {COLORS['accent_green'] if data.get('signal') == 'BUY' else COLORS['accent_red'] if data.get('signal') == 'SELL' else COLORS['accent_orange']};
                    font-size: 12px;
                    font-weight: 600;
                ">{data.get('signal', 'HOLD')}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# KEYBOARD SHORTCUTS
# =============================================================================

def inject_keyboard_shortcuts():
    """Inject keyboard shortcut handlers."""
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        // Ctrl+K - Quick search (will be handled by Streamlit)
        if (e.ctrlKey && e.key === 'k') {
            e.preventDefault();
            // Trigger search modal
            window.parent.postMessage({type: 'shortcut', action: 'search'}, '*');
        }
        
        // Number keys 1-5 for tabs
        if (!e.ctrlKey && !e.altKey && !e.metaKey) {
            const tabKeys = ['1', '2', '3', '4', '5'];
            if (tabKeys.includes(e.key)) {
                // Tab switching handled by Streamlit session state
                window.parent.postMessage({type: 'shortcut', action: 'tab', tab: parseInt(e.key)}, '*');
            }
        }
        
        // F key for fullscreen chart
        if (e.key === 'f' && !e.ctrlKey) {
            window.parent.postMessage({type: 'shortcut', action: 'fullscreen'}, '*');
        }
    });
    </script>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB STYLING
# =============================================================================

def get_enhanced_tab_css() -> str:
    """Get enhanced CSS for professional tab styling."""
    return f"""
    <style>
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        background: linear-gradient(180deg, {COLORS['bg_secondary']} 0%, {COLORS['bg_primary']} 100%);
        border-radius: 8px 8px 0 0;
        padding: 8px 8px 0 8px;
        gap: 4px;
        border-bottom: 2px solid {COLORS['border']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        color: {COLORS['text_muted']};
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: 500;
        border: 1px solid transparent;
        border-bottom: none;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: {COLORS['bg_hover']};
        color: {COLORS['text_secondary']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {COLORS['bg_primary']} !important;
        color: {COLORS['accent_orange']} !important;
        font-weight: 600;
        border: 1px solid {COLORS['border']};
        border-bottom: 2px solid {COLORS['bg_primary']};
        margin-bottom: -2px;
    }}
    
    /* Tab content padding */
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 20px;
    }}
    </style>
    """

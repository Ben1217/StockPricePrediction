"""
Status Bar Component
Top status bar with market status, indices, and alerts
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict
import yfinance as yf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.theme import COLORS


@st.cache_data(ttl=30)
def get_market_status() -> Dict:
    """Get current market status."""
    try:
        spy = yf.Ticker('SPY')
        info = spy.info
        market_state = info.get('marketState', 'UNKNOWN')
        
        status_map = {
            'CLOSED': ('🔵', 'Closed', COLORS['accent_blue']),
            'PRE': ('🟡', 'Pre-Market', '#FFD54F'),
            'POST': ('🟡', 'After-Hours', '#FFD54F'),
            'REGULAR': ('🟢', 'Open', COLORS['accent_green']),
        }
        
        emoji, text, color = status_map.get(market_state, ('⚪', 'Unknown', COLORS['text_muted']))
        
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
            'color': COLORS['text_muted'],
            'state': 'UNKNOWN'
        }


@st.cache_data(ttl=60)
def get_major_indices() -> List[Dict]:
    """Get quotes for major market indices."""
    indices = [
        ('SPY', 'S&P 500'),
        ('QQQ', 'NASDAQ'),
        ('IWM', 'Russell'),
        ('^VIX', 'VIX')
    ]
    
    results = []
    for symbol, name in indices:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='2d')
            
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                
                results.append({
                    'symbol': symbol.replace('^', ''),
                    'name': name,
                    'price': current,
                    'change': change
                })
        except Exception:
            pass
    
    return results


def render_status_bar(alert_count: int = 0, account_value: float = None):
    """
    Render the top status bar using Streamlit columns for better compatibility.
    Includes live clock with US Eastern and Malaysia time zones.
    
    Parameters
    ----------
    alert_count : int
        Number of active alerts
    account_value : float, optional
        Account balance to display
    """
    import streamlit.components.v1 as components
    
    market = get_market_status()
    indices = get_major_indices()
    
    # Use Streamlit columns for layout
    cols = st.columns([2, 1, 1, 1, 1, 2, 1])
    
    # Market status
    with cols[0]:
        st.markdown(f"""
        <div style="
            background: {market['color']}20;
            border: 1px solid {market['color']}60;
            border-radius: 4px;
            padding: 4px 10px;
            display: inline-block;
        ">
            <span style="color: {market['color']}; font-size: 12px; font-weight: 600;">
                {market['emoji']} {market['text']}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Index quotes
    for i, idx in enumerate(indices[:4]):  # Show up to 4 indices
        if i + 1 < len(cols) - 2:
            with cols[i + 1]:
                change_color = COLORS['accent_green'] if idx['change'] >= 0 else COLORS['accent_red']
                st.markdown(f"""
                <div style="text-align: center;">
                    <span style="color: {COLORS['text_muted']}; font-size: 11px;">{idx['symbol']}</span>
                    <span style="color: {change_color}; font-size: 12px; font-weight: 600; margin-left: 4px;">
                        {idx['change']:+.2f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
    
    # Live Clock with Dual Time Zones (using components.html for JavaScript)
    with cols[-2]:
        clock_html = f"""
        <div id="live-clock" style="
            text-align: center; 
            color: {COLORS['text_primary']}; 
            font-size: 11px; 
            font-family: 'Roboto Mono', monospace;
            white-space: nowrap;
            background: transparent;
        ">
            Loading...
        </div>
        <script>
            function updateClock() {{
                const now = new Date();
                
                // US Eastern Time
                const etOptions = {{ 
                    timeZone: 'America/New_York', 
                    hour: '2-digit', 
                    minute: '2-digit', 
                    second: '2-digit',
                    hour12: true 
                }};
                const etTime = now.toLocaleTimeString('en-US', etOptions);
                
                // Malaysia Time
                const mytOptions = {{ 
                    timeZone: 'Asia/Kuala_Lumpur', 
                    hour: '2-digit', 
                    minute: '2-digit', 
                    second: '2-digit',
                    hour12: true 
                }};
                const mytTime = now.toLocaleTimeString('en-US', mytOptions);
                
                // Update the clock element
                const clockEl = document.getElementById('live-clock');
                if (clockEl) {{
                    clockEl.innerHTML = '<span style="margin-right: 8px;">🇺🇸 ' + etTime + ' ET</span><span>🇲🇾 ' + mytTime + ' MYT</span>';
                }}
            }}
            
            // Update immediately and then every second
            updateClock();
            setInterval(updateClock, 1000);
        </script>
        """
        components.html(clock_html, height=25)
    
    # Alerts
    with cols[-1]:
        alert_bg = 'rgba(255,82,82,0.15)' if alert_count > 0 else 'transparent'
        badge = f'<span style="background: #FF5252; color: white; border-radius: 50%; padding: 2px 6px; font-size: 10px; margin-left: 4px;">{alert_count}</span>' if alert_count > 0 else ''
        st.markdown(f"""
        <div style="text-align: center; background: {alert_bg}; border-radius: 4px; padding: 2px;">
            🔔{badge}
        </div>
        """, unsafe_allow_html=True)
    
    # Divider line
    st.markdown(f"""
    <hr style="border: none; border-top: 1px solid {COLORS['border']}; margin: 8px 0 16px 0;">
    """, unsafe_allow_html=True)

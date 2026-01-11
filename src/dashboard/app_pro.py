"""
Stock Trading Dashboard - Professional Edition
Enhanced dashboard with 5-tab structure and institutional-grade UI
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.data_loader import download_stock_data
from src.features.technical_indicators import add_all_technical_indicators
from src.dashboard.theme import COLORS, get_custom_css, get_fonts_css
from src.dashboard.charts import create_candlestick_chart, add_moving_averages

# Phase 1 Components
from src.dashboard.components.status_bar import render_status_bar
from src.dashboard.components.watchlist import (
    render_watchlist_panel,
    render_mini_watchlist,
    init_watchlist_state
)
from src.dashboard.components.alerts import (
    init_alerts_state,
    render_alerts_sidebar,
    render_alerts_panel,
    get_active_alerts
)
from src.dashboard.components.drawing_tools import (
    init_drawings_state,
    apply_drawings_to_chart,
    render_drawing_toolbar,
    render_drawings_list
)
from src.dashboard.components.keyboard_shortcuts import (
    inject_keyboard_shortcuts,
    render_shortcuts_hint
)

# Phase 2 Components
from src.dashboard.components.advanced_indicators import (
    init_indicators_state,
    create_chart_with_indicators,
    render_indicator_toolbar
)
from src.dashboard.components.timeframe import (
    init_timeframe_state,
    get_selected_timeframe,
    set_timeframe,
    get_data_for_timeframe,
    render_timeframe_switcher,
    TIMEFRAMES
)
from src.dashboard.components.quick_search import (
    init_search_state,
    render_search_modal,
    render_search_bar
)
from src.dashboard.components.portfolio_analytics import (
    render_portfolio_analytics
)

# Phase 3 Components
from src.dashboard.components.workspace_manager import (
    init_workspace_state as init_ws_state,
    render_workspace_manager,
    save_workspace
)
from src.dashboard.components.export_manager import (
    render_export_panel,
    render_quick_export
)
from src.dashboard.components.responsive import (
    inject_responsive_css
)# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="TradeView Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme CSS
st.markdown(get_fonts_css(), unsafe_allow_html=True)
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Enhanced tab styling
st.markdown(f"""
<style>
/* Enhanced Professional Tab Styling */
.stTabs [data-baseweb="tab-list"] {{
    background: linear-gradient(180deg, {COLORS['bg_secondary']} 0%, {COLORS['bg_primary']} 100%);
    border-radius: 8px 8px 0 0;
    padding: 6px 6px 0 6px;
    gap: 2px;
    border-bottom: 2px solid {COLORS['border']};
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {COLORS['text_muted']};
    border-radius: 6px 6px 0 0;
    padding: 10px 18px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid transparent;
    border-bottom: none;
    transition: all 0.15s ease;
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

/* Compact metric cards */
[data-testid="stMetric"] {{
    background: {COLORS['bg_secondary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 12px;
}}

[data-testid="stMetric"] label {{
    font-size: 11px !important;
}}

[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    font-size: 20px !important;
}}

/* Sidebar enhancements */
section[data-testid="stSidebar"] {{
    background: {COLORS['bg_sidebar']};
    border-right: 1px solid {COLORS['border']};
}}

section[data-testid="stSidebar"] .stButton button {{
    font-size: 12px;
    padding: 6px 12px;
}}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# INITIALIZE STATE
# =============================================================================

# Phase 1 state
init_watchlist_state()
init_alerts_state()
init_drawings_state()

# Phase 2 state
init_indicators_state()
init_timeframe_state()
init_search_state()

# Phase 3 state
init_ws_state()

if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'SPY'

# Inject keyboard shortcuts (Ctrl+K, number keys, etc.)
inject_keyboard_shortcuts(st.session_state.get('watchlist_symbols', []))

# Inject responsive CSS for mobile/tablet
inject_responsive_css()


# =============================================================================
# TOP STATUS BAR
# =============================================================================

render_status_bar(
    alert_count=len(get_active_alerts()),
    account_value=None  # Can be connected to portfolio
)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    # Logo/Title
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 12px 0 16px 0;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 16px;
    ">
        <h1 style="
            margin: 0;
            color: {COLORS['accent_orange']};
            font-size: 20px;
            font-weight: 700;
        ">📈 TradeView Pro</h1>
        <p style="
            margin: 4px 0 0 0;
            color: {COLORS['text_muted']};
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        ">Professional Trading</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Symbol Selection (Quick)
    st.markdown(f"""
    <div style="color: {COLORS['text_secondary']}; font-size: 12px; margin-bottom: 4px;">
        Active Symbol
    </div>
    """, unsafe_allow_html=True)
    
    symbol = st.text_input(
        "Symbol",
        value=st.session_state.selected_symbol,
        key="symbol_input",
        label_visibility="collapsed",
        placeholder="Enter symbol..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        load_btn = st.button("📊 Load", use_container_width=True, type="primary")
    with col2:
        if st.button("➕ Watch", use_container_width=True):
            from src.dashboard.components.watchlist import add_to_watchlist
            add_to_watchlist(symbol)
            st.rerun()
    
    st.markdown(f"<hr style='border-color: {COLORS['border']}; margin: 16px 0;'>", unsafe_allow_html=True)
    
    # Watchlist
    def on_symbol_select(sym):
        st.session_state.selected_symbol = sym
        st.session_state.ticker = sym
        st.rerun()
    
    render_watchlist_panel(on_symbol_select=on_symbol_select)
    
    st.markdown(f"<hr style='border-color: {COLORS['border']}; margin: 16px 0;'>", unsafe_allow_html=True)
    
    # Alerts section
    render_alerts_sidebar()
    
    st.markdown(f"<hr style='border-color: {COLORS['border']}; margin: 16px 0;'>", unsafe_allow_html=True)
    
    # Workspaces section
    render_workspace_manager()
    
    # Quick Settings
    with st.expander("⚙️ Chart Settings", expanded=False):
        timeframe = st.selectbox(
            "Timeframe",
            options=["1d", "1wk", "1mo"],
            format_func=lambda x: {"1d": "Daily", "1wk": "Weekly", "1mo": "Monthly"}[x],
            key="timeframe"
        )
        
        show_volume = st.checkbox("Show Volume", value=True, key="show_volume")
        show_ma = st.checkbox("Moving Averages", value=True, key="show_ma")
        
        if show_ma:
            ma_periods = st.multiselect(
                "MA Periods",
                options=[10, 20, 50, 100, 200],
                default=[20, 50],
                key="ma_periods"
            )
        else:
            ma_periods = []
        
        chart_height = st.slider("Chart Height", 400, 800, 550, 50, key="chart_height")
    
    # Date Range (collapsed by default)
    with st.expander("📅 Date Range", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                value=datetime.now() - timedelta(days=365),
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End",
                value=datetime.now(),
                key="end_date"
            )


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=300)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load and process stock data"""
    df = download_stock_data(ticker, start, end)
    if not df.empty:
        df = add_all_technical_indicators(df)
    return df


# Load data on button click or from session
if load_btn or 'data' not in st.session_state:
    with st.spinner(f"Loading {symbol} data..."):
        df = load_data(symbol, str(start_date), str(end_date))
        if not df.empty:
            st.session_state['data'] = df
            st.session_state['data_with_indicators'] = df
            st.session_state['ticker'] = symbol
            st.session_state['selected_symbol'] = symbol


# =============================================================================
# MAIN CONTENT - 5 TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market",
    "📈 Analysis",
    "🔮 Signals",
    "💼 Portfolio",
    "⚙️ Settings"
])


# =============================================================================
# TAB 1: MARKET OVERVIEW
# =============================================================================

with tab1:
    st.markdown(f"""
    <h2 style="color: {COLORS['text_primary']}; margin-bottom: 16px; font-size: 20px;">
        Market Overview
    </h2>
    """, unsafe_allow_html=True)
    
    # Import and render heatmap
    try:
        from src.dashboard.heatmap import render_heatmap_page
        render_heatmap_page()
    except Exception as e:
        st.error(f"Failed to load market heatmap: {e}")
        st.info("The Market Overview tab displays the S&P 500 market heatmap with sector performance.")


# =============================================================================
# TAB 2: TECHNICAL ANALYSIS
# =============================================================================

with tab2:
    if 'data' in st.session_state:
        df = st.session_state['data']
        ticker = st.session_state.get('ticker', symbol)
        
        # Header with current price
        current_price = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        change_color = COLORS['accent_green'] if price_change >= 0 else COLORS['accent_red']
        
        st.markdown(f"""
        <div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 20px;">
            <h2 style="color: {COLORS['text_primary']}; margin: 0; font-size: 24px;">{ticker}</h2>
            <span style="color: {COLORS['text_primary']}; font-size: 28px; font-weight: 700; font-family: 'Roboto Mono', monospace;">
                ${current_price:.2f}
            </span>
            <span style="color: {change_color}; font-size: 18px; font-weight: 600;">
                {price_change:+.2f} ({price_change_pct:+.2f}%)
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("High", f"${float(df['High'].iloc[-1]):.2f}")
        with col2:
            st.metric("Low", f"${float(df['Low'].iloc[-1]):.2f}")
        with col3:
            high_52w = float(df['High'].max())
            st.metric("52W High", f"${high_52w:.2f}")
        with col4:
            low_52w = float(df['Low'].min())
            st.metric("52W Low", f"${low_52w:.2f}")
        with col5:
            vol = int(df['Volume'].iloc[-1])
            vol_str = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.0f}K"
            st.metric("Volume", vol_str)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Timeframe Switcher
        render_timeframe_switcher()
        
        # Drawing Tools & Indicators Toolbar Row
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("✏️ Drawing Tools", expanded=False):
                render_drawing_toolbar(
                    symbol=ticker,
                    current_price=current_price,
                    high=float(df['High'].max()),
                    low=float(df['Low'].min())
                )
                render_drawings_list(ticker)
        
        with col2:
            render_indicator_toolbar()
        
        # Main Chart with Advanced Indicators
        fig = create_chart_with_indicators(df, title="", height=chart_height)
        
        # Apply saved drawings to chart
        fig = apply_drawings_to_chart(fig, ticker, df.index)
        
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
        })
        
        # Technical Indicators Panel
        st.markdown(f"""
        <h3 style="color: {COLORS['text_secondary']}; margin: 20px 0 10px 0; font-size: 16px;">
            Technical Indicators
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sma_20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else None
            if sma_20 and not pd.isna(sma_20):
                trend = "↑ Above" if current_price > sma_20 else "↓ Below"
                st.metric("SMA 20", f"${sma_20:.2f}", trend)
            else:
                st.metric("SMA 20", "N/A")
        
        with col2:
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else None
            if rsi and not pd.isna(rsi):
                rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                st.metric("RSI", f"{rsi:.1f}", rsi_status)
            else:
                st.metric("RSI", "N/A")
        
        with col3:
            macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else None
            macd_signal = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns else None
            if macd and not pd.isna(macd):
                macd_trend = "Bullish" if macd > (macd_signal or 0) else "Bearish"
                st.metric("MACD", f"{macd:.3f}", macd_trend)
            else:
                st.metric("MACD", "N/A")
        
        with col4:
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else None
            if atr and not pd.isna(atr):
                st.metric("ATR", f"${atr:.2f}")
            else:
                st.metric("ATR", "N/A")
    
    else:
        st.info("👈 Enter a symbol in the sidebar and click 'Load' to begin analysis")


# =============================================================================
# TAB 3: TRADING SIGNALS
# =============================================================================

with tab3:
    if 'data' in st.session_state:
        st.markdown(f"""
        <h2 style="color: {COLORS['text_primary']}; margin-bottom: 16px; font-size: 20px;">
            🔮 AI Trading Signals
        </h2>
        """, unsafe_allow_html=True)
        
        try:
            # Store data for signals page
            st.session_state['stock_data'] = st.session_state['data']
            from src.dashboard.signal_display import render_signals_page
            render_signals_page()
        except Exception as e:
            st.error(f"Failed to load signals: {e}")
            
            # Fallback basic signals
            df = st.session_state['data']
            
            st.markdown(f"""
            <div style="
                background: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 20px;
            ">
                <h4 style="color: {COLORS['accent_orange']}; margin: 0 0 12px 0;">
                    📊 Technical Signal Summary
                </h4>
            """, unsafe_allow_html=True)
            
            # Simple signal based on indicators
            signals = []
            
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                if not pd.isna(rsi):
                    if rsi < 30:
                        signals.append(('RSI', 'BUY', 'Oversold'))
                    elif rsi > 70:
                        signals.append(('RSI', 'SELL', 'Overbought'))
                    else:
                        signals.append(('RSI', 'HOLD', 'Neutral'))
            
            if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                macd = df['MACD'].iloc[-1]
                macd_sig = df['MACD_Signal'].iloc[-1]
                if not pd.isna(macd) and not pd.isna(macd_sig):
                    if macd > macd_sig:
                        signals.append(('MACD', 'BUY', 'Bullish crossover'))
                    else:
                        signals.append(('MACD', 'SELL', 'Bearish crossover'))
            
            for indicator, signal, reason in signals:
                signal_color = COLORS['accent_green'] if signal == 'BUY' else COLORS['accent_red'] if signal == 'SELL' else COLORS['accent_orange']
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid {COLORS['border']};">
                    <span style="color: {COLORS['text_secondary']};">{indicator}</span>
                    <span style="color: {signal_color}; font-weight: 600;">{signal} - {reason}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.info("👈 Load stock data to see trading signals")


# =============================================================================
# TAB 4: PORTFOLIO
# =============================================================================

with tab4:
    if 'data' in st.session_state:
        try:
            df = st.session_state['data']
            ticker = st.session_state.get('ticker', 'Stock')
            
            # Use enhanced portfolio analytics
            render_portfolio_analytics(df, ticker)
            
        except Exception as e:
            st.error(f"Failed to load portfolio analytics: {e}")
    
    else:
        st.info("👈 Load stock data to see portfolio analytics")


# =============================================================================
# TAB 5: SETTINGS
# =============================================================================

with tab5:
    st.markdown(f"""
    <h2 style="color: {COLORS['text_primary']}; margin-bottom: 16px; font-size: 20px;">
        ⚙️ Settings
    </h2>
    """, unsafe_allow_html=True)
    
    # General Settings
    with st.expander("🎨 Display Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox(
                "Theme",
                options=["Dark (Professional)", "Light (Coming Soon)"],
                index=0,
                disabled=True,
                help="Light theme coming in future update"
            )
        
        with col2:
            st.selectbox(
                "Default Timeframe",
                options=["Daily", "Weekly", "Monthly"],
                index=0
            )
    
    with st.expander("📊 Chart Preferences", expanded=False):
        st.checkbox("Auto-load last viewed symbol", value=True)
        st.checkbox("Show gridlines", value=True)
        st.checkbox("Enable crosshair", value=True)
        st.multiselect(
            "Default Indicators",
            options=["SMA 20", "SMA 50", "SMA 200", "EMA 12", "RSI", "MACD", "Bollinger Bands"],
            default=["SMA 20", "SMA 50"]
        )
    
    with st.expander("🔔 Alerts (Coming Soon)", expanded=False):
        st.info("Price alerts and notification settings will be available in a future update.")
        st.checkbox("Enable price alerts", value=False, disabled=True)
        st.checkbox("Enable signal alerts", value=False, disabled=True)
        st.text_input("Email for notifications", disabled=True, placeholder="your@email.com")
    
    with st.expander("🔧 Advanced Settings", expanded=False):
        st.number_input("Data cache duration (seconds)", value=300, min_value=60, max_value=3600)
        st.checkbox("Show debug information", value=False)
        st.selectbox(
            "Data Provider",
            options=["Yahoo Finance (Free)", "Alpha Vantage (API Key Required)"],
            index=0
        )
    
    # Export & Reporting Section (Phase 3)
    st.markdown(f"""
    <h3 style="color: {COLORS['text_secondary']}; margin: 20px 0 12px 0; font-size: 16px;">
        📥 Export & Reports
    </h3>
    """, unsafe_allow_html=True)
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        ticker = st.session_state.get('ticker', 'Stock')
        
        render_export_panel(df, ticker)
    else:
        st.info("👈 Load stock data to enable export options")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # About
    st.markdown(f"""
    <div style="
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 16px;
    ">
        <h4 style="color: {COLORS['text_primary']}; margin: 0 0 8px 0;">📈 TradeView Pro</h4>
        <p style="color: {COLORS['text_muted']}; margin: 0; font-size: 13px;">
            Version 2.0 | Professional Trading Dashboard
        </p>
        <p style="color: {COLORS['text_muted']}; margin: 8px 0 0 0; font-size: 12px;">
            Data provided by Yahoo Finance. Prices may be delayed.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(f"""
<div style="
    margin-top: 40px;
    padding: 16px 0;
    border-top: 1px solid {COLORS['border']};
    text-align: center;
">
    <p style="color: {COLORS['text_muted']}; font-size: 11px; margin: 0;">
        📈 TradeView Pro v2.0 | © 2024 | Data: Yahoo Finance
    </p>
</div>
""", unsafe_allow_html=True)

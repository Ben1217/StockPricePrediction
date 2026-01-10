"""
Stock Prediction Dashboard - Version 2.0
Redesigned with dark theme and professional UI
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


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Stock Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme CSS
st.markdown(get_fonts_css(), unsafe_allow_html=True)
st.markdown(get_custom_css(), unsafe_allow_html=True)


# =============================================================================
# SIDEBAR - SETTINGS
# =============================================================================

with st.sidebar:
    # Logo/Title
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid {COLORS['border']}; margin-bottom: 20px;">
        <h1 style="margin: 0; color: {COLORS['accent_orange']}; font-size: 24px;">📈 TradingView</h1>
        <p style="margin: 5px 0 0 0; color: {COLORS['text_muted']}; font-size: 12px;">Stock Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Settings Section
    st.markdown(f"""
    <div style="padding: 10px 0;">
        <h3 style="color: {COLORS['text_primary']}; margin-bottom: 15px;">
            ⚙️ Settings
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Symbol Selection
    symbol = st.selectbox(
        "Select Symbol",
        options=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
        index=0
    )
    
    # Timeframe Selection
    timeframe = st.selectbox(
        "Timeframe",
        options=["1d", "1wk", "1mo"],
        format_func=lambda x: {"1d": "Daily", "1wk": "Weekly", "1mo": "Monthly"}[x],
        index=0
    )
    
    # Date Range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load Data Button
    load_button = st.button("📊 Load Data", use_container_width=True, type="primary")
    
    # Divider
    st.markdown(f"""
    <hr style="border-color: {COLORS['border']}; margin: 20px 0;">
    """, unsafe_allow_html=True)
    
    # Additional Settings (collapsed)
    with st.expander("📈 Chart Settings"):
        show_volume = st.checkbox("Show Volume", value=True)
        show_ma = st.checkbox("Show Moving Averages", value=True)
        ma_periods = st.multiselect(
            "MA Periods",
            options=[10, 20, 50, 100, 200],
            default=[20, 50]
        )
    
    with st.expander("🎨 Display Options"):
        chart_height = st.slider("Chart Height", 400, 800, 600, 50)


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


# Load data on button click or first load
if load_button or 'data' not in st.session_state:
    with st.spinner(f"Loading {symbol} data..."):
        df = load_data(symbol, str(start_date), str(end_date))
        if not df.empty:
            st.session_state['data'] = df
            st.session_state['symbol'] = symbol


# =============================================================================
# MAIN CONTENT - TABS
# =============================================================================

# Header
st.markdown(f"""
<div style="padding: 10px 0 20px 0;">
    <h1 style="margin: 0; color: {COLORS['text_primary']};">
        {st.session_state.get('symbol', symbol)} Analysis
    </h1>
</div>
""", unsafe_allow_html=True)

# Tabs - Consolidated to 4
tab1, tab2, tab3 = st.tabs([
    "📊 Market Overview",
    "📈 Analysis & Signals",
    "💼 Portfolio"
])


# =============================================================================
# TAB 1: MARKET OVERVIEW
# =============================================================================

with tab1:
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Top Metrics Row
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <h3 style="color: {COLORS['text_secondary']}; font-weight: 500;">Key Metrics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate metrics
        current_price = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        high_52w = float(df['High'].max())
        low_52w = float(df['Low'].min())
        volume = int(df['Volume'].iloc[-1])
        
        # Metrics Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Close Price",
                value=f"${current_price:.2f}"
            )
        
        with col2:
            delta_color = "normal" if price_change >= 0 else "inverse"
            st.metric(
                label="Change",
                value=f"${abs(price_change):.2f}",
                delta=f"{price_change_pct:+.2f}%"
            )
        
        with col3:
            st.metric(
                label="52W High",
                value=f"${high_52w:.2f}"
            )
        
        with col4:
            st.metric(
                label="52W Low",
                value=f"${low_52w:.2f}"
            )
        
        with col5:
            volume_str = f"{volume/1e6:.1f}M" if volume >= 1e6 else f"{volume/1e3:.0f}K"
            st.metric(
                label="Volume",
                value=volume_str
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Candlestick Chart
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <h3 style="color: {COLORS['text_secondary']}; font-weight: 500;">Candlestick Chart</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create chart
        fig = create_candlestick_chart(
            df,
            title="",
            show_volume=show_volume,
            height=chart_height
        )
        
        # Add moving averages if enabled
        if show_ma and ma_periods:
            fig = add_moving_averages(fig, df, periods=ma_periods)
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
        })
        
        # Results Panel (placeholder for Phase 2)
        st.markdown(f"""
        <div style="
            background-color: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        ">
            <h4 style="color: {COLORS['accent_orange']}; margin: 0 0 10px 0;">
                📊 Quick Stats
            </h4>
            <p style="color: {COLORS['text_secondary']}; margin: 0;">
                <span style="color: {COLORS['text_primary']};">Daily Range:</span> 
                ${float(df['Low'].iloc[-1]):.2f} - ${float(df['High'].iloc[-1]):.2f}
                &nbsp;|&nbsp;
                <span style="color: {COLORS['text_primary']};">Avg Volume:</span> 
                {df['Volume'].mean()/1e6:.1f}M
                &nbsp;|&nbsp;
                <span style="color: {COLORS['text_primary']};">Data Points:</span> 
                {len(df)}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("👈 Select a symbol and click 'Load Data' to get started")


# =============================================================================
# TAB 2: ANALYSIS & SIGNALS
# =============================================================================

with tab2:
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        st.markdown(f"""
        <div style="padding: 20px; background-color: {COLORS['bg_secondary']}; border-radius: 8px;">
            <h3 style="color: {COLORS['accent_orange']};">📈 Technical Analysis & ML Signals</h3>
            <p style="color: {COLORS['text_secondary']};">
                Full analysis features coming in Phase 2.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical Indicators Preview
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Get indicator values safely
        sma_20_val = f"${df['SMA_20'].iloc[-1]:.2f}" if 'SMA_20' in df.columns and not pd.isna(df['SMA_20'].iloc[-1]) else "N/A"
        rsi_val = f"{df['RSI'].iloc[-1]:.1f}" if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else "N/A"
        
        with col1:
            st.markdown(f"""
            <div style="background-color: {COLORS['bg_secondary']}; padding: 20px; border-radius: 8px; border: 1px solid {COLORS['border']};">
                <h4 style="color: {COLORS['text_primary']};">Trend Indicators</h4>
                <p style="color: {COLORS['text_muted']};">SMA 20: {sma_20_val}</p>
                <p style="color: {COLORS['text_muted']};">RSI: {rsi_val}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: {COLORS['bg_secondary']}; padding: 20px; border-radius: 8px; border: 1px solid {COLORS['border']};">
                <h4 style="color: {COLORS['text_primary']};">Signal Status</h4>
                <p style="color: {COLORS['accent_orange']};">⏳ Pending ML integration</p>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.info("👈 Load data first to see analysis")


# =============================================================================
# TAB 3: PORTFOLIO
# =============================================================================

with tab3:
    st.markdown(f"""
    <div style="padding: 20px; background-color: {COLORS['bg_secondary']}; border-radius: 8px;">
        <h3 style="color: {COLORS['accent_orange']};">💼 Portfolio Management</h3>
        <p style="color: {COLORS['text_secondary']};">
            Portfolio tracking and performance features coming in Phase 3.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(f"""
<div style="
    margin-top: 50px;
    padding: 20px 0;
    border-top: 1px solid {COLORS['border']};
    text-align: center;
">
    <p style="color: {COLORS['text_muted']}; font-size: 12px; margin: 0;">
        📊 Stock Trading Dashboard v2.0 | Data provided by Yahoo Finance
    </p>
</div>
""", unsafe_allow_html=True)

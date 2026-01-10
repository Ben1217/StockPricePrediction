"""
Signal Display Module for Dashboard

Provides Streamlit UI components for displaying trading signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.signals.signal_generator import TradingSignalGenerator, check_uptrend, check_downtrend
from src.signals.position_sizing import PositionSizeCalculator, RiskProfile, get_default_risk_profile


# Signal color mapping
SIGNAL_COLORS = {
    'STRONG_BUY': '#00C851',   # Green
    'BUY': '#7CB342',          # Light green
    'HOLD': '#FFB300',         # Amber
    'SELL': '#FF8800',         # Orange
    'STRONG_SELL': '#FF4444',  # Red
    'NEUTRAL': '#9E9E9E'       # Gray
}


def get_signal_color(action: str) -> str:
    """Get color for signal action."""
    return SIGNAL_COLORS.get(action, '#9E9E9E')


def display_mode_banner(mode: str, ml_available: Dict[str, bool] = None):
    """
    Display banner indicating which signal mode is active.
    
    Parameters
    ----------
    mode : str
        'FULL' or 'TECHNICAL_ONLY'
    ml_available : dict, optional
        Dict of model name -> bool availability
    """
    if mode == 'FULL':
        st.success("🤖 **FULL MODE**: Technical Analysis + ML Predictions Active")
    else:
        st.warning("""
        ⚠️ **TECHNICAL-ONLY MODE**: ML models not trained yet
        
        Showing signals based on technical analysis only. Confidence scores are capped at 80%.
        To enable ML predictions, go to the "Predictions" tab and train models.
        """)
        
        if ml_available:
            with st.expander("📊 Model Status"):
                for model, available in ml_available.items():
                    status = "✅ Ready" if available else "❌ Not Trained"
                    st.write(f"**{model.title()}**: {status}")


def display_trading_signal(signal: Dict, calculator: PositionSizeCalculator = None):
    """
    Display trading signal card with entry/stop/target.
    
    Parameters
    ----------
    signal : dict
        Signal from TradingSignalGenerator
    calculator : PositionSizeCalculator, optional
        For position sizing display
    """
    action = signal.get('action', 'HOLD')
    confidence = signal.get('confidence', 0)
    mode = signal.get('mode', 'TECHNICAL_ONLY')
    
    color = get_signal_color(action)
    
    # Main signal header
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}44);
        border-left: 5px solid {color};
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    ">
        <h2 style="margin: 0; color: #333;">
            {signal.get('symbol', 'N/A')} - {action}
        </h2>
        <p style="margin: 5px 0; font-size: 18px; color: #555;">
            Confidence: <strong>{confidence:.0f}%</strong>
            <span style="font-size: 14px; opacity: 0.7;"> ({mode})</span>
        </p>
        <p style="margin: 0; font-size: 14px; color: #666;">
            Current Price: ${signal.get('current_price', 0):.2f}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Warning if present
    if signal.get('warning'):
        st.info(f"ℹ️ {signal['warning']}")
    
    # Show reason for HOLD
    if action == 'HOLD':
        reason = signal.get('reason', 'No clear trading setup detected')
        st.info(f"💡 **Reason**: {reason}")
        return
    
    # Entry details for actionable signals
    if action in ['STRONG_BUY', 'BUY', 'SELL', 'STRONG_SELL']:
        display_entry_details(signal, calculator)
        display_trend_indicators(signal)


def display_entry_details(signal: Dict, calculator: PositionSizeCalculator = None):
    """Display entry, stop loss, and target details."""
    
    entry = signal.get('entry_price')
    stop = signal.get('stop_loss')
    target = signal.get('target')
    
    if entry is None or stop is None:
        st.warning("Entry/stop details not available")
        return
    
    # Calculate risk/reward
    risk = abs(entry - stop)
    reward = abs(target - entry) if target else risk * 2.5
    rr_ratio = reward / risk if risk > 0 else 0
    
    # Entry details columns
    st.subheader("📋 Trade Setup")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Entry Price", f"${entry:.2f}")
    
    with col2:
        st.metric("Stop Loss", f"${stop:.2f}", 
                  delta=f"-${risk:.2f}", delta_color="inverse")
    
    with col3:
        st.metric("Target", f"${target:.2f}" if target else "N/A",
                  delta=f"+${reward:.2f}" if target else None)
    
    with col4:
        st.metric("Risk/Reward", f"{rr_ratio:.1f}:1")
    
    # Position sizing
    if calculator:
        st.subheader("📊 Position Sizing")
        
        position = calculator.calculate_shares(
            entry_price=entry,
            stop_loss=stop,
            signal_confidence=signal.get('confidence', 80)
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Shares", f"{position['shares']:,}")
        
        with col2:
            st.metric("Position Value", f"${position['position_value']:,.0f}")
        
        with col3:
            st.metric("Dollar Risk", f"${position['dollar_risk']:.2f}")
        
        with col4:
            st.metric("Profit Target", f"${position['potential_profit']:.2f}")
        
        # Warnings
        for warning in position.get('warnings', []):
            st.warning(warning)
    
    # Action instructions
    st.subheader("✅ Next Actions")
    
    action = signal.get('action', 'HOLD')
    action_type = 'BUY' if 'BUY' in action else 'SELL/SHORT'
    
    st.markdown(f"""
    1. Place **{action_type} LIMIT** order at **${entry:.2f}**
    2. Set **STOP LOSS** at **${stop:.2f}**
    3. Set **TARGET** at **${target:.2f}** (optional)
    4. Monitor and trail stop as price moves in your favor
    """)


def display_trend_indicators(signal: Dict):
    """Display trend strength indicators."""
    
    tech_details = signal.get('technical_details', {})
    trend_score = tech_details.get('trend_score', {})
    
    uptrend = trend_score.get('uptrend', 0)
    downtrend = trend_score.get('downtrend', 0)
    patterns = tech_details.get('patterns', [])
    
    with st.expander("📈 Technical Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Uptrend Strength", f"{uptrend:.0f}/100")
            st.progress(int(uptrend))
        
        with col2:
            st.metric("Downtrend Strength", f"{downtrend:.0f}/100")
            st.progress(int(downtrend))
        
        if patterns:
            st.markdown("**Detected Patterns:**")
            for pattern in patterns:
                st.write(f"✅ {pattern.replace('_', ' ').title()}")
    
    # ML details if available
    ml_details = signal.get('ml_details')
    if ml_details:
        with st.expander("🤖 ML Prediction Details", expanded=False):
            st.write(f"**Predicted Price**: ${ml_details.predicted_price:.2f}")
            st.write(f"**Expected Return**: {ml_details.predicted_return*100:+.2f}%")
            st.write(f"**ML Confidence**: {ml_details.confidence_score*100:.0f}%")
            st.write(f"**Direction**: {ml_details.directional_signal}")


def create_entry_chart(df: pd.DataFrame, signal: Dict) -> go.Figure:
    """
    Create chart showing entry point and levels.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    signal : dict
        Trading signal with entry/stop/target
    
    Returns
    -------
    go.Figure
        Plotly figure with candlesticks and levels
    """
    # Use last 60 days of data
    df_plot = df.tail(60).copy()
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        name='Price',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # 20 MA
    if 'SMA_20' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot['SMA_20'],
            name='20 MA',
            line=dict(color='#2196F3', width=2)
        ))
    
    # 200 MA (if available)
    if 'SMA_200' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot['SMA_200'],
            name='200 MA',
            line=dict(color='#9C27B0', width=1.5, dash='dot')
        ))
    
    # Entry/Stop/Target lines for actionable signals
    action = signal.get('action', 'HOLD')
    
    if action in ['STRONG_BUY', 'BUY', 'SELL', 'STRONG_SELL']:
        entry = signal.get('entry_price')
        stop = signal.get('stop_loss')
        target = signal.get('target')
        
        if entry:
            fig.add_hline(
                y=entry,
                line_dash="dash",
                line_color="#4CAF50",
                annotation_text="ENTRY",
                annotation_position="right"
            )
        
        if stop:
            fig.add_hline(
                y=stop,
                line_dash="dash",
                line_color="#F44336",
                annotation_text="STOP LOSS",
                annotation_position="right"
            )
        
        if target:
            fig.add_hline(
                y=target,
                line_dash="dash",
                line_color="#FFD700",
                annotation_text="TARGET",
                annotation_position="right"
            )
    
    # Layout
    fig.update_layout(
        title=f"{signal.get('symbol', 'Stock')} - Trading Setup",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def render_risk_settings_sidebar():
    """Render risk management settings in sidebar."""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Risk Settings")
    
    with st.sidebar.expander("Position Sizing", expanded=False):
        account_balance = st.number_input(
            "Account Balance ($)",
            min_value=1000,
            max_value=10000000,
            value=30000,
            step=1000,
            key="risk_account"
        )
        
        risk_method = st.radio(
            "Risk Method",
            options=['Fixed Dollar', 'Percentage'],
            key="risk_method"
        )
        
        if risk_method == 'Fixed Dollar':
            risk_per_trade = st.number_input(
                "Risk Per Trade ($)",
                min_value=10,
                max_value=10000,
                value=100,
                step=10,
                key="risk_amount"
            )
            method = 'fixed_dollar'
        else:
            risk_pct = st.slider(
                "Risk Per Trade (%)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="risk_pct"
            )
            risk_per_trade = account_balance * (risk_pct / 100)
            method = 'percentage'
        
        max_position = st.number_input(
            "Max Position ($)",
            min_value=500,
            max_value=int(account_balance),
            value=min(5000, int(account_balance * 0.2)),
            step=500,
            key="max_position"
        )
        
        confidence_scaling = st.checkbox(
            "Scale by Confidence",
            value=True,
            key="conf_scaling",
            help="Reduce position size for lower confidence signals"
        )
    
    # Create and return risk profile
    return RiskProfile(
        account_balance=account_balance,
        risk_per_trade=risk_per_trade,
        risk_method=method,
        max_position_size=max_position,
        confidence_scaling=confidence_scaling
    )


def render_signals_page():
    """
    Render the complete Trading Signals page.
    """
    st.subheader("🎯 Trading Signals")
    
    # Initialize generator
    generator = TradingSignalGenerator()
    
    # Display mode banner
    ml_available = generator.check_ml_models_available()
    display_mode_banner(generator.mode, ml_available)
    
    # Get risk settings from sidebar
    risk_profile = render_risk_settings_sidebar()
    calculator = PositionSizeCalculator(risk_profile)
    
    # Stock selector
    if 'stock_data' not in st.session_state or st.session_state.stock_data is None:
        st.info("📊 Please load stock data first using the sidebar controls.")
        return
    
    df = st.session_state.stock_data
    symbol = st.session_state.get('selected_symbol', 'SPY')
    
    # Add required indicators if missing
    if 'SMA_20' not in df.columns:
        df['SMA_20'] = df['Close'].rolling(20).mean()
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = df['Close'].rolling(200).mean()
    
    st.markdown("---")
    
    # Generate signal
    with st.spinner("Analyzing patterns..."):
        signal = generator.analyze_stock(symbol, df)
    
    # Display signal
    display_trading_signal(signal, calculator)
    
    st.markdown("---")
    
    # Display chart
    st.subheader("📈 Chart with Entry Levels")
    fig = create_entry_chart(df, signal)
    st.plotly_chart(fig, use_container_width=True)
    
    # Disclaimer
    st.caption("""
    ⚠️ **Disclaimer**: Trading signals are for educational purposes only. 
    Past performance does not guarantee future results. 
    Always do your own research and manage risk appropriately.
    """)


if __name__ == "__main__":
    st.set_page_config(page_title="Trading Signals", layout="wide")
    render_signals_page()

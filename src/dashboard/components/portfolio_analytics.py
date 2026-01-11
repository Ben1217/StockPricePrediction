"""
Enhanced Portfolio Analytics Component
Advanced metrics, charts, and risk analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.theme import COLORS


# =============================================================================
# METRICS CALCULATIONS
# =============================================================================

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns."""
    return prices.pct_change().dropna()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns."""
    return (1 + returns).cumprod() - 1


def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """Calculate drawdown from peak."""
    cumulative = prices / prices.iloc[0]
    running_max = cumulative.cummax()
    return (cumulative - running_max) / running_max


def calculate_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive portfolio metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    
    Returns
    -------
    dict
        Dictionary of metrics
    """
    returns = calculate_returns(df['Close'])
    
    # Basic metrics
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    
    # Annualized metrics
    trading_days = len(returns)
    annual_factor = 252 / trading_days if trading_days > 0 else 1
    
    annual_return = returns.mean() * 252 * 100
    annual_volatility = returns.std() * np.sqrt(252) * 100
    
    # Risk-adjusted
    risk_free_rate = 0.05  # Assume 5% risk-free rate
    sharpe_ratio = (annual_return / 100 - risk_free_rate) / (annual_volatility / 100) if annual_volatility != 0 else 0
    
    # Drawdown
    drawdown = calculate_drawdown(df['Close'])
    max_drawdown = drawdown.min() * 100
    
    # Downside risk
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annual_return / 100 - risk_free_rate) / downside_std if downside_std != 0 else 0
    
    # Calmar ratio
    calmar_ratio = (annual_return / 100) / abs(max_drawdown / 100) if max_drawdown != 0 else 0
    
    # Value at Risk
    var_95 = np.percentile(returns, 5) * 100 if len(returns) > 0 else 0
    var_99 = np.percentile(returns, 1) * 100 if len(returns) > 0 else 0
    
    # Win rate
    winning_days = (returns > 0).sum()
    total_days = len(returns)
    win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
    
    # Best/Worst days
    best_day = returns.max() * 100 if len(returns) > 0 else 0
    worst_day = returns.min() * 100 if len(returns) > 0 else 0
    
    # Average gain/loss
    avg_gain = returns[returns > 0].mean() * 100 if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() * 100 if len(returns[returns < 0]) > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'var_99': var_99,
        'win_rate': win_rate,
        'best_day': best_day,
        'worst_day': worst_day,
        'avg_gain': avg_gain,
        'avg_loss': avg_loss,
        'trading_days': trading_days
    }


# =============================================================================
# CHARTS
# =============================================================================

def create_performance_chart(df: pd.DataFrame, height: int = 350) -> go.Figure:
    """Create cumulative returns chart."""
    returns = calculate_returns(df['Close'])
    cum_returns = calculate_cumulative_returns(returns)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cum_returns.index,
        y=cum_returns * 100,
        fill='tozeroy',
        name='Cumulative Return',
        line=dict(color=COLORS['accent_green'], width=2),
        fillcolor='rgba(76, 175, 80, 0.15)'
    ))
    
    fig.update_layout(
        title="Cumulative Returns",
        height=height,
        paper_bgcolor=COLORS['bg_primary'],
        plot_bgcolor=COLORS['bg_primary'],
        font=dict(color=COLORS['text_secondary']),
        xaxis=dict(gridcolor=COLORS['grid'], showgrid=True),
        yaxis=dict(gridcolor=COLORS['grid'], showgrid=True, title='Return (%)', side='right'),
        margin=dict(l=10, r=10, t=40, b=30),
        showlegend=False
    )
    
    return fig


def create_drawdown_chart(df: pd.DataFrame, height: int = 200) -> go.Figure:
    """Create drawdown chart."""
    drawdown = calculate_drawdown(df['Close'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color=COLORS['accent_red'], width=1.5),
        fillcolor='rgba(244, 67, 54, 0.15)'
    ))
    
    fig.update_layout(
        title="Drawdown",
        height=height,
        paper_bgcolor=COLORS['bg_primary'],
        plot_bgcolor=COLORS['bg_primary'],
        font=dict(color=COLORS['text_secondary']),
        xaxis=dict(gridcolor=COLORS['grid'], showgrid=True),
        yaxis=dict(gridcolor=COLORS['grid'], showgrid=True, title='Drawdown (%)', side='right'),
        margin=dict(l=10, r=10, t=40, b=30),
        showlegend=False
    )
    
    return fig


def create_returns_distribution(df: pd.DataFrame, height: int = 250) -> go.Figure:
    """Create returns distribution histogram."""
    returns = calculate_returns(df['Close']) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Daily Returns',
        marker_color=COLORS['accent_blue'],
        opacity=0.7
    ))
    
    # Add vertical lines for mean and VaR
    mean_return = returns.mean()
    var_95 = np.percentile(returns, 5)
    
    fig.add_vline(x=mean_return, line=dict(color=COLORS['accent_orange'], dash='dash'), 
                  annotation_text=f"Mean: {mean_return:.2f}%")
    fig.add_vline(x=var_95, line=dict(color=COLORS['accent_red'], dash='dash'),
                  annotation_text=f"VaR 95%: {var_95:.2f}%")
    
    fig.update_layout(
        title="Returns Distribution",
        height=height,
        paper_bgcolor=COLORS['bg_primary'],
        plot_bgcolor=COLORS['bg_primary'],
        font=dict(color=COLORS['text_secondary']),
        xaxis=dict(gridcolor=COLORS['grid'], showgrid=True, title='Daily Return (%)'),
        yaxis=dict(gridcolor=COLORS['grid'], showgrid=True, title='Frequency'),
        margin=dict(l=10, r=10, t=40, b=30),
        showlegend=False
    )
    
    return fig


def create_monthly_heatmap(df: pd.DataFrame, height: int = 300) -> go.Figure:
    """Create monthly returns heatmap."""
    returns = calculate_returns(df['Close'])
    
    # Group by year and month
    returns_df = returns.to_frame('returns')
    returns_df['year'] = returns_df.index.year
    returns_df['month'] = returns_df.index.month
    
    monthly = returns_df.groupby(['year', 'month'])['returns'].sum() * 100
    monthly_pivot = monthly.unstack()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure(data=go.Heatmap(
        z=monthly_pivot.values,
        x=months[:monthly_pivot.shape[1]],
        y=monthly_pivot.index.astype(str),
        colorscale=[
            [0, COLORS['accent_red']],
            [0.5, COLORS['bg_secondary']],
            [1, COLORS['accent_green']]
        ],
        zmid=0,
        text=np.round(monthly_pivot.values, 1),
        texttemplate="%{text}%",
        textfont={"size": 10},
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="Monthly Returns Heatmap",
        height=height,
        paper_bgcolor=COLORS['bg_primary'],
        plot_bgcolor=COLORS['bg_primary'],
        font=dict(color=COLORS['text_secondary']),
        margin=dict(l=10, r=10, t=40, b=30)
    )
    
    return fig


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_metrics_cards(metrics: Dict):
    """Render metrics as cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        color = COLORS['accent_green'] if metrics['total_return'] >= 0 else COLORS['accent_red']
        st.metric("Total Return", f"{metrics['total_return']:+.2f}%")
    
    with col2:
        st.metric("Annual Return", f"{metrics['annual_return']:+.2f}%")
    
    with col3:
        st.metric("Volatility", f"{metrics['annual_volatility']:.2f}%")
    
    with col4:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    
    with col5:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")


def render_risk_metrics(metrics: Dict):
    """Render risk metrics section."""
    st.markdown(f"""
    <h4 style="color: {COLORS['text_secondary']}; margin: 20px 0 12px 0;">
        📊 Risk Metrics
    </h4>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("VaR (95%)", f"{metrics['var_95']:.2f}%", help="Daily Value at Risk")
    
    with col2:
        st.metric("VaR (99%)", f"{metrics['var_99']:.2f}%", help="Daily Value at Risk")
    
    with col3:
        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}", help="Downside risk-adjusted return")
    
    with col4:
        st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}", help="Return / Max Drawdown")


def render_trading_stats(metrics: Dict):
    """Render trading statistics."""
    st.markdown(f"""
    <h4 style="color: {COLORS['text_secondary']}; margin: 20px 0 12px 0;">
        📈 Trading Statistics
    </h4>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    
    with col2:
        st.metric("Best Day", f"{metrics['best_day']:+.2f}%")
    
    with col3:
        st.metric("Worst Day", f"{metrics['worst_day']:.2f}%")
    
    with col4:
        st.metric("Trading Days", metrics['trading_days'])


def render_portfolio_analytics(df: pd.DataFrame, symbol: str = ""):
    """
    Render complete portfolio analytics view.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    symbol : str
        Stock symbol for display
    """
    st.markdown(f"""
    <h2 style="color: {COLORS['text_primary']}; margin-bottom: 16px;">
        💼 Portfolio Analytics {f'- {symbol}' if symbol else ''}
    </h2>
    """, unsafe_allow_html=True)
    
    # Calculate all metrics
    metrics = calculate_metrics(df)
    
    # Key metrics cards
    render_metrics_cards(metrics)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_performance_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_drawdown_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk metrics
    render_risk_metrics(metrics)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_returns_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_monthly_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trading stats
    render_trading_stats(metrics)

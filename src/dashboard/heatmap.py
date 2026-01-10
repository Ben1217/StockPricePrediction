"""
Market Heatmap Visualization Module
Interactive treemap visualization for market overview
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Tuple
from datetime import datetime
import yfinance as yf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.market_data import (
    get_sp500_constituents,
    get_market_heatmap_data,
    calculate_period_returns,
    fetch_batch_quotes,
    fetch_market_caps
)


def get_market_status() -> Tuple[str, str, str]:
    """
    Get current market status and data freshness indicator.
    
    Returns
    -------
    Tuple[str, str, str]
        (status_emoji, status_text, description)
    """
    try:
        spy = yf.Ticker('SPY')
        info = spy.info
        market_state = info.get('marketState', 'UNKNOWN')
        
        # Get last trade time
        last_trade_time = None
        if 'regularMarketTime' in info:
            last_trade_time = datetime.fromtimestamp(info['regularMarketTime'])
        
        if market_state == 'CLOSED':
            if last_trade_time:
                time_str = last_trade_time.strftime('%b %d, %I:%M %p')
                return "🔵", "Market Closed", f"Showing closing prices from {time_str}"
            return "🔵", "Market Closed", "Showing latest closing prices"
        
        elif market_state == 'PRE':
            return "🟡", "Pre-Market", "Pre-market trading session - limited volume"
        
        elif market_state == 'POST':
            return "🟡", "After-Hours", "Extended hours trading - prices may differ from regular session"
        
        elif market_state == 'REGULAR':
            return "🟢", "Market Open", "Live data - updated every 5 minutes"
        
        else:
            return "⚪", "Unknown", "Market status unavailable"
            
    except Exception as e:
        return "⚪", "Unknown", f"Could not determine market status"


def render_market_status():
    """
    Render market status indicator at the top of the heatmap.
    """
    emoji, status, description = get_market_status()
    
    # Create a styled status bar
    if "Open" in status:
        st.success(f"{emoji} **{status}** - {description}")
    elif "Closed" in status:
        st.info(f"{emoji} **{status}** - {description}")
    elif "Pre" in status or "After" in status:
        st.warning(f"{emoji} **{status}** - {description}")
    else:
        st.info(f"{emoji} **{status}** - {description}")


def create_heatmap_figure(
    df: pd.DataFrame,
    color_column: str = 'DailyReturn',
    size_column: str = 'MarketCap',
    title: str = 'S&P 500 Market Heatmap'
) -> go.Figure:
    """
    Create a Plotly treemap figure for market heatmap.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Symbol, Sector, MarketCap, DailyReturn columns
    color_column : str
        Column to use for color coding
    size_column : str
        Column to use for tile sizing
    title : str
        Chart title
    
    Returns
    -------
    go.Figure
        Plotly treemap figure
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available. Click 'Refresh Data' to load.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=600)
        return fig
    
    # Clamp returns for color scale (-10% to +10%)
    df = df.copy()
    df['ColorValue'] = df[color_column].clip(-10, 10)
    
    # Create custom hover text
    df['HoverText'] = df.apply(
        lambda x: (
            f"<b>{x['Symbol']}</b><br>"
            f"<b>{x['Company']}</b><br>"
            f"─────────────<br>"
            f"Price: ${x['Price']:.2f}<br>"
            f"Change: {x['DailyReturn']:+.2f}%<br>"
            f"Volume: {x['VolumeM']:.1f}M<br>"
            f"Market Cap: ${x['MarketCapB']:.1f}B<br>"
            f"Sector: {x['Sector']}"
        ),
        axis=1
    )
    
    # Create treemap
    fig = px.treemap(
        df,
        path=[px.Constant("Market"), 'Sector', 'Symbol'],
        values=size_column,
        color='ColorValue',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        custom_data=['HoverText', 'DailyReturn', 'Company'],
        title=title
    )
    
    # Update traces for better display
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[1]:+.1f}%",
        textposition="middle center",
        hovertemplate="%{customdata[0]}<extra></extra>",
        textfont=dict(size=12),
        marker=dict(
            line=dict(width=1, color='white'),
            cornerradius=3
        )
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        margin=dict(t=50, l=10, r=10, b=10),
        coloraxis_colorbar=dict(
            title="Return %",
            ticksuffix="%",
            len=0.6,
            thickness=15,
            tickvals=[-10, -5, 0, 5, 10],
            ticktext=["-10%", "-5%", "0%", "+5%", "+10%"]
        ),
        font=dict(family="Arial, sans-serif")
    )
    
    return fig


def render_summary_metrics(df: pd.DataFrame):
    """
    Render summary metrics bar above the heatmap.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data DataFrame
    """
    if df.empty:
        st.warning("No data available")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    gainers = (df['DailyReturn'] > 0).sum()
    decliners = (df['DailyReturn'] < 0).sum()
    unchanged = (df['DailyReturn'] == 0).sum()
    avg_return = df['DailyReturn'].mean()
    total_volume = df['Volume'].sum() / 1e9  # In billions
    
    with col1:
        st.metric("📈 Gainers", gainers, delta=None)
    
    with col2:
        st.metric("📉 Decliners", decliners, delta=None)
    
    with col3:
        st.metric("➡️ Unchanged", unchanged, delta=None)
    
    with col4:
        delta_color = "normal" if avg_return >= 0 else "inverse"
        st.metric("📊 Avg Return", f"{avg_return:+.2f}%")
    
    with col5:
        st.metric("📦 Volume", f"{total_volume:.1f}B")


def render_sector_breakdown(df: pd.DataFrame):
    """
    Render sector performance breakdown.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data DataFrame
    """
    if df.empty:
        return
    
    sector_stats = df.groupby('Sector').agg({
        'DailyReturn': 'mean',
        'Symbol': 'count',
        'MarketCap': 'sum'
    }).rename(columns={'Symbol': 'Stocks', 'DailyReturn': 'Avg Return'})
    
    sector_stats = sector_stats.sort_values('Avg Return', ascending=False)
    sector_stats['Avg Return'] = sector_stats['Avg Return'].apply(lambda x: f"{x:+.2f}%")
    sector_stats['MarketCap'] = (sector_stats['MarketCap'] / 1e12).apply(lambda x: f"${x:.1f}T")
    
    st.dataframe(
        sector_stats,
        use_container_width=True,
        column_config={
            'Avg Return': st.column_config.TextColumn("Avg Return"),
            'Stocks': st.column_config.NumberColumn("Stocks"),
            'MarketCap': st.column_config.TextColumn("Market Cap")
        }
    )


def render_heatmap_page():
    """
    Render the complete market heatmap page.
    """
    st.subheader("🗺️ Market Heatmap")
    
    # Market status indicator
    render_market_status()
    
    # Control row
    col1, col2, col3, col4 = st.columns([2, 2, 3, 1])
    
    with col1:
        market = st.selectbox(
            "Market",
            ["S&P 500"],  # Russell 2000 in Phase 2
            key="heatmap_market"
        )
    
    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            ["1D", "5D", "1M", "3M", "YTD", "1Y"],
            key="heatmap_timeframe"
        )
    
    with col3:
        # Get all sectors for filter
        constituents = get_sp500_constituents()
        all_sectors = sorted(constituents['Sector'].unique().tolist()) if not constituents.empty else []
        
        selected_sectors = st.multiselect(
            "Filter Sectors",
            options=all_sectors,
            default=all_sectors,
            key="heatmap_sectors"
        )
    
    with col4:
        st.write("")  # Spacer
        st.write("")  # Spacer
        refresh = st.button("🔄 Refresh", key="heatmap_refresh", use_container_width=True)
    
    # Clear cache on refresh
    if refresh:
        st.cache_data.clear()
        st.rerun()
    
    # Load data with progress
    with st.spinner("Loading market data..."):
        df = get_market_heatmap_data(market)
    
    if df.empty:
        st.error("Failed to load market data. Please try refreshing.")
        return
    
    # Apply sector filter
    if selected_sectors:
        df = df[df['Sector'].isin(selected_sectors)]
    
    if df.empty:
        st.warning("No stocks match the selected filters.")
        return
    
    # Calculate returns for selected timeframe if not 1D
    if timeframe != "1D":
        symbols = df['Symbol'].tolist()
        period_returns = calculate_period_returns(symbols, timeframe)
        
        if not period_returns.empty:
            df = df.merge(
                period_returns[['Symbol', 'PeriodReturn']], 
                on='Symbol', 
                how='left'
            )
            df['DailyReturn'] = df['PeriodReturn'].fillna(df['DailyReturn'])
            df['DisplayText'] = df.apply(
                lambda x: f"{x['Symbol']}<br>{x['DailyReturn']:+.1f}%", 
                axis=1
            )
    
    # Summary metrics
    render_summary_metrics(df)
    
    st.markdown("---")
    
    # Create and display heatmap
    title = f"{market} Market Heatmap ({timeframe})"
    fig = create_heatmap_figure(df, title=title)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector breakdown (expandable)
    with st.expander("📊 Sector Performance Breakdown"):
        render_sector_breakdown(df)
    
    # Top movers
    with st.expander("🔥 Top Movers"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 Gainers**")
            top_gainers = df.nlargest(10, 'DailyReturn')[['Symbol', 'Company', 'DailyReturn', 'Price']]
            top_gainers['DailyReturn'] = top_gainers['DailyReturn'].apply(lambda x: f"{x:+.2f}%")
            top_gainers['Price'] = top_gainers['Price'].apply(lambda x: f"${x:.2f}")
            st.dataframe(top_gainers, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Top 10 Decliners**")
            top_losers = df.nsmallest(10, 'DailyReturn')[['Symbol', 'Company', 'DailyReturn', 'Price']]
            top_losers['DailyReturn'] = top_losers['DailyReturn'].apply(lambda x: f"{x:+.2f}%")
            top_losers['Price'] = top_losers['Price'].apply(lambda x: f"${x:.2f}")
            st.dataframe(top_losers, use_container_width=True, hide_index=True)
    
    # Footer with timestamp and disclaimer
    st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Data cached for 5 minutes")
    st.caption("📊 Data provided by Yahoo Finance. Prices may be delayed up to 15 minutes during market hours.")


if __name__ == "__main__":
    st.set_page_config(page_title="Market Heatmap", layout="wide")
    render_heatmap_page()

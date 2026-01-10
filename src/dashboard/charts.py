"""
Professional Chart Components
Creates TradingView-style candlestick charts with dark theme
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from src.dashboard.theme import COLORS


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "",
    show_volume: bool = True,
    height: int = 600
) -> go.Figure:
    """
    Create a professional candlestick chart with optional volume bars.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: Open, High, Low, Close, Volume
    title : str
        Chart title
    show_volume : bool
        Whether to include volume subplot
    height : int
        Chart height in pixels
    
    Returns
    -------
    go.Figure
        Plotly figure with candlestick chart
    """
    # Create subplots if volume is shown
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25]
        )
    else:
        fig = go.Figure()
    
    # Candlestick chart
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing=dict(
            line=dict(color=COLORS['candle_up'], width=1),
            fillcolor=COLORS['candle_up']
        ),
        decreasing=dict(
            line=dict(color=COLORS['candle_down'], width=1),
            fillcolor=COLORS['candle_down']
        ),
        showlegend=False
    )
    
    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Volume bars (if enabled)
    if show_volume and 'Volume' in df.columns:
        # Determine colors based on price direction
        colors = [
            COLORS['volume_up'] if close >= open else COLORS['volume_down']
            for close, open in zip(df['Close'], df['Open'])
        ]
        
        volume_bars = go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker=dict(
                color=colors,
                opacity=0.6
            ),
            showlegend=False
        )
        fig.add_trace(volume_bars, row=2, col=1)
    
    # Layout styling
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color=COLORS['text_primary'])
        ),
        height=height,
        paper_bgcolor=COLORS['bg_primary'],
        plot_bgcolor=COLORS['bg_primary'],
        font=dict(
            family='Inter, sans-serif',
            color=COLORS['text_secondary'],
            size=12
        ),
        xaxis=dict(
            gridcolor=COLORS['grid'],
            showgrid=True,
            zeroline=False,
            rangeslider=dict(visible=False),
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor=COLORS['accent_orange'],
            spikethickness=1,
            spikedash='dot'
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            showgrid=True,
            zeroline=False,
            side='right',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor=COLORS['accent_orange'],
            spikethickness=1,
            spikedash='dot'
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor=COLORS['bg_secondary'],
            bordercolor=COLORS['border'],
            font=dict(color=COLORS['text_primary'], size=12)
        ),
        margin=dict(l=10, r=60, t=50, b=30),
        showlegend=False
    )
    
    # Volume subplot styling
    if show_volume:
        fig.update_xaxes(
            gridcolor=COLORS['grid'],
            showgrid=True,
            zeroline=False,
            row=2, col=1
        )
        fig.update_yaxes(
            gridcolor=COLORS['grid'],
            showgrid=False,
            zeroline=False,
            side='right',
            row=2, col=1
        )
        # Remove y-axis title for volume
        fig.update_yaxes(title_text="", row=2, col=1)
    
    # Enable modebar tools
    fig.update_layout(
        modebar=dict(
            bgcolor=COLORS['bg_secondary'],
            color=COLORS['text_secondary'],
            activecolor=COLORS['accent_orange']
        )
    )
    
    return fig


def add_moving_averages(
    fig: go.Figure,
    df: pd.DataFrame,
    periods: List[int] = [20, 50, 200],
    row: int = 1
) -> go.Figure:
    """
    Add moving average lines to chart.
    
    Parameters
    ----------
    fig : go.Figure
        Existing figure
    df : pd.DataFrame
        Price data
    periods : list
        MA periods to add
    row : int
        Subplot row (1 for main chart)
    
    Returns
    -------
    go.Figure
        Updated figure
    """
    ma_colors = {
        20: '#2196f3',   # Blue
        50: '#9c27b0',   # Purple
        200: '#ff9800'   # Orange
    }
    
    for period in periods:
        col_name = f'SMA_{period}'
        if col_name in df.columns:
            ma_data = df[col_name]
        else:
            ma_data = df['Close'].rolling(period).mean()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ma_data,
                name=f'{period} MA',
                line=dict(
                    color=ma_colors.get(period, '#ffffff'),
                    width=1.5
                ),
                hovertemplate=f'{period} MA: $%{{y:.2f}}<extra></extra>'
            ),
            row=row, col=1
        )
    
    return fig


def add_support_resistance_lines(
    fig: go.Figure,
    levels: List[Dict],
    row: int = 1
) -> go.Figure:
    """
    Add horizontal support/resistance lines.
    
    Parameters
    ----------
    fig : go.Figure
        Existing figure
    levels : list
        List of dicts with 'price', 'label', 'type' (support/resistance)
    row : int
        Subplot row
    
    Returns
    -------
    go.Figure
        Updated figure
    """
    for level in levels:
        price = level['price']
        label = level.get('label', f'${price:.2f}')
        level_type = level.get('type', 'support')
        
        color = COLORS['accent_green'] if level_type == 'support' else COLORS['accent_red']
        
        fig.add_hline(
            y=price,
            line=dict(
                color=color,
                width=1,
                dash='dash'
            ),
            annotation=dict(
                text=label,
                font=dict(color=color, size=10),
                bgcolor=COLORS['bg_secondary']
            ),
            row=row, col=1
        )
    
    return fig


def add_trade_markers(
    fig: go.Figure,
    trades: List[Dict],
    row: int = 1
) -> go.Figure:
    """
    Add entry/TP/SL markers to chart.
    
    Parameters
    ----------
    fig : go.Figure
        Existing figure
    trades : list
        List of trade dicts with 'date', 'price', 'type' (entry/tp/sl)
    row : int
        Subplot row
    
    Returns
    -------
    go.Figure
        Updated figure
    """
    for trade in trades:
        date = trade['date']
        price = trade['price']
        trade_type = trade.get('type', 'entry')
        
        if trade_type == 'entry':
            color = COLORS['accent_green']
            symbol = 'triangle-up'
            text = 'Entry'
        elif trade_type == 'tp':
            color = COLORS['accent_green']
            symbol = 'circle'
            text = 'TP Hit'
        else:  # sl
            color = COLORS['accent_red']
            symbol = 'x'
            text = 'SL Hit'
        
        fig.add_trace(
            go.Scatter(
                x=[date],
                y=[price],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=color,
                    symbol=symbol
                ),
                text=[text],
                textposition='top center',
                textfont=dict(color=color, size=10),
                showlegend=False,
                hovertemplate=f'{text}<br>${price:.2f}<extra></extra>'
            ),
            row=row, col=1
        )
    
    return fig


def create_simple_line_chart(
    df: pd.DataFrame,
    column: str = 'Close',
    title: str = "",
    height: int = 400
) -> go.Figure:
    """
    Create a simple line chart with dark theme.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with datetime index
    column : str
        Column to plot
    title : str
        Chart title
    height : int
        Chart height
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=column,
            line=dict(color=COLORS['accent_green'], width=2),
            fill='tozeroy',
            fillcolor=f"rgba(76, 175, 80, 0.1)"
        )
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=height,
        paper_bgcolor=COLORS['bg_primary'],
        plot_bgcolor=COLORS['bg_primary'],
        font=dict(color=COLORS['text_secondary']),
        xaxis=dict(gridcolor=COLORS['grid'], showgrid=True),
        yaxis=dict(gridcolor=COLORS['grid'], showgrid=True, side='right'),
        margin=dict(l=10, r=60, t=50, b=30),
        hovermode='x unified'
    )
    
    return fig

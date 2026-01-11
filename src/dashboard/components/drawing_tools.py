"""
Chart Drawing Tools Component
Trendlines, fibonacci, shapes, and annotations for charts
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.theme import COLORS


# =============================================================================
# DRAWING TYPES
# =============================================================================

class DrawingType:
    HORIZONTAL_LINE = "h_line"
    TRENDLINE = "trendline"
    RECTANGLE = "rectangle"
    FIBONACCI = "fibonacci"
    TEXT = "text"


DRAWING_LABELS = {
    DrawingType.HORIZONTAL_LINE: "📏 Horizontal Line",
    DrawingType.TRENDLINE: "📈 Trendline",
    DrawingType.RECTANGLE: "⬜ Rectangle Zone",
    DrawingType.FIBONACCI: "🔢 Fibonacci Retracement",
    DrawingType.TEXT: "📝 Text Note"
}

DRAWING_COLORS = {
    'support': COLORS['accent_green'],
    'resistance': COLORS['accent_red'],
    'neutral': COLORS['accent_orange'],
    'blue': COLORS['accent_blue'],
    'white': COLORS['text_primary']
}


# =============================================================================
# DRAWING STATE
# =============================================================================

def init_drawings_state():
    """Initialize drawings in session state."""
    if 'chart_drawings' not in st.session_state:
        st.session_state.chart_drawings = {}  # symbol -> list of drawings


def get_drawings_for_symbol(symbol: str) -> List[Dict]:
    """Get all drawings for a specific symbol."""
    init_drawings_state()
    return st.session_state.chart_drawings.get(symbol.upper(), [])


def add_drawing(
    symbol: str,
    drawing_type: str,
    params: Dict
) -> Dict:
    """
    Add a drawing to a symbol's chart.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    drawing_type : str
        Type of drawing
    params : dict
        Drawing parameters (varies by type)
    
    Returns
    -------
    dict
        The created drawing
    """
    init_drawings_state()
    symbol = symbol.upper()
    
    if symbol not in st.session_state.chart_drawings:
        st.session_state.chart_drawings[symbol] = []
    
    drawing = {
        'id': len(st.session_state.chart_drawings[symbol]) + 1,
        'type': drawing_type,
        'params': params,
        'created_at': datetime.now().isoformat(),
        'visible': True
    }
    
    st.session_state.chart_drawings[symbol].append(drawing)
    return drawing


def remove_drawing(symbol: str, drawing_id: int) -> bool:
    """Remove a drawing by ID."""
    init_drawings_state()
    symbol = symbol.upper()
    
    if symbol not in st.session_state.chart_drawings:
        return False
    
    drawings = st.session_state.chart_drawings[symbol]
    for i, d in enumerate(drawings):
        if d['id'] == drawing_id:
            drawings.pop(i)
            return True
    return False


def clear_all_drawings(symbol: str):
    """Clear all drawings for a symbol."""
    init_drawings_state()
    st.session_state.chart_drawings[symbol.upper()] = []


# =============================================================================
# APPLY DRAWINGS TO CHART
# =============================================================================

def apply_horizontal_line(fig: go.Figure, params: Dict, row: int = 1) -> go.Figure:
    """Add horizontal line to chart."""
    price = params.get('price', 0)
    color = DRAWING_COLORS.get(params.get('color', 'neutral'), COLORS['accent_orange'])
    label = params.get('label', f"${price:.2f}")
    line_style = params.get('style', 'dash')
    
    fig.add_hline(
        y=price,
        line=dict(color=color, width=1.5, dash=line_style),
        annotation=dict(
            text=label,
            font=dict(color=color, size=10),
            bgcolor=COLORS['bg_secondary'],
            bordercolor=color
        ),
        row=row, col=1
    )
    return fig


def apply_trendline(fig: go.Figure, params: Dict, df_index) -> go.Figure:
    """Add trendline between two points."""
    start_idx = params.get('start_idx', 0)
    start_price = params.get('start_price', 0)
    end_idx = params.get('end_idx', len(df_index) - 1)
    end_price = params.get('end_price', 0)
    color = DRAWING_COLORS.get(params.get('color', 'neutral'), COLORS['accent_orange'])
    
    # Calculate line across chart
    if start_idx < len(df_index) and end_idx < len(df_index):
        fig.add_trace(go.Scatter(
            x=[df_index[start_idx], df_index[end_idx]],
            y=[start_price, end_price],
            mode='lines',
            line=dict(color=color, width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    return fig


def apply_rectangle(fig: go.Figure, params: Dict, df_index) -> go.Figure:
    """Add rectangle zone to chart."""
    start_idx = params.get('start_idx', 0)
    end_idx = params.get('end_idx', len(df_index) - 1)
    top_price = params.get('top_price', 0)
    bottom_price = params.get('bottom_price', 0)
    color = DRAWING_COLORS.get(params.get('color', 'neutral'), COLORS['accent_orange'])
    
    if start_idx < len(df_index) and end_idx < len(df_index):
        fig.add_shape(
            type="rect",
            x0=df_index[start_idx],
            x1=df_index[end_idx],
            y0=bottom_price,
            y1=top_price,
            line=dict(color=color, width=1),
            fillcolor=f"{color}20",
            layer="below"
        )
    
    return fig


def apply_fibonacci(fig: go.Figure, params: Dict, row: int = 1) -> go.Figure:
    """Add fibonacci retracement levels."""
    high_price = params.get('high', 0)
    low_price = params.get('low', 0)
    
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    fib_colors = ['#ef5350', '#ff9800', '#ffeb3b', '#4caf50', '#2196f3', '#9c27b0', '#4caf50']
    
    diff = high_price - low_price
    
    for level, color in zip(fib_levels, fib_colors):
        price = high_price - (diff * level)
        
        fig.add_hline(
            y=price,
            line=dict(color=color, width=1, dash='dot'),
            annotation=dict(
                text=f"{level:.1%} (${price:.2f})",
                font=dict(color=color, size=9),
                bgcolor=COLORS['bg_secondary']
            ),
            row=row, col=1
        )
    
    return fig


def apply_text_annotation(fig: go.Figure, params: Dict, df_index) -> go.Figure:
    """Add text annotation to chart."""
    idx = params.get('idx', 0)
    price = params.get('price', 0)
    text = params.get('text', '')
    color = DRAWING_COLORS.get(params.get('color', 'white'), COLORS['text_primary'])
    
    if idx < len(df_index):
        fig.add_annotation(
            x=df_index[idx],
            y=price,
            text=text,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor=color,
            font=dict(color=color, size=11),
            bgcolor=COLORS['bg_secondary'],
            bordercolor=color,
            borderwidth=1
        )
    
    return fig


def apply_drawings_to_chart(
    fig: go.Figure,
    symbol: str,
    df_index,
    row: int = 1
) -> go.Figure:
    """
    Apply all saved drawings to a chart figure.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure to modify
    symbol : str
        Stock symbol
    df_index : pd.Index
        DataFrame index for date references
    row : int
        Subplot row number
    
    Returns
    -------
    go.Figure
        Modified figure with drawings
    """
    drawings = get_drawings_for_symbol(symbol)
    
    for drawing in drawings:
        if not drawing.get('visible', True):
            continue
        
        dtype = drawing['type']
        params = drawing['params']
        
        if dtype == DrawingType.HORIZONTAL_LINE:
            fig = apply_horizontal_line(fig, params, row)
        elif dtype == DrawingType.TRENDLINE:
            fig = apply_trendline(fig, params, df_index)
        elif dtype == DrawingType.RECTANGLE:
            fig = apply_rectangle(fig, params, df_index)
        elif dtype == DrawingType.FIBONACCI:
            fig = apply_fibonacci(fig, params, row)
        elif dtype == DrawingType.TEXT:
            fig = apply_text_annotation(fig, params, df_index)
    
    return fig


# =============================================================================
# DRAWING UI COMPONENTS
# =============================================================================

def render_drawing_toolbar(symbol: str, current_price: float = 0, high: float = 0, low: float = 0):
    """
    Render the drawing tools toolbar.
    
    Parameters
    ----------
    symbol : str
        Current stock symbol
    current_price : float
        Current stock price for default values
    high : float
        Recent high for fibonacci
    low : float
        Recent low for fibonacci
    """
    init_drawings_state()
    
    st.markdown(f"""
    <div style="
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 16px;
    ">
        <div style="color: {COLORS['text_primary']}; font-weight: 600; font-size: 13px; margin-bottom: 10px;">
            ✏️ Drawing Tools
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        drawing_type = st.selectbox(
            "Tool",
            options=list(DRAWING_LABELS.keys()),
            format_func=lambda x: DRAWING_LABELS[x],
            key="drawing_tool_type",
            label_visibility="collapsed"
        )
    
    with col2:
        color = st.selectbox(
            "Color",
            options=list(DRAWING_COLORS.keys()),
            format_func=lambda x: x.title(),
            key="drawing_color",
            label_visibility="collapsed"
        )
    
    # Dynamic parameters based on tool
    if drawing_type == DrawingType.HORIZONTAL_LINE:
        with col3:
            price = st.number_input(
                "Price",
                value=float(current_price) if current_price else 100.0,
                step=0.50,
                key="hline_price"
            )
        with col4:
            label = st.text_input("Label", value="", key="hline_label", placeholder="S/R Level")
        
        if st.button("Add Line", key="add_hline", use_container_width=True):
            add_drawing(symbol, DrawingType.HORIZONTAL_LINE, {
                'price': price,
                'color': color,
                'label': label if label else f"${price:.2f}",
                'style': 'dash'
            })
            st.rerun()
    
    elif drawing_type == DrawingType.FIBONACCI:
        with col3:
            fib_high = st.number_input("High", value=float(high) if high else 150.0, key="fib_high")
        with col4:
            fib_low = st.number_input("Low", value=float(low) if low else 100.0, key="fib_low")
        
        if st.button("Add Fibonacci", key="add_fib", use_container_width=True):
            add_drawing(symbol, DrawingType.FIBONACCI, {
                'high': fib_high,
                'low': fib_low
            })
            st.rerun()
    
    elif drawing_type == DrawingType.RECTANGLE:
        col3a, col3b = col3.columns(2)
        with col3a:
            rect_top = st.number_input("Top $", value=float(high) if high else 110.0, key="rect_top")
        with col3b:
            rect_bottom = st.number_input("Bottom $", value=float(low) if low else 90.0, key="rect_bottom")
        
        with col4:
            st.markdown(f"<small style='color:{COLORS['text_muted']}'>Uses full chart range</small>", unsafe_allow_html=True)
        
        if st.button("Add Zone", key="add_rect", use_container_width=True):
            add_drawing(symbol, DrawingType.RECTANGLE, {
                'start_idx': 0,
                'end_idx': 100,  # Will be capped to actual length
                'top_price': rect_top,
                'bottom_price': rect_bottom,
                'color': color
            })
            st.rerun()
    
    elif drawing_type == DrawingType.TEXT:
        with col3:
            text_price = st.number_input("At Price", value=float(current_price), key="text_price")
        with col4:
            text_content = st.text_input("Text", key="text_content", placeholder="Note")
        
        if st.button("Add Note", key="add_text", use_container_width=True):
            if text_content:
                add_drawing(symbol, DrawingType.TEXT, {
                    'idx': 50,  # Middle of chart
                    'price': text_price,
                    'text': text_content,
                    'color': color
                })
                st.rerun()


def render_drawings_list(symbol: str):
    """Render list of drawings with delete option."""
    init_drawings_state()
    
    drawings = get_drawings_for_symbol(symbol)
    
    if not drawings:
        return
    
    st.markdown(f"""
    <div style="color: {COLORS['text_secondary']}; font-size: 12px; margin: 8px 0;">
        Active Drawings ({len(drawings)})
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns([4, 1])
    
    for drawing in drawings:
        with cols[0]:
            dtype_label = DRAWING_LABELS.get(drawing['type'], drawing['type'])
            params = drawing['params']
            
            if drawing['type'] == DrawingType.HORIZONTAL_LINE:
                detail = f"${params.get('price', 0):.2f}"
            elif drawing['type'] == DrawingType.FIBONACCI:
                detail = f"${params.get('low', 0):.0f} - ${params.get('high', 0):.0f}"
            elif drawing['type'] == DrawingType.RECTANGLE:
                detail = f"${params.get('bottom_price', 0):.0f} - ${params.get('top_price', 0):.0f}"
            else:
                detail = ""
            
            st.markdown(f"""
            <span style="color: {COLORS['text_muted']}; font-size: 11px;">
                {dtype_label.split(' ')[0]} {detail}
            </span>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            if st.button("✕", key=f"del_draw_{drawing['id']}", help="Remove"):
                remove_drawing(symbol, drawing['id'])
                st.rerun()
    
    if st.button("🗑️ Clear All", key="clear_drawings", use_container_width=True):
        clear_all_drawings(symbol)
        st.rerun()

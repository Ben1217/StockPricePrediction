"""
Alerts System Component
Price alerts, signal notifications, and alert management
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Callable
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.theme import COLORS


# =============================================================================
# ALERT DATA STRUCTURES
# =============================================================================

class AlertType:
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PERCENT_CHANGE = "percent_change"
    RSI_ABOVE = "rsi_above"
    RSI_BELOW = "rsi_below"
    SIGNAL_CHANGE = "signal_change"


ALERT_TYPE_LABELS = {
    AlertType.PRICE_ABOVE: "Price Above",
    AlertType.PRICE_BELOW: "Price Below",
    AlertType.PERCENT_CHANGE: "% Change",
    AlertType.RSI_ABOVE: "RSI Above",
    AlertType.RSI_BELOW: "RSI Below",
    AlertType.SIGNAL_CHANGE: "Signal Change"
}


# =============================================================================
# ALERT STATE MANAGEMENT
# =============================================================================

def init_alerts_state():
    """Initialize alerts in session state."""
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    if 'triggered_alerts' not in st.session_state:
        st.session_state.triggered_alerts = []
    
    if 'alert_count' not in st.session_state:
        st.session_state.alert_count = 0


def add_alert(
    symbol: str,
    alert_type: str,
    value: float,
    note: str = ""
) -> Dict:
    """
    Add a new alert.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    alert_type : str
        Type of alert (from AlertType class)
    value : float
        Threshold value
    note : str
        Optional note
    
    Returns
    -------
    dict
        The created alert
    """
    init_alerts_state()
    
    alert = {
        'id': len(st.session_state.alerts) + 1,
        'symbol': symbol.upper(),
        'type': alert_type,
        'value': value,
        'note': note,
        'created_at': datetime.now().isoformat(),
        'active': True,
        'triggered': False
    }
    
    st.session_state.alerts.append(alert)
    st.session_state.alert_count = len([a for a in st.session_state.alerts if a['active']])
    
    return alert


def remove_alert(alert_id: int) -> bool:
    """Remove an alert by ID."""
    init_alerts_state()
    
    for i, alert in enumerate(st.session_state.alerts):
        if alert['id'] == alert_id:
            st.session_state.alerts.pop(i)
            st.session_state.alert_count = len([a for a in st.session_state.alerts if a['active']])
            return True
    return False


def toggle_alert(alert_id: int) -> bool:
    """Toggle alert active status."""
    init_alerts_state()
    
    for alert in st.session_state.alerts:
        if alert['id'] == alert_id:
            alert['active'] = not alert['active']
            st.session_state.alert_count = len([a for a in st.session_state.alerts if a['active']])
            return True
    return False


def check_alerts(current_data: Dict[str, Dict]) -> List[Dict]:
    """
    Check all alerts against current data.
    
    Parameters
    ----------
    current_data : dict
        Dict of symbol -> {price, change, rsi, signal}
    
    Returns
    -------
    list
        List of triggered alerts
    """
    init_alerts_state()
    triggered = []
    
    for alert in st.session_state.alerts:
        if not alert['active'] or alert['triggered']:
            continue
        
        symbol = alert['symbol']
        if symbol not in current_data:
            continue
        
        data = current_data[symbol]
        alert_type = alert['type']
        value = alert['value']
        
        is_triggered = False
        
        if alert_type == AlertType.PRICE_ABOVE and data.get('price', 0) > value:
            is_triggered = True
        elif alert_type == AlertType.PRICE_BELOW and data.get('price', 0) < value:
            is_triggered = True
        elif alert_type == AlertType.PERCENT_CHANGE and abs(data.get('change', 0)) > value:
            is_triggered = True
        elif alert_type == AlertType.RSI_ABOVE and data.get('rsi', 50) > value:
            is_triggered = True
        elif alert_type == AlertType.RSI_BELOW and data.get('rsi', 50) < value:
            is_triggered = True
        
        if is_triggered:
            alert['triggered'] = True
            alert['triggered_at'] = datetime.now().isoformat()
            triggered.append(alert)
            st.session_state.triggered_alerts.append(alert)
    
    return triggered


def get_active_alerts() -> List[Dict]:
    """Get all active (non-triggered) alerts."""
    init_alerts_state()
    return [a for a in st.session_state.alerts if a['active'] and not a['triggered']]


def get_triggered_alerts() -> List[Dict]:
    """Get all triggered alerts."""
    init_alerts_state()
    return st.session_state.triggered_alerts


def clear_triggered_alerts():
    """Clear triggered alerts history."""
    init_alerts_state()
    st.session_state.triggered_alerts = []
    for alert in st.session_state.alerts:
        alert['triggered'] = False


# =============================================================================
# ALERT UI COMPONENTS
# =============================================================================

def render_create_alert_form(default_symbol: str = "SPY"):
    """Render the create alert form."""
    init_alerts_state()
    
    st.markdown(f"""
    <div style="
        color: {COLORS['text_primary']};
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 12px;
    ">🔔 Create Alert</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input(
            "Symbol",
            value=default_symbol,
            key="alert_symbol",
            placeholder="SPY"
        )
    
    with col2:
        alert_type = st.selectbox(
            "Alert Type",
            options=list(ALERT_TYPE_LABELS.keys()),
            format_func=lambda x: ALERT_TYPE_LABELS[x],
            key="alert_type"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if alert_type in [AlertType.PRICE_ABOVE, AlertType.PRICE_BELOW]:
            value = st.number_input("Price", min_value=0.0, value=100.0, step=1.0, key="alert_value")
        elif alert_type == AlertType.PERCENT_CHANGE:
            value = st.number_input("% Threshold", min_value=0.0, value=5.0, step=0.5, key="alert_value")
        elif alert_type in [AlertType.RSI_ABOVE, AlertType.RSI_BELOW]:
            value = st.number_input("RSI Level", min_value=0, max_value=100, value=70 if "above" in alert_type else 30, key="alert_value")
        else:
            value = st.number_input("Value", value=0.0, key="alert_value")
    
    with col2:
        note = st.text_input("Note (optional)", key="alert_note", placeholder="e.g., Buy signal")
    
    if st.button("➕ Create Alert", key="create_alert_btn", use_container_width=True, type="primary"):
        alert = add_alert(symbol, alert_type, value, note)
        st.success(f"✅ Alert created for {symbol}")
        st.rerun()


def render_alerts_list():
    """Render the list of active alerts."""
    init_alerts_state()
    
    active_alerts = get_active_alerts()
    
    if not active_alerts:
        st.markdown(f"""
        <div style="
            color: {COLORS['text_muted']};
            text-align: center;
            padding: 20px;
            font-size: 13px;
        ">No active alerts. Create one above.</div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"""
    <div style="
        color: {COLORS['text_primary']};
        font-weight: 600;
        font-size: 14px;
        margin: 16px 0 12px 0;
    ">Active Alerts ({len(active_alerts)})</div>
    """, unsafe_allow_html=True)
    
    for alert in active_alerts:
        alert_id = alert['id']
        symbol = alert['symbol']
        alert_type = ALERT_TYPE_LABELS.get(alert['type'], alert['type'])
        value = alert['value']
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"""
            <div style="
                background: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px 12px;
            ">
                <span style="color: {COLORS['accent_orange']}; font-weight: 600;">{symbol}</span>
                <span style="color: {COLORS['text_muted']}; font-size: 12px; margin-left: 8px;">
                    {alert_type}: {value}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if alert.get('note'):
                st.markdown(f"""
                <div style="color: {COLORS['text_muted']}; font-size: 11px; padding-top: 8px;">
                    📝 {alert['note']}
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if st.button("🗑️", key=f"del_alert_{alert_id}", help="Delete alert"):
                remove_alert(alert_id)
                st.rerun()


def render_triggered_alerts():
    """Render triggered alerts notification."""
    init_alerts_state()
    
    triggered = get_triggered_alerts()
    
    if not triggered:
        return
    
    st.markdown(f"""
    <div style="
        background: {COLORS['accent_red']}20;
        border: 1px solid {COLORS['accent_red']};
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 16px;
    ">
        <div style="color: {COLORS['accent_red']}; font-weight: 600; margin-bottom: 8px;">
            🚨 Triggered Alerts ({len(triggered)})
        </div>
    """, unsafe_allow_html=True)
    
    for alert in triggered[-5:]:  # Show last 5
        st.markdown(f"""
        <div style="
            color: {COLORS['text_secondary']};
            font-size: 12px;
            padding: 4px 0;
            border-bottom: 1px solid {COLORS['border']};
        ">
            <strong>{alert['symbol']}</strong> - {ALERT_TYPE_LABELS.get(alert['type'], alert['type'])}: {alert['value']}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Clear All", key="clear_triggered", use_container_width=True):
        clear_triggered_alerts()
        st.rerun()


def render_alerts_panel():
    """Render the complete alerts panel."""
    init_alerts_state()
    
    st.markdown(f"""
    <div style="
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 16px;
    ">
    """, unsafe_allow_html=True)
    
    # Show triggered alerts first
    render_triggered_alerts()
    
    # Create alert form
    render_create_alert_form(
        default_symbol=st.session_state.get('selected_symbol', 'SPY')
    )
    
    # Active alerts list
    render_alerts_list()
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_alerts_sidebar():
    """Render compact alerts section for sidebar."""
    init_alerts_state()
    
    active_count = len(get_active_alerts())
    triggered_count = len(get_triggered_alerts())
    
    # Header
    st.markdown(f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    ">
        <span style="color: {COLORS['text_primary']}; font-weight: 600; font-size: 13px;">
            🔔 Alerts
        </span>
        <span style="
            background: {COLORS['accent_orange'] if active_count > 0 else COLORS['bg_hover']};
            color: {'#000' if active_count > 0 else COLORS['text_muted']};
            border-radius: 10px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: 600;
        ">{active_count}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Triggered notification
    if triggered_count > 0:
        st.markdown(f"""
        <div style="
            background: {COLORS['accent_red']}20;
            border: 1px solid {COLORS['accent_red']};
            border-radius: 4px;
            padding: 6px 10px;
            margin-bottom: 8px;
            font-size: 12px;
            color: {COLORS['accent_red']};
        ">🚨 {triggered_count} alert(s) triggered!</div>
        """, unsafe_allow_html=True)
    
    # Quick add
    with st.expander("➕ Quick Alert", expanded=False):
        symbol = st.text_input("Symbol", value=st.session_state.get('selected_symbol', 'SPY'), key="sidebar_alert_sym")
        
        alert_type = st.selectbox(
            "Type",
            options=[AlertType.PRICE_ABOVE, AlertType.PRICE_BELOW],
            format_func=lambda x: ALERT_TYPE_LABELS[x],
            key="sidebar_alert_type"
        )
        
        value = st.number_input("Price", min_value=0.0, value=100.0, step=1.0, key="sidebar_alert_val")
        
        if st.button("Create", key="sidebar_create_alert", use_container_width=True):
            add_alert(symbol, alert_type, value)
            st.success(f"Alert created!")
            st.rerun()

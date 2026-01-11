"""
Workspace Manager Component
Save, load, and manage custom dashboard layouts
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.theme import COLORS


# =============================================================================
# WORKSPACE DATA STRUCTURE
# =============================================================================

DEFAULT_WORKSPACE = {
    'name': 'Default',
    'created_at': None,
    'settings': {
        'selected_symbol': 'SPY',
        'timeframe': '1y',
        'chart_height': 550,
        'show_volume': True,
        'show_ma': True,
        'ma_periods': [20, 50],
        'active_overlay_indicators': ['SMA'],
        'active_subchart_indicators': ['RSI'],
        'watchlist_symbols': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']
    }
}

WORKSPACE_FILE = Path(__file__).parent.parent.parent.parent / 'data' / 'workspaces.json'


# =============================================================================
# STORAGE FUNCTIONS
# =============================================================================

def ensure_workspace_dir():
    """Ensure data directory exists."""
    WORKSPACE_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_workspaces() -> Dict[str, Dict]:
    """Load saved workspaces from file."""
    ensure_workspace_dir()
    
    if WORKSPACE_FILE.exists():
        try:
            with open(WORKSPACE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {'Default': DEFAULT_WORKSPACE}
    
    return {'Default': DEFAULT_WORKSPACE}


def save_workspaces(workspaces: Dict[str, Dict]):
    """Save workspaces to file."""
    ensure_workspace_dir()
    
    with open(WORKSPACE_FILE, 'w') as f:
        json.dump(workspaces, f, indent=2, default=str)


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def init_workspace_state():
    """Initialize workspace state."""
    if 'workspaces' not in st.session_state:
        st.session_state.workspaces = load_workspaces()
    
    if 'current_workspace' not in st.session_state:
        st.session_state.current_workspace = 'Default'


def get_current_workspace() -> Dict:
    """Get the current workspace settings."""
    init_workspace_state()
    name = st.session_state.current_workspace
    return st.session_state.workspaces.get(name, DEFAULT_WORKSPACE)


def get_workspace_names() -> List[str]:
    """Get list of workspace names."""
    init_workspace_state()
    return list(st.session_state.workspaces.keys())


def capture_current_settings() -> Dict:
    """Capture current dashboard settings."""
    return {
        'selected_symbol': st.session_state.get('selected_symbol', 'SPY'),
        'timeframe': st.session_state.get('selected_timeframe', '1y'),
        'chart_height': st.session_state.get('chart_height', 550),
        'show_volume': st.session_state.get('show_volume', True),
        'show_ma': st.session_state.get('show_ma', True),
        'ma_periods': st.session_state.get('ma_periods', [20, 50]),
        'active_overlay_indicators': st.session_state.get('active_overlay_indicators', ['SMA']),
        'active_subchart_indicators': st.session_state.get('active_subchart_indicators', ['RSI']),
        'watchlist_symbols': st.session_state.get('watchlist_symbols', [])
    }


def apply_workspace_settings(settings: Dict):
    """Apply workspace settings to session state."""
    for key, value in settings.items():
        st.session_state[key] = value


def save_workspace(name: str) -> bool:
    """
    Save current settings as a workspace.
    
    Parameters
    ----------
    name : str
        Workspace name
    
    Returns
    -------
    bool
        Success status
    """
    init_workspace_state()
    
    workspace = {
        'name': name,
        'created_at': datetime.now().isoformat(),
        'settings': capture_current_settings()
    }
    
    st.session_state.workspaces[name] = workspace
    save_workspaces(st.session_state.workspaces)
    st.session_state.current_workspace = name
    
    return True


def load_workspace(name: str) -> bool:
    """
    Load a saved workspace.
    
    Parameters
    ----------
    name : str
        Workspace name
    
    Returns
    -------
    bool
        Success status
    """
    init_workspace_state()
    
    if name not in st.session_state.workspaces:
        return False
    
    workspace = st.session_state.workspaces[name]
    apply_workspace_settings(workspace['settings'])
    st.session_state.current_workspace = name
    
    return True


def delete_workspace(name: str) -> bool:
    """Delete a workspace."""
    init_workspace_state()
    
    if name == 'Default':
        return False  # Can't delete default
    
    if name in st.session_state.workspaces:
        del st.session_state.workspaces[name]
        save_workspaces(st.session_state.workspaces)
        
        if st.session_state.current_workspace == name:
            st.session_state.current_workspace = 'Default'
        
        return True
    
    return False


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_workspace_selector():
    """Render workspace dropdown selector."""
    init_workspace_state()
    
    names = get_workspace_names()
    current = st.session_state.current_workspace
    
    selected = st.selectbox(
        "Workspace",
        options=names,
        index=names.index(current) if current in names else 0,
        key="workspace_selector",
        label_visibility="collapsed"
    )
    
    if selected != current:
        load_workspace(selected)
        st.rerun()


def render_workspace_manager():
    """Render full workspace manager panel."""
    init_workspace_state()
    
    st.markdown(f"""
    <div style="color: {COLORS['text_primary']}; font-weight: 600; font-size: 14px; margin-bottom: 12px;">
        💾 Workspaces
    </div>
    """, unsafe_allow_html=True)
    
    # Current workspace indicator
    st.markdown(f"""
    <div style="
        background: {COLORS['accent_orange']}20;
        border: 1px solid {COLORS['accent_orange']};
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 12px;
    ">
        <small style="color: {COLORS['text_muted']};">Current:</small>
        <span style="color: {COLORS['accent_orange']}; font-weight: 600; margin-left: 8px;">
            {st.session_state.current_workspace}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Workspace selector
    render_workspace_selector()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Save new workspace
    with st.expander("💾 Save Current Layout", expanded=False):
        new_name = st.text_input(
            "Workspace Name",
            placeholder="My Trading Setup",
            key="new_workspace_name"
        )
        
        if st.button("Save Workspace", key="save_workspace_btn", use_container_width=True, type="primary"):
            if new_name:
                save_workspace(new_name)
                st.success(f"✅ Saved '{new_name}'")
                st.rerun()
            else:
                st.error("Enter a name")
    
    # List workspaces
    st.markdown(f"<small style='color:{COLORS['text_muted']}'>Saved Workspaces</small>", unsafe_allow_html=True)
    
    for name in get_workspace_names():
        is_current = name == st.session_state.current_workspace
        workspace = st.session_state.workspaces[name]
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            marker = "🔵 " if is_current else ""
            st.markdown(f"""
            <div style="
                padding: 6px 0;
                color: {COLORS['accent_orange'] if is_current else COLORS['text_secondary']};
                font-weight: {'600' if is_current else '400'};
                font-size: 13px;
            ">{marker}{name}</div>
            """, unsafe_allow_html=True)
        
        with col2:
            if name != 'Default' and not is_current:
                if st.button("🗑️", key=f"del_ws_{name}", help="Delete"):
                    delete_workspace(name)
                    st.rerun()


def render_workspace_quick_save():
    """Render quick save button for toolbar."""
    init_workspace_state()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <small style="color: {COLORS['text_muted']};">
            Workspace: <strong>{st.session_state.current_workspace}</strong>
        </small>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("💾", key="quick_save", help="Quick save current workspace"):
            save_workspace(st.session_state.current_workspace)
            st.success("Saved!")

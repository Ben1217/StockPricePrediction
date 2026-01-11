"""
Mobile Responsive Component
Responsive CSS and layout adjustments for mobile/tablet
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.theme import COLORS


def get_responsive_css() -> str:
    """
    Get responsive CSS for mobile and tablet layouts.
    
    Returns
    -------
    str
        CSS string
    """
    return f"""
    <style>
    /* =========================================
       MOBILE RESPONSIVE STYLES
       ========================================= */
    
    /* Mobile First - Base styles for small screens */
    @media (max-width: 768px) {{
        /* Reduce padding */
        .main .block-container {{
            padding: 0.5rem 1rem !important;
        }}
        
        /* Sidebar collapses by default on mobile */
        section[data-testid="stSidebar"] {{
            width: 100% !important;
            min-width: 100% !important;
        }}
        
        section[data-testid="stSidebar"] > div {{
            width: 100% !important;
        }}
        
        /* Stack columns vertically */
        .stColumn {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}
        
        /* Reduce font sizes */
        h1 {{
            font-size: 1.5rem !important;
        }}
        
        h2 {{
            font-size: 1.25rem !important;
        }}
        
        h3 {{
            font-size: 1.1rem !important;
        }}
        
        /* Metrics - smaller on mobile */
        div[data-testid="stMetricValue"] {{
            font-size: 1.2rem !important;
        }}
        
        div[data-testid="stMetricLabel"] {{
            font-size: 0.75rem !important;
        }}
        
        /* Tabs - scrollable */
        .stTabs [data-baseweb="tab-list"] {{
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            -webkit-overflow-scrolling: touch;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            flex-shrink: 0 !important;
            padding: 8px 12px !important;
            font-size: 12px !important;
        }}
        
        /* Charts - full width */
        .js-plotly-plot {{
            width: 100% !important;
        }}
        
        /* Status bar - simplified */
        .status-bar {{
            flex-wrap: wrap !important;
            gap: 8px !important;
            padding: 8px !important;
        }}
        
        .status-bar .index-quote {{
            display: none !important;
        }}
        
        /* Buttons - larger touch targets */
        .stButton button {{
            min-height: 44px !important;
            font-size: 14px !important;
        }}
        
        /* Input fields - larger */
        .stTextInput input {{
            min-height: 44px !important;
            font-size: 16px !important; /* Prevents zoom on iOS */
        }}
        
        /* Select boxes */
        .stSelectbox {{
            font-size: 16px !important;
        }}
        
        /* Expanders - full width */
        .streamlit-expanderHeader {{
            font-size: 14px !important;
        }}
        
        /* Hide keyboard shortcuts hint on mobile */
        .shortcuts-hint {{
            display: none !important;
        }}
        
        /* Watchlist - more compact */
        .watchlist-item {{
            padding: 8px !important;
        }}
    }}
    
    /* Tablet - Medium screens */
    @media (min-width: 769px) and (max-width: 1024px) {{
        .main .block-container {{
            padding: 1rem !important;
        }}
        
        /* 2-column layout still works */
        .stColumn {{
            min-width: 48% !important;
        }}
        
        /* Slightly smaller fonts */
        h1 {{
            font-size: 1.75rem !important;
        }}
        
        div[data-testid="stMetricValue"] {{
            font-size: 1.4rem !important;
        }}
    }}
    
    /* Desktop - Large screens */
    @media (min-width: 1025px) {{
        .main .block-container {{
            max-width: 1600px !important;
        }}
    }}
    
    /* Ultra-wide - Very large screens */
    @media (min-width: 1921px) {{
        .main .block-container {{
            max-width: 2000px !important;
        }}
        
        /* Larger fonts for readability */
        body {{
            font-size: 15px !important;
        }}
    }}
    
    /* =========================================
       TOUCH-FRIENDLY ENHANCEMENTS
       ========================================= */
    
    /* Larger clickable areas */
    @media (pointer: coarse) {{
        .stButton button,
        .stSelectbox,
        .streamlit-expanderHeader {{
            min-height: 48px !important;
        }}
        
        /* More space between interactive elements */
        .stCheckbox {{
            padding: 8px 0 !important;
        }}
        
        /* Larger radio buttons */
        .stRadio > div {{
            gap: 12px !important;
        }}
    }}
    
    /* =========================================
       PRINT STYLES
       ========================================= */
    
    @media print {{
        /* Hide sidebar */
        section[data-testid="stSidebar"] {{
            display: none !important;
        }}
        
        /* Full width content */
        .main .block-container {{
            max-width: 100% !important;
            padding: 0 !important;
        }}
        
        /* Hide interactive elements */
        .stButton,
        .stTextInput,
        .stSelectbox,
        .streamlit-expanderHeader {{
            display: none !important;
        }}
        
        /* Charts at full width */
        .js-plotly-plot {{
            width: 100% !important;
            page-break-inside: avoid;
        }}
        
        /* Black text for printing */
        * {{
            color: #000 !important;
            background: #fff !important;
        }}
    }}
    
    /* =========================================
       DARK MODE SAFE
       ========================================= */
    
    @media (prefers-color-scheme: dark) {{
        /* Already dark theme, no changes needed */
    }}
    
    /* =========================================
       REDUCED MOTION
       ========================================= */
    
    @media (prefers-reduced-motion: reduce) {{
        * {{
            animation: none !important;
            transition: none !important;
        }}
    }}
    </style>
    """


def inject_responsive_css():
    """Inject responsive CSS into the page."""
    st.markdown(get_responsive_css(), unsafe_allow_html=True)


def get_layout_mode() -> str:
    """
    Detect current layout mode based on viewport.
    Uses JavaScript to detect and store in session state.
    
    Returns
    -------
    str
        'mobile', 'tablet', or 'desktop'
    """
    # Default to desktop
    return st.session_state.get('layout_mode', 'desktop')


def render_layout_toggle():
    """Render a layout mode toggle for testing."""
    st.markdown(f"""
    <div style="
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 8px;
        margin-bottom: 12px;
    ">
        <small style="color: {COLORS['text_muted']};">Layout Mode</small>
    </div>
    """, unsafe_allow_html=True)
    
    mode = st.radio(
        "Layout",
        options=['Auto', 'Desktop', 'Tablet', 'Mobile'],
        horizontal=True,
        key="layout_mode_selector",
        label_visibility="collapsed"
    )
    
    return mode.lower() if mode != 'Auto' else None

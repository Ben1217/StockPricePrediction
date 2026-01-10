"""
Dashboard Theme Configuration
Dark theme colors and styling for professional trading interface
"""

# =============================================================================
# COLOR PALETTE
# =============================================================================

COLORS = {
    # Backgrounds
    'bg_primary': '#1a1a1a',      # Main background
    'bg_secondary': '#2d2d2d',    # Cards, panels
    'bg_sidebar': '#1e1e1e',      # Sidebar
    'bg_hover': '#3d3d3d',        # Hover state
    
    # Text
    'text_primary': '#ffffff',    # Headings
    'text_secondary': '#b0b0b0',  # Body text
    'text_muted': '#808080',      # Labels
    
    # Accents
    'accent_orange': '#ff9800',   # Primary action
    'accent_orange_hover': '#ffb74d',
    'accent_green': '#4caf50',    # Positive/Bullish
    'accent_red': '#f44336',      # Negative/Bearish
    'accent_blue': '#2196f3',     # Info
    
    # Chart
    'candle_up': '#26a69a',       # Bullish candle (teal)
    'candle_down': '#ef5350',     # Bearish candle (red)
    'volume_up': '#26a69a',       # Volume bullish
    'volume_down': '#ef5350',     # Volume bearish
    'grid': '#333333',            # Grid lines
    'border': '#404040',          # Borders
}


# =============================================================================
# PLOTLY CHART TEMPLATE
# =============================================================================

PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': COLORS['bg_primary'],
        'plot_bgcolor': COLORS['bg_primary'],
        'font': {
            'color': COLORS['text_secondary'],
            'family': 'Inter, Roboto, sans-serif',
            'size': 12
        },
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'showgrid': True,
            'zeroline': False,
            'color': COLORS['text_muted'],
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'showgrid': True,
            'zeroline': False,
            'color': COLORS['text_muted'],
            'side': 'right'
        },
        'legend': {
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': COLORS['text_secondary']}
        }
    }
}


# =============================================================================
# CSS STYLES
# =============================================================================

def get_custom_css() -> str:
    """
    Get custom CSS for Streamlit dashboard.
    Applies dark theme and orange accent styling.
    """
    return f"""
    <style>
    /* =========================
       ROOT VARIABLES
       ========================= */
    :root {{
        --bg-primary: {COLORS['bg_primary']};
        --bg-secondary: {COLORS['bg_secondary']};
        --bg-sidebar: {COLORS['bg_sidebar']};
        --text-primary: {COLORS['text_primary']};
        --text-secondary: {COLORS['text_secondary']};
        --accent-orange: {COLORS['accent_orange']};
        --accent-green: {COLORS['accent_green']};
        --accent-red: {COLORS['accent_red']};
        --border-color: {COLORS['border']};
    }}

    /* =========================
       MAIN BACKGROUND
       ========================= */
    .stApp {{
        background-color: var(--bg-primary) !important;
    }}
    
    .main .block-container {{
        background-color: var(--bg-primary);
        padding: 1rem 2rem;
    }}

    /* =========================
       SIDEBAR
       ========================= */
    [data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color);
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: var(--text-secondary);
    }}
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: var(--text-primary) !important;
    }}

    /* =========================
       TYPOGRAPHY
       ========================= */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary) !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }}
    
    p, span, div {{
        color: var(--text-secondary);
    }}
    
    .stMarkdown {{
        color: var(--text-secondary);
    }}

    /* =========================
       TABS
       ========================= */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: var(--bg-secondary);
        padding: 8px;
        border-radius: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: var(--text-secondary);
        border-radius: 4px;
        padding: 8px 16px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: var(--accent-orange) !important;
        color: #000000 !important;
        font-weight: bold;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: var(--bg-hover);
    }}

    /* =========================
       METRIC CARDS
       ========================= */
    [data-testid="stMetric"] {{
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
    }}
    
    [data-testid="stMetric"] label {{
        color: var(--text-muted) !important;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: var(--text-primary) !important;
        font-family: 'Roboto Mono', monospace;
        font-size: 24px;
        font-weight: 700;
    }}
    
    [data-testid="stMetric"] [data-testid="stMetricDelta"] svg {{
        display: none;
    }}

    /* =========================
       BUTTONS
       ========================= */
    .stButton > button {{
        background-color: var(--accent-orange) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 4px;
        font-weight: bold;
        transition: all 0.2s ease;
    }}
    
    .stButton > button:hover {{
        background-color: {COLORS['accent_orange_hover']} !important;
        transform: scale(1.02);
    }}
    
    /* Secondary button style */
    .stButton > button[kind="secondary"] {{
        background-color: transparent !important;
        color: var(--accent-orange) !important;
        border: 1px solid var(--accent-orange) !important;
    }}

    /* =========================
       SELECTBOX / DROPDOWN
       ========================= */
    [data-testid="stSelectbox"] label {{
        color: var(--text-secondary) !important;
    }}
    
    [data-testid="stSelectbox"] > div > div {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }}

    /* =========================
       SCROLLBAR
       ========================= */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--bg-secondary);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--accent-orange);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['accent_orange_hover']};
    }}

    /* =========================
       EXPANDER
       ========================= */
    .streamlit-expanderHeader {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color);
        border-radius: 4px;
    }}
    
    .streamlit-expanderContent {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color);
        border-top: none;
    }}

    /* =========================
       DIVIDER
       ========================= */
    hr {{
        border-color: var(--border-color) !important;
    }}

    /* =========================
       CHART CONTAINER
       ========================= */
    [data-testid="stPlotlyChart"] {{
        background-color: var(--bg-primary);
        border-radius: 8px;
    }}

    /* =========================
       INFO/WARNING/ERROR
       ========================= */
    .stAlert {{
        background-color: var(--bg-secondary);
        border-radius: 4px;
    }}

    /* =========================
       DATAFRAME
       ========================= */
    [data-testid="stDataFrame"] {{
        background-color: var(--bg-secondary);
    }}

    </style>
    """


def get_fonts_css() -> str:
    """
    Google Fonts import for Inter and Roboto Mono.
    """
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre, .stCode {
        font-family: 'Roboto Mono', monospace !important;
    }
    </style>
    """

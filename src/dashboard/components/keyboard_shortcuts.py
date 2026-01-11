"""
Keyboard Shortcuts Component
Global keyboard shortcuts and quick search modal
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.theme import COLORS


# =============================================================================
# KEYBOARD SHORTCUTS CSS & JS
# =============================================================================

def get_keyboard_shortcuts_css() -> str:
    """Get CSS for keyboard shortcuts modal."""
    return f"""
    <style>
    /* Quick Search Modal */
    .quick-search-overlay {{
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: 9999;
        justify-content: center;
        align-items: flex-start;
        padding-top: 100px;
    }}
    
    .quick-search-overlay.active {{
        display: flex;
    }}
    
    .quick-search-box {{
        background: {COLORS['bg_secondary']};
        border: 2px solid {COLORS['accent_orange']};
        border-radius: 12px;
        width: 500px;
        max-width: 90%;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    }}
    
    .quick-search-input {{
        width: 100%;
        padding: 16px 20px;
        font-size: 18px;
        border: none;
        border-radius: 10px;
        background: {COLORS['bg_primary']};
        color: {COLORS['text_primary']};
        outline: none;
    }}
    
    .quick-search-input::placeholder {{
        color: {COLORS['text_muted']};
    }}
    
    .quick-search-results {{
        max-height: 300px;
        overflow-y: auto;
    }}
    
    .quick-search-item {{
        padding: 12px 20px;
        cursor: pointer;
        border-top: 1px solid {COLORS['border']};
        color: {COLORS['text_secondary']};
        transition: background 0.1s;
    }}
    
    .quick-search-item:hover {{
        background: {COLORS['bg_hover']};
    }}
    
    .quick-search-item .symbol {{
        color: {COLORS['accent_orange']};
        font-weight: 600;
        margin-right: 8px;
    }}
    
    /* Keyboard shortcut hints */
    .kbd {{
        display: inline-block;
        background: {COLORS['bg_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 11px;
        font-family: 'Roboto Mono', monospace;
        color: {COLORS['text_muted']};
    }}
    
    /* Shortcut help panel */
    .shortcuts-help {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 16px;
        z-index: 1000;
        display: none;
        min-width: 250px;
    }}
    
    .shortcuts-help.visible {{
        display: block;
    }}
    
    .shortcut-row {{
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid {COLORS['border']};
    }}
    
    .shortcut-row:last-child {{
        border-bottom: none;
    }}
    </style>
    """


def get_keyboard_shortcuts_js() -> str:
    """Get JavaScript for keyboard shortcuts handling."""
    return """
    <script>
    (function() {
        // Avoid duplicate initialization
        if (window.shortcutsInitialized) return;
        window.shortcutsInitialized = true;
        
        // Track current tab index
        let currentTab = 0;
        
        document.addEventListener('keydown', function(e) {
            // Skip if typing in input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                // Escape closes inputs
                if (e.key === 'Escape') {
                    e.target.blur();
                }
                return;
            }
            
            // Ctrl+K - Quick search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                toggleQuickSearch();
                return;
            }
            
            // ? - Show shortcuts help
            if (e.key === '?') {
                e.preventDefault();
                toggleShortcutsHelp();
                return;
            }
            
            // Number keys 1-5 for tabs
            if (!e.ctrlKey && !e.altKey && !e.metaKey) {
                const tabNum = parseInt(e.key);
                if (tabNum >= 1 && tabNum <= 5) {
                    e.preventDefault();
                    switchToTab(tabNum - 1);
                    return;
                }
            }
            
            // Escape - Close modals
            if (e.key === 'Escape') {
                closeQuickSearch();
                closeShortcutsHelp();
                return;
            }
            
            // R - Refresh data
            if (e.key === 'r' || e.key === 'R') {
                // Will be handled by Streamlit
                return;
            }
        });
        
        function switchToTab(index) {
            const tabs = document.querySelectorAll('[data-baseweb="tab"]');
            if (tabs[index]) {
                tabs[index].click();
                currentTab = index;
            }
        }
        
        function toggleQuickSearch() {
            const overlay = document.getElementById('quick-search-overlay');
            if (overlay) {
                overlay.classList.toggle('active');
                if (overlay.classList.contains('active')) {
                    const input = overlay.querySelector('input');
                    if (input) input.focus();
                }
            }
        }
        
        function closeQuickSearch() {
            const overlay = document.getElementById('quick-search-overlay');
            if (overlay) {
                overlay.classList.remove('active');
            }
        }
        
        function toggleShortcutsHelp() {
            const help = document.getElementById('shortcuts-help');
            if (help) {
                help.classList.toggle('visible');
            }
        }
        
        function closeShortcutsHelp() {
            const help = document.getElementById('shortcuts-help');
            if (help) {
                help.classList.remove('visible');
            }
        }
        
        // Make functions globally available
        window.toggleQuickSearch = toggleQuickSearch;
        window.closeQuickSearch = closeQuickSearch;
        window.toggleShortcutsHelp = toggleShortcutsHelp;
    })();
    </script>
    """


def get_quick_search_html(symbols: list = None) -> str:
    """Get HTML for quick search modal."""
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD']
    
    results_html = ""
    for sym in symbols[:10]:
        results_html += f"""
        <div class="quick-search-item" onclick="window.location.href='?symbol={sym}'">
            <span class="symbol">{sym}</span>
            <span>Load chart</span>
        </div>
        """
    
    return f"""
    <div id="quick-search-overlay" class="quick-search-overlay" onclick="if(event.target.id === 'quick-search-overlay') closeQuickSearch()">
        <div class="quick-search-box">
            <input 
                type="text" 
                class="quick-search-input" 
                placeholder="Search symbol... (Press Esc to close)"
                autofocus
            >
            <div class="quick-search-results">
                <div style="padding: 8px 20px; color: {COLORS['text_muted']}; font-size: 11px; text-transform: uppercase;">
                    Quick Access
                </div>
                {results_html}
            </div>
        </div>
    </div>
    """


def get_shortcuts_help_html() -> str:
    """Get HTML for shortcuts help panel."""
    shortcuts = [
        ('Ctrl+K', 'Quick search'),
        ('1-5', 'Switch tabs'),
        ('Esc', 'Close modals'),
        ('?', 'Show shortcuts'),
    ]
    
    rows = ""
    for key, description in shortcuts:
        rows += f"""
        <div class="shortcut-row">
            <span class="kbd">{key}</span>
            <span style="color: {COLORS['text_secondary']}; font-size: 12px;">{description}</span>
        </div>
        """
    
    return f"""
    <div id="shortcuts-help" class="shortcuts-help">
        <div style="color: {COLORS['text_primary']}; font-weight: 600; margin-bottom: 10px; font-size: 13px;">
            ⌨️ Keyboard Shortcuts
        </div>
        {rows}
        <div style="color: {COLORS['text_muted']}; font-size: 10px; margin-top: 8px; text-align: center;">
            Press ? to toggle
        </div>
    </div>
    """


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def inject_keyboard_shortcuts(watchlist_symbols: list = None):
    """
    Inject keyboard shortcuts CSS and JavaScript.
    
    Parameters
    ----------
    watchlist_symbols : list, optional
        Symbols to show in quick search
    """
    # Only inject once per session to prevent duplicates
    if st.session_state.get('_shortcuts_injected'):
        return
    
    st.session_state['_shortcuts_injected'] = True
    
    st.markdown(get_keyboard_shortcuts_css(), unsafe_allow_html=True)
    st.markdown(get_keyboard_shortcuts_js(), unsafe_allow_html=True)
    st.markdown(get_quick_search_html(watchlist_symbols), unsafe_allow_html=True)
    st.markdown(get_shortcuts_help_html(), unsafe_allow_html=True)


def render_shortcuts_hint():
    """Render a small hint about keyboard shortcuts."""
    st.markdown(f"""
    <div style="
        position: fixed;
        bottom: 10px;
        right: 10px;
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 10px;
        color: {COLORS['text_muted']};
        opacity: 0.7;
        z-index: 100;
    ">
        Press <span class="kbd">?</span> for shortcuts
    </div>
    """, unsafe_allow_html=True)

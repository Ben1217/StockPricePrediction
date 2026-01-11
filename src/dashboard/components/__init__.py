"""
Dashboard Components Package
"""

# Phase 1 Components
from src.dashboard.components.watchlist import (
    render_watchlist_panel,
    render_mini_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    init_watchlist_state
)

from src.dashboard.components.status_bar import (
    render_status_bar,
    get_market_status,
    get_major_indices
)

from src.dashboard.components.alerts import (
    init_alerts_state,
    add_alert,
    remove_alert,
    check_alerts,
    get_active_alerts,
    get_triggered_alerts,
    render_alerts_panel,
    render_alerts_sidebar,
    AlertType
)

from src.dashboard.components.drawing_tools import (
    init_drawings_state,
    add_drawing,
    remove_drawing,
    get_drawings_for_symbol,
    apply_drawings_to_chart,
    render_drawing_toolbar,
    render_drawings_list,
    DrawingType
)

from src.dashboard.components.keyboard_shortcuts import (
    inject_keyboard_shortcuts,
    render_shortcuts_hint
)

# Phase 2 Components
from src.dashboard.components.advanced_indicators import (
    init_indicators_state,
    get_active_overlays,
    get_active_subcharts,
    add_indicator,
    remove_indicator,
    create_chart_with_indicators,
    render_indicator_selector,
    render_indicator_toolbar
)

from src.dashboard.components.timeframe import (
    init_timeframe_state,
    get_selected_timeframe,
    set_timeframe,
    get_data_for_timeframe,
    render_timeframe_switcher,
    render_timeframe_bar,
    TIMEFRAMES
)

from src.dashboard.components.quick_search import (
    init_search_state,
    open_search,
    close_search,
    search_symbols,
    render_search_modal,
    render_search_trigger,
    render_search_bar
)

from src.dashboard.components.portfolio_analytics import (
    calculate_metrics,
    create_performance_chart,
    create_drawdown_chart,
    create_returns_distribution,
    create_monthly_heatmap,
    render_portfolio_analytics,
    render_metrics_cards,
    render_risk_metrics
)

# Phase 3 Components
from src.dashboard.components.workspace_manager import (
    init_workspace_state,
    get_current_workspace,
    save_workspace,
    load_workspace,
    delete_workspace,
    render_workspace_manager,
    render_workspace_selector
)

from src.dashboard.components.export_manager import (
    export_to_csv,
    export_to_excel,
    generate_report_data,
    render_export_panel,
    render_quick_export
)

from src.dashboard.components.responsive import (
    inject_responsive_css,
    get_layout_mode
)

__all__ = [
    # Phase 1
    'render_watchlist_panel', 'render_mini_watchlist', 'add_to_watchlist',
    'remove_from_watchlist', 'init_watchlist_state',
    'render_status_bar', 'get_market_status', 'get_major_indices',
    'init_alerts_state', 'add_alert', 'remove_alert', 'check_alerts',
    'get_active_alerts', 'get_triggered_alerts', 'render_alerts_panel',
    'render_alerts_sidebar', 'AlertType',
    'init_drawings_state', 'add_drawing', 'remove_drawing',
    'get_drawings_for_symbol', 'apply_drawings_to_chart',
    'render_drawing_toolbar', 'render_drawings_list', 'DrawingType',
    'inject_keyboard_shortcuts', 'render_shortcuts_hint',
    # Phase 2
    'init_indicators_state', 'get_active_overlays', 'get_active_subcharts',
    'add_indicator', 'remove_indicator', 'create_chart_with_indicators',
    'render_indicator_selector', 'render_indicator_toolbar',
    'init_timeframe_state', 'get_selected_timeframe', 'set_timeframe',
    'get_data_for_timeframe', 'render_timeframe_switcher',
    'render_timeframe_bar', 'TIMEFRAMES',

    'init_search_state', 'open_search', 'close_search', 'search_symbols',
    'render_search_modal', 'render_search_trigger', 'render_search_bar',
    'calculate_metrics', 'create_performance_chart', 'create_drawdown_chart',
    'create_returns_distribution', 'create_monthly_heatmap',
    'render_portfolio_analytics', 'render_metrics_cards', 'render_risk_metrics',
    # Phase 3
    'init_workspace_state', 'get_current_workspace', 'save_workspace',
    'load_workspace', 'delete_workspace', 'render_workspace_manager',
    'render_workspace_selector',
    'export_to_csv', 'export_to_excel', 'generate_report_data',
    'render_export_panel', 'render_quick_export',
    'inject_responsive_css', 'get_layout_mode'
]

"""
Export & Reporting Component
Export data, charts, and reports
"""

import streamlit as st
import pandas as pd
import io
from datetime import datetime
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dashboard.theme import COLORS


# =============================================================================
# DATA EXPORT FUNCTIONS
# =============================================================================

def export_to_csv(df: pd.DataFrame, filename: str = None) -> bytes:
    """
    Export DataFrame to CSV bytes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to export
    filename : str, optional
        Suggested filename
    
    Returns
    -------
    bytes
        CSV file content
    """
    return df.to_csv().encode('utf-8')


def export_to_excel(df: pd.DataFrame) -> bytes:
    """Export DataFrame to Excel bytes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='Data')
    return output.getvalue()


def generate_report_data(df: pd.DataFrame, symbol: str) -> dict:
    """
    Generate report data for a stock.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    symbol : str
        Stock symbol
    
    Returns
    -------
    dict
        Report data
    """
    returns = df['Close'].pct_change().dropna()
    
    report = {
        'symbol': symbol,
        'generated_at': datetime.now().isoformat(),
        'period': {
            'start': str(df.index[0].date()) if hasattr(df.index[0], 'date') else str(df.index[0]),
            'end': str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1]),
            'days': len(df)
        },
        'price': {
            'current': float(df['Close'].iloc[-1]),
            'open': float(df['Open'].iloc[0]),
            'high': float(df['High'].max()),
            'low': float(df['Low'].min()),
            'change': float((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100)
        },
        'performance': {
            'total_return': float((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100),
            'annual_return': float(returns.mean() * 252 * 100),
            'volatility': float(returns.std() * (252 ** 0.5) * 100),
            'sharpe_ratio': float((returns.mean() * 252) / (returns.std() * (252 ** 0.5))) if returns.std() > 0 else 0
        },
        'risk': {
            'max_drawdown': float(((df['Close'] / df['Close'].cummax()) - 1).min() * 100),
            'var_95': float(returns.quantile(0.05) * 100),
            'downside_std': float(returns[returns < 0].std() * (252 ** 0.5) * 100) if len(returns[returns < 0]) > 0 else 0
        },
        'volume': {
            'average': float(df['Volume'].mean()),
            'total': float(df['Volume'].sum())
        }
    }
    
    return report


def generate_report_text(report: dict) -> str:
    """Generate text report from report data."""
    lines = [
        f"=" * 60,
        f"STOCK ANALYSIS REPORT",
        f"=" * 60,
        f"",
        f"Symbol: {report['symbol']}",
        f"Generated: {report['generated_at']}",
        f"",
        f"PERIOD",
        f"-" * 40,
        f"Start Date: {report['period']['start']}",
        f"End Date: {report['period']['end']}",
        f"Trading Days: {report['period']['days']}",
        f"",
        f"PRICE SUMMARY",
        f"-" * 40,
        f"Current Price: ${report['price']['current']:.2f}",
        f"Period High: ${report['price']['high']:.2f}",
        f"Period Low: ${report['price']['low']:.2f}",
        f"Total Change: {report['price']['change']:+.2f}%",
        f"",
        f"PERFORMANCE METRICS",
        f"-" * 40,
        f"Total Return: {report['performance']['total_return']:+.2f}%",
        f"Annualized Return: {report['performance']['annual_return']:+.2f}%",
        f"Annualized Volatility: {report['performance']['volatility']:.2f}%",
        f"Sharpe Ratio: {report['performance']['sharpe_ratio']:.2f}",
        f"",
        f"RISK METRICS",
        f"-" * 40,
        f"Max Drawdown: {report['risk']['max_drawdown']:.2f}%",
        f"Value at Risk (95%): {report['risk']['var_95']:.2f}%",
        f"Downside Volatility: {report['risk']['downside_std']:.2f}%",
        f"",
        f"VOLUME",
        f"-" * 40,
        f"Average Daily Volume: {report['volume']['average']:,.0f}",
        f"Total Volume: {report['volume']['total']:,.0f}",
        f"",
        f"=" * 60,
    ]
    
    return "\n".join(lines)


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_export_panel(df: pd.DataFrame, symbol: str = "Stock"):
    """
    Render the export panel.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to export
    symbol : str
        Stock symbol
    """
    st.markdown(f"""
    <div style="color: {COLORS['text_primary']}; font-weight: 600; font-size: 14px; margin-bottom: 12px;">
        📥 Export Data
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Export
        csv_data = export_to_csv(df)
        filename = f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.csv"
        
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel Export
        try:
            excel_data = export_to_excel(df)
            filename = f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.xlsx"
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.button("📊 Excel (install openpyxl)", disabled=True, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Report Export
    st.markdown(f"""
    <div style="color: {COLORS['text_primary']}; font-weight: 600; font-size: 14px; margin-bottom: 12px;">
        📋 Export Report
    </div>
    """, unsafe_allow_html=True)
    
    report = generate_report_data(df, symbol)
    report_text = generate_report_text(report)
    
    # Text Report
    st.download_button(
        label="📝 Download Text Report",
        data=report_text,
        file_name=f"{symbol}_report_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    # JSON Report
    import json
    report_json = json.dumps(report, indent=2)
    
    st.download_button(
        label="📦 Download JSON Report",
        data=report_json,
        file_name=f"{symbol}_report_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
        use_container_width=True
    )
    
    # Preview
    with st.expander("👁️ Preview Report", expanded=False):
        st.code(report_text, language=None)


def render_quick_export(df: pd.DataFrame, symbol: str = "Stock"):
    """Render quick export buttons."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = export_to_csv(df)
        st.download_button(
            "📄 CSV",
            data=csv_data,
            file_name=f"{symbol}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        report = generate_report_data(df, symbol)
        report_text = generate_report_text(report)
        st.download_button(
            "📝 Report",
            data=report_text,
            file_name=f"{symbol}_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        import json
        report = generate_report_data(df, symbol)
        st.download_button(
            "📦 JSON",
            data=json.dumps(report, indent=2),
            file_name=f"{symbol}.json",
            mime="application/json",
            use_container_width=True
        )

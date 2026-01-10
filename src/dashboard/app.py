"""
Stock Prediction Dashboard
Main Streamlit application with Dark Theme
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.data_loader import download_stock_data
from src.features.technical_indicators import add_all_technical_indicators
from src.portfolio.performance_metrics import calculate_portfolio_metrics
from src.dashboard.heatmap import render_heatmap_page
from src.dashboard.signal_display import render_signals_page
from src.dashboard.theme import COLORS, get_custom_css, get_fonts_css
from src.dashboard.charts import create_candlestick_chart, add_moving_averages


# Page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Dark Theme CSS
st.markdown(get_fonts_css(), unsafe_allow_html=True)
st.markdown(get_custom_css(), unsafe_allow_html=True)


def main():
    """Main dashboard application"""

    # Header with dark theme styling
    st.markdown(f'''
    <h1 style="
        font-size: 2rem;
        font-weight: bold;
        color: {COLORS['accent_orange']};
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
    ">📈 Stock Prediction Dashboard</h1>
    ''', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown(f'''
    <div style="text-align: center; padding: 10px 0; border-bottom: 1px solid {COLORS['border']}; margin-bottom: 15px;">
        <h2 style="color: {COLORS['accent_orange']}; margin: 0;">⚙️ Settings</h2>
    </div>
    ''', unsafe_allow_html=True)

    # Stock selection
    ticker = st.sidebar.text_input("Stock Symbol", value="SPY", max_chars=10)

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start = st.date_input("Start Date", value=start_date)
    with col2:
        end = st.date_input("End Date", value=end_date)

    # Load data button
    if st.sidebar.button("🔄 Load Data", type="primary"):
        with st.spinner(f"Downloading {ticker} data..."):
            data = download_stock_data(ticker, str(start), str(end))

            if data is not None and not data.empty:
                st.session_state['data'] = data
                st.session_state['ticker'] = ticker

                # Add technical indicators
                data_with_indicators = add_all_technical_indicators(data)
                st.session_state['data_with_indicators'] = data_with_indicators

                st.sidebar.success(f"✅ Loaded {len(data)} days of data")
            else:
                st.sidebar.error("Failed to load data")

    # Main content
    if 'data' in st.session_state:
        data = st.session_state['data']
        ticker = st.session_state.get('ticker', 'Unknown')

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", "📈 Technical Analysis", "🤖 Predictions", 
            "💼 Portfolio", "🗺️ Market Heatmap", "🎯 Trading Signals"
        ])

        with tab1:
            render_overview(data, ticker)

        with tab2:
            render_technical_analysis()

        with tab3:
            render_predictions()

        with tab4:
            render_portfolio()

        with tab5:
            render_heatmap_page()

        with tab6:
            # Store data for signals page
            st.session_state['stock_data'] = data
            st.session_state['selected_symbol'] = ticker
            render_signals_page()

    else:
        st.info("👈 Enter a stock symbol and click 'Load Data' to get started")

        # Show sample
        st.markdown("---")
        st.subheader("📌 Sample Stocks")
        cols = st.columns(4)
        for i, (symbol, name) in enumerate([
            ("SPY", "S&P 500 ETF"),
            ("QQQ", "NASDAQ ETF"),
            ("AAPL", "Apple Inc"),
            ("MSFT", "Microsoft")
        ]):
            with cols[i]:
                st.metric(symbol, name)


def render_overview(data: pd.DataFrame, ticker: str):
    """Render overview tab"""
    st.subheader(f"📊 {ticker} Overview")

    # Key metrics
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
        st.metric("Close Price", f"${latest['Close']:.2f}", f"{change:+.2f}%")

    with col2:
        st.metric("Volume", f"{latest['Volume']:,.0f}")

    with col3:
        high_52w = data['High'].tail(252).max()
        st.metric("52W High", f"${high_52w:.2f}")

    with col4:
        low_52w = data['Low'].tail(252).min()
        st.metric("52W Low", f"${low_52w:.2f}")

    # Price chart
    st.subheader("📈 Price History")

    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    ))

    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        paper_bgcolor=COLORS['bg_primary'],
        plot_bgcolor=COLORS['bg_primary'],
        xaxis=dict(gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid'], side='right')
    )

    st.plotly_chart(fig, use_container_width=True)

    # Volume chart
    fig_vol = go.Figure()
    colors = ['red' if row['Close'] < row['Open'] else 'green'
              for _, row in data.iterrows()]
    fig_vol.add_trace(go.Bar(x=data.index, y=data['Volume'], marker_color=colors, name='Volume'))
    fig_vol.update_layout(height=200, template='plotly_dark', paper_bgcolor=COLORS['bg_primary'], plot_bgcolor=COLORS['bg_primary'])

    st.plotly_chart(fig_vol, use_container_width=True)


def render_technical_analysis():
    """Render technical analysis tab"""
    st.subheader("📈 Technical Analysis")

    if 'data_with_indicators' not in st.session_state:
        st.warning("Please load data first")
        return

    data = st.session_state['data_with_indicators']

    # Indicator selection
    indicator = st.selectbox("Select Indicator", [
        "Moving Averages", "RSI", "MACD", "Bollinger Bands"
    ])

    fig = go.Figure()

    if indicator == "Moving Averages":
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
        if 'SMA_20' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'))
        if 'SMA_50' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50'))

    elif indicator == "RSI":
        if 'RSI' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
            fig.add_hline(y=70, line_dash="dash", line_color="red")
            fig.add_hline(y=30, line_dash="dash", line_color="green")

    elif indicator == "MACD":
        if 'MACD' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'))
        if 'MACD_Signal' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal'))

    elif indicator == "Bollinger Bands":
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
        if 'BB_High' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], name='Upper Band',
                                    line=dict(dash='dash')))
        if 'BB_Low' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], name='Lower Band',
                                    line=dict(dash='dash')))

    fig.update_layout(height=500, template='plotly_dark', paper_bgcolor=COLORS['bg_primary'], plot_bgcolor=COLORS['bg_primary'], xaxis=dict(gridcolor=COLORS['grid']), yaxis=dict(gridcolor=COLORS['grid']))
    st.plotly_chart(fig, use_container_width=True)


def render_predictions():
    """Render predictions tab with actual price predictions"""
    st.subheader("🤖 Price Predictions")
    
    if 'data_with_indicators' not in st.session_state:
        st.warning("Please load data first")
        return
    
    data = st.session_state['data_with_indicators']
    ticker = st.session_state.get('ticker', 'Stock')
    
    # Check if we have trained models
    import os
    import joblib
    
    models_available = {
        'xgboost': os.path.exists('models/saved_models/xgboost/xgboost_model.json'),
        'random_forest': os.path.exists('models/saved_models/random_forest/rf_model.pkl'),
        'lstm': os.path.exists('models/saved_models/lstm/lstm_model.keras')
    }
    
    has_models = any(models_available.values())
    
    if has_models:
        st.success("✅ Trained models found!")
        
        # Try to load and use models
        try:
            # Load scaler
            scaler_path = 'models/scalers/feature_scaler.pkl'
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
            
            # Prepare features from the loaded data
            feature_cols = ['Returns', 'SMA_20', 'SMA_50', 'EMA_12', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'ATR']
            
            # Check which features are available
            available_features = [col for col in feature_cols if col in data.columns]
            
            if len(available_features) >= 5:
                X_latest = data[available_features].dropna().tail(30)
                
                predictions = {}
                
                # XGBoost predictions
                if models_available['xgboost']:
                    try:
                        import xgboost as xgb
                        xgb_model = xgb.XGBRegressor()
                        xgb_model.load_model('models/saved_models/xgboost/xgboost_model.json')
                        X_scaled = scaler.fit_transform(X_latest) if not hasattr(scaler, 'mean_') else scaler.transform(X_latest)
                        xgb_preds = xgb_model.predict(X_scaled)
                        predictions['XGBoost'] = xgb_preds
                    except Exception as e:
                        st.warning(f"XGBoost prediction error: {e}")
                
                # Random Forest predictions
                if models_available['random_forest']:
                    try:
                        rf_model = joblib.load('models/saved_models/random_forest/rf_model.pkl')
                        X_scaled = scaler.fit_transform(X_latest) if not hasattr(scaler, 'mean_') else scaler.transform(X_latest)
                        rf_preds = rf_model.predict(X_scaled)
                        predictions['Random Forest'] = rf_preds
                    except Exception as e:
                        st.warning(f"Random Forest prediction error: {e}")
                
                if predictions:
                    # Display predictions
                    st.subheader("📊 Predicted Returns (Next Day)")
                    
                    # Latest predictions
                    col1, col2, col3 = st.columns(3)
                    
                    current_price = data['Close'].iloc[-1]
                    
                    for i, (model_name, preds) in enumerate(predictions.items()):
                        with [col1, col2, col3][i % 3]:
                            latest_pred = preds[-1] * 100  # Convert to percentage
                            predicted_price = current_price * (1 + preds[-1])
                            
                            st.metric(
                                f"{model_name} Prediction",
                                f"{latest_pred:+.2f}%",
                                f"→ ${predicted_price:.2f}"
                            )
                    
                    # Plot historical predictions vs actual
                    st.subheader("📈 Prediction vs Actual Returns")
                    
                    fig = go.Figure()
                    
                    # Actual returns
                    if 'Returns' in data.columns:
                        actual_returns = data['Returns'].dropna().tail(30) * 100
                        fig.add_trace(go.Scatter(
                            x=list(range(len(actual_returns))),
                            y=actual_returns,
                            name='Actual Returns',
                            line=dict(color='blue', width=2)
                        ))
                    
                    # Model predictions
                    colors = {'XGBoost': 'green', 'Random Forest': 'orange', 'LSTM': 'purple'}
                    for model_name, preds in predictions.items():
                        fig.add_trace(go.Scatter(
                            x=list(range(len(preds))),
                            y=preds * 100,
                            name=f'{model_name} Predicted',
                            line=dict(color=colors.get(model_name, 'red'), dash='dash')
                        ))
                    
                    fig.update_layout(
                        height=400,
                        template='plotly_white',
                        xaxis_title='Days',
                        yaxis_title='Return (%)',
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction summary
                    st.subheader("📋 Prediction Summary")
                    
                    summary_data = []
                    for model_name, preds in predictions.items():
                        summary_data.append({
                            'Model': model_name,
                            'Latest Prediction (%)': f"{preds[-1] * 100:+.3f}",
                            'Avg Prediction (%)': f"{np.mean(preds) * 100:+.3f}",
                            'Std Dev (%)': f"{np.std(preds) * 100:.3f}",
                            'Signal': '📈 BUY' if preds[-1] > 0 else '📉 SELL'
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
            else:
                st.warning("Not enough features in data. Please ensure technical indicators are calculated.")
                
        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        # No models trained - show simple prediction based on technical analysis
        st.warning("⚠️ No trained models found. Showing technical analysis-based signals.")
        st.info("Train models using the notebook to get ML-based predictions.")
        
        # Simple technical signal
        if 'RSI' in data.columns and 'MACD' in data.columns:
            latest = data.iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rsi = latest.get('RSI', 50)
                if rsi < 30:
                    st.metric("RSI Signal", f"{rsi:.1f}", "📈 Oversold - BUY")
                elif rsi > 70:
                    st.metric("RSI Signal", f"{rsi:.1f}", "📉 Overbought - SELL")
                else:
                    st.metric("RSI Signal", f"{rsi:.1f}", "➡️ Neutral")
            
            with col2:
                macd = latest.get('MACD', 0)
                macd_signal = latest.get('MACD_Signal', 0)
                if macd > macd_signal:
                    st.metric("MACD Signal", f"{macd:.3f}", "📈 Bullish")
                else:
                    st.metric("MACD Signal", f"{macd:.3f}", "📉 Bearish")
            
            with col3:
                close = latest['Close']
                sma_20 = latest.get('SMA_20', close)
                if close > sma_20:
                    st.metric("Trend", f"${close:.2f}", f"📈 Above SMA20 (${sma_20:.2f})")
                else:
                    st.metric("Trend", f"${close:.2f}", f"📉 Below SMA20 (${sma_20:.2f})")
            
            # Composite signal
            st.markdown("---")
            st.subheader("🎯 Composite Signal")
            
            signals = []
            if rsi < 30: signals.append(1)
            elif rsi > 70: signals.append(-1)
            else: signals.append(0)
            
            if macd > macd_signal: signals.append(1)
            else: signals.append(-1)
            
            if close > sma_20: signals.append(1)
            else: signals.append(-1)
            
            avg_signal = np.mean(signals)
            
            if avg_signal > 0.3:
                st.success("🟢 **BULLISH** - Technical indicators suggest upward momentum")
            elif avg_signal < -0.3:
                st.error("🔴 **BEARISH** - Technical indicators suggest downward momentum")
            else:
                st.warning("🟡 **NEUTRAL** - Mixed signals, wait for clearer direction")


def render_portfolio():
    """Render portfolio tab with actual portfolio analysis"""
    st.subheader("💼 Portfolio Optimization")
    
    if 'data' not in st.session_state:
        st.warning("Please load data first")
        return
    
    data = st.session_state['data']
    ticker = st.session_state.get('ticker', 'Stock')
    
    # Calculate returns
    if 'Close' in data.columns:
        returns = data['Close'].pct_change().dropna()
    else:
        st.error("No price data available")
        return
    
    # Portfolio metrics for single stock (or use as baseline)
    st.subheader(f"📊 {ticker} Performance Metrics")
    
    # Calculate key metrics
    total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    annual_return = returns.mean() * 252 * 100
    annual_volatility = returns.std() * np.sqrt(252) * 100
    sharpe_ratio = (annual_return / 100) / (annual_volatility / 100) if annual_volatility != 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Return", f"{total_return:+.2f}%")
    with col2:
        st.metric("Annual Return", f"{annual_return:+.2f}%")
    with col3:
        st.metric("Annual Volatility", f"{annual_volatility:.2f}%")
    with col4:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    with col5:
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    
    # Cumulative returns chart
    st.subheader("📈 Cumulative Returns")
    
    cumulative_returns = (1 + returns).cumprod() - 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns * 100,
        fill='tozeroy',
        name='Cumulative Return'
    ))
    
    fig.update_layout(
        height=350,
        template='plotly_white',
        yaxis_title='Return (%)',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown chart
    st.subheader("📉 Drawdown Analysis")
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='red')
    ))
    
    fig_dd.update_layout(
        height=250,
        template='plotly_white',
        yaxis_title='Drawdown (%)',
        showlegend=False
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Risk metrics
    st.subheader("⚠️ Risk Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5) * 100
        st.metric("VaR (95%)", f"{var_95:.2f}%", help="Daily Value at Risk at 95% confidence")
    
    with col2:
        # Sortino Ratio (downside risk adjusted)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (annual_return / 100) / downside_std if downside_std != 0 else 0
        st.metric("Sortino Ratio", f"{sortino:.2f}", help="Risk-adjusted return using downside deviation")
    
    with col3:
        # Calmar Ratio
        calmar = (annual_return / 100) / abs(max_drawdown / 100) if max_drawdown != 0 else 0
        st.metric("Calmar Ratio", f"{calmar:.2f}", help="Return divided by max drawdown")
    
    # Distribution of returns
    st.subheader("📊 Returns Distribution")
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Daily Returns'
    ))
    
    # Add normal distribution overlay
    import scipy.stats as stats
    x_range = np.linspace(returns.min() * 100, returns.max() * 100, 100)
    normal_curve = stats.norm.pdf(x_range, returns.mean() * 100, returns.std() * 100) * len(returns) * (returns.max() - returns.min()) * 100 / 50
    
    fig_hist.add_trace(go.Scatter(
        x=x_range,
        y=normal_curve,
        name='Normal Distribution',
        line=dict(color='red', width=2)
    ))
    
    fig_hist.update_layout(
        height=300,
        template='plotly_white',
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Monthly returns heatmap
    st.subheader("📅 Monthly Returns Heatmap")
    
    try:
        monthly_returns = data['Close'].resample('M').last().pct_change().dropna()
        
        # Create monthly returns matrix
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values * 100
        })
        
        pivot_table = monthly_df.pivot_table(values='Return', index='Year', columns='Month', aggfunc='first')
        pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot_table.columns)]
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_table.values, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>"
        ))
        
        fig_heatmap.update_layout(
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not generate monthly heatmap: {e}")
    
    # Summary statistics
    st.subheader("📋 Summary Statistics")
    
    stats_data = {
        'Metric': ['Mean Daily Return', 'Median Daily Return', 'Std Dev', 'Skewness', 'Kurtosis', 
                   'Best Day', 'Worst Day', 'Positive Days', 'Negative Days'],
        'Value': [
            f"{returns.mean() * 100:.4f}%",
            f"{returns.median() * 100:.4f}%",
            f"{returns.std() * 100:.4f}%",
            f"{returns.skew():.2f}",
            f"{returns.kurtosis():.2f}",
            f"{returns.max() * 100:.2f}%",
            f"{returns.min() * 100:.2f}%",
            f"{(returns > 0).sum()} ({(returns > 0).mean() * 100:.1f}%)",
            f"{(returns < 0).sum()} ({(returns < 0).mean() * 100:.1f}%)"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()


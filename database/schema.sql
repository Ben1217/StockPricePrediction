-- ================================
-- STOCK METADATA
-- ================================
CREATE TABLE IF NOT EXISTS stocks (
    symbol VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    index_membership VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    added_date DATE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_stocks_sector ON stocks(sector);
CREATE INDEX IF NOT EXISTS idx_stocks_index ON stocks(index_membership);

-- ================================
-- PRICE DATA
-- ================================
CREATE TABLE IF NOT EXISTS daily_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(12, 4),
    high DECIMAL(12, 4),
    low DECIMAL(12, 4),
    close DECIMAL(12, 4),
    adjusted_close DECIMAL(12, 4),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date),
    FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_daily_symbol_date ON daily_prices(symbol, date DESC);

CREATE TABLE IF NOT EXISTS intraday_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    interval VARCHAR(5) NOT NULL,
    open DECIMAL(12, 4),
    high DECIMAL(12, 4),
    low DECIMAL(12, 4),
    close DECIMAL(12, 4),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval),
    FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_intraday_symbol_timestamp ON intraday_prices(symbol, timestamp DESC, interval);

-- ================================
-- TECHNICAL INDICATORS
-- ================================
CREATE TABLE IF NOT EXISTS technical_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_value DECIMAL(12, 4),
    timeframe VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date, indicator_name, timeframe),
    FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_indicators_symbol_date ON technical_indicators(symbol, date DESC);

-- ================================
-- MODEL PREDICTIONS
-- ================================
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    predicted_price DECIMAL(12, 4),
    predicted_return DECIMAL(8, 6),
    confidence_score DECIMAL(5, 4),
    actual_price DECIMAL(12, 4),
    actual_return DECIMAL(8, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, prediction_date, target_date, model_name),
    FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol, target_date DESC);

-- ================================
-- MODEL METADATA
-- ================================
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(50) NOT NULL,
    training_date DATE NOT NULL,
    validation_mae DECIMAL(10, 6),
    validation_rmse DECIMAL(10, 6),
    test_mae DECIMAL(10, 6),
    test_rmse DECIMAL(10, 6),
    directional_accuracy DECIMAL(5, 4),
    sharpe_ratio DECIMAL(8, 4),
    hyperparameters TEXT,
    feature_importance TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_perf_name_date ON model_performance(model_name, training_date DESC);

-- ================================
-- PORTFOLIO
-- ================================
CREATE TABLE IF NOT EXISTS portfolios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_name VARCHAR(100) NOT NULL,
    creation_date DATE NOT NULL,
    strategy VARCHAR(50),
    initial_capital DECIMAL(15, 2),
    current_value DECIMAL(15, 2),
    total_return DECIMAL(8, 4),
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INT NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    allocation_date DATE NOT NULL,
    weight DECIMAL(6, 4),
    shares DECIMAL(12, 4),
    entry_price DECIMAL(12, 4),
    current_price DECIMAL(12, 4),
    unrealized_pnl DECIMAL(12, 2),
    stop_loss DECIMAL(12, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
    FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_holdings_portfolio ON portfolio_holdings(portfolio_id, allocation_date DESC);

-- ================================
-- TRADES
-- ================================
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INT NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    trade_date TIMESTAMP NOT NULL,
    trade_type VARCHAR(10) NOT NULL,
    quantity DECIMAL(12, 4),
    price DECIMAL(12, 4),
    commission DECIMAL(8, 2),
    slippage DECIMAL(8, 4),
    realized_pnl DECIMAL(12, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
);

CREATE INDEX IF NOT EXISTS idx_trades_portfolio_date ON trades(portfolio_id, trade_date DESC);

-- ================================
-- BACKTEST RESULTS
-- ================================
CREATE TABLE IF NOT EXISTS backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_name VARCHAR(100),
    strategy VARCHAR(50),
    start_date DATE,
    end_date DATE,
    initial_capital DECIMAL(15, 2),
    final_value DECIMAL(15, 2),
    total_return DECIMAL(8, 4),
    sharpe_ratio DECIMAL(8, 4),
    sortino_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    win_rate DECIMAL(5, 4),
    total_trades INT,
    benchmark_return DECIMAL(8, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

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
    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_stocks_sector ON stocks(sector);
CREATE INDEX IF NOT EXISTS idx_stocks_index ON stocks(index_membership);

-- ================================
-- PRICE DATA (TimescaleDB Hypertables)
-- ================================

-- 1. Daily Prices
CREATE TABLE IF NOT EXISTS daily_prices (
    symbol VARCHAR(10) NOT NULL,
    date TIMESTAMPTZ NOT NULL,
    open NUMERIC(12, 4),
    high NUMERIC(12, 4),
    low NUMERIC(12, 4),
    close NUMERIC(12, 4),
    adjusted_close NUMERIC(12, 4),
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT daily_prices_pk PRIMARY KEY (symbol, date),
    FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

-- Convert to Hypertable (partition by time only, chunk size 1 year for daily)
SELECT create_hypertable('daily_prices', 'date', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 year');


-- 2. Intraday Prices
CREATE TABLE IF NOT EXISTS intraday_prices (
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    interval VARCHAR(5) NOT NULL, -- e.g., '1m', '5m'
    open NUMERIC(12, 4),
    high NUMERIC(12, 4),
    low NUMERIC(12, 4),
    close NUMERIC(12, 4),
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT intraday_prices_pk PRIMARY KEY (symbol, timestamp, interval),
    FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

-- Convert to Hypertable (partition by time, chunk size 1 week for intraday)
SELECT create_hypertable('intraday_prices', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 week');

-- Index for fast symbol lookups ordered by time
CREATE INDEX IF NOT EXISTS idx_intraday_symbol_time ON intraday_prices (symbol, timestamp DESC);

-- ================================
-- CONTINUOUS AGGREGATES
-- ================================

-- Example: Hourly candles from 1-minute data (Assuming '1m' interval in intraday_prices)
-- Note: Requires specific interval data. We create the view definition but might need to adjust based on actual data interval presence.
/*
CREATE MATERIALIZED VIEW hourly_candles
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    symbol,
    FIRST(open, timestamp) as open,
    MAX(high) as high,
    MIN(low) as low,
    LAST(close, timestamp) as close,
    SUM(volume) as volume
FROM intraday_prices
WHERE interval = '1m'
GROUP BY bucket, symbol;

-- Refresh policy (e.g., refresh last 3 days every hour)
SELECT add_continuous_aggregate_policy('hourly_candles',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
*/

-- ================================
-- TECHNICAL INDICATORS
-- ================================
CREATE TABLE IF NOT EXISTS technical_indicators (
    symbol VARCHAR(10) NOT NULL,
    date TIMESTAMPTZ NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_value NUMERIC(12, 4),
    timeframe VARCHAR(10),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date, indicator_name, timeframe),
    FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

SELECT create_hypertable('technical_indicators', 'date', if_not_exists => TRUE);


-- ================================
-- TRADES & PORTFOLIO (Standard Tables)
-- ================================
-- (Keeping these as standard tables unless trade volume is massive)

CREATE TABLE IF NOT EXISTS portfolios (
    id SERIAL PRIMARY KEY,
    portfolio_name VARCHAR(100) NOT NULL,
    creation_date DATE NOT NULL,
    initial_capital NUMERIC(15, 2),
    current_value NUMERIC(15, 2),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id SERIAL PRIMARY KEY,
    portfolio_id INT NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    allocation_date DATE NOT NULL,
    shares NUMERIC(12, 4),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    portfolio_id INT NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    trade_date TIMESTAMPTZ NOT NULL,
    trade_type VARCHAR(10) NOT NULL,
    quantity NUMERIC(12, 4),
    price NUMERIC(12, 4),
    commission NUMERIC(8, 2),
    realized_pnl NUMERIC(12, 2),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE
);

-- Convert trades to hypertable? Probably not needed unless HFT.

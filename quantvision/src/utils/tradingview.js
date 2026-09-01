/**
 * Mapping our tickers onto TradingView's symbol namespace.
 *
 * The backend speaks Yahoo Finance: `^GSPC`, `BRK-B`, `BTC-USD`, `EURUSD=X`.
 * TradingView speaks exchange-prefixed pairs: `SP:SPX`, `BRK.B`, `CRYPTO:BTCUSD`,
 * `FX:EURUSD`. Handing the widget a Yahoo index symbol renders an "invalid
 * symbol" panel rather than an error anyone can act on, so the translation
 * happens here rather than at each call site.
 *
 * Plain US equity tickers are passed through bare. TradingView resolves those
 * against its own listing search, which is more reliable than us guessing
 * between NASDAQ and NYSE from a symbol alone — a wrong prefix fails where no
 * prefix succeeds.
 */

/** Indices, where Yahoo's caret symbols have no relationship to TradingView's. */
const INDEX_SYMBOLS = {
    "^GSPC": "SP:SPX",
    "^SPX": "SP:SPX",
    "^IXIC": "NASDAQ:IXIC",
    "^NDX": "NASDAQ:NDX",
    "^DJI": "DJ:DJI",
    "^RUT": "TVC:RUT",
    "^VIX": "TVC:VIX",
    "^FTSE": "TVC:UKX",
    "^GDAXI": "XETR:DAX",
    "^N225": "TVC:NI225",
    "^HSI": "TVC:HSI",
    "^STOXX50E": "TVC:SX5E",
};

/** ETFs that stand in for an index often enough to be worth pinning. */
const EXCHANGE_OVERRIDES = {
    SPY: "AMEX:SPY",
    QQQ: "NASDAQ:QQQ",
    DIA: "AMEX:DIA",
    IWM: "AMEX:IWM",
};

/**
 * The TradingView symbol for one of our tickers.
 *
 * Returns a string the Advanced Chart widget accepts. An empty or unusable
 * input falls back to the S&P 500, because a widget with no symbol renders a
 * blank panel with no explanation.
 */
export function toTradingViewSymbol(ticker) {
    const raw = String(ticker || "").trim().toUpperCase();
    if (!raw) return INDEX_SYMBOLS["^GSPC"];

    if (INDEX_SYMBOLS[raw]) return INDEX_SYMBOLS[raw];
    if (EXCHANGE_OVERRIDES[raw]) return EXCHANGE_OVERRIDES[raw];

    // Forex, Yahoo style: EURUSD=X
    if (raw.endsWith("=X")) return `FX:${raw.slice(0, -2)}`;

    // Futures, Yahoo style: ES=F. TradingView's continuous contracts use a
    // trailing 1!, and the root is the same.
    if (raw.endsWith("=F")) return `${raw.slice(0, -2)}1!`;

    // Crypto pairs, Yahoo style: BTC-USD. The hyphen is the discriminator —
    // a share class (BRK-B) never has a fiat quote on its right-hand side.
    const crypto = raw.match(/^([A-Z0-9]{2,10})-(USD|USDT|EUR|GBP|JPY)$/);
    if (crypto) return `CRYPTO:${crypto[1]}${crypto[2]}`;

    // Share classes: Yahoo writes BRK-B, TradingView writes BRK.B.
    if (/^[A-Z]{1,5}-[A-Z]$/.test(raw)) return raw.replace("-", ".");

    // An already-prefixed symbol (NASDAQ:NVDA) is passed through untouched.
    return raw;
}

/** Timeframes we offer, mapped onto the widget's interval codes. */
const INTERVALS = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
    "1d": "D",
    "1wk": "W",
    "1mo": "M",
};

export function toTradingViewInterval(timeframe) {
    return INTERVALS[timeframe] || "D";
}

/** The timeframes the chart toolbar offers, in the order they are shown. */
export const CHART_TIMEFRAMES = ["1m", "15m", "1h", "1d", "1wk", "1mo"];

/** A link to the same chart on tradingview.com, for the "open in TradingView" action. */
export function tradingViewUrl(ticker) {
    return `https://www.tradingview.com/chart/?symbol=${encodeURIComponent(toTradingViewSymbol(ticker))}`;
}

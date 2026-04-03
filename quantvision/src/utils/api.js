/**
 * API client for QuantVision backend.
 * All endpoints hit the FastAPI server at localhost:8000.
 */

const API_BASE = "http://localhost:8000/api";

const INTERVAL_LIMITS = {
    "1m": { priceDays: [7, 7], indicatorDays: [60, 120], lookback: [60, 90], sentimentDays: [120, 120] },
    "5m": { priceDays: [30, 60], indicatorDays: [60, 120], lookback: [60, 120], sentimentDays: [120, 180] },
    "15m": { priceDays: [30, 60], indicatorDays: [60, 120], lookback: [60, 120], sentimentDays: [120, 180] },
    "1h": { priceDays: [180, 730], indicatorDays: [120, 240], lookback: [120, 365], sentimentDays: [240, 730] },
    "4h": { priceDays: [180, 730], indicatorDays: [120, 240], lookback: [120, 365], sentimentDays: [240, 730] },
    "1d": { priceDays: [30, 420], indicatorDays: [120, 320], lookback: [120, 420], sentimentDays: [240, 420] },
    "1wk": { priceDays: [730, 3650], indicatorDays: [120, 300], lookback: [180, 500], sentimentDays: [800, 2600] },
    "1mo": { priceDays: [1825, 3650], indicatorDays: [120, 180], lookback: [180, 500], sentimentDays: [1200, 3650] },
};

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function getIntervalLimit(interval, key) {
    return INTERVAL_LIMITS[interval]?.[key] || INTERVAL_LIMITS["1d"][key];
}

async function apiFetch(path, options = {}) {
    const url = `${API_BASE}${path}`;
    const res = await fetch(url, {
        headers: { "Content-Type": "application/json", ...options.headers },
        ...options,
    });
    if (!res.ok) {
        const body = await res.text();
        throw new Error(`API ${res.status}: ${body}`);
    }
    return res.json();
}

// ── Data ────────────────────────────────────────────────────
export async function fetchPrices(symbol, source = "yfinance", days = 120, interval = "1d") {
    const [minDays, maxDays] = getIntervalLimit(interval, "priceDays");
    const safeDays = clamp(days, minDays, maxDays);
    const url = `/data/prices/${symbol}?source=${source}&days=${safeDays}&interval=${interval}`;
    try {
        return await apiFetch(url);
    } catch (err) {
        const msg = String(err?.message || "");
        const shouldRetry = msg.includes("API 404") || msg.includes("API 422") || msg.includes("API 500") || msg.includes("API 502");
        if (!shouldRetry) throw err;
        const fallbackDays = maxDays;
        return apiFetch(`/data/prices/${symbol}?source=${source}&days=${fallbackDays}&interval=${interval}`);
    }
}

export async function fetchLiveQuote(symbol, source = "yfinance") {
    return apiFetch(`/data/quote/${symbol}?source=${source}`);
}

export async function fetchExtendedQuote(symbol, source = "yfinance") {
    return apiFetch(`/data/extended-quote/${symbol}?source=${source}`);
}

export async function fetchIndicators(symbol, days = 120, interval = "1d") {
    const [minDays, maxDays] = getIntervalLimit(interval, "indicatorDays");
    const safeDays = clamp(days, minDays, maxDays);
    const url = `/data/indicators/${symbol}?days=${safeDays}&interval=${interval}`;
    try {
        return await apiFetch(url);
    } catch (err) {
        const msg = String(err?.message || "");
        const shouldRetry = msg.includes("API 422") || msg.includes("API 500") || msg.includes("API 502");
        if (!shouldRetry) throw err;
        const fallbackDays = maxDays;
        return apiFetch(`/data/indicators/${symbol}?days=${fallbackDays}&interval=${interval}`);
    }
}

export async function fetchSP500() {
    return apiFetch("/data/sp500");
}

export async function fetchDataSources() {
    return apiFetch("/data/sources");
}

export async function uploadDataset(file) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API_BASE}/data/upload`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
}

// ── Training ────────────────────────────────────────────────
export async function triggerTraining(params) {
    return apiFetch("/training/train", {
        method: "POST",
        body: JSON.stringify(params),
    });
}

export async function getTrainingStatus(jobId) {
    return apiFetch(`/training/status/${jobId}`);
}

export async function listModels() {
    return apiFetch("/training/models");
}

// ── Predictions ─────────────────────────────────────────────
export async function fetchPredictions(symbol, modelType = "xgboost", horizon = 30) {
    return apiFetch("/predict", {
        method: "POST",
        body: JSON.stringify({ symbol, model_type: modelType, horizon }),
    });
}

export async function fetchHistoricalSignals(symbol, days = 90, modelType = "xgboost") {
    return apiFetch(`/predict/historical-signals/${symbol}?days=${days}&model_type=${modelType}`);
}

export async function fetchPatterns(symbol, tf = "1d") {
    return apiFetch(`/patterns/${symbol}?tf=${tf}`);
}

export async function fetchConfluence(symbol) {
    return apiFetch(`/patterns/confluence/${symbol}`);
}

export async function fetchSupportResistance(symbol, interval = "1d", lookback = 180) {
    const [minLookback, maxLookback] = getIntervalLimit(interval, "lookback");
    const safeLookback = clamp(lookback, minLookback, maxLookback);
    return apiFetch(`/patterns/support-resistance/${symbol}?interval=${interval}&lookback=${safeLookback}`);
}

export async function fetchSentiment(symbol, days = 400, interval = "1d") {
    const [minDays, maxDays] = getIntervalLimit(interval, "sentimentDays");
    const safeDays = clamp(days, minDays, maxDays);
    const url = `/sentiment/${symbol}?days=${safeDays}&interval=${interval}`;
    try {
        return await apiFetch(url);
    } catch (err) {
        const msg = String(err?.message || "");
        const shouldRetry = msg.includes("API 400") || msg.includes("API 422") || msg.includes("API 500") || msg.includes("API 502");
        if (!shouldRetry) throw err;
        const fallbackDays = maxDays;
        return apiFetch(`/sentiment/${symbol}?days=${fallbackDays}&interval=${interval}`);
    }
}

// ── Backtesting ─────────────────────────────────────────────
export async function runBacktest(params) {
    return apiFetch("/backtest/run", {
        method: "POST",
        body: JSON.stringify(params),
    });
}

export async function listBacktests() {
    return apiFetch("/backtest/results");
}

export async function getBacktestResults(backtestId) {
    return apiFetch(`/backtest/results/${backtestId}`);
}

// ── Portfolio ───────────────────────────────────────────────
export async function optimizePortfolio(params) {
    return apiFetch("/portfolio/optimize", {
        method: "POST",
        body: JSON.stringify(params),
    });
}

export async function fetchFrontier(params) {
    return apiFetch("/portfolio/frontier", {
        method: "POST",
        body: JSON.stringify(params),
    });
}

export async function fetchPortfolioMetrics(symbols, lookback = 252) {
    const sym = Array.isArray(symbols) ? symbols.join(",") : symbols;
    return apiFetch(`/portfolio/metrics?symbols=${sym}&lookback=${lookback}`);
}

// ── Export ───────────────────────────────────────────────────
export function getCSVExportURL(resource, symbol = "SPY") {
    return `${API_BASE}/export/csv/${resource}?symbol=${symbol}`;
}

export function getPDFExportURL(resource, symbol = "SPY") {
    return `${API_BASE}/export/pdf/${resource}?symbol=${symbol}`;
}

/**
 * API client for QuantVision backend.
 * All endpoints hit the FastAPI server at localhost:8000.
 */

const API_BASE = "http://localhost:8000/api";

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
export async function fetchPrices(symbol, source = "yfinance", days = 120) {
    return apiFetch(`/data/prices/${symbol}?source=${source}&days=${days}`);
}

export async function fetchIndicators(symbol, days = 120) {
    return apiFetch(`/data/indicators/${symbol}?days=${days}`);
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

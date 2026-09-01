/**
 * API client for QuantVision backend.
 * All endpoints hit the FastAPI server at localhost:8000.
 */

import { DEFAULT_INDEX_SYMBOL } from "./data";

/** Origin of the FastAPI server. Override with VITE_API_ORIGIN at build time. */
export const API_ORIGIN = import.meta.env.VITE_API_ORIGIN ?? "http://localhost:8000";
const API_BASE = `${API_ORIGIN}/api`;

/**
 * FastAPI reports errors as {"detail": "..."}. Pull that out so panels render the
 * server's explanation instead of a raw JSON envelope; `body` keeps the original.
 */
function errorDetail(body) {
    if (!body) return "";
    try {
        const detail = JSON.parse(body)?.detail;
        if (typeof detail === "string") return detail;
        // Pydantic validation errors arrive as [{loc, msg, type}, ...].
        if (Array.isArray(detail)) return detail.map(d => d?.msg || JSON.stringify(d)).join("; ");
    } catch {
        // Not JSON — the raw body is the best message available.
    }
    return body;
}

/** Error carrying the HTTP status, so callers branch on a number rather than a message. */
export class ApiError extends Error {
    constructor(status, body, url) {
        super(`API ${status}: ${errorDetail(body) || "request failed"}`);
        this.name = "ApiError";
        this.status = status;
        this.body = body;
        this.url = url;
    }
}

/** Statuses worth a second attempt. 404 and 422 are deterministic — retrying just doubles latency. */
const TRANSIENT_STATUSES = new Set([502, 503, 504]);
const isTransient = (err) => err instanceof ApiError && TRANSIENT_STATUSES.has(err.status);

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

function encodeSymbol(symbol) {
    return encodeURIComponent(String(symbol || "").trim());
}

/** Sent only when the server is configured to require it (VITE_API_KEY). */
const API_KEY = import.meta.env.VITE_API_KEY ?? "";
const authHeaders = () => (API_KEY ? { "X-API-Key": API_KEY } : {});

async function apiFetch(path, options = {}) {
    const url = `${API_BASE}${path}`;
    const res = await fetch(url, {
        headers: { "Content-Type": "application/json", ...authHeaders(), ...options.headers },
        ...options,
    });
    if (!res.ok) {
        const body = await res.text();
        throw new ApiError(res.status, body, url);
    }
    return res.json();
}

/** Liveness probe against the server root (not under /api). */
export async function fetchHealth(options = {}) {
    const res = await fetch(`${API_ORIGIN}/health`, { headers: authHeaders(), ...options });
    if (!res.ok) throw new ApiError(res.status, await res.text(), `${API_ORIGIN}/health`);
    return res.json();
}

// ── Data ────────────────────────────────────────────────────
export async function fetchPrices(symbol, source = "yfinance", days = 120, interval = "1d") {
    const [minDays, maxDays] = getIntervalLimit(interval, "priceDays");
    const safeDays = clamp(days, minDays, maxDays);
    const encodedSymbol = encodeSymbol(symbol);
    const url = `/data/prices/${encodedSymbol}?source=${source}&days=${safeDays}&interval=${interval}`;
    try {
        return await apiFetch(url);
    } catch (err) {
        if (!isTransient(err)) throw err;
        return apiFetch(`/data/prices/${encodedSymbol}?source=${source}&days=${maxDays}&interval=${interval}`);
    }
}

/**
 * Batch quote lookup — one request for many symbols.
 * Replaces the per-ticker fan-out in the ticker bar and the sector heatmap.
 */
export async function fetchQuotes(symbols, options = {}) {
    const list = (Array.isArray(symbols) ? symbols : [symbols])
        .map(s => String(s || "").trim().toUpperCase())
        .filter(Boolean);
    if (list.length === 0) return {};
    const data = await apiFetch(`/data/quotes?symbols=${encodeURIComponent(list.join(","))}`, options);
    // Normalise to the shape the ticker bar and heatmap consume.
    const out = {};
    for (const [symbol, q] of Object.entries(data.quotes || {})) {
        out[symbol] = { price: q.price, change: q.change_pct, vol: q.volume };
    }
    return out;
}

export async function fetchLiveQuote(symbol, source = "yfinance") {
    return apiFetch(`/data/quote/${encodeSymbol(symbol)}?source=${source}`);
}

export async function fetchExtendedQuote(symbol, source = "yfinance") {
    return apiFetch(`/data/extended-quote/${encodeSymbol(symbol)}?source=${source}`);
}

export async function fetchIndicators(symbol, days = 120, interval = "1d") {
    const [minDays, maxDays] = getIntervalLimit(interval, "indicatorDays");
    const safeDays = clamp(days, minDays, maxDays);
    const encodedSymbol = encodeSymbol(symbol);
    const url = `/data/indicators/${encodedSymbol}?days=${safeDays}&interval=${interval}`;
    try {
        return await apiFetch(url);
    } catch (err) {
        if (!isTransient(err)) throw err;
        return apiFetch(`/data/indicators/${encodedSymbol}?days=${maxDays}&interval=${interval}`);
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
    // No Content-Type header — the browser sets the multipart boundary itself.
    const res = await fetch(`${API_BASE}/data/upload`, {
        method: "POST",
        headers: authHeaders(),
        body: form,
    });
    if (!res.ok) throw new ApiError(res.status, await res.text(), `${API_BASE}/data/upload`);
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

// ── Model readiness & automatic preparation ──────────────────
/**
 * What a symbol can serve, and whatever preparation job is attached to it.
 *
 * Read-only. This is the call the dashboard makes on every ticker change, so it
 * must never train anything by itself — `prepareModels` below is the one that
 * starts work.
 */
export async function fetchModelReadiness(symbol) {
    return apiFetch(`/models/${encodeSymbol(symbol)}`);
}

/**
 * Train whatever `symbol` is missing, in the background.
 *
 * Idempotent: called while a job runs it returns that same job, and called when
 * nothing needs training it returns readiness with no job. So the UI can fire it
 * on every ticker change without tracking state itself.
 *
 * `force` skips the server's post-attempt cooldown. It belongs on a Retry button
 * a user pressed after reading an error, and nowhere else — automatic retries
 * with force would defeat the guard that stops a doomed run repeating.
 */
export async function prepareModels(symbol, { force = false } = {}) {
    return apiFetch(`/models/${encodeSymbol(symbol)}/prepare`, {
        method: "POST",
        body: JSON.stringify({ force }),
    });
}

/** Poll one preparation job. 404 once it ages out of the server's tracker. */
export async function getPreparationStatus(jobId) {
    return apiFetch(`/models/prepare/${encodeURIComponent(jobId)}`);
}

// ── Predictions ─────────────────────────────────────────────
export async function fetchPredictions(symbol, modelType = "xgboost", horizon = 1) {
    return apiFetch("/predict", {
        method: "POST",
        body: JSON.stringify({ symbol, model_type: modelType, horizon }),
    });
}

export async function fetchHistoricalSignals(symbol, days = 90, modelType = "xgboost") {
    return apiFetch(`/predict/historical-signals/${encodeSymbol(symbol)}?days=${days}&model_type=${modelType}`);
}

// ── Next-day direction ───────────────────────────────────────
/**
 * Walk-forward evaluation, the gated P(up tomorrow) gauge, the rolling hit-rate
 * strip, and the equity curve for one symbol.
 *
 * 404 means no walk-forward report has been generated yet; the server's detail
 * message names the command that produces one. That is not an error state to
 * paper over — the gauge is meaningless without the evaluation beside it.
 */
export async function fetchDirection(symbol, model = "logistic", includeGauge = true) {
    return apiFetch(
        `/direction/${encodeSymbol(symbol)}?model=${encodeURIComponent(model)}` +
        `&include_gauge=${includeGauge ? "true" : "false"}`
    );
}

/** Every symbol/model pair with a stored direction report. */
export async function listDirectionReports() {
    return apiFetch("/direction/");
}

/**
 * The reasoned direction call: UP / DOWN / NEUTRAL with the evidence behind it.
 *
 * Distinct from `fetchDirection` above, which serves one model's gated gauge and
 * its walk-forward record. This serves the combined read — trend, momentum,
 * volume, price action, support/resistance, volatility and historical analogs,
 * each with the percentage points it contributed — blended with that classifier
 * by measured skill.
 *
 * It answers whether or not a walk-forward report exists, so the panel is never
 * blocked on training; `blend.classifier.weight` is 0 and `classifier_note`
 * explains why when the classifier is not yet part of the answer.
 */
export async function fetchDirectionAnalysis(symbol, { model = "logistic", refresh = false } = {}) {
    return apiFetch(
        `/direction/${encodeSymbol(symbol)}/analysis?model=${encodeURIComponent(model)}` +
        (refresh ? "&refresh=true" : "")
    );
}

// ── Ensemble Predictions ─────────────────────────────────────
export async function fetchEnsemblePrediction(symbol, horizon = 30) {
    return apiFetch("/predict/ensemble", {
        method: "POST",
        body: JSON.stringify({ symbol, horizon }),
    });
}

export async function triggerEnsembleTraining(symbol, horizons = [7, 15, 30, 60], modelTypes = ["xgboost", "random_forest", "lstm"]) {
    return apiFetch("/predict/ensemble/train", {
        method: "POST",
        body: JSON.stringify({ symbol, horizons, model_types: modelTypes, lookback_days: 1825 }),
    });
}

export async function getEnsembleTrainingStatus(jobId) {
    return apiFetch(`/predict/ensemble/train/status/${jobId}`);
}

// ── Patterns ─────────────────────────────────────────────────
export async function fetchPatterns(symbol, tf = "1d") {
    return apiFetch(`/patterns/${encodeSymbol(symbol)}?tf=${tf}`);
}

export async function fetchConfluence(symbol) {
    return apiFetch(`/patterns/confluence/${encodeSymbol(symbol)}`);
}

export async function fetchSupportResistance(symbol, interval = "1d", lookback = 180) {
    const [minLookback, maxLookback] = getIntervalLimit(interval, "lookback");
    const safeLookback = clamp(lookback, minLookback, maxLookback);
    return apiFetch(`/patterns/support-resistance/${encodeSymbol(symbol)}?interval=${interval}&lookback=${safeLookback}`);
}

export async function fetchSentiment(symbol, days = 400, interval = "1d") {
    const [minDays, maxDays] = getIntervalLimit(interval, "sentimentDays");
    const safeDays = clamp(days, minDays, maxDays);
    const encodedSymbol = encodeSymbol(symbol);
    const url = `/sentiment/${encodedSymbol}?days=${safeDays}&interval=${interval}`;
    try {
        return await apiFetch(url);
    } catch (err) {
        if (!isTransient(err)) throw err;
        return apiFetch(`/sentiment/${encodedSymbol}?days=${maxDays}&interval=${interval}`);
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

// ── Agent ────────────────────────────────────────────────────
export async function askAgent(question, options = {}) {
    return apiFetch("/agent/query", {
        method: "POST",
        body: JSON.stringify({ question }),
        ...options,
    });
}

// ── Export ───────────────────────────────────────────────────
export function getCSVExportURL(resource, symbol = DEFAULT_INDEX_SYMBOL) {
    return `${API_BASE}/export/csv/${resource}?symbol=${encodeSymbol(symbol)}`;
}

export function getPDFExportURL(resource, symbol = DEFAULT_INDEX_SYMBOL) {
    return `${API_BASE}/export/pdf/${resource}?symbol=${encodeSymbol(symbol)}`;
}

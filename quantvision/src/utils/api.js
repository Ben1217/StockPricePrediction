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
        // The server's own explanation, already unwrapped from FastAPI's
        // {detail: ...}. Callers that want to show why a request was refused
        // read this instead of re-parsing the body or restating the reason
        // themselves — a hardcoded sentence goes stale the moment a status
        // code starts covering a second cause.
        this.detail = errorDetail(body);
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

/**
 * The S&P 500 constituents the server currently reads, with company and sector.
 *
 * Returns the `symbols` array directly, because the `count` beside it is just
 * its length. Callers fall back to the bundled `SP500_LIST` when this throws —
 * the picker is more useful stale than empty.
 */
export async function fetchSP500() {
    const body = await apiFetch("/data/sp500");
    return Array.isArray(body?.symbols) ? body.symbols : [];
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

/**
 * The best performing model's forecast, ready to superimpose on the chart.
 *
 * Two winners come back, not one. The trajectory and its bands belong to
 * whichever model won MAE/RMSE/MAPE/R-squared at this horizon; the up/down
 * arrow belongs to whichever won accuracy/F1/AUC. They are frequently different
 * models, and `selection` carries the full ranking table behind both so the
 * chart can say who was beaten and on what.
 *
 * `horizon` is the dashboard's 7/15/30/60-day window. It scopes the comparison
 * as well as the forecast: a model scored at another horizon is never ranked
 * against these, because a 30-day error and a 1-day error are not the same
 * measurement.
 *
 * A 200 with `status: "unavailable"` is a normal answer, not a failure — it is
 * what the server says when every candidate failed its out-of-sample skill gate,
 * and `message` explains which and why.
 */
export async function fetchBestModelForecast(symbol, horizon = 30) {
    return apiFetch(`/predict/best/${encodeSymbol(symbol)}?horizon=${horizon}`);
}

/**
 * The candles the Predictions tab draws — no models, no live quote.
 *
 * Deliberately separate from `fetchSimpleForecast`. These come off a cached
 * OHLCV download in milliseconds; the forecast is seconds of transformer
 * sampling. Asking for them together is what made the chart wait on the box.
 *
 * Same server-side download the models read, so the bars are the bars the
 * forecast was built on rather than a differently-adjusted series.
 */
export async function fetchForecastHistory(symbol, days = 252) {
    return apiFetch(`/predict/history/${encodeSymbol(symbol)}?days=${days}`);
}

/**
 * The forecast alone: next bar, direction, and the band around it.
 *
 * The server runs the whole pipeline — OHLCV, the technical-analysis features,
 * Kronos, Chronos-2 and TimesFM 2.5, then the aggregation — and answers with
 * only what is shown. There is deliberately nothing here to reconcile
 * client-side.
 *
 * A 200 with `status: "unavailable"` is a normal answer, and the chart is
 * unaffected by it: the candles arrive from their own request.
 */
export async function fetchSimpleForecast(symbol) {
    return apiFetch(`/predict/forecast/${encodeSymbol(symbol)}`);
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

/**
 * The walk-forward record for a symbol — the out-of-sample scorecard, not a
 * trading simulation.
 *
 * Distinct from `runBacktest`, which simulates one strategy over one period.
 * This reads what the benchmark already measured: purged folds, excess over
 * base rate, the verdict against the random walk, and the forecast's worth
 * after costs.
 *
 * `models` comes back empty rather than as a 404 when nothing has scored the
 * symbol yet; `message` carries the command that would fix that. A record
 * written before the null tests existed reports those fields as null rather
 * than as zero, and the message says so.
 */
export async function fetchBacktestEvidence(symbol, model) {
    const query = model ? `?model=${encodeURIComponent(model)}` : "";
    return apiFetch(`/backtest/evidence/${encodeSymbol(symbol)}${query}`);
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

function symbolList(symbols) {
    const list = Array.isArray(symbols) ? symbols : String(symbols || "").split(",");
    return list.map((item) => String(item || "").trim().toUpperCase()).filter(Boolean);
}

/**
 * How a given split of money would have performed over the lookback window.
 *
 * `weights` is the whole point: pass a ticker→weight map and the server scores
 * that exact allocation, which is what makes "your current split" and "the
 * optimizer's split" comparable — the same window, the same maths, one variable
 * changed. Omit it and the server assumes equal weights.
 *
 * Every number that comes back is measured on history. None of it is a forecast.
 */
export async function fetchPortfolioMetrics(
    symbols,
    { lookback = 252, weights = null, includeAttribution = false } = {},
) {
    const params = new URLSearchParams({
        symbols: symbolList(symbols).join(","),
        lookback: String(lookback),
    });
    if (weights) params.set("weights", JSON.stringify(weights));
    if (includeAttribution) params.set("include_attribution", "true");
    return apiFetch(`/portfolio/metrics?${params.toString()}`);
}

/**
 * How closely the holdings move together, pairwise.
 *
 * This is the number that explains diversification without jargon: two stocks
 * at 0.9 are very nearly one position wearing two names, and owning both buys
 * far less protection than the position count suggests. `high_corr_pairs`
 * names the offenders directly.
 */
export async function fetchCorrelation(symbols, lookbackDays = 90) {
    const params = new URLSearchParams({
        symbols: symbolList(symbols).join(","),
        lookback_days: String(lookbackDays),
    });
    return apiFetch(`/portfolio/correlation?${params.toString()}`);
}

/**
 * Risk-limit breaches for one allocation — concentration, sector, drawdown.
 *
 * Answers "what is wrong with how my money is split right now" in sentences a
 * reader can act on, rather than leaving them to infer it from a weights table.
 */
export async function fetchRiskAlerts(symbols, weights, lookbackDays = 90) {
    const params = new URLSearchParams({
        symbols: symbolList(symbols).join(","),
        weights: JSON.stringify(weights || {}),
        lookback_days: String(lookbackDays),
    });
    return apiFetch(`/portfolio/alerts?${params.toString()}`);
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

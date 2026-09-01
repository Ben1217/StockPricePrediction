import { useEffect, useMemo, useState } from "react";
import { C } from "../utils/data";
import { fetchEnsemblePrediction, fetchPredictions } from "../utils/api";
import { tradingViewUrl } from "../utils/tradingview";
import TradingViewChart from "../components/TradingViewChart";
import DirectionAnalysisPanel from "../components/DirectionAnalysisPanel";
import DirectionPanel from "../components/DirectionPanel";
import ModelPreparation from "../components/ModelPreparation";

const HORIZONS = [7, 15, 30, 60];

/** Multi-horizon regression bundles. These draw the forecast fan. */
const LEGACY_MODEL_KEYS = ["xgboost", "random_forest", "lstm"];

/**
 * Unified next-timeframe models: one price and one P(up) for the next bar.
 *
 * Fetched only when one is actually selected, never as part of the "All
 * Models" sweep. Kronos runs a transformer forward pass per request, so
 * firing it on every ticker change would stall the tab for everyone who
 * never opened it.
 */
const UNIFIED_MODEL_KEYS = ["unified_xgboost", "unified_random_forest", "unified_lstm", "unified_kronos"];

const MODEL_KEYS = [...LEGACY_MODEL_KEYS, ...UNIFIED_MODEL_KEYS];

const isUnifiedModel = (modelType) => UNIFIED_MODEL_KEYS.includes(modelType);
const MODEL_OPTIONS = [
    { value: "all", label: "All Models" },
    { value: "ensemble", label: "Ensemble" },
    { value: "lstm", label: "LSTM" },
    { value: "xgboost", label: "XGBoost" },
    { value: "random_forest", label: "Random Forest" },
    { value: "unified_xgboost", label: "Unified XGBoost" },
    { value: "unified_random_forest", label: "Unified Random Forest" },
    { value: "unified_lstm", label: "Unified LSTM" },
    { value: "unified_kronos", label: "Unified Kronos" },
];

const COLORS = {
    historical: "#9CA3AF",
    ensemble: "#F5C842",
    lstm: "#60A5FA",
    xgboost: "#F59E0B",
    random_forest: "#34D399",
    unified_xgboost: "#FB923C",
    unified_random_forest: "#4ADE80",
    unified_lstm: "#818CF8",
    unified_kronos: "#E879F9",
    band: "#6366F1",
    scenario: "#7C8AA5",
    surface: "#161B22",
    panel: "#0F1623",
};

const WEIGHT_LABELS = {
    lstm: "PyTorch LSTM",
    xgboost: "XGBoost",
    random_forest: "Random Forest",
};

const MODEL_LABELS = {
    historical: "Historical",
    prediction: "Prediction",
    ensemble: "Ensemble",
    lstm: "LSTM",
    xgboost: "XGBoost",
    random_forest: "Random Forest",
    unified_xgboost: "Unified XGBoost",
    unified_random_forest: "Unified Random Forest",
    unified_lstm: "Unified LSTM",
    unified_kronos: "Unified Kronos",
};

function formatPrice(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return `$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatPct(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(2)}%`;
}

function toFiniteNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

function firstFiniteNumber(...values) {
    for (const value of values) {
        const number = toFiniteNumber(value);
        if (number !== null) return number;
    }
    return null;
}

function modelLabel(model) {
    return MODEL_LABELS[model] || MODEL_LABELS.ensemble;
}

function getFinalForecastPoint(points) {
    return Array.isArray(points) && points.length ? points[points.length - 1] : null;
}

function normalizeSingleForecast(payload, modelType) {
    const points = (payload?.forecasts || []).map((point) => ({
        date: point.date,
        predicted: firstFiniteNumber(point.predicted, point.prediction),
        lower_95: point.lower95,
        upper_95: point.upper95,
        lower_68: point.lower68,
        upper_68: point.upper68,
        [modelType]: firstFiniteNumber(point.predicted, point.prediction),
    }));
    return {
        ...payload,
        forecast_points: points,
    };
}

function normalizeEnsemblePoint(point) {
    const prediction = firstFiniteNumber(point.prediction, point.predicted, point.ensemble);
    return {
        ...point,
        ensemble: prediction,
        prediction,
        predicted: prediction,
        lower_95: point.lower_95 ?? point.lower_90,
        upper_95: point.upper_95 ?? point.upper_90,
        lower_68: point.lower_68 ?? point.lower_90,
        upper_68: point.upper_68 ?? point.upper_90,
    };
}

function buildFallbackEnsemble(models) {
    const available = MODEL_KEYS
        .map((key) => [key, models?.[key]])
        .filter(([, payload]) => payload?.status === "ok" && Array.isArray(payload.forecast_points) && payload.forecast_points.length);
    if (!available.length) return null;

    const first = available[0][1];
    const points = first.forecast_points.map((point, index) => {
        const values = available
            .map(([key, payload]) => [key, firstFiniteNumber(
                payload.forecast_points[index]?.prediction,
                payload.forecast_points[index]?.predicted,
                payload.forecast_points[index]?.[key],
            )])
            .filter(([, value]) => value !== null);
        const predicted = values.reduce((sum, [, value]) => sum + Number(value), 0) / Math.max(values.length, 1);
        // Averaging point forecasts gives a centre but no interval. Reusing the
        // centre as both bounds drew a zero-width band and printed the forecast
        // price into the Upper 95% and Lower 95% cards as though it were an
        // interval, which is why all three read the same number. Null is the
        // honest answer: this path has no calibrated uncertainty.
        const bounds = available.reduce(
            (acc, [, payload]) => {
                const p = payload.forecast_points[index];
                acc.lower_95.push(toFiniteNumber(p?.lower_95));
                acc.upper_95.push(toFiniteNumber(p?.upper_95));
                acc.lower_68.push(toFiniteNumber(p?.lower_68));
                acc.upper_68.push(toFiniteNumber(p?.upper_68));
                return acc;
            },
            { lower_95: [], upper_95: [], lower_68: [], upper_68: [] },
        );
        const widest = (list, pick) => {
            const finite = list.filter((v) => v !== null);
            return finite.length ? pick(...finite) : null;
        };
        const row = {
            date: point.date,
            ensemble: predicted,
            prediction: predicted,
            predicted,
            lower_95: widest(bounds.lower_95, Math.min),
            upper_95: widest(bounds.upper_95, Math.max),
            lower_68: widest(bounds.lower_68, Math.min),
            upper_68: widest(bounds.upper_68, Math.max),
        };
        values.forEach(([key, value]) => {
            row[key] = Number(value);
        });
        return row;
    });
    const currentPrice = first.current_price;
    const finalPoint = getFinalForecastPoint(points);
    const changePct = finalPoint && currentPrice
        ? ((finalPoint.predicted - currentPrice) / currentPrice) * 100
        : 0;

    return {
        status: "ok",
        model_available: true,
        current_price: currentPrice,
        current_price_source: first.current_price_source,
        forecast_points: points,
        ensemble: {
            target: finalPoint?.predicted,
            change_pct: changePct,
            upper_95: finalPoint?.upper_95,
            lower_95: finalPoint?.lower_95,
            upper_68: finalPoint?.upper_68,
            lower_68: finalPoint?.lower_68,
            upper_90: finalPoint?.upper_95,
            lower_90: finalPoint?.lower_95,
            signal: changePct >= 0 ? "Bullish" : "Bearish",
            reliability: available.length === MODEL_KEYS.length ? "Medium" : "Low",
            consensus: `${available.length} model${available.length === 1 ? "" : "s"} available`,
        },
        weights: available.reduce((acc, [key]) => {
            acc[key] = 1 / available.length;
            return acc;
        }, {}),
    };
}

function resolveDisplayData(data, selectedModel) {
    if (!data) return {};
    const ensemblePayload = data.ensemblePayload?.status === "ok"
        ? {
            ...data.ensemblePayload,
            forecast_points: (data.ensemblePayload.forecast_points || []).map(normalizeEnsemblePoint),
        }
        : buildFallbackEnsemble(data.models);

    if (MODEL_KEYS.includes(selectedModel)) {
        const payload = data.models?.[selectedModel];
        const points = payload?.forecast_points || [];
        const finalPoint = getFinalForecastPoint(points);
        const changePct = finalPoint && payload?.current_price
            ? ((finalPoint.predicted - payload.current_price) / payload.current_price) * 100
            : payload?.expected_change_pct;
        return {
            payload,
            points,
            currentPrice: payload?.current_price,
            currentPriceSource: payload?.current_price_source,
            target: finalPoint?.predicted ?? payload?.target_price ?? payload?.predicted_price,
            changePct,
            lower95: finalPoint?.lower_95 ?? payload?.lower95,
            upper95: finalPoint?.upper_95 ?? payload?.upper95,
            lower68: finalPoint?.lower_68 ?? payload?.lower68,
            upper68: finalPoint?.upper_68 ?? payload?.upper68,
            signal: payload?.signal || (Number(changePct) >= 0 ? "Bullish" : "Bearish"),
            reliability: payload?.status === "ok" ? "Model" : "Unavailable",
            tableLabel: modelLabel(selectedModel),
            chartModel: selectedModel,
            unavailable: payload?.status !== "ok",
            message: payload?.message || payload?.model_info?.message,
            // The single-model route carries provenance on model_info.
            pathType: payload?.model_info?.path_type,
            perStepPredictions: payload?.model_info?.per_step_predictions,
            modelOutputCount: payload?.model_info?.model_output_count,
            scenarioPaths: payload?.scenario_paths,
            probabilityUp: payload?.probability_up,
            probabilityDown: payload?.probability_down,
            headsNote: payload?.model_info?.heads_note,
        };
    }

    const finalPoint = getFinalForecastPoint(ensemblePayload?.forecast_points);
    const summary = ensemblePayload?.ensemble;
    return {
        payload: ensemblePayload,
        points: ensemblePayload?.forecast_points || [],
        currentPrice: ensemblePayload?.current_price,
        currentPriceSource: ensemblePayload?.current_price_source,
        target: summary?.target ?? finalPoint?.predicted,
        changePct: summary?.change_pct,
        lower95: summary?.lower_95 ?? summary?.lower_90 ?? finalPoint?.lower_95,
        upper95: summary?.upper_95 ?? summary?.upper_90 ?? finalPoint?.upper_95,
        lower68: summary?.lower_68 ?? finalPoint?.lower_68,
        upper68: summary?.upper_68 ?? finalPoint?.upper_68,
        signal: summary?.signal,
        reliability: summary?.reliability,
        consensus: summary?.consensus,
        tableLabel: selectedModel === "ensemble" ? "Ensemble" : "Forecast",
        chartModel: selectedModel === "all" ? "all" : "ensemble",
        unavailable: !ensemblePayload || ensemblePayload.status !== "ok",
        message: ensemblePayload?.message,
        pathType: ensemblePayload?.path_type,
        perStepPredictions: ensemblePayload?.per_step_predictions,
        modelOutputCount: ensemblePayload?.model_output_count,
        scenarioPaths: ensemblePayload?.scenario_paths,
        degraded: ensemblePayload?.degraded,
        modelsAvailable: ensemblePayload?.models_available,
        modelsUnavailable: ensemblePayload?.models_unavailable,
    };
}

/**
 * Say where the daily points came from.
 *
 * In the default mode each model emits a single cumulative horizon-day return.
 * There are no daily predictions behind it, which is why the panel above draws
 * a range at the horizon rather than a path to it - but the reader still needs
 * to know which of the two kinds of forecast they are looking at, because a
 * per-step model is making a genuinely different (and more compounding-prone)
 * claim from one that emitted a single endpoint.
 */
function ForecastProvenanceNote({ pathType, perStepPredictions, modelOutputCount, horizon, scenarioCount }) {
    if (!pathType) return null;
    // per_step_predictions is authoritative; path_type is the readable fallback.
    const perStep = perStepPredictions ?? pathType === "recursive_per_step";
    const interpolated = !perStep;
    return (
        <div
            style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "8px 12px",
                borderRadius: 8,
                border: `1px solid ${interpolated ? C.amber || "#8a6d3b" : C.border}`,
                background: "rgba(255,255,255,0.03)",
                color: C.sub,
                fontSize: 12,
                lineHeight: 1.5,
            }}
        >
            <span style={{ fontWeight: 900, color: interpolated ? C.amber || "#d6a13a" : C.green }}>
                {interpolated ? "Projected path" : "Per-step forecast"}
            </span>
            <span>
                {interpolated
                    ? `Each model produced exactly one number — the ${horizon}-day return${
                          modelOutputCount ? ` (${modelOutputCount} model outputs total)` : ""
                      }. There is no day-by-day forecast underneath it, so none is drawn: what the
                       model claims is an endpoint and an interval around it, and that is what the
                       range shows.`
                    : `Every point is a separate model prediction${
                          modelOutputCount ? ` (${modelOutputCount} model outputs)` : ""
                      }. Later steps are conditioned on earlier predicted bars, so error compounds —
                       which is why the range at the horizon, not the sequence, is what is drawn.`}
                {scenarioCount > 0 && (
                    <>
                        {" "}
                        The ticks under the axis are where {scenarioCount} simulated outcomes landed.
                        Their spread is the useful part; the paths that produced them are not shown
                        because none of them is a prediction.
                    </>
                )}
            </span>
        </div>
    );
}

function MetricCard({ label, value, sub, color }) {
    return (
        <div style={{
            background: COLORS.surface,
            border: `1px solid ${C.border}`,
            borderRadius: 8,
            padding: "16px 18px",
            minHeight: 96,
            display: "grid",
            alignContent: "center",
            gap: 7,
        }}>
            <div style={{ fontSize: 11, color: C.textDim, fontWeight: 700, letterSpacing: "0.04em", textTransform: "uppercase" }}>
                {label}
            </div>
            <div style={{ fontSize: 24, lineHeight: 1.1, color: color || C.text, fontWeight: 800 }}>
                {value}
            </div>
            {sub && <div style={{ fontSize: 12, color: C.textMid }}>{sub}</div>}
        </div>
    );
}

/**
 * The forecast, drawn as what it is: a range, not a path.
 *
 * This replaces a line chart that ran from today's close to the horizon-day
 * prediction. That line was not a forecast of the days it passed through —
 * outside per-step mode each model emits exactly one number, and the shape
 * between was P(0) x (1 + r)^(t/H), monotone by construction. Drawing it
 * invited the reader to trust a candle-by-candle claim the model never made,
 * and no amount of dashing or footnoting fixed that; the picture said one thing
 * while the caption said another.
 *
 * So the picture now says the same thing as the model. One axis of price, the
 * last close marked on it, the prediction marked on it, and the calibrated 68%
 * and 95% intervals drawn around it. Where the price goes in between is not
 * claimed, because it is not known.
 *
 * When the payload carries Monte Carlo scenarios, their *endpoints* are plotted
 * as a strip of ticks under the axis. Endpoints, not paths: the useful content
 * of those simulations is the spread of where price lands, and drawing the
 * paths themselves puts 200 fictional price histories on screen.
 */
function ForecastRange({ currentPrice, finalPoint, horizon, scenarioPaths, interpolated, label }) {
    const target = firstFiniteNumber(finalPoint?.prediction, finalPoint?.predicted, finalPoint?.ensemble);
    const lower95 = toFiniteNumber(finalPoint?.lower_95);
    const upper95 = toFiniteNumber(finalPoint?.upper_95);
    const lower68 = toFiniteNumber(finalPoint?.lower_68);
    const upper68 = toFiniteNumber(finalPoint?.upper_68);
    const anchor = toFiniteNumber(currentPrice);

    // Scenario endpoints only. The last element of each simulated path is where
    // that scenario finished; the elements before it are the fiction.
    const endpoints = (Array.isArray(scenarioPaths) ? scenarioPaths : [])
        .map((path) => toFiniteNumber(Array.isArray(path) ? path[path.length - 1] : null))
        .filter((value) => value !== null);

    const candidates = [target, lower95, upper95, lower68, upper68, anchor, ...endpoints]
        .filter((value) => value !== null);
    if (!candidates.length || target === null) {
        return (
            <div style={{ background: COLORS.panel, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18, color: C.textDim, fontSize: 12 }}>
                This model produced no forecast to draw a range from.
            </div>
        );
    }

    const min = Math.min(...candidates);
    const max = Math.max(...candidates);
    const span = max - min || Math.max(Math.abs(max) * 0.02, 1);
    const pad = span * 0.08;
    const low = min - pad;
    const high = max + pad;
    const position = (value) => ((value - low) / (high - low)) * 100;

    const band = (lo, hi, opacity) => (
        lo === null || hi === null ? null : (
            <div style={{
                position: "absolute", top: 26, height: 26,
                left: `${position(lo)}%`, width: `${position(hi) - position(lo)}%`,
                background: COLORS.band, opacity, borderRadius: 4,
            }} />
        )
    );

    const marker = (value, colour, caption, above) => (
        value === null ? null : (
            <div style={{ position: "absolute", left: `${position(value)}%`, top: above ? 0 : 52, transform: "translateX(-50%)", textAlign: "center" }}>
                {above && <div style={{ color: colour, fontSize: 11, fontWeight: 800, whiteSpace: "nowrap" }}>{formatPrice(value)}</div>}
                <div style={{ width: 2, height: 26, background: colour, margin: "0 auto" }} />
                {!above && <div style={{ color: colour, fontSize: 11, fontWeight: 800, whiteSpace: "nowrap", marginTop: 2 }}>{formatPrice(value)}</div>}
                <div style={{ color: C.textDim, fontSize: 10, whiteSpace: "nowrap", marginTop: above ? 0 : 2 }}>{caption}</div>
            </div>
        )
    );

    return (
        <div style={{ background: COLORS.panel, border: `1px solid ${C.border}`, borderRadius: 8, padding: "18px 22px 16px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 10, marginBottom: 10 }}>
                <div style={{ color: C.text, fontSize: 13, fontWeight: 800 }}>
                    {label || "Forecast"} — where price could be in {horizon} days
                </div>
                <div style={{ color: C.textDim, fontSize: 11 }}>
                    {interpolated
                        ? "one model output for the whole horizon"
                        : "per-step model outputs, endpoint shown"}
                </div>
            </div>

            <div style={{ position: "relative", height: 108, marginTop: 6 }}>
                {/* 95% then 68%, drawn widest first so the inner band reads darker */}
                {band(lower95, upper95, 0.10)}
                {band(lower68, upper68, 0.22)}

                {/* The price axis itself */}
                <div style={{ position: "absolute", top: 38, left: 0, right: 0, height: 2, background: C.border }} />

                {marker(anchor, C.amber, "last close", true)}
                {marker(target, COLORS.ensemble, `${horizon}-day forecast`, false)}

                {/* Scenario endpoints: one tick each, no paths */}
                {endpoints.map((value, index) => (
                    <div
                        key={index}
                        title={`Simulated outcome ${formatPrice(value)}`}
                        style={{
                            position: "absolute", top: 96, left: `${position(value)}%`,
                            width: 1, height: 10, background: COLORS.scenario, opacity: 0.4,
                        }}
                    />
                ))}
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: C.textDim, marginTop: 4 }}>
                <span>{formatPrice(lower95)} · 95% lower</span>
                {endpoints.length > 0 && <span>{endpoints.length} simulated outcomes</span>}
                <span>95% upper · {formatPrice(upper95)}</span>
            </div>
        </div>
    );
}

function WeightsPanel({ weights, consensus, signal, reliability }) {
    // A model missing from `weights` did not contribute — it was excluded from a
    // partial ensemble. Falling back to its nominal share showed an excluded LSTM
    // at 40% alongside two renormalised members, so the bars summed to 140% and
    // credited a model that never ran. Absent means zero.
    const hasWeights = weights && Object.keys(weights).length > 0;
    const resolved = {
        lstm: Number(hasWeights ? weights.lstm ?? 0 : 0.4),
        xgboost: Number(hasWeights ? weights.xgboost ?? 0 : 0.35),
        random_forest: Number(hasWeights ? weights.random_forest ?? 0 : 0.25),
    };
    const signalColor = signal === "Bearish" ? C.red : signal === "Neutral" ? C.amber : C.green;

    return (
        <div style={{ background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
            <div style={{ fontSize: 13, fontWeight: 800, color: C.text, marginBottom: 16 }}>Ensemble Weights</div>
            <div style={{ display: "grid", gap: 13 }}>
                {["lstm", "xgboost", "random_forest"].map((key) => {
                    const pct = Math.round(resolved[key] * 100);
                    const excluded = pct === 0;
                    return (
                        <div key={key} style={{ display: "grid", gridTemplateColumns: "128px 1fr 66px", alignItems: "center", gap: 12 }}>
                            <div style={{ color: excluded ? C.textDim : C.textMid, fontSize: 12, fontWeight: 700 }}>
                                {WEIGHT_LABELS[key]}
                            </div>
                            <div style={{ height: 10, background: "#0A0F18", borderRadius: 999, overflow: "hidden", border: "1px solid rgba(255,255,255,.05)" }}>
                                <div style={{ width: `${pct}%`, height: "100%", background: COLORS[key] }} />
                            </div>
                            <div style={{ color: excluded ? C.textDim : C.text, fontSize: 12, fontWeight: 800, textAlign: "right" }}>
                                {excluded ? "excl." : `${pct}%`}
                            </div>
                        </div>
                    );
                })}
            </div>
            <div style={{ marginTop: 18, borderTop: `1px solid ${C.border}`, paddingTop: 14 }}>
                <div style={{ color: signalColor, fontSize: 13, fontWeight: 800 }}>
                    {consensus || `${reliability || "Medium"} reliability`}
                </div>
            </div>
        </div>
    );
}

function ForecastTable({ rows, modelKey, label }) {
    const tableRows = (rows || []).slice(0, 7);
    return (
        <div style={{ background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18, overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 620, fontSize: 12 }}>
                <thead>
                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                        <th style={{ color: C.textDim, textAlign: "left", padding: "0 10px 10px 0", fontWeight: 800 }}>DATE</th>
                        <th style={{ color: COLORS[modelKey] || COLORS.ensemble, textAlign: "right", padding: "0 10px 10px", fontWeight: 800 }}>{label || "PREDICTED PRICE"}</th>
                        <th style={{ color: C.red, textAlign: "right", padding: "0 10px 10px", fontWeight: 800 }}>LOWER 95%</th>
                        <th style={{ color: C.green, textAlign: "right", padding: "0 10px 10px", fontWeight: 800 }}>UPPER 95%</th>
                        <th style={{ color: C.textDim, textAlign: "right", padding: "0 0 10px 10px", fontWeight: 800 }}>RANGE</th>
                    </tr>
                </thead>
                <tbody>
                    {tableRows.map((row) => {
                        const predicted = row.predicted ?? row.ensemble ?? row[modelKey];
                        const lower = row.lower_95 ?? row.lower_90;
                        const upper = row.upper_95 ?? row.upper_90;
                        const range = Number.isFinite(Number(upper)) && Number.isFinite(Number(lower)) ? Number(upper) - Number(lower) : null;
                        return (
                            <tr key={row.date} style={{ borderBottom: "1px solid rgba(255,255,255,.055)" }}>
                                <td style={{ color: C.textMid, padding: "10px 10px 10px 0", fontVariantNumeric: "tabular-nums" }}>{row.date}</td>
                                <td style={{ color: COLORS[modelKey] || COLORS.ensemble, textAlign: "right", padding: "10px", fontWeight: 800 }}>{formatPrice(predicted)}</td>
                                <td style={{ color: C.red, textAlign: "right", padding: "10px" }}>{formatPrice(lower)}</td>
                                <td style={{ color: C.green, textAlign: "right", padding: "10px" }}>{formatPrice(upper)}</td>
                                <td style={{ color: C.textMid, textAlign: "right", padding: "10px 0 10px 10px" }}>{formatPrice(range)}</td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export default function PredictionsTab({ selectedTicker, apiConnected, priceData, modelPrep }) {
    const [horizon, setHorizon] = useState(30);
    const [selectedModel, setSelectedModel] = useState("all");
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Preparation is App's, shared with the Analysis tab. `readyVersion` ticks
    // when a training run finishes, which is what re-runs the fetch below —
    // there is no button, and nothing here decides what gets trained.
    const readyVersion = modelPrep?.readyVersion ?? 0;
    // Only a live training run suppresses the normal loading state. The brief
    // readiness check on every ticker change must not flash a panel over a
    // forecast that is about to render perfectly well.
    const preparing = modelPrep?.status === "preparing";

    // Only the models the current view actually shows. A unified model is a
    // single next-day number that the fan chart has no room for, so selecting
    // one asks for that model alone; everything else asks for the fan.
    const activeModelKeys = useMemo(
        () => (isUnifiedModel(selectedModel) ? [selectedModel] : LEGACY_MODEL_KEYS),
        [selectedModel],
    );

    useEffect(() => {
        if (!apiConnected) return;
        setLoading(true);
        setError(null);
        Promise.allSettled([
            fetchEnsemblePrediction(selectedTicker, horizon),
            ...activeModelKeys.map((modelType) =>
                fetchPredictions(selectedTicker, modelType, isUnifiedModel(modelType) ? 1 : horizon),
            ),
        ])
            .then((results) => {
                const [ensembleResult, ...modelResults] = results;
                const models = {};
                const errors = {};
                activeModelKeys.forEach((modelType, index) => {
                    const result = modelResults[index];
                    if (result.status === "fulfilled") {
                        models[modelType] = normalizeSingleForecast(result.value, modelType);
                    } else {
                        models[modelType] = {
                            status: "unavailable",
                            model_available: false,
                            message: result.reason?.message || "This model produced no forecast for the current request.",
                            forecasts: [],
                            forecast_points: [],
                        };
                        errors[modelType] = models[modelType].message;
                    }
                });
                setData({
                    ensemblePayload: ensembleResult.status === "fulfilled" ? ensembleResult.value : null,
                    models,
                    errors,
                });
                setLoading(false);
            })
            .catch((err) => {
                setError(err?.message || "Forecast request failed.");
                setLoading(false);
            });
    }, [selectedTicker, horizon, apiConnected, readyVersion, activeModelKeys]);

    if (!apiConnected) {
        return (
            <div style={{ padding: 48, color: C.textDim, textAlign: "center" }}>
                Connect to the API server to view predictions.
            </div>
        );
    }

    const display = resolveDisplayData(data, selectedModel);
    const currentPrice = display.currentPrice ?? priceData?.bars?.[priceData.bars.length - 1]?.close;
    const modelUnavailable = !loading && !error && display.unavailable;
    const bullish = Number(display.changePct || 0) >= 0;
    const reliabilityColor = display.reliability === "Low" ? C.red : display.reliability === "Medium" ? C.amber : C.green;
    // A unified model answers for the next bar whatever the horizon selector says.
    const horizonLocked = isUnifiedModel(selectedModel);
    // per_step_predictions is authoritative; path_type is the readable fallback.
    const perStepForecast = display.perStepPredictions
        ?? display.pathType === "recursive_per_step";

    return (
        <div style={{ display: "grid", gap: 18, paddingBottom: 36 }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap", alignItems: "end" }}>
                <div style={{ display: "grid", gap: 6 }}>
                    <div style={{ color: C.textDim, fontSize: 12, fontWeight: 800, letterSpacing: "0.06em", textTransform: "uppercase" }}>Predictions</div>
                    <div style={{ color: C.text, fontSize: 25, fontWeight: 900, lineHeight: 1 }}>{selectedTicker}</div>
                </div>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                    <select
                        value={selectedModel}
                        onChange={(event) => setSelectedModel(event.target.value)}
                        style={{
                            background: COLORS.surface,
                            color: C.text,
                            border: `1px solid ${C.border}`,
                            borderRadius: 8,
                            padding: "8px 10px",
                            fontSize: 13,
                            fontWeight: 700,
                            outline: "none",
                        }}
                    >
                        {MODEL_OPTIONS.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                    </select>
                    <div style={{ display: "inline-flex", background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 3, gap: 3 }}>
                        {HORIZONS.map((days) => (
                            <button
                                key={days}
                                type="button"
                                disabled={horizonLocked}
                                title={horizonLocked ? "Unified models forecast the next bar only" : undefined}
                                onClick={() => setHorizon(days)}
                                style={{
                                    background: horizon === days ? COLORS.ensemble : "transparent",
                                    color: horizon === days ? "#10131A" : C.textMid,
                                    border: "none",
                                    borderRadius: 6,
                                    padding: "7px 12px",
                                    minWidth: 42,
                                    fontSize: 12,
                                    fontWeight: 900,
                                    cursor: horizonLocked ? "not-allowed" : "pointer",
                                    opacity: horizonLocked ? 0.5 : 1,
                                }}
                            >
                                {days}D
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* ── The chart. TradingView's, not ours. ──────────────
                The official Advanced Chart widget, pointed at whichever ticker
                is selected. Candles, volume, timeframes, drawing tools and
                studies are all theirs; nothing on this tab redraws price, and
                nothing on this chart produces a number the panels below read.
                The widget is the visualisation half of the split — our backend
                is the analysis half, and the two never cross. */}
            <div style={{
                background: COLORS.panel, border: `1px solid ${C.border}`,
                borderRadius: 8, padding: 14,
            }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginBottom: 10 }}>
                    <div style={{ color: C.text, fontSize: 13, fontWeight: 900 }}>
                        {selectedTicker} — live chart
                    </div>
                    <a
                        href={tradingViewUrl(selectedTicker)}
                        target="_blank"
                        rel="noreferrer noopener"
                        style={{ color: COLORS.ensemble, fontSize: 11, fontWeight: 700 }}
                    >Open on TradingView ↗</a>
                </div>
                <TradingViewChart symbol={selectedTicker} interval="1d" height={520} />
            </div>

            {/* ── The analysis. Ours, not TradingView's. ───────────
                Direction, probability, confidence and the evidence behind them,
                from GET /api/direction/{symbol}/analysis: trend, momentum,
                volume, price action, support/resistance, volatility regime and
                historical analogs, blended with the classifier by measured
                out-of-sample skill. */}
            <DirectionAnalysisPanel symbol={selectedTicker} apiConnected={apiConnected} />

            {/* The classifier's own track record, under the combined call it
                feeds. This is the panel that says whether the model reading
                tomorrow has ever been right: the gauge, the rolling hit rate
                against the base rate, and equity against buy & hold. The
                analysis above weights it by exactly these numbers, so the two
                belong next to each other. */}
            <DirectionPanel symbol={selectedTicker} modelPrep={modelPrep} />

            {/* One panel covers every reason there is nothing to draw, and the
                server decides which: training under way with its stages, a real
                failure with a retry, or a model that trained and did not clear
                its out-of-sample baseline. None of them is a button asking the
                user to start a training run.

                It renders while preparation is live even if a partial forecast
                is also on screen — a degraded ensemble filling in its missing
                member should say so while it happens. */}
            {(preparing || (!loading && !error && modelUnavailable)) && (
                <ModelPreparation preparation={modelPrep} context="forecast" />
            )}

            {!preparing && loading && (
                <div style={{ padding: 46, color: C.textDim, background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, textAlign: "center" }}>
                    Loading forecast...
                </div>
            )}

            {!preparing && !loading && error && (
                <div style={{ padding: 18, color: C.red, background: "rgba(244,63,94,.08)", border: "1px solid rgba(244,63,94,.35)", borderRadius: 8 }}>
                    {error}
                </div>
            )}

            {!loading && !error && !modelUnavailable && display.payload && (
                <>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
                        <MetricCard
                            label="Current"
                            value={formatPrice(currentPrice)}
                            sub={display.currentPriceSource ? display.currentPriceSource.replace("_", " ") : null}
                        />
                        <MetricCard
                            label={`${modelLabel(display.chartModel)} ${horizonLocked ? "Next Bar" : "Forecast"}`}
                            value={formatPrice(display.target)}
                            sub={`${formatPct(display.changePct)} ${display.signal || ""} | ${display.reliability || "Model"}`}
                            color={bullish ? COLORS.ensemble : C.red}
                        />
                        {Number.isFinite(Number(display.probabilityUp)) ? (
                            <MetricCard
                                label="Direction"
                                value={display.probabilityUp >= 0.5 ? "UP" : "DOWN"}
                                sub={`Up ${(display.probabilityUp * 100).toFixed(1)}% | Down ${((display.probabilityDown ?? 1 - display.probabilityUp) * 100).toFixed(1)}%`}
                                color={display.probabilityUp >= 0.5 ? C.green : C.red}
                            />
                        ) : (
                            <MetricCard label="Upper 95%" value={formatPrice(display.upper95)} color={C.green} />
                        )}
                        <MetricCard
                            label={horizonLocked ? "Forecast Horizon" : "Lower 95%"}
                            value={horizonLocked ? "Next 1 Bar" : formatPrice(display.lower95)}
                            color={horizonLocked ? COLORS.band : C.red}
                        />
                    </div>

                    <ForecastRange
                        currentPrice={currentPrice}
                        finalPoint={getFinalForecastPoint((display.points || []).slice(0, horizon))}
                        horizon={horizon}
                        scenarioPaths={display.scenarioPaths}
                        label={modelLabel(display.chartModel)}
                        interpolated={!perStepForecast}
                    />

                    <ForecastProvenanceNote
                        pathType={display.pathType}
                        perStepPredictions={display.perStepPredictions}
                        modelOutputCount={display.modelOutputCount}
                        horizon={horizon}
                        scenarioCount={display.scenarioPaths?.length ?? 0}
                    />

                    {display.headsNote && (
                        <div style={{
                            padding: "8px 12px",
                            borderRadius: 8,
                            background: "rgba(245, 200, 66, 0.08)",
                            border: `1px solid ${COLORS.ensemble}44`,
                            color: C.sub,
                            fontSize: 12,
                        }}>
                            {display.headsNote}
                        </div>
                    )}

                    {display.degraded && display.modelsUnavailable && (
                        <div style={{
                            padding: "8px 12px",
                            borderRadius: 8,
                            border: `1px solid ${C.amber || "#8a6d3b"}`,
                            background: "rgba(255,255,255,0.03)",
                            color: C.sub,
                            fontSize: 12,
                            lineHeight: 1.5,
                        }}>
                            <span style={{ fontWeight: 900, color: C.amber || "#d6a13a", marginRight: 8 }}>
                                Partial ensemble
                            </span>
                            Served by {(display.modelsAvailable || []).map((m) => MODEL_LABELS[m] || m).join(", ")}.{" "}
                            {Object.entries(display.modelsUnavailable)
                                .map(([m, why]) => `${MODEL_LABELS[m] || m} excluded — ${why}`)
                                .join("; ")}.
                        </div>
                    )}

                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 14, alignItems: "start" }}>
                        <WeightsPanel
                            weights={display.payload?.weights}
                            consensus={display.consensus}
                            signal={display.signal}
                            reliability={display.reliability}
                        />
                        <div style={{ display: "grid", gap: 10 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
                                <div style={{ color: C.text, fontSize: 13, fontWeight: 900 }}>
                                    {perStepForecast ? "Forecast Table" : "Model Output"}
                                </div>
                                <div style={{ color: reliabilityColor, fontSize: 12, fontWeight: 800 }}>
                                    {display.reliability || "Model"}
                                </div>
                            </div>
                            {/* Only a per-step model has a row per day. Listing the
                                interpolated points would put the removed path back
                                on screen as a table, which is the same fiction with
                                gridlines: the model emitted one number, so one row
                                is what there is to show. */}
                            <ForecastTable
                                rows={perStepForecast
                                    ? (display.points || [])
                                    : [getFinalForecastPoint(display.points || [])].filter(Boolean)}
                                modelKey={display.chartModel === "all" ? "ensemble" : display.chartModel}
                                label={`${display.tableLabel || "Predicted"} Price`}
                            />
                            {!perStepForecast && (
                                <div style={{ color: C.textDim, fontSize: 11, lineHeight: 1.5 }}>
                                    One row, because the model produced one number: the {horizon}-day
                                    endpoint and its interval. The days in between were never predicted.
                                </div>
                            )}
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}

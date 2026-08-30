import { useEffect, useMemo, useRef, useState } from "react";
import {
    Area,
    CartesianGrid,
    ComposedChart,
    Line,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import { C } from "../utils/data";
import { ApiError, fetchEnsemblePrediction, fetchPredictions, getEnsembleTrainingStatus, triggerEnsembleTraining } from "../utils/api";

/** Stop reporting progress after this long. The server-side job is unaffected. */
const POLL_TIMEOUT_MS = 30 * 60 * 1000;

const HORIZONS = [7, 15, 30, 60];
const MODEL_KEYS = ["xgboost", "random_forest", "lstm"];
const MODEL_OPTIONS = [
    { value: "all", label: "All Models" },
    { value: "ensemble", label: "Ensemble" },
    { value: "lstm", label: "LSTM" },
    { value: "xgboost", label: "XGBoost" },
    { value: "random_forest", label: "Random Forest" },
];

const COLORS = {
    historical: "#9CA3AF",
    ensemble: "#F5C842",
    lstm: "#60A5FA",
    xgboost: "#F59E0B",
    random_forest: "#34D399",
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
};

function formatPrice(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return `$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatPct(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(2)}%`;
}

function formatDateLabel(date) {
    if (typeof date !== "string" || date.length < 10) return date;
    return date.slice(5).replace("-", "/");
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
 * In the default mode each model emits a single cumulative horizon-day return
 * and the days in between are compounded toward it. Those points are not daily
 * predictions, and a chart that draws them like one invites the reader to trust
 * detail the model never produced.
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
                    ? `Each model produced one ${horizon}-day forecast${
                          modelOutputCount ? ` (${modelOutputCount} model outputs total)` : ""
                      }; the intermediate days are a compounded path to that endpoint, not daily predictions.`
                    : `Every point is a separate model prediction${
                          modelOutputCount ? ` (${modelOutputCount} model outputs)` : ""
                      }. Later steps are conditioned on earlier predicted bars, so error compounds.`}
                {scenarioCount > 0 && (
                    <>
                        {" "}
                        The centre line is an average outcome and is smooth by
                        construction — the {scenarioCount} simulated paths behind it show the
                        day-to-day volatility any real price path would carry.
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

function TrainButton({ symbol, onComplete }) {
    const [state, setState] = useState({ loading: false, jobId: null, message: null, error: null });
    const pollRef = useRef(null);

    async function handleTrain() {
        // A second click would otherwise leave the previous interval running forever.
        clearInterval(pollRef.current);
        setState({ loading: true, jobId: null, message: "Starting ensemble training...", error: null });
        try {
            const result = await triggerEnsembleTraining(symbol);
            const jobId = result?.job_id;
            setState((s) => ({ ...s, jobId, message: result?.message || "Training started." }));

            const startedAt = Date.now();
            pollRef.current = setInterval(async () => {
                try {
                    const status = await getEnsembleTrainingStatus(jobId);
                    if (status.status === "completed") {
                        clearInterval(pollRef.current);
                        setState({ loading: false, jobId, message: "Training complete.", error: null });
                        setTimeout(() => onComplete?.(), 1200);
                    } else if (status.status === "failed") {
                        clearInterval(pollRef.current);
                        setState({ loading: false, jobId, message: null, error: status.error || "Training failed." });
                    } else if (Date.now() - startedAt > POLL_TIMEOUT_MS) {
                        // The job outlives this page, so this is a reporting limit, not a failure.
                        clearInterval(pollRef.current);
                        setState({
                            loading: false,
                            jobId,
                            message: "Still training on the server. Reload later to pick up the result.",
                            error: null,
                        });
                    }
                } catch (err) {
                    // A 4xx will repeat forever, so stop and say why. Anything else is
                    // treated as a blip worth another poll.
                    if (err instanceof ApiError && err.status >= 400 && err.status < 500) {
                        clearInterval(pollRef.current);
                        setState({
                            loading: false,
                            jobId,
                            message: null,
                            error:
                                err.status === 429
                                    ? "Rate limited while checking training progress. Training may still be running on the server."
                                    : `Could not read training status (HTTP ${err.status}).`,
                        });
                    }
                }
            }, 3000);
        } catch (err) {
            const rateLimited = err instanceof ApiError && err.status === 429;
            setState({
                loading: false,
                jobId: null,
                message: null,
                error: rateLimited
                    ? "Training is rate limited (10 runs per hour). Wait for the window to clear, then try again."
                    : err?.message || "Training request failed.",
            });
        }
    }

    useEffect(() => () => clearInterval(pollRef.current), []);

    return (
        <div style={{ display: "grid", gap: 8, justifyItems: "start" }}>
            <button
                type="button"
                onClick={handleTrain}
                disabled={state.loading}
                style={{
                    background: state.loading ? C.bg3 : COLORS.ensemble,
                    color: state.loading ? C.textDim : "#10131A",
                    border: "none",
                    borderRadius: 8,
                    padding: "9px 16px",
                    fontSize: 13,
                    fontWeight: 800,
                    cursor: state.loading ? "default" : "pointer",
                }}
            >
                {state.loading ? "Training in progress..." : `Train ensemble for ${symbol}`}
            </button>
            {state.message && <div style={{ fontSize: 12, color: C.textMid }}>{state.message}</div>}
            {state.error && <div style={{ fontSize: 12, color: C.red }}>{state.error}</div>}
        </div>
    );
}

function TooltipContent({ active, payload, label }) {
    if (!active || !payload?.length) return null;
    const rows = payload
        .filter((item) => MODEL_LABELS[item.dataKey] && item.value !== null && item.value !== undefined)
        .map((item) => ({
            key: item.dataKey,
            label: MODEL_LABELS[item.dataKey],
            value: item.value,
            color: item.color,
        }));
    if (!rows.length) return null;

    return (
        <div style={{
            background: "#0B101A",
            border: `1px solid ${C.border}`,
            borderRadius: 8,
            padding: "10px 12px",
            boxShadow: "0 12px 28px rgba(0,0,0,.28)",
            minWidth: 170,
        }}>
            <div style={{ color: C.text, fontWeight: 700, fontSize: 12, marginBottom: 8 }}>{label}</div>
            <div style={{ display: "grid", gap: 5 }}>
                {rows.map((row) => (
                    <div key={row.key} style={{ display: "flex", justifyContent: "space-between", gap: 18, fontSize: 12 }}>
                        <span style={{ color: row.color }}>{row.label}</span>
                        <span style={{ color: C.text }}>{formatPrice(row.value)}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function ForecastChart({ priceData, forecastPoints, selectedModel, scenarioPaths, showScenarios, onToggleScenarios }) {
    const { chartData, todayDate, yDomain, scenarioKeys } = useMemo(() => {
        const history = (priceData?.bars || []).slice(-60);
        const future = (forecastPoints || [])
            .filter((point) => point?.date)
            .slice()
            .sort((a, b) => String(a.date).localeCompare(String(b.date)));
        const rows = [];
        let boundary = null;

        // Monte Carlo paths, one series per scenario. Each path is [today, ...steps],
        // so element 0 pins to the last historical bar and element i+1 to forecast
        // point i — that alignment is what makes the fan start at today's price
        // instead of floating free of the history.
        const paths = showScenarios && Array.isArray(scenarioPaths) ? scenarioPaths : [];
        const keys = paths.map((_, index) => `scenario_${index}`);

        history.forEach((bar, index) => {
            const isLast = index === history.length - 1;
            if (isLast) boundary = bar.date;
            const row = {
                date: bar.date,
                historical: bar.close,
                ensemble: isLast && forecastPoints.length ? bar.close : null,
                lstm: isLast && forecastPoints.length ? bar.close : null,
                xgboost: isLast && forecastPoints.length ? bar.close : null,
                random_forest: isLast && forecastPoints.length ? bar.close : null,
                upper_90: isLast && forecastPoints.length ? bar.close : null,
                lower_90: isLast && forecastPoints.length ? bar.close : null,
                upper_95: isLast && forecastPoints.length ? bar.close : null,
                lower_95: isLast && forecastPoints.length ? bar.close : null,
                upper_68: isLast && forecastPoints.length ? bar.close : null,
                lower_68: isLast && forecastPoints.length ? bar.close : null,
            };
            keys.forEach((key, pathIndex) => {
                row[key] = isLast ? toFiniteNumber(paths[pathIndex]?.[0]) ?? bar.close : null;
            });
            rows.push(row);
        });

        future.forEach((point, stepIndex) => {
            const prediction = firstFiniteNumber(point.prediction, point.predicted, point.ensemble);
            const row = {
                date: point.date,
                historical: null,
                ensemble: prediction,
                lstm: firstFiniteNumber(point.lstm, selectedModel === "lstm" ? prediction : null),
                xgboost: firstFiniteNumber(point.xgboost, selectedModel === "xgboost" ? prediction : null),
                random_forest: firstFiniteNumber(point.random_forest, selectedModel === "random_forest" ? prediction : null),
                upper_90: point.upper_90 ?? point.upper_95,
                lower_90: point.lower_90 ?? point.lower_95,
                upper_95: point.upper_95 ?? point.upper_90,
                lower_95: point.lower_95 ?? point.lower_90,
                upper_68: point.upper_68,
                lower_68: point.lower_68,
            };
            keys.forEach((key, pathIndex) => {
                row[key] = toFiniteNumber(paths[pathIndex]?.[stepIndex + 1]);
            });
            rows.push(row);
        });

        const visibleKeys = selectedModel === "all"
            ? ["historical", "ensemble", "lstm", "xgboost", "random_forest"]
            : ["historical", selectedModel];
        // Scenario paths are deliberately included in the domain: clipping the fan
        // would understate exactly the spread it exists to show.
        const domainKeys = [...visibleKeys, ...keys];
        const lineValues = rows.flatMap((row) => domainKeys.map((key) => toFiniteNumber(row[key]))).filter((value) => value !== null);
        const min = Math.min(...lineValues);
        const max = Math.max(...lineValues);
        const span = Number.isFinite(max - min) ? max - min : 0;
        const pad = Math.max(span * 0.08, Math.abs(max || min || 0) * 0.01, 1);
        const domain = lineValues.length
            ? [Math.max(0, Number((min - pad).toFixed(2))), Number((max + pad).toFixed(2))]
            : ["auto", "auto"];

        return { chartData: rows, todayDate: boundary, yDomain: domain, scenarioKeys: keys };
    }, [priceData, forecastPoints, selectedModel, scenarioPaths, showScenarios]);

    const show = (model) => selectedModel === "all" || selectedModel === model;
    const scenarioCount = Array.isArray(scenarioPaths) ? scenarioPaths.length : 0;

    return (
        <div style={{ background: COLORS.panel, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 14, fontSize: 12, color: C.textMid, alignItems: "center" }}>
                {[
                    ["historical", "Historical", null],
                    ["ensemble", "Ensemble", null],
                    ["lstm", "LSTM", "5 5"],
                    ["xgboost", "XGBoost", "5 5"],
                    ["random_forest", "Random Forest", "5 5"],
                ].map(([key, label, dash]) => (
                    <div key={key} style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
                        <span style={{ width: 18, height: 0, borderTop: `3px ${dash ? "dashed" : "solid"} ${COLORS[key]}` }} />
                        <span>{label}</span>
                    </div>
                ))}
                {scenarioCount > 0 && (
                    <button
                        type="button"
                        onClick={onToggleScenarios}
                        title="Monte Carlo paths simulated by resampling this stock's own recent daily moves. They show the volatility a forecast has to live inside; the centre line is the average outcome, not a path the market is expected to trace."
                        style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 7,
                            marginLeft: "auto",
                            background: "transparent",
                            border: `1px solid ${C.border}`,
                            borderRadius: 6,
                            padding: "4px 9px",
                            color: showScenarios ? C.text : C.textDim,
                            fontSize: 12,
                            cursor: "pointer",
                        }}
                    >
                        <span style={{
                            width: 18,
                            height: 0,
                            borderTop: `2px solid ${COLORS.scenario}`,
                            opacity: showScenarios ? 0.9 : 0.3,
                        }} />
                        <span>{scenarioCount} simulated paths</span>
                    </button>
                )}
            </div>
            <ResponsiveContainer width="100%" height={366}>
                <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                    <CartesianGrid stroke={C.border} strokeDasharray="3 3" opacity={0.28} vertical={false} />
                    <XAxis
                        dataKey="date"
                        tick={{ fill: C.textDim, fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={formatDateLabel}
                        minTickGap={22}
                    />
                    <YAxis
                        orientation="right"
                        tick={{ fill: C.textDim, fontSize: 11 }}
                        domain={yDomain}
                        allowDataOverflow
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={(value) => `$${Number(value).toFixed(0)}`}
                    />
                    <Tooltip content={<TooltipContent />} />
                    <Area
                        type="monotone"
                        dataKey="upper_95"
                        stroke="none"
                        fill={COLORS.band}
                        fillOpacity={0.06}
                        connectNulls
                        isAnimationActive={false}
                        tooltipType="none"
                    />
                    <Area
                        type="monotone"
                        dataKey="lower_95"
                        stroke="none"
                        fill={COLORS.panel}
                        fillOpacity={1}
                        connectNulls
                        isAnimationActive={false}
                        tooltipType="none"
                    />
                    <Area
                        type="monotone"
                        dataKey="upper_68"
                        stroke="none"
                        fill={COLORS.band}
                        fillOpacity={0.14}
                        connectNulls
                        isAnimationActive={false}
                        tooltipType="none"
                    />
                    <Area
                        type="monotone"
                        dataKey="lower_68"
                        stroke="none"
                        fill={COLORS.panel}
                        fillOpacity={1}
                        connectNulls
                        isAnimationActive={false}
                        tooltipType="none"
                    />
                    {todayDate && (
                        <ReferenceLine
                            x={todayDate}
                            stroke={C.textDim}
                            strokeDasharray="4 4"
                            label={{ value: "Today", position: "top", fill: C.textMid, fontSize: 11 }}
                        />
                    )}
                    {/* Simulated paths first, so the forecast lines draw over them.
                        `type="linear"` keeps each daily step as its own segment —
                        monotone smoothing would round off the very step-to-step
                        movement these paths exist to show. */}
                    {scenarioKeys.map((key) => (
                        <Line
                            key={key}
                            type="linear"
                            dataKey={key}
                            stroke={COLORS.scenario}
                            strokeWidth={1}
                            strokeOpacity={0.32}
                            dot={false}
                            connectNulls
                            isAnimationActive={false}
                            legendType="none"
                            tooltipType="none"
                        />
                    ))}
                    <Line type="monotone" dataKey="historical" stroke={COLORS.historical} strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line hide={!show("lstm")} type="monotone" dataKey="lstm" stroke={COLORS.lstm} strokeWidth={2} strokeDasharray="6 5" dot={false} connectNulls isAnimationActive={false} />
                    <Line hide={!show("xgboost")} type="monotone" dataKey="xgboost" stroke={COLORS.xgboost} strokeWidth={2} strokeDasharray="6 5" dot={false} connectNulls isAnimationActive={false} />
                    <Line hide={!show("random_forest")} type="monotone" dataKey="random_forest" stroke={COLORS.random_forest} strokeWidth={2} strokeDasharray="6 5" dot={false} connectNulls isAnimationActive={false} />
                    <Line hide={!show("ensemble")} type="monotone" dataKey="ensemble" stroke={COLORS.ensemble} strokeWidth={4} dot={false} connectNulls isAnimationActive={false} />
                </ComposedChart>
            </ResponsiveContainer>
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

export default function PredictionsTab({ selectedTicker, apiConnected, priceData }) {
    const [horizon, setHorizon] = useState(30);
    const [selectedModel, setSelectedModel] = useState("all");
    const [showScenarios, setShowScenarios] = useState(true);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [refreshKey, setRefreshKey] = useState(0);

    useEffect(() => {
        if (!apiConnected) return;
        setLoading(true);
        setError(null);
        Promise.allSettled([
            fetchEnsemblePrediction(selectedTicker, horizon),
            ...MODEL_KEYS.map((modelType) => fetchPredictions(selectedTicker, modelType, horizon)),
        ])
            .then((results) => {
                const [ensembleResult, ...modelResults] = results;
                const models = {};
                const errors = {};
                MODEL_KEYS.forEach((modelType, index) => {
                    const result = modelResults[index];
                    if (result.status === "fulfilled") {
                        models[modelType] = normalizeSingleForecast(result.value, modelType);
                    } else {
                        models[modelType] = {
                            status: "unavailable",
                            model_available: false,
                            message: result.reason?.message || "Prediction model not available. Please train or load model bundle.",
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
    }, [selectedTicker, horizon, apiConnected, refreshKey]);

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
                                    cursor: "pointer",
                                }}
                            >
                                {days}D
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {loading && (
                <div style={{ padding: 46, color: C.textDim, background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, textAlign: "center" }}>
                    Loading forecast...
                </div>
            )}

            {!loading && error && (
                <div style={{ padding: 18, color: C.red, background: "rgba(244,63,94,.08)", border: "1px solid rgba(244,63,94,.35)", borderRadius: 8 }}>
                    {error}
                </div>
            )}

            {!loading && !error && modelUnavailable && (
                <div style={{ padding: 28, background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, display: "grid", gap: 16 }}>
                    <div>
                        <div style={{ color: C.text, fontSize: 18, fontWeight: 900, marginBottom: 6 }}>Prediction model unavailable</div>
                        <div style={{ color: C.textMid, fontSize: 13 }}>
                            {display.message || "Prediction model not available. Please train or load model bundle."}
                        </div>
                    </div>
                    <TrainButton symbol={selectedTicker} onComplete={() => setRefreshKey((key) => key + 1)} />
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
                            label={`${modelLabel(display.chartModel)} Forecast`}
                            value={formatPrice(display.target)}
                            sub={`${formatPct(display.changePct)} ${display.signal || ""} | ${display.reliability || "Model"}`}
                            color={bullish ? COLORS.ensemble : C.red}
                        />
                        <MetricCard label="Upper 95%" value={formatPrice(display.upper95)} color={C.green} />
                        <MetricCard label="Lower 95%" value={formatPrice(display.lower95)} color={C.red} />
                    </div>

                    <ForecastChart
                        priceData={priceData}
                        forecastPoints={(display.points || []).slice(0, horizon)}
                        selectedModel={display.chartModel}
                        scenarioPaths={display.scenarioPaths}
                        showScenarios={showScenarios}
                        onToggleScenarios={() => setShowScenarios((on) => !on)}
                    />

                    <ForecastProvenanceNote
                        pathType={display.pathType}
                        perStepPredictions={display.perStepPredictions}
                        modelOutputCount={display.modelOutputCount}
                        horizon={horizon}
                        scenarioCount={showScenarios ? (display.scenarioPaths?.length ?? 0) : 0}
                    />

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
                                <div style={{ color: C.text, fontSize: 13, fontWeight: 900 }}>Forecast Table</div>
                                <div style={{ color: reliabilityColor, fontSize: 12, fontWeight: 800 }}>
                                    {display.reliability || "Model"}
                                </div>
                            </div>
                            <ForecastTable
                                rows={display.points || []}
                                modelKey={display.chartModel === "all" ? "ensemble" : display.chartModel}
                                label={`${display.tableLabel || "Predicted"} Price`}
                            />
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
